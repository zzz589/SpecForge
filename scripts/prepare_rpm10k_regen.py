#!/usr/bin/env python3
"""Clean RPM-10K and prepare nested SpecForge regeneration inputs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261109)
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def clean_row(
    row: dict,
    *,
    split: str,
    source_index: int,
    image_root: Path,
) -> dict:
    image_name = str(row["image"]).strip()
    image_path = (image_root / image_name).resolve()
    query = " ".join(str(row["query"]).split())
    answer = str(row["reading"]).strip()
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"image", "query", "reading", "detail"}
    }
    return {
        "id": f"rpm10k_{Path(image_name).stem}",
        "_dataset_source": "rpm10k",
        "_rpm10k_split": split,
        "_rpm10k_source_index": source_index,
        "image": str(image_path),
        "reference_answer": answer,
        "rpm10k_metadata": metadata,
        "conversations": [
            {"role": "user", "content": f"<image>\n{query}"},
            {"role": "assistant", "content": answer},
        ],
    }


def token_lengths(tokenizer, rows: list[dict], raw_by_image: dict[str, dict]) -> dict:
    query_lengths: list[int] = []
    answer_lengths: list[int] = []
    assistant_span_lengths: list[int] = []
    detail_lengths: list[int] = []

    for row in rows:
        user = row["conversations"][0]
        assistant = row["conversations"][1]
        query_lengths.append(
            len(tokenizer.encode(user["content"], add_special_tokens=False))
        )
        answer_lengths.append(
            len(tokenizer.encode(assistant["content"], add_special_tokens=False))
        )
        try:
            prompt_ids = tokenizer.apply_chat_template(
                [user],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_ids = tokenizer.apply_chat_template(
                [user, assistant],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            span_length = max(0, len(full_ids) - len(prompt_ids))
            if span_length == 0:
                # Some VLM tokenizer-only chat templates omit multimodal
                # assistant content.  The answer tokens plus the assistant
                # terminator are the reliable lower-level estimate here.
                span_length = answer_lengths[-1] + 1
            assistant_span_lengths.append(span_length)
        except Exception:
            assistant_span_lengths.append(answer_lengths[-1] + 1)

        raw = raw_by_image[Path(row["image"]).name]
        detail = raw.get("detail")
        if isinstance(detail, str) and detail.strip():
            detail_lengths.append(
                len(tokenizer.encode(detail, add_special_tokens=False))
            )

    valid_anchor_counts = [max(0, length - 1) for length in assistant_span_lengths]
    return {
        "tokenizer": str(tokenizer.name_or_path),
        "query_tokens": summarize(query_lengths),
        "reference_answer_tokens": summarize(answer_lengths),
        "assistant_supervised_span_tokens": summarize(assistant_span_lengths),
        "raw_detail_tokens": summarize(detail_lengths),
        "estimated_valid_anchor_positions": summarize(valid_anchor_counts),
        "valid_anchor_coverage": {
            str(limit): {
                "rows_with_at_least_limit": sum(value >= limit for value in valid_anchor_counts),
                "fraction_with_at_least_limit": (
                    sum(value >= limit for value in valid_anchor_counts) / len(valid_anchor_counts)
                ),
                "mean_effective_anchors": statistics.fmean(
                    min(value, limit) for value in valid_anchor_counts
                ),
            }
            for limit in (4, 8, 16, 32)
        },
        "note": (
            "These anchor counts describe the original reference answers. "
            "Recompute after target-model regeneration before fixing num_anchors."
        ),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_root = input_root / "images"
    label_root = input_root / "labels"

    if args.gap <= 0:
        raise ValueError("--gap must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = output_dir / "subsets"
    id_dir = output_dir / "subset_ids"
    subset_dir.mkdir()
    id_dir.mkdir()

    raw_train = json.loads((label_root / "train.json").read_text(encoding="utf-8"))
    raw_test = json.loads((label_root / "test.json").read_text(encoding="utf-8"))
    test_images = {str(row["image"]).strip() for row in raw_test}
    overlap_images = sorted(
        {str(row["image"]).strip() for row in raw_train} & test_images
    )

    rejected: list[dict] = []
    image_sizes: list[tuple[int, int]] = []
    clean_train: list[dict] = []
    clean_test: list[dict] = []

    for split, source_rows, destination in (
        ("train", raw_train, clean_train),
        ("test", raw_test, clean_test),
    ):
        seen_images: set[str] = set()
        for source_index, row in enumerate(source_rows):
            try:
                image_name = str(row.get("image", "")).strip()
                if not image_name:
                    raise ValueError("missing image")
                if split == "train" and image_name in test_images:
                    rejected.append(
                        {
                            "split": split,
                            "source_index": source_index,
                            "image": image_name,
                            "reason": "train_test_overlap_kept_in_test",
                        }
                    )
                    continue
                if image_name in seen_images:
                    raise ValueError("duplicate image within split")
                if not str(row.get("query", "")).strip():
                    raise ValueError("empty query")
                if row.get("reading") is None or not str(row["reading"]).strip():
                    raise ValueError("empty reading")
                image_path = image_root / image_name
                if not image_path.is_file():
                    raise ValueError("missing image file")
                image_sizes.append(verify_image(image_path))
                destination.append(
                    clean_row(
                        row,
                        split=split,
                        source_index=source_index,
                        image_root=image_root,
                    )
                )
                seen_images.add(image_name)
            except Exception as exc:
                rejected.append(
                    {
                        "split": split,
                        "source_index": source_index,
                        "image": row.get("image") if isinstance(row, dict) else None,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

    rng = random.Random(args.seed)
    rng.shuffle(clean_train)

    train_path = output_dir / f"rpm10k-train-clean-{len(clean_train)}.jsonl"
    test_path = output_dir / f"rpm10k-test-clean-{len(clean_test)}.jsonl"
    write_jsonl(train_path, clean_train)
    write_jsonl(test_path, clean_test)

    subset_sizes = list(range(args.gap, len(clean_train) + 1, args.gap))
    if not subset_sizes or subset_sizes[-1] != len(clean_train):
        subset_sizes.append(len(clean_train))
    subset_manifest = []
    for size in subset_sizes:
        subset = clean_train[:size]
        subset_path = subset_dir / f"rpm10k-train-clean-{size}.jsonl"
        ids_path = id_dir / f"rpm10k-train-clean-{size}.ids"
        write_jsonl(subset_path, subset)
        ids_path.write_text(
            "".join(f"{row['id']}\n" for row in subset), encoding="utf-8"
        )
        subset_manifest.append(
            {
                "size": size,
                "data_path": str(subset_path),
                "ids_path": str(ids_path),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    raw_train_by_image = {str(row["image"]): row for row in raw_train}
    length_report = token_lengths(tokenizer, clean_train, raw_train_by_image)
    length_report["image_width"] = summarize([width for width, _ in image_sizes])
    length_report["image_height"] = summarize([height for _, height in image_sizes])
    write_json(output_dir / "length_report.json", length_report)

    all_image_names = {path.name for path in image_root.glob("*.jpg")}
    referenced = {
        Path(row["image"]).name for row in clean_train + clean_test
    }
    cleaning_report = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "gap": args.gap,
        "raw_train_rows": len(raw_train),
        "raw_test_rows": len(raw_test),
        "clean_train_rows": len(clean_train),
        "clean_test_rows": len(clean_test),
        "overlap_images": overlap_images,
        "unreferenced_images": sorted(all_image_names - referenced),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "raw_detail_policy": (
            "Dropped from regeneration conversations because some chain-of-thought "
            "text contradicts its final answer. The stable reading is kept as "
            "reference_answer; the target model will regenerate assistant content."
        ),
        "subset_manifest": subset_manifest,
        "full_train_path": str(train_path),
        "test_path": str(test_path),
    }
    write_json(output_dir / "cleaning_report.json", cleaning_report)
    write_json(output_dir / "subset_manifest.json", subset_manifest)

    print(json.dumps(cleaning_report, ensure_ascii=False, indent=2))
    print(json.dumps(length_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
