#!/usr/bin/env python3
"""Compact regenerated RPM-10K data and publish deterministic nested subsets."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean

from PIL import Image
from transformers import AutoTokenizer


SIZES = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8729)


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def load_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not ids or any(not sample_id for sample_id in ids):
        raise ValueError(f"invalid or empty ID manifest: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in manifest: {path}")
    return ids


def extract_text_content(content, *, sample_id: str) -> str:
    if isinstance(content, str):
        return content.replace("<image>\n", "").replace("<image>", "").strip()
    if not isinstance(content, list):
        raise ValueError(f"{sample_id}: unsupported user content type")
    text_parts = []
    image_parts = 0
    for part in content:
        if not isinstance(part, dict):
            raise ValueError(f"{sample_id}: malformed multimodal content part")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{sample_id}: text part is not a string")
            text_parts.append(text)
        elif part_type == "image_url":
            image_parts += 1
        else:
            raise ValueError(f"{sample_id}: unsupported content part {part_type!r}")
    if image_parts != 1:
        raise ValueError(f"{sample_id}: expected one embedded image, got {image_parts}")
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError(f"{sample_id}: empty user text")
    return text


def compact_record(row: dict, *, line_number: int) -> tuple[str, dict, str]:
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_number}: missing sample ID")
    if row.get("status") != "success":
        raise ValueError(f"{sample_id}: regeneration status is not success")
    image = row.get("image") or row.get("image_path")
    if not isinstance(image, str) or not image:
        raise ValueError(f"{sample_id}: missing image path")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"{sample_id}: expected exactly one user/assistant turn")
    user, assistant = conversations
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError(f"{sample_id}: unexpected conversation role sequence")
    question = extract_text_content(user.get("content"), sample_id=sample_id)
    answer = assistant.get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{sample_id}: empty regenerated answer")

    compact = {
        key: value
        for key, value in row.items()
        if key not in {"conversations", "status", "error", "image_path", "images"}
    }
    compact["image"] = image
    compact["conversations"] = [
        {"role": "user", "content": f"<image>\n{question}"},
        {"role": "assistant", "content": answer.strip()},
    ]
    return sample_id, compact, answer.strip()


def resize_if_needed(
    image_path: Path,
    *,
    output_dir: Path,
    max_pixels: int,
) -> tuple[str, tuple[int, int], tuple[int, int], bool]:
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        if width * height <= max_pixels:
            return str(image_path), (width, height), (width, height), False
        scale = math.sqrt(max_pixels / float(width * height))
        new_width = max(1, int(math.floor(width * scale)))
        new_height = max(1, int(math.floor(height * scale)))
        while new_width * new_height > max_pixels:
            if new_width >= new_height:
                new_width -= 1
            else:
                new_height -= 1
        resized = image.convert("RGB").resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        destination = output_dir / image_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        resized.save(temporary, format="JPEG", quality=95, subsampling=0)
        os.replace(temporary, destination)
        return (
            str(destination),
            (width, height),
            (new_width, new_height),
            True,
        )


def write_jsonl(path: Path, ids: list[str], records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in ids:
            handle.write(json.dumps(records[sample_id], ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-image-pixels", type=int, default=802_816)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    regen_path = (
        data_root / "regen" / "rpm10k-train-clean-8729-qwen3vl4b-greedy.jsonl"
    )
    train_ready = data_root / "train-ready"
    output_full = train_ready / "rpm10k-qwen3vl4b-regen-train-8729.jsonl"
    report_path = train_ready / "train_ready_report.json"
    if not args.overwrite and (output_full.exists() or report_path.exists()):
        raise FileExistsError("train-ready outputs already exist; pass --overwrite")

    ids_by_size = {
        size: load_ids(
            data_root / "subset_ids" / f"rpm10k-train-clean-{size}.ids"
        )
        for size in SIZES
    }
    for size, ids in ids_by_size.items():
        if len(ids) != size:
            raise ValueError(f"{size}: ID manifest contains {len(ids)} rows")
    for previous, current in zip(SIZES, SIZES[1:]):
        if ids_by_size[current][:previous] != ids_by_size[previous]:
            raise ValueError(f"ID manifests are not prefix-nested: {previous}, {current}")

    records: dict[str, dict] = {}
    answers: dict[str, str] = {}
    with regen_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            sample_id, compact, answer = compact_record(
                json.loads(line), line_number=line_number
            )
            if sample_id in records:
                raise ValueError(f"duplicate regenerated ID: {sample_id}")
            records[sample_id] = compact
            answers[sample_id] = answer

    expected = ids_by_size[SIZES[-1]]
    missing = sorted(set(expected) - set(records))
    unexpected = sorted(set(records) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"regenerated/manifest ID mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    resized_dir = train_ready / "resized-images-max802816"
    resized_count = 0
    before_pixels = []
    after_pixels = []
    for sample_id in expected:
        source = Path(records[sample_id]["image"])
        if not source.is_file():
            raise FileNotFoundError(f"{sample_id}: image does not exist: {source}")
        image, before, after, resized = resize_if_needed(
            source,
            output_dir=resized_dir,
            max_pixels=args.max_image_pixels,
        )
        records[sample_id]["image"] = image
        before_pixels.append(before[0] * before[1])
        after_pixels.append(after[0] * after[1])
        resized_count += int(resized)

    write_jsonl(output_full, expected, records)
    subset_paths = {}
    for size in SIZES:
        path = train_ready / "subsets" / f"rpm10k-qwen3vl4b-regen-train-{size}.jsonl"
        write_jsonl(path, ids_by_size[size], records)
        subset_paths[str(size)] = str(path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    answer_tokens = [
        len(tokenizer.encode(answers[sample_id], add_special_tokens=False))
        for sample_id in expected
    ]
    supervised_span = [length + 1 for length in answer_tokens]
    report = {
        "status": "pass",
        "source_regenerated_path": str(regen_path),
        "source_rows": len(records),
        "full_output_path": str(output_full),
        "subset_paths": subset_paths,
        "nested_by_prefix": True,
        "max_image_pixels": args.max_image_pixels,
        "resized_images": resized_count,
        "original_image_pixels": describe(before_pixels),
        "train_image_pixels": describe(after_pixels),
        "regenerated_answer_tokens": describe(answer_tokens),
        "estimated_supervised_span_tokens": describe(supervised_span),
        "anchor_coverage": {
            str(limit): {
                "rows_with_at_least_limit": sum(
                    length >= limit for length in answer_tokens
                ),
                "fraction_with_at_least_limit": sum(
                    length >= limit for length in answer_tokens
                ) / len(answer_tokens),
                "mean_effective_anchors": mean(
                    min(length, limit) for length in answer_tokens
                ),
            }
            for limit in (8, 16, 32, 64, 128, 256)
        },
        "optimizer_steps": {
            str(size): (size * 3) // (2 * 1 * 2) for size in SIZES
        },
        "schedule_contract": {
            "num_epochs": 3,
            "trainer_dp_size": 2,
            "per_rank_batch_size": 1,
            "accumulation_steps": 2,
            "global_optimizer_batch": 4,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
