#!/usr/bin/env python3
"""Convert and validate LLaVA data for the SpecForge VLM pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path


# Import only the standalone data-validation modules. Importing the top-level
# specforge package would unnecessarily initialize Torch/CUDA model modules.
repository_root = Path(__file__).resolve().parents[1]
data_module_root = repository_root / "specforge" / "data"

if str(data_module_root) not in sys.path:
    sys.path.insert(0, str(data_module_root))

from pipeline_issues import (
    IssueCode,
    IssueReporter,
    PipelineIssue,
    write_json_atomic,
)
from pipeline_validation import (
    SampleRejected,
    convert_llava_conversations,
    normalize_llava_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert and validate LLaVA JSON data for SpecForge VLM."
        )
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--image-prefix")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--rejected-path")
    parser.add_argument("--summary-path")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Only check image existence; do not verify image decoding.",
    )
    parser.add_argument(
        "--max-rejected-ratio",
        type=float,
        default=None,
        help=(
            "Fail without publishing the accepted output if the rejected "
            "ratio exceeds this value."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_conversations(messages):
    """Backward-compatible wrapper used by older callers."""
    try:
        return convert_llava_conversations(messages)
    except SampleRejected:
        return None


def sidecar_path(output_path: Path, suffix: str) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(suffix)
    return Path(str(output_path) + suffix)


def validate_args(args, paths):
    input_path, image_root, output_path, rejected_path, summary_path = paths

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root not found: {image_root}")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")
    if (
        args.max_rejected_ratio is not None
        and not 0.0 <= args.max_rejected_ratio <= 1.0
    ):
        raise ValueError("--max-rejected-ratio must be in [0, 1]")

    existing_outputs = [
        path
        for path in (output_path, rejected_path, summary_path)
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing_outputs)
        )


def main():
    args = parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().absolute()
    output_path = Path(args.output_path).expanduser().resolve()

    rejected_path = (
        Path(args.rejected_path).expanduser().resolve()
        if args.rejected_path
        else sidecar_path(output_path, ".rejected.jsonl")
    )
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else sidecar_path(output_path, ".summary.json")
    )

    paths = (
        input_path,
        image_root,
        output_path,
        rejected_path,
        summary_path,
    )
    validate_args(args, paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + ".tmp")
    temporary_path.unlink(missing_ok=True)

    with input_path.open(encoding="utf-8") as input_file:
        samples = json.load(input_file)

    if not isinstance(samples, list):
        raise ValueError("Input must contain a top-level JSON array")

    stats = {
        "input_samples_total": len(samples),
        "inspected_rows": 0,
        "prefix_excluded_rows": 0,
        "candidate_rows": 0,
        "written_rows": 0,
        "unknown_source_rows": 0,
    }

    first_row_by_id = {}

    try:
        with (
            temporary_path.open("w", encoding="utf-8") as output_file,
            IssueReporter(rejected_path) as reporter,
        ):
            for row_index, sample in enumerate(samples, 1):
                stats["inspected_rows"] += 1

                try:
                    normalized = normalize_llava_sample(
                        sample,
                        image_root=image_root,
                        image_prefix=args.image_prefix,
                        source_name=args.source_name,
                        decode_image=not args.skip_image_decode,
                    )

                    if normalized is None:
                        stats["prefix_excluded_rows"] += 1
                        continue

                    stats["candidate_rows"] += 1
                    sample_id = normalized["id"]

                    if sample_id in first_row_by_id:
                        raise SampleRejected(
                            IssueCode.RAW_DUPLICATE_ID,
                            f"Duplicate sample id: {sample_id}",
                            context={
                                "first_source_row": first_row_by_id[
                                    sample_id
                                ],
                            },
                        )

                    first_row_by_id[sample_id] = row_index

                    if normalized["_dataset_source"] == "unknown":
                        stats["unknown_source_rows"] += 1

                    output_file.write(
                        json.dumps(normalized, ensure_ascii=False) + "\n"
                    )
                    reporter.record_accepted()
                    stats["written_rows"] += 1

                except SampleRejected as exc:
                    raw_id = None
                    raw_image = None

                    if isinstance(sample, dict):
                        if sample.get("id") is not None:
                            raw_id = str(sample.get("id"))
                        raw_image = sample.get("image")

                    context = dict(exc.context)
                    if raw_image is not None:
                        context.setdefault("image", raw_image)

                    reporter.record(
                        PipelineIssue(
                            stage="raw_preflight",
                            error_code=exc.error_code,
                            message=str(exc),
                            retryable=False,
                            action="rejected",
                            sample_id=raw_id,
                            source_file=str(input_path),
                            source_row=row_index,
                            context=context or None,
                        )
                    )

                except Exception as exc:
                    reporter.record(
                        PipelineIssue(
                            stage="raw_preflight",
                            error_code=IssueCode.UNEXPECTED_ERROR,
                            message=str(exc),
                            retryable=False,
                            action="rejected",
                            source_file=str(input_path),
                            source_row=row_index,
                            context={
                                "exception_type": type(exc).__name__,
                            },
                        )
                    )

                if (
                    args.num_samples is not None
                    and stats["written_rows"] >= args.num_samples
                ):
                    break

            summary = reporter.summary(
                input_rows=stats["inspected_rows"]
            )

        summary.update(stats)
        summary.update({
            "stage": "raw_preflight",
            "status": "complete",
            "input_path": str(input_path),
            "image_root": str(image_root),
            "output_path": str(output_path),
            "rejected_path": str(rejected_path),
            "source_name": args.source_name,
            "image_decode_checked": not args.skip_image_decode,
        })

        evaluated_rows = (
            summary["accepted_rows"]
            + summary["rejected_rows"]
            + summary["retryable_rows"]
        )
        rejected_rows = (
            summary["rejected_rows"]
            + summary["retryable_rows"]
        )
        rejected_ratio = (
            rejected_rows / evaluated_rows
            if evaluated_rows
            else 0.0
        )
        summary["rejected_ratio"] = rejected_ratio

        threshold_exceeded = (
            args.max_rejected_ratio is not None
            and rejected_ratio > args.max_rejected_ratio
        )

        if (
            args.num_samples is not None
            and stats["written_rows"] != args.num_samples
        ):
            summary["status"] = "failed"
            summary["failure_reason"] = (
                "requested valid sample count was not reached"
            )
            write_json_atomic(summary_path, summary)
            raise RuntimeError(
                f"Requested {args.num_samples} valid samples, "
                f"but produced {stats['written_rows']}"
            )

        if threshold_exceeded:
            summary["status"] = "failed"
            summary["failure_reason"] = (
                "maximum rejected ratio exceeded"
            )
            write_json_atomic(summary_path, summary)
            raise RuntimeError(
                f"Rejected ratio {rejected_ratio:.2%} exceeds "
                f"limit {args.max_rejected_ratio:.2%}"
            )

        os.replace(temporary_path, output_path)
        write_json_atomic(summary_path, summary)

    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print("Input:", input_path)
    print("Image root:", image_root)
    print("Output:", output_path)
    print("Rejected:", rejected_path)
    print("Summary:", summary_path)

    for key in (
        "input_samples_total",
        "inspected_rows",
        "prefix_excluded_rows",
        "candidate_rows",
        "written_rows",
        "unknown_source_rows",
    ):
        print(f"{key}: {stats[key]}")

    print(f"rejected_ratio: {summary['rejected_ratio']:.2%}")
    print("issues_by_code:", summary["issues_by_code"])
    print("LLaVA VLM raw preflight: PASS")


if __name__ == "__main__":
    main()
