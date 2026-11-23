#!/usr/bin/env python3
"""Stream the exact SpecForge VLM prompt builder without retaining base64 payloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean


def percentile(values: list[int], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(values[lower])
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def describe(values: list[int]) -> dict:
    return {
        "count": len(values),
        "min": min(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from transformers import AutoProcessor, AutoTokenizer
    from specforge.data.template import TEMPLATE_REGISTRY
    from specforge.data.vlm_preprocessing import _prepare_one_record

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    template = TEMPLATE_REGISTRY.get("qwen")
    pad_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    merge_size = int(processor.image_processor.merge_size)

    prepared_count = 0
    skipped_count = 0
    input_lengths: list[int] = []
    supervised_tokens: list[int] = []
    image_tokens: list[int] = []
    with open(args.data, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            prepared = _prepare_one_record(
                record,
                source=f"{args.data}:{line_number}",
                tokenizer=tokenizer,
                processor=processor,
                template=template,
                pad_token_id=pad_token_id,
                merge_size=merge_size,
                max_length=args.max_length,
                min_loss_tokens=2,
            )
            if prepared is None:
                skipped_count += 1
                continue
            payload = prepared["payload"]
            input_lengths.append(len(payload["input_ids"]))
            supervised_tokens.append(sum(payload["loss_mask"]))
            image_tokens.append(
                len(payload["input_ids"]) - len(payload["request_input_ids"])
            )
            prepared_count += 1
            if line_number % 500 == 0:
                print(f"preflight {line_number} rows", flush=True)

    report = {
        "status": "pass" if skipped_count == 0 else "fail",
        "data": str(Path(args.data).resolve()),
        "max_length": args.max_length,
        "prepared": prepared_count,
        "skipped": skipped_count,
        "input_length": describe(input_lengths),
        "supervised_tokens": describe(supervised_tokens),
        "expanded_image_tokens": describe(image_tokens),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2))
    if skipped_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
