#!/usr/bin/env python3
"""Evaluate RPM-10K through a running SGLang DFlash2-VL server."""

from __future__ import annotations

import argparse
import base64
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import statistics
import time

import requests


NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?"


def data_uri(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if mime is None:
        raise ValueError(f"unsupported image suffix: {path}")
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def parse_decimal(value):
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def extract_reading(text: str):
    patterns = [
        rf"\\boxed\{{\s*({NUMBER})\s*\}}",
        rf"(?:final\s+answer|answer|reading|meter\s+reads|value)\s*(?:is|=|:)?\s*({NUMBER})",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).replace(",", "")
    matches = re.findall(NUMBER, text)
    return matches[-1].replace(",", "") if matches else None


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--training-size", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    acceptance_lengths = []
    accepted_total = proposed_total = drafts_total = completion_tokens = correct = completed = 0
    started_all = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as output:
        for local_index, row in enumerate(rows, 1):
            global_index = args.start_index + local_index - 1
            question = next(
                item["content"] for item in row["conversations"] if item["role"] == "user"
            ).replace("<image>\n", "").replace("<image>", "").strip()
            body = {
                "model": args.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri(row["image"])}},
                        {"type": "text", "text": question},
                    ],
                }],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
                "return_meta_info": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.perf_counter()
            try:
                response = requests.post(endpoint, json=body, timeout=args.timeout)
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                text = choice["message"].get("content") or ""
                meta = choice.get("meta_info") or payload.get("meta_info") or {}
                usage = payload.get("usage") or {}
                verify_count = float(meta.get("spec_verify_ct") or 0)
                acceptance_length = meta.get("spec_accept_length")
                acceptance_length = float(acceptance_length) if acceptance_length is not None else None
                accepted = max(0.0, acceptance_length - 1.0) * verify_count if acceptance_length is not None else 0.0
                proposed = verify_count * args.num_speculative_tokens
                predicted = extract_reading(text)
                expected_decimal = parse_decimal(row.get("reference_answer"))
                predicted_decimal = parse_decimal(predicted)
                is_correct = (
                    predicted_decimal == expected_decimal
                    if predicted_decimal is not None and expected_decimal is not None
                    else False
                )
                absolute_error = (
                    float(abs(predicted_decimal - expected_decimal))
                    if predicted_decimal is not None and expected_decimal is not None
                    else None
                )
                completion = int(usage.get("completion_tokens") or meta.get("completion_tokens") or 0)
                record = {
                    "schema_version": 1,
                    "id": row["id"],
                    "dataset_index": global_index,
                    "training_size": args.training_size,
                    "image": row["image"],
                    "question": question,
                    "reference_answer": row.get("reference_answer"),
                    "predicted_answer": predicted,
                    "correct": is_correct,
                    "absolute_error": absolute_error,
                    "target_output": text,
                    "finish_reason": choice.get("finish_reason"),
                    "latency_s": elapsed,
                    "completion_tokens": completion,
                    "speculation": {
                        "num_drafts": verify_count,
                        "proposed_tokens": proposed,
                        "accepted_tokens": accepted,
                        "average_acceptance_length": acceptance_length,
                        "draft_token_accept_rate": accepted / proposed if proposed else None,
                    },
                    "meta_info": meta,
                    "metadata": row.get("rpm10k_metadata", {}),
                }
                completed += 1
                correct += int(is_correct)
                completion_tokens += completion
                drafts_total += verify_count
                accepted_total += accepted
                proposed_total += proposed
                latencies.append(elapsed)
                if acceptance_length is not None:
                    acceptance_lengths.append(acceptance_length)
                print(
                    f"[{local_index}/{len(rows)}] {row['id']} correct={is_correct} "
                    f"aal={acceptance_length} latency={elapsed:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                record = {
                    "schema_version": 1,
                    "id": row.get("id"),
                    "dataset_index": global_index,
                    "training_size": args.training_size,
                    "image": row.get("image"),
                    "question": question,
                    "reference_answer": row.get("reference_answer"),
                    "metadata": row.get("rpm10k_metadata", {}),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[{local_index}/{len(rows)}] ERROR {record['error']}", flush=True)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    elapsed_all = time.perf_counter() - started_all
    summary = {
        "schema_version": 1,
        "runtime": "SGLang DFlash2-VL selector",
        "dataset": str(Path(args.data).resolve()),
        "server": args.base_url,
        "model": args.model,
        "training_size": args.training_size,
        "start_index": args.start_index,
        "requested": len(rows),
        "completed": completed,
        "errors": len(rows) - completed,
        "correct": correct,
        "accuracy": correct / completed if completed else None,
        "average_acceptance_length": 1.0 + accepted_total / drafts_total if drafts_total else None,
        "draft_token_accept_rate": accepted_total / proposed_total if proposed_total else None,
        "sample_acceptance_median": statistics.median(acceptance_lengths) if acceptance_lengths else None,
        "sample_acceptance_p10": percentile(acceptance_lengths, 0.10),
        "sample_acceptance_p90": percentile(acceptance_lengths, 0.90),
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed_all,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p90_s": percentile(latencies, 0.90),
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
