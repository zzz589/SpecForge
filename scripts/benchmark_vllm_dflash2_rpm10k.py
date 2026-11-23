#!/usr/bin/env python3
"""Evaluate DFlash2 acceptance and numeric meter-reading quality on RPM-10K."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import re
import statistics
import time
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import vllm.model_executor.layers.logits_processor as vllm_logits_processor


if hasattr(vllm_logits_processor, "_flashinfer_topk"):
    vllm_logits_processor._flashinfer_topk.cache_clear()
    vllm_logits_processor._flashinfer_topk = lambda: None


NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?"


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


def metric_snapshot(llm: LLM) -> dict[str, object]:
    snapshot = {}
    for metric in llm.get_metrics():
        if hasattr(metric, "values"):
            snapshot[metric.name] = [float(value) for value in metric.values]
        elif hasattr(metric, "value"):
            snapshot[metric.name] = float(metric.value)
    return snapshot


def metric_delta(before, after):
    result = {}
    for name, value in after.items():
        old = before.get(name)
        if isinstance(value, list):
            old_values = old if isinstance(old, list) else []
            result[name] = [
                item - (old_values[index] if index < len(old_values) else 0.0)
                for index, item in enumerate(value)
            ]
        else:
            result[name] = float(value) - (float(old) if old is not None else 0.0)
    return result


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
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--training-size", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = rows[args.start_index:]
    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    processor = AutoProcessor.from_pretrained(
        args.target, trust_remote_code=True, local_files_only=True
    )
    llm = LLM(
        model=args.target,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config={
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(
        temperature=0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=42,
    )

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
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            before = metric_snapshot(llm)
            started = time.perf_counter()
            try:
                with Image.open(row["image"]) as opened:
                    image = opened.convert("RGB")
                request = llm.generate(
                    {"prompt": prompt, "multi_modal_data": {"image": image}},
                    sampling,
                    use_tqdm=False,
                )[0]
                elapsed = time.perf_counter() - started
                generated = request.outputs[0]
                text = generated.text
                token_ids = list(generated.token_ids)
                metrics = metric_delta(before, metric_snapshot(llm))
                drafts = float(metrics.get("vllm:spec_decode_num_drafts", 0.0))
                accepted = float(metrics.get("vllm:spec_decode_num_accepted_tokens", 0.0))
                proposed = float(metrics.get("vllm:spec_decode_num_draft_tokens", 0.0))
                per_position = metrics.get("vllm:spec_decode_num_accepted_tokens_per_pos", [])
                acceptance_length = 1.0 + accepted / drafts if drafts else None
                accept_rate = accepted / proposed if proposed else None
                predicted = extract_reading(text)
                expected_decimal = parse_decimal(row["reference_answer"])
                predicted_decimal = parse_decimal(predicted)
                absolute_error = (
                    float(abs(predicted_decimal - expected_decimal))
                    if predicted_decimal is not None and expected_decimal is not None
                    else None
                )
                is_correct = (
                    predicted_decimal == expected_decimal
                    if predicted_decimal is not None and expected_decimal is not None
                    else False
                )
                record = {
                    "schema_version": 1,
                    "id": row["id"],
                    "dataset_index": global_index,
                    "training_size": args.training_size,
                    "image": row["image"],
                    "question": question,
                    "reference_answer": row["reference_answer"],
                    "predicted_answer": predicted,
                    "correct": is_correct,
                    "absolute_error": absolute_error,
                    "target_output": text,
                    "finish_reason": getattr(generated, "finish_reason", None),
                    "latency_s": elapsed,
                    "completion_tokens": len(token_ids),
                    "speculation": {
                        "num_drafts": drafts,
                        "proposed_tokens": proposed,
                        "accepted_tokens": accepted,
                        "average_acceptance_length": acceptance_length,
                        "draft_token_accept_rate": accept_rate,
                        "acceptance_survival_per_position": (
                            [float(value) / drafts for value in per_position]
                            if drafts and isinstance(per_position, list)
                            else []
                        ),
                    },
                    "metadata": row.get("rpm10k_metadata", {}),
                }
                completed += 1
                correct += int(is_correct)
                completion_tokens += len(token_ids)
                drafts_total += drafts
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
        "dataset": str(Path(args.data).resolve()),
        "target": args.target,
        "draft": args.draft,
        "training_size": args.training_size,
        "start_index": args.start_index,
        "requested": len(rows),
        "completed": completed,
        "correct": correct,
        "accuracy": correct / completed if completed else None,
        "max_tokens": args.max_tokens,
        "num_speculative_tokens": args.num_speculative_tokens,
        "wall_seconds": elapsed_all,
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p90": percentile(latencies, 0.9),
        },
        "completion_tokens": completion_tokens,
        "num_drafts": drafts_total,
        "num_draft_tokens": proposed_total,
        "num_accepted_tokens": accepted_total,
        "average_acceptance_length": 1.0 + accepted_total / drafts_total if drafts_total else None,
        "draft_token_accept_rate": accepted_total / proposed_total if proposed_total else None,
        "sample_acceptance_length": {
            "mean": statistics.fmean(acceptance_lengths) if acceptance_lengths else None,
            "median": statistics.median(acceptance_lengths) if acceptance_lengths else None,
            "p10": percentile(acceptance_lengths, 0.1),
            "p90": percentile(acceptance_lengths, 0.9),
        },
    }
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
