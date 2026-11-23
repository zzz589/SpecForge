#!/usr/bin/env python3
"""Evaluate DFlash2 on the single-image MMMU JSONL with vLLM."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import vllm.model_executor.layers.logits_processor as vllm_logits_processor


# The cluster's FlashInfer wheel tries to JIT-compile top-k with CUDA headers
# newer than the PyTorch build.  DFlash2 only needs ordinary top-k here, so use
# PyTorch's implementation without changing the shared vLLM environment.
if hasattr(vllm_logits_processor, "_flashinfer_topk"):
    vllm_logits_processor._flashinfer_topk.cache_clear()
    vllm_logits_processor._flashinfer_topk = lambda: None


def answer_letter(text: str):
    for pattern in (
        r"^\s*[\(\[]?([A-D])[\)\].,:;\s]",
        r"\b(?:answer|答案|option|选项)\s*(?:is|为|:|：)?\s*[\(\[]?([A-D])\b",
        r"\b([A-D])\b",
    ):
        found = re.search(pattern, text.upper())
        if found:
            return found.group(1)
    return None


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def metric_snapshot(llm: LLM) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for metric in llm.get_metrics():
        if hasattr(metric, "values"):
            snapshot[metric.name] = [float(value) for value in metric.values]
        elif hasattr(metric, "value"):
            snapshot[metric.name] = float(metric.value)
    return snapshot


def metric_delta(before: dict[str, object], after: dict[str, object]):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = rows[args.start_index :]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

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
    completion_tokens = 0
    correct = 0
    completed = 0
    before = metric_snapshot(llm)
    started_all = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            expected = row.get("answer")
            user_turn = next(item for item in row["conversations"] if item["role"] == "user")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_turn["content"]},
                ],
            }]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            started = time.perf_counter()
            try:
                with Image.open(row["image"]) as opened:
                    image = opened.convert("RGB")
                result = llm.generate(
                    {"prompt": prompt, "multi_modal_data": {"image": image}},
                    sampling,
                    use_tqdm=False,
                )[0]
                elapsed = time.perf_counter() - started
                text = result.outputs[0].text
                tokens = list(result.outputs[0].token_ids)
                predicted = answer_letter(text)
                is_correct = predicted == expected
                completed += 1
                correct += int(is_correct)
                completion_tokens += len(tokens)
                latencies.append(elapsed)
                record = {
                    "id": row["id"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                    "text": text,
                    "latency_s": elapsed,
                    "completion_tokens": len(tokens),
                }
                print(
                    f"[{index}/{len(rows)}] {row['id']} correct={is_correct} "
                    f"latency={elapsed:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                record = {
                    "id": row.get("id"),
                    "expected": expected,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[{index}/{len(rows)}] ERROR {record['error']}", flush=True)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    elapsed_all = time.perf_counter() - started_all
    metrics = metric_delta(before, metric_snapshot(llm))
    drafts = float(metrics.get("vllm:spec_decode_num_drafts", 0.0))
    accepted = float(metrics.get("vllm:spec_decode_num_accepted_tokens", 0.0))
    proposed = float(metrics.get("vllm:spec_decode_num_draft_tokens", 0.0))
    per_pos = metrics.get("vllm:spec_decode_num_accepted_tokens_per_pos", [])
    summary = {
        "dataset": str(Path(args.data).resolve()),
        "target": args.target,
        "draft": args.draft,
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
        "output_throughput_tok_s": completion_tokens / sum(latencies) if latencies else None,
        "num_drafts": drafts,
        "num_draft_tokens": proposed,
        "num_accepted_tokens": accepted,
        "average_acceptance_length": 1.0 + accepted / drafts if drafts else None,
        "draft_token_accept_rate": accepted / proposed if proposed else None,
        "acceptance_survival_per_position": (
            [float(value) / drafts for value in per_pos]
            if drafts and isinstance(per_pos, list)
            else []
        ),
    }
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
