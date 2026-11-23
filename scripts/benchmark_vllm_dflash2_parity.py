#!/usr/bin/env python3
"""Measure DFlash2 acceptance and deterministic output parity with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def messages_from_row(row: dict) -> list[dict[str, str]]:
    turns = row.get("conversations") or row.get("messages") or []
    if len(turns) < 2:
        raise ValueError("expected a conversation ending with an assistant turn")
    role_map = {"human": "user", "gpt": "assistant"}
    result = []
    for turn in turns[:-1]:
        role = role_map.get(turn.get("role"), turn.get("role"))
        content = turn.get("content", turn.get("value", ""))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role!r}")
        result.append({"role": role, "content": str(content)})
    return result


def load_prompt_ids(path: str, tokenizer, limit: int) -> list[list[int]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) != limit:
        raise ValueError(f"requested {limit} rows, found {len(rows)}")
    return [
        tokenizer.apply_chat_template(
            messages_from_row(row),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        for row in rows
    ]


def metric_snapshot(llm: LLM) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for metric in llm.get_metrics():
        if hasattr(metric, "values"):
            snapshot[metric.name] = [float(value) for value in metric.values]
        elif hasattr(metric, "value"):
            snapshot[metric.name] = float(metric.value)
    return snapshot


def delta(after: object, before: object | None) -> object:
    if isinstance(after, list):
        old = before if isinstance(before, list) else []
        return [
            value - (old[index] if index < len(old) else 0.0)
            for index, value in enumerate(after)
        ]
    return float(after) - (float(before) if before is not None else 0.0)


def metric_delta(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    return {name: delta(value, before.get(name)) for name, value in after.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8194)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.target, trust_remote_code=True, local_files_only=True
    )
    datasets = {
        path: load_prompt_ids(path, tokenizer, args.limit) for path in args.dataset
    }
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
    )
    sampling = SamplingParams(
        temperature=0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=42,
    )

    reports = []
    for path, prompt_ids in datasets.items():
        before = metric_snapshot(llm)
        started = time.perf_counter()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], sampling
        )
        elapsed = time.perf_counter() - started
        after = metric_snapshot(llm)
        metrics = metric_delta(before, after)
        drafts = float(metrics.get("vllm:spec_decode_num_drafts", 0.0))
        accepted = float(
            metrics.get("vllm:spec_decode_num_accepted_tokens", 0.0)
        )
        proposed = float(metrics.get("vllm:spec_decode_num_draft_tokens", 0.0))
        per_pos = metrics.get("vllm:spec_decode_num_accepted_tokens_per_pos", [])
        token_rows = [list(output.outputs[0].token_ids) for output in outputs]
        reports.append(
            {
                "dataset": path,
                "samples": len(outputs),
                "max_tokens": args.max_tokens,
                "num_speculative_tokens": args.num_speculative_tokens,
                "wall_seconds": elapsed,
                "total_output_tokens": sum(map(len, token_rows)),
                "output_tokens_per_second": sum(map(len, token_rows)) / elapsed,
                "num_drafts": drafts,
                "num_draft_tokens": proposed,
                "num_accepted_tokens": accepted,
                "verification_weighted_acceptance_length": (
                    1.0 + accepted / drafts if drafts else None
                ),
                "draft_token_accept_rate": (
                    accepted / proposed if proposed else None
                ),
                "acceptance_survival_per_position": (
                    [float(value) / drafts for value in per_pos]
                    if drafts and isinstance(per_pos, list)
                    else []
                ),
                "output_token_sha256": [
                    hashlib.sha256(
                        json.dumps(tokens, separators=(",", ":")).encode()
                    ).hexdigest()
                    for tokens in token_rows
                ],
            }
        )
    payload = {
        "target": args.target,
        "draft": args.draft,
        "reports": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
