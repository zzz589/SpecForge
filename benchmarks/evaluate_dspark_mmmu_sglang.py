#!/usr/bin/env python3
"""Sequential MMMU single-image evaluation against an SGLang chat server."""
import argparse
import base64
import json
import re
import statistics
import time
from pathlib import Path

import requests


def data_uri(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if mime is None:
        raise ValueError(f"unsupported image suffix: {path}")
    return "data:" + mime + ";base64," + base64.b64encode(
        Path(path).read_bytes()
    ).decode("ascii")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
            if line.strip()]
    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latencies, correct, completed = [], 0, 0
    completion_tokens, verify_count = 0, 0
    acceptance_lengths = []

    with output_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            user_turn = next(x for x in row["conversations"] if x["role"] == "user")
            expected = row.get("answer")
            body = {
                "model": args.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": data_uri(row["image"])}},
                        {"type": "text", "text": user_turn["content"]},
                    ],
                }],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
                "return_meta_info": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            start = time.perf_counter()
            try:
                response = requests.post(endpoint, json=body, timeout=args.timeout)
                elapsed = time.perf_counter() - start
                response.raise_for_status()
                payload = response.json()
                message = payload["choices"][0]["message"]
                text = message.get("content") or ""
                meta = (payload["choices"][0].get("meta_info") or payload.get("meta_info") or {})
                predicted = answer_letter(text)
                is_correct = predicted == expected
                completed += 1
                correct += int(is_correct)
                latencies.append(elapsed)
                completion_tokens += int(meta.get("completion_tokens") or 0)
                verify_count += int(meta.get("spec_verify_ct") or 0)
                if meta.get("spec_accept_length") is not None:
                    acceptance_lengths.append(float(meta["spec_accept_length"]))
                record = {"id": row["id"], "expected": expected, "predicted": predicted,
                          "correct": is_correct, "text": text, "latency_s": elapsed,
                          "meta_info": meta}
                print(f"[{index}/{len(rows)}] {row['id']} "
                      f"correct={is_correct} latency={elapsed:.3f}s", flush=True)
            except Exception as exc:
                record = {"id": row["id"], "expected": expected,
                          "error": f"{type(exc).__name__}: {exc}"}
                print(f"[{index}/{len(rows)}] {row['id']} ERROR: {exc}", flush=True)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    summary = {
        "dataset": str(Path(args.data).resolve()),
        "server": args.base_url,
        "model": args.model,
        "requested": len(rows),
        "completed": completed,
        "accuracy": correct / completed if completed else None,
        "correct": correct,
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p90": percentile(latencies, 0.9),
        },
        "completion_tokens": completion_tokens,
        "output_throughput_tok_s": completion_tokens / sum(latencies) if latencies else None,
        "average_acceptance_length": (
            statistics.fmean(acceptance_lengths) if acceptance_lengths else None),
        "spec_verify_count": verify_count or None,
    }
    Path(args.summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
