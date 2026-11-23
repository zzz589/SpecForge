#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys


NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?"


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def decimal_or_none(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def numeric_values(text):
    return [
        Decimal(match.replace(",", ""))
        for match in re.findall(NUMBER, str(text or ""))
    ]


def extract_final_values(text, expected_count):
    """Prefer explicit final-answer regions, then boxed/bold values, then a fallback."""
    text = str(text or "")
    final_regions = re.findall(
        r"(?is)(?:final\s+answer|answer|therefore|thus|hence)\s*[:：-]?(.{0,350})",
        text,
    )
    if final_regions:
        values = numeric_values(final_regions[-1])
        if len(values) >= expected_count:
            return values[-expected_count:], "final_answer_region", "high"
    boxed = numeric_values(" ".join(re.findall(r"\\boxed\{([^}]*)\}", text)))
    if len(boxed) >= expected_count:
        return boxed[-expected_count:], "boxed", "high"
    bold = numeric_values(" ".join(re.findall(r"\*\*([^*]+)\*\*", text)))
    if len(bold) >= expected_count:
        return bold[-expected_count:], "bold", "medium"
    values = numeric_values(text)
    if len(values) >= expected_count:
        return values[-expected_count:], "last_numbers_fallback", "low"
    return values, "insufficient_numeric_values", "low"


def audit_semantic_result(row):
    expected = numeric_values(row.get("reference_answer"))
    if not expected:
        return {
            "correct": False,
            "expected_values": [],
            "predicted_values": [],
            "parse_method": "reference_parse_failure",
            "parse_confidence": "low",
        }
    predicted, method, confidence = extract_final_values(
        row.get("target_output", ""), len(expected)
    )
    correct = len(predicted) == len(expected) and predicted == expected
    return {
        "correct": correct,
        "expected_values": [str(value) for value in expected],
        "predicted_values": [str(value) for value in predicted],
        "parse_method": method,
        "parse_confidence": confidence,
    }


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("metadata", {}).get(key, "unknown"))].append(row)
    result = {}
    for name, items in groups.items():
        completed = [item for item in items if "error" not in item]
        acceptance = [
            item.get("speculation", {}).get("average_acceptance_length")
            for item in completed
        ]
        acceptance = [value for value in acceptance if value is not None]
        result[name] = {
            "count": len(items),
            "accuracy": (
                sum(bool(item.get("correct")) for item in completed) / len(completed)
                if completed else None
            ),
            "mean_acceptance_length": statistics.fmean(acceptance) if acceptance else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--training-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--doc-script", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda item: item["dataset_index"])
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError("duplicate IDs in evaluation shards")

    completed = [row for row in rows if "error" not in row]
    for row in completed:
        row["benchmark_correct"] = row.get("correct")
        semantic = audit_semantic_result(row)
        row["semantic_evaluation"] = semantic
        row["correct"] = semantic["correct"]
        row["predicted_answer"] = "\n".join(semantic["predicted_values"])
    acceptance = [
        row["speculation"]["average_acceptance_length"]
        for row in completed
        if row.get("speculation", {}).get("average_acceptance_length") is not None
    ]
    latencies = [row["latency_s"] for row in completed]
    p10_acceptance = percentile(acceptance, 0.10)
    p90_latency = percentile(latencies, 0.90)

    category_counts = Counter()
    badcases = []
    for row in rows:
        labels = []
        domains = set()
        score = 0.0
        if "error" in row:
            labels.append("runtime_error")
            domains.add("system")
            score += 100
        else:
            if not row.get("predicted_answer"):
                labels.append("answer_parse_failure")
                domains.add("semantic")
                score += 40
            if not row.get("correct"):
                labels.append("semantic_incorrect")
                domains.add("semantic")
                score += 30
            if row.get("finish_reason") == "length":
                labels.append("output_truncated")
                domains.add("semantic")
                score += 3
            sample_aal = row.get("speculation", {}).get("average_acceptance_length")
            if sample_aal is not None and p10_acceptance is not None and sample_aal <= p10_acceptance:
                labels.append("low_acceptance_bottom10")
                domains.add("speculation")
                score += max(0.0, 20.0 - 5.0 * sample_aal)
            if sample_aal is not None and sample_aal < 2.0:
                labels.append("very_low_acceptance")
                domains.add("speculation")
                score += 20
            if p90_latency is not None and row.get("latency_s", 0) >= p90_latency:
                labels.append("slow_case_top10")
                domains.add("system")
                score += 5
        for label in labels:
            category_counts[label] += 1
        row["badcase"] = {
            "is_badcase": bool(labels),
            "labels": labels,
            "domains": sorted(domains),
            "priority_score": score,
            "review_status": "new",
        }
        if labels:
            badcases.append(row)
    badcases.sort(key=lambda item: (-item["badcase"]["priority_score"], item["id"]))

    drafts = sum(row.get("speculation", {}).get("num_drafts", 0) for row in completed)
    accepted = sum(row.get("speculation", {}).get("accepted_tokens", 0) for row in completed)
    proposed = sum(row.get("speculation", {}).get("proposed_tokens", 0) for row in completed)
    summary = {
        "schema_version": 1,
        "status": "complete" if len(rows) == 2000 else "partial",
        "metrics": {
            "requested": len(rows),
            "completed": len(completed),
            "errors": len(rows) - len(completed),
            "correct": sum(bool(row.get("correct")) for row in completed),
            "accuracy": sum(bool(row.get("correct")) for row in completed) / len(completed) if completed else None,
            "average_acceptance_length": 1.0 + accepted / drafts if drafts else None,
            "draft_token_accept_rate": accepted / proposed if proposed else None,
            "sample_acceptance_p10": p10_acceptance,
            "sample_acceptance_median": statistics.median(acceptance) if acceptance else None,
            "sample_acceptance_p90": percentile(acceptance, 0.90),
            "latency_mean_s": statistics.fmean(latencies) if latencies else None,
            "latency_p90_s": p90_latency,
        },
        "badcase_count": len(badcases),
        "badcase_categories": dict(sorted(category_counts.items())),
        "grouped_metrics": {
            "meter_type": grouped(completed, "meter_type"),
            "environment_conditions": grouped(completed, "environment_conditions"),
            "range": grouped(completed, "range"),
        },
        "interpretation": {
            "semantic_incorrect": "Target-model task error; not a direct DFlash2 quality metric.",
            "low_acceptance_bottom10": "DFlash2 acceleration weakness; prioritize for data analysis.",
            "runtime_error": "Engineering failure; exclude from model-quality denominator.",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    badcases_path = output_dir / "badcases.jsonl"
    summary_path = output_dir / "summary.json"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with badcases_path.open("w", encoding="utf-8") as handle:
        for row in badcases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    subprocess.run([
        sys.executable,
        args.doc_script,
        "--checkpoint-dir", args.checkpoint_dir,
        "--experiment-dir", args.experiment_dir,
        "--training-size", str(args.training_size),
        "--steps", str(args.steps),
        "--evaluation-summary", str(summary_path),
        "--results-jsonl", str(results_path),
        "--badcases-jsonl", str(badcases_path),
    ], check=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
