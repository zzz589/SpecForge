#!/usr/bin/env python3
"""Create the machine-readable manifest and human experiment card."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re


INIT = "/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/experiments_results/qwen3-4b-redhatai-dflash2-specforge-b9-20261105/Qwen3-4B-speculator.dflash2-specforge-b9-qwen3vl-mrope-v1"
TARGET = "/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/models/Qwen3-VL-4B-Instruct"
TEST = "/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/data/RPM-10K/processed-dflash2-v1/train-ready/test/rpm10k-test-ready-2000.jsonl"


def log_times(path: Path):
    found = []
    if path.is_file():
        pattern = re.compile(r"2026-\d\d-\d\d \d\d:\d\d:\d\d")
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    found.append(match.group(0))
    if found:
        return found[0], found[-1]
    if path.exists():
        timestamp = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
        return None, timestamp
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--training-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--evaluation-summary")
    parser.add_argument("--results-jsonl")
    parser.add_argument("--badcases-jsonl")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_dir).resolve()
    experiment = Path(args.experiment_dir).resolve()
    summary = None
    if args.evaluation_summary and Path(args.evaluation_summary).is_file():
        summary = json.loads(Path(args.evaluation_summary).read_text(encoding="utf-8"))
    started, ended = log_times(experiment / "train.log")
    run_id = checkpoint.name.rsplit("-step", 1)[0]
    manifest = {
        "schema_version": 1,
        "experiment_id": run_id,
        "status": "evaluated" if summary else "trained_pending_evaluation",
        "training": {
            "dataset": "RPM-10K",
            "training_size": args.training_size,
            "nested_subset_seed": 20261109,
            "init_checkpoint": INIT,
            "independent_from_other_scales": True,
            "target_model": TARGET,
            "method": "DFlash2",
            "modality": "multimodal",
            "block_size": 9,
            "num_epochs": 3,
            "optimizer_steps": args.steps,
            "per_rank_batch_size": 1,
            "trainer_dp_size": 2,
            "gradient_accumulation": 2,
            "global_batch_size": 4,
            "learning_rate": 1e-5,
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.03,
            "num_anchors": 32,
            "max_length": 4096,
            "seed": 20261109,
            "capture_gpu": 0,
            "trainer_gpus": [1, 7],
            "started_at": started,
            "ended_at": ended,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "global_step": args.steps,
            "exported_draft": str(checkpoint / "exported-draft"),
        },
        "evaluation": {
            "status": "complete" if summary else "pending",
            "test_dataset": TEST,
            "num_speculative_tokens": 8,
            "max_model_len": 4096,
            "decoding": "greedy",
            "runtime_note": (
                "vLLM 0.30.0 feeds images to the Qwen3-VL target, but reports that "
                "DFlash2Qwen3ForCausalLM does not accept external multimodal embeddings; "
                "the draft side uses the runtime's text-only draft-input fallback."
            ),
            "summary": summary,
            "results_jsonl": args.results_jsonl,
            "badcases_jsonl": args.badcases_jsonl,
        },
    }
    (checkpoint / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {run_id}",
        "",
        "## 结论状态",
        "",
        f"- 状态：{'训练和测试完成' if summary else '训练完成，测试待执行'}",
        f"- Checkpoint：`{checkpoint}`",
        "- 所有数据规模实验均独立从同一个公开 DFlash2-VL mRoPE init 开始；不是逐级续训。",
        "",
        "## 训练配置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 训练数据量 | {args.training_size} |",
        "| Epochs | 3 |",
        f"| Optimizer steps | {args.steps} |",
        "| Global batch | 4（2 trainer × batch 1 × accumulation 2） |",
        "| Learning rate | 1e-5 |",
        "| Scheduler | cosine |",
        "| Warmup ratio | 0.03 |",
        "| Anchors | 32 |",
        "| DFlash2 block size | 9 |",
        "| Capture / Trainer GPU | 0 / 1,7 |",
        f"| 开始时间 | {started or '日志未记录'} |",
        f"| 结束时间 | {ended or '日志未记录'} |",
        f"| Init checkpoint | `{INIT}` |",
        f"| Target model | `{TARGET}` |",
        "",
        "## 测试结果",
        "",
        "> 评测给 Qwen3-VL target 输入真实图片；当前 vLLM 0.30.0 同时警告 "
        "DFlash2 draft 不接收 external multimodal embeddings，draft 输入采用运行时的 "
        "text-only fallback。接受长度代表当前实际部署链路，不能解释成图像 embedding "
        "被直接送入 draft。",
        "",
    ]
    if summary:
        metrics = summary.get("metrics", summary)
        lines.extend([
            "| 指标 | 值 |",
            "|---|---:|",
            f"| 测试样本 | {metrics.get('completed')} / {metrics.get('requested')} |",
            f"| 数值答案准确率 | {metrics.get('accuracy')} |",
            f"| 平均接受长度 | {metrics.get('average_acceptance_length')} |",
            f"| Draft token 接受率 | {metrics.get('draft_token_accept_rate')} |",
            f"| 平均延迟（秒） | {metrics.get('latency_mean_s')} |",
            f"| Badcase 数量 | {summary.get('badcase_count')} |",
        ])
    else:
        lines.append("待在 RPM-10K 独立测试集 2,000 条上评测。")
    lines.extend([
        "",
        "## Badcase 使用原则",
        "",
        "- `semantic_incorrect`：目标模型读表错误，反映目标模型/标注/提示词问题，不直接代表 DFlash2 失败。",
        "- `low_acceptance_bottom10`：目标答案可以正确，但 DFlash2 接受长度低，是增加类似训练样本时最重要的样本。",
        "- `answer_parse_failure`：优先检查输出格式和解析器，不应直接当成模型读错。",
        "- `runtime_error`：工程链路问题，必须和模型效果分开统计。",
        "- `output_truncated`：生成达到 max_tokens，分析语义准确率时需谨慎。",
        "",
        "详细逐样本结果与 badcase 位于 `evaluation/rpm10k-test2000/`。",
    ])
    (checkpoint / "EXPERIMENT_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
