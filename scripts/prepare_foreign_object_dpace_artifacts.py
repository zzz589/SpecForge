#!/usr/bin/env python3
import argparse, collections, datetime, hashlib, json, os, pathlib, platform, subprocess
import yaml

ROOT = "/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj"
INIT = ROOT + "/experiments_results/qwen3-4b-redhatai-dflash2-specforge-b9-20261105/Qwen3-4B-speculator.dflash2-specforge-b9-qwen3vl-mrope-v1"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    for name in ("size", "steps"):
        ap.add_argument("--" + name, type=int, required=True)
    for name in ("exp", "data", "base-config", "repo", "env-root"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--status", choices=["prepared", "complete"], required=True)
    a = ap.parse_args()
    exp = pathlib.Path(a.exp)
    for d in ("config", "commands", "provenance", "training_curves", "validation", "evaluation", "runtime_audit"):
        (exp / d).mkdir(parents=True, exist_ok=True)
    data = pathlib.Path(a.data)
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if len(rows) != a.size:
        raise ValueError(f"expected {a.size} rows, got {len(rows)}")
    sources = collections.Counter(str(r.get("_foreign_source", "unknown")) for r in rows)
    labels = collections.Counter(label for r in rows for label in r.get("foreign_object_metadata", {}).get("labels", []))
    answers = [str(r.get("conversations", [{}, {}])[-1].get("content", r.get("reference_answer", ""))) for r in rows]
    manifest = {
        "schema_version": 1, "dataset": "Foreign Object Detection", "split": "train",
        "rows": len(rows), "images_present": sum(bool(r.get("image")) for r in rows),
        "nested_subset_seed": 20261116, "nested_subsets": "1000 ⊂ 2000 ⊂ ... ⊂ 8000",
        "data_path": str(data), "sha256": sha(data),
        "id_sha256": hashlib.sha256(("\n".join(str(r.get("id")) for r in rows) + "\n").encode()).hexdigest(),
        "source_distribution": dict(sorted(sources.items())), "label_distribution_multilabel": dict(sorted(labels.items())),
        "answer_chars": {"min": min(map(len, answers)), "max": max(map(len, answers)), "mean": sum(map(len, answers)) / len(answers)},
        "filtering": "Rail official train only; exact SHA256 dedup; _aug_3_1 near-duplicate variants removed; invalid/empty boxes removed; minimum edge enforced; AIGC capped at 1600; Rail official validation excluded."
    }
    (exp / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 数据分布说明", "", f"- 实际训练行数：{a.size}", f"- 数据文件：`{data}`", f"- 数据 SHA256：`{manifest['sha256']}`", "- 每行包含一张图、一个英文电力异物检查问题和 Qwen3-VL-4B 贪心重生成回答。", "- 子集固定种子 20261116，严格保持 1k ⊂ 2k ⊂ ... ⊂ 8k。", "- RailFOD23 官方验证集没有进入训练，留作后续独立评测。", "", "## 来源分布", "", "| 来源 | 数量 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(sources.items())]
    lines += ["", "## 异物标签分布（多标签，不可相加作为样本总数）", "", "| 标签 | 数量 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(labels.items())]
    (exp / "DATA_DISTRIBUTION.md").write_text("\n".join(lines) + "\n")
    base = pathlib.Path(a.base_config)
    (exp / "config" / "base.yaml").write_text(base.read_text())
    cfg = yaml.safe_load(base.read_text())
    run = f"qwen3-vl-4b-dflash2-dpace-foreign-object-n{a.size}-e3-lr1e-5-20261116"
    cfg["data"]["train_data_path"] = str(data)
    cfg["data"]["cache_dir"] = str(exp / "cache")
    cfg["training"]["max_steps"] = a.steps
    cfg["training"]["save_interval"] = a.steps
    cfg["run_id"] = run
    cfg["output_dir"] = str(exp / "output")
    cfg["deployment"]["disaggregated"]["control_dir"] = str(exp / "control")
    cfg["deployment"]["disaggregated"]["consumer_state_dir"] = str(exp / "consumer-state")
    (exp / "config" / "resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = f"cd {a.repo}\nexport CUDA_HOME={a.env_root}/lib/python3.11/site-packages/nvidia/cu13\nexport PATH={a.env_root}/bin:$CUDA_HOME/bin:$PATH\nexport LD_LIBRARY_PATH=$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${{LD_LIBRARY_PATH:-}}\n{a.env_root}/bin/python -m specforge.cli train -c {exp}/config/resolved.yaml --role auto\n"
    (exp / "commands" / "train.sh").write_text(cmd)
    os.chmod(exp / "commands" / "train.sh", 0o755)
    commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", a.repo, "status", "--short"], capture_output=True, text=True).stdout
    (exp / "provenance" / "code.txt").write_text(f"commit={commit}\n\n{status}")
    (exp / "provenance" / "environment.txt").write_text(f"generated_at={datetime.datetime.now().astimezone().isoformat()}\npython={platform.python_version()}\nenv={a.env_root}\ncapture_gpu=0\ntrainer_gpus=3,4\n")
    card = [f"# {run}", "", f"- 状态：{a.status}", f"- 数据量：{a.size}", f"- Optimizer steps：{a.steps}", "- Epochs：3", "- Global batch：4", "- LR：1e-5，cosine，warmup 0.03", "- 目标：DFlash2-VL + D-PACE（loss_type=dpace，dpace_alpha=0.5）", "- LK：lambda，kl_scale=0.1，kl_decay=0.0", "- Selector：alpha=1.0，stop_gradient=false", "- Anchors：32；DFlash2 block size：9", "- GPU：capture 0；trainer 3,4", "- Init：RedHatAI DFlash2 b9 Qwen3-VL mRoPE v1", "- 与其他规模独立训练，不从前一规模续训。", "", "数据详情见 `DATA_DISTRIBUTION.md` 与 `data_manifest.json`。"]
    (exp / "EXPERIMENT_CARD.md").write_text("\n".join(card) + "\n")
    expman = {"schema_version": 1, "experiment_id": run, "status": a.status, "training": {"size": a.size, "steps": a.steps, "epochs": 3, "global_batch": 4, "learning_rate": 1e-5, "loss_type": "dpace", "dpace_alpha": 0.5, "init_checkpoint": INIT, "independent_from_other_scales": True}, "data_manifest": str(exp / "data_manifest.json"), "resolved_config": str(exp / "config" / "resolved.yaml")}
    (exp / "experiment_manifest.json").write_text(json.dumps(expman, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    main()
