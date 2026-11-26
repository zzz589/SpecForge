#!/usr/bin/env python3
"""Adapt a SpecForge DFlash2 draft checkpoint to a Qwen3-VL target.

The draft tensors are copied byte-for-byte. Only target-owned configuration is
updated: language depth/token ids and Qwen3-VL RoPE parameters. The output
report records hashes so the conversion is auditable and reversible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target_model).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")

    source_config_path = source / "config.json"
    target_config_path = target / "config.json"
    source_weights = source / "model.safetensors"
    for required in (source_config_path, target_config_path, source_weights):
        if not required.is_file():
            raise FileNotFoundError(required)

    draft_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    target_config = json.loads(target_config_path.read_text(encoding="utf-8"))
    text_config = target_config.get("text_config", target_config)
    if draft_config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("source must be a converted DFlash2DraftModel checkpoint")

    rope_keys = (
        "max_position_embeddings",
        "rope_parameters",
        "rope_scaling",
        "rope_theta",
    )
    before = {key: draft_config.get(key) for key in rope_keys}
    for key in rope_keys:
        if key in text_config:
            draft_config[key] = text_config[key]
        else:
            draft_config.pop(key, None)
    draft_config["num_target_layers"] = int(text_config["num_hidden_layers"])
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in text_config:
            draft_config[key] = text_config[key]

    output.mkdir(parents=True)
    output_config_path = output / "config.json"
    output_config_path.write_text(
        json.dumps(draft_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_weights = output / source_weights.name
    shutil.copy2(source_weights, output_weights)
    for path in source.iterdir():
        if path.name in {"config.json", source_weights.name}:
            continue
        destination = output / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)

    source_hash = sha256(source_weights)
    output_hash = sha256(output_weights)
    report = {
        "conversion": "SpecForge DFlash2 text target -> Qwen3-VL target config",
        "source": str(source),
        "target_model": str(target),
        "output": str(output),
        "source_weights_sha256": source_hash,
        "output_weights_sha256": output_hash,
        "draft_tensors_bitwise_unchanged": source_hash == output_hash,
        "rope_before": before,
        "rope_after": {key: draft_config.get(key) for key in rope_keys},
        "num_target_layers": draft_config["num_target_layers"],
    }
    (output / "vl_adaptation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
