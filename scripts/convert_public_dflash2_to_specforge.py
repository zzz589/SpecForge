#!/usr/bin/env python3
"""Convert a Speculators DFlash2 release to the SpecForge/SGLang HF contract.

The source directory is read-only. Draft-owned tensors are copied exactly;
the two verifier-owned tensors serialized by Speculators are verified against
the target Qwen model and intentionally omitted because SpecForge/SGLang alias
the target embedding and LM head at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from specforge.modeling.auto import AutoDraftModel
from specforge.training.model_loading import (
    load_draft_config_source,
    warm_start_draft_model,
)


VERIFIER_OWNED = {"embed_tokens.weight", "lm_head.weight"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_manifest(path: Path) -> dict[str, dict[str, object]]:
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        return {
            key: {
                "shape": list(stream.get_slice(key).get_shape()),
                "dtype": str(stream.get_slice(key).get_dtype()),
            }
            for key in stream.keys()
        }


def load_target_embedding(target: Path) -> torch.Tensor:
    index_path = target / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index["weight_map"]["model.embed_tokens.weight"]
        with safe_open(
            str(target / shard_name), framework="pt", device="cpu"
        ) as stream:
            return stream.get_tensor("model.embed_tokens.weight")
    weights = target / "model.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as stream:
        return stream.get_tensor("model.embed_tokens.weight")


def translated_config(source_cfg: dict, target_cfg: dict) -> dict:
    if source_cfg.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("source is not a DFlash2DraftModel release")
    if source_cfg.get("speculators_model_type") != "dflash2":
        raise ValueError("source is not a Speculators dflash2 checkpoint")
    transformer = dict(source_cfg["transformer_layer_config"])
    block_size = int(source_cfg["block_size"])
    if source_cfg.get("sample_from_anchor") is not False:
        raise ValueError(
            "this converter currently requires sample_from_anchor=false"
        )
    speculators_aux_ids = list(source_cfg["aux_hidden_state_layer_ids"])
    if len(speculators_aux_ids) != int(transformer["num_hidden_layers"]):
        raise ValueError("target-layer count does not match draft-layer count")
    # Speculators records hidden-state tuple indices (embedding is index 0).
    # SpecForge/SGLang's public DFlash contract records zero-based transformer
    # layer ids, so this is the one semantic index translation in the config.
    target_layer_ids = [int(index) - 1 for index in speculators_aux_ids]
    if min(target_layer_ids) < 0:
        raise ValueError("invalid Speculators auxiliary hidden-state index")

    # Keep the released attention computation. Speculators' true
    # sliding_window_non_causal means a bidirectional local draft mask; in
    # SpecForge this is expressed by top-level is_causal=false.
    sliding_non_causal = bool(source_cfg.get("sliding_window_non_causal", False))
    config = transformer
    config.update(
        {
            "architectures": ["DFlash2DraftModel"],
            "block_size": block_size,
            "draft_vocab_size": int(source_cfg["draft_vocab_size"]),
            "dtype": source_cfg.get("dtype", "bfloat16"),
            "num_target_layers": int(target_cfg["num_hidden_layers"]),
            "eagle_aux_hidden_state_layer_ids": speculators_aux_ids,
            "sample_from_anchor": False,
            "sliding_window_non_causal": sliding_non_causal,
            "is_causal": False if sliding_non_causal else None,
            "bos_token_id": target_cfg.get("bos_token_id"),
            "eos_token_id": target_cfg.get("eos_token_id"),
            "pad_token_id": target_cfg.get("pad_token_id"),
            "dflash_config": {
                "block_size": block_size,
                "conv_group_size": int(source_cfg["conv_group_size"]),
                "conv_kernel_size": int(source_cfg["conv_kernel_size"]),
                "mask_token_id": int(source_cfg["mask_token_id"]),
                "sample_from_anchor": False,
                "causal": not sliding_non_causal,
                "selector_rank": int(source_cfg["selector_rank"]),
                "selector_top_k": int(source_cfg["selector_top_k"]),
                "target_layer_ids": target_layer_ids,
            },
        }
    )
    # Qwen3Config accepts either representation, but the current Transformers
    # version normalizes legacy rope_scaling into rope_parameters.
    if "rope_parameters" not in config and target_cfg.get("rope_scaling") is not None:
        config["rope_scaling"] = target_cfg["rope_scaling"]
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target_model).resolve()
    output = Path(args.output).resolve()
    source_weights = source / "model.safetensors"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)

    source_cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    target_cfg = json.loads((target / "config.json").read_text(encoding="utf-8"))
    output_cfg = translated_config(source_cfg, target_cfg)
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(output_cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_state = load_file(str(source_weights), device="cpu")
    verifier_embedding = load_target_embedding(target)
    verifier_equal = {
        key: bool(torch.equal(source_state[key], verifier_embedding))
        for key in sorted(VERIFIER_OWNED)
    }
    if not all(verifier_equal.values()):
        raise RuntimeError(
            "source verifier tensors differ from the declared Qwen3-4B target: "
            f"{verifier_equal}"
        )
    del verifier_embedding

    draft_cfg = load_draft_config_source(str(output / "config.json"))
    model = AutoDraftModel.from_config(draft_cfg, torch_dtype=torch.bfloat16)
    expected = model.state_dict()
    source_keys = set(source_state)
    expected_keys = set(expected)
    dropped = source_keys - expected_keys
    missing = expected_keys - source_keys
    if dropped != VERIFIER_OWNED or missing:
        raise RuntimeError(
            f"checkpoint contract mismatch: dropped={sorted(dropped)}, "
            f"missing={sorted(missing)}"
        )
    shape_mismatches = {
        key: [list(source_state[key].shape), list(expected[key].shape)]
        for key in sorted(expected_keys)
        if source_state[key].shape != expected[key].shape
    }
    if shape_mismatches:
        raise RuntimeError(f"shape mismatches: {shape_mismatches}")

    retained = {key: source_state[key] for key in sorted(expected_keys)}
    output_weights = output / "model.safetensors"
    save_file(retained, str(output_weights), metadata={"format": "pt"})
    saved_state = load_file(str(output_weights), device="cpu")
    unequal = [
        key
        for key in sorted(expected_keys)
        if not torch.equal(source_state[key], saved_state[key])
    ]
    if unequal:
        raise RuntimeError(f"retained tensors changed during save: {unequal}")

    for name in ("README.md", "train_command.txt", "val_metrics.json"):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, output / f"SOURCE_{name}")
    shutil.copy2(source / "config.json", output / "SOURCE_config.json")

    report = {
        "conversion": "Speculators DFlash2 -> SpecForge/SGLang DFlash2",
        "source": str(source),
        "source_sha256": sha256(source_weights),
        "output": str(output),
        "output_sha256": sha256(output_weights),
        "source_tensor_count": len(source_state),
        "output_tensor_count": len(saved_state),
        "retained_tensor_count": len(expected_keys),
        "retained_tensors_bitwise_equal": True,
        "dropped_verifier_owned_tensors": sorted(dropped),
        "dropped_tensors_equal_declared_target": verifier_equal,
        "block_size": int(output_cfg["block_size"]),
        "speculative_tokens": int(output_cfg["block_size"]) - 1,
        "sample_from_anchor": False,
        "is_causal": output_cfg["is_causal"],
        "sliding_window": output_cfg.get("sliding_window"),
        "target_layer_ids": output_cfg["dflash_config"]["target_layer_ids"],
        "speculators_aux_hidden_state_layer_ids": output_cfg[
            "eagle_aux_hidden_state_layer_ids"
        ],
        "source_manifest": tensor_manifest(source_weights),
        "output_manifest": tensor_manifest(output_weights),
    }
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    del expected, retained, saved_state, source_state, model
    fresh = AutoDraftModel.from_config(draft_cfg, torch_dtype=torch.bfloat16)
    strict = warm_start_draft_model(
        fresh,
        str(output),
        draft_config=draft_cfg,
        strategy="dflash",
        trust_remote_code=True,
    )
    strict_payload = {
        "source": strict.source,
        "checkpoint_format": strict.checkpoint_format,
        "loaded_keys": strict.loaded_keys,
        "missing_keys": list(strict.missing_keys),
        "loaded_embedding": strict.loaded_embedding,
    }
    (output / "strict_reload_report.json").write_text(
        json.dumps(strict_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "strict_reload": strict_payload}, indent=2))


if __name__ == "__main__":
    main()
