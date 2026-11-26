"""Checkpoint identity for DFlash-family resolved computation configurations."""

import hashlib
import json
from pathlib import Path

from .dflash_mask import resolve_dflash_is_causal

COMPUTE_CONFIG_KEY = "dflash_compute_config"


def dflash_compute_config(config):
    """Keep computation fields while dropping serialization/runtime metadata."""
    from .dflash import get_rope_scaling_value

    payload = config.to_dict() if hasattr(config, "to_dict") else dict(vars(config))
    if payload.get("draft_vocab_size") is None:
        payload["draft_vocab_size"] = payload.get("vocab_size")
    for key in list(payload):
        if key.startswith("_") or key in {
            "architectures", "auto_map", "transformers_version", "dtype",
            "torch_dtype", "use_cache", "output_attentions", "output_hidden_states",
            "return_dict", "tie_word_embeddings",
        }:
            payload.pop(key)
    rope = dict(getattr(config, "rope_scaling", None) or {})
    rope.update(getattr(config, "rope_parameters", None) or {})
    rope["rope_theta"] = get_rope_scaling_value(config, "rope_theta", getattr(config, "rope_theta", 10000.0))
    rope["rope_type"] = rope.get("rope_type", rope.get("type", "default"))
    rope.pop("type", None)
    payload.pop("rope_scaling", None)
    payload.pop("rope_theta", None)
    payload["rope_parameters"] = rope
    layers = payload.get("layer_types", [])
    payload["is_causal"] = [resolve_dflash_is_causal(payload.get("is_causal"), layer) for layer in layers]
    method = dict(payload.get("dflash_config") or {})
    if payload.get("block_size") is None:
        payload["block_size"] = method.get("block_size")
    method.pop("block_size", None)
    payload["dflash_config"] = method
    # JSON normalizes tuple/list representations across checkpoint writers.
    return json.loads(json.dumps(payload, sort_keys=True))


def _legacy_config(state, config_path=None, *, algorithm="dflash"):
    """Recover the original JSON only when its recorded content hash matches."""
    from specforge.modeling.auto import AutoDraftModelConfig

    name = "DSpark" if algorithm == "dspark" else "DFlash"
    provenance = dict(state.get("specforge_model_provenance") or ())
    identity = provenance.get("draft_config")
    if not identity or identity[0] not in {"file", "file-stat-v1", "file-stat-v2"}:
        raise ValueError(f"{name} checkpoint lacks a resolved compute config and verifiable original config provenance")
    path = Path(identity[1])
    record = identity[2]
    if identity[0] != "file" and record[0] != "sha256":
        raise ValueError(f"{name} original config provenance must contain a SHA-256 digest")
    if not path.is_file() and config_path is not None:
        path = Path(config_path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record[-1]:
        raise ValueError(f"{name} original config is missing or differs from checkpoint provenance")
    config = AutoDraftModelConfig.from_file(str(path))
    depth = state.get(f"{algorithm}_draft_num_hidden_layers", config.num_hidden_layers)
    if len(config.layer_types) != depth:
        if len(set(config.layer_types)) != 1:
            raise ValueError(f"cannot reconstruct legacy {name} mixed attention layout")
        config.layer_types = [config.layer_types[0]] * depth
    config.num_hidden_layers = depth
    config.block_size = state.get(f"{algorithm}_block_size", getattr(config, "block_size", None))
    method = dict(getattr(config, "dflash_config", None) or {})
    for field in ("mask_token_id", "target_layer_ids"):
        if algorithm + "_" + field in state:
            method[field] = state[algorithm + "_" + field]
    config.dflash_config = method
    return config


def validate_dflash_compute_config(state, config, *, config_path=None, algorithm="dflash"):
    recorded = state.get(f"{algorithm}_compute_config")
    if recorded is None:
        recorded = dflash_compute_config(_legacy_config(state, config_path, algorithm=algorithm))
    current = dflash_compute_config(config)
    differences = sorted(key for key in recorded.keys() | current.keys() if recorded.get(key) != current.get(key))
    if differences:
        name = "DSpark" if algorithm == "dspark" else "DFlash"
        raise ValueError(f"{name} checkpoint/config computation mismatch: " + ", ".join(differences))
