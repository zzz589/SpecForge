"""Verify real training changed DFlash2 selector and backbone weights."""

import sys
from pathlib import Path

import torch
from safetensors import safe_open


def weight_file(directory):
    files = list(Path(directory).glob("*.safetensors"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one safetensors file in {directory}: {files}")
    return files[0]


with safe_open(weight_file(sys.argv[1]), framework="pt", device="cpu") as source:
    with safe_open(weight_file(sys.argv[2]), framework="pt", device="cpu") as trained:
        keys = sorted(set(source.keys()) & set(trained.keys()))
        for prefix in ("candidate_selector.hidden_projection.weight", "layers.0."):
            candidate = next((key for key in keys if key.startswith(prefix)), None)
            if candidate is None:
                raise RuntimeError(f"Missing {prefix} tensor in source or trained model")
            before = source.get_tensor(candidate)
            after = trained.get_tensor(candidate)
            changed = int(torch.count_nonzero(before != after).item())
            print({"tensor": candidate, "shape": tuple(before.shape), "changed_elements": changed})
            if changed == 0:
                raise RuntimeError(f"Training did not change {candidate}")
