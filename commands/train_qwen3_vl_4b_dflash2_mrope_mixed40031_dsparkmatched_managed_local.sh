#!/usr/bin/env bash
set -euo pipefail

DFLASH2_REPO=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/src/SpecForge-pr785-dflash2-vl-mrope
DFLASH2_ENV=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/envs/specforge785-dflash2-vl
DFLASH2_EXP=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/experiments/qwen3-vl-4b-dflash2-mrope-mixed40031-dsparkmatched-4gpu-20261108
DFLASH2_CONFIG=examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-mixed40031-dsparkmatched.yaml

cd "$DFLASH2_REPO"
mkdir -p "$DFLASH2_EXP"

export PATH="$DFLASH2_ENV/bin:$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DFLASH2_REPO${PYTHONPATH:+:$PYTHONPATH}"

# Use python -m so this checkout is authoritative.  The environment's old
# `specforge` console script may still point at SpecForge-main.
exec "$DFLASH2_ENV/bin/python" -m specforge.cli train \
  -c "$DFLASH2_CONFIG" \
  --role auto \
  2>&1 | tee "$DFLASH2_EXP/train.log"
