#!/usr/bin/env bash
set -euo pipefail

DFLASH2_ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
DFLASH2_REPO="$DFLASH2_ROOT/src/SpecForge-pr785-dflash2-vl-mrope"
DFLASH2_ENV="$DFLASH2_ROOT/envs/specforge785-dflash2-vl"
DFLASH2_EXP="$DFLASH2_ROOT/experiments/qwen3-vl-4b-dflash2-mrope-textvqa100-4gpu-20261108"

cd "$DFLASH2_REPO"
export PYTHONPATH="$DFLASH2_REPO"
export PATH="$DFLASH2_ENV/bin:$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$DFLASH2_EXP"

"$DFLASH2_ENV/bin/python" -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-redhatai-dflash2-b9-mrope-textvqa100-4gpu.yaml \
  --role auto \
  2>&1 | tee "$DFLASH2_EXP/launcher.log"
