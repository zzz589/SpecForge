#!/usr/bin/env bash
# One source/environment for DFlash-VL, DFlash2-mRoPE-VL and DSpark-VL.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
XJ=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
PYTHON="$XJ/envs/specforge785-dflash2-vl/bin/python"
SGLANG="$XJ/src/sglang-dflash2-vl-mrope/python"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG.yaml [specforge train options]" >&2
  exit 2
fi
CONFIG="$1"
shift
cd "$ROOT"
CAPTURE="$($PYTHON "$ROOT/scripts/prepare_sglang_capture_overlay.py" --base "${SGLANG%/python}")"
export PYTHONPATH="$ROOT:$CAPTURE:$SGLANG${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c 'import pathlib,sys,specforge,sglang; root=pathlib.Path(sys.argv[1]).resolve(); sg=pathlib.Path(sys.argv[2]).resolve(); assert pathlib.Path(specforge.__file__).resolve().is_relative_to(root); assert pathlib.Path(sglang.__file__).resolve().is_relative_to(sg); print("SpecForge:",specforge.__file__); print("SGLang:",sglang.__file__)' "$ROOT" "$SGLANG"
exec "$PYTHON" -m specforge.cli train -c "$CONFIG" "$@"
