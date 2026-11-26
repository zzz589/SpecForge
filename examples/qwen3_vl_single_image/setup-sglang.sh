#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
destination=${1:-third_party/sglang}
if [[ -e "$destination" ]]; then echo "Use a new destination: $destination" >&2; exit 1; fi
git clone --no-checkout https://github.com/sgl-project/sglang.git "$destination"
git -C "$destination" fetch "$here/runtime/sglang-base.bundle" HEAD
git -C "$destination" checkout --detach 01889693a51fbf1be2af03f13537d738d49b0978
git -C "$destination" apply "$here/runtime/dspark.patch"
cp "$here/runtime/dspark_mrope.py" "$destination/python/sglang/srt/speculative/dspark_components/dspark_mrope.py"
python scripts/prepare_sglang_capture_overlay.py --base "$destination"
