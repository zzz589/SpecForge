#!/usr/bin/env bash
set -euo pipefail
: "${TARGET_MODEL_PATH:?set TARGET_MODEL_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SGLANG_MM_AVOID_RETOKENIZE=1
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
export MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-http://127.0.0.1:35880/metadata}"
export MOONCAKE_MASTER_SERVER_ADDR="${MOONCAKE_MASTER_SERVER_ADDR:-127.0.0.1:35551}"
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-34359738368}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-1073741824}"
PORT="${PORT:-30000}"; TP_SIZE="${TP_SIZE:-1}"; MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
exec python -m sglang.launch_server  --model-path "${TARGET_MODEL_PATH}" --host 0.0.0.0 --port "${PORT}"  --tp-size "${TP_SIZE}" --dtype bfloat16 --trust-remote-code  --mem-fraction-static "${MEM_FRACTION_STATIC}" --chunked-prefill-size -1  --enable-spec-capture --spec-capture-method dspark  --spec-capture-aux-layer-ids 1 8 15 22 29
