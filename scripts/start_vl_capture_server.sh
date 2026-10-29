#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_MODEL_PATH:?set TARGET_MODEL_PATH to Qwen3-VL-4B-Instruct}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export SGLANG_MM_AVOID_RETOKENIZE="${SGLANG_MM_AVOID_RETOKENIZE:-1}"

export MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-http://127.0.0.1:35880/metadata}"
export MOONCAKE_MASTER_SERVER_ADDR="${MOONCAKE_MASTER_SERVER_ADDR:-127.0.0.1:35551}"
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}"
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-68719476736}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-1073741824}"
export MC_TRANSFER_TIMEOUT="${MC_TRANSFER_TIMEOUT:-300}"

exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${TARGET_MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --dtype bfloat16 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --context-length "${CONTEXT_LENGTH}" \
  --chunked-prefill-size -1 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --enable-spec-capture \
  --spec-capture-method dspark \
  --spec-capture-aux-layer-ids 1 9 17 25 33
