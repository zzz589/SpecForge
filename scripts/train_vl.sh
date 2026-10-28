#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TARGET_MODEL_PATH:?set TARGET_MODEL_PATH}"
: "${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH to regenerated VL JSONL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/configs/online/disaggregated/external/qwen3.5-4b-vl-dspark.yaml}"
DRAFT_CONFIG="${DRAFT_CONFIG:-${ROOT_DIR}/configs/qwen3.5-4b-dspark.json}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000}"
RUN_ID="${RUN_ID:-xiaojing-vl-$(date +%Y%m%d-%H%M%S)}"
PLAN_ARGS=(); [[ "${PLAN:-0}" == 1 ]] && PLAN_ARGS+=(--plan)
mkdir -p "${OUTPUT_DIR}"
exec specforge train --config "${CONFIG}" "${PLAN_ARGS[@]}"  "model.target_model_path=${TARGET_MODEL_PATH}"  "model.draft_model_config=${DRAFT_CONFIG}"  "data.train_data_path=${TRAIN_DATA_PATH}"  "data.cache_dir=${OUTPUT_DIR}/cache"  "run_id=${RUN_ID}" "output_dir=${OUTPUT_DIR}"  "deployment.disaggregated.control_dir=${OUTPUT_DIR}/control"  "deployment.disaggregated.consumer_state_dir=${OUTPUT_DIR}/consumer-state"  "deployment.disaggregated.server_urls=[\"${SERVER_URL}\"]" "$@"
