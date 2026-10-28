#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TARGET_MODEL_PATH:?set TARGET_MODEL_PATH}"
: "${PROMPTS_PATH:?set PROMPTS_PATH to a .prompts.jsonl file}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/local/qwen3-1.7b-dflash-online-dp1-100k-20261027.yaml}"
DRAFT_CONFIG="${DRAFT_CONFIG:-${ROOT_DIR}/configs/local/qwen3-1.7b-dflash-5layer.json}"
RUN_ID="${RUN_ID:-xiaojing-text-$(date +%Y%m%d-%H%M%S)}"
PLAN_ARGS=(); [[ "${PLAN:-0}" == 1 ]] && PLAN_ARGS+=(--plan)
mkdir -p "${OUTPUT_DIR}"
exec specforge train --config "${CONFIG}" "${PLAN_ARGS[@]}"  "model.target_model_path=${TARGET_MODEL_PATH}"  "model.draft_model_config=${DRAFT_CONFIG}"  "data.prompts_path=${PROMPTS_PATH}"  "data.cache_dir=${OUTPUT_DIR}/cache"  "run_id=${RUN_ID}" "output_dir=${OUTPUT_DIR}"  "deployment.disaggregated.control_dir=${OUTPUT_DIR}/control" "$@"
