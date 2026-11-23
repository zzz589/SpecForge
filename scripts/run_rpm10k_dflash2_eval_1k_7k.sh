#!/usr/bin/env bash
set -euo pipefail

ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
DATA="$ROOT/data/RPM-10K/processed-dflash2-v1"
E1="$ROOT/experiments/rpm10k-dflash2-scale-20261109"
E2="$ROOT/experiments/rpm10k-dflash2-dpace-scale-20261110"
REPO="$ROOT/src/SpecForge-pr785-dflash2-vl-mrope"
SGLREPO="$ROOT/src/sglang-pr35371-dflash2"
ENV="$ROOT/envs/specforge785-dflash2-vl"
TARGET="$ROOT/models/Qwen3-VL-4B-Instruct"
TESTDATA="$DATA/train-ready/test/rpm10k-test-ready-2000.jsonl"
GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-30005}"

export PATH="$ENV/bin:$ENV/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$ENV/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

checkpoint_for() {
  local size="$1"
  if (( size <= 2000 )); then
    find "$E1/n${size}/output" -mindepth 1 -maxdepth 1 -type d -name '*step*' | sort | tail -1
  else
    find "$E2/n${size}/output" -mindepth 1 -maxdepth 1 -type d -name '*step*' | sort | tail -1
  fi
}

experiment_for() {
  local size="$1"
  if (( size <= 2000 )); then
    echo "$E1/n${size}"
  else
    echo "$E2/n${size}"
  fi
}

steps_for() {
  case "$1" in
    1000) echo 750 ;;
    2000) echo 1500 ;;
    3000) echo 2250 ;;
    4000) echo 3000 ;;
    5000) echo 3750 ;;
    6000) echo 4500 ;;
    7000) echo 5250 ;;
    *) return 1 ;;
  esac
}

for size in 1000 2000 3000 4000 5000 6000 7000; do
  ckpt="$(checkpoint_for "$size")"
  experiment="$(experiment_for "$size")"
  steps="$(steps_for "$size")"
  eval_dir="$experiment/evaluation/rpm10k-test-2000-dflash2-vl"
  mkdir -p "$eval_dir"

  if [[ -f "$eval_dir/final/summary.json" ]] && \
     "$ENV/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("status")=="complete" else 1)' "$eval_dir/final/summary.json"; then
    echo "[$(date -Is)] n${size}: complete result already exists; skipping"
    continue
  fi

  cleanup
  server_pid=""
  rm -f "$eval_dir/server.ready"
  echo "[$(date -Is)] n${size}: starting DFlash2-VL server from $ckpt/exported-draft"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONPATH="$SGLREPO/python:${PYTHONPATH:-}" \
  nohup "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$TARGET" \
    --dtype bfloat16 \
    --trust-remote-code \
    --tp-size 1 \
    --context-length 4096 \
    --attention-backend triton \
    --speculative-draft-attention-backend triton \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$ckpt/exported-draft" \
    --speculative-dflash-block-size 9 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --mem-fraction-static 0.72 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    > "$eval_dir/server.log" 2>&1 &
  server_pid=$!
  echo "$server_pid" > "$eval_dir/server.pid"

  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "n${size}: server exited before readiness" >&2
      tail -160 "$eval_dir/server.log" >&2
      exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    echo "n${size}: server readiness timeout" >&2
    tail -160 "$eval_dir/server.log" >&2
    exit 1
  fi
  touch "$eval_dir/server.ready"

  grep -q 'type=DFlash2DraftModel' "$eval_dir/server.log"
  grep -q 'block_size=9' "$eval_dir/server.log"
  echo "[$(date -Is)] n${size}: server ready; running 2000 independent test samples"
  "$ENV/bin/python" "$REPO/scripts/benchmark_sglang_dflash2_rpm10k.py" \
    --data "$TESTDATA" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "$TARGET" \
    --output-jsonl "$eval_dir/raw-results.jsonl" \
    --summary-json "$eval_dir/raw-summary.json" \
    --training-size "$size" \
    --max-tokens 256 \
    --num-speculative-tokens 8 \
    --timeout 300 \
    > "$eval_dir/client.log" 2>&1

  "$ENV/bin/python" "$REPO/scripts/analyze_rpm10k_eval.py" \
    --inputs "$eval_dir/raw-results.jsonl" \
    --output-dir "$eval_dir/final" \
    --checkpoint-dir "$ckpt" \
    --experiment-dir "$experiment" \
    --training-size "$size" \
    --steps "$steps" \
    --doc-script "$REPO/scripts/update_rpm10k_experiment_docs.py" \
    > "$eval_dir/analyze.log" 2>&1

  "$ENV/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d.get("metrics",{}), ensure_ascii=False))' "$eval_dir/final/summary.json"
  echo "[$(date -Is)] n${size}: evaluation complete"
done

echo "[$(date -Is)] all requested checkpoints complete"
