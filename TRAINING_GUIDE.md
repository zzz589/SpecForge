# Vision-language training workflow

## 1. VL JSONL contract

The v1 path accepts zero or one image per sample. Use image, image_path, or a one-element images list. The first user turn must contain <image>.

~~~json
{"image":"/data/images/0001.jpg","conversations":[{"role":"user","content":"<image>\nDescribe this image."},{"role":"assistant","content":"..."}]}
~~~

Paths must be visible to regeneration and training processes. Base64/data-URI strings are accepted. Multiple images are rejected, not silently downgraded.

## 2. Regenerate VL answers

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server  --model-path /models/Qwen3.5-4B --host 127.0.0.1 --port 30000  --tp-size 1 --dtype bfloat16 --trust-remote-code
~~~

Then:

~~~bash
python scripts/regenerate_train_data.py  --model /models/Qwen3.5-4B --reasoning disable  --temperature 0 --concurrency 32 --max-tokens 4096  --server-address 127.0.0.1:30000  --input-file-path /data/train_vl.jsonl  --output-file-path /data/train_vl_regen.jsonl

python scripts/validate_regenerated_data.py  --data-path /data/train_vl_regen.jsonl  --expect-non-reasoning --strict-think-markers
~~~

The regeneration script attaches image bytes to the first user turn containing <image>. Before training, sample-check that every image exists and the placeholder remains.

## 3. Start Mooncake services

The default external recipe expects metadata at http://127.0.0.1:35880/metadata, master at 127.0.0.1:35551, and TCP. Start the site's Mooncake metadata/master services first. If endpoints differ, export matching MOONCAKE_METADATA_SERVER and MOONCAKE_MASTER_SERVER_ADDR and keep the training configuration consistent.

## 4. Start the patched capture server

~~~bash
export TARGET_MODEL_PATH=/models/Qwen3.5-4B
export CUDA_VISIBLE_DEVICES=0
export PORT=30000
bash scripts/start_vl_capture_server.sh
~~~

This sets SGLANG_MM_AVOID_RETOKENIZE=1, enables DSpark spec capture, disables chunked prefill as required, and captures target layers [1, 8, 15, 22, 29]. Wait for readiness. For RDMA, set MOONCAKE_PROTOCOL=rdma and explicitly set MOONCAKE_RDMA_DEVICES.

## 5. Plan and train

~~~bash
export TARGET_MODEL_PATH=/models/Qwen3.5-4B
export TRAIN_DATA_PATH=/data/train_vl_regen.jsonl
export OUTPUT_DIR=/outputs/qwen3.5-4b-vl-dspark
export SERVER_URL=http://127.0.0.1:30000
PLAN=1 bash scripts/train_vl.sh
bash scripts/train_vl.sh
~~~

The default recipe is examples/configs/online/disaggregated/external/qwen3.5-4b-vl-dspark.yaml. It launches four trainers and consumes input_ids, loss_mask, hidden_states and target_last_hidden_states.

Append typed overrides as needed:

~~~bash
bash scripts/train_vl.sh training.max_steps=1000 training.log_interval=1
~~~

## 6. Resume

~~~bash
bash scripts/train_vl.sh training.resume_from=/outputs/previous-run
~~~

Keep the same data, server topology, Mooncake endpoints and branch revision.

## 7. Export

~~~bash
specforge export --to sglang  --checkpoint /outputs/run/checkpoint-1000  --draft-config configs/qwen3.5-4b-dspark.json  --output-dir /outputs/exported-vl-draft
~~~

## 8. MMMU evaluation

Launch the target plus exported draft in SGLang, then:

~~~bash
python benchmarks/evaluate_dspark_mmmu_sglang.py  --data /data/mmmu.jsonl --base-url http://127.0.0.1:30000  --model /models/Qwen3.5-4B  --output-jsonl /outputs/mmmu_predictions.jsonl  --summary-json /outputs/mmmu_summary.json  --max-samples 100 --max-tokens 2048
~~~

## Preflight checklist

- Environment and exact patch verification pass.
- Each image row has one readable image and a first-user <image> placeholder.
- Regeneration and conversation validation pass.
- Mooncake services and the patched capture server are reachable.
- PLAN=1 shows four trainers and the intended endpoint.
- OUTPUT_DIR is new, or resume is explicit.
