# Text training workflow

## 1. Prepare ShareGPT JSONL

Each line is one object:

~~~json
{"conversations":[{"role":"user","content":"Explain speculative decoding."},{"role":"assistant","content":"..."}]}
~~~

Built-in data:

~~~bash
python scripts/prepare_data.py --dataset ultrachat --output-path cache/dataset --split-eval
~~~

Custom ShareGPT data:

~~~bash
python scripts/prepare_data.py --dataset sharegpt --data-path /data/source.jsonl --output-path cache/dataset
~~~

## 2. Regenerate with the target model

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path /models/Qwen3-1.7B --host 127.0.0.1 --port 30000   --tp-size 1 --dtype bfloat16 --trust-remote-code
~~~

In another terminal:

~~~bash
python scripts/regenerate_train_data.py   --model /models/Qwen3-1.7B --reasoning disable   --temperature 0 --concurrency 64 --max-tokens 4096   --server-address 127.0.0.1:30000   --input-file-path cache/dataset/sharegpt_train.jsonl   --output-file-path cache/dataset/sharegpt_train_regen.jsonl
~~~

Use --resume only when intentionally continuing the same output. Success, error and skipped rows must account for every input row.

## 3. Validate

~~~bash
python scripts/validate_regenerated_data.py   --data-path cache/dataset/sharegpt_train_regen.jsonl   --expect-non-reasoning --strict-think-markers
~~~

For reasoning targets, regenerate with --reasoning save and validate with --expect-reasoning.

## 4. Create the reusable prompt cache

~~~bash
python scripts/prepare_prompt_cache.py   --input-path cache/dataset/sharegpt_train_regen.jsonl   --output-dir cache/prompts/qwen3-1.7b   --model-path /models/Qwen3-1.7B   --chat-template qwen --max-length 3072 --num-proc 16
~~~

Keep the generated metadata file. It records the input SHA-256 and preprocessing settings.

## 5. Plan and train

~~~bash
export TARGET_MODEL_PATH=/models/Qwen3-1.7B
export PROMPTS_PATH=$PWD/cache/prompts/qwen3-1.7b/sharegpt_train_regen.prompts.jsonl
export OUTPUT_DIR=/outputs/qwen3-1.7b-dflash
PLAN=1 bash scripts/train_text.sh
bash scripts/train_text.sh
~~~

The default config is configs/local/qwen3-1.7b-dflash-online-dp1-100k-20261027.yaml. It uses managed-local disaggregated training with trainer GPU 7 and capture GPU 1; copy and edit the config for another GPU layout. Append typed overrides when needed:

~~~bash
bash scripts/train_text.sh training.max_steps=1000 training.log_interval=1
~~~

## 6. Resume

~~~bash
bash scripts/train_text.sh training.resume_from=/outputs/previous-run
~~~

Do not combine training.resume_from with model.draft_checkpoint_path.

## 7. Export and benchmark

~~~bash
specforge export --to sglang   --checkpoint /outputs/run/checkpoint-1000   --draft-config configs/local/qwen3-1.7b-dflash-5layer.json   --output-dir /outputs/exported-draft

specforge benchmark --model /models/Qwen3-1.7B   --dataset gsm8k --base-url http://127.0.0.1:30000   --num-prompts 100 --concurrency 8 --output-json /outputs/benchmark.json
~~~

## Preflight checklist

- scripts/verify_environment.py passes.
- Regeneration accounts for every row and validation passes.
- Prompt-cache metadata matches the source and target model.
- PLAN=1 shows the intended paths, GPUs and processes.
- OUTPUT_DIR is new, or resume is explicit.
