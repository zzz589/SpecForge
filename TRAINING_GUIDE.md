# 视觉语言训练流程  

## 0.代码下载以及环境安装
### 下载代码
~~~bash
git clone \
  --branch xiaojing/vl-training \
  --single-branch \
  https://github.com/zzz589/SpecForge.git \
  SpecForge-vl-training

cd SpecForge-vl-training
git rev-parse HEAD
~~~
### 安装环境
~~~bash
uv venv -p 3.11 --seed
source .venv/bin/activate
uv pip install -v -e . --prerelease=allow
~~~
## 1. 数据准备 
~~~bash
python scripts/prepare_llava_vlm_data.py \
  --input-path /data/llava-1.5-665k/raw/llava_v1_5_mix665k_shuffled.json \
  --image-root /data/llava-1.5-665k/images \
  --image-prefix textvqa/ \
  --output-path /data/llava-1.5-665k/subsets/textvqa-100-flow-test.jsonl \
  --source-name textvqa \
  --num-samples 100 \
  --overwrite
~~~

## 2. 重生成 VL 回答

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server  --model-path /models/Qwen3-VL-4B-Instruct --host 127.0.0.1 --port 30000  --tp-size 1 --dtype bfloat16 --trust-remote-code
~~~

然后执行：

~~~bash
python scripts/regenerate_train_data.py \
  --model /HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/models/Qwen3-VL-4B-Instruct \
  --reasoning disable \
  --temperature 0 \
  --concurrency 32 \
  --max-tokens 4096 \
  --server-address 127.0.0.1:30000 \
  --input-file-path /data/llava-1.5-665k/subsets/textvqa-100-flow-test.jsonl \
  --output-file-path data/regen_textqva-100-flow-test.jsonl

python scripts/validate_regenerated_data.py  --data-path /data/train_vl_regen.jsonl  --expect-non-reasoning --strict-think-markers
~~~

重生成脚本会把图片字节附加到第一条包含 `<image>` 的用户消息中。训练前应抽样检查：每张图片都真实存在，并且 `<image>` 占位符仍然保留。

## 3.启动训练（managed-loacl和extern二选一）
~~~bash
bash scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18
~~~
原始 SGLang 只负责生成文本，不会把指定层的 hidden states 发布给 SpecForge。

这个脚本会修改当前 Python 环境中已安装的 SGLang，增加：
- --enable-spec-capture
- --spec-capture-method
- --spec-capture-aux-layer-ids
- 向 Mooncake 发布 hidden states 的逻辑
- SpecForge 所需的 capture 返回协议
通常一个环境只需应用一次。重新安装或升级 SGLang 后才需要重新应用。
### managed-local 模式
~~~bash
export MC_TRANSFER_TIMEOUT=300

"$PY785" -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-dspark-b16.yaml \
  --role auto
~~~
### extern 模式
#### 1. 启动 Mooncake 服务

“目标模型 Capture Server”和“DSpark Trainer”之间的高速张量中转站。
~~~bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=127.0.0.1 \
  --rpc_port=35551 \
  --http_metadata_server_port=35880 \
  --metrics_port=35903 \
  --default_kv_lease_ttl=10s
~~~
#### 2.启动训练用 Capture Server
~~~bash
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=68719476736
export MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
export MC_TRANSFER_TIMEOUT=300
export SGLANG_MM_AVOID_RETOKENIZE=1

CUDA_VISIBLE_DEVICES=4 \
python -m sglang.launch_server \
  --model-path models/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --tp-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --context-length 4096 \
  --chunked-prefill-size -1 \
  --mem-fraction-static 0.85 \
  --enable-spec-capture \
  --spec-capture-method dspark \
  --spec-capture-aux-layer-ids 1 9 17 25 33
  ~~~
#### 3.训练   
~~~bash
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
export MC_TRANSFER_TIMEOUT=300

CUDA_VISIBLE_DEVICES=0,1 \
"$PY785" -m specforge.cli train \
  -c /examples/configs/online/disaggregated/external/qwen3-4b-dspark.yaml \
  --role auto \
  "data.train_data_path=/data/regen_textqva-100-flow-test.jsonl" \
  "data.cache_dir=$RUN/cache" \
  "training.num_epochs=1" \
  "training.save_interval=10" \
  "run_id=qwen3-vl-4b-dspark-b16-textvqa100-smoke" \
  "output_dir=/output" \
  "deployment.disaggregated.control_dir=/control" \
  "deployment.disaggregated.consumer_state_dir=/consumer-state"
~~~

