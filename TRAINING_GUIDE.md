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
  --role auto 
~~~

## 4. DFlash2 + Qwen3-VL 多模态训练

下面的命令已使用 100 条重生成 TextVQA 数据完成实跑。DFlash2 使用独立代码目录和独立环境，不修改原来的 `xj/envs/specforge785`：

- 代码：`/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/src/SpecForge-pr785-dflash2-vl`
- 环境：`/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/envs/specforge785-dflash2-vl`
- 数据：`/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/data/regen_textqva-100-flow-test.jsonl`

### 4.1 公共环境变量

~~~bash
export DFLASH2_ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
export DFLASH2_REPO="$DFLASH2_ROOT/src/SpecForge-pr785-dflash2-vl"
export DFLASH2_ENV="$DFLASH2_ROOT/envs/specforge785-dflash2-vl"
export DFLASH2_PY="$DFLASH2_ENV/bin/python"
export DFLASH2_EXP="$DFLASH2_ROOT/experiments/qwen3-vl-4b-dflash2-textvqa100-pr785-smoke"

export PYTHONPATH="$DFLASH2_REPO"
export PATH="$DFLASH2_ENV/bin:$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MC_TCP_BIND_ADDRESS=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=68719476736
export MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
export MC_TRANSFER_TIMEOUT=300

mkdir -p "$DFLASH2_EXP"
~~~

### 4.2 启动 Mooncake

在第一个终端设置 4.1 的环境变量，然后运行：

~~~bash
: "${DFLASH2_EXP:?请先在当前终端执行 4.1 公共环境变量}"
mkdir -p "$DFLASH2_EXP"

"$DFLASH2_PY" -m mooncake.cli \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=127.0.0.1 \
  --rpc_port=35551 \
  --http_metadata_server_port=35880 \
  --metrics_port=35903 \
  --default_kv_lease_ttl=10000 \
  > "$DFLASH2_EXP/mooncake.log" 2>&1
~~~

### 4.3 启动 DFlash2 Capture Server

DFlash2 与 DFlash 共用 `--spec-capture-method dflash`。实际 draft 架构由 trainer 配置中的 `DFlash2DraftModel` 决定。

在第二个终端设置 4.1 的环境变量，然后运行：

~~~bash
: "${DFLASH2_EXP:?请先在当前终端执行 4.1 公共环境变量}"
mkdir -p "$DFLASH2_EXP"

export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_MM_AVOID_RETOKENIZE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

CUDA_VISIBLE_DEVICES=5 \
"$DFLASH2_PY" -m sglang.launch_server \
  --model-path "$DFLASH2_ROOT/models/Qwen3-VL-4B-Instruct" \
  --dtype bfloat16 \
  --trust-remote-code \
  --tp-size 1 \
  --context-length 4111 \
  --attention-backend triton \
  --chunked-prefill-size -1 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --enable-spec-capture \
  --spec-capture-method dflash \
  --spec-capture-aux-layer-ids 3 10 18 25 32 \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.70 \
  > "$DFLASH2_EXP/capture-server.log" 2>&1
~~~

服务启动后检查：

~~~bash
curl http://127.0.0.1:30000/health
~~~

### 4.4 使用 100 条重生成 TextVQA 数据训练

在第三个终端设置 4.1 的环境变量，然后运行：

~~~bash
cd "$DFLASH2_REPO"

CUDA_VISIBLE_DEVICES=3 \
"$DFLASH2_PY" -m specforge.cli train \
  -c examples/configs/online/disaggregated/external/qwen3-vl-4b-dflash2-b16.yaml \
  training.num_epochs=1 \
  training.max_steps=50 \
  training.save_interval=50 \
  training.log_interval=10 \
  deployment.trainer.nproc_per_node=1 \
  data.train_data_path="$DFLASH2_ROOT/data/regen_textqva-100-flow-test.jsonl" \
  data.cache_dir="$DFLASH2_EXP/cache" \
  data.build_dataset_num_proc=4 \
  data.dataloader_num_workers=1 \
  run_id=qwen3-vl-4b-dflash2-textvqa100-pr785-smoke \
  output_dir="$DFLASH2_EXP/output" \
  deployment.disaggregated.control_dir="$DFLASH2_EXP/control" \
  deployment.disaggregated.consumer_state_dir="$DFLASH2_EXP/consumer-state" \
  > "$DFLASH2_EXP/train.log" 2>&1
~~~

上述命令把标准输出和错误都重定向到 `train.log`，因此训练期间当前终端不会打印内容。可在另一个终端实时查看：

~~~bash
tail -f "$DFLASH2_EXP/train.log"
~~~

训练命令返回提示符后，应立即检查退出码和最后的训练状态：

~~~bash
echo $?
grep -E '^step |Traceback|FAILED|RuntimeError' "$DFLASH2_EXP/train.log" | tail -n 20
~~~

退出码为 `0` 且日志出现最终 step、没有异常，即表示 trainer 正常完成。external 模式只退出 trainer，不会自动关闭手动启动的 Mooncake 和 Capture Server；不再继续训练时，需要在它们各自的终端按 `Ctrl+C`。

该配置使用 `DFlash2DraftModel`、block size 16、卷积核大小 2、卷积分组 16、selector rank 256、selector top-k 16。batch size 为 1，梯度累积为 2，因此 100 条样本对应 50 个 optimizer step。

训练成功后，检查点位于：

~~~text
/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/experiments/qwen3-vl-4b-dflash2-textvqa100-pr785-smoke/output/qwen3-vl-4b-dflash2-textvqa100-pr785-smoke-step50/training_state.pt
~~~

### 4.5 managed-local 一键训练（已实跑）

managed-local 模式不需要手动分别启动 Mooncake 和 Capture Server。`specforge` supervisor 会依次启动 Mooncake、Capture Server、producer 和 consumer，并在训练成功或失败后清理它们。

当前 recipe 使用 GPU1 训练 DFlash2、GPU2 运行 Qwen3-VL Capture Server，直接读取同一份 100 条重生成数据。每次运行必须使用新的 `control_dir`；下面用时间戳生成独立实验目录。同时显式设置 `PYTHONPATH`，避免当前终端仍激活旧 `(specforge785)` 环境时错误导入 `SpecForge-main`：

~~~bash
export DFLASH2_ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
export DFLASH2_REPO="$DFLASH2_ROOT/src/SpecForge-pr785-dflash2-vl"
export DFLASH2_ENV="$DFLASH2_ROOT/envs/specforge785-dflash2-vl"
export DFLASH2_PY="$DFLASH2_ENV/bin/python"

cd "$DFLASH2_REPO"

export PYTHONPATH="$DFLASH2_REPO"
export PATH="$DFLASH2_ENV/bin:$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="$DFLASH2_ENV/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

RUN_TAG="qwen3-vl-4b-dflash2-textvqa100-pr785-managed-local-$(date +%Y%m%d-%H%M%S)"
EXP="$DFLASH2_ROOT/experiments/$RUN_TAG"

"$DFLASH2_PY" -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-dflash2-b16.yaml \
  --role auto \
  run_id="$RUN_TAG" \
  output_dir="$EXP/output" \
  data.cache_dir="$EXP/cache" \
  deployment.disaggregated.control_dir="$EXP/control" \
  deployment.disaggregated.consumer_state_dir="$EXP/consumer-state"
~~~

启动前可用下面的命令确认没有串到其他 SpecForge 仓库：

~~~bash
"$DFLASH2_PY" -c 'import specforge; print(specforge.__file__)'
~~~

输出路径必须包含 `SpecForge-pr785-dflash2-vl/specforge/__init__.py`，不能是 `SpecForge-main`。

这条命令已完整训练 100 条样本并完成 50 个 optimizer step。本次运行的检查点位于：

~~~text
$EXP/output/${RUN_TAG}-step50/training_state.pt
~~~

实跑验收结果：`prepared=100`、`skipped=0`、`committed=100`、`acked=100`、`global_step=50`，日志无 `Traceback`、`FAILED` 或 `RuntimeError`。训练完成后端口 30000、35551、35880、35903 均由 supervisor 自动释放。



单轮单图的 DFlash2、DFlash、DSpark 训练与重生成，请查看[已验证的迁移指南](examples/qwen3_vl_single_image/README.md)。
