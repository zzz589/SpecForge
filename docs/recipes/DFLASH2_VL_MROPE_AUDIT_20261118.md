# DFlash2-VL 草稿 mRoPE：隔离实现与真实链路核验（2026-11-18）

## 结论与边界

- 隔离训练目录：`/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/src/SpecForge-pr785-dflash2-vl-mrope`。
- 隔离推理目录：`/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/src/sglang-dflash2-vl-mrope`。原有训练和推理目录没有被本次 mRoPE 修改覆盖。
- 真实图文数据完成了 managed-local 采集、三轴位置传递、DFlash2 主干和 selector 反向传播、检查点保存、导出和 SGLang 投机推理。该试验只有 **2 个优化器 step**，用来验证链路，不能用于判断接受长度或模型质量。
- 与同配置的原模型逐词推理比较：100 条真实图片提示、`temperature=0`、最多生成 32 token，**85/100 条完整输出相同**。15 条的首次分叉处，原模型前两候选的对数分数差均不超过 0.25；两条重复 3 次仍稳定分叉。批量验证与逐词解码在 BF16 下存在数值差异。因此可以确认链路能运行，**不能声称逐 token 严格等价**。后续性能实验应以相同推理设置重新测量。

## 逐阶段证据

| 阶段 | 核验与结果 |
|---|---|
| 训练输入 | Qwen3-VL 图像网格生成三轴位置；采集后 `position_ids` 以 `[B,S,3]` 进入 collator。最初发现 collator 丢弃该字段并修复；启用 mRoPE 时缺失或形状错误会直接报错，避免悄悄退化成一维 RoPE。相关 6 项多模态和 collation 单元测试通过。 |
| 旋转计算 | 用同一组真实配置的三轴位置比较 SpecForge 草稿 Q/K 与 SGLang 草稿 Q/K，GPU 数值误差在 `1e-5` 范围内。 |
| 视觉 hidden states | managed-local 采集服务加载 Qwen3-VL 目标模型，启用 `spec_capture_method=dflash`，采集层 `[0,8,16,24,32]`；导出草稿配置中的 `target_layer_ids` 同样为 `[0,8,16,24,32]`。100 条图文提示准备完成，0 条跳过，100 条采集成功。 |
| DFlash2 训练 | 配置 `architectures=DFlash2DraftModel`、`block_size=9`、`selector_rank=256`、`selector_top_k=16`，selector loss 为有限值；2 个优化器 step 结束并保存。训练前后比较，`candidate_selector.hidden_projection.weight` 有 647835 个元素变化，`layers.0.attention_conv.base_kernel` 有 4794 个元素变化，证明 selector 和主干都实际更新。 |
| 推理 | 草稿使用图像提示中的三轴位置和目标模型生成阶段的 `mrope_position_delta`；mRoPE 模式禁用只支持一维位置的融合 KV/QK 路径。真实图片请求成功，投机验证累计调用数大于 0。此两步检查点的最近一次指标约为接受长度 1.875（含 bonus），只是烟雾测试结果。 |
| 输出一致性 | 100 条真实图文请求有 85 条逐字一致。15 条首次分叉时，原模型前两词对数分数差为 0 至 0.25；需要把这种 BF16 数值敏感性和是否要求严格等价区分开。 |
| 额外请求 | 一条纯文本问答在两条推理路径下均得到 `The capital of France is Paris.`；把两条真实样本的图片合成一个双图请求后，两条路径均成功推理，48 token 输出一致。 |

回归测试：训练侧 DFlash2 模型 30 项通过，多模态与 collation 6 项通过；推理侧三轴位置映射与 Q/K 旋转数值测试通过。核验结束后，我启动的 31055、31056 两个推理服务均已关闭，端口已释放。

## 目录与产物

```text
ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
训练配置：$ROOT/src/SpecForge-pr785-dflash2-vl-mrope/examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d.yaml
初始化：$ROOT/experiments_results/qwen3-4b-redhatai-dflash2-specforge-b9-20261105/Qwen3-4B-speculator.dflash2-specforge-b9-qwen3vl-mrope-v1
检查点：$ROOT/experiments/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d/output/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d-step2
导出模型：$ROOT/experiments/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d/export
采集日志：$ROOT/experiments/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d/control/logs/capture-server-0.log
推理对比脚本：$ROOT/src/sglang-dflash2-vl-mrope/test/registered/unit/spec/test_dflash2_mrope_target_parity.py
```

## 重现训练

以下命令在 **hn47** 上执行。managed-local 自动启动采集服务；本次配置为 GPU 6 采集、GPU 7 训练。运行前重新检查 GPU 占用。

```bash
source /etc/profile.d/modules.sh
module unload CUDA
module load CUDA/13.0
export ROOT=/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj
export REPO="$ROOT/src/SpecForge-pr785-dflash2-vl-mrope"
export PY="$ROOT/envs/specforge785-dflash2-vl/bin/python"
export PYTHONPATH="$REPO"
export CUDA_HOME=/APP/u22/ai_x86/CUDA/13.0
export CC=/usr/bin/gcc
cd "$REPO"
"$PY" -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d.yaml \
  --role auto
```

## 导出与推理核验

```bash
"$PY" -m specforge.cli export --to hf \
  --checkpoint "$ROOT/experiments/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d/output/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d-step2" \
  --draft-config "$ROOT/experiments_results/qwen3-4b-redhatai-dflash2-specforge-b9-20261105/Qwen3-4B-speculator.dflash2-specforge-b9-qwen3vl-mrope-v1/config.json" \
  --embedding-source target \
  --embedding-key model.language_model.embed_tokens.weight \
  --output-dir "$ROOT/experiments/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d/export"

export PYTHONPATH="$ROOT/src/sglang-dflash2-vl-mrope/python"
CUDA_VISIBLE_DEVICES=5 "$PY" -m sglang.launch_server \
  --model-path "$ROOT/models/Qwen3-VL-4B-Instruct" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$ROOT/experiments/qwen3-vl-4b-dflash2-mrope-parity-smoke-20261118d/export" \
  --speculative-num-draft-tokens 9 \
  --attention-backend triton --disable-cuda-graph --disable-radix-cache \
  --chunked-prefill-size -1 --context-length 4111 \
  --max-running-requests 1 --mem-fraction-static 0.40 \
  --enable-metrics --host 127.0.0.1 --port 31055 --trust-remote-code

# 另一终端，在已确认空闲的 GPU 上启动相同目标模型的逐词基线：
CUDA_VISIBLE_DEVICES=4 "$PY" -m sglang.launch_server \
  --model-path "$ROOT/models/Qwen3-VL-4B-Instruct" \
  --attention-backend triton --disable-cuda-graph --disable-radix-cache \
  --chunked-prefill-size -1 --context-length 4111 \
  --max-running-requests 1 --mem-fraction-static 0.40 \
  --host 127.0.0.1 --port 31056 --trust-remote-code

"$PY" "$ROOT/src/sglang-dflash2-vl-mrope/test/registered/unit/spec/test_dflash2_mrope_target_parity.py" \
  "$ROOT/data/regen_textqva-100-flow-test.jsonl" \
  "$ROOT/models/Qwen3-VL-4B-Instruct" 100 31055 31056 summary
```

`--speculative-num-draft-tokens 9` 对应初始化模型的 `block_size=9`，即最多 8 个草稿 token 加 1 个 bonus；这里没有把 DFlash2 错当成普通 DFlash：草稿架构和 selector 权重均已核对。`--speculative-algorithm DFLASH` 是 SGLang 为该家族共用的调度入口。

## 后续正式实验前

这份两步检查点仅证明端到端链路与权重更新。正式训练需用相同数据、学习率和步数控制与一维 RoPE 对照组的可比性，再在同一测试集测接受长度。由于 100 条确定性输出并非全部严格一致，报告接受长度时应附上目标模型和草稿服务的精度、推理设置以及输出一致性结果。
