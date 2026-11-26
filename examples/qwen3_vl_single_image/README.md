# PR785：用 Qwen3-VL-4B-Instruct 训练单图问答草稿模型

以下命令从 **SpecForge 仓库根目录**执行。目标模型冻结；更新的是 DFlash2、DFlash 或 DSpark 草稿。三个算法使用同一个目标模型、图像处理器、模板和答案，因此**答案只重新生成一次**。训练时各自启动目标模型采集隐藏状态；不可把三种算法的特征缓存、优化器状态或输出目录混用。

## 1. 拉代码、安装环境

适用 Linux、NVIDIA CUDA。使用包含本目录及 PR785 本地修复的仓库和分支，不要换成官方 SpecForge main。以下 URL 和分支由上传者填写。

```bash
export TRAIN_REPO='替换为你上传后的 GitHub 克隆地址'
export TRAIN_REF='替换为包含本配方的分支或标签'
git clone --branch "$TRAIN_REF" "$TRAIN_REPO" SpecForge
cd SpecForge
git rev-parse HEAD
export GUIDE=examples/qwen3_vl_single_image

# 已有 uv 可略过第一行；这里用独立 venv，不改系统 Python。
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate

# 先加载本机 CUDA 13.2 工具链（有 modules 的机器可用 module load CUDA/13.2）。
nvidia-smi
nvcc --version
uv pip install -e . --prerelease=allow -c "$GUIDE/constraints.txt" \
    mooncake-transfer-engine-cuda13 ninja
command -v mooncake_master
python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'

# 还原实测 SGLang：上游源码 + 两个固定提交 + DSpark mRoPE 修改。
# 使用新的目标目录；脚本最后会校验提交、补丁和 capture 基础文件。
bash "$GUIDE/setup-sglang.sh" third_party/sglang
export SG=third_party/sglang
export PYTHONPATH="$PWD:$PWD/$SG/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
python -c 'import sglang,specforge;print(sglang.__file__,specforge.__file__)'
```

基础安装依据是仓库的 `pyproject.toml` 和 `docs/sections/get_started/installation.md`；`constraints.txt` 固定 hn47 已用过的关键版本，并非完整系统镜像。需要可访问匹配平台的 wheel 源；驱动、CUDA toolkit、编译器及系统共享库由服务器提供。若 CUDA 或包版本不符，先解决环境问题。训练脚本自动构建目标特征采集用 overlay，不需要手工改第三方源码。每次重新登录后重新激活 venv 并设置 GUIDE、SG、PYTHONPATH。

## 2. 准备模型与数据

```bash
export TARGET=models/Qwen3-VL-4B-Instruct
export INIT=models/dflash2-b9-fullattn-mrope
export RAW=data/questions.jsonl
export DATA=data/regenerated.jsonl
export RUNS=runs/single-image-smoke
# 如未准备目标模型：
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "$TARGET"
```

`INIT` 只给 DFlash2 使用：把既有 B9 全注意力 mRoPE 初始化目录（HF 权重及 config.json）复制到这里。本目录附带 `dflash2-b9-config.json` 供核对；**配置不是权重**。不要用 B16 配置或普通文本草稿代替。DFlash 和 DSpark 直接读取仓库 `configs/qwen3-vl-4b-{dflash,dspark}-mrope.json`，无需该初始化权重。

`RAW` 是逐行 JSON，每行一个问题、一张实际存在的图片，无需旧答案。图片路径相对**仓库根目录**。先准备四条有充分回答空间的问题。示例（图片需自行准备）：

```json
{"id":"qa-001","image":"data/images/001.jpg","question":"请描述画面中的设备，说明判断依据。"}
```

## 3. 共用一次答案重生成

```bash
CUDA_VISIBLE_DEVICES=2 python "$GUIDE/regenerate.py" \
    --target "$TARGET" --input "$RAW" --output "$DATA" --max-new-tokens 1024
```

本命令在本机启动 SGLang 目标模型，按“文本、图片”顺序提交，温度为 0，不发送旧答案。保存原图引用、模板参数、原始 prompt IDs 和响应 IDs，用 PR785 的单轮校验逻辑检查后再发布 JSONL。截断、空答案或模板不匹配会报错；失败后检查 `.partial`，用**新输出文件名**重跑。正式数据需确保图像展开后的长度和回答长度适合训练配置；仅有很短回答的样本可能不满足训练锚点条件。

## 4. 按顺序训练、导出

```bash
# 示例：一张卡训练，一张卡独占目标模型采集。卡号为物理 GPU 编号。
for algo in dflash2 dflash dspark; do
    python "$GUIDE/train.py" "$algo" --target "$TARGET" --init "$INIT" \
        --data "$DATA" --sglang "$SG" --out "$RUNS" \
        --train-gpus 0 --capture-gpu 2 --steps 2 --epochs 1 || break
done
```

每种算法的入口相同：先从本目录的 `base.yaml` 渲染 `train.yaml`，再执行 `python -m specforge.cli train -c … --role auto`、`python -m specforge.cli export --to hf …` 和 serving 配置归一化。DFlash2 用 `strategy=dflash` 加 B9 卷积、selector 和 DPaCE；DFlash 用 B16 DFlash 目标；DSpark 用自己的策略、Markov 头及置信度目标。三者均使用 interleaved mRoPE [24,20,20]。

每项打印 `TRAIN_EXPORT_OK` 才算完成。结果位于 `$RUNS/<算法>/exported/`，可以整目录搬走；继续训练所需的完整 checkpoint 在同级 `output/`。正式训练请换新的 RUNS，按样本量、全局 batch 和期望 epoch 设置 `--steps`、`--epochs`，保证实际可用样本能走到指定的导出步；多卡可传 `--train-gpus 0,1`，采集卡不能重叠。示例的两步和小缓存是冒烟设置。

## 5. 验证导出权重

训练结束并释放显存后，重用同一份数据的第一条图文问题，**只提交用户提示和图片**，每次只加载一个草稿。

```bash
for algo in dflash2 dflash dspark; do
    CUDA_VISIBLE_DEVICES=2 python "$GUIDE/verify_export.py" \
        --target "$TARGET" --draft "$RUNS/$algo/exported" --data "$DATA" || break
done
```

发布时保留本目录（含 runtime）、仓库 configs、scripts、patches 以及修改过的 specforge 代码，不要只上传本说明；无需上传 venv、third_party 工作副本、图像、缓存、输出或模型权重。
