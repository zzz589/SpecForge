#!/usr/bin/env python3
import datetime, json, pathlib, shutil

ROOT = pathlib.Path('/HOME/nsccgz_zgchen/nsccgz_zgchen_8/HDD_POOL/xj/experiments/rpm10k-dflash2-dpace-scale-20261110')
SIZES = [3000, 4000, 5000, 6000, 7000, 8000, 8729]
STEPS = {3000:2250,4000:3000,5000:3750,6000:4500,7000:5250,8000:6000,8729:6546}

table = ['# RPM-10K DFlash2 + D-PACE 数据规模实验汇总', '',
         '所有规模均从同一个 RedHatAI DFlash2 b9 Qwen3-VL mRoPE v1 初始化独立训练；不是逐级续训。', '',
         '| 数据量 | Steps | 末10点 loss | 末10点 acc | 末10点 selector loss | 末10点 selector accepted length |',
         '|---:|---:|---:|---:|---:|---:|']
for size in SIZES:
    exp = ROOT / f'n{size}'
    summary = json.loads((exp/'training_curves/summary.json').read_text())
    data = json.loads((exp/'data_manifest.json').read_text())
    ckpt = next((exp/'output').iterdir())
    started = json.loads((exp/'experiment_manifest.json').read_text()).get('status', 'complete')
    completed_at = datetime.datetime.fromtimestamp((ckpt/'exported-draft/model.safetensors').stat().st_mtime).astimezone().isoformat(timespec='seconds')
    loss = summary['loss']['last_10_mean']; acc = summary['acc']['last_10_mean']
    sloss = summary['dflash2/selector/loss']['last_10_mean']
    alen = summary['dflash2/selector/greedy_accepted_length']['last_10_mean']
    table.append(f'| {size} | {STEPS[size]} | {loss:.4f} | {acc:.4f} | {sloss:.4f} | {alen:.4f} |')
    ranges = ', '.join(f'{k}: {v}' for k,v in data['range_distribution'].items())
    text = f'''# n{size} 训练结果说明

- 状态：complete
- 完成时间：{completed_at}
- 数据：{size} 条，3 epochs，global batch 4，optimizer steps {STEPS[size]}
- 初始化：固定 RedHatAI DFlash2 b9 Qwen3-VL mRoPE v1；与其他数据规模独立训练
- 目标：DFlash2 + D-PACE (`loss_type=dpace`, `dpace_alpha=0.5`)；selector alpha 1.0
- 优化：LR 1e-5，cosine，warmup 0.03；anchors 32；block size 9
- GPU：capture 6；trainers 3,5
- 数据 SHA256：`{data['sha256']}`
- 量程分布：{ranges}

## 训练指标

| 指标 | 首点 | 末点 | 末 10 点均值 |
|---|---:|---:|---:|
| loss | {summary['loss']['first']:.4f} | {summary['loss']['last']:.4f} | {loss:.4f} |
| token acc | {summary['acc']['first']:.4f} | {summary['acc']['last']:.4f} | {acc:.4f} |
| selector loss | {summary['dflash2/selector/loss']['first']:.4f} | {summary['dflash2/selector/loss']['last']:.4f} | {sloss:.4f} |
| selector greedy accepted length | {summary['dflash2/selector/greedy_accepted_length']['first']:.4f} | {summary['dflash2/selector/greedy_accepted_length']['last']:.4f} | {alen:.4f} |

训练曲线位于 `training_curves/loss_function.png` 与 `training_curves/training_metrics.png`，原始抽样指标为 `metrics.jsonl`。

## 测试与 badcase

尚未执行独立测试集推理，因此这里不报告测试接受长度、测试准确率或 badcase。训练 acc 不能替代测试结果。后续评测结果应写入 `evaluation/`，badcase 应包含样本 ID、图片、问题、标准答案、模型输出、拒绝位置/首错位置和错误类型。
'''
    (exp/'TRAINING_RESULT.zh-CN.md').write_text(text)
    shutil.copy2(exp/'TRAINING_RESULT.zh-CN.md', ckpt/'TRAINING_RESULT.zh-CN.md')

table += ['', '注意：表中的 accepted length 是训练 batch 上的 selector 指标，不是独立测试集接受长度。测试集结果与 badcase 仍需单独评测。', '']
(ROOT/'SCALE_SUMMARY.zh-CN.md').write_text('\n'.join(table))
