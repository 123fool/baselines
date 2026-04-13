# Combined Innovation 1+2: 6ch ControlNet + BTR

## 概述

将两个已独立验证有效的创新点组合：

- **Innovation 1**: 6通道空间条件 (starting_z + age + atrophy_rate + vent_rate)
- **Innovation 2**: 双向时间正则化 BTR (L_fwd + 0.5 × L_bwd)

## 独立结果回顾

| 方法               | SSIM   | PSNR  | MAE    | ROI_SSIM |
| ------------------ | ------ | ----- | ------ | -------- |
| Baseline           | 0.8990 | 25.22 | 0.0356 | 0.7969   |
| Innovation 1 (6ch) | 0.9153 | 26.54 | 0.0290 | 0.8116   |
| Innovation 2 (BTR) | 0.9282 | 27.30 | 0.0262 | 0.8277   |

## 关键设计

### 前向 (A→B)

- 6ch condition: `[starting_z, starting_age, atrophy_rate, vent_rate]`
- context: `[followup_age, sex, followup_diag, ...]`
- Target: predict noise in followup_z

### 反向 (B→A)

- 6ch condition: `[followup_z, followup_age, -atrophy_rate, -vent_rate]`
- context: `[starting_age, sex, starting_diag, ...]`
- Target: predict noise in starting_z
- 注意: 反向率取负号（疾病进展方向反转）

### 初始化

- 从 baseline 4ch ControlNet → 6ch (新通道 zero-init)
- 使用改进 AE (autoencoder-ep-2.pth)

## 文件结构

```
scripts/
  train_controlnet_6ch_btr.py  — 训练脚本
  evaluate_6ch_btr.py          — 评估脚本
src/
  mci_conditioning.py          — 6ch ControlNet 初始化 + 条件构建
  bidirectional_temporal.py    — BTR 模块 (仅作参考)
_start_training.py             — 一键上传+训练
```

## 运行

```bash
# 本地: 上传代码到服务器并启动训练
python _start_training.py

# 服务器: 手动运行
bash train.sh        # 训练
bash eval.sh 4       # 评估 epoch 4
```
