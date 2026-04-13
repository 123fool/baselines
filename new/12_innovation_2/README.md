# Innovation 2: 双向时间正则化 (Bidirectional Temporal Regularization)

## 核心思想

BrLP 的 ControlNet 仅训练「正向预测」(t1→t2)，缺少时间一致性约束。
本创新在训练阶段同时学习正向 (A→B) 和反向 (B→A) 预测，并引入双向
时间一致性损失 (Bidirectional Temporal Consistency Loss)。

核心约束：如果模型从 A 生成了 B，那么从 B 也应当能还原 A。

## 技术细节

### 训练损失

$$L_{total} = L_{forward} + \lambda_{btc} \cdot L_{backward}$$

其中：
- $L_{forward} = MSE(\hat{\epsilon}_{A\to B}, \epsilon)$：正向噪声预测（与 BrLP 一致）
- $L_{backward} = MSE(\hat{\epsilon}_{B\to A}, \epsilon')$：反向噪声预测
- $\lambda_{btc}$：双向一致性权重（默认 0.5）

### 反向对的构建

对于每个训练样本 (starting→followup)，我们自动构建反向对：
- 反向 starting_latent = 原 followup_latent
- 反向 followup_latent = 原 starting_latent
- 反向 starting_age = 原 followup_age
- 反向 context = 原 starting 的 covariates

### 推理时双向平均 (可选)

1. 正向预测：A → B̂
2. 可用于多时间点预测的一致性检查

## 参考文献

- Temporally-Aware Diffusion Model for Brain Progression Modelling
  with Bidirectional Temporal Regularisation (2025-09)
- SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image
  Generation (IPMI 2023)

## 目录结构

```
12_innovation_2/
├── README.md
├── src/
│   └── bidirectional_temporal.py   # 双向损失与反向对构建
├── scripts/
│   ├── train_controlnet_btr.py     # ControlNet 双向训练
│   └── evaluate_btr.py             # 评估脚本
├── train.sh
└── eval.sh
```
