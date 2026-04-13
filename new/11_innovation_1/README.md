# Innovation 1: MCI 转化动态条件引导

## 核心思想

BrLP 的 ControlNet 空间条件仅包含 starting latent (3ch) + age (1ch) = 4 通道。
本创新扩展为 6 通道，新增：

- **海马萎缩速率** (hippocampal atrophy rate): 量化海马体积变化速度
- **脑室扩张速率** (ventricular expansion rate): 量化脑室膨胀速度

这两个指标是 MCI→AD 转化最核心的影像生物标志物，为扩散模型提供了
"大脑应该以什么速度退化"的显式指导。

## 技术细节

| 参数                                 | 基线 (BrLP)     | 本创新          |
| ------------------------------------ | --------------- | --------------- |
| `conditioning_embedding_in_channels` | 4               | **6**           |
| `cross_attention_dim`                | 8               | 8 (不变)        |
| UNet 结构                            | 不变            | 不变 (冻结)     |
| 训练方式                             | ControlNet only | ControlNet only |

### 新增条件通道

训练时：

```
channel 4 (原 age): starting_age
channel 5 (新增): hippocampal_atrophy_rate = (start_hipp - followup_hipp) / time_delta
channel 6 (新增): ventricular_expansion_rate = (followup_vent - start_vent) / time_delta
```

推理时：

- atrophy rate 由辅助模型 (Leaspy) 的预测轨迹估算

## 目录结构

```
11_innovation_1/
├── README.md
├── src/
│   └── mci_conditioning.py    # 条件计算与网络修改
├── scripts/
│   ├── prepare_mci_conditions.py  # 数据预处理 (计算 atrophy rates)
│   ├── train_controlnet_mci.py    # ControlNet 训练
│   └── evaluate_mci.py            # 评估脚本
├── configs/
│   └── train_mci.yaml
├── train.sh
└── eval.sh
```

## 运行方式

```bash
# 1. 数据预处理
bash train.sh prepare

# 2. 训练
bash train.sh train

# 3. 评估
bash eval.sh [epoch]
```
