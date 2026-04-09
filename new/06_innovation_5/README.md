# Innovation 5: Hippocampal Region Attention Weighting

## 目标

在 BrLP 的 AutoEncoder 和 ControlNet 训练中引入海马体及内嗅皮层的区域加权损失，
使模型更关注 MCI→AD 转化中最关键的脑结构变化。

## 目录结构

```
innovation_5/
├── README.md
├── dashboard/
│   └── app.py                      # 实时监控网页（Flask）
├── src/
│   ├── region_weights.py           # 区域权重图生成
│   └── weighted_losses.py          # 加权损失函数
├── scripts/
│   ├── train_autoencoder_regional.py   # 改进的 AE 训练
│   ├── train_controlnet_regional.py    # 改进的 ControlNet 训练
│   ├── evaluate_regional.py            # 区域级评估
│   └── precompute_weight_maps.py       # 预计算权重图缓存
├── configs/
│   └── train_mci.yaml              # MCI 训练配置
├── changelog.json                  # 修改日志
└── run.sh                          # 一键执行脚本
```

## 核心改动

1. **AE 训练**: L1 loss 替换为区域加权 L1 loss，海马体+杏仁核区域权重 3x
2. **ControlNet 训练**: MSE 噪声预测 loss 替换为潜空间区域加权 MSE
3. **评估**: 新增海马体子区域 SSIM/MAE/体积误差指标

## 使用方式

```bash
# 在服务器上
cd /home/wangchong/data/fwz/code/innovation_5
bash run.sh
```
