# 创新点 4+5 联合重训实验（方案 A）

## 目标

在 Innovation 4 微调后的 AE（Decoder 优化）基础上，重新训练 Innovation 5 的 ControlNet。

这样 ControlNet 的噪声预测目标会与 Inn4 的 Decoder 特征分布对齐，解决上次联合推理时的 **Decoder Mismatch 问题**。

## 与上次联合实验的区别

| 对比项          | 上次（联合推理）                        | 本次（联合重训）                     |
| --------------- | --------------------------------------- | ------------------------------------ |
| AE 来源         | Inn4 AE (ep-4)                          | Inn4 AE (ep-4) ✅ 相同               |
| ControlNet 来源 | Inn5 ControlNet (训练时用基线 AE)       | **本次重新训练**（训练时用 Inn4 AE） |
| 潜空间对齐      | ❌ ControlNet 隐含了基线 Decoder 的分布 | ✅ ControlNet 与 Inn4 Decoder 一致   |
| 预期效果        | 无叠加收益（已验证）                    | **预期超越 Inn5 单独使用**           |

## Checkpoint 路径

| 模型          | 路径                                                    |
| ------------- | ------------------------------------------------------- |
| Inn4 AE       | `/output/innovation_4/ae_training/autoencoder-ep-4.pth` |
| Baseline UNet | `/brlp-train/pretrained/latentdiffusion.pth`            |
| 训练输出      | `/output/combined_retrain/controlnet/cnet-ep-*.pth`     |

## 运行方式

```bash
# 训练（约 1 小时）
bash /home/wangchong/data/fwz/code/combined_retrain/train.sh

# 评估 (自动在训练完成后运行)
bash /home/wangchong/data/fwz/code/combined_retrain/eval.sh
```

## 预期指标（参考上轮实验）

| 指标         | Baseline | Inn4 v1    | Inn5 v2    | 联合推理 | **联合重训（预期）** |
| ------------ | -------- | ---------- | ---------- | -------- | -------------------- |
| Overall SSIM | 0.9015   | 0.9081     | **0.9145** | 0.9123   | **>0.9145**          |
| ROI SSIM     | 0.7983   | **0.8184** | 0.8141     | 0.8059   | **>0.8184**          |
