# 创新点 4+5 联合实验

## 核心思路

| 创新点  | 作用位置                              | 关键约束                     |
| ------- | ------------------------------------- | ---------------------------- |
| 创新点4 | AE Decoder精调（3D感知损失+频域约束） | **Encoder冻结** → 潜空间不变 |
| 创新点5 | ControlNet区域加权（海马体/杏仁核）   | 训练时使用预提取的NPZ潜向量  |

**关键洞察**：由于创新点4的Encoder完全冻结，潜表示与基线完全一致。  
创新点5的ControlNet在潜空间做区域加权噪声预测，与Decoder无关。  
因此两者在**推理阶段**可以直接组合，无需任何重新训练。

## Checkpoint

| 模型        | 路径                                                   | 说明                                   |
| ----------- | ------------------------------------------------------ | -------------------------------------- |
| AutoEncoder | `output/innovation_4/ae_training/autoencoder-ep-4.pth` | Inn4 v1最后epoch（SSIM=0.9081）        |
| ControlNet  | `output/innovation_5/controlnet/cnet-ep-3.pth`         | Inn5 v2最优epoch（valid loss_w=0.031） |
| Diffusion   | `brlp-train/pretrained/latentdiffusion.pth`            | 基线UNet（不变）                       |

## 评估数据

- CSV: `output/innovation_5/prepared/B_mci.csv`（50对测试样本）
- 指标: overall SSIM/PSNR/MAE + 海马体/杏仁核/ROI区域SSIM/MAE

## 运行

```bash
bash /home/wangchong/data/fwz/code/combined_4_5/run.sh
```

## 预期结果

| 实验                | Overall SSIM    | ROI SSIM        |
| ------------------- | --------------- | --------------- |
| 基线                | 0.9015          | 0.7983          |
| 创新点4 v1          | 0.9081 (+0.73%) | 0.8184 (+2.52%) |
| 创新点5 v2          | 0.9145 (+1.44%) | 0.8141 (+1.98%) |
| **4+5联合（预期）** | **>0.9145**     | **>0.8184**     |
