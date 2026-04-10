# 3D MedDiffusion → BrLP 集成指南

## 概述

本文档描述如何将 3D MedDiffusion 的核心技术组件迁移到 BrLP 框架中，
用于改进 MCI 纵向脑 MRI 预测任务。

---

## 1. 技术桥接点

### 1.1 可迁移组件

| 3D MedDiffusion 组件 | BrLP 目标位置 | 迁移难度 | 价值 |
|---|---|---|---|
| DiTBlock (adaLN-Zero) | UNet bottleneck 注入 | ⭐⭐⭐ | 全局空间注意力 |
| PatchEmbed_Voxel | DiT 增强模块前端 | ⭐⭐ | 3D patch 化 |
| MedicalNetPerceptual | AE 训练感知损失 | ⭐ | 真 3D 感知损失 |
| cosine beta schedule | 噪声调度替换 | ⭐ | 平滑训练 |
| Feature Fusion | UNet-DiT 特征融合 | ⭐⭐⭐ | 多尺度融合 |

### 1.2 不可直接迁移的组件

| 组件 | 原因 |
|---|---|
| BiFlowNet 完整架构 | 非条件生成 vs. 条件进展预测，架构不兼容 |
| PatchVolume AE | VQ-VAE vs. KL-VAE，codebook 机制不同 |
| 类别嵌入 | BrLP 用 cross-attention 条件化，不同机制 |

---

## 2. 详细迁移方案

### 2.1 DiT 增强模块（创新点 3）

**源码位置**: `3D-MedDiffusion-main/ddpm/BiFlowNet.py` → `DiTBlock` 类

**迁移步骤**:

1. **提取 DiTBlock**: 从 BiFlowNet.py 提取 `DiTBlock` 和 `PatchEmbed_Voxel` 类
2. **适配维度**: 
   - BiFlowNet 的 hidden_size=1536, BrLP UNet bottleneck channel=768
   - 需要缩小 DiT 维度: 1536 → 768
3. **适配条件化**:
   - BiFlowNet: time_mlp + class_embed + resolution_embed → cond
   - BrLP: timestep + 10维协变量 → 通过 DiTConditionAdapter 映射
4. **注入方式**: 使用零初始化 gate 的残差连接

**关键代码差异**:

```python
# BiFlowNet 原始 (非条件/类别条件)
cond = self.time_mlp(t) + self.class_embed(y) + self.res_embed(r)
# 适配为 BrLP (时间步+协变量条件)
time_emb = timestep_embedding(t)
context_flat = context.squeeze(1)  # (B, 10)
cond = adapter(time_emb, context_flat)  # (B, 256)
```

**参见**: `BrLP-main/new/code/intermediate/dit_enhancement.py`

### 2.2 MedicalNet 3D 感知损失（创新点 4）

**源码位置**: `3D-MedDiffusion-main/AutoEncoder/model/MedicalNetPerceptual.py`

**迁移步骤**:

1. **复制模型**: 将 `warvito_MedicalNet-models_main/` 目录引入
2. **适配输入**: 
   - 3D MedDiffusion: 多类别条件生成
   - BrLP: 单通道脑 MRI，输入尺寸 (120, 144, 120)
3. **替换 BrLP 的 fake3D 感知损失**: 
   - 原始: squeeze + fake3D → 2D VGG 切片感知损失
   - 改进: 直接用 MedicalNet ResNet-10 的 3D 特征

```python
# BrLP 原始 (train_autoencoder.py L-约100)
perceptual_loss = PerceptualLoss(
    spatial_dims=3, network_type="squeeze", 
    fake_3d_ratio=0.2
)
# 替换为 (train_autoencoder.py)
from MedicalNetPerceptual import MedicalNetPerceptualSimilarity
perceptual_loss_3d = MedicalNetPerceptualSimilarity()
```

**预训练权重**: `3D-MedDiffusion-main/warvito_MedicalNet-models_main/medicalnet/resnet_10_23dataset.pth`

**参见**: `BrLP-main/new/code/intermediate/frequency_loss.py`

### 2.3 噪声调度优化

**源码位置**: `3D-MedDiffusion-main/ddpm/BiFlowNet.py` → `cosine_beta_schedule()`

**迁移说明**:

BrLP 使用 `scaled_linear_beta` 调度；3D MedDiffusion 使用 cosine 调度。
Cosine 调度在训练初期和末期更平滑，可能对细微变化的学习更有利。

```python
# 当前 BrLP
scheduler = DDPMScheduler(
    schedule='scaled_linear_beta',
    beta_start=0.0015, beta_end=0.0205
)
# 可实验: cosine schedule
# 注意: MONAI DDPMScheduler 不直接支持 cosine
# 需要手动设置 betas
import numpy as np
def cosine_betas(T=1000, s=0.008):
    steps = np.arange(T + 1) / T
    alphas_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return np.clip(betas, 0, 0.999)
```

---

## 3. 风险与注意事项

### 3.1 框架差异
- 3D MedDiffusion: PyTorch Lightning + 自定义训练
- BrLP: MONAI Generative + 原生 PyTorch 训练循环
- **需要额外适配 MONAI 的 ControlNet 接口**

### 3.2 任务差异
- 3D MedDiffusion: **无条件/类别条件** 的 3D 医学图像生成
- BrLP: **纵向条件** 的脑 MRI 进展预测
- DiT 模块需要接受 **时间步 + 协变量** 而非 **类别标签**

### 3.3 计算开销
- DiT 全局自注意力: O(N²) 其中 N = patch 数量
  - BrLP bottleneck (4×5×4) → 80 patches → 可接受
  - 若在更高分辨率层使用 → 需要限制 patch 数

### 3.4 MedicalNet 依赖
- 预训练权重: ResNet-10 在 23 个医学图像数据集上预训练
- 文件: `resnet_10_23dataset.pth` (约 45MB)
- License: MIT

---

## 4. 推荐实施优先级

1. **MedicalNet 感知损失** → 最低风险、最容易集成、即刻可用
2. **增强条件向量** → 需要数据预处理但实现简单
3. **双向时间正则化** → 训练损失改进，代码改动中等
4. **DiT 增强模块** → 最复杂，需要修改 MONAI 源码或 hook
5. **Cosine 噪声调度** → 消融实验级别，最后考虑
