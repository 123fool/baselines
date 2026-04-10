# MCI 纵向预测研究：改进路线图与差异化策略

> 日期: 2026-04-09
> 基线: BrLP (MICCAI 2024 / MedIA 2025)
> 目标: 在 BrLP 基础上，针对 MCI 纵向预测进行有针对性的改进

---

## 一、差异化定位

### 当前竞争格局

| 方法                | 定位                                |  MCI 特异性  |
| ------------------- | ----------------------------------- | :----------: |
| BrLP                | 通用脑 MRI 纵向预测                 |      ❌       |
| AG-LDM              | 更高效的通用脑 MRI 纵向预测         |      ❌       |
| TADM-3D             | 通用脑退行性预测 + 双向时间建模     |      ❌       |
| Linguistic Compass  | MCI 转化预测 (2D 表示级)            | ✅ 但非影像级 |
| **你的工作 (目标)** | **MCI 特异化的 3D 脑 MRI 纵向预测** | **✅ 影像级** |

### 你的核心卖点

> **第一个将扩散模型纵向预测框架按 MCI 病理特点进行全栈特异化的工作**
>
> 从潜空间编码 → 条件引导 → 时间建模 → 损失函数四个层面，
> 系统性地解决 BrLP 在 MCI 预测中的不足。

---

## 二、改进实施计划

### 第一阶段: AE 层面改进 (1-2 周)

**目标**: 让潜空间对 MCI 相关结构变化更敏感

#### 改进 A: 海马体区域加权损失
- **修改文件**: `scripts/training/train_autoencoder.py`
- **做法**: 利用 SynthSeg 分割 mask，在 L1 重建损失中对海马体、杏仁核、脑室区域加权 (2-5x)
- **工作量**: ~1 天
- **风险**: 极低

#### 改进 B: 3D 感知损失替换
- **修改文件**: `scripts/training/train_autoencoder.py`
- **做法**: 将 MONAI 的 `PerceptualLoss(fake_3d_ratio=0.2)` 替换为 MedicalNet ResNet-10 3D 感知损失
- **依赖**: 3D MedDiffusion 中的 `MedicalNetPerceptual` 模块
- **工作量**: ~2 天
- **风险**: 低

#### 改进 C: 频域一致性损失 (可选)
- **做法**: 增加拉普拉斯金字塔多尺度损失或 FFT 频域约束
- **工作量**: ~1 天
- **风险**: 低

**第一阶段评估指标**:
- AE 重建 SSIM/PSNR (全脑 + 海马体亚区)
- 海马体体积估计误差 (与 SynthSeg 真值对比)

---

### 第二阶段: 条件引导改进 (2-3 周)

**目标**: 让扩散模型理解 MCI 转化动态

#### 改进 D: MCI 转化条件增强
- **修改文件**: `src/brlp/const.py`, `scripts/training/train_controlnet.py`, `scripts/training/train_diffusion_unet.py`
- **做法**:
  1. 在条件向量中增加新的标量条件:
     - `conversion_probability`: MCI→AD 转化概率 (基于 baseline 数据的预测)
     - `hippocampal_atrophy_rate`: 海马体年化萎缩率估计
     - `cognitive_score`: 标准化认知评分 (MMSE 或 ADAS-Cog)
     - `apoe4_status`: ApoE4 基因型 (0/1/2 个等位基因)
  2. 修改 `cross_attention_dim` 从 8 → 12
  3. 修改 UNet 和 ControlNet 的 `cross_attention_dim` 参数
- **工作量**: ~3-5 天
- **风险**: 中 (需要重新训练 UNet 和 ControlNet)

#### 改进 E: 双向时间正则化 (需要与 TADM-3D 差异化)
- **修改文件**: `scripts/training/train_controlnet.py`
- **做法**:
  1. 在 B.csv 数据加载时，同时生成正向 (A→B) 和反向 (B→A) 训练对
  2. 增加双向一致性损失
  3. **差异化**: 将双向约束与 MCI 转化条件结合——pMCI 组给予更强的前向约束 (因为退化方向更确定)，sMCI 组给予更强的双向对称约束 (因为变化较缓慢)
- **工作量**: ~5 天
- **风险**: 中

**第二阶段评估指标**:
- MCI 子组 (pMCI, sMCI) 上的体积预测 MAE
- 与 BrLP 原始在 MCI 子组上的对比
- 生成影像的 SSIM/PSNR
- 临床协变量利用率 (参考 AG-LDM 的评估方式)

---

### 第三阶段: 消融实验 (1-2 周)

**目标**: 证明每个改进模块的独立贡献

#### 消融设计

| 实验             |   A   |   B   |   C   |   D   |   E   |
| ---------------- | :---: | :---: | :---: | :---: | :---: |
| BrLP (baseline)  |       |       |       |       |       |
| + 区域加权       |   ✅   |       |       |       |       |
| + 3D 感知损失    |       |   ✅   |       |       |       |
| + 频域约束       |       |       |   ✅   |       |       |
| + MCI 条件增强   |       |       |       |   ✅   |       |
| + 双向时间正则化 |       |       |       |       |   ✅   |
| Full model       |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |

---

## 三、数据准备要点

### 需要准备的额外数据

| 数据               | 来源                                  | 用于   |
| ------------------ | ------------------------------------- | ------ |
| pMCI/sMCI 标签     | ADNI (基于 36 个月 followup 转化情况) | 改进 D |
| MMSE/ADAS-Cog 评分 | ADNI 临床数据                         | 改进 D |
| ApoE4 基因型       | ADNI 遗传数据                         | 改进 D |
| SynthSeg 分割 mask | BrLP 预处理已包含                     | 改进 A |

### pMCI/sMCI 定义

- **pMCI**: 在 baseline 时诊断为 MCI，36 个月内转化为 AD
- **sMCI**: 在 baseline 时诊断为 MCI，36 个月内仍为 MCI 或恢复为 CN

---

## 四、论文写作建议

### 论文标题方向

> "MCI-Aware Brain Latent Progression: Tailoring Latent Diffusion for Mild Cognitive Impairment Longitudinal Prediction"

### 核心贡献表述

1. 针对 MCI 纵向脑 MRI 预测任务，对 BrLP 的潜空间学习目标进行了任务特异化增强
2. 提出 MCI 转化动态条件引导机制，将 pMCI/sMCI 异质性编码到扩散模型条件中
3. 设计了与 MCI 病理特点相适配的双向时间正则化策略
4. 在 ADNI 数据集上系统验证了各改进模块在 MCI 子组上的效果

### 需要对标的方法 (Related Work & Experiments)

| 对标方法                  | 对标维度                  |  代码可得性  |
| ------------------------- | ------------------------- | :----------: |
| BrLP (2024)               | 主要基线                  |      ✅       |
| TADM-3D (2025)            | 双向时间建模对比          |  ✅ (需确认)  |
| AG-LDM (2026)             | 条件利用率 + 体积误差对比 | ❌ 需自己实现 |
| Linguistic Compass (2025) | MCI 转化预测对比          |      ❌       |

---

## 五、风险清单

| 风险                              | 概率  | 影响  | 缓解措施                                   |
| --------------------------------- | :---: | :---: | ------------------------------------------ |
| AG-LDM 在正式发表前公开代码       |  中   |  高   | 强调 MCI 特异性是你的差异化                |
| TADM-3D 发布 MCI 特异版本         |  低   |  高   | 尽快出结果，抢占 MCI 赛道                  |
| MCI 条件增强效果不显著            |  中   |  中   | 确保消融实验设计合理，至少有 AE 改进兜底   |
| ControlNet 维度修改导致训练不稳定 |  低   |  中   | 从少量新条件开始逐步增加，密切监控训练曲线 |
