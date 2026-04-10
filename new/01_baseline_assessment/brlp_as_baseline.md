# BrLP 作为 MCI 纵向预测基线模型的可行性评估

> 评估日期: 2026-04-09
> 评估基于: BrLP 源码分析 + 近两年文献调研

---

## 一、结论：BrLP 非常适合作为基线模型

**可行性评分: ★★★★★ (5/5)**

BrLP 是目前 3D 脑 MRI 纵向预测领域公认的 SOTA 方法，完全适合作为 MCI 纵向预测研究的基线模型。

---

## 二、支撑理由

### 2.1 学术认可度极高

| 指标   | 详情                                                                           |
| ------ | ------------------------------------------------------------------------------ |
| 发表   | MICCAI 2024 Oral (top 4%) + Medical Image Analysis 2025 (IF=11.8)              |
| 荣誉   | MICCAI 2024 Best Paper 提名 (top <1%)，MICCAI 2025 MedIA Best Paper Award 亚军 |
| 引用   | MICCAI 版本 51 次引用，MedIA 版本 18 次引用 (截至 2026-04)                     |
| 复现   | Vanderbilt University 在 BLSA 数据集独立复现成功 (SPIE 2025)                   |
| 被对比 | AG-LDM (arXiv 2026)、TADM-3D (CMIG 2025) 均以 BrLP 作为主要对比基线            |

### 2.2 架构适合 MCI 研究

从源码分析可知，BrLP 的模块化架构天然适合"外科手术式"改进：

```
AutoencoderKL (潜空间编码)
    ↓
DiffusionModelUNet (潜空间扩散)
    ↓
ControlNet (个体化条件注入: 基线 latent + age)
    ↓
Auxiliary DCM (辅助模型: 分 CN/MCI/AD 三组建模脑区体积变化)
```

每个模块可以独立替换或增强，不需要全系统重构。

### 2.3 数据流程完全覆盖 MCI

BrLP 的数据组织方式直接支持 MCI 研究：
- 支持 ADNI 1/2/3/GO、OASIS-3、AIBL 三大 AD/MCI 公开数据集
- 诊断编码: CN=0, MCI=0.5, AD=1
- 辅助模型已对 MCI 组独立建模 (dcm_mci.json)
- 条件变量包含海马体、杏仁核、脑室等 MCI 核心标志物

### 2.4 代码成熟度高

| 维度         | 评估                                       |
| ------------ | ------------------------------------------ |
| GitHub Stars | 123 (LemuelPuglisi/BrLP)                   |
| 代码质量     | 清晰的模块划分，文档完善                   |
| 依赖         | MONAI GenerativeModels + PyTorch，主流框架 |
| CLI 工具     | 提供 `brlp` 命令行推理工具                 |
| 预训练模型   | 已公开 AE + UNet + ControlNet + DCM 权重   |

---

## 三、BrLP 用于 MCI 纵向预测的已知局限

以下是从源码和论文中提取的、与 MCI 研究直接相关的局限性：

### 3.1 MCI 异质性未被充分建模

**问题**: 辅助模型 (train_aux.py) 将 MCI 视为一个整体 (last_diagnosis == 0.5)，但 MCI 的核心挑战在于:
- pMCI (进展型): 1-3 年内转化为 AD
- sMCI (稳定型): 长期保持 MCI 状态

当前条件向量中 diagnosis=0.5 无法区分这两类。

**源码证据**: `train_aux.py` L79-82:
```python
mci_df = train_df[train_df.last_diagnosis == 0.5]
mci_data = prepare_dcm_data(mci_df)
mci_leaspy = train_leaspy(mci_data, 'mci', logs_path)
```

### 3.2 条件向量维度有限

**问题**: 条件向量仅 8 维 (age, sex, diagnosis, 5 个脑区体积)，缺少:
- 认知评分 (MMSE, ADAS-Cog)
- ApoE4 基因型
- CSF 生物标志物 (Aβ42, tau, p-tau)
- MCI 转化概率

**源码证据**: `const.py` L68-76:
```python
CONDITIONING_VARIABLES = [
    "age", "sex", "diagnosis",
    "cerebral_cortex", "hippocampus", "amygdala",
    "cerebral_white_matter", "lateral_ventricle",
]
```

### 3.3 AutoEncoder 对细微变化不够敏感

**问题**: AE 的所有 attention_levels 为 False，网络容量限制了对 MCI 早期细微萎缩的捕捉。

**源码证据**: `networks.py` L46:
```python
attention_levels=(False, False, False, False),
```

### 3.4 感知损失使用切片级近似

**问题**: 使用 fake3D perceptual loss (2D 切片抽样 20%)，对 3D 结构连续性的约束有限。

### 3.5 时间建模隐式

**问题**: 时间信息仅通过 age 条件变量隐式编码，无显式时间连续性约束。多时间点预测间缺乏双向一致性保证。

---

## 四、与竞争方法的对比

| 方法         | 类型                          |  3D   | 个体化 | 纵向  | MCI 适配 | 代码  | 适合作基线 |
| ------------ | ----------------------------- | :---: | :----: | :---: | :------: | :---: | :--------: |
| **BrLP**     | Latent Diffusion + ControlNet |   ✅   |   ✅    |   ✅   |    ⚠️     |   ✅   | **✅ 最佳** |
| AG-LDM       | Latent Diffusion + WarpSeg    |   ✅   |   ✅    |   ✅   |    ⚠️     |   ❌   |  ⚠️ 无代码  |
| TADM/TADM-3D | Diffusion + Brain-Age         |   ✅   |   ✅    |   ✅   |    ❌     |   ✅   |   ✅ 可选   |
| SADM         | Sequence-Aware Diffusion      | ❌(2D) |   ⚠️    |   ✅   |    ❌     |   ✅   |    ❌ 2D    |
| TaDiff-Net   | Treatment-Aware DDPM          |   ✅   |   ⚠️    |   ✅   |    ❌     |   ✅   | ❌ 面向肿瘤 |

---

## 五、推荐的使用方式

1. **直接使用 BrLP 作为基线**: 下载预训练权重，在你的 MCI 数据集上测试
2. **以 BrLP 为基础进行改进**: 针对 MCI 异质性、条件引导、时间建模等方面做针对性增强
3. **对比实验设计**: 将 BrLP 原始结果作为 baseline，依次消融各改进模块

---

## 六、关键参考

1. Puglisi et al., "Brain Latent Progression", MICCAI 2024 / MedIA 2025
2. McMaster et al., "A technical assessment of latent diffusion for AD progression", SPIE 2025
3. Wan et al., "AG-LDM: Anatomically Guided Latent Diffusion", arXiv:2601.14584, 2026
4. Litrico et al., "TADM-3D: Temporally-aware Diffusion Model with Bidirectional Temporal Regularisation", CMIG 2025
