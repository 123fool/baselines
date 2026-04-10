# 近两年文献综述：扩散模型 × 脑影像 × MCI 纵向预测

> 调研日期: 2026-04-09
> 覆盖范围: 2024-2026 年发表或预印的相关论文
> 调研方法: Google Scholar + GitHub 代码搜索

---

## 一、核心对标论文 (直接竞争/互补)

### 1. BrLP — Brain Latent Progression [基线模型]
- **作者**: Puglisi, Alexander, Ravì
- **发表**: MICCAI 2024 (Oral, Best Paper 提名) → MedIA 2025 (IF=11.8)
- **核心**: 3D KL-AutoEncoder + Latent Diffusion UNet + ControlNet + Leaspy DCM 辅助模型
- **数据**: ADNI 1/2/3/GO + OASIS-3 + AIBL，11,730 张 T1w MRI，2,805 名被试
- **引用**: 51 (MICCAI) + 18 (MedIA)
- **代码**: https://github.com/LemuelPuglisi/BrLP (123 stars)
- **论文**: https://arxiv.org/abs/2405.03328

### 2. AG-LDM — Anatomically Guided Latent Diffusion Model
- **作者**: Wan, Jafrasteh, Adeli, Zhang, Zhao (Stanford)
- **发表**: arXiv:2601.14584, 2026-01 (尚未正式发表)
- **核心**: 分割引导框架 (WarpSeg)，不需要 ControlNet 或辅助模型，直接在输入层融合基线解剖、噪声随访和临床协变量
- **关键发现**:
  - 匹配或超越 BrLP，体积误差降低 15-20%
  - 对时间和临床协变量的利用率比 BrLP 高 31.5x
  - 在 31,713 ADNI 纵向对上训练，OASIS-3 零样本验证
- **代码**: 暂未公开
- **论文**: https://arxiv.org/abs/2601.14584

### 3. TADM — Temporally-Aware Diffusion Model [MICCAI 2024 版]
- **作者**: Litrico, Guarnera, Giuffrida, Ravì, Battiato
- **发表**: MICCAI 2024
- **核心**: 2D 切片级时间感知扩散，引入 Brain-Age Estimator 作为时间条件
- **引用**: 21
- **代码**: https://github.com/MattiaLitrico/TADM-Temporally-Aware-Diffusion-Model-for-Neurodegenerative-Progression-on-Brain-MRI (14 stars)
- **论文**: https://arxiv.org/abs/2406.12411

### 4. TADM-3D — Temporally-Aware Diffusion Model with Bidirectional Temporal Regularisation [扩展版]
- **作者**: Litrico, Guarnera, Giuffrida, Ravì, Battiato
- **发表**: Computerized Medical Imaging and Graphics (CMIG), 2025
- **核心**: 将 TADM 扩展到 3D，增加双向时间正则化 (前向 t1→t2 + 反向 t2→t1)
- **关键改进**: 预训练 Brain-Age Estimator 提供更精确的时间条件，双向一致性损失保证时序合理性
- **引用**: 1
- **代码**: 同 TADM 仓库 (待确认 3D 版本是否发布)
- **论文**: https://doi.org/10.1016/j.compmedimag.2025... (Elsevier)

### 5. Vanderbilt Replication — BrLP BLSA 复现
- **作者**: McMaster, Puglisi, Gao et al. (Vanderbilt University)
- **发表**: SPIE Medical Imaging 2025
- **核心**: 在 Baltimore Longitudinal Study of Aging (BLSA) 数据集上独立复现 BrLP
- **引用**: 3
- **论文**: https://doi.org/10.1117/12.3047135

---

## 二、重要参考论文 (方法借鉴)

### 6. 3D MedDiffusion — 3D Medical Latent Diffusion Model
- **作者**: ShanghaiTech IMPACT Lab
- **发表**: IEEE TMI 2025
- **核心**: Patch-Volume AE (VQ-VAE) + BiFlowNet (DiT + UNet 混合架构)，面向 3D 医学图像可控生成
- **与本研究的关系**: AE 训练策略（真 3D MedicalNet 感知损失）和 DiT 增强思路的来源
- **代码**: https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion (122 stars)

### 7. SADM — Sequence-Aware Diffusion Model
- **作者**: UBC TEA Lab
- **发表**: IPMI 2023
- **核心**: 将序列历史信息注入扩散模型条件，用于纵向医学影像生成
- **局限**: 2D 切片级，直接用于 3D 需大量调整
- **代码**: https://github.com/ubc-tea/SADM-Longitudinal-Medical-Image-Generation (72 stars)

### 8. TaDiff-Net — Treatment-Aware Diffusion Probabilistic Model
- **作者**: Liu, Fuster-Garcia, Hovden et al.
- **发表**: IEEE TMI 2025
- **核心**: 治疗感知的纵向 MRI 生成 + 弥漫性胶质瘤生长预测
- **与本研究的关系**: 条件设计参考（如何将外部临床信息注入扩散模型）
- **代码**: https://github.com/samleoqh/TaDiff-Net (17 stars)

### 9. Loci-DiffCom — Longitudinal Consistency-Informed Diffusion Model
- **作者**: Zhu, Tao et al.
- **发表**: MICCAI 2024
- **核心**: 纵向一致性引导的 3D 婴儿脑 MRI 补全
- **与本研究的关系**: 纵向一致性建模思路
- **引用**: 6
- **代码**: 暂未公开

### 10. USB — Unified Synthetic Brain Framework
- **作者**: Wang, Liu
- **发表**: arXiv:2512.00269, 2025-12
- **核心**: 双向病理-健康脑生成与编辑统一框架
- **代码**: 暂未公开

### 11. Diffusion with a Linguistic Compass
- **作者**: Tang, Li, Zhang, Zhang
- **发表**: arXiv:2506.05428, 2025
- **核心**: 语言引导的扩散模型生成临床合理的未来 sMRI 表示，用于早期 MCI 转化预测
- **与本研究的关系**: 直接面向 MCI 转化预测，是最直接的竞争/互补工作
- **代码**: 暂未公开

### 12. MRExtrap — Longitudinal Aging of Brain MRIs using Linear Modeling in Latent Space
- **作者**: Kapoor, Macke, Baumgartner
- **发表**: arXiv:2508.19482, 2025
- **核心**: 用潜空间线性建模做脑 MRI 纵向外推，方法极简
- **代码**: 暂未公开

### 13. Conditional Latent Diffusion for Irregularly Spaced Longitudinal Data
- **作者**: Mouadden, Laousy, Marini, Ong et al.
- **发表**: MICCAI 2025
- **核心**: 处理不规则时间间隔的条件潜扩散模型
- **与本研究的关系**: 实际临床数据中随访间隔往往不规则

### 14. Development-Driven Diffusion Model (DDM)
- **作者**: Zhang, Chen, Huang, Zhu et al.
- **发表**: IEEE TMI 2024
- **核心**: 发育驱动扩散模型用于胎儿脑 MRI 纵向预测（无配对数据）
- **引用**: 10

---

## 三、综述类论文

### 15. Early Detection of AD Using Generative Models: A Review of GANs and Diffusion Models
- **发表**: Algorithms, 2025
- **引用**: 11

### 16. Generative AI in Medical Imaging: Foundations, Progress, and Clinical Translation
- **发表**: arXiv:2508.09177, 2025
- **引用**: 5

### 17. Shape Modeling of Longitudinal Medical Images: From Diffeomorphic Metric Mapping to Deep Learning
- **发表**: Frontiers in AI, 2025

---

## 四、文献图谱总结

```
                    ┌─────────────────────────────────────┐
                    │    扩散模型 × 脑影像 × 纵向预测      │
                    └───────────────┬─────────────────────┘
                                    │
            ┌───────────────────────┼──────────────────────┐
            │                       │                      │
    ┌───────▼──────┐      ┌────────▼───────┐     ┌────────▼──────────┐
    │  3D纵向预测   │      │  2D纵向预测     │     │  条件/生成增强     │
    │  (核心赛道)   │      │  (早期工作)     │     │  (技术借鉴)       │
    └───────┬──────┘      └────────┬───────┘     └────────┬──────────┘
            │                      │                      │
    BrLP (2024)            SADM (2023)           3D MedDiffusion (2025)
    AG-LDM (2026)          TADM (2024)           TaDiff-Net (2025)
    TADM-3D (2025)         Ling.Compass(2025)    USB (2025)
    Loci-DiffCom(2024)     MRExtrap (2025)       DDM (2024)
```

---

## 五、关键趋势

1. **从 2D 到 3D**: SADM (2D, 2023) → BrLP (3D, 2024) → TADM-3D (3D, 2025)，3D 已成主流
2. **从复杂到简洁**: BrLP (4 阶段训练) → AG-LDM (端到端)，趋向简化流水线
3. **解剖一致性**: AG-LDM 的 WarpSeg 和 BrLP 的 SynthSeg 都强调分割引导
4. **时间建模增强**: TADM-3D 的双向时间正则化代表了时间建模的最新进展
5. **MCI 特异化**: Linguistic Compass (2025) 是第一个直接面向 MCI 转化预测的扩散模型工作
6. **条件注入多样化**: 从简单的类别标签 → 脑区体积 → 治疗信息 → 语言引导
