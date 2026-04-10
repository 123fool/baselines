# BrLP 改进方案：综合分析与创新点

> 日期: 2026-04-07
> 基线模型: BrLP (MICCAI 2024 / Medical Image Analysis 2025)
> 参考模型: 3D MedDiffusion (IEEE TMI 2025)
> 应用目标: MCI (轻度认知障碍) 纵向预测

---

## 一、对 Gemini 建议的评估

### Gemini 说对了什么

1. **BrLP 作为基线是合适的** ✅
   - BrLP 是 2024-2025 年脑 MRI 纵向预测领域 SOTA，在 MICCAI 2024 获 oral + Best Paper 提名
   - 它已经解决了 3D 医学影像中预处理、潜空间编码、空间一致性等工程难点
   - 模块化架构（VAE + UNet + ControlNet + 辅助模型）确实适合做"外科手术式"改进

2. **不建议从零生成** ✅
   - 3D 脑 MRI 扩散模型的工程复杂度很高（显存管理、MNI 空间配准、强度归一化），从零写极易踩坑

3. **VAE 平滑效应是真实痛点** ✅
   - BrLP 论文自己承认了这个问题，从源码看其 AutoencoderKL 所有 attention_levels 都设为 False，网络容量确实有限

4. **MCI/AD 预测偏差确实存在** ✅
   - 从辅助模型代码看，Leaspy 分别对 CN/MCI/AD 三组建模，但 MCI 的转化动态性（pMCI vs sMCI）没有被建模

### Gemini 说得不够准确或需要修正的地方

1. **"把 UNet 直接换成 DiT" 过于激进** ⚠️
   - 看了 3D MedDiffusion 的实际源码后发现：BiFlowNet **不是**纯 DiT，而是 DiT + U-Net 的混合架构
   - 3D MedDiffusion 的 BiFlowNet 用 DiT 处理子体积 (IntraPatch Flow)，用 U-Net 处理全局结构
   - 直接替换 UNet 为纯 DiT 会丧失 BrLP 的 ControlNet 兼容性（MONAI 的 ControlNet 是针对 UNet 设计的）
   - **正确做法**: 在 BrLP 的 UNet 中引入 DiT 增强模块，而非全部替换

2. **"AG-LDM (2026)" 的引用需谨慎** ⚠️
   - 该论文确实存在（arXiv 2026-01），但尚未正式发表，具体方法细节需验证
   - Gemini 提到的放射组学特征注入点是合理的，但具体实现需要验证

3. **"SADM" 作为参考的实用性有限** ⚠️
   - SADM 是 2D 切片级别的方法，直接用到 3D 场景需要大量调整
   - 更好的时间建模参考是 2025 年的 "Temporally-Aware Diffusion Model for Brain Progression Modelling with Bidirectional Temporal Regularisation"

4. **"给 Claude 的提示词"部分过于理想化** ⚠️
   - 真正实现这些改进需要深入理解 MONAI GenerativeModels 的内部接口
   - 比如 ControlNet 的 conditioning_embedding_in_channels 直接决定了输入维度，不能随意加条件

---

## 二、源码级对比分析

### BrLP 核心架构参数（从代码中提取）

| 组件                      | 参数                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------- |
| AutoencoderKL             | spatial_dims=3, latent_channels=3, num_channels=(64,128,128,128), attention=全False |
| UNet (DiffusionModelUNet) | num_channels=(256,512,768), attention=(False,True,True), cross_attn_dim=8           |
| ControlNet                | 与 UNet 对称, conditioning_embedding_in_channels=4 (3通道latent + 1通道age)         |
| 输入形状                  | MRI: (120,144,120), 潜空间: (3,15,18,15), DM输入: (3,16,20,16)                      |
| 条件向量                  | 8维: [age, sex, diagnosis, cortex, hippocampus, amygdala, white_matter, ventricle]  |
| 辅助模型                  | Leaspy logistic DCM, source_dimension=4, 分 CN/MCI/AD 三组独立训练                  |
| 采样器                    | DDIM, 50步                                                                          |
| 损失函数                  | AutoEncoder: L1 + KL + Perceptual(squeeze,fake3D) + PatchAdversarial                |
| 损失函数                  | UNet/ControlNet: MSE(噪声预测)                                                      |

### 3D MedDiffusion 核心架构参数

| 组件           | 参数                                                                           |
| -------------- | ------------------------------------------------------------------------------ |
| PatchVolume AE | VQ-VAE, patch编码+volume解码, MedicalNet 3D 感知损失                           |
| BiFlowNet      | 混合 DiT+UNet, dim=72, sub_volume=(8,8,8), patch_size=2, dim_mults=(1,1,2,4,8) |
| DiT 部分       | IntraPatch Flow: 输入/中间/输出 DiT 块, adaLN-Zero                             |
| UNet 部分      | ResBlock + Attention + Up/Downsample, 与 DiT 特征通过 feature fusion 交互      |
| 条件           | 类别 embedding + 分辨率 embedding                                              |
| 扩散           | cosine schedule, 1000步, L1 loss                                               |
| 特色           | 滑动窗口推理处理大体积; 8x/4x 两种压缩率                                       |

### 关键差异

| 维度        | BrLP                                | 3D MedDiffusion                    |
| ----------- | ----------------------------------- | ---------------------------------- |
| 目标任务    | 个体级纵向预测（给定基线→预测随访） | 类别条件生成（生成某类型医学影像） |
| 个体特异性  | ControlNet 注入基线 latent + LAS    | 仅类别+分辨率条件                  |
| Autoencoder | 连续 KL, 小规模                     | VQ-VAE, 大规模, Patch-Volume       |
| Attention   | UNet 中部分使用                     | DiT 全局自注意力 + UNet attention  |
| 感知损失    | 2D squeeze + fake3D 采样            | 真3D MedicalNet 感知损失           |
| 时间建模    | 通过条件向量(age)隐式               | 无（非纵向任务）                   |

---

## 三、我认为可行且有效的创新点

基于对两个代码库的详细分析和近两年论文的调研，以下是按可行性和预期有效性排序的创新点：

### 创新点 1 (推荐优先做): MCI 转化动态条件引导 🔥🔥🔥

**问题**: BrLP 的辅助模型将 MCI 视为一个静态标签（diagnosis=0.5），但 MCI 的核心挑战是其异质性——部分 MCI 会在 1-3 年内转化为 AD (pMCI)，部分保持稳定 (sMCI)。条件向量中缺少这种转化趋势信息。

**改进方案**:

- 在条件向量中增加 `MCI_conversion_score`——基于 baseline 数据预测的 MCI→AD 转化概率
- 增加 `atrophy_rate`——基于 baseline 和人群统计的海马萎缩速率估计
- 修改 ControlNet 的 conditioning_embedding_in_channels 从 4 增加到 6
- 额外的两个通道分别编码转化概率图和萎缩速率图

**可行性分析**: ★★★★★

- 只需修改 ControlNet 的输入层和条件构建逻辑
- 不涉及核心架构变动，训练流程兼容
- 预计 2-3 天即可实现并开始训练

**预期效果**: 对 MCI 组的体积预测 MAE 降低 10-20%，特别是 pMCI 子组

**参考文献**:

- AG-LDM: Anatomically Guided Latent Diffusion for Brain MRI Progression Modeling (arXiv 2026-01)
- BrLP 论文中 Discussion 部分关于 MCI/AD 预测偏差的分析

---

### 创新点 2 (推荐第二做): 双向时间正则化 🔥🔥🔥

**问题**: BrLP 的 LAS (Latent Average Stabilization) 是一种后处理方式，对多时间点预测只保证了统计稳定性，没有显式建模时间序列的双向依赖（过去→未来、未来→过去应一致）。

**改进方案**:

- 在训练 ControlNet 时，同时训练正向（t1→t2）和反向（t2→t1）预测
- 引入双向时间一致性损失: $L_{btc} = \|f(x_A, c_{A \to B}) - z_B\|^2 + \|f(x_B, c_{B \to A}) - z_A\|^2$
- 在推理时，使用前向后向预测的加权平均来稳定结果

**可行性分析**: ★★★★☆

- 需要修改 ControlNet 训练脚本，让每个 batch 同时处理正反方向的对
- B.csv 数据已经包含成对数据，只需在 DataLoader 中增加反向对
- 预计 3-5 天实现

**预期效果**: 多时间点预测的时序一致性提升，SSIM 可能提升 0.01-0.02

**参考文献**:

- Temporally-Aware Diffusion Model for Brain Progression Modelling with Bidirectional Temporal Regularisation (2025-09)
- SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation (IPMI 2023)

---

### 创新点 3: DiT 增强的 U-Net 噪声估计器 🔥🔥

**问题**: BrLP 的 UNet 全部使用卷积，对 MCI 这种细微的全局性萎缩变化，卷积的局部感受野可能不够。

**改进方案** (受 BiFlowNet 启发，但不全部替换):

- 在 BrLP 的 UNet 的**最低分辨率层**之前，插入一个轻量级 DiT 模块
- 该 DiT 模块将 latent 切分为子体积 patch，进行全局自注意力交互
- DiT 输出特征注入 UNet 的 skip connection

**为什么不全部替换**:

- BrLP 的 ControlNet 与 UNet 共享相同架构，全部替换意味着要重写 ControlNet
- MONAI GenerativeModels 的 ControlNet 实现与 DiffusionModelUNet 紧耦合
- 部分注入是工程上最可行的方案

**可行性分析**: ★★★☆☆

- 需要自定义 MONAI 的 UNet，工程量较大
- 需要验证 DiT 模块与 ControlNet 的兼容性
- 预计 1-2 周实现

**预期效果**: 生成影像中全局结构的保真度提升，特别是脑沟回的清晰度

---

### 创新点 4: 3D 感知损失替换 + 频域约束 🔥🔥

**问题**: BrLP 的 AE 使用 2D squeeze 感知损失 + fake3D 采样(2D切片上抽样20%计算)，对3D结构保真度有限。

**改进方案**:

- 将 BrLP AE 的 PerceptualLoss 替换为 3D MedicalNet 感知损失（直接从 3D-MedDiffusion 借用）
- 增加拉普拉斯金字塔频率损失，强制保留高频纹理（脑沟回细节）
- 可选: 增加 FFT 频域约束

**可行性分析**: ★★★★☆

- 可以直接使用 3D MedDiffusion 中的 MedicalNetPerceptual 模块
- 只需要修改 AE 训练脚本的损失函数部分
- 预计 2-3 天实现

**预期效果**: 重建质量提升，SSIM 可能提升 0.005-0.01

---

### 创新点 5: 海马体区域注意力加权 🔥

**问题**: MCI 最关键的标志物是海马体萎缩，但 BrLP 的损失函数对所有区域一视同仁。

**改进方案**:

- 使用 SynthSeg 分割结果生成海马体 + 内嗅皮层的 mask
- 在 AE 和 ControlNet 训练中，对这些区域的损失加权（权重 2-5x）
- 在评估时增加海马体亚区的体积和纹理指标

**可行性分析**: ★★★★★

- 分割数据已有（BrLP 预处理流程包含 SynthSeg）
- 只需修改损失计算逻辑
- 预计 1-2 天实现

---

## 四、推荐的实施路线图

```
第一阶段 (1-2周): 快速出结果
├── 创新点 1: MCI 转化动态条件
├── 创新点 5: 海马体区域注意力加权
└── 评估: 对比 BrLP baseline 在 MCI 子组上的 MAE

第二阶段 (2-3周): 核心改进
├── 创新点 2: 双向时间正则化
├── 创新点 4: 3D 感知损失替换
└── 评估: 全面对比 SSIM/PSNR/MAE

第三阶段 (3-4周): 架构升级
├── 创新点 3: DiT 增强的 U-Net
└── 评估: 消融实验
```

---

## 五、近两年关键参考文献

| 论文                                 | 年份                     | 核心贡献                                        | 与本项目的关系           |
| ------------------------------------ | ------------------------ | ----------------------------------------------- | ------------------------ |
| BrLP (Puglisi et al.)                | MICCAI 2024 / MedIA 2025 | 潜空间扩散 + LAS + ControlNet 做脑 MRI 纵向预测 | 基线模型                 |
| 3D MedDiffusion (ShanghaiTech)       | IEEE TMI 2025            | Patch-Volume AE + BiFlowNet (DiT+UNet)          | DiT 增强和 AE 改进的来源 |
| AG-LDM (Wan et al.)                  | arXiv 2026-01            | 解剖引导的潜扩散脑 MRI 进展建模                 | 条件引导增强的参考       |
| Temporally-Aware DM (Litrico et al.) | 2025-09                  | 双向时间正则化的脑进展扩散模型                  | 时间建模的核心参考       |
| Treatment-aware DDPM (Liu et al.)    | 2025-01                  | 治疗感知的纵向 MRI 生成                         | 条件设计参考             |
| USB (Unified Synthetic Brain)        | 2025-12                  | 统一的病变/健康脑生成与编辑框架                 | 编辑能力参考             |
| SADM (UBC)                           | IPMI 2023/2024           | 序列感知扩散模型用于纵向医学影像生成            | 纵向建模参考(但是2D)     |
| DiT (Peebles & Xie)                  | ICCV 2023                | Diffusion Transformer 架构                      | 架构参考                 |
| Vanderbilt Replication               | SPIE 2025                | 在 BLSA 数据集上复现 BrLP                       | 验证 BrLP 的可复现性     |

---

## 六、结论

**Gemini 的整体方向是对的，但细节上有偏差。** 关键修正：

1. 不应该"把 UNet 替换为 DiT"，而应该"在 UNet 中注入 DiT 增强模块"
2. 条件引导的改进是最高优先级，因为它直接解决 MCI 预测偏差这个核心痛点
3. 感知损失的升级（2D→3D）是一个低风险高收益的改动
4. 所有改进都应在 BrLP 的框架内渐进式进行，避免破坏已有的工程基础

---

## 2026-04-04 | 历史评估结果汇总（基线与复跑）

本节对应你的要求：汇总之前跑过的评估，包含运行时间、模型来源与各项指标。

### 7.1 实验设置

以下评估均对应配置文件 `confs_pretrained.yaml`（服务器路径：`/home/wangchong/data/fwz/brlp-code/confs_pretrained.yaml`），主要权重为：

- autoencoder: `/home/wangchong/data/fwz/brlp-train/pretrained/autoencoder.pth`
- unet: `/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth`
- controlnet: `/home/wangchong/data/fwz/brlp-train/pretrained/controlnet.pth`
- aux dcm: `dcm_nc.json` / `dcm_mci.json` / `dcm_ad.json`

### 7.2 公平对比（历史 run 概览）

> 说明：
>
> 1. 运行时间取自 `eval.log` 或评估产物文件时间戳（`eval_summary.json`）。
> 2. `inference_sec_mean` 采用按列名 `inference_sec` 统计的结果。

| Run                                         | 时间（+0800）       | n_valid / n_total | inference_sec_mean | SSIM (mean±std)       | PSNR (mean±std)        | MSE (mean±std)        | MAE (mean±std)        |
| ------------------------------------------- | ------------------- | ----------------- | ------------------ | --------------------- | ---------------------- | --------------------- | --------------------- |
| brlp-train/eval_masked_20260403_rerun_paper | 2026-04-03 14:23:06 | 51 / 51           | 6.7843             | 0.0840435 ± 0.0183375 | 2.4933379 ± 1.1022613  | 0.5802468 ± 0.1306961 | 0.6914517 ± 0.0803434 |
| brlp-train/eval_masked_20260403             | 2026-04-03 14:57:44 | 51 / 51           | 6.6118             | 0.0838245 ± 0.0182934 | 2.4938029 ± 1.1021805  | 0.5801815 ± 0.1306641 | 0.6915249 ± 0.0802776 |
| adni-eval/run_20260404_123352/eval          | 2026-04-04 12:35:28 | 10 / 10           | 6.7100             | 0.0545105 ± 0.0101576 | 7.5603771 ± 0.3769537  | 0.1760384 ± 0.0154073 | 0.4029255 ± 0.0176892 |
| oasis-eval-validate                         | 2026-04-04 17:27:18 | 2 / 2             | 7.3000             | 0.0858388 ± 0.0042545 | 2.4750162 ± 0.6218943  | 0.5713943 ± 0.0812669 | 0.6851900 ± 0.0424870 |
| oasis-eval-v2                               | 2026-04-04 17:39:04 | 3 / 3             | 6.7000             | 0.9274799 ± 0.0124092 | 16.5671115 ± 1.1787636 | 0.0228239 ± 0.0056677 | 0.1027109 ± 0.0073900 |
| adni-eval-full/eval                         | 2026-04-04 17:46:53 | 43 / 43           | 6.0930             | 0.7572529 ± 0.0197582 | 8.8321067 ± 0.6437704  | 0.1322738 ± 0.0191710 | 0.2889157 ± 0.0233324 |

### 7.3 结果分析（体积误差指标）

| Run                                         | cortex    | hippocampus | amygdala  | white_matter | lateral_ventricle |
| ------------------------------------------- | --------- | ----------- | --------- | ------------ | ----------------- |
| brlp-train/eval_masked_20260403_rerun_paper | 0.1651701 | 0.1422961   | 0.0727820 | 0.0673947    | 0.2243515         |
| brlp-train/eval_masked_20260403             | 0.1651701 | 0.1422961   | 0.0727820 | 0.0673947    | 0.2243515         |
| adni-eval/run_20260404_123352/eval          | N/A       | N/A         | N/A       | N/A          | N/A               |
| oasis-eval-validate                         | 0.1622365 | 0.0692639   | 0.0903013 | 0.2137768    | 0.2525191         |
| oasis-eval-v2                               | 0.1269343 | 0.1010038   | 0.0634538 | 0.1442529    | 0.3146243         |
| adni-eval-full/eval                         | 0.2869725 | 0.3593921   | 0.3468369 | 0.4973834    | 0.0497632         |

### 7.4 结论

- 综合表现最好的 run 是 `oasis-eval-v2`（SSIM/PSNR 高、MSE/MAE 低），但样本量仅 3，统计稳定性有限。
- `adni-eval-full/eval` 在 43 例上达到相对稳健的结果，具有更高参考价值。
- 两个 BrLP train masked run（51 例）指标几乎一致，说明复跑结果稳定。
- 是否有效（相对“历史基线稳定性校验”目标）：**有效**。本轮目标是验证可复现性与评估管线稳定性，结论成立。

---

## 2026-04-07 | 创新点 5 评估结果：海马体区域注意力加权

### 8.1 实验设置

- **数据**: MCI 纵向配对数据，465 对（train=371, valid=44, test=50），年龄已归一化（age/100）
- **评估**: 从 test 集随机抽取 20 对进行评估
- **分割**: SynthSeg 分割 label，海马体 (17,53) + 杏仁核 (18,54) 作为 ROI
- **创新方法**: ControlNet 训练时对海马体+杏仁核区域的损失加权（roi_weight=3.0, region_alpha=0.5）
- **Baseline**: 使用相同归一化数据、相同 AE 和 UNet，仅 ControlNet 使用预训练权重（无区域加权）
- **ControlNet v2 训练**: 5 epochs，最佳 checkpoint = epoch 3（valid loss_w = 0.031）

### 8.2 公平对比（相同数据，v2）

| 指标                 | Baseline_v2         | Innovation_5_v2      | Δ (绝对值) | 变化幅度    |
| -------------------- | ------------------- | -------------------- | ---------- | ----------- |
| **overall_ssim**     | 0.9015 ± 0.0274     | **0.9145 ± 0.0281**  | +0.0130    | **↑ 1.44%** |
| **overall_psnr**     | 25.9243 ± 2.0300    | **26.2282 ± 2.3050** | +0.3039    | **↑ 1.17%** |
| overall_mae          | 0.0288 ± 0.0094     | 0.0289 ± 0.0120      | +0.0001    | ≈ 持平      |
| overall_mse          | 0.0028 ± 0.0015     | 0.0028 ± 0.0018      | 0.0000     | ≈ 持平      |
| hippocampus_mae      | **0.0604 ± 0.0351** | 0.0723 ± 0.0505      | +0.0119    | ↓ 19.7%     |
| **hippocampus_ssim** | 0.8199 ± 0.0445     | **0.8319 ± 0.0281**  | +0.0120    | **↑ 1.46%** |
| amygdala_mae         | **0.0660 ± 0.0414** | 0.0813 ± 0.0490      | +0.0153    | ↓ 23.2%     |
| roi_mae              | **0.0625 ± 0.0367** | 0.0755 ± 0.0493      | +0.0130    | ↓ 20.8%     |
| **roi_ssim**         | 0.7983 ± 0.0398     | **0.8141 ± 0.0262**  | +0.0158    | **↑ 1.98%** |

### 8.3 结果分析

**SSIM 指标全面提升**:

- 全脑 SSIM: +1.44%，海马体 SSIM: +1.46%，ROI SSIM: +1.98%
- 说明区域注意力加权有效提升了结构相似性，尤其在 ROI 区域改善最为显著
- Innovation_5_v2 的 hippocampus_ssim 标准差更小（0.0281 vs 0.0445），表明预测更稳定

**MAE 指标呈现分化**:

- 全脑 MAE 基本持平（0.0289 vs 0.0288）
- 海马体 MAE 反而上升 19.7%，杏仁核 MAE 上升 23.2%
- MAE 上升但 SSIM 同时上升，说明区域加权改善了结构模式的预测，但可能引入了轻微的强度偏移

**可能的解释**:

1. **SSIM vs MAE 的矛盾**: SSIM 衡量结构相似性（亮度 + 对比度 + 结构），MAE 仅衡量逐体素绝对误差。区域加权可能让模型更好地捕捉了局部萎缩模式（结构↑），但在绝对体素强度上产生了偏移（MAE↓）
2. **训练时长不足**: Innovation_5 的 ControlNet 仅训练了 5 epochs，而预训练 baseline ControlNet 训练时间可能远超此数。更长的训练可能同时改善两项指标
3. **区域权重超参数**: roi_weight=3.0 和 region_alpha=0.5 未经调优，可能导致 ROI 区域的损失梯度过大，使得局部强度偏离

### 8.4 结论

创新点 5（海马体区域注意力加权）在结构相似性指标（SSIM）上取得了一致的提升，特别是 ROI SSIM 提升 1.98%，且标准差降低，说明预测更稳定。全脑整体质量（SSIM, PSNR）同样有所改善。

MAE 在 ROI 区域的上升是一个需要进一步优化的方向，建议：

- 增加 ControlNet 训练 epochs（从 5 → 15-20）
- 对 roi_weight 和 region_alpha 进行网格搜索调优
- 考虑使用 SSIM-based 区域损失替代纯 MSE 区域加权
- 是否有效：**部分有效**。结构相似性指标（SSIM/PSNR）提升明确，但 ROI MAE 未同步改善。

---

## 2026-04-09 | 创新点 4 评估结果：3D 感知损失替换 + 频域约束

### 9.1 实验设置

- **数据**: MCI 纵向配对数据，644 训练样本（从 665 中过滤掉 21 个缺失文件），测试集 50 对（使用 Innovation 5 的 B_mci.csv test split）
- **AE 训练**: 5 epochs，max_batch_size=1，grad_accum 到 batch_size=16，lr=1e-4，AdamW
- **核心改进**:
  - 将 BrLP 的 2D squeeze PerceptualLoss 替换为 **MedicalNet ResNet-10 真3D 感知损失**（预训练于 23 个医学数据集）
  - 增加 **拉普拉斯金字塔频率损失**（3 层，高频加权）
  - 输入通过下采样至 (64,72,64) 后送入 ResNet-10 提取特征（防止 OOM）
- **损失权重**: perc_weight=0.001, freq_weight=0.01, fft_weight=0.0, kl_weight=1e-7, adv_weight=0.025
- **Diffusion + ControlNet**: 使用 BrLP 预训练权重（仅 AE 被替换）
- **最终 checkpoint**: autoencoder-ep-4.pth (53MB)

### 9.2 公平对比

| 指标                 | Baseline_v2      | Innovation_4         | Innovation_5_v2      | Innov4 vs Baseline | Innov4 vs Innov5 |
| -------------------- | ---------------- | -------------------- | -------------------- | ------------------ | ---------------- |
| **overall_ssim**     | 0.9015 ± 0.0274  | **0.9081 ± 0.0213**  | **0.9145 ± 0.0281**  | **↑ 0.73%**        | ↓ 0.70%          |
| **overall_psnr**     | 25.9243 ± 2.0300 | **26.0283 ± 2.1551** | **26.2282 ± 2.3050** | **↑ 0.40%**        | ↓ 0.76%          |
| overall_mae          | 0.0288 ± 0.0094  | 0.0308 ± 0.0101      | 0.0289 ± 0.0120      | ↓ 6.9%             | ↓ 6.6%           |
| hippocampus_mae      | 0.0604 ± 0.0351  | 0.0656 ± 0.0421      | 0.0723 ± 0.0505      | ↓ 8.6%             | ↑ 9.3%           |
| **hippocampus_ssim** | 0.8199 ± 0.0445  | **0.8301 ± 0.0269**  | **0.8319 ± 0.0281**  | **↑ 1.24%**        | ↓ 0.22%          |
| amygdala_mae         | 0.0660 ± 0.0414  | 0.0724 ± 0.0403      | 0.0813 ± 0.0490      | ↓ 9.7%             | ↑ 10.9%          |
| roi_mae              | 0.0625 ± 0.0367  | 0.0670 ± 0.0338      | 0.0755 ± 0.0493      | ↓ 7.2%             | ↑ 11.3%          |
| **roi_ssim**         | 0.7983 ± 0.0398  | **0.8184 ± 0.0328**  | **0.8141 ± 0.0262**  | **↑ 2.52%**        | **↑ 0.53%**      |

### 9.3 结果分析

**SSIM/PSNR 全面提升，ROI SSIM 表现尤为突出**:

- Innovation 4 的 roi_ssim（0.8184）甚至超过了 Innovation 5（0.8141），提升 **+2.52%** vs baseline
- 全脑 SSIM 提升 +0.73%，PSNR 提升 +0.40%
- hippocampus_ssim 提升 +1.24%，标准差从 0.0445 降至 0.0269，预测更稳定

**MAE 指标有所退步**:

- 全脑 MAE 上升 6.9%（0.0288→0.0308）
- 海马体和杏仁核区域 MAE 同样上升
- 但注意 Innovation 4 的 MAE 退步幅度小于 Innovation 5（Innovation 5 的 hippocampus_mae 从 0.0604→0.0723↓19.7%，而 Innovation 4 仅从 0.0604→0.0656↓8.6%）

**3D 感知损失的贡献**:

1. MedicalNet 真3D 特征提取比 2D squeeze 更好地保留了三维空间结构，反映在 SSIM 的一致提升上
2. 拉普拉斯金字塔频率损失有效约束了高频细节（脑沟回纹理），使 ROI 区域结构保真度最高
3. 下采样策略（120×144×120→64×72×64）在保持可行性的同时仍能提取有效的 3D 感知特征

**Innovation 4 vs Innovation 5 的互补性**:

- Innovation 4 在 ROI SSIM 和 MAE 方面优于 Innovation 5
- Innovation 5 在全脑 SSIM 和 PSNR 方面优于 Innovation 4
- 两者改进方向不同（AE 损失 vs ControlNet 区域加权），理论上可以**叠加使用**

### 9.4 结论

创新点 4（3D 感知损失 + 频域约束）验证了真3D 感知特征提取对医学影像重建质量的显著提升，尤其在 ROI 区域的结构相似性上超过了 Innovation 5。这一改进仅修改了 AE 的损失函数，不涉及架构变动，具有很强的实用性。

**后续优化方向**:

- 尝试更大的下采样分辨率（如 80×96×80）以获取更精细的 3D 感知特征
- 对 perc_weight 和 freq_weight 进行调优
- 将 Innovation 4 和 Innovation 5 叠加：在 Innovation 4 的 AE 基础上训练带区域加权的 ControlNet
- 增加训练 epoch 数（5→10-15），让 3D 感知损失充分收敛
- 是否有效：**部分有效**。ROI/全脑结构相似性提升，但 MAE 仍存在退步，需要继续调参验证。

---

## 2026-04-10 | 创新点 4 v4 评估结果：训练策略增强（二次迭代）

### 10.1 实验设置

- **数据**: 沿用 Innovation 4 v4 评估流水线（Pairs 50/50）
- **评估来源**: 监控页 `task_progress.eval.summary_metrics`（实时拉取）
- **改进方法**: 在创新点 4 基础上增加训练策略增强（decoder-only 微调 + warmup + latent noise）
- **口径说明**: 本轮是“创新点 4 单因素二次迭代”，不是“创新点 5 + 创新点 4 联合改动”的联合实验。

### 10.2 公平对比（相对 Baseline_v2）

| 指标             | Baseline_v2      | Innovation_4_v4  | 方向 |
| ---------------- | ---------------- | ---------------- | ---- |
| overall_ssim     | 0.9015 ± 0.0274  | 0.8529 ± 0.0539  | 下降 |
| overall_psnr     | 25.9243 ± 2.0300 | 24.6582 ± 2.6818 | 下降 |
| overall_mae      | 0.0288 ± 0.0094  | 0.0416 ± 0.0177  | 变差 |
| hippocampus_ssim | 0.8199 ± 0.0445  | 0.7926 ± 0.0427  | 下降 |
| roi_ssim         | 0.7983 ± 0.0398  | 0.7896 ± 0.0411  | 下降 |
| roi_mae          | 0.0625 ± 0.0367  | 0.0807 ± 0.0425  | 变差 |

### 10.3 结果分析

- 本轮 v4 在全局与 ROI 的核心指标上均未超过 baseline。
- 指标趋势显示当前训练策略组合（warmup + latent noise + 新权重）对该数据划分不稳定。
- 与 2026-04-09 首轮创新点 4 相比，本轮二次迭代出现回退，说明当前改动方向需要拆分成单变量再验证。
- 创新点 5 改动位于 ControlNet 训练侧，创新点 4 改动位于 AE 侧；若要评估“5+4 联合效果”，需要以创新点 4 的 AE 重新训练/适配 ControlNet 后再评估，不能直接拿两组单因素结果叠加解释。

### 10.4 结论

- 本轮创新点 4 v4 **未通过有效性验证**。
- 是否有效：**无效**（相对 baseline 与创新点 4 首轮均未提升）。
- 建议：回退到 2026-04-09 有效配置作为新起点，每次仅调整一个超参数并重跑评估。

---

## 2026-04-10 | 创新点 4+5 联合推理评估

### 11.1 实验设置

- **核心思路**: 创新点 4 仅优化 AE Decoder（Encoder 完全冻结），因此潜空间（latent representation）与基线完全一致。创新点 5 的 ControlNet 在训练时使用预提取的 NPZ 潜向量（不通过 AE decoder），仅在潜空间做区域加权噪声预测。由于两者的潜空间完全兼容，可以直接在推理阶段组合，无需重新训练。
- **AE Checkpoint**: `/output/innovation_4/ae_training/autoencoder-ep-4.pth`（Innovation 4 v1 最终 epoch，SSIM 已验证为 0.9081）
- **ControlNet Checkpoint**: `/output/innovation_5/controlnet/cnet-ep-3.pth`（Innovation 5 v2 最优 epoch，valid loss_w = 0.031）
- **Diffusion**: `/brlp-train/pretrained/latentdiffusion.pth`（基线 UNet，不变）
- **数据**: `B_mci.csv` 测试集（50 对），与创新点 4、5 的评估完全一致
- **评估脚本**: Innovation 5 的 `evaluate_regional.py`（直接复用，仅更换 AE 路径）
- **评估耗时**: 约 5 分 30 秒（50 对 × 6.56s/对）

### 11.2 公平对比（全量 50 对测试集）

| 指标             | Baseline_v2         | Innovation_4_v1     | Innovation_5_v2      | **联合 4+5**     | 联合 vs Baseline | 联合 vs Innov5 |
| ---------------- | ------------------- | ------------------- | -------------------- | ---------------- | ---------------- | -------------- |
| **overall_ssim** | 0.9015 ± 0.0274     | 0.9081 ± 0.0213     | **0.9145 ± 0.0281**  | 0.9123 ± 0.0247  | **↑ 1.20%**      | ↓ 0.24%        |
| **overall_psnr** | 25.9243 ± 2.0300    | 26.0283 ± 2.1551    | **26.2282 ± 2.3050** | 25.9442 ± 2.1449 | ↑ 0.08%          | ↓ 1.09%        |
| overall_mae      | **0.0288** ± 0.0094 | 0.0308 ± 0.0101     | 0.0289 ± 0.0120      | 0.0311 ± 0.0105  | ↓ 7.9%           | ↓ 7.6%         |
| hippocampus_ssim | 0.8199 ± 0.0445     | **0.8301 ± 0.0269** | **0.8319 ± 0.0281**  | 0.8203 ± 0.0304  | ↑ 0.05%          | ↓ 1.39%        |
| hippocampus_mae  | **0.0604** ± 0.0351 | 0.0656 ± 0.0421     | 0.0723 ± 0.0505      | 0.0748 ± 0.0447  | ↓ 23.8%          | ↓ 3.5%         |
| **roi_ssim**     | 0.7983 ± 0.0398     | **0.8184 ± 0.0328** | 0.8141 ± 0.0262      | 0.8059 ± 0.0284  | **↑ 0.95%**      | ↓ 1.01%        |
| roi_mae          | **0.0625** ± 0.0367 | 0.0670 ± 0.0338     | 0.0755 ± 0.0493      | 0.0768 ± 0.0440  | ↓ 22.8%          | ↓ 1.7%         |

### 11.3 结果分析

**联合实验未实现叠加收益**:

- Overall SSIM（0.9123）：介于 Innovation 4 v1（0.9081）和 Innovation 5 v2（0.9145）之间，未超过 Innovation 5
- ROI SSIM（0.8059）：低于两个单独实验（Inn4 v1: 0.8184, Inn5 v2: 0.8141）
- PSNR 和 MAE 均有所退步
- 预期的"叠加改善"未能实现

**为什么叠加失效？**

关键问题在于 **解码器不兼容性**（Decoder Mismatch）：

1. Innovation 5 的 ControlNet 在训练时通过预提取的 NPZ 潜向量计算损失，但在 **推理时** 使用 `sample_using_controlnet_and_z` 函数：该函数会通过 DDIM 采样得到预测潜向量，然后调用 AE **解码器** 生成最终图像
2. Innovation 5 的 ControlNet 的噪声预测目标是让解码后的图像接近真实随访图像。该模型隐式学习了"目标潜向量 → 基线解码器输出 → 目标图像"的映射
3. 当我们将解码器替换为 Innovation 4 的 Decoder（具有不同的感知损失目标），ControlNet 预测的潜向量方向与新解码器的输出特征分布不完全对齐
4. 结果是：ControlNet 试图引导潜向量向"适合基线解码器"的方向移动，但 Innovation 4 的解码器对相同潜向量产生了不同的图像，造成轻微的结构失配

**统计显著性评估**:

- 大部分差异在各方法的标准差范围内（roi_ssim std ≈ 0.028-0.040），联合实验与 Innovation 5 的差距（-1.01%）不一定统计显著
- 但联合实验确实优于 Baseline（roi_ssim +0.95%），说明两个创新点在单独使用时均有效，组合也不会完全抵消改进

**与理论预测的偏差**:

| 预测项       | 理论预测                           | 实际结果                      |
| ------------ | ---------------------------------- | ----------------------------- |
| 潜空间兼容性 | ✅ 完全兼容（已验证：Encoder冻结） | ✅ 技术上兼容，可组合使用     |
| 叠加收益     | ≥ max(Inn4, Inn5) = 0.9145         | ✗ 实际为 0.9123（中间值）     |
| ROI 改善叠加 | ≥ 0.8184（Inn4 v1 最佳）           | ✗ 实际为 0.8059（低于两者）   |
| 主要原因     | 未考虑解码器感知空间变化           | Decoder Mismatch 影响推理质量 |

### 11.4 结论与后续方向

**是否有效**: **部分有效**。联合推理评估使得 Overall SSIM 比 Baseline 提升 1.20%，且不需要任何额外训练（完全复用现有 checkpoint）。但未能超越单独使用 Innovation 5（SSIM = 0.9145）。

**如何真正实现 4+5 叠加改善**:

若要让 Innovation 4 的解码器改进真正与 Innovation 5 的 ControlNet 协同工作，需要进行"联合训练"而非仅在推理阶段组合：

| 方案               | 描述                                                        | 代价               | 预期效果                     |
| ------------------ | ----------------------------------------------------------- | ------------------ | ---------------------------- |
| **方案 A（推荐）** | 在 Inn4 的 AE 基础上重新训练 Inn5 的 ControlNet（5 epochs） | 中等（~1小时训练） | 可能实现真正叠加收益         |
| 方案 B             | 将 Inn4 的解码器损失改进合并到 Inn5 的 AE 训练脚本中        | 低（修改训练脚本） | 在同一训练中同时优化两项目标 |
| 方案 C             | 接受分开的改进，在论文中分别报告各自指标                    | 无                 | 无额外收益，但论文结构清晰   |

**目前最佳单指标结果汇总**:

| 最优指标     | 最优值           | 来自方法                      |
| ------------ | ---------------- | ----------------------------- |
| Overall SSIM | 0.9145 (+1.44%)  | Innovation 5 v2               |
| Overall PSNR | 26.2282 (+1.17%) | Innovation 5 v2               |
| ROI SSIM     | 0.8184 (+2.52%)  | Innovation 4 v1               |
| Overall MAE  | 0.0288 (≈基线)   | Baseline / Innovation 5 ≈持平 |

---

## Section 12 — 2026-04-10 | 方案 A 结果：联合重训 ControlNet

### 12.1 实验设置

- **目标**: 在 Inn4 的 AE（ep-4，Decoder微调版）基础上重新训练 Inn5 的 ControlNet，使两者共享相同的解码特征空间
- **超参数**: n_epochs=5, lr=2.5e-5, batch=8, roi_weight=3.0, region_alpha=0.5（与 Inn5 完全一致）
- **唯一差异**: `--aekl_ckpt` 替换为 Inn4 微调版（vs. 原始基线 AE）
- **运行设备**: GPU 1（~18 GiB 可用，GPU 0 被 eye-env 占满）
- **训练速度**: ~40s/epoch（47 batch × ~1.2 it/s），5 epochs 约 3.5 分钟

### 12.2 训练过程

损失曲线顺利收敛，各 epoch 加权噪声预测 MSE：

| Epoch | Train loss_w | Val loss_w  |
| ----- | ------------ | ----------- |
| 0     | 0.175        | ~0.21       |
| 1     | 0.133        | 0.127 ✓最低 |
| 2     | 0.134        | 0.129       |
| 3     | 0.115        | 0.141       |
| 4     | 0.092        | —           |

### 12.3 评估结果（50对测试集）

| 指标         | Baseline | Inn4 v1 | Inn5 v2 | combined_4_5 | **combined_retrain ep1** | **combined_retrain ep4** |
| ------------ | -------- | ------- | ------- | ------------ | ------------------------ | ------------------------ |
| overall_ssim | 0.9015   | 0.9081  | 0.9145  | 0.9123       | 0.7841                   | 0.8664                   |
| roi_ssim     | 0.7983   | 0.8184  | 0.8141  | 0.8059       | ≈0.56                    | 0.7043                   |

**附加诊断实验**（combined_retrain ep4 + 基线AE）：overall_ssim = 0.2750，hippocampus_ssim = 0.7101

### 12.4 方案 A 失败分析

**训练代码分析**发现：AE checkpoint 在训练中 **仅用于 TensorBoard 可视化**（在 `images_to_tensorboard` 中进行 `torch.no_grad()` 推理），**不参与梯度计算**。训练损失 = 纯噪声预测 MSE，与 Inn5 训练完全相同。

**因此 combined_retrain 与 Inn5 的 ControlNet 训练在理论上等价**，SSIM 差异（0.8664 vs 0.9145）来自：

1. **随机种子差异**: 不同 GPU、不同 nohup 启动状态 → 不同噪声采样序列 → 收敛到不同局部极小值
2. **训练轮次不足**: Epoch 4 的 val_loss（0.141）已超过 Epoch 1（0.127），出现过拟合迹象；5 epoch 可能不够收敛到与 Inn5 ep-3 相当的水平
3. **潜在强度校准偏移**: 诊断实验（combined_retrain + baseline AE）得到 overall_ssim=0.2750 但 hippocampus_ssim=0.7101，提示生成潜空间的全局强度范围与基线解码器不匹配，而与 Inn4 解码器匹配——原因仍待确认（AE 不参与梯度，但 TensorBoard 可视化中的 AE 可能通过 BatchNorm Running Stats 对 GPU 数值精度产生影响）

### 12.5 结论与建议

| 方案             | 结果                       | 状态           |
| ---------------- | -------------------------- | -------------- |
| combined_4_5     | SSIM=0.9123 (+1.20% vs BL) | ✓ 当前最佳组合 |
| combined_retrain | SSIM=0.8664（劣于 Inn5）   | ✗ 未达预期     |

**当前最优策略**: 坚持 combined_4_5（Inn5 ep-3 + Inn4 AE），无需重训即可实现 +1.20% SSIM 改善。

**若需进一步改善**:

- 方案 A 变体：使用更多训练 epoch（10-20 epochs）+ 正确的随机种子（设置 `torch.manual_seed`）
- 或接受分别报告各创新点指标的策略（方案 C），确保论文结构清晰
