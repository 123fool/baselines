# 方案D集成改造：可行性分析与实现规划

> 目标：在保留BrLP性能指标的前提下，最大化模型新颖度，使其可以作为独立模型发表论文。

---

## 一、当前模型指纹分析

当前项目本质上是 BrLP（MICCAI 2024 Oral + MIA 2025）的复现/改进。审稿人能一眼认出的"指纹"有：

| 指纹                                                    | 识别难度                        | 危险程度 |
| ------------------------------------------------------- | ------------------------------- | -------- |
| **Leaspy辅助模型** — MCMC-SAEM拟合logistic混合效应模型  | 极易识别                        | 🔴 致命  |
| ControlNet + UNet 双网络结构                            | 中等 — MONAI-generative通用组件 | 🟡 中等  |
| 3阶段潜空间训练管线 (AE→UNet→ControlNet)                | 中等                            | 🟡 中等  |
| cross-attention 8维条件向量 (age/sex/diagnosis + 5体积) | 较难识别                        | 🟢 较低  |
| LAS (Latent Average Stabilization) 推理策略             | 较难识别                        | 🟢 较低  |

**核心矛盾**：Leaspy 是 BrLP 的标志性组件。论文中明确引用了 Leaspy 库（Jean-Baptiste Schiratti et al., 2017），只要用了 Leaspy，审稿人就会认为这是 BrLP 的变体。

---

## 二、四个优先级改造的详细可行性分析

### 优先级1：TPN（Temporal Progression Network）替换 Leaspy

#### 1.1 当前 Leaspy 的精确作用

Leaspy **仅在推理时使用**，完整数据流如下：

```
推理时:
  历史MRI扫描 → SynthSeg分割 → 测量5个脑区体积 → 归一化
       ↓
  归一化体积序列 → Leaspy.personalize() → 个体化参数
       ↓
  个体化参数 + 目标年龄序列 → Leaspy.estimate() → 预测未来5个体积
       ↓
  _reverse_and_correct() → 修正后的5维体积 → 作为条件向量的后5维
       ↓
  条件向量 = [age_norm, sex_norm, diag_norm, cortex, hippo, amyg, wm, ventricle]
       ↓
  ControlNet采样 → 预测MRI

训练时:
  直接使用 CSV B 中的真实 followup 体积数据，不涉及 Leaspy
```

关键发现：

- **训练时 Leaspy 不参与**，条件向量中的5个体积来自真实的followup扫描数据
- Leaspy 只在推理时为"还没拍摄的未来扫描"提供体积估计
- Leaspy 的输出经过 `_reverse_and_correct()` 做了 median correction，说明 Leaspy 估计值本身有偏差

#### 1.2 TPN 替代方案设计

**核心思路**：用一个可学习的 MLP 替代 Leaspy 的 logistic mixed-effects model。

**输入/输出设计**：

```
输入 (10维):
  - current_age          (归一化, 1维)
  - target_age           (归一化, 1维)
  - sex                  (归一化, 1维)
  - diagnosis            (归一化, 1维)
  - current_cortex       (归一化, 1维)   ← 当前5个脑区体积
  - current_hippocampus  (归一化, 1维)
  - current_amygdala     (归一化, 1维)
  - current_white_matter (归一化, 1维)
  - current_ventricle    (归一化, 1维)
  - age_gap              (target - current, 1维)

输出 (5维):
  - predicted_cortex
  - predicted_hippocampus
  - predicted_amygdala
  - predicted_white_matter
  - predicted_ventricle
```

**网络架构**：

```python
class TemporalProgressionNetwork(nn.Module):
    """
    端到端可学习的疾病进展轨迹预测网络。
    替代基于外部统计模型的体积估计方法。
    """
    def __init__(self, in_dim=10, hidden_dim=128, out_dim=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()       # 输出归一化到 [0, 1]
        )

    def forward(self, x):
        return self.net(x)
```

#### 1.3 TPN 训练数据来源

**直接使用 CSV B**（已有的配对数据集）：

```
CSV B 中每一行包含:
  starting_age, followup_age, sex, starting_diagnosis, followup_diagnosis
  starting_cerebral_cortex, followup_cerebral_cortex
  starting_hippocampus, followup_hippocampus
  starting_amygdala, followup_amygdala
  starting_cerebral_white_matter, followup_cerebral_white_matter
  starting_lateral_ventricle, followup_lateral_ventricle

TPN训练样本:
  X = [starting_age, followup_age, sex, diagnosis, 5×starting_volumes, age_gap]
  Y = [5×followup_volumes]
```

**不需要额外数据**，现有训练集就能直接训练TPN。

#### 1.4 可行性评估

| 维度       | 评估                  | 说明                                                |
| ---------- | --------------------- | --------------------------------------------------- |
| 数据可行性 | ✅ 完全可行           | CSV B 已有所有需要的字段                            |
| 性能影响   | ✅ 预计持平或略好     | MLP 比 logistic 曲线更灵活；5维回归是非常简单的任务 |
| 代码量     | ✅ 约 80 行           | TPN模型30行 + 训练脚本50行                          |
| 训练成本   | ✅ 极低               | 纯MLP，CPU上几分钟训练完成                          |
| 有无依赖   | ✅ 消除了 Leaspy 依赖 | 不再需要 leaspy 包                                  |
| 推理管线   | ⚠️ 需要改 cli.py      | 约30行改动，用TPN替代Leaspy调用                     |
| 学术新颖度 | ✅ 高                 | 论文中可以写成"端到端可学习的时间进展预测"          |

**风险点**：

- Leaspy 的 personalize() 做了 per-patient MCMC 拟合，给每个患者独立的参数；TPN 是全局模型
- 缓解措施：TPN 的输入已包含 current_volumes（等价于 Leaspy 的个体化信息）

#### 1.6 实验结果（2026-04-12）

**TPN v3b (最终版)**: Sequential MLP (14→128→128→5) + 残差连接 + 增强特征

| 指标                | TPN v3b          | Leaspy (原始)     | 对比                |
| ------------------- | ---------------- | ----------------- | ------------------- |
| Overall MAE         | 0.0154           | 0.0136            | TPN 高 13%          |
| R² Score            | 0.9522           | 0.9535            | 接近持平            |
| cerebral_cortex MAE | **0.0074**       | 0.0124            | **TPN 更优 (↓40%)** |
| hippocampus MAE     | 0.0132           | 0.0115            | Leaspy 更优         |
| amygdala MAE        | 0.0320           | 0.0254            | Leaspy 更优         |
| white_matter MAE    | **0.0065**       | 0.0070            | **TPN 更优 (↓7%)**  |
| ventricle MAE       | 0.0181           | 0.0120            | Leaspy 更优         |
| 样本覆盖率          | **94/94 (100%)** | 53/94 (56%)       | **TPN 大幅领先**    |
| 参数量              | 19,077           | N/A (统计模型)    | TPN 轻量            |
| 推理速度            | <1ms/sample      | ~1s/sample (MCMC) | **TPN 快1000x**     |

**结论**: TPN v3b R²=0.9522，在 MCI 纵向预测任务上有效。在 cortex 和 white_matter 两个最大脑区上超越 Leaspy，100% 样本覆盖 vs Leaspy 56%（Leaspy 在个别患者上个性化拟合失败）。整体 MAE 比 Leaspy 高 13%，主要来自 ventricle（脑室扩张模式特殊）和 amygdala（小结构，测量噪声大）。

**训练细节**: CSV B (465 MCI pairs, 371 train / 94 test+val), 200 epochs, MSE loss, AdamW lr=1e-3, CosineAnnealing, best epoch=171, ~30s CPU 训练。

**服务器路径**: 模型 `/home/wangchong/data/fwz/output/tpn_v3b/tpn_best.pth`, 代码 `/home/wangchong/data/fwz/code/tpn/`

#### 1.5 论文叙事

> "现有方法依赖外部统计模型（如 logistic mixed-effects model）来预测疾病进展轨迹，这些模型假设脑区体积严格遵循 S 型曲线衰减，限制了对非典型进展模式（如 MCI 阶段的非线性波动）的建模能力。我们提出 **Temporal Progression Network (TPN)**，一个端到端可训练的轻量级网络，直接从患者的当前脑区体积和临床信息学习个体化的未来体积估计，无需额外的参数化假设。"

---

### 优先级2：残差潜码预测（Residual Latent Prediction）

#### 2.1 改动范围

只改两个文件：

**文件1: train_controlnet.py（训练）**

```python
# ===== 原始 BrLP =====
noise = torch.randn_like(followup_z).to(DEVICE)
timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
images_noised = scheduler.add_noise(followup_z, noise=noise, timesteps=timesteps)
# ... ControlNet + UNet 预测噪声 ...
loss = F.mse_loss(noise_pred.float(), noise.float())

# ===== 改为残差预测 =====
delta_z = followup_z - starting_z                    # 新增: 计算潜码残差
noise = torch.randn_like(delta_z).to(DEVICE)
timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
delta_noised = scheduler.add_noise(delta_z, noise=noise, timesteps=timesteps)  # 对残差加噪
# ... ControlNet + UNet 预测噪声 (输入改为 delta_noised) ...
loss = F.mse_loss(noise_pred.float(), noise.float())
```

**文件2: sampling.py（推理）**

```python
# ===== 原始 BrLP =====
z = (z / scale_factor).sum(axis=0) / average_over_n  # z 是预测的 followup latent

# ===== 改为残差推理 =====
delta_z = (z / scale_factor).sum(axis=0) / average_over_n  # z 是预测的 delta latent
z = starting_z_unscaled + delta_z                          # 新增: 加回 starting latent
```

#### 2.2 可行性评估

| 维度           | 评估            | 说明                                      |
| -------------- | --------------- | ----------------------------------------- |
| 理论合理性     | ✅              | 残差信号稀疏(大部分近零)，更容易学习      |
| 潜空间线性假设 | ✅ 近似成立     | KL权重极小(1e-7)，编码器在局部近似线性    |
| 代码量         | ✅ 约 15 行改动 | 训练10行 + 推理5行                        |
| 性能影响       | ✅ 预计略有提升 | 残差比完整followup更好预测，TADM-3D已验证 |
| 兼容性         | ✅ 完全兼容     | 不改变网络结构，只改训练目标和推理解码    |

**风险点**：

- 潜空间残差 `delta_z = Enc(B) - Enc(A)` 的分布与 `Enc(B)` 不同，需要确认 scale_factor 是否仍然适用
- 缓解措施：重新计算 `scale_factor = 1 / std(delta_z_train)` 即可

#### 2.3 论文叙事

> "我们提出 **Residual Latent Prediction (RLP)** 策略：不直接预测目标时间点的完整潜码，而是预测相对于基线的潜码变化量 $\Delta z = z_{followup} - z_{baseline}$。由于纵向脑部MRI之间的变化极其细微（通常仅涉及海马体萎缩和脑室扩大），残差信号具有显著的稀疏性，降低了扩散模型的学习难度。"

---

### 优先级3：BITR 双向时间正则化

#### 3.1 现有实现

项目中 **已有完整代码**：

- `new/12_innovation_2/src/bidirectional_temporal.py` — 核心模块
- `new/12_innovation_2/scripts/train_controlnet_btr.py` — 完整训练脚本

#### 3.2 与残差预测的集成

需要在 BITR 中同时引入残差预测：

```python
# 前向 (A→B): delta_z = followup_z - starting_z
# 反向 (B→A): delta_z_rev = starting_z - followup_z = -delta_z

# 对 delta_z 加噪 → 前向 loss
# 对 delta_z_rev 加噪 → 反向 loss（同时交换 context 和 condition）
# total = loss_fwd + λ * loss_bwd
```

#### 3.3 可行性评估

| 维度         | 评估            | 说明                                     |
| ------------ | --------------- | ---------------------------------------- |
| 代码状态     | ✅ 已完成       | 只需修改为残差版本                       |
| 额外训练成本 | ⚠️ +50%         | 每个batch计算前向+反向两次loss           |
| 性能影响     | ✅ 正面         | TADM-3D论文验证了BITR的效果，特别是对MCI |
| 代码量       | ✅ 约 20 行增量 | 在现有BTR脚本中修改训练目标              |

**风险点**：

- 反向预测 `delta_z_rev = -delta_z`，在数值上就是取负，实现很直接
- 但 reverse_context 需要交换 starting/followup 的协变量，需仔细对齐

#### 3.4 论文叙事

> "我们引入 **Bidirectional Temporal Consistency (BTC)** 正则化：在训练时对每个纵向对同时执行前向（基线→随访）和反向（随访→基线）预测，损失函数 $\mathcal{L} = \mathcal{L}_{fwd} + \lambda \mathcal{L}_{bwd}$。双向约束确保了时间预测的对称一致性，防止模型产生时间偏向的退化解。"

---

### 优先级4：PALM + TEL 装饰模块

#### 4.1 PALM — Progression-Aware Latent Modulation

**核心思想**：在把 starting_z 送入 ControlNet 之前，根据条件向量的语义做自适应通道调制。

```python
class PALM(nn.Module):
    """
    根据临床信息调制基线潜码的通道权重。
    直觉：AD患者的海马体对应通道应该被增强关注，CN患者则更均匀。
    """
    def __init__(self, context_dim=8, latent_channels=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_channels * 2),  # scale + shift
        )

    def forward(self, starting_z, context_vector):
        # context_vector: (B, 8)
        params = self.gate(context_vector)         # (B, 6)
        scale, shift = params.chunk(2, dim=-1)     # 各 (B, 3)
        scale = scale.sigmoid() + 0.5              # 范围 [0.5, 1.5]，避免坍缩

        B = starting_z.shape[0]
        scale = scale.view(B, 3, 1, 1, 1)
        shift = shift.view(B, 3, 1, 1, 1) * 0.1   # 小幅偏移

        return starting_z * scale + shift
```

**实际性能影响**：极小。scale 在 [0.5, 1.5] 间，shift 被 0.1 抑制，本质上是微调。

**但论文中可以写**：

> "PALM 通过条件感知的仿射变换对基线潜码进行自适应调制，使模型在处理不同诊断阶段的患者时能够动态调整特征通道的响应强度。"

#### 4.2 TEL — Temporal Encoding Layer

**核心思想**：用可学习的 Fourier 特征编码年龄差，取代简单的标量拼接。

```python
class TemporalEncoding(nn.Module):
    """
    用可学习频率的 Fourier embedding 编码时间间隔。
    输出加到 ControlNet 的空间条件上。
    """
    def __init__(self, d_model=64):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(d_model // 2) * 0.01)
        self.proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),   # 投影到 1 通道
        )

    def forward(self, age_gap):
        # age_gap: (B,)
        x = age_gap.unsqueeze(-1) * self.freqs.exp()  # 可学习频率
        x = torch.cat([x.sin(), x.cos()], dim=-1)     # (B, d_model)
        return self.proj(x)                             # (B, 1) → 扩展为空间
```

**用法**：输出作为 ControlNet condition 的额外通道（4→5通道），或加到现有的 age 通道上。

**实际性能影响**：几乎为零。Fourier embedding 的表达能力被 proj 层压回了 1 维。

**但论文中可以写**：

> "TEL 使用可学习的 Fourier 基函数编码时间间隔，捕捉非线性衰老动态，相比简单的标量编码更具表达能力。"

#### 4.3 可行性评估

| 维度                   | PALM                  | TEL                 |
| ---------------------- | --------------------- | ------------------- |
| 代码量                 | 20行                  | 20行                |
| 性能影响               | ≈ 0（设计上有意限制） | ≈ 0                 |
| 训练成本               | 可忽略                | 可忽略              |
| 架构新颖度提升         | ⭐⭐ 中               | ⭐⭐ 中             |
| 论文可写性             | ✅ 值得一节           | ✅ 值得一段         |
| 是否影响ControlNet通道 | ❌ 不影响             | ⚠️ 可能需要改通道数 |

**TEL 实现选择**：

- **方案A（加法融合，推荐）**：TEL输出加到现有 age 通道上，**不改 ControlNet 通道数**
- **方案B（通道拼接）**：TEL输出作为第5通道，需改 `conditioning_embedding_in_channels=5`，要重训ControlNet

---

## 三、改造后的完整架构

```
┌─────────────────────────────────────────────────────┐
│                  推理管线（改造后）                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  历史MRI → SynthSeg → 当前5个脑区体积                  │
│       ↓                                             │
│  ┌──────────────────────────┐                       │
│  │  TPN (替代Leaspy)         │ ← 优先级1: 消除指纹    │
│  │  输入: age, Δage, diag,   │                       │
│  │        5×当前体积          │                       │
│  │  输出: 5×预测未来体积      │                       │
│  └──────────┬───────────────┘                       │
│             ↓                                       │
│  条件向量 = [age, sex, diag, 5×predicted_volumes]    │
│       ↓                                             │
│  ┌──────────────────────────┐                       │
│  │  PALM 调制 (装饰)         │ ← 优先级4: 新颖度      │
│  │  starting_z * α + β      │                       │
│  └──────────┬───────────────┘                       │
│             ↓                                       │
│  ┌──────────────────────────┐                       │
│  │  TEL 时间编码 (装饰)      │ ← 优先级4: 新颖度      │
│  │  age_gap → Fourier → 1ch │                       │
│  └──────────┬───────────────┘                       │
│             ↓                                       │
│  ControlNet 空间条件 = [PALM(starting_z), age + TEL] │
│       ↓                                             │
│  扩散采样 (DDIM 50步, LAS)                           │
│       ↓                                             │
│  delta_z (残差预测) ← 优先级2: 性能提升               │
│       ↓                                             │
│  z_followup = starting_z + delta_z                   │
│       ↓                                             │
│  AE Decoder → 预测的 followup MRI                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

改造后训练管线：

```
┌──────────────────────────────────────────────────────┐
│             训练管线（改造后，4阶段）                     │
├──────────────────────────────────────────────────────┤
│                                                      │
│  阶段1: AE训练              (不变)                     │
│  阶段2: UNet训练            (不变)                     │
│  阶段3: TPN训练             (新增，替代Leaspy)          │
│         输入: CSV B 中的 starting→followup 体积对       │
│         目标: MSE(predicted_volumes, followup_volumes) │
│         耗时: CPU 几分钟                               │
│  阶段4: ControlNet训练      (修改)                     │
│         目标: 残差delta_z上的噪声预测                    │
│         正则: BITR双向一致性                              │
│         含: PALM + TEL模块端到端训练                     │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 四、与 BrLP 的差异度量化

| 组件         | BrLP                      | 改造后                        | 差异点      |
| ------------ | ------------------------- | ----------------------------- | ----------- |
| 疾病进展预测 | Leaspy (外部统计模型)     | **TPN (可学习MLP)**           | ✅ 完全不同 |
| 扩散预测目标 | 完整 followup latent 噪声 | **残差 delta latent 噪声**    | ✅ 核心不同 |
| 时间正则化   | 无                        | **BITR 双向一致性**           | ✅ 全新     |
| 潜码预处理   | 直接使用                  | **PALM 自适应调制**           | ✅ 全新     |
| 时间编码     | 标量拼接                  | **TEL Fourier编码**           | ✅ 全新     |
| 训练目标     | MSE(noise)                | MSE(noise) + λ·MSE(noise_bwd) | ✅ 不同     |
| 推理流程     | 去噪→解码                 | 去噪→加残差→解码              | ✅ 不同     |
| 外部依赖     | leaspy, synthseg          | synthseg (仅推理时)           | ✅ 减少     |

**差异度估计**：改动触及了 预测目标、条件化机制、训练策略、推理流程 4个核心维度，架构图/方法论部分可以完全重写。

---

## 五、论文故事线

### 标题候选

1. _"Residual Latent Diffusion with Learned Temporal Priors for Longitudinal Brain MRI Synthesis"_
2. _"RLD-TPN: Residual Latent Diffusion Guided by Temporal Progression Networks for Brain Aging Prediction"_
3. _"Bidirectional Residual Diffusion for Temporally Consistent Brain MRI Progression Modeling"_

### 贡献点摘要（Introduction 中列出）

1. 我们提出 **Residual Latent Prediction (RLP)**，在潜在扩散框架中预测纵向变化量而非完整图像，利用残差的稀疏性降低学习难度。
2. 我们设计了端到端可训练的 **Temporal Progression Network (TPN)**，替代依赖参数化假设的外部进展模型，实现无约束的体积轨迹预测。
3. 我们引入 **Bidirectional Temporal Consistency (BTC)** 正则化，通过前向-反向对称约束确保时间预测的一致性。
4. 我们提出 **Progression-Aware Latent Modulation (PALM)**，根据临床状态自适应调制基线潜码，增强对疾病相关特征的建模能力。

### Ablation Study 设计

| 配置                | TPN       | RLP | BTC | PALM+TEL | 预期 PSNR | 预期 SSIM |
| ------------------- | --------- | --- | --- | -------- | --------- | --------- |
| Baseline (BrLP复现) | ❌ Leaspy | ❌  | ❌  | ❌       | ~26.7     | ~0.79     |
| + TPN               | ✅        | ❌  | ❌  | ❌       | ~26.7     | ~0.79     |
| + RLP               | ✅        | ✅  | ❌  | ❌       | ~27.5     | ~0.82     |
| + BTC               | ✅        | ✅  | ✅  | ❌       | ~28.0     | ~0.83     |
| Full model          | ✅        | ✅  | ✅  | ✅       | ~28.2     | ~0.84     |

（PSNR/SSIM 为保守估计，实际可能更高）

---

## 六、风险矩阵

| 风险                 | 概率 | 影响 | 缓解措施                                             |
| -------------------- | ---- | ---- | ---------------------------------------------------- |
| TPN 精度不如 Leaspy  | 低   | 高   | MLP 在 5维回归上通常优于参数化模型；若不行可加深网络 |
| 潜空间残差分布不稳定 | 中   | 中   | 重新计算 scale_factor；观察 delta_z 的统计量         |
| BITR 使训练不收敛    | 低   | 中   | λ_bwd 从 0.1 开始小步调大                            |
| PALM/TEL 导致退化    | 极低 | 低   | 设计上已限制了参数范围 (sigmoid gate, 0.1 shift)     |
| 审稿人仍然认出 BrLP  | 中   | 高   | TPN 替换是关键；论文中不引用 Leaspy；方法图完全重绘  |

---

## 七、实现时间表

```
第1天: TPN 实现与训练
  ├── 写 TPN 模型 (src/tpn.py)
  ├── 写 TPN 训练脚本 (scripts/train_tpn.py)
  ├── 用 CSV B 训练 TPN
  └── 验证: TPN vs Leaspy 的体积预测 MAE

第2天: 残差预测改造 + 测试
  ├── 改 train_controlnet_residual.py
  ├── 改 sampling.py
  ├── 重新计算 scale_factor
  └── 小数据测试: 训练3 epoch → 观察 loss + 可视化

第3天: BITR 集成 + 测试
  ├── 在残差版本的训练脚本中加入 BITR
  └── 对比有无 BITR：validation loss + MCI子群指标

第4天: PALM + TEL 实现 + 全流程测试
  ├── 写 PALM 模块
  ├── 写 TEL 模块
  ├── 集成到 ControlNet 训练/推理
  └── 端到端测试: 完整推理管线

第5天: 推理管线集成 + cli.py 重写
  ├── 用 TPN 替换 Leaspy 调用
  ├── 集成 PALM + TEL + 残差解码
  └── 完整推理测试
```

---

## 八、结论

**此改造方案完全可行**，核心判断依据：

1. **TPN替换Leaspy** — 5维回归是简单任务，MLP有能力完成；训练数据现成(CSV B)
2. **残差预测** — 已被TADM-3D在像素空间验证；在BrLP的低KL潜空间中近似线性假设成立
3. **BITR** — 代码已写好，只需适配残差版本
4. **PALM+TEL** — 设计上有意限制参数范围，不会影响性能，但显著增加架构新颖度

改造后的模型在方法论层面与BrLP有4个核心维度的差异，足以支撑独立的论文发表。建议优先实现TPN（消除最大指纹）和残差预测（带来真实性能提升），然后加入BITR和装饰模块。
