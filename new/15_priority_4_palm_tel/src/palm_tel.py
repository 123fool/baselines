"""
Priority 4: PALM + TEL Decoration Modules
==========================================

PALM — Progression-Aware Latent Modulation
  Modulates starting_z channels based on clinical context (8-dim).
  Channel-wise affine: out = z * scale + shift
  scale ∈ [0.5, 1.5] via sigmoid gate, shift ∈ small range (×0.1).

TEL — Temporal Encoding Layer
  Fourier embedding of age_gap with learnable frequencies.
  Output (scalar per sample) added to the age channel of ControlNet condition.
"""

import torch
import torch.nn as nn


class PALM(nn.Module):
    """
    Progression-Aware Latent Modulation.

    根据临床状态 (8-dim cross-attention context) 自适应调制基线潜码的通道权重。
    AD 患者的海马体对应通道应被增强关注，CN 患者则更均匀。

    设计上有意限制参数范围 (sigmoid gate → [0.5,1.5], shift × 0.1) 以保证训练稳定性。
    """

    def __init__(self, context_dim=8, latent_channels=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.GELU(),
            nn.Linear(32, latent_channels * 2),  # scale + shift
        )

    def forward(self, starting_z, context_vector):
        """
        Args:
            starting_z:     (B, C, D, H, W) latent tensor
            context_vector: (B, 8) clinical covariates
        Returns:
            modulated_z:    (B, C, D, H, W)
        """
        context_vector = context_vector.to(dtype=self.gate[0].weight.dtype)
        params = self.gate(context_vector)              # (B, 2C)
        scale, shift = params.chunk(2, dim=-1)          # each (B, C)
        scale = scale.sigmoid() + 0.5                   # [0.5, 1.5]

        B = starting_z.shape[0]
        scale = scale.view(B, -1, 1, 1, 1)
        shift = shift.view(B, -1, 1, 1, 1) * 0.1       # small perturbation

        return starting_z * scale + shift


class TEL(nn.Module):
    """
    Temporal Encoding Layer.

    用可学习频率的 Fourier 基函数编码时间间隔 (age_gap)，
    捕捉非线性衰老动态，相比简单的标量 age 编码更具表达能力。
    输出加到 ControlNet 空间条件的 age 通道上 (加法融合，不改通道数)。
    """

    def __init__(self, d_model=64):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(d_model // 2) * 0.01)
        self.proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, age_gap):
        """
        Args:
            age_gap: (B,) normalized age difference (followup_age - starting_age)
        Returns:
            encoding: (B, 1) temporal encoding to add to age channel
        """
        age_gap = age_gap.to(dtype=self.freqs.dtype)
        x = age_gap.unsqueeze(-1) * self.freqs.exp()       # (B, d_model//2)
        x = torch.cat([x.sin(), x.cos()], dim=-1)          # (B, d_model)
        return self.proj(x)                                  # (B, 1)
