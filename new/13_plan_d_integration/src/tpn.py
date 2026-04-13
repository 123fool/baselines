"""
Temporal Progression Network (TPN) v3
======================================
可学习的 MLP 替代 Leaspy logistic mixed-effects model，
用于预测未来脑区体积。

v3 = v1 的简单 Sequential 架构 + 4 个额外特征
  - 输入 14 维: 原始 10 维 + age_ratio, vol_mean, vol_std, age_gap²
  - 纯 Sequential MLP (无内部 skip connection)
  - 仅输出层有残差 (out = current_vols + delta)
"""

import torch
import torch.nn as nn


class TemporalProgressionNetwork(nn.Module):
    """
    端到端可学习的疾病进展轨迹预测网络。
    替代基于外部统计模型 (Leaspy) 的体积估计方法。
    """

    def __init__(self, in_dim=14, hidden_dim=128, out_dim=5, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.use_residual = True

    def forward(self, x):
        """
        Args:
            x: (B, 14) — [current_age, target_age, sex, diagnosis,
                           cortex, hippo, amyg, wm, vent,
                           age_gap, age_ratio, vol_mean, vol_std, age_gap_sq]
        Returns:
            (B, 5) — predicted future volumes, clamped to [0, 1]
        """
        delta = self.net(x)
        if self.use_residual:
            out = x[:, 4:9] + delta
        else:
            out = delta
        return out.clamp(0.0, 1.0)

    @staticmethod
    def build_input(current_age, target_age, sex, diagnosis, current_volumes):
        """
        构建 TPN v3 输入向量 (14维)。
        """
        if not isinstance(current_age, torch.Tensor):
            current_age = torch.tensor([current_age], dtype=torch.float32)
        if not isinstance(target_age, torch.Tensor):
            target_age = torch.tensor([target_age], dtype=torch.float32)
        if not isinstance(sex, torch.Tensor):
            sex = torch.tensor([sex], dtype=torch.float32)
        if not isinstance(diagnosis, torch.Tensor):
            diagnosis = torch.tensor([diagnosis], dtype=torch.float32)
        if not isinstance(current_volumes, torch.Tensor):
            current_volumes = torch.tensor(current_volumes, dtype=torch.float32)

        if current_age.dim() == 0:
            current_age = current_age.unsqueeze(0)
        if target_age.dim() == 0:
            target_age = target_age.unsqueeze(0)
        if sex.dim() == 0:
            sex = sex.unsqueeze(0)
        if diagnosis.dim() == 0:
            diagnosis = diagnosis.unsqueeze(0)
        if current_volumes.dim() == 1:
            current_volumes = current_volumes.unsqueeze(0)

        age_gap = target_age - current_age
        age_ratio = age_gap / (current_age + 1e-8)
        vol_mean = current_volumes.mean(dim=-1, keepdim=True)
        vol_std = current_volumes.std(dim=-1, keepdim=True)
        age_gap_sq = age_gap ** 2

        return torch.cat([
            current_age, target_age, sex, diagnosis,
            current_volumes,
            age_gap, age_ratio, vol_mean, vol_std, age_gap_sq,
        ], dim=-1)
