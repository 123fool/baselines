"""
Weighted Loss Functions for Hippocampal Region Attention.

Provides region-weighted L1 and MSE losses that apply higher weight
to hippocampus + amygdala regions critical for MCI progression.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RegionWeightedL1Loss(nn.Module):
    """
    L1 loss with spatial region weighting.
    
    weight_map should have shape broadcastable to the input,
    typically (1, 1, D, H, W) or (N, 1, D, H, W).
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weight_map: Tensor,
    ) -> Tensor:
        """
        Args:
            prediction: Model output (N, C, D, H, W)
            target: Ground truth (N, C, D, H, W)
            weight_map: Spatial weights (N, 1, D, H, W) or broadcastable
            
        Returns:
            Weighted L1 loss scalar
        """
        diff = torch.abs(prediction - target)
        weighted_diff = diff * weight_map
        
        if self.reduction == 'mean':
            # Normalize by sum of weights to avoid scale dependency
            return weighted_diff.sum() / weight_map.sum()
        elif self.reduction == 'sum':
            return weighted_diff.sum()
        else:
            return weighted_diff


class RegionWeightedMSELoss(nn.Module):
    """
    MSE loss with spatial region weighting for latent space.
    
    Used in ControlNet training where the noise prediction loss
    is computed in latent space.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weight_map: Tensor,
    ) -> Tensor:
        """
        Args:
            prediction: Predicted noise (N, C, D, H, W)
            target: Actual noise (N, C, D, H, W)
            weight_map: Spatial weights (N, C, D, H, W) or broadcastable
            
        Returns:
            Weighted MSE loss scalar
        """
        diff_sq = (prediction - target) ** 2
        weighted_diff = diff_sq * weight_map
        
        if self.reduction == 'mean':
            return weighted_diff.sum() / weight_map.sum()
        elif self.reduction == 'sum':
            return weighted_diff.sum()
        else:
            return weighted_diff


class CombinedRegionLoss(nn.Module):
    """
    Combines standard uniform loss with region-weighted loss.
    
    total_loss = (1 - alpha) * uniform_loss + alpha * region_weighted_loss
    
    This ensures the model still learns overall reconstruction quality
    while focusing more on hippocampal regions.
    """
    
    def __init__(self, alpha: float = 0.5, loss_type: str = 'l1'):
        """
        Args:
            alpha: Blending factor. 0 = fully uniform, 1 = fully region-weighted.
            loss_type: 'l1' or 'mse'
        """
        super().__init__()
        self.alpha = alpha
        if loss_type == 'l1':
            self.uniform_loss = nn.L1Loss()
            self.weighted_loss = RegionWeightedL1Loss()
        elif loss_type == 'mse':
            self.uniform_loss = nn.MSELoss()
            self.weighted_loss = RegionWeightedMSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        weight_map: Tensor,
    ) -> Tensor:
        uniform = self.uniform_loss(prediction, target)
        weighted = self.weighted_loss(prediction, target, weight_map)
        return (1 - self.alpha) * uniform + self.alpha * weighted
