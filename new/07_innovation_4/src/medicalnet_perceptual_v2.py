"""
3D MedicalNet Perceptual Loss v2 — Multi-Scale Feature Matching.

v2 improvements over v1:
  1. Multi-scale feature extraction (layers 1-4) instead of final layer only
  2. L1 distance instead of L2 (more robust, matches VGG perceptual loss behavior)
  3. Less aggressive downsampling (80x96x80 instead of 64x72x64)
  4. Per-layer feature normalization before distance computation

This module AUGMENTS (not replaces) the original MONAI PerceptualLoss.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ============================================================
# 3D ResNet-10 (self-contained, MedicalNet architecture)
# ============================================================

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation,
                     stride=stride, padding=dilation, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet3D(nn.Module):
    """3D ResNet for multi-scale feature extraction."""

    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def extract_features(self, x) -> List[torch.Tensor]:
        """Extract multi-scale features from all 4 residual layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # 64-ch
        f2 = self.layer2(f1)  # 128-ch
        f3 = self.layer3(f2)  # 256-ch
        f4 = self.layer4(f3)  # 512-ch

        return [f1, f2, f3, f4]


def _load_resnet10(pretrained_path: Optional[str] = None) -> ResNet3D:
    """Build ResNet-10 and optionally load MedicalNet pretrained weights."""
    model = ResNet3D(BasicBlock, [1, 1, 1, 1])
    if pretrained_path is not None and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return model


# ============================================================
# MedicalNet 3D Perceptual Loss v2 — Multi-Scale
# ============================================================

class MedicalNet3DPerceptualLoss(nn.Module):
    """
    Multi-scale 3D perceptual loss using MedicalNet ResNet-10.

    v2 improvements:
      - Extracts features from layers 1-4 (multi-scale, fine→coarse)
      - Uses L1 distance (robust, consistent with VGG perceptual losses)
      - Per-layer instance normalization for scale invariance
      - Moderate downsampling (80×96×80) to preserve more detail

    This loss is designed to AUGMENT the original MONAI PerceptualLoss,
    not replace it. Use with a small weight (e.g., 0.0005).

    Args:
        pretrained_path: Path to MedicalNet resnet_10_23dataset.pth
        downsample_size: Input downsampling target. (80,96,80) balances
                        memory and detail preservation.
        layer_weights: Per-layer loss weights [layer1, layer2, layer3, layer4].
                      Default gives equal weight to all scales.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        downsample_size: Tuple[int, int, int] = (80, 96, 80),
        layer_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.model = _load_resnet10(pretrained_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.downsample_size = downsample_size

        if layer_weights is None:
            # Equal weight across 4 layers
            layer_weights = [0.25, 0.25, 0.25, 0.25]
        self.layer_weights = layer_weights

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_size is not None:
            return F.interpolate(x, size=self.downsample_size,
                                 mode='trilinear', align_corners=False)
        return x

    @staticmethod
    def _instance_norm(feat: torch.Tensor) -> torch.Tensor:
        """Instance normalization per sample for scale invariance."""
        b, c = feat.shape[:2]
        feat_flat = feat.view(b, c, -1)
        mean = feat_flat.mean(dim=-1, keepdim=True)
        std = feat_flat.std(dim=-1, keepdim=True) + 1e-7
        normed = (feat_flat - mean) / std
        return normed.view_as(feat)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_ds = self._downsample(prediction)
        tgt_ds = self._downsample(target)

        n_channels = pred_ds.shape[1]
        total_loss = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)

        for ch in range(n_channels):
            pred_feats = self.model.extract_features(pred_ds[:, ch:ch+1])
            with torch.no_grad():
                tgt_feats = self.model.extract_features(tgt_ds[:, ch:ch+1])

            for i, (pf, tf) in enumerate(zip(pred_feats, tgt_feats)):
                pf_norm = self._instance_norm(pf)
                tf_norm = self._instance_norm(tf)
                total_loss = total_loss + self.layer_weights[i] * F.l1_loss(pf_norm, tf_norm)

        return total_loss / max(n_channels, 1)
