"""
3D MedicalNet Perceptual Loss for Innovation 4.

Adapted from 3D-MedDiffusion (IEEE TMI 2025):
  - Uses MedicalNet ResNet-10 pretrained on 23 medical imaging datasets
  - True 3D feature extraction (no fake-3D slice sampling)
  - Channel-wise processing for multi-channel inputs

Key difference from BrLP's original PerceptualLoss:
  BrLP:  2D VGG squeeze + fake_3d_ratio=0.2 → only 20% slices, 2D features
  Ours:  3D ResNet-10 → full volumetric feature matching

Reference:
  Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis", arXiv:1904.00625
  ShanghaiTech IMPACT, "3D MedDiffusion", IEEE TMI 2025
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# 3D ResNet-10 (self-contained, no torch.hub needed)
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
    """3D ResNet for feature extraction (MedicalNet architecture)."""

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _load_resnet10(pretrained_path: Optional[str] = None) -> ResNet3D:
    """Build ResNet-10 and optionally load MedicalNet pretrained weights."""
    model = ResNet3D(BasicBlock, [1, 1, 1, 1])
    if pretrained_path is not None and os.path.exists(pretrained_path):
        device = 'cpu'
        state = torch.load(pretrained_path, map_location=device, weights_only=False)
        if 'state_dict' in state:
            state = state['state_dict']
        # Strip DataParallel 'module.' prefix if present
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return model


# ============================================================
# MedicalNet 3D Perceptual Loss
# ============================================================

def _mednet_normalize(x):
    """Z-score normalization per sample."""
    return (x - x.mean()) / (x.std() + 1e-7)


def _feature_normalize(x, eps=1e-7):
    """L2-normalize features along channel dimension."""
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm + eps)


class MedicalNet3DPerceptualLoss(nn.Module):
    """
    True 3D perceptual loss using MedicalNet ResNet-10.

    Replaces MONAI's fake-3D PerceptualLoss (2D VGG squeeze + 20% slice sampling)
    with a genuine 3D feature extractor pretrained on 23 medical imaging datasets.

    To manage GPU memory, inputs are downsampled before feature extraction.
    The ResNet is frozen (no grad), but gradients flow through the downsample→diff path.

    Args:
        pretrained_path: Path to resnet_10_23dataset.pth weights file.
        downsample_size: Tuple (D,H,W) to downsample inputs to before ResNet.
                        Default (64,72,64) keeps memory manageable on 24GB GPUs.

    Input shapes:
        prediction, target: (N, C, D, H, W) where C can be 1 or more.

    Output:
        Scalar loss value (mean L2 distance in normalized feature space).
    """

    def __init__(self, pretrained_path: Optional[str] = None,
                 downsample_size: tuple = (64, 72, 64)):
        super().__init__()
        self.model = _load_resnet10(pretrained_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.downsample_size = downsample_size

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample input to reduce memory. Gradients flow through interpolation."""
        if self.downsample_size is not None:
            return F.interpolate(x, size=self.downsample_size, mode='trilinear', align_corners=False)
        return x

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Downsample to manageable size first
        prediction_ds = self._downsample(prediction)
        target_ds = self._downsample(target)

        prediction_ds = _mednet_normalize(prediction_ds)
        target_ds = _mednet_normalize(target_ds)

        # Process each channel independently (ResNet expects 1-channel input)
        all_feats_pred = []
        all_feats_tgt = []
        for ch in range(prediction_ds.shape[1]):
            pred_ch = prediction_ds[:, ch:ch+1, ...]
            tgt_ch = target_ds[:, ch:ch+1, ...]
            all_feats_pred.append(self.model(pred_ch))
            all_feats_tgt.append(self.model(tgt_ch))

        feats_pred = torch.cat(all_feats_pred, dim=1)
        feats_tgt = torch.cat(all_feats_tgt, dim=1)

        feats_pred = _feature_normalize(feats_pred)
        feats_tgt = _feature_normalize(feats_tgt)

        diff = (feats_pred - feats_tgt) ** 2
        return diff.sum(dim=1, keepdim=True).mean()
