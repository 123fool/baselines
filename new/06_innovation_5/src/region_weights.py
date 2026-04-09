"""
Region Weight Map Generation for Hippocampal Attention Weighting.

Generates spatial weight maps from SynthSeg segmentation that emphasize
hippocampus + amygdala regions for MCI-focused training.
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from typing import Tuple, Optional

# SynthSeg label IDs for regions of interest
HIPPOCAMPUS_LABELS = [17, 53]  # left_hippocampus, right_hippocampus
AMYGDALA_LABELS = [18, 54]     # left_amygdala, right_amygdala
ENTORHINAL_LABELS = []          # SynthSeg does not segment entorhinal directly

# Combined ROI labels for MCI-relevant structures
MCI_ROI_LABELS = HIPPOCAMPUS_LABELS + AMYGDALA_LABELS

# Default spatial shapes
AE_INPUT_SHAPE = (120, 144, 120)
LATENT_SHAPE_AE = (15, 18, 15)    # before padding
LATENT_SHAPE_DM = (16, 20, 16)    # after padding to be divisible by 4


def load_segmentation(segm_path: str) -> np.ndarray:
    """Load a SynthSeg segmentation NIfTI file."""
    segm = nib.load(segm_path)
    return segm.get_fdata().round().astype(np.int32)


def create_roi_mask(
    segm: np.ndarray,
    roi_labels: list = None,
) -> np.ndarray:
    """
    Create a binary mask for the specified ROI labels.
    
    Args:
        segm: SynthSeg segmentation array (integer labels)
        roi_labels: List of label IDs to include in the mask
        
    Returns:
        Binary mask (float32), shape matching segm
    """
    if roi_labels is None:
        roi_labels = MCI_ROI_LABELS
    mask = np.zeros_like(segm, dtype=np.float32)
    for label in roi_labels:
        mask[segm == label] = 1.0
    return mask


def create_weight_map_image_space(
    segm_path: str,
    target_shape: Tuple[int, int, int] = AE_INPUT_SHAPE,
    roi_weight: float = 3.0,
    background_weight: float = 1.0,
    smooth_sigma: float = 2.0,
    roi_labels: list = None,
) -> torch.Tensor:
    """
    Create a weight map in image space for AE training.
    
    ROI regions (hippocampus + amygdala) get higher weight.
    A Gaussian smoothing is applied at the boundary to avoid
    sharp transitions.
    
    Args:
        segm_path: Path to the SynthSeg segmentation NIfTI
        target_shape: Target spatial shape (must match AE input)
        roi_weight: Weight multiplier for ROI regions
        background_weight: Weight for non-ROI brain regions
        smooth_sigma: Gaussian smoothing sigma for boundary
        roi_labels: List of ROI label IDs
        
    Returns:
        Weight map tensor of shape (1, D, H, W)
    """
    if roi_labels is None:
        roi_labels = MCI_ROI_LABELS

    segm = load_segmentation(segm_path)
    roi_mask = create_roi_mask(segm, roi_labels)
    
    # Create a brain mask (any non-zero label)
    brain_mask = (segm > 0).astype(np.float32)
    
    # Build weight map: background_weight for brain, roi_weight for ROI
    weight_map = brain_mask * background_weight
    weight_map[roi_mask > 0] = roi_weight
    
    # Convert to tensor and resize to target shape
    weight_tensor = torch.from_numpy(weight_map).float().unsqueeze(0).unsqueeze(0)
    
    if weight_tensor.shape[2:] != target_shape:
        weight_tensor = F.interpolate(
            weight_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
    
    # Apply Gaussian smoothing at boundaries for stable gradients
    if smooth_sigma > 0:
        weight_tensor = _gaussian_smooth_3d(weight_tensor, sigma=smooth_sigma)
    
    # Ensure minimum weight of background_weight
    weight_tensor = weight_tensor.clamp(min=background_weight)
    
    # Normalize so mean weight = 1 (preserves effective learning rate)
    weight_tensor = weight_tensor / weight_tensor.mean()
    
    return weight_tensor.squeeze(0)  # (1, D, H, W)


def create_weight_map_latent_space(
    segm_path: str,
    latent_channels: int = 3,
    roi_weight: float = 3.0,
    background_weight: float = 1.0,
    roi_labels: list = None,
) -> torch.Tensor:
    """
    Create a weight map in latent space for ControlNet training.
    
    Downsamples the segmentation mask to the latent space dimensions
    using average pooling, then pads to match DM input shape.
    
    Args:
        segm_path: Path to the SynthSeg segmentation NIfTI
        latent_channels: Number of latent channels
        roi_weight: Weight multiplier for ROI regions
        background_weight: Weight for non-ROI regions
        roi_labels: List of ROI label IDs
        
    Returns:
        Weight map tensor of shape (C, D, H, W) matching LATENT_SHAPE_DM
    """
    if roi_labels is None:
        roi_labels = MCI_ROI_LABELS

    segm = load_segmentation(segm_path)
    roi_mask = create_roi_mask(segm, roi_labels)
    
    # Resize to AE input shape first
    mask_tensor = torch.from_numpy(roi_mask).float().unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(
        mask_tensor,
        size=AE_INPUT_SHAPE,
        mode='trilinear',
        align_corners=False
    )
    
    # Average pool to latent space dimensions (compression ratio = 8x)
    # AE has 3 downsampling layers, each 2x, so total 8x
    mask_latent = F.adaptive_avg_pool3d(mask_tensor, LATENT_SHAPE_AE)
    
    # Convert soft mask to weight map
    # Where mask_latent > 0, the region partially overlaps with ROI
    weight_map = torch.ones_like(mask_latent) * background_weight
    weight_map = weight_map + mask_latent * (roi_weight - background_weight)
    
    # Pad to DM input shape (divisible by 4)
    pad_d = LATENT_SHAPE_DM[0] - LATENT_SHAPE_AE[0]  # 1
    pad_h = LATENT_SHAPE_DM[1] - LATENT_SHAPE_AE[1]  # 2
    pad_w = LATENT_SHAPE_DM[2] - LATENT_SHAPE_AE[2]  # 1
    # Pad: (w_left, w_right, h_left, h_right, d_left, d_right)
    weight_map = F.pad(
        weight_map,
        (0, pad_w, 0, pad_h, 0, pad_d),
        mode='constant',
        value=background_weight
    )
    
    # Normalize so mean weight = 1
    weight_map = weight_map / weight_map.mean()
    
    # Expand to match latent channels
    weight_map = weight_map.squeeze(0).expand(latent_channels, -1, -1, -1)
    
    return weight_map  # (C, D, H, W)


def batch_weight_maps_image_space(
    segm_paths: list,
    **kwargs
) -> torch.Tensor:
    """
    Create a batch of weight maps for AE training.
    
    Args:
        segm_paths: List of segmentation file paths
        **kwargs: Arguments passed to create_weight_map_image_space
        
    Returns:
        Batched weight maps tensor of shape (N, 1, D, H, W)
    """
    maps = []
    for path in segm_paths:
        wmap = create_weight_map_image_space(path, **kwargs)
        maps.append(wmap)
    return torch.stack(maps, dim=0)


def batch_weight_maps_latent_space(
    segm_paths: list,
    **kwargs
) -> torch.Tensor:
    """
    Create a batch of weight maps for ControlNet training.
    
    Args:
        segm_paths: List of segmentation file paths
        **kwargs: Arguments passed to create_weight_map_latent_space
        
    Returns:
        Batched weight maps tensor of shape (N, C, D, H, W)
    """
    maps = []
    for path in segm_paths:
        wmap = create_weight_map_latent_space(path, **kwargs)
        maps.append(wmap)
    return torch.stack(maps, dim=0)


def _gaussian_smooth_3d(
    tensor: torch.Tensor,
    sigma: float = 2.0,
    kernel_size: int = 0,
) -> torch.Tensor:
    """
    Apply 3D Gaussian smoothing to a tensor.
    
    Args:
        tensor: Input tensor (N, C, D, H, W)
        sigma: Gaussian standard deviation
        kernel_size: Kernel size (auto-computed if 0)
        
    Returns:
        Smoothed tensor
    """
    if kernel_size == 0:
        kernel_size = int(2 * round(3 * sigma) + 1)
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create separable 3D kernel
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)
    
    padding = kernel_size // 2
    channels = tensor.shape[1]
    
    # Apply per-channel
    kernel_3d = kernel_3d.expand(channels, 1, -1, -1, -1).to(tensor.device)
    smoothed = F.conv3d(
        tensor,
        kernel_3d,
        padding=padding,
        groups=channels
    )
    
    return smoothed


def precompute_latent_weight_map(
    segm_path: str,
    save_path: str,
    **kwargs
) -> None:
    """
    Pre-compute and save a latent space weight map for faster training.
    
    Args:
        segm_path: Path to SynthSeg segmentation
        save_path: Path to save the weight map (.npz)
        **kwargs: Arguments passed to create_weight_map_latent_space
    """
    weight_map = create_weight_map_latent_space(segm_path, **kwargs)
    np.savez_compressed(save_path, data=weight_map.numpy())
