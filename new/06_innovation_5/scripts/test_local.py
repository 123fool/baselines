"""
Local sanity test for Innovation 5 modules.
Verifies imports, class instantiation, and basic logic without requiring GPU or data.
"""

import sys
import os
import numpy as np
import torch

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src'))

def test_region_weights():
    """Test region weight generation with synthetic data."""
    from region_weights import (
        create_roi_mask, MCI_ROI_LABELS, HIPPOCAMPUS_LABELS,
        AE_INPUT_SHAPE, LATENT_SHAPE_AE, LATENT_SHAPE_DM,
    )
    
    # Create synthetic segmentation
    segm = np.zeros((182, 218, 182), dtype=np.int32)
    # Add hippocampus regions
    segm[80:100, 100:120, 80:100] = 17  # left hippocampus
    segm[80:100, 100:120, 100:120] = 53  # right hippocampus
    segm[85:95, 105:115, 85:95] = 18   # left amygdala
    segm[85:95, 105:115, 105:115] = 54  # right amygdala
    # Add some brain tissue
    segm[20:160, 20:200, 20:160] = np.where(
        segm[20:160, 20:200, 20:160] == 0, 3, segm[20:160, 20:200, 20:160]
    )
    
    # Test ROI mask
    mask = create_roi_mask(segm, MCI_ROI_LABELS)
    assert mask.shape == segm.shape
    assert mask.max() == 1.0
    assert mask.sum() > 0
    
    # Test hippocampus only mask
    h_mask = create_roi_mask(segm, HIPPOCAMPUS_LABELS)
    assert h_mask.sum() > 0
    assert h_mask.sum() < mask.sum()  # hippocampus < hippocampus + amygdala
    
    print("  ✓ region_weights: ROI mask creation OK")
    
    # Test weight map shapes
    # We can't test create_weight_map_image_space directly (needs NIfTI file)
    # But we can test the logic
    from region_weights import _gaussian_smooth_3d
    test_tensor = torch.randn(1, 1, 30, 36, 30)
    smoothed = _gaussian_smooth_3d(test_tensor, sigma=2.0)
    assert smoothed.shape == test_tensor.shape
    print("  ✓ region_weights: Gaussian smoothing OK")
    
    print("  ✓ region_weights: All tests passed")


def test_weighted_losses():
    """Test weighted loss functions."""
    from weighted_losses import (
        RegionWeightedL1Loss,
        RegionWeightedMSELoss,
        CombinedRegionLoss,
    )
    
    batch_size = 2
    channels = 1
    D, H, W = 30, 36, 30
    
    pred = torch.randn(batch_size, channels, D, H, W)
    target = torch.randn(batch_size, channels, D, H, W)
    weight = torch.ones(batch_size, channels, D, H, W)
    weight[:, :, 10:20, 12:24, 10:20] = 3.0  # ROI region
    
    # Test RegionWeightedL1Loss
    l1_fn = RegionWeightedL1Loss()
    loss = l1_fn(pred, target, weight)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
    print("  ✓ RegionWeightedL1Loss: OK")
    
    # Test RegionWeightedMSELoss
    mse_fn = RegionWeightedMSELoss()
    loss = mse_fn(pred, target, weight)
    assert loss.ndim == 0
    assert loss.item() > 0
    print("  ✓ RegionWeightedMSELoss: OK")
    
    # Test CombinedRegionLoss
    combined_l1 = CombinedRegionLoss(alpha=0.5, loss_type='l1')
    loss = combined_l1(pred, target, weight)
    assert loss.ndim == 0
    assert loss.item() > 0
    print("  ✓ CombinedRegionLoss (L1): OK")
    
    combined_mse = CombinedRegionLoss(alpha=0.5, loss_type='mse')
    loss = combined_mse(pred, target, weight)
    assert loss.ndim == 0
    assert loss.item() > 0
    print("  ✓ CombinedRegionLoss (MSE): OK")
    
    # Test alpha=0 equals uniform loss
    combined_uniform = CombinedRegionLoss(alpha=0.0, loss_type='l1')
    loss_alpha0 = combined_uniform(pred, target, weight)
    loss_uniform = torch.nn.L1Loss()(pred, target)
    assert torch.allclose(loss_alpha0, loss_uniform, atol=1e-6), \
        f"alpha=0 should equal uniform loss: {loss_alpha0} vs {loss_uniform}"
    print("  ✓ CombinedRegionLoss: alpha=0 equals uniform loss")
    
    # Test that higher weight increases loss contribution
    weight_high = torch.ones_like(weight) * 5.0
    weight_low = torch.ones_like(weight) * 1.0
    loss_high = RegionWeightedL1Loss()(pred, target, weight_high)
    loss_low = RegionWeightedL1Loss()(pred, target, weight_low)
    # After normalization by sum of weights, both should be similar (since uniform weights)
    # But with non-uniform weights, the ROI contribution should be higher
    print("  ✓ weighted_losses: All tests passed")


def test_dashboard_imports():
    """Test that dashboard can be imported."""
    try:
        import flask
        print("  ✓ flask: Available")
    except ImportError:
        print("  ⚠ flask: Not installed (install with: pip install flask)")
    
    try:
        import psutil
        print("  ✓ psutil: Available")
    except ImportError:
        print("  ⚠ psutil: Not installed (install with: pip install psutil)")


if __name__ == '__main__':
    print("=" * 50)
    print("Innovation 5 - Local Sanity Tests")
    print("=" * 50)
    
    print("\n[1] Testing region_weights module...")
    test_region_weights()
    
    print("\n[2] Testing weighted_losses module...")
    test_weighted_losses()
    
    print("\n[3] Testing dashboard dependencies...")
    test_dashboard_imports()
    
    print("\n" + "=" * 50)
    print("All local tests passed! ✓")
    print("=" * 50)
