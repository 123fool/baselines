#!/usr/bin/env python3
"""Debug script: check image shapes and spacing."""
import sys
sys.path.insert(0, '/home/wangchong/data/fwz/code/brlp_src/src')
import nibabel as nib
import numpy as np
import torch
from monai import transforms

# Load a real image
img_path = '/home/wangchong/data/fwz/data/mci_longitudinal/005_S_0572/2006-06-20/t1w_final.nii.gz'
img = nib.load(img_path)
print(f"Shape: {img.shape}")
print(f"Affine:\n{img.affine}")
print(f"Pixdim: {img.header.get_zooms()}")
print(f"Data range: [{img.get_fdata().min():.4f}, {img.get_fdata().max():.4f}]")

# Test Spacing transform
gt_np = img.get_fdata()
gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).float()
resample_fn = transforms.Spacing(pixdim=1.5)
gt_resampled = resample_fn(gt_tensor).squeeze(0).numpy()
print(f"\nOriginal shape: {gt_np.shape}")
print(f"After Spacing(1.5) shape: {gt_resampled.shape}")

# Now test what the model output looks like
from brlp import utils
print(f"\nMNI 1.5mm trick target dim: (122, 146, 122)")
