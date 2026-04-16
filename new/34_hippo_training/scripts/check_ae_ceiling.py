"""
Check AutoencoderKL reconstruction ceiling for hippocampus.

Replicates exact preprocessing from extract_latents.py:
  LoadImage -> Spacing(1.5mm) -> ResizeWithPadOrCrop(120,144,120) -> ScaleIntensity[0,1]
Then encode -> decode and measure H-SSIM.

Usage:
    python check_ae_ceiling.py \
        --dataset_csv /path/to/B_mci.csv \
        --aekl_ckpt /path/to/autoencoder.pth \
        --n_test 10
"""
import os
import argparse
import numpy as np
import nibabel as nib
import torch
from torch.cuda.amp import autocast
from monai import transforms
from skimage.metrics import structural_similarity as compute_ssim

import sys
sys.path.insert(0, '/home/wangchong/data/fwz/code')

from brlp import networks, const

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIPPO_LABELS = [17, 53]


def compute_hippo_ssim(img1, img2, seg_data):
    """Compute SSIM within hippocampus region bounding box."""
    hippo_mask = np.isin(seg_data, HIPPO_LABELS)
    if hippo_mask.sum() == 0:
        return float('nan')

    coords = np.where(hippo_mask)
    margin = 5
    slices = []
    for dim in range(3):
        lo = max(0, coords[dim].min() - margin)
        hi = min(hippo_mask.shape[dim], coords[dim].max() + margin + 1)
        slices.append(slice(lo, hi))

    roi1 = img1[slices[0], slices[1], slices[2]]
    roi2 = img2[slices[0], slices[1], slices[2]]
    mask_roi = hippo_mask[slices[0], slices[1], slices[2]]

    data_range = max(roi1.max(), roi2.max()) - min(roi1.min(), roi2.min())
    if data_range < 1e-8:
        return 1.0

    ssim_map = compute_ssim(roi1, roi2, data_range=data_range, full=True)[1]
    return float(ssim_map[mask_roi].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--n_test', default=10, type=int)
    args = parser.parse_args()

    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    autoencoder.to(DEVICE)
    autoencoder.eval()

    import pandas as pd
    df = pd.read_csv(args.dataset_csv)
    test_df = df[df.split == 'test'].head(args.n_test)

    # Exact same preprocessing as extract_latents.py
    preprocess = transforms.Compose([
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    results = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        img_path = row['followup_image']
        seg_path = row['followup_segm']

        # Preprocess image exactly like latent extraction
        processed = preprocess({'image': img_path})
        img_tensor = processed['image']  # (1, 120, 144, 120), [0,1]
        img_np = img_tensor.squeeze().cpu().numpy()

        seg_data = nib.load(seg_path).get_fdata()

        # Encode (get mu) then decode
        with torch.no_grad():
            with autocast(enabled=True):
                x = img_tensor.unsqueeze(0).to(DEVICE)  # (1,1,120,144,120)
                z, _ = autoencoder.encode(x)  # z = mu
                recon = autoencoder.decode(z)
                recon_np = recon.squeeze().cpu().float().numpy()

        # Trim if shapes differ slightly
        s = img_np.shape
        if recon_np.shape != s:
            recon_np = recon_np[:s[0], :s[1], :s[2]]

        h_ssim = compute_hippo_ssim(img_np, recon_np, seg_data)
        overall = compute_ssim(img_np, recon_np, data_range=1.0)

        results.append({'h_ssim': h_ssim, 'overall': overall})
        print(f'Pair {idx}: H-SSIM={h_ssim:.4f}, Overall={overall:.4f}')

    avg_h = np.mean([r['h_ssim'] for r in results])
    avg_o = np.mean([r['overall'] for r in results])
    std_h = np.std([r['h_ssim'] for r in results])
    print(f'\n=== AE Reconstruction Ceiling ===')
    print(f'H-SSIM: {avg_h:.4f} +/- {std_h:.4f}')
    print(f'Overall SSIM: {avg_o:.4f}')
    print(f'This is the UPPER BOUND for diffusion-based hippocampus quality.')
    if avg_h < 0.90:
        print(f'WARNING: AE ceiling ({avg_h:.4f}) < 0.9 -> AE fine-tuning may be needed!')
    else:
        print(f'AE ceiling ({avg_h:.4f}) >= 0.9 -> ControlNet training can reach target.')


if __name__ == '__main__':
    main()
