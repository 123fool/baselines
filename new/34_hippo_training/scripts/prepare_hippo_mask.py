"""
Pre-compute static hippocampus mask in latent space.

Creates a mask by averaging hippocampus segmentations from training data,
downsampling to latent dimensions, and applying DivisiblePad alignment.

Usage:
    python prepare_hippo_mask.py \
        --csv /path/to/B_mci.csv \
        --output /path/to/hippo_latent_mask.npy \
        --n_samples 50
"""
import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

HIPPO_LABELS = [17, 53]  # SynthSeg: 17=L-Hippocampus, 53=R-Hippocampus
INPUT_SHAPE = (120, 144, 120)
AE_DOWNSAMPLE = 8
LATENT_PADDED = (16, 20, 16)


def seg_to_latent_mask(seg_path):
    """Load segmentation, extract hippocampus, downsample to latent space."""
    seg_data = nib.load(seg_path).get_fdata()

    hippo = np.isin(seg_data, HIPPO_LABELS).astype(np.float32)

    # Ensure correct shape (should be 120x144x120 at 1.5mm)
    if hippo.shape != INPUT_SHAPE:
        hippo_t = torch.from_numpy(hippo).float().unsqueeze(0).unsqueeze(0)
        hippo_t = F.interpolate(hippo_t, size=INPUT_SHAPE, mode='nearest')
        hippo = hippo_t.squeeze().numpy()

    # Average pool to latent dims (divide by 8): (120,144,120) -> (15,18,15)
    hippo_t = torch.from_numpy(hippo).float().unsqueeze(0).unsqueeze(0)
    hippo_latent = F.avg_pool3d(hippo_t, kernel_size=AE_DOWNSAMPLE)  # (1,1,15,18,15)

    # DivisiblePad(k=4) alignment: 15->16, 18->20, 15->16
    # MONAI pads: dim 15->(0,1), dim 18->(1,1), dim 15->(0,1)
    # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
    hippo_latent = F.pad(hippo_latent, (0, 1, 1, 1, 0, 1), mode='constant', value=0)

    return hippo_latent.squeeze().numpy()  # (16, 20, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--n_samples', default=50, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    train_df = df[df.split == 'train']

    n = min(args.n_samples, len(train_df))
    sample_df = train_df.sample(n, random_state=42)

    masks = []
    for _, row in tqdm(sample_df.iterrows(), total=n, desc='Computing masks'):
        seg_path = str(row['followup_segm'])
        if not os.path.exists(seg_path):
            print(f'  Skipping (not found): {seg_path}')
            continue
        mask = seg_to_latent_mask(seg_path)
        masks.append(mask)

    avg_mask = np.mean(masks, axis=0)  # (16, 20, 16), soft [0,1]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, avg_mask)

    total = np.prod(LATENT_PADDED)
    print(f'Averaged {len(masks)} masks')
    print(f'Shape: {avg_mask.shape}')
    print(f'Max: {avg_mask.max():.4f}, Mean: {avg_mask.mean():.6f}')
    print(f'Voxels > 0.01: {(avg_mask > 0.01).sum()} ({100*(avg_mask > 0.01).sum()/total:.1f}%)')
    print(f'Voxels > 0.05: {(avg_mask > 0.05).sum()} ({100*(avg_mask > 0.05).sum()/total:.1f}%)')
    print(f'Saved to: {args.output}')


if __name__ == '__main__':
    main()
