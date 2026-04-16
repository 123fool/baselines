"""
Fine-tune AE DECODER with hippocampus-weighted reconstruction loss.

Strategy:
  - Freeze encoder + quant_conv_mu (latent representation stays fixed)
  - Train decoder + quant_conv_post with hippocampus emphasis
  - Use static hippocampus mask in image space (from averaged segmentations)
  - Combined loss: hippocampus-weighted L1 + perceptual + KL

This preserves compatibility with existing latents and ControlNet checkpoints.
Only the decoder is swapped at inference time.

Usage:
    python train_ae_decoder_hippo.py \
        --dataset_csv B_mci.csv \
        --cache_dir /path/to/cache \
        --output_dir /path/to/output \
        --aekl_ckpt autoencoder.pth \
        --alpha 30 --n_epochs 3 --lr 5e-5
"""
import os
import argparse
import warnings

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.nn import L1Loss

# Patch torch.load for PyTorch 2.6 + MONAI cache compatibility
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai import transforms
from monai.utils import set_determinism
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/wangchong/data/fwz/code')

# Fix PyTorch 2.6 weights_only=True for MONAI cached data
try:
    from monai.data.meta_tensor import MetaTensor
    from monai.utils.enums import MetaKeys, SpaceKeys
    torch.serialization.add_safe_globals([MetaTensor, MetaKeys, SpaceKeys])
except Exception:
    pass

from brlp import const, utils
from brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, get_dataset_from_pd
)

set_determinism(0)
warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIPPO_LABELS = [17, 53]


def compute_static_hippo_mask(csv_path, n_samples=50):
    """Compute averaged hippocampus mask in image space (120,144,120)."""
    df = pd.read_csv(csv_path)
    train_df = df[df.split == 'train'].sample(min(n_samples, len(df[df.split=='train'])),
                                                random_state=42)
    masks = []
    for _, row in train_df.iterrows():
        seg_path = str(row.get('followup_segm', ''))
        if not os.path.exists(seg_path):
            continue
        seg_data = nib.load(seg_path).get_fdata()
        hippo = np.isin(seg_data, HIPPO_LABELS).astype(np.float32)
        if hippo.shape == tuple(const.INPUT_SHAPE_AE):
            masks.append(hippo)

    avg_mask = np.mean(masks, axis=0) if masks else np.zeros(const.INPUT_SHAPE_AE)
    print(f'Image-space hippo mask: averaged {len(masks)} samples, '
          f'max={avg_mask.max():.3f}, nonzero={(avg_mask>0.01).sum()}')
    return avg_mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      required=True, type=str)
    parser.add_argument('--alpha',          default=30.0,  type=float,
                        help='Hippocampus weight multiplier')
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=3,     type=int)
    parser.add_argument('--max_batch_size', default=1,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=5e-5,  type=float)
    args = parser.parse_args()

    print(f'=== AE Decoder Fine-tuning ===')
    print(f'Alpha={args.alpha}, LR={args.lr}, Epochs={args.n_epochs}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Compute static hippocampus mask in image space
    # ------------------------------------------------------------------
    hippo_np = compute_static_hippo_mask(args.dataset_csv)
    hippo_mask = torch.from_numpy(hippo_np).float().to(DEVICE)
    hippo_mask = hippo_mask.unsqueeze(0).unsqueeze(0)  # (1,1,120,144,120)

    weight_map = 1.0 + args.alpha * hippo_mask
    weight_map = weight_map / weight_map.mean()
    print(f'Weight map: min={weight_map.min():.2f}, max={weight_map.max():.2f}')

    # ------------------------------------------------------------------
    # Data: use followup_image paths (same preprocessing as extract_latents)
    # ------------------------------------------------------------------
    # Create a temporary column 'image_path' from followup_image for compatibility
    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train'].copy()

    # Use both starting and followup images to double the data
    rows = []
    for _, row in train_df.iterrows():
        for col in ['starting_image', 'followup_image']:
            if col in row and pd.notna(row[col]):
                rows.append({'image_path': row[col]})
    img_df = pd.DataFrame(rows)
    img_df['split'] = 'train'
    print(f'Training images: {len(img_df)}')

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    trainset = get_dataset_from_pd(img_df, transforms_fn, args.cache_dir)
    train_loader = DataLoader(
        trainset, num_workers=args.num_workers,
        batch_size=args.max_batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True)

    # ------------------------------------------------------------------
    # Model: freeze encoder, train only decoder
    # ------------------------------------------------------------------
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)

    # Freeze encoder + quant_conv_mu
    for name, p in autoencoder.named_parameters():
        if 'encoder' in name or 'quant_conv_mu' in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainable = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in autoencoder.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    kl_weight = 1e-7
    perceptual_weight = 0.001

    l1_loss_fn = L1Loss(reduction='none')  # per-element for weighting
    kl_loss_fn = KLDivergenceLoss()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from generative.losses import PerceptualLoss
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3, network_type="squeeze",
            is_fake_3d=True, fake_3d_ratio=0.2).to(DEVICE)

    # Only optimize trainable parameters
    decoder_params = [p for p in autoencoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(decoder_params, lr=args.lr)

    gradacc = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer,
        grad_scaler=GradScaler())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(args.n_epochs):
        autoencoder.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Ep{epoch}')

        for step, batch in pbar:
            with autocast(enabled=True):
                images = batch['image'].to(DEVICE)  # (B,1,120,144,120), [0,1]

                # Forward through full AE (encoder frozen, decoder trainable)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # Hippocampus-weighted L1 reconstruction loss
                l1_per_voxel = l1_loss_fn(reconstruction.float(), images.float())
                # weight_map broadcasts: (1,1,120,144,120) * (B,1,120,144,120)
                weighted_l1 = (weight_map * l1_per_voxel).mean()

                # KL divergence (encoder is frozen, but KL still regularizes)
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)

                # Perceptual loss
                per_loss = perceptual_weight * perc_loss_fn(
                    reconstruction.float(), images.float())

                loss = weighted_l1 + kld_loss + per_loss

            gradacc.step(loss, step)
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (step + 1)})

        epoch_loss /= len(train_loader)
        print(f'  Epoch {epoch} loss: {epoch_loss:.6f}')

        ckpt_path = os.path.join(args.output_dir,
                                 f'ae-hippo-dec-a{int(args.alpha)}-ep{epoch}.pth')
        torch.save(autoencoder.state_dict(), ckpt_path)
        print(f'  Saved: {ckpt_path}')

    print('\n=== AE Decoder fine-tuning complete ===')
