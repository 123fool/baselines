"""
AE Decoder Fine-tuning V2 — SSIM Loss & Multi-Region Support.

Enhancements over V1 (Section 34):
  1. Differentiable 3D SSIM loss (directly optimize structural similarity)
  2. Multi-region mask support (hippocampus + amygdala + thalamus + ventricle)
  3. Combined loss modes: l1, ssim, l1_ssim, freq_l1
  4. Configurable region selection

Usage:
    # SSIM loss + hippocampus only
    python train_ae_decoder_v2.py --loss_type ssim --regions hippo ...

    # L1 loss + multi-region
    python train_ae_decoder_v2.py --loss_type l1 --regions multi ...

    # Combined SSIM+L1 + multi-region
    python train_ae_decoder_v2.py --loss_type l1_ssim --regions multi ...
"""
import os, sys, argparse, warnings
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import L1Loss

# Patch torch.load for PyTorch 2.6 + MONAI cache
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

sys.path.insert(0, '/home/wangchong/data/fwz/code')

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

# SynthSeg label definitions for AD-relevant regions
REGION_LABELS = {
    'hippo':     [17, 53],          # hippocampus
    'amygdala':  [18, 54],          # amygdala
    'thalamus':  [10, 49],          # thalamus
    'ventricle': [4, 43],           # lateral ventricles
    'caudate':   [11, 50],          # caudate
    'putamen':   [12, 51],          # putamen
}

REGION_PRESETS = {
    'hippo': ['hippo'],
    'multi': ['hippo', 'amygdala', 'thalamus', 'ventricle'],
    'all_subcort': ['hippo', 'amygdala', 'thalamus', 'ventricle', 'caudate', 'putamen'],
}


# ──────────────────────────────────────────────────────────────────────
# Differentiable 3D SSIM Loss
# ──────────────────────────────────────────────────────────────────────
def _fspecial_gauss_3d(size, sigma):
    """Create 3D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return kernel


class SSIM3DLoss(torch.nn.Module):
    """Differentiable 3D Structural Similarity Loss.

    Returns 1 - SSIM(x, y) so that minimizing this maximizes SSIM.
    Supports optional spatial weighting mask.
    """
    def __init__(self, window_size=7, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        kernel = _fspecial_gauss_3d(window_size, sigma)
        # Shape: (1, 1, W, W, W) for conv3d
        self.register_buffer('window', kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x, y, weight_map=None):
        """
        x, y: (B, 1, D, H, W) in [0, 1]
        weight_map: (1, 1, D, H, W) optional spatial weighting
        Returns: scalar loss (1 - weighted_mean_SSIM)
        """
        pad = self.window.shape[-1] // 2
        mu_x = F.conv3d(x, self.window, padding=pad)
        mu_y = F.conv3d(y, self.window, padding=pad)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv3d(x * x, self.window, padding=pad) - mu_x_sq
        sigma_y_sq = F.conv3d(y * y, self.window, padding=pad) - mu_y_sq
        sigma_xy = F.conv3d(x * y, self.window, padding=pad) - mu_xy

        # Clamp for numerical stability
        sigma_x_sq = torch.clamp(sigma_x_sq, min=0)
        sigma_y_sq = torch.clamp(sigma_y_sq, min=0)

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))

        if weight_map is not None:
            return 1.0 - (weight_map * ssim_map).sum() / weight_map.sum()
        else:
            return 1.0 - ssim_map.mean()


# ──────────────────────────────────────────────────────────────────────
# Frequency Domain Loss (emphasize high-frequency details)
# ──────────────────────────────────────────────────────────────────────
class FrequencyLoss(torch.nn.Module):
    """L1 loss in frequency domain, weighted toward high frequencies."""
    def __init__(self, high_freq_weight=2.0):
        super().__init__()
        self.high_freq_weight = high_freq_weight

    def forward(self, x, y):
        # 3D FFT
        fx = torch.fft.fftn(x, dim=(-3, -2, -1))
        fy = torch.fft.fftn(y, dim=(-3, -2, -1))

        # Magnitude difference
        diff = torch.abs(fx - fy)

        # Create frequency weighting (higher weight for high frequencies)
        D, H, W = x.shape[-3:]
        freq_d = torch.fft.fftfreq(D, device=x.device).reshape(-1, 1, 1)
        freq_h = torch.fft.fftfreq(H, device=x.device).reshape(1, -1, 1)
        freq_w = torch.fft.fftfreq(W, device=x.device).reshape(1, 1, -1)
        freq_mag = torch.sqrt(freq_d**2 + freq_h**2 + freq_w**2)

        # Weight: 1 for low freq, high_freq_weight for high freq
        weight = 1.0 + (self.high_freq_weight - 1.0) * freq_mag / (freq_mag.max() + 1e-8)

        return (weight * diff).mean()


# ──────────────────────────────────────────────────────────────────────
# Mask computation
# ──────────────────────────────────────────────────────────────────────
def compute_region_mask(csv_path, region_labels_list, n_samples=50):
    """Compute averaged region mask in image space (120,144,120).

    Args:
        region_labels_list: flat list of all SynthSeg label IDs to include
    """
    df = pd.read_csv(csv_path)
    train_df = df[df.split == 'train'].sample(
        min(n_samples, len(df[df.split=='train'])), random_state=42)

    masks = []
    for _, row in train_df.iterrows():
        seg_path = str(row.get('followup_segm', ''))
        if not os.path.exists(seg_path):
            continue
        seg_data = nib.load(seg_path).get_fdata()
        region_mask = np.isin(seg_data, region_labels_list).astype(np.float32)
        if region_mask.shape == tuple(const.INPUT_SHAPE_AE):
            masks.append(region_mask)

    avg_mask = np.mean(masks, axis=0) if masks else np.zeros(const.INPUT_SHAPE_AE)
    print(f'Region mask: averaged {len(masks)} samples, '
          f'max={avg_mask.max():.3f}, nonzero={(avg_mask>0.01).sum()}')
    return avg_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      required=True, type=str)
    parser.add_argument('--alpha',          default=30.0,  type=float,
                        help='Region weight multiplier')
    parser.add_argument('--regions',        default='hippo', type=str,
                        choices=['hippo', 'multi', 'all_subcort'],
                        help='Region preset for mask')
    parser.add_argument('--loss_type',      default='l1', type=str,
                        choices=['l1', 'ssim', 'l1_ssim', 'freq_l1'],
                        help='Loss function type')
    parser.add_argument('--ssim_weight',    default=1.0, type=float,
                        help='SSIM loss weight (for l1_ssim mode)')
    parser.add_argument('--freq_weight',    default=0.1, type=float,
                        help='Frequency loss weight (for freq_l1 mode)')
    parser.add_argument('--num_workers',    default=8,  type=int)
    parser.add_argument('--n_epochs',       default=3,  type=int)
    parser.add_argument('--max_batch_size', default=1,  type=int)
    parser.add_argument('--batch_size',     default=16, type=int)
    parser.add_argument('--lr',             default=5e-5, type=float)
    parser.add_argument('--exp_name',       default=None, type=str)
    args = parser.parse_args()

    exp_name = args.exp_name or f'{args.loss_type}_{args.regions}_a{int(args.alpha)}'
    print(f'=== AE Decoder V2: {exp_name} ===')
    print(f'Loss={args.loss_type}, Regions={args.regions}, Alpha={args.alpha}')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Build region mask ──
    region_preset = REGION_PRESETS[args.regions]
    all_labels = []
    for rname in region_preset:
        all_labels.extend(REGION_LABELS[rname])
    print(f'Target regions: {region_preset}, labels: {all_labels}')

    mask_np = compute_region_mask(args.dataset_csv, all_labels)
    mask_t = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    weight_map = 1.0 + args.alpha * mask_t
    weight_map = weight_map / weight_map.mean()
    print(f'Weight map: min={weight_map.min():.2f}, max={weight_map.max():.2f}')

    # For SSIM loss: also create a normalized weight for SSIM weighting
    ssim_weight_map = weight_map / weight_map.sum() * weight_map.numel()

    # ── Data ──
    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train'].copy()

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

    # ── Model ──
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)

    for name, p in autoencoder.named_parameters():
        if 'encoder' in name or 'quant_conv_mu' in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainable = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in autoencoder.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    # ── Loss functions ──
    kl_weight = 1e-7
    perceptual_weight = 0.001

    l1_loss_fn = L1Loss(reduction='none')
    kl_loss_fn = KLDivergenceLoss()

    ssim_loss_fn = None
    freq_loss_fn = None

    if args.loss_type in ('ssim', 'l1_ssim'):
        ssim_loss_fn = SSIM3DLoss(window_size=7, sigma=1.5).to(DEVICE)
        print(f'SSIM loss enabled (weight={args.ssim_weight})')

    if args.loss_type == 'freq_l1':
        freq_loss_fn = FrequencyLoss(high_freq_weight=2.0).to(DEVICE)
        print(f'Frequency loss enabled (weight={args.freq_weight})')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from generative.losses import PerceptualLoss
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3, network_type="squeeze",
            is_fake_3d=True, fake_3d_ratio=0.2).to(DEVICE)

    decoder_params = [p for p in autoencoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(decoder_params, lr=args.lr)

    gradacc = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer,
        grad_scaler=GradScaler())

    # ── Training ──
    for epoch in range(args.n_epochs):
        autoencoder.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Ep{epoch}')

        for step, batch in pbar:
            with autocast(enabled=True):
                images = batch['image'].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # ── Reconstruction loss based on loss_type ──
                if args.loss_type == 'l1':
                    l1_per_voxel = l1_loss_fn(reconstruction.float(), images.float())
                    rec_loss = (weight_map * l1_per_voxel).mean()

                elif args.loss_type == 'ssim':
                    rec_loss = ssim_loss_fn(
                        reconstruction.float(), images.float(),
                        weight_map=ssim_weight_map)

                elif args.loss_type == 'l1_ssim':
                    l1_per_voxel = l1_loss_fn(reconstruction.float(), images.float())
                    l1_loss = (weight_map * l1_per_voxel).mean()
                    ssim_loss = ssim_loss_fn(
                        reconstruction.float(), images.float(),
                        weight_map=ssim_weight_map)
                    rec_loss = l1_loss + args.ssim_weight * ssim_loss

                elif args.loss_type == 'freq_l1':
                    l1_per_voxel = l1_loss_fn(reconstruction.float(), images.float())
                    l1_loss = (weight_map * l1_per_voxel).mean()
                    freq_loss = freq_loss_fn(reconstruction.float(), images.float())
                    rec_loss = l1_loss + args.freq_weight * freq_loss

                # KL divergence
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)

                # Perceptual loss
                per_loss = perceptual_weight * perc_loss_fn(
                    reconstruction.float(), images.float())

                loss = rec_loss + kld_loss + per_loss

            gradacc.step(loss, step)
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (step + 1)})

        epoch_loss /= len(train_loader)
        print(f'  Epoch {epoch} loss: {epoch_loss:.6f}')

        ckpt_path = os.path.join(args.output_dir,
                                 f'ae-v2-{exp_name}-ep{epoch}.pth')
        torch.save(autoencoder.state_dict(), ckpt_path)
        print(f'  Saved: {ckpt_path}')

    print(f'\n=== AE Decoder V2 training complete: {exp_name} ===')
