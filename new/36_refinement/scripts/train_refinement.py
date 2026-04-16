"""
Section 36: Image-Space Refinement Network Training
=====================================================
Train a lightweight 3D U-Net to refine AE decoded outputs using BL as guidance.

Core idea: AE reconstruction ceiling (H-SSIM=0.8288) is the bottleneck.
The refinement network learns to restore high-frequency details lost by the AE
by leveraging the real BL image (which shares most structure with FU).

Architecture:
  Input:  concat(AE_decoded, real_BL) → (B, 2, 120, 144, 120)
  Output: refined                     → (B, 1, 120, 144, 120)
  Residual learning: output = AE_decoded + correction_network(input)

Training data (on-the-fly):
  For each (BL, FU) pair in training CSV:
    1. AE encode(FU) → z → AE decode(z) → recon_FU  (simulates AE bottleneck)
    2. Optional noise augmentation on recon_FU  (simulates diffusion errors)
    3. Input: concat(recon_FU, BL)
    4. Target: real FU

Usage:
    python train_refinement.py \
        --csv /path/to/B_mci.csv \
        --ae_ckpt /path/to/autoencoder.pth \
        --output_dir /path/to/output \
        --exp_name RefA \
        --loss_type l1_region \
        --epochs 5 --lr 1e-4 --gpu 0
"""
import os, sys, json, time, argparse, warnings
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Patch torch.load for PyTorch 2.6 + MONAI cache
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, '/home/wangchong/data/fwz/code')

try:
    from monai.data.meta_tensor import MetaTensor
    from monai.utils.enums import MetaKeys, SpaceKeys
    torch.serialization.add_safe_globals([MetaTensor, MetaKeys, SpaceKeys])
except Exception:
    pass

from brlp import const
from brlp import init_autoencoder

warnings.filterwarnings("ignore")

# ── SynthSeg label definitions ──
REGION_LABELS = {
    'hippocampus': [17, 53],
    'amygdala': [18, 54],
    'thalamus': [10, 49],
    'lateral_ventricle': [4, 43],
}

# ── Target spatial dimensions ──
TARGET_SHAPE = const.INPUT_SHAPE_AE  # (120, 144, 120)


# ═══════════════════════════════════════════════
# Network Architecture
# ═══════════════════════════════════════════════

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)


class RefinementUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for post-AE refinement with residual learning.
    Input:  (B, 2, 120, 144, 120)  — [generated_AD, real_BL]
    Output: (B, 1, 120, 144, 120)  — refined_AD = generated + correction
    """
    def __init__(self, in_ch=2, out_ch=1, base_ch=32):
        super().__init__()
        c = base_ch

        # Encoder
        self.enc0 = nn.Sequential(nn.Conv3d(in_ch, c, 3, padding=1), ResBlock3D(c))
        self.down0 = nn.Conv3d(c, c*2, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(ResBlock3D(c*2))
        self.down1 = nn.Conv3d(c*2, c*4, 3, stride=2, padding=1)

        # Bottleneck
        self.mid = nn.Sequential(ResBlock3D(c*4), ResBlock3D(c*4))

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose3d(c*4, c*2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(c*4, c*2, 3, padding=1), ResBlock3D(c*2))
        self.up0 = nn.ConvTranspose3d(c*2, c, 2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv3d(c*2, c, 3, padding=1), ResBlock3D(c))

        # Output head — initialized near zero for identity start
        self.out_conv = nn.Conv3d(c, out_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        pred_ad = x[:, 0:1]  # (B, 1, D, H, W)

        e0 = self.enc0(x)                          # (B, c, 120, 144, 120)
        e1 = self.enc1(self.down0(e0))              # (B, 2c, 60, 72, 60)
        m = self.mid(self.down1(e1))                # (B, 4c, 30, 36, 30)

        d1 = self.dec1(torch.cat([self.up1(m), e1], dim=1))   # (B, 2c, 60, 72, 60)
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))  # (B, c, 120, 144, 120)

        correction = self.out_conv(d0)              # (B, 1, 120, 144, 120)
        return pred_ad + correction


# ═══════════════════════════════════════════════
# SSIM Loss (from Section 35)
# ═══════════════════════════════════════════════

class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=7, sigma=1.5):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))  # (1,1,W,W,W)
        self.pad = window_size // 2
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _gaussian_filter(self, x):
        ch = x.shape[1]
        k = self.kernel.expand(ch, -1, -1, -1, -1).to(x.device, x.dtype)
        return F.conv3d(x, k, padding=self.pad, groups=ch)

    def forward(self, pred, target, mask=None):
        mu_p = self._gaussian_filter(pred)
        mu_t = self._gaussian_filter(target)
        sigma_pp = self._gaussian_filter(pred * pred) - mu_p * mu_p
        sigma_tt = self._gaussian_filter(target * target) - mu_t * mu_t
        sigma_pt = self._gaussian_filter(pred * target) - mu_p * mu_t

        num = (2 * mu_p * mu_t + self.C1) * (2 * sigma_pt + self.C2)
        den = (mu_p ** 2 + mu_t ** 2 + self.C1) * (sigma_pp + sigma_tt + self.C2)
        ssim_map = num / (den + 1e-8)

        if mask is not None:
            m = mask.to(ssim_map.device, ssim_map.dtype)
            if m.dim() == 3:
                m = m.unsqueeze(0).unsqueeze(0)
            return 1 - (ssim_map * m).sum() / (m.sum() + 1e-8)
        return 1 - ssim_map.mean()


# ═══════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════

def load_and_crop(path, target_shape=TARGET_SHAPE):
    img = nib.load(path).get_fdata().astype(np.float32)
    s = img.shape
    if s == target_shape:
        return img
    if s == (122, 146, 122):
        return img[1:121, 1:145, 1:121]
    starts = [(s[i] - target_shape[i]) // 2 for i in range(3)]
    return img[starts[0]:starts[0]+target_shape[0],
               starts[1]:starts[1]+target_shape[1],
               starts[2]:starts[2]+target_shape[2]]


def normalize_01(img):
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return img
    return (img - mn) / (mx - mn)


class RefinementDataset(Dataset):
    """Load BL/FU image pairs + optional segmentation for refinement training."""
    def __init__(self, csv_path, exclude_last_n=5):
        df = pd.read_csv(csv_path)
        # Use test/train split if available, otherwise exclude last N
        if 'split' in df.columns:
            self.df = df[df['split'] == 'train'].reset_index(drop=True)
        else:
            self.df = df.iloc[:-exclude_last_n].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bl = normalize_01(load_and_crop(row['starting_image']))
        fu = normalize_01(load_and_crop(row['followup_image']))

        result = {
            'bl': torch.tensor(bl, dtype=torch.float32).unsqueeze(0),
            'fu': torch.tensor(fu, dtype=torch.float32).unsqueeze(0),
        }

        # Load segmentation if available
        if 'followup_segm' in row and pd.notna(row['followup_segm']):
            seg_path = row['followup_segm']
            if os.path.exists(seg_path):
                seg = load_and_crop(seg_path)
                result['seg'] = torch.tensor(seg, dtype=torch.float32)

        return result


def build_region_mask(seg_tensor, device):
    """Build binary mask for AD-relevant regions from SynthSeg labels."""
    mask = torch.zeros_like(seg_tensor, dtype=torch.float32)
    for labels in REGION_LABELS.values():
        for lbl in labels:
            mask += (seg_tensor == lbl).float()
    return (mask > 0).float().unsqueeze(0).unsqueeze(0).to(device)


# ═══════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════

def train(args):
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load AE (frozen)
    autoencoder = init_autoencoder(args.ae_ckpt)
    if args.ae_decoder_ckpt:
        autoencoder.load_state_dict(
            torch.load(args.ae_decoder_ckpt, map_location='cpu'))
    for p in autoencoder.parameters():
        p.requires_grad = False
    autoencoder.to(device).eval()

    # Refinement network
    model = RefinementUNet3D(in_ch=2, out_ch=1, base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'RefinementUNet3D: {n_params:,} trainable params (base_ch={args.base_ch})')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # Loss functions
    l1_loss_fn = nn.L1Loss()
    ssim_loss_fn = SSIM3DLoss().to(device)

    # Dataset
    dataset = RefinementDataset(args.csv, exclude_last_n=args.n_test_hold)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    print(f'Training samples: {len(dataset)}')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            bl = batch['bl'].to(device)       # (1, 1, 120, 144, 120)
            fu = batch['fu'].to(device)       # (1, 1, 120, 144, 120)

            # Generate AE reconstruction (simulates AE bottleneck)
            with torch.no_grad():
                with autocast(enabled=True):
                    z, _ = autoencoder.encode(fu)
                    recon_fu = autoencoder.decode_stage_2_outputs(z)

            # Noise augmentation to simulate diffusion errors
            if args.noise_aug > 0 and torch.rand(1).item() < args.noise_aug:
                noise_std = torch.rand(1).item() * 0.02  # random scale up to 0.02
                recon_fu = recon_fu + torch.randn_like(recon_fu) * noise_std

            # Forward pass
            x_input = torch.cat([recon_fu, bl], dim=1)  # (1, 2, 120, 144, 120)

            with autocast(enabled=True):
                refined = model(x_input)  # (1, 1, 120, 144, 120)

                # ── Compute loss ──
                loss_l1 = l1_loss_fn(refined, fu)

                loss = loss_l1

                if 'region' in args.loss_type:
                    if 'seg' in batch:
                        mask = build_region_mask(batch['seg'][0], device)
                        loss_region = l1_loss_fn(refined * mask, fu * mask)
                    else:
                        loss_region = torch.tensor(0.0, device=device)
                    loss = loss + args.region_alpha * loss_region

                if 'ssim' in args.loss_type:
                    if 'seg' in batch and args.ssim_masked:
                        mask = build_region_mask(batch['seg'][0], device)
                        loss_ssim = ssim_loss_fn(refined, fu, mask=mask)
                    else:
                        loss_ssim = ssim_loss_fn(refined, fu)
                    loss = loss + args.ssim_weight * loss_ssim

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 20 == 0:
                avg = np.mean(epoch_losses[-20:])
                print(f'  Ep {epoch} [{batch_idx+1}/{len(loader)}] loss={avg:.4f}')

        scheduler.step()
        mean_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0
        print(f'Epoch {epoch}: loss={mean_loss:.4f}, time={elapsed:.0f}s, lr={scheduler.get_last_lr()[0]:.2e}')

        # Save checkpoint
        ckpt_path = os.path.join(exp_dir, f'refnet-{args.exp_name}-ep{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f'  Saved: {ckpt_path}')

    print(f'\nTraining complete: {args.exp_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Section 36: Refinement Network Training')
    parser.add_argument('--csv', required=True, type=str, help='B_mci.csv path')
    parser.add_argument('--ae_ckpt', required=True, type=str, help='AE checkpoint')
    parser.add_argument('--ae_decoder_ckpt', default=None, type=str, help='Fine-tuned AE decoder (optional)')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--loss_type', default='l1_ssim_region', type=str,
                        choices=['l1', 'l1_region', 'l1_ssim', 'l1_ssim_region'])
    parser.add_argument('--region_alpha', default=10.0, type=float)
    parser.add_argument('--ssim_weight', default=1.0, type=float)
    parser.add_argument('--ssim_masked', default=False, action='store_true')
    parser.add_argument('--noise_aug', default=0.0, type=float, help='Prob of noise augmentation')
    parser.add_argument('--base_ch', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n_test_hold', default=5, type=int, help='N test pairs to exclude')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    train(args)
