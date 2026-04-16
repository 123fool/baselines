"""
Section 37: Improved Refinement Network Training
=================================================
Key improvements over S36:
1. 20 epochs (vs 5) with validation set monitoring
2. Early stopping based on validation loss (patience=5)
3. Proper train/valid split from CSV 'split' column (371 train, 44 valid)
4. More noise augmentation options (simulate diffusion domain gap)
5. Save best model based on validation loss, not just every epoch
6. Validation includes region-weighted metrics
7. Logging of train_loss, val_loss, val_region_loss per epoch

Usage:
    python train_refinement_v2.py \
        --csv /path/to/B_mci.csv \
        --ae_ckpt /path/to/autoencoder.pth \
        --output_dir /path/to/output \
        --exp_name RefC_v2 \
        --loss_type l1_ssim_region \
        --noise_aug 0.5 \
        --epochs 20 --lr 1e-4 --gpu 0
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

REGION_LABELS = {
    'hippocampus': [17, 53],
    'amygdala': [18, 54],
    'thalamus': [10, 49],
    'lateral_ventricle': [4, 43],
}
TARGET_SHAPE = const.INPUT_SHAPE_AE


# ═══════════════════════════════════════════════
# Network Architecture (same as S36)
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
    def __init__(self, in_ch=2, out_ch=1, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc0 = nn.Sequential(nn.Conv3d(in_ch, c, 3, padding=1), ResBlock3D(c))
        self.down0 = nn.Conv3d(c, c*2, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(ResBlock3D(c*2))
        self.down1 = nn.Conv3d(c*2, c*4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResBlock3D(c*4), ResBlock3D(c*4))
        self.up1 = nn.ConvTranspose3d(c*4, c*2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(c*4, c*2, 3, padding=1), ResBlock3D(c*2))
        self.up0 = nn.ConvTranspose3d(c*2, c, 2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv3d(c*2, c, 3, padding=1), ResBlock3D(c))
        self.out_conv = nn.Conv3d(c, out_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        pred_ad = x[:, 0:1]
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        m = self.mid(self.down1(e1))
        d1 = self.dec1(torch.cat([self.up1(m), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return pred_ad + self.out_conv(d0)


# ═══════════════════════════════════════════════
# SSIM Loss
# ═══════════════════════════════════════════════

class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=7, sigma=1.5):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
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
    def __init__(self, csv_path, split='train'):
        df = pd.read_csv(csv_path)
        if 'split' in df.columns:
            self.df = df[df['split'] == split].reset_index(drop=True)
        else:
            # Fallback: 80/10/10
            n = len(df)
            if split == 'train':
                self.df = df.iloc[:int(n*0.8)].reset_index(drop=True)
            elif split == 'valid':
                self.df = df.iloc[int(n*0.8):int(n*0.9)].reset_index(drop=True)
            else:
                self.df = df.iloc[int(n*0.9):].reset_index(drop=True)

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
        if 'followup_segm' in row and pd.notna(row['followup_segm']):
            seg_path = row['followup_segm']
            if os.path.exists(seg_path):
                seg = load_and_crop(seg_path)
                result['seg'] = torch.tensor(seg, dtype=torch.float32)
        return result


def build_region_mask(seg_tensor, device):
    mask = torch.zeros_like(seg_tensor, dtype=torch.float32)
    for labels in REGION_LABELS.values():
        for lbl in labels:
            mask += (seg_tensor == lbl).float()
    return (mask > 0).float().unsqueeze(0).unsqueeze(0).to(device)


# ═══════════════════════════════════════════════
# Training with Validation
# ═══════════════════════════════════════════════

def evaluate_on_split(model, autoencoder, loader, device, l1_fn, ssim_fn, args):
    """Compute average loss on a data split (val or test)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            bl = batch['bl'].to(device)
            fu = batch['fu'].to(device)
            with autocast(enabled=True):
                z, _ = autoencoder.encode(fu)
                recon_fu = autoencoder.decode_stage_2_outputs(z)
            x_input = torch.cat([recon_fu, bl], dim=1)
            with autocast(enabled=True):
                refined = model(x_input)
                loss = l1_fn(refined, fu)
                if 'ssim' in args.loss_type:
                    loss = loss + args.ssim_weight * ssim_fn(refined, fu)
                if 'region' in args.loss_type and 'seg' in batch:
                    mask = build_region_mask(batch['seg'][0], device)
                    loss = loss + args.region_alpha * l1_fn(refined * mask, fu * mask)
            losses.append(loss.item())
    model.train()
    return np.mean(losses) if losses else float('inf')


def train(args):
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load AE (frozen)
    autoencoder = init_autoencoder(args.ae_ckpt)
    if args.ae_decoder_ckpt:
        autoencoder.load_state_dict(torch.load(args.ae_decoder_ckpt, map_location='cpu'))
    for p in autoencoder.parameters():
        p.requires_grad = False
    autoencoder.to(device).eval()

    # Refinement network
    model = RefinementUNet3D(in_ch=2, out_ch=1, base_ch=args.base_ch).to(device)
    
    # Optionally resume from S36 checkpoint
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        model.load_state_dict(torch.load(args.resume_ckpt, map_location='cpu'))
        print(f'Resumed from: {args.resume_ckpt}')
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'RefinementUNet3D: {n_params:,} params (base_ch={args.base_ch})')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    l1_fn = nn.L1Loss()
    ssim_fn = SSIM3DLoss().to(device)

    # Datasets: train + valid
    train_dataset = RefinementDataset(args.csv, split='train')
    valid_dataset = RefinementDataset(args.csv, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(f'Train: {len(train_dataset)}, Valid: {len(valid_dataset)}')

    # Training log
    log_path = os.path.join(exp_dir, 'training_log.json')
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'epoch_time': []}

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            bl = batch['bl'].to(device)
            fu = batch['fu'].to(device)

            with torch.no_grad():
                with autocast(enabled=True):
                    z, _ = autoencoder.encode(fu)
                    recon_fu = autoencoder.decode_stage_2_outputs(z)

            # Noise augmentation
            if args.noise_aug > 0 and torch.rand(1).item() < args.noise_aug:
                noise_std = torch.rand(1).item() * 0.02
                recon_fu = recon_fu + torch.randn_like(recon_fu) * noise_std

            x_input = torch.cat([recon_fu, bl], dim=1)

            with autocast(enabled=True):
                refined = model(x_input)
                loss = l1_fn(refined, fu)
                if 'region' in args.loss_type and 'seg' in batch:
                    mask = build_region_mask(batch['seg'][0], device)
                    loss = loss + args.region_alpha * l1_fn(refined * mask, fu * mask)
                if 'ssim' in args.loss_type:
                    if 'seg' in batch and args.ssim_masked:
                        mask = build_region_mask(batch['seg'][0], device)
                        loss = loss + args.ssim_weight * ssim_fn(refined, fu, mask=mask)
                    else:
                        loss = loss + args.ssim_weight * ssim_fn(refined, fu)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 50 == 0:
                avg = np.mean(epoch_losses[-50:])
                print(f'  Ep {epoch} [{batch_idx+1}/{len(train_loader)}] loss={avg:.4f}')

        scheduler.step()
        train_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0

        # Validation
        val_loss = evaluate_on_split(model, autoencoder, valid_loader, device, l1_fn, ssim_fn, args)

        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
              f'time={elapsed:.0f}s, lr={lr_now:.2e}')

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['lr'].append(float(lr_now))
        history['epoch_time'].append(float(elapsed))

        # Save log every epoch
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(exp_dir, f'refnet-{args.exp_name}-ep{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  Saved: {ckpt_path}')

        # Best model tracking + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(exp_dir, f'refnet-{args.exp_name}-best.pth')
            torch.save(model.state_dict(), best_path)
            print(f'  New best (val_loss={val_loss:.4f}) → {best_path}')
        else:
            patience_counter += 1
            print(f'  No improvement ({patience_counter}/{args.patience})')

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Save final
    final_path = os.path.join(exp_dir, f'refnet-{args.exp_name}-final.pth')
    torch.save(model.state_dict(), final_path)
    print(f'\nTraining complete: {args.exp_name}')
    print(f'  Best val_loss: {best_val_loss:.4f}')
    print(f'  Best model: {best_path}')
    print(f'  Final model: {final_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--ae_ckpt', required=True)
    parser.add_argument('--ae_decoder_ckpt', default=None)
    parser.add_argument('--resume_ckpt', default=None, help='Resume from S36 checkpoint')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--loss_type', default='l1_ssim_region',
                        choices=['l1', 'l1_region', 'l1_ssim', 'l1_ssim_region'])
    parser.add_argument('--region_alpha', default=10.0, type=float)
    parser.add_argument('--ssim_weight', default=1.0, type=float)
    parser.add_argument('--ssim_masked', default=False, action='store_true')
    parser.add_argument('--noise_aug', default=0.0, type=float)
    parser.add_argument('--base_ch', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=5, type=int, help='Early stopping patience')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    train(args)
