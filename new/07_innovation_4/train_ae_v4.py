"""
Innovation 4 v4: Improved decoder-only fine-tuning with SSIM loss + loss warmup.

Key improvements over v3:
  1. Explicit SSIM reconstruction loss (directly targets evaluation metric)
  2. Loss warmup: first 3 epochs use only original losses, then gradually add 3D+freq
  3. Reduced freq_weight (0.005 → 0.001) to prevent MAE regression
  4. L1 weight boosted (1.0 → 1.5) for stronger pixel accuracy
  5. Cosine LR schedule for better convergence
  6. 10 epochs (vs 5 in v3)
  7. Latent noise augmentation for robustness to diffusion-generated latents
"""

import os
import sys
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai import transforms
from monai.losses import PatchAdversarialLoss
from generative.losses import PerceptualLoss

# Fix PyTorch 2.6+ weights_only=True default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from brlp import const, utils
from brlp.data import get_dataset_from_pd
from brlp.networks import init_autoencoder, init_patch_discriminator
from brlp.gradacc import GradientAccumulation
from brlp.losses import KLDivergenceLoss
from medicalnet_perceptual_v2 import MedicalNet3DPerceptualLoss
from frequency_losses import LaplacianPyramidLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3D SSIM Loss
# ============================================================
class SSIMLoss3D(torch.nn.Module):
    """
    3D Structural Similarity Index loss.
    Returns 1 - SSIM (so minimizing the loss maximizes SSIM).

    Uses a sliding window approach with Gaussian-weighted patches.
    """

    def __init__(self, window_size=7, sigma=1.5, channel=1):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, sigma, channel))

    @staticmethod
    def _gaussian_kernel_1d(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        return g / g.sum()

    @staticmethod
    def _create_window(size, sigma, channel):
        g1d = SSIMLoss3D._gaussian_kernel_1d(size, sigma)
        g3d = g1d.unsqueeze(-1).unsqueeze(-1) * g1d.unsqueeze(0).unsqueeze(-1) * g1d.unsqueeze(0).unsqueeze(0)
        window = g3d.unsqueeze(0).unsqueeze(0).expand(channel, 1, size, size, size).contiguous()
        return window

    def forward(self, prediction, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ch = prediction.size(1)

        if self.window.shape[0] != ch:
            window = self._create_window(self.window_size, self.sigma, ch).to(prediction.device)
        else:
            window = self.window.to(prediction.device)

        pad = self.window_size // 2
        mu_pred = F.conv3d(prediction, window, padding=pad, groups=ch)
        mu_tgt = F.conv3d(target, window, padding=pad, groups=ch)

        mu_pred_sq = mu_pred ** 2
        mu_tgt_sq = mu_tgt ** 2
        mu_cross = mu_pred * mu_tgt

        sigma_pred_sq = F.conv3d(prediction ** 2, window, padding=pad, groups=ch) - mu_pred_sq
        sigma_tgt_sq = F.conv3d(target ** 2, window, padding=pad, groups=ch) - mu_tgt_sq
        sigma_cross = F.conv3d(prediction * target, window, padding=pad, groups=ch) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_pred_sq + mu_tgt_sq + C1) * (sigma_pred_sq + sigma_tgt_sq + C2))

        return 1.0 - ssim_map.mean()


# ============================================================
# Loss warmup scheduler
# ============================================================
def get_warmup_weight(epoch, warmup_start, warmup_end, max_weight):
    """
    Linear warmup for auxiliary losses.
    Returns 0 if epoch < warmup_start, linearly increases to max_weight
    from warmup_start to warmup_end, then stays at max_weight.
    """
    if epoch < warmup_start:
        return 0.0
    elif epoch < warmup_end:
        progress = (epoch - warmup_start) / max(warmup_end - warmup_start, 1)
        return max_weight * progress
    else:
        return max_weight


def validate_epoch(autoencoder, val_loader, l1_loss_fn, ssim_loss_fn):
    autoencoder.eval()
    total_l1 = 0.0
    total_ssim = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            with autocast(enabled=True):
                reconstruction, _, _ = autoencoder(images)
                l1 = l1_loss_fn(reconstruction.float(), images.float())
                ssim_l = ssim_loss_fn(reconstruction.float(), images.float())
            total_l1 += l1.item()
            total_ssim += (1.0 - ssim_l.item())  # actual SSIM value
            n_batches += 1
    autoencoder.train()
    return total_l1 / max(n_batches, 1), total_ssim / max(n_batches, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Innovation 4 v4: Improved decoder-only fine-tuning')
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=4,     type=int)
    parser.add_argument('--n_epochs',       default=10,    type=int)
    parser.add_argument('--max_batch_size', default=1,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=2e-5,  type=float)
    # Innovation 4 v4 args
    parser.add_argument('--mednet_ckpt',    required=True, type=str)
    parser.add_argument('--perc3d_weight',  default=0.0005, type=float,
                        help='3D MedicalNet perceptual weight (max after warmup)')
    parser.add_argument('--freq_weight',    default=0.001,  type=float,
                        help='Laplacian pyramid frequency loss weight (reduced from v3)')
    parser.add_argument('--ssim_weight',    default=0.5,    type=float,
                        help='SSIM reconstruction loss weight')
    parser.add_argument('--l1_weight',      default=1.5,    type=float,
                        help='L1 reconstruction loss weight (boosted from v3)')
    parser.add_argument('--lap_levels',     default=3,      type=int)
    parser.add_argument('--warmup_start',   default=3,      type=int,
                        help='Epoch to start adding auxiliary losses')
    parser.add_argument('--warmup_end',     default=6,      type=int,
                        help='Epoch at which auxiliary losses reach full weight')
    parser.add_argument('--latent_noise_std', default=0.01, type=float,
                        help='Std of Gaussian noise added to encoder output (0 to disable)')
    parser.add_argument('--latent_noise_prob', default=0.5, type=float,
                        help='Probability of adding latent noise per batch')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data loading ----
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image'}, names=['image_orig']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE,
                                        mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid'] if 'valid' in dataset_df.split.values else None

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    train_loader = DataLoader(
        dataset=trainset, num_workers=args.num_workers,
        batch_size=args.max_batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True
    )

    val_loader = None
    if valid_df is not None and len(valid_df) > 0:
        valset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)
        val_loader = DataLoader(
            dataset=valset, num_workers=2,
            batch_size=args.max_batch_size, shuffle=False,
            persistent_workers=False, pin_memory=True
        )

    # ---- Models ----
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    # ========================================
    # v4: Freeze encoder to preserve latent space (same as v3)
    # ========================================
    frozen_count = 0
    trainable_count = 0
    for name, param in autoencoder.named_parameters():
        if (name.startswith('encoder')
            or name.startswith('quant_conv_mu')
            or name.startswith('quant_conv_log_sigma')):
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()

    total_params = frozen_count + trainable_count
    print(f"[v4] Encoder FROZEN: {frozen_count:,} params")
    print(f"[v4] Decoder trainable: {trainable_count:,} params")
    print(f"[v4] Total: {total_params:,} ({100*trainable_count/total_params:.1f}% trainable)")

    # ---- Loss weights ----
    adv_weight = 0.025
    orig_perc_weight = 0.001
    kl_weight = 1e-7
    l1_weight = args.l1_weight
    ssim_weight = args.ssim_weight
    perc3d_max_weight = args.perc3d_weight
    freq_max_weight = args.freq_weight

    # ---- Loss functions ----
    l1_loss_fn = torch.nn.L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    ssim_loss_fn = SSIMLoss3D(window_size=7, sigma=1.5, channel=1).to(DEVICE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig_perc_fn = PerceptualLoss(
            spatial_dims=3, network_type="squeeze",
            is_fake_3d=True, fake_3d_ratio=0.2
        ).to(DEVICE)

    print(f"[v4] Loading MedicalNet ResNet-10 from {args.mednet_ckpt}")
    perc3d_fn = MedicalNet3DPerceptualLoss(
        pretrained_path=args.mednet_ckpt,
        downsample_size=(80, 96, 80),
    ).to(DEVICE)

    freq_fn = LaplacianPyramidLoss(num_levels=args.lap_levels).to(DEVICE)

    # ---- Optimizers (only decoder params) ----
    decoder_params = [p for p in autoencoder.parameters() if p.requires_grad]
    optimizer_g = torch.optim.Adam(decoder_params, lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # v4: Cosine LR schedule
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=args.n_epochs, eta_min=args.lr * 0.01)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, T_max=args.n_epochs, eta_min=args.lr * 0.01)

    gradacc_g = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g, grad_scaler=GradScaler()
    )
    gradacc_d = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d, grad_scaler=GradScaler()
    )

    avgloss = utils.AverageLoss()
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    total_counter = 0
    best_val_loss = float('inf')

    print(f"[Innovation 4 v4] Improved decoder-only fine-tuning")
    print(f"  L1 weight:                       {l1_weight}")
    print(f"  SSIM weight:                     {ssim_weight}")
    print(f"  Original PerceptualLoss weight:  {orig_perc_weight}")
    print(f"  3D MedicalNet weight (max):      {perc3d_max_weight}")
    print(f"  Lap frequency weight (max):      {freq_max_weight}")
    print(f"  Warmup: epochs {args.warmup_start}-{args.warmup_end}")
    print(f"  Latent noise: std={args.latent_noise_std}, prob={args.latent_noise_prob}")
    print(f"  Learning rate: {args.lr} (cosine schedule)")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Train samples: {len(train_df)}")
    if valid_df is not None:
        print(f"  Valid samples: {len(valid_df)}")

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        autoencoder.train()

        # v4: Compute warmup weights for this epoch
        perc3d_w = get_warmup_weight(epoch, args.warmup_start, args.warmup_end, perc3d_max_weight)
        freq_w = get_warmup_weight(epoch, args.warmup_start, args.warmup_end, freq_max_weight)
        print(f"\n  [Epoch {epoch}] perc3d_w={perc3d_w:.6f}, freq_w={freq_w:.6f}, lr={optimizer_g.param_groups[0]['lr']:.2e}")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            with autocast(enabled=True):
                images = batch["image"].to(DEVICE)

                # v4: Latent noise augmentation
                # Simulates the noisy latent distribution from diffusion sampling
                if args.latent_noise_std > 0 and torch.rand(1).item() < args.latent_noise_prob:
                    with torch.no_grad():
                        z_mu, z_sigma = autoencoder.encode(images)
                        z = autoencoder.sampling(z_mu, z_sigma)
                        noise = torch.randn_like(z) * args.latent_noise_std
                        z_noisy = z + noise
                    reconstruction = autoencoder.decode(z_noisy)
                    # Still need z_mu, z_sigma for KL loss
                else:
                    reconstruction, z_mu, z_sigma = autoencoder(images)

                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # === Original BrLP losses ===
                rec_loss = l1_weight * l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss_orig = orig_perc_weight * orig_perc_fn(
                    reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(
                    logits_fake, target_is_real=True, for_discriminator=False)

                # v4: SSIM reconstruction loss (always active)
                ssim_loss = ssim_weight * ssim_loss_fn(
                    reconstruction.float(), images.float())

                # === Innovation 4 v4: Auxiliary losses with warmup ===
                if perc3d_w > 0:
                    per_loss_3d = perc3d_w * perc3d_fn(
                        reconstruction.float(), images.float())
                else:
                    per_loss_3d = torch.tensor(0.0, device=DEVICE)

                if freq_w > 0:
                    frq_loss = freq_w * freq_fn(
                        reconstruction.float(), images.float())
                else:
                    frq_loss = torch.tensor(0.0, device=DEVICE)

                loss_g = (rec_loss + kld_loss + per_loss_orig
                          + ssim_loss + per_loss_3d + frq_loss + gen_loss)

            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False,
                                          for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True,
                                          for_discriminator=True)
                loss_d = adv_weight * (d_loss_fake + d_loss_real) * 0.5

            gradacc_d.step(loss_d, step)

            # Logging
            avgloss.put('G/rec',           rec_loss.item())
            avgloss.put('G/ssim',          ssim_loss.item())
            avgloss.put('G/perc_orig',     per_loss_orig.item())
            avgloss.put('G/perc_3d',       per_loss_3d.item() if torch.is_tensor(per_loss_3d) else per_loss_3d)
            avgloss.put('G/freq',          frq_loss.item() if torch.is_tensor(frq_loss) else frq_loss)
            avgloss.put('G/adv',           gen_loss.item())
            avgloss.put('G/kl',            kld_loss.item())
            avgloss.put('D/loss',          loss_d.item())

            if total_counter % 10 == 0:
                tb_step = total_counter // 10
                avgloss.to_tensorboard(writer, tb_step)
                utils.tb_display_reconstruction(
                    writer, tb_step,
                    images[0].detach().cpu(),
                    reconstruction[0].detach().cpu()
                )
            total_counter += 1

        # v4: Step LR schedulers
        scheduler_g.step()
        scheduler_d.step()

        # ---- Validation ----
        val_l1_str = "N/A"
        val_ssim_str = "N/A"
        if val_loader is not None:
            val_l1, val_ssim = validate_epoch(autoencoder, val_loader, l1_loss_fn, ssim_loss_fn)
            val_l1_str = f"{val_l1:.6f}"
            val_ssim_str = f"{val_ssim:.4f}"
            writer.add_scalar('val/rec_loss', val_l1, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)
            # Use combined metric for best model: lower L1 + higher SSIM
            # We want to minimize (L1 - ssim_bonus), so val_combined is lower = better
            val_combined = val_l1 - 0.1 * val_ssim
            if val_combined < best_val_loss:
                best_val_loss = val_combined
                torch.save(autoencoder.state_dict(),
                           os.path.join(args.output_dir, 'autoencoder-best.pth'))
                torch.save(discriminator.state_dict(),
                           os.path.join(args.output_dir, 'discriminator-best.pth'))
                print(f"  [Epoch {epoch}] New best! val_l1={val_l1_str}, val_ssim={val_ssim_str}, saved best checkpoint")

        torch.save(autoencoder.state_dict(),
                   os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
        torch.save(discriminator.state_dict(),
                   os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        print(f"  [Epoch {epoch}] val_l1={val_l1_str}, val_ssim={val_ssim_str}, saved checkpoint")

    print(f"\n[Innovation 4 v4] Training complete.")
    print(f"  Best val_combined={best_val_loss:.6f}")
    print(f"  Checkpoints in {args.output_dir}")
