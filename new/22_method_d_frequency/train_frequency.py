#!/usr/bin/env python3
"""
Method D: Multi-Scale Frequency Loss ControlNet Training
=========================================================
Adds a frequency-domain loss to ControlNet training to improve
structural fidelity of generated brain MRI.

Inspired by:
  - Forecasting Future Anatomies (Ravi, 2025): Multi-scale structural loss
  - AD-DAE (Das, CMIG 2025): Feature-level consistency
  
Key idea: Standard MSE on noise predictions treats all spatial frequencies equally.
Brain atrophy patterns involve both low-frequency (overall volume) and
high-frequency (cortical folding) changes. A frequency-aware loss weights
these appropriately.

Loss: L_total = L_mse + λ_freq * L_freq + λ_smooth * L_smooth

where L_freq penalizes frequency-domain differences between predicted and
actual noise, and L_smooth encourages smooth temporal transitions.

Uses time-aware context (no auxiliary model needed).

Usage on server:
  conda activate fwz
  cd /home/wangchong/data/fwz/code/brlp_src
  python -m scripts.method_d_frequency_loss \
    --dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv \
    --cache_dir /home/wangchong/data/fwz/brlp-data/cache_freq \
    --output_dir /home/wangchong/data/fwz/output/method_d_freq/controlnet \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --n_epochs 5 --batch_size 4 --lr 2.5e-5
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from datetime import datetime

from brlp import const, utils, networks
from brlp import get_dataset_from_pd

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_FILE = None


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def concat_covariates_time_aware(_dict):
    """Time-aware context (no brain volumes)."""
    followup_age = float(_dict['followup_age'])
    starting_age = float(_dict['starting_age'])
    sex = float(_dict['sex'])
    followup_diag = float(_dict.get('followup_diagnosis', _dict.get('starting_diagnosis', 0.5)))
    starting_diag = float(_dict.get('starting_diagnosis', followup_diag))

    time_delta = followup_age - starting_age
    age_ratio = followup_age / max(starting_age, 50.0)
    norm_time_delta = time_delta / 10.0
    diag_change = followup_diag - starting_diag

    conditions = [
        followup_age, sex, followup_diag,
        time_delta, age_ratio, starting_age, diag_change, norm_time_delta,
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    return _dict


def frequency_loss(pred, target):
    """
    Frequency-domain loss using 3D FFT.
    Compares the magnitude spectrum of predicted vs target noise.
    This emphasizes structural patterns at different scales.
    """
    # 3D FFT on the spatial dimensions
    pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
    target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
    
    # Magnitude spectrum
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    # L1 loss on log-magnitude (log-scale better for frequency analysis)
    eps = 1e-8
    loss = F.l1_loss(torch.log(pred_mag + eps), torch.log(target_mag + eps))
    
    return loss


def gradient_smoothness_loss(noise_pred, starting_z):
    """
    Smoothness loss: the noise prediction should be spatially smooth
    in regions where the starting scan has consistent structure.
    
    This prevents the model from introducing artificial high-frequency
    artifacts in brain regions.
    """
    # Compute spatial gradients of noise prediction (3D)
    dx = noise_pred[:, :, 1:, :, :] - noise_pred[:, :, :-1, :, :]
    dy = noise_pred[:, :, :, 1:, :] - noise_pred[:, :, :, :-1, :]
    dz = noise_pred[:, :, :, :, 1:] - noise_pred[:, :, :, :, :-1]
    
    # Weight by inverse gradient of starting_z (smooth where brain is uniform)
    with torch.no_grad():
        sx = torch.abs(starting_z[:, :, 1:, :, :] - starting_z[:, :, :-1, :, :])
        sy = torch.abs(starting_z[:, :, :, 1:, :] - starting_z[:, :, :, :-1, :])
        sz = torch.abs(starting_z[:, :, :, :, 1:] - starting_z[:, :, :, :, :-1])
        
        # Inverse weighting: penalize noise gradients more where brain is smooth
        wx = torch.exp(-sx * 5.0)
        wy = torch.exp(-sy * 5.0)
        wz = torch.exp(-sz * 5.0)
    
    smooth_loss = (
        (dx ** 2 * wx).mean() + 
        (dy ** 2 * wy).mean() + 
        (dz ** 2 * wz).mean()
    ) / 3.0
    
    return smooth_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True,   type=str)
    parser.add_argument('--cache_dir',   required=True,   type=str)
    parser.add_argument('--output_dir',  required=True,   type=str)
    parser.add_argument('--aekl_ckpt',   required=True,   type=str)
    parser.add_argument('--diff_ckpt',   required=True,   type=str)
    parser.add_argument('--cnet_ckpt',   default=None,    type=str)
    parser.add_argument('--num_workers', default=8,       type=int)
    parser.add_argument('--n_epochs',    default=5,       type=int)
    parser.add_argument('--batch_size',  default=4,       type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    parser.add_argument('--lambda_freq',   default=0.01,  type=float,
                        help='Weight for frequency loss')
    parser.add_argument('--lambda_smooth', default=0.005, type=float,
                        help='Weight for smoothness loss')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    LOG_FILE = os.path.join(args.output_dir, "train_freq.log")

    log(f"[METHOD-D] Frequency Loss ControlNet Training")
    log(f"[METHOD-D] lambda_freq={args.lambda_freq}, lambda_smooth={args.lambda_smooth}")
    log(f"[METHOD-D] Device: {DEVICE}")

    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates_time_aware),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=validset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              persistent_workers=True,
                              pin_memory=True)

    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion   = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet  = networks.init_controlnet()

    if args.cnet_ckpt is not None:
        log('[METHOD-D] Resuming from checkpoint...')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        log('[METHOD-D] Copying weights from diffusion model')
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    for p in diffusion.parameters():
        p.requires_grad = False

    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]['followup_latent']

    scale_factor = 1 / torch.std(z)
    log(f"[METHOD-D] Scale factor: {scale_factor:.4f}")

    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              schedule='scaled_linear_beta',
                              beta_start=0.0015,
                              beta_end=0.0205)

    best_val_loss = float('inf')

    for epoch in range(args.n_epochs):
        for mode in ['train', 'valid']:
            loader = train_loader if mode == 'train' else valid_loader
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_mse = 0.
            epoch_freq = 0.
            epoch_smooth = 0.
            epoch_total = 0.
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch} [{mode}]")

            for step, batch in progress_bar:
                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(mode == 'train'):
                    starting_z = batch['starting_latent'].to(DEVICE) * scale_factor
                    followup_z = batch['followup_latent'].to(DEVICE) * scale_factor
                    context    = batch['context'].to(DEVICE)
                    starting_a = batch['starting_age'].to(DEVICE)

                    n = starting_z.shape[0]

                    with autocast(enabled=True):
                        concatenating_age    = starting_a.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
                        controlnet_condition = torch.cat([starting_z, concatenating_age], dim=1)

                        noise = torch.randn_like(followup_z).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
                        images_noised = scheduler.add_noise(followup_z, noise=noise, timesteps=timesteps)

                        down_h, mid_h = controlnet(
                            x=images_noised.float(),
                            timesteps=timesteps,
                            context=context.float(),
                            controlnet_cond=controlnet_condition.float()
                        )

                        noise_pred = diffusion(
                            x=images_noised.float(),
                            timesteps=timesteps,
                            context=context.float(),
                            down_block_additional_residuals=down_h,
                            mid_block_additional_residual=mid_h
                        )

                        # Primary loss
                        loss_mse = F.mse_loss(noise_pred.float(), noise.float())

                        # Frequency loss
                        loss_freq = frequency_loss(noise_pred.float(), noise.float())

                        # Gradient smoothness loss
                        loss_smooth = gradient_smoothness_loss(noise_pred.float(), starting_z.float())

                        # Total
                        loss = loss_mse + args.lambda_freq * loss_freq + args.lambda_smooth * loss_smooth

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                epoch_mse += loss_mse.item()
                epoch_freq += loss_freq.item()
                epoch_smooth += loss_smooth.item()
                epoch_total += loss.item()
                progress_bar.set_postfix({
                    "mse": epoch_mse / (step + 1),
                    "freq": epoch_freq / (step + 1),
                })

            n_steps = len(loader)
            log(f"[METHOD-D] Epoch {epoch} [{mode}] "
                f"total={epoch_total/n_steps:.6f} "
                f"mse={epoch_mse/n_steps:.6f} "
                f"freq={epoch_freq/n_steps:.6f} "
                f"smooth={epoch_smooth/n_steps:.6f}")

            if mode == 'valid':
                val_loss = epoch_mse / n_steps  # track MSE for best model selection
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    savepath = os.path.join(args.output_dir, f'cnet-freq-best.pth')
                    torch.save(controlnet.state_dict(), savepath)
                    log(f"[METHOD-D] Saved best model (val_mse={best_val_loss:.6f})")

        if epoch >= 1:
            savepath = os.path.join(args.output_dir, f'cnet-freq-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            log(f"[METHOD-D] Saved epoch {epoch} checkpoint")

    log("[METHOD-D] Training complete!")
