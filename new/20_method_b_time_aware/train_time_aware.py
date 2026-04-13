#!/usr/bin/env python3
"""
Method B: Time-Aware Context ControlNet Training
=================================================
Replaces the 5 auxiliary-model brain volumes in the 8-dim context vector with
temporal features derived directly from available data:

Original 8-dim:  [followup_age, sex, diagnosis, ctx_vol, hipp_vol, amyg_vol, wm_vol, vent_vol]
New 8-dim:       [followup_age, sex, diagnosis, time_delta, age_ratio, baseline_age, diag_change_flag, norm_time_delta]

This completely eliminates the need for the Leaspy auxiliary model at inference time.
The ControlNet architecture and UNet remain unchanged (cross_attention_dim=8).

Borrowed idea from:
  - TADM (Litrico, MICCAI 2024): Uses age difference and cognitive state as primary temporal conditions
  - AD-DAE (Das, CMIG 2025): Uses progression attribute (delta_age + cognitive_change)

Usage on server:
  conda activate fwz
  cd /home/wangchong/data/fwz/code/brlp_src
  python -m scripts.method_b_time_aware_context \
    --dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv \
    --cache_dir /home/wangchong/data/fwz/brlp-data/cache_time_aware \
    --output_dir /home/wangchong/data/fwz/output/method_b_time_aware/controlnet \
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
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from datetime import datetime

from brlp import const, utils, networks
from brlp import get_dataset_from_pd, sample_using_controlnet_and_z

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
    """
    New context vector that uses temporal features instead of brain volumes.
    
    Original: [followup_age, sex, diagnosis, ctx, hipp, amyg, wm, vent]
                ↑ requires auxiliary model at inference
    
    New: [followup_age, sex, diagnosis, time_delta, age_ratio, baseline_age, diag_change, norm_time]
         ↑ all computable from metadata alone, no auxiliary model needed
    """
    followup_age = float(_dict['followup_age'])
    starting_age = float(_dict['starting_age'])
    sex = float(_dict['sex'])
    
    # Diagnosis handling
    followup_diag = float(_dict.get('followup_diagnosis', _dict.get('starting_diagnosis', 0.5)))
    starting_diag = float(_dict.get('starting_diagnosis', followup_diag))
    
    # Temporal features
    time_delta = followup_age - starting_age  # years between visits
    age_ratio = followup_age / max(starting_age, 50.0)  # normalized age progression
    norm_time_delta = time_delta / 10.0  # normalized to ~[0, 1] for typical AD study range
    
    # Diagnosis change flag: 0=stable, 0.5=progressed by one step, 1.0=progressed by two
    diag_change = followup_diag - starting_diag
    
    conditions = [
        followup_age,       # target age (same as original)
        sex,                # sex (same as original)
        followup_diag,      # diagnosis at followup (same as original)
        time_delta,         # NEW: years between baseline and followup
        age_ratio,          # NEW: followup_age / starting_age
        starting_age,       # NEW: baseline age (helps model understand starting point)
        diag_change,        # NEW: diagnosis progression indicator
        norm_time_delta,    # NEW: normalized time gap
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    return _dict


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

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    LOG_FILE = os.path.join(args.output_dir, "train_time_aware.log")
    
    log("[METHOD-B] Time-Aware Context ControlNet Training")
    log(f"[METHOD-B] Device: {DEVICE}")
    log(f"[METHOD-B] Output: {args.output_dir}")

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

    # Initialize models — same architecture, just different context
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion   = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet  = networks.init_controlnet()

    if args.cnet_ckpt is not None:
        log('[METHOD-B] Resuming from checkpoint...')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        log('[METHOD-B] Copying weights from diffusion model')
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    # Freeze UNet
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
    log(f"[METHOD-B] Scale factor: {scale_factor:.4f}")

    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              schedule='scaled_linear_beta',
                              beta_start=0.0015,
                              beta_end=0.0205)

    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.n_epochs):
        for mode in ['train', 'valid']:
            loader = train_loader if mode == 'train' else valid_loader
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss = 0.
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

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            epoch_loss = epoch_loss / len(loader)
            log(f"[METHOD-B] Epoch {epoch} [{mode}] loss={epoch_loss:.6f}")

            if mode == 'valid' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                savepath = os.path.join(args.output_dir, f'cnet-time-aware-best.pth')
                torch.save(controlnet.state_dict(), savepath)
                log(f"[METHOD-B] Saved best model (val_loss={best_val_loss:.6f})")

        # Save every epoch after epoch 1
        if epoch >= 1:
            savepath = os.path.join(args.output_dir, f'cnet-time-aware-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            log(f"[METHOD-B] Saved epoch {epoch} checkpoint")

    log("[METHOD-B] Training complete!")
