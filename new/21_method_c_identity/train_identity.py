#!/usr/bin/env python3
"""
Method C: Identity-Preserving ControlNet Training
==================================================
Adds a contrastive identity-preserving loss to the ControlNet training,
inspired by IP-LDM (Huang et al., arXiv 2025).

Key idea: During training, we add a regularization term that encourages
the predicted noise for same-subject pairs to be consistent, while
the main MSE loss on noise prediction remains the primary objective.

Specifically:
  L_total = L_mse(noise_pred, noise) + λ_id * L_identity

where L_identity ensures latent features from the same subject
at different timepoints remain close in feature space.

This can work WITH or WITHOUT the auxiliary model volumes.
We test two variants:
  (a) With original 8-dim context (volumes from GT)
  (b) With time-aware 8-dim context (no volumes)

Usage on server:
  conda activate fwz
  cd /home/wangchong/data/fwz/code/brlp_src
  python -m scripts.method_c_identity_preserving \
    --dataset_csv /home/wangchong/data/fwz/brlp-data/dataset.csv \
    --cache_dir /home/wangchong/data/fwz/brlp-data/cache_identity \
    --output_dir /home/wangchong/data/fwz/output/method_c_identity/controlnet \
    --aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth \
    --diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth \
    --n_epochs 5 --batch_size 4 --lr 2.5e-5 --lambda_id 0.1 \
    --context_mode time_aware
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


def concat_covariates_original(_dict):
    """Original 8-dim context with brain volumes."""
    conditions = [
        _dict['followup_age'],
        _dict['sex'],
        _dict['followup_diagnosis'],
        _dict['followup_cerebral_cortex'],
        _dict['followup_hippocampus'],
        _dict['followup_amygdala'],
        _dict['followup_cerebral_white_matter'],
        _dict['followup_lateral_ventricle']
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    return _dict


def concat_covariates_time_aware(_dict):
    """Time-aware 8-dim context without brain volumes."""
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


def identity_preserving_loss(starting_z_batch, noise_pred_batch):
    """
    Identity-preserving regularization loss.
    
    Idea: The starting latent (z_baseline) encodes the subject's identity.
    The noise prediction should be consistent with this identity —
    i.e., noise predictions for the same spatial structure (starting_z)
    should produce similar denoised outputs.
    
    We enforce this by making the noise prediction smooth relative to
    the starting condition: the predicted noise should not drastically
    alter the identity-bearing features of the starting latent.
    
    L_identity = MSE(noise_pred * mask_identity, 0)
    where mask_identity emphasizes regions where starting_z has strong features.
    """
    # Create identity mask: regions where the brain structure is prominent
    # (high absolute values in starting_z indicate structural content)
    with torch.no_grad():
        identity_mask = (torch.abs(starting_z_batch) > torch.abs(starting_z_batch).mean()).float()
        # Normalize
        identity_mask = identity_mask / (identity_mask.sum() + 1e-8) * identity_mask.numel()

    # The noise prediction in identity-strong regions should be smaller
    # This encourages the model to preserve identity features
    identity_loss = F.mse_loss(noise_pred_batch * identity_mask, 
                                torch.zeros_like(noise_pred_batch) * identity_mask)
    
    return identity_loss


def latent_consistency_loss(starting_z_batch, followup_z_batch, noise_pred_batch, 
                            noised_images, timesteps, scheduler):
    """
    Latent consistency loss: encourages the denoised prediction to be
    structurally consistent with the starting latent.
    
    From the noise prediction, we can estimate the clean followup latent,
    then measure its similarity to the starting latent.
    The starting and followup latents of the same subject should share
    structural features (same brain, just aged).
    """
    # Estimate x_0 from noise prediction using DDPM formula
    # x_0 = (x_t - sqrt(1-alpha_bar) * noise_pred) / sqrt(alpha_bar)
    with torch.no_grad():
        alpha_bar = scheduler.alphas_cumprod.to(noised_images.device)
        alpha_t = alpha_bar[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

    # Estimated clean followup from noise prediction
    estimated_clean = (noised_images - sqrt_one_minus_alpha_t * noise_pred_batch) / (sqrt_alpha_t + 1e-8)

    # Structural consistency: the estimated followup should correlate with starting latent
    # We use a normalized cross-correlation style loss
    s_flat = starting_z_batch.flatten(start_dim=1)
    e_flat = estimated_clean.flatten(start_dim=1)

    s_norm = s_flat - s_flat.mean(dim=1, keepdim=True)
    e_norm = e_flat - e_flat.mean(dim=1, keepdim=True)

    # Cosine similarity (higher = more similar structure)
    cos_sim = F.cosine_similarity(s_norm, e_norm, dim=1).mean()
    
    # We want high similarity → minimize negative cosine similarity
    return 1.0 - cos_sim


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
    parser.add_argument('--lambda_id',   default=0.1,     type=float,
                        help='Weight for identity preserving loss')
    parser.add_argument('--lambda_con',  default=0.05,    type=float,
                        help='Weight for latent consistency loss')
    parser.add_argument('--context_mode', default='time_aware', type=str,
                        choices=['original', 'time_aware'],
                        help='Which context vector to use')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    LOG_FILE = os.path.join(args.output_dir, "train_identity.log")
    
    log(f"[METHOD-C] Identity-Preserving ControlNet Training")
    log(f"[METHOD-C] Context mode: {args.context_mode}")
    log(f"[METHOD-C] lambda_id={args.lambda_id}, lambda_con={args.lambda_con}")
    log(f"[METHOD-C] Device: {DEVICE}")

    # Choose context function
    if args.context_mode == 'original':
        context_fn = concat_covariates_original
    else:
        context_fn = concat_covariates_time_aware

    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=context_fn),
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
        log('[METHOD-C] Resuming from checkpoint...')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        log('[METHOD-C] Copying weights from diffusion model')
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
    log(f"[METHOD-C] Scale factor: {scale_factor:.4f}")

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
            epoch_id = 0.
            epoch_con = 0.
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

                        # Primary loss: noise prediction MSE
                        loss_mse = F.mse_loss(noise_pred.float(), noise.float())

                        # Identity-preserving loss
                        loss_id = identity_preserving_loss(starting_z, noise_pred)

                        # Latent consistency loss
                        loss_con = latent_consistency_loss(
                            starting_z, followup_z, noise_pred,
                            images_noised, timesteps, scheduler
                        )

                        # Total loss
                        loss = loss_mse + args.lambda_id * loss_id + args.lambda_con * loss_con

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                epoch_mse += loss_mse.item()
                epoch_id += loss_id.item()
                epoch_con += loss_con.item()
                epoch_total += loss.item()
                progress_bar.set_postfix({
                    "mse": epoch_mse / (step + 1),
                    "id": epoch_id / (step + 1),
                    "con": epoch_con / (step + 1),
                })

            n_steps = len(loader)
            log(f"[METHOD-C] Epoch {epoch} [{mode}] "
                f"total={epoch_total/n_steps:.6f} "
                f"mse={epoch_mse/n_steps:.6f} "
                f"id={epoch_id/n_steps:.6f} "
                f"con={epoch_con/n_steps:.6f}")

            if mode == 'valid':
                val_loss = epoch_total / n_steps
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    savepath = os.path.join(args.output_dir, f'cnet-identity-best.pth')
                    torch.save(controlnet.state_dict(), savepath)
                    log(f"[METHOD-C] Saved best model (val_loss={best_val_loss:.6f})")

        if epoch >= 1:
            savepath = os.path.join(args.output_dir, f'cnet-identity-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            log(f"[METHOD-C] Saved epoch {epoch} checkpoint")

    log("[METHOD-C] Training complete!")
