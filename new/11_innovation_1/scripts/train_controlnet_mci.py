"""
Innovation 1: MCI Dynamic Conditioning - ControlNet Training.

Extends ControlNet spatial conditioning from 4 to 6 channels:
  ch 0-2: starting latent
  ch  3:  starting age
  ch  4:  hippocampal atrophy rate   (NEW)
  ch  5:  ventricular expansion rate (NEW)

Changes vs original train_controlnet.py:
  1. Uses init_controlnet_mci (conditioning_embedding_in_channels=6)
  2. Computes atrophy/expansion rates from CSV columns
  3. Builds 6-channel controlnet_condition using build_controlnet_condition()
  4. Everything else (UNet, cross_attention, scheduler) unchanged
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime

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

# Fix PyTorch 2.6+ weights_only=True default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
INNOV_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, INNOV_SRC)

from brlp import const, utils, networks
from brlp import get_dataset_from_pd, sample_using_controlnet_and_z
from mci_conditioning import (
    init_controlnet_mci,
    init_controlnet_mci_from_pretrained,
    build_controlnet_condition,
)

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """
    Build cross-attention context (8-dim, unchanged from baseline).
    Also pass through the new rate columns for controlnet conditioning.
    """
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

    # Innovation 1: pass rate columns through as scalars
    _dict['hippocampal_atrophy_rate'] = float(_dict.get('hippocampal_atrophy_rate', 0.0))
    _dict['ventricular_expansion_rate'] = float(_dict.get('ventricular_expansion_rate', 0.0))

    return _dict


def images_to_tensorboard(
    writer, epoch, mode,
    autoencoder, diffusion, controlnet,
    dataset, scale_factor
):
    """Visualize generation on tensorboard (uses baseline sampling for viz)."""
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice(range(len(dataset)), min(3, len(dataset)))

    for tag_i, i in enumerate(random_indices):
        starting_z = dataset[i]['starting_latent'] * scale_factor
        context = dataset[i]['context'].squeeze(0)
        starting_a = dataset[i]['starting_age']

        starting_image = torch.from_numpy(
            nib.load(dataset[i]['starting_image']).get_fdata()
        ).unsqueeze(0)
        followup_image = torch.from_numpy(
            nib.load(dataset[i]['followup_image']).get_fdata()
        ).unsqueeze(0)
        starting_image = resample_fn(starting_image).squeeze(0)
        followup_image = resample_fn(followup_image).squeeze(0)

        # For TensorBoard viz, use the innovation-1 sampling with rates
        atrophy_rate = torch.tensor([dataset[i]['hippocampal_atrophy_rate']])
        vent_rate = torch.tensor([dataset[i]['ventricular_expansion_rate']])

        predicted_image = sample_using_controlnet_mci(
            autoencoder=autoencoder,
            diffusion=diffusion,
            controlnet=controlnet,
            starting_z=starting_z,
            starting_a=starting_a,
            context=context,
            atrophy_rate=atrophy_rate.item(),
            ventricular_rate=vent_rate.item(),
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_cond_generation(
            writer=writer,
            step=epoch,
            tag=f'{mode}/comparison_{tag_i}',
            starting_image=starting_image,
            followup_image=followup_image,
            predicted_image=predicted_image
        )


@torch.no_grad()
def sample_using_controlnet_mci(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context,
    atrophy_rate, ventricular_rate,
    device, scale_factor=1,
    average_over_n=1,
    num_training_steps=1000, num_inference_steps=50,
    schedule='scaled_linear_beta',
    beta_start=0.0015, beta_end=0.0205,
    verbose=True
):
    """
    Innovation 1 inference: like sample_using_controlnet_and_z but with
    6-channel controlnet_condition (latent + age + atrophy_rate + vent_rate).
    """
    from generative.networks.schedulers import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Prepare 6-channel controlnet condition
    starting_z_t = starting_z.unsqueeze(0).to(device)
    age_t = torch.tensor([starting_a]).float().to(device)
    atrophy_t = torch.tensor([atrophy_rate]).float().to(device)
    vent_t = torch.tensor([ventricular_rate]).float().to(device)

    controlnet_condition = build_controlnet_condition(
        starting_z_t, age_t, atrophy_t, vent_t
    )

    context_t = context.unsqueeze(0).unsqueeze(0).to(device)

    if average_over_n > 1:
        context_t = context_t.repeat(average_over_n, 1, 1)
        controlnet_condition = controlnet_condition.repeat(average_over_n, 1, 1, 1, 1)

    z = torch.randn(average_over_n, *starting_z_t.shape[1:]).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with autocast(enabled=True):
            timestep = torch.tensor([t]).repeat(average_over_n).to(device)

            down_h, mid_h = controlnet(
                x=z.float(),
                timesteps=timestep,
                context=context_t,
                controlnet_cond=controlnet_condition.float()
            )

            noise_pred = diffusion(
                x=z.float(),
                timesteps=timestep,
                context=context_t.float(),
                down_block_additional_residuals=down_h,
                mid_block_additional_residual=mid_h
            )

            z, _ = scheduler.step(noise_pred, t, z)

    z = (z / scale_factor).sum(axis=0) / average_over_n
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Innovation 1: ControlNet with MCI dynamic conditioning')
    parser.add_argument('--dataset_csv', required=True, type=str,
                        help='Path to B_mci_inn1.csv (with atrophy/expansion rate columns)')
    parser.add_argument('--cache_dir',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str, help='AutoEncoder checkpoint')
    parser.add_argument('--diff_ckpt',   required=True, type=str, help='UNet diffusion checkpoint')
    parser.add_argument('--cnet_ckpt',   default=None,  type=str, help='Resume from Innovation 1 ControlNet ckpt')
    parser.add_argument('--pretrained_cnet', default=None, type=str, help='Pretrained 4ch ControlNet to initialize from')
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=5,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=2.5e-5, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)

    # Verify Innovation 1 columns exist
    for col in ['hippocampal_atrophy_rate', 'ventricular_expansion_rate']:
        if col not in dataset_df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Run prepare_mci_conditions.py first.")

    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    print(f"[Innovation 1] Training: {len(train_df)} pairs | Validation: {len(valid_df)} pairs")

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        dataset=trainset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True
    )
    valid_loader = DataLoader(
        dataset=validset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True
    )

    # ---- Models ----
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt)

    # Innovation 1: Use modified ControlNet with 6 conditioning channels
    if args.cnet_ckpt is not None:
        print('Resuming from Innovation 1 checkpoint...')
        controlnet = init_controlnet_mci(args.cnet_ckpt)
    elif args.pretrained_cnet is not None:
        print('Initializing 6ch ControlNet from pretrained 4ch ControlNet...')
        controlnet = init_controlnet_mci_from_pretrained(args.pretrained_cnet)
    else:
        # Fallback: init from UNet (not recommended - conditioning_embedding is random)
        print('WARNING: No pretrained ControlNet provided. Initializing from UNet (slow convergence).')
        controlnet = init_controlnet_mci()
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
    print(f"Scaling factor set to {scale_factor}")

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    global_counter = {'train': 0, 'valid': 0}
    loaders = {'train': train_loader, 'valid': valid_loader}
    datasets = {'train': trainset, 'valid': validset}

    print(f"[Innovation 1] MCI Dynamic Conditioning - 6ch ControlNet")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {args.n_epochs} | BS: {args.batch_size} | LR: {args.lr}")

    # ---- Training loop ----
    for epoch in range(args.n_epochs):

        for mode in loaders.keys():
            print(f'mode: {mode}')
            loader = loaders[mode]
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss = 0.
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:

                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(mode == 'train'):

                    starting_z = batch['starting_latent'].to(DEVICE) * scale_factor
                    followup_z = batch['followup_latent'].to(DEVICE) * scale_factor
                    context = batch['context'].to(DEVICE)
                    starting_a = batch['starting_age'].to(DEVICE)

                    # Innovation 1: get rate columns from batch
                    atrophy_rate = batch['hippocampal_atrophy_rate'].float().to(DEVICE)
                    vent_rate = batch['ventricular_expansion_rate'].float().to(DEVICE)

                    n = starting_z.shape[0]

                    with autocast(enabled=True):
                        # Innovation 1: 6-channel conditioning
                        controlnet_condition = build_controlnet_condition(
                            starting_z, starting_a, atrophy_rate, vent_rate
                        )

                        noise = torch.randn_like(followup_z).to(DEVICE)
                        timesteps = torch.randint(
                            0, scheduler.num_train_timesteps, (n,), device=DEVICE
                        ).long()
                        images_noised = scheduler.add_noise(
                            followup_z, noise=noise, timesteps=timesteps
                        )

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

                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            images_to_tensorboard(
                writer=writer, epoch=epoch, mode=mode,
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet, dataset=datasets[mode],
                scale_factor=scale_factor
            )

        # Save checkpoints from epoch 1 onwards
        if epoch >= 1:
            savepath = os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            print(f"  Checkpoint saved: {savepath}")

    print("[Innovation 1] ControlNet training complete.")
