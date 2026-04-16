"""
Fine-tune ControlNet with hippocampus-weighted noise prediction loss.

Methods:
  H1: Weighted noise MSE in latent space (spatial hippocampus emphasis)
  H2: H1 + biased timestep sampling toward low timesteps (fine details)

Usage:
    python train_controlnet_hippo.py \
        --dataset_csv B_mci.csv \
        --cache_dir /path/to/cache \
        --output_dir /path/to/output \
        --aekl_ckpt ae.pth \
        --diff_ckpt diffusion.pth \
        --cnet_ckpt cnet-btr.pth \
        --hippo_mask hippo_latent_mask.npy \
        --method H1 --alpha 30 --n_epochs 3 --lr 1e-5
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Patch torch.load for PyTorch 2.6 + MONAI cache compatibility
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load
import nibabel as nib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
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

from brlp import const, utils, networks
from brlp import get_dataset_from_pd

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
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


def weighted_mse_loss(pred, target, weight_map):
    """Weighted MSE: higher weight on hippocampus latent voxels."""
    se = (pred.float() - target.float()) ** 2
    return (weight_map * se).mean()


def sample_timesteps_biased(n, num_steps, low_ratio=0.5, low_max=200, device='cuda'):
    """Bias 50% of timesteps toward [0, low_max)."""
    n_low = int(n * low_ratio)
    n_high = n - n_low
    t_low = torch.randint(0, low_max, (n_low,), device=device)
    t_high = torch.randint(low_max, num_steps, (n_high,), device=device)
    timesteps = torch.cat([t_low, t_high])
    return timesteps[torch.randperm(n, device=device)].long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--cache_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str,
                        help='BTR ControlNet checkpoint to fine-tune from')
    parser.add_argument('--hippo_mask', required=True, type=str,
                        help='Pre-computed hippocampus latent mask (.npy)')
    parser.add_argument('--method', default='H1', choices=['H1', 'H2'],
                        help='H1=weighted noise, H2=weighted+timestep bias')
    parser.add_argument('--alpha', default=30.0, type=float,
                        help='Hippocampus weight multiplier')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    args = parser.parse_args()

    tag = f'{args.method}_a{int(args.alpha)}'
    print(f'=== Training: {tag} ===')
    print(f'Method={args.method}, Alpha={args.alpha}, LR={args.lr}, Epochs={args.n_epochs}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load hippocampus mask and build weight map
    # ------------------------------------------------------------------
    hippo_np = np.load(args.hippo_mask)  # (16, 20, 16) soft values
    hippo_mask = torch.from_numpy(hippo_np).float().to(DEVICE)
    hippo_mask = hippo_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 20, 16)

    # weight = 1 + alpha * mask, then normalize so mean(weight) = 1
    weight_map = 1.0 + args.alpha * hippo_mask
    weight_map = weight_map / weight_map.mean()
    print(f'Weight map: min={weight_map.min():.2f}, max={weight_map.max():.2f}')
    print(f'Hippo mask nonzero: {(hippo_np > 0.01).sum()} / {hippo_np.size}')

    # ------------------------------------------------------------------
    # Data pipeline (identical to original train_controlnet.py)
    # ------------------------------------------------------------------
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'],
                              reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'],
                                       channel_dim=0),
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'],
                                 k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(trainset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True,
                              persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(validset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True,
                              persistent_workers=True, pin_memory=True)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet = networks.init_controlnet()

    print('Loading BTR checkpoint for fine-tuning...')
    controlnet.load_state_dict(torch.load(args.cnet_ckpt, map_location='cpu'))

    for p in diffusion.parameters():
        p.requires_grad = False

    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    # Scale factor
    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]['followup_latent']
    scale_factor = 1 / torch.std(z)
    print(f'Scale factor: {scale_factor:.4f}')

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    loaders = {'train': train_loader, 'valid': valid_loader}
    global_step = {'train': 0, 'valid': 0}

    for epoch in range(args.n_epochs):
        for mode in ['train', 'valid']:
            loader = loaders[mode]
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss = 0.0
            pbar = tqdm(enumerate(loader), total=len(loader),
                        desc=f'Ep{epoch} [{mode}]')

            for step, batch in pbar:
                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(mode == 'train'):
                    starting_z = batch['starting_latent'].to(DEVICE) * scale_factor
                    followup_z = batch['followup_latent'].to(DEVICE) * scale_factor
                    context = batch['context'].to(DEVICE)
                    starting_a = batch['starting_age'].to(DEVICE)
                    n = starting_z.shape[0]

                    with autocast(enabled=True):
                        cat_age = starting_a.view(n, 1, 1, 1, 1).expand(
                            n, 1, *starting_z.shape[-3:])
                        cnet_cond = torch.cat([starting_z, cat_age], dim=1)

                        noise = torch.randn_like(followup_z).to(DEVICE)

                        if args.method == 'H2':
                            timesteps = sample_timesteps_biased(
                                n, scheduler.num_train_timesteps, device=DEVICE)
                        else:
                            timesteps = torch.randint(
                                0, scheduler.num_train_timesteps,
                                (n,), device=DEVICE).long()

                        noised = scheduler.add_noise(
                            followup_z, noise=noise, timesteps=timesteps)

                        down_h, mid_h = controlnet(
                            x=noised.float(), timesteps=timesteps,
                            context=context.float(),
                            controlnet_cond=cnet_cond.float())

                        noise_pred = diffusion(
                            x=noised.float(), timesteps=timesteps,
                            context=context.float(),
                            down_block_additional_residuals=down_h,
                            mid_block_additional_residual=mid_h)

                        # === Hippocampus-weighted loss ===
                        loss = weighted_mse_loss(noise_pred, noise, weight_map)

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': epoch_loss / (step + 1)})
                global_step[mode] += 1

            epoch_loss /= len(loader)
            print(f'  Epoch {epoch} [{mode}] loss: {epoch_loss:.6f}')

        # Save every epoch
        ckpt_path = os.path.join(args.output_dir,
                                 f'cnet-hippo-{tag}-ep{epoch}.pth')
        torch.save(controlnet.state_dict(), ckpt_path)
        print(f'  Saved: {ckpt_path}')

    print(f'\n=== Training {tag} complete ===')
