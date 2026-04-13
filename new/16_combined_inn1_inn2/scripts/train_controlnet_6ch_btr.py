"""
Combined Innovation 1+2: 6-Channel ControlNet + Bidirectional Temporal Regularization.

Merges:
  - Innovation 1: 6ch spatial conditioning (starting_z + age + atrophy_rate + vent_rate)
  - Innovation 2: BTR bidirectional loss (L_fwd + λ * L_bwd)

Forward (A→B):
  condition_6ch = [starting_z, starting_age, atrophy_rate, vent_rate]
  context = [followup_age, sex, followup_diag, followup_cortex, followup_hipp, ...]
  Target: predict noise in followup_z

Backward (B→A):
  condition_6ch = [followup_z, followup_age, -atrophy_rate, -vent_rate]
  context = [starting_age, sex, starting_diag, starting_cortex, starting_hipp, ...]
  Target: predict noise in starting_z

Usage:
    python train_controlnet_6ch_btr.py \\
        --dataset_csv /path/to/B_mci_inn1.csv \\
        --cache_dir   /path/to/cache \\
        --output_dir  /path/to/output \\
        --aekl_ckpt   /path/to/autoencoder.pth \\
        --diff_ckpt   /path/to/latentdiffusion.pth \\
        --btc_weight  0.5
"""

import os
import sys
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

# Fix PyTorch 2.6+ weights_only default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'brlp_src'))
INNOV_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, INNOV_SRC)

from brlp import const, utils, networks
from brlp import get_dataset_from_pd
from mci_conditioning import (
    init_controlnet_mci,
    init_controlnet_mci_from_pretrained,
    build_controlnet_condition,
)

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """Build forward cross-attention context (8-dim) + pass through rate columns."""
    conditions = [
        _dict['followup_age'],
        _dict['sex'],
        _dict['followup_diagnosis'],
        _dict['followup_cerebral_cortex'],
        _dict['followup_hippocampus'],
        _dict['followup_amygdala'],
        _dict['followup_cerebral_white_matter'],
        _dict['followup_lateral_ventricle'],
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    _dict['hippocampal_atrophy_rate'] = float(_dict.get('hippocampal_atrophy_rate', 0.0))
    _dict['ventricular_expansion_rate'] = float(_dict.get('ventricular_expansion_rate', 0.0))
    return _dict


def build_reverse_context(batch):
    """Build cross-attention context for the reverse direction (B→A)."""
    conditions = [
        batch['starting_age'],
        batch['sex'],
        batch['starting_diagnosis'],
        batch['starting_cerebral_cortex'],
        batch['starting_hippocampus'],
        batch['starting_amygdala'],
        batch['starting_cerebral_white_matter'],
        batch['starting_lateral_ventricle'],
    ]
    return torch.stack(conditions, dim=-1).unsqueeze(1)


def combined_6ch_btr_loss(
    controlnet, diffusion, scheduler,
    starting_z, followup_z,
    forward_context, forward_condition_6ch,
    reverse_context, reverse_condition_6ch,
    device, btc_weight=0.5,
):
    """
    Combined forward + backward noise-prediction loss with 6-channel conditioning.

    Args:
        forward_condition_6ch: (N, 6, D, H, W) — [starting_z, starting_age, atrophy, vent]
        reverse_condition_6ch: (N, 6, D, H, W) — [followup_z, followup_age, -atrophy, -vent]
    """
    n = starting_z.shape[0]

    # Forward: A→B (predict noise in followup_z)
    noise_fwd = torch.randn_like(followup_z).to(device)
    t_fwd = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()
    noised_fwd = scheduler.add_noise(followup_z, noise=noise_fwd, timesteps=t_fwd)

    down_h, mid_h = controlnet(
        x=noised_fwd.float(), timesteps=t_fwd,
        context=forward_context.float(),
        controlnet_cond=forward_condition_6ch.float()
    )
    pred_fwd = diffusion(
        x=noised_fwd.float(), timesteps=t_fwd,
        context=forward_context.float(),
        down_block_additional_residuals=down_h,
        mid_block_additional_residual=mid_h
    )
    loss_fwd = F.mse_loss(pred_fwd.float(), noise_fwd.float())

    # Backward: B→A (predict noise in starting_z)
    noise_bwd = torch.randn_like(starting_z).to(device)
    t_bwd = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()
    noised_bwd = scheduler.add_noise(starting_z, noise=noise_bwd, timesteps=t_bwd)

    down_h_b, mid_h_b = controlnet(
        x=noised_bwd.float(), timesteps=t_bwd,
        context=reverse_context.float(),
        controlnet_cond=reverse_condition_6ch.float()
    )
    pred_bwd = diffusion(
        x=noised_bwd.float(), timesteps=t_bwd,
        context=reverse_context.float(),
        down_block_additional_residuals=down_h_b,
        mid_block_additional_residual=mid_h_b
    )
    loss_bwd = F.mse_loss(pred_bwd.float(), noise_bwd.float())

    total = loss_fwd + btc_weight * loss_bwd
    return total, loss_fwd, loss_bwd


def images_to_tensorboard(writer, epoch, mode, autoencoder, diffusion,
                          controlnet, dataset, scale_factor):
    """Visualize generation on TensorBoard using 6ch sampling."""
    from generative.networks.schedulers import DDIMScheduler
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice(range(len(dataset)), min(3, len(dataset)))

    for tag_i, i in enumerate(random_indices):
        starting_z = dataset[i]['starting_latent'] * scale_factor
        context = dataset[i]['context'].squeeze(0)
        starting_a = dataset[i]['starting_age']
        atrophy_rate = dataset[i]['hippocampal_atrophy_rate']
        vent_rate = dataset[i]['ventricular_expansion_rate']

        starting_image = torch.from_numpy(
            nib.load(dataset[i]['starting_image']).get_fdata()
        ).unsqueeze(0)
        followup_image = torch.from_numpy(
            nib.load(dataset[i]['followup_image']).get_fdata()
        ).unsqueeze(0)
        starting_image = resample_fn(starting_image).squeeze(0)
        followup_image = resample_fn(followup_image).squeeze(0)

        predicted_image = sample_6ch(
            autoencoder=autoencoder, diffusion=diffusion,
            controlnet=controlnet, starting_z=starting_z,
            starting_a=starting_a, context=context,
            atrophy_rate=atrophy_rate, ventricular_rate=vent_rate,
            device=DEVICE, scale_factor=scale_factor,
            num_inference_steps=50, verbose=False
        )

        utils.tb_display_cond_generation(
            writer=writer, step=epoch,
            tag=f'{mode}/comparison_{tag_i}',
            starting_image=starting_image,
            followup_image=followup_image,
            predicted_image=predicted_image
        )


@torch.no_grad()
def sample_6ch(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context,
    atrophy_rate, ventricular_rate,
    device, scale_factor=1, average_over_n=1,
    num_inference_steps=50, verbose=True
):
    """Inference with 6-channel conditioning."""
    from generative.networks.schedulers import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205,
        clip_sample=False
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

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
                x=z.float(), timesteps=timestep,
                context=context_t,
                controlnet_cond=controlnet_condition.float()
            )
            noise_pred = diffusion(
                x=z.float(), timesteps=timestep,
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
    parser = argparse.ArgumentParser(
        description='Combined Inn1+Inn2: 6ch ControlNet + BTR')
    parser.add_argument('--dataset_csv', required=True, type=str,
                        help='Path to B_mci_inn1.csv (with rate columns)')
    parser.add_argument('--cache_dir',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   default=None,  type=str,
                        help='Resume from existing 6ch ControlNet ckpt')
    parser.add_argument('--pretrained_cnet_4ch', default=None, type=str,
                        help='Pretrained 4ch ControlNet to expand to 6ch')
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=5,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=2.5e-5, type=float)
    parser.add_argument('--btc_weight',  default=0.5,   type=float,
                        help='Weight for backward temporal consistency loss')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Combined Inn1+Inn2] 6ch ControlNet + BTR")
    print(f"  btc_weight (λ): {args.btc_weight}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {args.n_epochs} | BS: {args.batch_size} | LR: {args.lr}")

    # ─── Data ───
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(
            keys=['starting_latent', 'followup_latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(
            keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(
            keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    for col in ['hippocampal_atrophy_rate', 'ventricular_expansion_rate']:
        if col not in dataset_df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. Use B_mci_inn1.csv from Innovation 1.")

    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    print(f"  Training: {len(train_df)} pairs | Validation: {len(valid_df)} pairs")

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        dataset=trainset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(
        dataset=validset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True)

    # ─── Models ───
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt)

    # 6-channel ControlNet initialization
    if args.cnet_ckpt is not None:
        print(f'  Resuming from 6ch checkpoint: {args.cnet_ckpt}')
        controlnet = init_controlnet_mci(args.cnet_ckpt)
    elif args.pretrained_cnet_4ch is not None:
        print(f'  Expanding 4ch→6ch from: {args.pretrained_cnet_4ch}')
        controlnet = init_controlnet_mci_from_pretrained(args.pretrained_cnet_4ch)
    else:
        print('  WARNING: No pretrained ControlNet. Initializing from UNet.')
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
    print(f"  Scale factor: {scale_factor:.4f}")

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    global_counter = {'train': 0, 'valid': 0}
    loaders = {'train': train_loader, 'valid': valid_loader}
    datasets = {'train': trainset, 'valid': validset}

    # ─── Training loop ───
    for epoch in range(args.n_epochs):
        for mode in loaders.keys():
            print(f'mode: {mode}')
            loader = loaders[mode]
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss_total = 0.
            epoch_loss_fwd = 0.
            epoch_loss_bwd = 0.
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(mode == 'train'):
                    starting_z = batch['starting_latent'].to(DEVICE) * scale_factor
                    followup_z = batch['followup_latent'].to(DEVICE) * scale_factor
                    forward_context = batch['context'].to(DEVICE)
                    starting_a = batch['starting_age'].to(DEVICE)
                    followup_a = batch['followup_age'].to(DEVICE)
                    atrophy_rate = batch['hippocampal_atrophy_rate'].float().to(DEVICE)
                    vent_rate = batch['ventricular_expansion_rate'].float().to(DEVICE)
                    n = starting_z.shape[0]

                    with autocast(enabled=True):
                        # Forward 6ch condition: [starting_z, starting_age, atrophy, vent]
                        fwd_condition = build_controlnet_condition(
                            starting_z, starting_a, atrophy_rate, vent_rate
                        )

                        # Backward 6ch condition: [followup_z, followup_age, -atrophy, -vent]
                        # Negate rates because disease progression is reversed
                        bwd_condition = build_controlnet_condition(
                            followup_z, followup_a, -atrophy_rate, -vent_rate
                        )

                        # Reverse context (starting covariates for B→A)
                        reverse_context = build_reverse_context(batch).to(DEVICE)

                        # Combined bidirectional loss with 6ch conditioning
                        total_loss, loss_fwd, loss_bwd = combined_6ch_btr_loss(
                            controlnet=controlnet,
                            diffusion=diffusion,
                            scheduler=scheduler,
                            starting_z=starting_z,
                            followup_z=followup_z,
                            forward_context=forward_context,
                            forward_condition_6ch=fwd_condition,
                            reverse_context=reverse_context,
                            reverse_condition_6ch=bwd_condition,
                            device=DEVICE,
                            btc_weight=args.btc_weight,
                        )

                if mode == 'train':
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-total', total_loss.item(),
                                  global_counter[mode])
                writer.add_scalar(f'{mode}/batch-fwd', loss_fwd.item(),
                                  global_counter[mode])
                writer.add_scalar(f'{mode}/batch-bwd', loss_bwd.item(),
                                  global_counter[mode])
                epoch_loss_total += total_loss.item()
                epoch_loss_fwd += loss_fwd.item()
                epoch_loss_bwd += loss_bwd.item()
                progress_bar.set_postfix({
                    "total": epoch_loss_total / (step + 1),
                    "fwd": epoch_loss_fwd / (step + 1),
                    "bwd": epoch_loss_bwd / (step + 1),
                })
                global_counter[mode] += 1

            n_steps = len(loader)
            writer.add_scalar(f'{mode}/epoch-total', epoch_loss_total / n_steps, epoch)
            writer.add_scalar(f'{mode}/epoch-fwd', epoch_loss_fwd / n_steps, epoch)
            writer.add_scalar(f'{mode}/epoch-bwd', epoch_loss_bwd / n_steps, epoch)

            print(f"  [Epoch {epoch}] {mode}: "
                  f"total={epoch_loss_total/n_steps:.6f}  "
                  f"fwd={epoch_loss_fwd/n_steps:.6f}  "
                  f"bwd={epoch_loss_bwd/n_steps:.6f}")

            images_to_tensorboard(
                writer=writer, epoch=epoch, mode=mode,
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet, dataset=datasets[mode],
                scale_factor=scale_factor)

        # Save from epoch 1 onwards
        if epoch >= 1:
            savepath = os.path.join(args.output_dir, f'cnet-6ch-btr-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            print(f"  Checkpoint: {savepath}")

    print("[Combined Inn1+Inn2] Training complete.")
