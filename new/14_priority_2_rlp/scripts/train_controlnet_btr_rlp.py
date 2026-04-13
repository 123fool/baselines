"""
BTR + RLP Combined Training: Bidirectional Temporal Regularization
with Residual Latent Prediction.

Combines Innovation 2 (BTR) with Priority 2 (RLP):
  - Forward: predict noise on delta_z = followup_z - starting_z
  - Backward: predict noise on -delta_z (reverse direction)
  - Total loss = L_fwd + lambda * L_bwd

Reference: TADM-3D uses both residual prediction AND bidirectional training.
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

# Fix PyTorch 2.6+ weights_only default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Paths — support both local workspace and server layout
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
BRLP_SRC_ALT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'brlp_src'))
INNOV2_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '12_innovation_2', 'src'))
INNOV2_SRC_ALT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'innov2_src'))
RLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
for p in [BRLP_SRC, BRLP_SRC_ALT, INNOV2_SRC, INNOV2_SRC_ALT, RLP_SRC]:
    if os.path.isdir(p):
        sys.path.insert(0, p)

from brlp import const, utils, networks
from brlp import get_dataset_from_pd
from bidirectional_temporal import build_reverse_context
from sampling_rlp import sample_using_controlnet_and_z_rlp

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """Build forward cross-attention context (8-dim)."""
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
    return _dict


def compute_residual_scale_factor(dataset, n_samples=50):
    """Compute scale_factor = 1 / std(delta_z) for RLP."""
    deltas = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        delta = sample['followup_latent'] - sample['starting_latent']
        deltas.append(delta)
    deltas = torch.stack(deltas)
    sf = 1.0 / torch.std(deltas)
    return sf


def bidirectional_rlp_loss(
    controlnet, diffusion, scheduler,
    starting_z, followup_z,
    forward_context, forward_condition,
    reverse_context, reverse_condition,
    device, btc_weight=0.5,
):
    """
    Combined BTR + RLP loss: bidirectional noise prediction on residual latents.

    Forward: noise on delta_z = followup_z - starting_z
    Backward: noise on -delta_z = starting_z - followup_z
    """
    n = starting_z.shape[0]

    # Compute residual
    delta_z = followup_z - starting_z

    # ── Forward: predict noise on delta_z conditioned on starting_z ──
    noise_fwd = torch.randn_like(delta_z).to(device)
    t_fwd = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()
    noised_fwd = scheduler.add_noise(delta_z, noise=noise_fwd, timesteps=t_fwd)

    down_h, mid_h = controlnet(
        x=noised_fwd.float(), timesteps=t_fwd,
        context=forward_context.float(),
        controlnet_cond=forward_condition.float(),
    )
    pred_fwd = diffusion(
        x=noised_fwd.float(), timesteps=t_fwd,
        context=forward_context.float(),
        down_block_additional_residuals=down_h,
        mid_block_additional_residual=mid_h,
    )
    loss_fwd = F.mse_loss(pred_fwd.float(), noise_fwd.float())

    # ── Backward: predict noise on -delta_z conditioned on followup_z ──
    neg_delta_z = -delta_z
    noise_bwd = torch.randn_like(neg_delta_z).to(device)
    t_bwd = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()
    noised_bwd = scheduler.add_noise(neg_delta_z, noise=noise_bwd, timesteps=t_bwd)

    down_h_b, mid_h_b = controlnet(
        x=noised_bwd.float(), timesteps=t_bwd,
        context=reverse_context.float(),
        controlnet_cond=reverse_condition.float(),
    )
    pred_bwd = diffusion(
        x=noised_bwd.float(), timesteps=t_bwd,
        context=reverse_context.float(),
        down_block_additional_residuals=down_h_b,
        mid_block_additional_residual=mid_h_b,
    )
    loss_bwd = F.mse_loss(pred_bwd.float(), noise_bwd.float())

    total = loss_fwd + btc_weight * loss_bwd
    return total, loss_fwd, loss_bwd


def images_to_tensorboard(writer, epoch, mode, autoencoder, diffusion,
                          controlnet, dataset, scale_factor):
    """Visualize RLP generation on TensorBoard."""
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice(range(len(dataset)), min(3, len(dataset)))

    for tag_i, i in enumerate(random_indices):
        starting_z = dataset[i]['starting_latent'] * scale_factor
        starting_z_unscaled = dataset[i]['starting_latent']
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

        predicted_image = sample_using_controlnet_and_z_rlp(
            autoencoder=autoencoder, diffusion=diffusion,
            controlnet=controlnet, starting_z=starting_z,
            starting_z_unscaled=starting_z_unscaled,
            starting_a=starting_a, context=context,
            device=DEVICE, scale_factor=scale_factor
        )

        utils.tb_display_cond_generation(
            writer=writer, step=epoch,
            tag=f'{mode}/comparison_{tag_i}',
            starting_image=starting_image,
            followup_image=followup_image,
            predicted_image=predicted_image
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='BTR + RLP Combined: Bidirectional Temporal + Residual Latent')
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--cache_dir',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   default=None,  type=str)
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=5,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=2.5e-5, type=float)
    parser.add_argument('--btc_weight',  default=0.5,   type=float,
                        help='Weight for backward temporal consistency loss')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[BTR + RLP] Bidirectional Temporal + Residual Latent")
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
    controlnet = networks.init_controlnet()

    if args.cnet_ckpt is not None:
        print(f'  Resuming from {args.cnet_ckpt}')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        print('  Copying weights from diffusion model (default init)')
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    for p in diffusion.parameters():
        p.requires_grad = False

    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    # RLP: residual-based scale factor
    scale_factor = compute_residual_scale_factor(trainset, n_samples=min(50, len(trainset)))
    print(f"  Scale factor (residual): {scale_factor:.4f}")

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
                    n = starting_z.shape[0]

                    with autocast(enabled=True):
                        # Forward condition: [starting_z, starting_age]
                        fwd_age = starting_a.view(n, 1, 1, 1, 1).expand(
                            n, 1, *starting_z.shape[-3:])
                        fwd_condition = torch.cat([starting_z, fwd_age], dim=1)

                        # Reverse condition: [followup_z, followup_age]
                        bwd_age = followup_a.view(n, 1, 1, 1, 1).expand(
                            n, 1, *followup_z.shape[-3:])
                        bwd_condition = torch.cat([followup_z, bwd_age], dim=1)

                        # Reverse context
                        reverse_context = build_reverse_context(batch).to(DEVICE)

                        # Combined bidirectional + residual loss
                        total_loss, loss_fwd, loss_bwd = bidirectional_rlp_loss(
                            controlnet=controlnet,
                            diffusion=diffusion,
                            scheduler=scheduler,
                            starting_z=starting_z,
                            followup_z=followup_z,
                            forward_context=forward_context,
                            forward_condition=fwd_condition,
                            reverse_context=reverse_context,
                            reverse_condition=bwd_condition,
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
            savepath = os.path.join(args.output_dir, f'cnet-btr-rlp-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            print(f"  Checkpoint: {savepath}")

    print("[BTR + RLP] Training complete.")
