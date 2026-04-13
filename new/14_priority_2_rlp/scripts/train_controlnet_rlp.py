"""
Priority 2: Residual Latent Prediction (RLP) — ControlNet Training.

Key difference from baseline train_controlnet.py:
  Instead of predicting noise on followup_z directly, we predict noise on
  the residual delta_z = followup_z - starting_z. This leverages the
  sparsity of longitudinal brain changes for easier diffusion learning.

  Reference: TADM-3D (CMIG 2026, arXiv:2509.03141) validated residual
  diffusion for brain MRI prediction.

Usage:
    python train_controlnet_rlp.py \
        --dataset_csv /path/to/B_mci.csv \
        --cache_dir   /path/to/cache \
        --output_dir  /path/to/output \
        --aekl_ckpt   /path/to/autoencoder.pth \
        --diff_ckpt   /path/to/latentdiffusion.pth
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
for p in [BRLP_SRC, BRLP_SRC_ALT]:
    if os.path.isdir(p):
        sys.path.insert(0, p)

from brlp import const, utils, networks
from brlp import get_dataset_from_pd

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """Build cross-attention context (8-dim), same as BrLP baseline."""
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
    """
    Compute scale_factor = 1 / std(delta_z) for residual latent prediction.
    
    Unlike baseline BrLP which uses 1/std(followup_z), RLP needs the scale
    factor computed on the residual delta_z = followup_z - starting_z.
    """
    deltas = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        delta = sample['followup_latent'] - sample['starting_latent']
        deltas.append(delta)
    deltas = torch.stack(deltas)
    sf = 1.0 / torch.std(deltas)
    return sf


def images_to_tensorboard_rlp(writer, epoch, mode, autoencoder, diffusion,
                               controlnet, dataset, scale_factor):
    """Visualize RLP generation on TensorBoard."""
    # Import the RLP sampling function
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src')))
    from sampling_rlp import sample_using_controlnet_and_z_rlp

    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice(range(len(dataset)), min(3, len(dataset)))

    for tag_i, i in enumerate(random_indices):
        starting_z = dataset[i]['starting_latent'] * scale_factor
        context = dataset[i]['context'].squeeze(0)
        starting_a = dataset[i]['starting_age']
        # For RLP, we need unscaled starting latent for residual reconstruction
        starting_z_unscaled = dataset[i]['starting_latent']

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
        description='Priority 2: ControlNet with Residual Latent Prediction (RLP)')
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--cache_dir',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   default=None,  type=str,
                        help='Resume from existing ControlNet ckpt')
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=5,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=2.5e-5, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Priority 2] Residual Latent Prediction (RLP)")
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

    # ─── RLP: Use residual-based scale factor ───
    # Key difference: scale_factor is computed from delta_z, not followup_z
    scale_factor = compute_residual_scale_factor(trainset, n_samples=min(50, len(trainset)))
    print(f"  Scale factor (residual): {scale_factor:.4f}")

    # Also compute baseline scale_factor for comparison
    with torch.no_grad():
        with autocast(enabled=True):
            z_baseline = trainset[0]['followup_latent']
    sf_baseline = 1 / torch.std(z_baseline)
    print(f"  Scale factor (baseline): {sf_baseline:.4f}")

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
                    n = starting_z.shape[0]

                    with autocast(enabled=True):
                        # ControlNet spatial condition: [starting_z, starting_age]
                        concatenating_age = starting_a.view(n, 1, 1, 1, 1).expand(
                            n, 1, *starting_z.shape[-3:])
                        controlnet_condition = torch.cat(
                            [starting_z, concatenating_age], dim=1)

                        # ============ RLP: Residual target ============
                        # Instead of adding noise to followup_z, we add noise
                        # to delta_z = followup_z - starting_z
                        delta_z = followup_z - starting_z

                        noise = torch.randn_like(delta_z).to(DEVICE)
                        timesteps = torch.randint(
                            0, scheduler.num_train_timesteps,
                            (n,), device=DEVICE).long()

                        # Noise is added to delta_z, not followup_z
                        delta_noised = scheduler.add_noise(
                            delta_z, noise=noise, timesteps=timesteps)
                        # =============================================

                        down_h, mid_h = controlnet(
                            x=delta_noised.float(),
                            timesteps=timesteps,
                            context=context.float(),
                            controlnet_cond=controlnet_condition.float()
                        )

                        noise_pred = diffusion(
                            x=delta_noised.float(),
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

                writer.add_scalar(f'{mode}/batch-mse', loss.item(),
                                  global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            print(f"  [Epoch {epoch}] {mode}: loss={epoch_loss:.6f}")

            images_to_tensorboard_rlp(
                writer=writer, epoch=epoch, mode=mode,
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet, dataset=datasets[mode],
                scale_factor=scale_factor)

        # Save from epoch 1 onwards
        if epoch >= 1:
            savepath = os.path.join(args.output_dir, f'cnet-rlp-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            print(f"  Checkpoint: {savepath}")

    print("[Priority 2] RLP Training complete.")
