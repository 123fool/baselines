"""
Priority 4: BTR + PALM + TEL — ControlNet Training.

Combines Innovation 2 (Bidirectional Temporal Regularization) with
Priority 4 decoration modules (PALM + TEL).

Key additions over Innovation 2 BTR:
  - PALM modulates starting_z/followup_z based on clinical context before
    building the ControlNet spatial condition.
  - TEL adds a learnable Fourier temporal encoding to the age channel.
  - Both modules are trained end-to-end with ControlNet.

Usage:
    python train_controlnet_btc_palm_tel.py \
        --dataset_csv /path/to/B_mci.csv \
        --cache_dir   /path/to/cache \
        --output_dir  /path/to/output \
        --aekl_ckpt   /path/to/autoencoder.pth \
        --diff_ckpt   /path/to/latentdiffusion.pth \
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

# ─── Paths ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'brlp_src'))
INNOV2_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'innov2_src'))
P4_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, INNOV2_SRC)
sys.path.insert(0, P4_SRC)

from brlp import const, utils, networks
from brlp import get_dataset_from_pd, sample_using_controlnet_and_z
from bidirectional_temporal import build_reverse_context, bidirectional_controlnet_loss
from palm_tel import PALM, TEL

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """Build forward cross-attention context (8-dim), same as BrLP baseline."""
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


def images_to_tensorboard(writer, epoch, mode, autoencoder, diffusion,
                          controlnet, dataset, scale_factor):
    """Visualize generation on TensorBoard (uses standard sampling for viz)."""
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

        predicted_image = sample_using_controlnet_and_z(
            autoencoder=autoencoder, diffusion=diffusion,
            controlnet=controlnet, starting_z=starting_z,
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
        description='Priority 4: ControlNet with BTR + PALM + TEL')
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--cache_dir',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   default=None,  type=str,
                        help='Resume from checkpoint (plain or PALM+TEL dict)')
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=5,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=2.5e-5, type=float)
    parser.add_argument('--btc_weight',  default=0.5,  type=float,
                        help='Weight for backward temporal consistency loss')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Priority 4] BTR + PALM + TEL ControlNet Training")
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

    # Priority 4 modules
    palm = PALM(context_dim=8, latent_channels=3)
    tel = TEL(d_model=64)

    if args.cnet_ckpt is not None:
        ckpt = torch.load(args.cnet_ckpt)
        if isinstance(ckpt, dict) and 'controlnet' in ckpt:
            # PALM+TEL checkpoint format
            controlnet.load_state_dict(ckpt['controlnet'])
            palm.load_state_dict(ckpt['palm'])
            tel.load_state_dict(ckpt['tel'])
            print(f'  Resumed from PALM+TEL checkpoint: {args.cnet_ckpt}')
        else:
            # Plain ControlNet checkpoint (e.g. from BTR)
            controlnet.load_state_dict(ckpt)
            print(f'  Loaded ControlNet from: {args.cnet_ckpt}')
            print(f'  PALM + TEL initialized from scratch')
    else:
        print('  Copying weights from diffusion model (default init)')
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    for p in diffusion.parameters():
        p.requires_grad = False

    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)
    palm.to(DEVICE)
    tel.to(DEVICE)

    # Combined optimizer for ControlNet + PALM + TEL
    params_to_train = (
        list(controlnet.parameters()) +
        list(palm.parameters()) +
        list(tel.parameters())
    )
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(params_to_train, lr=args.lr)

    print(f"  PALM params: {sum(p.numel() for p in palm.parameters())}")
    print(f"  TEL params:  {sum(p.numel() for p in tel.parameters())}")

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
            if mode == 'train':
                controlnet.train()
                palm.train()
                tel.train()
            else:
                controlnet.eval()
                palm.eval()
                tel.eval()

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
                    forward_context = batch['context'].to(DEVICE)    # (N, 1, 8)
                    starting_a = batch['starting_age'].to(DEVICE)
                    followup_a = batch['followup_age'].to(DEVICE)
                    n = starting_z.shape[0]

                    with autocast(enabled=True):
                        # ── PALM: modulate latents based on clinical context ──
                        palm_fwd_ctx = forward_context.squeeze(1)    # (N, 8)
                        reverse_context = build_reverse_context(batch).to(DEVICE)
                        palm_rev_ctx = reverse_context.squeeze(1)    # (N, 8)

                        modulated_starting_z = palm(starting_z, palm_fwd_ctx)
                        modulated_followup_z = palm(followup_z, palm_rev_ctx)

                        # ── TEL: enhance age channel with temporal encoding ──
                        fwd_age_gap = followup_a - starting_a        # (N,)
                        bwd_age_gap = starting_a - followup_a        # (N,)
                        fwd_tel = tel(fwd_age_gap).squeeze(-1)       # (N,)
                        bwd_tel = tel(bwd_age_gap).squeeze(-1)       # (N,)

                        fwd_age_enhanced = starting_a + fwd_tel      # (N,)
                        bwd_age_enhanced = followup_a + bwd_tel      # (N,)

                        # ── Build spatial conditions ──
                        fwd_age = fwd_age_enhanced.view(n, 1, 1, 1, 1).expand(
                            n, 1, *starting_z.shape[-3:])
                        fwd_condition = torch.cat([modulated_starting_z, fwd_age], dim=1)

                        bwd_age = bwd_age_enhanced.view(n, 1, 1, 1, 1).expand(
                            n, 1, *followup_z.shape[-3:])
                        bwd_condition = torch.cat([modulated_followup_z, bwd_age], dim=1)

                        # ── Combined bidirectional loss ──
                        total_loss, loss_fwd, loss_bwd = bidirectional_controlnet_loss(
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
            savepath = os.path.join(args.output_dir,
                                    f'cnet-btc-palm-tel-ep-{epoch}.pth')
            torch.save({
                'controlnet': controlnet.state_dict(),
                'palm': palm.state_dict(),
                'tel': tel.state_dict(),
            }, savepath)
            print(f"  Checkpoint: {savepath}")

    print("[Priority 4] BTR + PALM + TEL Training complete.")
