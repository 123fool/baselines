"""
Modified AutoEncoder Training with 3D Perceptual Loss + Frequency Domain Constraint.

Innovation 4: Replace fake-3D PerceptualLoss (2D VGG squeeze + 20% slice sampling)
with true 3D MedicalNet ResNet-10 perceptual loss + add Laplacian pyramid frequency
constraint to preserve high-frequency brain structures.

Changes vs original train_autoencoder.py:
  1. Replace PerceptualLoss(is_fake_3d=True) with MedicalNet3DPerceptualLoss
  2. Add LaplacianPyramidLoss for multi-scale frequency preservation
  3. Add optional FFTFrequencyLoss for spectral consistency
  4. Log all new loss components to TensorBoard + dashboard changelog
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism

# Fix PyTorch 2.6+ weights_only=True default for MONAI PersistentDataset
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter

# Add BrLP source and Innovation 4 source to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
INNOV_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, INNOV_SRC)

from brlp import const
from brlp import utils
from brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
    get_dataset_from_pd
)
from medicalnet_perceptual import MedicalNet3DPerceptualLoss
from frequency_losses import LaplacianPyramidLoss, FFTFrequencyLoss


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def log_change(changelog_path, entry):
    """Append an entry to the changelog JSON file."""
    if os.path.exists(changelog_path):
        with open(changelog_path, 'r') as f:
            log = json.load(f)
    else:
        log = []
    log.append(entry)
    with open(changelog_path, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AE training with 3D perceptual loss + frequency constraint')
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str, help='Pretrained AE checkpoint to finetune')
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--aug_p',          default=0.8,   type=float)
    # Innovation 4 specific args
    parser.add_argument('--mednet_ckpt',    required=True, type=str, help='Path to resnet_10_23dataset.pth')
    parser.add_argument('--perc_weight',    default=0.001, type=float, help='MedicalNet perceptual loss weight')
    parser.add_argument('--freq_weight',    default=0.01,  type=float, help='Laplacian pyramid frequency loss weight')
    parser.add_argument('--fft_weight',     default=0.0,   type=float, help='FFT frequency loss weight (0 = disabled)')
    parser.add_argument('--lap_levels',     default=3,     type=int, help='Laplacian pyramid levels')
    parser.add_argument('--changelog',      default=None,  type=str, help='Path to changelog.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Log this run
    if args.changelog:
        log_change(args.changelog, {
            "timestamp": datetime.now().isoformat(),
            "action": "start_ae_training",
            "innovation": "4_3d_perceptual_frequency",
            "params": {
                "perc_weight": args.perc_weight,
                "freq_weight": args.freq_weight,
                "fft_weight": args.fft_weight,
                "lap_levels": args.lap_levels,
                "lr": args.lr,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "aekl_ckpt": args.aekl_ckpt,
                "mednet_ckpt": args.mednet_ckpt,
            },
            "description": f"AE finetuning with 3D MedicalNet perceptual loss (w={args.perc_weight}) + Laplacian freq (w={args.freq_weight}) + FFT (w={args.fft_weight})"
        })

    # ---- Data loading ----
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=args.max_batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )

    # ---- Models ----
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    # ---- Loss weights ----
    adv_weight = 0.025
    perceptual_weight = args.perc_weight
    freq_weight = args.freq_weight
    fft_weight = args.fft_weight
    kl_weight = 1e-7

    # ---- Loss functions ----
    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    # Innovation 4: True 3D MedicalNet perceptual loss (replaces fake-3D VGG)
    print(f"[Innovation 4] Loading MedicalNet ResNet-10 from {args.mednet_ckpt}")
    perc_loss_fn = MedicalNet3DPerceptualLoss(pretrained_path=args.mednet_ckpt).to(DEVICE)

    # Innovation 4: Laplacian pyramid frequency loss
    freq_loss_fn = LaplacianPyramidLoss(num_levels=args.lap_levels).to(DEVICE)

    # Innovation 4: Optional FFT frequency loss
    fft_loss_fn = None
    if fft_weight > 0:
        fft_loss_fn = FFTFrequencyLoss(weight_high_freq=True).to(DEVICE)

    # ---- Optimizers ----
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    gradacc_g = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler()
    )

    gradacc_d = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d,
        grad_scaler=GradScaler()
    )

    avgloss = utils.AverageLoss()
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    total_counter = 0

    print(f"[Innovation 4] Training AE with 3D MedicalNet perceptual + frequency constraints")
    print(f"  Perceptual weight: {perceptual_weight} | Freq weight: {freq_weight} | FFT weight: {fft_weight}")
    print(f"  Laplacian levels: {args.lap_levels}")
    print(f"  Device: {DEVICE}")
    print(f"  Training samples: {len(train_df)}")

    for epoch in range(args.n_epochs):

        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:

            with autocast(enabled=True):

                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # L1 reconstruction loss
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())

                # KL divergence regularization
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)

                # Innovation 4: True 3D perceptual loss (MedicalNet)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())

                # Innovation 4: Laplacian pyramid frequency loss
                frq_loss = freq_weight * freq_loss_fn(reconstruction.float(), images.float())

                # Innovation 4: Optional FFT frequency loss
                if fft_loss_fn is not None:
                    fft_loss = fft_weight * fft_loss_fn(reconstruction.float(), images.float())
                else:
                    fft_loss = torch.tensor(0.0, device=DEVICE)

                # Adversarial loss
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

                loss_g = rec_loss + kld_loss + per_loss + frq_loss + fft_loss + gen_loss

            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging
            avgloss.put('Generator/reconstruction_loss',    rec_loss.item())
            avgloss.put('Generator/perceptual_3d_loss',     per_loss.item())
            avgloss.put('Generator/frequency_loss',         frq_loss.item())
            avgloss.put('Generator/fft_loss',               fft_loss.item() if isinstance(fft_loss, torch.Tensor) else fft_loss)
            avgloss.put('Generator/adversarial_loss',       gen_loss.item())
            avgloss.put('Generator/kl_regularization',      kld_loss.item())
            avgloss.put('Discriminator/adversarial_loss',   loss_d.item())

            if total_counter % 10 == 0:
                tb_step = total_counter // 10
                avgloss.to_tensorboard(writer, tb_step)
                utils.tb_display_reconstruction(writer, tb_step, images[0].detach().cpu(), reconstruction[0].detach().cpu())

            total_counter += 1

        # Save the model after each epoch
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
        print(f"  [Epoch {epoch}] Saved checkpoints to {args.output_dir}")

        # Log epoch completion
        if args.changelog:
            log_change(args.changelog, {
                "timestamp": datetime.now().isoformat(),
                "action": "epoch_complete",
                "innovation": "4_3d_perceptual_frequency",
                "epoch": epoch,
                "metrics": avgloss.get_latest_dict() if hasattr(avgloss, 'get_latest_dict') else {}
            })

    print(f"[Innovation 4] Training complete. Models saved to {args.output_dir}")
