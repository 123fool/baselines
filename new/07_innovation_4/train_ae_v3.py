"""
Innovation 4 v3: Decoder-only fine-tuning with 3D perceptual + frequency losses.

Key difference from v2: FREEZES ENCODER to preserve the latent space
that diffusion/controlnet models were trained on. Only fine-tunes the decoder
and post_quant_conv so that decoded images are enhanced by 3D perceptual guidance
while remaining compatible with the existing pipeline.
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai import transforms
from monai.losses import PatchAdversarialLoss
from generative.losses import PerceptualLoss

# Fix PyTorch 2.6+ weights_only=True default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from brlp import const, utils
from brlp.data import get_dataset_from_pd
from brlp.networks import init_autoencoder, init_patch_discriminator
from brlp.gradacc import GradientAccumulation
from brlp.losses import KLDivergenceLoss, L1Loss
from medicalnet_perceptual_v2 import MedicalNet3DPerceptualLoss, LaplacianPyramidLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_epoch(autoencoder, val_loader, l1_loss_fn):
    autoencoder.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            with autocast(enabled=True):
                reconstruction, _, _ = autoencoder(images)
                loss = l1_loss_fn(reconstruction.float(), images.float())
            total_loss += loss.item()
            n_batches += 1
    autoencoder.train()
    return total_loss / max(n_batches, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Innovation 4 v3: Decoder-only fine-tuning')
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=4,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=1,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=2e-5,  type=float)
    # Innovation 4 v3 args
    parser.add_argument('--mednet_ckpt',    required=True, type=str)
    parser.add_argument('--perc3d_weight',  default=0.0005, type=float,
                        help='3D MedicalNet perceptual weight (auxiliary)')
    parser.add_argument('--freq_weight',    default=0.005,  type=float,
                        help='Laplacian pyramid frequency loss weight')
    parser.add_argument('--lap_levels',     default=3,      type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data loading ----
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE,
                                        mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid'] if 'valid' in dataset_df.split.values else None

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    train_loader = DataLoader(
        dataset=trainset, num_workers=args.num_workers,
        batch_size=args.max_batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True
    )

    val_loader = None
    if valid_df is not None and len(valid_df) > 0:
        valset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)
        val_loader = DataLoader(
            dataset=valset, num_workers=2,
            batch_size=args.max_batch_size, shuffle=False,
            persistent_workers=False, pin_memory=True
        )

    # ---- Models ----
    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    # ========================================
    # v3 KEY: Freeze encoder to preserve latent space
    # Only decoder + post_quant_conv are trainable
    # ========================================
    frozen_count = 0
    trainable_count = 0
    for name, param in autoencoder.named_parameters():
        if (name.startswith('encoder')
            or name.startswith('quant_conv_mu')
            or name.startswith('quant_conv_log_sigma')):
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()

    total_params = frozen_count + trainable_count
    print(f"[v3] Encoder FROZEN: {frozen_count:,} params")
    print(f"[v3] Decoder trainable: {trainable_count:,} params")
    print(f"[v3] Total: {total_params:,} ({100*trainable_count/total_params:.1f}% trainable)")

    # ---- Loss weights ----
    adv_weight = 0.025
    orig_perc_weight = 0.001
    kl_weight = 1e-7
    perc3d_weight = args.perc3d_weight
    freq_weight = args.freq_weight

    # ---- Loss functions ----
    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig_perc_fn = PerceptualLoss(
            spatial_dims=3, network_type="squeeze",
            is_fake_3d=True, fake_3d_ratio=0.2
        ).to(DEVICE)

    print(f"[v3] Loading MedicalNet ResNet-10 from {args.mednet_ckpt}")
    perc3d_fn = MedicalNet3DPerceptualLoss(
        pretrained_path=args.mednet_ckpt,
        downsample_size=(80, 96, 80),
    ).to(DEVICE)

    freq_fn = LaplacianPyramidLoss(num_levels=args.lap_levels).to(DEVICE)

    # ---- Optimizers (v3: only decoder params for generator) ----
    decoder_params = [p for p in autoencoder.parameters() if p.requires_grad]
    optimizer_g = torch.optim.Adam(decoder_params, lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    gradacc_g = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g, grad_scaler=GradScaler()
    )
    gradacc_d = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d, grad_scaler=GradScaler()
    )

    avgloss = utils.AverageLoss()
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    total_counter = 0
    best_val_loss = float('inf')

    print(f"[Innovation 4 v3] Decoder-only fine-tuning")
    print(f"  Original PerceptualLoss weight: {orig_perc_weight}")
    print(f"  3D MedicalNet weight (auxiliary): {perc3d_weight}")
    print(f"  Laplacian frequency weight:      {freq_weight}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Train samples: {len(train_df)}")
    if valid_df is not None:
        print(f"  Valid samples: {len(valid_df)}")

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            with autocast(enabled=True):
                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # === Original BrLP losses ===
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss_orig = orig_perc_weight * orig_perc_fn(
                    reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(
                    logits_fake, target_is_real=True, for_discriminator=False)

                # === Innovation 4 v3: Auxiliary losses ===
                per_loss_3d = perc3d_weight * perc3d_fn(
                    reconstruction.float(), images.float())
                frq_loss = freq_weight * freq_fn(
                    reconstruction.float(), images.float())

                loss_g = (rec_loss + kld_loss + per_loss_orig
                          + per_loss_3d + frq_loss + gen_loss)

            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False,
                                          for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True,
                                          for_discriminator=True)
                loss_d = adv_weight * (d_loss_fake + d_loss_real) * 0.5

            gradacc_d.step(loss_d, step)

            # Logging
            avgloss.put('G/rec',           rec_loss.item())
            avgloss.put('G/perc_orig',     per_loss_orig.item())
            avgloss.put('G/perc_3d',       per_loss_3d.item())
            avgloss.put('G/freq',          frq_loss.item())
            avgloss.put('G/adv',           gen_loss.item())
            avgloss.put('G/kl',            kld_loss.item())
            avgloss.put('D/loss',          loss_d.item())

            if total_counter % 10 == 0:
                tb_step = total_counter // 10
                avgloss.to_tensorboard(writer, tb_step)
                utils.tb_display_reconstruction(
                    writer, tb_step,
                    images[0].detach().cpu(),
                    reconstruction[0].detach().cpu()
                )
            total_counter += 1

        # ---- Validation ----
        val_loss_str = "N/A"
        if val_loader is not None:
            val_loss = validate_epoch(autoencoder, val_loader, l1_loss_fn)
            val_loss_str = f"{val_loss:.6f}"
            writer.add_scalar('val/rec_loss', val_loss, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(autoencoder.state_dict(),
                           os.path.join(args.output_dir, 'autoencoder-best.pth'))
                torch.save(discriminator.state_dict(),
                           os.path.join(args.output_dir, 'discriminator-best.pth'))
                print(f"  [Epoch {epoch}] New best val_loss={val_loss:.6f}, saved best checkpoint")

        torch.save(autoencoder.state_dict(),
                   os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
        torch.save(discriminator.state_dict(),
                   os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        print(f"  [Epoch {epoch}] val_loss={val_loss_str}, saved checkpoint")

    print(f"[Innovation 4 v3] Training complete. Best val_loss={best_val_loss:.6f}")
    print(f"  Checkpoints in {args.output_dir}")
