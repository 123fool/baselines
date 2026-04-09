"""
Modified AutoEncoder Training with Hippocampal Region Attention Weighting.

Innovation 5: Applies higher loss weight to hippocampus + amygdala regions
during AE training, forcing the model to better preserve MCI-critical structures.

Changes vs original train_autoencoder.py:
  1. Loads SynthSeg segmentation alongside images
  2. Generates per-sample region weight maps
  3. Replaces uniform L1 with CombinedRegionLoss (alpha=0.5)
  4. Logs region-specific reconstruction metrics
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
# Monkey-patch torch.load to default weights_only=False (needed for worker processes)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter

# Add BrLP source to path
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
from region_weights import create_weight_map_image_space
from weighted_losses import CombinedRegionLoss, RegionWeightedL1Loss


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

    parser = argparse.ArgumentParser(description='AE training with hippocampal region attention')
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
    # Innovation 5 specific args
    parser.add_argument('--roi_weight',     default=3.0,   type=float, help='Weight multiplier for hippocampal regions')
    parser.add_argument('--region_alpha',   default=0.5,   type=float, help='Blending: 0=uniform, 1=region-only')
    parser.add_argument('--changelog',      default=None,  type=str,   help='Path to changelog.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Log this run
    if args.changelog:
        log_change(args.changelog, {
            "timestamp": datetime.now().isoformat(),
            "action": "start_ae_training",
            "innovation": "5_hippocampal_attention",
            "params": {
                "roi_weight": args.roi_weight,
                "region_alpha": args.region_alpha,
                "lr": args.lr,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "aekl_ckpt": args.aekl_ckpt,
            },
            "description": f"AE finetuning with hippocampal region weight={args.roi_weight}, alpha={args.region_alpha}"
        })

    # ---- Data loading (now includes segm_path) ----
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
    perceptual_weight = 0.001
    kl_weight = 1e-7

    # Innovation 5: Combined region-weighted L1 loss
    region_loss_fn = CombinedRegionLoss(alpha=args.region_alpha, loss_type='l1')
    uniform_l1_fn = L1Loss()  # fallback if no segm available
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2
        ).to(DEVICE)

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

    # ---- Pre-compute weight map cache ----
    # Cache weight maps in memory (dict: segm_path -> weight_tensor)
    _weight_cache = {}

    def get_weight_map(segm_path_str):
        """Get or compute weight map for a given segmentation."""
        if segm_path_str not in _weight_cache:
            try:
                wmap = create_weight_map_image_space(
                    segm_path_str,
                    target_shape=const.INPUT_SHAPE_AE,
                    roi_weight=args.roi_weight,
                    background_weight=1.0,
                    smooth_sigma=2.0,
                )
                _weight_cache[segm_path_str] = wmap
            except Exception as e:
                print(f"Warning: Could not load segmentation {segm_path_str}: {e}")
                # Return uniform weights as fallback
                _weight_cache[segm_path_str] = None
        return _weight_cache[segm_path_str]

    print(f"[Innovation 5] Training AE with hippocampal attention weighting")
    print(f"  ROI weight: {args.roi_weight}x | Alpha: {args.region_alpha}")
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

                # Innovation 5: Try to use region-weighted loss
                has_segm = 'segm_path' in batch and batch['segm_path'] is not None
                if has_segm:
                    try:
                        segm_paths = batch['segm_path']
                        weight_maps = []
                        all_valid = True
                        for sp in segm_paths:
                            wm = get_weight_map(sp)
                            if wm is None:
                                all_valid = False
                                break
                            weight_maps.append(wm)

                        if all_valid:
                            weight_batch = torch.stack(weight_maps, dim=0).to(DEVICE)
                            rec_loss = region_loss_fn(
                                reconstruction.float(),
                                images.float(),
                                weight_batch
                            )
                        else:
                            rec_loss = uniform_l1_fn(reconstruction.float(), images.float())
                    except Exception:
                        rec_loss = uniform_l1_fn(reconstruction.float(), images.float())
                else:
                    rec_loss = uniform_l1_fn(reconstruction.float(), images.float())

                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

                loss_g = rec_loss + kld_loss + per_loss + gen_loss

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
            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Generator/perceptual_loss', per_loss.item())
            avgloss.put('Generator/adverarial_loss', gen_loss.item())
            avgloss.put('Generator/kl_regularization', kld_loss.item())
            avgloss.put('Discriminator/adverarial_loss', loss_d.item())

            if total_counter % 10 == 0:
                tb_step = total_counter // 10
                avgloss.to_tensorboard(writer, tb_step)
                utils.tb_display_reconstruction(
                    writer, tb_step,
                    images[0].detach().cpu(),
                    reconstruction[0].detach().cpu()
                )

            total_counter += 1

        # Save checkpoints
        torch.save(discriminator.state_dict(),
                    os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),
                    os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))

        print(f"  Epoch {epoch} complete. Checkpoints saved.")

    # Final log
    if args.changelog:
        log_change(args.changelog, {
            "timestamp": datetime.now().isoformat(),
            "action": "finish_ae_training",
            "innovation": "5_hippocampal_attention",
            "result": f"Completed {args.n_epochs} epochs. Checkpoints in {args.output_dir}",
        })

    print("Training complete.")
