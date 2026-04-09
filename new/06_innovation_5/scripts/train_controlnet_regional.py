"""
Modified ControlNet Training with Hippocampal Region Attention Weighting.

Innovation 5: Applies higher loss weight to hippocampus + amygdala regions
in latent space during ControlNet training, focusing noise prediction on
MCI-critical brain structures.

Changes vs original train_controlnet.py:
  1. Loads SynthSeg segmentation paths from CSV
  2. Generates per-sample latent space region weight maps
  3. Replaces uniform MSE with CombinedRegionLoss (alpha=0.5)
  4. Logs region-weighted vs uniform loss comparison
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

# Fix PyTorch 2.6+ weights_only=True default for MONAI PersistentDataset
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Add BrLP source and innovation source to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
INNOV_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, INNOV_SRC)

from brlp import const
from brlp import utils
from brlp import networks
from brlp import (
    get_dataset_from_pd,
    sample_using_controlnet_and_z
)
from region_weights import create_weight_map_latent_space
from weighted_losses import CombinedRegionLoss


warnings.filterwarnings("ignore")
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


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
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
    return _dict


def images_to_tensorboard(
    writer, epoch, mode,
    autoencoder, diffusion, controlnet,
    dataset, scale_factor
):
    """Visualize generation on tensorboard."""
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice(range(len(dataset)), min(3, len(dataset)))

    for tag_i, i in enumerate(random_indices):
        starting_z = dataset[i]['starting_latent'] * scale_factor
        # squeeze context back to 1D (8,) because sample_using_controlnet_and_z
        # internally does unsqueeze(0).unsqueeze(0) to create (1,1,8)
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
            autoencoder=autoencoder,
            diffusion=diffusion,
            controlnet=controlnet,
            starting_z=starting_z,
            starting_a=starting_a,
            context=context,
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ControlNet training with hippocampal attention')
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
    # Innovation 5 specific args
    parser.add_argument('--roi_weight',   default=3.0,  type=float, help='Weight for hippocampal regions')
    parser.add_argument('--region_alpha', default=0.5,  type=float, help='Blending: 0=uniform, 1=region-only')
    parser.add_argument('--changelog',    default=None, type=str,   help='Path to changelog.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Log this run
    if args.changelog:
        log_change(args.changelog, {
            "timestamp": datetime.now().isoformat(),
            "action": "start_controlnet_training",
            "innovation": "5_hippocampal_attention",
            "params": {
                "roi_weight": args.roi_weight,
                "region_alpha": args.region_alpha,
                "lr": args.lr,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
            },
            "description": f"ControlNet training with latent-space hippocampal weight={args.roi_weight}, alpha={args.region_alpha}"
        })

    # ---- Data loading ----
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    valid_df = dataset_df[dataset_df.split == 'valid']
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )
    valid_loader = DataLoader(
        dataset=validset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )

    # ---- Models ----
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet = networks.init_controlnet()

    if args.cnet_ckpt is not None:
        print('Resuming training...')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        print('Copying weights from diffusion model')
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

    # Innovation 5: Region-weighted MSE loss
    region_mse_fn = CombinedRegionLoss(alpha=args.region_alpha, loss_type='mse')
    uniform_mse_fn = F.mse_loss

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    global_counter = {'train': 0, 'valid': 0}
    loaders = {'train': train_loader, 'valid': valid_loader}
    datasets = {'train': trainset, 'valid': validset}

    # ---- Weight map cache ----
    _weight_cache = {}

    def get_latent_weight_map(segm_path_str):
        """Get or compute latent-space weight map."""
        if segm_path_str not in _weight_cache:
            try:
                wmap = create_weight_map_latent_space(
                    segm_path_str,
                    latent_channels=3,
                    roi_weight=args.roi_weight,
                    background_weight=1.0,
                )
                _weight_cache[segm_path_str] = wmap
            except Exception as e:
                print(f"Warning: Could not compute weight map for {segm_path_str}: {e}")
                _weight_cache[segm_path_str] = None
        return _weight_cache[segm_path_str]

    print(f"[Innovation 5] Training ControlNet with hippocampal attention weighting")
    print(f"  ROI weight: {args.roi_weight}x | Alpha: {args.region_alpha}")
    print(f"  Device: {DEVICE}")
    print(f"  Training samples: {len(train_df)} | Validation: {len(valid_df)}")

    for epoch in range(args.n_epochs):

        for mode in loaders.keys():
            print('mode:', mode)
            loader = loaders[mode]
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss = 0.
            epoch_loss_uniform = 0.
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

                        concatenating_age = starting_a.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
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

                        # Innovation 5: Apply region-weighted MSE loss
                        # CSV B has 'starting_segm' and 'followup_segm' (path columns)
                        use_segm = False
                        segm_key = None
                        for candidate in ['followup_segm', 'starting_segm',
                                           'followup_segm_path', 'starting_segm_path']:
                            if candidate in batch:
                                segm_key = candidate
                                use_segm = True
                                break

                        if use_segm and segm_key:
                            segm_paths = batch[segm_key]
                            weight_maps = []
                            all_valid = True
                            for sp in segm_paths:
                                wm = get_latent_weight_map(sp)
                                if wm is None:
                                    all_valid = False
                                    break
                                weight_maps.append(wm)

                            if all_valid:
                                weight_batch = torch.stack(weight_maps, dim=0).to(DEVICE)
                                loss = region_mse_fn(noise_pred.float(), noise.float(), weight_batch)
                            else:
                                loss = uniform_mse_fn(noise_pred.float(), noise.float())
                        else:
                            loss = uniform_mse_fn(noise_pred.float(), noise.float())

                        # Also compute uniform loss for comparison logging
                        loss_uniform = uniform_mse_fn(noise_pred.float(), noise.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-mse-weighted', loss.item(), global_counter[mode])
                writer.add_scalar(f'{mode}/batch-mse-uniform', loss_uniform.item(), global_counter[mode])
                epoch_loss += loss.item()
                epoch_loss_uniform += loss_uniform.item()
                progress_bar.set_postfix({
                    "loss_w": epoch_loss / (step + 1),
                    "loss_u": epoch_loss_uniform / (step + 1)
                })
                global_counter[mode] += 1

            epoch_loss = epoch_loss / len(loader)
            epoch_loss_uniform = epoch_loss_uniform / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse-weighted', epoch_loss, epoch)
            writer.add_scalar(f'{mode}/epoch-mse-uniform', epoch_loss_uniform, epoch)

            images_to_tensorboard(
                writer=writer,
                epoch=epoch,
                mode=mode,
                autoencoder=autoencoder,
                diffusion=diffusion,
                controlnet=controlnet,
                dataset=datasets[mode],
                scale_factor=scale_factor
            )

        if epoch >= 1:
            savepath = os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)
            print(f"  Checkpoint saved: {savepath}")

    if args.changelog:
        log_change(args.changelog, {
            "timestamp": datetime.now().isoformat(),
            "action": "finish_controlnet_training",
            "innovation": "5_hippocampal_attention",
            "result": f"Completed {args.n_epochs} epochs. Final weighted loss: {epoch_loss:.6f}"
        })

    print("ControlNet training complete.")
