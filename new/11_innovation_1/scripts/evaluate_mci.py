"""
Innovation 1: MCI Dynamic Conditioning - Evaluation Script.

Evaluates the 6-channel ControlNet against baseline using:
  - Overall SSIM, PSNR, MAE
  - Hippocampal ROI SSIM, MAE
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
from monai import transforms
from monai.data.image_reader import NumpyReader
from torch.cuda.amp import autocast
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Fix torch.load
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
INNOV_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, INNOV_SRC)

from brlp import const, networks, utils
from mci_conditioning import init_controlnet_mci, build_controlnet_condition

# ROI labels from SynthSeg
HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]
MCI_ROI_LABELS = HIPPOCAMPUS_LABELS + AMYGDALA_LABELS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_roi_mask(segm_data, labels):
    """Create binary mask for specified SynthSeg labels."""
    mask = np.zeros_like(segm_data, dtype=bool)
    for label in labels:
        mask |= (segm_data.round() == label)
    return mask


def compute_region_metrics(pred, target, mask):
    """Compute metrics within a masked region."""
    if mask.sum() == 0:
        return {'mae': float('nan'), 'ssim': float('nan')}
    coords = np.where(mask > 0)
    slices = tuple(slice(c.min(), c.max() + 1) for c in coords)
    pred_roi = pred[slices]
    target_roi = target[slices]
    pred_masked = pred[mask > 0]
    target_masked = target[mask > 0]
    mae = np.abs(pred_masked - target_masked).mean()
    data_range = max(target_roi.max() - target_roi.min(), 1e-8)
    try:
        ssim_val = ssim(target_roi, pred_roi, data_range=data_range)
    except Exception:
        ssim_val = float('nan')
    return {'mae': float(mae), 'ssim': float(ssim_val)}


@torch.no_grad()
def sample_using_controlnet_mci(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context,
    atrophy_rate, ventricular_rate,
    device, scale_factor=1, average_over_n=1,
    num_inference_steps=50, verbose=False
):
    """Innovation 1 inference."""
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

    for t in scheduler.timesteps:
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


def evaluate_pair(
    autoencoder, diffusion, controlnet,
    row, scale_factor, load_latent, las_m=1
):
    """Evaluate a single pair."""
    starting_latent = load_latent(row['starting_latent']) * scale_factor

    context = torch.tensor([
        row['followup_age'], row['sex'], row['followup_diagnosis'],
        row['followup_cerebral_cortex'], row['followup_hippocampus'],
        row['followup_amygdala'], row['followup_cerebral_white_matter'],
        row['followup_lateral_ventricle']
    ])

    atrophy_rate = row.get('hippocampal_atrophy_rate', 0.0)
    vent_rate = row.get('ventricular_expansion_rate', 0.0)

    predicted = sample_using_controlnet_mci(
        autoencoder=autoencoder, diffusion=diffusion, controlnet=controlnet,
        starting_z=starting_latent.float(),
        starting_a=row['starting_age'],
        context=context.float(),
        atrophy_rate=atrophy_rate,
        ventricular_rate=vent_rate,
        device=DEVICE, scale_factor=scale_factor,
        average_over_n=las_m,
        num_inference_steps=50, verbose=False
    )

    predicted_np = predicted.numpy().clip(0, 1)

    # Load ground truth with same preprocessing as AE training pipeline
    load_gt = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])
    followup_tensor = load_gt(row['followup_image']).squeeze(0)
    followup = followup_tensor.numpy()

    # Ensure shapes match (pred from to_mni_space_1p5mm_trick may differ from resampled followup)
    min_shape = tuple(min(a, b) for a, b in zip(predicted_np.shape, followup.shape))
    predicted_np = predicted_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    followup = followup[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Overall metrics
    data_range = max(followup.max() - followup.min(), 1e-8)
    overall_ssim = ssim(followup, predicted_np, data_range=data_range)
    overall_psnr = psnr(followup, predicted_np, data_range=data_range)
    overall_mae = np.abs(followup - predicted_np).mean()

    result = {
        'subject_id': row['subject_id'],
        'overall_ssim': overall_ssim,
        'overall_psnr': overall_psnr,
        'overall_mae': overall_mae,
    }

    # ROI metrics
    segm_key = 'followup_segm' if 'followup_segm' in row else None
    if segm_key and pd.notna(row[segm_key]) and os.path.exists(str(row[segm_key])):
        segm_tensor = torch.from_numpy(
            nib.load(row[segm_key]).get_fdata().astype(np.float32)
        ).unsqueeze(0)
        resample_segm = transforms.Compose([
            transforms.Spacing(pixdim=const.RESOLUTION),
            transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        ])
        segm_resampled = resample_segm(segm_tensor).squeeze(0).numpy().round().astype(np.int32)

        # Align segm to same shape as predicted/followup
        segm_resampled = segm_resampled[:min_shape[0], :min_shape[1], :min_shape[2]]

        roi_mask = create_roi_mask(segm_resampled, MCI_ROI_LABELS)
        roi_metrics = compute_region_metrics(predicted_np, followup, roi_mask)
        result['roi_ssim'] = roi_metrics['ssim']
        result['roi_mae'] = roi_metrics['mae']

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Innovation 1 ControlNet')
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--max_pairs',   default=50,    type=int)
    parser.add_argument('--las_m',       default=1,     type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Innovation 1] Evaluation")
    print(f"  CSV: {args.dataset_csv}")
    print(f"  ControlNet: {args.cnet_ckpt}")
    print(f"  Max pairs: {args.max_pairs}")

    # Load models
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    controlnet = init_controlnet_mci(args.cnet_ckpt).to(DEVICE).eval()

    # Compute scale factor
    npz_reader = NumpyReader(npz_keys=['data'])
    load_latent = transforms.Compose([
        transforms.LoadImage(reader=npz_reader, image_only=True),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    test_df = dataset_df[dataset_df.split == 'test']
    if len(test_df) == 0:
        test_df = dataset_df[dataset_df.split == 'valid']

    # Scale factor from first training sample
    train_df = dataset_df[dataset_df.split == 'train']
    first_latent = load_latent(train_df.iloc[0]['followup_latent'])
    scale_factor = 1 / torch.std(first_latent)
    print(f"  Scale factor: {scale_factor:.4f}")

    # Select pairs to evaluate
    eval_pairs = test_df.head(args.max_pairs)
    print(f"  Evaluating {len(eval_pairs)} test pairs")

    results = []
    for idx, row in tqdm(eval_pairs.iterrows(), total=len(eval_pairs), desc="Evaluating"):
        try:
            res = evaluate_pair(
                autoencoder, diffusion, controlnet,
                row, scale_factor, load_latent, las_m=args.las_m
            )
            results.append(res)
        except Exception as e:
            print(f"  Error on {row['subject_id']}: {e}")

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'eval_innovation_1.csv')
    results_df.to_csv(csv_path, index=False)

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'innovation': '1_mci_dynamic_conditioning',
        'controlnet_ckpt': args.cnet_ckpt,
        'n_pairs': len(results),
        'overall_ssim': float(results_df['overall_ssim'].mean()),
        'overall_psnr': float(results_df['overall_psnr'].mean()),
        'overall_mae': float(results_df['overall_mae'].mean()),
    }
    if 'roi_ssim' in results_df.columns:
        summary['roi_ssim'] = float(results_df['roi_ssim'].mean())
        summary['roi_mae'] = float(results_df['roi_mae'].mean())

    json_path = os.path.join(args.output_dir, 'summary_innovation_1.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Innovation 1 Evaluation Results ({len(results)} pairs):")
    print(f"  Overall SSIM: {summary['overall_ssim']:.4f}")
    print(f"  Overall PSNR: {summary['overall_psnr']:.4f}")
    print(f"  Overall MAE:  {summary['overall_mae']:.6f}")
    if 'roi_ssim' in summary:
        print(f"  ROI SSIM:     {summary['roi_ssim']:.4f}")
        print(f"  ROI MAE:      {summary['roi_mae']:.6f}")
    print(f"{'='*50}")
