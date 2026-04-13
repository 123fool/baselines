"""
Priority 2: Residual Latent Prediction (RLP) — Evaluation Script.

Uses the RLP sampling pipeline (sample_using_controlnet_and_z_rlp) where
the diffusion denoises to delta_z and then reconstructs z_followup = z_starting + delta_z.

Key difference from evaluate_btr.py:
  1. Uses RLP sampling function instead of standard BrLP sampling
  2. Scale factor is computed from delta_z distribution
  3. Passes starting_z_unscaled for residual reconstruction
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
BRLP_SRC_ALT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'brlp_src'))
RLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
for p in [BRLP_SRC, BRLP_SRC_ALT, RLP_SRC]:
    if os.path.isdir(p):
        sys.path.insert(0, p)

from brlp import const, networks, utils
from sampling_rlp import sample_using_controlnet_and_z_rlp

# ROI labels from SynthSeg
HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]
MCI_ROI_LABELS = HIPPOCAMPUS_LABELS + AMYGDALA_LABELS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_roi_mask(segm_data, labels):
    mask = np.zeros_like(segm_data, dtype=bool)
    for label in labels:
        mask |= (segm_data.round() == label)
    return mask


def compute_region_metrics(pred, target, mask):
    if mask.sum() == 0:
        return {'mae': float('nan'), 'ssim': float('nan')}
    coords = np.where(mask > 0)
    slices = tuple(slice(c.min(), c.max() + 1) for c in coords)
    pred_roi = pred[slices]
    target_roi = target[slices]
    mae = np.abs(pred[mask > 0] - target[mask > 0]).mean()
    data_range = max(target_roi.max() - target_roi.min(), 1e-8)
    try:
        ssim_val = ssim(target_roi, pred_roi, data_range=data_range)
    except Exception:
        ssim_val = float('nan')
    return {'mae': float(mae), 'ssim': float(ssim_val)}


def compute_residual_scale_factor_from_df(df, load_latent, n_samples=50):
    """Compute scale_factor = 1/std(delta_z) from training data."""
    deltas = []
    n = min(n_samples, len(df))
    for idx in range(n):
        row = df.iloc[idx]
        starting = load_latent(row['starting_latent'])
        followup = load_latent(row['followup_latent'])
        delta = followup - starting
        deltas.append(delta)
    deltas = torch.stack(deltas)
    sf = 1.0 / torch.std(deltas)
    return sf


def evaluate_pair_rlp(autoencoder, diffusion, controlnet,
                      row, scale_factor, load_latent, las_m=1):
    """Evaluate one pair using RLP sampling pipeline."""
    starting_latent_raw = load_latent(row['starting_latent'])
    starting_latent_scaled = starting_latent_raw * scale_factor

    context = torch.tensor([
        row['followup_age'], row['sex'], row['followup_diagnosis'],
        row['followup_cerebral_cortex'], row['followup_hippocampus'],
        row['followup_amygdala'], row['followup_cerebral_white_matter'],
        row['followup_lateral_ventricle']
    ])

    predicted = sample_using_controlnet_and_z_rlp(
        autoencoder=autoencoder, diffusion=diffusion, controlnet=controlnet,
        starting_z=starting_latent_scaled.float(),
        starting_z_unscaled=starting_latent_raw.float(),
        starting_a=row['starting_age'],
        context=context.float(),
        device=DEVICE, scale_factor=scale_factor,
        average_over_n=las_m,
        num_inference_steps=50, verbose=False
    )
    predicted_np = predicted.numpy().clip(0, 1)

    load_gt = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])
    followup = load_gt(row['followup_image']).squeeze(0).numpy()

    min_shape = tuple(min(a, b) for a, b in
                      zip(predicted_np.shape, followup.shape))
    predicted_np = predicted_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    followup = followup[:min_shape[0], :min_shape[1], :min_shape[2]]

    data_range = max(followup.max() - followup.min(), 1e-8)
    overall_ssim = ssim(followup, predicted_np, data_range=data_range)
    overall_psnr = psnr(followup, predicted_np, data_range=data_range)
    overall_mae = np.abs(followup - predicted_np).mean()
    overall_mse = ((followup - predicted_np) ** 2).mean()

    result = {
        'subject_id': row['subject_id'],
        'overall_ssim': overall_ssim,
        'overall_psnr': overall_psnr,
        'overall_mae': overall_mae,
        'overall_mse': overall_mse,
    }

    segm_key = 'followup_segm' if 'followup_segm' in row else None
    if segm_key and pd.notna(row[segm_key]) and os.path.exists(str(row[segm_key])):
        segm_tensor = torch.from_numpy(
            nib.load(row[segm_key]).get_fdata().astype(np.float32)
        ).unsqueeze(0)
        resample_segm = transforms.Compose([
            transforms.Spacing(pixdim=const.RESOLUTION),
            transforms.ResizeWithPadOrCrop(
                spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        ])
        segm = resample_segm(segm_tensor).squeeze(0).numpy().round().astype(np.int32)
        segm = segm[:min_shape[0], :min_shape[1], :min_shape[2]]

        hipp_mask = create_roi_mask(segm, HIPPOCAMPUS_LABELS)
        amyg_mask = create_roi_mask(segm, AMYGDALA_LABELS)
        roi_mask = create_roi_mask(segm, MCI_ROI_LABELS)

        hipp = compute_region_metrics(predicted_np, followup, hipp_mask)
        amyg = compute_region_metrics(predicted_np, followup, amyg_mask)
        roi = compute_region_metrics(predicted_np, followup, roi_mask)

        result.update({
            'hippocampus_mae': hipp['mae'], 'hippocampus_ssim': hipp['ssim'],
            'amygdala_mae': amyg['mae'],
            'roi_mae': roi['mae'], 'roi_ssim': roi['ssim'],
        })

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RLP ControlNet')
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--max_pairs',   default=50,    type=int)
    parser.add_argument('--model_name',  default='priority_2_rlp', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Priority 2] RLP Evaluation")
    print(f"  CSV: {args.dataset_csv}")
    print(f"  ControlNet: {args.cnet_ckpt}")
    print(f"  AE: {args.aekl_ckpt}")
    print(f"  Max pairs: {args.max_pairs}")

    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    controlnet = networks.init_controlnet(args.cnet_ckpt).to(DEVICE).eval()

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

    train_df = dataset_df[dataset_df.split == 'train']

    # RLP: compute residual scale factor
    scale_factor = compute_residual_scale_factor_from_df(train_df, load_latent,
                                                         n_samples=min(50, len(train_df)))
    print(f"  Scale factor (residual): {scale_factor:.4f}")

    eval_pairs = test_df.head(args.max_pairs)
    print(f"  Evaluating {len(eval_pairs)} test pairs")

    results = []
    for idx, row in tqdm(eval_pairs.iterrows(), total=len(eval_pairs),
                         desc="Evaluating pairs"):
        try:
            res = evaluate_pair_rlp(
                autoencoder, diffusion, controlnet,
                row, scale_factor, load_latent)
            results.append(res)
        except Exception as e:
            print(f"  Error on {row['subject_id']}: {e}")

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f'eval_{args.model_name}.csv')
    results_df.to_csv(csv_path, index=False)

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'method': 'priority_2_residual_latent_prediction',
        'controlnet_ckpt': args.cnet_ckpt,
        'n_pairs': len(results),
    }
    for col in results_df.columns:
        if col != 'subject_id':
            summary[col] = f"{results_df[col].mean():.6f} ± {results_df[col].std():.6f}"

    json_path = os.path.join(args.output_dir, f'summary_{args.model_name}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation Results: {args.model_name}")
    print(f"{'='*60}")
    print(f"Pairs evaluated: {len(results)}")
    for col in ['overall_ssim', 'overall_psnr', 'overall_mae', 'overall_mse',
                'hippocampus_mae', 'hippocampus_ssim', 'amygdala_mae',
                'roi_mae', 'roi_ssim']:
        if col in results_df.columns:
            print(f"  {col}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")
    print(f"{'='*60}")

    print(f"\nResults saved to {csv_path}")
