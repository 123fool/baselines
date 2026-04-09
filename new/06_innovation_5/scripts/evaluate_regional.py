"""
Regional Evaluation Script for Innovation 5.

Evaluates model performance with hippocampal-specific metrics:
  - Overall SSIM, PSNR, MAE
  - Hippocampal region SSIM, MAE
  - Amygdala region MAE
  
Uses paired CSV (B_mci.csv) format with starting_*/followup_* columns.
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

# Fix PyTorch 2.6+ weights_only=True default
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
from brlp import sample_using_controlnet_and_z
from region_weights import load_segmentation, create_roi_mask, HIPPOCAMPUS_LABELS, AMYGDALA_LABELS, MCI_ROI_LABELS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Compute metrics within a masked region.
    
    Args:
        pred: Predicted volume
        target: Ground truth volume
        mask: Binary region mask
        
    Returns:
        Dict with MAE, SSIM within region
    """
    if mask.sum() == 0:
        return {'mae': float('nan'), 'ssim': float('nan')}
    
    # Create bounding box around the mask for SSIM computation
    coords = np.where(mask > 0)
    slices = tuple(slice(c.min(), c.max() + 1) for c in coords)
    
    pred_roi = pred[slices]
    target_roi = target[slices]
    
    # MAE within mask
    pred_masked = pred[mask > 0]
    target_masked = target[mask > 0]
    mae = np.abs(pred_masked - target_masked).mean()
    
    # SSIM within bounding box
    data_range = max(target_roi.max() - target_roi.min(), 1e-8)
    try:
        ssim_val = ssim(target_roi, pred_roi, data_range=data_range)
    except Exception:
        ssim_val = float('nan')
    
    return {'mae': float(mae), 'ssim': float(ssim_val)}


def evaluate_pair(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context,
    followup_image_path, followup_segm_path,
    scale_factor, las_m=1,
):
    """
    Evaluate a single starting→followup prediction.
    
    Returns:
        Dict with overall and region-specific metrics
    """
    # Generate prediction
    pred_image = sample_using_controlnet_and_z(
        autoencoder=autoencoder,
        diffusion=diffusion,
        controlnet=controlnet,
        starting_z=starting_z,
        starting_a=starting_a,
        context=context,
        device=DEVICE,
        scale_factor=scale_factor,
        average_over_n=las_m,
        num_inference_steps=50,
        verbose=False
    )
    pred_np = pred_image.numpy().clip(0, 1)
    
    # Load ground truth with same preprocessing as AE training pipeline
    load_gt = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])
    target_tensor = load_gt(followup_image_path).squeeze(0)
    target_np = target_tensor.numpy()
    
    # Ensure shapes match (pred is from to_mni_space_1p5mm_trick → INPUT_SHAPE_1p5mm)
    min_shape = tuple(min(a, b) for a, b in zip(pred_np.shape, target_np.shape))
    pred_np = pred_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    target_np = target_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # Overall metrics
    data_range = max(target_np.max() - target_np.min(), 1e-8)
    overall_ssim = ssim(target_np, pred_np, data_range=data_range)
    overall_psnr = psnr(target_np, pred_np, data_range=data_range)
    overall_mae = np.abs(pred_np - target_np).mean()
    overall_mse = ((pred_np - target_np) ** 2).mean()
    
    metrics = {
        'overall_ssim': float(overall_ssim),
        'overall_psnr': float(overall_psnr),
        'overall_mae': float(overall_mae),
        'overall_mse': float(overall_mse),
    }
    
    # Region-specific metrics using segmentation
    if followup_segm_path and os.path.exists(followup_segm_path):
        segm = load_segmentation(followup_segm_path)
        
        # Resample segm to match prediction shape
        segm_tensor = torch.from_numpy(segm.astype(np.float32)).unsqueeze(0)
        resample_segm = transforms.Compose([
            transforms.Spacing(pixdim=const.RESOLUTION),
            transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        ])
        segm_resampled = resample_segm(segm_tensor).squeeze(0).numpy().round().astype(np.int32)
        segm_resampled = segm_resampled[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Hippocampus metrics
        hippo_mask = create_roi_mask(segm_resampled, HIPPOCAMPUS_LABELS)
        hippo_metrics = compute_region_metrics(pred_np, target_np, hippo_mask)
        metrics['hippocampus_mae'] = hippo_metrics['mae']
        metrics['hippocampus_ssim'] = hippo_metrics['ssim']
        
        # Amygdala metrics
        amyg_mask = create_roi_mask(segm_resampled, AMYGDALA_LABELS)
        amyg_metrics = compute_region_metrics(pred_np, target_np, amyg_mask)
        metrics['amygdala_mae'] = amyg_metrics['mae']
        metrics['amygdala_ssim'] = amyg_metrics['ssim']
        
        # Combined ROI metrics
        roi_mask = create_roi_mask(segm_resampled, MCI_ROI_LABELS)
        roi_metrics = compute_region_metrics(pred_np, target_np, roi_mask)
        metrics['roi_mae'] = roi_metrics['mae']
        metrics['roi_ssim'] = roi_metrics['ssim']
    
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Regional evaluation for Innovation 5')
    parser.add_argument('--dataset_csv', required=True, type=str, help='Paired CSV (B_mci.csv format)')
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   required=True, type=str)
    parser.add_argument('--max_samples', default=0,     type=int, help='Max pairs to evaluate (0=all)')
    parser.add_argument('--las_m',       default=1,     type=int, help='LAS parameter m')
    parser.add_argument('--model_name',  default='innovation_5', type=str, help='Label for this model')
    parser.add_argument('--raw_age',     action='store_true', help='Use raw age values from CSV (no /100 norm)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print(f"Loading models...")
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    controlnet = networks.init_controlnet(args.cnet_ckpt).to(DEVICE).eval()

    # Load paired dataset (B_mci.csv format)
    dataset_df = pd.read_csv(args.dataset_csv)
    test_df = dataset_df[dataset_df.split == 'test'].reset_index(drop=True)
    
    if args.max_samples > 0:
        test_df = test_df.head(args.max_samples)
    
    print(f"Evaluating {len(test_df)} test pairs with model: {args.model_name}")

    # Latent loading pipeline
    npz_reader = NumpyReader(npz_keys=['data'])
    load_latent = transforms.Compose([
        transforms.LoadImage(reader=npz_reader),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])

    # Compute scale_factor from first test sample
    first_latent_col = 'starting_latent'
    sample_z = load_latent(test_df.iloc[0][first_latent_col])
    scale_factor = 1 / torch.std(sample_z)
    print(f"Scale factor: {scale_factor}")

    # Evaluate each pair directly from CSV rows
    all_metrics = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Evaluating pairs'):
        # Load starting latent
        starting_latent_path = str(row['starting_latent'])
        followup_image_path = str(row['followup_image'])
        
        if not os.path.exists(starting_latent_path) or not os.path.exists(followup_image_path):
            print(f"  Skipping row {idx}: missing files")
            continue
        
        starting_z = load_latent(starting_latent_path) * scale_factor
        # BrLP convention: starting_age is age/100 (see REPR-DATA.md)
        # --raw_age skips normalization (for models trained with raw age)
        raw_age = float(row['starting_age'])
        if args.raw_age:
            starting_a = raw_age
        else:
            starting_a = raw_age / 100.0 if raw_age > 1.0 else raw_age
        
        # Build context vector — normalize age to BrLP convention (age/100)
        raw_followup_age = float(row['followup_age'])
        if args.raw_age:
            followup_age_norm = raw_followup_age
        else:
            followup_age_norm = raw_followup_age / 100.0 if raw_followup_age > 1.0 else raw_followup_age
        
        context = torch.tensor([
            followup_age_norm,
            float(row['sex']),
            float(row['followup_diagnosis']),
            float(row['followup_cerebral_cortex']),
            float(row['followup_hippocampus']),
            float(row['followup_amygdala']),
            float(row['followup_cerebral_white_matter']),
            float(row['followup_lateral_ventricle']),
        ])
        
        followup_segm_path = str(row.get('followup_segm', ''))
        if not os.path.exists(followup_segm_path):
            followup_segm_path = None
        
        try:
            metrics = evaluate_pair(
                autoencoder, diffusion, controlnet,
                starting_z, starting_a, context,
                followup_image_path, followup_segm_path,
                scale_factor, las_m=args.las_m,
            )
            metrics['subject_id'] = str(row.get('subject_id', ''))
            metrics['model'] = args.model_name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  Error evaluating row {idx}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(all_metrics)
    results_path = os.path.join(args.output_dir, f'eval_{args.model_name}.csv')
    results_df.to_csv(results_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {args.model_name}")
    print(f"{'='*60}")
    print(f"Pairs evaluated: {len(all_metrics)}")
    
    if len(all_metrics) > 0:
        summary = {}
        for col in ['overall_ssim', 'overall_psnr', 'overall_mae', 'overall_mse',
                     'hippocampus_mae', 'hippocampus_ssim', 'amygdala_mae', 'roi_mae', 'roi_ssim']:
            if col in results_df.columns:
                vals = results_df[col].dropna()
                if len(vals) > 0:
                    summary[col] = f"{vals.mean():.4f} ± {vals.std():.4f}"
                    print(f"  {col}: {summary[col]}")
        
        # Save summary
        summary_path = os.path.join(args.output_dir, f'summary_{args.model_name}.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'model': args.model_name,
                'timestamp': datetime.now().isoformat(),
                'num_pairs': len(all_metrics),
                'metrics': summary
            }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
