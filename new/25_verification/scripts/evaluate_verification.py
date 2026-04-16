"""
Verification Mechanism Evaluation — Compare All Methods.

Tests each verification scheme on data WITH ground truth to validate that
the selection mechanism picks better samples. Methods compared:

  1. LAS (original):    Blind average of m=3 latents (BrLP baseline)
  2. Single random:     Single random sample (no averaging, no selection)
  3. Best-of-N (best1): Generate N, pick best by composite score
  4. Best-of-N (topk):  Generate N, average top-K by composite score
  5. Best-of-N (weighted): Weighted average using composite scores
  6. Round-Trip BoN:    Generate N, pick best by round-trip consistency

For each method, we compute:
  - Overall SSIM, PSNR, MAE (vs ground-truth follow-up)
  - ROI SSIM, MAE (hippocampus, amygdala)
  - Wall-clock time per sample

Usage:
  python evaluate_verification.py \
    --dataset_csv /path/to/dataset.csv \
    --aekl_ckpt /path/to/autoencoder.pth \
    --diff_ckpt /path/to/diffusion.pth \
    --cnet_ckpt /path/to/controlnet.pth \
    --output_dir /path/to/output \
    --max_pairs 20 \
    --n_candidates 8
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
from monai import transforms
from monai.data.image_reader import NumpyReader
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
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, SCRIPT_DIR)

from brlp import const, networks, utils
from brlp import sample_using_controlnet_and_z
from sampling_bon import sample_best_of_n, sample_best_of_n_batched
from sampling_roundtrip import round_trip_best_of_n

# ROI labels
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


def compute_gt_metrics(predicted_np, followup, segm=None):
    """Compute all metrics against ground truth."""
    min_shape = tuple(min(a, b) for a, b in
                      zip(predicted_np.shape, followup.shape))
    pred = predicted_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    gt = followup[:min_shape[0], :min_shape[1], :min_shape[2]]

    data_range = max(gt.max() - gt.min(), 1e-8)
    result = {
        'overall_ssim': float(ssim(gt, pred, data_range=data_range)),
        'overall_psnr': float(psnr(gt, pred, data_range=data_range)),
        'overall_mae': float(np.abs(gt - pred).mean()),
    }

    if segm is not None:
        segm = segm[:min_shape[0], :min_shape[1], :min_shape[2]]
        hipp_mask = create_roi_mask(segm, HIPPOCAMPUS_LABELS)
        amyg_mask = create_roi_mask(segm, AMYGDALA_LABELS)
        roi_mask = create_roi_mask(segm, MCI_ROI_LABELS)
        hipp = compute_region_metrics(pred, gt, hipp_mask)
        amyg = compute_region_metrics(pred, gt, amyg_mask)
        roi = compute_region_metrics(pred, gt, roi_mask)
        result.update({
            'hippocampus_mae': hipp['mae'], 'hippocampus_ssim': hipp['ssim'],
            'amygdala_mae': amyg['mae'],
            'roi_mae': roi['mae'], 'roi_ssim': roi['ssim'],
        })

    return result


def load_ground_truth(row):
    """Load and preprocess the ground truth follow-up image."""
    load_gt = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])
    return load_gt(row['followup_image']).squeeze(0).numpy()


def load_segmentation(row, min_shape):
    """Load and preprocess segmentation if available."""
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
        return segm
    return None


def build_context(row):
    """Build conditioning context from CSV row."""
    return torch.tensor([
        row['followup_age'], row['sex'], row['followup_diagnosis'],
        row['followup_cerebral_cortex'], row['followup_hippocampus'],
        row['followup_amygdala'], row['followup_cerebral_white_matter'],
        row['followup_lateral_ventricle'],
    ])


def build_reverse_context(row):
    """Build reverse context (for round-trip: predicting back to starting time)."""
    return torch.tensor([
        row['starting_age'], row['sex'], row['starting_diagnosis'],
        row['starting_cerebral_cortex'], row['starting_hippocampus'],
        row['starting_amygdala'], row['starting_cerebral_white_matter'],
        row['starting_lateral_ventricle'],
    ])


def evaluate_methods(
    autoencoder, diffusion, controlnet,
    row, scale_factor, load_latent, 
    n_candidates=8, las_m=3,
    methods=None,
):
    """Evaluate all verification methods for one test pair.

    Returns dict mapping method_name -> {metrics, time_sec, ...}
    """
    if methods is None:
        methods = ['las', 'single', 'bon_best1', 'bon_topk', 'bon_weighted',
                   'roundtrip_bon']

    starting_latent = load_latent(row['starting_latent']) * scale_factor
    context = build_context(row)
    followup = load_ground_truth(row)
    min_shape = tuple(min(a, b) for a, b in
                      zip(const.INPUT_SHAPE_1p5mm, followup.shape))
    segm = load_segmentation(row, min_shape)

    results = {}

    for method in methods:
        t0 = time.time()

        if method == 'las':
            # Original LAS: blind average of m latents
            predicted = sample_using_controlnet_and_z(
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet,
                starting_z=starting_latent.float(),
                starting_a=row['starting_age'],
                context=context.float(),
                device=DEVICE, scale_factor=scale_factor,
                average_over_n=las_m,
                num_inference_steps=50, verbose=False,
            )
            predicted_np = predicted.numpy().clip(0, 1)

        elif method == 'single':
            # Single random sample
            predicted = sample_using_controlnet_and_z(
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet,
                starting_z=starting_latent.float(),
                starting_a=row['starting_age'],
                context=context.float(),
                device=DEVICE, scale_factor=scale_factor,
                average_over_n=1,
                num_inference_steps=50, verbose=False,
            )
            predicted_np = predicted.numpy().clip(0, 1)

        elif method.startswith('bon_'):
            sel = method.replace('bon_', '')
            # Map short names to selection strategy names
            sel_map = {'best1': 'best1', 'topk': 'topk_avg', 'weighted': 'weighted'}
            sel = sel_map.get(sel, sel)
            bon_result = sample_best_of_n_batched(
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet,
                starting_z=starting_latent.float(),
                starting_a=row['starting_age'],
                context=context.float(),
                device=DEVICE, scale_factor=scale_factor,
                n_candidates=n_candidates,
                batch_size=min(4, n_candidates),
                selection=sel,
                num_inference_steps=50,
                verbose=False,
            )
            predicted_np = bon_result['image']
            results[method + '_scores'] = bon_result['composite_values']
            results[method + '_selected'] = bon_result['selected_idx']

        elif method == 'roundtrip_bon':
            reverse_ctx = build_reverse_context(row)
            rt_result = round_trip_best_of_n(
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet,
                starting_z=starting_latent.float(),
                starting_a=row['starting_age'],
                forward_context=context.float(),
                reverse_context=reverse_ctx.float(),
                reverse_age=int(row['followup_age']),
                device=DEVICE, scale_factor=scale_factor,
                n_candidates=min(n_candidates, 5),  # fewer due to 2x cost
                num_inference_steps=50,
                verbose=False,
            )
            predicted_np = rt_result['image']
            results[method + '_roundtrip_ssim'] = rt_result['roundtrip_ssim']

        else:
            continue

        elapsed = time.time() - t0

        # Compute GT metrics
        metrics = compute_gt_metrics(predicted_np, followup, segm)
        metrics['time_sec'] = elapsed
        results[method] = metrics

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate verification mechanisms against ground truth')
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--max_pairs', default=20, type=int)
    parser.add_argument('--n_candidates', default=8, type=int,
                        help='Number of candidates for Best-of-N')
    parser.add_argument('--las_m', default=3, type=int,
                        help='LAS m parameter for baseline comparison')
    parser.add_argument('--methods', default=None, type=str,
                        help='Comma-separated method names (default: all)')
    parser.add_argument('--model_name', default='verification_eval', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'eval_verification.log')

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',')]

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_file, 'a') as f:
            f.write(line + '\n')

    log(f"=== Verification Mechanism Evaluation ===")
    log(f"CSV: {args.dataset_csv}")
    log(f"ControlNet: {args.cnet_ckpt}")
    log(f"N candidates: {args.n_candidates}")
    log(f"LAS m: {args.las_m}")
    log(f"Methods: {methods or 'all'}")

    # Load models
    log("Loading models...")
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    controlnet = networks.init_controlnet(args.cnet_ckpt).to(DEVICE).eval()

    npz_reader = NumpyReader(npz_keys=['data'])
    load_latent = transforms.Compose([
        transforms.LoadImage(reader=npz_reader, image_only=True),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])

    # Load dataset
    dataset_df = pd.read_csv(args.dataset_csv)
    test_df = dataset_df[dataset_df.split == 'test']
    if len(test_df) == 0:
        test_df = dataset_df[dataset_df.split == 'valid']

    train_df = dataset_df[dataset_df.split == 'train']
    first_latent = load_latent(train_df.iloc[0]['followup_latent'])
    scale_factor = 1 / torch.std(first_latent)
    log(f"Scale factor: {scale_factor:.4f}")

    eval_pairs = test_df.head(args.max_pairs)
    log(f"Evaluating {len(eval_pairs)} test pairs")

    # Run evaluation
    all_results = []
    for idx, (_, row) in enumerate(tqdm(eval_pairs.iterrows(),
                                         total=len(eval_pairs),
                                         desc="Evaluating")):
        try:
            pair_results = evaluate_methods(
                autoencoder, diffusion, controlnet,
                row, scale_factor, load_latent,
                n_candidates=args.n_candidates,
                las_m=args.las_m,
                methods=methods,
            )
            pair_results['subject_id'] = row.get('subject_id', f'pair_{idx}')
            all_results.append(pair_results)

            # Log progress
            for method_key, metrics in pair_results.items():
                if isinstance(metrics, dict) and 'overall_ssim' in metrics:
                    log(f"  Pair {idx}: {method_key} SSIM={metrics['overall_ssim']:.4f} "
                        f"MAE={metrics['overall_mae']:.4f} "
                        f"time={metrics.get('time_sec', 0):.1f}s")

        except Exception as e:
            log(f"  Pair {idx} ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results
    log("\n=== SUMMARY ===")
    method_keys = [k for k in all_results[0].keys()
                   if isinstance(all_results[0].get(k), dict)
                   and 'overall_ssim' in all_results[0][k]]

    summary = {}
    for method_key in method_keys:
        vals = [r[method_key] for r in all_results if method_key in r]
        if not vals:
            continue
        avg = {}
        for metric in ['overall_ssim', 'overall_psnr', 'overall_mae',
                       'hippocampus_ssim', 'hippocampus_mae',
                       'roi_ssim', 'roi_mae', 'time_sec']:
            metric_vals = [v[metric] for v in vals
                          if metric in v and v[metric] is not None
                          and not np.isnan(v[metric])]
            if metric_vals:
                avg[metric] = float(np.mean(metric_vals))
                avg[metric + '_std'] = float(np.std(metric_vals))
        summary[method_key] = avg

        log(f"\n{method_key}:")
        log(f"  SSIM:  {avg.get('overall_ssim', 0):.4f} ± "
            f"{avg.get('overall_ssim_std', 0):.4f}")
        log(f"  PSNR:  {avg.get('overall_psnr', 0):.2f} ± "
            f"{avg.get('overall_psnr_std', 0):.2f}")
        log(f"  MAE:   {avg.get('overall_mae', 0):.4f} ± "
            f"{avg.get('overall_mae_std', 0):.4f}")
        if 'roi_ssim' in avg:
            log(f"  ROI SSIM: {avg.get('roi_ssim', 0):.4f}")
            log(f"  ROI MAE:  {avg.get('roi_mae', 0):.4f}")
        log(f"  Time:  {avg.get('time_sec', 0):.1f}s/pair")

    # Save results
    output = {
        'config': {
            'dataset_csv': args.dataset_csv,
            'cnet_ckpt': args.cnet_ckpt,
            'n_candidates': args.n_candidates,
            'las_m': args.las_m,
            'max_pairs': args.max_pairs,
            'timestamp': datetime.now().isoformat(),
        },
        'summary': summary,
        'per_pair': [{
            'subject_id': r.get('subject_id'),
            **{k: v for k, v in r.items()
               if isinstance(v, dict) and 'overall_ssim' in v}
        } for r in all_results],
    }

    output_path = os.path.join(args.output_dir,
                               f'summary_{args.model_name}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to: {output_path}")

    # Also save a comparison CSV
    rows = []
    for method_key, avg in summary.items():
        row_data = {'method': method_key}
        row_data.update(avg)
        rows.append(row_data)
    if rows:
        comp_df = pd.DataFrame(rows)
        comp_path = os.path.join(args.output_dir, 'comparison.csv')
        comp_df.to_csv(comp_path, index=False)
        log(f"Comparison CSV: {comp_path}")


if __name__ == '__main__':
    main()
