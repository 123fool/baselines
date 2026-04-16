"""
Large-scale BoN Weighted evaluation — full 50-pair MCI test set.

Compares LAS (m=3) vs BoN Weighted (N=8) on the complete B_mci.csv test set.
Outputs per-pair results + summary JSON.

Usage (on server):
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python run_bon_fullscale.py \
        > /home/wangchong/data/fwz/output/verification/fullscale_50/eval.log 2>&1

Server paths are hardcoded below.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from datetime import datetime
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, SCRIPT_DIR)

from brlp import const, utils
from brlp.networks import init_autoencoder, init_latent_diffusion, init_controlnet
from brlp.sampling import sample_using_controlnet_and_z
from sampling_bon_integrated import sample_bon_weighted, sample_bon_weighted_with_details

from monai import transforms

# ── Server paths ──
AEKL_CKPT = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT = "/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth"
CSV_PATH  = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
OUTPUT_DIR = "/home/wangchong/data/fwz/output/verification/fullscale_50"

# Synthseg ROI label map
HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]

# ── Config ──
N_CANDIDATES = 8
LAS_M = 3
MAX_PAIRS = 50     # full test set
SCALE_FACTOR = 1.0469
START_PAIR = 27    # resume from pair 27 (first 27 done before OOM)


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def compute_metrics(pred, gt):
    """Compute SSIM, PSNR, MAE between two 3D volumes."""
    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
    gt_np = gt.numpy() if isinstance(gt, torch.Tensor) else gt
    # Ensure shapes match by center-cropping to the smaller shape
    min_shape = tuple(min(a, b) for a, b in zip(pred_np.shape, gt_np.shape))
    pred_np = _center_crop(pred_np, min_shape)
    gt_np = _center_crop(gt_np, min_shape)
    pred_np = pred_np.clip(0, 1).astype(np.float64)
    gt_np = gt_np.clip(0, 1).astype(np.float64)
    return {
        'overall_ssim': float(ssim_fn(pred_np, gt_np, data_range=1.0)),
        'overall_psnr': float(psnr_fn(gt_np, pred_np, data_range=1.0)),
        'overall_mae': float(np.abs(pred_np - gt_np).mean()),
    }


def _center_crop(vol, target_shape):
    """Center crop a 3D volume to target_shape."""
    starts = [(s - t) // 2 for s, t in zip(vol.shape, target_shape)]
    slices = tuple(slice(max(0, s), max(0, s) + t) for s, t in zip(starts, target_shape))
    return vol[slices]


def compute_roi_metrics(pred, gt, segm_path):
    """Compute hippocampus and amygdala SSIM/MAE."""
    if not os.path.exists(segm_path):
        return {}
    segm = nib.load(segm_path).get_fdata().round().astype(np.int16)
    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
    gt_np = gt.numpy() if isinstance(gt, torch.Tensor) else gt

    # Crop to shared valid region
    min_shape = tuple(min(a, b, c) for a, b, c in zip(pred_np.shape, gt_np.shape, segm.shape))
    pred_np = pred_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    gt_np = gt_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    segm = segm[:min_shape[0], :min_shape[1], :min_shape[2]]

    results = {}
    for name, labels in [('hippocampus', HIPPOCAMPUS_LABELS), ('amygdala', AMYGDALA_LABELS)]:
        mask = np.isin(segm, labels)
        if mask.sum() < 50:
            continue
        p_roi = pred_np[mask].astype(np.float64)
        g_roi = gt_np[mask].astype(np.float64)
        results[f'{name}_mae'] = float(np.abs(p_roi - g_roi).mean())
        # Compute ROI SSIM on bounding box
        coords = np.where(mask)
        slices = tuple(slice(c.min(), c.max() + 1) for c in coords)
        p_bb = pred_np[slices].astype(np.float64)
        g_bb = gt_np[slices].astype(np.float64)
        if min(p_bb.shape) >= 7:
            results[f'{name}_ssim'] = float(ssim_fn(p_bb, g_bb, data_range=1.0))

    # Combined ROI
    roi_mask = np.isin(segm, HIPPOCAMPUS_LABELS + AMYGDALA_LABELS)
    if roi_mask.sum() >= 50:
        coords = np.where(roi_mask)
        slices = tuple(slice(c.min(), c.max() + 1) for c in coords)
        p_bb = pred_np[slices].astype(np.float64)
        g_bb = gt_np[slices].astype(np.float64)
        if min(p_bb.shape) >= 7:
            results['roi_ssim'] = float(ssim_fn(p_bb, g_bb, data_range=1.0))
        results['roi_mae'] = float(np.abs(pred_np[roi_mask] - gt_np[roi_mask]).mean())

    return results


def load_models(device):
    """Load all models."""
    ae = init_autoencoder(AEKL_CKPT).to(device)
    ae.eval()

    dm = init_latent_diffusion(DIFF_CKPT).to(device)
    dm.eval()

    cn = init_controlnet(CNET_CKPT).to(device)
    cn.eval()

    return ae, dm, cn


def prepare_pair(row, ae, device):
    """Prepare input data for one pair."""
    loader = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    # Starting (baseline) volume
    start_data = loader({'image_path': row['starting_image']})
    start_img = start_data['image'].unsqueeze(0).to(device)
    start_z = ae.encode(start_img)[0]
    start_z = transforms.DivisiblePad(k=4, mode='constant')(start_z.squeeze(0))

    # Follow-up (target) volume for GT comparison
    follow_data = loader({'image_path': row['followup_image']})
    follow_img = follow_data['image']

    # Build context vector matching CONDITIONING_VARIABLES order in const.py:
    # [age, sex, diagnosis, cerebral_cortex, hippocampus, amygdala, cerebral_white_matter, lateral_ventricle]
    # CSV values are already normalized to 0-1 range.
    context = torch.tensor([
        row['followup_age'],
        row['sex'],
        row['starting_diagnosis'],
        row['followup_cerebral_cortex'],
        row['followup_hippocampus'],
        row['followup_amygdala'],
        row['followup_cerebral_white_matter'],
        row['followup_lateral_ventricle'],
    ], dtype=torch.float32)

    return start_z, row['starting_age'], context, follow_img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{ts()} Device: {device}")
    print(f"{ts()} Loading models...")

    ae, dm, cn = load_models(device)
    print(f"{ts()} Models loaded.")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    n_pairs = min(len(df), MAX_PAIRS)
    print(f"{ts()} CSV has {len(df)} pairs, using {n_pairs}")

    # Results storage
    all_results = {'las': [], 'bon_weighted': []}
    pair_results = []

    for idx in range(START_PAIR, n_pairs):
        row = df.iloc[idx].to_dict()
        subj = row.get('subject_id', row.get('ptid', f'pair_{idx}'))
        t0 = time.time()

        with torch.no_grad():
            start_z, start_a, context, gt_img = prepare_pair(row, ae, device)
            # gt_img is [C,H,W,D] from MONAI loader — squeeze to [H,W,D]
            gt_np = gt_img.squeeze(0).numpy().clip(0, 1)

            # Segmentation path for ROI metrics
            segm_path = row.get('followup_segm', '')

            pair_data = {'pair_idx': idx, 'subject_id': subj}

            # ── Method 1: LAS (m=3) ──
            t1 = time.time()
            las_img = sample_using_controlnet_and_z(
                autoencoder=ae, diffusion=dm, controlnet=cn,
                starting_z=start_z.float(), starting_a=start_a,
                context=context.float(), device=device,
                scale_factor=SCALE_FACTOR, average_over_n=LAS_M,
                num_inference_steps=50, verbose=False,
            )
            las_time = time.time() - t1
            las_np = las_img.numpy().clip(0, 1)
            las_metrics = compute_metrics(las_img, torch.from_numpy(gt_np))
            las_roi = compute_roi_metrics(las_np, gt_np, segm_path)
            las_metrics.update(las_roi)
            las_metrics['time_sec'] = las_time
            pair_data['las'] = las_metrics
            all_results['las'].append(las_metrics)
            print(f"{ts()}   Pair {idx}: las SSIM={las_metrics['overall_ssim']:.4f} "
                  f"MAE={las_metrics['overall_mae']:.4f} time={las_time:.1f}s")

            # ── Method 2: BoN Weighted (N=8) ──
            t2 = time.time()
            bon_img = sample_bon_weighted(
                autoencoder=ae, diffusion=dm, controlnet=cn,
                starting_z=start_z.float(), starting_a=start_a,
                context=context.float(), device=device,
                scale_factor=SCALE_FACTOR, n_candidates=N_CANDIDATES,
                num_inference_steps=50, verbose=False,
            )
            bon_time = time.time() - t2
            bon_np = bon_img.numpy().clip(0, 1)
            bon_metrics = compute_metrics(bon_img, torch.from_numpy(gt_np))
            bon_roi = compute_roi_metrics(bon_np, gt_np, segm_path)
            bon_metrics.update(bon_roi)
            bon_metrics['time_sec'] = bon_time
            pair_data['bon_weighted'] = bon_metrics
            all_results['bon_weighted'].append(bon_metrics)
            print(f"{ts()}   Pair {idx}: bon_w SSIM={bon_metrics['overall_ssim']:.4f} "
                  f"MAE={bon_metrics['overall_mae']:.4f} time={bon_time:.1f}s")

            # Winner
            winner = 'bon_weighted' if bon_metrics['overall_ssim'] > las_metrics['overall_ssim'] else 'las'
            pair_data['winner_ssim'] = winner
            pair_results.append(pair_data)

        elapsed = time.time() - t0
        done_count = idx - START_PAIR + 1
        remaining = (elapsed / done_count) * (n_pairs - idx - 1) if done_count > 0 else 0
        print(f"{ts()} Pair {idx}/{n_pairs} done ({elapsed:.0f}s) — "
              f"ETA: {remaining/60:.1f}min — Winner: {winner}")

        # Memory cleanup to prevent OOM fragmentation
        del start_z, start_a, context, gt_img, gt_np
        del las_img, las_np, bon_img, bon_np
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{ts()} === SUMMARY ({n_pairs} pairs) ===\n")
    summary = {}
    for method, results in all_results.items():
        ssims = [r['overall_ssim'] for r in results]
        psnrs = [r['overall_psnr'] for r in results]
        maes = [r['overall_mae'] for r in results]
        times = [r['time_sec'] for r in results]

        roi_ssims = [r['roi_ssim'] for r in results if 'roi_ssim' in r]
        hipp_ssims = [r['hippocampus_ssim'] for r in results if 'hippocampus_ssim' in r]
        roi_maes = [r['roi_mae'] for r in results if 'roi_mae' in r]
        hipp_maes = [r['hippocampus_mae'] for r in results if 'hippocampus_mae' in r]

        s = {
            'overall_ssim': float(np.mean(ssims)),
            'overall_ssim_std': float(np.std(ssims)),
            'overall_psnr': float(np.mean(psnrs)),
            'overall_psnr_std': float(np.std(psnrs)),
            'overall_mae': float(np.mean(maes)),
            'overall_mae_std': float(np.std(maes)),
            'time_sec': float(np.mean(times)),
            'time_sec_std': float(np.std(times)),
        }
        if roi_ssims:
            s['roi_ssim'] = float(np.mean(roi_ssims))
            s['roi_ssim_std'] = float(np.std(roi_ssims))
        if hipp_ssims:
            s['hippocampus_ssim'] = float(np.mean(hipp_ssims))
            s['hippocampus_ssim_std'] = float(np.std(hipp_ssims))
        if roi_maes:
            s['roi_mae'] = float(np.mean(roi_maes))
        if hipp_maes:
            s['hippocampus_mae'] = float(np.mean(hipp_maes))

        summary[method] = s
        print(f"{method}:")
        print(f"  SSIM:  {s['overall_ssim']:.4f} ± {s['overall_ssim_std']:.4f}")
        print(f"  PSNR:  {s['overall_psnr']:.2f} ± {s['overall_psnr_std']:.2f}")
        print(f"  MAE:   {s['overall_mae']:.4f} ± {s['overall_mae_std']:.4f}")
        if 'roi_ssim' in s:
            print(f"  ROI SSIM: {s['roi_ssim']:.4f}")
        if 'hippocampus_ssim' in s:
            print(f"  Hipp SSIM: {s['hippocampus_ssim']:.4f}")
        print(f"  Time:  {s['time_sec']:.1f}s/pair")
        print()

    # Wins count
    bon_wins = sum(1 for p in pair_results if p['winner_ssim'] == 'bon_weighted')
    las_wins = n_pairs - bon_wins
    print(f"SSIM wins: bon_weighted {bon_wins}/{n_pairs}, las {las_wins}/{n_pairs}")

    # Paired t-test
    from scipy import stats
    las_ssims = [r['overall_ssim'] for r in all_results['las']]
    bon_ssims = [r['overall_ssim'] for r in all_results['bon_weighted']]
    t_stat, p_value = stats.ttest_rel(bon_ssims, las_ssims)
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")
    print(f"{'Statistically significant (p<0.05)' if p_value < 0.05 else 'NOT significant'}")

    # Save
    output = {
        'config': {
            'n_candidates': N_CANDIDATES,
            'las_m': LAS_M,
            'max_pairs': n_pairs,
            'scale_factor': SCALE_FACTOR,
            'csv': CSV_PATH,
            'cnet': CNET_CKPT,
            'timestamp': datetime.now().isoformat(),
        },
        'summary': summary,
        'per_pair': pair_results,
        'statistics': {
            'bon_wins': bon_wins,
            'las_wins': las_wins,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
        },
    }

    out_path = os.path.join(OUTPUT_DIR, 'summary_verification_eval.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n{ts()} Results saved to: {out_path}")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'comparison.csv')
    rows = []
    for p in pair_results:
        row = {'pair': p['pair_idx'], 'subject': p['subject_id'], 'winner': p['winner_ssim']}
        for method in ['las', 'bon_weighted']:
            if method in p:
                for k, v in p[method].items():
                    row[f'{method}_{k}'] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"{ts()} CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
