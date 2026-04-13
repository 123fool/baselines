"""
去辅助模型端到端验证
===================
对比 4 种推理时 context 来源，验证去掉 Leaspy 后 ControlNet 图像质量不下降。

Methods:
  GT     — 真实 followup 体积（oracle 上界）
  TPN    — TPN v3 预测的 followup 体积（学习替代 Leaspy）
  Skip   — 直接使用 starting 体积（最简基线，假设无变化）
  Linear — 线性外推（基于年龄差异方向外推体积变化趋势）

Usage:
    python evaluate_no_aux.py \
        --dataset_csv  /path/to/B_mci.csv \
        --aekl_ckpt    /path/to/autoencoder.pth \
        --diff_ckpt    /path/to/latentdiffusion.pth \
        --cnet_ckpt    /path/to/cnet-btr-ep-1.pth \
        --tpn_ckpt     /path/to/tpn_best.pth \
        --output_dir   /path/to/output \
        --max_pairs    50

日志格式 (用于 dashboard 解析):
    [NO_AUX] Method=GT  | Pair 1/50 | SSIM=0.9312
    [NO_AUX] Method=TPN | Pair 1/50 | SSIM=0.9298
    [NO_AUX] === SUMMARY === GT: 0.9282 | TPN: 0.9275 | Skip: 0.9180 | Linear: 0.9200
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from monai.data.image_reader import NumpyReader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Fix torch.load for older checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BrLP src: reuse from existing tpn code directory
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'tpn', 'brlp_src'))
sys.path.insert(0, BRLP_SRC)
# TPN model src
TPN_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'tpn', 'src'))
sys.path.insert(0, TPN_SRC)

from brlp import const, networks
from brlp import sample_using_controlnet_and_z

VOLUME_REGIONS = const.CONDITIONING_REGIONS  # [cortex, hippo, amyg, wm, vent]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ROI labels from SynthSeg
HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]
MCI_ROI_LABELS = HIPPOCAMPUS_LABELS + AMYGDALA_LABELS


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


# ---- Context methods ----

def context_gt(row):
    """Oracle: use ground truth followup volumes."""
    return torch.tensor([
        row['followup_age'], row['sex'], row['followup_diagnosis'],
        row['followup_cerebral_cortex'], row['followup_hippocampus'],
        row['followup_amygdala'], row['followup_cerebral_white_matter'],
        row['followup_lateral_ventricle']
    ], dtype=torch.float32)


def context_skip(row):
    """Simplest: use starting volumes as-is (assume no change)."""
    return torch.tensor([
        row['followup_age'], row['sex'], row['followup_diagnosis'],
        row['starting_cerebral_cortex'], row['starting_hippocampus'],
        row['starting_amygdala'], row['starting_cerebral_white_matter'],
        row['starting_lateral_ventricle']
    ], dtype=torch.float32)


def context_linear(row, train_slopes=None):
    """Linear extrapolation from starting volumes using age gap."""
    age_gap = row['followup_age'] - row['starting_age']
    vols = []
    for region in VOLUME_REGIONS:
        start_val = row[f'starting_{region}']
        if train_slopes is not None and region in train_slopes:
            slope = train_slopes[region]
        else:
            slope = 0.0  # fallback: no change
        predicted = start_val + slope * age_gap
        vols.append(max(0.0, min(1.0, predicted)))
    return torch.tensor([
        row['followup_age'], row['sex'], row['followup_diagnosis'],
        *vols
    ], dtype=torch.float32)


def context_tpn(row, tpn_model, device='cpu'):
    """Use TPN to predict followup volumes."""
    from tpn import TemporalProgressionNetwork

    current_age = row['starting_age']
    target_age = row['followup_age']
    sex = row['sex']
    # Use starting_diagnosis for TPN input
    diag_col = 'starting_diagnosis' if 'starting_diagnosis' in row.index else 'starting_last_diagnosis'
    diag = row[diag_col] if diag_col in row.index else 0.5

    # Normalize sex/diagnosis same way as TPN training
    sex_norm = (sex - const.SEX_MIN) / const.SEX_DELTA
    diag_norm = diag if diag <= 1 else (diag - const.DIA_MIN) / const.DIA_DELTA

    current_vols = [row[f'starting_{r}'] for r in VOLUME_REGIONS]

    # Build TPN v3 input (14-dim)
    age_gap = target_age - current_age
    age_ratio = age_gap / (current_age + 1e-8)
    vol_mean = np.mean(current_vols)
    vol_std = np.std(current_vols)
    age_gap_sq = age_gap ** 2

    x = torch.tensor([
        current_age, target_age, sex_norm, diag_norm,
        *current_vols,
        age_gap, age_ratio, vol_mean, vol_std, age_gap_sq
    ], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_vols = tpn_model(x).cpu().numpy().flatten()

    return torch.tensor([
        row['followup_age'], row['sex'], row['followup_diagnosis'],
        *predicted_vols.tolist()
    ], dtype=torch.float32)


def compute_linear_slopes(train_df):
    """Compute average volume change slopes from training data."""
    slopes = {}
    for region in VOLUME_REGIONS:
        start_col = f'starting_{region}'
        follow_col = f'followup_{region}'
        if start_col in train_df.columns and follow_col in train_df.columns:
            valid = train_df.dropna(subset=[start_col, follow_col, 'starting_age', 'followup_age'])
            age_gaps = valid['followup_age'] - valid['starting_age']
            vol_changes = valid[follow_col] - valid[start_col]
            # Avoid division by zero
            mask = age_gaps.abs() > 1e-6
            if mask.sum() > 0:
                slopes[region] = float((vol_changes[mask] / age_gaps[mask]).mean())
            else:
                slopes[region] = 0.0
        else:
            slopes[region] = 0.0
    return slopes


def evaluate_pair_multi(autoencoder, diffusion, controlnet,
                        row, scale_factor, load_latent, load_gt_transform,
                        tpn_model=None, train_slopes=None, methods=None):
    """Evaluate one pair with multiple context methods."""
    if methods is None:
        methods = ['GT', 'TPN', 'Skip', 'Linear']

    # Load starting latent (shared across all methods)
    starting_latent = load_latent(row['starting_latent']) * scale_factor

    # Load GT followup image (shared across all methods)
    followup = load_gt_transform(row['followup_image']).squeeze(0).numpy()

    # Load segmentation if available
    segm = None
    segm_key = 'followup_segm' if 'followup_segm' in row.index else None
    if segm_key and pd.notna(row[segm_key]) and os.path.exists(str(row[segm_key])):
        import nibabel as nib
        segm_tensor = torch.from_numpy(
            nib.load(row[segm_key]).get_fdata().astype(np.float32)
        ).unsqueeze(0)
        resample_segm = transforms.Compose([
            transforms.Spacing(pixdim=const.RESOLUTION),
            transforms.ResizeWithPadOrCrop(
                spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        ])
        segm = resample_segm(segm_tensor).squeeze(0).numpy().round().astype(np.int32)

    results = {}
    for method in methods:
        # Build context vector
        if method == 'GT':
            ctx = context_gt(row)
        elif method == 'TPN':
            ctx = context_tpn(row, tpn_model, device=DEVICE)
        elif method == 'Skip':
            ctx = context_skip(row)
        elif method == 'Linear':
            ctx = context_linear(row, train_slopes)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Run ControlNet inference
        predicted = sample_using_controlnet_and_z(
            autoencoder=autoencoder, diffusion=diffusion, controlnet=controlnet,
            starting_z=starting_latent.float(),
            starting_a=row['starting_age'],
            context=ctx.float(),
            device=DEVICE, scale_factor=scale_factor,
            average_over_n=1,  # LAS m=1 for speed
            num_inference_steps=50, verbose=False
        )
        predicted_np = predicted.numpy().clip(0, 1)

        # Align shapes
        min_shape = tuple(min(a, b) for a, b in
                          zip(predicted_np.shape, followup.shape))
        pred_crop = predicted_np[:min_shape[0], :min_shape[1], :min_shape[2]]
        fu_crop = followup[:min_shape[0], :min_shape[1], :min_shape[2]]

        # Compute metrics
        data_range = max(fu_crop.max() - fu_crop.min(), 1e-8)
        overall_ssim = ssim(fu_crop, pred_crop, data_range=data_range)
        overall_psnr = psnr(fu_crop, pred_crop, data_range=data_range)
        overall_mae = float(np.abs(fu_crop - pred_crop).mean())

        res = {
            'subject_id': row['subject_id'],
            'method': method,
            'overall_ssim': overall_ssim,
            'overall_psnr': overall_psnr,
            'overall_mae': overall_mae,
            'context_volumes': ctx[3:].tolist(),  # the 5 volume values used
        }

        # ROI metrics
        if segm is not None:
            segm_crop = segm[:min_shape[0], :min_shape[1], :min_shape[2]]
            hipp_mask = create_roi_mask(segm_crop, HIPPOCAMPUS_LABELS)
            amyg_mask = create_roi_mask(segm_crop, AMYGDALA_LABELS)
            roi_mask = create_roi_mask(segm_crop, MCI_ROI_LABELS)
            hipp = compute_region_metrics(pred_crop, fu_crop, hipp_mask)
            amyg = compute_region_metrics(pred_crop, fu_crop, amyg_mask)
            roi = compute_region_metrics(pred_crop, fu_crop, roi_mask)
            res.update({
                'hippocampus_ssim': hipp['ssim'],
                'amygdala_mae': amyg['mae'],
                'roi_ssim': roi['ssim'],
            })

        results[method] = res
        print(f"[NO_AUX] Method={method:<6} | SSIM={overall_ssim:.4f} | PSNR={overall_psnr:.2f}")

    return results


def load_tpn_model(ckpt_path, device='cpu'):
    """Load TPN v3 model."""
    from tpn import TemporalProgressionNetwork
    model = TemporalProgressionNetwork(
        in_dim=14, hidden_dim=128, out_dim=5, n_layers=3, dropout=0.0
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='去辅助模型端到端验证')
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str)
    parser.add_argument('--tpn_ckpt', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--max_pairs', default=50, type=int)
    parser.add_argument('--methods', default='GT,TPN,Skip,Linear', type=str,
                        help='Comma-separated methods to test')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(',')]
    log_path = os.path.join(args.output_dir, 'eval_no_aux.log')

    print(f"[NO_AUX] 去辅助模型端到端验证")
    print(f"  CSV: {args.dataset_csv}")
    print(f"  ControlNet: {args.cnet_ckpt}")
    print(f"  TPN: {args.tpn_ckpt}")
    print(f"  Methods: {methods}")
    print(f"  Max pairs: {args.max_pairs}")

    # Load models
    print("[NO_AUX] Loading models...")
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    controlnet = networks.init_controlnet(args.cnet_ckpt).to(DEVICE).eval()

    tpn_model = None
    if 'TPN' in methods:
        tpn_model = load_tpn_model(args.tpn_ckpt, device=DEVICE)
        print(f"[NO_AUX] TPN loaded: {args.tpn_ckpt}")

    # Setup data loading
    npz_reader = NumpyReader(npz_keys=['data'])
    load_latent = transforms.Compose([
        transforms.LoadImage(reader=npz_reader, image_only=True),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])
    load_gt_transform = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    # Load CSV
    dataset_df = pd.read_csv(args.dataset_csv)
    test_df = dataset_df[dataset_df.split == 'test']
    if len(test_df) == 0:
        test_df = dataset_df[dataset_df.split == 'valid']
    train_df = dataset_df[dataset_df.split == 'train']

    # Scale factor
    first_latent = load_latent(train_df.iloc[0]['followup_latent'])
    scale_factor = 1 / torch.std(first_latent)
    print(f"[NO_AUX] Scale factor: {scale_factor:.4f}")

    # Compute linear slopes from training data
    train_slopes = compute_linear_slopes(train_df) if 'Linear' in methods else None
    if train_slopes:
        print(f"[NO_AUX] Linear slopes: {train_slopes}")

    # Evaluate
    eval_pairs = test_df.head(args.max_pairs)
    n_pairs = len(eval_pairs)
    print(f"[NO_AUX] Evaluating {n_pairs} pairs × {len(methods)} methods = {n_pairs * len(methods)} total inferences")

    all_results = []
    for idx, (_, row) in enumerate(tqdm(eval_pairs.iterrows(), total=n_pairs,
                                         desc="Evaluating")):
        try:
            pair_results = evaluate_pair_multi(
                autoencoder, diffusion, controlnet,
                row, scale_factor, load_latent, load_gt_transform,
                tpn_model=tpn_model, train_slopes=train_slopes,
                methods=methods
            )
            for method, res in pair_results.items():
                all_results.append(res)

            # Progress log (for dashboard parsing)
            gt_ssim = pair_results.get('GT', {}).get('overall_ssim', 0)
            tpn_ssim = pair_results.get('TPN', {}).get('overall_ssim', 0)
            skip_ssim = pair_results.get('Skip', {}).get('overall_ssim', 0)
            progress_line = (f"[NO_AUX] Pair {idx+1}/{n_pairs} | "
                           f"GT={gt_ssim:.4f} TPN={tpn_ssim:.4f} Skip={skip_ssim:.4f}")
            print(progress_line)

            # Append to log file
            with open(log_path, 'a') as f:
                f.write(progress_line + '\n')

        except Exception as e:
            print(f"[NO_AUX] Error on {row['subject_id']}: {e}")
            import traceback
            traceback.print_exc()

    # Save detailed results
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, 'eval_no_aux_detailed.csv')
    results_df.to_csv(csv_path, index=False)

    # Summary per method
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'no_aux_model_verification',
        'controlnet': args.cnet_ckpt,
        'tpn': args.tpn_ckpt,
        'n_pairs': n_pairs,
        'methods': {}
    }

    print(f"\n{'='*70}")
    print(f"[NO_AUX] === FINAL SUMMARY ===")
    print(f"{'='*70}")

    summary_line_parts = []
    for method in methods:
        method_df = results_df[results_df.method == method]
        if len(method_df) == 0:
            continue
        m_summary = {}
        for col in ['overall_ssim', 'overall_psnr', 'overall_mae']:
            if col in method_df.columns:
                m_summary[col] = {
                    'mean': float(method_df[col].mean()),
                    'std': float(method_df[col].std()),
                }
        for col in ['hippocampus_ssim', 'roi_ssim']:
            if col in method_df.columns and method_df[col].notna().sum() > 0:
                m_summary[col] = {
                    'mean': float(method_df[col].mean()),
                    'std': float(method_df[col].std()),
                }
        summary['methods'][method] = m_summary

        ssim_mean = m_summary.get('overall_ssim', {}).get('mean', 0)
        ssim_std = m_summary.get('overall_ssim', {}).get('std', 0)
        psnr_mean = m_summary.get('overall_psnr', {}).get('mean', 0)
        mae_mean = m_summary.get('overall_mae', {}).get('mean', 0)

        print(f"  {method:<8} SSIM={ssim_mean:.4f}±{ssim_std:.4f}  "
              f"PSNR={psnr_mean:.2f}  MAE={mae_mean:.6f}")
        summary_line_parts.append(f"{method}={ssim_mean:.4f}")

    # One-line summary for dashboard
    summary_line = "[NO_AUX] === SUMMARY === " + " | ".join(summary_line_parts)
    print(summary_line)
    with open(log_path, 'a') as f:
        f.write(summary_line + '\n')

    print(f"{'='*70}")

    # Save JSON summary
    json_path = os.path.join(args.output_dir, 'summary_no_aux.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Check SSIM threshold
    for method in methods:
        m = summary['methods'].get(method, {})
        ssim_val = m.get('overall_ssim', {}).get('mean', 0)
        status = "✓ PASS" if ssim_val >= 0.92 else "✗ FAIL"
        print(f"  {method}: SSIM={ssim_val:.4f} {status} (threshold=0.92)")

    print(f"\nResults: {csv_path}")
    print(f"Summary: {json_path}")
    print(f"Log:     {log_path}")


if __name__ == '__main__':
    main()
