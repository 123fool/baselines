"""
多时间点连续生成与验证
=======================
从基线出发，在多个未来时间点生成脑MRI图像，与真实多时间点数据对比。

Methods:
  Direct-Skip    — 从基线直接生成各时间点，使用基线体积（假设不变）
  Direct-Linear  — 从基线直接生成各时间点，线性外推体积变化
  Direct-TPN     — 从基线直接生成各时间点，TPN预测体积
  Auto-Linear    — 自回归链式生成：上一步输出的潜在表示作为下一步的空间条件

评估指标:
  SSIM / PSNR / MAE（整体）
  按时间间隔分组: 0-6月, 6-12月, 12-24月, 24月+

Usage:
    python evaluate_multi_timepoint.py \
        --dataset_csv  /path/to/B_mci.csv \
        --aekl_ckpt    /path/to/autoencoder.pth \
        --diff_ckpt    /path/to/latentdiffusion.pth \
        --cnet_ckpt    /path/to/cnet-btr-ep-1.pth \
        --tpn_ckpt     /path/to/tpn_best.pth \
        --output_dir   /path/to/output \
        --min_visits   3

日志格式 (用于 dashboard 解析):
    [MULTI_TP] subject visit method | SSIM=x.xxxx | gap=Nmo
    [MULTI_TP] === SUMMARY === Direct-Skip=x.xxxx | Direct-Linear=x.xxxx | ...
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
from torch.cuda.amp.autocast_mode import autocast

# Patch torch.load for older checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'tpn', 'brlp_src'))
sys.path.insert(0, BRLP_SRC)
TPN_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'tpn', 'src'))
sys.path.insert(0, TPN_SRC)

from brlp import const, networks, utils
from generative.networks.schedulers import DDIMScheduler

VOLUME_REGIONS = const.CONDITIONING_REGIONS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]
MCI_ROI_LABELS = HIPPOCAMPUS_LABELS + AMYGDALA_LABELS


# ========================= Utilities =========================

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
    mae = np.abs(pred[mask] - target[mask]).mean()
    data_range = max(target_roi.max() - target_roi.min(), 1e-8)
    try:
        ssim_val = ssim(target_roi, pred_roi, data_range=data_range)
    except Exception:
        ssim_val = float('nan')
    return {'mae': float(mae), 'ssim': float(ssim_val)}


def compute_linear_slopes(train_df):
    """Compute average volume change slopes from training data."""
    slopes = {}
    for region in VOLUME_REGIONS:
        s_col = f'starting_{region}'
        f_col = f'followup_{region}'
        if s_col in train_df.columns and f_col in train_df.columns:
            valid = train_df.dropna(subset=[s_col, f_col, 'starting_age', 'followup_age'])
            age_gaps = valid['followup_age'] - valid['starting_age']
            vol_changes = valid[f_col] - valid[s_col]
            mask = age_gaps.abs() > 1e-6
            if mask.sum() > 0:
                slopes[region] = float((vol_changes[mask] / age_gaps[mask]).mean())
            else:
                slopes[region] = 0.0
        else:
            slopes[region] = 0.0
    return slopes


def load_tpn_model(ckpt_path, device='cpu'):
    from tpn import TemporalProgressionNetwork
    model = TemporalProgressionNetwork(
        in_dim=14, hidden_dim=128, out_dim=5, n_layers=3, dropout=0.0)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


# ========================= Context Builders =========================

def build_context_skip(baseline_info, target_age, target_diag):
    """Use baseline volumes unchanged."""
    return torch.tensor([
        target_age, baseline_info['sex'], target_diag,
        *[baseline_info['volumes'][r] for r in VOLUME_REGIONS]
    ], dtype=torch.float32)


def build_context_linear(baseline_info, target_age, target_diag, train_slopes):
    """Linear extrapolation from baseline using average slopes."""
    age_gap = target_age - baseline_info['age']
    vols = []
    for r in VOLUME_REGIONS:
        s = baseline_info['volumes'][r]
        slope = train_slopes.get(r, 0.0)
        vols.append(max(0.0, min(1.0, s + slope * age_gap)))
    return torch.tensor([target_age, baseline_info['sex'], target_diag, *vols],
                        dtype=torch.float32)


def build_context_tpn(baseline_info, target_age, target_diag, tpn_model):
    """TPN-predicted volumes."""
    current_age = baseline_info['age']
    sex = baseline_info['sex']
    diag = baseline_info.get('diagnosis', 0.5)

    sex_norm = (sex - const.SEX_MIN) / const.SEX_DELTA
    diag_norm = diag if diag <= 1 else (diag - const.DIA_MIN) / const.DIA_DELTA

    current_vols = [baseline_info['volumes'][r] for r in VOLUME_REGIONS]
    age_gap = target_age - current_age
    age_ratio = age_gap / (current_age + 1e-8)
    vol_mean = np.mean(current_vols)
    vol_std = np.std(current_vols)

    x = torch.tensor([
        current_age, target_age, sex_norm, diag_norm,
        *current_vols,
        age_gap, age_ratio, vol_mean, vol_std, age_gap ** 2
    ], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predicted = tpn_model(x).cpu().numpy().flatten()
    return torch.tensor([target_age, baseline_info['sex'], target_diag,
                         *predicted.tolist()], dtype=torch.float32)


def build_context_gt(baseline_info, target_info):
    """Oracle: use true followup volumes."""
    return torch.tensor([
        target_info['age'], baseline_info['sex'], target_info['diagnosis'],
        *[target_info['volumes'][r] for r in VOLUME_REGIONS]
    ], dtype=torch.float32)


# ========================= Sampling =========================

@torch.no_grad()
def sample_controlnet(autoencoder, diffusion, controlnet,
                      starting_z, starting_a, context, scale_factor,
                      num_inference_steps=50, return_latent=False):
    """
    ControlNet inference. Optionally return raw denoised latent for
    autoregressive chaining.

    Args:
        starting_z: [C, D, H, W] latent * scale_factor
        starting_a: scalar, normalized baseline age
        context: [8] tensor
        return_latent: if True, also return raw denoised z (in scaled space)

    Returns:
        image: [D, H, W] decoded MRI
        (optional) raw_z: [C, D, H, W] denoised latent in scaled space
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    starting_z_4d = starting_z.unsqueeze(0).to(DEVICE)
    concat_age = (torch.tensor([starting_a])
                  .view(1, 1, 1, 1, 1)
                  .expand(1, 1, *starting_z_4d.shape[-3:])
                  .to(DEVICE))
    cnet_cond = torch.cat([starting_z_4d, concat_age], dim=1).to(DEVICE)
    ctx = context.unsqueeze(0).unsqueeze(0).to(DEVICE)

    z = torch.randn(1, *starting_z_4d.shape[1:]).to(DEVICE)

    for t in scheduler.timesteps:
        with autocast(enabled=True):
            timestep = torch.tensor([t]).to(DEVICE)
            down_h, mid_h = controlnet(
                x=z.float(), timesteps=timestep,
                context=ctx, controlnet_cond=cnet_cond.float())
            noise_pred = diffusion(
                x=z.float(), timesteps=timestep, context=ctx.float(),
                down_block_additional_residuals=down_h,
                mid_block_additional_residual=mid_h)
            z, _ = scheduler.step(noise_pred, t, z)

    # Save raw latent for autoregressive chaining (in scaled ≈ diffusion space)
    raw_z = z.squeeze(0).cpu().clone() if return_latent else None

    # Decode
    z_dec = z / scale_factor
    z_dec = z_dec.squeeze(0)
    z_dec = utils.to_vae_latent_trick(z_dec.cpu())
    image = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(DEVICE))
    image = utils.to_mni_space_1p5mm_trick(image.squeeze(0).cpu()).squeeze(0)

    if return_latent:
        return image, raw_z
    return image


# ========================= Multi-visit Subject Discovery =========================

def get_multi_visit_subjects(df, min_visits=3):
    """
    Build multi-visit timelines per subject.
    Returns {subject_id: [(rounded_day_key, visit_info), ...]} sorted by time.
    """
    timelines = {}
    for _, row in df.iterrows():
        sid = row['subject_id']
        if sid not in timelines:
            timelines[sid] = {}

        s_days = int(round(row['starting_days_from_first_visit']))
        f_days = int(round(row['followup_days_from_first_visit']))

        # Round to nearest 30 days to group nearby visits
        s_key = round(s_days / 30) * 30
        f_key = round(f_days / 30) * 30

        if s_key not in timelines[sid]:
            timelines[sid][s_key] = {
                'days': s_days,
                'age': row['starting_age'],
                'latent': row['starting_latent'],
                'image': row['starting_image'],
                'volumes': {r: row[f'starting_{r}'] for r in VOLUME_REGIONS},
                'sex': row['sex'],
                'diagnosis': row.get('starting_diagnosis',
                                     row.get('starting_last_diagnosis', 0.5)),
                'split': row['split'],
            }
        if f_key not in timelines[sid]:
            timelines[sid][f_key] = {
                'days': f_days,
                'age': row['followup_age'],
                'latent': row['followup_latent'],
                'image': row['followup_image'],
                'volumes': {r: row[f'followup_{r}'] for r in VOLUME_REGIONS},
                'sex': row['sex'],
                'diagnosis': row.get('followup_diagnosis', 0.5),
                'split': row['split'],
            }

    result = {}
    for sid, timeline in timelines.items():
        sorted_visits = sorted(timeline.items(), key=lambda x: x[0])
        if len(sorted_visits) >= min_visits:
            result[sid] = sorted_visits
    return result


# ========================= Evaluation =========================

def evaluate_image(predicted_np, followup_np):
    """Compute SSIM / PSNR / MAE between predicted and GT images."""
    min_shape = tuple(min(a, b) for a, b in
                      zip(predicted_np.shape, followup_np.shape))
    p = predicted_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    f = followup_np[:min_shape[0], :min_shape[1], :min_shape[2]]

    data_range = max(f.max() - f.min(), 1e-8)
    return {
        'overall_ssim': float(ssim(f, p, data_range=data_range)),
        'overall_psnr': float(psnr(f, p, data_range=data_range)),
        'overall_mae':  float(np.abs(f - p).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='多时间点连续生成与验证')
    parser.add_argument('--dataset_csv', required=True)
    parser.add_argument('--aekl_ckpt', required=True)
    parser.add_argument('--diff_ckpt', required=True)
    parser.add_argument('--cnet_ckpt', required=True)
    parser.add_argument('--tpn_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_visits', default=3, type=int)
    parser.add_argument('--splits', default='test,valid',
                        help='Comma-separated splits to evaluate')
    parser.add_argument('--methods',
                        default='Direct-Skip,Direct-Linear,Direct-TPN,Auto-Linear',
                        help='Comma-separated methods')
    parser.add_argument('--gpu', default=0, type=int, help='GPU index')
    args = parser.parse_args()

    global DEVICE
    if torch.cuda.is_available():
        DEVICE = f'cuda:{args.gpu}'
        torch.cuda.set_device(args.gpu)

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(',')]
    splits = [s.strip() for s in args.splits.split(',')]
    log_path = os.path.join(args.output_dir, 'eval_multi_tp.log')

    print(f"[MULTI_TP] 多时间点连续生成与验证")
    print(f"  CSV: {args.dataset_csv}")
    print(f"  ControlNet: {args.cnet_ckpt}")
    print(f"  TPN: {args.tpn_ckpt}")
    print(f"  Methods: {methods}")
    print(f"  Splits: {splits}")
    print(f"  Min visits: {args.min_visits}")
    print(f"  Device: {DEVICE}")

    # ---- Load models ----
    print("[MULTI_TP] Loading models...")
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE).eval()
    controlnet = networks.init_controlnet(args.cnet_ckpt).to(DEVICE).eval()

    tpn_model = None
    if any('TPN' in m for m in methods):
        tpn_model = load_tpn_model(args.tpn_ckpt, device=DEVICE)
        print(f"[MULTI_TP] TPN loaded")

    # ---- Transforms ----
    npz_reader = NumpyReader(npz_keys=['data'])
    load_latent = transforms.Compose([
        transforms.LoadImage(reader=npz_reader, image_only=True),
        transforms.EnsureChannelFirst(channel_dim=0),
        transforms.DivisiblePad(k=4, mode='constant'),
    ])
    load_gt = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    # ---- Data ----
    df = pd.read_csv(args.dataset_csv)
    train_df = df[df.split == 'train']

    first_latent = load_latent(train_df.iloc[0]['followup_latent'])
    scale_factor = 1 / torch.std(first_latent)
    print(f"[MULTI_TP] Scale factor: {scale_factor:.4f}")

    train_slopes = compute_linear_slopes(train_df)
    print(f"[MULTI_TP] Linear slopes: {train_slopes}")

    # ---- Multi-visit subjects ----
    multi_subjects = get_multi_visit_subjects(df, min_visits=args.min_visits)

    eval_subjects = {}
    for sid, visits in multi_subjects.items():
        subj_split = visits[0][1]['split']
        if subj_split in splits:
            eval_subjects[sid] = visits

    total_visits = sum(len(v) - 1 for v in eval_subjects.values())
    total_inferences = total_visits * len(methods)
    print(f"[MULTI_TP] Subjects: {len(eval_subjects)} ({len(multi_subjects)} total multi-visit)")
    print(f"[MULTI_TP] Visit pairs: {total_visits}, total inferences: {total_inferences}")

    all_results = []
    inference_count = 0

    for subj_idx, (sid, visits) in enumerate(sorted(eval_subjects.items())):
        n_visits = len(visits)
        baseline = visits[0][1]

        print(f"\n[MULTI_TP] === Subject {subj_idx+1}/{len(eval_subjects)}: "
              f"{sid} [{baseline['split']}] {n_visits} visits ===")

        # Load baseline latent (always used for Direct methods)
        baseline_latent = load_latent(baseline['latent']) * scale_factor
        baseline_age = baseline['age']

        # Autoregressive state: track current latent and age per method
        auto_state = {}
        for m in methods:
            if m.startswith('Auto'):
                auto_state[m] = {
                    'latent': baseline_latent.clone(),
                    'age': baseline_age,
                }

        for visit_idx in range(1, n_visits):
            target = visits[visit_idx][1]
            days_gap = target['days'] - baseline['days']
            months_gap = days_gap / 30.44

            # Load real followup image
            try:
                followup_np = load_gt(target['image']).squeeze(0).numpy()
            except Exception as e:
                print(f"  [WARN] Cannot load visit {visit_idx} image: {e}")
                continue

            for method in methods:
                try:
                    is_auto = method.startswith('Auto-')
                    ctx_type = method.split('-', 1)[1]  # Skip / Linear / TPN

                    # Choose starting point
                    if is_auto:
                        starting_z = auto_state[method]['latent'].float()
                        starting_a = auto_state[method]['age']
                    else:
                        starting_z = baseline_latent.float()
                        starting_a = baseline_age

                    # Target info
                    target_age = target['age']
                    target_diag = target.get('diagnosis', 0.5)

                    # Build context
                    if ctx_type == 'Skip':
                        ctx = build_context_skip(baseline, target_age, target_diag)
                    elif ctx_type == 'Linear':
                        ctx = build_context_linear(baseline, target_age, target_diag, train_slopes)
                    elif ctx_type == 'TPN':
                        ctx = build_context_tpn(baseline, target_age, target_diag, tpn_model)
                    elif ctx_type == 'GT':
                        ctx = build_context_gt(baseline, target)
                    else:
                        raise ValueError(f"Unknown context type: {ctx_type}")

                    # Generate
                    result = sample_controlnet(
                        autoencoder, diffusion, controlnet,
                        starting_z, starting_a, ctx.float(), scale_factor,
                        num_inference_steps=50,
                        return_latent=is_auto)

                    if is_auto:
                        predicted_img, raw_z = result
                        # Chain: use generated latent as next starting point
                        auto_state[method]['latent'] = raw_z
                        auto_state[method]['age'] = target_age
                    else:
                        predicted_img = result

                    # Evaluate
                    pred_np = predicted_img.numpy().clip(0, 1)
                    metrics = evaluate_image(pred_np, followup_np)

                    res = {
                        'subject_id': sid,
                        'split': baseline['split'],
                        'method': method,
                        'visit_idx': visit_idx,
                        'n_visits': n_visits,
                        'days_from_baseline': days_gap,
                        'months_from_baseline': round(months_gap, 1),
                        'baseline_age': baseline_age,
                        'target_age': target_age,
                        **metrics,
                    }
                    all_results.append(res)
                    inference_count += 1

                    line = (f"[MULTI_TP] {sid} v{visit_idx} {method:<16} | "
                            f"SSIM={metrics['overall_ssim']:.4f} "
                            f"PSNR={metrics['overall_psnr']:.2f} | "
                            f"gap={months_gap:.0f}mo "
                            f"[{inference_count}/{total_inferences}]")
                    print(line)
                    with open(log_path, 'a') as lf:
                        lf.write(line + '\n')

                except Exception as e:
                    print(f"  [ERROR] {sid} v{visit_idx} {method}: {e}")
                    import traceback
                    traceback.print_exc()

    # =================== Save Results ===================
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, 'eval_multi_timepoint.csv')
    results_df.to_csv(csv_path, index=False)

    # =================== Summary ===================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'multi_timepoint_generation',
        'controlnet': args.cnet_ckpt,
        'n_subjects': len(eval_subjects),
        'total_inferences': inference_count,
        'methods': {},
    }

    print(f"\n{'='*70}")
    print(f"[MULTI_TP] === FINAL SUMMARY ===")
    print(f"{'='*70}")

    summary_parts = []
    for method in methods:
        mdf = results_df[results_df.method == method]
        if len(mdf) == 0:
            continue

        m_sum = {
            'n': len(mdf),
            'overall_ssim': {
                'mean': float(mdf.overall_ssim.mean()),
                'std':  float(mdf.overall_ssim.std()),
            },
            'overall_psnr': {
                'mean': float(mdf.overall_psnr.mean()),
                'std':  float(mdf.overall_psnr.std()),
            },
            'overall_mae': {
                'mean': float(mdf.overall_mae.mean()),
                'std':  float(mdf.overall_mae.std()),
            },
            'by_time_gap': {},
        }

        # Breakdown by time gap
        gap_bins = [
            ('0-6mo',   0,   183),
            ('6-12mo',  183, 365),
            ('12-24mo', 365, 730),
            ('24mo+',   730, 99999),
        ]
        for label, lo, hi in gap_bins:
            subset = mdf[(mdf.days_from_baseline >= lo) & (mdf.days_from_baseline < hi)]
            if len(subset) > 0:
                m_sum['by_time_gap'][label] = {
                    'n': len(subset),
                    'ssim_mean': float(subset.overall_ssim.mean()),
                    'ssim_std':  float(subset.overall_ssim.std()),
                    'psnr_mean': float(subset.overall_psnr.mean()),
                    'mae_mean':  float(subset.overall_mae.mean()),
                }

        summary['methods'][method] = m_sum
        ssim_m = m_sum['overall_ssim']['mean']
        ssim_s = m_sum['overall_ssim']['std']
        psnr_m = m_sum['overall_psnr']['mean']
        print(f"  {method:<16} SSIM={ssim_m:.4f}±{ssim_s:.4f}  "
              f"PSNR={psnr_m:.2f}  n={len(mdf)}")
        summary_parts.append(f"{method}={ssim_m:.4f}")

    # Time-gap breakdown
    print(f"\n--- SSIM by time gap ---")
    header = f"{'Method':<16}"
    for label, _, _ in gap_bins:
        header += f" | {label:>12}"
    print(header)
    for method in methods:
        m = summary['methods'].get(method, {})
        by_gap = m.get('by_time_gap', {})
        row = f"{method:<16}"
        for label, _, _ in gap_bins:
            g = by_gap.get(label, {})
            if g.get('n', 0) > 0:
                row += f" | {g['ssim_mean']:.4f}(n={g['n']:>2})"
            else:
                row += f" | {'---':>12}"
        print(row)

    # Per-subject summary
    print(f"\n--- Per-subject SSIM ---")
    for method in methods:
        mdf = results_df[results_df.method == method]
        by_subj = mdf.groupby('subject_id')['overall_ssim'].mean()
        subj_str = ', '.join(f"{s}={v:.4f}" for s, v in by_subj.items())
        print(f"  {method:<16} {subj_str}")

    # One-line summary for dashboard
    summary_line = "[MULTI_TP] === SUMMARY === " + " | ".join(summary_parts)
    print(f"\n{summary_line}")
    with open(log_path, 'a') as lf:
        lf.write(summary_line + '\n')

    # Threshold check
    print(f"\n--- SSIM ≥ 0.92 check ---")
    for method in methods:
        m = summary['methods'].get(method, {})
        s = m.get('overall_ssim', {}).get('mean', 0)
        status = "PASS" if s >= 0.92 else "FAIL"
        print(f"  {method:<16} SSIM={s:.4f} {status}")

    # Save
    json_path = os.path.join(args.output_dir, 'summary_multi_timepoint.json')
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results: {csv_path}")
    print(f"Summary: {json_path}")
    print(f"Log:     {log_path}")


if __name__ == '__main__':
    main()
