"""
Section 33: Hippocampus SSIM Improvement — V3
==============================================
V3 新策略: 探索LAS参数空间和候选多样性

V2结论: Best-of-16 with LAS m=3 只提升0.0035 H-SSIM
       Oracle上界也只有0.8689, 说明候选多样性不足

V3核心思路:
  1. 尝试不同LAS m值 (m=1,3,5,7) — 找最优平均数
  2. 低LAS+多候选选择 — 增大候选多样性再选最佳
  3. 混合LAS — 不同m的候选混合选择
  4. 后处理增强 — 在海马体区域做局部增强

Methods:
  A:  Baseline LAS m=3  (对照)
  J:  LAS m=5  (更多平均)
  K:  LAS m=7  (最大平均)
  L:  Best-of-16, LAS m=1 per candidate + hippo scoring (最大多样性)
  N:  Best-of-16, LAS m=1 per candidate + oracle scoring (多样性上界)
  P:  LAS m=5 + Best-of-8 with oracle scoring (高质量+选择)
"""

import os, sys, json, time, argparse, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from skimage.metrics import structural_similarity as ssim_fn
from monai import transforms
from monai.data.meta_tensor import MetaTensor

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel in [['..', '..', 'src'], ['..', '..', '..', 'src']]:
    _p = os.path.abspath(os.path.join(SCRIPT_DIR, *_rel))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from brlp import const, utils
from brlp.networks import init_autoencoder, init_latent_diffusion, init_controlnet

# ── Server paths ──
AEKL_CKPT = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT = "/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth"
CSV_PATH  = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
OUTPUT_DIR = "/home/wangchong/data/fwz/output/33_hippocampus_v3"

SCALE_FACTOR = 1.0469
HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def create_roi_mask(segm_data, labels):
    mask = np.zeros_like(segm_data, dtype=bool)
    for label in labels:
        mask |= (segm_data.round() == label)
    return mask


def get_roi_bbox(mask):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return tuple(slice(c.min(), c.max() + 1) for c in coords)


def compute_region_metrics(pred, target, mask):
    if mask.sum() == 0:
        return {'mae': float('nan'), 'ssim': float('nan')}
    bbox = get_roi_bbox(mask)
    if bbox is None:
        return {'mae': float('nan'), 'ssim': float('nan')}
    pred_roi = pred[bbox]
    target_roi = target[bbox]
    mae = np.abs(pred[mask > 0] - target[mask > 0]).mean()
    data_range = max(target_roi.max() - target_roi.min(), 1e-8)
    try:
        ssim_val = ssim_fn(target_roi, pred_roi, data_range=data_range)
    except Exception:
        ssim_val = float('nan')
    return {'mae': float(mae), 'ssim': float(ssim_val)}


def load_and_resample_segm(segm_path, target_shape):
    segm_tensor = torch.from_numpy(
        nib.load(segm_path).get_fdata().astype(np.float32)
    ).unsqueeze(0)
    resample = transforms.Compose([
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
    ])
    segm = resample(segm_tensor).squeeze(0).numpy().round().astype(np.int32)
    min_shape = tuple(min(a, b) for a, b in zip(segm.shape, target_shape))
    return segm[:min_shape[0], :min_shape[1], :min_shape[2]]


def compute_full_metrics(pred_np, gt_np, segm=None):
    min_shape = tuple(min(a, b) for a, b in zip(pred_np.shape, gt_np.shape))
    pred_np = pred_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    gt_np = gt_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    data_range = max(gt_np.max() - gt_np.min(), 1e-8)
    result = {
        'overall_ssim': float(ssim_fn(gt_np, pred_np, data_range=data_range)),
        'overall_mae': float(np.abs(gt_np - pred_np).mean()),
    }
    if segm is not None:
        segm = segm[:min_shape[0], :min_shape[1], :min_shape[2]]
        hipp_mask = create_roi_mask(segm, HIPPOCAMPUS_LABELS)
        amyg_mask = create_roi_mask(segm, AMYGDALA_LABELS)
        roi_mask = hipp_mask | amyg_mask
        hipp = compute_region_metrics(pred_np, gt_np, hipp_mask)
        amyg = compute_region_metrics(pred_np, gt_np, amyg_mask)
        roi = compute_region_metrics(pred_np, gt_np, roi_mask)
        result.update({
            'hippocampus_ssim': hipp['ssim'],
            'hippocampus_mae': hipp['mae'],
            'amygdala_mae': amyg['mae'],
            'roi_ssim': roi['ssim'],
            'roi_mae': roi['mae'],
        })
    return result


def score_hippo_ssim(generated_np, reference_np, ref_segm):
    min_shape = tuple(min(a, b, c) for a, b, c in
                      zip(generated_np.shape, reference_np.shape, ref_segm.shape))
    g = generated_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    r = reference_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    s = ref_segm[:min_shape[0], :min_shape[1], :min_shape[2]]
    mask = create_roi_mask(s, HIPPOCAMPUS_LABELS)
    m = compute_region_metrics(g, r, mask)
    return m['ssim'] if not np.isnan(m['ssim']) else 0.5


# ═══════════════════════════════════════════════════════════════
#  Sampling Functions
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_las(autoencoder, diffusion, controlnet, starting_z, starting_a,
                 context, device, scale_factor=1.0, las_m=3,
                 num_inference_steps=50, seed=None):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    sz = starting_z.unsqueeze(0).to(device)
    age_vol = (torch.tensor([starting_a]).view(1,1,1,1,1)
               .expand(1, 1, *sz.shape[-3:]).to(device))
    cnet_cond = torch.cat([sz, age_vol], dim=1)
    ctx = context.unsqueeze(0).unsqueeze(0).to(device)

    batch = max(1, las_m)
    if batch > 1:
        ctx = ctx.repeat(batch, 1, 1)
        cnet_cond = cnet_cond.repeat(batch, 1, 1, 1, 1)

    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(batch, *sz.shape[1:]).to(device)

    for t in scheduler.timesteps:
        with autocast(enabled=True):
            timestep = torch.tensor([t]).repeat(batch).to(device)
            dh, mh = controlnet(
                x=z.float(), timesteps=timestep,
                context=ctx, controlnet_cond=cnet_cond.float())
            noise_pred = diffusion(
                x=z.float(), timesteps=timestep, context=ctx.float(),
                down_block_additional_residuals=dh,
                mid_block_additional_residual=mh)
            z, _ = scheduler.step(noise_pred, t, z)

    z = (z / scale_factor).sum(axis=0) / batch
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x.numpy().clip(0, 1)


@torch.no_grad()
def generate_n_las_candidates(autoencoder, diffusion, controlnet,
                              starting_z, starting_a, context, device,
                              scale_factor=1.0, n_candidates=16, las_m=3,
                              num_inference_steps=50):
    candidates = []
    for i in range(n_candidates):
        seed = 42 + i * 7919
        img = generate_las(
            autoencoder, diffusion, controlnet,
            starting_z, starting_a, context, device,
            scale_factor=scale_factor, las_m=las_m,
            num_inference_steps=num_inference_steps, seed=seed)
        candidates.append(img)
        torch.cuda.empty_cache()
    return candidates


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def load_models(device):
    ae = init_autoencoder(AEKL_CKPT).to(device).eval()
    dm = init_latent_diffusion(DIFF_CKPT).to(device).eval()
    cn = init_controlnet(CNET_CKPT).to(device).eval()
    return ae, dm, cn


def prepare_pair(row, ae, device):
    loader = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])
    start_data = loader({'image_path': row['starting_image']})
    start_img = start_data['image'].unsqueeze(0).to(device)
    start_z = ae.encode(start_img)[0]
    start_z = transforms.DivisiblePad(k=4, mode='constant')(start_z.squeeze(0))
    source_np = transforms.ResizeWithPadOrCrop(
        spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'
    )(start_data['image']).squeeze(0).numpy().clip(0, 1)
    follow_data = loader({'image_path': row['followup_image']})
    follow_np = transforms.ResizeWithPadOrCrop(
        spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'
    )(follow_data['image']).squeeze(0).numpy().clip(0, 1)
    context = torch.tensor([
        row['followup_age'],
        row['sex'],
        row.get('followup_diagnosis', row.get('starting_diagnosis', 2)),
        row['followup_cerebral_cortex'],
        row['followup_hippocampus'],
        row['followup_amygdala'],
        row['followup_cerebral_white_matter'],
        row['followup_lateral_ventricle'],
    ], dtype=torch.float32)
    return start_z, row['starting_age'], context, source_np, follow_np


def get_segmentation(row, key_prefix, target_shape):
    for suffix in ['segm', 'segm_path']:
        key = f'{key_prefix}{suffix}'
        if key in row and pd.notna(row[key]) and os.path.exists(str(row[key])):
            return load_and_resample_segm(row[key], target_shape)
    return None


# ═══════════════════════════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════════════════════════

def run_experiment(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "experiment.log")
    methods = [m.strip().upper() for m in args.methods.split(',')]

    def log(msg):
        line = f"{ts()} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"═══ Hippocampus Improvement V3 ═══")
    log(f"Methods: {methods}")
    log(f"Max pairs: {args.max_pairs}")
    log(f"Device: {device}")

    ae, dm, cn = load_models(device)
    log("Models loaded.")

    df = pd.read_csv(args.csv_path)
    if 'split' in df.columns:
        test_df = df[df.split == 'test']
        if len(test_df) == 0:
            test_df = df[df.split == 'valid']
        if len(test_df) == 0:
            test_df = df
    else:
        test_df = df

    n_pairs = min(len(test_df), args.max_pairs)
    log(f"Using {n_pairs} test pairs")

    all_results = {m: [] for m in methods}

    for idx in range(n_pairs):
        row = test_df.iloc[idx].to_dict()
        subj = row.get('subject_id', f'pair_{idx}')
        log(f"\n{'='*60}")
        log(f"Pair {idx+1}/{n_pairs}: {subj}")

        try:
            with torch.no_grad():
                start_z, start_a, context, source_np, gt_np = prepare_pair(row, ae, device)

            eval_segm = get_segmentation(row, 'followup_', gt_np.shape)
            source_segm = get_segmentation(row, 'starting_', source_np.shape)
            if source_segm is None:
                source_segm = eval_segm
            if source_segm is None:
                source_segm = np.zeros_like(source_np, dtype=np.int32)
                log("  WARNING: No segmentation found")

            # Cache candidates for different LAS m values
            cache_m1_16 = None  # 16 candidates with LAS m=1
            cache_m5_8 = None   # 8 candidates with LAS m=5

            for method in methods:
                t0 = time.time()
                log(f"  Method {method}...")

                try:
                    if method == 'A':
                        # Baseline: LAS m=3
                        pred_np = generate_las(
                            ae, dm, cn, start_z, start_a, context, device,
                            scale_factor=SCALE_FACTOR, las_m=3, seed=42)

                    elif method == 'J':
                        # LAS m=5
                        pred_np = generate_las(
                            ae, dm, cn, start_z, start_a, context, device,
                            scale_factor=SCALE_FACTOR, las_m=5, seed=42)

                    elif method == 'K':
                        # LAS m=7
                        pred_np = generate_las(
                            ae, dm, cn, start_z, start_a, context, device,
                            scale_factor=SCALE_FACTOR, las_m=7, seed=42)

                    elif method == 'L':
                        # Best-of-16, LAS m=1 (max diversity) + hippo scoring
                        if cache_m1_16 is None:
                            cache_m1_16 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=16, las_m=1)
                        scores = [score_hippo_ssim(c, source_np, source_segm)
                                  for c in cache_m1_16]
                        best_idx = np.argmax(scores)
                        pred_np = cache_m1_16[best_idx]
                        log(f"    Best hippo score: {scores[best_idx]:.4f} "
                            f"(range: {min(scores):.4f} - {max(scores):.4f})")

                    elif method == 'N':
                        # Oracle Best-of-16, LAS m=1 (diversity upper bound)
                        if cache_m1_16 is None:
                            cache_m1_16 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=16, las_m=1)
                        if eval_segm is not None:
                            oracle_scores = []
                            for c in cache_m1_16:
                                m = compute_full_metrics(c, gt_np, eval_segm)
                                oracle_scores.append(m.get('hippocampus_ssim', 0))
                            best_idx = np.argmax(oracle_scores)
                            pred_np = cache_m1_16[best_idx]
                            log(f"    Oracle best: {max(oracle_scores):.4f} "
                                f"worst: {min(oracle_scores):.4f} "
                                f"range: {max(oracle_scores)-min(oracle_scores):.4f}")
                        else:
                            pred_np = cache_m1_16[0]

                    elif method == 'P':
                        # LAS m=5 + Best-of-8 with oracle scoring
                        if cache_m5_8 is None:
                            cache_m5_8 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=8, las_m=5)
                        if eval_segm is not None:
                            oracle_scores = []
                            for c in cache_m5_8:
                                m = compute_full_metrics(c, gt_np, eval_segm)
                                oracle_scores.append(m.get('hippocampus_ssim', 0))
                            best_idx = np.argmax(oracle_scores)
                            pred_np = cache_m5_8[best_idx]
                            log(f"    Oracle best: {max(oracle_scores):.4f} "
                                f"worst: {min(oracle_scores):.4f}")
                        else:
                            pred_np = cache_m5_8[0]

                    else:
                        log(f"    Unknown method {method}, skipping")
                        continue

                    elapsed = time.time() - t0
                    metrics = compute_full_metrics(pred_np, gt_np, eval_segm)
                    metrics['method'] = method
                    metrics['pair_idx'] = idx
                    metrics['subject'] = subj
                    metrics['time_sec'] = round(elapsed, 2)
                    all_results[method].append(metrics)

                    h_ssim = metrics.get('hippocampus_ssim', float('nan'))
                    o_ssim = metrics.get('overall_ssim', float('nan'))
                    h_mae = metrics.get('hippocampus_mae', float('nan'))
                    log(f"    SSIM={o_ssim:.4f} H-SSIM={h_ssim:.4f} H-MAE={h_mae:.4f} "
                        f"T={elapsed:.1f}s")

                except Exception as e:
                    log(f"    ERROR: {e}")
                    traceback.print_exc()
                    all_results[method].append({
                        'method': method, 'pair_idx': idx,
                        'subject': subj, 'error': str(e)})

                torch.cuda.empty_cache()

            del cache_m1_16, cache_m5_8
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"  ERROR preparing: {e}")
            traceback.print_exc()
            for method in methods:
                all_results[method].append({
                    'method': method, 'pair_idx': idx,
                    'subject': subj, 'error': str(e)})

    # ── Summary ──
    log(f"\n{'='*80}")
    log(f"EXPERIMENT SUMMARY (V3)")
    log(f"{'='*80}")

    summary = {}
    for method in methods:
        valid = [r for r in all_results[method] if 'error' not in r]
        if not valid:
            log(f"Method {method}: No valid results")
            continue
        mdf = pd.DataFrame(valid)
        summary[method] = {}
        log(f"\nMethod {method} ({len(valid)} pairs):")
        for col in ['overall_ssim', 'overall_mae', 'hippocampus_ssim',
                     'hippocampus_mae', 'roi_ssim', 'time_sec']:
            if col in mdf.columns:
                mean_val = mdf[col].mean()
                std_val = mdf[col].std()
                log(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
                summary[method][f'{col}_mean'] = round(float(mean_val), 6)
                summary[method][f'{col}_std'] = round(float(std_val), 6)

        csv_path = os.path.join(args.output_dir, f'results_{method}.csv')
        mdf.to_csv(csv_path, index=False)

    summary['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'n_pairs': n_pairs,
        'methods': methods,
        'scale_factor': SCALE_FACTOR,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    log(f"\n{'='*80}")
    log(f"{'Method':<8} {'SSIM':>8} {'H-SSIM':>8} {'H-MAE':>8} {'Time':>8}")
    log(f"{'-'*40}")
    for method in methods:
        if method in summary and 'overall_ssim_mean' in summary[method]:
            s = summary[method]
            log(f"{method:<8} {s['overall_ssim_mean']:>8.4f} "
                f"{s.get('hippocampus_ssim_mean', 0):>8.4f} "
                f"{s.get('hippocampus_mae_mean', 0):>8.4f} "
                f"{s.get('time_sec_mean', 0):>8.1f}")
    log(f"\nDone: {datetime.now().isoformat()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='A,J,K,L,N,P')
    parser.add_argument('--max-pairs', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--csv-path', type=str, default=CSV_PATH)
    args = parser.parse_args()
    run_experiment(args)
