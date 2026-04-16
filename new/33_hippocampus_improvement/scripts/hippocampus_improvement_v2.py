"""
Section 33: Hippocampus SSIM Improvement — V2
==============================================
改进版本:
- 方法B改为每个候选使用LAS m=3平均
- 增加Oracle方法(用GT评分,作为上界参考)
- 增加方法G: Best-of-N with overall SSIM scoring (instead of hippo-specific)
- 增加方法H: 增大候选数量 (N=32, 每个LAS m=3)
- 增加方法I: 加权融合top-K候选

Methods:
  A: Baseline LAS m=3
  B: Best-of-16 with LAS m=3 per candidate, hippo scoring
  D: Hippo-Weighted ET-BoN (N=16→6, cp=10)
  E: CCI Selection (N=8, forward+reverse cycle consistency)
  F: Region-Adaptive Fusion (N=16, LAS m=3)
  G: Best-of-16 with overall SSIM scoring + LAS m=3
  H: Best-of-32 with overall SSIM scoring + LAS m=3
  I: Weighted Fusion top-5 from N=16 (LAS m=3)
  O: Oracle Best-of-16 (GT scoring, upper bound only)
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
from tqdm import tqdm

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
OUTPUT_DIR = "/home/wangchong/data/fwz/output/33_hippocampus_v2"

SCALE_FACTOR = 1.0469
LAS_M = 3

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


# ═══════════════════════════════════════════════════════════════
#  Scoring Functions
# ═══════════════════════════════════════════════════════════════

def score_overall_ssim(generated_np, reference_np):
    """Overall SSIM vs reference (source image)."""
    min_shape = tuple(min(a, b) for a, b in zip(generated_np.shape, reference_np.shape))
    g = generated_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    r = reference_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    return float(ssim_fn(g, r, data_range=1.0))


def score_hippo_ssim(generated_np, reference_np, ref_segm):
    """Hippocampus SSIM vs reference."""
    min_shape = tuple(min(a, b, c) for a, b, c in
                      zip(generated_np.shape, reference_np.shape, ref_segm.shape))
    g = generated_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    r = reference_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    s = ref_segm[:min_shape[0], :min_shape[1], :min_shape[2]]
    mask = create_roi_mask(s, HIPPOCAMPUS_LABELS)
    m = compute_region_metrics(g, r, mask)
    return m['ssim'] if not np.isnan(m['ssim']) else 0.5


def score_composite(generated_np, reference_np, ref_segm):
    """Combined score: 50% overall SSIM + 30% hippo SSIM + 20% intensity match."""
    o_ssim = score_overall_ssim(generated_np, reference_np)
    h_ssim = score_hippo_ssim(generated_np, reference_np, ref_segm)

    # Intensity similarity
    min_shape = tuple(min(a, b) for a, b in zip(generated_np.shape, reference_np.shape))
    g = generated_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    r = reference_np[:min_shape[0], :min_shape[1], :min_shape[2]].clip(0, 1)
    g_mask = g > 0.01
    r_mask = r > 0.01
    if g_mask.sum() > 100 and r_mask.sum() > 100:
        intens = max(0, 1.0 - abs(float(g[g_mask].mean() - r[r_mask].mean())) * 4)
    else:
        intens = 0.5

    return 0.50 * o_ssim + 0.30 * h_ssim + 0.20 * intens


# ═══════════════════════════════════════════════════════════════
#  Sampling Functions
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_las(autoencoder, diffusion, controlnet, starting_z, starting_a,
                 context, device, scale_factor=1.0, las_m=3,
                 num_inference_steps=50, seed=None):
    """Generate with LAS averaging (standard BrLP pipeline)."""
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    sz = starting_z.unsqueeze(0).to(device)
    age_vol = (torch.tensor([starting_a]).view(1,1,1,1,1)
               .expand(1, 1, *sz.shape[-3:]).to(device))
    cnet_cond = torch.cat([sz, age_vol], dim=1)
    ctx = context.unsqueeze(0).unsqueeze(0).to(device)

    if las_m > 1:
        ctx = ctx.repeat(las_m, 1, 1)
        cnet_cond = cnet_cond.repeat(las_m, 1, 1, 1, 1)

    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(las_m, *sz.shape[1:]).to(device)

    for t in scheduler.timesteps:
        with autocast(enabled=True):
            timestep = torch.tensor([t]).repeat(las_m).to(device)
            dh, mh = controlnet(
                x=z.float(), timesteps=timestep,
                context=ctx, controlnet_cond=cnet_cond.float())
            noise_pred = diffusion(
                x=z.float(), timesteps=timestep, context=ctx.float(),
                down_block_additional_residuals=dh,
                mid_block_additional_residual=mh)
            z, _ = scheduler.step(noise_pred, t, z)

    z = (z / scale_factor).sum(axis=0) / las_m
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x.numpy().clip(0, 1)


@torch.no_grad()
def generate_n_las_candidates(autoencoder, diffusion, controlnet,
                              starting_z, starting_a, context, device,
                              scale_factor=1.0, n_candidates=16, las_m=3,
                              num_inference_steps=50):
    """Generate N independent LAS-averaged candidates."""
    candidates = []
    for i in range(n_candidates):
        seed = 42 + i * 7919  # prime spacing for independence
        img = generate_las(
            autoencoder, diffusion, controlnet,
            starting_z, starting_a, context, device,
            scale_factor=scale_factor, las_m=las_m,
            num_inference_steps=num_inference_steps, seed=seed)
        candidates.append(img)
        torch.cuda.empty_cache()
    return candidates


@torch.no_grad()
def generate_et_bon_hippo(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context, device,
    source_np, source_segm,
    scale_factor=1.0,
    n_initial=16, n_survivors=6, checkpoint_step=10,
    num_inference_steps=50):
    """
    Hippo-Weighted ET-BoN: Early filtering + hippocampus-emphasis scoring.
    Uses LAS m=1 per candidate (speed tradeoff for more candidates).
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    sz = starting_z.unsqueeze(0).to(device)
    age_vol = (torch.tensor([starting_a]).view(1,1,1,1,1)
               .expand(1, 1, *sz.shape[-3:]).to(device))
    cnet_cond = torch.cat([sz, age_vol], dim=1)
    ctx = context.unsqueeze(0).unsqueeze(0).to(device)
    timesteps = list(scheduler.timesteps)

    # Phase 1: All N candidates until checkpoint
    latents = []
    for _ in range(n_initial):
        z = torch.randn(1, *sz.shape[1:]).to(device)
        for t in timesteps[:checkpoint_step]:
            with autocast(enabled=True):
                ts = torch.tensor([t]).to(device)
                dh, mh = controlnet(x=z.float(), timesteps=ts,
                                    context=ctx, controlnet_cond=cnet_cond.float())
                noise_pred = diffusion(x=z.float(), timesteps=ts, context=ctx.float(),
                                       down_block_additional_residuals=dh,
                                       mid_block_additional_residual=mh)
                z, _ = scheduler.step(noise_pred, t, z)
        latents.append(z.clone())
        del z
        torch.cuda.empty_cache()

    # Phase 2: Score and filter
    scores = []
    for z in latents:
        z_dec = utils.to_vae_latent_trick((z / scale_factor).squeeze(0).cpu())
        img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
        img_np = utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu()).squeeze(0).numpy().clip(0, 1)
        sc = score_composite(img_np, source_np, source_segm)
        scores.append(sc)
        del z_dec, img
        torch.cuda.empty_cache()

    ranked = sorted(range(n_initial), key=lambda i: scores[i], reverse=True)
    survivor_idx = ranked[:n_survivors]
    survivor_latents = [latents[i] for i in survivor_idx]
    del latents
    torch.cuda.empty_cache()

    # Phase 3: Survivors complete
    completed = []
    for z in survivor_latents:
        for t in timesteps[checkpoint_step:]:
            with autocast(enabled=True):
                ts = torch.tensor([t]).to(device)
                dh, mh = controlnet(x=z.float(), timesteps=ts,
                                    context=ctx, controlnet_cond=cnet_cond.float())
                noise_pred = diffusion(x=z.float(), timesteps=ts, context=ctx.float(),
                                       down_block_additional_residuals=dh,
                                       mid_block_additional_residual=mh)
                z, _ = scheduler.step(noise_pred, t, z)
        z_dec = utils.to_vae_latent_trick((z / scale_factor).squeeze(0).cpu())
        img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
        img_np = utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu()).squeeze(0).numpy().clip(0, 1)
        completed.append(img_np)
        del z, z_dec, img
        torch.cuda.empty_cache()

    # Phase 4: Weighted fusion
    final_scores = [score_composite(c, source_np, source_segm) for c in completed]
    weights = np.array(final_scores)
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()
    result = sum(w * c for w, c in zip(weights, completed))
    return result.clip(0, 1)


@torch.no_grad()
def generate_cci_selection(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context, device,
    source_np, source_segm, row,
    scale_factor=1.0, n_candidates=8, las_m=3, num_inference_steps=50):
    """CCI selection: generate candidates with LAS, pick by cycle consistency."""
    # Forward: generate N candidates with LAS
    candidates = generate_n_las_candidates(
        autoencoder, diffusion, controlnet,
        starting_z, starting_a, context, device,
        scale_factor=scale_factor, n_candidates=n_candidates, las_m=las_m,
        num_inference_steps=num_inference_steps)

    # Build reverse context
    rev_context = torch.tensor([
        row['starting_age'],
        row['sex'],
        row.get('starting_diagnosis', 2),
        row['starting_cerebral_cortex'],
        row['starting_hippocampus'],
        row['starting_amygdala'],
        row['starting_cerebral_white_matter'],
        row['starting_lateral_ventricle'],
    ], dtype=torch.float32)

    cci_scores = []
    for i, fwd_np in enumerate(candidates):
        # Encode the forward prediction
        enc_loader = transforms.Compose([
            transforms.EnsureChannelFirst(channel_dim='no_channel'),
            transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        ])
        fwd_tensor = enc_loader(torch.from_numpy(fwd_np).float()).unsqueeze(0).to(device)
        fwd_z = autoencoder.encode(fwd_tensor)[0]
        fwd_z = transforms.DivisiblePad(k=4, mode='constant')(fwd_z.squeeze(0))

        # Reverse generation
        rev_age = row.get('followup_age', row.get('starting_age', 0.7))
        rev_img = generate_las(
            autoencoder, diffusion, controlnet,
            fwd_z * scale_factor, rev_age, rev_context, device,
            scale_factor=scale_factor, las_m=1,  # LAS m=1 for speed
            num_inference_steps=num_inference_steps, seed=42 + i)

        # Cycle error in hippocampus
        min_shape = tuple(min(a, b, c) for a, b, c in
                          zip(source_np.shape, rev_img.shape, source_segm.shape))
        src = source_np[:min_shape[0], :min_shape[1], :min_shape[2]]
        rev = rev_img[:min_shape[0], :min_shape[1], :min_shape[2]]
        seg = source_segm[:min_shape[0], :min_shape[1], :min_shape[2]]

        h_mask = create_roi_mask(seg, HIPPOCAMPUS_LABELS)
        if h_mask.sum() > 0:
            cycle_h_ssim = compute_region_metrics(rev, src, h_mask)['ssim']
            if np.isnan(cycle_h_ssim):
                cycle_h_ssim = 0.5
            cycle_h_mae = np.abs(rev[h_mask] - src[h_mask]).mean()
            # Also measure forward quality
            fwd_quality = score_composite(fwd_np, source_np, source_segm)
            cci_score = (0.35 * cycle_h_ssim +
                        0.25 * max(0, 1 - cycle_h_mae * 10) +
                        0.40 * fwd_quality)
        else:
            cci_score = score_overall_ssim(fwd_np, source_np)

        cci_scores.append(cci_score)
        del fwd_tensor, fwd_z
        torch.cuda.empty_cache()

    best_idx = np.argmax(cci_scores)
    return candidates[best_idx]


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
    """Load segmentation with given prefix (starting_ or followup_)."""
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

    log(f"═══ Hippocampus Improvement V2 ═══")
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
                source_segm = eval_segm  # fallback
            if source_segm is None:
                source_segm = np.zeros_like(source_np, dtype=np.int32)
                log(f"  WARNING: No segmentation found")

            # For methods that generate N candidates with LAS, cache them
            cached_candidates_16 = None
            cached_candidates_32 = None

            for method in methods:
                t0 = time.time()
                log(f"  Method {method}...")

                try:
                    if method == 'A':
                        pred_np = generate_las(
                            ae, dm, cn, start_z, start_a, context, device,
                            scale_factor=SCALE_FACTOR, las_m=LAS_M)

                    elif method == 'B':
                        # Best-of-16 with LAS m=3, hippo scoring
                        if cached_candidates_16 is None:
                            cached_candidates_16 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=16, las_m=LAS_M)
                        scores = [score_hippo_ssim(c, source_np, source_segm)
                                  for c in cached_candidates_16]
                        pred_np = cached_candidates_16[np.argmax(scores)]
                        log(f"    Best hippo score: {max(scores):.4f}")

                    elif method == 'D':
                        # Hippo-Weighted ET-BoN
                        pred_np = generate_et_bon_hippo(
                            ae, dm, cn, start_z, start_a, context, device,
                            source_np, source_segm,
                            scale_factor=SCALE_FACTOR,
                            n_initial=16, n_survivors=6, checkpoint_step=10)

                    elif method == 'E':
                        # CCI Selection (N=6 for speed since each needs reverse pass)
                        pred_np = generate_cci_selection(
                            ae, dm, cn, start_z, start_a, context, device,
                            source_np, source_segm, row,
                            scale_factor=SCALE_FACTOR, n_candidates=6, las_m=LAS_M)

                    elif method == 'F':
                        # Region-Adaptive Fusion from N=16 LAS candidates
                        if cached_candidates_16 is None:
                            cached_candidates_16 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=16, las_m=LAS_M)
                        # Get best-hippo and best-overall from cached candidates
                        h_scores = [score_hippo_ssim(c, source_np, source_segm)
                                    for c in cached_candidates_16]
                        o_scores = [score_overall_ssim(c, source_np)
                                    for c in cached_candidates_16]
                        best_h = np.argmax(h_scores)
                        best_o = np.argmax(o_scores)
                        if best_h == best_o:
                            pred_np = cached_candidates_16[best_h]
                        else:
                            from scipy.ndimage import gaussian_filter
                            min_shape = tuple(
                                min(cached_candidates_16[best_h].shape[d],
                                    cached_candidates_16[best_o].shape[d],
                                    source_segm.shape[d]) for d in range(3))
                            h_img = cached_candidates_16[best_h][:min_shape[0], :min_shape[1], :min_shape[2]]
                            o_img = cached_candidates_16[best_o][:min_shape[0], :min_shape[1], :min_shape[2]]
                            seg = source_segm[:min_shape[0], :min_shape[1], :min_shape[2]]
                            roi = create_roi_mask(seg, HIPPOCAMPUS_LABELS + AMYGDALA_LABELS).astype(np.float32)
                            blend = gaussian_filter(roi, sigma=3.0)
                            blend = blend / max(blend.max(), 1e-8)
                            pred_np = (o_img * (1 - blend) + h_img * blend).clip(0, 1)

                    elif method == 'G':
                        # Best-of-16 with overall SSIM scoring + LAS m=3
                        if cached_candidates_16 is None:
                            cached_candidates_16 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=16, las_m=LAS_M)
                        scores = [score_overall_ssim(c, source_np)
                                  for c in cached_candidates_16]
                        pred_np = cached_candidates_16[np.argmax(scores)]
                        log(f"    Best overall score: {max(scores):.4f}")

                    elif method == 'H':
                        # Best-of-32 with overall SSIM scoring + LAS m=3
                        if cached_candidates_32 is None:
                            cached_candidates_32 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=32, las_m=LAS_M)
                        scores = [score_overall_ssim(c, source_np)
                                  for c in cached_candidates_32]
                        pred_np = cached_candidates_32[np.argmax(scores)]
                        log(f"    Best overall score: {max(scores):.4f}")

                    elif method == 'I':
                        # Weighted fusion of top-5 from N=16 LAS candidates
                        if cached_candidates_16 is None:
                            cached_candidates_16 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=16, las_m=LAS_M)
                        scores = [score_composite(c, source_np, source_segm)
                                  for c in cached_candidates_16]
                        top_k = 5
                        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                        top_indices = ranked[:top_k]
                        top_scores = np.array([scores[i] for i in top_indices])
                        top_scores = top_scores - top_scores.min() + 1e-6
                        weights = top_scores / top_scores.sum()
                        pred_np = sum(w * cached_candidates_16[i]
                                      for w, i in zip(weights, top_indices)).clip(0, 1)
                        log(f"    Top-5 weights: {[f'{w:.3f}' for w in weights]}")

                    elif method == 'O':
                        # Oracle: Best-of-16 using GT hippocampus SSIM (upper bound)
                        if cached_candidates_16 is None:
                            cached_candidates_16 = generate_n_las_candidates(
                                ae, dm, cn, start_z, start_a, context, device,
                                scale_factor=SCALE_FACTOR, n_candidates=16, las_m=LAS_M)
                        if eval_segm is not None:
                            oracle_scores = []
                            for c in cached_candidates_16:
                                m = compute_full_metrics(c, gt_np, eval_segm)
                                oracle_scores.append(m.get('hippocampus_ssim', 0))
                            pred_np = cached_candidates_16[np.argmax(oracle_scores)]
                            log(f"    Oracle best hippo SSIM: {max(oracle_scores):.4f}")
                            log(f"    Oracle worst hippo SSIM: {min(oracle_scores):.4f}")
                        else:
                            pred_np = cached_candidates_16[0]

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

            # Clear cached candidates
            del cached_candidates_16, cached_candidates_32
            cached_candidates_16 = None
            cached_candidates_32 = None
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
    log(f"EXPERIMENT SUMMARY (V2)")
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
        'las_m': LAS_M,
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
    parser.add_argument('--methods', default='A,B,G,I,O', type=str)
    parser.add_argument('--max-pairs', default=10, type=int)
    parser.add_argument('--csv-path', default=CSV_PATH, type=str)
    parser.add_argument('--output-dir', default=OUTPUT_DIR, type=str)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    run_experiment(args)
