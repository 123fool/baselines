"""
Section 33: Hippocampus SSIM Improvement Experiment
====================================================
多种推理时方法对比，目标：将海马体 SSIM 从 0.8409 提升至 0.9+

Methods:
  A: Baseline LAS m=3 (参考)
  B: Best-of-N Hippo Selection (N=16) — 生成16个候选，选海马体最优
  C: Best-of-N Hippo Selection (N=32) — 更大候选池
  D: Hippo-Weighted ET-BoN — 修改ET-BoN评分函数强调海马体
  E: CCI Selection (N=8) — 循环一致性引导选择
  F: Region-Adaptive Fusion — 海马体区域融合
  G: Guided Denoising — 海马体引导去噪

Usage:
  cd /home/wangchong/data/fwz/code/33_hippocampus/scripts
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python hippocampus_improvement.py --methods A,B,D --max-pairs 10
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

# Fix torch.load
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
OUTPUT_DIR = "/home/wangchong/data/fwz/output/33_hippocampus"

SCALE_FACTOR = 1.0469
LAS_M = 3

# ── SynthSeg ROI labels ──
HIPPOCAMPUS_LABELS = [17, 53]
AMYGDALA_LABELS = [18, 54]


# ═══════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════

def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def create_roi_mask(segm_data, labels):
    mask = np.zeros_like(segm_data, dtype=bool)
    for label in labels:
        mask |= (segm_data.round() == label)
    return mask


def get_roi_bbox(mask):
    """Get bounding box slices for a binary mask."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return tuple(slice(c.min(), c.max() + 1) for c in coords)


def compute_region_metrics(pred, target, mask):
    """Compute SSIM and MAE for a masked region (using bounding box)."""
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
    """Load segmentation and resample to match prediction shape."""
    segm_tensor = torch.from_numpy(
        nib.load(segm_path).get_fdata().astype(np.float32)
    ).unsqueeze(0)
    resample = transforms.Compose([
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'),
    ])
    segm = resample(segm_tensor).squeeze(0).numpy().round().astype(np.int32)
    # Crop to match target shape
    min_shape = tuple(min(a, b) for a, b in zip(segm.shape, target_shape))
    segm = segm[:min_shape[0], :min_shape[1], :min_shape[2]]
    return segm


def _center_crop(vol, target_shape):
    starts = [(s - t) // 2 for s, t in zip(vol.shape, target_shape)]
    slices = tuple(slice(max(0, s), max(0, s) + t) for s, t in zip(starts, target_shape))
    return vol[slices]


def compute_full_metrics(pred_np, gt_np, segm=None):
    """Compute all metrics: overall + hippocampus + amygdala."""
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
#  Hippocampus Scoring Functions (for candidate selection)
# ═══════════════════════════════════════════════════════════════

def hippo_source_ssim(generated_np, source_np, source_segm):
    """Hippocampus SSIM between generated image and source image."""
    hipp_mask = create_roi_mask(source_segm, HIPPOCAMPUS_LABELS)
    if hipp_mask.sum() == 0:
        return 0.5
    min_shape = tuple(min(a, b) for a, b in zip(generated_np.shape, source_np.shape))
    gen = generated_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    src = source_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    mask = hipp_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
    return compute_region_metrics(gen, src, mask)['ssim']


def hippo_structural_score(generated_np, source_np, source_segm):
    """Combined hippocampus structural quality score."""
    hipp_mask = create_roi_mask(source_segm, HIPPOCAMPUS_LABELS)
    if hipp_mask.sum() == 0:
        return 0.5
    min_shape = tuple(min(a, b) for a, b in zip(generated_np.shape, source_np.shape))
    gen = generated_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    src = source_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    mask = hipp_mask[:min_shape[0], :min_shape[1], :min_shape[2]]

    # SSIM in hippocampus bounding box
    h_ssim = compute_region_metrics(gen, src, mask)['ssim']
    if np.isnan(h_ssim):
        h_ssim = 0.5

    # Intensity match in hippocampus voxels
    h_gen = gen[mask]
    h_src = src[mask]
    if len(h_gen) > 0 and len(h_src) > 0:
        intensity_match = max(0, 1.0 - abs(h_gen.mean() - h_src.mean()) * 5)
        std_match = max(0, 1.0 - abs(h_gen.std() - h_src.std()) * 5)
    else:
        intensity_match = 0.5
        std_match = 0.5

    # Smoothness in hippocampus bounding box
    bbox = get_roi_bbox(mask)
    if bbox is not None:
        roi = gen[bbox]
        gx = np.abs(np.diff(roi, axis=0)).mean() if roi.shape[0] > 1 else 0
        gy = np.abs(np.diff(roi, axis=1)).mean() if roi.shape[1] > 1 else 0
        gz = np.abs(np.diff(roi, axis=2)).mean() if roi.shape[2] > 1 else 0
        smoothness = max(0, 1.0 - (gx + gy + gz) / 3.0 * 20)
    else:
        smoothness = 0.5

    # Overall SSIM as regularizer
    overall_ssim = float(ssim_fn(gen.clip(0,1), src.clip(0,1), data_range=1.0))

    # Weighted combination favoring hippocampus
    score = (0.40 * h_ssim +
             0.20 * intensity_match +
             0.10 * std_match +
             0.10 * smoothness +
             0.20 * overall_ssim)
    return score


def overall_composite_score(generated_np, source_np):
    """Overall quality composite (from ET-BoN)."""
    s_ssim = float(ssim_fn(generated_np.clip(0,1), source_np.clip(0,1), data_range=1.0))
    g_mask = generated_np > 0.01
    s_mask = source_np > 0.01
    if g_mask.sum() < 100 or s_mask.sum() < 100:
        return s_ssim
    g_mean, s_mean = generated_np[g_mask].mean(), source_np[s_mask].mean()
    g_std, s_std = generated_np[g_mask].std(), source_np[s_mask].std()
    intens = 0.6 * max(0, 1 - abs(float(g_mean - s_mean)) * 4) + \
             0.4 * max(0, 1 - abs(float(g_std - s_std)) * 4)
    gr = g_mask.sum() / max(generated_np.size, 1)
    sr = s_mask.sum() / max(source_np.size, 1)
    cover = max(0, 1.0 - abs(float(gr - sr)) / max(float(sr), 1e-6) * 5)
    return 0.45 * s_ssim + 0.25 * intens + 0.15 * cover + 0.15 * 0.7  # smoothness approx


# ═══════════════════════════════════════════════════════════════
#  Core Sampling Functions
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_single(autoencoder, diffusion, controlnet, starting_z, starting_a,
                    context, device, scale_factor=1.0, las_m=1,
                    num_inference_steps=50, seed=None):
    """Generate a single MRI using standard pipeline."""
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
def generate_n_candidates(autoencoder, diffusion, controlnet,
                          starting_z, starting_a, context, device,
                          scale_factor=1.0, n_candidates=16,
                          num_inference_steps=50):
    """Generate N independent candidate images from different seeds."""
    candidates = []
    for i in range(n_candidates):
        seed = 42 + i * 1000 + int(time.time() * 100) % 10000
        img = generate_single(
            autoencoder, diffusion, controlnet,
            starting_z, starting_a, context, device,
            scale_factor=scale_factor, las_m=1,
            num_inference_steps=num_inference_steps, seed=seed)
        candidates.append(img)
        torch.cuda.empty_cache()
    return candidates


@torch.no_grad()
def generate_et_bon_hippo_weighted(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context, device,
    source_np, source_segm,
    scale_factor=1.0,
    n_initial=16, n_survivors=6, checkpoint_step=10,
    num_inference_steps=50):
    """
    ET-BoN with hippocampus-weighted scoring.
    - Phase 1: N candidates run until checkpoint
    - Phase 2: Score with hippocampus emphasis, keep top-K
    - Phase 3: Survivors complete denoising
    - Phase 4: Weighted fusion emphasizing hippocampus quality
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

    # Phase 1: All N candidates run until checkpoint
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

    # Phase 2: Score intermediate latents with hippocampus emphasis
    scores = []
    for z in latents:
        z_dec = utils.to_vae_latent_trick((z / scale_factor).squeeze(0).cpu())
        img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
        img_np = utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu()).squeeze(0).numpy().clip(0, 1)

        # Hippocampus-weighted score
        h_score = hippo_structural_score(img_np, source_np, source_segm)
        o_score = overall_composite_score(img_np, source_np)
        # 60% hippocampus + 40% overall
        combined = 0.60 * h_score + 0.40 * o_score
        scores.append(combined)
        del z_dec, img
        torch.cuda.empty_cache()

    ranked = sorted(range(n_initial), key=lambda i: scores[i], reverse=True)
    survivor_indices = ranked[:n_survivors]
    survivor_latents = [latents[i] for i in survivor_indices]
    survivor_scores = [scores[i] for i in survivor_indices]
    del latents
    torch.cuda.empty_cache()

    # Phase 3: Survivors complete remaining steps
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

    # Phase 4: Hippocampus-weighted fusion
    final_scores = []
    for img_np in completed:
        h_score = hippo_structural_score(img_np, source_np, source_segm)
        o_score = overall_composite_score(img_np, source_np)
        final_scores.append(0.60 * h_score + 0.40 * o_score)

    weights = np.array(final_scores)
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()
    result = sum(w * c for w, c in zip(weights, completed))

    return result.clip(0, 1)


@torch.no_grad()
def generate_cci_selection(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context, device,
    source_np, source_segm,
    row,  # need row for reverse context
    scale_factor=1.0, n_candidates=8, num_inference_steps=50):
    """
    CCI (Cycle Consistency Inference) Selection.
    For each candidate:
      1. Forward: predict followup from starting
      2. Encode followup prediction
      3. Reverse: predict back to starting from predicted followup
      4. Measure cycle consistency error in hippocampus region
      5. Select candidate with lowest cycle error
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205, clip_sample=False)

    sz = starting_z.unsqueeze(0).to(device)
    age_vol = (torch.tensor([starting_a]).view(1,1,1,1,1)
               .expand(1, 1, *sz.shape[-3:]).to(device))
    cnet_cond = torch.cat([sz, age_vol], dim=1)
    ctx = context.unsqueeze(0).unsqueeze(0).to(device)

    # Build reverse context (starting -> followup mapping reversed)
    # For reverse: use starting conditions as target
    rev_context = torch.tensor([
        row['starting_age'],
        row['sex'],
        row['starting_diagnosis'],
        row['starting_cerebral_cortex'],
        row['starting_hippocampus'],
        row['starting_amygdala'],
        row['starting_cerebral_white_matter'],
        row['starting_lateral_ventricle'],
    ], dtype=torch.float32)

    candidates_forward = []
    cci_scores = []

    for i in range(n_candidates):
        # Forward generation
        seed = 42 + i * 2000
        torch.manual_seed(seed)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        z = torch.randn(1, *sz.shape[1:]).to(device)
        for t in scheduler.timesteps:
            with autocast(enabled=True):
                ts = torch.tensor([t]).to(device)
                dh, mh = controlnet(x=z.float(), timesteps=ts,
                                    context=ctx, controlnet_cond=cnet_cond.float())
                noise_pred = diffusion(x=z.float(), timesteps=ts, context=ctx.float(),
                                       down_block_additional_residuals=dh,
                                       mid_block_additional_residual=mh)
                z, _ = scheduler.step(noise_pred, t, z)

        # Decode forward prediction
        z_dec = utils.to_vae_latent_trick((z / scale_factor).squeeze(0).cpu())
        fwd_img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
        fwd_np = utils.to_mni_space_1p5mm_trick(fwd_img.squeeze(0).cpu()).squeeze(0).numpy().clip(0, 1)
        candidates_forward.append(fwd_np)

        # Encode forward prediction for reverse pass
        loader = transforms.Compose([
            transforms.EnsureChannelFirst(channel_dim='no_channel'),
            transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        ])
        fwd_tensor = loader(torch.from_numpy(fwd_np).float()).unsqueeze(0).to(device)
        fwd_z = autoencoder.encode(fwd_tensor)[0]
        fwd_z = transforms.DivisiblePad(k=4, mode='constant')(fwd_z.squeeze(0))



        # Reverse: predict back to starting
        rev_age = row.get('followup_age', row.get('starting_age', 0.7))
        age_vol_rev = (torch.tensor([rev_age]).view(1,1,1,1,1)
                       .expand(1, 1, *fwd_z.unsqueeze(0).shape[-3:]).to(device))
        cnet_cond_rev = torch.cat([fwd_z.unsqueeze(0) * scale_factor, age_vol_rev], dim=1)
        ctx_rev = rev_context.unsqueeze(0).unsqueeze(0).to(device)

        torch.manual_seed(seed + 1)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        z_rev = torch.randn(1, *sz.shape[1:]).to(device)

        for t in scheduler.timesteps:
            with autocast(enabled=True):
                ts = torch.tensor([t]).to(device)
                dh, mh = controlnet(x=z_rev.float(), timesteps=ts,
                                    context=ctx_rev, controlnet_cond=cnet_cond_rev.float())
                noise_pred = diffusion(x=z_rev.float(), timesteps=ts, context=ctx_rev.float(),
                                       down_block_additional_residuals=dh,
                                       mid_block_additional_residual=mh)
                z_rev, _ = scheduler.step(noise_pred, t, z_rev)

        z_rev_dec = utils.to_vae_latent_trick((z_rev / scale_factor).squeeze(0).cpu())
        rev_img = autoencoder.decode_stage_2_outputs(z_rev_dec.unsqueeze(0).to(device))
        rev_np = utils.to_mni_space_1p5mm_trick(rev_img.squeeze(0).cpu()).squeeze(0).numpy().clip(0, 1)

        # Compute cycle consistency in hippocampus region
        min_shape = tuple(min(a, b, c) for a, b, c in
                          zip(source_np.shape, rev_np.shape, source_segm.shape))
        src_crop = source_np[:min_shape[0], :min_shape[1], :min_shape[2]]
        rev_crop = rev_np[:min_shape[0], :min_shape[1], :min_shape[2]]
        segm_crop = source_segm[:min_shape[0], :min_shape[1], :min_shape[2]]

        hipp_mask = create_roi_mask(segm_crop, HIPPOCAMPUS_LABELS)
        if hipp_mask.sum() > 0:
            # Lower cycle error = better consistency
            cycle_error_hippo = np.abs(src_crop[hipp_mask] - rev_crop[hipp_mask]).mean()
            cycle_ssim_hippo = compute_region_metrics(rev_crop, src_crop, hipp_mask)['ssim']
            if np.isnan(cycle_ssim_hippo):
                cycle_ssim_hippo = 0.5

            # Also consider forward quality
            h_quality = hippo_structural_score(fwd_np, source_np, source_segm)

            # CCI score: high cycle consistency + high hippocampus quality
            cci_score = (0.40 * cycle_ssim_hippo +
                        0.30 * (1.0 - min(cycle_error_hippo * 10, 1.0)) +
                        0.30 * h_quality)
        else:
            cci_score = overall_composite_score(fwd_np, source_np)

        cci_scores.append(cci_score)

        del z, z_dec, fwd_img, fwd_z, fwd_tensor
        del z_rev, z_rev_dec, rev_img
        torch.cuda.empty_cache()

    # Select best candidate
    best_idx = np.argmax(cci_scores)
    return candidates_forward[best_idx]


@torch.no_grad()
def generate_region_adaptive_fusion(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context, device,
    source_np, source_segm,
    scale_factor=1.0, n_candidates=16, num_inference_steps=50):
    """
    Region-Adaptive Fusion: generate N candidates, then:
    - For hippocampus region: use the candidate with best hippo SSIM
    - For rest of brain: use the candidate with best overall SSIM
    - Gaussian-blended transition between regions
    """
    candidates = generate_n_candidates(
        autoencoder, diffusion, controlnet,
        starting_z, starting_a, context, device,
        scale_factor=scale_factor, n_candidates=n_candidates,
        num_inference_steps=num_inference_steps)

    # Score all candidates
    hippo_scores = []
    overall_scores = []
    for img_np in candidates:
        h = hippo_structural_score(img_np, source_np, source_segm)
        o = overall_composite_score(img_np, source_np)
        hippo_scores.append(h)
        overall_scores.append(o)

    best_hippo_idx = np.argmax(hippo_scores)
    best_overall_idx = np.argmax(overall_scores)

    if best_hippo_idx == best_overall_idx:
        return candidates[best_hippo_idx]

    # Create blended result
    min_shape = tuple(min(candidates[best_hippo_idx].shape[d],
                          candidates[best_overall_idx].shape[d],
                          source_segm.shape[d]) for d in range(3))

    hippo_img = candidates[best_hippo_idx][:min_shape[0], :min_shape[1], :min_shape[2]]
    overall_img = candidates[best_overall_idx][:min_shape[0], :min_shape[1], :min_shape[2]]
    segm = source_segm[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Create hippocampus + amygdala mask with Gaussian blurring for smooth transition
    roi_mask = create_roi_mask(segm, HIPPOCAMPUS_LABELS + AMYGDALA_LABELS).astype(np.float32)

    # Dilate mask with Gaussian blur for smooth blending
    from scipy.ndimage import gaussian_filter
    blend_mask = gaussian_filter(roi_mask, sigma=3.0)
    blend_mask = blend_mask / max(blend_mask.max(), 1e-8)

    # Blend: hippocampus region from best-hippo, rest from best-overall
    result = overall_img * (1 - blend_mask) + hippo_img * blend_mask

    return result.clip(0, 1)


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def load_models(device):
    ae = init_autoencoder(AEKL_CKPT).to(device).eval()
    dm = init_latent_diffusion(DIFF_CKPT).to(device).eval()
    cn = init_controlnet(CNET_CKPT).to(device).eval()
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

    # Load and encode starting image
    start_data = loader({'image_path': row['starting_image']})
    start_img = start_data['image'].unsqueeze(0).to(device)
    start_z = ae.encode(start_img)[0]
    start_z = transforms.DivisiblePad(k=4, mode='constant')(start_z.squeeze(0))

    # Decode source for scoring
    source_np = start_data['image'].squeeze(0).numpy().clip(0, 1)
    # Resize to match output
    source_np = transforms.ResizeWithPadOrCrop(
        spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'
    )(start_data['image']).squeeze(0).numpy().clip(0, 1)

    # Load follow-up ground truth
    follow_data = loader({'image_path': row['followup_image']})
    follow_np = transforms.ResizeWithPadOrCrop(
        spatial_size=const.INPUT_SHAPE_1p5mm, mode='minimum'
    )(follow_data['image']).squeeze(0).numpy().clip(0, 1)

    # Build context
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


def get_segmentation(row, target_shape):
    """Try to load follow-up segmentation, fall back to starting."""
    for key in ['followup_segm_path', 'followup_segm',
                'starting_segm_path', 'starting_segm']:
        if key in row and pd.notna(row[key]) and os.path.exists(str(row[key])):
            return load_and_resample_segm(row[key], target_shape)
    return None


def get_source_segmentation(row, target_shape):
    """Load starting/source segmentation for scoring."""
    for key in ['starting_segm_path', 'starting_segm',
                'followup_segm_path', 'followup_segm']:
        if key in row and pd.notna(row[key]) and os.path.exists(str(row[key])):
            return load_and_resample_segm(row[key], target_shape)
    return None


# ═══════════════════════════════════════════════════════════════
#  Main Experiment Runner
# ═══════════════════════════════════════════════════════════════

def run_experiment(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "hippocampus_experiment.log")

    methods = [m.strip().upper() for m in args.methods.split(',')]

    def log(msg):
        line = f"{ts()} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Hippocampus Improvement Experiment")
    log(f"Methods: {methods}")
    log(f"Max pairs: {args.max_pairs}")
    log(f"Device: {device}")

    ae, dm, cn = load_models(device)
    log("Models loaded.")

    df = pd.read_csv(args.csv_path)
    # Try test split, fall back to full
    if 'split' in df.columns:
        test_df = df[df.split == 'test']
        if len(test_df) == 0:
            test_df = df[df.split == 'valid']
        if len(test_df) == 0:
            test_df = df
    else:
        test_df = df

    n_pairs = min(len(test_df), args.max_pairs)
    log(f"CSV has {len(test_df)} test pairs, using {n_pairs}")

    # Results storage
    all_results = {method: [] for method in methods}

    for idx in range(n_pairs):
        row = test_df.iloc[idx].to_dict()
        subj = row.get('subject_id', row.get('ptid', f'pair_{idx}'))
        log(f"\n{'='*60}")
        log(f"Pair {idx+1}/{n_pairs}: {subj}")

        try:
            with torch.no_grad():
                start_z, start_a, context, source_np, gt_np = prepare_pair(row, ae, device)

            # Load segmentation for evaluation
            eval_segm = get_segmentation(row, gt_np.shape)
            source_segm = get_source_segmentation(row, source_np.shape)

            if source_segm is None:
                log(f"  WARNING: No segmentation found, skipping hippo-specific methods")
                source_segm = np.zeros_like(source_np, dtype=np.int32)

            for method in methods:
                t0 = time.time()
                log(f"  Method {method}...", )

                try:
                    if method == 'A':
                        # Baseline LAS m=3
                        pred_np = generate_single(
                            ae, dm, cn, start_z, start_a, context, device,
                            scale_factor=SCALE_FACTOR, las_m=LAS_M,
                            num_inference_steps=50)

                    elif method == 'B':
                        # Best-of-N Hippo Selection (N=16)
                        candidates = generate_n_candidates(
                            ae, dm, cn, start_z, start_a, context, device,
                            scale_factor=SCALE_FACTOR, n_candidates=16,
                            num_inference_steps=50)
                        scores = [hippo_structural_score(c, source_np, source_segm) for c in candidates]
                        best_idx = np.argmax(scores)
                        pred_np = candidates[best_idx]
                        log(f"    Best candidate: {best_idx}, score: {scores[best_idx]:.4f}")

                    elif method == 'C':
                        # Best-of-N Hippo Selection (N=32)
                        candidates = generate_n_candidates(
                            ae, dm, cn, start_z, start_a, context, device,
                            scale_factor=SCALE_FACTOR, n_candidates=32,
                            num_inference_steps=50)
                        scores = [hippo_structural_score(c, source_np, source_segm) for c in candidates]
                        best_idx = np.argmax(scores)
                        pred_np = candidates[best_idx]
                        log(f"    Best candidate: {best_idx}, score: {scores[best_idx]:.4f}")

                    elif method == 'D':
                        # Hippo-Weighted ET-BoN
                        pred_np = generate_et_bon_hippo_weighted(
                            ae, dm, cn, start_z, start_a, context, device,
                            source_np, source_segm,
                            scale_factor=SCALE_FACTOR,
                            n_initial=16, n_survivors=6, checkpoint_step=10)

                    elif method == 'E':
                        # CCI Selection (N=8)
                        pred_np = generate_cci_selection(
                            ae, dm, cn, start_z, start_a, context, device,
                            source_np, source_segm, row,
                            scale_factor=SCALE_FACTOR, n_candidates=8)

                    elif method == 'F':
                        # Region-Adaptive Fusion
                        pred_np = generate_region_adaptive_fusion(
                            ae, dm, cn, start_z, start_a, context, device,
                            source_np, source_segm,
                            scale_factor=SCALE_FACTOR, n_candidates=16)

                    else:
                        log(f"    Unknown method {method}, skipping")
                        continue

                    elapsed = time.time() - t0

                    # Compute metrics
                    metrics = compute_full_metrics(pred_np, gt_np, eval_segm)
                    metrics['method'] = method
                    metrics['pair_idx'] = idx
                    metrics['subject'] = subj
                    metrics['time_sec'] = round(elapsed, 2)

                    all_results[method].append(metrics)

                    h_ssim = metrics.get('hippocampus_ssim', float('nan'))
                    o_ssim = metrics.get('overall_ssim', float('nan'))
                    log(f"    Overall SSIM={o_ssim:.4f}, Hippo SSIM={h_ssim:.4f}, "
                        f"Time={elapsed:.1f}s")

                except Exception as e:
                    log(f"    ERROR in method {method}: {e}")
                    traceback.print_exc()
                    all_results[method].append({
                        'method': method, 'pair_idx': idx,
                        'subject': subj, 'error': str(e)
                    })

                torch.cuda.empty_cache()

        except Exception as e:
            log(f"  ERROR preparing pair: {e}")
            traceback.print_exc()
            for method in methods:
                all_results[method].append({
                    'method': method, 'pair_idx': idx,
                    'subject': subj, 'error': str(e)
                })

    # ── Summary ──
    log(f"\n{'='*80}")
    log(f"EXPERIMENT SUMMARY")
    log(f"{'='*80}")

    summary = {}
    for method in methods:
        valid = [r for r in all_results[method] if 'error' not in r]
        if not valid:
            log(f"Method {method}: No valid results")
            continue

        method_df = pd.DataFrame(valid)
        summary[method] = {}

        log(f"\nMethod {method} ({len(valid)} valid pairs):")
        for col in ['overall_ssim', 'overall_mae', 'hippocampus_ssim',
                     'hippocampus_mae', 'roi_ssim', 'roi_mae', 'time_sec']:
            if col in method_df.columns:
                mean_val = method_df[col].mean()
                std_val = method_df[col].std()
                log(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
                summary[method][col] = f"{mean_val:.4f} ± {std_val:.4f}"
                summary[method][f'{col}_mean'] = round(float(mean_val), 6)

        # Save per-pair results
        csv_path = os.path.join(args.output_dir, f'results_method_{method}.csv')
        method_df.to_csv(csv_path, index=False)
        log(f"  Saved to {csv_path}")

    # Save overall summary
    summary['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'n_pairs': n_pairs,
        'methods': methods,
        'csv_path': args.csv_path,
        'scale_factor': SCALE_FACTOR,
    }
    json_path = os.path.join(args.output_dir, 'experiment_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to {json_path}")

    # Comparison table
    log(f"\n{'='*80}")
    log(f"COMPARISON TABLE")
    log(f"{'='*80}")
    log(f"{'Method':<10} {'Overall SSIM':>15} {'Hippo SSIM':>15} {'Hippo MAE':>12} {'Time(s)':>10}")
    log(f"{'-'*62}")
    for method in methods:
        if method in summary and 'overall_ssim_mean' in summary[method]:
            s = summary[method]
            log(f"{method:<10} {s.get('overall_ssim_mean', 0):>15.4f} "
                f"{s.get('hippocampus_ssim_mean', 0):>15.4f} "
                f"{s.get('hippocampus_mae_mean', 0):>12.4f} "
                f"{s.get('time_sec_mean', 0):>10.1f}")

    log(f"\nExperiment complete: {datetime.now().isoformat()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hippocampus SSIM Improvement Experiment')
    parser.add_argument('--methods', default='A,B,D', type=str,
                        help='Comma-separated methods: A,B,C,D,E,F')
    parser.add_argument('--max-pairs', default=10, type=int)
    parser.add_argument('--csv-path', default=CSV_PATH, type=str)
    parser.add_argument('--output-dir', default=OUTPUT_DIR, type=str)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    run_experiment(args)
