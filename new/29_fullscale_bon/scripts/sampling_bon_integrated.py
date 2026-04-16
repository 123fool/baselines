"""
BoN Weighted Sampling — integrated into BrLP sampling pipeline.

Drop-in replacement for `sample_using_controlnet_and_z()` from brlp.sampling.
Same interface, just adds n_candidates + selection parameters.

Usage:
    from sampling_bon_integrated import sample_bon_weighted
    img = sample_bon_weighted(autoencoder, diffusion, controlnet,
                              starting_z, starting_a, context, device,
                              scale_factor=sf, n_candidates=8)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
if BRLP_SRC not in sys.path:
    sys.path.insert(0, BRLP_SRC)

from brlp import utils, const
from skimage.metrics import structural_similarity as ssim_fn


# ── Quality metrics (inlined to avoid extra imports) ──

def _source_ssim(generated, source):
    """SSIM between generated and starting baseline."""
    return float(ssim_fn(generated, source, data_range=1.0))


def _intensity_score(generated, source):
    """Intensity distribution consistency."""
    g_mask = generated > 0.01
    s_mask = source > 0.01
    if g_mask.sum() < 100 or s_mask.sum() < 100:
        return 0.5
    g_mean, s_mean = generated[g_mask].mean(), source[s_mask].mean()
    g_std, s_std = generated[g_mask].std(), source[s_mask].std()
    mean_sc = max(0, 1.0 - abs(float(g_mean - s_mean)) * 4)
    std_sc = max(0, 1.0 - abs(float(g_std - s_std)) * 4)
    return 0.6 * mean_sc + 0.4 * std_sc


def _coverage_score(generated, source):
    """Brain / background ratio consistency."""
    gr = (generated > 0.01).sum() / max(generated.size, 1)
    sr = (source > 0.01).sum() / max(source.size, 1)
    if sr < 1e-6:
        return 0.5
    return max(0, 1.0 - abs(float(gr - sr)) / float(sr) * 5)


def _smoothness_score(generated):
    """Gradient-based smoothness (low = good)."""
    gx = np.abs(np.diff(generated, axis=0)).mean()
    gy = np.abs(np.diff(generated, axis=1)).mean()
    gz = np.abs(np.diff(generated, axis=2)).mean()
    mg = (gx + gy + gz) / 3.0
    return max(0, 1.0 - float(mg) * 15)


def _latent_norm_score(norm, expected_mean=1.0, expected_std=0.3):
    z = abs(norm - expected_mean) / expected_std
    return max(0, 1.0 - z * 0.3)


def _composite(generated, source, latent_norm=None):
    """Compute composite quality score. Returns float in [0,1]."""
    s_ssim = _source_ssim(generated, source)
    intens = _intensity_score(generated, source)
    cover = _coverage_score(generated, source)
    smooth = _smoothness_score(generated)

    if latent_norm is not None:
        ln = _latent_norm_score(latent_norm)
        score = 0.40 * s_ssim + 0.20 * intens + 0.15 * cover + 0.15 * smooth + 0.10 * ln
    else:
        score = 0.45 * s_ssim + 0.22 * intens + 0.18 * cover + 0.15 * smooth
    return score


# ── Main function ──

@torch.no_grad()
def sample_bon_weighted(
    autoencoder: nn.Module,
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_a: int,
    context: torch.Tensor,
    device: str,
    scale_factor: int = 1,
    n_candidates: int = 8,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    verbose: bool = False,
) -> torch.Tensor:
    """
    BoN Weighted sampling — drop-in replacement for sample_using_controlnet_and_z.

    Generates N candidates, scores each, returns quality-weighted fusion.
    Returns torch.Tensor (same shape as original function).
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule, beta_start=beta_start, beta_end=beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Prepare controlnet spatial condition
    sz = starting_z.unsqueeze(0).to(device)
    age_vol = (torch.tensor([starting_a]).view(1, 1, 1, 1, 1)
               .expand(1, 1, *sz.shape[-3:]).to(device))
    cnet_cond = torch.cat([sz, age_vol], dim=1)
    ctx = context.unsqueeze(0).unsqueeze(0).to(device)

    # Decode baseline for quality scoring
    src_z = utils.to_vae_latent_trick((sz / scale_factor).squeeze(0).cpu())
    src_img = autoencoder.decode_stage_2_outputs(src_z.unsqueeze(0).to(device))
    source_np = (utils.to_mni_space_1p5mm_trick(src_img.squeeze(0).cpu())
                 .squeeze(0).numpy().clip(0, 1))

    # Generate N candidates
    candidates = []
    latent_norms = []

    for _i in range(n_candidates):
        z = torch.randn(1, *sz.shape[1:]).to(device)
        latent_norms.append(float(z.norm().cpu()))

        for t in scheduler.timesteps:
            with autocast(enabled=True):
                ts = torch.tensor([t]).to(device)
                dh, mh = controlnet(
                    x=z.float(), timesteps=ts,
                    context=ctx, controlnet_cond=cnet_cond.float(),
                )
                noise_pred = diffusion(
                    x=z.float(), timesteps=ts, context=ctx.float(),
                    down_block_additional_residuals=dh,
                    mid_block_additional_residual=mh,
                )
                z, _ = scheduler.step(noise_pred, t, z)

        # Decode
        z_dec = utils.to_vae_latent_trick((z / scale_factor).squeeze(0).cpu())
        img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
        img_np = (utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu())
                  .squeeze(0).numpy().clip(0, 1))
        candidates.append(img_np)

    # Score & weight
    scores = [_composite(c, source_np, latent_norms[i]) for i, c in enumerate(candidates)]
    weights = np.array(scores)
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()

    # Weighted fusion
    result = sum(w * c for w, c in zip(weights, candidates))
    return torch.from_numpy(result).float()


@torch.no_grad()
def sample_bon_weighted_with_details(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context, device,
    scale_factor=1, n_candidates=8,
    num_inference_steps=50, verbose=False,
    **kwargs,
) -> dict:
    """Same as sample_bon_weighted but returns detailed info for evaluation."""
    scheduler = DDIMScheduler(
        num_train_timesteps=kwargs.get('num_training_steps', 1000),
        schedule=kwargs.get('schedule', 'scaled_linear_beta'),
        beta_start=kwargs.get('beta_start', 0.0015),
        beta_end=kwargs.get('beta_end', 0.0205),
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    sz = starting_z.unsqueeze(0).to(device)
    age_vol = (torch.tensor([starting_a]).view(1, 1, 1, 1, 1)
               .expand(1, 1, *sz.shape[-3:]).to(device))
    cnet_cond = torch.cat([sz, age_vol], dim=1)
    ctx = context.unsqueeze(0).unsqueeze(0).to(device)

    src_z = utils.to_vae_latent_trick((sz / scale_factor).squeeze(0).cpu())
    src_img = autoencoder.decode_stage_2_outputs(src_z.unsqueeze(0).to(device))
    source_np = (utils.to_mni_space_1p5mm_trick(src_img.squeeze(0).cpu())
                 .squeeze(0).numpy().clip(0, 1))

    candidates = []
    latent_norms = []

    for _i in range(n_candidates):
        z = torch.randn(1, *sz.shape[1:]).to(device)
        latent_norms.append(float(z.norm().cpu()))
        for t in scheduler.timesteps:
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
        img_np = (utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu())
                  .squeeze(0).numpy().clip(0, 1))
        candidates.append(img_np)

    scores = [_composite(c, source_np, latent_norms[i]) for i, c in enumerate(candidates)]
    weights = np.array(scores)
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()
    result = sum(w * c for w, c in zip(weights, candidates))

    return {
        'image': result,
        'source': source_np,
        'candidates': candidates,
        'scores': scores,
        'weights': weights.tolist(),
        'latent_norms': latent_norms,
    }
