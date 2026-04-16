"""
Best-of-N Sampling with Quality-Based Selection (Scheme A).

Replaces the blind LAS averaging in BrLP with intelligent selection:
  1. Generate N independent samples
  2. Score each sample using no-GT quality metrics
  3. Return the best sample (or weighted average of top-K)

Three selection strategies:
  - 'best1':    Return the single best sample
  - 'topk_avg': Average the top-K samples (K = N//3 or at least 2)
  - 'weighted': Weighted average using quality scores as weights
"""

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, SCRIPT_DIR)

from brlp import utils, const
from quality_metrics import composite_score


@torch.no_grad()
def sample_best_of_n(
    autoencoder: nn.Module,
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_a: int,
    context: torch.Tensor,
    device: str,
    scale_factor: int = 1,
    n_candidates: int = 8,
    selection: str = 'best1',
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    source_image: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """Generate N candidate samples and select the best one.

    Unlike the original LAS (blind averaging), this method evaluates each
    sample individually and selects based on quality metrics.

    Args:
        autoencoder, diffusion, controlnet: trained model components.
        starting_z: latent of baseline MRI.
        starting_a: starting age.
        context: conditioning vector.
        device: 'cuda' or 'cpu'.
        scale_factor: latent scale factor.
        n_candidates: number of candidates to generate (N).
        selection: 'best1', 'topk_avg', or 'weighted'.
        source_image: decoded starting MRI (numpy). If None, will be decoded.
        verbose: show progress.

    Returns:
        dict with keys:
            'image':        selected/fused MRI (numpy)
            'scores':       list of score dicts for each candidate
            'selected_idx': index of the selected candidate (or list for topk)
            'all_images':   list of all candidate images (numpy)
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Prepare controlnet condition (same for all candidates)
    starting_z_dev = starting_z.unsqueeze(0).to(device)
    concat_age = (torch.tensor([starting_a])
                  .view(1, 1, 1, 1, 1)
                  .expand(1, 1, *starting_z_dev.shape[-3:])
                  .to(device))
    controlnet_cond = torch.cat([starting_z_dev, concat_age], dim=1).to(device)
    ctx = context.unsqueeze(0).unsqueeze(0).to(device)

    # Decode source image if not provided (for quality metrics)
    if source_image is None:
        src_z = starting_z.unsqueeze(0).to(device) / scale_factor
        src_z = utils.to_vae_latent_trick(src_z.squeeze(0).cpu())
        source_image = autoencoder.decode_stage_2_outputs(
            src_z.unsqueeze(0).to(device)
        )
        source_image = (utils.to_mni_space_1p5mm_trick(source_image.squeeze(0).cpu())
                        .squeeze(0).numpy().clip(0, 1))

    # Generate N candidates sequentially to save GPU memory
    candidates = []
    latent_norms = []

    iterator = range(n_candidates)
    if verbose:
        iterator = tqdm(iterator, desc=f"Generating {n_candidates} candidates")

    for i in iterator:
        # Fresh random noise for each candidate
        z = torch.randn(1, *starting_z_dev.shape[1:]).to(device)
        latent_norms.append(float(z.norm().cpu()))

        ctx_i = ctx.clone()
        cond_i = controlnet_cond.clone()

        for t in scheduler.timesteps:
            with autocast(enabled=True):
                timestep = torch.tensor([t]).to(device)
                down_h, mid_h = controlnet(
                    x=z.float(),
                    timesteps=timestep,
                    context=ctx_i,
                    controlnet_cond=cond_i.float(),
                )
                noise_pred = diffusion(
                    x=z.float(),
                    timesteps=timestep,
                    context=ctx_i.float(),
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h,
                )
                z, _ = scheduler.step(noise_pred, t, z)

        # Decode latent to image
        z_dec = z / scale_factor
        z_dec = utils.to_vae_latent_trick(z_dec.squeeze(0).cpu())
        img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
        img = (utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu())
               .squeeze(0).numpy().clip(0, 1))
        candidates.append(img)

    # Score each candidate
    all_scores = []
    for i, cand in enumerate(candidates):
        scores = composite_score(
            generated=cand,
            source=source_image,
            latent_norm=latent_norms[i],
        )
        all_scores.append(scores)

    composite_values = [s['composite'] for s in all_scores]

    # Selection strategy
    if selection == 'best1':
        best_idx = int(np.argmax(composite_values))
        result_image = candidates[best_idx]
        selected_idx = best_idx

    elif selection == 'topk_avg':
        k = max(2, n_candidates // 3)
        top_indices = np.argsort(composite_values)[-k:]
        result_image = np.mean([candidates[i] for i in top_indices], axis=0)
        selected_idx = top_indices.tolist()

    elif selection == 'weighted':
        weights = np.array(composite_values)
        weights = weights - weights.min() + 1e-6  # shift to positive
        weights = weights / weights.sum()
        result_image = sum(w * c for w, c in zip(weights, candidates))
        selected_idx = list(range(n_candidates))

    else:
        raise ValueError(f"Unknown selection strategy: {selection}")

    return {
        'image': result_image,
        'scores': all_scores,
        'selected_idx': selected_idx,
        'all_images': candidates,
        'composite_values': composite_values,
        'selection': selection,
    }


@torch.no_grad()
def sample_best_of_n_batched(
    autoencoder: nn.Module,
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_a: int,
    context: torch.Tensor,
    device: str,
    scale_factor: int = 1,
    n_candidates: int = 8,
    batch_size: int = 4,
    selection: str = 'best1',
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    source_image: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """Batched version: generate candidates in parallel batches for speed.

    Same logic as sample_best_of_n but processes batch_size candidates
    at once through the diffusion process.
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    starting_z_dev = starting_z.unsqueeze(0).to(device)
    concat_age = (torch.tensor([starting_a])
                  .view(1, 1, 1, 1, 1)
                  .expand(1, 1, *starting_z_dev.shape[-3:])
                  .to(device))
    controlnet_cond_single = torch.cat([starting_z_dev, concat_age], dim=1).to(device)
    ctx_single = context.unsqueeze(0).unsqueeze(0).to(device)

    # Decode source if needed
    if source_image is None:
        src_z = starting_z.unsqueeze(0).to(device) / scale_factor
        src_z = utils.to_vae_latent_trick(src_z.squeeze(0).cpu())
        source_image = autoencoder.decode_stage_2_outputs(
            src_z.unsqueeze(0).to(device)
        )
        source_image = (utils.to_mni_space_1p5mm_trick(source_image.squeeze(0).cpu())
                        .squeeze(0).numpy().clip(0, 1))

    candidates = []
    latent_norms = []

    n_batches = (n_candidates + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if verbose:
        iterator = tqdm(iterator, desc=f"Batched gen ({n_candidates} cands, bs={batch_size})")

    for b in iterator:
        bs = min(batch_size, n_candidates - b * batch_size)

        ctx_b = ctx_single.repeat(bs, 1, 1)
        cond_b = controlnet_cond_single.repeat(bs, 1, 1, 1, 1)
        z = torch.randn(bs, *starting_z_dev.shape[1:]).to(device)

        for j in range(bs):
            latent_norms.append(float(z[j].norm().cpu()))

        for t in scheduler.timesteps:
            with autocast(enabled=True):
                timestep = torch.tensor([t]).repeat(bs).to(device)
                down_h, mid_h = controlnet(
                    x=z.float(),
                    timesteps=timestep,
                    context=ctx_b,
                    controlnet_cond=cond_b.float(),
                )
                noise_pred = diffusion(
                    x=z.float(),
                    timesteps=timestep,
                    context=ctx_b.float(),
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h,
                )
                z, _ = scheduler.step(noise_pred, t, z)

        # Decode each sample in the batch
        for j in range(bs):
            z_j = z[j:j+1] / scale_factor
            z_j = utils.to_vae_latent_trick(z_j.squeeze(0).cpu())
            img = autoencoder.decode_stage_2_outputs(z_j.unsqueeze(0).to(device))
            img = (utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu())
                   .squeeze(0).numpy().clip(0, 1))
            candidates.append(img)

    # Score and select (same as sequential version)
    all_scores = []
    for i, cand in enumerate(candidates):
        scores = composite_score(
            generated=cand,
            source=source_image,
            latent_norm=latent_norms[i],
        )
        all_scores.append(scores)

    composite_values = [s['composite'] for s in all_scores]

    if selection == 'best1':
        best_idx = int(np.argmax(composite_values))
        result_image = candidates[best_idx]
        selected_idx = best_idx
    elif selection == 'topk_avg':
        k = max(2, n_candidates // 3)
        top_indices = np.argsort(composite_values)[-k:]
        result_image = np.mean([candidates[i] for i in top_indices], axis=0)
        selected_idx = top_indices.tolist()
    elif selection == 'weighted':
        weights = np.array(composite_values)
        weights = weights - weights.min() + 1e-6
        weights = weights / weights.sum()
        result_image = sum(w * c for w, c in zip(weights, candidates))
        selected_idx = list(range(n_candidates))
    else:
        raise ValueError(f"Unknown selection: {selection}")

    return {
        'image': result_image,
        'scores': all_scores,
        'selected_idx': selected_idx,
        'all_images': candidates,
        'composite_values': composite_values,
        'selection': selection,
    }
