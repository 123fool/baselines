"""
Round-Trip Consistency Verification (Scheme B).

Given a baseline MRI at time t0, generate a follow-up at t1.
Then, use the generated follow-up as a new starting point and predict
back to t0. Compare the round-trip reconstruction with the original
baseline. High consistency = the model is confident in its prediction.

This can be used to:
  1. Score individual samples (higher round-trip SSIM = better)
  2. Combined with Best-of-N: generate N, round-trip each, pick highest
  3. As a standalone quality gate: reject samples below threshold
"""

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from skimage.metrics import structural_similarity as ssim

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
sys.path.insert(0, BRLP_SRC)

from brlp import utils, const


@torch.no_grad()
def _denoise_single(
    diffusion, controlnet, z, controlnet_cond, context,
    scheduler, device
):
    """Run the full denoising loop for a single sample."""
    for t in scheduler.timesteps:
        with autocast(enabled=True):
            timestep = torch.tensor([t]).to(device)
            down_h, mid_h = controlnet(
                x=z.float(),
                timesteps=timestep,
                context=context,
                controlnet_cond=controlnet_cond.float(),
            )
            noise_pred = diffusion(
                x=z.float(),
                timesteps=timestep,
                context=context.float(),
                down_block_additional_residuals=down_h,
                mid_block_additional_residual=mid_h,
            )
            z, _ = scheduler.step(noise_pred, t, z)
    return z


@torch.no_grad()
def _decode_latent(autoencoder, z, scale_factor, device):
    """Decode latent z to MRI image (numpy [0,1])."""
    z_dec = z / scale_factor
    z_dec = utils.to_vae_latent_trick(z_dec.squeeze(0).cpu())
    img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
    img = (utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu())
           .squeeze(0).numpy().clip(0, 1))
    return img


@torch.no_grad()
def _encode_image_to_latent(autoencoder, image_tensor, scale_factor, device):
    """Encode an image tensor to latent space.
    
    image_tensor: torch.Tensor shape (C, D, H, W) in [0, 1]
    Returns: latent tensor (C, d, h, w) padded to DM-compatible shape
    """
    with autocast(enabled=True):
        latent = autoencoder.encode_stage_2_inputs(
            image_tensor.unsqueeze(0).to(device)
        )
    # The raw AE latent is (3,15,18,15) but the DM needs (3,16,20,16)
    # Apply DivisiblePad(k=4) to match the padded shape used in training
    from monai.transforms import DivisiblePad
    padder = DivisiblePad(k=4)
    latent = padder(latent.squeeze(0)).unsqueeze(0)
    return latent.squeeze(0) * scale_factor


@torch.no_grad()
def round_trip_score(
    autoencoder: nn.Module,
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_a: int,
    forward_context: torch.Tensor,
    reverse_context: torch.Tensor,
    reverse_age: int,
    device: str,
    scale_factor: int = 1,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
) -> dict:
    """Perform a round-trip: baseline -> followup -> baseline_reconstructed.

    Args:
        autoencoder, diffusion, controlnet: trained model components.
        starting_z: latent of baseline MRI (scaled).
        starting_a: age at baseline.
        forward_context: conditioning vector for forward prediction (baseline->followup).
        reverse_context: conditioning vector for reverse prediction (followup->baseline).
        reverse_age: age to use as "starting age" for the reverse direction.
        device: 'cuda' or 'cpu'.
        scale_factor: latent scale factor.

    Returns:
        dict:
            'forward_image':    generated follow-up (numpy)
            'roundtrip_image':  reconstructed baseline (numpy)
            'source_image':     decoded original baseline (numpy)
            'roundtrip_ssim':   SSIM between source and round-trip
            'roundtrip_mae':    MAE between source and round-trip
            'forward_source_ssim': SSIM between forward and source
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # -- Step 1: Decode source baseline --
    source_image = _decode_latent(autoencoder, starting_z.unsqueeze(0).to(device),
                                  scale_factor, device)

    # -- Step 2: Forward pass (baseline -> followup) --
    starting_z_dev = starting_z.unsqueeze(0).to(device)
    concat_age_fwd = (torch.tensor([starting_a])
                      .view(1, 1, 1, 1, 1)
                      .expand(1, 1, *starting_z_dev.shape[-3:])
                      .to(device))
    cond_fwd = torch.cat([starting_z_dev, concat_age_fwd], dim=1)
    ctx_fwd = forward_context.unsqueeze(0).unsqueeze(0).to(device)

    z_fwd = torch.randn(1, *starting_z_dev.shape[1:]).to(device)
    z_fwd = _denoise_single(diffusion, controlnet, z_fwd, cond_fwd, ctx_fwd,
                            scheduler, device)

    # Decode forward prediction
    forward_image = _decode_latent(autoencoder, z_fwd, scale_factor, device)

    # -- Step 3: Encode forward prediction back to latent --
    # Re-encode the generated follow-up
    from monai import transforms as T
    forward_tensor = torch.tensor(forward_image).unsqueeze(0).float()
    # Resize to AE input shape if needed
    resize = T.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode='minimum')
    forward_tensor = resize(forward_tensor)

    followup_z = _encode_image_to_latent(
        autoencoder, forward_tensor, scale_factor, device
    )

    # -- Step 4: Reverse pass (followup -> baseline) --
    followup_z_dev = followup_z.unsqueeze(0).to(device)
    concat_age_rev = (torch.tensor([reverse_age])
                      .view(1, 1, 1, 1, 1)
                      .expand(1, 1, *followup_z_dev.shape[-3:])
                      .to(device))
    cond_rev = torch.cat([followup_z_dev, concat_age_rev], dim=1)
    ctx_rev = reverse_context.unsqueeze(0).unsqueeze(0).to(device)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    z_rev = torch.randn(1, *followup_z_dev.shape[1:]).to(device)
    z_rev = _denoise_single(diffusion, controlnet, z_rev, cond_rev, ctx_rev,
                            scheduler, device)

    roundtrip_image = _decode_latent(autoencoder, z_rev, scale_factor, device)

    # -- Step 5: Compute round-trip metrics --
    min_shape = tuple(min(a, b, c) for a, b, c in
                      zip(source_image.shape, forward_image.shape,
                          roundtrip_image.shape))
    src = source_image[:min_shape[0], :min_shape[1], :min_shape[2]]
    fwd = forward_image[:min_shape[0], :min_shape[1], :min_shape[2]]
    rtp = roundtrip_image[:min_shape[0], :min_shape[1], :min_shape[2]]

    data_range = max(src.max() - src.min(), 1e-8)
    rt_ssim = float(ssim(src, rtp, data_range=data_range))
    rt_mae = float(np.abs(src - rtp).mean())
    fwd_ssim = float(ssim(src, fwd, data_range=data_range))

    return {
        'forward_image': forward_image,
        'roundtrip_image': roundtrip_image,
        'source_image': source_image,
        'roundtrip_ssim': rt_ssim,
        'roundtrip_mae': rt_mae,
        'forward_source_ssim': fwd_ssim,
    }


@torch.no_grad()
def round_trip_best_of_n(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a,
    forward_context, reverse_context, reverse_age,
    device, scale_factor=1,
    n_candidates=5,
    num_inference_steps=50,
    verbose=True,
    **scheduler_kwargs,
) -> dict:
    """Generate N forward predictions, round-trip each, pick the one
    with highest round-trip SSIM.

    This combines Scheme A (Best-of-N) with Scheme B (Round-Trip).
    """
    scheduler_defaults = dict(
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.0205,
    )
    scheduler_defaults.update(scheduler_kwargs)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        **scheduler_defaults,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Decode source
    source_image = _decode_latent(autoencoder, starting_z.unsqueeze(0).to(device),
                                  scale_factor, device)

    # Prepare common conditions
    starting_z_dev = starting_z.unsqueeze(0).to(device)
    concat_age_fwd = (torch.tensor([starting_a])
                      .view(1, 1, 1, 1, 1)
                      .expand(1, 1, *starting_z_dev.shape[-3:])
                      .to(device))
    cond_fwd = torch.cat([starting_z_dev, concat_age_fwd], dim=1)
    ctx_fwd = forward_context.unsqueeze(0).unsqueeze(0).to(device)

    from monai import transforms as T
    resize = T.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode='minimum')

    results = []
    iterator = range(n_candidates)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Round-trip BoN ({n_candidates} cands)")

    for i in iterator:
        # Forward pass
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        z_fwd = torch.randn(1, *starting_z_dev.shape[1:]).to(device)
        z_fwd = _denoise_single(diffusion, controlnet, z_fwd, cond_fwd, ctx_fwd,
                                scheduler, device)
        forward_image = _decode_latent(autoencoder, z_fwd, scale_factor, device)

        # Encode forward
        fwd_tensor = torch.tensor(forward_image).unsqueeze(0).float()
        fwd_tensor = resize(fwd_tensor)
        followup_z = _encode_image_to_latent(
            autoencoder, fwd_tensor, scale_factor, device
        )

        # Reverse pass
        followup_z_dev = followup_z.unsqueeze(0).to(device)
        concat_age_rev = (torch.tensor([reverse_age])
                          .view(1, 1, 1, 1, 1)
                          .expand(1, 1, *followup_z_dev.shape[-3:])
                          .to(device))
        cond_rev = torch.cat([followup_z_dev, concat_age_rev], dim=1)
        ctx_rev = reverse_context.unsqueeze(0).unsqueeze(0).to(device)

        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        z_rev = torch.randn(1, *followup_z_dev.shape[1:]).to(device)
        z_rev = _denoise_single(diffusion, controlnet, z_rev, cond_rev, ctx_rev,
                                scheduler, device)
        roundtrip_image = _decode_latent(autoencoder, z_rev, scale_factor, device)

        # Compute scores
        min_shape = tuple(min(a, b, c) for a, b, c in
                          zip(source_image.shape, forward_image.shape,
                              roundtrip_image.shape))
        src = source_image[:min_shape[0], :min_shape[1], :min_shape[2]]
        fwd = forward_image[:min_shape[0], :min_shape[1], :min_shape[2]]
        rtp = roundtrip_image[:min_shape[0], :min_shape[1], :min_shape[2]]

        data_range = max(src.max() - src.min(), 1e-8)
        rt_ssim = float(ssim(src, rtp, data_range=data_range))
        fwd_ssim = float(ssim(src, fwd, data_range=data_range))

        results.append({
            'idx': i,
            'forward_image': forward_image,
            'roundtrip_image': roundtrip_image,
            'roundtrip_ssim': rt_ssim,
            'forward_source_ssim': fwd_ssim,
        })

    # Select best by roundtrip SSIM
    best = max(results, key=lambda r: r['roundtrip_ssim'])

    return {
        'image': best['forward_image'],
        'source_image': source_image,
        'roundtrip_image': best['roundtrip_image'],
        'selected_idx': best['idx'],
        'roundtrip_ssim': best['roundtrip_ssim'],
        'all_results': [{
            'idx': r['idx'],
            'roundtrip_ssim': r['roundtrip_ssim'],
            'forward_source_ssim': r['forward_source_ssim'],
        } for r in results],
    }
