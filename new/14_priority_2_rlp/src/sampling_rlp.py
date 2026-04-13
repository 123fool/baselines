"""
Residual Latent Prediction (RLP) — Modified Sampling/Inference.

Key difference from baseline sampling.py:
  The diffusion model now denoises to get delta_z (the residual latent),
  not the complete followup_z. The final followup latent is reconstructed
  as: z_followup = z_starting + delta_z

  Reference: TADM-3D test_diff_model.py line:
      pred_image = torch.clamp(context + diff_pred_image, 0, 1)
"""

import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
BRLP_SRC_ALT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'brlp_src'))
for p in [BRLP_SRC, BRLP_SRC_ALT]:
    if os.path.isdir(p):
        sys.path.insert(0, p)

from brlp import utils, const


@torch.no_grad()
def sample_using_controlnet_and_z_rlp(
    autoencoder: nn.Module,
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_z_unscaled: torch.Tensor,
    starting_a: int,
    context: torch.Tensor,
    device: str,
    scale_factor: int = 1,
    average_over_n: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    verbose: bool = True,
) -> torch.Tensor:
    """
    RLP inference: Denoise to get delta_z, then z_followup = z_starting + delta_z.

    Args:
        autoencoder: the KL autoencoder
        diffusion: the UNet
        controlnet: the ControlNet
        starting_z: the latent from starting visit, SCALED (starting_latent * scale_factor)
        starting_z_unscaled: the latent from starting visit, UNSCALED (original)
        starting_a: the starting age
        context: the covariates (8-dim)
        device: 'cuda' or 'cpu'
        scale_factor: scale factor computed from delta_z distribution
        average_over_n: LAS parameter m
        num_training_steps: T parameter (1000)
        num_inference_steps: DDIM steps (50)
        schedule: noise schedule
        beta_start: noise starting level
        beta_end: noise ending level
        verbose: print progress bar

    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Prepare controlnet spatial condition
    starting_z_scaled = starting_z.unsqueeze(0).to(device)
    concatenating_age = torch.tensor([starting_a]).view(1, 1, 1, 1, 1).expand(
        1, 1, *starting_z_scaled.shape[-3:]).to(device)
    controlnet_condition = torch.cat(
        [starting_z_scaled, concatenating_age], dim=1).to(device)

    # Cross-attention context
    context = context.unsqueeze(0).unsqueeze(0).to(device)

    # LAS: repeat inputs if averaging over multiple samples
    if average_over_n > 1:
        context = context.repeat(average_over_n, 1, 1)
        controlnet_condition = controlnet_condition.repeat(
            average_over_n, 1, 1, 1, 1)

    # z_T: starting noise (this will be denoised to delta_z)
    z = torch.randn(average_over_n, *starting_z_scaled.shape[1:]).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)

                down_h, mid_h = controlnet(
                    x=z.float(),
                    timesteps=timestep,
                    context=context,
                    controlnet_cond=controlnet_condition.float()
                )

                noise_pred = diffusion(
                    x=z.float(),
                    timesteps=timestep,
                    context=context.float(),
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h
                )

                z, _ = scheduler.step(noise_pred, t, z)

    # ============ RLP: Residual reconstruction ============
    # z is now the denoised delta_z (in scaled space)
    # Average over LAS samples and unscale
    delta_z = (z / scale_factor).sum(axis=0) / average_over_n

    # Reconstruct followup latent: z_followup = z_starting + delta_z
    starting_z_raw = starting_z_unscaled.to(device)
    z_followup = starting_z_raw + delta_z
    # =====================================================

    z_followup = utils.to_vae_latent_trick(z_followup.squeeze(0).cpu())

    # Decode using AE
    x = autoencoder.decode_stage_2_outputs(z_followup.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x
