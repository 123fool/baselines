"""
Sampling with PALM + TEL for inference.

Extends BrLP's sample_using_controlnet_and_z to:
  1. Modulate starting_z through PALM before creating ControlNet condition
  2. Enhance age channel with TEL temporal encoding
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm

from brlp import utils, const


@torch.no_grad()
def sample_using_controlnet_and_z_palm_tel(
    autoencoder: nn.Module,
    diffusion: nn.Module,
    controlnet: nn.Module,
    palm: nn.Module,
    tel: nn.Module,
    starting_z: torch.Tensor,
    starting_a: float,
    context: torch.Tensor,
    age_gap: float,
    device: str,
    scale_factor: float = 1,
    average_over_n: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Inference with PALM + TEL enhanced ControlNet.

    Like sample_using_controlnet_and_z but with:
      - PALM modulates starting_z based on clinical context
      - TEL enhances age channel with learnable temporal encoding

    Args:
        palm: PALM module (Progression-Aware Latent Modulation)
        tel: TEL module (Temporal Encoding Layer)
        age_gap: followup_age - starting_age (normalized)
        (others same as original)
    """
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Prepare tensors
    starting_z = starting_z.unsqueeze(0).to(device)           # (1, C, D, H, W)
    context_8d = context.unsqueeze(0).to(device)               # (1, 8)
    context_ca = context.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 8)

    # PALM: modulate starting_z based on clinical context
    modulated_z = palm(starting_z, context_8d)                 # (1, C, D, H, W)

    # TEL: enhance age channel with temporal encoding
    age_gap_t = torch.tensor([age_gap], dtype=torch.float32).to(device)
    tel_enc = tel(age_gap_t).squeeze(-1)                       # (1,)
    age_enhanced = torch.tensor([starting_a], dtype=torch.float32).to(device) + tel_enc

    # Build spatial condition: [PALM(starting_z), age + TEL(age_gap)]
    age_spatial = age_enhanced.view(1, 1, 1, 1, 1).expand(1, 1, *modulated_z.shape[-3:])
    controlnet_condition = torch.cat([modulated_z, age_spatial], dim=1).to(device)

    # LAS (Latent Average Stabilization)
    if average_over_n > 1:
        context_ca = context_ca.repeat(average_over_n, 1, 1)
        controlnet_condition = controlnet_condition.repeat(average_over_n, 1, 1, 1, 1)

    # Starting noise z_T ~ N(0, I)
    z = torch.randn(average_over_n, *starting_z.shape[1:]).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)

                down_h, mid_h = controlnet(
                    x=z.float(),
                    timesteps=timestep,
                    context=context_ca,
                    controlnet_cond=controlnet_condition.float(),
                )

                noise_pred = diffusion(
                    x=z.float(),
                    timesteps=timestep,
                    context=context_ca.float(),
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h,
                )

                z, _ = scheduler.step(noise_pred, t, z)

    # LAS averaging + decode
    z = (z / scale_factor).sum(axis=0) / average_over_n
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())
    x = autoencoder.decode_stage_2_outputs(z.unsqueeze(0).to(device))
    x = utils.to_mni_space_1p5mm_trick(x.squeeze(0).cpu()).squeeze(0)
    return x
