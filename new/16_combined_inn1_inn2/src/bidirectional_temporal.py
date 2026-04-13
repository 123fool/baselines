"""
Bidirectional Temporal Regularization Module — Innovation 2.

Provides:
  1. build_reverse_batch(): swap starting↔followup in a batch for reverse training
  2. build_reverse_context(): construct reverse cross-attention context
  3. BidirectionalTemporalLoss: combined forward + backward MSE
"""

import torch
import torch.nn.functional as F


def build_reverse_context(batch):
    """
    Build the cross-attention context vector for the *reverse* direction.

    In BrLP's forward direction the context is:
        [followup_age, sex, followup_diagnosis, followup_cortex,
         followup_hippocampus, followup_amygdala, followup_white_matter,
         followup_lateral_ventricle]

    For the reverse (B→A) we want:
        [starting_age, sex, starting_diagnosis, starting_cortex,
         starting_hippocampus, starting_amygdala, starting_white_matter,
         starting_lateral_ventricle]

    The batch dict comes from MONAI DataLoader; scalar fields are already
    tensors of shape (N,).
    """
    conditions = [
        batch['starting_age'],
        batch['sex'],
        batch['starting_diagnosis'],
        batch['starting_cerebral_cortex'],
        batch['starting_hippocampus'],
        batch['starting_amygdala'],
        batch['starting_cerebral_white_matter'],
        batch['starting_lateral_ventricle'],
    ]
    # Stack to (N, 8) then add seq dim → (N, 1, 8)
    return torch.stack(conditions, dim=-1).unsqueeze(1)


def bidirectional_controlnet_loss(
    controlnet, diffusion, scheduler,
    starting_z, followup_z,
    forward_context, forward_condition,
    reverse_context, reverse_condition,
    device, btc_weight=0.5,
):
    """
    Compute combined forward + backward noise-prediction loss in a single call.

    Args:
        controlnet:        the ControlNet being trained
        diffusion:         the frozen UNet
        scheduler:         DDPMScheduler for noise addition
        starting_z:        (N, C, D, H, W) starting latent * scale_factor
        followup_z:        (N, C, D, H, W) followup latent * scale_factor
        forward_context:   (N, 1, 8) cross-attention context (direction A→B)
        forward_condition: (N, 4, D, H, W) spatial condition (direction A→B)
        reverse_context:   (N, 1, 8) cross-attention context (direction B→A)
        reverse_condition: (N, 4, D, H, W) spatial condition (direction B→A)
        device:            cuda / cpu
        btc_weight:        weight for the backward loss (default 0.5)

    Returns:
        total_loss, forward_loss, backward_loss  (all scalar tensors)
    """
    n = starting_z.shape[0]

    # ── Forward: predict noise on followup_z conditioned on starting_z ──
    noise_fwd = torch.randn_like(followup_z).to(device)
    t_fwd = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()
    noised_fwd = scheduler.add_noise(followup_z, noise=noise_fwd, timesteps=t_fwd)

    down_h, mid_h = controlnet(
        x=noised_fwd.float(), timesteps=t_fwd,
        context=forward_context.float(),
        controlnet_cond=forward_condition.float(),
    )
    pred_fwd = diffusion(
        x=noised_fwd.float(), timesteps=t_fwd,
        context=forward_context.float(),
        down_block_additional_residuals=down_h,
        mid_block_additional_residual=mid_h,
    )
    loss_fwd = F.mse_loss(pred_fwd.float(), noise_fwd.float())

    # ── Backward: predict noise on starting_z conditioned on followup_z ──
    noise_bwd = torch.randn_like(starting_z).to(device)
    t_bwd = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()
    noised_bwd = scheduler.add_noise(starting_z, noise=noise_bwd, timesteps=t_bwd)

    down_h_b, mid_h_b = controlnet(
        x=noised_bwd.float(), timesteps=t_bwd,
        context=reverse_context.float(),
        controlnet_cond=reverse_condition.float(),
    )
    pred_bwd = diffusion(
        x=noised_bwd.float(), timesteps=t_bwd,
        context=reverse_context.float(),
        down_block_additional_residuals=down_h_b,
        mid_block_additional_residual=mid_h_b,
    )
    loss_bwd = F.mse_loss(pred_bwd.float(), noise_bwd.float())

    total = loss_fwd + btc_weight * loss_bwd
    return total, loss_fwd, loss_bwd
