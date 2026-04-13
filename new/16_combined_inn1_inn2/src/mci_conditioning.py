"""
MCI Conditioning Module - Innovation 1.

Provides:
  1. Modified ControlNet init with conditioning_embedding_in_channels=6
  2. Utility functions to compute atrophy rate & ventricular expansion rate
  3. Functions to build extended controlnet_condition tensors
"""

import torch
import torch.nn as nn
from generative.networks.nets import ControlNet


def init_controlnet_mci(checkpoints_path=None):
    """
    ControlNet with 6 spatial conditioning channels:
      ch 0-2: starting latent (3 channels)
      ch 3:   starting age
      ch 4:   hippocampal atrophy rate
      ch 5:   ventricular expansion rate

    The extra 2 channels provide disease-progression-velocity information
    that helps the model predict MCI brains with varying decline speeds.
    """
    controlnet = ControlNet(
        spatial_dims=3,
        in_channels=3,
        num_res_blocks=2,
        num_channels=(256, 512, 768),
        attention_levels=(False, True, True),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=(0, 512, 768),
        transformer_num_layers=1,
        with_conditioning=True,
        cross_attention_dim=8,          # unchanged from baseline
        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False,
        conditioning_embedding_in_channels=6,   # 4 -> 6
        conditioning_embedding_num_channels=(256,),
    )

    if checkpoints_path is not None:
        import os
        assert os.path.exists(checkpoints_path), f'Invalid path: {checkpoints_path}'
        device = next(controlnet.parameters()).device
        state = torch.load(checkpoints_path, map_location=device)
        controlnet.load_state_dict(state, strict=True)

    return controlnet


def init_controlnet_mci_from_pretrained(pretrained_4ch_path):
    """
    Initialize 6-channel ControlNet from pretrained 4-channel ControlNet.

    Strategy: load all weights, expand conv_in from [256,4,3,3,3] to [256,6,3,3,3],
    zero-initializing the 2 new input channels (atrophy_rate, vent_rate).
    This preserves the learned baseline behavior and only requires learning
    the impact of the new biomarker channels.
    """
    import os
    assert os.path.exists(pretrained_4ch_path), f'Not found: {pretrained_4ch_path}'

    # Create 6-channel ControlNet
    controlnet = init_controlnet_mci()

    # Load pretrained 4-channel state dict
    pretrained_sd = torch.load(pretrained_4ch_path, map_location='cpu')

    # Build new state dict: copy all matching keys, expand conv_in
    new_sd = controlnet.state_dict()
    conv_in_key = 'controlnet_cond_embedding.conv_in.conv.weight'

    for key in pretrained_sd:
        if key == conv_in_key:
            # Expand [256, 4, 3, 3, 3] -> [256, 6, 3, 3, 3]
            old_w = pretrained_sd[key]  # [256, 4, 3, 3, 3]
            new_w = new_sd[key]         # [256, 6, 3, 3, 3]
            nn.init.zeros_(new_w)
            new_w[:, :old_w.shape[1]] = old_w  # copy first 4 channels
            new_sd[key] = new_w
            print(f'  Expanded {key}: {old_w.shape} -> {new_w.shape} (new channels zero-init)')
        elif key in new_sd:
            new_sd[key] = pretrained_sd[key]

    controlnet.load_state_dict(new_sd, strict=True)
    print(f'  Loaded pretrained ControlNet from {pretrained_4ch_path}')
    return controlnet


def compute_atrophy_rate(start_hippocampus, followup_hippocampus, time_delta):
    """
    Compute hippocampal atrophy rate.

    Args:
        start_hippocampus: normalized hippocampus vol at starting visit
        followup_hippocampus: normalized hippocampus vol at followup visit
        time_delta: time between visits (in years, normalized 0-1 scale)

    Returns:
        Atrophy rate (positive = volume loss).
        Clamped to [-1, 1] to avoid extreme values from short intervals.
    """
    if time_delta < 1e-6:
        return 0.0
    rate = (start_hippocampus - followup_hippocampus) / time_delta
    return max(-1.0, min(1.0, rate))


def compute_ventricular_rate(start_ventricle, followup_ventricle, time_delta):
    """
    Compute ventricular expansion rate.

    Args:
        start_ventricle: normalized lateral ventricle vol at starting visit
        followup_ventricle: normalized lateral ventricle vol at followup visit
        time_delta: time between visits (in years, normalized 0-1 scale)

    Returns:
        Expansion rate (positive = volume gain).
        Clamped to [-1, 1].
    """
    if time_delta < 1e-6:
        return 0.0
    rate = (followup_ventricle - start_ventricle) / time_delta
    return max(-1.0, min(1.0, rate))


def build_controlnet_condition(starting_z, starting_age, atrophy_rate, ventricular_rate):
    """
    Build the 6-channel ControlNet spatial condition tensor.

    Args:
        starting_z: (B, 3, D, H, W) starting latent
        starting_age: (B,) starting ages
        atrophy_rate: (B,) hippocampal atrophy rates
        ventricular_rate: (B,) ventricular expansion rates

    Returns:
        (B, 6, D, H, W) condition tensor
    """
    n = starting_z.shape[0]
    spatial_shape = starting_z.shape[-3:]

    # Broadcast scalars to spatial maps
    age_map = starting_age.view(n, 1, 1, 1, 1).expand(n, 1, *spatial_shape)
    atrophy_map = atrophy_rate.view(n, 1, 1, 1, 1).expand(n, 1, *spatial_shape)
    vent_map = ventricular_rate.view(n, 1, 1, 1, 1).expand(n, 1, *spatial_shape)

    return torch.cat([starting_z, age_map, atrophy_map, vent_map], dim=1)
