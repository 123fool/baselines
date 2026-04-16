import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn

from . import utils
from . import const


@torch.no_grad()
def sample_using_diffusion(
    autoencoder: nn.Module, 
    diffusion: nn.Module, 
    context: torch.Tensor,
    device: str, 
    scale_factor: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True
) -> torch.Tensor: 
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    context = context.unsqueeze(0).to(device).to(device)

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device)
    
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                timestep = torch.tensor([t]).to(device)
                
                # predict the noise
                noise_pred = diffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context.float(), 
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)
    
    # decode the latent
    z = z / scale_factor
    z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
    x = autoencoder.decode_stage_2_outputs( z.unsqueeze(0).to(device) )
    x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x


@torch.no_grad()
def sample_using_controlnet_and_z(
    autoencoder: nn.Module, 
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
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
    verbose: bool = True
) -> torch.Tensor:
    """
    The inference process described in the paper.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        controlnet (nn.Module): the ControlNet
        starting_z (torch.Tensor): the latent from the MRI of the starting visit 
        starting_a (int): the starting age
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        average_over_n (int, optional): LAS parameter m. Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.

    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # preparing controlnet spatial condition.
    starting_z             = starting_z.unsqueeze(0).to(device)
    concatenating_age      = torch.tensor([ starting_a ]).view(1, 1, 1, 1, 1).expand(1, 1, *starting_z.shape[-3:]).to(device)
    controlnet_condition   = torch.cat([ starting_z, concatenating_age ], dim=1).to(device)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    context = context.unsqueeze(0).unsqueeze(0).to(device)

    # if performing LAS, we repeat the inputs for the diffusion process
    # m times (as specified in the paper) and perform the reverse diffusion
    # process in parallel to avoid overheads.
    if average_over_n > 1:
        context               = context.repeat(average_over_n, 1, 1)
        controlnet_condition  = controlnet_condition.repeat(average_over_n, 1, 1, 1, 1) 
    
    # this is z_T - the starting noise.
    z = torch.randn(average_over_n, *starting_z.shape[1:]).to(device)

    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                # convert the timestep to a tensor.
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)

                # get the intermediate features from the ControlNet
                # by feeding the starting latent, the covariates and the timestep
                down_h, mid_h = controlnet(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context,
                    controlnet_cond=controlnet_condition.float()
                )

                # the diffusion takes the intermediate features and predicts
                # the noise. This is why we conceptualize the two networks as
                # as a unified network.
                noise_pred = diffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context.float(), 
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)

    # Here we conclude Latent Average Stabilization by averaging 
    # m different latents from m different samplings.
    z = (z / scale_factor).sum(axis=0) / average_over_n
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())

    # decode the latent using the Decoder block from the KL autoencoder.
    x = autoencoder.decode_stage_2_outputs( z.unsqueeze(0).to(device) )
    x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x


# ═══════════════════════════════════════════════════════════════════════
# ET-BoN (Early-Timestep Best-of-N) Sampling
# ═══════════════════════════════════════════════════════════════════════
#
# Core idea:
#   1. Generate N candidate initial noises
#   2. All candidates denoise simultaneously via DDIM
#   3. At an early checkpoint (e.g. step 10/50), decode intermediate
#      latents and evaluate quality
#   4. Eliminate the worst candidates, keeping only top-K survivors
#   5. Only K survivors complete the remaining denoising steps
#   6. Final weighted fusion of K results
#
# Computational cost: N*T_early + K*T_remain  vs  N*T_total (standard BoN)
# Example: N=8→K=5, T_early=10: 8×10 + 5×40 = 280 steps
#          vs BoN: 8×50 = 400 steps (30% savings)
# ═══════════════════════════════════════════════════════════════════════

def _source_ssim(generated, source):
    return float(ssim_fn(generated, source, data_range=1.0))

def _intensity_score(generated, source):
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
    gr = (generated > 0.01).sum() / max(generated.size, 1)
    sr = (source > 0.01).sum() / max(source.size, 1)
    if sr < 1e-6:
        return 0.5
    return max(0, 1.0 - abs(float(gr - sr)) / float(sr) * 5)

def _smoothness_score(generated):
    gx = np.abs(np.diff(generated, axis=0)).mean()
    gy = np.abs(np.diff(generated, axis=1)).mean()
    gz = np.abs(np.diff(generated, axis=2)).mean()
    mg = (gx + gy + gz) / 3.0
    return max(0, 1.0 - float(mg) * 15)

def _latent_norm_score(norm, expected_mean=1.0, expected_std=0.3):
    z = abs(norm - expected_mean) / expected_std
    return max(0, 1.0 - z * 0.3)

def _composite(generated, source, latent_norm=None):
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

def _latent_proxy_score(intermediate_z, source_z, autoencoder, device, scale_factor):
    """Decode-based quality scoring at intermediate timestep."""
    z_dec = utils.to_vae_latent_trick(
        (intermediate_z / scale_factor).squeeze(0).cpu()
    )
    img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
    img_np = (utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu())
              .squeeze(0).numpy().clip(0, 1))

    src_dec = utils.to_vae_latent_trick(
        (source_z / scale_factor).squeeze(0).cpu()
    )
    src_img = autoencoder.decode_stage_2_outputs(src_dec.unsqueeze(0).to(device))
    src_np = (utils.to_mni_space_1p5mm_trick(src_img.squeeze(0).cpu())
              .squeeze(0).numpy().clip(0, 1))

    ssim_val = _source_ssim(img_np, src_np)
    intens = _intensity_score(img_np, src_np)
    cover = _coverage_score(img_np, src_np)
    return 0.50 * ssim_val + 0.30 * intens + 0.20 * cover

def _latent_proxy_fast(intermediate_z, source_z):
    """Fast latent-space proxy scoring (no decoding required)."""
    z_flat = intermediate_z.flatten().float()
    s_flat = source_z.flatten().float()
    cos_sim = float(torch.nn.functional.cosine_similarity(
        z_flat.unsqueeze(0), s_flat.unsqueeze(0)
    ))
    l2_dist = float((z_flat - s_flat).norm())
    return 0.6 * (cos_sim + 1) / 2 + 0.4 * max(0, 1.0 - l2_dist / (s_flat.norm() + 1e-6))


@torch.no_grad()
def sample_et_bon_weighted(
    autoencoder: nn.Module,
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z: torch.Tensor,
    starting_a: int,
    context: torch.Tensor,
    device: str,
    scale_factor: int = 1,
    n_initial: int = 8,
    n_survivors: int = 5,
    checkpoint_step: int = 10,
    use_decoded_proxy: bool = True,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Early-Timestep Best-of-N (ET-BoN) weighted sampling.

    Phase 1: N candidates run denoising until checkpoint_step
    Phase 2: Score intermediate latents and keep top-K survivors
    Phase 3: Survivors complete remaining denoising steps
    Phase 4: Weighted fusion of K decoded images

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet
        controlnet (nn.Module): the ControlNet
        starting_z (torch.Tensor): latent from the baseline MRI
        starting_a (int): the starting age
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): latent scale factor. Defaults to 1.
        n_initial (int, optional): number of initial candidates N. Defaults to 8.
        n_survivors (int, optional): top-K to keep after early filtering. Defaults to 5.
        checkpoint_step (int, optional): step at which to evaluate and filter. Defaults to 10.
        use_decoded_proxy (bool, optional): decode for scoring (precise) vs latent-space (fast). Defaults to True.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progress info. Defaults to False.

    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    assert n_survivors <= n_initial, f"n_survivors({n_survivors}) > n_initial({n_initial})"
    assert checkpoint_step < num_inference_steps, f"checkpoint({checkpoint_step}) >= total({num_inference_steps})"

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

    # Decode baseline for final scoring
    src_z = utils.to_vae_latent_trick((sz / scale_factor).squeeze(0).cpu())
    src_img = autoencoder.decode_stage_2_outputs(src_z.unsqueeze(0).to(device))
    source_np = (utils.to_mni_space_1p5mm_trick(src_img.squeeze(0).cpu())
                 .squeeze(0).numpy().clip(0, 1))

    timesteps = list(scheduler.timesteps)

    # ── Phase 1: All N candidates run until checkpoint ──
    latents = []
    latent_norms = []
    for _i in range(n_initial):
        z = torch.randn(1, *sz.shape[1:]).to(device)
        latent_norms.append(float(z.norm().cpu()))

        for step_idx, t in enumerate(timesteps[:checkpoint_step]):
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

        latents.append(z.clone())
        del z
        torch.cuda.empty_cache()

    if verbose:
        print(f"[ET-BoN] Phase 1 done: {n_initial} candidates × {checkpoint_step} steps")

    # ── Phase 2: Score intermediate latents and select survivors ──
    proxy_scores = []
    for i, z in enumerate(latents):
        if use_decoded_proxy:
            score = _latent_proxy_score(z, sz, autoencoder, device, scale_factor)
        else:
            score = _latent_proxy_fast(z, sz)
        proxy_scores.append(score)
        torch.cuda.empty_cache()

    ranked = sorted(range(n_initial), key=lambda i: proxy_scores[i], reverse=True)
    survivor_indices = ranked[:n_survivors]

    if verbose:
        print(f"[ET-BoN] Phase 2: Scores = {[f'{s:.4f}' for s in proxy_scores]}")
        print(f"[ET-BoN] Survivors: {survivor_indices}")

    survivor_latents = [latents[i] for i in survivor_indices]
    survivor_norms = [latent_norms[i] for i in survivor_indices]
    del latents
    torch.cuda.empty_cache()

    # ── Phase 3: Survivors complete remaining steps ──
    completed_images = []
    for j, z in enumerate(survivor_latents):
        for t in timesteps[checkpoint_step:]:
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

        z_dec = utils.to_vae_latent_trick((z / scale_factor).squeeze(0).cpu())
        img = autoencoder.decode_stage_2_outputs(z_dec.unsqueeze(0).to(device))
        img_np = (utils.to_mni_space_1p5mm_trick(img.squeeze(0).cpu())
                  .squeeze(0).numpy().clip(0, 1))
        completed_images.append(img_np)

        del z, z_dec, img
        torch.cuda.empty_cache()

    if verbose:
        print(f"[ET-BoN] Phase 3 done: {n_survivors} survivors × {num_inference_steps - checkpoint_step} steps")

    # ── Phase 4: Score & weighted fusion ──
    final_scores = [
        _composite(c, source_np, survivor_norms[i])
        for i, c in enumerate(completed_images)
    ]
    weights = np.array(final_scores)
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()

    result = sum(w * c for w, c in zip(weights, completed_images))

    if verbose:
        print(f"[ET-BoN] Phase 4: Final scores = {[f'{s:.4f}' for s in final_scores]}")
        print(f"[ET-BoN] Weights = {[f'{w:.3f}' for w in weights]}")

    return torch.from_numpy(result).float()