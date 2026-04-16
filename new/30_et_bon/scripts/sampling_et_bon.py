"""
Early-Timestep BoN (ET-BoN) Sampling — 早期时间步筛选 + 加权融合

核心思路：
  1. 生成 N 个候选初始噪声
  2. 所有候选同时开始 DDIM 去噪
  3. 在早期检查点 (如 step 10/50) 解码中间 latent，评估质量
  4. 淘汰质量最差的候选，只保留 top-K
  5. 仅对 K 个幸存者继续完成剩余去噪步骤
  6. 最终对 K 个结果做加权融合

相比 BoN Weighted:
  - 计算量: N*T_early + K*T_remain  vs  N*T_total
  - 例: 8→3, T_early=10: 8*10 + 3*40 = 200 steps vs 8*50 = 400 steps (节省50%)
  - 越激进的淘汰越省算力

Usage:
    from sampling_et_bon import sample_et_bon_weighted
    img = sample_et_bon_weighted(autoencoder, diffusion, controlnet,
                                  starting_z, starting_a, context, device,
                                  scale_factor=sf,
                                  n_initial=8, n_survivors=3, checkpoint_step=10)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Server: scripts/ is at code/et_bon/scripts/, src at code/src/ -> ../../src
# Local: scripts/ is at new/30_et_bon/scripts/, src at src/ -> ../../../src
# Try both paths
for _rel in ['..', '..', 'src'], ['..', '..', '..', 'src']:
    _p = os.path.abspath(os.path.join(SCRIPT_DIR, *_rel))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

from brlp import utils, const
from skimage.metrics import structural_similarity as ssim_fn


# ── Quality metrics (与 BoN integrated 一致) ──

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


# ── Early-step proxy metric (latent space) ──

def _latent_proxy_score(intermediate_z, source_z, autoencoder, device, scale_factor):
    """
    在早期步骤评估中间 latent 质量。
    解码中间 latent 并与 source 做 SSIM 比较。
    虽然中间步骤的图像还很 noisy，但相对排序已有信号。
    """
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

    # 综合评分：SSIM + 强度一致性
    ssim_val = _source_ssim(img_np, src_np)
    intens = _intensity_score(img_np, src_np)
    cover = _coverage_score(img_np, src_np)
    return 0.50 * ssim_val + 0.30 * intens + 0.20 * cover


def _latent_proxy_fast(intermediate_z, source_z):
    """
    快速 latent 空间代理评分（不需要解码）。
    使用余弦相似度 + L2 距离作为排序信号。
    """
    z_flat = intermediate_z.flatten().float()
    s_flat = source_z.flatten().float()
    cos_sim = float(torch.nn.functional.cosine_similarity(
        z_flat.unsqueeze(0), s_flat.unsqueeze(0)
    ))
    l2_dist = float((z_flat - s_flat).norm())
    # 归一化: cos_sim in [-1,1], 越高越好; l2越小越好
    return 0.6 * (cos_sim + 1) / 2 + 0.4 * max(0, 1.0 - l2_dist / (s_flat.norm() + 1e-6))


# ── Main ET-BoN function ──

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
    n_survivors: int = 3,
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
    Early-Timestep BoN Weighted sampling.

    Phase 1: N candidates run denoising until checkpoint_step
    Phase 2: Score and keep top-K survivors
    Phase 3: Survivors complete remaining steps
    Phase 4: Weighted fusion of K decoded images

    Args:
        n_initial: 初始候选数 N
        n_survivors: 筛选后保留数 K
        checkpoint_step: 在第几步做筛选 (基于0索引，即step 10 = 第10步)
        use_decoded_proxy: True用解码评分(精确但慢), False用latent空间评分(快)
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

        # Free GPU memory
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

        # Free decoded intermediates
        torch.cuda.empty_cache()

    # Select top-K
    ranked = sorted(range(n_initial), key=lambda i: proxy_scores[i], reverse=True)
    survivor_indices = ranked[:n_survivors]
    eliminated_indices = ranked[n_survivors:]

    if verbose:
        print(f"[ET-BoN] Phase 2: Scores = {[f'{s:.4f}' for s in proxy_scores]}")
        print(f"[ET-BoN] Survivors: {survivor_indices}, Eliminated: {eliminated_indices}")

    # Free eliminated latents
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

        # Decode
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


@torch.no_grad()
def sample_et_bon_with_details(
    autoencoder, diffusion, controlnet,
    starting_z, starting_a, context, device,
    scale_factor=1, n_initial=8, n_survivors=3,
    checkpoint_step=10, use_decoded_proxy=True,
    num_inference_steps=50, verbose=False,
    **kwargs,
) -> dict:
    """Same as sample_et_bon_weighted but returns detailed info for analysis."""
    import time
    t0 = time.time()

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

    timesteps = list(scheduler.timesteps)

    # Phase 1
    latents = []
    latent_norms = []
    for _i in range(n_initial):
        z = torch.randn(1, *sz.shape[1:]).to(device)
        latent_norms.append(float(z.norm().cpu()))
        for t in timesteps[:checkpoint_step]:
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

    phase1_time = time.time() - t0

    # Phase 2
    t1 = time.time()
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

    survivor_latents = [latents[i] for i in survivor_indices]
    survivor_norms = [latent_norms[i] for i in survivor_indices]
    del latents
    torch.cuda.empty_cache()
    phase2_time = time.time() - t1

    # Phase 3
    t2 = time.time()
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

    phase3_time = time.time() - t2

    # Phase 4
    final_scores = [
        _composite(c, source_np, survivor_norms[i])
        for i, c in enumerate(completed_images)
    ]
    weights = np.array(final_scores)
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()
    result = sum(w * c for w, c in zip(weights, completed_images))

    total_time = time.time() - t0

    # 计算理论步数
    total_steps = n_initial * checkpoint_step + n_survivors * (num_inference_steps - checkpoint_step)
    baseline_steps = n_initial * num_inference_steps  # BoN
    las_steps = num_inference_steps  # LAS M=1

    return {
        "image": torch.from_numpy(result).float(),
        "config": {
            "n_initial": n_initial,
            "n_survivors": n_survivors,
            "checkpoint_step": checkpoint_step,
            "use_decoded_proxy": use_decoded_proxy,
            "num_inference_steps": num_inference_steps,
        },
        "timing": {
            "phase1_sec": round(phase1_time, 2),
            "phase2_sec": round(phase2_time, 2),
            "phase3_sec": round(phase3_time, 2),
            "total_sec": round(total_time, 2),
        },
        "steps": {
            "et_bon_steps": total_steps,
            "bon_steps": baseline_steps,
            "step_savings_pct": round((1 - total_steps / baseline_steps) * 100, 1),
        },
        "proxy_scores": proxy_scores,
        "survivor_indices": survivor_indices,
        "final_scores": final_scores,
        "weights": weights.tolist(),
        "source_np": source_np,
        "completed_images": completed_images,
    }
