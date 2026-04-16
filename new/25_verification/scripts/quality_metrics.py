"""
No-Ground-Truth Quality Metrics for Brain MRI Generation.

These metrics evaluate generated brain MRI quality WITHOUT requiring the real
follow-up image. Used for Best-of-N sample selection (Scheme A) and quality
gating across all verification schemes.

Metrics:
  1. Source SSIM:         Structural similarity to the starting (baseline) image
  2. Intensity Score:     How well intensity stats match expected brain distribution
  3. Brain Coverage:      Ratio of non-zero voxels (brain region)
  4. Smoothness Score:    Gradient-based smoothness (natural brains are smooth)
  5. Latent Norm Score:   Deviation of latent from expected norm range
  6. Composite Score:     Weighted combination of all metrics
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def source_ssim(generated: np.ndarray, source: np.ndarray) -> float:
    """SSIM between generated follow-up and the starting (baseline) image.

    Brain structure changes slowly in MCI (6-24 months), so a good prediction
    should retain high structural similarity with the source while showing
    subtle volumetric changes.
    """
    min_shape = tuple(min(a, b) for a, b in zip(generated.shape, source.shape))
    gen = generated[:min_shape[0], :min_shape[1], :min_shape[2]]
    src = source[:min_shape[0], :min_shape[1], :min_shape[2]]
    data_range = max(src.max() - src.min(), 1e-8)
    try:
        return float(ssim(src, gen, data_range=data_range))
    except Exception:
        return 0.0


def intensity_score(generated: np.ndarray, source: np.ndarray) -> float:
    """Score how well intensity statistics match the source distribution.

    Compares mean and std of non-zero voxels. A bad sample often has
    abnormal intensity patterns (too dark, too bright, or flat).
    Returns value in [0, 1], higher is better.
    """
    gen_mask = generated > 0.01
    src_mask = source > 0.01

    if gen_mask.sum() < 100 or src_mask.sum() < 100:
        return 0.0

    gen_vals = generated[gen_mask]
    src_vals = source[src_mask]

    mean_diff = abs(gen_vals.mean() - src_vals.mean())
    std_diff = abs(gen_vals.std() - src_vals.std())

    # Normalize: mean_diff and std_diff both expected in [0, 0.5] for [0,1] images
    mean_score = max(0, 1.0 - mean_diff * 4)    # penalty 4x mean_diff
    std_score = max(0, 1.0 - std_diff * 4)
    return float(0.6 * mean_score + 0.4 * std_score)


def brain_coverage_score(generated: np.ndarray, source: np.ndarray) -> float:
    """Score brain/background ratio consistency.

    Brain coverage should be similar between source and generated.
    Returns value in [0, 1], higher is better.
    """
    gen_ratio = (generated > 0.01).sum() / max(generated.size, 1)
    src_ratio = (source > 0.01).sum() / max(source.size, 1)

    if src_ratio < 0.01:
        return 0.5  # source is mostly empty, can't judge

    ratio_diff = abs(gen_ratio - src_ratio) / src_ratio
    return float(max(0, 1.0 - ratio_diff * 5))


def smoothness_score(generated: np.ndarray) -> float:
    """Gradient-based smoothness score for the generated image.

    Natural brain MRIs have smooth intensity transitions. Noisy or
    artifact-laden images have high gradient magnitudes.
    Returns value in [0, 1], higher is better (smoother).
    """
    if generated.max() < 0.01:
        return 0.0

    # Compute gradient magnitude along each axis
    gx = np.diff(generated, axis=0)
    gy = np.diff(generated, axis=1)
    gz = np.diff(generated, axis=2)

    mean_grad = (np.abs(gx).mean() + np.abs(gy).mean() + np.abs(gz).mean()) / 3

    # For [0,1] brain MRI, typical mean gradient is ~0.01-0.03
    # High gradient (>0.05) indicates artifacts
    return float(max(0, 1.0 - mean_grad * 15))


def latent_norm_score(latent_norm: float,
                      expected_mean: float = 1.0,
                      expected_std: float = 0.3) -> float:
    """Score latent vector norm relative to expected distribution.

    Latents with extreme norms (too large or too small) tend to produce
    poor quality samples.
    Returns value in [0, 1], higher is better.
    """
    z_score = abs(latent_norm - expected_mean) / max(expected_std, 1e-8)
    return float(max(0, 1.0 - z_score * 0.3))


def composite_score(generated: np.ndarray,
                    source: np.ndarray,
                    latent_norm: float = None,
                    weights: dict = None) -> dict:
    """Compute all metrics and a weighted composite score.

    Args:
        generated: Generated follow-up MRI as numpy array [0, 1].
        source: Starting (baseline) MRI as numpy array [0, 1].
        latent_norm: Optional L2 norm of the latent vector.
        weights: Optional dict of metric weights. Defaults to balanced.

    Returns:
        Dict with individual scores and composite.
    """
    if weights is None:
        weights = {
            'source_ssim': 0.40,
            'intensity': 0.20,
            'coverage': 0.15,
            'smoothness': 0.15,
            'latent_norm': 0.10,
        }

    scores = {
        'source_ssim': source_ssim(generated, source),
        'intensity': intensity_score(generated, source),
        'coverage': brain_coverage_score(generated, source),
        'smoothness': smoothness_score(generated),
    }

    if latent_norm is not None:
        scores['latent_norm'] = latent_norm_score(latent_norm)
    else:
        scores['latent_norm'] = 0.5  # neutral if not provided
        weights['source_ssim'] = 0.45
        weights['intensity'] = 0.22
        weights['coverage'] = 0.18
        weights['smoothness'] = 0.15
        weights['latent_norm'] = 0.0

    total = sum(weights.get(k, 0) * scores[k] for k in scores)
    scores['composite'] = total

    return scores
