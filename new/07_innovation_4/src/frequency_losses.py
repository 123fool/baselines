"""
Frequency Domain Loss Functions for Innovation 4.

Two complementary losses to combat AE over-smoothing:

1. LaplacianPyramidLoss: Multi-scale frequency decomposition.
   Separates high/low frequencies at each level, penalizes mismatch at each scale.
   High-frequency layers capture brain sulci, gyri edges, fine textures.

2. FFTFrequencyLoss: Direct spectral consistency in Fourier domain.
   Penalizes mismatch in frequency magnitude spectrum.

Reference:
  3D MedDiffusion (IEEE TMI 2025) - loss design philosophy
  Laplacian Pyramid for style transfer (Ling et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPyramidLoss(nn.Module):
    """
    3D Laplacian Pyramid Loss.

    Builds a Gaussian pyramid by repeated downsampling, then computes
    Laplacian (high-frequency) residuals at each level. The loss is
    a weighted sum of L1 differences at each pyramid level.

    Higher levels capture coarse structure; lower levels capture fine detail.
    By default, finer levels get higher weight (emphasize high-freq preservation).

    Args:
        num_levels: Number of pyramid levels (default: 3)
        weight_mode: 'equal' or 'emphasize_high_freq'
    """

    def __init__(self, num_levels: int = 3, weight_mode: str = 'emphasize_high_freq'):
        super().__init__()
        self.num_levels = num_levels
        self.weight_mode = weight_mode

        if weight_mode == 'emphasize_high_freq':
            # Higher weight for finer levels (high-freq)
            # Level 0 (finest): weight 4, Level 1: 2, Level 2: 1, ...
            weights = [2 ** (num_levels - 1 - i) for i in range(num_levels)]
        else:
            weights = [1.0] * num_levels
        total = sum(weights)
        self.register_buffer('level_weights', torch.tensor([w / total for w in weights]))

    def _downsample_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by 2x using average pooling."""
        return F.avg_pool3d(x, kernel_size=2, stride=2, padding=0)

    def _upsample_3d(self, x: torch.Tensor, target_size) -> torch.Tensor:
        """Upsample to target size using trilinear interpolation."""
        return F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)

    def _build_pyramid(self, x: torch.Tensor):
        """Build Laplacian pyramid for input tensor."""
        gaussians = [x]
        current = x
        for _ in range(self.num_levels - 1):
            down = self._downsample_3d(current)
            gaussians.append(down)
            current = down

        laplacians = []
        for i in range(self.num_levels - 1):
            upsampled = self._upsample_3d(gaussians[i + 1], gaussians[i].shape[2:])
            laplacian = gaussians[i] - upsampled
            laplacians.append(laplacian)
        # Coarsest level is the residual Gaussian
        laplacians.append(gaussians[-1])
        return laplacians

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: (N, C, D, H, W)
            target: (N, C, D, H, W)

        Returns:
            Scalar loss
        """
        pred_pyramid = self._build_pyramid(prediction)
        tgt_pyramid = self._build_pyramid(target)

        loss = torch.tensor(0.0, device=prediction.device)
        for i in range(self.num_levels):
            level_loss = F.l1_loss(pred_pyramid[i], tgt_pyramid[i])
            loss = loss + self.level_weights[i] * level_loss

        return loss


class FFTFrequencyLoss(nn.Module):
    """
    3D FFT Frequency Domain Consistency Loss.

    Computes L1 distance between magnitude spectra of prediction and target
    in the Fourier domain. This directly penalizes frequency content mismatch.

    Args:
        weight_high_freq: If True, applies a high-pass weighting in frequency
                         domain to emphasize high-frequency consistency.
    """

    def __init__(self, weight_high_freq: bool = True):
        super().__init__()
        self.weight_high_freq = weight_high_freq
        self._freq_weight_cache = {}

    def _get_freq_weight(self, shape, device):
        """Create frequency-domain weight map (higher weight for high freq)."""
        key = (shape, device)
        if key not in self._freq_weight_cache:
            D, H, W = shape
            # Create distance from DC component (center of spectrum)
            dz = torch.linspace(-1, 1, D, device=device)
            dy = torch.linspace(-1, 1, H, device=device)
            dx = torch.linspace(-1, 1, W, device=device)
            grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij')
            dist = torch.sqrt(grid_z**2 + grid_y**2 + grid_x**2)
            # Weight: 1 at DC, 2 at Nyquist; emphasize high freq mildly
            freq_weight = 1.0 + dist
            self._freq_weight_cache[key] = freq_weight
        return self._freq_weight_cache[key]

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: (N, C, D, H, W)
            target: (N, C, D, H, W)

        Returns:
            Scalar loss
        """
        # Compute 3D FFT
        pred_fft = torch.fft.fftn(prediction, dim=(-3, -2, -1))
        tgt_fft = torch.fft.fftn(target, dim=(-3, -2, -1))

        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        tgt_mag = torch.abs(tgt_fft)

        if self.weight_high_freq:
            spatial_shape = prediction.shape[-3:]
            freq_w = self._get_freq_weight(spatial_shape, prediction.device)
            diff = (pred_mag - tgt_mag).abs() * freq_w
        else:
            diff = (pred_mag - tgt_mag).abs()

        return diff.mean()


class CombinedFrequencyLoss(nn.Module):
    """
    Combines Laplacian pyramid and FFT frequency losses.

    total = lap_weight * LaplacianPyramidLoss + fft_weight * FFTFrequencyLoss

    Args:
        lap_weight: Weight for Laplacian pyramid loss
        fft_weight: Weight for FFT loss
        num_levels: Laplacian pyramid levels
    """

    def __init__(self, lap_weight: float = 1.0, fft_weight: float = 0.1,
                 num_levels: int = 3):
        super().__init__()
        self.lap_loss = LaplacianPyramidLoss(num_levels=num_levels)
        self.fft_loss = FFTFrequencyLoss(weight_high_freq=True)
        self.lap_weight = lap_weight
        self.fft_weight = fft_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lap = self.lap_loss(prediction, target)
        fft = self.fft_loss(prediction, target)
        return self.lap_weight * lap + self.fft_weight * fft
