"""ReconstructionTask — uses model's decoder directly.

No probe head. Pixel loss + optional SSIM between input and reconstruction.
check_compatible: model must have decoder (has_decoder=True).

Reads RECONSTRUCTION and SPATIAL from the model_output dict — no re-encode.
Reconstruction is NOT special — it's just another task.

Supports two pixel-loss functions:
  - 'mse' (default): MSE penalizes large errors quadratically, preventing
    zero-collapse on sparse images like MNIST.
  - 'l1': L1 loss. Prone to zero-collapse when images are mostly black
    (outputting all-zeros gives low L1 ≈ 0.12 on MNIST).
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.eval_metric import EvalMetric
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


# ---------------------------------------------------------------------------
# SSIM implementation (Wang et al. 2004)
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """1D Gaussian kernel, normalized to sum=1."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """2D Gaussian kernel for SSIM, shape [1, 1, size, size]."""
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)  # outer product
    return k2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute mean SSIM between x and y.

    Args:
        x, y: [B, C, H, W] tensors in [0, 1] range.
        window_size: Gaussian window size (default 7 for 28x28 images).
        sigma: Gaussian sigma.

    Returns:
        Scalar mean SSIM (higher = more similar, range roughly [-1, 1]).
    """
    C1 = 0.01**2  # stabilization constants (assuming [0,1] range)
    C2 = 0.03**2

    kernel = _gaussian_kernel_2d(window_size, sigma).to(x.device, x.dtype)
    C = x.shape[1]
    # Expand kernel for depthwise conv across all channels
    kernel = kernel.expand(C, -1, -1, -1)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    """1 - SSIM. Range [0, 2], 0 = identical."""
    return 1.0 - ssim(x, y, **kwargs)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class ReconstructionTask(Task):
    """Reconstruction via the model's decoder. Pixel loss + optional SSIM.

    Requires autoencoder with a decoder (has_decoder=True).
    No probe head — reads RECONSTRUCTION directly from model forward output.

    Loss = pixel_loss + ssim_weight * (1 - SSIM).

    Args:
        loss_fn: 'mse' (default) or 'l1'. MSE prevents zero-collapse on
            sparse images; L1 is prone to it.
        ssim_weight: Weight for SSIM loss term. 0 = pure pixel loss.
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        ssim_weight: float = 0.0,
        loss_fn: str = "mse",
        **kwargs,
    ):
        super().__init__(name, dataset, **kwargs)
        self.ssim_weight = ssim_weight
        if loss_fn not in ("mse", "l1"):
            raise ValueError(f"loss_fn must be 'mse' or 'l1', got {loss_fn!r}")
        self.loss_fn = loss_fn

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        if not autoencoder.has_decoder:
            raise TaskError(
                f"ReconstructionTask requires model with decoder, "
                f"but model.has_decoder=False"
            )

    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        return None

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        """Pixel loss + optional SSIM reconstruction loss."""
        images = batch[0]
        recon = model_output[ModelOutput.RECONSTRUCTION]

        if self.loss_fn == "mse":
            pixel_loss = F.mse_loss(recon, images)
        else:
            pixel_loss = F.l1_loss(recon, images)

        if self.ssim_weight > 0:
            # Clamp to [0,1] for SSIM (it assumes this range for C1/C2)
            recon_clamped = recon.clamp(0, 1)
            s_loss = ssim_loss(recon_clamped, images)
            return pixel_loss + self.ssim_weight * s_loss
        return pixel_loss

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Compute PSNR, L1, and SSIM on eval split."""
        autoencoder.eval()

        total_l1 = 0.0
        total_mse = 0.0
        total_ssim = 0.0
        n_batches = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images = batch[0].to(device)
            model_output = autoencoder(images)
            recon = model_output[ModelOutput.RECONSTRUCTION]

            recon_clamped = recon.clamp(0, 1)
            total_l1 += F.l1_loss(recon_clamped, images).item()
            total_mse += F.mse_loss(recon_clamped, images).item()
            total_ssim += ssim(recon_clamped, images).item()
            n_batches += 1

        autoencoder.train()

        avg_l1 = total_l1 / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)
        avg_ssim = total_ssim / max(n_batches, 1)
        psnr = 10 * math.log10(1.0 / max(avg_mse, 1e-10))

        return {EvalMetric.L1: avg_l1, EvalMetric.PSNR: psnr, "ssim": avg_ssim}
