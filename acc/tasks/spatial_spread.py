"""SpatialSpreadTask — latent spread must match image aspect ratio/extent.

The second spatial moment (variance) of the latent activation should match
the second moment of the input pixel distribution. Captures aspect ratio
and spatial extent: a "1" is tall and narrow (small x-variance, large
y-variance), a "0" is round (similar variance in both axes).

Position (CenterOfMass) tells you *where*. Spread tells you *how the
mass is distributed*. This encodes shape without explicitly defining it.

Reads ModelOutput.SPATIAL (pooled to 3x3) and batch[0] (input image).
No probe head.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


def _compute_variance(
    weights: torch.Tensor, coords: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """Compute weighted variance around a center.

    Args:
        weights: [B, H, W] non-negative weights.
        coords: [H, W] coordinate values.
        center: [B] center of mass values.

    Returns:
        [B] variance values.
    """
    B = weights.shape[0]
    w_flat = weights.reshape(B, -1)  # [B, N]
    c_flat = coords.reshape(-1)  # [N]

    # Deviation from center
    dev = c_flat.unsqueeze(0) - center.unsqueeze(1)  # [B, N]
    w_sum = w_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
    var = (w_flat * dev**2).sum(dim=-1) / w_sum.squeeze(-1)  # [B]
    return var


def _compute_com(weights: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Compute center of mass."""
    B = weights.shape[0]
    w_flat = weights.reshape(B, -1)
    c_flat = coords.reshape(-1)
    w_sum = w_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (w_flat * c_flat.unsqueeze(0)).sum(dim=-1) / w_sum.squeeze(-1)


class SpatialSpreadTask(Task):
    """Penalize mismatch between image spatial spread and latent spatial spread.

    Loss = MSE(var_pred_x, var_gt_x) + MSE(var_pred_y, var_gt_y)
    """

    def __init__(self, name: str, dataset: AccDataset, **kwargs):
        super().__init__(name, dataset, **kwargs)
        self._coords_cached: dict[str, torch.Tensor] = {}

    def _get_coords(self, H: int, W: int, device: torch.device):
        key = f"{H}_{W}_{device}"
        if key not in self._coords_cached:
            y_coords = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, W)
            x_coords = torch.linspace(-1, 1, W, device=device).view(1, W).expand(H, W)
            self._coords_cached[key] = (x_coords, y_coords)
        return self._coords_cached[key]

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        pass

    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        return None

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        spatial = model_output.get(ModelOutput.SPATIAL)
        images = batch[0]
        if spatial is None:
            return torch.tensor(0.0, device=images.device)

        device = images.device

        # Ground truth spread from input pixels
        B, _, H_img, W_img = images.shape
        x_img, y_img = self._get_coords(H_img, W_img, device)
        img = images[:, 0]  # [B, H, W]
        gt_cx = _compute_com(img, x_img)
        gt_cy = _compute_com(img, y_img)
        gt_var_x = _compute_variance(img, x_img, gt_cx)
        gt_var_y = _compute_variance(img, y_img, gt_cy)

        # Predicted spread from latent
        pooled = F.adaptive_avg_pool2d(spatial.abs(), 3)  # [B, C, 3, 3]
        z = pooled.mean(dim=1)  # [B, 3, 3]
        x_z, y_z = self._get_coords(3, 3, device)
        pred_cx = _compute_com(z, x_z)
        pred_cy = _compute_com(z, y_z)
        pred_var_x = _compute_variance(z, x_z, pred_cx)
        pred_var_y = _compute_variance(z, y_z, pred_cy)

        loss = F.mse_loss(pred_var_x, gt_var_x) + F.mse_loss(pred_var_y, gt_var_y)
        return loss

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        autoencoder.eval()
        total_error = 0.0
        n = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images = batch[0].to(device)
            output = autoencoder(images)
            spatial = output.get(ModelOutput.SPATIAL)
            if spatial is None:
                break

            B, _, H_img, W_img = images.shape
            x_img, y_img = self._get_coords(H_img, W_img, device)
            img = images[:, 0]
            gt_cx = _compute_com(img, x_img)
            gt_cy = _compute_com(img, y_img)
            gt_vx = _compute_variance(img, x_img, gt_cx)
            gt_vy = _compute_variance(img, y_img, gt_cy)

            pooled = F.adaptive_avg_pool2d(spatial.abs(), 3)
            z = pooled.mean(dim=1)
            x_z, y_z = self._get_coords(3, 3, device)
            pred_cx = _compute_com(z, x_z)
            pred_cy = _compute_com(z, y_z)
            pred_vx = _compute_variance(z, x_z, pred_cx)
            pred_vy = _compute_variance(z, y_z, pred_cy)

            err = ((pred_vx - gt_vx) ** 2 + (pred_vy - gt_vy) ** 2).mean().item()
            total_error += err
            n += 1
            if n >= 10:
                break

        autoencoder.train()
        return {"spread_mse": total_error / max(n, 1)}
