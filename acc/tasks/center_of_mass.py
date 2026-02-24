"""CenterOfMassTask — latent spatial pattern must match image position.

The center of mass of the latent spatial activation must match the center
of mass of the input image pixels. This gives the 3x3 latent spatial
meaning: position (0,0) corresponds to top-left, a centered digit produces
centered latent activation.

This is a "stepping stone" regularity — establish spatial correspondence
before asking the network to reconstruct.

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


def _compute_com(weights: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Compute center of mass along a coordinate grid.

    Args:
        weights: [B, ...] non-negative weights.
        coords: [...] coordinate values matching spatial dims of weights.

    Returns:
        [B] center of mass values.
    """
    # Flatten spatial dims
    B = weights.shape[0]
    w_flat = weights.reshape(B, -1)  # [B, N]
    c_flat = coords.reshape(-1)  # [N]

    w_sum = w_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
    com = (w_flat * c_flat.unsqueeze(0)).sum(dim=-1) / w_sum.squeeze(-1)  # [B]
    return com


class CenterOfMassTask(Task):
    """Penalize mismatch between image COM and latent COM.

    Loss = MSE(com_pred_x, com_gt_x) + MSE(com_pred_y, com_gt_y)
    """

    def __init__(self, name: str, dataset: AccDataset, **kwargs):
        super().__init__(name, dataset, **kwargs)
        # Coordinate grids — will be created lazily on correct device
        self._coords_cached: dict[str, torch.Tensor] = {}

    def _get_coords(self, H: int, W: int, device: torch.device):
        """Get or create coordinate grids for given resolution."""
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
        images = batch[0]  # [B, 1, 28, 28]
        if spatial is None:
            return torch.tensor(0.0, device=images.device)

        device = images.device

        # Ground truth COM from input pixels
        B, C_img, H_img, W_img = images.shape
        x_img, y_img = self._get_coords(H_img, W_img, device)
        img_flat = images[:, 0]  # [B, H, W] — single channel
        gt_cx = _compute_com(img_flat, x_img)  # [B]
        gt_cy = _compute_com(img_flat, y_img)  # [B]

        # Predicted COM from latent spatial map (pooled to 3x3)
        pooled = F.adaptive_avg_pool2d(spatial.abs(), 3)  # [B, C, 3, 3]
        # Average across channels
        z = pooled.mean(dim=1)  # [B, 3, 3]
        x_z, y_z = self._get_coords(3, 3, device)
        pred_cx = _compute_com(z, x_z)  # [B]
        pred_cy = _compute_com(z, y_z)  # [B]

        loss = F.mse_loss(pred_cx, gt_cx) + F.mse_loss(pred_cy, gt_cy)
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
            img_flat = images[:, 0]
            gt_cx = _compute_com(img_flat, x_img)
            gt_cy = _compute_com(img_flat, y_img)

            pooled = F.adaptive_avg_pool2d(spatial.abs(), 3)
            z = pooled.mean(dim=1)
            x_z, y_z = self._get_coords(3, 3, device)
            pred_cx = _compute_com(z, x_z)
            pred_cy = _compute_com(z, y_z)

            err = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).mean().item()
            total_error += err
            n += 1
            if n >= 10:
                break

        autoencoder.train()
        return {"com_mse": total_error / max(n, 1)}
