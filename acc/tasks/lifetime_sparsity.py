"""LifetimeSparsityTask — force latent positions to specialize.

Across a batch, each latent spatial position should only activate
significantly for a small fraction of images. Penalizes diffuse,
always-on features that are the core FER signature.

Reads ModelOutput.SPATIAL (encoder output pre-pool), pools to 3x3,
applies soft threshold, computes per-position lifetime (fraction of
batch that activates it), and penalizes deviation from target.

No probe head. Operates directly on spatial activations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


class LifetimeSparsityTask(Task):
    """Penalize latent positions that activate for too many images.

    Loss = mean((lifetime - target_lifetime)^2)
    where lifetime = mean_over_batch(sigmoid(sharpness * z_pooled))

    Args:
        target_lifetime: Target fraction of batch each position should
            activate for. Lower = more specialized. Default 0.1.
        sharpness: Steepness of the soft threshold sigmoid. Higher = more
            binary activation decision. Default 10.0.
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        target_lifetime: float = 0.1,
        sharpness: float = 10.0,
        **kwargs,
    ):
        super().__init__(name, dataset, **kwargs)
        self.target_lifetime = target_lifetime
        self.sharpness = sharpness

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        # We need SPATIAL in the model output. Can't check at attach time
        # since we don't have a forward pass, but ConvCPPN always emits it.
        pass

    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        return None

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        spatial = model_output.get(ModelOutput.SPATIAL)
        if spatial is None:
            return torch.tensor(0.0, device=batch[0].device)

        # Pool to 3x3 to match bottleneck resolution
        pooled = F.adaptive_avg_pool2d(spatial, 3)  # [B, C, 3, 3]

        # Soft binary activation: sigmoid(sharpness * z)
        active = torch.sigmoid(self.sharpness * pooled)  # [B, C, 3, 3]

        # Lifetime: fraction of batch that activates each position
        lifetime = active.mean(dim=0)  # [C, 3, 3]

        # Penalize deviation from target
        loss = ((lifetime - self.target_lifetime) ** 2).mean()
        return loss

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        autoencoder.eval()
        total_lifetime = None
        n_batches = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images = batch[0].to(device)
            output = autoencoder(images)
            spatial = output.get(ModelOutput.SPATIAL)
            if spatial is None:
                break
            pooled = F.adaptive_avg_pool2d(spatial, 3)
            active = torch.sigmoid(self.sharpness * pooled)
            lt = active.mean(dim=0)
            if total_lifetime is None:
                total_lifetime = lt
            else:
                total_lifetime = total_lifetime + lt
            n_batches += 1
            if n_batches >= 10:
                break

        autoencoder.train()

        if total_lifetime is None or n_batches == 0:
            return {"lifetime_mean": 0.0, "lifetime_std": 0.0}

        avg_lifetime = total_lifetime / n_batches
        return {
            "lifetime_mean": avg_lifetime.mean().item(),
            "lifetime_std": avg_lifetime.std().item(),
        }

    def describe(self) -> dict:
        info = super().describe()
        info["target_lifetime"] = self.target_lifetime
        info["sharpness"] = self.sharpness
        return info
