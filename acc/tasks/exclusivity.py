"""WithinImageExclusivityTask — force concentrated per-image activation.

For each individual image, the spatial activation pattern across the 3x3
latent should be concentrated: a few positions highly active, the rest quiet.
Minimizes entropy of the per-image spatial activation distribution.

Works with LifetimeSparsity: lifetime says "don't fire for everything",
exclusivity says "when you fire, be decisive about where."

Reads ModelOutput.SPATIAL, pools to 3x3, flattens spatial, applies
temperature-controlled softmax, computes entropy.

No probe head.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


class WithinImageExclusivityTask(Task):
    """Minimize entropy of per-image spatial activation distribution.

    Loss = mean(entropy(softmax(z_flat / temperature, dim=spatial)))

    Args:
        temperature: Controls softmax sharpness. Lower = sharper
            distributions penalized less. Default 0.5.
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        temperature: float = 0.5,
        **kwargs,
    ):
        super().__init__(name, dataset, **kwargs)
        self.temperature = temperature

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        pass

    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        return None

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        spatial = model_output.get(ModelOutput.SPATIAL)
        if spatial is None:
            return torch.tensor(0.0, device=batch[0].device)

        # Pool to 3x3, flatten spatial dims
        pooled = F.adaptive_avg_pool2d(spatial, 3)  # [B, C, 3, 3]
        B, C = pooled.shape[:2]
        z_flat = pooled.view(B, C, 9)  # [B, C, 9]

        # Temperature-scaled softmax over spatial positions
        p = F.softmax(z_flat / self.temperature, dim=-1)  # [B, C, 9]

        # Entropy: -sum(p * log(p))
        # Clamp for numerical stability
        log_p = torch.log(p.clamp(min=1e-8))
        entropy = -(p * log_p).sum(dim=-1)  # [B, C]

        # Mean entropy across batch and channels
        return entropy.mean()

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        autoencoder.eval()
        total_entropy = 0.0
        n = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images = batch[0].to(device)
            output = autoencoder(images)
            spatial = output.get(ModelOutput.SPATIAL)
            if spatial is None:
                break
            pooled = F.adaptive_avg_pool2d(spatial, 3)
            B, C = pooled.shape[:2]
            z_flat = pooled.view(B, C, 9)
            p = F.softmax(z_flat / self.temperature, dim=-1)
            log_p = torch.log(p.clamp(min=1e-8))
            entropy = -(p * log_p).sum(dim=-1).mean().item()
            total_entropy += entropy
            n += 1
            if n >= 10:
                break

        autoencoder.train()
        return {"mean_entropy": total_entropy / max(n, 1)}

    def describe(self) -> dict:
        info = super().describe()
        info["temperature"] = self.temperature
        return info
