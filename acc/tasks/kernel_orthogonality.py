"""KernelOrthogonalityTask — decorrelate conv kernels across channels.

Computes the Gram matrix of normalized, flattened kernels and penalizes
off-diagonal elements. Ensures different channels learn different features.

Reads model parameters directly (encoder conv layer weights).
No-op when encoder has 1 channel.
No probe head.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


class KernelOrthogonalityTask(Task):
    """Penalize correlated conv kernels across output channels.

    Loss = ||W_norm @ W_norm.T - I||^2 (off-diagonal only)

    This reads the encoder conv weights directly. When the encoder has
    only 1 output channel, the loss is zero (nothing to decorrelate).

    The task stores a reference to the autoencoder (set during attach)
    and accesses its encoder layers' weights at compute_loss time.
    """

    def __init__(self, name: str, dataset: AccDataset, **kwargs):
        super().__init__(name, dataset, **kwargs)
        self._autoencoder: Optional[nn.Module] = None

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        # Need encoder_layers with conv weights
        if not hasattr(autoencoder, "encoder_layers"):
            raise TaskError(
                f"KernelOrthogonalityTask requires model with encoder_layers attribute"
            )

    def attach(self, autoencoder: nn.Module) -> None:
        super().attach(autoencoder)
        self._autoencoder = autoencoder

    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        return None

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        if self._autoencoder is None:
            return torch.tensor(0.0, device=batch[0].device)

        total_loss = torch.tensor(0.0, device=batch[0].device)
        n_layers = 0

        for layer in self._autoencoder.encoder_layers:  # type: ignore
            weight = layer.conv.weight  # [C_out, C_in, K, K]
            C_out = weight.shape[0]
            if C_out <= 1:
                continue

            # Flatten each kernel: [C_out, C_in*K*K]
            W_flat = weight.view(C_out, -1)

            # Normalize rows
            W_norm = F.normalize(W_flat, dim=-1)  # [C_out, D]

            # Gram matrix
            gram = W_norm @ W_norm.T  # [C_out, C_out]

            # Off-diagonal penalty: ||gram - I||^2
            identity = torch.eye(C_out, device=gram.device)
            off_diag = gram - identity
            total_loss = total_loss + (off_diag**2).mean()
            n_layers += 1

        return total_loss / max(n_layers, 1)

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        total_off_diag = 0.0
        n_layers = 0

        if not hasattr(autoencoder, "encoder_layers"):
            return {"kernel_ortho": 0.0}

        for layer in autoencoder.encoder_layers:
            weight = layer.conv.weight
            C_out = weight.shape[0]
            if C_out <= 1:
                continue
            W_flat = weight.view(C_out, -1)
            W_norm = F.normalize(W_flat, dim=-1)
            gram = W_norm @ W_norm.T
            identity = torch.eye(C_out, device=gram.device)
            off_diag_mag = ((gram - identity) ** 2).mean().item()
            total_off_diag += off_diag_mag
            n_layers += 1

        return {"kernel_ortho": total_off_diag / max(n_layers, 1)}
