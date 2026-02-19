"""LinearAutoencoder — simplest possible autoencoder for baseline experiments.

Linear encoder (flatten -> Linear -> ReLU -> hidden) and decoder (Linear -> Sigmoid).
Returns {LATENT, RECONSTRUCTION}. No spatial features, no VAE, no factor groups.

This is the cheapest model for testing gradient gating: one hidden layer,
fast to train, easy to inspect weights and activations.
"""

import torch
import torch.nn as nn

from acc.model_output import ModelOutput


class LinearAutoencoder(nn.Module):
    """Flat linear autoencoder.

    Args:
        in_dim: Flattened input dimension (e.g., 784 for 28x28 MNIST).
        hidden_dim: Hidden layer / latent dimension.
        image_shape: Original image shape (C, H, W) for reshaping reconstruction.
            If None, reconstruction is returned flat.
    """

    def __init__(
        self,
        in_dim: int = 784,
        hidden_dim: int = 64,
        image_shape: tuple[int, ...] | None = (1, 28, 28),
    ):
        super().__init__()
        self._in_dim = in_dim
        self._hidden_dim = hidden_dim
        self._image_shape = image_shape

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),
        )

    @property
    def has_decoder(self) -> bool:
        return True

    @property
    def latent_dim(self) -> int:
        return self._hidden_dim

    def config(self) -> dict:
        """Serializable architectural config for checkpoint metadata."""
        return {
            "class": type(self).__name__,
            "in_dim": self._in_dim,
            "hidden_dim": self._hidden_dim,
            "image_shape": list(self._image_shape) if self._image_shape else None,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W] or already flat [B, D].

        Returns:
            {LATENT: [B, hidden_dim], RECONSTRUCTION: [B, C, H, W]}
        """
        # Flatten if spatial input
        if x.ndim > 2:
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
        else:
            batch_size = x.shape[0]
            x_flat = x

        latent = self.encoder(x_flat)  # [B, hidden_dim]
        recon_flat = self.decoder(latent)  # [B, in_dim]

        # Reshape reconstruction back to image shape
        if self._image_shape is not None:
            recon = recon_flat.view(batch_size, *self._image_shape)
        else:
            recon = recon_flat

        return {
            ModelOutput.LATENT: latent,
            ModelOutput.RECONSTRUCTION: recon,
        }
