"""FactorHead — projects spatial features to a single factor's latent slice.

Pools spatially then projects to the factor's latent dim.
One FactorHead per factor group in the encoder.

Supports VAE mode: outputs mu + logvar and reparameterizes.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FactorHeadOutput:
    """Output from a FactorHead. Always has z. Has mu/logvar if VAE mode."""

    z: torch.Tensor  # (B, factor_dim) — sampled if VAE, deterministic otherwise
    mu: Optional[torch.Tensor] = None  # (B, factor_dim) — only in VAE mode
    logvar: Optional[torch.Tensor] = None  # (B, factor_dim) — only in VAE mode


class FactorHead(nn.Module):
    """Project spatial features to a factor's latent slice.

    AdaptiveAvgPool2d(1) -> flatten -> Linear -> SiLU -> Linear -> z

    When vae=True, the final linear outputs 2*factor_dim (mu + logvar),
    and z is sampled via reparameterization trick.

    Args:
        spatial_channels: Number of channels from the backbone.
        factor_dim: Latent dimension for this factor.
        hidden_dim: Hidden layer size.
        vae: If True, output mu/logvar and reparameterize.
    """

    def __init__(
        self,
        spatial_channels: int,
        factor_dim: int,
        hidden_dim: int = 256,
        vae: bool = False,
    ):
        super().__init__()
        self.factor_dim = factor_dim
        self.vae = vae

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Linear(spatial_channels, hidden_dim),
            nn.SiLU(),
        )

        if vae:
            self.to_mu = nn.Linear(hidden_dim, factor_dim)
            self.to_logvar = nn.Linear(hidden_dim, factor_dim)
        else:
            self.to_z = nn.Linear(hidden_dim, factor_dim)

    def forward(self, features: torch.Tensor) -> FactorHeadOutput:
        """features: (B, C, H, W) -> FactorHeadOutput"""
        x = self.pool(features).flatten(1)
        h = self.shared(x)

        if self.vae:
            mu = self.to_mu(h)
            logvar = self.to_logvar(h)
            z = self._reparameterize(mu, logvar)
            return FactorHeadOutput(z=z, mu=mu, logvar=logvar)
        else:
            z = self.to_z(h)
            return FactorHeadOutput(z=z)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
