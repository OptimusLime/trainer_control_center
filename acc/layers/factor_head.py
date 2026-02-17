"""FactorHead — projects spatial features to a single factor's latent slice.

Pools spatially then projects to the factor's latent dim.
One FactorHead per factor group in the encoder.
"""

import torch
import torch.nn as nn


class FactorHead(nn.Module):
    """Project spatial features to a factor's latent slice.

    AdaptiveAvgPool2d(1) -> flatten -> Linear -> SiLU -> Linear -> factor_dim
    """

    def __init__(self, spatial_channels: int, factor_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(spatial_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, factor_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, H, W) -> (B, factor_dim)"""
        x = self.pool(features).flatten(1)
        return self.proj(x)
