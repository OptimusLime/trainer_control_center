"""ResBlock — residual block with GroupNorm + SiLU for deeper architectures."""

import torch.nn as nn


class ResBlock(nn.Module):
    """Standard residual block: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(num_groups, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(num_groups, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)
