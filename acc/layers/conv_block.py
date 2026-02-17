"""Convolutional building blocks for autoencoders.

ConvBlock: Conv2d + BatchNorm + ReLU (encoder building block)
ConvTransposeBlock: ConvTranspose2d + BatchNorm + ReLU (decoder building block)
"""

from dataclasses import dataclass

import torch.nn as nn


class ConvBlock(nn.Module):
    """Encoder building block: Conv2d -> BatchNorm2d -> ReLU."""

    @dataclass
    class Config:
        in_channels: int
        out_channels: int
        kernel_size: int = 3
        stride: int = 1
        padding: int = 1

    def __init__(self, config: "ConvBlock.Config"):
        super().__init__()
        self.config = config
        self.conv = nn.Conv2d(
            config.in_channels,
            config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
        )
        self.bn = nn.BatchNorm2d(config.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    """Decoder building block: ConvTranspose2d -> BatchNorm2d -> ReLU.

    For the final decoder layer, use output_activation='sigmoid' to clamp
    output to [0, 1] for image reconstruction.
    """

    @dataclass
    class Config:
        in_channels: int
        out_channels: int
        kernel_size: int = 3
        stride: int = 1
        padding: int = 1
        output_padding: int = 0
        output_activation: str = "relu"  # "relu" or "sigmoid"

    def __init__(self, config: "ConvTransposeBlock.Config"):
        super().__init__()
        self.config = config
        self.conv_t = nn.ConvTranspose2d(
            config.in_channels,
            config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            output_padding=config.output_padding,
        )
        self.bn = nn.BatchNorm2d(config.out_channels)
        if config.output_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv_t(x)))
