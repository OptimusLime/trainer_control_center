"""Autoencoder nn.Module.

The autoencoder is composed from layer modules. Reconstruction is NOT intrinsic —
it's just another Task. The decoder is optional (encoder-only models are valid).

forward() returns a dict[str, Tensor] keyed by ModelOutput enum values.
This is the contract that makes the Trainer model-agnostic.

print(autoencoder) shows every layer — PyTorch gives us this for free.
"""

from typing import Optional

import torch
import torch.nn as nn

from acc.model_output import ModelOutput


class Autoencoder(nn.Module):
    """Autoencoder with modular encoder/decoder built from nn.Module layers.

    Args:
        encoder_layers: List of nn.Module layers for the encoder.
        decoder_layers: Optional list of nn.Module layers for the decoder.
            If None, model is encoder-only and ReconstructionTask will refuse
            to attach.
    """

    def __init__(
        self,
        encoder_layers: list[nn.Module],
        decoder_layers: Optional[list[nn.Module]] = None,
    ):
        super().__init__()
        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = (
            nn.ModuleList(decoder_layers) if decoder_layers is not None else None
        )
        self.pool = nn.AdaptiveAvgPool2d(4)

        # Lazily computed
        self._latent_dim: Optional[int] = None

    @property
    def has_decoder(self) -> bool:
        return self.decoder is not None

    @property
    def latent_dim(self) -> int:
        """Dimension of the pooled flat latent vector. Computed lazily via dummy forward."""
        if self._latent_dim is None:
            self._compute_latent_dim()
        assert self._latent_dim is not None
        return self._latent_dim

    def _compute_latent_dim(self):
        """Run a dummy forward pass to determine latent_dim."""
        # Guess input shape: try 1-channel 64x64 first, common for our use case
        # Put dummy tensor on same device as model parameters
        device = next(self.parameters()).device
        dummy = torch.zeros(1, self._guess_in_channels(), 64, 64, device=device)
        with torch.no_grad():
            was_training = self.training
            self.eval()
            spatial = self._encode_spatial(dummy)
            pooled = self.pool(spatial)
            self._latent_dim = pooled.view(1, -1).shape[1]
            if was_training:
                self.train()

    def _guess_in_channels(self) -> int:
        """Infer input channels from the first encoder layer."""
        first_layer = self.encoder[0]
        # Walk into the layer to find the first Conv2d
        for module in first_layer.modules():
            if isinstance(module, nn.Conv2d):
                return module.in_channels
        # Fallback
        return 1

    def _encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to spatial feature map (before pooling)."""
        h = x
        for layer in self.encoder:
            h = layer(h)
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to spatial feature map."""
        return self._encode_spatial(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode spatial feature map to reconstructed image.

        Raises:
            RuntimeError: If model has no decoder.
        """
        if self.decoder is None:
            raise RuntimeError("Cannot decode: model has no decoder.")
        h = z
        for layer in self.decoder:
            h = layer(h)
        return h

    def get_pooled_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to flat [B, D] latent vector for probes."""
        spatial = self._encode_spatial(x)
        pooled = self.pool(spatial)
        return pooled.view(pooled.size(0), -1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass. Returns dict keyed by ModelOutput enum.

        Always contains: LATENT [B, D], SPATIAL [B, C, h, w].
        Contains RECONSTRUCTION [B, C, H, W] if model has decoder.
        """
        spatial = self._encode_spatial(x)
        pooled = self.pool(spatial)
        latent = pooled.view(pooled.size(0), -1)

        output = {
            ModelOutput.LATENT: latent,
            ModelOutput.SPATIAL: spatial,
        }

        if self.has_decoder:
            output[ModelOutput.RECONSTRUCTION] = self.decode(spatial)

        return output

    def get_encoder_layer_output(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Get the output of a specific encoder layer."""
        h = x
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i == layer_idx:
                return h
        raise IndexError(
            f"Encoder layer index {layer_idx} out of range (have {len(self.encoder)} layers)"
        )

    def freeze_encoder_up_to(self, layer_idx: int):
        """Freeze encoder layers 0..layer_idx (inclusive)."""
        for i, layer in enumerate(self.encoder):
            for param in layer.parameters():
                param.requires_grad = i > layer_idx

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def insert_encoder_layer(self, idx: int, layer: nn.Module):
        """Insert a layer into the encoder at position idx. (M6)"""
        layers = list(self.encoder)
        layers.insert(idx, layer)
        self.encoder = nn.ModuleList(layers)
        self._latent_dim = None  # invalidate cached dim

    def insert_decoder_layer(self, idx: int, layer: nn.Module):
        """Insert a layer into the decoder at position idx. (M6)"""
        if self.decoder is None:
            raise RuntimeError("Cannot insert decoder layer: model has no decoder.")
        layers = list(self.decoder)
        layers.insert(idx, layer)
        self.decoder = nn.ModuleList(layers)
