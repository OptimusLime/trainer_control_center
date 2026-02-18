"""FactorSlotAutoencoder — cross-attention VAE with designated factor groups.

A completely different nn.Module from the simple Autoencoder, but returns the
same ModelOutput dict. The Trainer doesn't care which model it trains — it just
calls forward() and passes the dict to tasks.

Architecture:
    Encoder:  image -> CNN backbone (keeps spatial) -> conv mu/logvar -> reparameterize -> z
              Factor heads read per-factor slices from the flat z vector.
    Decoder:  z reshaped to spatial -> [upsample + cross-attn(factor embeds)] x N -> image

Key design: the bottleneck is spatial (e.g., 8x8 with latent_channels channels),
NOT a global-avg-pooled vector. This preserves spatial information through the
bottleneck while still factoring the latent space into named groups.
"""

import torch
import torch.nn as nn

from acc.model_output import ModelOutput
from acc.factor_group import FactorGroup, validate_factor_groups
from acc.layers.res_block import ResBlock
from acc.layers.cross_attention import FactorEmbedder, CrossAttentionBlock


class FactorSlotAutoencoder(nn.Module):
    """Factor-Slot Cross-Attention VAE with spatial bottleneck.

    The encoder produces a spatial feature map, then a 1x1 conv projects to
    latent_channels to get mu/logvar at each spatial position. The total
    latent dim = latent_channels * spatial_h * spatial_w, but the factor
    groups index into the CHANNEL dimension only (shared across spatial
    positions).

    The flat z vector (for probes, KL, classification) is obtained by
    reshaping the spatial latent. Factor slices are extracted by channel
    indexing then spatial pooling.

    Args:
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB).
        factor_groups: List of FactorGroup defining channel-wise latent layout.
        backbone_channels: Channel progression for the CNN backbone.
        embed_dim: Shared embedding dimension for cross-attention.
        image_size: Expected input image size (for computing spatial dims).
        detach_factor_grad: If True, detach factor slices before feeding to
            the decoder's cross-attention (FactorEmbedder). This blocks
            reconstruction gradients from flowing back into the factor
            channels, so only probe/task gradients shape what those channels
            encode. The decoder still READS factor tokens as context, but
            can't corrupt them into reconstruction-optimal representations.
    """

    def __init__(
        self,
        in_channels: int,
        factor_groups: list[FactorGroup],
        backbone_channels: list[int] | None = None,
        embed_dim: int = 64,
        image_size: int = 32,
        use_cross_attention: bool = True,
        detach_factor_grad: bool = False,
    ):
        super().__init__()

        # Default backbone: 2 stages for 32x32 (→ 8x8 spatial), 3 for 64x64
        if backbone_channels is None:
            if image_size <= 32:
                backbone_channels = [64, 128]
            else:
                backbone_channels = [64, 128, 256]

        self.factor_groups = factor_groups
        self.use_cross_attention = use_cross_attention
        self.detach_factor_grad = detach_factor_grad
        # latent_channels = sum of factor group dims (channel-wise)
        self.latent_channels = sum(fg.latent_dim for fg in factor_groups)
        validate_factor_groups(factor_groups, self.latent_channels)

        self._in_channels = in_channels
        self._image_size = image_size
        self._backbone_channels = list(backbone_channels)
        self.embed_dim = embed_dim

        # Spatial size after backbone
        self._spatial_size = image_size // (2 ** len(backbone_channels))
        # Total flat latent dim = channels * spatial * spatial
        self.total_latent_dim = self.latent_channels * self._spatial_size * self._spatial_size

        # --- Encoder backbone ---
        backbone_layers = []
        ch_in = in_channels
        for ch_out in backbone_channels:
            backbone_layers.extend([
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                ResBlock(ch_out),
            ])
            ch_in = ch_out
        self.backbone = nn.Sequential(*backbone_layers)
        self.backbone_out_channels = backbone_channels[-1]

        # --- Spatial VAE bottleneck (1x1 conv to mu/logvar) ---
        self.to_mu = nn.Conv2d(self.backbone_out_channels, self.latent_channels, 1)
        self.to_logvar = nn.Conv2d(self.backbone_out_channels, self.latent_channels, 1)

        # --- Factor embedder for cross-attention ---
        # Built even when use_cross_attention=False so factor_slices still work
        factor_dims = {fg.name: fg.latent_dim for fg in factor_groups}
        self.factor_embedder = FactorEmbedder(factor_dims, embed_dim)

        # --- Decoder ---
        # Decoder starts from the spatial latent (latent_channels, h, w)
        # and upsamples back to image resolution
        decoder_channels = list(reversed(backbone_channels))

        self.decoder_input = nn.Sequential(
            nn.Conv2d(self.latent_channels, decoder_channels[0], 3, padding=1),
            ResBlock(decoder_channels[0]),
        )

        self.decoder_stages = nn.ModuleList()
        self.cross_attn_stages = nn.ModuleList()

        ch_in = decoder_channels[0]
        for i, ch_out in enumerate(decoder_channels):
            self.decoder_stages.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
                ResBlock(ch_out),
            ))
            # Cross-attention at all but the last (highest res) stage
            if use_cross_attention and i < len(decoder_channels) - 1:
                self.cross_attn_stages.append(CrossAttentionBlock(ch_out, embed_dim))
            else:
                self.cross_attn_stages.append(None)
            ch_in = ch_out

        # Final output conv — no sigmoid!
        # Sigmoid causes gradient death (pre-sigmoid values go very negative,
        # sigmoid saturates to 0, gradient vanishes, decoder never recovers).
        # Without sigmoid, outputs naturally stay in ~[-0.3, 1.3] range
        # because targets are in [0, 1]. Reconstruction task clamps for eval.
        self.to_output = nn.Conv2d(decoder_channels[-1], in_channels, 1)

    @property
    def has_decoder(self) -> bool:
        return True

    @property
    def latent_dim(self) -> int:
        return self.total_latent_dim

    def config(self) -> dict:
        """Return architectural config as a serializable dict.

        Used by CheckpointStore.save() to persist model config alongside
        weights so you can understand what a checkpoint contains without
        needing the recipe source code.
        """
        return {
            "class": type(self).__name__,
            "in_channels": self._in_channels,
            "latent_channels": self.latent_channels,
            "image_size": self._image_size,
            "backbone_channels": self._backbone_channels,
            "embed_dim": self.embed_dim,
            "use_cross_attention": self.use_cross_attention,
            "detach_factor_grad": self.detach_factor_grad,
            "factor_groups": [
                {"name": fg.name, "start": fg.latent_start, "end": fg.latent_end, "dim": fg.latent_dim}
                for fg in self.factor_groups
            ],
        }

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _extract_factor_slices(self, z_spatial: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract per-factor vectors from spatial latent by channel slicing + global avg pool.

        z_spatial: (B, latent_channels, h, w)
        Returns: dict mapping factor_name -> (B, factor_dim) pooled vectors.
        """
        slices = {}
        for fg in self.factor_groups:
            # Channel slice
            factor_spatial = z_spatial[:, fg.latent_start:fg.latent_end, :, :]  # (B, factor_dim, h, w)
            # Global average pool to get a vector per factor
            factor_vec = factor_spatial.mean(dim=(2, 3))  # (B, factor_dim)
            slices[fg.name] = factor_vec
        return slices

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass. Returns ModelOutput dict.

        Always contains: LATENT, SPATIAL, RECONSTRUCTION, FACTOR_SLICES, MU, LOGVAR.
        """
        B = x.shape[0]

        # Encode to spatial features
        spatial = self.backbone(x)  # (B, C, h, w)

        # Spatial mu/logvar (per-pixel VAE bottleneck)
        mu_spatial = self.to_mu(spatial)       # (B, latent_channels, h, w)
        logvar_spatial = self.to_logvar(spatial)  # (B, latent_channels, h, w)
        z_spatial = self._reparameterize(mu_spatial, logvar_spatial)  # (B, latent_channels, h, w)

        # Flat vectors for tasks (KL, classification, etc.)
        mu = mu_spatial.flatten(1)       # (B, latent_channels * h * w)
        logvar = logvar_spatial.flatten(1)  # (B, latent_channels * h * w)
        z = z_spatial.flatten(1)         # (B, total_latent_dim)

        # Per-factor slices (channel groups, spatially pooled)
        factor_slices = self._extract_factor_slices(z_spatial)

        # Factor embeddings for cross-attention.
        # When detach_factor_grad is True, we detach factor slices before
        # feeding to the decoder's FactorEmbedder. This blocks reconstruction
        # gradients from corrupting factor channels — only probe/task
        # gradients shape what those channels encode.
        if self.detach_factor_grad:
            decoder_slices = {k: v.detach() for k, v in factor_slices.items()}
        else:
            decoder_slices = factor_slices
        factor_embeds = self.factor_embedder(decoder_slices)  # (B, N, D)

        # Decode from spatial latent
        h = self.decoder_input(z_spatial)  # (B, decoder_ch, h, w)

        for stage, cross_attn in zip(self.decoder_stages, self.cross_attn_stages):
            h = stage(h)
            if cross_attn is not None:
                h = cross_attn(h, factor_embeds)

        reconstruction = self.to_output(h)

        return {
            ModelOutput.LATENT: z,
            ModelOutput.SPATIAL: spatial,
            ModelOutput.RECONSTRUCTION: reconstruction,
            ModelOutput.FACTOR_SLICES: factor_slices,
            ModelOutput.MU: mu,
            ModelOutput.LOGVAR: logvar,
        }
