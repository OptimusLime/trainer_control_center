"""FactorSlotAutoencoder — cross-attention VAE with designated factor groups.

A completely different nn.Module from the simple Autoencoder, but returns the
same ModelOutput dict. The Trainer doesn't care which model it trains — it just
calls forward() and passes the dict to tasks.

Architecture:
    Encoder:  image -> CNN backbone -> per-factor projection heads -> mu/logvar -> z (reparameterized)
    Decoder:  z -> initial spatial -> [upsample + cross-attn(factor embeds)] x N -> image
"""

import torch
import torch.nn as nn

from acc.model_output import ModelOutput
from acc.factor_group import FactorGroup, validate_factor_groups
from acc.layers.res_block import ResBlock
from acc.layers.factor_head import FactorHead
from acc.layers.cross_attention import FactorEmbedder, CrossAttentionBlock


class FactorSlotAutoencoder(nn.Module):
    """Factor-Slot Cross-Attention VAE.

    Each factor group has a dedicated FactorHead that outputs mu/logvar
    and reparameterizes. The KLDivergenceTask reads MU/LOGVAR from the
    model output dict to compute per-factor or full-latent KL.

    Args:
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB).
        factor_groups: List of FactorGroup defining the latent layout.
        backbone_channels: Channel progression for the CNN backbone.
        embed_dim: Shared embedding dimension for cross-attention.
        image_size: Expected input image size (for computing spatial dims).
    """

    def __init__(
        self,
        in_channels: int,
        factor_groups: list[FactorGroup],
        backbone_channels: list[int] | None = None,
        embed_dim: int = 64,
        image_size: int = 32,
    ):
        super().__init__()

        # Default backbone: 2 stages for 32x32 (→ 8x8 spatial), 3 for 64x64
        if backbone_channels is None:
            if image_size <= 32:
                backbone_channels = [64, 128]
            else:
                backbone_channels = [64, 128, 256]

        self.factor_groups = factor_groups
        self.total_latent_dim = sum(fg.latent_dim for fg in factor_groups)
        validate_factor_groups(factor_groups, self.total_latent_dim)

        self._in_channels = in_channels
        self.embed_dim = embed_dim

        # --- Encoder backbone ---
        backbone_layers = []
        ch_in = in_channels
        for ch_out in backbone_channels:
            backbone_layers.extend(
                [
                    nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                    ResBlock(ch_out),
                ]
            )
            ch_in = ch_out
        self.backbone = nn.Sequential(*backbone_layers)
        self.backbone_out_channels = backbone_channels[-1]

        # --- Per-factor projection heads (VAE mode: mu/logvar + reparameterize) ---
        self.factor_heads = nn.ModuleDict(
            {
                fg.name: FactorHead(
                    self.backbone_out_channels, fg.latent_dim, vae=True
                )
                for fg in factor_groups
            }
        )

        # --- Decoder ---
        spatial_size = image_size // (2 ** len(backbone_channels))
        self._decoder_init_spatial = spatial_size
        decoder_init_ch = backbone_channels[-1]
        self.decoder_init = nn.Sequential(
            nn.Linear(
                self.total_latent_dim, decoder_init_ch * spatial_size * spatial_size
            ),
            nn.SiLU(),
        )
        self._decoder_init_ch = decoder_init_ch

        # Factor embedder for cross-attention
        factor_dims = {fg.name: fg.latent_dim for fg in factor_groups}
        self.factor_embedder = FactorEmbedder(factor_dims, embed_dim)

        # Upsample stages with cross-attention
        decoder_channels = list(reversed(backbone_channels))

        self.decoder_stages = nn.ModuleList()
        self.cross_attn_stages = nn.ModuleList()

        ch_in = decoder_init_ch
        for i, ch_out in enumerate(decoder_channels):
            self.decoder_stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(ch_in, ch_out, 3, padding=1),
                    ResBlock(ch_out),
                )
            )
            # Cross-attention at all but the last (highest res) stage
            if i < len(decoder_channels) - 1:
                self.cross_attn_stages.append(CrossAttentionBlock(ch_out, embed_dim))
            else:
                self.cross_attn_stages.append(None)
            ch_in = ch_out

        # Final output conv
        self.to_output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], in_channels, 1),
            nn.Sigmoid(),
        )

    @property
    def has_decoder(self) -> bool:
        return True

    @property
    def latent_dim(self) -> int:
        return self.total_latent_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass. Returns ModelOutput dict.

        Always contains: LATENT, SPATIAL, RECONSTRUCTION, FACTOR_SLICES, MU, LOGVAR.
        """
        B = x.shape[0]

        # Encode
        spatial = self.backbone(x)  # (B, C, h, w)

        # Per-factor projection with reparameterization
        factor_slices = {}
        z_parts = []
        mu_parts = []
        logvar_parts = []

        for fg in self.factor_groups:
            head_out = self.factor_heads[fg.name](spatial)
            factor_slices[fg.name] = head_out.z
            z_parts.append(head_out.z)
            mu_parts.append(head_out.mu)
            logvar_parts.append(head_out.logvar)

        z = torch.cat(z_parts, dim=1)  # (B, total_latent_dim)
        mu = torch.cat(mu_parts, dim=1)  # (B, total_latent_dim)
        logvar = torch.cat(logvar_parts, dim=1)  # (B, total_latent_dim)

        # Decode
        factor_embeds = self.factor_embedder(factor_slices)  # (B, N, D)

        h = self.decoder_init(z).view(
            B,
            self._decoder_init_ch,
            self._decoder_init_spatial,
            self._decoder_init_spatial,
        )

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
