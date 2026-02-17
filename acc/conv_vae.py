"""ConvVAE — standard convolutional VAE baseline.

Simple, well-understood architecture for validating that our training pipeline
works.  If this can't get recon L1 < 0.03 on MNIST 32x32 in 10 epochs,
something is fundamentally broken in the pipeline.

Architecture (for 32x32 input):
    Encoder: Conv(1->32, s2) -> Conv(32->64, s2) -> Flatten -> FC -> mu, logvar
    Decoder: FC -> Unflatten -> ConvT(64->32, s2) -> ConvT(32->1, s2) -> Sigmoid

Returns the standard ModelOutput dict with LATENT, RECONSTRUCTION, MU, LOGVAR.
"""

import torch
import torch.nn as nn

from acc.model_output import ModelOutput


class ConvVAE(nn.Module):
    """Standard convolutional VAE.

    Args:
        in_channels: Input image channels (1 for grayscale).
        latent_dim: Dimension of the latent vector z.
        image_size: Expected input image size (must be 32 or 64).
        base_channels: Base channel count. Encoder doubles at each stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        image_size: int = 32,
        base_channels: int = 32,
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self._image_size = image_size

        # Number of downsampling stages
        if image_size == 32:
            n_stages = 2  # 32 -> 16 -> 8
        elif image_size == 64:
            n_stages = 3  # 64 -> 32 -> 16 -> 8
        else:
            raise ValueError(f"Unsupported image_size={image_size}, must be 32 or 64")

        # --- Encoder ---
        enc_layers = []
        ch_in = in_channels
        ch_out = base_channels
        for _ in range(n_stages):
            enc_layers.extend([
                nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
            ch_out = ch_in * 2

        self.encoder = nn.Sequential(*enc_layers)

        # Spatial size after encoding
        self._enc_spatial = image_size // (2 ** n_stages)  # 8 for 32x32
        self._enc_channels = ch_in  # 64 for 32x32 with base=32
        enc_flat_dim = self._enc_channels * self._enc_spatial * self._enc_spatial

        self.fc_mu = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, enc_flat_dim),
            nn.ReLU(inplace=True),
        )

        dec_layers = []
        ch_in = self._enc_channels
        for i in range(n_stages):
            ch_out = ch_in // 2 if i < n_stages - 1 else in_channels
            if i < n_stages - 1:
                dec_layers.extend([
                    nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                ])
            else:
                # Last stage: no batchnorm, sigmoid output
                dec_layers.extend([
                    nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                    nn.Sigmoid(),
                ])
            ch_in = ch_out

        self.decoder = nn.Sequential(*dec_layers)

    @property
    def has_decoder(self) -> bool:
        return True

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.shape[0]

        # Encode
        h = self.encoder(x)
        h_flat = h.view(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self._reparameterize(mu, logvar)

        # Decode
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(B, self._enc_channels, self._enc_spatial, self._enc_spatial)
        reconstruction = self.decoder(h_dec)

        return {
            ModelOutput.LATENT: z,
            ModelOutput.SPATIAL: h,  # encoder feature map
            ModelOutput.RECONSTRUCTION: reconstruction,
            ModelOutput.MU: mu,
            ModelOutput.LOGVAR: logvar,
        }
