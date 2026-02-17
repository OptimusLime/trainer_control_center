"""Cross-attention layers for the Factor-Slot decoder.

FactorEmbedder: projects each factor slice to shared embed_dim tokens.
CrossAttentionBlock: spatial features (Q) attend to factor embeddings (K, V).
"""

import torch
import torch.nn as nn


class FactorEmbedder(nn.Module):
    """Projects each factor slice into a shared embedding space.

    Produces N factor tokens that serve as K,V for cross-attention.
    One MLP per factor group.
    """

    def __init__(self, factor_dims: dict[str, int], embed_dim: int):
        """
        Args:
            factor_dims: dict mapping factor_name -> latent_dim for that factor.
            embed_dim: shared embedding dimension D.
        """
        super().__init__()
        self.factor_names = list(factor_dims.keys())
        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for name, dim in factor_dims.items()
            }
        )

    def forward(self, factor_slices: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            factor_slices: dict mapping factor_name -> (B, factor_dim) tensors.

        Returns:
            (B, N_factors, embed_dim) — one token per factor group.
        """
        tokens = []
        for name in self.factor_names:
            token = self.projectors[name](factor_slices[name])  # (B, D)
            tokens.append(token.unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, N, D)


class CrossAttentionBlock(nn.Module):
    """Spatial features (Q) attend to factor embeddings (K, V).

    Each spatial position learns which factors are relevant to it.
    This is the mechanism that forces spatial specialization per factor.

    Set `store_attn = True` before a forward pass to capture attention
    weights in `self.last_attn_weights` (B, heads, HW, N_factors).
    """

    def __init__(self, spatial_channels: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.to_q = nn.Linear(spatial_channels, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, spatial_channels)

        self.norm_spatial = nn.LayerNorm(spatial_channels)
        self.norm_factors = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(spatial_channels),
            nn.Linear(spatial_channels, spatial_channels * 4),
            nn.GELU(),
            nn.Linear(spatial_channels * 4, spatial_channels),
        )

        # Attention capture
        self.store_attn: bool = False
        self.last_attn_weights: torch.Tensor | None = None
        self.last_spatial_shape: tuple[int, int] | None = None

    def forward(
        self,
        spatial: torch.Tensor,  # (B, C, H, W)
        factor_embeds: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        B, C, H, W = spatial.shape
        N = factor_embeds.shape[1]

        # Flatten spatial to sequence
        x = spatial.flatten(2).transpose(1, 2)  # (B, HW, C)

        x_norm = self.norm_spatial(x)
        f_norm = self.norm_factors(factor_embeds)

        Q = self.to_q(x_norm)  # (B, HW, D)
        K = self.to_k(f_norm)  # (B, N, D)
        V = self.to_v(f_norm)  # (B, N, D)

        # Multi-head attention
        head_dim = self.embed_dim // self.num_heads
        Q = Q.view(B, H * W, self.num_heads, head_dim).transpose(
            1, 2
        )  # (B, heads, HW, hd)
        K = K.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, heads, N, hd)
        V = V.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, heads, N, hd)

        attn = (Q @ K.transpose(-2, -1)) / (head_dim**0.5)  # (B, heads, HW, N)
        attn = attn.softmax(dim=-1)

        # Optionally store attention weights for visualization
        if self.store_attn:
            self.last_attn_weights = attn.detach()  # (B, heads, HW, N)
            self.last_spatial_shape = (H, W)

        out = attn @ V  # (B, heads, HW, hd)
        out = out.transpose(1, 2).reshape(B, H * W, self.embed_dim)  # (B, HW, D)

        # Residual + project back to spatial channels
        x = x + self.out_proj(out)
        x = x + self.ffn(x)

        return x.transpose(1, 2).view(B, C, H, W)
