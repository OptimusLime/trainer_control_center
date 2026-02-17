"""Attention map extraction for FactorSlotAutoencoder.

Extracts per-factor spatial attention maps from cross-attention layers.
The maps show which spatial regions each factor controls in the decoder.

Usage:
    maps = extract_attention_maps(model, images)
    # maps["thickness"] -> (B, H_out, W_out) tensor of attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.layers.cross_attention import CrossAttentionBlock


def extract_attention_maps(
    model: nn.Module,
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extract per-factor attention heatmaps from a FactorSlotAutoencoder.

    Runs a forward pass with attention storage enabled on all cross-attention
    stages, then averages attention weights across heads and stages.

    Args:
        model: A FactorSlotAutoencoder (must have factor_groups and cross_attn_stages).
        images: (B, C, H, W) input images on the model's device.

    Returns:
        dict mapping factor_name -> (B, H_out, W_out) attention heatmaps.
        H_out, W_out match the output image resolution.
        Values are in [0, 1] — for each spatial position, attention sums
        to 1 across factors.
    """
    if not hasattr(model, "cross_attn_stages"):
        raise ValueError("Model has no cross_attn_stages — not a FactorSlotAutoencoder")
    if not hasattr(model, "factor_groups"):
        raise ValueError("Model has no factor_groups")

    # Enable attention storage
    attn_blocks = []
    for block in model.cross_attn_stages:
        if block is not None:
            block.store_attn = True
            attn_blocks.append(block)

    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(images)

        # Collect attention weights from all stages
        # Each block has: last_attn_weights (B, heads, HW, N_factors)
        #                  last_spatial_shape (H, W) at that stage's resolution
        factor_names = [fg.name for fg in model.factor_groups]
        n_factors = len(factor_names)
        B = images.shape[0]
        out_h, out_w = images.shape[2], images.shape[3]

        # Accumulate per-factor maps at output resolution
        accumulated = torch.zeros(B, n_factors, out_h, out_w, device=images.device)
        n_stages = 0

        for block in attn_blocks:
            if block.last_attn_weights is None:
                continue

            weights = block.last_attn_weights  # (B, heads, HW, N)
            H, W = block.last_spatial_shape

            # Average across heads -> (B, HW, N)
            avg_attn = weights.mean(dim=1)

            # Reshape to spatial -> (B, N, H, W)
            spatial_attn = avg_attn.transpose(1, 2).view(B, n_factors, H, W)

            # Upsample to output resolution
            if H != out_h or W != out_w:
                spatial_attn = F.interpolate(
                    spatial_attn, size=(out_h, out_w), mode="bilinear", align_corners=False
                )

            accumulated += spatial_attn
            n_stages += 1

        if n_stages > 0:
            accumulated /= n_stages

        # Normalize: for each spatial position, attention should sum to 1 across factors
        # (softmax was already applied in the attention computation, but averaging
        # across stages may break the sum-to-1 property)
        accumulated = accumulated / (accumulated.sum(dim=1, keepdim=True) + 1e-8)

        # Return as dict
        result = {}
        for i, name in enumerate(factor_names):
            result[name] = accumulated[:, i]  # (B, H_out, W_out)

        return result

    finally:
        # Always disable attention storage (don't leak memory during training)
        for block in attn_blocks:
            block.store_attn = False
            block.last_attn_weights = None
            block.last_spatial_shape = None
