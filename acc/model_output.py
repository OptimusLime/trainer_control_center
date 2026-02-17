"""ModelOutput — enum keys for the dict returned by any model's forward().

This is the contract between models and tasks. Every model returns a dict
keyed by these values. Tasks read what they need. The Trainer is a dumb pipe.

Every model MUST populate LATENT. Everything else is optional.
Tasks check for what they need in check_compatible().
"""

from enum import Enum


class ModelOutput(str, Enum):
    LATENT = "latent"  # [B, D] full pooled latent vector. Always present.
    RECONSTRUCTION = (
        "reconstruction"  # [B, C, H, W] decoded image. Present if model has decoder.
    )
    SPATIAL = "spatial"  # [B, C, h, w] encoder spatial features before pooling.
    FACTOR_SLICES = "factor_slices"  # dict[str, Tensor] named factor slices. Present if model has factor groups.
