"""Competitive Gradient Gating — activation-proportional gradient scaling.

During the forward pass, capture each layer's output activations.
During the backward pass, scale each channel's gradient by how strongly
that channel activated, so strongly-activated weights receive
proportionally more gradient signal.

The mechanism is invisible to the Trainer: it attaches via standard
PyTorch hooks on the module and tensor. The Trainer calls forward(),
backward(), step() as normal.

Usage:
    from acc.gradient_gating import attach_competitive_gating

    model = LinearAutoencoder(784, 64)
    gating = attach_competitive_gating(model, layer_configs={
        "encoder.0": {"temperature": 1.0, "gate_strength": 1.0},
    })

    # ... train normally ...

    gating.remove()  # cleanup hooks when done
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class GateConfig:
    """Configuration for a single layer's gradient gate.

    Args:
        temperature: Softmax temperature for the gate mask. Lower = sharper
            competition (winner-take-all). Higher = more uniform.
        gate_strength: Interpolation between ungated (0.0) and fully gated (1.0).
            At 0.0, gradients pass through unchanged. At 1.0, gradients are
            fully scaled by activation strength.
    """
    temperature: float = 1.0
    gate_strength: float = 1.0


class CompetitiveGradientGating:
    """Attaches forward+backward hooks to a set of modules for gradient gating.

    For each configured layer:
    1. Forward hook captures output activations.
    2. Tensor hook on the output scales gradients by a softmax-normalized
       activation mask during backward.

    The gate mask for a channel c is:
        mask[c] = softmax(mean_activation[c] / temperature)
    scaled by gate_strength:
        effective_grad[c] = grad[c] * (1 - gate_strength + gate_strength * N * mask[c])

    where N is the number of channels. The N * mask[c] term ensures that
    the expected gradient magnitude is preserved (softmax sums to 1, so
    N * mask averages to 1).

    This class manages all hooks and provides a single remove() to clean up.
    """

    def __init__(self):
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._layer_configs: dict[str, GateConfig] = {}
        # Store last gate masks for inspection/debugging
        self.last_gate_masks: dict[str, torch.Tensor] = {}

    def attach(
        self,
        module: nn.Module,
        layer_name: str,
        config: GateConfig,
    ) -> None:
        """Attach gating hooks to a specific sub-module.

        Args:
            module: The nn.Module to hook (e.g., a Linear or Conv2d layer).
            layer_name: Human-readable name for debugging/inspection.
            config: Gate configuration (temperature, strength).
        """
        self._layer_configs[layer_name] = config

        def make_forward_hook(name: str, cfg: GateConfig):
            """Create a forward hook that registers a backward hook on the output tensor."""

            def forward_hook(mod: nn.Module, input: tuple, output: torch.Tensor):
                if not output.requires_grad:
                    return

                # Compute per-channel mean activation magnitude.
                # Works for both 2D (Linear: [B, C]) and 4D (Conv: [B, C, H, W]).
                if output.ndim == 2:
                    # [B, C] -> mean over batch -> [C]
                    channel_means = output.abs().mean(dim=0)
                elif output.ndim >= 3:
                    # [B, C, ...] -> mean over batch and spatial dims -> [C]
                    reduce_dims = tuple(range(output.ndim))
                    reduce_dims = (0,) + tuple(range(2, output.ndim))
                    channel_means = output.abs().mean(dim=reduce_dims)
                else:
                    return  # 1D output, nothing to gate

                # Softmax gate mask: sharp competition at low temperature
                gate_mask = torch.softmax(channel_means / cfg.temperature, dim=0)
                n_channels = gate_mask.shape[0]

                # Store for inspection
                self.last_gate_masks[name] = gate_mask.detach()

                # Register backward hook on the output tensor
                def grad_hook(grad: torch.Tensor) -> torch.Tensor:
                    # Scale gradient by gate mask.
                    # Reshape mask to broadcast: [C] -> [1, C] for 2D, [1, C, 1, 1] for 4D
                    shape = [1] * grad.ndim
                    shape[1] = n_channels
                    mask_broadcast = gate_mask.view(*shape)

                    # Effective scaling: (1 - s) + s * N * mask
                    # At s=0: pass-through. At s=1: full gating.
                    # N * mask ensures expected magnitude = 1 (softmax sums to 1).
                    scale = (1 - cfg.gate_strength) + cfg.gate_strength * n_channels * mask_broadcast
                    return grad * scale.detach()

                output.register_hook(grad_hook)

            return forward_hook

        hook = module.register_forward_hook(make_forward_hook(layer_name, config))
        self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all hooks. Call when done with gated training."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_configs.clear()
        self.last_gate_masks.clear()

    def describe(self) -> dict:
        """Return gating configuration as serializable dict."""
        return {
            layer_name: {
                "temperature": cfg.temperature,
                "gate_strength": cfg.gate_strength,
            }
            for layer_name, cfg in self._layer_configs.items()
        }


def attach_competitive_gating(
    model: nn.Module,
    layer_configs: dict[str, dict],
    temperature: float = 1.0,
) -> CompetitiveGradientGating:
    """Convenience: walk named modules and attach gating per config.

    Args:
        model: The model to attach gating to.
        layer_configs: Dict mapping module name (as in model.named_modules())
            to config overrides. Keys: "temperature", "gate_strength".
            Example: {"encoder.0": {"gate_strength": 1.0}}
        temperature: Default temperature for layers not specifying one.

    Returns:
        CompetitiveGradientGating instance. Call .remove() when done.

    Raises:
        ValueError: If a layer_config key doesn't match any named module.
    """
    gating = CompetitiveGradientGating()

    # Build lookup of named modules
    named_modules = dict(model.named_modules())

    for layer_name, overrides in layer_configs.items():
        if layer_name not in named_modules:
            available = [n for n, _ in model.named_modules() if n]
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available: {available}"
            )
        module = named_modules[layer_name]
        config = GateConfig(
            temperature=overrides.get("temperature", temperature),
            gate_strength=overrides.get("gate_strength", 1.0),
        )
        gating.attach(module, layer_name, config)

    return gating
