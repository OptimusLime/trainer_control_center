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
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from acc.training_metrics import TrainingMetricsAccumulator


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
        effective_grad[c] = grad[c] * (1 - gate_strength + gate_strength * mask[c])

    The raw softmax mask is used WITHOUT rescaling. A feature that wins fewer
    images gets smaller total gradient — that's correct. Niche features learn
    slower because they see less data, popular features learn faster because
    they see more. Rescaling (N * mask) would amplify noise from small batches.

    This class manages all hooks and provides a single remove() to clean up.
    """

    def __init__(self, metrics: Optional["TrainingMetricsAccumulator"] = None):
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._layer_configs: dict[str, GateConfig] = {}
        # Store last gate masks for inspection/debugging
        self.last_gate_masks: dict[str, torch.Tensor] = {}
        # Optional training-time metrics accumulator
        self.metrics: Optional["TrainingMetricsAccumulator"] = metrics
        # Store last grad norms for metrics (populated after backward)
        self.last_grad_norms: dict[str, torch.Tensor] = {}

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
                def grad_hook(grad: torch.Tensor, _name=name) -> torch.Tensor:
                    # Scale gradient by raw gate mask (no rescaling).
                    # Reshape mask to broadcast: [C] -> [1, C] for 2D, [1, C, 1, 1] for 4D
                    shape = [1] * grad.ndim
                    shape[1] = n_channels
                    mask_broadcast = gate_mask.view(*shape)

                    # Effective scaling: (1 - s) + s * mask
                    # At s=0: pass-through. At s=1: full gating by raw softmax mask.
                    # No N* rescaling — features that win fewer images get less
                    # gradient, which is correct (niche features learn slower).
                    scale = (1 - cfg.gate_strength) + cfg.gate_strength * mask_broadcast
                    scaled_grad = grad * scale.detach()

                    # Capture per-feature gradient norms for metrics
                    if grad.ndim == 2:
                        # [B, C] -> per-feature norm over batch
                        norms = scaled_grad.detach().abs().mean(dim=0)  # [C]
                    elif grad.ndim >= 3:
                        reduce_dims = (0,) + tuple(range(2, grad.ndim))
                        norms = scaled_grad.detach().abs().mean(dim=reduce_dims)  # [C]
                    else:
                        norms = None
                    if norms is not None:
                        self.last_grad_norms[_name] = norms.cpu()

                    return scaled_grad

                output.register_hook(grad_hook)

            return forward_hook

        hook = module.register_forward_hook(make_forward_hook(layer_name, config))
        self._hooks.append(hook)

    def record_step_metrics(self, step: int) -> Optional[dict]:
        """Call after loss.backward() to feed metrics accumulator.

        Passes the last gate masks and grad norms to the accumulator.
        Returns a summary dict if the accumulator says it's time, else None.

        Args:
            step: Current training step (1-indexed).

        Returns:
            Dict of training metrics if a summary is due, else None.
        """
        if self.metrics is None:
            return None

        self.metrics.on_step(
            step=step,
            gate_masks=self.last_gate_masks if self.last_gate_masks else None,
            grad_norms=self.last_grad_norms if self.last_grad_norms else None,
        )

        if self.metrics.should_summarize(step):
            return self.metrics.summarize()
        return None

    def remove(self) -> None:
        """Remove all hooks. Call when done with gated training."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_configs.clear()
        self.last_gate_masks.clear()
        self.last_grad_norms.clear()

    def describe(self) -> dict:
        """Return gating configuration as serializable dict."""
        return {
            layer_name: {
                "temperature": cfg.temperature,
                "gate_strength": cfg.gate_strength,
            }
            for layer_name, cfg in self._layer_configs.items()
        }


@dataclass
class NeighborhoodConfig:
    """Configuration for neighborhood-based gradient gating.

    Args:
        neighborhood_k: Number of nearest neighbors per feature (in weight space).
        recompute_every: Recompute neighborhoods every N forward passes.
        gate_strength: Interpolation between ungated (0.0) and fully gated (1.0).
        temperature: Sigmoid sharpness for margin-based competition.
            Higher = sharper winner/loser distinction.
    """
    neighborhood_k: int = 8
    recompute_every: int = 50
    gate_strength: float = 1.0
    temperature: float = 5.0


class NeighborhoodGating:
    """Gradient gating with per-image neighborhood competition.

    Each feature competes against its k-nearest neighbors in weight space,
    independently per image. Winners (features that beat their neighbors on
    a given image) get full gradient. Losers get attenuated gradient via soft
    sigmoid on the activation margin.

    For each configured layer:
    1. Periodically compute neighborhoods: cosine similarity of weight rows,
       take top-k neighbors per feature.
    2. Forward hook captures per-image activations [B, D].
    3. Backward hook: for each (image, feature) pair, compute margin =
       my_activation - max_neighbor_activation. Apply sigmoid(margin * temp)
       as gradient scale. Winners (positive margin) get ~1.0, losers get ~0.0.

    Dead features are NOT handled by the gating mechanism. Use
    ResidualPCAReplacer to periodically replace dead features with
    directions derived from reconstruction residuals.

    Same interface as CompetitiveGradientGating: attach(), remove(), describe(),
    record_step_metrics(), last_gate_masks, metrics.
    """

    def __init__(self, metrics: Optional["TrainingMetricsAccumulator"] = None):
        self._hooks: list = []
        self._layer_configs: dict[str, NeighborhoodConfig] = {}
        self._layer_modules: dict[str, nn.Module] = {}
        self.last_gate_masks: dict[str, torch.Tensor] = {}  # [D] batch-mean for metrics compat
        self.last_gate_masks_2d: dict[str, torch.Tensor] = {}  # [B, D] full per-image masks
        self.last_grad_norms: dict[str, torch.Tensor] = {}
        self.metrics: Optional["TrainingMetricsAccumulator"] = metrics

        # Neighborhood state per layer
        self._neighbors: dict[str, torch.Tensor] = {}  # [D, k] indices
        self._prev_neighbors: dict[str, torch.Tensor] = {}
        self._step_count: int = 0

    def _compute_neighborhoods(self, layer_name: str, module: nn.Module,
                                config: NeighborhoodConfig) -> None:
        """Recompute k-nearest neighbors in weight space for a layer."""
        W = module.weight.detach()  # [D, in_features] for Linear
        W_flat = W.view(W.size(0), -1)  # flatten for Conv2d compatibility
        W_norm = torch.nn.functional.normalize(W_flat, dim=1)
        sim = W_norm @ W_norm.T  # [D, D]
        sim.fill_diagonal_(-1.0)
        k = min(config.neighborhood_k, sim.size(0) - 1)
        _top_sims, indices = sim.topk(k, dim=1)  # [D, k]

        if layer_name in self._neighbors:
            self._prev_neighbors[layer_name] = self._neighbors[layer_name]
        self._neighbors[layer_name] = indices

    def _compute_neighborhood_stability(self, layer_name: str) -> Optional[float]:
        """Fraction of neighbors unchanged since last recomputation."""
        if layer_name not in self._prev_neighbors:
            return None
        prev = self._prev_neighbors[layer_name]  # [D, k]
        curr = self._neighbors[layer_name]        # [D, k]
        same = (prev.unsqueeze(-1) == curr.unsqueeze(-2)).any(dim=-1)  # [D, k]
        return same.float().mean().item()

    def attach(
        self,
        module: nn.Module,
        layer_name: str,
        config: NeighborhoodConfig,
    ) -> None:
        """Attach neighborhood gating hooks to a module."""
        self._layer_configs[layer_name] = config
        self._layer_modules[layer_name] = module
        self._compute_neighborhoods(layer_name, module, config)

        def make_forward_hook(name: str, cfg: NeighborhoodConfig):
            def forward_hook(mod: nn.Module, input: tuple, output: torch.Tensor):
                if not output.requires_grad:
                    return

                self._step_count += 1
                if self._step_count % cfg.recompute_every == 0:
                    self._compute_neighborhoods(name, mod, cfg)

                # Per-image, per-feature activation strength [B, D]
                if output.ndim == 2:
                    strength = output.abs()  # [B, D]
                elif output.ndim >= 3:
                    spatial_dims = tuple(range(2, output.ndim))
                    strength = output.abs().mean(dim=spatial_dims)  # [B, D]
                else:
                    return

                B, D = strength.shape
                neighbors = self._neighbors[name]  # [D, k]

                # Per-image neighborhood competition:
                # For each (image, feature), compare against neighbors
                neighbors_exp = neighbors.unsqueeze(0).expand(B, -1, -1)  # [B, D, k]
                neighbor_strengths = torch.gather(
                    strength.unsqueeze(1).expand(-1, D, -1),
                    dim=2,
                    index=neighbors_exp,
                )  # [B, D, k]

                # Margin: how much stronger am I than my best neighbor?
                max_neighbor = neighbor_strengths.max(dim=2).values  # [B, D]
                margin = strength - max_neighbor  # [B, D], positive = winning
                soft_wins = torch.sigmoid(margin * cfg.temperature)  # [B, D]

                # Store for metrics
                self.last_gate_masks[name] = soft_wins.detach().mean(dim=0).cpu()  # [D]
                self.last_gate_masks_2d[name] = soft_wins.detach().cpu()  # [B, D]

                n_channels = D
                def grad_hook(grad: torch.Tensor, _name=name) -> torch.Tensor:
                    if grad.ndim == 2:
                        mask_b = soft_wins  # [B, D]
                    elif grad.ndim >= 3:
                        extra_dims = grad.ndim - 2
                        mask_b = soft_wins.view(B, D, *([1] * extra_dims))
                    else:
                        return grad

                    scale = (1 - cfg.gate_strength) + cfg.gate_strength * mask_b
                    scaled_grad = grad * scale.detach()

                    # Capture grad norms
                    if grad.ndim == 2:
                        norms = scaled_grad.detach().abs().mean(dim=0)
                    elif grad.ndim >= 3:
                        rd = (0,) + tuple(range(2, grad.ndim))
                        norms = scaled_grad.detach().abs().mean(dim=rd)
                    else:
                        norms = None
                    if norms is not None:
                        self.last_grad_norms[_name] = norms.cpu()

                    return scaled_grad

                output.register_hook(grad_hook)

            return forward_hook

        hook = module.register_forward_hook(make_forward_hook(layer_name, config))
        self._hooks.append(hook)

    def record_step_metrics(self, step: int) -> Optional[dict]:
        """Same interface as CompetitiveGradientGating.record_step_metrics."""
        if self.metrics is None:
            return None

        # Pass [B, D] masks for per-image metrics
        masks = self.last_gate_masks_2d if self.last_gate_masks_2d else (
            self.last_gate_masks if self.last_gate_masks else None
        )
        self.metrics.on_step(
            step=step,
            gate_masks=masks,
            grad_norms=self.last_grad_norms if self.last_grad_norms else None,
        )

        if self.metrics.should_summarize(step):
            summary = self.metrics.summarize()
            for layer_name in self._layer_configs:
                stab = self._compute_neighborhood_stability(layer_name)
                if stab is not None:
                    summary["neighborhood_stability"] = round(stab, 4)
            return summary
        return None

    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_configs.clear()
        self._layer_modules.clear()
        self.last_gate_masks.clear()
        self.last_gate_masks_2d.clear()
        self.last_grad_norms.clear()
        self._neighbors.clear()
        self._prev_neighbors.clear()

    def describe(self) -> dict:
        """Return gating configuration as serializable dict."""
        return {
            layer_name: {
                "mechanism": "neighborhood",
                "neighborhood_k": cfg.neighborhood_k,
                "recompute_every": cfg.recompute_every,
                "gate_strength": cfg.gate_strength,
                "temperature": cfg.temperature,
            }
            for layer_name, cfg in self._layer_configs.items()
        }


class ResidualPCAReplacer:
    """Periodically replaces dead features with the top principal component
    of reconstruction errors in the neighborhood that needs the most help.

    Dead features are identified by win rate (from FeatureHealthTracker).
    For each dead feature:
    1. Find the neighborhood with the highest total reconstruction error.
    2. Collect error vectors (x - x_hat) for images that neighborhood handles.
    3. Compute top-1 PCA of those errors.
    4. Replace the dead feature's encoder + decoder weights with this direction.
    5. Reset the feature's Adam optimizer state.

    The top PC of reconstruction error in a neighborhood is literally the
    direction that would most reduce that neighborhood's error if a feature
    existed to capture it. It's the feature the system NEEDS, derived from
    the system's own failures.

    Usage:
        replacer = ResidualPCAReplacer(encoder_layer, decoder_layer, optimizer)
        # Every N epochs:
        replacements = replacer.check_and_replace(
            health_tracker, dataloader, model, device
        )
    """

    def __init__(
        self,
        encoder_layer: nn.Linear,
        decoder_layer: nn.Linear,
        optimizer: torch.optim.Optimizer,
        dead_threshold: float = 0.01,
        min_error_samples: int = 10,
    ):
        """
        Args:
            encoder_layer: The encoder Linear layer whose weights get replaced.
            decoder_layer: The decoder Linear layer (transpose relationship).
            optimizer: The Adam optimizer (for resetting momentum state).
            dead_threshold: Win rate below this = dead (fraction, default 1%).
            min_error_samples: Minimum images in a neighborhood to compute PCA.
        """
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.optimizer = optimizer
        self.dead_threshold = dead_threshold
        self.min_error_samples = min_error_samples

    @torch.no_grad()
    def check_and_replace(
        self,
        win_rates: torch.Tensor,
        dataloader,
        model: nn.Module,
        device: torch.device,
    ) -> list[dict]:
        """Identify dead features, compute residual PCA, replace weights.

        Args:
            win_rates: [D] tensor of per-feature win rates (from health tracker).
            dataloader: Training data loader for collecting error vectors.
            model: The full model (for forward passes to compute errors).
            device: Torch device.

        Returns:
            List of replacement event dicts for logging/tracking.
        """
        dead_mask = win_rates < self.dead_threshold
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        if len(dead_indices) == 0:
            return []

        # Collect reconstruction errors grouped by winning feature
        from collections import defaultdict
        neighborhood_errors: dict[int, list[torch.Tensor]] = defaultdict(list)

        model.eval()
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)

            out = model(x)
            recon = out.get("reconstruction", out.get("RECONSTRUCTION"))
            if recon is None:
                break
            recon_flat = recon.view(recon.size(0), -1)
            error = x_flat - recon_flat  # [B, in_dim]

            # Assign each image to its strongest-activating feature
            latent = out.get("latent", out.get("LATENT"))
            if latent is None:
                break
            winners = latent.abs().argmax(dim=1)  # [B]

            for b in range(x.size(0)):
                neighborhood_errors[winners[b].item()].append(error[b].cpu())

        model.train()

        if not neighborhood_errors:
            return []

        # Compute total error norm per neighborhood
        nbr_error_norms = {}
        for nbr_id, errors in neighborhood_errors.items():
            err_stack = torch.stack(errors)
            nbr_error_norms[nbr_id] = err_stack.norm().item()

        # Sort neighborhoods by error (worst first)
        sorted_nbrs = sorted(
            nbr_error_norms.keys(),
            key=lambda k: nbr_error_norms[k],
            reverse=True,
        )

        replacements = []
        used_nbrs = set()

        for dead_idx in dead_indices:
            dead_idx_int = dead_idx.item()

            # Find next unused neighborhood with enough samples
            donor_nbr = None
            for nbr_id in sorted_nbrs:
                if nbr_id in used_nbrs:
                    continue
                if len(neighborhood_errors.get(nbr_id, [])) >= self.min_error_samples:
                    donor_nbr = nbr_id
                    break

            if donor_nbr is None:
                continue

            used_nbrs.add(donor_nbr)

            # PCA on that neighborhood's errors
            err_matrix = torch.stack(neighborhood_errors[donor_nbr])
            err_matrix = err_matrix - err_matrix.mean(dim=0)

            try:
                U, S, V = torch.pca_lowrank(err_matrix, q=1)
                missing_direction = V[:, 0]
            except RuntimeError:
                # PCA can fail on degenerate matrices
                continue

            # Scale to match existing feature magnitudes
            missing_direction = missing_direction.to(device)
            existing_scale = self.encoder_layer.weight.norm(dim=1).mean()
            missing_direction = missing_direction * existing_scale

            # Replace encoder weights [D, in_dim]
            self.encoder_layer.weight[dead_idx_int] = missing_direction
            if self.encoder_layer.bias is not None:
                self.encoder_layer.bias[dead_idx_int] = 0.0

            # Replace decoder weights [in_dim, D]
            self.decoder_layer.weight[:, dead_idx_int] = missing_direction
            # decoder bias is shared across all features, don't touch

            # Reset Adam state for this feature
            self._reset_adam_state(dead_idx_int)

            replacements.append({
                "dead_idx": dead_idx_int,
                "donor_neighborhood": donor_nbr,
                "error_norm": nbr_error_norms[donor_nbr],
                "num_error_samples": len(neighborhood_errors[donor_nbr]),
            })

        return replacements

    def _reset_adam_state(self, feature_idx: int) -> None:
        """Zero out Adam momentum for a specific feature index."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                if "exp_avg" not in state:
                    continue

                ea = state["exp_avg"]
                easq = state["exp_avg_sq"]

                # Encoder weight: [D, in_dim] — zero row
                if ea.shape == self.encoder_layer.weight.shape:
                    ea[feature_idx] = 0
                    easq[feature_idx] = 0
                # Decoder weight: [in_dim, D] — zero column
                elif ea.dim() == 2 and ea.shape[1] == self.encoder_layer.weight.shape[0]:
                    ea[:, feature_idx] = 0
                    easq[:, feature_idx] = 0
                # Bias: [D] — zero element
                elif ea.dim() == 1 and ea.shape[0] == self.encoder_layer.weight.shape[0]:
                    ea[feature_idx] = 0
                    easq[feature_idx] = 0


def attach_neighborhood_gating(
    model: nn.Module,
    layer_configs: dict[str, dict],
    neighborhood_k: int = 8,
    recompute_every: int = 50,
    temperature: float = 1.0,
    metrics: Optional["TrainingMetricsAccumulator"] = None,
) -> NeighborhoodGating:
    """Convenience: attach neighborhood-based gating to named modules.

    Args:
        model: The model to attach gating to.
        layer_configs: Dict mapping module name to config overrides.
            Keys: "neighborhood_k", "recompute_every", "gate_strength", "temperature".
        neighborhood_k: Default number of neighbors per feature.
        recompute_every: Default steps between neighborhood recomputation.
        temperature: Default softmax temperature for local competition.
        metrics: Optional training-time metrics accumulator.

    Returns:
        NeighborhoodGating instance. Call .remove() when done.
    """
    gating = NeighborhoodGating(metrics=metrics)
    named_modules = dict(model.named_modules())

    for layer_name, overrides in layer_configs.items():
        if layer_name not in named_modules:
            available = [n for n, _ in model.named_modules() if n]
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available: {available}"
            )
        module = named_modules[layer_name]
        config = NeighborhoodConfig(
            neighborhood_k=overrides.get("neighborhood_k", neighborhood_k),
            recompute_every=overrides.get("recompute_every", recompute_every),
            gate_strength=overrides.get("gate_strength", 1.0),
            temperature=overrides.get("temperature", temperature),
        )
        gating.attach(module, layer_name, config)

    return gating


def attach_competitive_gating(
    model: nn.Module,
    layer_configs: dict[str, dict],
    temperature: float = 1.0,
    metrics: Optional["TrainingMetricsAccumulator"] = None,
) -> CompetitiveGradientGating:
    """Convenience: walk named modules and attach gating per config.

    Args:
        model: The model to attach gating to.
        layer_configs: Dict mapping module name (as in model.named_modules())
            to config overrides. Keys: "temperature", "gate_strength".
            Example: {"encoder.0": {"gate_strength": 1.0}}
        temperature: Default temperature for layers not specifying one.
        metrics: Optional training-time metrics accumulator. If provided,
            the gating mechanism will feed it gate masks and grad norms
            after each backward pass.

    Returns:
        CompetitiveGradientGating instance. Call .remove() when done.

    Raises:
        ValueError: If a layer_config key doesn't match any named module.
    """
    gating = CompetitiveGradientGating(metrics=metrics)

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
