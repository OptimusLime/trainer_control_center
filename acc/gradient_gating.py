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
        floor: Minimum gradient scale for losing features (0.0 = full gating,
            0.1 = losers keep 10% gradient). Prevents total gradient starvation
            so losers can still slowly differentiate.
    """
    neighborhood_k: int = 8
    recompute_every: int = 50
    gate_strength: float = 1.0
    temperature: float = 5.0
    floor: float = 0.1


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

                # Dissimilarity floor: losers keep at least floor fraction of
                # gradient. Prevents total starvation so they can differentiate.
                if cfg.floor > 0:
                    soft_wins = cfg.floor + (1.0 - cfg.floor) * soft_wins

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


@dataclass
class BCLConfig:
    """Configuration for Bidirectional Competitive Learning.

    Args:
        neighborhood_k: Number of nearest neighbors per feature (weight space).
        temperature: Sigmoid sharpness for margin competition. Higher = sharper.
        som_lr: SOM learning rate for loser weight updates.
        novelty_clamp: Maximum novelty multiplier (caps influence of rare images).
        recompute_every: Steps between neighborhood recomputation.
    """
    neighborhood_k: int = 8
    temperature: float = 5.0
    som_lr: float = 0.005
    novelty_clamp: float = 3.0
    recompute_every: int = 50


class BCL:
    """Bidirectional Competitive Learning.

    Attaches to an nn.Linear layer. A single forward hook computes local
    competition (sigmoid margin) and novelty modulation, then registers a
    backward hook that does BOTH gradient masking (for winners) AND SOM
    weight update (for losers) in one location.

    Winners: gradient masked by rank_score * novelty.
    Losers: weight pulled toward raw inputs, weighted by
            (1 - rank_score) * in_neighborhood * novelty.

    The backward hook fires AFTER gradients are computed from the original
    forward-pass weights. Modifying module.weight inside it is safe — it
    only affects the NEXT forward pass. One hook, one location, no bifurcation.

    Usage:
        bcl = BCL(model.encoder[0], BCLConfig(som_lr=0.005))
        # train normally — no apply_som() needed
        bcl.remove()
    """

    def __init__(self, module: nn.Module, config: BCLConfig):
        self.module = module
        self.config = config
        self._step = 0
        self._neighbors: Optional[torch.Tensor] = None  # [D, k]
        self._last_metrics: Optional[dict[str, torch.Tensor]] = None
        self._handle = module.register_forward_hook(self._forward_hook)

    def _compute_neighborhoods(self) -> None:
        """Recompute k-nearest neighbors in weight space."""
        W = self.module.weight.detach()
        W_flat = W.view(W.size(0), -1)
        W_norm = torch.nn.functional.normalize(W_flat, dim=1)
        sim = W_norm @ W_norm.T
        sim.fill_diagonal_(-float('inf'))
        k = min(self.config.neighborhood_k, sim.size(0) - 1)
        self._neighbors = sim.topk(k, dim=1).indices

    def _forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor):
        if not output.requires_grad:
            return

        layer_input = input[0].detach()  # [B, in_features]
        act = output.detach()            # [B, D]

        if act.dim() == 2:
            strength = act.abs()  # [B, D]
        elif act.dim() == 4:
            strength = act.abs().mean(dim=(2, 3))  # [B, D]
            layer_input = layer_input.view(layer_input.size(0), -1)
        else:
            return

        B, D = strength.shape
        cfg = self.config

        # --- Neighborhoods ---
        if self._neighbors is None or self._step % cfg.recompute_every == 0:
            self._compute_neighborhoods()
        self._step += 1
        neighbors = self._neighbors  # [D, k]

        # --- Local competition ---
        neighbors_exp = neighbors.unsqueeze(0).expand(B, -1, -1)  # [B, D, k]
        neighbor_strengths = torch.gather(
            strength.unsqueeze(1).expand(-1, D, -1),
            dim=2, index=neighbors_exp,
        )  # [B, D, k]
        max_neighbor = neighbor_strengths.max(dim=2).values  # [B, D]
        margin = strength - max_neighbor  # [B, D]
        rank_score = torch.sigmoid(margin * cfg.temperature)  # [B, D]

        # --- Image novelty ---
        image_crowding = rank_score.sum(dim=1)  # [B]
        novelty = 1.0 / (image_crowding + 1e-8)  # [B]
        novelty = novelty / (novelty.mean() + 1e-8)  # normalize mean=1
        novelty = novelty.clamp(max=cfg.novelty_clamp)  # [B]

        # --- Step 5: Feature status + force blending weights ---
        win_rate = rank_score.mean(dim=0)  # [D]
        gradient_weight = win_rate                              # [D] — peaks for winners
        rotation_weight = win_rate * (1 - win_rate) * 4         # [D] — peaks at wr=0.5
        attraction_weight = 1.0 - win_rate                      # [D] — peaks for dead

        # --- Step 6: Force 1 — Gradient (for winners) ---
        grad_mask = (
            rank_score * novelty.unsqueeze(1) * gradient_weight.unsqueeze(0)
        )  # [B, D]

        # --- Shared: in_neighborhood mask ---
        winners_per_image = rank_score.argmax(dim=1)  # [B]
        winner_nbrs = neighbors[winners_per_image]  # [B, k]
        in_nbr = torch.zeros(B, D, device=act.device)
        in_nbr.scatter_(1, winner_nbrs, 1.0)  # [B, D]

        # --- Step 7: Force 2 — Global attraction (for dead features) ---
        # NO in_neighborhood gate — dead features search globally
        attract_pull = (
            (1.0 - rank_score) * novelty.unsqueeze(1) * attraction_weight.unsqueeze(0)
        )  # [B, D]
        attract_norm = attract_pull.sum(dim=0, keepdim=True) + 1e-8  # [1, D]
        attract_pull = attract_pull / attract_norm  # [B, D]
        attract_target = attract_pull.T @ layer_input  # [D, in_features]

        # --- Step 8: Force 3 — Local rotation (for contenders) ---
        # Bully direction: weighted blend of neighbors that beat you
        strength_exp = strength.unsqueeze(2).expand_as(neighbor_strengths)  # [B, D, k]
        beat_margin = (neighbor_strengths - strength_exp).clamp(min=0)     # [B, D, k]
        beat_mean = beat_margin.mean(dim=0)  # [D, k]
        beat_weights = beat_mean / (beat_mean.sum(dim=1, keepdim=True) + 1e-8)  # [D, k]

        W = module.weight.detach()  # [D, in_features]
        neighbor_weights = W[neighbors]  # [D, k, in_features]
        bully_raw = torch.einsum('dk,dki->di', beat_weights, neighbor_weights)  # [D, in_features]
        bully_direction = torch.nn.functional.normalize(bully_raw, dim=1)  # [D, in_features]

        # Project bully direction out of inputs
        overlap = layer_input @ bully_direction.T  # [B, D]
        rotated_input = (
            layer_input.unsqueeze(1)
            - overlap.unsqueeze(2) * bully_direction.unsqueeze(0)
        )  # [B, D, in_features]

        # Pull toward rotated inputs — LOCAL, needs in_neighborhood
        rotate_pull = (
            (1.0 - rank_score) * novelty.unsqueeze(1)
            * in_nbr * rotation_weight.unsqueeze(0)
        )  # [B, D]
        rotate_norm = rotate_pull.sum(dim=0, keepdim=True) + 1e-8  # [1, D]
        rotate_pull = rotate_pull / rotate_norm  # [B, D]
        rotate_target = torch.einsum('bd,bdi->di', rotate_pull, rotated_input)  # [D, in_features]

        # --- Step 9: Combine SOM targets ---
        aw = attraction_weight.unsqueeze(1)  # [D, 1]
        rw = rotation_weight.unsqueeze(1)    # [D, 1]
        som_targets = (aw * attract_target + rw * rotate_target) / (aw + rw + 1e-8)
        # [D, in_features]

        # Total SOM weight for metrics (combine both forces)
        som_weight = (
            (1.0 - rank_score) * novelty.unsqueeze(1)
            * (attraction_weight.unsqueeze(0) + in_nbr * rotation_weight.unsqueeze(0))
        )  # [B, D]

        # --- Store metrics for this step ---
        bully_magnitude = bully_raw.norm(dim=1)  # [D]
        self._last_metrics = {
            'rank_score': rank_score.detach(),       # [B, D]
            'novelty': novelty.detach(),             # [B]
            'grad_mask': grad_mask.detach(),         # [B, D]
            'som_weight': som_weight.detach(),       # [B, D]
            'strength': strength.detach(),           # [B, D]
            'in_nbr': in_nbr.detach(),               # [B, D]
            'bully_magnitude': bully_magnitude.detach(),  # [D]
        }

        # --- Register backward hook: gradient mask + SOM update ---
        # Capture everything by value for the closure
        _mask = grad_mask.detach()
        _som_targets = som_targets.detach()
        _som_lr = cfg.som_lr
        _module = module

        def backward_hook(grad: torch.Tensor) -> torch.Tensor:
            # SOM update for losers (safe: grad already computed from unmodified weights)
            with torch.no_grad():
                _module.weight += _som_lr * (_som_targets - _module.weight)
            # Gradient mask for winners
            return grad * _mask

        output.register_hook(backward_hook)

    def get_step_metrics(self) -> Optional[dict]:
        """Per-step metrics from last forward pass. Returns None if unavailable."""
        if self._last_metrics is None:
            return None

        m = self._last_metrics
        rs = m['rank_score']     # [B, D]
        gm = m['grad_mask']      # [B, D]
        sw = m['som_weight']     # [B, D]
        st = m['strength']       # [B, D]
        nov = m['novelty']       # [B]
        bm = m['bully_magnitude']  # [D]

        win_rate = rs.mean(dim=0)              # [D]
        grad_magnitude = gm.sum(dim=0)         # [D]
        som_magnitude = sw.sum(dim=0)          # [D]
        mean_activation = st.mean(dim=0)       # [D]

        return {
            'win_rate': win_rate,               # [D]
            'grad_magnitude': grad_magnitude,   # [D]
            'som_magnitude': som_magnitude,     # [D]
            'mean_activation': mean_activation, # [D]
            'bully_magnitude': bm,             # [D]
            'mean_novelty': nov.mean().item(),
            'novelty_std': nov.std().item(),
            'mean_crowding': rs.sum(dim=1).mean().item(),
            'crowding_std': rs.sum(dim=1).std().item(),
        }

    def remove(self) -> None:
        """Remove all hooks. Call when done with BCL training."""
        self._handle.remove()
        self._neighbors = None
        self._last_metrics = None

    def describe(self) -> dict:
        """Return BCL configuration as serializable dict."""
        return {
            "mechanism": "bcl",
            "neighborhood_k": self.config.neighborhood_k,
            "temperature": self.config.temperature,
            "som_lr": self.config.som_lr,
            "novelty_clamp": self.config.novelty_clamp,
            "recompute_every": self.config.recompute_every,
        }


def attach_bcl(
    model: nn.Module,
    layer_name: str,
    config: Optional[BCLConfig] = None,
) -> BCL:
    """Convenience: attach BCL to a named module.

    Args:
        model: The model containing the target layer.
        layer_name: Dot-separated module name (e.g. "encoder.0").
        config: BCL configuration. Uses defaults if None.

    Returns:
        BCL instance. Call .remove() when done.
    """
    if config is None:
        config = BCLConfig()

    named_modules = dict(model.named_modules())
    if layer_name not in named_modules:
        available = [n for n, _ in model.named_modules() if n]
        raise ValueError(
            f"Layer '{layer_name}' not found in model. Available: {available}"
        )
    module = named_modules[layer_name]
    return BCL(module, config)


def attach_neighborhood_gating(
    model: nn.Module,
    layer_configs: dict[str, dict],
    neighborhood_k: int = 8,
    recompute_every: int = 50,
    temperature: float = 1.0,
    floor: float = 0.1,
    metrics: Optional["TrainingMetricsAccumulator"] = None,
) -> NeighborhoodGating:
    """Convenience: attach neighborhood-based gating to named modules.

    Args:
        model: The model to attach gating to.
        layer_configs: Dict mapping module name to config overrides.
            Keys: "neighborhood_k", "recompute_every", "gate_strength",
            "temperature", "floor".
        neighborhood_k: Default number of neighbors per feature.
        recompute_every: Default steps between neighborhood recomputation.
        temperature: Default softmax temperature for local competition.
        floor: Default minimum gradient scale for losers (0.1 = 10%).
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
            floor=overrides.get("floor", floor),
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
