"""Training-time metrics accumulators — pluggable per-step data collection.

These accumulate data during Trainer.train() and emit summary metrics
periodically. They flow through the same step_info → JobManager → API →
Dashboard pipeline as loss data.

Usage:
    from acc.training_metrics import GatingMetricsAccumulator

    accumulator = GatingMetricsAccumulator(num_features=64, epoch_length=469)
    gating = attach_competitive_gating(model, ..., metrics=accumulator)

    # During training, Trainer calls accumulator.on_step() after each backward.
    # At epoch boundaries, accumulator.summarize() returns a dict of metrics
    # which gets injected into step_info["training_metrics"].
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch


class TrainingMetricsAccumulator(ABC):
    """Protocol for accumulating training-time metrics.

    Implementations collect data every step and produce summary dicts
    at configurable intervals. The Trainer calls on_step() after each
    backward pass, and periodically calls summarize() to get a metrics
    dict that gets injected into the step_info stream.
    """

    @abstractmethod
    def on_step(
        self,
        step: int,
        gate_masks: Optional[dict[str, torch.Tensor]] = None,
        grad_norms: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """Called after each training step.

        Args:
            step: Current training step (1-indexed).
            gate_masks: Per-layer gate masks from gating mechanism.
                Keys are layer names, values are [C] tensors (softmax output).
            grad_norms: Per-layer per-feature gradient L2 norms.
                Keys are layer names, values are [C] tensors.
        """
        ...

    @abstractmethod
    def should_summarize(self, step: int) -> bool:
        """Whether a summary should be emitted at this step."""
        ...

    @abstractmethod
    def summarize(self) -> dict:
        """Produce a summary dict and reset accumulators for next interval.

        Returns:
            Dict with string keys and float/list values, suitable for JSON.
            Example: {"assignment_entropy": 0.87, "gradient_cv": 0.32}
        """
        ...


class GatingMetricsAccumulator(TrainingMetricsAccumulator):
    """Accumulates gating-specific training metrics.

    Tracks:
    - Assignment entropy: entropy of the gate-mass distribution across features.
      1.0 = perfectly uniform, 0.0 = total collapse.
    - Per-feature gradient magnitude: mean abs gradient per feature.
    - Gradient CV: coefficient of variation (std/mean) of per-feature gradient norms.
    - Win counts: how many times each feature "won" (had highest gate weight).
    - Coverage CV: coefficient of variation of per-image coverage (how evenly
      images are served by the feature population). Lower = more uniform.
    - Explorer graduation: features that transitioned from dead to alive.

    Handles both [D] masks (old-style batch-averaged) and [B, D] masks
    (per-image, from NeighborhoodExplorerGating).
    """

    def __init__(self, num_features: int, summary_every: int = 469):
        """
        Args:
            num_features: Number of features (channels) in the gated layer.
            summary_every: Emit a summary every N steps.
                Default 469 = one MNIST epoch at batch_size=128.
        """
        self.num_features = num_features
        self.summary_every = summary_every

        # Accumulators — reset after each summarize()
        self._win_counts = torch.zeros(num_features)
        self._gate_mass_sum = torch.zeros(num_features)  # sum of gate values per feature
        self._grad_norm_sum = torch.zeros(num_features)
        self._grad_norm_sq_sum = torch.zeros(num_features)
        self._n_steps = 0

        # Coverage tracking (for [B, D] masks)
        self._coverage_sum = 0.0   # sum of per-image coverage values
        self._coverage_sq_sum = 0.0  # sum of squares for variance
        self._coverage_count = 0   # total number of images

        # Explorer graduation tracking
        self._prev_alive: Optional[torch.Tensor] = None  # [D] bool from last summary
        self._graduations = 0  # count of dead->alive transitions this interval

    def on_step(
        self,
        step: int,
        gate_masks: Optional[dict[str, torch.Tensor]] = None,
        grad_norms: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        self._n_steps += 1

        if gate_masks is not None:
            for _layer_name, mask in gate_masks.items():
                m = mask.detach().cpu()

                if m.ndim == 2:
                    # [B, D] per-image mask from NeighborhoodExplorerGating
                    B, D = m.shape

                    # Per-image argmax wins
                    winners = m.argmax(dim=1)  # [B]
                    for w in winners:
                        self._win_counts[w.item()] += 1

                    # Gate mass: sum over batch for each feature
                    self._gate_mass_sum += m.sum(dim=0)  # [D]

                    # Coverage: sum of gate values per image
                    image_coverage = m.sum(dim=1)  # [B]
                    self._coverage_sum += image_coverage.sum().item()
                    self._coverage_sq_sum += (image_coverage ** 2).sum().item()
                    self._coverage_count += B

                elif m.ndim == 1:
                    # [D] batch-averaged mask (old-style)
                    winner = m.argmax().item()
                    self._win_counts[winner] += 1
                    self._gate_mass_sum += m

                break  # Only track first layer

        if grad_norms is not None:
            for _layer_name, norms in grad_norms.items():
                n = norms.detach().cpu()
                self._grad_norm_sum += n
                self._grad_norm_sq_sum += n ** 2
                break

    def should_summarize(self, step: int) -> bool:
        return self._n_steps > 0 and step % self.summary_every == 0

    def summarize(self) -> dict:
        result = {}

        if self._n_steps == 0:
            return result

        max_entropy = math.log(self.num_features) if self.num_features > 0 else 1.0

        # Assignment entropy — based on GATE MASS (total gradient share)
        gate_total = self._gate_mass_sum.sum()
        if gate_total > 0:
            p_gate = self._gate_mass_sum / gate_total
            p_gate_nz = p_gate[p_gate > 0]
            gate_entropy = -(p_gate_nz * p_gate_nz.log()).sum().item()
            result["assignment_entropy"] = round(
                max(0.0, gate_entropy / max_entropy), 4
            )

        # Win counts (argmax)
        total_wins = self._win_counts.sum()
        if total_wins > 0:
            result["win_counts"] = self._win_counts.tolist()
            result["dead_features"] = int((self._win_counts == 0).sum().item())

            p_wins = self._win_counts / total_wins
            p_wins_nz = p_wins[p_wins > 0]
            win_entropy = -(p_wins_nz * p_wins_nz.log()).sum().item()
            result["win_entropy"] = round(
                max(0.0, win_entropy / max_entropy), 4
            )

        # Explorer graduation: track dead->alive transitions
        # "Alive" = won more than 1% of total wins
        if total_wins > 0:
            alive_threshold = total_wins * 0.01
            currently_alive = self._win_counts >= alive_threshold  # [D] bool
            if self._prev_alive is not None:
                # Features that were dead last interval but alive now
                graduated = (~self._prev_alive & currently_alive).sum().item()
                self._graduations += int(graduated)
            self._prev_alive = currently_alive.clone()
            result["explorer_graduations"] = self._graduations

        # Gradient-starved features
        if self._grad_norm_sum.sum() > 0:
            mean_norms = self._grad_norm_sum / self._n_steps
            overall_mean = mean_norms.mean()
            if overall_mean > 0:
                starved = (mean_norms < 0.01 * overall_mean).sum().item()
                result["gradient_starved_features"] = int(starved)

        # Gradient CV
        if self._grad_norm_sum.sum() > 0:
            mean_norms = self._grad_norm_sum / self._n_steps
            mean_sq = self._grad_norm_sq_sum / self._n_steps
            variance = mean_sq - mean_norms ** 2
            std_norms = variance.clamp(min=0).sqrt()
            overall_mean = mean_norms.mean()
            overall_std = std_norms.mean()
            if overall_mean > 0:
                result["gradient_cv"] = round(
                    (overall_std / overall_mean).item(), 4
                )
            result["per_feature_grad_norms"] = mean_norms.tolist()

        # Coverage CV (only available with [B, D] masks)
        if self._coverage_count > 1:
            mean_cov = self._coverage_sum / self._coverage_count
            mean_cov_sq = self._coverage_sq_sum / self._coverage_count
            cov_var = max(0.0, mean_cov_sq - mean_cov ** 2)
            cov_std = cov_var ** 0.5
            if mean_cov > 0:
                result["coverage_cv"] = round(cov_std / mean_cov, 4)

        # Reset accumulators (but NOT _prev_alive or _graduations — those persist)
        self._win_counts.zero_()
        self._gate_mass_sum.zero_()
        self._grad_norm_sum.zero_()
        self._grad_norm_sq_sum.zero_()
        self._coverage_sum = 0.0
        self._coverage_sq_sum = 0.0
        self._coverage_count = 0
        self._n_steps = 0
        self._graduations = 0

        return result


class FeatureHealthTracker(GatingMetricsAccumulator):
    """Extended metrics accumulator with per-feature lifecycle tracking.

    Builds on GatingMetricsAccumulator, adding:
    - Per-feature win rate history [D, num_epochs] for heatmap visualization
    - Feature status classification (winner/partial/stale/dead)
    - Replacement event logging with post-replacement success tracking
    - Gini coefficient and top-k concentration of win distribution
    - Stale feature detection (win rate unchanged over N epochs)

    The epoch boundary is driven by the recipe calling end_epoch(), not by
    the step-based summary_every. The parent's summarize() is called by
    record_step_metrics() for the dashboard stream; end_epoch() is called
    by the recipe for lifecycle tracking.
    """

    def __init__(self, num_features: int, summary_every: int = 469):
        super().__init__(num_features=num_features, summary_every=summary_every)

        # Per-feature epoch-level tracking
        self.win_rate_history: list[list[float]] = []  # [num_epochs][D]
        self.epoch_summaries: list[dict] = []

        # Per-feature birth tracking
        self.birth_epoch = [-1] * num_features  # -1 = original feature
        self.replacement_count = [0] * num_features

        # Replacement event log
        self.replacement_log: list[dict] = []

        # Epoch-level snapshot: captured by summarize() before reset,
        # consumed by end_epoch(). This solves the ordering problem where
        # summarize() fires at step 469 (resetting accumulators) and then
        # end_epoch() is called after training returns.
        self._epoch_win_counts: torch.Tensor = torch.zeros(num_features)
        self._epoch_coverage_count: int = 0

    def summarize(self) -> dict:
        """Override parent to snapshot accumulators and add lifecycle metrics.

        The parent's summarize() resets _win_counts et al. But end_epoch()
        needs the epoch's accumulated data. So we snapshot here, then call
        super() which resets. We also inject gini and top5_share into the
        dashboard stream.
        """
        self._epoch_win_counts = self._win_counts.clone()
        self._epoch_coverage_count = self._coverage_count

        # Compute gini + concentration before parent resets
        wc = self._win_counts
        sorted_wins = wc.sort().values
        n = len(sorted_wins)
        total = sorted_wins.sum().item()
        if total > 0:
            cumulative = sorted_wins.cumsum(0)
            gini = 1.0 - 2.0 * cumulative.sum().item() / (n * total) + 1.0 / n
            top5_share = sorted_wins[-5:].sum().item() / total
        else:
            gini = 0.0
            top5_share = 0.0

        result = super().summarize()
        result["gini"] = round(gini, 4)
        result["top5_share"] = round(top5_share, 4)
        result["replacement_count"] = len(self.replacement_log)
        return result

    def end_epoch(self, epoch_num: int) -> dict:
        """Compute epoch-level summary from snapshot taken at last summarize().

        Called by the recipe between training chunks. Uses the snapshot
        taken by summarize() (which fires at step 469, just before reset).

        Args:
            epoch_num: Current epoch number (0-indexed).

        Returns:
            Epoch summary dict with health metrics.
        """
        D = self.num_features
        # Use snapshot from summarize(), fall back to live accumulators if
        # summarize() hasn't fired (e.g. epoch shorter than summary_every)
        win_counts = self._epoch_win_counts if self._epoch_win_counts.sum() > 0 else self._win_counts
        coverage_count = self._epoch_coverage_count if self._epoch_coverage_count > 0 else self._coverage_count
        total_images = max(coverage_count, 1) if coverage_count > 0 else max(int(win_counts.sum().item()), 1)

        # Win rates per feature
        win_rates = win_counts / max(total_images, 1)  # [D]

        # Store for heatmap
        self.win_rate_history.append(win_rates.tolist())

        # Dead features (win rate < 1%)
        dead_count = int((win_rates < 0.01).sum().item())

        # Stale features: win rate hasn't changed much over last 5 epochs
        stale_count = 0
        if len(self.win_rate_history) >= 5:
            for f in range(D):
                recent = [self.win_rate_history[-(i + 1)][f]
                          for i in range(min(5, len(self.win_rate_history)))]
                if len(recent) >= 3:
                    t = torch.tensor(recent)
                    if t.var().item() < 1e-6 and t.mean().item() > 0.01:
                        stale_count += 1

        # Gini coefficient of win distribution
        sorted_wins = win_counts.sort().values
        n = len(sorted_wins)
        cumulative = sorted_wins.cumsum(0)
        total = cumulative[-1].item()
        if total > 0:
            gini = 1.0 - 2.0 * cumulative.sum().item() / (n * total) + 1.0 / n
        else:
            gini = 0.0

        # Top-k concentration
        if total > 0:
            top5_share = sorted_wins[-5:].sum().item() / total
            top10_share = sorted_wins[-10:].sum().item() / total
        else:
            top5_share = 0.0
            top10_share = 0.0

        # Replacements this epoch
        replacements_this_epoch = len([
            r for r in self.replacement_log if r.get("epoch") == epoch_num
        ])

        summary = {
            "epoch": epoch_num,
            "dead_count": dead_count,
            "stale_count": stale_count,
            "gini": round(gini, 4),
            "top5_share": round(top5_share, 4),
            "top10_share": round(top10_share, 4),
            "replacements_this_epoch": replacements_this_epoch,
            "win_rates": win_rates.tolist(),
        }

        self.epoch_summaries.append(summary)
        return summary

    def get_win_rates(self) -> torch.Tensor:
        """Current per-feature win rates for ResidualPCAReplacer.

        Uses the epoch snapshot if available (after summarize() has reset
        the live accumulators), falls back to live accumulators.
        """
        wc = self._epoch_win_counts if self._epoch_win_counts.sum() > 0 else self._win_counts
        total = max(wc.sum().item(), 1.0)
        return wc / total

    def record_replacements(self, replacements: list[dict], epoch_num: int) -> None:
        """Record replacement events from ResidualPCAReplacer.

        Args:
            replacements: List of replacement dicts from check_and_replace().
            epoch_num: The epoch when replacement occurred.
        """
        for r in replacements:
            r["epoch"] = epoch_num
            self.replacement_log.append(r)

            idx = r["dead_idx"]
            self.birth_epoch[idx] = epoch_num
            self.replacement_count[idx] += 1
            # Clear this feature's recent win history (new life)
            for hist in self.win_rate_history:
                if idx < len(hist):
                    pass  # keep history, but birth_epoch marks the reset

    def get_feature_statuses(self) -> list[str]:
        """Classify each feature: winner/partial/stale/dead."""
        if not self.win_rate_history:
            return ["unknown"] * self.num_features

        latest_wr = self.win_rate_history[-1]
        statuses = []
        for f in range(self.num_features):
            wr = latest_wr[f]
            if wr < 0.01:
                statuses.append("dead")
            elif wr > 0.05:
                statuses.append("winner")
            else:
                # Check staleness
                if len(self.win_rate_history) >= 5:
                    recent = [self.win_rate_history[-(i + 1)][f]
                              for i in range(min(5, len(self.win_rate_history)))]
                    if torch.tensor(recent).var().item() < 1e-6:
                        statuses.append("stale")
                    else:
                        statuses.append("partial")
                else:
                    statuses.append("partial")
        return statuses

    def get_replacement_success_rate(self, lookback_epochs: int = 5) -> Optional[float]:
        """Fraction of replacements where the feature became alive (wr > 1%)
        within lookback_epochs after replacement."""
        if not self.replacement_log or not self.win_rate_history:
            return None

        current_epoch = len(self.win_rate_history) - 1
        successes = 0
        eligible = 0

        for r in self.replacement_log:
            rep_epoch = r["epoch"]
            check_epoch = rep_epoch + lookback_epochs
            if check_epoch > current_epoch:
                continue  # too recent to evaluate
            eligible += 1
            idx = r["dead_idx"]
            if check_epoch < len(self.win_rate_history):
                if self.win_rate_history[check_epoch][idx] >= 0.01:
                    successes += 1

        if eligible == 0:
            return None
        return successes / eligible
