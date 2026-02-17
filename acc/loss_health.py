"""Loss health classification and summary statistics.

Pure library module. No dependencies on trainer, UI, or API.

LossHealth enum classifies loss values as healthy/warning/critical.
LossSummary dataclass holds per-task summary statistics.
classify_loss() maps (task_type, loss_value) -> LossHealth.
compute_loss_summary() computes per-task summaries from a list of step_infos.

Thresholds are per-task-type. The same thresholds are used everywhere:
trainer step_info, UI display, job history, checkpoint metadata.
"""

from dataclasses import dataclass
from enum import Enum


class LossHealth(str, Enum):
    """Health classification for a loss value."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def css_class(self) -> str:
        return f"loss-{self.value}"

    @property
    def color(self) -> str:
        """Hex color for this health level."""
        return _HEALTH_COLORS[self]


_HEALTH_COLORS = {
    LossHealth.HEALTHY: "#7ee787",
    LossHealth.WARNING: "#f0883e",
    LossHealth.CRITICAL: "#f85149",
}


# Thresholds per task type: (healthy_upper, warning_upper)
# Below healthy_upper -> HEALTHY
# Between healthy_upper and warning_upper -> WARNING
# Above warning_upper -> CRITICAL
#
# These assume:
# - ReconstructionTask: L1 loss on [0,1] images.
#   0.02 = excellent, 0.05 = decent, 0.15 = blurry, >0.15 = garbage
# - KLDivergenceTask: KL divergence per dim.
#   <5.0 = normal VAE, 5-15 = high but sometimes intentional, >15 = collapsed/exploded
# - ClassificationTask: cross-entropy loss.
#   <0.5 = good, 0.5-2.0 = learning, >2.0 = not learning
# - RegressionTask: MSE loss.
#   <0.1 = good, 0.1-0.5 = learning, >0.5 = not learning
_THRESHOLDS: dict[str, tuple[float, float]] = {
    "ReconstructionTask": (0.05, 0.15),
    "KLDivergenceTask": (5.0, 15.0),
    "ClassificationTask": (0.5, 2.0),
    "RegressionTask": (0.1, 0.5),
}

# Default thresholds for unknown task types
_DEFAULT_THRESHOLDS = (0.1, 1.0)


def classify_loss(task_type: str, loss_value: float) -> LossHealth:
    """Classify a loss value as healthy/warning/critical.

    Args:
        task_type: The task class name (e.g., "ReconstructionTask").
        loss_value: The scalar loss value.

    Returns:
        LossHealth classification.
    """
    healthy_upper, warning_upper = _THRESHOLDS.get(task_type, _DEFAULT_THRESHOLDS)

    if loss_value < healthy_upper:
        return LossHealth.HEALTHY
    elif loss_value < warning_upper:
        return LossHealth.WARNING
    else:
        return LossHealth.CRITICAL


@dataclass
class LossSummary:
    """Per-task summary statistics from a training run.

    Computed from a list of step_info dicts for a single task.
    """

    task_name: str
    task_type: str
    mean: float
    final: float
    min_val: float
    max_val: float
    trend: str  # "improving", "worsening", "flat"
    health: LossHealth
    n_steps: int

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "task_type": self.task_type,
            "mean": self.mean,
            "final": self.final,
            "min": self.min_val,
            "max": self.max_val,
            "trend": self.trend,
            "health": self.health.value,
            "n_steps": self.n_steps,
        }


def compute_loss_summary(losses: list[dict]) -> dict[str, LossSummary]:
    """Compute per-task summary statistics from a list of step_info dicts.

    Args:
        losses: List of step_info dicts, each with keys:
            "step", "task_name", "task_loss", "task_type" (optional), "health" (optional)

    Returns:
        Dict mapping task_name -> LossSummary.
    """
    if not losses:
        return {}

    # Group by task_name
    by_task: dict[str, list[dict]] = {}
    for entry in losses:
        name = entry["task_name"]
        by_task.setdefault(name, []).append(entry)

    summaries = {}
    for task_name, entries in by_task.items():
        values = [e["task_loss"] for e in entries]
        n = len(values)
        task_type = entries[-1].get("task_type", "unknown")

        mean_val = sum(values) / n
        final_val = values[-1]
        min_val = min(values)
        max_val = max(values)

        # Trend: compare first-half mean to second-half mean
        if n >= 4:
            mid = n // 2
            first_half_mean = sum(values[:mid]) / mid
            second_half_mean = sum(values[mid:]) / (n - mid)
            # Use 5% relative threshold for "flat"
            if first_half_mean > 0:
                change = (second_half_mean - first_half_mean) / first_half_mean
            else:
                change = 0.0

            if change < -0.05:
                trend = "improving"
            elif change > 0.05:
                trend = "worsening"
            else:
                trend = "flat"
        else:
            trend = "flat"

        health = classify_loss(task_type, final_val)

        summaries[task_name] = LossSummary(
            task_name=task_name,
            task_type=task_type,
            mean=mean_val,
            final=final_val,
            min_val=min_val,
            max_val=max_val,
            trend=trend,
            health=health,
            n_steps=n,
        )

    return summaries
