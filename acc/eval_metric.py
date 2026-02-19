"""EvalMetric — well-known evaluation metric names.

Same pattern as ModelOutput: an enum that serves as the canonical source
of truth for metric keys returned by Task.evaluate(). No magic strings.

Each metric knows its own direction (higher_is_better) and display properties.
This is the contract between tasks (which produce metrics) and the UI
(which displays and compares them).
"""

from enum import Enum


class EvalMetric(str, Enum):
    """Well-known evaluation metric names.

    Inherits from str so it can be used as a dict key and serializes
    cleanly to JSON. The .value is the canonical string representation.
    """

    ACCURACY = "accuracy"
    L1 = "l1"
    PSNR = "psnr"
    MAE = "mae"
    MSE = "mse"
    KL = "kl"
    UFR = "ufr"
    DISENTANGLEMENT = "disentanglement"
    COMPLETENESS = "completeness"
    WEIGHT_COSINE_SIM = "weight_cosine_sim"
    ACTIVATION_SPARSITY = "activation_sparsity"
    SPARSITY_VARIANCE = "sparsity_variance"
    EFFECTIVE_RANK = "effective_rank"

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        return self in _HIGHER_IS_BETTER

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return _DISPLAY_NAMES.get(self, self.value)


# Metrics where higher = better. Everything else is lower = better.
_HIGHER_IS_BETTER = {
    EvalMetric.ACCURACY,
    EvalMetric.PSNR,
    EvalMetric.UFR,
    EvalMetric.DISENTANGLEMENT,
    EvalMetric.COMPLETENESS,
    EvalMetric.SPARSITY_VARIANCE,
    EvalMetric.EFFECTIVE_RANK,
}

_DISPLAY_NAMES = {
    EvalMetric.ACCURACY: "Accuracy",
    EvalMetric.L1: "L1",
    EvalMetric.PSNR: "PSNR",
    EvalMetric.MAE: "MAE",
    EvalMetric.MSE: "MSE",
    EvalMetric.KL: "KL",
    EvalMetric.UFR: "UFR",
    EvalMetric.DISENTANGLEMENT: "Disentanglement",
    EvalMetric.COMPLETENESS: "Completeness",
    EvalMetric.WEIGHT_COSINE_SIM: "Weight Cosine Sim",
    EvalMetric.ACTIVATION_SPARSITY: "Activation Sparsity",
    EvalMetric.SPARSITY_VARIANCE: "Sparsity Variance",
    EvalMetric.EFFECTIVE_RANK: "Effective Rank",
}
