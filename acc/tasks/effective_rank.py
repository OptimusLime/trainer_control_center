"""EffectiveRankTask — effective rank of a layer's weight matrix via SVD.

Evaluation-only task (extends EvalOnlyTask — never sampled during training).
At eval time, extracts the named layer's weight matrix, computes SVD,
and returns exp(entropy(normalized_singular_values)).

Higher effective rank = more dimensions actively used = less redundancy.
A rank-1 matrix has effective rank 1.0. A matrix with all equal singular
values has effective rank = min(rows, cols).

Works with any layer that has a .weight attribute (Linear, Conv2d, etc.).
For Conv2d, kernels are flattened to 2D: [out_channels, in_channels*kH*kW].

This is a library abstraction — works with any model, not just our architectures.
"""

import math

import torch
import torch.nn as nn

from acc.dataset import AccDataset
from acc.eval_metric import EvalMetric
from acc.tasks.base import EvalOnlyTask, TaskError


def _get_weight_matrix(module: nn.Module) -> torch.Tensor:
    """Extract weight as 2D matrix [out_features, in_features].

    Linear: already [out, in].
    Conv2d: reshape [out_channels, in_channels, kH, kW] -> [out_channels, in*kH*kW].
    """
    w = module.weight  # type: ignore[union-attr]
    if w.ndim == 2:
        return w
    return w.view(w.shape[0], -1)


def effective_rank(weight_matrix: torch.Tensor) -> float:
    """Compute effective rank via Shannon entropy of normalized singular values.

    effective_rank = exp(H(p)) where p_i = sigma_i / sum(sigma)
    and H(p) = -sum(p_i * log(p_i)).

    Args:
        weight_matrix: [M, N] weight matrix.

    Returns:
        Effective rank (float). Range [1, min(M, N)].
    """
    # SVD — only need singular values
    sigma = torch.linalg.svdvals(weight_matrix.float())

    # Remove near-zero singular values (numerical noise)
    sigma = sigma[sigma > 1e-10]
    if len(sigma) == 0:
        return 0.0

    # Normalize to probability distribution
    p = sigma / sigma.sum()

    # Shannon entropy
    entropy = -(p * p.log()).sum().item()

    return math.exp(entropy)


class EffectiveRankTask(EvalOnlyTask):
    """Measure effective rank of a specific layer's weight matrix.

    Evaluation-only (extends EvalOnlyTask). At eval time, extracts the
    named layer's weight matrix and reports its effective rank via SVD.

    Args:
        name: Task name.
        dataset: Dataset (required by Task interface, not used for weight-only eval).
        layer_name: Name of the module to inspect (as in model.named_modules()).
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        layer_name: str,
        weight: float = 0.0,
    ):
        super().__init__(name=name, dataset=dataset, weight=weight)
        self.layer_name = layer_name

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        named = dict(autoencoder.named_modules())
        if self.layer_name not in named:
            available = [n for n, _ in autoencoder.named_modules() if n]
            raise TaskError(
                f"EffectiveRankTask '{self.name}' targets layer '{self.layer_name}' "
                f"but it was not found in model. Available: {available}"
            )
        module = named[self.layer_name]
        if not hasattr(module, "weight"):
            raise TaskError(
                f"EffectiveRankTask '{self.name}' targets layer '{self.layer_name}' "
                f"({type(module).__name__}) which has no .weight attribute."
            )

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Compute effective rank of the target layer's weight matrix."""
        named = dict(autoencoder.named_modules())
        module = named[self.layer_name]
        weight_matrix = _get_weight_matrix(module)
        rank = effective_rank(weight_matrix)
        return {EvalMetric.EFFECTIVE_RANK: rank}

    def describe(self) -> dict:
        info = super().describe()
        info["layer_name"] = self.layer_name
        return info
