"""WeightDiversityTask — mean pairwise cosine similarity of weight rows.

Evaluation-only task (extends EvalOnlyTask — never sampled during training).
At eval time, extracts the named layer's weight matrix and computes
mean pairwise cosine similarity between rows (output units).

Lower cosine similarity = more diverse weight vectors = more specialized units.

Works with any layer that has a .weight attribute (Linear, Conv2d, etc.).
For Conv2d, kernels are flattened to 2D: [out_channels, in_channels*kH*kW].

This is a library abstraction — works with any model, not just our architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # Conv2d or higher-dim: flatten all dims after the first
    return w.view(w.shape[0], -1)


def mean_pairwise_cosine_similarity(weight_matrix: torch.Tensor) -> float:
    """Compute mean pairwise cosine similarity between rows of a weight matrix.

    Args:
        weight_matrix: [N, D] where N is number of units, D is feature dim.

    Returns:
        Mean cosine similarity (float). Range [-1, 1], typically [0, 1] for
        ReLU networks. Lower = more diverse.
    """
    # Normalize rows to unit vectors
    normed = F.normalize(weight_matrix, p=2, dim=1)  # [N, D]
    # Pairwise cosine sim = normed @ normed^T
    sim_matrix = normed @ normed.T  # [N, N]
    n = sim_matrix.shape[0]
    if n < 2:
        return 0.0
    # Extract upper triangle (excluding diagonal) and take mean
    mask = torch.triu(torch.ones(n, n, device=sim_matrix.device, dtype=torch.bool), diagonal=1)
    return sim_matrix[mask].mean().item()


class WeightDiversityTask(EvalOnlyTask):
    """Measure weight diversity of a specific layer.

    Evaluation-only (extends EvalOnlyTask). At eval time, extracts the
    named layer's weight matrix and reports mean pairwise cosine similarity.

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
                f"WeightDiversityTask '{self.name}' targets layer '{self.layer_name}' "
                f"but it was not found in model. Available: {available}"
            )
        module = named[self.layer_name]
        if not hasattr(module, "weight"):
            raise TaskError(
                f"WeightDiversityTask '{self.name}' targets layer '{self.layer_name}' "
                f"({type(module).__name__}) which has no .weight attribute."
            )

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Compute mean pairwise cosine similarity of the target layer's weights."""
        named = dict(autoencoder.named_modules())
        module = named[self.layer_name]
        weight_matrix = _get_weight_matrix(module)
        cosine_sim = mean_pairwise_cosine_similarity(weight_matrix)
        return {EvalMetric.WEIGHT_COSINE_SIM: cosine_sim}

    def describe(self) -> dict:
        info = super().describe()
        info["layer_name"] = self.layer_name
        return info
