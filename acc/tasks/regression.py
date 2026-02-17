"""RegressionTask — Linear probe for regression.

Linear(latent_dim, output_dim), MSE loss, MAE eval.
check_compatible: dataset must have float targets.

Same class supports full latent or sliced via latent_slice config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.eval_metric import EvalMetric
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


class RegressionTask(Task):
    """Train a linear probe for regression on latent representations.

    Requires dataset with float targets (target_type == 'float').

    Args:
        name: Task name.
        dataset: Dataset with float targets.
        output_dim: Dimension of regression output. If None, inferred from dataset.
        weight: Loss weight.
        latent_slice: Optional (start, end) to read a factor slice.
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        output_dim: int | None = None,
        weight: float = 1.0,
        latent_slice: tuple[int, int] | None = None,
    ):
        super().__init__(name, dataset, weight=weight, latent_slice=latent_slice)
        self._output_dim = output_dim

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        if dataset.target_type != "float":
            raise TaskError(
                f"RegressionTask requires dataset with float targets "
                f"(target_type='float'), but dataset '{dataset.name}' "
                f"has target_type='{dataset.target_type}'"
            )

    def _build_head(self, latent_dim: int) -> nn.Module:
        if self._output_dim is not None:
            out_dim = self._output_dim
        elif self.dataset.targets is not None and self.dataset.targets.ndim > 1:
            out_dim = self.dataset.targets.shape[1]
        else:
            out_dim = 1
        return nn.Linear(latent_dim, out_dim)

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        """MSE loss on linear probe output."""
        _, targets = batch
        assert self.head is not None
        latent = self._get_latent(model_output)
        prediction = self.head(latent)

        # Handle shape: targets might be [B] or [B, D]
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        return F.mse_loss(prediction, targets.float())

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Compute MAE and MSE on eval split."""
        assert self.head is not None
        autoencoder.eval()
        self.head.eval()

        total_mae = 0.0
        total_mse = 0.0
        n_samples = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images, targets = batch
            images, targets = images.to(device), targets.to(device).float()
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

            model_output = autoencoder(images)
            latent = self._get_latent(model_output)
            prediction = self.head(latent)

            total_mae += (prediction - targets).abs().sum().item()
            total_mse += ((prediction - targets) ** 2).sum().item()
            n_samples += targets.shape[0] * targets.shape[1]

        autoencoder.train()
        self.head.train()

        mae = total_mae / max(n_samples, 1)
        mse = total_mse / max(n_samples, 1)
        return {EvalMetric.MAE: mae, EvalMetric.MSE: mse}
