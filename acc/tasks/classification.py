"""ClassificationTask — Linear probe for classification.

Linear(latent_dim, num_classes), cross-entropy loss, accuracy eval.
check_compatible: dataset must have integer targets.

Supports latent_slice for factor-targeted classification — same class,
just pass latent_slice=(start, end) at construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.eval_metric import EvalMetric
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


class ClassificationTask(Task):
    """Train a linear probe to classify latent representations.

    Requires dataset with integer targets (target_type == 'classes').
    """

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        if dataset.target_type != "classes":
            raise TaskError(
                f"ClassificationTask requires dataset with integer targets "
                f"(target_type='classes'), but dataset '{dataset.name}' "
                f"has target_type='{dataset.target_type}'"
            )

    def _build_head(self, latent_dim: int) -> nn.Module:
        num_classes = self.dataset.num_classes
        assert num_classes is not None
        return nn.Linear(latent_dim, num_classes)

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        """Cross-entropy loss on linear probe output."""
        _, targets = batch
        assert self.head is not None
        latent = self._get_latent(model_output)
        logits = self.head(latent)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Compute accuracy on eval split."""
        assert self.head is not None
        autoencoder.eval()
        self.head.eval()

        correct = 0
        total = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            model_output = autoencoder(images)
            latent = self._get_latent(model_output)
            logits = self.head(latent)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        autoencoder.train()
        self.head.train()

        accuracy = correct / total if total > 0 else 0.0
        return {EvalMetric.ACCURACY: accuracy}
