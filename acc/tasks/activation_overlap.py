"""ActivationOverlapDiagnostic — measure UFR vs FER properties.

Evaluation-only task that measures:
1. Lifetime: fraction of dataset each latent position activates for
2. Pairwise cosine similarity of activation patterns across dataset
3. Co-activation: fraction of images where position pairs fire simultaneously

UFR target: low cosine similarity AND low co-activation.
FER signature: low cosine similarity BUT high co-activation.

Reads ModelOutput.SPATIAL, pools to 3x3.
No gradient — pure measurement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.dataset import AccDataset
from acc.model_output import ModelOutput
from acc.tasks.base import EvalOnlyTask, TaskError


class ActivationOverlapDiagnostic(EvalOnlyTask):
    """Measure UFR vs FER properties of the latent space.

    Reports:
    - lifetime_mean: average fraction of dataset each position activates for
    - cosine_sim_mean: average pairwise cosine similarity across positions
    - coactivation_mean: average pairwise co-activation rate

    Interpretation:
    - UFR: low lifetime, low cosine sim, low co-activation
    - FER: low cosine sim but HIGH co-activation (orthogonal but overlapping)
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        threshold: float = 0.1,
        n_eval_batches: int = 20,
        **kwargs,
    ):
        super().__init__(name, dataset, **kwargs)
        self.threshold = threshold
        self.n_eval_batches = n_eval_batches

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        pass

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        autoencoder.eval()

        # Collect pooled activations across dataset
        all_activations: list[torch.Tensor] = []

        for i, batch in enumerate(self.dataset.eval_loader(batch_size=256)):
            if i >= self.n_eval_batches:
                break
            images = batch[0].to(device)
            output = autoencoder(images)
            spatial = output.get(ModelOutput.SPATIAL)
            if spatial is None:
                break
            pooled = F.adaptive_avg_pool2d(spatial, 3)  # [B, C, 3, 3]
            # Flatten to [B, C*9] — each position is a "unit"
            flat = pooled.reshape(pooled.shape[0], -1)  # [B, N]
            all_activations.append(flat.cpu())

        autoencoder.train()

        if not all_activations:
            return {
                "lifetime_mean": 0.0,
                "cosine_sim_mean": 0.0,
                "coactivation_mean": 0.0,
            }

        A = torch.cat(all_activations, dim=0)  # [total_images, N]
        N_images, N_units = A.shape

        # 1. Lifetime: fraction of images each unit activates for
        active = (A.abs() > self.threshold).float()  # [N_images, N_units]
        lifetime = active.mean(dim=0)  # [N_units]

        # 2. Pairwise cosine similarity of activation patterns
        # Each unit has an activation pattern across images: A[:, i]
        A_norm = F.normalize(A.T, dim=-1)  # [N_units, N_images]
        if N_units > 1:
            cos_sim = A_norm @ A_norm.T  # [N_units, N_units]
            # Mean of off-diagonal elements
            mask = ~torch.eye(N_units, dtype=torch.bool)
            cos_sim_mean = cos_sim[mask].mean().item()
        else:
            cos_sim_mean = 0.0

        # 3. Co-activation: fraction of images where pairs fire simultaneously
        if N_units > 1:
            # For each pair (i, j), co-activation = mean(active[:, i] * active[:, j])
            coact_matrix = (active.T @ active) / N_images  # [N_units, N_units]
            mask = ~torch.eye(N_units, dtype=torch.bool)
            coact_mean = coact_matrix[mask].mean().item()
        else:
            coact_mean = 0.0

        return {
            "lifetime_mean": lifetime.mean().item(),
            "cosine_sim_mean": cos_sim_mean,
            "coactivation_mean": coact_mean,
        }

    def describe(self) -> dict:
        info = super().describe()
        info["threshold"] = self.threshold
        info["n_eval_batches"] = self.n_eval_batches
        return info
