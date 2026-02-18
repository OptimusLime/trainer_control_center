"""KLDivergenceTask — KL regularization toward N(0,1).

Reads ModelOutput.MU and ModelOutput.LOGVAR from the model output dict.
Weight controls beta: weight=1.0 is standard VAE, weight>1.0 is beta-VAE.

Supports per-factor KL via latent_slice. Create separate instances with
different latent_slice and weight values for per-factor control:

    KLDivergenceTask("kl_digit", ds, weight=4.0, latent_slice=(0, 4))
    KLDivergenceTask("kl_free", ds, weight=0.1, latent_slice=(10, 16))

KL annealing: set warmup_steps to linearly ramp KL contribution from
0 → full weight over N training steps. This lets the model learn good
reconstruction before KL regularization kicks in.

    KLDivergenceTask("kl", ds, weight=1.0, warmup_steps=5000)

KL is NOT a dataset-dependent task — it doesn't use the batch data at all.
It only reads mu/logvar from the model output. But the Task base class
requires a dataset (for the round-robin), so pass any loaded dataset.
"""

from typing import Optional

import torch
import torch.nn as nn

from acc.dataset import AccDataset
from acc.eval_metric import EvalMetric
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


class KLDivergenceTask(Task):
    """KL divergence regularization toward N(0,1).

    Computes: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))

    KL annealing: when warmup_steps > 0, the KL loss is multiplied by
    min(1.0, step / warmup_steps) where step counts how many times
    compute_loss has been called. This gives the reconstruction task
    time to learn before KL starts fighting it.

    Args:
        name: Task name.
        dataset: Any loaded dataset (needed for round-robin scheduling).
        weight: Loss weight. Acts as beta in beta-VAE.
        warmup_steps: If > 0, linearly anneal KL from 0 → full over this many steps.
        latent_slice: Optional (start, end) for per-factor KL.
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        weight: float = 1.0,
        warmup_steps: int = 0,
        latent_slice: Optional[tuple[int, int]] = None,
    ):
        super().__init__(name, dataset, weight=weight, latent_slice=latent_slice)
        self.warmup_steps = warmup_steps
        self._train_step = 0

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        """KL requires model to output MU and LOGVAR."""
        # We can't check at attach time since we don't have a model output yet.
        # We'll check at compute_loss time. But we can verify the model has
        # latent_dim > 0.
        if not hasattr(autoencoder, "latent_dim") or autoencoder.latent_dim <= 0:
            raise TaskError(
                "KLDivergenceTask requires model with latent_dim > 0"
            )

    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        """KL has no probe head."""
        return None

    @property
    def anneal_factor(self) -> float:
        """Current annealing multiplier (0.0 to 1.0)."""
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, self._train_step / self.warmup_steps)

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        """Compute KL divergence from mu/logvar in model output.

        Returns scalar KL loss (mean over batch and dims), scaled by
        annealing factor if warmup_steps > 0.
        """
        self._train_step += 1

        if ModelOutput.MU not in model_output or ModelOutput.LOGVAR not in model_output:
            raise TaskError(
                "KLDivergenceTask requires model output to contain MU and LOGVAR. "
                "Is the model a VAE with reparameterization?"
            )

        mu = model_output[ModelOutput.MU]
        logvar = model_output[ModelOutput.LOGVAR]

        # Apply latent_slice if set
        if self.latent_slice is not None:
            start, end = self.latent_slice
            mu = mu[:, start:end]
            logvar = logvar[:, start:end]

        # KL(q(z|x) || p(z)) where q = N(mu, sigma^2), p = N(0, 1)
        # Per-dim: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        #
        # We sum over latent dims then mean over batch, but normalize
        # by the number of latent dims D. This makes the loss scale
        # independent of latent size, so weight=1.0 means "standard VAE"
        # whether the latent is 16-dim or 2048-dim.
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        D = max(kl_per_dim.shape[-1], 1)
        raw_kl = kl_per_dim.sum(dim=-1).mean() / D

        # Apply annealing: 0 → 1 over warmup_steps
        return raw_kl * self.anneal_factor

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Compute average KL on eval split."""
        autoencoder.eval()

        total_kl = 0.0
        n_batches = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images = batch[0].to(device)
            model_output = autoencoder(images)

            if ModelOutput.MU not in model_output:
                return {EvalMetric.KL: 0.0}

            mu = model_output[ModelOutput.MU]
            logvar = model_output[ModelOutput.LOGVAR]

            if self.latent_slice is not None:
                start, end = self.latent_slice
                mu = mu[:, start:end]
                logvar = logvar[:, start:end]

            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            D = max(kl_per_dim.shape[-1], 1)
            total_kl += (kl_per_dim.sum(dim=-1).mean() / D).item()
            n_batches += 1

        autoencoder.train()
        return {EvalMetric.KL: total_kl / max(n_batches, 1)}
