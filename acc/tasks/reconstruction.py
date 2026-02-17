"""ReconstructionTask — uses model's decoder directly.

No probe head. L1 loss between input and reconstruction. PSNR eval.
check_compatible: model must have decoder (has_decoder=True).

Reads RECONSTRUCTION and SPATIAL from the model_output dict — no re-encode.
Reconstruction is NOT special — it's just another task.
"""

from typing import Optional
import math

import torch
import torch.nn as nn

from acc.dataset import AccDataset
from acc.model_output import ModelOutput
from acc.tasks.base import Task, TaskError


class ReconstructionTask(Task):
    """Reconstruction via the model's decoder. L1 loss, PSNR eval.

    Requires autoencoder with a decoder (has_decoder=True).
    No probe head — reads RECONSTRUCTION directly from model forward output.
    """

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        if not autoencoder.has_decoder:
            raise TaskError(
                f"ReconstructionTask requires model with decoder, "
                f"but model.has_decoder=False"
            )

    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        return None

    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        """L1 reconstruction loss. Reads RECONSTRUCTION from model output dict."""
        images = batch[0]
        recon = model_output[ModelOutput.RECONSTRUCTION]
        return nn.functional.l1_loss(recon, images)

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Compute PSNR and L1 on eval split."""
        autoencoder.eval()

        total_l1 = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in self.dataset.eval_loader(batch_size=256):
            images = batch[0].to(device)
            model_output = autoencoder(images)
            recon = model_output[ModelOutput.RECONSTRUCTION]

            total_l1 += nn.functional.l1_loss(recon, images).item()
            total_mse += nn.functional.mse_loss(recon, images).item()
            n_batches += 1

        autoencoder.train()

        avg_l1 = total_l1 / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)
        psnr = 10 * math.log10(1.0 / max(avg_mse, 1e-10))

        return {"l1": avg_l1, "psnr": psnr}
