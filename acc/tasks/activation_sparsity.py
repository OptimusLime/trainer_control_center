"""ActivationSparsityTask — fraction of channels above mean activation.

Evaluation-only task (extends EvalOnlyTask — never sampled during training).
At eval time, runs test images through the model, captures activations
at the named layer via a forward hook, and computes:

1. Activation sparsity: fraction of channels above mean activation,
   averaged across images. Lower = sparser = more specialized.
2. Sparsity variance: variance of per-image sparsity. Higher = different
   images activate different channel subsets = more specialization.

Works with any model — captures activations by hooking a named module.
Handles both 2D (Linear: [B, C]) and 4D (Conv2d: [B, C, H, W]) outputs.

This is a library abstraction — works with any model, not just our architectures.
"""

import torch
import torch.nn as nn

from acc.dataset import AccDataset
from acc.eval_metric import EvalMetric
from acc.tasks.base import EvalOnlyTask, TaskError


class ActivationSparsityTask(EvalOnlyTask):
    """Measure activation sparsity of a specific layer.

    Evaluation-only (extends EvalOnlyTask). At eval time, hooks the
    named layer to capture activations, runs eval data through the model,
    and reports sparsity statistics.

    Args:
        name: Task name.
        dataset: Dataset to run through model for activation capture.
        layer_name: Name of the module to hook (as in model.named_modules()).
        n_eval_batches: Max batches to process during eval (caps compute cost).
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        layer_name: str,
        weight: float = 0.0,
        n_eval_batches: int = 10,
    ):
        super().__init__(name=name, dataset=dataset, weight=weight)
        self.layer_name = layer_name
        self.n_eval_batches = n_eval_batches

    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        named = dict(autoencoder.named_modules())
        if self.layer_name not in named:
            available = [n for n, _ in autoencoder.named_modules() if n]
            raise TaskError(
                f"ActivationSparsityTask '{self.name}' targets layer '{self.layer_name}' "
                f"but it was not found in model. Available: {available}"
            )

    @torch.no_grad()
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Capture activations and compute sparsity metrics.

        For each image, compute the fraction of channels whose mean
        activation exceeds the layer-wide mean. Then report:
        - mean sparsity across images (ACTIVATION_SPARSITY)
        - variance of sparsity across images (SPARSITY_VARIANCE)
        """
        autoencoder.eval()

        # Hook to capture activations
        captured: list[torch.Tensor] = []
        named = dict(autoencoder.named_modules())
        target_module = named[self.layer_name]

        def capture_hook(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            captured.append(out.detach())

        handle = target_module.register_forward_hook(capture_hook)

        try:
            per_image_sparsities: list[float] = []

            for i, batch in enumerate(self.dataset.eval_loader(batch_size=256)):
                if i >= self.n_eval_batches:
                    break

                images = batch[0].to(device)
                captured.clear()
                autoencoder(images)

                if not captured:
                    continue
                activations = captured[0]  # [B, C, ...] or [B, C]

                # Compute per-channel mean activation for each image
                if activations.ndim == 2:
                    # [B, C] — already per-channel
                    channel_act = activations.abs()  # [B, C]
                elif activations.ndim >= 3:
                    # [B, C, H, W, ...] — mean over spatial dims
                    spatial_dims = tuple(range(2, activations.ndim))
                    channel_act = activations.abs().mean(dim=spatial_dims)  # [B, C]
                else:
                    continue

                # Per-image sparsity: fraction of channels above mean
                # For each image: which channels are above that image's mean?
                image_means = channel_act.mean(dim=1, keepdim=True)  # [B, 1]
                above_mean = (channel_act > image_means).float()  # [B, C]
                sparsity_per_image = above_mean.mean(dim=1)  # [B] — fraction above mean

                per_image_sparsities.extend(sparsity_per_image.cpu().tolist())
        finally:
            handle.remove()
            autoencoder.train()

        if not per_image_sparsities:
            return {EvalMetric.ACTIVATION_SPARSITY: 0.0, EvalMetric.SPARSITY_VARIANCE: 0.0}

        sparsity_tensor = torch.tensor(per_image_sparsities)
        mean_sparsity = sparsity_tensor.mean().item()
        var_sparsity = sparsity_tensor.var().item() if len(per_image_sparsities) > 1 else 0.0

        return {
            EvalMetric.ACTIVATION_SPARSITY: mean_sparsity,
            EvalMetric.SPARSITY_VARIANCE: var_sparsity,
        }

    def describe(self) -> dict:
        info = super().describe()
        info["layer_name"] = self.layer_name
        info["n_eval_batches"] = self.n_eval_batches
        return info
