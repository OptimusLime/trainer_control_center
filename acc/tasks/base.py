"""Task base class and TaskError.

A Task binds a dataset to a loss function and an evaluation metric.
It owns a probe head that it builds when it attaches to the autoencoder.

Critical: tasks check compatibility. task.attach(autoencoder) calls
task.check_compatible(autoencoder, dataset) first. If incompatible,
it raises TaskError with a human-readable explanation.

Tasks receive the full model_output dict (keyed by ModelOutput enum) and
pick what they need. Two ways to target a latent subset:

1. latent_slice=(start, end) — reads LATENT[:, start:end] (flat vector).
   Use for simple models without factor groups (e.g., ConvVAE).

2. factor_name="digit" — reads FACTOR_SLICES["digit"] (spatially pooled
   factor vector). Use for FactorSlot models. The model must have
   factor_groups and the forward output must contain FACTOR_SLICES.

Same class, different config.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from acc.dataset import AccDataset
from acc.model_output import ModelOutput


class TaskError(Exception):
    """Raised when a task can't attach to a model or dataset.

    Message must clearly explain what's wrong, e.g.:
    "ReconstructionTask requires model with decoder, but model.has_decoder=False"
    """

    pass


class Task(ABC):
    """Base class for all tasks.

    Subclasses must implement:
        - check_compatible(autoencoder, dataset): raise TaskError if incompatible
        - _build_head(latent_dim): return nn.Module probe head (or None)
        - compute_loss(model_output, batch): return scalar loss tensor
        - evaluate(autoencoder, device): return dict of metric_name -> float

    Args:
        name: Human-readable task name.
        dataset: The dataset this task trains/evals on.
        weight: Loss weight multiplier.
        latent_slice: Optional (start, end) tuple. When set, the probe reads
            latent[:, start:end] instead of the full latent vector. When None,
            reads the full latent. Same class supports both — config, not inheritance.
    """

    def __init__(
        self,
        name: str,
        dataset: AccDataset,
        weight: float = 1.0,
        latent_slice: Optional[tuple[int, int]] = None,
        factor_name: Optional[str] = None,
    ):
        if latent_slice is not None and factor_name is not None:
            raise ValueError(
                "Cannot set both latent_slice and factor_name. "
                "Use latent_slice for flat vector indexing, "
                "factor_name for spatially-pooled factor vectors."
            )
        self.name = name
        self.dataset = dataset
        self.weight = weight
        self.latent_slice = latent_slice
        self.factor_name = factor_name
        self.enabled = True
        self.head: Optional[nn.Module] = None
        self._attached = False

    def attach(self, autoencoder: nn.Module) -> None:
        """Build the probe head and attach to the autoencoder.

        Calls check_compatible first — raises TaskError if the model or
        dataset is incompatible with this task.

        Head input dim is determined by:
        - factor_name: reads from model's factor_groups to get dim
        - latent_slice: (end - start)
        - otherwise: full latent_dim
        """
        self.check_compatible(autoencoder, self.dataset)

        full_latent_dim = autoencoder.latent_dim

        if self.factor_name is not None:
            # Resolve head_dim from the model's factor_groups
            if not hasattr(autoencoder, "factor_groups"):
                raise TaskError(
                    f"Task '{self.name}' uses factor_name='{self.factor_name}' "
                    f"but model has no factor_groups attribute. "
                    f"Use latent_slice for non-factor models."
                )
            fg_map = {fg.name: fg for fg in autoencoder.factor_groups}
            if self.factor_name not in fg_map:
                available = [fg.name for fg in autoencoder.factor_groups]
                raise TaskError(
                    f"Task '{self.name}' uses factor_name='{self.factor_name}' "
                    f"but model has no such factor group. "
                    f"Available: {available}"
                )
            head_dim = fg_map[self.factor_name].latent_dim
        elif self.latent_slice is not None:
            start, end = self.latent_slice
            if start < 0 or end > full_latent_dim or start >= end:
                raise TaskError(
                    f"latent_slice=({start}, {end}) is out of bounds for "
                    f"model with latent_dim={full_latent_dim}"
                )
            head_dim = end - start
        else:
            head_dim = full_latent_dim

        self.head = self._build_head(head_dim)
        self._autoencoder = autoencoder
        self._attached = True

    def _get_latent(self, model_output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the latent tensor this task should read.

        Priority: factor_name > latent_slice > full latent.
        This is the single place where latent extraction happens — all tasks call this.
        """
        if self.factor_name is not None:
            # FACTOR_SLICES is dict[str, Tensor] inside the model_output dict.
            # The outer dict type annotation can't express this, so we cast.
            factor_slices: dict[str, torch.Tensor] | None = model_output.get(  # type: ignore[assignment]
                ModelOutput.FACTOR_SLICES
            )
            if factor_slices is None:
                raise TaskError(
                    f"Task '{self.name}' uses factor_name='{self.factor_name}' "
                    f"but model output has no FACTOR_SLICES."
                )
            if self.factor_name not in factor_slices:
                raise TaskError(
                    f"Task '{self.name}' uses factor_name='{self.factor_name}' "
                    f"but FACTOR_SLICES keys are: {list(factor_slices.keys())}"
                )
            return factor_slices[self.factor_name]

        latent = model_output[ModelOutput.LATENT]
        if self.latent_slice is not None:
            start, end = self.latent_slice
            return latent[:, start:end]
        return latent

    @abstractmethod
    def check_compatible(self, autoencoder: nn.Module, dataset: AccDataset) -> None:
        """Raise TaskError with clear message if model or dataset is incompatible."""
        ...

    @abstractmethod
    def _build_head(self, latent_dim: int) -> Optional[nn.Module]:
        """Build and return the probe head nn.Module, or None if not needed."""
        ...

    @abstractmethod
    def compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: tuple
    ) -> torch.Tensor:
        """Compute task loss given model forward output dict and a data batch.

        Use self._get_latent(model_output) to get the (possibly sliced) latent.
        Returns a scalar loss tensor.
        """
        ...

    @abstractmethod
    def evaluate(
        self, autoencoder: nn.Module, device: torch.device
    ) -> dict[str, float]:
        """Run evaluation on the eval split. Return dict of metric_name -> value."""
        ...

    def head_parameters(self) -> list[nn.Parameter]:
        """Return probe head parameters for separate optimizer group."""
        if self.head is None:
            return []
        return list(self.head.parameters())

    def describe(self) -> dict:
        """Metadata for dashboard display."""
        info = {
            "name": self.name,
            "type": type(self).__name__,
            "enabled": self.enabled,
            "weight": self.weight,
            "attached": self._attached,
            "dataset": self.dataset.name,
        }
        if self.latent_slice is not None:
            info["latent_slice"] = list(self.latent_slice)
        if self.factor_name is not None:
            info["factor_name"] = self.factor_name
        return info
