"""Task base class and TaskError.

A Task binds a dataset to a loss function and an evaluation metric.
It owns a probe head that it builds when it attaches to the autoencoder.

Critical: tasks check compatibility. task.attach(autoencoder) calls
task.check_compatible(autoencoder, dataset) first. If incompatible,
it raises TaskError with a human-readable explanation.

Tasks receive the full model_output dict (keyed by ModelOutput enum) and
pick what they need. A task with latent_slice=(8, 24) reads z[:, 8:24]
instead of the full latent. Same class, different config.
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
    ):
        self.name = name
        self.dataset = dataset
        self.weight = weight
        self.latent_slice = latent_slice
        self.enabled = True
        self.head: Optional[nn.Module] = None
        self._attached = False

    def attach(self, autoencoder: nn.Module) -> None:
        """Build the probe head and attach to the autoencoder.

        Calls check_compatible first — raises TaskError if the model or
        dataset is incompatible with this task.

        Head input dim is determined by latent_slice if set, else full latent_dim.
        """
        self.check_compatible(autoencoder, self.dataset)

        full_latent_dim = autoencoder.latent_dim

        if self.latent_slice is not None:
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

        If latent_slice is set, returns the slice. Otherwise returns full latent.
        This is the single place where slicing happens — all tasks call this.
        """
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
        return info
