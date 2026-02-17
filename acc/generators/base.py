"""DatasetGenerator base class.

A DatasetGenerator produces AccDatasets from configurable parameters.
Subclasses implement generate() with their specific logic.

The GeneratorRegistry discovers DatasetGenerator subclasses from
acc/generators/ and makes them available via the dashboard.
"""

from abc import ABC, abstractmethod
from typing import Any

from acc.dataset import AccDataset


class DatasetGenerator(ABC):
    """Base class for dataset generators.

    Subclasses must define:
        - name: Human-readable name (class attribute)
        - description: One-line description (class attribute)
        - parameters: dict of parameter name -> {type, default, description}
        - generate(**params) -> AccDataset

    The parameters dict tells the dashboard what form fields to render.
    """

    name: str = "unnamed"
    description: str = ""
    parameters: dict[str, dict[str, Any]] = {}

    @abstractmethod
    def generate(self, **params) -> AccDataset:
        """Generate a dataset with the given parameters.

        Args:
            **params: Generator-specific parameters matching self.parameters keys.

        Returns:
            AccDataset ready for training/eval.
        """
        ...

    def describe(self) -> dict:
        """Metadata for dashboard/API display."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
