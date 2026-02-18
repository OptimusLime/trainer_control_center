"""CheckpointStore — saves model + optimizer + probe states to disk.

Tracks parent-child relationships for tree view (M4).
Checkpoints are .pt files on disk from day 1.
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

from acc.trainer import Trainer


@dataclass
class Checkpoint:
    """Metadata for a saved checkpoint.

    Core fields (always present):
        id, tag, parent_id, step, timestamp

    Rich metadata (populated by recipes, may be empty for manual saves):
        recipe_name:  Which recipe created this checkpoint
        description:  Human-readable purpose (e.g. "20ch, no stop-grad, control branch")
        model_config: Architectural config dict from model.config()
        tasks_snapshot: List of {name, type, dataset, weight, latent_slice} at save time
        metrics:      Loss summary, eval results, etc.
    """

    id: str
    tag: str
    parent_id: Optional[str] = None
    step: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    recipe_name: Optional[str] = None
    description: Optional[str] = None
    model_config: dict = field(default_factory=dict)
    tasks_snapshot: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tag": self.tag,
            "parent_id": self.parent_id,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "recipe_name": self.recipe_name,
            "description": self.description,
            "model_config": self.model_config,
            "tasks_snapshot": self.tasks_snapshot,
            "metrics": self.metrics,
        }


def _snapshot_tasks(tasks: list) -> list[dict]:
    """Capture current task configuration as serializable dicts."""
    snapshot = []
    for t in tasks:
        info = {
            "name": t.name,
            "type": type(t).__name__,
        }
        if hasattr(t, "dataset") and t.dataset is not None:
            info["dataset"] = t.dataset.name if hasattr(t.dataset, "name") else str(t.dataset)
        if hasattr(t, "weight"):
            info["weight"] = t.weight
        if hasattr(t, "latent_slice") and t.latent_slice is not None:
            info["latent_slice"] = list(t.latent_slice)
        snapshot.append(info)
    return snapshot


class CheckpointStore:
    """Disk-backed checkpoint store with parent-child tracking.

    Args:
        directory: Path to directory where .pt files are saved.
    """

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._checkpoints: dict[str, Checkpoint] = {}
        self._current_id: Optional[str] = None

    def save(
        self,
        autoencoder: nn.Module,
        trainer: Trainer,
        tag: str,
        parent_id: Optional[str] = None,
        recipe_name: Optional[str] = None,
        description: Optional[str] = None,
        model_config: Optional[dict] = None,
        tasks_snapshot: Optional[list] = None,
        metrics: Optional[dict] = None,
    ) -> Checkpoint:
        """Save a checkpoint to disk.

        Args:
            autoencoder: The model (any nn.Module).
            trainer: The trainer (for optimizer + probe states).
            tag: Human-readable tag.
            parent_id: Parent checkpoint ID (for tree tracking).
            recipe_name: Which recipe created this checkpoint.
            description: Human-readable purpose of this checkpoint.
            model_config: Model architectural config (from model.config()).
            tasks_snapshot: List of task configs at save time.
            metrics: Pre-computed metrics (loss_summary, eval results).

        Returns:
            The Checkpoint metadata.
        """
        cp_id = uuid.uuid4().hex[:12]

        # Get current step from the latest job losses if available
        step = 0
        if trainer.model_optimizer.state:
            # Approximate step from optimizer state
            for group in trainer.model_optimizer.param_groups:
                if "step" in group:
                    step = group["step"]
                    break

        # Extract model config if not provided and model supports it
        if model_config is None and hasattr(autoencoder, "config"):
            model_config = autoencoder.config()

        # Snapshot current tasks if not provided
        if tasks_snapshot is None:
            tasks_snapshot = _snapshot_tasks(trainer.tasks)

        checkpoint = Checkpoint(
            id=cp_id,
            tag=tag,
            parent_id=parent_id or self._current_id,
            step=step,
            recipe_name=recipe_name,
            description=description,
            model_config=model_config or {},
            tasks_snapshot=tasks_snapshot or [],
            metrics=metrics or {},
        )

        # Save full state to disk — metadata is complete BEFORE torch.save
        state = {
            "trainer_state": trainer.state_dict(),
            "checkpoint_meta": checkpoint.to_dict(),
        }
        path = os.path.join(self.directory, f"{cp_id}.pt")
        torch.save(state, path)

        self._checkpoints[cp_id] = checkpoint
        self._current_id = cp_id
        return checkpoint

    def load(
        self,
        checkpoint_id: str,
        autoencoder: nn.Module,
        trainer: Trainer,
        device: Optional[torch.device] = None,
    ) -> Checkpoint:
        """Load a checkpoint from disk and restore model + trainer state.

        Args:
            checkpoint_id: The checkpoint ID to load.
            autoencoder: The model to restore into.
            trainer: The trainer to restore into.
            device: Target device for map_location. If None, uses trainer.device.

        Returns:
            The Checkpoint metadata.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        path = os.path.join(self.directory, f"{checkpoint_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        target_device = device or trainer.device
        state = torch.load(path, weights_only=False, map_location=target_device)
        trainer.load_state_dict(state["trainer_state"])

        # Ensure model and probe heads are on the target device after load
        autoencoder.to(target_device)
        for task in trainer.tasks:
            if task.head is not None:
                task.head.to(target_device)

        self._current_id = checkpoint_id
        return self._checkpoints.get(
            checkpoint_id,
            Checkpoint(
                id=checkpoint_id,
                tag=state["checkpoint_meta"].get("tag", "unknown"),
            ),
        )

    def fork(self, checkpoint_id: str, new_tag: str) -> Checkpoint:
        """Create a fork point from an existing checkpoint.

        The fork checkpoint has the same parent as the source,
        creating a branch in the checkpoint tree.
        """
        source = self._checkpoints.get(checkpoint_id)
        if source is None:
            raise KeyError(f"Checkpoint {checkpoint_id} not found in store")

        fork_id = uuid.uuid4().hex[:12]
        fork_cp = Checkpoint(
            id=fork_id,
            tag=new_tag,
            parent_id=checkpoint_id,
            step=source.step,
            metrics=dict(source.metrics),
        )

        # Copy the .pt file
        src_path = os.path.join(self.directory, f"{checkpoint_id}.pt")
        dst_path = os.path.join(self.directory, f"{fork_id}.pt")
        if os.path.exists(src_path):
            import shutil

            shutil.copy2(src_path, dst_path)

        self._checkpoints[fork_id] = fork_cp
        self._current_id = fork_id
        return fork_cp

    def tree(self) -> list[Checkpoint]:
        """Return flat list of all checkpoints with parent_id links."""
        return list(self._checkpoints.values())

    @property
    def current_id(self) -> Optional[str]:
        return self._current_id
