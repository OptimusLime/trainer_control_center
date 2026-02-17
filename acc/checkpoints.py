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
    """Metadata for a saved checkpoint."""

    id: str
    tag: str
    parent_id: Optional[str] = None
    step: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tag": self.tag,
            "parent_id": self.parent_id,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


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
    ) -> Checkpoint:
        """Save a checkpoint to disk.

        Args:
            autoencoder: The model (any nn.Module).
            trainer: The trainer (for optimizer + probe states).
            tag: Human-readable tag.
            parent_id: Parent checkpoint ID (for tree tracking).

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

        checkpoint = Checkpoint(
            id=cp_id,
            tag=tag,
            parent_id=parent_id or self._current_id,
            step=step,
        )

        # Save full state to disk
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
    ) -> Checkpoint:
        """Load a checkpoint from disk and restore model + trainer state.

        Args:
            checkpoint_id: The checkpoint ID to load.
            autoencoder: The model to restore into.
            trainer: The trainer to restore into.

        Returns:
            The Checkpoint metadata.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        path = os.path.join(self.directory, f"{checkpoint_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        state = torch.load(path, weights_only=False)
        trainer.load_state_dict(state["trainer_state"])

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
