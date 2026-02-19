"""Recipe base class — a program that operates on the checkpoint tree.

A recipe creates models, loads datasets, attaches tasks, forks checkpoints,
trains, evaluates, and forks again. The checkpoint tree IS the experiment.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

import torch.nn as nn

from acc.dataset import AccDataset
from acc.tasks.base import Task


class Recipe(ABC):
    """Base class for all recipes.

    Subclasses implement run(ctx) which uses ctx to perform tree operations:
    create_model, load_dataset, attach_task, fork, train, evaluate.
    """

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def run(self, ctx: "RecipeContext") -> None:
        """Execute the recipe."""
        ...


class RecipeContext:
    """Operations available to a recipe. Wraps TrainerAPI state.

    This is the interface between recipes and the trainer process.
    Recipes call these methods to build the checkpoint tree.
    """

    def __init__(self, api: "TrainerAPI", recipe: Recipe):  # noqa: F821 — forward ref
        self._api = api
        self._recipe = recipe
        self._phase = "initializing"
        self._phases_completed: list[str] = []
        self._checkpoints_created: list[str] = []
        self._stopped = False
        # Branch tracking
        self._current_branch: Optional[str] = None
        self._branch_index: int = 0
        self._total_branches: int = 0
        self._branches: list[dict] = []  # [{name, description, phases, results}]
        self._branch_results: dict = {}  # branch_name -> eval results

    @property
    def phase(self) -> str:
        return self._phase

    @phase.setter
    def phase(self, name: str) -> None:
        if self._phase != "initializing":
            self._phases_completed.append(self._phase)
        self._phase = name

    @contextmanager
    def branch(self, name: str, description: str = "", total: int = 0):
        """Declare a branch context. Groups phases under a named branch.

        Usage:
            with ctx.branch("baseline", "20ch, no stop-grad", total=3):
                ctx.phase = "Build model"
                ...
        """
        # Auto-detect total from first call if not set
        if total > 0 and self._total_branches == 0:
            self._total_branches = total
        self._branch_index += 1
        self._current_branch = name
        # Flush current phase so setup phases before this branch are counted
        if self._phase != "initializing":
            self._phases_completed.append(self._phase)
            self._phase = "initializing"
        branch_entry = {
            "name": name,
            "description": description,
            "phase_start": len(self._phases_completed),
        }
        self._branches.append(branch_entry)
        try:
            yield
        finally:
            # Flush the last phase inside the branch
            if self._phase != "initializing":
                self._phases_completed.append(self._phase)
                self._phase = "initializing"
            branch_entry["phase_end"] = len(self._phases_completed)
            self._current_branch = None

    def record_results(self, branch_name: str, results: dict) -> None:
        """Store eval results for a branch. Shown in comparison summary."""
        self._branch_results[branch_name] = results

    @property
    def current_checkpoint_id(self) -> Optional[str]:
        if self._api.checkpoints is None:
            return None
        return self._api.checkpoints.current_id

    def create_model(self, builder: Callable[[], nn.Module]) -> None:
        """Set the current model. builder() returns an nn.Module.

        Resets checkpoint lineage so the first save after a new model
        becomes a new root node in the checkpoint tree (not chained to
        whatever was saved last).
        """
        model = builder()
        model = model.to(self._api.device)
        self._api.autoencoder = model
        self._api.trainer = None
        self._api.tasks.clear()
        # Reset checkpoint lineage — new model = new root
        if self._api.checkpoints is not None:
            self._api.checkpoints._current_id = None

    def load_dataset(self, name: str, loader: Callable[[], AccDataset]) -> AccDataset:
        """Load or generate a dataset. Returns it and registers with the API."""
        if name in self._api.datasets:
            return self._api.datasets[name]
        ds = loader()
        self._api.datasets[name] = ds
        return ds

    def attach_task(self, task: Task) -> None:
        """Attach a task to the current model."""
        if self._api.autoencoder is None:
            raise RuntimeError("No model loaded. Call create_model() first.")
        task.attach(self._api.autoencoder)
        self._api.tasks[task.name] = task

    def detach_all_tasks(self) -> None:
        """Remove all tasks."""
        self._api.tasks.clear()
        self._api.trainer = None

    def save_checkpoint(self, tag: str, parent_id: Optional[str] = None,
                        description: Optional[str] = None) -> str:
        """Save current state, return checkpoint_id.

        Recipe name, model config, task snapshot, and loss summary are all
        persisted IN the .pt file so the checkpoint is self-describing.

        Args:
            tag: Human-readable tag for the checkpoint.
            parent_id: Explicit parent checkpoint ID. If None, uses the
                checkpoint store's current lineage (last saved/loaded).
            description: Human-readable purpose of this checkpoint.
        """
        if self._api.autoencoder is None:
            raise RuntimeError("No model to checkpoint.")
        self._ensure_trainer()
        if self._api.checkpoints is None:
            from acc.checkpoints import CheckpointStore
            self._api.checkpoints = CheckpointStore("./acc/checkpoints_data")

        # Build metrics (loss summary + full history) BEFORE save so it's in the .pt file
        metrics = {}
        from acc.loss_health import compute_loss_summary
        recent_jobs = self._api.jobs.list()
        for j in recent_jobs:
            if j.losses:
                summaries = compute_loss_summary(j.losses)
                metrics["loss_summary"] = {
                    name: s.to_dict() for name, s in summaries.items()
                }
                metrics["loss_history"] = j.losses
                break

        cp = self._api.checkpoints.save(
            self._api.autoencoder,
            self._api.trainer,
            tag=tag,
            parent_id=parent_id,
            recipe_name=self._recipe.name,
            description=description or self._phase,
            metrics=metrics,
        )
        self._checkpoints_created.append(cp.id)
        return cp.id

    def fork(self, checkpoint_id: str, tag: str) -> str:
        """Fork from a checkpoint, load that state, return new checkpoint_id."""
        if self._api.checkpoints is None:
            raise RuntimeError("No checkpoint store.")
        # Fork creates a copy of the .pt file
        fork_cp = self._api.checkpoints.fork(checkpoint_id, tag)
        # Load the forked state
        self._ensure_trainer()
        self._api.checkpoints.load(fork_cp.id, self._api.autoencoder, self._api.trainer)
        self._checkpoints_created.append(fork_cp.id)
        return fork_cp.id

    def train(
        self,
        steps: int,
        lr: float = 1e-3,
        probe_lr: float = 1e-3,
        batch_size: int = 64,
        task_weights: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        """Train for N steps. Routes through JobManager so losses are visible in the dashboard.

        Args:
            steps: Number of training steps.
            lr: Learning rate for model parameters.
            probe_lr: Learning rate for probe heads.
            batch_size: Batch size for dataloaders.
            task_weights: Optional dict mapping task_name -> sampling weight.
                Weights are relative — {"recon": 9, "kl": 1} means recon
                is sampled 90% of steps.  If None, uniform sampling.
        """
        if self._stopped:
            return []
        self._ensure_trainer(batch_size=batch_size)
        trainer = self._api.trainer
        if trainer.lr != lr or trainer.probe_lr != probe_lr:
            trainer.lr = lr
            trainer.probe_lr = probe_lr
            trainer._build_optimizers()
        if trainer.batch_size != batch_size:
            trainer.batch_size = batch_size
        # Route through JobManager: creates a job, wires on_step for SSE + loss history,
        # runs synchronously (blocking=True). This makes recipe training visible in the
        # dashboard's loss chart, loss log, job history, and loss summary.
        job = self._api.jobs.start(
            trainer,
            steps=steps,
            checkpoint_id=self.current_checkpoint_id,
            blocking=True,
            task_weights=task_weights,
        )
        return job.losses

    def evaluate(self) -> dict[str, dict[str, float]]:
        """Run eval on all tasks."""
        self._ensure_trainer()
        return self._api.trainer.evaluate_all()

    def log(self, message: str) -> None:
        """Log a message (for now just print, later SSE)."""
        print(f"[Recipe] {message}")

    def stop(self) -> None:
        """Signal the recipe to stop after current operation."""
        self._stopped = True
        if self._api.trainer is not None:
            self._api.trainer.stop()

    def _ensure_trainer(self, batch_size: int = 64) -> None:
        """Rebuild the trainer if tasks changed."""
        from acc.trainer import Trainer
        tasks = list(self._api.tasks.values())
        if not tasks:
            if self._api.trainer is None:
                # Create a minimal trainer even without tasks (for checkpoint save/load)
                self._api.trainer = Trainer(
                    self._api.autoencoder, [], self._api.device,
                    batch_size=batch_size,
                )
            return
        if self._api.trainer is None:
            self._api.trainer = Trainer(
                self._api.autoencoder, tasks, self._api.device,
                batch_size=batch_size,
            )
        else:
            # Update task list if it changed
            self._api.trainer.tasks = tasks
            self._api.trainer._build_optimizers()


@dataclass
class RecipeJob:
    """Tracks a running recipe execution."""

    id: str
    recipe_name: str
    state: str = "running"  # running, completed, stopped, failed
    current_phase: str = "initializing"
    phases_completed: list[str] = field(default_factory=list)
    checkpoints_created: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    # Branch tracking
    current_branch: Optional[str] = None
    branch_index: int = 0
    total_branches: int = 0
    branches: list[dict] = field(default_factory=list)
    branch_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "recipe_name": self.recipe_name,
            "state": self.state,
            "current_phase": self.current_phase,
            "phases_completed": self.phases_completed,
            "checkpoints_created": self.checkpoints_created,
            "started_at": self.started_at.isoformat(),
            "error": self.error,
            "current_branch": self.current_branch,
            "branch_index": self.branch_index,
            "total_branches": self.total_branches,
            "branches": self.branches,
            "branch_results": self.branch_results,
        }
