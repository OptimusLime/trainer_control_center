"""Trainer — multi-task training with round-robin task scheduling.

All losses come from tasks. No special reconstruction carve-out.
Task losses round-robin: each step samples from the next enabled task's dataloader.
"""

from typing import Callable, Optional
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from acc.tasks.base import Task
from acc.loss_health import classify_loss


class Trainer:
    """Trains an autoencoder on multiple tasks simultaneously.

    Model-agnostic: any nn.Module whose forward() returns dict[str, Tensor]
    keyed by ModelOutput enum values works. The Trainer is a dumb pipe.

    Args:
        autoencoder: The model to train (any nn.Module with forward()->dict).
        tasks: List of attached Task instances.
        device: Torch device (cuda/cpu).
        lr: Learning rate for model parameters.
        probe_lr: Learning rate for probe head parameters.
    """

    def __init__(
        self,
        autoencoder: nn.Module,
        tasks: list[Task],
        device: torch.device,
        lr: float = 1e-3,
        probe_lr: float = 1e-3,
        batch_size: int = 64,
    ):
        self.autoencoder = autoencoder
        self.tasks = tasks
        self.device = device
        self.lr = lr
        self.probe_lr = probe_lr
        self.batch_size = batch_size
        self._stop_requested = False

        # Move model to device
        self.autoencoder.to(device)

        # Move probe heads to device
        for task in self.tasks:
            if task.head is not None:
                task.head.to(device)

        # Build optimizers
        self._build_optimizers()

    def _build_optimizers(self):
        """Build separate optimizers for model params and probe params."""
        self.model_optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr)

        # Collect all probe head parameters
        probe_params = []
        for task in self.tasks:
            probe_params.extend(task.head_parameters())

        if probe_params:
            self.probe_optimizer = optim.Adam(probe_params, lr=self.probe_lr)
        else:
            self.probe_optimizer = None

    def _enabled_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.enabled]

    def train(
        self,
        steps: int,
        on_step: Optional[Callable[[dict], None]] = None,
    ) -> list[dict]:
        """Train for the given number of steps with round-robin task scheduling.

        Args:
            steps: Number of training steps.
            on_step: Optional callback called after each step with step info dict.

        Returns:
            List of per-step loss dicts.
        """
        self._stop_requested = False
        self.autoencoder.train()
        for task in self.tasks:
            if task.head is not None:
                task.head.train()

        enabled = self._enabled_tasks()
        if not enabled:
            return []

        # Build dataloaders for enabled tasks (round-robin iterators)
        task_iters = {}
        task_loaders = {}
        for task in enabled:
            loader = task.dataset.train_loader(self.batch_size)
            task_loaders[task.name] = loader
            task_iters[task.name] = iter(loader)

        # Round-robin cycle through tasks
        task_cycle = itertools.cycle(enabled)
        loss_history = []

        for step in range(1, steps + 1):
            if self._stop_requested:
                break

            task = next(task_cycle)

            # Get next batch, restart iterator if exhausted
            try:
                batch = next(task_iters[task.name])
            except StopIteration:
                task_iters[task.name] = iter(task_loaders[task.name])
                batch = next(task_iters[task.name])

            # Move batch to device
            batch = tuple(t.to(self.device) for t in batch)

            # Forward through autoencoder — model returns a dict, task picks what it needs
            model_output = self.autoencoder(batch[0])

            # Compute task loss
            loss = task.compute_loss(model_output, batch) * task.weight

            # Backward and step
            self.model_optimizer.zero_grad()
            if self.probe_optimizer is not None:
                self.probe_optimizer.zero_grad()

            loss.backward()

            self.model_optimizer.step()
            if self.probe_optimizer is not None:
                self.probe_optimizer.step()

            # Record
            loss_val = loss.item()
            task_type = type(task).__name__
            step_info = {
                "step": step,
                "task_name": task.name,
                "task_type": task_type,
                "task_loss": loss_val,
                "health": classify_loss(task_type, loss_val).value,
            }
            loss_history.append(step_info)

            if on_step is not None:
                on_step(step_info)

        return loss_history

    def stop(self):
        """Request training to stop after current step."""
        self._stop_requested = True

    @torch.no_grad()
    def evaluate_all(self) -> dict[str, dict[str, float]]:
        """Run evaluation for all enabled tasks.

        Returns:
            Dict mapping task_name -> {metric_name: value}
        """
        results = {}
        for task in self._enabled_tasks():
            results[task.name] = task.evaluate(self.autoencoder, self.device)
        return results

    def state_dict(self) -> dict:
        """Full state for checkpointing: model + optimizer + probe heads."""
        state = {
            "autoencoder": self.autoencoder.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "lr": self.lr,
            "probe_lr": self.probe_lr,
        }
        if self.probe_optimizer is not None:
            state["probe_optimizer"] = self.probe_optimizer.state_dict()

        # Save probe head states
        probe_states = {}
        for task in self.tasks:
            if task.head is not None:
                probe_states[task.name] = task.head.state_dict()
        state["probe_heads"] = probe_states

        return state

    def load_state_dict(self, state: dict):
        """Restore full state from checkpoint."""
        self.autoencoder.load_state_dict(state["autoencoder"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])

        if "probe_optimizer" in state and self.probe_optimizer is not None:
            self.probe_optimizer.load_state_dict(state["probe_optimizer"])

        # Restore probe head states
        probe_states = state.get("probe_heads", {})
        for task in self.tasks:
            if task.head is not None and task.name in probe_states:
                task.head.load_state_dict(probe_states[task.name])
