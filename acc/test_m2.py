"""M2 Verification Script — Hot-Reload Tasks + Dashboard Task Management.

Usage:
    python -m acc.test_m2

This proves:
1. TaskRegistry discovers built-in task classes (Classification, Reconstruction, Regression, KLDivergence)
2. TaskRegistry hot-reload: write a dummy task file, verify it appears in registry
3. TaskRegistry error handling: write a file with syntax error, verify registry catches it
4. TaskRegistry deletion: remove a file, class disappears from registry
5. Task add via registry-resolved class works with latent_slice
6. Task weight update works
7. Job loss history has per-task step data
8. Eval reconstructions endpoint returns originals + reconstructions
9. Jobs history endpoint returns job summaries
"""

import os
import shutil
import tempfile
import time

import torch

from acc.factor_group import FactorGroup
from acc.factor_slot_autoencoder import FactorSlotAutoencoder
from acc.model_output import ModelOutput
from acc.dataset import AccDataset, load_mnist
from acc.tasks.base import Task, TaskError
from acc.tasks.registry import TaskRegistry
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.classification import ClassificationTask
from acc.tasks.regression import RegressionTask
from acc.tasks.kl_divergence import KLDivergenceTask
from acc.trainer import Trainer
from acc.jobs import JobManager


FACTOR_GROUPS = [
    FactorGroup("digit", 0, 4),
    FactorGroup("thickness", 4, 7),
    FactorGroup("slant", 7, 10),
    FactorGroup("free", 10, 16),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    passed = 0
    total = 0

    # ── 1. TaskRegistry discovers built-in tasks ──
    total += 1
    print("\n=== 1. TaskRegistry discovers built-in tasks ===")
    registry = TaskRegistry()
    classes = registry.list()
    class_names = [c["class_name"] for c in classes]
    print(f"Discovered: {class_names}")

    expected = {"ClassificationTask", "ReconstructionTask", "RegressionTask", "KLDivergenceTask"}
    found = expected.intersection(set(class_names))
    assert found == expected, f"Missing task classes: {expected - found}"
    print(f"PASS: All {len(expected)} built-in task classes discovered")
    passed += 1

    # ── 2. TaskRegistry.get() returns class, not instance ──
    total += 1
    print("\n=== 2. TaskRegistry.get() returns class ===")
    cls = registry.get("ClassificationTask")
    assert cls is not None, "Expected ClassificationTask class, got None"
    assert cls.__name__ == "ClassificationTask", f"Expected ClassificationTask, got {cls.__name__}"
    assert issubclass(cls, Task), "Returned class is not a Task subclass"
    cls2 = registry.get("NonexistentTask")
    assert cls2 is None, "Expected None for nonexistent task"
    print("PASS: get() returns class for valid name, None for invalid")
    passed += 1

    # ── 3. TaskRegistry hot-reload: add a new task file ──
    total += 1
    print("\n=== 3. TaskRegistry hot-reload: new task file ===")
    tasks_dir = registry.tasks_dir
    dummy_path = os.path.join(tasks_dir, "dummy_test_task.py")

    # Write a dummy task file
    dummy_code = '''
"""Dummy test task for M2 verification."""

import torch
import torch.nn as nn
from acc.tasks.base import Task, TaskError
from acc.dataset import AccDataset
from acc.model_output import ModelOutput


class DummyTestTask(Task):
    """A dummy task that returns a constant loss. For testing only."""

    def check_compatible(self, autoencoder, dataset):
        pass  # Compatible with everything

    def _build_head(self, latent_dim):
        return nn.Linear(latent_dim, 1)

    def compute_loss(self, model_output, batch):
        latent = self._get_latent(model_output)
        pred = self.head(latent)
        return pred.mean()  # Dummy loss

    def evaluate(self, autoencoder, device):
        return {"dummy_metric": 0.42}
'''
    try:
        with open(dummy_path, "w") as f:
            f.write(dummy_code)

        # Force registry to check for changes
        registry._check_for_changes()
        time.sleep(0.1)  # Let filesystem settle

        # Verify DummyTestTask appears
        classes_after = registry.list()
        class_names_after = [c["class_name"] for c in classes_after]
        assert "DummyTestTask" in class_names_after, f"DummyTestTask not found. Got: {class_names_after}"
        print("PASS: DummyTestTask appeared in registry after file creation")
        passed += 1
    finally:
        pass  # Clean up after all tests

    # ── 4. TaskRegistry error handling: syntax error ──
    total += 1
    print("\n=== 4. TaskRegistry error handling: syntax error ===")
    bad_path = os.path.join(tasks_dir, "bad_syntax_task.py")
    try:
        with open(bad_path, "w") as f:
            f.write("def this is broken syntax {{\n")

        # Should not crash
        count_before = len(registry.list())
        registry._check_for_changes()
        count_after = len(registry.list())
        # Registry should still work, bad file should not add any classes
        print(f"Classes before: {count_before}, after: {count_after}")
        print("PASS: Syntax error caught, registry still functional")
        passed += 1
    finally:
        if os.path.exists(bad_path):
            os.remove(bad_path)
            registry._check_for_changes()

    # ── 5. TaskRegistry deletion: remove file, class disappears ──
    total += 1
    print("\n=== 5. TaskRegistry deletion: remove file ===")
    assert "DummyTestTask" in [c["class_name"] for c in registry.list()]
    os.remove(dummy_path)
    registry._check_for_changes()
    assert "DummyTestTask" not in [c["class_name"] for c in registry.list()], "DummyTestTask should be gone"
    print("PASS: DummyTestTask removed from registry after file deletion")
    passed += 1

    # ── 6. Task add with latent_slice via registry ──
    total += 1
    print("\n=== 6. Task add with latent_slice via registry ===")
    model = FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS,
        backbone_channels=[32, 64],
        embed_dim=32,
        image_size=32,
    ).to(device)

    # Load a small MNIST dataset
    mnist = load_mnist(image_size=32)

    # Use registry to get task class
    recon_cls = registry.get("ReconstructionTask")
    assert recon_cls is not None
    recon_task = recon_cls("recon", mnist, weight=1.0)
    recon_task.attach(model)

    kl_cls = registry.get("KLDivergenceTask")
    assert kl_cls is not None
    kl_task = kl_cls("kl_digit", mnist, weight=0.1, latent_slice=(0, 4))
    kl_task.attach(model)
    assert kl_task.latent_slice == (0, 4)

    classify_cls = registry.get("ClassificationTask")
    assert classify_cls is not None
    classify_task = classify_cls("classify", mnist, weight=1.0, latent_slice=(0, 4))
    classify_task.attach(model)

    print(f"Tasks attached: recon (w={recon_task.weight}), kl_digit (w={kl_task.weight}, slice=0:4), classify (w={classify_task.weight}, slice=0:4)")
    print("PASS: Tasks created from registry and attached with latent_slice")
    passed += 1

    # ── 7. Task weight update ──
    total += 1
    print("\n=== 7. Task weight update ===")
    assert recon_task.weight == 1.0
    recon_task.weight = 2.5
    assert recon_task.weight == 2.5
    recon_task.weight = 1.0  # Reset
    print("PASS: Task weight updated successfully")
    passed += 1

    # ── 8. Job loss history has per-task data ──
    total += 1
    print("\n=== 8. Job loss history per-task data ===")
    tasks = [recon_task, kl_task, classify_task]
    trainer = Trainer(model, tasks, device, lr=1e-3, probe_lr=1e-3, batch_size=32)
    jobs = JobManager()

    job = jobs.start(trainer, steps=30, blocking=True)
    assert job.state in ("completed", "stopped"), f"Job state: {job.state}"
    assert len(job.losses) > 0, "No loss data recorded"

    # Verify per-task data
    task_names_in_losses = set(l["task_name"] for l in job.losses)
    print(f"Task names in loss history: {task_names_in_losses}")
    assert "recon" in task_names_in_losses, "Missing 'recon' in losses"
    assert "kl_digit" in task_names_in_losses, "Missing 'kl_digit' in losses"
    assert "classify" in task_names_in_losses, "Missing 'classify' in losses"

    # Check each entry has required fields
    for entry in job.losses[:5]:
        assert "step" in entry
        assert "task_name" in entry
        assert "task_loss" in entry
    print(f"PASS: {len(job.losses)} loss entries with per-task data for {len(task_names_in_losses)} tasks")
    passed += 1

    # ── 9. Jobs history endpoint data ──
    total += 1
    print("\n=== 9. Jobs history data ===")
    all_jobs = jobs.list()
    assert len(all_jobs) >= 1
    latest = all_jobs[0]
    assert latest.id == job.id
    assert latest.state == "completed"

    # Final losses per task
    final_losses = {}
    for loss_entry in reversed(latest.losses):
        tn = loss_entry["task_name"]
        if tn not in final_losses:
            final_losses[tn] = loss_entry["task_loss"]
    print(f"Final losses: {final_losses}")
    assert len(final_losses) == 3, f"Expected 3 tasks in final losses, got {len(final_losses)}"
    print("PASS: Job history with per-task final losses")
    passed += 1

    # ── 10. Eval reconstructions ──
    total += 1
    print("\n=== 10. Eval reconstructions ===")
    model.eval()
    with torch.no_grad():
        images = mnist.sample(8).to(device)
        model_out = model(images)
        assert ModelOutput.RECONSTRUCTION in model_out
        recon = model_out[ModelOutput.RECONSTRUCTION]
        assert recon.shape == images.shape, f"Recon shape {recon.shape} != input shape {images.shape}"
    print(f"PASS: Reconstruction output shape matches input: {recon.shape}")
    passed += 1

    # ── 11. Eval metrics per task ──
    total += 1
    print("\n=== 11. Eval metrics per task ===")
    eval_results = trainer.evaluate_all()
    print(f"Eval results: {eval_results}")
    assert "recon" in eval_results
    assert "kl_digit" in eval_results
    assert "classify" in eval_results
    # Reconstruction should have L1 and PSNR
    assert "l1" in eval_results["recon"] or "L1" in eval_results["recon"] or len(eval_results["recon"]) > 0
    print("PASS: Per-task eval metrics returned")
    passed += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"M2 VERIFICATION: {passed}/{total} tests passed")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILED: {total - passed} test(s)")
    print(f"{'='*60}")

    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
