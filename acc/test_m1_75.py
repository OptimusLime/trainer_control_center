"""M1.75 Verification Script — Factor-Slot Autoencoder through the model-agnostic pipeline.

Usage:
    python -m acc.test_m1_75

This proves:
1. FactorSlotAutoencoder builds with factor groups and returns correct ModelOutput dict
2. Synthetic shapes dataset generates with float targets
3. RegressionTask attaches with latent_slice targeting factor groups
4. ClassificationTask attaches to shape factor slice (same class, different config)
5. ReconstructionTask attaches to the factor-slot model (has decoder)
6. All tasks train together through the same Trainer (model-agnostic)
7. JobManager tracks the training run identically to M1
8. Evaluation produces real metrics for all task types
9. Checkpoint save/load works with the factor-slot model
10. Metrics revert on checkpoint load
11. Gradient isolation: probe on position slice has zero gradient on shape dims
"""

import os
import shutil

import torch

from acc.factor_group import FactorGroup
from acc.factor_slot_autoencoder import FactorSlotAutoencoder
from acc.model_output import ModelOutput
from acc.generators.shapes import generate_shapes
from acc.tasks.classification import ClassificationTask
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.regression import RegressionTask
from acc.tasks.base import TaskError
from acc.eval_metric import EvalMetric
from acc.trainer import Trainer
from acc.jobs import JobManager
from acc.checkpoints import CheckpointStore
from acc.dataset import AccDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_checkpoint_dir = "./test_checkpoints_m175"
    if os.path.exists(test_checkpoint_dir):
        shutil.rmtree(test_checkpoint_dir)

    # ── 1. Build Factor-Slot model ──
    print("\n=== 1. Build FactorSlotAutoencoder ===")
    factor_groups = [
        FactorGroup("position", 0, 8),  # x_pos, y_pos regression
        FactorGroup("scale", 8, 12),  # scale regression
        FactorGroup("shape", 12, 16),  # shape_class classification
        FactorGroup("free", 16, 32),  # unstructured
    ]

    model = FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=factor_groups,
        backbone_channels=[16, 32, 64],
        embed_dim=32,
        image_size=64,
    )

    assert model.has_decoder, "FactorSlotAutoencoder should have decoder"
    assert model.latent_dim == 32, f"Expected latent_dim=32, got {model.latent_dim}"
    print(f"Model built. latent_dim={model.latent_dim}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Verify forward returns correct dict
    dummy = torch.randn(4, 1, 64, 64)
    out = model(dummy)
    assert isinstance(out, dict), f"forward() should return dict, got {type(out)}"
    assert ModelOutput.LATENT in out
    assert ModelOutput.SPATIAL in out
    assert ModelOutput.RECONSTRUCTION in out
    assert ModelOutput.FACTOR_SLICES in out
    assert out[ModelOutput.LATENT].shape == (4, 32)
    assert out[ModelOutput.RECONSTRUCTION].shape == (4, 1, 64, 64)

    # Verify LATENT is concat of FACTOR_SLICES
    z = out[ModelOutput.LATENT]
    slices = out[ModelOutput.FACTOR_SLICES]
    z_from_slices = torch.cat([slices[fg.name] for fg in factor_groups], dim=1)
    assert torch.allclose(z, z_from_slices), "LATENT should be concat of factor slices"
    print(f"Output keys: {list(out.keys())}")
    print(f"Factor slices: {list(slices.keys())}")
    print("PASS: FactorSlotAutoencoder builds and returns correct dict")

    # ── 2. Generate synthetic shapes dataset ──
    print("\n=== 2. Generate synthetic shapes ===")
    shapes = generate_shapes(n=2000, image_size=64)
    desc = shapes.describe()
    assert desc["target_type"] == "float"
    assert shapes.targets.shape == (2000, 4)
    print(f"Dataset: {desc}")
    print("PASS: Shapes dataset generated with float targets")

    # ── 3. Create classification dataset for shape_class ──
    # Shape class is the 4th column (index 3) — we need int targets for classification
    print("\n=== 3. Build classification dataset for shape_class ===")
    shape_classes = shapes.targets[:, 3].long()  # float -> int class labels
    shape_cls_dataset = AccDataset(shapes.images, shape_classes, name="shape_classes")
    assert shape_cls_dataset.target_type == "classes"
    assert shape_cls_dataset.num_classes == 3
    print(f"Shape classification dataset: {shape_cls_dataset.describe()}")
    print("PASS: Shape classification dataset created")

    # ── 4. Create position regression dataset ──
    # Position is columns 0-1 (x_pos, y_pos)
    print("\n=== 4. Build position regression dataset ===")
    position_targets = shapes.targets[:, :2]  # [N, 2]
    position_dataset = AccDataset(
        shapes.images, position_targets, name="position_targets"
    )
    assert position_dataset.target_type == "float"
    print(f"Position regression dataset: {position_dataset.describe()}")
    print("PASS: Position regression dataset created")

    # ── 5. Create scale regression dataset ──
    # Scale is column 2
    print("\n=== 5. Build scale regression dataset ===")
    scale_targets = shapes.targets[:, 2:3]  # [N, 1]
    scale_dataset = AccDataset(shapes.images, scale_targets, name="scale_targets")
    assert scale_dataset.target_type == "float"
    print(f"Scale regression dataset: {scale_dataset.describe()}")
    print("PASS: Scale regression dataset created")

    # ── 6. Attach sliced tasks ──
    print("\n=== 6. Attach tasks with latent_slice config ===")

    # Position regression on position slice z[0:8]
    position_task = RegressionTask(
        "position_probe",
        position_dataset,
        output_dim=2,
        latent_slice=factor_groups[0].latent_slice,  # (0, 8)
    )
    position_task.attach(model)
    # RegressionTask._build_head returns nn.Linear directly
    assert position_task.head.in_features == 8
    assert position_task.head.out_features == 2
    print(
        f"Position probe: Linear({position_task.head.in_features}, {position_task.head.out_features}) on z[0:8]"
    )

    # Scale regression on scale slice z[8:12]
    scale_task = RegressionTask(
        "scale_probe",
        scale_dataset,
        output_dim=1,
        latent_slice=factor_groups[1].latent_slice,  # (8, 12)
    )
    scale_task.attach(model)
    assert scale_task.head.in_features == 4
    assert scale_task.head.out_features == 1
    print(
        f"Scale probe: Linear({scale_task.head.in_features}, {scale_task.head.out_features}) on z[8:12]"
    )

    # Shape classification on shape slice z[12:16]
    shape_task = ClassificationTask(
        "shape_probe",
        shape_cls_dataset,
        latent_slice=factor_groups[2].latent_slice,  # (12, 16)
    )
    shape_task.attach(model)
    assert shape_task.head.in_features == 4
    assert shape_task.head.out_features == 3
    print(
        f"Shape probe: Linear({shape_task.head.in_features}, {shape_task.head.out_features}) on z[12:16]"
    )

    # Reconstruction on full model (no slice — reads RECONSTRUCTION key)
    recon_task = ReconstructionTask("recon", shapes)
    recon_task.attach(model)
    print("Reconstruction task attached")

    print("PASS: All tasks attached with correct slice targeting")

    # ── 7. Verify out-of-bounds slice rejection ──
    print("\n=== 7. Out-of-bounds slice rejection ===")
    try:
        bad_task = RegressionTask(
            "bad",
            position_dataset,
            output_dim=2,
            latent_slice=(0, 999),
        )
        bad_task.attach(model)
        assert False, "Should have raised TaskError"
    except TaskError as e:
        print(f"Correctly rejected: {e}")

    # RegressionTask on classification dataset should fail
    try:
        bad_task2 = RegressionTask(
            "bad2",
            shape_cls_dataset,  # int targets, not float
            output_dim=1,
        )
        bad_task2.attach(model)
        assert False, "Should have raised TaskError"
    except TaskError as e:
        print(f"Correctly rejected: {e}")
    print("PASS: Bad configs raise TaskError")

    # ── 8. Train ──
    print("\n=== 8. Train 200 steps ===")
    jobs = JobManager()
    ckpts = CheckpointStore(test_checkpoint_dir)
    all_tasks = [position_task, scale_task, shape_task, recon_task]
    trainer = Trainer(model, all_tasks, device, lr=1e-3, probe_lr=1e-3, batch_size=32)

    job = jobs.start(trainer, steps=200)
    assert job.state == "completed", f"Expected 'completed', got '{job.state}'"
    assert job.current_step == 200

    # Verify all tasks appear in loss history
    task_names_in_losses = set(l["task_name"] for l in job.losses)
    assert "position_probe" in task_names_in_losses
    assert "scale_probe" in task_names_in_losses
    assert "shape_probe" in task_names_in_losses
    assert "recon" in task_names_in_losses
    print(f"Job {job.id}: {job.state}, {job.current_step} steps")
    print(f"Last loss: {job.losses[-1]}")
    print(f"Task names in losses: {task_names_in_losses}")
    print("PASS: Training completed with all tasks in loss history")

    # ── 9. Evaluate ──
    print("\n=== 9. Evaluate all tasks ===")
    results = trainer.evaluate_all()

    assert "position_probe" in results
    assert EvalMetric.MAE in results["position_probe"]
    assert EvalMetric.MSE in results["position_probe"]

    assert "scale_probe" in results
    assert EvalMetric.MAE in results["scale_probe"]

    assert "shape_probe" in results
    assert EvalMetric.ACCURACY in results["shape_probe"]

    assert "recon" in results
    assert EvalMetric.PSNR in results["recon"]

    print(f"Position MAE: {results['position_probe'][EvalMetric.MAE]:.4f}")
    print(f"Scale MAE: {results['scale_probe'][EvalMetric.MAE]:.4f}")
    print(f"Shape accuracy: {results['shape_probe'][EvalMetric.ACCURACY]:.4f}")
    print(f"Recon PSNR: {results['recon'][EvalMetric.PSNR]:.1f}")
    print("PASS: All task types produce real eval metrics")

    # ── 10. Checkpoint save ──
    print("\n=== 10. Checkpoint save ===")
    cp1 = ckpts.save(model, trainer, tag="factor_slot_run1")
    cp1_path = os.path.join(test_checkpoint_dir, f"{cp1.id}.pt")
    assert os.path.exists(cp1_path)
    print(f"Checkpoint saved: {cp1.tag} ({cp1.id})")

    # Save metrics for comparison after revert
    pos_mae_at_cp1 = results["position_probe"][EvalMetric.MAE]
    shape_acc_at_cp1 = results["shape_probe"][EvalMetric.ACCURACY]

    # ── 11. Train more and verify metrics change ──
    print("\n=== 11. Train 200 more steps ===")
    job2 = jobs.start(trainer, steps=200, checkpoint_id=cp1.id)
    results2 = trainer.evaluate_all()
    print(
        f"Position MAE after 400 total steps: {results2['position_probe'][EvalMetric.MAE]:.4f}"
    )
    print(
        f"Shape accuracy after 400 total steps: {results2['shape_probe'][EvalMetric.ACCURACY]:.4f}"
    )
    print(f"Recon PSNR after 400 total steps: {results2['recon'][EvalMetric.PSNR]:.1f}")
    print("PASS: Additional training completed")

    # ── 12. Load checkpoint and verify revert ──
    print("\n=== 12. Load checkpoint and verify revert ===")
    ckpts.load(cp1.id, model, trainer)
    results_reverted = trainer.evaluate_all()

    pos_mae_reverted = results_reverted["position_probe"][EvalMetric.MAE]
    shape_acc_reverted = results_reverted["shape_probe"][EvalMetric.ACCURACY]

    print(f"Position MAE reverted: {pos_mae_reverted:.4f} (was {pos_mae_at_cp1:.4f})")
    print(f"Shape acc reverted: {shape_acc_reverted:.4f} (was {shape_acc_at_cp1:.4f})")

    pos_diff = abs(pos_mae_reverted - pos_mae_at_cp1)
    acc_diff = abs(shape_acc_reverted - shape_acc_at_cp1)
    assert pos_diff < 0.05, f"Position MAE diff after revert too large: {pos_diff:.4f}"
    assert acc_diff < 0.05, (
        f"Shape accuracy diff after revert too large: {acc_diff:.4f}"
    )
    print("PASS: Checkpoint load reverts metrics correctly")

    # ── 13. Gradient isolation test ──
    # Prove: when only the position probe fires, gradients flow ONLY through
    # the position factor head, not through other factor heads.
    print("\n=== 13. Gradient isolation test ===")
    model.zero_grad()
    x = torch.randn(4, 1, 64, 64, device=device)
    model_output = model(x)

    # Manually compute position probe loss
    pos_latent = model_output[ModelOutput.LATENT][:, 0:8]  # position slice
    pos_pred = position_task.head(pos_latent)
    dummy_target = torch.rand(4, 2, device=device)
    pos_loss = torch.nn.functional.mse_loss(pos_pred, dummy_target)
    pos_loss.backward()

    # Check: position factor head should have gradients
    pos_head = model.factor_heads["position"]
    pos_head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in pos_head.parameters()
    )
    assert pos_head_has_grad, "Position factor head should have gradients"

    # Check: shape factor head should have ZERO gradients
    # Because the loss only touches z[0:8] and the shape head produces z[12:16]
    shape_head = model.factor_heads["shape"]
    shape_head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in shape_head.parameters()
    )
    assert not shape_head_has_grad, (
        "Shape factor head should have ZERO gradients when only position probe fires"
    )

    # Check: scale factor head should also have ZERO gradients
    scale_head = model.factor_heads["scale"]
    scale_head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in scale_head.parameters()
    )
    assert not scale_head_has_grad, (
        "Scale factor head should have ZERO gradients when only position probe fires"
    )

    print("Position head has gradients: True")
    print("Shape head has gradients: False (isolated)")
    print("Scale head has gradients: False (isolated)")
    print("PASS: Gradient isolation confirmed — probes only affect their factor slice")

    # ── 14. Job history ──
    print("\n=== 14. Job history ===")
    all_jobs = jobs.list()
    assert len(all_jobs) == 2, f"Expected 2 jobs, got {len(all_jobs)}"
    for j in all_jobs:
        print(f"  {j.id}: {j.state}, {j.current_step} steps")
    print("PASS: Job history intact")

    # Cleanup
    shutil.rmtree(test_checkpoint_dir)

    print("\n" + "=" * 60)
    print("M1.75 PASSED: Factor-Slot Autoencoder trains through")
    print("the same model-agnostic pipeline as the simple Autoencoder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
