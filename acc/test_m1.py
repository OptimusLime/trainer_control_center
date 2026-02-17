"""M1 Verification Script — exercises the full training loop from code.

Usage:
    python -m acc.test_m1

This is the primary verification for M1. It proves:
1. Autoencoder builds, encodes, decodes, reports latent_dim
2. AccDataset loads MNIST with describe/sample
3. Tasks attach with compatibility checking (and fail gracefully)
4. Trainer runs multi-task training with round-robin
5. JobManager tracks training runs with full loss history
6. Evaluation produces real metrics
7. Checkpoints save/load to disk
8. Metrics revert on checkpoint load
9. Job history survives across runs
"""

import os
import shutil
import sys

import torch

from acc.autoencoder import Autoencoder
from acc.layers.conv_block import ConvBlock, ConvTransposeBlock
from acc.dataset import load_mnist
from acc.tasks.classification import ClassificationTask
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.base import TaskError
from acc.trainer import Trainer
from acc.jobs import JobManager
from acc.checkpoints import CheckpointStore
from acc.eval_metric import EvalMetric


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_checkpoint_dir = "./test_checkpoints"
    if os.path.exists(test_checkpoint_dir):
        shutil.rmtree(test_checkpoint_dir)

    # ── 1. Build model ──
    print("\n=== 1. Build model ===")
    encoder = [ConvBlock(ConvBlock.Config(1, 32, stride=2))]
    decoder = [
        ConvTransposeBlock(
            ConvTransposeBlock.Config(
                32, 1, stride=2, output_padding=1, output_activation="sigmoid"
            )
        )
    ]
    model = Autoencoder(encoder, decoder)
    print(model)
    assert model.has_decoder, "Model should have decoder"
    assert model.latent_dim > 0, f"latent_dim should be > 0, got {model.latent_dim}"
    print(f"latent_dim = {model.latent_dim}")
    print("PASS: Model builds correctly")

    # ── 2. Load dataset ──
    print("\n=== 2. Load dataset ===")
    mnist = load_mnist(image_size=64)
    assert len(mnist) == 60000, f"Expected 60000 images, got {len(mnist)}"
    assert mnist.sample(5).shape[0] == 5
    desc = mnist.describe()
    print(desc)
    assert desc["name"] == "mnist"
    assert desc["target_type"] == "classes"
    assert desc["image_shape"] == [1, 64, 64]
    print("PASS: MNIST loaded correctly")

    # ── 3. Create tasks + verify compatibility checking ──
    print("\n=== 3. Tasks + compatibility checking ===")
    classify = ClassificationTask("digits", mnist)
    recon = ReconstructionTask("recon", mnist)

    classify.attach(model)
    print(f"ClassificationTask attached: head = {classify.head}")
    recon.attach(model)
    print(f"ReconstructionTask attached: head = {recon.head}")

    # Verify incompatible attachment fails
    encoder_only = Autoencoder(
        [ConvBlock(ConvBlock.Config(1, 32, stride=2))], decoder_layers=None
    )
    assert not encoder_only.has_decoder, "Encoder-only model should not have decoder"

    try:
        bad_recon = ReconstructionTask("bad", mnist)
        bad_recon.attach(encoder_only)
        assert False, "Should have raised TaskError"
    except TaskError as e:
        print(f"Correctly rejected: {e}")

    print("PASS: Task compatibility checking works")

    # ── 4. Train ──
    print("\n=== 4. Train 500 steps ===")
    jobs = JobManager()
    ckpts = CheckpointStore(test_checkpoint_dir)
    trainer = Trainer(model, [classify, recon], device, lr=1e-3, probe_lr=1e-3)

    job = jobs.start(trainer, steps=500)
    assert job.state == "completed", f"Expected 'completed', got '{job.state}'"
    assert job.current_step == 500, f"Expected step 500, got {job.current_step}"
    assert len(job.losses) == 500, f"Expected 500 loss records, got {len(job.losses)}"

    # Verify both tasks appear in losses
    task_names_in_losses = set(l["task_name"] for l in job.losses)
    assert "digits" in task_names_in_losses, "digits task missing from loss history"
    assert "recon" in task_names_in_losses, "recon task missing from loss history"
    print(f"Job {job.id}: {job.state}, {job.current_step} steps")
    print(f"Last loss: {job.losses[-1]}")
    print("PASS: Training completed with loss history")

    # ── 5. Evaluate ──
    print("\n=== 5. Evaluate ===")
    results = trainer.evaluate_all()
    assert "digits" in results, "digits missing from eval results"
    assert EvalMetric.ACCURACY in results["digits"], "accuracy missing from digits eval"
    assert results["digits"][EvalMetric.ACCURACY] > 0.1, (
        f"Accuracy too low: {results['digits'][EvalMetric.ACCURACY]}"
    )
    assert "recon" in results, "recon missing from eval results"
    assert EvalMetric.PSNR in results["recon"], "psnr missing from recon eval"
    print(f"Accuracy: {results['digits'][EvalMetric.ACCURACY]:.4f}")
    print(f"PSNR: {results['recon'][EvalMetric.PSNR]:.1f}")
    print(f"L1: {results['recon'][EvalMetric.L1]:.4f}")
    print("PASS: Evaluation produces real metrics")

    # ── 6. Checkpoint ──
    print("\n=== 6. Checkpoint ===")
    cp1 = ckpts.save(model, trainer, tag="run1")
    cp1_path = os.path.join(test_checkpoint_dir, f"{cp1.id}.pt")
    assert os.path.exists(cp1_path), f"Checkpoint file not found: {cp1_path}"
    print(f"Checkpoint saved: {cp1.tag} ({cp1.id})")
    print("PASS: Checkpoint saved to disk")

    # Save metrics at this point for later comparison
    accuracy_at_cp1 = results["digits"][EvalMetric.ACCURACY]

    # ── 7. Train more ──
    print("\n=== 7. Train 500 more steps ===")
    job2 = jobs.start(trainer, steps=500, checkpoint_id=cp1.id)
    results2 = trainer.evaluate_all()
    print(f"Accuracy after 1000 total steps: {results2['digits'][EvalMetric.ACCURACY]:.4f}")
    print(f"PSNR after 1000 total steps: {results2['recon'][EvalMetric.PSNR]:.1f}")

    # Accuracy should generally improve (or at least not collapse)
    # Being lenient here — 500 more steps on a tiny model might not always improve
    assert results2["digits"][EvalMetric.ACCURACY] > 0.1, (
        "Accuracy collapsed after more training"
    )
    print("PASS: Additional training completed")

    # ── 8. Load old checkpoint — metrics should revert ──
    print("\n=== 8. Load checkpoint and verify revert ===")
    ckpts.load(cp1.id, model, trainer)
    results_reverted = trainer.evaluate_all()
    print(f"Accuracy after revert: {results_reverted['digits'][EvalMetric.ACCURACY]:.4f}")
    print(f"Expected ~{accuracy_at_cp1:.4f}")

    # Should be close to the accuracy at checkpoint time
    diff = abs(results_reverted["digits"][EvalMetric.ACCURACY] - accuracy_at_cp1)
    assert diff < 0.05, f"Accuracy diff after revert too large: {diff:.4f}"
    print("PASS: Checkpoint load reverts metrics correctly")

    # ── 9. Verify job history survives ──
    print("\n=== 9. Job history ===")
    all_jobs = jobs.list()
    assert len(all_jobs) == 2, f"Expected 2 jobs, got {len(all_jobs)}"
    assert all_jobs[0].state == "completed"
    assert all_jobs[1].state == "completed"
    print(f"Job history: {len(all_jobs)} jobs")
    for j in all_jobs:
        print(f"  {j.id}: {j.state}, {j.current_step} steps")
    print("PASS: Job history persists")

    # Cleanup
    shutil.rmtree(test_checkpoint_dir)

    print("\n" + "=" * 50)
    print("M1 PASSED: full loop works from code.")
    print("=" * 50)


if __name__ == "__main__":
    main()
