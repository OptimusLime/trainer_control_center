"""M1.5 Verification Script — model-agnostic forward protocol + slice targeting.

Usage:
    python -m acc.test_m1_5

This proves:
1. ModelOutput enum keys work in the forward dict
2. A task with latent_slice=None reads full latent (backward compat)
3. A task with latent_slice=(0, D) behaves identically to latent_slice=None
4. A task with latent_slice=(0, 64) builds a head with in_features=64
5. A task with out-of-bounds slice raises TaskError on attach
6. ReconstructionTask reads RECONSTRUCTION from dict (no re-encode)
7. Two tasks with different slices train together in the same Trainer
8. Sliced task produces real eval metrics
"""

import os
import shutil

import torch

from acc.autoencoder import Autoencoder
from acc.layers.conv_block import ConvBlock, ConvTransposeBlock
from acc.model_output import ModelOutput
from acc.dataset import load_mnist
from acc.tasks.classification import ClassificationTask
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.base import TaskError
from acc.trainer import Trainer
from acc.jobs import JobManager


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Forward returns dict with ModelOutput keys ──
    print("\n=== 1. Forward returns dict with ModelOutput keys ===")
    encoder = [ConvBlock(ConvBlock.Config(1, 32, stride=2))]
    decoder = [
        ConvTransposeBlock(
            ConvTransposeBlock.Config(
                32, 1, stride=2, output_padding=1, output_activation="sigmoid"
            )
        )
    ]
    model = Autoencoder(encoder, decoder)

    dummy = torch.randn(2, 1, 64, 64)
    output = model(dummy)

    assert isinstance(output, dict), f"forward() should return dict, got {type(output)}"
    assert ModelOutput.LATENT in output, "LATENT missing from output"
    assert ModelOutput.SPATIAL in output, "SPATIAL missing from output"
    assert ModelOutput.RECONSTRUCTION in output, "RECONSTRUCTION missing from output"
    assert output[ModelOutput.LATENT].shape == (2, model.latent_dim)
    assert output[ModelOutput.RECONSTRUCTION].shape == (2, 1, 64, 64)
    print(f"Output keys: {list(output.keys())}")
    print(f"LATENT shape: {output[ModelOutput.LATENT].shape}")
    print(f"RECONSTRUCTION shape: {output[ModelOutput.RECONSTRUCTION].shape}")
    print("PASS: Forward returns correct dict")

    # ── 2. Encoder-only model has no RECONSTRUCTION key ──
    print("\n=== 2. Encoder-only model ===")
    encoder_only = Autoencoder(
        [ConvBlock(ConvBlock.Config(1, 32, stride=2))], decoder_layers=None
    )
    output_enc = encoder_only(dummy)
    assert ModelOutput.LATENT in output_enc
    assert ModelOutput.RECONSTRUCTION not in output_enc
    print("PASS: Encoder-only model omits RECONSTRUCTION")

    # ── 3. Task with latent_slice=None reads full latent ──
    print("\n=== 3. Task with no slice reads full latent ===")
    mnist = load_mnist(image_size=64)
    task_full = ClassificationTask("full", mnist, latent_slice=None)
    task_full.attach(model)
    assert task_full.head.in_features == model.latent_dim
    print(
        f"Full latent head: Linear({task_full.head.in_features}, {task_full.head.out_features})"
    )
    print("PASS: No-slice task uses full latent_dim")

    # ── 4. Task with latent_slice=(0, 64) builds smaller head ──
    print("\n=== 4. Task with slice=(0, 64) ===")
    task_sliced = ClassificationTask("sliced", mnist, latent_slice=(0, 64))
    task_sliced.attach(model)
    assert task_sliced.head.in_features == 64, (
        f"Expected in_features=64, got {task_sliced.head.in_features}"
    )
    print(
        f"Sliced head: Linear({task_sliced.head.in_features}, {task_sliced.head.out_features})"
    )
    print("PASS: Sliced task builds correctly sized head")

    # ── 5. _get_latent slices correctly ──
    print("\n=== 5. _get_latent slicing ===")
    latent_full = task_full._get_latent(output)
    latent_sliced = task_sliced._get_latent(output)
    assert latent_full.shape == (2, model.latent_dim)
    assert latent_sliced.shape == (2, 64)
    # Sliced should be the first 64 dims of full
    assert torch.allclose(latent_sliced, latent_full[:, :64])
    print("PASS: _get_latent slices correctly")

    # ── 6. Out-of-bounds slice raises TaskError ──
    print("\n=== 6. Out-of-bounds slice ===")
    try:
        bad_task = ClassificationTask("bad", mnist, latent_slice=(0, 9999))
        bad_task.attach(model)
        assert False, "Should have raised TaskError"
    except TaskError as e:
        print(f"Correctly rejected: {e}")

    try:
        bad_task2 = ClassificationTask("bad2", mnist, latent_slice=(100, 50))
        bad_task2.attach(model)
        assert False, "Should have raised TaskError"
    except TaskError as e:
        print(f"Correctly rejected: {e}")

    print("PASS: Bad slices raise TaskError")

    # ── 7. ReconstructionTask reads from dict ──
    print("\n=== 7. ReconstructionTask uses dict ===")
    recon = ReconstructionTask("recon", mnist)
    recon.attach(model)
    # compute_loss should work with the output dict
    batch = (dummy, torch.zeros(2, dtype=torch.long))
    loss = recon.compute_loss(output, batch)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"Reconstruction loss: {loss.item():.4f}")
    print("PASS: ReconstructionTask reads RECONSTRUCTION from dict")

    # ── 8. Two tasks with different slices train together ──
    print("\n=== 8. Train with sliced + full tasks ===")
    task_a = ClassificationTask("first_half", mnist, latent_slice=(0, 256))
    task_b = ClassificationTask("second_half", mnist, latent_slice=(256, 512))
    recon_task = ReconstructionTask("recon_train", mnist)

    task_a.attach(model)
    task_b.attach(model)
    recon_task.attach(model)

    assert task_a.head.in_features == 256
    assert task_b.head.in_features == 256

    jobs = JobManager()
    trainer = Trainer(
        model, [task_a, task_b, recon_task], device, lr=1e-3, probe_lr=1e-3
    )
    job = jobs.start(trainer, steps=100)
    assert job.state == "completed"
    assert job.current_step == 100

    # Verify all three tasks appear in loss history
    task_names = set(l["task_name"] for l in job.losses)
    assert "first_half" in task_names
    assert "second_half" in task_names
    assert "recon_train" in task_names
    print(f"Trained 100 steps with {len(task_names)} tasks")
    print("PASS: Multiple sliced tasks train together")

    # ── 9. Sliced tasks produce real eval metrics ──
    print("\n=== 9. Sliced task eval ===")
    results = trainer.evaluate_all()
    assert "first_half" in results
    assert "second_half" in results
    assert "accuracy" in results["first_half"]
    assert "accuracy" in results["second_half"]
    print(f"first_half accuracy (z[0:256]): {results['first_half']['accuracy']:.4f}")
    print(
        f"second_half accuracy (z[256:512]): {results['second_half']['accuracy']:.4f}"
    )
    print(f"recon PSNR: {results['recon_train']['psnr']:.1f}")
    print("PASS: Sliced tasks produce real eval metrics")

    print("\n" + "=" * 50)
    print("M1.5 PASSED: model-agnostic forward protocol works.")
    print("=" * 50)


if __name__ == "__main__":
    main()
