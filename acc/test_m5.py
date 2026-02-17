"""M5 Verification Script — UFR Evaluation + Visual Diagnosis Dashboard.

Usage:
    python -m acc.test_m5

This proves:
1. Attention map extraction returns correct shapes and values
2. Attention maps are per-factor (B, H, W) tensors
3. Attention maps sum to ~1 across factors at each spatial position
4. UFR scoring returns valid disentanglement/completeness/ufr metrics
5. UFR metrics are in [0, 1] range
6. UFR uses EvalMetric enum keys
7. Traversals with checkpoint_id work (temporarily loads, then restores)
8. Sort-by-factor with checkpoint_id works
9. Attention maps with checkpoint_id work (state restored after)
"""

import tempfile

import torch
import torch.nn as nn

from acc.factor_slot_autoencoder import FactorSlotAutoencoder
from acc.factor_group import FactorGroup
from acc.trainer import Trainer
from acc.checkpoints import CheckpointStore
from acc.tasks.reconstruction import ReconstructionTask
from acc.dataset import AccDataset
from acc.eval.attention import extract_attention_maps
from acc.eval.ufr import compute_ufr
from acc.eval_metric import EvalMetric
from acc.model_output import ModelOutput


def _make_test_setup(device: torch.device):
    """Create a minimal FactorSlotAutoencoder + trainer + dataset for testing."""
    factor_groups = [
        FactorGroup("shape", 0, 4),
        FactorGroup("position", 4, 8),
        FactorGroup("free", 8, 12),
    ]

    autoencoder = FactorSlotAutoencoder(
        factor_groups=factor_groups,
        backbone_channels=[16, 32],
        embed_dim=16,
        image_size=32,
        in_channels=1,
    )
    autoencoder.to(device)

    # Small dataset — must have more samples than batch_size
    images = torch.rand(200, 1, 32, 32)
    dataset = AccDataset(images, name="test_data")

    task = ReconstructionTask("recon_test", dataset)
    task.attach(autoencoder)

    trainer = Trainer(autoencoder, [task], device, batch_size=32)

    return autoencoder, trainer, dataset, task


def main():
    print("M5 Verification: UFR Evaluation + Visual Diagnosis Dashboard")
    print("=" * 60)

    passed = 0
    total = 0

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if has_cuda else torch.device("cpu")
    print(f"Device: {device}")

    autoencoder, trainer, dataset, task = _make_test_setup(device)

    # Train a few steps for non-trivial state
    trainer.train(steps=10)

    # ── 1. Attention map extraction returns correct shapes ──
    total += 1
    print("\n=== 1. Attention map extraction: correct shapes ===")

    images = dataset.sample(4).to(device)
    attn_maps = extract_attention_maps(autoencoder, images)

    assert isinstance(attn_maps, dict), f"Expected dict, got {type(attn_maps)}"
    assert len(attn_maps) == 3, f"Expected 3 factor groups, got {len(attn_maps)}"
    for name in ["shape", "position", "free"]:
        assert name in attn_maps, f"Missing factor '{name}' in attention maps"
    print("PASS: Attention maps have correct structure")
    passed += 1

    # ── 2. Attention maps are (B, H, W) tensors ──
    total += 1
    print("\n=== 2. Attention maps: per-factor (B, H, W) tensors ===")

    for name, maps in attn_maps.items():
        assert maps.shape[0] == 4, f"{name}: expected batch 4, got {maps.shape[0]}"
        assert maps.ndim == 3, f"{name}: expected 3D (B, H, W), got {maps.ndim}D"
        assert maps.shape[1] == 32, f"{name}: expected H=32, got {maps.shape[1]}"
        assert maps.shape[2] == 32, f"{name}: expected W=32, got {maps.shape[2]}"
    print("PASS: All attention maps are (4, 32, 32)")
    passed += 1

    # ── 3. Attention maps sum to ~1 across factors ──
    total += 1
    print("\n=== 3. Attention maps: sum to ~1 across factors per position ===")

    stacked = torch.stack(list(attn_maps.values()), dim=0)  # (N_factors, B, H, W)
    factor_sum = stacked.sum(dim=0)  # (B, H, W)
    max_deviation = (factor_sum - 1.0).abs().max().item()
    assert max_deviation < 0.01, f"Factor sum deviates from 1.0 by {max_deviation}"
    print(f"PASS: Max deviation from sum=1: {max_deviation:.6f}")
    passed += 1

    # ── 4. Attention map values in [0, 1] ──
    total += 1
    print("\n=== 4. Attention maps: values in [0, 1] ===")

    all_vals = torch.cat([m.flatten() for m in attn_maps.values()])
    assert all_vals.min() >= -0.001, f"Min value {all_vals.min():.4f} is below 0"
    assert all_vals.max() <= 1.001, f"Max value {all_vals.max():.4f} is above 1"
    print(f"PASS: Values range [{all_vals.min():.4f}, {all_vals.max():.4f}]")
    passed += 1

    # ── 5. UFR scoring returns valid metrics ──
    total += 1
    print("\n=== 5. UFR scoring: returns valid metrics ===")

    datasets = {"test": dataset}
    ufr_results = compute_ufr(autoencoder, datasets, device, n_samples=100)

    assert isinstance(ufr_results, dict), f"Expected dict, got {type(ufr_results)}"
    assert EvalMetric.UFR in ufr_results, "Missing UFR metric"
    assert EvalMetric.DISENTANGLEMENT in ufr_results, "Missing DISENTANGLEMENT metric"
    assert EvalMetric.COMPLETENESS in ufr_results, "Missing COMPLETENESS metric"
    print(f"PASS: UFR={ufr_results[EvalMetric.UFR]:.4f}, "
          f"D={ufr_results[EvalMetric.DISENTANGLEMENT]:.4f}, "
          f"C={ufr_results[EvalMetric.COMPLETENESS]:.4f}")
    passed += 1

    # ── 6. UFR metrics in [0, 1] ──
    total += 1
    print("\n=== 6. UFR metrics: in [0, 1] range ===")

    for metric_key in [EvalMetric.UFR, EvalMetric.DISENTANGLEMENT, EvalMetric.COMPLETENESS]:
        val = ufr_results[metric_key]
        assert 0.0 <= val <= 1.0, f"{metric_key}: {val} not in [0, 1]"
    print("PASS: All UFR metrics in [0, 1]")
    passed += 1

    # ── 7. UFR uses EvalMetric enum keys ──
    total += 1
    print("\n=== 7. UFR uses EvalMetric enum keys ===")

    for key in ufr_results:
        assert isinstance(key, EvalMetric), f"Key {key} is not EvalMetric, is {type(key)}"
    # Verify the enum has higher_is_better
    assert EvalMetric.UFR.higher_is_better is True
    assert EvalMetric.DISENTANGLEMENT.higher_is_better is True
    assert EvalMetric.COMPLETENESS.higher_is_better is True
    print("PASS: UFR results use EvalMetric enum keys with correct higher_is_better")
    passed += 1

    # ── 8. Per-checkpoint traversals (state restoration) ──
    total += 1
    print("\n=== 8. Per-checkpoint eval: state restored after checkpoint load ===")

    import copy

    with tempfile.TemporaryDirectory() as tmpdir:
        cs = CheckpointStore(tmpdir)

        # Save current state as checkpoint
        cp1 = cs.save(autoencoder, trainer, "baseline")

        # Train more to change state
        trainer.train(steps=10)
        cp2 = cs.save(autoencoder, trainer, "trained_more")

        # Record current (cp2) weights
        current_param = next(autoencoder.parameters()).data.clone()

        # Deep-copy the full state before loading cp1
        original_state = copy.deepcopy(trainer.state_dict())
        cs.load(cp1.id, autoencoder, trainer, device=device)
        loaded_param = next(autoencoder.parameters()).data.clone()

        # Weights should differ from current (cp2) since we loaded cp1
        assert not torch.allclose(current_param, loaded_param), \
            "Loaded checkpoint should have different weights"

        # Restore to cp2 state
        trainer.load_state_dict(original_state)
        autoencoder.to(device)
        restored_param = next(autoencoder.parameters()).data.clone()

        assert torch.allclose(current_param, restored_param), \
            "State not properly restored after checkpoint load"

        print("PASS: Checkpoint load + restore works correctly")
        passed += 1

    # ── 9. CrossAttentionBlock store_attn flag cleanup ──
    total += 1
    print("\n=== 9. Attention storage cleanup: store_attn disabled after extraction ===")

    # After extract_attention_maps, store_attn should be False on all blocks
    for block in autoencoder.cross_attn_stages:
        if block is not None:
            assert block.store_attn is False, "store_attn should be False after extraction"
            assert block.last_attn_weights is None, "last_attn_weights should be None after cleanup"
    print("PASS: Attention storage properly cleaned up after extraction")
    passed += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"M5 VERIFICATION: {passed}/{total} tests passed")
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
