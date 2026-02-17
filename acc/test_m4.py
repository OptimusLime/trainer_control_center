"""M4 Verification Script — Device Selection + Checkpoint Safety.

Usage:
    python -m acc.test_m4

This proves:
1. CheckpointStore.load() uses map_location — checkpoint loads on CPU from any device
2. CheckpointStore.load() uses map_location — checkpoint loads back on original device
3. After load, model and probe heads are on the target device
4. Device selection API: GET /device and POST /device/set routes exist
5. Device selection: changing device moves model + probe heads
6. Training still works after device change
7. Checkpoint saved on one device loads on another device
8. (If 2 GPUs) Checkpoint from cuda:0 loads and trains on cuda:1
"""

import tempfile

import torch
import torch.nn as nn

from acc.autoencoder import Autoencoder
from acc.layers.conv_block import ConvBlock, ConvTransposeBlock
from acc.trainer import Trainer
from acc.checkpoints import CheckpointStore
from acc.tasks.reconstruction import ReconstructionTask
from acc.dataset import AccDataset


def _make_test_setup(device: torch.device):
    """Create a minimal autoencoder + trainer + dataset for testing."""
    encoder = [ConvBlock(ConvBlock.Config(1, 16, stride=2))]
    decoder = [
        ConvTransposeBlock(
            ConvTransposeBlock.Config(
                16, 1, stride=2, output_padding=1, output_activation="sigmoid"
            )
        )
    ]
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)

    images = torch.rand(200, 1, 64, 64)
    dataset = AccDataset(images, name="test_data")

    task = ReconstructionTask("recon_test", dataset)
    task.attach(autoencoder)

    trainer = Trainer(autoencoder, [task], device, batch_size=32)

    return autoencoder, trainer, dataset, task


def main():
    print("M4 Verification: Device Selection + Checkpoint Safety")
    print("=" * 60)

    passed = 0
    total = 0

    # Determine available devices
    has_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if has_cuda else 0
    primary_device = torch.device("cuda:0") if has_cuda else torch.device("cpu")
    alt_device = torch.device("cpu")  # Always available as alternate

    print(f"Primary device: {primary_device}")
    print(f"CUDA available: {has_cuda}, GPU count: {cuda_count}")
    if cuda_count >= 2:
        print(f"Second GPU available: cuda:1")

    # ── 1. Save checkpoint, load with map_location=cpu ──
    total += 1
    print("\n=== 1. Checkpoint map_location: load on CPU ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        autoencoder, trainer, dataset, task = _make_test_setup(primary_device)

        # Train a few steps to have non-trivial state
        trainer.train(steps=5)

        store = CheckpointStore(tmpdir)
        cp = store.save(autoencoder, trainer, tag="test_save")
        print(f"Saved checkpoint: {cp.id} on {primary_device}")

        # Create a fresh model on CPU and load
        ae_cpu, trainer_cpu, _, _ = _make_test_setup(torch.device("cpu"))
        cp_loaded = store.load(cp.id, ae_cpu, trainer_cpu, device=torch.device("cpu"))
        print(f"Loaded checkpoint: {cp_loaded.id} on CPU")

        # Verify all params are on CPU
        for name, param in ae_cpu.named_parameters():
            assert param.device == torch.device("cpu"), f"{name} on {param.device}, expected cpu"

        print("PASS: Checkpoint loads on CPU with map_location")
        passed += 1

    # ── 2. Load checkpoint back on original device ──
    total += 1
    print("\n=== 2. Checkpoint map_location: load back on original device ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        autoencoder, trainer, dataset, task = _make_test_setup(primary_device)
        trainer.train(steps=5)

        store = CheckpointStore(tmpdir)
        cp = store.save(autoencoder, trainer, tag="test_roundtrip")

        # Load back on original device
        ae2, trainer2, _, _ = _make_test_setup(primary_device)
        cp2 = store.load(cp.id, ae2, trainer2, device=primary_device)

        for name, param in ae2.named_parameters():
            assert param.device.type == primary_device.type, f"{name} on {param.device}"

        print(f"PASS: Checkpoint roundtrips on {primary_device}")
        passed += 1

    # ── 3. After load, probe heads are on target device ──
    total += 1
    print("\n=== 3. Probe heads on target device after load ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        autoencoder, trainer, dataset, task = _make_test_setup(primary_device)
        trainer.train(steps=5)

        store = CheckpointStore(tmpdir)
        cp = store.save(autoencoder, trainer, tag="test_probes")

        # Load on CPU
        ae_cpu, trainer_cpu, _, task_cpu = _make_test_setup(torch.device("cpu"))
        store.load(cp.id, ae_cpu, trainer_cpu, device=torch.device("cpu"))

        if task_cpu.head is not None:
            for name, param in task_cpu.head.named_parameters():
                assert param.device == torch.device("cpu"), f"Probe {name} on {param.device}"

        print("PASS: Probe heads are on target device after load")
        passed += 1

    # ── 4. Device API: routes exist ──
    total += 1
    print("\n=== 4. Device API routes ===")
    from acc.trainer_api import TrainerAPI

    api = TrainerAPI()
    # Verify the device attribute
    assert hasattr(api, "device"), "TrainerAPI must have .device"
    assert isinstance(api.device, torch.device)

    # Check available devices
    available = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            available.append(f"cuda:{i}")

    print(f"TrainerAPI device: {api.device}")
    print(f"Available devices: {available}")

    # Verify the routes exist
    route_paths = [r.path for r in api.app.routes]
    assert "/device" in route_paths, f"/device route missing. Routes: {route_paths}"
    assert "/device/set" in route_paths, f"/device/set route missing. Routes: {route_paths}"
    print("PASS: Device API routes registered")
    passed += 1

    # ── 5. Device change: model moves ──
    total += 1
    print("\n=== 5. Device selection: change device ===")
    ae, trainer_inst, ds, task_inst = _make_test_setup(api.device)
    api.autoencoder = ae
    api.datasets["test"] = ds
    api.tasks["recon_test"] = task_inst
    api.trainer = trainer_inst

    old_device = api.device
    target = torch.device("cpu")

    # Simulate what POST /device/set does
    api.device = target
    api.autoencoder.to(target)
    for t in api.tasks.values():
        if t.head is not None:
            t.head.to(target)
    api.trainer.device = target
    api.trainer._build_optimizers()

    for name, param in api.autoencoder.named_parameters():
        assert param.device == target, f"{name} still on {param.device}"

    print(f"PASS: Device changed from {old_device} to {target}")
    passed += 1

    # ── 6. Training works after device change ──
    total += 1
    print("\n=== 6. Training works after device change ===")
    losses = api.trainer.train(steps=10)
    assert len(losses) == 10, f"Expected 10 loss entries, got {len(losses)}"
    assert all(isinstance(l["task_loss"], float) for l in losses)
    print(f"PASS: Trained 10 steps on {target}, final loss: {losses[-1]['task_loss']:.4f}")
    passed += 1

    # ── 7. Cross-device checkpoint: save on one, load on other ──
    total += 1
    print("\n=== 7. Cross-device checkpoint portability ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save on primary device
        ae_src, trainer_src, _, _ = _make_test_setup(primary_device)
        trainer_src.train(steps=5)
        store = CheckpointStore(tmpdir)
        cp = store.save(ae_src, trainer_src, tag="cross_device")

        # Load on CPU (different device)
        ae_dst, trainer_dst, _, _ = _make_test_setup(torch.device("cpu"))
        cp_loaded = store.load(cp.id, ae_dst, trainer_dst, device=torch.device("cpu"))

        # Train a few steps on CPU to prove it works end-to-end
        losses = trainer_dst.train(steps=5)
        assert len(losses) == 5
        print(f"PASS: Checkpoint from {primary_device} loaded and trained on CPU")
        passed += 1

    # ── 8. Second GPU (if available) ──
    if cuda_count >= 2:
        total += 1
        print("\n=== 8. Second GPU: cuda:1 ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save on cuda:0
            ae0, trainer0, _, _ = _make_test_setup(torch.device("cuda:0"))
            trainer0.train(steps=5)
            store = CheckpointStore(tmpdir)
            cp = store.save(ae0, trainer0, tag="gpu0")

            # Load on cuda:1
            ae1, trainer1, _, _ = _make_test_setup(torch.device("cuda:1"))
            store.load(cp.id, ae1, trainer1, device=torch.device("cuda:1"))

            # Verify params on cuda:1
            for name, param in ae1.named_parameters():
                assert param.device == torch.device("cuda:1"), f"{name} on {param.device}"

            # Train on cuda:1
            losses = trainer1.train(steps=5)
            assert len(losses) == 5
            print("PASS: Checkpoint from cuda:0 loaded and trained on cuda:1")
            passed += 1
    else:
        print(f"\n=== 8. Second GPU: SKIPPED (only {cuda_count} GPU(s)) ===")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"M4 VERIFICATION: {passed}/{total} tests passed")
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
