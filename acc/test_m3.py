"""M3 Verification Script — Generator Hot-Reload + Dataset Dashboard.

Usage:
    python -m acc.test_m3

This proves:
1. DatasetGenerator base class works (describe, generate)
2. GeneratorRegistry discovers built-in generators (thickness, slant, shapes)
3. GeneratorRegistry hot-reload: write a dummy generator file, verify it appears
4. GeneratorRegistry error handling: syntax error doesn't crash
5. GeneratorRegistry deletion: remove file, generator disappears
6. Generate a dataset via GeneratorRegistry and verify it's a valid AccDataset
7. Generated dataset has correct shape, targets, and sample() works
8. Existing generate_thickness() and generate_slant() functions still work (backward compat)
9. Multiple generators produce different datasets that can coexist
"""

import os
import time

import torch

from acc.dataset import AccDataset
from acc.generators.base import DatasetGenerator
from acc.generators.registry import GeneratorRegistry
from acc.generators.thickness import generate_thickness, ThicknessGenerator
from acc.generators.slant import generate_slant, SlantGenerator
from acc.generators.shapes import generate_shapes, ShapesGenerator


def main():
    print("M3 Verification: Generator Hot-Reload + Dataset Dashboard")
    print("=" * 60)

    passed = 0
    total = 0

    # ── 1. DatasetGenerator base class ──
    total += 1
    print("\n=== 1. DatasetGenerator base class ===")
    gen = ThicknessGenerator()
    desc = gen.describe()
    assert desc["name"] == "thickness"
    assert "parameters" in desc
    assert "n" in desc["parameters"]
    assert "image_size" in desc["parameters"]
    print(f"ThicknessGenerator.describe(): {desc}")
    print("PASS: DatasetGenerator base class works")
    passed += 1

    # ── 2. GeneratorRegistry discovers built-in generators ──
    total += 1
    print("\n=== 2. GeneratorRegistry discovers built-in generators ===")
    registry = GeneratorRegistry()
    generators = registry.list()
    gen_names = [g["name"] for g in generators]
    print(f"Discovered: {gen_names}")

    expected = {"thickness", "slant", "shapes"}
    found = expected.intersection(set(gen_names))
    assert found == expected, f"Missing generators: {expected - found}"
    print(f"PASS: All {len(expected)} built-in generators discovered")
    passed += 1

    # ── 3. GeneratorRegistry.get() returns instance ──
    total += 1
    print("\n=== 3. GeneratorRegistry.get() returns instance ===")
    gen_inst = registry.get("thickness")
    assert gen_inst is not None, "Expected ThicknessGenerator instance"
    assert isinstance(gen_inst, DatasetGenerator)
    assert gen_inst.name == "thickness"
    none_inst = registry.get("nonexistent")
    assert none_inst is None
    print("PASS: get() returns instance for valid name, None for invalid")
    passed += 1

    # ── 4. GeneratorRegistry hot-reload: add a new generator file ──
    total += 1
    print("\n=== 4. GeneratorRegistry hot-reload: new generator file ===")
    gen_dir = registry.generators_dir
    dummy_path = os.path.join(gen_dir, "dummy_test_gen.py")

    dummy_code = '''
"""Dummy test generator for M3 verification."""

import torch
from acc.dataset import AccDataset
from acc.generators.base import DatasetGenerator


class DummyTestGenerator(DatasetGenerator):
    """A dummy generator for testing."""

    name = "dummy_test"
    description = "Generates random noise images for testing"
    parameters = {
        "n": {"type": "int", "default": 100, "description": "Number of images"},
    }

    def generate(self, **params):
        n = int(params.get("n", 100))
        images = torch.rand(n, 1, 16, 16)
        targets = torch.rand(n)
        return AccDataset(images, targets, name="dummy_test_data")
'''
    try:
        with open(dummy_path, "w") as f:
            f.write(dummy_code)

        registry._check_for_changes()
        time.sleep(0.1)

        gen_names_after = [g["name"] for g in registry.list()]
        assert "dummy_test" in gen_names_after, f"dummy_test not found. Got: {gen_names_after}"
        print("PASS: DummyTestGenerator appeared in registry after file creation")
        passed += 1
    finally:
        pass

    # ── 5. GeneratorRegistry error handling: syntax error ──
    total += 1
    print("\n=== 5. GeneratorRegistry error handling: syntax error ===")
    bad_path = os.path.join(gen_dir, "bad_syntax_gen.py")
    try:
        with open(bad_path, "w") as f:
            f.write("def this is broken syntax {{\n")

        count_before = len(registry.list())
        registry._check_for_changes()
        count_after = len(registry.list())
        print(f"Generators before: {count_before}, after: {count_after}")
        print("PASS: Syntax error caught, registry still functional")
        passed += 1
    finally:
        if os.path.exists(bad_path):
            os.remove(bad_path)
            registry._check_for_changes()

    # ── 6. GeneratorRegistry deletion: remove file ──
    total += 1
    print("\n=== 6. GeneratorRegistry deletion: remove file ===")
    assert "dummy_test" in [g["name"] for g in registry.list()]
    os.remove(dummy_path)
    registry._check_for_changes()
    assert "dummy_test" not in [g["name"] for g in registry.list()], "dummy_test should be gone"
    print("PASS: DummyTestGenerator removed from registry after file deletion")
    passed += 1

    # ── 7. Generate a dataset via registry ──
    total += 1
    print("\n=== 7. Generate dataset via registry ===")
    gen_inst = registry.get("thickness")
    assert gen_inst is not None
    dataset = gen_inst.generate(n=200, image_size=32)
    assert isinstance(dataset, AccDataset)
    assert len(dataset) == 200
    assert dataset.images.shape == (200, 1, 32, 32)
    assert dataset.targets is not None
    assert dataset.target_type == "float"
    print(f"Generated thickness dataset: {dataset.describe()}")
    print("PASS: Generated valid AccDataset via registry")
    passed += 1

    # ── 8. Dataset sample() works ──
    total += 1
    print("\n=== 8. Dataset sample() works ===")
    samples = dataset.sample(8)
    assert samples.shape == (8, 1, 32, 32)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0
    print(f"Sampled 8 images, shape: {samples.shape}, range: [{samples.min():.2f}, {samples.max():.2f}]")
    print("PASS: sample() returns valid image tensors")
    passed += 1

    # ── 9. Backward compatibility: standalone functions ──
    total += 1
    print("\n=== 9. Backward compatibility: standalone functions ===")
    ds_thickness = generate_thickness(n=100, image_size=32)
    assert isinstance(ds_thickness, AccDataset)
    assert len(ds_thickness) == 100
    assert ds_thickness.name == "thickness_synth"

    ds_slant = generate_slant(n=100, image_size=32)
    assert isinstance(ds_slant, AccDataset)
    assert len(ds_slant) == 100
    assert ds_slant.name == "slant_synth"

    ds_shapes = generate_shapes(n=100, image_size=32)
    assert isinstance(ds_shapes, AccDataset)
    assert len(ds_shapes) == 100
    assert ds_shapes.name == "synthetic_shapes"

    print("PASS: generate_thickness(), generate_slant(), generate_shapes() all work")
    passed += 1

    # ── 10. Multiple generators produce different datasets ──
    total += 1
    print("\n=== 10. Multiple generators coexist ===")
    datasets = {}
    for gen_name in ["thickness", "slant", "shapes"]:
        gen = registry.get(gen_name)
        assert gen is not None
        ds = gen.generate(n=50, image_size=32)
        datasets[gen_name] = ds
        print(f"  {gen_name}: {ds.name}, {len(ds)} images, shape={list(ds.images.shape)}")

    assert len(datasets) == 3
    # All datasets should have different names
    names = [ds.name for ds in datasets.values()]
    assert len(set(names)) == 3, f"Expected 3 unique names, got {names}"
    print("PASS: Multiple generators produce distinct datasets")
    passed += 1

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"M3 VERIFICATION: {passed}/{total} tests passed")
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
