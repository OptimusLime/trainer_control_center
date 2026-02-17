"""M1.95 Verification Script — Recipes + Checkpoint Tree + Experiment Runner.

Usage:
    python -m acc.test_m1_95

This proves:
1. FactorSlotAutoencoder with VAE reparameterization returns MU, LOGVAR, LATENT
2. KLDivergenceTask computes KL from MU/LOGVAR, supports per-factor slice
3. Synthetic thickness and slant generators produce valid AccDatasets
4. Recipe infrastructure: Recipe base, RecipeContext, RecipeRunner, RecipeRegistry
5. RecipeContext.fork() creates a branch in the checkpoint tree
6. A test recipe: create model, save root, fork two branches, train each, verify tree
7. RecipeRegistry discovers recipes from acc/recipes/ directory
8. Recipe hot-reload: file watcher detects new recipe files
9. Checkpoint tree has correct parent-child relationships after fork
"""

import os
import shutil
import tempfile
import time

import torch
import torch.nn as nn

from acc.factor_group import FactorGroup
from acc.factor_slot_autoencoder import FactorSlotAutoencoder
from acc.model_output import ModelOutput
from acc.dataset import AccDataset
from acc.generators.thickness import generate_thickness
from acc.generators.slant import generate_slant
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.kl_divergence import KLDivergenceTask
from acc.tasks.classification import ClassificationTask
from acc.tasks.regression import RegressionTask
from acc.trainer import Trainer
from acc.checkpoints import CheckpointStore
from acc.recipes.base import Recipe, RecipeContext, RecipeJob
from acc.recipes.runner import RecipeRunner
from acc.recipes.registry import RecipeRegistry


FACTOR_GROUPS = [
    FactorGroup("digit", 0, 4),
    FactorGroup("thickness", 4, 7),
    FactorGroup("slant", 7, 10),
    FactorGroup("free", 10, 16),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_checkpoint_dir = "./test_checkpoints_m195"
    if os.path.exists(test_checkpoint_dir):
        shutil.rmtree(test_checkpoint_dir)

    # ── 1. FactorSlotAutoencoder with VAE ──
    print("\n=== 1. FactorSlotAutoencoder with VAE reparameterization ===")
    model = FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS,
        backbone_channels=[32, 64],
        embed_dim=32,
        image_size=32,
    )
    model = model.to(device)

    assert model.has_decoder
    assert model.latent_dim == 16
    print(f"Model built. latent_dim={model.latent_dim}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randn(4, 1, 32, 32, device=device)
    out = model(dummy)
    assert ModelOutput.LATENT in out
    assert ModelOutput.MU in out
    assert ModelOutput.LOGVAR in out
    assert ModelOutput.RECONSTRUCTION in out
    assert ModelOutput.FACTOR_SLICES in out
    assert out[ModelOutput.LATENT].shape == (4, 16)
    assert out[ModelOutput.MU].shape == (4, 16)
    assert out[ModelOutput.LOGVAR].shape == (4, 16)
    assert out[ModelOutput.RECONSTRUCTION].shape == (4, 1, 32, 32)

    # Verify MU and LOGVAR are different (logvar shouldn't be all zeros)
    mu = out[ModelOutput.MU]
    logvar = out[ModelOutput.LOGVAR]
    assert not torch.allclose(mu, logvar), "MU and LOGVAR should differ"
    print(f"MU range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")
    print(f"LOGVAR range: [{logvar.min().item():.3f}, {logvar.max().item():.3f}]")
    print("PASS: FactorSlotAutoencoder returns MU, LOGVAR, LATENT")

    # ── 2. KLDivergenceTask ──
    print("\n=== 2. KLDivergenceTask ===")
    # Create a minimal dataset for KL task (it doesn't actually use the images/targets for loss)
    kl_dataset = AccDataset(
        torch.randn(100, 1, 32, 32), torch.zeros(100), name="dummy_kl"
    )
    kl_task = KLDivergenceTask("kl_full", kl_dataset, weight=1.0)
    kl_task.attach(model)
    kl_loss = kl_task.compute_loss(out, (dummy, torch.zeros(4, device=device)))
    assert kl_loss.shape == (), f"KL loss should be scalar, got {kl_loss.shape}"
    assert kl_loss.item() > 0, f"KL loss should be positive, got {kl_loss.item()}"
    print(f"KL loss (full): {kl_loss.item():.4f}")

    # Per-factor KL
    kl_digit = KLDivergenceTask(
        "kl_digit", kl_dataset, weight=1.0, latent_slice=(0, 4)
    )
    kl_digit.attach(model)
    kl_digit_loss = kl_digit.compute_loss(out, (dummy, torch.zeros(4, device=device)))
    assert kl_digit_loss.item() > 0
    print(f"KL loss (digit 0:4): {kl_digit_loss.item():.4f}")

    kl_free = KLDivergenceTask(
        "kl_free", kl_dataset, weight=0.1, latent_slice=(10, 16)
    )
    kl_free.attach(model)
    kl_free_loss = kl_free.compute_loss(out, (dummy, torch.zeros(4, device=device)))
    print(f"KL loss (free 10:16, weight=0.1): {kl_free_loss.item():.4f}")
    print("PASS: KLDivergenceTask computes per-factor KL correctly")

    # ── 3. Synthetic generators ──
    print("\n=== 3. Synthetic generators ===")
    thickness_ds = generate_thickness(n=200, image_size=32)
    assert thickness_ds.images.shape == (200, 1, 32, 32)
    assert thickness_ds.targets.shape == (200,) or thickness_ds.targets.shape == (200, 1)
    assert thickness_ds.images.min() >= 0 and thickness_ds.images.max() <= 1
    print(f"Thickness dataset: {thickness_ds.describe()}")

    slant_ds = generate_slant(n=200, image_size=32)
    assert slant_ds.images.shape == (200, 1, 32, 32)
    assert slant_ds.targets.shape == (200,) or slant_ds.targets.shape == (200, 1)
    assert slant_ds.images.min() >= 0 and slant_ds.images.max() <= 1
    print(f"Slant dataset: {slant_ds.describe()}")
    print("PASS: Thickness and slant generators produce valid AccDatasets")

    # ── 4. KL + Recon train together ──
    print("\n=== 4. Train with ReconTask + KLTask (200 steps) ===")
    recon_task = ReconstructionTask("recon", kl_dataset)
    recon_task.attach(model)

    tasks = [recon_task, kl_task]
    trainer = Trainer(model, tasks, device, lr=1e-3, batch_size=16)
    losses = trainer.train(steps=200)
    assert len(losses) == 200
    # Both tasks should appear
    task_names = set(l["task_name"] for l in losses)
    assert "recon" in task_names
    assert "kl_full" in task_names
    early_recon = [l["task_loss"] for l in losses[:20] if l["task_name"] == "recon"]
    late_recon = [l["task_loss"] for l in losses[-20:] if l["task_name"] == "recon"]
    if early_recon and late_recon:
        print(f"Recon loss: {sum(early_recon)/len(early_recon):.4f} -> {sum(late_recon)/len(late_recon):.4f}")
    print("PASS: ReconTask + KLTask train together")

    # ── 5. Recipe infrastructure (tiny recipe with fork) ──
    print("\n=== 5. Recipe infrastructure — tiny recipe with fork ===")

    # Create a minimal TrainerAPI-like object for RecipeContext
    class FakeAPI:
        def __init__(self):
            self.autoencoder = None
            self.trainer = None
            self.tasks = {}
            self.datasets = {}
            self.device = device
            self.checkpoints = CheckpointStore(test_checkpoint_dir)

    api = FakeAPI()

    ctx = RecipeContext(api)

    # Create model
    ctx.create_model(lambda: FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS,
        backbone_channels=[16, 32],
        embed_dim=16,
        image_size=32,
    ))
    assert api.autoencoder is not None
    print(f"Model created: latent_dim={api.autoencoder.latent_dim}")

    # Load dataset
    ds = ctx.load_dataset("tiny", lambda: AccDataset(
        torch.randn(100, 1, 32, 32), torch.zeros(100), name="tiny"
    ))
    assert "tiny" in api.datasets
    print(f"Dataset loaded: {ds.name}")

    # Attach tasks
    recon = ReconstructionTask("recon", ds)
    ctx.attach_task(recon)
    kl = KLDivergenceTask("kl", ds, weight=1.0)
    ctx.attach_task(kl)
    assert len(api.tasks) == 2
    print(f"Tasks attached: {list(api.tasks.keys())}")

    # Save root checkpoint
    ctx.phase = "Save root"
    root_id = ctx.save_checkpoint("root")
    assert root_id is not None
    print(f"Root checkpoint: {root_id[:8]} (tag='root')")

    # Fork to branch_a
    ctx.phase = "Fork branch_a"
    ctx.detach_all_tasks()
    branch_a_id = ctx.fork(root_id, "branch_a")
    ctx.attach_task(ReconstructionTask("recon", ds))
    ctx.attach_task(KLDivergenceTask("kl", ds, weight=1.0))
    print(f"Forked branch_a: {branch_a_id[:8]}")

    # Train branch_a briefly
    ctx.phase = "Train branch_a"
    ctx.train(steps=50, lr=1e-3)
    branch_a_save_id = ctx.save_checkpoint("branch_a_50")
    print(f"Branch_a trained 50 steps, saved: {branch_a_save_id[:8]}")

    # Fork to branch_b (from root, not from branch_a!)
    ctx.phase = "Fork branch_b"
    ctx.detach_all_tasks()
    branch_b_id = ctx.fork(root_id, "branch_b")
    ctx.attach_task(ReconstructionTask("recon", ds))
    ctx.attach_task(KLDivergenceTask("kl", ds, weight=2.0))  # Different beta
    print(f"Forked branch_b: {branch_b_id[:8]}")

    # Train branch_b briefly
    ctx.phase = "Train branch_b"
    ctx.train(steps=50, lr=1e-3)
    branch_b_save_id = ctx.save_checkpoint("branch_b_50")
    print(f"Branch_b trained 50 steps, saved: {branch_b_save_id[:8]}")

    # Verify checkpoint tree
    tree = api.checkpoints.tree()
    assert len(tree) == 5, f"Expected 5 checkpoints, got {len(tree)}"
    # root, branch_a (fork), branch_a_50 (save), branch_b (fork), branch_b_50 (save)

    tags = {cp.tag for cp in tree}
    assert "root" in tags
    assert "branch_a" in tags
    assert "branch_a_50" in tags
    assert "branch_b" in tags
    assert "branch_b_50" in tags

    # Verify tree structure
    by_id = {cp.id: cp for cp in tree}

    # branch_a's parent should be root
    branch_a_cp = next(cp for cp in tree if cp.tag == "branch_a")
    assert branch_a_cp.parent_id == root_id, (
        f"branch_a parent should be root ({root_id[:8]}), got {branch_a_cp.parent_id}"
    )

    # branch_b's parent should also be root (not branch_a!)
    branch_b_cp = next(cp for cp in tree if cp.tag == "branch_b")
    assert branch_b_cp.parent_id == root_id, (
        f"branch_b parent should be root ({root_id[:8]}), got {branch_b_cp.parent_id}"
    )

    # branch_a_50's parent should be branch_a
    branch_a_50_cp = next(cp for cp in tree if cp.tag == "branch_a_50")
    assert branch_a_50_cp.parent_id == branch_a_id, (
        f"branch_a_50 parent should be branch_a, got {branch_a_50_cp.parent_id}"
    )

    # branch_b_50's parent should be branch_b
    branch_b_50_cp = next(cp for cp in tree if cp.tag == "branch_b_50")
    assert branch_b_50_cp.parent_id == branch_b_id, (
        f"branch_b_50 parent should be branch_b, got {branch_b_50_cp.parent_id}"
    )

    print("Tree structure:")
    for cp in tree:
        parent_tag = by_id[cp.parent_id].tag if cp.parent_id and cp.parent_id in by_id else "-"
        print(f"  {cp.tag} ({cp.id[:8]}) <- parent: {parent_tag}")
    print("PASS: Recipe fork creates correct tree structure")

    # ── 6. RecipeRunner executes in background thread ──
    print("\n=== 6. RecipeRunner background execution ===")

    class TinyRecipe(Recipe):
        name = "tiny_test"
        description = "Minimal test recipe"

        def run(self, ctx: RecipeContext) -> None:
            ctx.phase = "Setup"
            ctx.create_model(lambda: FactorSlotAutoencoder(
                in_channels=1,
                factor_groups=FACTOR_GROUPS,
                backbone_channels=[16, 32],
                embed_dim=16,
                image_size=32,
            ))
            ds = ctx.load_dataset("tiny2", lambda: AccDataset(
                torch.randn(100, 1, 32, 32), torch.zeros(100), name="tiny2"
            ))
            ctx.attach_task(ReconstructionTask("recon", ds))
            ctx.phase = "Train"
            ctx.train(steps=20, lr=1e-3, batch_size=16)
            ctx.phase = "Done"

    # Clean up for runner test
    test_dir_2 = "./test_checkpoints_m195_runner"
    if os.path.exists(test_dir_2):
        shutil.rmtree(test_dir_2)

    api2 = FakeAPI()
    api2.checkpoints = CheckpointStore(test_dir_2)

    runner = RecipeRunner()
    job = runner.start(TinyRecipe(), api2)
    assert job.state == "running"
    print(f"Recipe job started: {job.id}")

    # Wait for completion
    runner.wait(timeout=30.0)
    final_job = runner.current()
    if final_job.state == "failed":
        print(f"Recipe FAILED. Error: {final_job.error}")
    assert final_job.state == "completed", f"Expected 'completed', got '{final_job.state}'. Error: {final_job.error}"
    print(f"Recipe completed. Phases: {final_job.phases_completed}")
    print("PASS: RecipeRunner executes recipe in background thread")

    shutil.rmtree(test_dir_2, ignore_errors=True)

    # ── 7. RecipeRegistry discovers recipes ──
    print("\n=== 7. RecipeRegistry discovery ===")
    registry = RecipeRegistry()
    recipes = registry.list()
    print(f"Discovered {len(recipes)} recipes: {[r['name'] for r in recipes]}")

    # Should find mnist_factor_experiment
    names = [r["name"] for r in recipes]
    assert "mnist_factor_experiment" in names, (
        f"Expected 'mnist_factor_experiment' in registry, got {names}"
    )
    print("PASS: RecipeRegistry discovers recipes from acc/recipes/")

    # ── 8. Recipe hot-reload ──
    print("\n=== 8. Recipe hot-reload (file watcher) ===")
    # Create a temp recipe file
    recipes_dir = os.path.join(os.path.dirname(__file__), "recipes")
    temp_recipe = os.path.join(recipes_dir, "hot_reload_test.py")

    # Start watcher
    registry2 = RecipeRegistry(recipes_dir)
    registry2.start_watcher(poll_interval=0.5)

    # Write a new recipe file
    recipe_code = '''
from acc.recipes.base import Recipe, RecipeContext

class HotReloadTestRecipe(Recipe):
    name = "hot_reload_test"
    description = "Test hot reload"

    def run(self, ctx: RecipeContext) -> None:
        ctx.phase = "test"
'''
    try:
        with open(temp_recipe, "w") as f:
            f.write(recipe_code)

        # Wait for watcher to pick it up
        time.sleep(2.0)

        recipes2 = registry2.list()
        names2 = [r["name"] for r in recipes2]
        assert "hot_reload_test" in names2, (
            f"Hot-reloaded recipe not found. Got: {names2}"
        )
        print(f"Hot-reloaded recipes: {names2}")
        print("PASS: Recipe hot-reload works")
    finally:
        # Clean up
        registry2.stop_watcher()
        if os.path.exists(temp_recipe):
            os.remove(temp_recipe)
        # Clean up __pycache__
        pycache = os.path.join(recipes_dir, "__pycache__")
        for f in os.listdir(pycache) if os.path.exists(pycache) else []:
            if "hot_reload_test" in f:
                os.remove(os.path.join(pycache, f))

    # ── 9. Eval on trained model ──
    print("\n=== 9. Eval on tiny trained model ===")
    results = ctx.evaluate()
    print(f"Eval results: {results}")
    print("PASS: Evaluation runs on recipe-trained model")

    # Cleanup
    shutil.rmtree(test_checkpoint_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("M1.95 PASSED: Recipes + Checkpoint Tree + Experiment Runner")
    print("  - FactorSlotAutoencoder with VAE (MU/LOGVAR)")
    print("  - KLDivergenceTask with per-factor slicing")
    print("  - Synthetic thickness/slant generators")
    print("  - Recipe infrastructure (base, context, runner, registry)")
    print("  - Fork creates correct checkpoint tree structure")
    print("  - Recipe hot-reload via file watcher")
    print("=" * 60)


if __name__ == "__main__":
    main()
