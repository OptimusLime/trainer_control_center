"""MNIST Factor Experiment — three-branch comparison with vanilla baseline.

Branch 0 (vanilla_vae): Standard ConvVAE baseline. Recon + KL only.
  If this can't get recon L1 < 0.03 in 5 epochs, the pipeline is broken.

Branch 1 (mnist_only): FactorSlotAutoencoder trained on MNIST only.
  Recon + KL + digit classification. Thickness/slant slots learn from
  recon+KL only.

Branch 2 (with_curriculum): IDENTICAL FactorSlot architecture, PLUS synthetic
  thickness and slant regression tasks. Same fork point, same initial weights.

Step count math:
  MNIST: 60k images, batch_size=64, 1 epoch = 937 batches.
  With N tasks round-robin, 1 recon epoch = 937 * N total steps.
  We train for RECON_EPOCHS epochs of recon per branch.
"""

from acc.recipes.base import Recipe, RecipeContext
from acc.factor_group import FactorGroup
from acc.factor_slot_autoencoder import FactorSlotAutoencoder
from acc.conv_vae import ConvVAE
from acc.dataset import load_mnist
from acc.generators.thickness import generate_thickness
from acc.generators.slant import generate_slant
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.classification import ClassificationTask
from acc.tasks.kl_divergence import KLDivergenceTask
from acc.tasks.regression import RegressionTask


FACTOR_GROUPS = [
    FactorGroup("digit", 0, 4),
    FactorGroup("thickness", 4, 7),
    FactorGroup("slant", 7, 10),
    FactorGroup("free", 10, 16),
]

# 60k MNIST / batch 64 = 937 batches per epoch
BATCHES_PER_EPOCH = 937
RECON_EPOCHS = 10


def _build_factor_model() -> FactorSlotAutoencoder:
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS,
        image_size=32,
    )


def _build_vanilla_vae() -> ConvVAE:
    return ConvVAE(
        in_channels=1,
        latent_dim=16,
        image_size=32,
        base_channels=32,
    )


class MNISTFactorExperiment(Recipe):
    name = "mnist_factor_experiment"
    description = (
        "Three-branch comparison: vanilla ConvVAE baseline, "
        "factor-slot MNIST-only, factor-slot + synthetic curriculum"
    )

    def run(self, ctx: RecipeContext) -> None:
        # === Common setup ===
        ctx.phase = "Load MNIST 32x32"
        mnist = ctx.load_dataset("mnist_32", lambda: load_mnist(image_size=32))

        # ============================================================
        # Branch 0: VANILLA VAE BASELINE
        # ============================================================
        # This is our sanity check. If a standard ConvVAE can't learn
        # MNIST recon through our pipeline, something is fundamentally
        # broken in the training loop, loss function, or data loading.

        ctx.phase = "Build vanilla ConvVAE"
        ctx.create_model(_build_vanilla_vae)

        ctx.phase = "Save vanilla root"
        vanilla_root = ctx.save_checkpoint("vanilla_root")

        ctx.detach_all_tasks()
        ctx.attach_task(ReconstructionTask("recon", mnist))
        ctx.attach_task(KLDivergenceTask("kl", mnist, weight=0.5))

        # 2 tasks round-robin, RECON_EPOCHS epochs of recon
        vanilla_steps = BATCHES_PER_EPOCH * 2 * RECON_EPOCHS
        ctx.phase = f"Train vanilla VAE ({vanilla_steps} steps = {RECON_EPOCHS} recon epochs)"
        ctx.train(steps=vanilla_steps, lr=1e-3)

        ctx.phase = "Save vanilla checkpoint"
        ctx.save_checkpoint(f"vanilla_{RECON_EPOCHS}ep")

        ctx.phase = "Eval vanilla"
        metrics_0 = ctx.evaluate()
        ctx.log(f"vanilla_vae metrics: {metrics_0}")

        # ============================================================
        # Branch 1: FACTOR-SLOT, MNIST ONLY
        # ============================================================
        ctx.phase = "Build FactorSlotAutoencoder"
        ctx.create_model(_build_factor_model)

        ctx.phase = "Save factor root"
        factor_root = ctx.save_checkpoint("factor_root")

        ctx.phase = "Fork -> mnist_only"
        ctx.fork(factor_root, "mnist_only")
        ctx.detach_all_tasks()

        ctx.attach_task(ReconstructionTask("recon", mnist))
        ctx.attach_task(KLDivergenceTask("kl", mnist, weight=1.0))
        ctx.attach_task(
            ClassificationTask("digit_classify", mnist, latent_slice=(0, 4))
        )

        # 3 tasks round-robin, RECON_EPOCHS epochs of recon
        factor_only_steps = BATCHES_PER_EPOCH * 3 * RECON_EPOCHS
        ctx.phase = f"Train mnist_only ({factor_only_steps} steps = {RECON_EPOCHS} recon epochs)"
        ctx.train(steps=factor_only_steps, lr=1e-3)

        ctx.phase = "Save mnist_only checkpoint"
        ctx.save_checkpoint(f"mnist_only_{RECON_EPOCHS}ep")

        ctx.phase = "Eval mnist_only"
        metrics_1 = ctx.evaluate()
        ctx.log(f"mnist_only metrics: {metrics_1}")

        # ============================================================
        # Branch 2: FACTOR-SLOT + CURRICULUM
        # ============================================================
        ctx.phase = "Fork -> with_curriculum"
        ctx.fork(factor_root, "with_curriculum")
        ctx.detach_all_tasks()

        ctx.attach_task(ReconstructionTask("recon", mnist))
        ctx.attach_task(KLDivergenceTask("kl", mnist, weight=1.0))
        ctx.attach_task(
            ClassificationTask("digit_classify", mnist, latent_slice=(0, 4))
        )

        ctx.phase = "Generate synthetic datasets"
        thickness_ds = ctx.load_dataset(
            "thickness_synth", lambda: generate_thickness(n=5000, image_size=32)
        )
        slant_ds = ctx.load_dataset(
            "slant_synth", lambda: generate_slant(n=5000, image_size=32)
        )

        ctx.attach_task(
            RegressionTask(
                "thickness", thickness_ds, output_dim=1, latent_slice=(4, 7)
            )
        )
        ctx.attach_task(
            RegressionTask("slant", slant_ds, output_dim=1, latent_slice=(7, 10))
        )

        # 5 tasks round-robin, RECON_EPOCHS epochs of recon
        curriculum_steps = BATCHES_PER_EPOCH * 5 * RECON_EPOCHS
        ctx.phase = f"Train with_curriculum ({curriculum_steps} steps = {RECON_EPOCHS} recon epochs)"
        ctx.train(steps=curriculum_steps, lr=1e-3)

        ctx.phase = "Save curriculum checkpoint"
        ctx.save_checkpoint(f"curriculum_{RECON_EPOCHS}ep")

        ctx.phase = "Eval with_curriculum"
        metrics_2 = ctx.evaluate()
        ctx.log(f"with_curriculum metrics: {metrics_2}")

        ctx.phase = f"Complete - compare vanilla vs mnist_only vs curriculum @ {RECON_EPOCHS} epochs"
