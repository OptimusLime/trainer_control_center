"""MNIST Factor Experiment — two-branch comparison.

Tests: does synthetic task curriculum improve disentanglement in a factor-slot VAE?

Branch 1 (mnist_only): FactorSlotAutoencoder trained on MNIST only.
  Recon + per-factor KL + digit classification. Thickness/slant slots learn
  from recon+KL only — whatever structure they find, they find on their own.

Branch 2 (with_curriculum): IDENTICAL architecture, PLUS synthetic thickness
  and slant regression tasks. Same fork point, same initial weights.

If branch 2 has cleaner thickness/slant traversals, the synthetic tasks worked.
"""

from acc.recipes.base import Recipe, RecipeContext
from acc.factor_group import FactorGroup
from acc.factor_slot_autoencoder import FactorSlotAutoencoder
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


def _build_model() -> FactorSlotAutoencoder:
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS,
        image_size=32,
    )


class MNISTFactorExperiment(Recipe):
    name = "mnist_factor_experiment"
    description = (
        "Two-branch comparison: factor-slot MNIST-only vs "
        "factor-slot + synthetic thickness/slant curriculum"
    )

    def run(self, ctx: RecipeContext) -> None:
        # === Common setup ===
        ctx.phase = "Load MNIST 32x32"
        mnist = ctx.load_dataset("mnist_32", lambda: load_mnist(image_size=32))

        ctx.phase = "Build FactorSlotAutoencoder"
        ctx.create_model(_build_model)

        ctx.phase = "Save root checkpoint"
        root_id = ctx.save_checkpoint("experiment_root")

        # === Branch 1: MNIST only ===
        ctx.phase = "Fork -> mnist_only"
        ctx.fork(root_id, "mnist_only")
        ctx.detach_all_tasks()

        ctx.attach_task(ReconstructionTask("recon", mnist))
        ctx.attach_task(KLDivergenceTask("kl", mnist, weight=1.0))
        ctx.attach_task(
            ClassificationTask("digit_classify", mnist, latent_slice=(0, 4))
        )

        ctx.phase = "Train mnist_only (5000 steps)"
        ctx.train(steps=5000, lr=1e-3)

        ctx.phase = "Save mnist_only checkpoint"
        ctx.save_checkpoint("mnist_only_5k")

        ctx.phase = "Eval mnist_only"
        metrics_1 = ctx.evaluate()
        ctx.log(f"mnist_only metrics: {metrics_1}")

        # === Branch 2: MNIST + synthetic curriculum ===
        ctx.phase = "Fork -> with_curriculum"
        ctx.fork(root_id, "with_curriculum")
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

        ctx.phase = "Train with_curriculum (5000 steps)"
        ctx.train(steps=5000, lr=1e-3)

        ctx.phase = "Save curriculum checkpoint"
        ctx.save_checkpoint("curriculum_5k")

        ctx.phase = "Eval with_curriculum"
        metrics_2 = ctx.evaluate()
        ctx.log(f"with_curriculum metrics: {metrics_2}")

        ctx.phase = "Complete - compare mnist_only_5k vs curriculum_5k"
