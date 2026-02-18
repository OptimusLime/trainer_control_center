"""MNIST Factor Experiment — no-free vs with-free factor groups.

Branch 0 (vanilla_vae): Standard ConvVAE baseline. Recon + KL only.

Branch 1 (mnist_only): FactorSlotAutoencoder (with free group) on MNIST only.
  Recon + KL + digit classification. No factor supervision.

Branch 2 (curriculum_free): FactorSlot WITH free group + full curriculum.
  digit + thickness + slant + 12 free channels.  Uniform task sampling.

Branch 3 (curriculum_nofree): FactorSlot WITHOUT free group + full curriculum.
  digit + thickness + slant only — 20 channels total. Every channel must
  serve a named factor. Forces the model to actually encode thickness in
  the thickness channels and slant in the slant channels, instead of
  dumping everything into free.

All curriculum branches use uniform task sampling (random.choices, equal weights).
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


# WITH free group: 32 channels, 8x8 spatial → 2048 flat latent dims
FACTOR_GROUPS_FREE = [
    FactorGroup("digit", 0, 8),          # 8 channels
    FactorGroup("thickness", 8, 14),     # 6 channels
    FactorGroup("slant", 14, 20),        # 6 channels
    FactorGroup("free", 20, 32),         # 12 channels
]

# WITHOUT free group: 20 channels, 8x8 spatial → 1280 flat latent dims
# Every channel belongs to a supervised factor.
FACTOR_GROUPS_NOFREE = [
    FactorGroup("digit", 0, 8),          # 8 channels
    FactorGroup("thickness", 8, 14),     # 6 channels
    FactorGroup("slant", 14, 20),        # 6 channels
]

BATCHES_PER_EPOCH = 937  # 60k / 64

# Total gradient steps per branch.
TOTAL_STEPS = BATCHES_PER_EPOCH * 3 * 10  # 28,110

# KL annealing warmup (in KL-task calls, not total steps).
KL_WARMUP_STEPS = 1000


def _build_factor_model_free() -> FactorSlotAutoencoder:
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS_FREE,
        image_size=32,
    )


def _build_factor_model_nofree() -> FactorSlotAutoencoder:
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS_NOFREE,
        image_size=32,
    )


def _build_vanilla_vae() -> ConvVAE:
    return ConvVAE(
        in_channels=1,
        latent_dim=128,
        image_size=32,
        base_channels=32,
    )


def _attach_curriculum_tasks(ctx, mnist, thickness_ds, slant_ds):
    """Attach the full 5-task curriculum to the current model."""
    ctx.detach_all_tasks()
    ctx.attach_task(ReconstructionTask("recon", mnist))
    ctx.attach_task(KLDivergenceTask(
        "kl", mnist, weight=0.5, warmup_steps=KL_WARMUP_STEPS,
    ))
    ctx.attach_task(
        ClassificationTask("digit_classify", mnist, factor_name="digit")
    )
    ctx.attach_task(
        RegressionTask("thickness", thickness_ds, output_dim=1, factor_name="thickness")
    )
    ctx.attach_task(
        RegressionTask("slant", slant_ds, output_dim=1, factor_name="slant")
    )


class MNISTFactorExperiment(Recipe):
    name = "mnist_factor_experiment"
    description = (
        "No-free vs with-free factor groups: does removing the free "
        "channel group force real disentanglement?"
    )

    def run(self, ctx: RecipeContext) -> None:
        # === Common setup ===
        ctx.phase = "Load MNIST 32x32"
        mnist = ctx.load_dataset("mnist_32", lambda: load_mnist(image_size=32))

        ctx.phase = "Generate synthetic datasets"
        thickness_ds = ctx.load_dataset(
            "thickness_synth", lambda: generate_thickness(n=5000, image_size=32)
        )
        slant_ds = ctx.load_dataset(
            "slant_synth", lambda: generate_slant(n=5000, image_size=32)
        )

        # ============================================================
        # Branch 0: VANILLA VAE BASELINE
        # ============================================================
        ctx.phase = "Build vanilla ConvVAE"
        ctx.create_model(_build_vanilla_vae)
        ctx.save_checkpoint("vanilla_root")

        ctx.detach_all_tasks()
        ctx.attach_task(ReconstructionTask("recon", mnist))
        ctx.attach_task(KLDivergenceTask("kl", mnist, weight=0.5))

        ctx.phase = f"Train vanilla ({TOTAL_STEPS} steps)"
        ctx.train(steps=TOTAL_STEPS, lr=1e-3)
        ctx.save_checkpoint("vanilla_trained")

        ctx.phase = "Eval vanilla"
        m_vanilla = ctx.evaluate()
        ctx.log(f"vanilla: {m_vanilla}")

        # ============================================================
        # Branch 1: FACTOR-SLOT, MNIST ONLY (with free group)
        # ============================================================
        ctx.phase = "Build FactorSlot (with free)"
        ctx.create_model(_build_factor_model_free)
        factor_root_free = ctx.save_checkpoint("factor_root_free")

        ctx.phase = "Fork -> mnist_only"
        ctx.fork(factor_root_free, "mnist_only")
        ctx.detach_all_tasks()
        ctx.attach_task(ReconstructionTask("recon", mnist))
        ctx.attach_task(KLDivergenceTask(
            "kl", mnist, weight=0.5, warmup_steps=KL_WARMUP_STEPS,
        ))
        ctx.attach_task(
            ClassificationTask("digit_classify", mnist, factor_name="digit")
        )

        ctx.phase = f"Train mnist_only ({TOTAL_STEPS} steps)"
        ctx.train(steps=TOTAL_STEPS, lr=1e-3)
        ctx.save_checkpoint("mnist_only_trained")

        ctx.phase = "Eval mnist_only"
        m_mnist = ctx.evaluate()
        ctx.log(f"mnist_only: {m_mnist}")

        # ============================================================
        # Branch 2: CURRICULUM WITH FREE GROUP (32ch)
        # ============================================================
        ctx.phase = "Fork -> curriculum_free"
        ctx.fork(factor_root_free, "curriculum_free")
        _attach_curriculum_tasks(ctx, mnist, thickness_ds, slant_ds)

        ctx.phase = f"Train curriculum_free ({TOTAL_STEPS} steps)"
        ctx.train(steps=TOTAL_STEPS, lr=1e-3)
        ctx.save_checkpoint("curriculum_free_trained")

        ctx.phase = "Eval curriculum_free"
        m_free = ctx.evaluate()
        ctx.log(f"curriculum_free: {m_free}")

        # ============================================================
        # Branch 3: CURRICULUM WITHOUT FREE GROUP (20ch)
        # ============================================================
        ctx.phase = "Build FactorSlot (no free)"
        ctx.create_model(_build_factor_model_nofree)
        factor_root_nofree = ctx.save_checkpoint("factor_root_nofree")

        ctx.phase = "Fork -> curriculum_nofree"
        ctx.fork(factor_root_nofree, "curriculum_nofree")
        _attach_curriculum_tasks(ctx, mnist, thickness_ds, slant_ds)

        ctx.phase = f"Train curriculum_nofree ({TOTAL_STEPS} steps)"
        ctx.train(steps=TOTAL_STEPS, lr=1e-3)
        ctx.save_checkpoint("curriculum_nofree_trained")

        ctx.phase = "Eval curriculum_nofree"
        m_nofree = ctx.evaluate()
        ctx.log(f"curriculum_nofree: {m_nofree}")

        # ============================================================
        # Summary
        # ============================================================
        ctx.phase = "Complete"
        ctx.log(f"COMPARISON @ {TOTAL_STEPS} steps, uniform sampling:")
        ctx.log(f"  vanilla:          {m_vanilla}")
        ctx.log(f"  mnist_only:       {m_mnist}")
        ctx.log(f"  curriculum_free:  {m_free}")
        ctx.log(f"  curriculum_nofree:{m_nofree}")
