"""MNIST Factor Experiment — weighted sampling comparison.

Branch 0 (vanilla_vae): Standard ConvVAE baseline. Recon + KL only.

Branch 1 (mnist_only): FactorSlotAutoencoder on MNIST only.
  Recon + KL (annealed) + digit classification. No factor supervision.

Branches 2a/2b/2c: FactorSlot + curriculum (recon + KL + digit + thickness + slant),
  forked from the same factor_root. SAME total gradient steps, but different
  task sampling weights:
    2a: 90% recon, rest uniform    — tests "recon-heavy" schedule
    2b: 50% recon, rest uniform    — tests "balanced" schedule
    2c: uniform (20% each)         — tests naive equal sampling

This isolates whether the curriculum branch's poor recon (L1=0.147 in
the uniform run) is caused by recon starvation vs something deeper.

Step count: all branches use TOTAL_STEPS gradient updates.  With weighted
sampling, the number of recon batches varies by weight, but the encoder
sees the same total number of gradient steps.
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


# Factor groups index into the CHANNEL dimension of the spatial bottleneck.
# 32 channels total, 8x8 spatial → 2048 total flat latent dims.
FACTOR_GROUPS = [
    FactorGroup("digit", 0, 8),          # 8 channels for digit identity
    FactorGroup("thickness", 8, 14),     # 6 channels for stroke thickness
    FactorGroup("slant", 14, 20),        # 6 channels for slant
    FactorGroup("free", 20, 32),         # 12 channels for uncontrolled variation
]

# 60k MNIST / batch 64 = 937 batches per epoch
BATCHES_PER_EPOCH = 937

# Total gradient steps per branch.  28,110 = 937 * 3 * 10, i.e. enough for
# ~10 recon epochs if recon gets ~1/3 of the steps.
TOTAL_STEPS = BATCHES_PER_EPOCH * 3 * 10

# KL annealing: ramp KL from 0 → full over this many KL-task calls.
# Lower than before so it finishes even when KL gets a small sampling share.
KL_WARMUP_STEPS = 1000


def _build_factor_model() -> FactorSlotAutoencoder:
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS,
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
        "Weighted sampling comparison: vanilla, mnist-only, "
        "curriculum @ 90%/50%/uniform recon weight"
    )

    def run(self, ctx: RecipeContext) -> None:
        # === Common setup ===
        ctx.phase = "Load MNIST 32x32"
        mnist = ctx.load_dataset("mnist_32", lambda: load_mnist(image_size=32))

        # ============================================================
        # Branch 0: VANILLA VAE BASELINE
        # ============================================================
        ctx.phase = "Build vanilla ConvVAE"
        ctx.create_model(_build_vanilla_vae)

        ctx.phase = "Save vanilla root"
        vanilla_root = ctx.save_checkpoint("vanilla_root")

        ctx.detach_all_tasks()
        ctx.attach_task(ReconstructionTask("recon", mnist))
        ctx.attach_task(KLDivergenceTask("kl", mnist, weight=0.5))

        # 2 tasks, uniform sampling → each gets ~50%
        # TOTAL_STEPS gives ~15 recon epochs at 50%
        ctx.phase = f"Train vanilla VAE ({TOTAL_STEPS} steps)"
        ctx.train(steps=TOTAL_STEPS, lr=1e-3)

        ctx.phase = "Save vanilla checkpoint"
        ctx.save_checkpoint("vanilla_trained")

        ctx.phase = "Eval vanilla"
        metrics_vanilla = ctx.evaluate()
        ctx.log(f"vanilla metrics: {metrics_vanilla}")

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
        ctx.attach_task(KLDivergenceTask(
            "kl", mnist, weight=0.5, warmup_steps=KL_WARMUP_STEPS,
        ))
        ctx.attach_task(
            ClassificationTask("digit_classify", mnist, factor_name="digit")
        )

        # 3 tasks, uniform → recon gets ~33%
        ctx.phase = f"Train mnist_only ({TOTAL_STEPS} steps)"
        ctx.train(steps=TOTAL_STEPS, lr=1e-3)

        ctx.phase = "Save mnist_only checkpoint"
        ctx.save_checkpoint("mnist_only_trained")

        ctx.phase = "Eval mnist_only"
        metrics_mnist = ctx.evaluate()
        ctx.log(f"mnist_only metrics: {metrics_mnist}")

        # ============================================================
        # Generate synthetic datasets (shared by all curriculum branches)
        # ============================================================
        ctx.phase = "Generate synthetic datasets"
        thickness_ds = ctx.load_dataset(
            "thickness_synth", lambda: generate_thickness(n=5000, image_size=32)
        )
        slant_ds = ctx.load_dataset(
            "slant_synth", lambda: generate_slant(n=5000, image_size=32)
        )

        # ============================================================
        # Curriculum branches: same tasks, different sampling weights
        # ============================================================
        # Weight configs: (tag, recon_share description, weights dict)
        # Remaining weight after recon is split uniformly among kl,
        # digit_classify, thickness, slant.
        curriculum_configs = [
            ("curr_90recon", "90% recon", {"recon": 36, "kl": 1, "digit_classify": 1, "thickness": 1, "slant": 1}),
            ("curr_50recon", "50% recon", {"recon": 4, "kl": 1, "digit_classify": 1, "thickness": 1, "slant": 1}),
            ("curr_uniform", "uniform",   None),  # None = equal weights
        ]

        metrics_curriculum = {}
        for tag, desc, weights in curriculum_configs:
            ctx.phase = f"Fork -> {tag}"
            ctx.fork(factor_root, tag)
            _attach_curriculum_tasks(ctx, mnist, thickness_ds, slant_ds)

            ctx.phase = f"Train {tag} ({TOTAL_STEPS} steps, {desc})"
            ctx.train(steps=TOTAL_STEPS, lr=1e-3, task_weights=weights)

            ctx.phase = f"Save {tag} checkpoint"
            ctx.save_checkpoint(f"{tag}_trained")

            ctx.phase = f"Eval {tag}"
            metrics_curriculum[tag] = ctx.evaluate()
            ctx.log(f"{tag} metrics: {metrics_curriculum[tag]}")

        # ============================================================
        # Summary
        # ============================================================
        ctx.phase = "Complete"
        ctx.log(f"COMPARISON @ {TOTAL_STEPS} total steps:")
        ctx.log(f"  vanilla:       {metrics_vanilla}")
        ctx.log(f"  mnist_only:    {metrics_mnist}")
        for tag, desc, _ in curriculum_configs:
            ctx.log(f"  {tag:16s} {metrics_curriculum[tag]}")
