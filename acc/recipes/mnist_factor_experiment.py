"""MNIST Factor Experiment — stop-gradient isolation for factor disentanglement.

Previous experiment showed that reconstruction gradients flowing through
cross-attention overwhelm the factor probe gradients, corrupting named
factor channels into reconstruction-optimal representations. Traversals
show no visible thickness/slant variation even with no free channels.

This experiment tests whether detaching factor slices from the decoder's
cross-attention path forces real disentanglement:

Branch 0 (baseline_nofree): Re-run of curriculum_nofree (20ch, no detach).
  Control branch to compare against on same random seed.

Branch 1 (stopgrad_20ch): Same 20ch no-free config, but with
  detach_factor_grad=True. Reconstruction gradients can't flow back
  through cross-attention tokens to corrupt factor channels.

Branch 2 (stopgrad_half): 10ch total (4 digit, 3 thickness, 3 slant),
  detach_factor_grad=True. Half the capacity with gradient isolation.
  Tests whether extreme compression + stop-grad forces stronger encoding.

All branches use uniform task sampling and the same training schedule.
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


# 20 channels, no free group (same as previous experiment)
FACTOR_GROUPS_NOFREE = [
    FactorGroup("digit", 0, 8),          # 8 channels
    FactorGroup("thickness", 8, 14),     # 6 channels
    FactorGroup("slant", 14, 20),        # 6 channels
]

# 10 channels, no free group, half capacity
FACTOR_GROUPS_HALF = [
    FactorGroup("digit", 0, 4),          # 4 channels
    FactorGroup("thickness", 4, 7),      # 3 channels
    FactorGroup("slant", 7, 10),         # 3 channels
]

BATCHES_PER_EPOCH = 937  # 60k / 64
TOTAL_STEPS = BATCHES_PER_EPOCH * 3 * 10  # 28,110
KL_WARMUP_STEPS = 1000


def _build_nofree() -> FactorSlotAutoencoder:
    """20ch, no free group, no detach (baseline control)."""
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS_NOFREE,
        image_size=32,
        detach_factor_grad=False,
    )


def _build_stopgrad_20ch() -> FactorSlotAutoencoder:
    """20ch, no free group, with detach_factor_grad."""
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS_NOFREE,
        image_size=32,
        detach_factor_grad=True,
    )


def _build_stopgrad_half() -> FactorSlotAutoencoder:
    """10ch, no free group, with detach_factor_grad."""
    return FactorSlotAutoencoder(
        in_channels=1,
        factor_groups=FACTOR_GROUPS_HALF,
        image_size=32,
        detach_factor_grad=True,
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
        "Stop-gradient isolation: does detaching factor slices from "
        "decoder cross-attention force real disentanglement in traversals?"
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
        # Branch 0: BASELINE — 20ch no-free, NO detach (control)
        # ============================================================
        with ctx.branch("baseline_nofree", "20ch, no stop-grad (control)", total=3):
            ctx.phase = "Build model"
            ctx.create_model(_build_nofree)
            root_nofree = ctx.save_checkpoint(
                "baseline_nofree_root",
                description="CONTROL: 20ch, no stop-grad, untrained",
            )

            ctx.phase = "Fork and attach tasks"
            ctx.fork(root_nofree, "baseline_nofree")
            _attach_curriculum_tasks(ctx, mnist, thickness_ds, slant_ds)

            ctx.phase = f"Train ({TOTAL_STEPS} steps)"
            ctx.train(steps=TOTAL_STEPS, lr=1e-3)
            ctx.save_checkpoint(
                "baseline_nofree_trained",
                description=f"CONTROL: 20ch, no stop-grad, trained {TOTAL_STEPS} steps",
            )

            ctx.phase = "Evaluate"
            m_baseline = ctx.evaluate()
            ctx.record_results("baseline_nofree", m_baseline)
            ctx.log(f"baseline_nofree: {m_baseline}")

        # ============================================================
        # Branch 1: STOP-GRAD 20ch — same arch, detach_factor_grad=True
        # ============================================================
        with ctx.branch("stopgrad_20ch", "20ch, stop-grad ON", total=3):
            ctx.phase = "Build model"
            ctx.create_model(_build_stopgrad_20ch)
            root_sg20 = ctx.save_checkpoint(
                "stopgrad_20ch_root",
                description="EXPERIMENT: 20ch, stop-grad ON, untrained",
            )

            ctx.phase = "Fork and attach tasks"
            ctx.fork(root_sg20, "stopgrad_20ch")
            _attach_curriculum_tasks(ctx, mnist, thickness_ds, slant_ds)

            ctx.phase = f"Train ({TOTAL_STEPS} steps)"
            ctx.train(steps=TOTAL_STEPS, lr=1e-3)
            ctx.save_checkpoint(
                "stopgrad_20ch_trained",
                description=f"EXPERIMENT: 20ch, stop-grad ON, trained {TOTAL_STEPS} steps",
            )

            ctx.phase = "Evaluate"
            m_sg20 = ctx.evaluate()
            ctx.record_results("stopgrad_20ch", m_sg20)
            ctx.log(f"stopgrad_20ch: {m_sg20}")

        # ============================================================
        # Branch 2: STOP-GRAD HALF — 10ch, detach_factor_grad=True
        # ============================================================
        with ctx.branch("stopgrad_half", "10ch, stop-grad ON, half capacity", total=3):
            ctx.phase = "Build model"
            ctx.create_model(_build_stopgrad_half)
            root_sghalf = ctx.save_checkpoint(
                "stopgrad_half_root",
                description="EXPERIMENT: 10ch, stop-grad ON, half capacity, untrained",
            )

            ctx.phase = "Fork and attach tasks"
            ctx.fork(root_sghalf, "stopgrad_half")
            _attach_curriculum_tasks(ctx, mnist, thickness_ds, slant_ds)

            ctx.phase = f"Train ({TOTAL_STEPS} steps)"
            ctx.train(steps=TOTAL_STEPS, lr=1e-3)
            ctx.save_checkpoint(
                "stopgrad_half_trained",
                description=f"EXPERIMENT: 10ch, stop-grad ON, half capacity, trained {TOTAL_STEPS} steps",
            )

            ctx.phase = "Evaluate"
            m_sghalf = ctx.evaluate()
            ctx.record_results("stopgrad_half", m_sghalf)
            ctx.log(f"stopgrad_half: {m_sghalf}")

        # ============================================================
        # Summary
        # ============================================================
        ctx.phase = "Complete"
        ctx.log(f"STOP-GRAD COMPARISON @ {TOTAL_STEPS} steps, uniform sampling:")
        ctx.log(f"  baseline_nofree (20ch, no detach): {m_baseline}")
        ctx.log(f"  stopgrad_20ch   (20ch, detach):    {m_sg20}")
        ctx.log(f"  stopgrad_half   (10ch, detach):    {m_sghalf}")
