"""Gradient Gating Level 0 — Linear Autoencoder baseline.

Simplest possible CGG experiment: linear AE on MNIST, standard vs gated.
Two branches, same model architecture, same training schedule, only
difference is whether gradient gating is attached to the encoder.

This validates:
1. LinearAutoencoder conforms to ModelOutput protocol
2. CompetitiveGradientGating hooks fire correctly during training
3. Both conditions converge (gated doesn't diverge)
4. Branch comparison shows L1/PSNR for both conditions
"""

from acc.recipes.base import Recipe, RecipeContext
from acc.models.linear_ae import LinearAutoencoder
from acc.gradient_gating import attach_competitive_gating
from acc.dataset import load_mnist
from acc.tasks.reconstruction import ReconstructionTask

IN_DIM = 784       # 28 * 28
HIDDEN_DIM = 64
TRAINING_STEPS = 25000
LR = 1e-3
BATCH_SIZE = 128


def _build_linear_ae() -> LinearAutoencoder:
    return LinearAutoencoder(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        image_shape=(1, 28, 28),
    )


class GradientGatingL0(Recipe):
    name = "gradient_gating_l0"
    description = (
        "Linear AE on MNIST: standard SGD vs competitive gradient gating. "
        "Tests whether activation-proportional gradient scaling changes "
        "reconstruction quality in the simplest possible setting."
    )

    def run(self, ctx: RecipeContext) -> None:
        # === Common setup ===
        ctx.phase = "Load MNIST 28x28"
        mnist = ctx.load_dataset("mnist_28", lambda: load_mnist(image_size=28))

        # ============================================================
        # Branch 0: STANDARD — no gating (control)
        # ============================================================
        with ctx.branch("standard", "Linear AE, no gating (control)", total=2):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            ctx.phase = "Attach reconstruction task"
            ctx.detach_all_tasks()
            ctx.attach_task(ReconstructionTask("recon", mnist))

            ctx.phase = f"Train ({TRAINING_STEPS} steps)"
            ctx.train(
                steps=TRAINING_STEPS,
                lr=LR,
                batch_size=BATCH_SIZE,
            )

            ctx.phase = "Evaluate"
            m_standard = ctx.evaluate()
            ctx.record_results("standard", m_standard)
            ctx.log(f"standard: {m_standard}")

        # ============================================================
        # Branch 1: GATED — competitive gradient gating on encoder
        # ============================================================
        with ctx.branch("gated", "Linear AE, gradient gating on encoder", total=2):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            ctx.phase = "Attach gating"
            # Gate the encoder's Linear layer (encoder.0 is the nn.Linear)
            gating = attach_competitive_gating(
                ctx._api.autoencoder,
                layer_configs={
                    "encoder.0": {"temperature": 1.0, "gate_strength": 1.0},
                },
            )

            ctx.phase = "Attach reconstruction task"
            ctx.detach_all_tasks()
            ctx.attach_task(ReconstructionTask("recon", mnist))

            ctx.phase = f"Train ({TRAINING_STEPS} steps)"
            ctx.train(
                steps=TRAINING_STEPS,
                lr=LR,
                batch_size=BATCH_SIZE,
            )

            ctx.phase = "Evaluate"
            m_gated = ctx.evaluate()
            ctx.record_results("gated", m_gated)
            ctx.log(f"gated: {m_gated}")

            # Cleanup hooks
            gating.remove()

        # ============================================================
        # Summary
        # ============================================================
        ctx.phase = "Complete"
        ctx.log(f"GRADIENT GATING L0 @ {TRAINING_STEPS} steps:")
        ctx.log(f"  standard (no gating): {m_standard}")
        ctx.log(f"  gated (temp=1.0, strength=1.0): {m_gated}")
