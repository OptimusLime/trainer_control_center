"""Gradient Gating Level 0 — Linear Autoencoder temperature sweep.

CGG experiment: linear AE on MNIST with varying gate temperatures.
Four branches, same architecture, same training schedule:
  - control:  no gating
  - soft:     temperature=1.0  (mild competition)
  - medium:   temperature=0.3  (sharp competition)
  - hard:     temperature=0.1  (near winner-take-all)

Evaluates reconstruction quality AND specialization metrics
(weight diversity, activation sparsity, effective rank) for each.
"""

from acc.recipes.base import Recipe, RecipeContext
from acc.models.linear_ae import LinearAutoencoder
from acc.gradient_gating import attach_competitive_gating
from acc.dataset import load_mnist
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.weight_diversity import WeightDiversityTask
from acc.tasks.activation_sparsity import ActivationSparsityTask
from acc.tasks.effective_rank import EffectiveRankTask

IN_DIM = 784       # 28 * 28
HIDDEN_DIM = 64
TRAINING_STEPS = 25000
LR = 1e-3
BATCH_SIZE = 128
ENCODER_LAYER = "encoder.0"  # nn.Linear(784, 64)

# Temperature sweep: (tag, description, temperature_or_None)
# None = no gating (control)
CONDITIONS = [
    ("control",    "No gating (control)",       None),
    ("soft-t1.0",  "Gated, temperature=1.0",    1.0),
    ("med-t0.3",   "Gated, temperature=0.3",    0.3),
    ("hard-t0.1",  "Gated, temperature=0.1",    0.1),
]


def _build_linear_ae() -> LinearAutoencoder:
    return LinearAutoencoder(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        image_shape=(1, 28, 28),
    )


class GradientGatingL0(Recipe):
    name = "gradient_gating_l0"
    description = (
        "Linear AE on MNIST: temperature sweep of competitive gradient gating. "
        "control / soft(t=1.0) / medium(t=0.3) / hard(t=0.1). "
        "Compares reconstruction + specialization metrics."
    )

    def run(self, ctx: RecipeContext) -> None:
        ctx.phase = "Load MNIST 28x28"
        mnist = ctx.load_dataset("mnist_28", lambda: load_mnist(image_size=28))

        n_conditions = len(CONDITIONS)
        results = {}

        for tag, desc, temperature in CONDITIONS:
            with ctx.branch(tag, desc, total=n_conditions):
                ctx.phase = "Build model"
                ctx.create_model(_build_linear_ae)

                # Attach gating if this is a gated condition
                gating = None
                if temperature is not None:
                    ctx.phase = f"Attach gating (t={temperature})"
                    gating = attach_competitive_gating(
                        ctx._api.autoencoder,
                        layer_configs={
                            ENCODER_LAYER: {
                                "temperature": temperature,
                                "gate_strength": 1.0,
                            },
                        },
                    )

                ctx.phase = "Attach tasks"
                ctx.detach_all_tasks()
                ctx.attach_task(ReconstructionTask("recon", mnist))
                ctx.attach_task(WeightDiversityTask("weight_div", mnist, ENCODER_LAYER))
                ctx.attach_task(ActivationSparsityTask("act_sparsity", mnist, ENCODER_LAYER))
                ctx.attach_task(EffectiveRankTask("eff_rank", mnist, ENCODER_LAYER))

                ctx.phase = f"Train ({TRAINING_STEPS} steps)"
                ctx.train(
                    steps=TRAINING_STEPS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                )

                ctx.phase = "Evaluate"
                metrics = ctx.evaluate()
                ctx.record_results(tag, metrics)
                ctx.save_checkpoint(tag, description=desc, eval_results=metrics)
                ctx.log(f"{tag}: {metrics}")
                results[tag] = metrics

                if gating is not None:
                    gating.remove()

        # Summary
        ctx.phase = "Complete"
        ctx.log(f"GRADIENT GATING L0 TEMPERATURE SWEEP @ {TRAINING_STEPS} steps:")
        for tag, _, temp in CONDITIONS:
            t_str = f"t={temp}" if temp is not None else "no gating"
            ctx.log(f"  {tag} ({t_str}): {results[tag]}")
