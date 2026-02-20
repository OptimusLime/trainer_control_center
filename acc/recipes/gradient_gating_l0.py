"""Gradient Gating Level 0 — Linear Autoencoder gating comparison.

BCL experiment v1: linear AE on MNIST comparing 4 conditions:
  - control:    no gating (standard SGD)
  - nbr-k8:     neighborhood gating (k=8) — best previous result
  - bcl-slow:   BCL (k=8, som_lr=0.001) — gentle SOM push for losers
  - bcl-med:    BCL (k=8, som_lr=0.005) — stronger SOM push for losers

Evaluates reconstruction quality AND specialization metrics
(weight diversity, activation sparsity, effective rank) for each.
Training-time metrics tracked live via FeatureHealthTracker (nbr-k8)
and BCLHealthTracker (BCL conditions).

All conditions run as a single ctx.train() call (one job, full 25k steps
in the dashboard). Epoch-boundary logic fires inside training_metrics_fn.
"""

from acc.recipes.base import Recipe, RecipeContext
from acc.models.linear_ae import LinearAutoencoder
from acc.gradient_gating import (
    attach_neighborhood_gating,
    attach_bcl,
    BCLConfig,
)
from acc.training_metrics import (
    FeatureHealthTracker,
    BCLHealthTracker,
    FeatureSnapshotRecorder,
)
from acc.dataset import load_mnist
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.weight_diversity import WeightDiversityTask
from acc.tasks.activation_sparsity import ActivationSparsityTask
from acc.tasks.effective_rank import EffectiveRankTask

IN_DIM = 784       # 28 * 28
HIDDEN_DIM = 64
NUM_EPOCHS = 53     # ~25000 steps at 469 steps/epoch
LR = 1e-3
BATCH_SIZE = 128
ENCODER_LAYER = "encoder.0"  # nn.Linear(784, 64)
DECODER_LAYER = "decoder.0"  # nn.Linear(64, 784)
# MNIST: 60000 / 128 = ~469 steps per epoch
EPOCH_LENGTH = 469



def _build_linear_ae() -> LinearAutoencoder:
    return LinearAutoencoder(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        image_shape=(1, 28, 28),
    )


def _attach_tasks(ctx: RecipeContext, mnist) -> None:
    """Attach the standard eval task battery."""
    ctx.detach_all_tasks()
    ctx.attach_task(ReconstructionTask("recon", mnist))
    ctx.attach_task(WeightDiversityTask("weight_div", mnist, ENCODER_LAYER))
    ctx.attach_task(ActivationSparsityTask("act_sparsity", mnist, ENCODER_LAYER))
    ctx.attach_task(EffectiveRankTask("eff_rank", mnist, ENCODER_LAYER))


def _run_gated_condition(
    ctx: RecipeContext,
    tag: str,
    mnist,
    health: FeatureHealthTracker,
    neighborhood_k: int = 8,
) -> dict:
    """Run a neighborhood-gated condition (no PCA, no BCL).

    Runs the FULL training in a single ctx.train() call. Epoch-boundary
    logic (health tracking) fires inside the training_metrics_fn.
    """
    gating = attach_neighborhood_gating(
        ctx._api.autoencoder,
        layer_configs={ENCODER_LAYER: {"neighborhood_k": neighborhood_k, "gate_strength": 1.0}},
        temperature=5.0,
        recompute_every=50,
        metrics=health,
    )

    _attach_tasks(ctx, mnist)

    total_steps = NUM_EPOCHS * EPOCH_LENGTH

    # Feature weight snapshots
    named_mods = dict(ctx._api.autoencoder.named_modules())
    snapshot_recorder = FeatureSnapshotRecorder(
        encoder_layer=named_mods[ENCODER_LAYER],
        every_n_steps=500,
        image_shape=(28, 28),
    )
    snapshot_recorder.snapshot(0, event="init")

    state = {"epoch": -1}

    def metrics_fn(step: int):
        result = gating.record_step_metrics(step)
        snapshot_recorder.maybe_snapshot(step)

        epoch = (step - 1) // EPOCH_LENGTH
        if epoch > state["epoch"]:
            state["epoch"] = epoch
            epoch_summary = health.end_epoch(epoch)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                ctx.log(f"{tag} epoch {epoch}: dead={epoch_summary['dead_count']} "
                        f"gini={epoch_summary['gini']} top5={epoch_summary['top5_share']}")
        return result

    ctx.phase = f"Train ({total_steps} steps)"
    ctx.train(steps=total_steps, lr=LR, batch_size=BATCH_SIZE, training_metrics_fn=metrics_fn)

    ctx.phase = "Evaluate"
    metrics = ctx.evaluate()

    statuses = health.get_feature_statuses()
    status_counts = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    ctx.log(f"{tag} feature statuses: {status_counts}")

    gating.remove()

    if not hasattr(ctx._api, 'snapshot_recorders'):
        ctx._api.snapshot_recorders = {}  # type: ignore[attr-defined]
    ctx._api.snapshot_recorders[tag] = snapshot_recorder  # type: ignore[attr-defined]
    ctx.log(f"{tag}: captured {snapshot_recorder.num_snapshots()} weight snapshots")

    return metrics


def _run_bcl_condition(
    ctx: RecipeContext,
    tag: str,
    mnist,
    health: BCLHealthTracker,
    som_lr: float = 0.005,
) -> dict:
    """Run a BCL condition.

    BCL's backward hook handles both gradient masking AND SOM weight update.
    No apply_som(), no post_step_fn. The training_metrics_fn collects
    BCL-specific metrics (signal scatter, win rate, dead diversity).
    """
    bcl_config = BCLConfig(
        neighborhood_k=8,
        temperature=5.0,
        som_lr=som_lr,
        novelty_clamp=3.0,
        recompute_every=50,
    )
    bcl = attach_bcl(ctx._api.autoencoder, ENCODER_LAYER, bcl_config)

    _attach_tasks(ctx, mnist)

    total_steps = NUM_EPOCHS * EPOCH_LENGTH

    # Feature weight snapshots
    named_mods = dict(ctx._api.autoencoder.named_modules())
    encoder_layer = named_mods[ENCODER_LAYER]
    snapshot_recorder = FeatureSnapshotRecorder(
        encoder_layer=encoder_layer,
        every_n_steps=500,
        image_shape=(28, 28),
    )
    snapshot_recorder.snapshot(0, event="init")

    state = {"epoch": -1}

    def metrics_fn(step: int):
        # Collect BCL step metrics
        bcl_metrics = bcl.get_step_metrics()
        if bcl_metrics is not None:
            health.record_bcl_step(
                step=step,
                metrics=bcl_metrics,
                encoder_weight=encoder_layer.weight,
            )

        snapshot_recorder.maybe_snapshot(step)

        # Check if parent accumulator wants to summarize
        if health.should_summarize(step):
            result = health.summarize()
        else:
            result = None

        # Epoch boundary
        epoch = (step - 1) // EPOCH_LENGTH
        if epoch > state["epoch"]:
            state["epoch"] = epoch
            epoch_summary = health.end_epoch(epoch)
            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                ctx.log(f"{tag} epoch {epoch}: dead={epoch_summary['dead_count']} "
                        f"gini={epoch_summary['gini']} top5={epoch_summary['top5_share']}")

        return result

    ctx.phase = f"Train ({total_steps} steps)"
    ctx.train(steps=total_steps, lr=LR, batch_size=BATCH_SIZE, training_metrics_fn=metrics_fn)

    ctx.phase = "Evaluate"
    metrics = ctx.evaluate()

    statuses = health.get_feature_statuses()
    status_counts = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    ctx.log(f"{tag} feature statuses: {status_counts}")

    # Log BCL-specific summary
    scatter = health.signal_scatter_log
    if scatter:
        last = scatter[-1]
        unreachable = sum(
            1 for g, s in zip(last['grad_magnitude'], last['som_magnitude'])
            if g < 0.1 and s < 0.1
        )
        ctx.log(f"{tag} final scatter: {len(scatter)} snapshots, "
                f"unreachable={unreachable}/64 at step {last['step']}")

    diversity = health.dead_diversity_log
    if diversity:
        ctx.log(f"{tag} dead diversity: first={diversity[0]['mean_similarity']:.3f} "
                f"last={diversity[-1]['mean_similarity']:.3f}")

    bcl.remove()

    if not hasattr(ctx._api, 'snapshot_recorders'):
        ctx._api.snapshot_recorders = {}  # type: ignore[attr-defined]
    ctx._api.snapshot_recorders[tag] = snapshot_recorder  # type: ignore[attr-defined]
    ctx.log(f"{tag}: captured {snapshot_recorder.num_snapshots()} weight snapshots")

    # Store BCL health tracker for API access (signal scatter, win rate heatmap)
    if not hasattr(ctx._api, 'bcl_trackers'):
        ctx._api.bcl_trackers = {}  # type: ignore[attr-defined]
    ctx._api.bcl_trackers[tag] = health  # type: ignore[attr-defined]

    return metrics


class GradientGatingL0(Recipe):
    name = "gradient_gating_l0"
    description = (
        "BCL experiment v1: control vs nbr-k8 vs bcl-slow (som_lr=0.001) "
        "vs bcl-med (som_lr=0.005). Tests whether SOM revives dead features."
    )

    def run(self, ctx: RecipeContext) -> None:
        ctx.phase = "Load MNIST 28x28"
        mnist = ctx.load_dataset("mnist_28", lambda: load_mnist(image_size=28))

        total_steps = NUM_EPOCHS * EPOCH_LENGTH
        results = {}
        tags = ["control", "nbr-k8", "bcl-slow", "bcl-med"]

        # ── Condition 1: Control (no gating, standard SGD) ────────────
        with ctx.branch("control", "No gating (standard SGD)", total=4):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            _attach_tasks(ctx, mnist)

            ctx.phase = f"Train ({total_steps} steps)"
            ctx.train(steps=total_steps, lr=LR, batch_size=BATCH_SIZE)

            ctx.phase = "Evaluate"
            metrics = ctx.evaluate()
            ctx.record_results("control", metrics)
            ctx.save_checkpoint("control", description="No gating (standard SGD)", eval_results=metrics)
            ctx.log(f"control: {metrics}")
            results["control"] = metrics

        # ── Condition 2: Neighborhood gating k=8 (baseline for comparison) ─
        with ctx.branch("nbr-k8", "Neighborhood gating (k=8)", total=4):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            health = FeatureHealthTracker(num_features=HIDDEN_DIM, summary_every=EPOCH_LENGTH)
            metrics = _run_gated_condition(ctx, "nbr-k8", mnist, health, neighborhood_k=8)
            ctx.record_results("nbr-k8", metrics)
            ctx.save_checkpoint("nbr-k8", description="Neighborhood gating (k=8)", eval_results=metrics)
            results["nbr-k8"] = metrics

        # ── Condition 3: BCL slow (som_lr=0.001) ─────────────────────
        with ctx.branch("bcl-slow", "BCL (som_lr=0.001)", total=4):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            health = BCLHealthTracker(num_features=HIDDEN_DIM, summary_every=EPOCH_LENGTH)
            metrics = _run_bcl_condition(ctx, "bcl-slow", mnist, health, som_lr=0.001)
            ctx.record_results("bcl-slow", metrics)
            ctx.save_checkpoint("bcl-slow", description="BCL (k=8, som_lr=0.001)", eval_results=metrics)
            results["bcl-slow"] = metrics

        # ── Condition 4: BCL medium (som_lr=0.005) ───────────────────
        with ctx.branch("bcl-med", "BCL (som_lr=0.005)", total=4):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            health = BCLHealthTracker(num_features=HIDDEN_DIM, summary_every=EPOCH_LENGTH)
            metrics = _run_bcl_condition(ctx, "bcl-med", mnist, health, som_lr=0.005)
            ctx.record_results("bcl-med", metrics)
            ctx.save_checkpoint("bcl-med", description="BCL (k=8, som_lr=0.005)", eval_results=metrics)
            results["bcl-med"] = metrics

        # Summary
        ctx.phase = "Complete"
        ctx.log(f"BCL EXPERIMENT v1 @ {NUM_EPOCHS} epochs:")
        for tag in tags:
            ctx.log(f"  {tag}: {results.get(tag, 'N/A')}")
