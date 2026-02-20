"""Gradient Gating Level 0 — Linear Autoencoder gating comparison.

CGG experiment: linear AE on MNIST comparing gating mechanisms.
5 conditions:
  - control:   no gating (standard SGD)
  - nbr-k8:    neighborhood gating (k=8) — per-image competition
  - nbr-k16:   neighborhood gating (k=16) — larger neighborhoods
  - pca-k8:    neighborhood gating (k=8) + residual PCA replacement every 5 epochs
  - pca-k16:   neighborhood gating (k=16) + residual PCA replacement every 5 epochs

Evaluates reconstruction quality AND specialization metrics
(weight diversity, activation sparsity, effective rank) for each.
Training-time metrics (assignment entropy, gradient CV, dead features,
gini, replacement events) tracked live via FeatureHealthTracker.

All conditions run as a single ctx.train() call (one job, full 25k steps
in the dashboard). Epoch-boundary logic (health tracking, PCA replacement)
fires inside the training_metrics_fn at step boundaries.
"""

from acc.recipes.base import Recipe, RecipeContext
from acc.models.linear_ae import LinearAutoencoder
from acc.gradient_gating import attach_neighborhood_gating, ResidualPCAReplacer
from acc.training_metrics import FeatureHealthTracker
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
REPLACEMENT_INTERVAL = 5  # epochs between PCA replacements


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
    with_pca: bool = False,
) -> dict:
    """Run a neighborhood-gated condition with optional PCA replacement.

    Runs the FULL training in a single ctx.train() call (one job, full
    loss history in the dashboard). Epoch-boundary logic (health tracking,
    PCA replacement) fires inside the training_metrics_fn at step boundaries.
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

    # State for epoch-boundary logic inside training_metrics_fn.
    # Mutable container so the closure can modify it.
    state = {"replacer": None, "epoch": -1}

    def metrics_fn(step: int):
        """Called after every backward pass. Delegates to gating metrics,
        then runs epoch-boundary logic at the right step boundaries."""
        result = gating.record_step_metrics(step)

        # Check for epoch boundary
        epoch = (step - 1) // EPOCH_LENGTH
        if epoch > state["epoch"]:
            state["epoch"] = epoch
            epoch_summary = health.end_epoch(epoch)

            # Create replacer after first epoch (optimizer now exists)
            if with_pca and state["replacer"] is None and ctx._api.trainer is not None:
                named_mods = dict(ctx._api.autoencoder.named_modules())
                state["replacer"] = ResidualPCAReplacer(
                    encoder_layer=named_mods[ENCODER_LAYER],
                    decoder_layer=named_mods[DECODER_LAYER],
                    optimizer=ctx._api.trainer.model_optimizer,
                )

            # Periodic PCA replacement
            replacer = state["replacer"]
            if (with_pca and replacer is not None
                    and epoch > 0 and epoch % REPLACEMENT_INTERVAL == 0):
                win_rates = health.get_win_rates()
                replacements = replacer.check_and_replace(
                    win_rates=win_rates,
                    dataloader=mnist.train_loader(batch_size=BATCH_SIZE),
                    model=ctx._api.autoencoder,
                    device=ctx._api.trainer.device,
                )
                if replacements:
                    health.record_replacements(replacements, epoch)
                    ctx.log(f"{tag} epoch {epoch}: replaced {len(replacements)} features: "
                            f"{[r['dead_idx'] for r in replacements]}")

            if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
                ctx.log(f"{tag} epoch {epoch}: dead={epoch_summary['dead_count']} "
                        f"gini={epoch_summary['gini']} top5={epoch_summary['top5_share']}")

        return result

    ctx.phase = f"Train ({total_steps} steps)"
    ctx.train(
        steps=total_steps,
        lr=LR,
        batch_size=BATCH_SIZE,
        training_metrics_fn=metrics_fn,
    )

    ctx.phase = "Evaluate"
    metrics = ctx.evaluate()

    # Log lifecycle summary
    statuses = health.get_feature_statuses()
    status_counts = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    ctx.log(f"{tag} feature statuses: {status_counts}")

    if with_pca and health.replacement_log:
        ctx.log(f"{tag} total replacements: {len(health.replacement_log)}")
        success_rate = health.get_replacement_success_rate()
        if success_rate is not None:
            ctx.log(f"{tag} replacement success rate (5-epoch lookback): {success_rate:.1%}")

    gating.remove()
    return metrics


class GradientGatingL0(Recipe):
    name = "gradient_gating_l0"
    description = (
        "Linear AE on MNIST: control vs neighborhood gating (k=8, k=16) "
        "vs neighborhood + residual PCA replacement (k=8, k=16). "
        "Compares reconstruction + specialization + feature lifecycle."
    )

    def run(self, ctx: RecipeContext) -> None:
        ctx.phase = "Load MNIST 28x28"
        mnist = ctx.load_dataset("mnist_28", lambda: load_mnist(image_size=28))

        total_steps = NUM_EPOCHS * EPOCH_LENGTH
        results = {}
        tags = ["control", "nbr-k8", "nbr-k16", "pca-k8", "pca-k16"]

        # ── Condition 1: Control (no gating, standard SGD) ────────────
        with ctx.branch("control", "No gating (standard SGD)", total=5):
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

        # ── Condition 2: Neighborhood gating k=8 (no replacement) ─────
        with ctx.branch("nbr-k8", "Neighborhood gating (k=8)", total=5):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            health = FeatureHealthTracker(num_features=HIDDEN_DIM, summary_every=EPOCH_LENGTH)
            metrics = _run_gated_condition(ctx, "nbr-k8", mnist, health, neighborhood_k=8)
            ctx.record_results("nbr-k8", metrics)
            ctx.save_checkpoint("nbr-k8", description="Neighborhood gating (k=8)", eval_results=metrics)
            results["nbr-k8"] = metrics

        # ── Condition 3: Neighborhood gating k=16 (no replacement) ────
        with ctx.branch("nbr-k16", "Neighborhood gating (k=16)", total=5):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            health = FeatureHealthTracker(num_features=HIDDEN_DIM, summary_every=EPOCH_LENGTH)
            metrics = _run_gated_condition(ctx, "nbr-k16", mnist, health, neighborhood_k=16)
            ctx.record_results("nbr-k16", metrics)
            ctx.save_checkpoint("nbr-k16", description="Neighborhood gating (k=16)", eval_results=metrics)
            results["nbr-k16"] = metrics

        # ── Condition 4: Neighborhood gating k=8 + PCA replacement ────
        with ctx.branch("pca-k8", "Neighborhood + PCA (k=8)", total=5):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            health = FeatureHealthTracker(num_features=HIDDEN_DIM, summary_every=EPOCH_LENGTH)
            metrics = _run_gated_condition(ctx, "pca-k8", mnist, health, neighborhood_k=8, with_pca=True)
            ctx.record_results("pca-k8", metrics)
            ctx.save_checkpoint("pca-k8", description="Neighborhood + PCA (k=8)", eval_results=metrics)
            results["pca-k8"] = metrics

        # ── Condition 5: Neighborhood gating k=16 + PCA replacement ───
        with ctx.branch("pca-k16", "Neighborhood + PCA (k=16)", total=5):
            ctx.phase = "Build model"
            ctx.create_model(_build_linear_ae)

            health = FeatureHealthTracker(num_features=HIDDEN_DIM, summary_every=EPOCH_LENGTH)
            metrics = _run_gated_condition(ctx, "pca-k16", mnist, health, neighborhood_k=16, with_pca=True)
            ctx.record_results("pca-k16", metrics)
            ctx.save_checkpoint("pca-k16", description="Neighborhood + PCA (k=16)", eval_results=metrics)
            results["pca-k16"] = metrics

        # Summary
        ctx.phase = "Complete"
        ctx.log(f"GRADIENT GATING L0 COMPARISON @ {NUM_EPOCHS} epochs:")
        for tag in tags:
            ctx.log(f"  {tag}: {results.get(tag, 'N/A')}")
