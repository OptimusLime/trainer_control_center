# Competitive Gradient Gating: Experiment Plan

## Summary

Competitive gradient gating (CGG) is a transparent modification to gradient flow: during backpropagation, scale each channel's gradient by how strongly that channel activated during the forward pass, so strongly-activated weights receive proportionally more gradient. The hypothesis is that this produces more diverse, more specialized hidden representations without changing the loss function or architecture.

This plan integrates CGG into ACC as a progression of four experiment levels (linear AE → conv AE → conv VAE → factor-slot VAE), each comparing standard SGD vs gated training. Every level produces quantitative metrics (weight diversity, activation sparsity, effective rank) alongside reconstruction quality, all visible in the dashboard.

## Context & Motivation

The existing ACC pipeline assumes a fixed model class (`FactorSlotAutoencoder`) with architectural gradient isolation (`detach_factor_grad`). CGG is a different, orthogonal approach: instead of detaching factor slices at the forward level, it modulates gradient magnitude at the backward level based on activation strength. If it works, it could synergize with the factor-slot architecture (Level 3) — architectural isolation in the latent, gradient competition in the encoder.

To test this rigorously, we need:
1. **New model classes** (linear AE, conv AE) that still conform to the `ModelOutput` protocol
2. **New evaluation metrics** (weight diversity, activation sparsity, effective rank) as first-class `EvalMetric` members
3. **New task types** that compute these metrics as evaluation-only tasks (no training loss, just measurement)
4. **A gradient gating module** that attaches to any `nn.Module` via hooks, invisible to the Trainer
5. **A recipe** that orchestrates the full experiment: build both conditions, train both, evaluate both, compare

## Design Constraints

### Invisible to the Trainer

The Trainer is a "dumb pipe." It calls `autoencoder.forward()`, gets a `dict[ModelOutput, Tensor]`, passes it to `task.compute_loss()`, calls `.backward()`, steps optimizers. **CGG must not change any of this.** The gating hooks attach to the model before training starts and fire transparently during backward. The Trainer never knows they exist.

### Invisible to the Task protocol

Tasks call `_get_latent(model_output)`, compute a loss, return a scalar. The new evaluation-only tasks follow the same `Task` interface — they just return `{}` from `compute_loss()` (zero loss, no training signal) and do their real work in `evaluate()`.

### New models are just nn.Modules

The linear AE and conv AE implement the same `forward() -> dict[ModelOutput, Tensor]` protocol. The Trainer, Tasks, CheckpointStore, and dashboard work with them unchanged. The models register no special behavior — they're just simpler architectures.

### Recipe-driven experimentation

The recipe creates both conditions (standard + gated), trains both, evaluates both, records results. The existing `ctx.branch()` mechanism handles this naturally — each condition is a branch.

## Naming Conventions

- **Module:** `acc/gradient_gating.py` — the `CompetitiveGradientGating` class and `attach_competitive_gating()` helper
- **Models:** `acc/models/linear_ae.py`, `acc/models/conv_ae.py`, `acc/models/conv_vae.py`
- **Eval tasks:** `acc/tasks/weight_diversity.py`, `acc/tasks/activation_sparsity.py`, `acc/tasks/effective_rank.py`
- **New EvalMetric members:** `WEIGHT_COSINE_SIM`, `ACTIVATION_SPARSITY`, `EFFECTIVE_RANK`, `SPARSITY_VARIANCE`
- **Recipe:** `acc/recipes/gradient_gating_experiment.py`
- **Config key in ModelOutput:** models expose `config()` returning `{"class": "LinearAutoencoder", "gating": {"temperature": 1.0, ...}}` or `{"gating": None}`

---

## Milestones

### M-CGG-1: Linear Autoencoder End-to-End

**Functionality:** I can run a recipe that trains a linear autoencoder on MNIST with and without gradient gating, and see reconstruction quality for both conditions in the dashboard comparison summary.

**Foundation:** `LinearAutoencoder` model class conforming to `ModelOutput` protocol. `CompetitiveGradientGating` class with tensor-level hooks. `attach_competitive_gating()` helper. Recipe uses `ctx.branch()` for standard vs gated conditions.

- `LinearAutoencoder(in_dim, hidden_dim)` — `Linear(784, 64) → ReLU → Linear(64, 784) → Sigmoid`. Forward returns `{LATENT: hidden_activations, RECONSTRUCTION: output}`.
- `CompetitiveGradientGating` — forward hook on any module that registers a `tensor.register_hook()` to scale gradients by activation-gated mask. Parameters: `temperature`, `gate_strength`.
- `attach_competitive_gating(model, temperature, layer_configs)` — walks named modules, attaches gating per config.
- Recipe: `gradient_gating_l0` — loads MNIST flat, creates two branches (standard, gated), trains 5000 steps each, evaluates reconstruction (L1, PSNR), records results.
- **Implements:** `acc/gradient_gating.py` (`CompetitiveGradientGating`, `attach_competitive_gating`), `acc/models/linear_ae.py` (`LinearAutoencoder`), initial recipe.
- **Proves:** Gradient hooks fire correctly during recipe-driven training. Model conforms to `ModelOutput` protocol. Both conditions train and converge. Dashboard shows comparison.

**Verification:**
- Run `gradient_gating_l0` recipe from dashboard
- Both branches complete without error
- Recipe comparison table shows L1 and PSNR for both conditions
- Gated reconstruction MSE is within 2x of standard (not diverging)

### M-CGG-1 Cleanup Audit
- [ ] Does `LinearAutoencoder.config()` return serializable dict? (needed for checkpoint metadata)
- [ ] Is `CompetitiveGradientGating.remove()` called anywhere? (hook cleanup for model reuse)

---

### M-CGG-2: Specialization Metrics

**Functionality:** I can see weight diversity, activation sparsity, and effective rank for any trained model — both during evaluation and in the recipe comparison table. I can answer "did gating produce more specialized representations?" with numbers, not vibes.

**Foundation:** Three new `EvalMetric` members (`WEIGHT_COSINE_SIM`, `ACTIVATION_SPARSITY`, `EFFECTIVE_RANK`, `SPARSITY_VARIANCE`). Three new evaluation-only task classes that measure model internals. These are **library abstractions** — they work with any model that has Linear or Conv2d layers, not just our specific architectures.

- `WeightDiversityTask(name, dataset, layer_name)` — at eval time, extracts named layer's weight matrix, computes mean pairwise cosine similarity. Lower = more diverse. Returns `{WEIGHT_COSINE_SIM: float}`.
- `ActivationSparsityTask(name, dataset, layer_name)` — at eval time, runs test images through model, captures activations via forward hook, computes fraction of channels above mean. Lower = sparser. Also reports variance across images (higher variance = different images activate different subsets). Returns `{ACTIVATION_SPARSITY: float, SPARSITY_VARIANCE: float}`.
- `EffectiveRankTask(name, dataset, layer_name)` — at eval time, extracts weight matrix, computes SVD, returns `exp(entropy(normalized_singular_values))`. Higher = more dimensions actively used. Returns `{EFFECTIVE_RANK: float}`.
- All three are **evaluation-only**: `compute_loss()` returns `torch.tensor(0.0)` (zero loss, no gradient). They attach to the model like any task but contribute no training signal.
- Recipe updated: attach specialization tasks targeting the hidden layer, evaluate after training, comparison table now shows diversity + sparsity + rank alongside reconstruction.
- **Implements:** `acc/tasks/weight_diversity.py`, `acc/tasks/activation_sparsity.py`, `acc/tasks/effective_rank.py`, new `EvalMetric` members, updated recipe.
- **Proves:** Specialization metrics are computable and comparable across conditions. The Task interface is flexible enough for evaluation-only measurement tasks.

**Verification:**
- Run `gradient_gating_l0` recipe
- Comparison table shows: L1, PSNR, weight_cosine_sim, activation_sparsity, sparsity_variance, effective_rank
- Weight cosine similarity for gated < standard (hypothesis direction)
- Activation sparsity for gated < standard (hypothesis direction)
- Effective rank for gated > standard (hypothesis direction)
- If hypothesis direction is wrong, that's a valid scientific result — the metrics themselves must be correct either way

### M-CGG-2 Cleanup Audit
- [ ] Can specialization tasks be attached from the dashboard [+ Task] panel? (layer_name needs a UI input)
- [ ] Should `LossHealth` thresholds be added for the new task types? (they produce zero training loss, health classification is N/A)
- [ ] Are the new EvalMetric members rendering correctly in the dashboard eval table?

---

### M-CGG-2.5a: Training-Time Metrics Infrastructure

**Functionality:** I can watch assignment entropy and per-feature gradient magnitude evolve in real-time during training in the dashboard. If collapse is happening, I see entropy dropping within the first epoch, not after 25k steps of wasted compute.

**Foundation:** `TrainingMetricsAccumulator` protocol — a pluggable object that accumulates per-step data from gating hooks and summarizes at epoch boundaries. Generic infrastructure for ANY training-time metric stream, not just CGG. The `step_info` dict gains an optional `training_metrics` field that flows through `JobManager → SSE → Dashboard` with zero changes to the pipe.

- `TrainingMetricsAccumulator` ABC in `acc/training_metrics.py` with `on_step(step, gate_masks, grad_norms)` and `summarize() -> dict` and `reset()`
- `GatingMetricsAccumulator(TrainingMetricsAccumulator)` — accumulates win counts per feature, computes assignment entropy at epoch boundaries, tracks per-feature gradient norms and gradient CV
- `CompetitiveGradientGating` extended: optional `metrics_accumulator` field, calls `accumulator.on_step()` after each backward pass with gate masks
- `Trainer.train()` extended: after `loss.backward()`, if model has gating with accumulator, calls `accumulator.on_step()`; periodically (every `metrics_every` steps, default=epoch length) injects `training_metrics` dict into `step_info`
- `step_info` dict gains optional `training_metrics: {assignment_entropy, gradient_cv, per_feature_grad_norms, win_counts}` field
- Dashboard: new training-metrics chart panel showing entropy + gradient CV curves alongside loss curve
- **Implements:** `acc/training_metrics.py` (`TrainingMetricsAccumulator`, `GatingMetricsAccumulator`), extensions to `CompetitiveGradientGating`, `Trainer.train()` extension, dashboard training-metrics panel
- **Proves:** Training-time metric data flows end-to-end: accumulator → gating → Trainer → JobManager → SSE → dashboard chart. Validates against known-broken softmax gating (entropy drops to ~0 for hard temperature).

**Verification:**
- Run current temperature sweep recipe
- Dashboard shows entropy curve dropping to ~0 for hard-t0.1 condition (confirming known collapse)
- Dashboard shows entropy staying high (~0.85+) for soft-t1.0 condition
- Gradient CV visible, high for hard conditions, low for soft/control
- No regression: recipe still completes, comparison table still works

### M-CGG-2.5a Cleanup Audit
- [ ] Is `TrainingMetricsAccumulator` generic enough for non-gating metrics (e.g., learning rate schedules, weight norm tracking)?
- [ ] Does the `training_metrics` field bloat the SSE stream? Should it be a separate endpoint?
- [ ] Should training metrics be persisted in checkpoint alongside eval results?

---

### M-CGG-2.5b: Neighborhood Gating Mechanism

**Functionality:** I can run a recipe with neighborhood-based gating and see whether it avoids the mode collapse that softmax gating produces. The recipe has 3+ conditions: control, softmax-t1.0 (old mechanism for reference), neighborhood (new mechanism). Training-time diagnostics from 2.5a show entropy staying high throughout.

**Foundation:** `NeighborhoodGating` class in `gradient_gating.py` — a drop-in alternative to `CompetitiveGradientGating` using SOM-style weight similarity neighborhoods instead of softmax activation competition. Same hook interface, same `describe()` output, same `remove()` cleanup, same `TrainingMetricsAccumulator` integration. The `attach_competitive_gating` function gains a `mechanism` parameter (`"softmax"` or `"neighborhood"`).

- `NeighborhoodGating` — forward hook computes neighborhoods based on weight cosine similarity (recomputed every `recompute_every` steps). Backward hook: for each feature, competition is only against its k nearest neighbors in weight space. Features that are dissimilar from neighbors but still activated get proportionally more gradient. This creates diversity pressure instead of winner-take-all.
- Parameters: `neighborhood_k` (default 8), `recompute_every` (default 50 steps), `similarity_temperature` (default 1.0)
- Implements `TrainingMetricsAccumulator` protocol: tracks neighborhood stability (fraction of neighbors unchanged between recomputations), assignment entropy, per-feature gradient norms
- Recipe updated: adds neighborhood condition(s) alongside existing softmax conditions
- **Implements:** `NeighborhoodGating` in `acc/gradient_gating.py`, updated recipe with neighborhood conditions
- **Proves:** Neighborhood gating runs without errors, produces non-degenerate gradients, entropy stays high, reconstruction quality comparable to control

**Verification:**
- Run updated recipe with neighborhood condition
- Neighborhood condition: assignment entropy > 0.80 throughout training (visible in 2.5a dashboard panel)
- Neighborhood condition: reconstruction L1 within 1.5x of control
- Neighborhood condition: effective rank > 50/64
- Neighborhood condition: weight features show distinct patterns (not blobs, not collapse) in features panel
- Neighborhood stability metric increases over training (features settling into niches)

### M-CGG-2.5b Cleanup Audit
- [ ] Is `NeighborhoodGating` truly drop-in compatible with `CompetitiveGradientGating`? Can recipe switch between them with one parameter?
- [ ] Should neighborhood_k be adaptive (start large, shrink as features specialize)?
- [ ] Does `recompute_every=50` interact badly with the training loop? Cost of recomputation?

---

### M-CGG-2.75: Neighborhood Gating + Residual PCA Feature Replacement

**Functionality:** I can run a recipe with 3 conditions (control, nbr-k8, pca-k8) and see whether dead features are revived by residual PCA replacement. The dashboard shows gini coefficient, replacement count, and per-epoch dead feature count alongside existing gating metrics. Recipe logs replacement events with success rate tracking.

**Foundation:** `ResidualPCAReplacer` class — periodic dead feature recovery via top-1 PCA of reconstruction errors in the neighborhood with highest error. `FeatureHealthTracker(GatingMetricsAccumulator)` — per-feature lifecycle tracking with win rate history, epoch-level snapshots, replacement event logging, Gini coefficient, stale detection, replacement success rate. Chunked epoch-by-epoch training pattern with loss accumulation across epochs.

- `ResidualPCAReplacer` in `acc/gradient_gating.py` — identifies dead features (win rate < 1%), finds donor neighborhood (highest reconstruction error), computes top-1 PCA of error vectors, replaces encoder+decoder weights, resets Adam state
- `FeatureHealthTracker` in `acc/training_metrics.py` — extends `GatingMetricsAccumulator` with `end_epoch()` for lifecycle tracking, `get_win_rates()` for PCA replacer, `record_replacements()` for event logging, `get_feature_statuses()` for classification, `get_replacement_success_rate()` for efficacy tracking. Overrides `summarize()` to snapshot accumulators before parent resets them (solving the summarize/end_epoch ordering problem)
- `NeighborhoodGating` upgraded to per-image `[B, D]` masks with sigmoid margin competition (from the explorer work — kept the good part, removed dissimilarity floor and explorer routing)
- Recipe `gradient_gating_l0` rewritten: 3 conditions (control, nbr-k8, pca-k8), epoch-by-epoch training with health tracking and PCA replacement every 5 epochs, loss accumulation across chunked training
- Dashboard: `TrainingMetrics` type extended with `gini`, `top5_share`, `replacement_count`. `TrainingMetricsChart` shows gini (0-1 y-axis, dashed yellow) and cumulative replacements (y2 axis, diamond markers)
- **Implements:** `ResidualPCAReplacer`, `FeatureHealthTracker`, updated `NeighborhoodGating`, updated recipe, dashboard updates
- **Proves:** PCA replacement mechanism runs without crashes, produces directional improvement (more living features, higher effective rank), replacement success rate is measurable

**Results (first run):**

| Metric | control | nbr-k8 | pca-k8 |
|--------|---------|--------|--------|
| L1 | 0.068 | 0.084 | 0.086 |
| PSNR | 12.75 | 11.97 | 11.94 |
| weight_cosine_sim | 0.024 | 0.369 | 0.296 |
| activation_sparsity | 0.473 | 0.144 | 0.107 |
| effective_rank | 57.3 | 7.2 | 9.5 |
| dead features | - | 62 | 59 |
| winners | - | 2 | 4 |
| total replacements | - | - | 20 |
| replacement success rate | - | - | 27.8% |

**Observations:** PCA replacement is directionally correct (9.5 vs 7.2 rank, 4 vs 2 winners, lower cosine sim) but the effect is small. Only 2 replacements per cycle because only 2-3 neighborhoods have enough error samples. Replaced features mostly die again within 1-2 epochs — the gating mechanism immediately punishes them for losing their first neighborhood competition. The core problem: entrenched winners capture >99% of images, leaving almost no error samples for other neighborhoods.

**Known issues for next iteration:**
- Replacement scaling: PCA features are scaled to match existing feature magnitudes, but that puts them at the AVERAGE magnitude — they need to be above-average to survive initial competition
- Gating attenuation: dead features should get a grace period after replacement (e.g., gate_strength=0 for 1 epoch) so they can learn without being immediately attenuated
- Min error samples too high: threshold of 10 excludes most neighborhoods; lower to 3-5 or use all training data for error collection
- Only 2 donor neighborhoods available: need to collect errors from ALL neighborhoods, not just the ones with active winners

**Verification:**
- Run `gradient_gating_l0` recipe from dashboard
- All 3 conditions complete without error
- pca-k8 logs replacement events with feature indices and donor neighborhoods
- pca-k8 effective_rank > nbr-k8 effective_rank (directional improvement)
- Dashboard gating metrics chart shows gini and replacement count for gated conditions

### M-CGG-2.75 Cleanup Audit
- [ ] Should `ResidualPCAReplacer` be a generic class that works with any encoder/decoder pair, or is it specific to linear AE?
- [ ] The chunked training + loss accumulation pattern is a workaround — should `RecipeContext.train()` support epoch callbacks natively?
- [ ] `FeatureHealthTracker.summarize()` override to snapshot before reset is fragile — if `summary_every` doesn't align with epoch length, snapshots may be stale. Consider a more robust solution.
- [ ] Explorer routing code was removed but `explorer_graduations` field persists in types/chart — clean up?

---

### M-CGG-2.9: Bidirectional Competitive Learning (BCL)

**Functionality:** I can run a recipe with 6 conditions (control, nbr-k8, pca-k8, bcl-lr005, bcl-lr01, bcl-lr001) and see whether BCL solves the dead feature problem without global gradient floor or PCA hackery. The dashboard shows the per-feature signal scatter (grad vs SOM magnitude — the single most important diagnostic), unreachable feature count, win rate heatmap, and feature velocity alongside existing gating metrics. I can watch feature weight snapshots evolve and see losers actively migrating toward useful input regions.

**Foundation:** `BCL` class — a single-hook mechanism that attaches to any `nn.Linear` layer. One forward hook computes competition and SOM targets, then registers ONE backward hook that does both gradient masking AND SOM weight update in a single location. No `apply_som()`, no `post_step_fn`, no bifurcation. `get_step_metrics()` returns per-step [D]-tensor diagnostics. This is a **library abstraction** — one class, two knobs (`temperature`, `som_lr`), no external dependencies. `BCLHealthTracker(FeatureHealthTracker)` in `training_metrics.py` accumulates the per-step metrics for dashboard streaming and epoch-level analysis.

**Why:** Our experiments showed:
1. Neighborhood gating produces sharp winners that beat control on reconstruction (pca-k8 achieved L1=0.049, 23% better than control).
2. But ~50% of features die because losers get no useful signal.
3. The 10% gradient floor works numerically but is a corrupting diffuse signal — it's just softened SGD.
4. PCA replacement creates alien blob features that freeze in place.

The core problem: losers have no LOCAL signal telling them where to go. BCL gives them one — the raw input that their neighbor's winner claimed, weighted by how NOVEL that image is. This creates two complementary pressures:
- **Winners freeze** via novelty attenuation (their territory becomes crowded, gradient signal drops).
- **Losers explore** via novelty amplification (novel images = uncovered territory, SOM signal is strongest there).

No global SGD. No gradient floor. No PCA. No explorer routing. One class. Two knobs.

**The Algorithm (BCL class):**

The BCL class attaches a single forward hook to an `nn.Linear` layer. The hook:

1. **Captures** `input[0].detach()` (the layer's input, for SOM targets) and `output.detach()` (activations, for competition).
2. **Recomputes neighborhoods** every `recompute_every` steps: cosine similarity of weight rows, top-k neighbors per feature.
3. **Local competition**: for each (image, feature), computes `margin = my_strength - max_neighbor_strength`, then `rank_score = sigmoid(margin * temperature)`. This is the same sigmoid margin as `NeighborhoodGating`.
4. **Image novelty**: `crowding = rank_score.sum(dim=1)` per image (how many features claim this image). `novelty = normalize(1/crowding)`, clamped to `novelty_clamp` (default 3.0). Novel images are uncrowded — few features claim them.
5. **Backward hook** (single hook does both gradient masking AND SOM update):
   - Computes `grad_mask = rank_score * novelty` for winners.
   - Computes SOM targets for losers: identify which features are in the per-image winner's neighborhood. SOM weight = `(1 - rank_score) * in_neighborhood * novelty`. Compute weighted average of inputs: `som_targets = normalized_som_weight.T @ layer_input` → [D, in_features].
   - Registers ONE backward hook on the output tensor that does both:
     ```python
     def _backward_hook(grad, mask, module, som_targets, som_lr):
         # SOM update for losers (safe: gradient already computed from unmodified weights)
         with torch.no_grad():
             module.weight += som_lr * (som_targets - module.weight)
         # Gradient mask for winners (returned to autograd)
         return grad * mask
     ```
   - This is safe because the backward hook fires AFTER the gradient is computed from the original (unmodified) forward-pass weights. Modifying `module.weight` here does not corrupt the gradient — it only affects the NEXT forward pass. One hook, one location, no bifurcation.
6. **`get_step_metrics()`**: returns per-step diagnostics from the last forward pass: `win_rate[D]`, `grad_magnitude[D]`, `som_magnitude[D]`, `mean_activation[D]`, `internal_diversity[D]`, `mean_novelty`, `novelty_std`, `mean_crowding`, `crowding_std`.

**Parameters:**
- `neighborhood_k`: int = 8 (number of weight-space neighbors per feature)
- `temperature`: float = 5.0 (sigmoid sharpness for margin competition)
- `som_lr`: float = 0.005 (SOM learning rate for losers)
- `novelty_clamp`: float = 3.0 (max novelty multiplier)
- `recompute_every`: int = 50 (steps between neighborhood recomputation)

**Training loop integration:**
```python
bcl = BCL(model.encoder[0], neighborhood_k=8, som_lr=0.005)
for step, batch in enumerate(loader):
    out = model(batch)
    loss = F.mse_loss(out, batch)
    optimizer.zero_grad()
    loss.backward()        # backward hook does BOTH: gradient masking + SOM weight update
    optimizer.step()       # Adam updates (winners effectively)
    # No apply_som() needed — SOM update already happened in the backward hook
```

**Changes from current codebase:**

1. **New class** `BCL` in `acc/gradient_gating.py`:
   - Self-contained: one `__init__`, one forward hook (which registers a backward hook each pass), one `get_step_metrics()`, one `remove()`. No `apply_som()` — SOM update happens inside the backward hook alongside gradient masking.
   - Does NOT inherit from `NeighborhoodGating`. Independent implementation. Shares the neighborhood computation pattern but the hook logic is fundamentally different (novelty modulation, in-hook SOM update, metrics capture).
   - `BCLConfig` dataclass for parameters: `neighborhood_k`, `temperature`, `som_lr`, `novelty_clamp`, `recompute_every`.
   - `attach_bcl(model, layer_configs, ...)` convenience function parallel to `attach_neighborhood_gating()`.

2. **New class** `BCLHealthTracker(FeatureHealthTracker)` in `acc/training_metrics.py`:
   - Accepts per-step metrics from `BCL.get_step_metrics()` via a new `record_bcl_step(metrics_dict)` method.
   - Accumulates per-epoch:
     - `grad_magnitude_sum[D]`, `som_magnitude_sum[D]` — for the signal scatter plot.
     - `unreachable_count` — features where both grad and SOM magnitude are below threshold.
     - `win_rate_history[D, epochs]` — for the win rate heatmap.
     - `feature_velocity[D]` — weight change between epoch start/end snapshots.
     - `internal_diversity_mean` — mean neighborhood diversity.
   - Overrides `summarize()` to inject BCL metrics into the dashboard stream: `som_magnitude_mean`, `grad_magnitude_mean`, `unreachable_count`, `mean_novelty`, `mean_crowding`, `internal_diversity`.
   - Overrides `end_epoch()` to snapshot epoch-level signal scatter data and velocity.

3. **Recipe update**: `gradient_gating_l0.py`:
   - New helper `_run_bcl_condition(ctx, tag, mnist, health, som_lr)`:
     - Creates `BCL` instance on encoder layer.
     - BCL's backward hook handles both gradient masking AND SOM update — no `apply_som()` call needed, no `post_step_fn` on Trainer. The `training_metrics_fn` calls `bcl.get_step_metrics()` → feeds to `BCLHealthTracker.record_bcl_step()`, then returns metrics summary when due.
     - Snapshot recorder wired in (reuses existing `FeatureSnapshotRecorder`).
     - Epoch boundary logic (health tracking) same pattern as `_run_gated_condition`.
   - **No Trainer changes needed.** SOM update happens inside the backward hook (fires after gradients are computed from unmodified weights, so weight modification is safe). One hook, one location, no bifurcation. The existing `training_metrics_fn` callback is sufficient for metrics collection.
   - 3 new conditions:
     - `bcl-lr005`: BCL k=8, som_lr=0.005
     - `bcl-lr01`: BCL k=8, som_lr=0.01
     - `bcl-lr001`: BCL k=8, som_lr=0.001
   - Trim old conditions to 3 (control, nbr-k8, pca-k8) for comparison. Total: 6 conditions.

4. **Dashboard**:
   - `TrainingMetrics` type extended with: `som_magnitude_mean`, `grad_magnitude_mean`, `unreachable_count`, `mean_novelty`, `mean_crowding`, `internal_diversity`.
   - `TrainingMetricsChart` gains new datasets: unreachable count (red, y2 axis), SOM magnitude (green dashed, y2 axis).
   - **New panel: Signal Scatter** — fetched on demand, shows grad_magnitude[D] vs som_magnitude[D] as a scatter plot at a selected step. Color = win_rate. This is the single most important BCL diagnostic.
   - Feature snapshot timeline already available from prior work.

**Implements:**
- `BCLConfig` dataclass in `acc/gradient_gating.py`
- `BCL` class in `acc/gradient_gating.py` (forward hook + backward hook for both gradient masking and SOM update)
- `attach_bcl()` helper function in `acc/gradient_gating.py`
- `BCLHealthTracker(FeatureHealthTracker)` in `acc/training_metrics.py`
- `_run_bcl_condition()` helper in `acc/recipes/gradient_gating_l0.py`
- 3 new conditions in `GradientGatingL0.run()`
- Dashboard type + chart extensions + signal scatter panel

**Metrics Tiers:**

Tier 1 — Must-have (validates whether BCL works at all):

| # | Metric | Source | Visualization | Target |
|---|--------|--------|---------------|--------|
| 1 | Reconstruction L1 + PSNR | Loss during training | Line plot (existing) | L1 < 0.050, PSNR > 14.5 |
| 2 | Per-feature signal scatter | `get_step_metrics()` grad_magnitude vs som_magnitude | Scatter plot, color=win_rate, at steps [500, 5000, 15000, 25000] | Two populations: top-left (winners) + bottom-right (losers). No dead zone (low grad, low SOM). |
| 3 | Unreachable feature count | Features where grad < threshold AND SOM < threshold | Line plot over steps | < 5 by step 10000 |
| 4 | Weight visualization 8x8 | Encoder weights reshaped to 28x28 | Feature grid with status-colored borders | > 50 features with recognizable structure |

Tier 2 — Important (validates dynamics are healthy):

| # | Metric | Source | Visualization |
|---|--------|--------|---------------|
| 5 | Win rate heatmap | `win_rate[D]` every 100 steps | [D x steps] heatmap — bright bands = stable winners, brightening = recovering losers |
| 6 | Neighborhood internal diversity | `internal_diversity[D]` | Two lines: mean diversity of top-20 vs bottom-20 features |
| 7 | Image crowding distribution | `mean_crowding`, `crowding_std` | Line plot over steps |
| 8 | Feature velocity | Weight snapshots, per-feature L2 distance | [D x steps] heatmap — all features should be nonzero |

Tier 3 — Nice-to-have (deep diagnostics):

| # | Metric | Tests |
|---|--------|-------|
| 9 | SOM target coherence within dead neighborhoods | H3: dead neighborhoods translate as blob? |
| 10 | Gradient norm vs SOM displacement ratio | H5: SOM dominates for dead features? |
| 11 | Win source entropy (per-feature, by digit class) | Specialist vs generic? |

**Implementation Scope for M-CGG-2.9:** Tier 1 (all 4) + Tier 2 items 5 and 8 (win rate heatmap and feature velocity, both already partially supported by existing `FeatureHealthTracker` and `FeatureSnapshotRecorder`). Tier 2 items 6-7 and Tier 3 are deferred to analysis after first run — we capture the raw data (via `get_step_metrics()`) but don't build dashboard panels until we know what we need.

**Hypotheses to test:**

- **H1**: Winners specialize and freeze via novelty attenuation. Evidence: win rate heatmap shows stable bright bands; feature velocity for top-20 decreases.
- **H2**: Near-winner losers find sub-niches via novelty-weighted SOM. Evidence: features brightening in win rate heatmap; signal scatter shows migration from SOM-dominant to gradient-dominant.
- **H3** (known risk): Dead neighborhoods translate coherently as a blob without dispersing. Evidence: all dead features move same direction at same speed. If confirmed, need per-feature noise injection in next iteration.
- **H6** (optimistic): Per-image novelty creates enough asymmetry to break dead neighborhoods over time. Evidence: internal diversity of dead neighborhoods INCREASES over training.

**Verification:**
- Run `gradient_gating_l0` recipe from dashboard
- All 6 conditions complete without error (control, nbr-k8, pca-k8, bcl-lr005, bcl-lr01, bcl-lr001)
- BCL conditions log per-step metrics (grad_magnitude, som_magnitude, unreachable count)
- Dashboard gating metrics chart shows unreachable_count and som_magnitude_mean
- Signal scatter panel shows two populations (winners top-left, losers bottom-right) — no dead zone
- Feature snapshot timeline shows loser features visibly moving over time (not frozen)
- At least one BCL condition beats pca-k8 on L1 AND has < 5 unreachable features
- If ALL BCL conditions fail, document which hypothesis failed and why

**Implementation order (subtasks):**

1. `BCLConfig` dataclass and `BCL` class in `acc/gradient_gating.py` — the core mechanism (forward hook computes competition + SOM targets, backward hook applies both gradient mask + SOM weight update)
2. `attach_bcl()` convenience function
3. `BCLHealthTracker(FeatureHealthTracker)` in `acc/training_metrics.py` — accumulates `get_step_metrics()` output
4. `_run_bcl_condition()` in recipe — wires BCL hook + health tracker + snapshot recorder
5. Wire 3 BCL conditions into `GradientGatingL0.run()`, trim to 6 total
6. Dashboard: `TrainingMetrics` type extensions + chart datasets
7. Dashboard: Signal scatter panel (grad vs SOM per feature)
8. Run recipe, verify all 6 conditions complete, analyze results against hypotheses

### M-CGG-2.9 Results (BCL Experiment v1)

**Run date:** 2026-02-20. Recipe: 4 conditions (control, nbr-k8, bcl-slow som_lr=0.001, bcl-med som_lr=0.005). All linear AE 784→64→784, Adam lr=1e-3, 25000 steps, batch 128.

| Metric | control | nbr-k8 | bcl-slow | bcl-med | Target |
|--------|---------|--------|----------|---------|--------|
| L1 | **0.062** | 0.068 | 0.115 | 0.109 | < 0.050 |
| PSNR | 13.27 | 12.75 | 10.61 | 10.78 | > 14.5 |
| weight_cosine_sim | 0.014 | 0.091 | 0.510 | 0.812 | low |
| activation_sparsity | 0.507 | 0.223 | 0.141 | 0.428 | - |
| effective_rank | 54.6 | 49.0 | **1.8** | **6.1** | > 50 |
| winners (>5% wr) | - | - | 2 | 4 | > 40 |
| dead (<1% wr) | - | - | 62 | 59 | < 10 |
| unreachable | - | - | 55 | 48 | < 5 |
| dead cos sim (final) | - | - | - | 0.767 | < 0.1 |

**Verdict: FAIL.** BCL v1 causes catastrophic collapse:

1. **H3 confirmed (blob translation):** Dead features all get SOM-pulled toward similar inputs. Weight cosine similarity 0.51-0.81 (near-identical features). Effective rank collapses to 1.8-6.1.
2. **Unreachable features:** 48-55/64 features get neither meaningful gradient NOR meaningful SOM signal. The SOM mechanism doesn't reach them.
3. **Stronger SOM = worse collapse:** bcl-med (som_lr=0.005) has HIGHER cosine similarity (0.812) than bcl-slow (0.510). More SOM = more blob.
4. **Root cause identified:** `som_targets = som_pull.T @ layer_input` computes a weighted average of RAW inputs. All dead features in similar neighborhoods see the same weighted average → move to the same place → blob.

**What's needed:** The SOM target must be DIFFERENT for each feature. Each feature needs to move toward inputs with the bully direction (the direction dominated by its strongest competitors) projected out. This is M3-0.

### M-CGG-2.9 Cleanup Audit
- [x] SOM is applied inside the backward hook (BEFORE optimizer.step()) — does Adam then overwrite the SOM update? Answer: partially. For losers with near-zero gradient, Adam's update is near-zero, so SOM dominates. For winners with strong gradient, Adam dominates. This is the desired behavior. CONFIRMED in experiment: SOM dominates losers, gradient dominates winners. The PROBLEM is that SOM pulls all losers to the same place.
- [ ] `novelty_clamp=3.0` — moot until blob translation is fixed.
- [ ] Should BCL reset Adam state for features that received large SOM displacement? — moot until SOM targets diverge.
- [x] Memory: `input[0].detach()` capture is [B, 784] per forward pass = 100KB at B=128. Confirmed negligible.
- [ ] Decoder BCL: Not included. Encoder BCL must work first.

---

### M3-0: Bully-Adjusted SOM Targets

**Functionality:** BCL's SOM mechanism gives each dead feature a UNIQUE target by projecting out the "bully direction" — the blend of competitor weight vectors that dominate it. Each feature moves toward the parts of the input that its bullies DON'T cover, breaking the blob translation that killed BCL v1.

**Foundation:** Surgical modification to `BCL._forward_hook` in `acc/gradient_gating.py`. Steps 1-5 (competition, novelty, gradient mask) and Step 9 (backward hook) are UNCHANGED. The change replaces the SOM target computation (old lines 730-739) with three new phases.

**Why this fixes the blob:** In BCL v1, all dead features in the same neighborhood computed `som_targets = weighted_avg(raw_inputs)` — the same weighted average, the same target, the same blob. In M3-0, each feature f first computes its "bully direction" (a blend of the neighbor weight vectors that beat it most), then projects that direction out of each input image before averaging. Feature 12's bully is different from Feature 31's bully, so their adjusted inputs are different, so their SOM targets diverge.

**The Algorithm (Steps 6-8, replacing old SOM target computation):**

```
STEP 6: WHO BEATS YOU AND BY HOW MUCH?
  # Feature 12 looks at its 8 neighbors across the whole batch.
  # Neighbor 7 beats feature 12 on 60 images, average margin 0.5
  # Neighbor 19 beats feature 12 on 40 images, average margin 0.3
  # Neighbor 3 beats feature 12 on 10 images, average margin 0.1
  # Feature 12's bully direction = weighted blend of these neighbors'
  # weight vectors, weighted by how badly they beat it.
  # This points toward "where the features that dominate me live."
  for each feature f:
    for each neighbor i:
      how_much_i_beats_f = mean over batch of max(0, strength[i] - strength[f])
    normalize into weights summing to 1
    bully_direction[f] = normalize(weighted sum of neighbor weight vectors)

STEP 7: ADJUST INPUTS BY REMOVING BULLY DIRECTION
  # Feature 12's bully direction points toward "generic 7 detector."
  # For each image, remove that direction:
  #   Image 45 is a 7. Remove the generic-7-detector component.
  #   What's left: the parts of this 7 that the bullies miss.
  #   The serif. The slant. The unusual thickness.
  #   THIS is where feature 12 should move to become unique.
  #
  # Every feature gets its OWN adjusted version of every image
  # because every feature has different bullies.
  overlap[image, feature] = dot(X[image], bully_direction[feature])
  adjusted_input[image, feature] = X[image] - overlap * bully_direction[feature]

STEP 8: COMPUTE SOM TARGETS FOR LOSERS (modified)
  # Same weighting as BCL v1 (1-rank_score, novelty, in_neighborhood)
  # but uses adjusted_input instead of raw layer_input.
  # Each feature's target is now UNIQUE because each sees different
  # adjusted inputs (different bully projections removed).
  pull[image, feature] = (1 - rank_score) * novelty * in_neighborhood
  pull = pull / sum_over_images(pull)
  target[feature] = weighted average of adjusted_input using pull weights
```

**What stays the same:**
- `BCLConfig` — no new parameters (bully direction derived from existing competition data)
- `BCLHealthTracker` — same metrics pipeline. Scatter/winrate/diversity still work.
- `get_step_metrics()` — add `bully_magnitude` to metrics dict for diagnostics
- Recipe — same 4 conditions (control, nbr-k8, bcl-slow, bcl-med)
- API endpoint `/eval/bcl/diagnostics` — unchanged
- Dashboard `BCLDiagnostics.tsx` — unchanged (same scatter/heatmap/diversity tabs)
- Backward hook — identical structure (`module.weight += som_lr * (target - module.weight)`)

**Memory cost:** `adjusted_input` is [B, D, in_features] = [128, 64, 784] = 25.2 MB float32. Fine for GPU with MNIST. For larger models, would need chunked computation over features.

**Files to modify:**
1. `acc/gradient_gating.py` — `BCL._forward_hook`: replace lines 730-739 with Steps 6-7-8 (~30 lines of tensor ops)
2. `docs/CGG_PLAN.md` — this milestone spec (done)

**Implementation (tensor ops for Steps 6-7-8):**

```python
# --- Step 6: Bully direction per feature ---
# neighbor_strengths is already [B, D, k] from Step 3
# strength is [B, D]
strength_exp = strength.unsqueeze(2).expand_as(neighbor_strengths)  # [B, D, k]
beat_margin = (neighbor_strengths - strength_exp).clamp(min=0)     # [B, D, k]
beat_mean = beat_margin.mean(dim=0)  # [D, k] — avg margin per neighbor

beat_weights = beat_mean / (beat_mean.sum(dim=1, keepdim=True) + 1e-8)  # [D, k]

W = module.weight.detach()  # [D, in_features]
neighbor_weights = W[neighbors]  # [D, k, in_features]
bully_raw = torch.einsum('dk,dki->di', beat_weights, neighbor_weights)  # [D, in_features]
bully_direction = F.normalize(bully_raw, dim=1)  # [D, in_features]

# --- Step 7: Adjusted inputs per feature ---
# layer_input is [B, in_features], bully_direction is [D, in_features]
overlap = layer_input @ bully_direction.T  # [B, D]
# adjusted_input[b, d, :] = layer_input[b, :] - overlap[b, d] * bully_direction[d, :]
adjusted_input = layer_input.unsqueeze(1) - overlap.unsqueeze(2) * bully_direction.unsqueeze(0)
# [B, D, in_features]

# --- Step 8: SOM targets using adjusted inputs ---
som_weight = (1.0 - rank_score) * in_nbr * novelty.unsqueeze(1)  # [B, D]
som_norm = som_weight.sum(dim=0, keepdim=True) + 1e-8  # [1, D]
som_pull = som_weight / som_norm  # [B, D]
# Per-feature weighted average of adjusted inputs
som_targets = torch.einsum('bd,bdi->di', som_pull, adjusted_input)  # [D, in_features]
```

**Hypotheses:**
- **H-M3-1:** Bully projection breaks blob symmetry. Dead feature cosine similarity should decrease from 0.77 → < 0.3 by end of training.
- **H-M3-2:** More features become reachable. Unreachable count should drop from 48-55 → < 20.
- **H-M3-3:** Reconstruction quality improves. L1 should drop from 0.109-0.115 → < 0.080 (closer to control's 0.062).
- **H-M3-4:** Effective rank increases. From 1.8-6.1 → > 20.

**Verification:**
- Run recipe with same 4 conditions
- BCL conditions: dead feature cosine similarity < 0.3 (H-M3-1)
- BCL conditions: unreachable < 20 (H-M3-2)
- BCL conditions: L1 < 0.080 (H-M3-3)
- BCL conditions: effective rank > 20 (H-M3-4)
- Signal scatter shows two populations (winners top-left, losers bottom-right) not a blob at origin
- If bully adjustment helps but doesn't fully solve: next step is per-feature noise or stochastic SOM targets

### M3-0 Cleanup Audit
- [ ] The `adjusted_input` tensor [B, D, 784] is 25 MB. For conv layers with larger in_features, this needs chunking.
- [ ] `bully_direction` is recomputed every forward pass. Could cache and recompute every `recompute_every` steps like neighborhoods, but the cost is dominated by the einsum, not the bully computation.
- [ ] Should `bully_magnitude` (per-feature norm of bully_raw before normalization) be tracked? High bully_magnitude = feature is heavily dominated. Low = feature is competitive. Could be a useful diagnostic.
- [ ] What if a feature has NO neighbors that beat it (all beat_margin = 0)? Then bully_direction is zero vector, adjusted_input = raw input, SOM target = old behavior. This is correct — winners shouldn't have their SOM targets adjusted.

---

### M-CGG-2.5c: Enriched Eval Metrics + Feature Utilization Map

**Functionality:** I can see Hoyer sparsity, top-k concentration, per-feature selectivity, and the feature-utilization heatmap in the dashboard for any checkpoint. The eval panel becomes a comprehensive specialization report — answering not just "is it specialized?" but "how is it specialized?" and "what did each feature learn?"

**Foundation:** Extended `EvalMetric` enum with `HOYER_SPARSITY`, `TOP_K_CONCENTRATION`, `FEATURE_SELECTIVITY`. New `FeatureUtilizationTask` (eval-only) producing a [D, 10] heatmap with per-feature selectivity scores. All library-generic — work with any model that has a hookable hidden layer.

- `ActivationSparsityTask` extended: adds Hoyer sparsity measure `(sqrt(D) - L1/L2) / (sqrt(D) - 1)` and top-k concentration (k=4, 8, 16)
- `EffectiveRankTask` extended: adds condition number `S[0]/S[-1]`, returns singular value array for SV plot
- `WeightDiversityTask` extended: adds std, max, min of pairwise cosine similarities
- New `FeatureUtilizationTask(EvalOnlyTask)` — runs test data through model, groups activations by class label (0-9), produces [D, 10] mean activation matrix, computes per-feature selectivity (1 - normalized_row_entropy), returns `{FEATURE_SELECTIVITY: mean_selectivity}` plus the raw heatmap data
- Dashboard: heatmap panel for feature utilization ([D, 10] grid), enriched eval comparison table with all new metrics, weight features panel gains hierarchical clustering sort option
- **Implements:** Extended eval tasks, `acc/tasks/feature_utilization.py`, new `EvalMetric` members (`HOYER_SPARSITY`, `TOP_K_CONCENTRATION`, `FEATURE_SELECTIVITY`), dashboard panels
- **Proves:** Comprehensive eval battery works as drop-in tasks. Feature utilization map shows clear block-diagonal structure for specialized models vs uniform rows for unspecialized.

**Verification:**
- Load any checkpoint from the temperature sweep recipe
- Eval panel shows Hoyer sparsity, top-k concentration, feature selectivity alongside existing metrics
- Feature utilization heatmap renders as [D, 10] grid in dashboard
- For collapsed models (hard-t0.1): heatmap shows 1-2 bright rows, rest dim; selectivity near 0
- For control: heatmap shows moderate variation; selectivity moderate
- For neighborhood gating (if 2.5b complete): heatmap shows distinct per-feature preferences; selectivity > 0.4

### M-CGG-2.5c Cleanup Audit
- [ ] Should the feature utilization heatmap be a separate panel or integrated into the existing features panel?
- [ ] Is hierarchical clustering sort worth the complexity, or is sorting by selectivity score sufficient?
- [ ] Do the new EvalMetric members render correctly in the sibling comparison table?

---

### M-CGG-2.5/2.75/2.9 Dependency and Relationship to Existing Plan

```
M-CGG-1: Linear AE + Gating mechanism + Recipe ✅
    ↓
M-CGG-2: Specialization metrics (weight diversity, sparsity, rank) ✅
    ↓
M-CGG-2.5a: Training-time metrics infrastructure (entropy, gradient CV, dashboard panel) ✅
    ↓
M-CGG-2.5b: Neighborhood gating mechanism (the fix for mode collapse) ✅
    ↓
M-CGG-2.75: Residual PCA replacement (dead feature recovery) ✅
    ↓
M-CGG-2.9: Bidirectional Competitive Learning (BCL) — local SOM signal for losers ✅ (FAILED: blob translation)
    ↓
M3-0: Bully-Adjusted SOM Targets — fix blob translation with per-feature bully projection
    ↓
M-CGG-2.5c: Enriched eval metrics + feature utilization map
    ↓
M-CGG-3: Conv AE + depth-dependent gating
    ...
```

**Note on M-CGG-5 (Periodic Training Metrics):** M-CGG-2.5a subsumes most of what M-CGG-5 was designed to do. After 2.5a, M-CGG-5 reduces to "periodic eval snapshots" (running full eval tasks every N steps during training) — a small delta on top of the training-time metrics stream. M-CGG-5 may be absorbed entirely or reduced to a minor enhancement.

---

### M-CGG-3: Convolutional Autoencoder + Depth-Dependent Gating

**Functionality:** I can run the same gating experiment on a convolutional autoencoder with depth-dependent gate strength (stronger near input, weaker near latent). The comparison shows whether gating works with spatial features and whether depth tapering matters.

**Foundation:** `ConvAutoencoder` model class with Conv2d encoder and ConvTranspose2d decoder, conforming to `ModelOutput` protocol. `attach_competitive_gating` already supports `layer_configs` with per-layer `gate_strength` — this milestone validates that design with conv layers.

- `ConvAutoencoder(in_channels, hidden_dim, image_size)` — Conv(1,32,3,s2,p1) → ReLU → Conv(32,64,3,s2,p1) → ReLU → Flatten → Linear(64\*7\*7, 64) encoder; mirror decoder with ConvTranspose. Forward returns `{LATENT, RECONSTRUCTION}`.
- Recipe: `gradient_gating_l1` — two branches: standard vs gated. Gated applies `gate_strength=1.0` on conv1, `gate_strength=0.5` on conv2. Same training schedule, same evaluation.
- Specialization tasks: `WeightDiversityTask` now handles Conv2d (flatten kernels), `ActivationSparsityTask` handles 4D activations. These are the same task classes from M-CGG-2, just exercised with conv layers.
- Recipe evaluates both conv layers separately (weight diversity per layer, sparsity per layer).
- **Implements:** `acc/models/conv_ae.py` (`ConvAutoencoder`), `gradient_gating_l1` recipe. Possible extensions to specialization tasks for Conv2d if needed.
- **Proves:** Gating mechanism works with Conv2d layers (4D activations, spatial gradient masks). Depth-dependent gate_strength produces measurably different results at different layers.

**Verification:**
- Run `gradient_gating_l1` recipe
- Both branches converge (reconstruction quality comparable)
- Comparison shows kernel diversity and sparsity per layer
- Conv1 (stronger gating) shows larger diversity difference than conv2 (weaker gating) — or documents that it doesn't

### M-CGG-3 Cleanup Audit
- [ ] Are the specialization tasks generic enough for both Linear and Conv2d, or did we add Conv2d-specific code paths?
- [ ] Does `ConvAutoencoder.config()` capture all architecture hyperparams?
- [ ] Any common encoder/decoder code between LinearAutoencoder and ConvAutoencoder that should be shared?

---

### M-CGG-4: Convolutional VAE + KL Interaction

**Functionality:** I can run the gating experiment on a VAE and see whether gradient gating interacts well with KL regularization. The comparison shows KL per dimension alongside specialization metrics — answering whether gating reduces posterior collapse.

**Foundation:** `ConvVAE` model class with reparameterization trick, conforming to `ModelOutput` protocol (outputs `MU`, `LOGVAR`). This validates that gating is compatible with the VAE training regime (KL + reconstruction).

- `ConvVAE(in_channels, latent_dim, image_size)` — same conv encoder but final layer outputs `mu(latent_dim)` and `logvar(latent_dim)`. Reparameterization. ConvTranspose decoder. Forward returns `{LATENT, RECONSTRUCTION, MU, LOGVAR}`.
- Recipe: `gradient_gating_l2` — standard VAE vs gated VAE. Gating on encoder conv layers only (decoder is not gated — gating is about encoder specialization). 10000 steps, β=0.05 for KL.
- KLDivergenceTask already exists and works with any model producing MU/LOGVAR. It slots in with zero new code.
- New metric opportunity: **per-dimension KL** to measure how many latent dims are "alive" (not collapsed to prior). This could be a new eval task or an extension of KLDivergenceTask.
- **Implements:** `acc/models/conv_vae.py` (`ConvVAE`), `gradient_gating_l2` recipe.
- **Proves:** Gating is compatible with the VAE objective. Gating hooks don't interfere with reparameterization gradient flow.

**Verification:**
- Run `gradient_gating_l2` recipe
- Both branches converge
- Comparison shows L1, PSNR, KL, weight diversity, sparsity, effective rank
- KL values are reasonable (not collapsed, not exploded) for both conditions

### M-CGG-4 Cleanup Audit
- [ ] Is there shared code between ConvAutoencoder and ConvVAE that should be factored out?
- [ ] Should per-dimension KL be a separate task or an extension of KLDivergenceTask?
- [ ] How does gating interact with KL warmup? (annealing schedule)

---

### M-CGG-5: Tracking Metrics During Training

**Functionality:** I can see how specialization metrics evolve DURING training, not just after. The dashboard shows weight diversity, activation sparsity, and effective rank updating periodically as training progresses — answering "when does specialization emerge?" and "does it emerge faster with gating?"

**Foundation:** Periodic evaluation hooks in the training loop. The existing `on_step` callback is extended or complemented by a periodic eval mechanism that runs specialization tasks every N steps without stopping training. These are lightweight measurements (no gradient, small batch) that can run between training steps.

- **Approach:** The recipe inserts periodic evaluation points: every `eval_every` steps (e.g., every 500), pause training briefly, run specialization tasks on a small batch, log the metrics. These appear in the job's loss/metric stream alongside per-step training losses.
- **Step info extension:** Add optional `eval_metrics` field to step info dict. The SSE stream already serializes whatever is in the dict. Dashboard JS needs minor update to display periodic eval data.
- **Or:** Simpler approach — the recipe breaks training into chunks (`ctx.train(500)` repeated), runs eval between chunks, logs metrics via `ctx.log()` or a new `ctx.record_metrics()`. The recipe comparison shows the final metrics; the dashboard's job history shows intermediate snapshots.
- **Implements:** `RecipeContext.evaluate_periodically()` or manual chunked-training pattern in recipe. Dashboard updates to display periodic metrics if needed.
- **Proves:** We can observe specialization dynamics, not just endpoints.

**Verification:**
- Run `gradient_gating_l0` recipe with periodic eval (every 500 steps for 5000 total = 10 measurement points)
- Dashboard shows metric evolution: weight diversity should decrease over training for gated condition
- Can visually confirm "specialization starts at step ~X" from the metric timeline

### M-CGG-5 Cleanup Audit
- [ ] Is periodic eval generic enough for all future recipes, or is it specific to this experiment?
- [ ] Should periodic metrics be persisted in checkpoint metadata?
- [ ] Does chunked training (multiple `ctx.train()` calls) interact correctly with optimizer state?

---

### M-CGG-6: Factor-Slot VAE + Gating (Conditional on L0-L2 Success)

**Functionality:** I can run gradient gating on the full `FactorSlotAutoencoder` and see whether encoder gradient competition combined with architectural factor isolation produces better disentanglement than either alone. Four conditions: standard, gating-only, detach-only, gating+detach.

**Foundation:** No new abstractions — this milestone is pure composition of everything built in M-CGG-1 through M-CGG-5. If the foundations are right, this is "just another recipe."

- Recipe: `gradient_gating_l3` — four branches:
  1. Standard (no gating, no detach)
  2. Gating only (competitive gating on encoder conv layers)
  3. Detach only (`detach_factor_grad=True`, existing mechanism)
  4. Gating + Detach (both)
- Uses `FactorSlotAutoencoder` directly — no new model code.
- Evaluation: all existing tasks (recon, KL, classification, regression) + all specialization tasks (weight diversity, sparsity, rank) + traversal quality.
- **Implements:** `gradient_gating_l3` recipe only.
- **Proves:** The full system composes cleanly. CGG foundations from M-CGG-1 work with the existing model without modification.

**Verification:**
- Run `gradient_gating_l3` recipe (will take significant time — 4 branches × full training)
- Comparison table shows all metrics across 4 conditions
- Traversal quality (visual) compared across conditions
- Scientific conclusion: does gating+detach outperform either alone?

### M-CGG-6 Cleanup Audit
- [ ] After 6 milestones, are there any one-off patterns that should become library abstractions?
- [ ] Is the `gradient_gating.py` module clean enough to be reusable for other gradient modification experiments?
- [ ] Should `attach_competitive_gating` become a model config option (like `detach_factor_grad`) rather than a recipe-level call?

---

## Milestone Dependencies

```
M-CGG-1: Linear AE + Gating mechanism + Recipe ✅
    ↓
M-CGG-2: Specialization metrics (weight diversity, sparsity, rank) ✅
    ↓
M-CGG-2.5a: Training-time metrics infrastructure (entropy, gradient CV) ✅
    ↓
M-CGG-2.5b: Neighborhood gating mechanism (fix for mode collapse) ✅
    ↓
M-CGG-2.75: Residual PCA replacement (dead feature recovery) ✅
    ↓
M-CGG-2.9: Bidirectional Competitive Learning (local SOM for losers) ✅ (FAILED: blob translation)
    ↓
M3-0: Bully-Adjusted SOM Targets (fix blob with per-feature bully projection)
    ↓
M-CGG-2.5c: Enriched eval metrics + feature utilization map
    ↓
M-CGG-3: Conv AE + depth-dependent gating
    ↓
M-CGG-4: Conv VAE + KL interaction
    ↓
M-CGG-5: (Subsumed by 2.5a — reduced to periodic eval snapshots if needed)
    ↓
M-CGG-6: Factor-Slot VAE + Gating (conditional on L0-L2 results)
```

M-CGG-1 and M-CGG-2 are the critical pair — confirmed: softmax gating causes mode collapse at sharp temperatures. M-CGG-2.5 is the diagnostic + fix cycle: build the instruments (2.5a), build the new mechanism (2.5b), enrich the measurement battery (2.5c). M-CGG-2.75 added dead feature recovery via PCA (works but feels like hackery). M-CGG-2.9 is the key algorithmic pivot: instead of global gradient floor + PCA replacement, give losers a LOCAL signal via SOM-style weight alignment. Every subsequent milestone adds one architectural element.

## Milestone Status

| Milestone | Status | What I Can Do After |
|-----------|--------|---------------------|
| M-CGG-1: Linear AE End-to-End | **DONE** | Train linear AE with/without gating, see reconstruction comparison |
| M-CGG-2: Specialization Metrics | **DONE** | See weight diversity, sparsity, effective rank in comparison table |
| M-CGG-2.5a: Training-Time Metrics | **DONE** | Watch entropy + gradient CV evolve in real-time during training |
| M-CGG-2.5b: Neighborhood Gating | **DONE** | Run neighborhood gating, see per-image competition with sigmoid margins |
| M-CGG-2.75: Residual PCA Replacement | **DONE** | Run PCA replacement for dead features, see replacement events + success rate |
| M-CGG-2.9: Bidirectional Competitive Learning | **DONE (FAILED)** | BCL v1 causes blob translation (H3). Dead features collapse to cos_sim 0.77. Effective rank 1.8-6.1. |
| M3-0: Bully-Adjusted SOM Targets | Not started | Fix blob translation by projecting out bully direction from SOM targets |
| M-CGG-2.5c: Enriched Eval Metrics | Not started | See Hoyer sparsity, top-k, selectivity, feature utilization heatmap |
| M-CGG-3: Conv AE + Depth Gating | Not started | Run gating on conv layers with depth-dependent strength |
| M-CGG-4: Conv VAE + KL | Not started | Test gating compatibility with VAE objective |
| M-CGG-5: Periodic Training Metrics | Subsumed by 2.5a | (Reduced scope: periodic eval snapshots only) |
| M-CGG-6: Factor-Slot + Gating | Not started | Test the full combination: architectural + gradient isolation |

## Parameter Defaults

```yaml
# Shared
dataset: MNIST (28×28 for linear, 32×32 for conv)
batch_size: 128
optimizer: Adam
learning_rate: 1e-3
random_seeds: [42, 123, 456]  # 3 seeds per condition for stability

# Level 0 (Linear AE)
hidden_dim: 64
training_steps: 25000
temperature: 1.0
gate_strength: 1.0  # single hidden layer, full gating

# Level 1 (Conv AE)
training_steps: 5000
temperature: 1.0
gate_strength: {conv1: 1.0, conv2: 0.5}  # depth-tapered

# Level 2 (Conv VAE)
latent_dim: 16
training_steps: 10000
beta: 0.05  # KL weight
temperature: 1.0
gate_strength: {conv1: 1.0, conv2: 0.5}  # encoder only

# Level 3 (Factor-Slot VAE)
# Same as existing FactorSlotAutoencoder config
training_steps: 28110  # 3 epochs × 10 repeats × 937 batches
```

## Sensitivity Sweeps (Post Initial Results)

After M-CGG-2 confirms (or denies) the basic effect:
- Temperature sweep: [0.3, 0.5, 1.0, 2.0, 5.0]
- Gate strength sweep: [0.25, 0.5, 0.75, 1.0]

These can be additional recipe branches or a separate sweep recipe.

## Integration Points (No Changes Needed)

These existing abstractions work as-is:
- **Trainer:** No changes. Gating is invisible (tensor hooks).
- **JobManager / SSE:** No changes. Step info dict is extensible.
- **CheckpointStore:** No changes. Models implement `config()`, checkpoints store metadata.
- **TaskRegistry:** No changes. New tasks auto-discovered from `acc/tasks/`.
- **RecipeRegistry:** No changes. New recipe auto-discovered from `acc/recipes/`.
- **Dashboard recipe panel:** No changes. Branch comparison already shows all eval metrics.
- **LossHealth:** Add thresholds for new task types (evaluation-only tasks always "healthy").

## Integration Points (Changes Needed)

- **`EvalMetric` enum:** Add 4 new members (`WEIGHT_COSINE_SIM`, `ACTIVATION_SPARSITY`, `SPARSITY_VARIANCE`, `EFFECTIVE_RANK`).
- **`acc/models/` directory:** New. Contains `linear_ae.py`, `conv_ae.py`, `conv_vae.py`.
- **`acc/gradient_gating.py`:** New. The gating mechanism.
- **`acc/tasks/`:** 3 new task files for specialization metrics.
- **`acc/recipes/`:** 1-4 new recipe files for experiment levels.
- **`RecipeContext.create_model()`:** Currently takes `builder: Callable[[], nn.Module]`. This is already generic — works with any model class. No change needed.

## Directory Structure (Anticipated)

```
acc/
├── gradient_gating.py              — CompetitiveGradientGating, attach_competitive_gating()
├── models/
│   ├── __init__.py
│   ├── linear_ae.py                — LinearAutoencoder (M-CGG-1)
│   ├── conv_ae.py                  — ConvAutoencoder (M-CGG-3)
│   └── conv_vae.py                 — ConvVAE (M-CGG-4)
├── tasks/
│   ├── weight_diversity.py         — WeightDiversityTask (M-CGG-2)
│   ├── activation_sparsity.py      — ActivationSparsityTask (M-CGG-2)
│   └── effective_rank.py           — EffectiveRankTask (M-CGG-2)
├── recipes/
│   ├── gradient_gating_l0.py       — Linear AE experiment (M-CGG-1)
│   ├── gradient_gating_l1.py       — Conv AE experiment (M-CGG-3)
│   ├── gradient_gating_l2.py       — Conv VAE experiment (M-CGG-4)
│   └── gradient_gating_l3.py       — Factor-Slot experiment (M-CGG-6)
├── eval_metric.py                  — +4 new enum members (M-CGG-2)
└── loss_health.py                  — +thresholds for new task types (M-CGG-2)
```

## How to Verify (Full Experiment)

1. `M-CGG-1:` Run `gradient_gating_l0` recipe → comparison shows L1/PSNR for standard vs gated
2. `M-CGG-2:` Same recipe, now with specialization tasks → comparison includes diversity/sparsity/rank
3. `M-CGG-3:` Run `gradient_gating_l1` recipe → conv results with depth-dependent gating
4. `M-CGG-4:` Run `gradient_gating_l2` recipe → VAE results with KL interaction data
5. `M-CGG-5:` Run any recipe with periodic eval → watch metrics evolve during training
6. `M-CGG-6:` Run `gradient_gating_l3` recipe → 4-condition factor-slot comparison

## Related Documents

- `docs/HOW_WE_WORK.md` — Core principles (dual nature, verification, hypothesis-driven debugging)
- `docs/WRITING_MILESTONES.md` — Milestone structure (functionality-indexed, not backend-indexed)
- `docs/ROADMAP.md` — Project roadmap (M1-M6)
- `docs/ui_refactor_plan.md` — Dashboard refactor (Phases 1-4 done, Phase 5 pending)
