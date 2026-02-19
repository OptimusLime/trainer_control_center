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
M-CGG-1: Linear AE + Gating mechanism + Recipe
    ↓
M-CGG-2: Specialization metrics (weight diversity, sparsity, rank)
    ↓
M-CGG-3: Conv AE + depth-dependent gating
    ↓
M-CGG-4: Conv VAE + KL interaction
    ↓
M-CGG-5: Periodic eval during training
    ↓
M-CGG-6: Factor-Slot VAE + Gating (conditional on L0-L2 results)
```

M-CGG-1 and M-CGG-2 are the critical pair. If gating doesn't produce specialization in a linear layer with proper measurement, the experiment is falsified at the cheapest possible cost. Every subsequent milestone adds one architectural element.

## Milestone Status

| Milestone | Status | What I Can Do After |
|-----------|--------|---------------------|
| M-CGG-1: Linear AE End-to-End | Not started | Train linear AE with/without gating, see reconstruction comparison |
| M-CGG-2: Specialization Metrics | Not started | See weight diversity, sparsity, effective rank in comparison table |
| M-CGG-3: Conv AE + Depth Gating | Not started | Run gating on conv layers with depth-dependent strength |
| M-CGG-4: Conv VAE + KL | Not started | Test gating compatibility with VAE objective |
| M-CGG-5: Periodic Training Metrics | Not started | Watch specialization emerge during training, not just after |
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
