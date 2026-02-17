# ACC Roadmap — Milestones M2 through M6

## Status

| Milestone | Status | Summary |
|-----------|--------|---------|
| M1 | Done | Full training loop from code + UI dashboard |
| M1.5 | Done | Model-agnostic forward protocol (`ModelOutput` dict, `latent_slice`) |
| M1.75 | Done | Factor-Slot autoencoder through same pipeline, gradient isolation |
| M1.9 | Done | Split-machine deployment via Tailscale |
| M1.95 | Next | Recipes + checkpoint tree + experiment runner (absorbs old M4) |
| M2 | Planned | Hot-reload tasks (wiring file watcher to `acc/tasks/`) |
| M3 | Planned | Generator hot-reload + dashboard generator UI |
| M5 | Planned | UFR evaluation + traversal/sort visualizations |
| M6 | Planned | Model expansion (layer addition) |

---

## M1.95: Recipes + Checkpoint Tree + Experiment Runner

**Functionality:** I can select a recipe from the dashboard, click Run, and watch it build a checkpoint tree — forking, configuring, training, and evaluating each branch. The built-in recipe runs the MNIST factor experiment: same FactorSlotAutoencoder forked into two branches (MNIST-only vs MNIST + synthetic curriculum), trained, and evaluated. I compare traversal quality between branches with my eyes.

**Foundation:** `Recipe` base class with tree operations. `RecipeRunner` executes recipes in background thread with SSE progress. `RecipeRegistry` with hot-reload. `CheckpointTree` visualization. `KLDivergenceTask` for VAE regularization. FactorHead reparameterization (mu/logvar). Synthetic generators (thickness, slant). This absorbs the old M4 (checkpoint tree + forking) because recipes require forks to operate.

See `docs/M1.95_PLAN.md` for full details.

**Verification:** `python -m acc.test_m1_95` runs the two-branch experiment programmatically and verifies checkpoint tree structure.

---

## M2: Hot-Reload Tasks

**Functionality:** I can write a new Task subclass in `acc/tasks/`, save it, and it appears in the dashboard's [+ Task] menu without restarting anything.

**Foundation:** `TaskRegistry` with file-watcher-driven `importlib.reload`. Same pattern as `RecipeRegistry` from M1.95, wired to a different directory.

- File watcher on `acc/tasks/` directory in the trainer process
- On `.py` file change: reload module, re-scan for Task subclasses, update registry
- New task class appears in dropdown within 2 seconds of file save
- Existing tasks not disrupted by registry update
- Syntax errors caught, don't crash the trainer

**Verification:**

1. System running from M1.95 (trainer + UI, model loaded, tasks active)
2. Create `acc/tasks/dummy_task.py` with a trivial Task subclass
3. Save → "DummyTask" appears in [+ Task] dropdown within 2 seconds
4. Add DummyTask → train → loss shows in chart
5. Model weights and existing task probes unchanged by the reload

---

## M3: Generator Hot-Reload + Dashboard Generator UI

**Functionality:** I can write a generator in `acc/generators/`, save it, and generate datasets from the dashboard without restarting.

**Foundation:** `GeneratorRegistry` (same pattern as TaskRegistry/RecipeRegistry). Dashboard UI for generator configuration and execution.

- File watcher on `acc/generators/`
- Dashboard: [+ Dataset] → pick generator → configure → [Generate] → dataset appears
- Generated datasets saved to `acc/data/` as .pt files
- Thickness/slant generators already exist from M1.95; this adds the dashboard UI and hot-reload

**Verification:**

1. System running from M2
2. Create `acc/generators/symmetry_gen.py` with a generator that produces symmetric patterns
3. Save → "SymmetryGenerator" appears in [+ Dataset] menu within 2 seconds
4. Generate dataset → create RegressionTask on it → train → MAE shows in eval

---

## M5: UFR Evaluation + Visual Diagnosis

**Functionality:** I can run evaluation on any checkpoint and see traversal grids, sort-by-activation grids, and attention maps in the dashboard. UFR score quantifies what I see.

**Foundation:** `FactoringEvaluator` with configurable concepts/contexts. Traversal grid renderer. Sort-by-activation renderer. Attention map visualizer. These are the eval tools from the MNIST factor experiment, generalized.

- Dashboard button: [Run Eval] on any checkpoint
- Traversal grids: per factor group, 5 seeds × 9 steps
- Sort-by-activation: per factor group, 20 lowest + 20 highest
- Cross-attention maps: per factor, overlaid on input
- UFR score: concept × context transfer matrix

**Verification:**

1. Load mnist_only_5k checkpoint → run eval → see traversals
2. Load curriculum_5k checkpoint → run eval → see traversals
3. Compare: curriculum_5k thickness traversal is cleaner than mnist_only_5k
4. UFR score is higher for curriculum_5k than mnist_only_5k

---

## M6: Model Expansion (Layer Addition)

**Functionality:** I can add layers to a trained model without starting over.

**Foundation:** `Autoencoder.expand()` method. Expansion as a checkpoint operation.

- Dashboard: [Expand Model] → select where to add layer → configure
- Existing weights preserved
- New layer initialized
- All probes detached and re-attached
- New checkpoint created automatically

**Verification:**

1. 1-layer model trained to reasonable accuracy
2. Expand → add layer → accuracy drops (expected)
3. Train → accuracy recovers and exceeds original

---

## The Loop

- M1 establishes the full loop from code
- M1.5/M1.75 prove the pipeline is model-agnostic
- M1.9 enables GPU training from a laptop
- **M1.95 enables experiments: recipes build checkpoint trees, forks isolate variables, visual eval with your eyes**
- M2 makes task creation instant (hot reload)
- M3 makes dataset creation instant (hot reload)
- M5 quantifies what you see (UFR score)
- M6 lets the model grow without starting over
