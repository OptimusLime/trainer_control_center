# ACC Roadmap — Milestones M2 through M7

## Status

| Milestone | Status | Summary |
|-----------|--------|---------|
| M1 | Done | Full training loop from code + UI dashboard |
| M1.5 | Done | Model-agnostic forward protocol (`ModelOutput` dict, `latent_slice`) |
| M1.75 | Done | Factor-Slot autoencoder through same pipeline, gradient isolation |
| M1.9 | Done | Split-machine deployment via Tailscale |
| M1.95 | Done | Recipes + checkpoint tree + experiment runner (absorbed old M4) |
| M2 | Done | Hot-reload tasks + dashboard task management |
| M3 | Done | Generator hot-reload + dataset dashboard |
| M4 | Next | Multi-GPU training + evaluation |
| M5 | Planned | UFR evaluation + visual diagnosis dashboard |
| M6 | Planned | Model expansion (layer addition) |

---

## M2: Hot-Reload Tasks + Dashboard Task Management

**Functionality:** I can write a new Task subclass in `acc/tasks/`, save it, and it appears in the dashboard within 2 seconds. From the dashboard, I can add tasks to the model, see per-task loss curves during training, adjust task weights, see per-task eval metrics, and manage the full task lifecycle without touching code.

**Foundation:** `TaskRegistry` with file-watcher-driven `importlib.reload`. Same pattern as `RecipeRegistry` from M1.95, wired to `acc/tasks/`. Dashboard task management panel with real-time training visibility.

- File watcher on `acc/tasks/` directory in the trainer process
- On `.py` file change: reload module, re-scan for Task subclasses, update registry
- New task class appears in [+ Task] dropdown within 2 seconds of file save
- Existing tasks not disrupted by registry update
- Syntax errors caught, don't crash the trainer
- Dashboard: per-task loss curves (not just aggregate), task weight sliders, add/remove/toggle tasks
- Dashboard: per-task eval metrics displayed after each eval run
- Dashboard: reconstruction comparison (input vs output side-by-side)
- Dashboard: persistent loss history across page reloads (trainer stores, UI fetches)

**Verification:** `python -m acc.test_m2`

1. System running from M1.95 (trainer + UI, model loaded, tasks active)
2. Create `acc/tasks/dummy_task.py` with a trivial Task subclass
3. Save → "DummyTask" appears in [+ Task] dropdown within 2 seconds
4. Add DummyTask from dashboard → train → per-task loss shows in chart
5. Model weights and existing task probes unchanged by the reload
6. Dashboard shows per-task metrics, loss curves, and reconstruction comparison

See `docs/M2_PLAN.md` for full details.

---

## M3: Generator Hot-Reload + Dataset Dashboard

**Functionality:** I can write a generator in `acc/generators/`, save it, and generate datasets from the dashboard without restarting. I can browse dataset samples visually, see target distributions, and generate new datasets with configurable parameters — all from the dashboard.

**Foundation:** `GeneratorRegistry` (same pattern as TaskRegistry/RecipeRegistry). Dashboard dataset management with visual browsing and generator configuration UI.

- File watcher on `acc/generators/`
- Dashboard: [+ Dataset] → pick generator → configure parameters → [Generate] → dataset appears
- Dashboard: dataset sample grid (browse actual images), target distribution histograms
- Dashboard: dataset comparison (side-by-side sample grids from different datasets)
- Generated datasets saved to `acc/data/` as .pt files
- Thickness/slant generators already exist from M1.95; this adds the dashboard UI and hot-reload

**Verification:** `python -m acc.test_m3`

1. System running from M2
2. Create `acc/generators/symmetry_gen.py` with a generator that produces symmetric patterns
3. Save → "SymmetryGenerator" appears in [+ Dataset] menu within 2 seconds
4. Generate dataset from dashboard → browse samples → create RegressionTask on it → train → MAE shows in eval
5. Dataset sample grid shows actual images with targets

---

## M4: Multi-GPU Training + Evaluation

**Functionality:** I can train on one GPU and run evaluation on the other simultaneously, or use both GPUs for faster training via data-parallel.

**Foundation:** `DeviceManager` that manages GPU allocation. Training and evaluation can run concurrently on separate devices.

- 2x NVIDIA RTX 3090 available (currently only using cuda:0)
- Mode 1: Train on cuda:0, eval on cuda:1 (concurrent training + evaluation)
- Mode 2: DataParallel across both GPUs for 2x batch throughput
- Dashboard: device selection, GPU utilization display
- Checkpoint save/load handles multi-GPU state correctly

**Verification:** `python -m acc.test_m4`

1. Train on cuda:0 while running traversal eval on cuda:1 — both complete without blocking
2. DataParallel mode: batch throughput measurably increases vs single GPU
3. Checkpoint from single-GPU loads correctly in multi-GPU mode and vice versa

---

## M5: UFR Evaluation + Visual Diagnosis Dashboard

**Functionality:** I can run evaluation on any checkpoint and see traversal grids, sort-by-activation grids, and attention maps in the dashboard. UFR score quantifies what I see. I can compare checkpoints side-by-side.

**Foundation:** `FactoringEvaluator` with configurable concepts/contexts. Traversal grid renderer. Sort-by-activation renderer. Attention map visualizer. Checkpoint comparison panel. These are the eval tools from the MNIST factor experiment, generalized.

- Dashboard button: [Run Eval] on any checkpoint
- Traversal grids: per factor group, 5 seeds x 9 steps
- Sort-by-activation: per factor group, 20 lowest + 20 highest
- Cross-attention maps: per factor, overlaid on input
- UFR score: concept x context transfer matrix
- Checkpoint comparison: side-by-side traversals from different branches

**Verification:** `python -m acc.test_m5`

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

**Verification:** `python -m acc.test_m6`

1. 1-layer model trained to reasonable accuracy
2. Expand → add layer → accuracy drops (expected)
3. Train → accuracy recovers and exceeds original

---

## The Loop

- M1 establishes the full loop from code
- M1.5/M1.75 prove the pipeline is model-agnostic
- M1.9 enables GPU training from a laptop
- M1.95 enables experiments: recipes build checkpoint trees, forks isolate variables
- **M2 gives you eyes on training: per-task loss curves, metrics, task management from dashboard**
- **M3 gives you eyes on data: dataset browsing, generator UI, visual dataset management**
- M4 uses both GPUs — concurrent train + eval, or faster training
- M5 quantifies what you see (UFR score) and enables checkpoint comparison
- M6 lets the model grow without starting over
