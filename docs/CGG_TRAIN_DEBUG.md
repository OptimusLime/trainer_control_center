# Step-wise Training Debugger Plan

## Summary

A new dashboard domain for batch-by-batch training inspection. Pause training at any step, advance one batch at a time, see complete per-batch diagnostics (force magnitudes, SOM targets, cosine similarity matrices, win dynamics, feature trajectories), and investigate BCL force dynamics in detail. Built as a generic `DebugController` abstraction that works with any training loop, with BCL-specific panels layered on top.

## Context & Motivation

After 4 BCL iterations (v1 -> M3-0 -> M3-0.1 -> M3-0.2), cosine similarity remains 0.94-0.99. Epoch-level summaries and per-500-step logs can't explain WHY features converge. We need to see what happens **inside a single batch** -- the actual tensors, the force magnitudes, the SOM targets, the weight updates. We need to pause, step one batch, inspect everything, hypothesize, step again.

The current infrastructure has no pause/resume capability. The trainer has a `_stop_requested` boolean that kills training entirely. There's no step-mode, no batch inspection, no way to run diagnostic queries against live state.

### What Exists Today

- `Trainer.train()` loop: `backward() -> training_metrics_fn(step) -> optimizer.step()`. One boolean `_stop_requested` for hard stop. No pause, no resume, no step mode.
- `BCL.get_step_metrics()` returns per-step `[D]`-tensor diagnostics (win_rate, grad_magnitude, som_magnitude, feature_novelty, etc.)
- `BCL._last_metrics` stores per-step `[B, D]` tensors (rank_score, grad_mask, som_weight, strength, in_nbr, feature_novelty)
- `BCLHealthTracker.record_bcl_step()` accumulates metrics, logs sparsely every 500 steps
- Dashboard has `BCLDiagnostics.tsx` (scatter, heatmap, diversity tabs) for post-hoc analysis
- SSE streaming is currently disabled due to lock contention; dashboard polls every 5 seconds

### What We Need

Pause/step/resume with per-batch inspection. Generic infrastructure that works for any training loop, with BCL-specific diagnostic panels built on top.

## Design Constraints

1. **Library-centric.** The `DebugController` is a generic training-loop interceptor. It works with ANY training loop, not just BCL. BCL diagnostics are panels that USE the infrastructure, not the infrastructure itself.

2. **Invisible to the Trainer.** The Trainer doesn't know about debugging. The debug controller is passed as an optional parameter. When `None`, zero overhead. When present, one `Event.is_set()` check per step (nanoseconds).

3. **The Dashboard Is The Tool.** All inspection happens in the dashboard. No ad-hoc scripts, no /tmp images, no console dumps.

4. **End-to-end from M-DBG-1.** First milestone produces a working pause/step/resume with batch visualization. BCL-specific panels come later, building on the foundation.

5. **Complexity over time.** Start with pause + batch images. Add metrics tables. Add force visualizations. Add matrices. Add trajectories. Each milestone verifiable independently.

## Naming Conventions

- **Module:** `acc/debug_controller.py` -- the `DebugController` class
- **API domain:** `/debug/*` endpoints in `acc/trainer_api.py`
- **Dashboard components:** `Debug*.tsx` prefix in `dashboard/src/components/`
- **Store additions:** `$debugState` computed store in `dashboard/src/lib/store.ts`
- **Types:** `DebugState`, `DebugStepData`, `DebugTrajectory` in `dashboard/src/lib/types.ts`

---

## Milestones

### M-DBG-1: Pause, Step, See Batch

**Functionality:** I can start training, click "Pause" in the dashboard, and training stops at the current step. I can then click "Step" to advance exactly one batch. After each step, I see: the batch images (a grid), the loss value, and the step number. I can click "Resume" to continue normal training.

**Foundation:** `DebugController` class -- a generic training-loop interceptor that adds pause/step/resume semantics to any `Trainer.train()` call. Uses `threading.Event` for zero-overhead when not paused. New API endpoints follow the existing pattern (`/debug/*`). Dashboard component follows the React island pattern.

- `DebugController` in `acc/debug_controller.py`:
  - `pause()`, `resume()`, `step()`, `is_paused: bool`
  - `wait_if_paused()` -- called inside the training loop. If not paused, returns immediately (one `Event.is_set()` check). If paused, blocks on `_step_event.wait()`.
  - `capture_batch(batch_tensor, loss, step)` -- stores the last batch + metadata for inspection
  - `get_batch() -> dict` -- returns the last batch tensor + metadata
  - `get_state() -> dict` -- returns `{paused, step, has_batch}`

- **Integration point:** `Trainer.train()` gains an optional `debug_controller: Optional[DebugController]` parameter. After `optimizer.step()` and `on_step()`, calls `debug_controller.wait_if_paused()`. The batch tensor and loss are captured via `debug_controller.capture_batch()` after `loss.backward()`. This is minimal integration -- two lines in the trainer.

- **API endpoints** in `acc/trainer_api.py`:
  - `POST /debug/pause` -- pauses training
  - `POST /debug/resume` -- resumes training
  - `POST /debug/step` -- advances one step while paused
  - `GET /debug/state` -- returns `{paused, step, has_batch}`
  - `GET /debug/batch` -- returns the current batch as base64 image grid + loss value + step number

- **Dashboard:** `DebugPanel.tsx` React island:
  - Pause/Resume/Step buttons (Pause and Resume toggle; Step only enabled when paused)
  - Step counter showing current step
  - Batch image grid (128 MNIST images at 28x28, arranged in rows)
  - Loss value for current step

- **Implements:** `acc/debug_controller.py` (`DebugController`), two-line integration in `Trainer.train()`, 5 API endpoints, `DebugPanel.tsx`
- **Proves:** Pause/step/resume works without deadlocks or race conditions. Training resumes cleanly. Dashboard shows real-time state.

**Verification:**
- Start recipe from dashboard
- Click Pause -- training stops (step counter freezes, no new loss entries in chart)
- Click Step -- step counter advances by 1, batch grid shows 128 MNIST images, loss value updates
- Click Step 5 more times -- each time, new images appear, loss changes
- Click Resume -- training continues at normal speed
- Recipe completes without error
- Run with control condition (no BCL) -- pause/step/resume still works identically

### M-DBG-1 Cleanup Audit
- [ ] Does `DebugController` work with non-BCL training (control condition)?
- [ ] Is `wait_if_paused()` truly zero-overhead when not paused? Profile the `Event.is_set()` call.
- [ ] Does pausing interact correctly with `_stop_requested`? (stop should override pause)
- [ ] Does `capture_batch` clone the tensor or hold a reference? (must clone -- tensor may be overwritten)

---

### M-DBG-2: Per-Step Metrics Snapshot

**Functionality:** When paused and stepping, I can see the full per-step metrics for the current batch: per-feature win_rate, grad_magnitude, som_magnitude, feature_novelty, effective_win, and the three blending weights (gradient_weight, contender_weight, attraction_weight). This is a table of 64 rows (one per feature), sortable by any column, color-coded by feature status.

**Foundation:** `DebugController.capture_step_data(key, data)` -- a generic key-value store for per-step diagnostic data. Any hook or callback can push data into the debug controller during a step. The dashboard reads it via `GET /debug/step_data/{key}`. This is the generic "ask questions about the current step" mechanism -- not BCL-specific.

- `DebugController` extended:
  - `_step_data: dict[str, Any]` -- cleared at the start of each step, populated by hooks/callbacks during the step
  - `capture_step_data(key: str, data: Any)` -- stores data under key
  - `get_step_data(key: str) -> Any` -- retrieves data for a key
  - `list_step_data_keys() -> list[str]` -- what data is available this step

- **BCL integration:** The recipe's `training_metrics_fn` pushes `bcl.get_step_metrics()` into the debug controller via `debug_controller.capture_step_data("bcl_metrics", bcl_metrics)`. Also pushes raw `_last_metrics` tensors (rank_score, grad_mask, etc.) under `"bcl_tensors"`.

- **API endpoints:**
  - `GET /debug/step_data/keys` -- lists available data keys for the current step
  - `GET /debug/step_data/{key}` -- returns the data for that key (tensors serialized to lists)

- **Dashboard:** `DebugMetricsTable.tsx`:
  - 64-row table with columns: feature_id, win_rate, grad_magnitude, som_magnitude, feature_novelty, effective_win, gradient_weight, contender_weight, attraction_weight
  - Sortable by any column (click header)
  - Color-coded: dead features (win_rate < 0.01) red, winners (win_rate > 0.05) green, contenders yellow
  - Summary row: mean, min, max of each column

- **Implements:** `DebugController` step_data mechanism, BCL integration in recipe, 2 API endpoints, `DebugMetricsTable.tsx`
- **Proves:** Generic step-data capture works. Any hook can push data. Dashboard reads and displays it.

**Verification:**
- Pause training, step once
- Debug panel shows 64-row table with all BCL per-feature metrics
- Sort by win_rate -- top features have high win_rate, bottom have near-zero
- Sort by som_magnitude -- dead features have high SOM signal
- Values match what `get_step_metrics()` returns (spot-check: sum grad_magnitude across features matches grad_mask.sum())
- `GET /debug/step_data/keys` returns `["bcl_metrics", "bcl_tensors"]`

### M-DBG-2 Cleanup Audit
- [ ] Is `capture_step_data` generic enough for non-BCL diagnostics (e.g., vanilla gradient norms)?
- [ ] Should large tensor data (rank_score [128, 64]) be serialized lazily (on request) or eagerly (at capture time)?
- [ ] Does clearing step_data at step start interact correctly with the `training_metrics_fn` calling order? (backward -> metrics_fn -> optimizer.step -> wait_if_paused: step_data must be populated before pause)

---

### M-DBG-3: Force Visualization -- Gradient vs SOM Per Feature

**Functionality:** When paused and stepping, I can see a scatter plot of grad_update_magnitude vs som_update_magnitude per feature for THIS batch. I can also see the actual weight update vectors: for a selected feature, the dashboard shows the gradient update direction, the SOM update direction, and the combined update direction as 28x28 images. I can answer: "for feature 17, is the gradient pulling it one way and the SOM pulling it another?"

**Foundation:** `BCL._forward_hook` extended to capture `som_targets` and `som_delta` in `_last_metrics`. Dashboard scatter plot reuses the Chart.js scatter pattern from `BCLDiagnostics.tsx`. Feature detail view reuses the 28x28 weight rendering pattern from `FeatureSnapshotTimeline.tsx`.

- **BCL extension:** Store additional tensors in `_last_metrics`:
  - `som_targets` [D, 784] -- the SOM target weight vectors
  - `som_delta` [D, 784] -- `som_lr * (som_targets - module.weight)`, the actual SOM weight update
  - These are already computed in the backward hook; just need to be captured before application.

- **API endpoints:**
  - `GET /debug/step_data/force_detail?feature={id}` -- returns `{grad_delta: float[784], som_delta: float[784], combined: float[784], current_weight: float[784], cos_grad_som: float}` for the selected feature

- **Dashboard:** `DebugForceViz.tsx`:
  - Scatter plot: x=grad_magnitude, y=som_magnitude per feature, color=win_rate (same axes as BCL scatter but for THIS single batch, not accumulated)
  - Click a point -> shows 4 images (28x28 each): current weight, grad delta, SOM delta, combined delta
  - Cosine similarity between grad and SOM directions displayed as a number per feature
  - Summary: how many features have grad-SOM cosine > 0 (forces agree) vs < 0 (forces fight)

- **Implements:** BCL tensor capture extension in `_last_metrics`, force_detail endpoint, `DebugForceViz.tsx`
- **Proves:** We can see the actual update directions competing for each feature. This is the core diagnostic for understanding BCL convergence.

**Verification:**
- Pause, step, see scatter plot of this batch's forces
- Click feature with high win_rate -- grad delta shows a recognizable pattern (digit-like), SOM delta is near-zero
- Click feature with low win_rate -- grad delta is near-zero, SOM delta shows a pattern pointing toward some input region
- Click a contender feature -- both forces visible, cosine similarity tells us if they agree or fight
- Values are plausible: no NaN, grad_delta norms correlate with grad_magnitude in the table

### M-DBG-3 Cleanup Audit
- [ ] Storing `som_targets` [D, 784] and `som_delta` [D, 784] increases memory. Only capture when debug controller is attached and paused?
- [ ] Should force detail be lazy (computed on request from stored _last_metrics) or eager (captured during step)?
- [ ] Can we reuse `BCLDiagnostics.tsx` scatter chart code, or should `DebugForceViz.tsx` use a shared `ScatterPlot` component? (Library-centric: extract shared component)

---

### M-DBG-4: Pairwise Cosine Similarity Matrix

**Functionality:** When paused, I can request a 64x64 pairwise cosine similarity heatmap of the current encoder weights. I can also request the 64x64 heatmap of the SOM targets -- showing how similar different features' SOM targets are. If SOM targets are near-identical (off-diagonal > 0.9), that's the convergence-to-average problem visualized directly.

**Foundation:** Reuses `DebugController.capture_step_data` from M-DBG-2. Introduces `HeatmapPanel.tsx` -- a generic reusable component that takes any 2D numeric array + labels and renders a color-coded matrix. Library abstraction: works for any square matrix, not just cosine similarity.

- **API endpoints:**
  - `GET /debug/cosine_matrix?source=weights` -- returns 64x64 matrix of pairwise cosine similarities of encoder weight rows
  - `GET /debug/cosine_matrix?source=som_targets` -- returns 64x64 matrix of SOM target pairwise similarities
  - `GET /debug/cosine_matrix?source=grad_delta` -- returns 64x64 matrix of gradient update direction similarities

- **Dashboard:** `HeatmapPanel.tsx` -- generic reusable component:
  - Canvas-rendered NxN heatmap with configurable color scale (blue=0, white=0.5, red=1 default)
  - Hover shows (row, col, value)
  - Dropdown to select source (weights, som_targets, grad_delta)
  - Summary stats below: mean off-diagonal, std, max off-diagonal
  - Sortable: option to reorder rows/cols by a metric (e.g., win_rate) for visual clustering

- **Implements:** 1 parameterized API endpoint, `HeatmapPanel.tsx` (reusable), integration into debug section of dashboard
- **Proves:** We can directly see whether features are converging (high off-diagonal) or diversifying (low off-diagonal). The generic heatmap is reusable for any future matrix visualization (e.g., attention maps, correlation matrices).

**Verification:**
- Pause at step 100, view weight cosine matrix -- should show low off-diagonal similarity (near-random init)
- Step to step 5000, view weight cosine matrix -- shows whatever convergence pattern exists
- View SOM target cosine matrix -- if off-diagonal mean > 0.9, convergence-to-average is confirmed visually
- Compare weight matrix vs SOM target matrix side-by-side -- SOM targets SHOULD be more diverse than weights (they're supposed to pull features apart). If they're less diverse, the algorithm is the problem.
- Sort by win_rate -- check if dead features cluster together in the matrix

### M-DBG-4 Cleanup Audit
- [ ] `HeatmapPanel` -- is it generic enough for non-square matrices? (not needed now, but consider the API)
- [ ] Computing 64x64 cosine similarity is cheap ([64, 784] matrix, ~3ms). For larger models, should this be paginated or downsampled?
- [ ] Should we store historical snapshots of the cosine matrix (e.g., every N steps when stepping) for trajectory analysis? Or is that M-DBG-5's domain?

---

### M-DBG-5: Feature Trajectory Tracker

**Functionality:** I can select 3-5 features and "watch" them across steps. As I step through batches, the dashboard accumulates a trajectory: for each watched feature, it tracks weight cosine similarity to its initial state, pairwise similarity to other watched features, win_rate, and force magnitudes. I can see a feature's journey from dead to alive (or from alive to convergent). The trajectory persists across resume -- I can watch features across an entire training run.

**Foundation:** `DebugController.add_watch(feature_id)` / `remove_watch(feature_id)` -- a watched-feature system. Each step (whether paused-and-stepping or running normally), for watched features, the controller snapshots their weight vectors and key metrics. Generic -- "watch" could apply to any per-feature diagnostic in the future.

- **DebugController extended:**
  - `_watched_features: set[int]` -- which features to track
  - `_watch_trajectories: dict[int, list[dict]]` -- per-feature list of snapshots
  - `_watch_initial_weights: dict[int, Tensor]` -- weight vector at time of watch start
  - `add_watch(feature_id: int)`, `remove_watch(feature_id: int)`, `get_watched() -> list[int]`
  - `get_trajectories() -> dict[int, list[dict]]` -- accumulated trajectory data
  - `clear_trajectories()` -- reset without removing watches
  - Each step, for watched features: append `{step, weight_cos_to_init, win_rate, grad_mag, som_mag, feature_novelty}` to trajectory

- **API endpoints:**
  - `POST /debug/watch` body `{feature_ids: [3, 17, 42]}` -- set watched features (replaces previous set)
  - `GET /debug/watch` -- returns current watched feature IDs
  - `GET /debug/trajectories` -- returns accumulated trajectory data for all watched features
  - `POST /debug/trajectories/clear` -- reset trajectory data

- **Dashboard:** `FeatureTrajectory.tsx`:
  - Feature selector: click feature IDs or type them to add to watch list
  - Multi-line chart: x=step, y=metric, one line per watched feature, colored by feature ID
  - Metric toggle: weight_cos_to_init, win_rate, grad_magnitude, som_magnitude, feature_novelty
  - Pairwise similarity subplot: N_watched x N_watched matrix showing similarity between watched features at current step
  - Step range selector to zoom into interesting regions

- **Implements:** Watch mechanism in `DebugController`, 4 endpoints, `FeatureTrajectory.tsx`
- **Proves:** We can track individual feature journeys. If features 3 and 42 start at cos_sim=0.1 and converge to cos_sim=0.95, we see exactly when and how fast.

**Verification:**
- Watch features [0, 15, 31, 47, 63] (spread across feature space)
- Step 20 times
- Trajectory chart shows 5 lines for win_rate -- some rising, some falling, some stable
- Switch to weight_cos_to_init -- shows how much each feature has moved from its initial weights
- Pairwise similarity shows whether watched features are converging toward each other
- Resume training, let it run 1000 steps, pause again -- trajectories have 1020 points, showing the full journey
- Remove watch on feature 0, add feature 10 -- feature 0's trajectory preserved in chart, feature 10 starts fresh

### M-DBG-5 Cleanup Audit
- [ ] Trajectory memory grows linearly with steps. Cap at N snapshots (e.g., 10000) with circular buffer? Or thin old entries?
- [ ] Should trajectories survive recipe completion? (Probably not -- clear on new recipe run)
- [ ] When watching during normal (non-paused) training, should we thin snapshots (every Nth step) to avoid excessive memory?
- [ ] Can trajectory data be exported (JSON download) for external analysis if needed?

---

## Milestone Dependencies

```
M-DBG-1: Pause/Step/Resume + Batch Viz
    |
    |--- Foundation: DebugController (pause, step, resume, capture_batch)
    |--- Foundation: /debug/* API pattern
    |--- Foundation: DebugPanel.tsx React island
    |
    v
M-DBG-2: Per-Step Metrics Table
    |
    |--- Foundation: capture_step_data() generic key-value store
    |--- Foundation: DebugMetricsTable.tsx sortable table
    |
    v
M-DBG-3: Force Visualization
    |
    |--- Foundation: BCL tensor capture (som_targets, som_delta)
    |--- Foundation: DebugForceViz.tsx (scatter + image detail)
    |
    v
M-DBG-4: Cosine Similarity Heatmap
    |
    |--- Foundation: HeatmapPanel.tsx (generic reusable matrix viz)
    |
    v
M-DBG-5: Feature Trajectory Tracker
    |
    |--- Foundation: Watch mechanism (add_watch, trajectories)
    |--- Foundation: FeatureTrajectory.tsx (multi-line temporal chart)
```

M-DBG-1 is the critical foundation. Everything else builds on pause/step/resume and the `DebugController` class. M-DBG-2 through M-DBG-5 follow a natural progression where each adds one concept, but could be reordered if needed.

## Milestone Status

| Milestone | Status | What I Can Do After |
|-----------|--------|---------------------|
| M-DBG-1: Pause/Step/Resume + Batch Viz | Not started | Pause training, step one batch, see batch images and loss |
| M-DBG-2: Per-Step Metrics Table | Not started | See 64-row table of all BCL metrics per feature, sortable |
| M-DBG-3: Force Visualization | Not started | See grad vs SOM forces per feature, click to view 28x28 update directions |
| M-DBG-4: Cosine Similarity Heatmap | Not started | See 64x64 weight/target similarity matrices, diagnose convergence |
| M-DBG-5: Feature Trajectory Tracker | Not started | Watch individual features across steps, see convergence trajectories |

## Full Outcome Across All Milestones

After all 5 milestones, I start a BCL training run, pause at step 500, and see:

1. **Batch grid** -- the 128 MNIST images in this batch, loss value, step number
2. **Metrics table** -- 64 rows, every feature's win_rate, forces, novelty, blending weights, sortable
3. **Force scatter** -- this batch's gradient vs SOM magnitude per feature, click for 28x28 update images
4. **Cosine matrices** -- 64x64 heatmaps of weight similarity and SOM target similarity
5. **Feature trajectories** -- multi-line charts tracking 5 watched features across all steps so far

I step forward one batch. Everything updates. I can see exactly what the algorithm is doing to each feature, at each step, with each batch. I can answer:

- "Why are features converging?" -- Look at SOM target similarity matrix. If off-diagonal > 0.9, targets are the problem.
- "Is the gradient fighting the SOM?" -- Click a feature, compare grad delta vs SOM delta cosine similarity.
- "When did feature 31 start dying?" -- Check its trajectory: win_rate dropped at step 340, never recovered.
- "Which images cause the most trouble?" -- Look at batch grid alongside force magnitudes.

## Directory Structure (Anticipated)

```
acc/
  debug_controller.py            -- DebugController class (M-DBG-1)
  trainer.py                     -- +2 lines: optional debug_controller param
  trainer_api.py                 -- +/debug/* endpoints (M-DBG-1 through M-DBG-5)
  gradient_gating.py             -- BCL: +som_targets/som_delta capture (M-DBG-3)
  recipes/gradient_gating_l0.py  -- Wire debug_controller into training_metrics_fn

dashboard/src/
  components/
    DebugPanel.tsx               -- Pause/Resume/Step + batch viz (M-DBG-1)
    DebugMetricsTable.tsx        -- 64-row sortable metrics table (M-DBG-2)
    DebugForceViz.tsx            -- Force scatter + 28x28 detail (M-DBG-3)
    HeatmapPanel.tsx             -- Generic NxN heatmap (M-DBG-4)
    FeatureTrajectory.tsx        -- Multi-line trajectory chart (M-DBG-5)
  lib/
    types.ts                     -- +DebugState, DebugStepData, DebugTrajectory
    store.ts                     -- +$debugState computed store, debug polling
    api.ts                       -- (unchanged, uses existing fetchJSON/postJSON)
  pages/
    index.astro                  -- +debug section with new components
```

## How to Verify (Full Debugger)

1. **M-DBG-1:** Start recipe -> Pause -> Step 5 times -> see 5 different batch grids -> Resume -> recipe completes
2. **M-DBG-2:** Pause -> Step -> 64-row table visible, sortable, values match get_step_metrics()
3. **M-DBG-3:** Pause -> Step -> scatter plot visible, click feature -> 28x28 delta images, cosine similarity displayed
4. **M-DBG-4:** Pause -> view weight cosine matrix -> view SOM target matrix -> compare off-diagonal means
5. **M-DBG-5:** Watch 5 features -> step 20 times -> trajectory chart shows 20 points per feature -> resume -> trajectories continue accumulating

## Related Documents

- `docs/HOW_WE_WORK.md` -- Core principles (dual nature, verification, dashboard is the tool)
- `docs/WRITING_MILESTONES.md` -- Milestone structure (functionality-indexed, library-centric)
- `docs/CGG_PLAN.md` -- Full CGG experiment plan with BCL iterations and results
