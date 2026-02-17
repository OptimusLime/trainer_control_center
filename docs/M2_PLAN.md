# M2: Hot-Reload Tasks + Dashboard Task Management

## Summary

Wire the `RecipeRegistry` hot-reload pattern to `acc/tasks/`, then build real dashboard panels that give the operator eyes on what's happening: per-task loss curves, eval metrics, reconstruction comparison, task lifecycle management, and persistent training history. When M2 is done, you are no longer training blind.

## Context & Motivation

M1.95 proved the recipe + fork pattern works. The MNIST factor experiment ran end-to-end. But the dashboard is operationally blind — it shows almost nothing about what the model is learning, what each task is contributing, or what the reconstructions look like compared to input. You can train, but you can't *see*.

M2 fixes this by shipping hot-reload tasks (foundation) alongside the dashboard panels that make training visible (functionality). Per WRITING_MILESTONES principles: dual nature, coupled features ship together.

## Naming Conventions

- `TaskRegistry` — auto-discovers `Task` subclasses from `acc/tasks/`, hot-reloads on file change
- Dashboard partials: `partial_task_detail`, `partial_loss_curves`, `partial_recon_comparison`
- Trainer API endpoints: `/registry/tasks` (updated to use TaskRegistry), `/tasks/{name}/detail`, `/jobs/{job_id}/loss_history`

## Phases

### Phase 1: TaskRegistry + Hot-Reload Wiring

**Outcome:** Write a `.py` file in `acc/tasks/`, save it, and the new Task class appears in the trainer's available tasks within 2 seconds. No restart.

**Foundation:** `TaskRegistry` — identical pattern to `RecipeRegistry`, pointed at `acc/tasks/`. Proves the registry pattern generalizes (M3 will clone it again for generators).

Tasks:
1. Create `acc/tasks/registry.py` — clone from `acc/recipes/registry.py`, adapted for `Task` subclasses
   - Scans `acc/tasks/` for `.py` files, skips `__init__`, `base.py`, `registry.py`
   - `_load_module()` finds `Task` subclasses (not `Task` itself)
   - File watcher polls for changes, calls `_load_module()` on modified files
   - `list()` returns available task classes with metadata
   - `get(class_name)` returns the class (not an instance — tasks need constructor args)
2. Wire `TaskRegistry` into `TrainerAPI.__init__()` — start watcher, replace hardcoded `_resolve_task_class()` with registry lookup
3. Update `/registry/tasks` endpoint to return from `TaskRegistry.list()` instead of hardcoded list
4. Update `/tasks/add` endpoint to resolve class from `TaskRegistry.get()` instead of `_resolve_task_class()`

**Verification:**
- Trainer running → create `acc/tasks/dummy_task.py` with a trivial Task subclass → within 2 seconds, `GET /registry/tasks` includes "DummyTask"
- `POST /tasks/add {"class_name": "DummyTask", ...}` succeeds
- Delete the file → "DummyTask" disappears from registry
- Syntax error in file → caught, logged, trainer continues running

### Phase 2: Dashboard Task Management Panel

**Outcome:** From the dashboard, I can see all available task classes, add a task to the model with a specific dataset, adjust weights, toggle enable/disable, remove tasks — full lifecycle without touching code.

**Foundation:** Task management UI components that generalize to any Task subclass. The [+ Task] form queries the registries for available classes and datasets.

Tasks:
1. New trainer API endpoint: `GET /registry/tasks` — already returns available classes (updated in Phase 1 to use TaskRegistry). Add constructor parameter hints (e.g., "needs dataset with target_type='classes'")
2. Dashboard [+ Task] panel:
   - Dropdown: select task class (from `/registry/tasks`)
   - Dropdown: select dataset (from `/datasets`)
   - Input: task name (auto-generated default)
   - Input: weight (default 1.0)
   - Input: latent_slice (optional, format "start:end")
   - [Add Task] button → `POST /tasks/add`
   - Error display if `TaskError` on attach
3. Enhanced task card in sidebar:
   - Shows: name, class, dataset, weight, enabled status, latent_slice
   - Weight adjustment: input field + update button → `POST /tasks/{name}/set_weight` (new endpoint)
   - Toggle enable/disable (already exists)
   - [Remove] button → `POST /tasks/{name}/remove`
4. New trainer API endpoint: `POST /tasks/{name}/set_weight` — adjusts task weight without remove/re-add

**Verification:**
- Dashboard shows [+ Task] form with task classes from registry + datasets
- Add a ClassificationTask from dashboard → task appears in sidebar
- Change weight via dashboard → loss contribution changes on next train
- Remove task from dashboard → disappears, training unaffected
- Hot-reload a new task file → it appears in the [+ Task] dropdown

### Phase 3: Per-Task Loss Curves + Persistent History

**Outcome:** During training, I see individual loss curves for each task (not just an aggregate). After training completes or I refresh the page, the loss history is still there — it doesn't vanish.

**Foundation:** Job loss history stored in `JobManager`, served via API, rendered as multi-series Chart.js. This is the foundation for all future training visualization.

Tasks:
1. Trainer API: `GET /jobs/{job_id}/loss_history` — returns full loss array `[{step, task_name, task_loss, total_loss}, ...]`
2. Trainer API: `GET /jobs/history` — returns last N completed jobs with summary stats (total steps, final losses per task, duration)
3. Update SSE stream to include `total_loss` alongside per-task loss
4. Update Chart.js in dashboard:
   - Multi-series chart: one line per task, colored distinctly
   - Y-axis shows loss, X-axis shows step
   - On page load, fetch `/jobs/{job_id}/loss_history` for current/latest job and populate chart (persistence)
   - During training, SSE appends points live
5. Loss history panel below chart:
   - List of recent jobs with summary (steps, duration, final losses)
   - Click a job → load its loss curves into the chart

**Verification:**
- Start training with 3 tasks → chart shows 3 distinct loss curves
- Refresh page mid-training → chart repopulates from history + SSE continues
- Training completes → chart persists, doesn't vanish
- Click a previous job → its loss curves load

### Phase 4: Eval Metrics + Reconstruction Comparison

**Outcome:** I can run eval and see per-task metrics clearly displayed. I can see input images side-by-side with their reconstructions to visually judge quality.

**Foundation:** Eval display components and reconstruction comparison. This is the visual feedback loop — you see metrics AND images.

Tasks:
1. Enhanced eval panel:
   - [Run Eval] button triggers `POST /eval/run`
   - Results displayed as a table: task name | metric name | value
   - Color coding: green for good (accuracy > 0.9, loss < 0.1), yellow for moderate, red for poor
   - History: last 5 eval results shown so you can see trends
2. New trainer API: `POST /eval/reconstructions` — encode + decode N images, return `{originals: [base64...], reconstructions: [base64...]}` 
3. Reconstruction comparison panel:
   - Side-by-side grid: original on left, reconstruction on right
   - 8 random images from each dataset
   - Auto-refresh button: [Refresh Reconstructions]
   - Visible difference highlighting (optional: diff image)
4. Wire eval panel to refresh after training completes (HTMX trigger on job done)

**Verification:**
- Run eval → table shows per-task metrics with values
- Reconstruction panel shows input vs output side-by-side for 8 images
- Visual quality assessment: can you tell them apart? That's the metric.
- After training 1000 steps, reconstructions should be recognizable

### Phase 5: Verification Script

**Outcome:** `python -m acc.test_m2` exercises the full M2 feature set programmatically.

**Foundation:** Test harness for M2 that validates all new functionality.

Tasks:
1. Create `acc/test_m2.py`:
   - Test TaskRegistry discovers built-in tasks (Classification, Reconstruction, Regression, KLDivergence)
   - Test TaskRegistry hot-reload: write a dummy task file, verify it appears in registry
   - Test TaskRegistry error handling: write a file with syntax error, verify registry catches it
   - Test `/tasks/add` via API with registry-resolved class
   - Test `/tasks/{name}/set_weight` endpoint
   - Test `/jobs/{job_id}/loss_history` returns per-task loss data
   - Test `/eval/reconstructions` returns originals + reconstructions
   - Cleanup: remove dummy task file

**Verification:** `python -m acc.test_m2` — all tests pass.

## Phase Cleanup Notes

Review at phase end:
- The hardcoded `_resolve_task_class()` in trainer_api.py should be fully replaced by TaskRegistry
- `RecipeRegistry` and `TaskRegistry` share 90% code — consider extracting a `BaseRegistry` generic class (defer if it doesn't block M3)
- Pre-existing LSP type errors in autoencoder.py, trainer_api.py, dataset.py — not M2 scope but flag for later

## Full Outcome Across All Phases

When M2 is done:
- **Hot reload:** Write a task .py file → it appears in the dashboard dropdown within 2 seconds
- **Task management:** Add, configure, weight-adjust, toggle, remove tasks entirely from the dashboard
- **Loss visibility:** Per-task loss curves during training, persistent across page reloads
- **Eval visibility:** Per-task metrics in a clear table after each eval run
- **Reconstruction visibility:** Input vs reconstruction side-by-side for visual quality assessment
- **Training history:** Recent jobs listed with summary stats, click to view loss curves

The operator is no longer training blind. They can see what each task contributes, how the model reconstructs, and what the metrics look like — all from the dashboard.

## Directory Structure (Anticipated)

```
acc/tasks/
├── __init__.py
├── base.py              (existing — Task ABC)
├── registry.py          (NEW — TaskRegistry with hot-reload)
├── classification.py    (existing)
├── reconstruction.py    (existing)
├── regression.py        (existing)
├── kl_divergence.py     (existing)
└── dummy_task.py        (test artifact, cleaned up)

acc/trainer_api.py       (modified — TaskRegistry, new endpoints)
acc/ui/app.py            (modified — new dashboard panels)
acc/test_m2.py           (NEW — verification script)
docs/M2_PLAN.md          (this file)
```

## How to Review

1. Phase 1: `GET /registry/tasks` returns all built-in tasks dynamically. Create/delete a task file → registry updates.
2. Phase 2: Dashboard [+ Task] form works, task cards show full info, weight adjustment works.
3. Phase 3: Train with multiple tasks → distinct loss curves. Refresh page → curves persist.
4. Phase 4: Run eval → metrics table. Reconstruction comparison panel shows input vs output.
5. Phase 5: `python -m acc.test_m2` passes.
