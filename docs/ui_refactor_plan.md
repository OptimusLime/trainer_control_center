# UI Refactor Plan: From Untrustworthy Monolith to Self-Describing Dashboard

## Summary

The ACC dashboard was a 1903-line monolith (`acc/ui/app.py`) that couldn't be trusted: panels disagreed on state, checkpoint loads missed panels, and checkpoints themselves were meaningless labels with no context. This refactor makes the dashboard a reliable source of truth for understanding what your experiments are doing.

## Context & Motivation

When I look at the dashboard, I need to answer:
- **What am I looking at?** Which checkpoint is loaded, what model config does it have, which recipe created it, what was it trained for?
- **What happened?** Per-task loss results, health classification, training steps completed.
- **What changed?** When I load a checkpoint or finish training, every panel should update. No stale data.

Before this refactor, the answer to all three was "unclear." Checkpoint names like `baseline_nofree_trained` conveyed nothing without reading recipe source code. Loading a checkpoint didn't refresh attention maps. Different panels could disagree on the current state.

## Naming Conventions

- **API client:** `acc/ui/api.py` — `call()` with GET cache, `is_error()`, `invalidate_all()`
- **Components:** `acc/ui/components.py` — `panel()`, `no_model_guard()`, `HEALTH_COLORS`, `loss_summary_table()`, `metric_color()`
- **Events:** `acc/ui/events.py` — `CHECKPOINT_CHANGED`, `TRAINING_DONE`, `MODEL_CHANGED`, `TASKS_CHANGED`, `DATASETS_CHANGED`
- **Partials:** `acc/ui/partials/{group}.py` — one file per panel group
- **Actions:** `acc/ui/actions/{group}.py` — one file per action group

## Phases

### Phase 1: Dashboard Tells the Truth (Done)
**Functionality:** I can trust that every panel shows the same state. Loading a checkpoint refreshes ALL panels, including attention maps (which was broken). No panel is ever "forgotten."
**Foundation:** `components.py` (single-source HTML builders, `HEALTH_COLORS`), `events.py` (event constants), `static/dashboard.js` and `dashboard.css` (extracted from inline). Event-driven refresh via `HX-Trigger` headers.
**Commits:** `d706080`, `a8b34cc`

### Phase 2: Dashboard is Maintainable (Done)
**Functionality:** I can find and edit any panel's code in under 10 seconds. `app.py` is 194 lines. Each panel group is a focused module. Adding a new panel is: create a file, register one route.
**Foundation:** Module structure (`partials/`, `actions/`, `api.py`). GET response caching in `api.py`. Consistent `is_error()` helper.
**Commits:** `29c11a5`, `14c284b`, `8a6bc3e`

### Phase 3: Checkpoints Explain Themselves (Done)
**Functionality:** I can look at the checkpoint tree and understand what each checkpoint is: which recipe created it, what model config it has (channels, stop-grad, factor groups), what training results it achieved. I don't need to read recipe source code to understand my experiments.
**Foundation:** `Checkpoint` dataclass with `recipe_name`, `description`, `model_config`, `tasks_snapshot`, `metrics` — all persisted in `.pt` files. `FactorSlotAutoencoder.config()` serializes architectural config. `CheckpointStore.save()` builds all metadata before `torch.save()`.
**Commits:** `0e1bc24`
**Bug fixed:** Loss summary was added AFTER `torch.save()` so it was never persisted to disk. Now built before save.

### Phase 4: Recipe Panel Shows What's Happening (Not Started)
**Functionality:** I can see what recipe is running, what phase it's on, which branch is being trained, how far along it is, and what the latest results are. When a recipe finishes, I can see the comparison summary.
**Foundation:** Recipe progress via SSE or polling. RecipeJob already tracks `current_phase`, `phases_completed`, `checkpoints_created`. UI needs to surface these meaningfully.

**Verification:**
- Run `mnist_factor_experiment` recipe → see "Branch 1 of 3: baseline (20ch, no stop-grad)" in the recipe panel
- Training progress shows live for each branch
- When complete, recipe panel shows comparison summary
- Recipe panel polls for updates while running, stops when done

### Phase 5: Checkpoint Comparison View (Not Started)
**Functionality:** I can select two checkpoints (e.g., baseline_nofree_trained vs stopgrad_20ch_trained) and see a side-by-side comparison: eval metrics, traversal grids, reconstruction quality, attention maps. I can answer "did stop-grad help?" directly from the dashboard.
**Foundation:** Comparison endpoint already exists (`/eval/checkpoint`). UI needs a dedicated comparison panel that shows both results side-by-side with winner highlighting.

**Verification:**
- Select baseline_nofree_trained and stopgrad_20ch_trained
- See side-by-side eval table with green highlighting on better values
- See side-by-side traversal grids to visually compare disentanglement
- Metrics that are better on the left are colored differently from metrics better on the right

## Phase Status

| Phase | Status | What I Can Do After |
|-------|--------|-------------------|
| Phase 1: Dashboard Tells the Truth | Done | Trust that checkpoint load refreshes all panels. No stale data. |
| Phase 2: Dashboard is Maintainable | Done | Find any panel's code in 10 seconds. Add a new panel without touching other code. |
| Phase 3: Checkpoints Explain Themselves | Done | See recipe name, model config, and results for every checkpoint in the tree. |
| Phase 4: Recipe Panel Shows What's Happening | Not started | See live recipe progress: which branch, what phase, how far along. |
| Phase 5: Checkpoint Comparison View | Not started | Compare two checkpoints side-by-side and answer "which is better?" |

## Phase 3 Cleanup Audit

**Done:**
- Deleted `state.py` (unused, duplicated `api.py` functionality)
- Added module-level GET cache (1s TTL) to `api.py` with `invalidate()`/`invalidate_all()`
- Added `is_error()` helper to `api.py`, adopted across all 11 modules
- Fixed bug: `model.py:12` had `"error" in data` that would crash on list responses
- All mutation actions call `invalidate_all()` after POST

**Deferred:**
- `panel()`/`empty()`/`error_div()` from `components.py` barely adopted (~50 raw HTML panel wrappers)
- Hardcoded color hex values scattered across partials (22 occurrences)
- `api_proxies.py` uses `HTMLResponse` for JSON instead of `JSONResponse`

## Directory Structure

```
acc/ui/
├── app.py              194 lines — thin router with imports
├── api.py               82 lines — API client, GET cache, is_error()
├── components.py       227 lines — HTML builders, HEALTH_COLORS, metric helpers
├── events.py            36 lines — event constants
├── static/
│   ├── dashboard.js    188 lines — Chart.js, SSE, health banner
│   └── dashboard.css         — all styles
├── partials/           7 files, ~1050 lines total
│   ├── model.py        — model info, task cards, add-task form
│   ├── training.py     — loss chart, step counter, job history
│   ├── eval.py         — eval metrics, reconstructions, traversals, sort, attention maps
│   ├── checkpoints.py  — checkpoint tree with rich metadata
│   ├── datasets.py     — dataset browser, sample thumbnails, generator form
│   ├── recipe.py       — recipe picker, run/stop, phase progress
│   └── health.py       — connection health + device selector
└── actions/            7 files, ~365 lines total
    ├── training.py     — train, stop
    ├── checkpoints.py  — save, load, fork (with cache invalidation)
    ├── tasks.py        — add, toggle, remove, set weight
    ├── datasets.py     — generate dataset, recipe run/stop
    ├── eval.py         — run eval, compare
    └── api_proxies.py  — SSE proxy, JSON pass-throughs for JS

acc/checkpoints.py      — Checkpoint dataclass with recipe_name, model_config, tasks_snapshot, metrics
acc/factor_slot_autoencoder.py — config() method for serializable architectural config
acc/recipes/base.py     — RecipeContext passes recipe metadata to CheckpointStore
```

## How to Verify

1. Open the dashboard. Every panel loads. No errors in browser console.
2. Load a checkpoint → ALL panels update (including attention maps, eval, traversals).
3. Checkpoint tree shows recipe name, description, model config, per-task loss with health colors.
4. Run training → loss chart updates live → when done, all result panels refresh.
5. `wc -l acc/ui/app.py` = 194. `find acc/ui -name "*.py" | wc -l` = 16 focused modules.

## Related Documents

- `docs/HOW_WE_WORK.md` — Core principles (dual nature, verification, hypothesis-driven debugging)
- `docs/WRITING_MILESTONES.md` — Milestone structure (functionality-indexed, not backend-indexed)
