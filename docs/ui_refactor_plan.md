# UI Refactor Plan: Centralized State & Event-Driven Dashboard

## Summary

`acc/ui/app.py` is a 1903-line monolith where every panel independently fetches its own data from the trainer API, renders its own HTML with inline styles, and has no coordination with other panels. The result is a dashboard you cannot trust: loading a checkpoint doesn't update all panels, different panels can disagree on what the "current" checkpoint is, and critical panels like attention maps are never refreshed by any event. This refactor introduces a shared state cache, an HTMX event system for declarative cross-panel refresh, and a component library to kill duplicated rendering logic.

## Context & Motivation

When a user loads a checkpoint, they expect the entire dashboard to reflect that checkpoint. Currently:
- 3 independent `/checkpoints/tree` fetches determine "current checkpoint" — they can disagree
- 7 independent `/health` fetches check the same `has_model` boolean
- Attention maps panel is NEVER refreshed on checkpoint load (missing from the imperative refresh list)
- Cross-panel refresh is ad-hoc `htmx.ajax()` calls in `<script>` blocks — every new panel or action must manually enumerate which other panels to refresh, and forgetting one is silent
- health_colors defined 4 times (3 Python, 1 JS), loss summary table built in 2 places, no-model guard copy-pasted 6 times, training panel skeleton duplicated

The root cause: there is no shared state layer and no event system. Each partial is a completely independent micro-app.

## Naming Conventions

- **State cache:** `DashboardState` in `acc/ui/state.py`
- **Components:** Functions in `acc/ui/components.py`, named `panel()`, `eval_table()`, `image_grid()`, `health_badge()`, `no_model_guard()`
- **Events:** Constants in `acc/ui/events.py`, named `CHECKPOINT_CHANGED`, `TRAINING_DONE`, `MODEL_CHANGED`, `TASKS_CHANGED`
- **Partials:** `acc/ui/partials/{group}.py` — one file per panel group
- **Actions:** `acc/ui/actions/{group}.py` — one file per action group
- **Static:** `acc/ui/static/dashboard.js`, `acc/ui/static/dashboard.css`
- **Templates:** `acc/ui/templates/index.html` (Jinja2)

## Audit Findings (Evidence)

### Duplicated API Calls

| API Path | Independent Fetch Count | Locations |
|----------|------------------------|-----------|
| `/health` | 7 | Lines 735, 796, 893, 1158, 1376, 1426, 1481 |
| `/checkpoints/tree` | 3 | Lines 597, 855, 1280 |
| `/jobs/current` | 3 | Lines 621, 1147, 1830 |
| `/datasets` | 3 | Lines 491, 744, 1102 |
| `/eval/run` | 2 | Lines 803, 900 |

### Duplicated Constants & Rendering

| What | Copy Count | Locations |
|------|-----------|-----------|
| `health_colors` dict | 4 | Lines 244 (JS), 641, 1061, 1303 (Python) |
| Loss summary table HTML | 2 | Lines 657-671 (Python), 356-371 (JS) |
| Training panel skeleton | 2 | Lines 716-730, 1561-1575 |
| "No model loaded" guard | 6 | Lines 735-739, 796-800, 893-897, 1376-1380, 1426-1431, 1481-1485 |
| Panel `<div class="panel"><h3>` wrapper | ~18 | Every partial |
| Error check `isinstance(x, dict) and "error" in x` | 6 | Lines 433, 742, 804, 901, 1103, 1281 |

### Stale Panels (load once, never auto-refresh)

| Panel | Refreshed On Checkpoint Load? | Refreshed On Training Done? |
|-------|------------------------------|----------------------------|
| Eval Metrics | Yes (line 1634) | Yes (line 389) |
| Reconstructions | Yes (line 1633) | Yes (line 390) |
| Traversals | Yes (line 1635) | No |
| Sort by Factor | Yes (line 1636) | No |
| **Attention Maps** | **NO — BUG** | **No** |

### Checkpoint Load Refresh — What Fires vs What's Missing

Currently `action_load_checkpoint` (lines 1628-1639) fires 7 imperative `htmx.ajax()` calls:

**Refreshed:** model, tasks, reconstructions, eval, traversals, sort_by_factor, training, checkpoints.

**NOT refreshed (bugs):**
- `attention_maps` — shows stale data from previous checkpoint
- `checkpoint_indicator` — relies on 3s polling, brief inconsistency window

### SSE "Training Done" Refresh — What Fires vs What's Missing

SSE done handler (lines 387-392) refreshes: training, eval, reconstructions, tasks, jobs_history.

**NOT refreshed:**
- model, checkpoints, traversals, sort_by_factor, attention_maps, datasets

## Phases

### Phase 1: Extract Infrastructure (No Behavior Change)

**Outcome:** User sees identical dashboard. New files exist. All duplicated logic has a single source. Partials delegate to shared abstractions.

**Foundation:** `DashboardState` (cached API proxy), `components.py` (HTML builders), `events.py` (event constants), static JS/CSS files. Future phases build ON these abstractions.

**Verification:** Open the UI, all panels render correctly. Visually identical to current state. Run `python -m acc.test_m1` — passes. Grep for duplicated `health_colors` — only in `components.py` and `dashboard.js`. Grep for `_api("/health")` in partials — zero hits (all go through `state.health()`).

Tasks:
1. Create `acc/ui/state.py` with `DashboardState` class — cached async proxy to trainer API with TTL and `invalidate()` method
2. Create `acc/ui/components.py` — extract `panel()`, `eval_table()`, `image_grid()`, `health_badge()`, `no_model_guard()`, `error_guard()`, `HEALTH_COLORS`
3. Create `acc/ui/events.py` — define event name constants: `CHECKPOINT_CHANGED`, `TRAINING_DONE`, `MODEL_CHANGED`, `TASKS_CHANGED`
4. Create `acc/ui/static/dashboard.js` — extract all inline JS from `_chart_js()` (lines 216-402). Reference `HEALTH_COLORS` from a single source.
5. Create `acc/ui/static/dashboard.css` — extract inline styles from HTML f-strings
6. Mount `StaticFiles` in `app.py` for `/static` path
7. Replace all `_api("/health")` calls in partials with `state.health()`
8. Replace all `_api("/checkpoints/tree")` calls with `state.checkpoint_tree()` / `state.current_checkpoint_id()`
9. Replace all 6 no-model guards with `components.no_model_guard(title)`
10. Replace all 4 health_colors dicts with `components.HEALTH_COLORS`
11. Replace both loss summary table renderers with `components.eval_table()`
12. Replace both training panel skeletons with a shared function

### Phase 2: Event-Driven Refresh

**Outcome:** Checkpoint load, training done, and model change correctly refresh ALL dependent panels. No panel is ever "forgotten." Attention maps bug is fixed.

**Foundation:** HTMX custom events (`HX-Trigger` response header) replace all imperative `htmx.ajax()` refresh lists. Panels self-declare what events they respond to. New panels automatically participate by adding `hx-trigger` attributes.

**Verification:**
- Load a checkpoint → ALL eval panels update, including attention maps (currently broken)
- Complete training → eval, reconstructions, traversals, sort_by_factor all refresh
- Count `htmx.ajax` calls in action handlers — should be zero (all replaced by `HX-Trigger` headers)
- Add a fake new panel with `hx-trigger="checkpoint-changed from:body"` — verify it refreshes on checkpoint load without touching any action handler code

Tasks:
1. Define panel-to-event mapping:
   - `CHECKPOINT_CHANGED` → model, tasks, eval, reconstructions, traversals, sort_by_factor, attention_maps, checkpoints, checkpoint_indicator, training
   - `TRAINING_DONE` → eval, reconstructions, traversals, sort_by_factor, attention_maps, training, jobs_history
   - `MODEL_CHANGED` → model, tasks, eval, reconstructions, traversals, sort_by_factor, attention_maps
   - `TASKS_CHANGED` → tasks, add_task, eval
2. Update main page template: add `hx-trigger="EVENT_NAME from:body"` to each panel div
3. Update `action_load_checkpoint`: remove all `<script>htmx.ajax()</script>` blocks, add `HX-Trigger: checkpoint-changed` response header, call `state.invalidate_all()`
4. Update `action_train` (SSE done handler in JS): emit `training-done` custom event on `document.body` via `htmx.trigger(document.body, 'training-done')`
5. Update `action_add_task`, `action_remove_task`, `action_toggle_task`: add `HX-Trigger: tasks-changed`
6. Update `action_save_checkpoint`, `action_fork_checkpoint`: add `HX-Trigger: checkpoint-changed`
7. Fix attention maps bug: panel now listens to `checkpoint-changed` — no code needed beyond the hx-trigger attribute from task 2
8. Remove all remaining imperative `htmx.ajax()` refresh calls from action handlers

### Phase 3: Split Partials Into Modules

**Outcome:** `app.py` is under 200 lines — just route registration and `DashboardState` instantiation. Each panel group lives in its own file.

**Foundation:** Partial modules import `DashboardState` and `components`, share no global mutable state. Adding a new panel = adding a new file + registering routes.

**Verification:** `app.py` line count < 200. Each partial file is self-contained. `wc -l acc/ui/partials/*.py` — no file over 300 lines. UI looks identical.

Tasks:
1. Create `acc/ui/partials/model.py` — move `partial_model`, `partial_tasks`, `partial_add_task`
2. Create `acc/ui/partials/training.py` — move `partial_training`, `partial_step`, `partial_jobs_history`
3. Create `acc/ui/partials/eval.py` — move `partial_eval`, `partial_reconstructions`, `partial_traversals`, `partial_sort_by_factor`, `partial_attention_maps`
4. Create `acc/ui/partials/checkpoints.py` — move `partial_checkpoints_tree`, `partial_checkpoint_indicator`
5. Create `acc/ui/partials/datasets.py` — move `partial_datasets`, `partial_dataset_samples`, `partial_generate`
6. Create `acc/ui/partials/recipe.py` — move `partial_recipe`
7. Create `acc/ui/actions/training.py` — move `action_train`, `action_stop`
8. Create `acc/ui/actions/checkpoints.py` — move `action_load_checkpoint`, `action_save_checkpoint`, `action_fork_checkpoint`
9. Create `acc/ui/actions/tasks.py` — move `action_add_task`, `action_toggle_task`, `action_remove_task`, `action_set_weight`
10. Create `acc/ui/actions/datasets.py` — move `action_generate_dataset`
11. Update `app.py` to import and register routes from all modules
12. Delete dead code from `app.py`

### Phase 4: Jinja2 Templates

**Outcome:** HTML is in `.html` files, not Python f-strings. Layout changes don't require touching Python.

**Foundation:** Jinja2 template inheritance. Base layout template, panel-specific templates. `components.py` functions become Jinja2 macros.

**Verification:** Same visual result. HTML lives in `acc/ui/templates/`. Zero f-string HTML in Python partial handlers (they call `template.render()` instead).

Tasks:
1. Add `jinja2` to dependencies (likely already available via Starlette)
2. Create `acc/ui/templates/base.html` — main page layout, static file includes, panel grid
3. Create `acc/ui/templates/partials/` — one template per partial
4. Create `acc/ui/templates/macros/` — `components.html` with Jinja2 macros for panel, eval_table, image_grid, health_badge
5. Convert each partial handler: replace f-string HTML with `templates.TemplateResponse()`
6. Verify visual parity

## Phase Cleanup Notes

Review at end of each phase:
- Are there remaining duplicated strings or patterns?
- Can any new abstraction be applied to existing code?
- Are there partials that should be combined or split differently?

### Cleanup Decision Template
- **Do now:** Items that block next phase or create significant tech debt
- **Defer:** Nice-to-have items that don't block progress
- **Drop:** Items that turned out to be unnecessary

## Full Outcome Across All Phases

After all four phases:
- **User can trust the dashboard.** Every panel reflects the current checkpoint, current model, and current training state. No stale data. No disagreements between panels.
- **Checkpoint load is atomic from the UI perspective.** One event, all panels respond.
- **Adding a new panel requires:** one new partial file + one `hx-trigger` declaration. Zero changes to action handlers or other panels.
- **`app.py` is a thin router** (~200 lines). Panel logic lives in focused modules (~100-300 lines each). HTML lives in templates. JS and CSS live in static files.
- **`DashboardState`** is the cached proxy pattern — reusable for any future UI that needs to talk to the trainer API.

## Directory Structure (Anticipated)

```
acc/ui/
├── app.py                      # ~200 lines: route registration, DashboardState init, static mount
├── state.py                    # DashboardState — cached async trainer API proxy
├── components.py               # panel(), eval_table(), image_grid(), health_badge(), HEALTH_COLORS
├── events.py                   # CHECKPOINT_CHANGED, TRAINING_DONE, MODEL_CHANGED, TASKS_CHANGED
├── __init__.py
├── partials/
│   ├── __init__.py
│   ├── model.py                # partial_model, partial_tasks, partial_add_task
│   ├── training.py             # partial_training, partial_step, partial_jobs_history
│   ├── eval.py                 # partial_eval, partial_reconstructions, partial_traversals,
│   │                           #   partial_sort_by_factor, partial_attention_maps
│   ├── checkpoints.py          # partial_checkpoints_tree, partial_checkpoint_indicator
│   ├── datasets.py             # partial_datasets, partial_dataset_samples, partial_generate
│   └── recipe.py               # partial_recipe
├── actions/
│   ├── __init__.py
│   ├── training.py             # action_train, action_stop
│   ├── checkpoints.py          # action_load_checkpoint, action_save_checkpoint, action_fork
│   ├── tasks.py                # action_add_task, action_toggle_task, action_remove_task, action_set_weight
│   └── datasets.py             # action_generate_dataset, action_recipe_run, action_recipe_stop
├── static/
│   ├── dashboard.js            # Chart.js setup, SSE handler, HTMX event triggers
│   └── dashboard.css           # All styles
└── templates/
    ├── index.html              # Main page layout (Jinja2)
    └── partials/               # Per-panel templates (Phase 4 only)
        ├── model.html
        ├── training.html
        ├── eval.html
        ├── checkpoints.html
        ├── datasets.html
        └── recipe.html
```

## How to Review

1. **Phase 1:** Diff `app.py` — should only show deletions (moved to new files) and delegation calls. New files should contain zero new logic — only extracted existing logic. Open UI, verify visual parity.
2. **Phase 2:** Search for `htmx.ajax` in action handlers — should find zero. Search for `HX-Trigger` in action handlers — should find one per action. Load checkpoint, verify attention maps refresh.
3. **Phase 3:** `wc -l app.py` should be < 200. Each partial file should be self-contained (no imports from other partials). UI visually identical.
4. **Phase 4:** Zero f-string HTML in Python handlers. All HTML in `templates/`. UI visually identical.

## Related Documents

- `docs/HOW_WE_WORK.md` — Core principles (dual nature, verification, hypothesis-driven debugging)
- `docs/WRITING_MILESTONES.md` — Milestone structure (functionality-indexed, not backend-indexed)
