# M3: Generator Hot-Reload + Dataset Dashboard

## Summary

Wire the registry hot-reload pattern to `acc/generators/`, introduce `DatasetGenerator` base class, and build dashboard panels for generating datasets, browsing samples, and managing data — all without touching code or restarting anything.

## Context & Motivation

M2 gave the operator eyes on training (per-task loss curves, eval metrics, reconstruction comparison). But datasets were still code-only — you had to write Python to generate or load data. M3 completes the data pipeline by making generators discoverable and usable from the dashboard.

## Phases

### Phase 1: DatasetGenerator base class + GeneratorRegistry

**Outcome:** `DatasetGenerator` ABC with `name`, `description`, `parameters`, `generate(**params)`. `GeneratorRegistry` discovers subclasses from `acc/generators/`, hot-reloads on file change.

**Foundation:** `DatasetGenerator` base class (same pattern as Task, Recipe). `GeneratorRegistry` (same pattern as TaskRegistry, RecipeRegistry).

- `acc/generators/base.py` — `DatasetGenerator` ABC
- `acc/generators/registry.py` — `GeneratorRegistry` with file watcher
- Existing generators (thickness, slant, shapes) wrapped as `DatasetGenerator` subclasses
- Backward compat: `generate_thickness()`, `generate_slant()`, `generate_shapes()` standalone functions still work
- Wired into `TrainerAPI.__init__()` with file watcher started

### Phase 2: Dashboard generator UI

**Outcome:** Pick a generator from dropdown, configure parameters, click Generate, dataset appears.

**Foundation:** Dashboard generator form that reads `parameters` metadata from the registry.

- `[+ Dataset]` panel in sidebar: generator dropdown, dynamic param form, Generate button
- `POST /generators/generate` API endpoint
- Generated dataset registered in trainer's datasets dict
- `[+ Task]` dropdown auto-updates with new dataset

### Phase 3: Dataset sample browser

**Outcome:** See actual images from each dataset in the dashboard.

**Foundation:** Per-dataset sample thumbnail grid (already implemented in M2 datasets panel).

- Dataset panel shows name, size, shape, target type, and 8 sample thumbnails per dataset
- Sample thumbnails loaded via `GET /datasets/{name}/sample`

### Phase 4: Verification script

`python -m acc.test_m3` — 10 tests covering all phases.

## Verification

`python -m acc.test_m3` — 10/10 tests pass:
1. DatasetGenerator base class works
2. GeneratorRegistry discovers built-in generators (thickness, slant, shapes)
3. GeneratorRegistry hot-reload: new file appears in registry
4. GeneratorRegistry error handling: syntax error caught
5. GeneratorRegistry deletion: file removed, generator disappears
6. Generate dataset via registry produces valid AccDataset
7. Generated dataset has correct shape, targets, sample() works
8. Backward compatibility: standalone functions still work
9. Multiple generators produce distinct coexisting datasets
