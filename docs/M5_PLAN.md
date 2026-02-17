# M5 Plan: UFR Evaluation + Visual Diagnosis Dashboard

## Summary

M5 gives the user a complete visual diagnosis workbench. Every factor group's behavior is visible: traversals show what varying a factor does, sort-by-factor shows what the model learned, attention maps show where in the image each factor acts, and UFR score quantifies disentanglement. All of this works per-checkpoint, enabling side-by-side comparison across experiment branches.

## Context & Motivation

We can train, hot-reload tasks/generators, manage datasets, and switch devices. The traversal and sort-by-factor endpoints already work. But the user can't yet:
- See **attention maps** (which spatial regions each factor controls)
- Compute a **UFR score** (quantitative disentanglement metric)
- Run **per-checkpoint eval** with visual outputs (traversals + sort from a specific checkpoint without switching)
- **Compare** two checkpoints side-by-side with visual grids

The roadmap says: "M5 quantifies what you see (UFR score) and enables checkpoint comparison." The visual tools already partially exist but need generalization and the two new capabilities (attention maps, UFR).

## Naming Conventions

- `acc/eval/ufr.py` — UFR scoring logic
- `acc/eval/attention.py` — Attention map extraction
- `EvalMetric.UFR` — new enum entry for UFR score
- `EvalMetric.COMPLETENESS`, `EvalMetric.DISENTANGLEMENT` — UFR sub-metrics
- API: `/eval/attention_maps` — new endpoint
- API: `/eval/traversals?checkpoint_id=X` — checkpoint param on existing endpoint
- API: `/eval/sort_by_factor?checkpoint_id=X` — checkpoint param on existing endpoint
- API: `/eval/ufr` — new endpoint
- UI: `partial_attention_maps()` — new panel

## Phases

### Phase 1: Attention Map Extraction + Visualization

**Outcome:** I can click [Attention Maps] in the dashboard and see per-factor heatmaps overlaid on input images, showing which spatial regions each factor controls.

**Foundation:** `CrossAttentionBlock` gains `store_attn` flag. `acc/eval/attention.py` provides `extract_attention_maps(model, images)` that returns per-factor spatial heatmaps. Reusable for any future attention visualization.

**Verification:** `python -m acc.test_m5` tests 1-3:
1. `extract_attention_maps()` returns dict of `{factor_name: (B, H, W)}` tensors
2. `GET /eval/attention_maps` returns base64 PNG heatmaps per factor
3. Heatmap values are in [0, 1] and sum to 1 across factors for each spatial position

Tasks:
1. Modify `CrossAttentionBlock.forward()` to optionally store attention weights in `self.last_attn_weights`
2. Create `acc/eval/attention.py` with `extract_attention_maps(model, images) -> dict[str, Tensor]`
   - Runs forward pass, collects `last_attn_weights` from each cross-attention stage
   - Averages across heads, reshapes to spatial dims, normalizes per-factor
   - Returns `{factor_name: (B, H, W)}` averaged across stages
3. Add `GET /eval/attention_maps` endpoint to `trainer_api.py`
   - Takes `n_images: int = 4`
   - Returns `{factor_name: [base64_heatmap_overlay, ...], originals: [base64, ...]}`
4. Add `partial_attention_maps()` to UI — grid of original images with per-factor heatmap overlays
5. Add CSS for heatmap display

### Phase 2: UFR Scoring

**Outcome:** I can click [Run UFR] and see a disentanglement score that quantifies how well factors separate concepts from contexts. The score appears in the eval panel alongside task metrics.

**Foundation:** `acc/eval/ufr.py` with `compute_ufr(model, datasets, factor_groups)`. `EvalMetric` gains UFR entries. The UFR computation is a reusable library function, not tied to the API.

**Verification:** `python -m acc.test_m5` tests 4-6:
4. `compute_ufr()` returns a dict with `EvalMetric.UFR`, `EvalMetric.DISENTANGLEMENT`, `EvalMetric.COMPLETENESS`
5. `POST /eval/ufr` returns the UFR results
6. UFR score is in [0, 1] range

Tasks:
1. Add `EvalMetric.UFR`, `EvalMetric.DISENTANGLEMENT`, `EvalMetric.COMPLETENESS` to `acc/eval_metric.py`
2. Create `acc/eval/ufr.py` with `compute_ufr(model, datasets, factor_groups, device)`:
   - For each factor group (concept), encode dataset with factor group varied vs fixed
   - Build concept x context transfer matrix
   - Compute disentanglement (row entropy), completeness (column entropy), UFR (harmonic mean)
3. Add `POST /eval/ufr` endpoint to `trainer_api.py`
4. Display UFR metrics in the eval panel alongside task metrics

### Phase 3: Per-Checkpoint Eval + Comparison

**Outcome:** I can pick any two checkpoints and see their traversals, sort-by-factor, and attention maps side-by-side. I don't have to load a checkpoint to see its visual eval.

**Foundation:** Existing traversal/sort/attention endpoints gain optional `checkpoint_id` param. `_with_checkpoint()` context manager pattern for temporarily loading a checkpoint. Checkpoint comparison UI panel.

**Verification:** `python -m acc.test_m5` tests 7-9:
7. `GET /eval/traversals?checkpoint_id=X` returns traversals from checkpoint X without changing current state
8. Checkpoint comparison API returns paired results
9. Current model state unchanged after per-checkpoint eval

Tasks:
1. Create `_with_checkpoint()` helper in `trainer_api.py` that temporarily loads a checkpoint, yields, then restores
2. Add `checkpoint_id` optional query param to `/eval/traversals`, `/eval/sort_by_factor`, `/eval/attention_maps`
3. Add `GET /eval/compare` endpoint — takes two checkpoint IDs, returns side-by-side traversals + metrics
4. Add comparison UI panel: two checkpoint dropdowns + side-by-side grids
5. Update existing eval comparison table to include visual comparison option

## Phase Cleanup Notes

After each phase, check:
- Can the attention map extraction pattern be reused for other layer visualizations?
- Is `_with_checkpoint()` general enough for M6 (model expansion) to use?
- Should the base64 image encoding be moved to a shared utility?

### Cleanup Decision (at phase end)

- **Do now:** Items that block next phase
- **Defer:** `_tensor_to_base64` lives in `trainer_api.py` but could be `acc/eval/viz.py`
- **Drop:** Items that turned out unnecessary

## Full Outcome Across All Phases

After M5, the user can:
1. See attention maps per factor group overlaid on input images
2. Get a UFR disentanglement score for the current model or any checkpoint
3. Run traversals, sort-by-factor, and attention maps on any checkpoint without loading it
4. Compare two checkpoints side-by-side with visual grids and metrics
5. All metrics (task eval + UFR) visible in one unified eval panel

## Directory Structure (Anticipated)

```
acc/eval/
├── __init__.py
├── attention.py    # extract_attention_maps()
└── ufr.py          # compute_ufr()
```

## How to Review

1. `python -m acc.test_m5` — all tests pass
2. Open dashboard → load a checkpoint → click [Attention Maps] → see per-factor heatmaps
3. Click [Run UFR] → see UFR score in eval panel
4. Pick two checkpoints → [Compare] → see side-by-side traversals
5. Verify: running per-checkpoint eval does NOT change the currently loaded model state
