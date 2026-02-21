# Step-wise Training Debugger Plan

## Summary

A batch-by-batch training inspector built as a new section of the dashboard. Uses the existing `Trainer.train()` loop — no modifications to the trainer. A `StepInspector` sits between the recipe and the trainer, intercepting each step's tensors through a capture API keyed by an enum (`StepTensorKey`). The frontend calls `POST /inspect/step` to advance one batch and receives back every captured tensor for that step. The same enum is shared (Python `StepTensorKey` / TypeScript `StepTensorKey`) so backend capture and frontend rendering are always in sync.

The first milestone is fully end-to-end: press "Step" in the dashboard, one batch runs through the real trainer, and the batch images appear in a new Inspector page. Everything after that adds more captured tensors and more visualizations — but the pipe is proven in M1.

## Context & Motivation

After 4 BCL iterations (v1 -> M3-0 -> M3-0.1 -> M3-0.2), cosine similarity remains 0.94-0.99. Epoch-level summaries and per-500-step logs can't explain WHY features converge. We need to see what happens **inside a single batch** — the actual tensors, the force magnitudes, the SOM targets, the weight updates. We need to step one batch, inspect everything, hypothesize, step again.

The existing `Trainer.train()` loop works fine for this. We don't need pause/resume — we need a mode where the *frontend* drives the loop: "run exactly one step, give me back the tensors." The trainer already accepts `training_metrics_fn` which fires after each backward. We already have `BCL._last_metrics` storing per-step tensors. The infrastructure exists; we just need a capture layer and a pipe to the UI.

### What Exists Today

- `Trainer.train()` loop: `forward -> backward -> training_metrics_fn(step) -> optimizer.step()`. Runs N steps synchronously.
- `BCL._forward_hook` computes all competition/novelty/force tensors, stores subset in `_last_metrics` dict.
- `BCL._last_metrics` keys: `rank_score` [B,D], `feature_novelty` [D], `grad_mask` [B,D], `som_weight` [B,D], `strength` [B,D], `in_nbr` [B,D].
- BCL backward hook computes `som_targets` [D,784] and `som_delta` [D,784] but does NOT store them — applies in-place and discards.
- `GradientGatingL0` recipe wires BCL via `attach_bcl()`, calls `ctx.train(steps=total_steps)`.
- Dashboard has `BCLDiagnostics.tsx` for post-hoc analysis of accumulated logs.
- Dashboard is Astro + React + nanostores. Pattern: React islands with `client:load`, nanostore atoms, `fetchJSON`/`postJSON` helpers.

### What We Need

A new "Inspector" section in the dashboard. Press Step -> one batch runs -> all captured tensors come back -> UI renders them. No polling. No pause/resume threading. Synchronous request-response: POST step, get tensors.

## Design Constraints

1. **Do not modify `Trainer.train()`**. The inspector calls `trainer.train(steps=1)` — the existing loop, one step at a time. The `training_metrics_fn` callback captures tensors into the `StepInspector`. No new parameters, no debug controller, no threading events.

2. **Enum-keyed tensor capture.** Both backend and frontend share a vocabulary of tensor keys (`StepTensorKey`). The backend captures tensors by key. The frontend knows how to render each key. Adding a new tensor = add enum value + capture call + render component. No stringly-typed ad-hoc dicts.

3. **The dashboard is the tool.** New Astro page `/inspect` with its own React components. Not bolted onto the existing dashboard page.

4. **End-to-end from M1.** First milestone: press Step, see batch images. Real trainer, real data, real model. Everything else builds on that pipe.

5. **Library-centric capture API.** `StepInspector.capture(key, tensor)` is generic — works for any tensor from any training loop, not just BCL. BCL-specific captures are calls made in the recipe's `training_metrics_fn`, not baked into the infrastructure.

## API Design

### Core Abstraction: `StepInspector`

```python
# acc/step_inspector.py

class StepTensorKey(str, Enum):
    """Every capturable tensor has an entry here.
    
    Adding a new tensor to inspect:
    1. Add enum value here
    2. Add inspector.capture(KEY, tensor) in the recipe's metrics_fn
    3. Add rendering in the frontend for that key
    """
    # -- Batch data --
    BATCH_IMAGES = "batch_images"           # [B, C, H, W] input images
    BATCH_LABELS = "batch_labels"           # [B] labels (if available)
    
    # -- Loss --
    LOSS = "loss"                           # scalar
    
    # -- Encoder weights --
    ENCODER_WEIGHTS = "encoder_weights"     # [D, 784] current encoder weight matrix
    
    # -- BCL competition --
    RANK_SCORE = "rank_score"               # [B, D] sigmoid margin competition result
    STRENGTH = "strength"                   # [B, D] |dot(W, X)| raw activation strength
    FEATURE_NOVELTY = "feature_novelty"     # [D] how unique each feature's wins are
    IMAGE_COVERAGE = "image_coverage"       # [B] how many features claim each image
    WIN_RATE = "win_rate"                   # [D] mean rank_score per feature
    
    # -- BCL neighborhoods --
    NEIGHBORS = "neighbors"                 # [D, k] neighbor indices
    LOCAL_COVERAGE = "local_coverage"       # [B, D] how many neighbors won each image
    LOCAL_NOVELTY = "local_novelty"         # [B, D] 1/(local_coverage + 1)
    IN_NEIGHBORHOOD = "in_nbr"             # [B, D] binary: is this feature in winner's nbr?
    
    # -- BCL blending weights --
    GRADIENT_WEIGHT = "gradient_weight"     # [D]
    CONTENDER_WEIGHT = "contender_weight"   # [D]
    ATTRACTION_WEIGHT = "attraction_weight" # [D]
    
    # -- BCL forces --
    GRAD_MASK = "grad_mask"                 # [B, D] gradient mask (force 1)
    LOCAL_TARGET = "local_target"           # [D, 784] SOM local pull target
    GLOBAL_TARGET = "global_target"         # [D, 784] SOM global attraction target
    SOM_TARGETS = "som_targets"             # [D, 784] blended SOM target
    SOM_DELTA = "som_delta"                 # [D, 784] actual weight update from SOM
    
    # -- Gradient (from autograd) --
    GRAD_RAW = "grad_raw"                   # [D, 784] raw gradient on encoder weights
    GRAD_MASKED = "grad_masked"             # [D, 784] gradient after BCL mask
    
    # -- Post-step --
    ENCODER_WEIGHTS_POST = "encoder_weights_post"  # [D, 784] weights after optimizer.step()


class StepInspector:
    """Captures tensors during training steps and accumulates history.
    
    Every step's captures are stored in a history list. The frontend can
    scroll through all past steps. Scalars (loss, win_rate, etc.) are
    stored compactly; large tensors (batch images, [B,D] matrices) are
    stored only for the current step to bound memory.
    
    Usage:
        inspector = StepInspector()
        
        # In recipe's training_metrics_fn:
        inspector.capture(StepTensorKey.RANK_SCORE, bcl._last_metrics['rank_score'])
        inspector.capture(StepTensorKey.LOSS, loss)
        
        # After trainer.train(steps=1):
        data = inspector.collect()       # -> serialized data for THIS step
        inspector.commit()               # archives to history, clears current
        
        # Later:
        inspector.get_history()          # -> list of all past step summaries
        inspector.get_step(42)           # -> full data for step 42 (if retained)
    """
    
    # Keys whose full tensor is retained in history (small: scalars + [D]-vectors)
    _HISTORY_RETAIN = {
        StepTensorKey.LOSS, StepTensorKey.WIN_RATE, StepTensorKey.FEATURE_NOVELTY,
        StepTensorKey.GRADIENT_WEIGHT, StepTensorKey.CONTENDER_WEIGHT,
        StepTensorKey.ATTRACTION_WEIGHT, StepTensorKey.IMAGE_COVERAGE,
    }
    
    def __init__(self):
        self._captures: dict[StepTensorKey, torch.Tensor] = {}
        self._step: int = 0
        # Full history: list of {step, keys, + serialized small tensors}
        self._history: list[dict] = []
        # Full tensor snapshots for the last N steps (ring buffer for large tensors)
        self._full_snapshots: dict[int, dict[str, Any]] = {}
        self._max_full_snapshots: int = 50  # keep last 50 steps' full data
    
    def capture(self, key: StepTensorKey, tensor: torch.Tensor) -> None:
        """Store a tensor for this step. Clones to avoid mutation."""
        self._captures[key] = tensor.detach().clone().cpu()
    
    def capture_scalar(self, key: StepTensorKey, value: float) -> None:
        """Store a scalar value."""
        self._captures[key] = torch.tensor(value)
    
    def collect(self) -> dict[str, Any]:
        """Serialize all captured tensors for JSON transport.
        
        Returns dict keyed by StepTensorKey.value (string).
        Tensors are converted based on shape:
        - Images [B, C, H, W] -> base64 PNG grid
        - Vectors [D] -> list of floats  
        - Matrices [B, D] or [D, F] -> nested lists
        - Scalars -> float
        """
        result = {}
        for key, tensor in self._captures.items():
            result[key.value] = _serialize_tensor(key, tensor)
        result["_step"] = self._step
        result["_keys"] = [k.value for k in self._captures.keys()]
        return result
    
    def commit(self) -> None:
        """Archive current step to history, then clear for next step.
        
        Small tensors (scalars, [D]-vectors) are stored in every history
        entry. Full step data (all tensors) is stored in a ring buffer
        of the last N steps. This means you can always plot loss/win_rate
        over all steps, and you can scroll back to any of the last 50
        steps to see the full tensor data (batch images, [B,D] matrices).
        """
        # Build history summary (small tensors only)
        summary = {"step": self._step, "keys": [k.value for k in self._captures.keys()]}
        for key, tensor in self._captures.items():
            if key in self._HISTORY_RETAIN:
                summary[key.value] = _serialize_tensor(key, tensor)
        self._history.append(summary)
        
        # Store full snapshot in ring buffer
        full = self.collect()
        self._full_snapshots[self._step] = full
        # Evict oldest if over capacity
        if len(self._full_snapshots) > self._max_full_snapshots:
            oldest = min(self._full_snapshots.keys())
            del self._full_snapshots[oldest]
        
        # Clear and advance
        self._captures.clear()
        self._step += 1
    
    def get_history(self) -> list[dict]:
        """Return all step summaries (small tensors only). For timeline charts."""
        return self._history
    
    def get_step(self, step: int) -> dict | None:
        """Return full tensor data for a specific step, if still in the ring buffer."""
        return self._full_snapshots.get(step)
    
    @property
    def step(self) -> int:
        return self._step
```

### HTTP API

All under `/inspect/*`. The inspector is a new domain, not mixed into existing endpoints.

```
POST /inspect/setup
  Body: { "recipe": "gradient_gating_l0", "condition": "bcl-med" }
  Response: { "status": "ready", "model_dim": 64, "image_shape": [1,28,28] }
  
  Sets up model + dataset + BCL hooks. Does NOT start training.
  Internally: creates model, loads dataset, attaches tasks, attaches BCL,
  stores references in TrainerAPI.inspector_state. Ready for stepping.

POST /inspect/step
  Body: {} (or { "n": 5 } to run 5 steps at once)
  Response: { 
    "step": 42,
    "keys": ["batch_images", "loss", "rank_score", ...],
    "batch_images": "<base64 grid>",
    "loss": 0.0834,
    "rank_score": [[0.92, 0.01, ...], ...],
    "feature_novelty": [0.83, 0.02, ...],
    ...
  }
  
  Runs exactly 1 (or N) steps of trainer.train(steps=1). Each step's
  captures are committed to history. Returns the LAST step's full data.
  This is synchronous — the response IS the step result.

GET /inspect/state
  Response: { "active": true, "step": 42, "condition": "bcl-med",
              "total_steps": 42, ... }
  
  Current inspector state. Is a session active? What step are we on?

GET /inspect/history
  Response: [
    {"step": 0, "keys": [...], "loss": 0.52, "win_rate": [...], ...},
    {"step": 1, "keys": [...], "loss": 0.49, "win_rate": [...], ...},
    ...
  ]
  
  Returns ALL step summaries. Each entry contains scalars and [D]-vectors
  (loss, win_rate, feature_novelty, blending weights) but NOT large tensors
  (batch images, [B,D] matrices). For timeline charts and scrolling.

GET /inspect/step/{step_num}
  Response: same shape as POST /inspect/step response (full tensor data)
  
  Fetch full tensor data for a past step. Available for the last 50 steps
  (ring buffer). Returns 404 if the step has been evicted.
  This is how the UI scrolls back: click step 17 in the timeline,
  GET /inspect/step/17, render all tensors for that step.

POST /inspect/teardown
  Response: { "status": "torn_down" }
  
  Cleans up model, BCL hooks, frees GPU memory. History is cleared.
```

### Frontend Types

```typescript
// dashboard/src/lib/inspect-types.ts

/** Must match Python StepTensorKey enum exactly. */
export enum StepTensorKey {
  BATCH_IMAGES = 'batch_images',
  BATCH_LABELS = 'batch_labels',
  LOSS = 'loss',
  ENCODER_WEIGHTS = 'encoder_weights',
  RANK_SCORE = 'rank_score',
  STRENGTH = 'strength',
  FEATURE_NOVELTY = 'feature_novelty',
  IMAGE_COVERAGE = 'image_coverage',
  WIN_RATE = 'win_rate',
  NEIGHBORS = 'neighbors',
  LOCAL_COVERAGE = 'local_coverage',
  LOCAL_NOVELTY = 'local_novelty',
  IN_NEIGHBORHOOD = 'in_nbr',
  GRADIENT_WEIGHT = 'gradient_weight',
  CONTENDER_WEIGHT = 'contender_weight',
  ATTRACTION_WEIGHT = 'attraction_weight',
  GRAD_MASK = 'grad_mask',
  LOCAL_TARGET = 'local_target',
  GLOBAL_TARGET = 'global_target',
  SOM_TARGETS = 'som_targets',
  SOM_DELTA = 'som_delta',
  GRAD_RAW = 'grad_raw',
  GRAD_MASKED = 'grad_masked',
  ENCODER_WEIGHTS_POST = 'encoder_weights_post',
}

export interface InspectStepResponse {
  step: number;
  keys: StepTensorKey[];
  [key: string]: unknown;  // tensor data keyed by StepTensorKey values
}

/** Summary for one step in the history timeline. Contains scalars + [D]-vectors only. */
export interface InspectStepSummary {
  step: number;
  keys: StepTensorKey[];
  loss?: number;
  win_rate?: number[];
  feature_novelty?: number[];
  gradient_weight?: number[];
  contender_weight?: number[];
  attraction_weight?: number[];
  image_coverage?: number[];
}

export interface InspectState {
  active: boolean;
  step: number;
  total_steps: number;
  condition: string | null;
  model_dim: number;
  image_shape: number[];
}
```

### How It Wires Together

```
Dashboard "Step" button
  -> POST /inspect/step
  -> TrainerAPI calls trainer.train(steps=1, training_metrics_fn=inspector_metrics_fn)
  -> Inside the one step:
       1. forward() fires BCL._forward_hook (computes all tensors)
       2. loss.backward() fires BCL backward hook (SOM update + grad mask)
       3. training_metrics_fn fires:
          - inspector.capture(BATCH_IMAGES, batch[0])
          - inspector.capture(LOSS, loss)
          - inspector.capture(RANK_SCORE, bcl._last_metrics['rank_score'])
          - inspector.capture(FEATURE_NOVELTY, bcl._last_metrics['feature_novelty'])
          - ... all other BCL tensors ...
          - inspector.capture(ENCODER_WEIGHTS_POST, encoder_layer.weight)
       4. optimizer.step()
  -> data = inspector.collect()     # serialize current step
  -> inspector.commit()             # archive to history, clear for next
  -> JSON response (data) back to frontend
  -> Frontend renders current step, updates timeline with history

Dashboard scrolling to past step:
  -> GET /inspect/step/17
  -> Returns full tensor data from ring buffer
  -> Frontend renders step 17's tensors

Dashboard timeline chart (loss over steps, win_rate over steps):
  -> GET /inspect/history
  -> Returns all step summaries (scalars + [D]-vectors)
  -> Frontend plots loss curve, per-feature win_rate evolution, etc.
```

No threading. No events. No polling. Request-response. Full history.

## Naming Conventions

- **Module:** `acc/step_inspector.py` — `StepInspector` class + `StepTensorKey` enum
- **API domain:** `/inspect/*` endpoints in `acc/trainer_api.py`
- **Dashboard page:** `dashboard/src/pages/inspect.astro`
- **Dashboard components:** `Inspect*.tsx` prefix in `dashboard/src/components/`
- **Store:** `dashboard/src/lib/inspect-store.ts` — separate from main store (not polled)
- **Types:** `dashboard/src/lib/inspect-types.ts` — `StepTensorKey` enum + response types

---

## Milestones

### M-DBG-1: Step and See Batch (End-to-End Pipe)

**Functionality:** I open `/inspect` in the dashboard. I click "Setup" which initializes a bcl-med model. I click "Step" and see the 128 MNIST images from that batch, the loss value, and the step number. I click "Step" again and see 128 different images. A loss timeline chart shows loss at every step so far. I can click any past step in the timeline and the batch images + loss for that step load back into view. I click "Step" 20 more times — the timeline has 22 points, I can scroll through all of them.

**Foundation:** `StepInspector` class with `capture()`/`collect()`/`commit()` and built-in history (ring buffer for full snapshots, unbounded list for scalar summaries). `StepTensorKey` enum (starting with `BATCH_IMAGES` and `LOSS`). `/inspect/setup`, `/inspect/step`, `/inspect/state`, `/inspect/history`, `/inspect/step/{n}` endpoints. `inspect.astro` page with `InspectPanel.tsx` React island. `inspect-store.ts` with action functions and step history array. This is the full pipe — everything after this adds tensor keys and render components, using the existing pipe.

**Captures:** `BATCH_IMAGES`, `LOSS`

- `acc/step_inspector.py`: `StepInspector` class (with history), `StepTensorKey` enum, `_serialize_tensor()` helper
- `/inspect/setup` endpoint: creates `LinearAutoencoder`, loads MNIST, attaches BCL, stores in `TrainerAPI._inspector` state
- `/inspect/step` endpoint: calls `trainer.train(steps=1, training_metrics_fn=...)`, commits to history, returns `inspector.collect()`
- `/inspect/state` endpoint: returns `{active, step, total_steps, condition}`
- `/inspect/history` endpoint: returns all step summaries (scalars only, for timeline)
- `/inspect/step/{n}` endpoint: returns full tensor data for past step N (from ring buffer)
- `/inspect/teardown` endpoint: cleans up, clears history
- `dashboard/src/pages/inspect.astro`: new page with layout, nav link back to main dashboard
- `dashboard/src/components/InspectPanel.tsx`: Setup button, Step button, step counter, batch image grid (8x16 = 128 images), loss display, **loss timeline chart** (clickable — click a point to load that step), **step slider** to scrub through history
- `dashboard/src/lib/inspect-store.ts`: `$inspectState` atom, `$inspectHistory` atom (array of summaries), `$inspectCurrentStep` atom, `setupInspector()`, `stepInspector()`, `loadStep(n)`, `fetchHistory()` actions
- `dashboard/src/lib/inspect-types.ts`: `StepTensorKey` enum (BATCH_IMAGES, LOSS only initially), `InspectStepResponse`, `InspectStepSummary`, `InspectState`

**Proves:** The entire pipe works. Frontend drives a real training step. Tensors flow back. No trainer modifications needed. `training_metrics_fn` is sufficient for capture.

**Critical implementation detail:** The `/inspect/step` endpoint must NOT conflict with the `_is_model_busy()` guard. The inspector session owns the model — no other jobs or recipes can run while inspector is active. `_is_model_busy()` should return True if inspector is active, and inspector endpoints should bypass the guard.

**Verification:**
- Open `localhost:4321/inspect`
- Click "Setup (bcl-med)" — status shows "Ready, step 0"
- Click "Step" — 128 MNIST images appear in a grid, loss value shown (should be ~0.5 on step 1), step counter shows 1
- Click "Step" 4 more times — images change each time, loss decreases, step counter increments
- **Loss timeline chart shows 5 points** — loss at each step, clickable
- Click step 2 in the timeline — batch images from step 2 reload, loss shows step 2's value
- Click step 5 (latest) — back to the most recent batch
- Click "Step" 15 more times (total 20 steps) — timeline shows 20 points, all scrollable
- Use step slider to scrub to step 10 — full data for step 10 loads (from ring buffer)
- Open main dashboard at `/` — model panel shows LinearAutoencoder(64), confirming the inspector set up a real model
- Click "Teardown" — status shows inactive, history cleared

### M-DBG-1 Cleanup Audit
- [ ] Does `_serialize_tensor` handle image tensors correctly? MNIST is [B,1,28,28] with values 0-1.
- [ ] Is the batch tensor captured BEFORE backward (so it's the actual input, not a mutated version)?
- [ ] Does the inspector session prevent concurrent recipe runs? (must — CUDA is not thread-safe)
- [ ] Response size for 128 28x28 images as base64 PNG — is it under 1MB? (Should be ~200KB)
- [ ] Ring buffer of 50 full snapshots — each snapshot ~200KB for images + small tensors. 50 * 200KB = 10MB. Acceptable.
- [ ] History list grows unbounded for scalars — even 10000 steps, each summary is ~1KB = 10MB. Fine.

---

### M-DBG-2: Per-Feature Metrics Table + BCL Competition Tensors

**Functionality:** After stepping, I see a sortable 64-row table showing each feature's win_rate, feature_novelty, gradient_weight, contender_weight, attraction_weight, grad_magnitude, and som_magnitude. I can sort by any column. Dead features (win_rate < 0.01) are highlighted red. I can immediately see: how many features are winning, how many are dead, how the blending weights partition the features.

**Foundation:** BCL tensor capture in `training_metrics_fn` — the recipe's metrics function now pumps BCL's `_last_metrics` and derived scalars into the `StepInspector`. Frontend `InspectMetricsTable.tsx` is a generic sortable table component that reads tensor keys from the step response. Adding future columns = adding captures, the table auto-discovers keys that are `[D]`-shaped vectors.

**Captures added:** `RANK_SCORE`, `STRENGTH`, `FEATURE_NOVELTY`, `WIN_RATE`, `GRADIENT_WEIGHT`, `CONTENDER_WEIGHT`, `ATTRACTION_WEIGHT`, `GRAD_MASK`, `IMAGE_COVERAGE`, `IN_NEIGHBORHOOD`, `ENCODER_WEIGHTS`

- Extend `StepTensorKey` enum with new entries
- Extend inspector `training_metrics_fn` to capture BCL tensors from `bcl._last_metrics` and derived values
- `InspectMetricsTable.tsx`: 64-row sortable table, auto-columns from `[D]`-shaped tensors, color coding, summary row (mean/min/max)
- Table sorts client-side — all data already in the step response

**Verification:**
- Step once. Table appears with 64 rows, 7+ columns.
- Sort by win_rate descending — top features have high win_rate, bottom near-zero.
- Sort by feature_novelty — high-novelty features are the ones with unique territory.
- Dead features (red rows) have near-zero win_rate AND near-zero gradient_weight. Their attraction_weight is high. This confirms the blending logic.
- Sum of gradient_weight + contender_weight + attraction_weight should be meaningful per feature (they don't need to sum to 1 — check the actual formulas).
- Cross-check: `win_rate[i]` should match `rank_score[:, i].mean()` from the raw tensor.

### M-DBG-2 Cleanup Audit
- [ ] Should the table auto-discover columns from `[D]`-shaped tensors, or use an explicit column config? Auto-discover is more library-centric but may show confusing columns. Consider: auto-discover with an optional hide list.
- [ ] `[B,D]` tensors (rank_score, grad_mask) — should these be summarized to `[D]` for the table (mean across batch) and also available as raw matrices? The table needs per-feature summaries; later milestones may want the full `[B,D]` matrix.

---

### M-DBG-3: Full Tensor Capture + Viz Primitives

**Functionality:** After stepping, every intermediate tensor from the BCL algorithm is captured and available in the step response. The two reusable visualization components — `InspectHeatmap` for [B,D] and [D,D] matrices and `InspectWeightGrid` for [D,784] tensors rendered as 28x28 thumbnails — are built, verified standalone with the tensors we already have, and ready for the Algorithm Trace in M-DBG-4.

**Why this is a separate milestone:** The backend changes (storing 6 new tensors in `_last_metrics`, capturing them in the inspector endpoint) and the two new viz components (canvas-based heatmap, weight-image grid) are the hard parts. If the [B,D] heatmap rendering has issues, or the [D,784] grid serialization is too large, we want to catch that before wiring 9 sections into a page layout.

**Backend — BCL `_last_metrics` additions:**

The BCL `_forward_hook` already computes all these tensors. They just aren't stored. Add to `_last_metrics`:

| Key | Tensor | Shape | Currently stored? |
|-----|--------|-------|-------------------|
| `neighbors` | `self._neighbors` | [D, k] | No (on `self`, not in dict) |
| `local_coverage` | `local_coverage` | [B, D] | No |
| `local_novelty` | `local_novelty` | [B, D] | No |
| `local_target` | `local_target` | [D, 784] | No |
| `global_target` | `global_target` | [D, 784] | No |
| `som_targets` | `som_targets` | [D, 784] | No |

Memory cost: `local_coverage` and `local_novelty` are [128, 64] = 32 KB each. `local_target`, `global_target`, `som_targets` are [64, 784] = 200 KB each. Total additional: ~664 KB per step. Combined with existing captures (~360 KB), total ~1 MB per step. 50 steps in ring buffer = 50 MB. Acceptable.

**Backend — Inspector endpoint additions:**

Add captures for all 6 new keys in the `/inspect/step` endpoint's BCL metrics loop.

**Frontend — Two new reusable viz components:**

1. **`InspectHeatmap.tsx`** — Canvas-based [M,N] heatmap.
   - Props: `data: number[][]`, `title: string`, `xLabel?: string`, `yLabel?: string`, `colorScale?: 'viridis' | 'hot' | 'diverging'`
   - Renders via `<canvas>` for performance (128×64 = 8192 cells). HTML table would be too slow.
   - Hover shows (row, col, value) in a tooltip.
   - Color scale: 0→dark, 1→bright by default. Diverging scale for signed data.
   - Used for: strength [B,D], rank_score [B,D], local_coverage [B,D], local_novelty [B,D], grad_mask [B,D], in_nbr [B,D], cosine similarity [D,D].
   - Cell size configurable (default 4px for [128,64], 6px for [64,64]).

2. **`InspectWeightGrid.tsx`** — [D, 784] tensor rendered as a grid of 28×28 thumbnails.
   - Props: `data: number[][]`, `title: string`, `cols?: number` (default 8), `imageSize?: number` (default 28)
   - Each row of the [D, 784] matrix is reshaped to 28×28 and rendered as a grayscale thumbnail.
   - 64 features in an 8×8 grid = fits in ~300px.
   - Used for: `encoder_weights` [D,784], `local_target` [D,784], `global_target` [D,784], `som_targets` [D,784].
   - Supports signed data (diverging colormap: blue=negative, white=zero, red=positive) for force targets.

**What this does NOT include:** The 9-step Algorithm Trace layout (that's M-DBG-4). This milestone proves the captures work and the viz components render correctly.

**Files to modify:**
- `acc/gradient_gating.py` — Add 6 keys to `_last_metrics` dict
- `acc/trainer_api.py` — Add 6 captures to inspector step endpoint
- `acc/step_inspector.py` — Add `NEIGHBORS` to `_IMAGE_KEYS` if rendering as weight thumbnails, otherwise handle [D,k] serialization

**Files to create:**
- `dashboard/src/components/InspectHeatmap.tsx` — Canvas [M,N] heatmap
- `dashboard/src/components/InspectWeightGrid.tsx` — [D,784] as 28×28 thumbnail grid

**Captures added:** `NEIGHBORS`, `LOCAL_COVERAGE`, `LOCAL_NOVELTY`, `LOCAL_TARGET`, `GLOBAL_TARGET`, `SOM_TARGETS`

**Verification:**
- Step once via curl. Response includes all new keys alongside existing ones.
- `local_target` and `global_target` are [64, 784] arrays. `som_targets` likewise.
- `neighbors` is [64, 8] array. `local_coverage` and `local_novelty` are [128, 64].
- In InspectPanel, temporarily render: `<InspectHeatmap data={strength} title="Strength [B,D]" />` and `<InspectWeightGrid data={encoderWeights} title="Encoder Weights" />`. Both render without errors.
- Heatmap hover shows correct values. Weight grid shows 64 28×28 thumbnails.
- Response size check: full step response is under 5 MB (the [D,784] matrices are the largest at ~200 KB each as JSON).

### M-DBG-3 Cleanup Audit
- [ ] `_last_metrics` now has ~16 keys. Memory per step is ~1 MB. Should BCL have an `inspector_mode` flag to only store the expensive [D,784] tensors when inspector is active?
- [ ] `InspectHeatmap` renders on canvas. Does it need a resize observer for responsive layout? Or is fixed width (cell_size × N_cols) sufficient?
- [ ] `InspectWeightGrid` renders [D,784] as images. For signed data (targets - weights = delta), need diverging colormap. Is this handled?
- [ ] Response size for 3× [64,784] as JSON = ~600 KB of nested arrays. Consider: should [D,784] tensors be serialized as base64 image grids (like batch_images) instead of raw arrays? Image grid is ~30 KB. But raw arrays are needed for hover-to-inspect values.

---

### M-DBG-4: Algorithm Trace — All 9 Steps Visualized

**Functionality:** I see the entire BCL algorithm's intermediate state laid out vertically, one section per computation step. Every [B,D] matrix is a heatmap. Every [D] vector is a bar chart. Every [D,784] tensor is a grid of 64 28×28 thumbnails. I scroll down through the algorithm's execution for this batch and can visually trace where symmetry appears, where information gets flattened, and where forces diverge or converge.

**The specific question this answers:** At which step in the computation does the symmetry appear? Is strength already uniform? Or does strength have variety but rank_score kills it? Or does rank_score have variety but feature_novelty kills it? The heatmaps show exactly where the information gets flattened.

**Layout (vertical scroll, one section per algorithm step):**

**Step 1: Strength** — `strength` [128, 64] rendered as `InspectHeatmap`. Rows = batch images, columns = features. Which features light up on which images? If this is already uniform, features are clones from the start.

**Step 2: Neighborhoods** — `neighbors` [64, 8] rendered as a grid. Each row shows one feature's 8 neighbor weight vectors as tiny 28×28 thumbnails (looked up from `encoder_weights`). Do all neighborhoods look the same? Are they all each other? Adjacent: `encoder_weights` [64, 784] as `InspectWeightGrid` for reference.

**Step 3: Rank Score** — `rank_score` [128, 64] rendered as `InspectHeatmap`. Where does each feature WIN its neighborhood? If features are identical, margins ≈ 0, scores ≈ 0.5. Nobody clearly wins or loses. Compare visually to Step 1 — did competition sharpen or flatten the signal?

**Step 4: Image Coverage** — `image_coverage` [128] as a bar chart. Which images are crowded vs underserved? With identical features all scoring 0.5, coverage ≈ 32 everywhere. Nothing is novel, nothing is underserved.

**Step 5: Feature Novelty** — `feature_novelty` [64] as a bar chart. With uniform coverage, every feature gets the same novelty score. The metric is blind. Compare to `win_rate` [64] side by side.

**Step 6: Local Coverage** — `local_coverage` [128, 64] as `InspectHeatmap`. Per feature, per image, how many neighbors also won. With identical features this is uniform. `local_novelty` [128, 64] as a second heatmap next to it.

**Step 7: SOM Targets** — `local_target` [64, 784] and `global_target` [64, 784] each as `InspectWeightGrid` (64 28×28 thumbnails). **This is the smoking gun.** If all 64 global targets look identical, the SOM is pulling everyone to the same place. If they're different, something is creating asymmetry. Show `som_targets` [64, 784] (the blend) as a third grid.

**Step 8: Blending Weights** — Three [64] bar charts side by side: `gradient_weight`, `contender_weight`, `attraction_weight` per feature. Who's getting which force? With identical features, these are all identical.

**Step 9: Final Forces** — `grad_mask` [128, 64] as `InspectHeatmap`. What gradient actually flows? And `som_targets` [64, 784] as `InspectWeightGrid` showing the final combined direction each feature is pushed.

**Foundation:** `InspectAlgorithmTrace.tsx` — a single component that takes the step response and renders all 9 sections using `InspectHeatmap`, `InspectWeightGrid`, and inline bar charts. This is the "full X-ray" of one batch through BCL. The component is self-contained: give it a step response dict, it renders everything.

**Inline bar chart:** Simple `<div>` bars for [D] and [B] vectors. No external chart library. Same pattern as the existing loss timeline bars in InspectPanel. Labeled with feature ID or image index.

**Files to create:**
- `dashboard/src/components/InspectAlgorithmTrace.tsx` — the 9-section vertical layout
- `dashboard/src/components/InspectBarChart.tsx` — reusable [N] bar chart (used for image_coverage, feature_novelty, blending weights)

**Files to modify:**
- `dashboard/src/components/InspectPanel.tsx` — add `<InspectAlgorithmTrace stepData={stepData} />` below the metrics table

**No backend changes.** All tensors are already captured in M-DBG-3. This is pure frontend layout.

**Verification:**
- Open `/inspect`. Auto-setup bcl-slow. Click Step.
- Scroll down. All 9 sections render:
  1. Strength heatmap [128×64] — cells colored by activation magnitude
  2. Neighborhoods grid — 64 rows × 8 neighbor thumbnails + full encoder weight grid
  3. Rank score heatmap [128×64] — compare visually to strength
  4. Image coverage bar chart [128] — one bar per batch image
  5. Feature novelty bar chart [64] — one bar per feature, side by side with win_rate
  6. Local coverage heatmap [128×64] + local novelty heatmap [128×64]
  7. Three 8×8 grids of 28×28 thumbnails: local_target, global_target, som_targets. **Are the 64 global targets all the same image?** This is the convergence-to-average diagnostic.
  8. Three [64] bar charts: gradient_weight, contender_weight, attraction_weight
  9. Grad mask heatmap [128×64] + som_targets grid (repeated for reference)
- Step 10 more times. Scroll through the trace again. Do the heatmaps change? Does symmetry break over time?
- At step 0 (random init): strength should be somewhat varied. Rank score should have some contrast. SOM targets should be varied.
- At step 50+: If features are converging, the SOM target grids show it — all 64 thumbnails look the same.

### M-DBG-4 Cleanup Audit
- [ ] 9 sections is a lot of vertical scroll. Should sections be collapsible (click header to expand/collapse)?
- [ ] The three [64, 784] grids at Step 7 are the most visually dense. Each is 8×8 thumbnails at 28×28 = 224×224 pixels. Three side-by-side = 672px. Fits in 1200px page width.
- [ ] [128] bar chart for image_coverage: 128 bars at 4px each = 512px wide. Fits fine.
- [ ] Performance: rendering 6 heatmaps × ~8000 cells each + 4 weight grids × 64 thumbnails. All canvas-based, should be fine. But test with step x10 (rapid re-renders).

---

### M-DBG-5: Cosine Similarity Matrices

**Functionality:** Below the algorithm trace, I see three 64×64 heatmaps:
1. **Weight cosine similarity** — how similar are the 64 features' current weights?
2. **SOM target cosine similarity** — how similar are the 64 features' SOM targets?
3. **Gradient cosine similarity** — how similar are the 64 features' masked gradient directions?

If weight cosine is low but SOM target cosine is high, the SOM is actively pulling features together (convergence-to-average confirmed). If gradient cosine is high, SGD is also contributing to convergence. Mean off-diagonal shown below each matrix.

**Foundation:** Uses `InspectHeatmap` from M-DBG-3. Cosine similarity matrices computed server-side (computing 64×64 from 64×784 is cheap in Python, expensive to ship raw 64×784 matrices to the client for no reason).

**Captures needed:** Uses `ENCODER_WEIGHTS` [D,784], `SOM_TARGETS` [D,784], `GRAD_MASKED` [D,784]. The first two are already captured. `GRAD_MASKED` needs to be captured — `encoder_layer.weight.grad` after `loss.backward()` IS the masked gradient (BCL backward hook applies the mask to the output gradient, which flows through to weight grad). Capture in the inspector endpoint.

**Server-side derived tensors:** Add to `StepInspector.collect()` or compute in the endpoint:
- `weight_cosine_matrix` [D, D] — cosine similarity of `encoder_weights` rows
- `som_target_cosine_matrix` [D, D] — cosine similarity of `som_targets` rows  
- `grad_cosine_matrix` [D, D] — cosine similarity of `grad_masked` rows

**Files to create:**
- `dashboard/src/components/InspectCosineMatrices.tsx` — three `InspectHeatmap` instances, mean off-diagonal stats

**Files to modify:**
- `acc/trainer_api.py` — capture `GRAD_MASKED` from `encoder_layer.weight.grad`, compute 3 cosine matrices
- `acc/step_inspector.py` — add `WEIGHT_COSINE_MATRIX`, `SOM_TARGET_COSINE_MATRIX`, `GRAD_COSINE_MATRIX` to enum
- `dashboard/src/lib/inspect-types.ts` — add new enum values
- `dashboard/src/components/InspectPanel.tsx` — add `<InspectCosineMatrices />` section

**Verification:**
- Step to step 0: weight cosine matrix shows low off-diagonal (< 0.3 at random init).
- Step to step 50+: compare off-diagonal means across all 3 matrices.
- **Key diagnostic:** If `mean_off_diagonal(som_targets) >> mean_off_diagonal(weights)`, SOM is driving convergence.

### M-DBG-5 Cleanup Audit
- [ ] `InspectHeatmap` reuse — same component for [B,D] heatmaps and [D,D] cosine matrices. Color scale may differ (0-1 for cosine vs 0-max for activations).
- [ ] Should cosine matrices be computed eagerly (every step) or lazily (on scroll to that section)? Eager adds ~5ms — negligible.

---

## Milestone Dependencies

```
M-DBG-1: Step + Batch Images (the pipe)  ✅
    |
    v
M-DBG-2: Per-Feature Metrics Table  ✅
    |
    v
M-DBG-3: Full Tensor Capture + Viz Primitives  ✅
    |     (backend: 6 new _last_metrics keys + captures)
    |     (frontend: InspectHeatmap + InspectWeightGrid)
    |
    v
M-DBG-4: Algorithm Trace — All 9 Steps Visualized
    |     (pure frontend: InspectAlgorithmTrace + InspectBarChart)
    |     (no backend changes — all tensors already captured)
    |
    v
M-DBG-5: Cosine Similarity Matrices
          (backend: GRAD_MASKED capture + 3 derived cosine matrices)
          (frontend: InspectCosineMatrices using InspectHeatmap)
```

M-DBG-1 is the pipe. M-DBG-2 adds the numbers. M-DBG-3 captures everything and builds the rendering primitives. M-DBG-4 is the full X-ray — you see every intermediate computation in the algorithm for all 64 features at once, and you can visually trace where symmetry appears. M-DBG-5 adds the global convergence diagnostic via cosine similarity matrices.

## Milestone Status

| Milestone | Status | What I Can Do After |
|-----------|--------|---------------------|
| M-DBG-1: Step + Batch Images | **DONE** | Press Step, see 128 MNIST images and loss, timeline, history scroll |
| M-DBG-2: Per-Feature Metrics Table | **DONE** | See 64-row sortable table with win_rate, novelty, blending weights, grad_mask, strength |
| M-DBG-3: Full Tensor Capture + Viz Primitives | **DONE** | All BCL intermediates captured. InspectHeatmap + InspectWeightGrid components verified. |
| M-DBG-4: Algorithm Trace (9 Steps) | Not started | Full X-ray of one batch through BCL: 6 heatmaps, 4 weight grids, 5 bar charts. See where symmetry appears. |
| M-DBG-5: Cosine Similarity Matrices | Not started | 3× 64x64 heatmaps: weight/SOM target/gradient similarity. The convergence-to-average diagnostic. |

## Full Outcome Across All Milestones

After all 5 milestones, I open `/inspect`, bcl-slow auto-loads, and I click Step. I see:

1. **Batch grid** — 128 MNIST images from this batch.
2. **Metrics table** — 64 rows, every feature's win_rate, novelty, blending weights, sortable. Dead features red.
3. **Algorithm Trace** — 9 sections, one per computation step of the BCL algorithm:
   - Strength [128×64] heatmap — which features activate on which images
   - Neighborhoods — each feature's 8 neighbors as 28×28 thumbnails
   - Rank score [128×64] heatmap — where each feature wins its competition
   - Image coverage [128] bar chart — crowded vs underserved images
   - Feature novelty [64] bar chart — unique vs redundant features
   - Local coverage/novelty [128×64] heatmaps — neighborhood-level competition
   - **SOM targets** — 64 local_target + 64 global_target + 64 som_target thumbnails. THE smoking gun.
   - Blending weights — three [64] bar charts showing force partition per feature
   - Grad mask [128×64] heatmap — what gradient actually flows
4. **Cosine matrices** — three 64×64 heatmaps: weight/SOM target/gradient similarity

I step forward. Everything updates. I scroll through the trace and answer:

- **"At which step does symmetry appear?"** — Strength heatmap has variety (features activate differently). Rank score heatmap flattens it (margins near zero → all scores ≈ 0.5). Feature novelty is blind (uniform coverage → uniform novelty). The flattening happens at rank_score because features are too similar for margin competition to separate them.
- **"Why are features converging?"** — Step 7 SOM targets: all 64 global targets look identical. The SOM is pulling everyone to the same place. Confirmed by cosine similarity matrix: `mean_off_diagonal(som_targets) >> mean_off_diagonal(weights)`.
- **"Is gradient fighting SOM?"** — Step 9 grad_mask is near-zero for dead features. The only force is SOM. And SOM points to the same target for all of them.
- **"Why don't dead features revive?"** — Step 8 blending weights: dead features have high attraction_weight but Step 4 shows uniform image_coverage, so the global pull has no gradient — all images are equally "underserved" when every feature claims them.

## Directory Structure (Anticipated)

```
acc/
  step_inspector.py              -- StepInspector + StepTensorKey enum (M-DBG-1)
  trainer.py                     -- UNCHANGED
  trainer_api.py                 -- +/inspect/* endpoints (M-DBG-1 through M-DBG-5)
  gradient_gating.py             -- BCL: +store all intermediates in _last_metrics (M-DBG-3)

dashboard/src/
  pages/
    index.astro                  -- UNCHANGED (existing dashboard)
    inspect.astro                -- New inspector page (M-DBG-1)
  components/
    InspectPanel.tsx             -- Auto-setup + condition switcher + batch grid + loss (M-DBG-1)
    InspectMetricsTable.tsx      -- 64-row sortable feature metrics (M-DBG-2)
    InspectHeatmap.tsx           -- Canvas-based [M,N] heatmap (M-DBG-3)
    InspectWeightGrid.tsx        -- [D,784] as 28x28 thumbnail grid (M-DBG-3)
    InspectAlgorithmTrace.tsx    -- 9-step vertical trace layout (M-DBG-4)
    InspectBarChart.tsx          -- Reusable [N] bar chart (M-DBG-4)
    InspectCosineMatrices.tsx    -- 3x 64x64 cosine similarity heatmaps (M-DBG-5)
  lib/
    inspect-types.ts             -- StepTensorKey enum + response types (M-DBG-1)
    inspect-store.ts             -- Inspector state + actions (M-DBG-1)
```

## How to Verify (Full Inspector)

1. **M-DBG-1:** Page auto-loads bcl-slow -> Step 5 times -> 5 different batch grids + decreasing loss -> change condition dropdown -> fresh session
2. **M-DBG-2:** Step -> 64-row table visible, sortable, dead features highlighted red, 9 columns with blending weights
3. **M-DBG-3:** Step via curl -> response includes `local_target`, `global_target`, `som_targets`, `neighbors`, `local_coverage`, `local_novelty`. InspectHeatmap renders [128,64] data. InspectWeightGrid renders [64,784] as 8×8 grid of thumbnails.
4. **M-DBG-4:** Step -> scroll down -> 9 algorithm trace sections visible. Strength heatmap, neighborhoods grid, rank_score heatmap, image_coverage bars, feature_novelty bars, local coverage heatmaps, **SOM target grids** (local + global + combined), blending weight bars, grad_mask heatmap. All update on each step.
5. **M-DBG-5:** Three 64×64 cosine similarity heatmaps visible below the trace. Mean off-diagonal displayed. Compare weight vs SOM target similarity.

## How to Run

```bash
# Terminal 1 (tmux: control-trainer)
./run_trainer.sh

# Terminal 2 (tmux: control-ui)
./run_ui.sh

# Browser
open http://localhost:4321/inspect
```

## Related Documents

- `docs/HOW_WE_WORK.md` — Core principles (dual nature, verification, dashboard is the tool)
- `docs/WRITING_MILESTONES.md` — Milestone structure (functionality-indexed, library-centric)
- `docs/CGG_PLAN.md` — Full CGG experiment plan with BCL iterations and results
