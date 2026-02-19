# UI Rewrite Plan: HTMX → Astro + Vanilla JS (React Islands If Needed)

## Why

The HTMX/Starlette UI is fundamentally broken for a training dashboard:
- **Server-side DOM replacement**: every poll replaces HTML. If the trainer is slow for one request, the panel flashes to "empty" then back. No client-side state to fall back on.
- **Proxy middleman**: the Starlette UI server proxies every request to the trainer, doubling latency and creating a single point of failure.
- **SSE lock convoy**: the streaming loss endpoint + training thread contend for the same lock on every step, starving the entire HTTP event loop.
- **Connection spam**: even with caching, HTMX polls fire 8+ concurrent requests every few seconds, overwhelming the trainer's single-threaded uvicorn.

The fix is architectural: **static HTML/JS that fetches JSON directly from the trainer API**. The browser holds state. Failed fetches are silently ignored. No middleman.

## Stack

- **Astro** — static site builder. Generates HTML/CSS/JS with zero runtime by default. React/Preact islands only where we need interactive components (loss chart, forms).
- **Chart.js** — keep for loss curves (already works well).
- **Vanilla JS fetch()** — poll JSON from trainer API (port 6060) directly. No proxy.
- **CSS** — keep the dark GitHub theme. Responsive down to iPad (768px).
- **Trainer API** — unchanged. Already returns JSON for everything. Just need CORS headers.

## Architecture

```
Browser ←→ Trainer API (port 6060)
              ↑
         Static files served by trainer
         (or separate static server, doesn't matter)
```

No UI server process. No Starlette. No proxy. The trainer serves its own dashboard at `/` or we serve it separately.

## Critical Docs

Read before implementing:
- `docs/HOW_WE_WORK.md` — dual nature, library-centric, end-to-end from M1
- `docs/WRITING_MILESTONES.md` — functionality-indexed, verification is first-class

---

## Milestones (Priority Order — What I Use Most Comes First)

### M-UI-1: "I can see live loss curves and know what's training"

**I can**: Open the dashboard during training and see loss curves updating, which model is loaded, current step count, and overall health — without the page flashing or freezing.

**Foundation**: Astro project scaffolded. Trainer serves static build. CORS enabled on trainer. Polling infrastructure (fetch + silent failure + DOM update pattern) established as reusable utilities. Chart.js loss chart component.

**Panels ported**:
1. Loss curves (Chart.js, poll `/jobs/current` + `/jobs/{id}/loss_history` + `/jobs/{id}/loss_summary`)
2. Step counter (poll `/jobs/current`)
3. Health indicator (poll `/health`)
4. Model info (poll `/model/describe`)
5. Loss summary table (from loss_summary endpoint)

**Verification**: Start a recipe via curl. Open dashboard. See loss chart updating every 5s. See step count incrementing. See model info. Close trainer — health shows disconnected. Reopen — recovers. No flashing at any point. Works at 768px width.

**Key decisions**:
- Poll interval: 5s for loss data, 10s for model/health (configurable)
- On fetch failure: log to console, keep existing DOM, retry on next interval
- Loss chart: append new points since last fetch, don't re-render entire history
- Layout: 2-column above 768px, single column below

---

### M-UI-2: "I can see recipe progress and experiment results side by side"

**I can**: See which recipe is running, which branch it's on, phase progress, and when it finishes see the comparison table showing metrics for all branches.

**Foundation**: Recipe status polling component. Branch progress rendering. Comparison table component (reusable for any A/B results).

**Panels ported**:
1. Recipe progress (poll `/recipes/current`, `/recipes`)
2. Job history (poll `/jobs/history?limit=10`, click to load curves)

**Verification**: Start gradient_gating_l0 recipe. Dashboard shows branch 1/2 "standard", phase progress updates. Branch completes, shows eval metrics. Branch 2 starts. When done, comparison table shows L1/PSNR for both conditions. Click a job in history — loss chart loads that job's data.

---

### M-UI-3: "I can see eval results and reconstructions"

**I can**: Run eval, see per-task metrics, see input vs reconstruction pairs, compare current model against a checkpoint.

**Foundation**: Image rendering component (base64 → img), eval table component with metric coloring.

**Panels ported**:
1. Eval metrics (POST `/eval/run`, display results table)
2. Reconstructions (POST `/eval/reconstructions`, display image pairs)
3. Checkpoint comparison (POST `/eval/checkpoint`, side-by-side table)

**Verification**: Load a trained checkpoint. Click "Run Eval" — see metrics table. See reconstruction pairs (input top, output bottom). Select a comparison checkpoint — see side-by-side metrics with best values highlighted.

---

### M-UI-4: "I can manage checkpoints and navigate experiment history"

**I can**: See the checkpoint tree, load/fork checkpoints, save new ones, see which checkpoint is current.

**Foundation**: Tree rendering component. Checkpoint action handlers.

**Panels ported**:
1. Checkpoint tree (GET `/checkpoints/tree`, load/fork buttons)
2. Checkpoint indicator (current checkpoint badge)
3. Save checkpoint form

**Verification**: See checkpoint tree with recipe grouping. Click "Load" on a checkpoint — model info updates. Click "Fork" — new branch appears. Save a checkpoint — appears in tree. Current checkpoint badge updates.

---

### M-UI-5: "I can manage tasks, datasets, and training config"

**I can**: Add/remove tasks, adjust weights, toggle tasks, generate synthetic datasets, view dataset samples, start/stop manual training runs.

**Foundation**: Form components (dropdowns, inputs, buttons). Dataset browser with thumbnails.

**Panels ported**:
1. Tasks panel (list, toggle, remove, weight adjustment)
2. Add task form
3. Datasets browser with sample thumbnails
4. Generate dataset form
5. Training start/stop controls
6. Device selector

**Verification**: Add a reconstruction task via the form. Toggle it off, on. Change weight. Remove it. Generate a thickness dataset. See its samples. Start a 100-step training run. Stop it early. Switch device to CPU and back.

---

### M-UI-6: "I can see detailed latent space visualizations"

**I can**: See latent traversals, sort-by-factor images, and attention maps for the current model.

**Foundation**: Image grid component for traversals/sorted/attention display.

**Panels ported**:
1. Latent traversals (GET `/eval/traversals`)
2. Sort by factor (GET `/eval/sort_by_factor`)
3. Attention maps (GET `/eval/attention_maps`)

**Verification**: Load a trained FactorSlotAutoencoder checkpoint. See traversal grids per factor. See highest/lowest activation sorts. See attention heatmaps. All images render at correct size with pixel-art rendering.

---

## Trainer-Side Changes Required

1. **CORS headers** — add `CORSMiddleware` to FastAPI app (allow origin `*` for dev).
2. **Serve static files** — mount the Astro build output at `/` on the trainer.
3. **Re-enable SSE (optional, M-UI-1 stretch)** — fix the lock convoy by replacing the blocking `jobs.stream()` iterator with an async queue. Not required for M-UI-1 (polling works fine at 5s).

## What Dies

- `acc/ui/` — entire directory (Starlette app, partials, actions, api client)
- `acc/ui_main.py` — UI process entry point
- `run_ui.sh` — UI launch script
- `control-ui` tmux session — no longer needed

## What Lives

- `acc/trainer_api.py` — all endpoints unchanged
- `acc/trainer_main.py` — unchanged
- `acc/ui/static/dashboard.css` — color values/theme ported to new CSS
- `acc/ui/components.py` — reference for health colors, metric thresholds

## File Structure (New)

```
dashboard/
  astro.config.mjs
  package.json
  src/
    layouts/
      Layout.astro          — page shell, dark theme, responsive grid
    pages/
      index.astro           — main dashboard page
    components/
      LossChart.tsx          — Chart.js wrapper (React island)
      StepCounter.astro      — inline step display
      HealthBadge.astro      — connection status
      ModelInfo.astro        — model description
      LossSummary.astro      — loss summary table
      RecipeProgress.astro   — recipe/branch/phase display
      JobHistory.astro       — clickable job list
      EvalTable.astro        — eval metrics table
      ReconPairs.astro       — reconstruction image pairs
      CheckpointTree.astro   — checkpoint tree with actions
      ... (one per panel)
    lib/
      api.ts                 — fetch wrapper with silent failure, retry, caching
      polling.ts             — setInterval-based polling with configurable intervals
      theme.ts               — HEALTH_COLORS, CHART_COLORS, metric thresholds
    styles/
      global.css             — dark theme, responsive grid, panel styles
  public/
    favicon.svg
```
