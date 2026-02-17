# ACC Roadmap — Milestones M2 through M6

## Status

| Milestone | Status | Summary |
|-----------|--------|---------|
| M1 | Done | Full training loop from code + UI dashboard |
| M1.5 | Done | Model-agnostic forward protocol (`ModelOutput` dict, `latent_slice`) |
| M1.75 | Done | Factor-Slot autoencoder through same pipeline, gradient isolation |
| M1.9 | Done | Split-machine deployment via Tailscale |
| M2 | Next | Hot-reload tasks + new task from code |
| M3 | Planned | Synthetic dataset generation + dataset hot reload |
| M4 | Planned | Checkpoints as a tree + forking |
| M5 | Planned | UFR evaluation + diagnosis panel |
| M6 | Planned | Model expansion (layer addition) |

---

## M2: Hot Reload Tasks + New Task from Code

**Functionality:** I can write a new Task subclass in a Python file inside `acc/tasks/`,
save it, and it appears in the dashboard's [+ Task] menu without restarting anything. My
AI assistant can do this too — it writes the file, the system picks it up.

**Foundation:** `TaskRegistry` with file-watcher-driven `importlib.reload`, task
auto-discovery via `Task` subclass scanning. This is the mechanism that makes AI-assisted
task creation possible.

**Details:**

- File watcher on `acc/tasks/` directory in the trainer process
- On `.py` file change: reload module, re-scan for Task subclasses, update registry
- Dashboard [+ Task] dropdown populated from registry
- New task class appears in dropdown within 2 seconds of file save
- Existing tasks (already attached to model) are not disrupted by registry update
- Test: create `acc/tasks/cosine_probe.py` with `CosineProbeTask`, save, verify it
  appears in dashboard
- **Implements:** `TaskRegistry`, file watcher integration in trainer process, importlib
  reload safety (try/except around reload — syntax errors don't crash the trainer)
- **Proves:** Hot reload works for task code without killing model state

**Verification:**

1. System running from M1 (trainer + UI, model loaded, tasks active)
2. Create file `acc/tasks/dummy_task.py`:
   ```python
   class DummyTask(Task):
       def _build_head(self, latent_dim): return nn.Linear(latent_dim, 1)
       def compute_loss(self, model_output, batch): return model_output["latent"].mean()
       def evaluate(self, ae, device): return {"dummy": 0.0}
   ```
3. Save file -> within 2 seconds, "DummyTask" appears in [+ Task] dropdown
4. Add DummyTask -> task card appears -> train -> dummy loss shows in chart
5. Model weights and existing task probes unchanged by the reload

**Cleanup Audit:**

- [ ] Does reload handle import errors gracefully? (display error in dashboard, don't crash)
- [ ] Can we hot-reload dataset generators the same way?
- [ ] Should the registry also scan for Autoencoder subclasses (new model architectures)?

---

## M3: Synthetic Dataset Generation + Dataset Hot Reload

**Functionality:** I can write a Python function that generates a synthetic dataset
(images + labels), save it in `acc/generators/`, and it appears in the [+ Dataset] menu. I
generate the dataset from the dashboard, browse the images, and create tasks on it.

**Foundation:** `DatasetGenerator` protocol with `generate(config) -> AccDataset`,
generator registry with hot reload (same mechanism as tasks). This is how we build
arbitrary training data on the fly.

**Details:**

- Generator protocol: a class with a `generate()` method that returns an `AccDataset`
- File watcher on `acc/generators/` — same importlib reload pattern as tasks
- Built-in generators: `MNISTGenerator`, `SyntheticShapesGenerator` (circles, squares,
  triangles on backgrounds)
- Dashboard: [+ Dataset] -> pick generator -> configure (form fields from generator's
  Config class) -> [Generate] -> progress bar -> dataset appears in browser with thumbnails
- Generated datasets saved to `acc/data/` as .pt files (persist across restarts)
- **Implements:** `DatasetGenerator` protocol, `SyntheticShapesGenerator`, generator
  registry, dataset persistence, dashboard generator UI
- **Proves:** Custom synthetic data creation works end-to-end. AI assistant can write a
  generator file and it's immediately usable.

**Verification:**

1. System running from M2
2. Click [+ Dataset] -> select "SyntheticShapes" -> set n=500, shapes=circle+square ->
   [Generate]
3. Progress bar fills -> dataset "shapes_500" appears in dataset browser -> thumbnails show
   circles and squares
4. Create `acc/generators/symmetry_gen.py` with a generator that produces symmetric
   patterns + axis labels
5. Save -> "SymmetryGenerator" appears in [+ Dataset] menu within 2 seconds
6. Generate symmetry dataset -> create RegressionTask on it -> train -> MAE shows in eval

**Cleanup Audit:**

- [ ] Is dataset persistence (.pt files) appropriate or should we use a more structured format?
- [ ] Should generators support incremental generation (add more images to existing dataset)?
- [ ] Can we preview generated images before committing the full dataset?

---

## M4: Checkpoints as a Tree + Forking

**Functionality:** I can see my checkpoint history as a tree (not a flat list), fork from
any checkpoint to try a different training path, and revert to any previous state. The tree
visualization shows which path I'm currently on.

**Foundation:** `CheckpointTree` with parent-child tracking, fork operation,
current-branch indicator. This enables the "explore, fail, revert, try again" workflow.

**Details:**

- Checkpoint tree rendered as a DAG in the sidebar
- Current checkpoint highlighted
- Fork button on any checkpoint -> creates new branch, loads that state
- Checkpoint metadata: tag, step count, timestamp, summary metrics
- Delete checkpoint (with confirmation, can't delete if it has children)
- **Implements:** `CheckpointTree` (parent_id tracking, tree rendering), fork operation,
  branch indicator
- **Proves:** Non-linear exploration of training paths works

**Verification:**

1. Train 1000 steps -> save "baseline"
2. Train 1000 more -> save "path_a"
3. Click "baseline" -> [Fork] -> creates "fork_b" branch
4. Train 1000 steps on fork_b -> save "path_b"
5. Tree shows: baseline -> path_a (branch 1), baseline -> path_b (branch 2)
6. Load path_a -> verify metrics match what they were when path_a was saved

**Cleanup Audit:**

- [ ] Is the tree visualization clear enough when there are 10+ checkpoints?
- [ ] Should we track which tasks were active at each checkpoint?
- [ ] Can we diff metrics between two checkpoints?

---

## M5: UFR Evaluation + Diagnosis Panel

**Functionality:** I can run the UFR factoring evaluation on my model and see a heatmap of
which concepts transfer across contexts and which are entangled. This tells me what tasks
to add next.

**Foundation:** `FactoringEvaluator` with configurable concepts/contexts, `UFRMetrics`,
diagnosis panel showing the factoring matrix. This is the metric that guides our task
selection.

**Details:**

- UFR evaluation as a dashboard action (button: [Run UFR Eval])
- Concept x context heatmap: rows = concepts (fluffy, round, etc.), columns = contexts
  (animal, vehicle, etc.), cells = probe transfer score
- Compositional independence matrix: concept pairs colored by independence score
- Single UFR score displayed prominently
- For MNIST: concepts = digit identities, contexts = clean/noisy/rotated
- Runs on eval split, takes ~30 seconds on a 3090
- **Implements:** `FactoringEvaluator`, `LinearProbeTrainer`, `UFRMetrics`, heatmap
  visualization in dashboard
- **Proves:** UFR score differentiates between training conditions (model trained with
  diverse tasks scores higher than model trained with classification only)

**Verification:**

1. Load "baseline" checkpoint (classification only, 1000 steps)
2. Run UFR eval -> UFR score = X, heatmap shows some entangled pairs
3. Load "path_b" checkpoint (classification + symmetry + shapes tasks)
4. Run UFR eval -> UFR score = Y
5. Verify Y > X (diverse tasks -> better factoring)
6. Heatmap shows specific improvements in the concept pairs targeted by added tasks

**Cleanup Audit:**

- [ ] Is the concept/context set appropriate for MNIST scale?
- [ ] Does the evaluation run fast enough for interactive use?
- [ ] Can we run UFR per layer (not just final latent)?

---

## M6: Model Expansion (Layer Addition)

**Functionality:** I can take my trained 1-layer autoencoder, add a second encoder/decoder
layer, preserve existing weights, and continue training. The model gets deeper without
starting over.

**Foundation:** `Autoencoder.expand()` method that adds layers while preserving existing
weights. Expansion as a checkpoint operation (new checkpoint with expanded architecture,
parent = pre-expansion checkpoint).

**Details:**

- Dashboard button: [Expand Model] -> select where to add layer -> configure new layer
- Existing weights preserved exactly
- New layer weights initialized (small random or identity-like)
- All existing task probes detached and re-attached (latent_dim may change)
- New checkpoint created automatically (expansion is a checkpoint event)
- **Implements:** `Autoencoder.insert_encoder_layer()`,
  `Autoencoder.insert_decoder_layer()`, probe re-attachment logic
- **Proves:** Progressive model growth works without losing learned features

**Verification:**

1. 1-layer model trained to 72% accuracy on MNIST
2. Click [Expand Model] -> add ConvBlock(32->64, stride=2) as encoder layer 2, matching
   decoder layer
3. New checkpoint created -> model now has 2 encoder layers
4. Run eval -> accuracy drops (expected — new layer is random)
5. Train 1000 steps -> accuracy recovers to > 65%
6. Train 2000 more -> accuracy exceeds original 72% (deeper model has more capacity)
7. Load pre-expansion checkpoint -> verify original 1-layer model loads correctly

**Cleanup Audit:**

- [ ] Does probe re-attachment handle latent_dim changes correctly?
- [ ] Should we support layer removal (compression) as well?
- [ ] Is weight initialization for new layers appropriate? (Kaiming? Identity? Zero?)

---

## The Loop

The milestones build toward one workflow:

**Task -> Dataset -> Train -> Evaluate -> Diagnose -> Task.**

- M1 establishes the full loop from code
- M1.5/M1.75 prove the pipeline is model-agnostic
- M1.9 enables GPU training from a laptop
- M2 makes task creation instant (hot reload)
- M3 makes dataset creation instant (hot reload)
- M4 makes exploration safe (checkpoint tree + fork)
- M5 tells you what to do next (UFR diagnosis)
- M6 lets the model grow without starting over

Each milestone adds a capability that accelerates the loop.
