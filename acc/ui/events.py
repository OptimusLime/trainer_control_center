"""Dashboard event constants for HTMX-driven cross-panel refresh.

Events are emitted via HX-Trigger response headers by action endpoints.
Panels declare which events they respond to via hx-trigger attributes.

This is the contract between actions (which cause state changes) and
panels (which display state). Adding a new panel requires only adding
an hx-trigger attribute — zero changes to action handlers.

Phase 2 of the UI refactor wires these into HX-Trigger headers.
Phase 1 defines the constants so all code references them instead of
magic strings.
"""

# Fired when a checkpoint is loaded, saved, or forked.
# Panels that depend on model weights, checkpoint metadata, or tree structure
# should listen to this event.
CHECKPOINT_CHANGED = "checkpoint-changed"

# Fired when a training job completes (SSE "done" event).
# Panels that display eval results, reconstructions, or training stats
# should listen to this event.
TRAINING_DONE = "training-done"

# Fired when a model is created, loaded, or its structure changes.
# Panels that depend on model existence or description should listen.
MODEL_CHANGED = "model-changed"

# Fired when tasks are added, removed, toggled, or weights changed.
# Panels that display task lists or task-dependent eval should listen.
TASKS_CHANGED = "tasks-changed"

# Fired when a dataset is generated or removed.
# Panels that display dataset lists or dataset-dependent forms should listen.
DATASETS_CHANGED = "datasets-changed"
