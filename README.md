# Autoencoder Control Center (ACC)

An interactive training workbench for variational autoencoders with human-in-the-loop guidance. Train, probe, checkpoint, fork, and steer autoencoder latent spaces from a live dashboard.

## Architecture

**Two-process design** (non-negotiable):

```
                    +-----------------+
  Browser -------->|  UI Process     |  :8080  (Starlette + HTMX + SSE)
                    |  Hot-reloads    |  Stateless — all state from trainer API
                    +--------+--------+
                             |
                             | HTTP / SSE
                             v
                    +-----------------+
                    | Trainer Process  |  :8787  (Flask JSON API)
                    | Never restarts   |  Owns: model, tasks, optimizer, checkpoints
                    +-----------------+
```

- **Trainer process** (:8787) — owns the model, tasks, datasets, optimizer, checkpoint store. Exposes a JSON API. Never restarts during a session.
- **UI process** (:8080) — Starlette + HTMX dashboard. Stateless proxy to the trainer API. Hot-reloads freely via `uvicorn --reload`.

## What's in here

The pipeline is **model-agnostic**: any `nn.Module` whose `forward()` returns `dict[str, Tensor]` keyed by `ModelOutput` enum values can be trained. Tasks read what they need from the dict. The Trainer is a dumb pipe.

**Two model architectures** proven through the same pipeline:

1. **Simple Autoencoder** — CNN encoder + decoder, flat latent vector. Trains on MNIST.
2. **Factor-Slot Cross-Attention Autoencoder** — partitions latent into named factor groups (position, scale, shape), cross-attention decoder. Trains on synthetic shapes with ground-truth factor labels.

**Task types** (config, not inheritance):

- `ClassificationTask` — linear probe, cross-entropy loss, accuracy eval
- `ReconstructionTask` — decoder output, L1 loss, PSNR eval
- `RegressionTask` — linear probe, MSE loss, MAE eval

Any task can target a latent slice via `latent_slice=(start, end)` config. A ClassificationTask on `z[12:16]` and one on all of `z` are the **same class with different config**.

## Running

### Both processes locally

```bash
# Terminal 1: trainer
python -m acc.trainer_main

# Terminal 2: UI (hot-reloads)
python -m acc.ui_main
```

Open http://localhost:8080 in a browser.

### Verification tests

```bash
python -m acc.test_m1      # Full training loop (simple autoencoder + MNIST)
python -m acc.test_m1_5    # Model-agnostic dict protocol + latent slicing
python -m acc.test_m1_75   # Factor-slot autoencoder through same pipeline
```

## Milestones

| Milestone | Status | What it proves |
|-----------|--------|----------------|
| M1 | Done | Full loop: model, tasks, train, eval, checkpoint, dashboard |
| M1.5 | Done | Model-agnostic forward protocol (`ModelOutput` dict, `latent_slice`) |
| M1.75 | Done | Factor-Slot autoencoder trains through same pipeline, gradient isolation |
| M1.9 | Next | Split-machine deployment: trainer on GPU, UI on laptop via Tailscale |

## Key design decisions

- **Reconstruction is not special** — it's just another task. No special-casing in the Trainer.
- **Tasks are config, not inheritance** — `latent_slice` replaces per-model task subclasses.
- **All `forward()` calls return `dict[str, Tensor]`** — the `ModelOutput` enum is the contract between models and tasks.
- **The Trainer is a dumb pipe** — calls `forward()`, passes dict to task, runs backward.
- **Two processes are non-negotiable** — trainer state survives UI changes.

## Directory structure

```
acc/
  model_output.py              # ModelOutput enum — the contract
  autoencoder.py               # Simple autoencoder
  factor_group.py              # FactorGroup dataclass
  factor_slot_autoencoder.py   # Factor-Slot cross-attention autoencoder
  dataset.py                   # AccDataset, load_mnist
  trainer.py                   # Trainer (model-agnostic)
  jobs.py                      # JobManager
  checkpoints.py               # CheckpointStore
  trainer_api.py               # Trainer HTTP API (Flask)
  trainer_main.py              # Trainer process entry
  ui_main.py                   # UI process entry
  tasks/
    base.py                    # Task ABC with latent_slice
    classification.py          # ClassificationTask
    reconstruction.py          # ReconstructionTask
    regression.py              # RegressionTask
  layers/
    conv_block.py              # ConvBlock, ConvTransposeBlock
    res_block.py               # ResBlock
    factor_head.py             # FactorHead
    cross_attention.py         # CrossAttentionBlock, FactorEmbedder
  generators/
    shapes.py                  # Synthetic shapes with factor labels
  ui/
    app.py                     # Starlette + HTMX dashboard
docs/
  HOW_WE_WORK.md               # Working principles
  WRITING_MILESTONES.md         # Milestone writing guide
  M1.5_M1.75_PLAN.md           # M1.5 + M1.75 plan (completed)
```
