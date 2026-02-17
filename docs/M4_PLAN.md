# M4 Plan — Device Selection + Checkpoint Safety (Thin)

## Summary

Fix the `map_location` bug in checkpoint loading and add device selection so we can target `cuda:0` or `cuda:1` explicitly. NOT full DataParallel — models are ~448K params, too small to benefit.

## Context & Motivation

- `checkpoints.py:125` uses `torch.load(path, weights_only=False)` with no `map_location`. Loading a CUDA checkpoint on CPU (or on a different GPU) will fail.
- After `load_state_dict`, no explicit `.to(device)` call — works by accident because model was already on device. Fragile.
- Device is auto-detected as `cuda` or `cpu` in `TrainerAPI.__init__()`. No way to target `cuda:1` on a 2-GPU machine.

## Functionality

**"I can choose which GPU to train on, and checkpoints load correctly on any device."**

## Foundation

- `map_location` in `CheckpointStore.load()` — checkpoint portability across devices.
- `GET /device` + `POST /device/set` API — device management infrastructure for future multi-GPU work.

## Changes

### Phase 1: Checkpoint `map_location` Fix

**File:** `acc/checkpoints.py`
- `load()` accepts optional `device` parameter
- Uses `torch.load(..., map_location=device)` 
- After `load_state_dict`, calls `autoencoder.to(device)` and `task.head.to(device)` for all tasks

### Phase 2: Device Selection API

**File:** `acc/trainer_api.py`
- `GET /device` — returns `{"current": "cuda:0", "available": ["cpu", "cuda:0", "cuda:1"]}`
- `POST /device/set` — body `{"device": "cuda:1"}` — validates device, moves model + probe heads, updates trainer

## Verification

`python -m acc.test_m4`

1. Save checkpoint on current device
2. Load checkpoint with explicit `map_location=cpu` — succeeds
3. Load checkpoint with `map_location` back to original device — succeeds
4. `GET /device` returns current device and available devices list
5. `POST /device/set` changes device, model moves to new device
6. Training still works after device change
7. Regression: M1.95, M2, M3 tests still pass
