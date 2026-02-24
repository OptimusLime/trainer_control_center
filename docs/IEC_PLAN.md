# Interactive Evolutionary CPPN Autoencoder — Plan

## Summary

Build an interactive convolutional autoencoder where the human grows the architecture from scratch — starting from a single hidden node — through mutation, targeted SGD bursts, and visual inspection. Convolution channels are CPPN nodes with heterogeneous per-channel activations (sin, cos, gaussian, tanh, etc.). The goal is to produce a Unified Factored Representation (UFR) by replacing SGD as the primary optimizer with human judgment operating through the dashboard.

## Context & Motivation

The Fractured Entangled Representation (FER) hypothesis (Kumar, Clune, Lehman, Stanley 2025) demonstrates that SGD-trained networks produce internally disorganized representations even when output quality is high. The only known process that produces clean UFRs is PicBreeder — interactive evolution with CPPNs where humans select on output quality and internal representation quality emerges as a side effect.

Nobody has ever produced a UFR autoencoder. We attempt it by giving the human fine-grained control over network growth.

### Why this is a surgical change from where we are

We already have:
- `Trainer` (multi-task, model-agnostic, takes any `nn.Module` — calls `model(batch[0])`, gets `dict[ModelOutput, Tensor]`, passes to `task.compute_loss()`)
- `CheckpointStore` (tree-structured persistence with fork/lineage)
- `Task` system (reconstruction via L1 loss, classification, eval-only metrics)
- `TrainerAPI` with `/train/*`, `/checkpoints/*`, `/eval/*` endpoints, `_is_model_busy()` guard
- `StepInspector` pattern (separate page, own store, action-driven not polled, mutually exclusive session via `self._inspector is not None` check)
- Astro frontend with shared patterns (nanostores, panel components)

We are NOT rewriting any of this. The ConvCPPN model is a new `nn.Module` that plugs into the existing `Trainer`. The genome is a new data structure that wraps the model. The `/iec` page is a new Astro page like `/inspect`. The API extensions are new endpoints under `/iec/*`.

## Naming Conventions

- Model: `ConvCPPN` (module), `ConvCPPNGenome` (topology descriptor)
- Layer: `ConvCPPNLayer` (single conv + mask + heterogeneous activation)
- Activation: `HeterogeneousActivation` (grouped dispatch module)
- Files: `acc/models/conv_cppn.py` (model + genome + mutations), `acc/iec.py` (IEC session manager)
- API: `/iec/*` endpoints in `trainer_api.py`
- Frontend: `dashboard/src/pages/iec.astro`, `dashboard/src/components/Iec*.tsx`, `dashboard/src/lib/iec-store.ts`, `dashboard/src/lib/iec-types.ts`
- Genome mutations: `add_channel`, `remove_channel`, `add_connection`, `remove_connection`, `change_activation`, `add_layer`, `remove_layer`

## The Simplest Starting Architecture

```
Input: [image, X, Y, Gauss] = 4 channels at 28x28
  |
  Conv2d(4, 1, kernel_size=3, padding=1)  +  activation (e.g. tanh)
  |
  AdaptiveAvgPool2d(3)  →  1 channel at 3x3  =  9 latent values
  |
  ConvTranspose2d(1+3, 1, ...) + Upsample to 28x28   (1 latent + X,Y,Gauss at 3x3)
  |
  Conv2d(1, 1, kernel_size=3, padding=1) + sigmoid  →  1 channel at 28x28 = reconstruction
```

One encoder conv layer, one hidden channel, average pool to 3x3 bottleneck, one decoder conv layer. This is the absolute minimum — a single CPPN node processing the spatial input. The human grows from here.

As topology grows, channels producing features <= 3x3 skip the average pool. The genome tracks spatial resolution per channel.

## Core Technical Design

### CPPN ↔ Conv Mapping

A CPPN node = a conv output channel. A CPPN connection = a weight slice `weight[j, i, :, :]` in the conv kernel. The conv operation already computes `out[j] = bias[j] + sum_i conv2d(input[i], weight[j, i])` — this IS a CPPN layer with shared activation. We add two things: **(1)** per-channel heterogeneous activation, **(2)** connection masking via a `[C_out, C_in]` binary buffer.

### Per-Channel Activation: Grouped Dispatch

After conv produces `[B, C_out, H, W]`, apply different activations per channel. Implementation: group channels by activation type, apply each activation to its group in one vectorized op.

```python
ACTIVATION_REGISTRY: dict[str, Callable] = {
    'identity': lambda x: x,
    'relu':     torch.relu,
    'sigmoid':  torch.sigmoid,
    'tanh':     torch.tanh,
    'sin':      torch.sin,
    'cos':      torch.cos,
    'gaussian': lambda x: torch.exp(-x * x / 2.0),
    'abs':      torch.abs,
    'softplus': nn.functional.softplus,
}
```

`HeterogeneousActivation(nn.Module)` stores `(act_name, fn)` tuples and registered index buffers (`idx_{act_name}`). Forward: `out[:, idx] = fn(x[:, idx])` per group. Cost: one kernel launch per unique activation type per layer (4-6 typical). Negligible at 28x28.

### Connection Masking

`ConvCPPNLayer` holds a `conn_mask` buffer of shape `[C_out, C_in, 1, 1]` (broadcast over kernel dims). Applied at forward time: `weight = self.conv.weight * self.conn_mask`. The mask is a registered buffer (not a parameter). Disabled connections have zero weight in the output regardless of optimizer state.

Gradient zeroing for masked connections: `zero_masked_grads()` called after `loss.backward()`, before `optimizer.step()`. This prevents Adam from accumulating momentum on dead connections.

### Skip Connections: Identity Relay Channels

A skip from layer S to layer T (T > S+1) is implemented as pass-through channels in each intermediate layer. A pass-through channel has:
- One incoming connection (from the source channel in the previous layer)
- A 1x1-centered 3x3 identity kernel: `[[0,0,0],[0,1,0],[0,0,0]]`
- Identity activation
- Frozen weights (excluded from optimizer or re-stamped)

Multi-layer skip from S to T costs T-S-1 relay channels. Pass-throughs inherit stride of their layer (spatial resolution halves if the layer has stride 2 — this is correct CPPN behavior).

### Coordinate Channel Injection

Precomputed at each spatial resolution as registered buffers:
- `X ∈ [-1,1]`, `Y ∈ [-1,1]`, `Gauss = exp(-(X²+Y²)/2)` — 3 channels
- Stored at 28x28 (encoder input), 3x3, 7x7, 14x14 (decoder scales)
- Encoder: concatenated once at input → `[B, 1+3, 28, 28]`
- Decoder: concatenated at every layer's input → each decoder layer sees `[features + 3 coords]` as input channels. The coord weight slices in the conv ARE trainable (the network learns how to use positional info).

### Stride and Resolution Map

```
Encoder:
  Input:  [B, 4, 28, 28]     (image + x,y,gauss)
  L1:     [B, C1, 14, 14]    stride=2, padding=1, kernel=3
  L2:     [B, C2, 7, 7]      stride=2, padding=1, kernel=3
  L3:     [B, C3, 3, 3]      stride=2, padding=0, kernel=3  (7→3)

Bottleneck: AdaptiveAvgPool2d(3) on encoder output → [B, C3, 3, 3]
Latent: flatten → [B, C3*9]

Decoder:
  Input:  [B, C3+3, 3, 3]    (latent + x,y,gauss at 3×3)
  L4:     [B, C2+3, 7, 7]    ConvTranspose2d stride=2, pad=0, kernel=3 → concat coords at 7×7
  L5:     [B, C1+3, 14, 14]  ConvTranspose2d stride=2, pad=1, out_pad=1, kernel=3 → concat coords at 14×14
  L6:     [B, 1, 28, 28]     ConvTranspose2d stride=2, pad=1, out_pad=1, kernel=3 → sigmoid
```

The starter 4→1→1 architecture uses only L1 (encoder) and L6 (decoder) with a single channel each.

### Model Protocol Compliance

`ConvCPPN` must satisfy the model protocol used by `Trainer` and `Task`:

```python
class ConvCPPN(nn.Module):
    @property
    def has_decoder(self) -> bool: ...      # Always True
    @property
    def latent_dim(self) -> int: ...        # C_bottleneck * 3 * 3 (e.g. 1*9=9 for starter)
    def config(self) -> dict: ...           # {"class": "ConvCPPN", "genome": genome.to_dict(), ...}
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # x: [B, 1, 28, 28] (raw image — coord channels concatenated internally)
        return {
            ModelOutput.LATENT: ...,         # [B, D] where D = latent_dim (pooled+flattened)
            ModelOutput.RECONSTRUCTION: ..., # [B, 1, 28, 28]
            ModelOutput.SPATIAL: ...,        # [B, C, h, w] encoder spatial features pre-pool
        }
```

**Critical**: The Trainer calls `self.autoencoder(batch[0])` where `batch[0]` is `[B, 1, 28, 28]` from MNIST. The ConvCPPN receives 1-channel input and internally concatenates the 3 coordinate channels. It does NOT expect the caller to provide coords.

`ReconstructionTask` reads `model_output[ModelOutput.RECONSTRUCTION]` and computes `F.l1_loss(recon, images)` against `batch[0]`. It checks `autoencoder.has_decoder` in `check_compatible()`. The `LATENT` key is used by probe-head tasks (classification, regression) via `task._get_latent(model_output)`.

## Genome Structure

```python
@dataclass
class ChannelDescriptor:
    activation: str                 # key into ACTIVATION_REGISTRY
    is_passthrough: bool = False    # identity relay for skip connections
    passthrough_source: int = -1    # source channel idx in previous layer
    frozen: bool = False            # pass-through weights don't train

@dataclass
class LayerGenome:
    channel_descriptors: list[ChannelDescriptor]
    connection_mask: list[list[int]]   # C_out × C_in binary, JSON-serializable
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1

@dataclass
class ConvCPPNGenome:
    encoder_layers: list[LayerGenome]
    decoder_layers: list[LayerGenome]
    metadata: dict = field(default_factory=dict)  # parent_id, generation, tags

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> 'ConvCPPNGenome': ...
```

Weights are NOT stored in the genome. The genome is topology only. Weights live in the `ConvCPPN` module's `state_dict()`. When persisting, we store `(genome.to_dict(), model.state_dict())` as a pair.

## Phases

**Build sequence: end-to-end first, complexify later.** Every milestone produces a working app you can see and interact with in the browser. Backend and frontend ship together. Stubs are fine — silent stubs are not. If a button exists, it either works or shows an explicit "not yet implemented" message.

### Phase 1: Static End-to-End (M-IEC-1) --- COMPLETED

**Functionality:** I can open `/iec` in the browser, see a 4→1→1 ConvCPPN architecture summary, see 8 reconstruction pairs (random noise output — untrained), and click a "Step" button that returns a hardcoded loss value.

**Foundation:** The full vertical pipeline proven working: `ConvCPPN(nn.Module)` with `from_genome()`/`to_genome()`, `default_genome()`, `ACTIVATION_REGISTRY`, `HeterogeneousActivation`, `ConvCPPNLayer` with `conn_mask` — all the core model abstractions. `IECSession` with `setup()` and `get_state()` and `get_reconstructions()`. `POST /iec/setup`, `GET /iec/state`, `GET /iec/reconstructions`, `POST /iec/teardown` endpoints. `iec-store.ts`, `iec-types.ts`, `IecPanel.tsx`, `iec.astro` page. **Proves:** ConvCPPN → Trainer → API → Store → UI integration works. Any shape mismatches, protocol violations, or wiring bugs surface here, not in Phase 4.

Tasks:
1. `acc/models/conv_cppn.py` — full model implementation:
   - `ACTIVATION_REGISTRY` dict (identity, relu, sigmoid, tanh, sin, cos, gaussian, abs, softplus)
   - `ChannelDescriptor`, `LayerGenome`, `ConvCPPNGenome` dataclasses with `to_dict()`/`from_dict()`
   - `default_genome()` → 4→1→1 starter
   - `HeterogeneousActivation(nn.Module)` — grouped dispatch
   - `ConvCPPNLayer(nn.Module)` — conv + conn_mask buffer + heterogeneous activation + `zero_masked_grads()`
   - `make_coord_channels(H, W)` helper
   - `ConvCPPN(nn.Module)` — encoder layers, AdaptiveAvgPool2d(3), decoder layers, coord buffers. `has_decoder=True`, `latent_dim`, `config()`. Forward: `[B,1,28,28]` in → `{LATENT, RECONSTRUCTION, SPATIAL}` out.
   - `ConvCPPN.from_genome()` / `to_genome()` round-trip
2. `acc/iec.py` — `IECSession` (partial — enough for this phase):
   - `setup(device, mnist_dataset, genome_dict=None)` — builds ConvCPPN, creates Trainer with ReconstructionTask, loads MNIST
   - `get_state()` — returns genome dict, step count, last_loss, latent_dim, architecture summary, undo_depth
   - `get_reconstructions(n=8)` — inference on eval images, returns base64 input/output grids
   - `step(n, lr)` — **stub**: returns `{"error": "not_implemented", "message": "Training not yet wired"}` with 501 status
   - `mutate(...)` — **stub**: returns 501
   - `undo()` — **stub**: returns 501
   - `teardown()` — cleanup refs
3. API endpoints in `trainer_api.py`:
   - `self._iec: Optional[IECSession] = None` field
   - `self._iec is not None` added to `_is_model_busy()`
   - `POST /iec/setup` — guard, create session, return state
   - `GET /iec/state` — return session state
   - `GET /iec/reconstructions` — return inference images
   - `POST /iec/step` — delegates to session (returns 501 stub for now)
   - `POST /iec/mutate` — delegates to session (returns 501 stub)
   - `POST /iec/undo` — delegates to session (returns 501 stub)
   - `POST /iec/teardown` — teardown session
4. Frontend — full page shell with static display:
   - `dashboard/src/lib/iec-types.ts` — TypeScript types for state, reconstructions, genome shape
   - `dashboard/src/lib/iec-store.ts` — nanostore `$iec` atom with actions: `setupIec()`, `teardownIec()`, `fetchState()`, `fetchReconstructions()`, `stepIec(n, lr)` (calls endpoint, handles 501 gracefully)
   - `dashboard/src/pages/iec.astro` — page shell
   - `dashboard/src/components/IecPanel.tsx`:
     - Auto-setup on mount (calls `POST /iec/setup`)
     - Architecture summary text: "Encoder: 4→1(tanh) | Bottleneck: 1ch @ 3x3 | Decoder: 4→1(sigmoid)"
     - Reconstruction grid: 8 input + 8 output images side by side
     - Toolbar: lr input, Step button (grayed out / shows "not yet implemented" toast), step counter, loss display
     - Mutation controls: Add Channel button, activation dropdown — all disabled with "Coming in M-IEC-3" tooltip
     - Undo button — disabled

**Verification:** Open `http://localhost:4321/iec` in browser. Page loads. Architecture reads "Encoder: 4→1(tanh)...". Reconstruction grid shows 8 MNIST digits on the left, 8 random-looking outputs on the right (untrained model). Step button exists but shows "not implemented" when clicked. No console errors. Teardown works (navigate away, come back, re-setup works).

Also verify model protocol in isolation: `python -c "from acc.models.conv_cppn import ConvCPPN, default_genome; g = default_genome(); m = ConvCPPN.from_genome(g); import torch; x = torch.randn(2,1,28,28); out = m(x); print(out['reconstruction'].shape, out['latent'].shape, m.has_decoder, m.latent_dim)"` prints `torch.Size([2, 1, 28, 28]) torch.Size([2, 9]) True 9`.

### Phase 2: Live Training (M-IEC-2) --- COMPLETED

**Functionality:** I can click "Step x10" on `/iec`, see the loss decrease in real time, and see reconstructions update from noise to vaguely digit-shaped blobs.

**Foundation:** `IECSession.step()` fully wired — creates Trainer, runs `trainer.train(steps=n)`, returns loss history. `POST /iec/step` returns losses + fresh reconstructions in one response. Store action `stepIec()` updates state + reconstructions atomically. **This is the first milestone where the human sees the model learn.** Foundation for all future training interactions.

Tasks:
1. `acc/iec.py` — implement `step()` for real:
   - If lr provided and differs from current, rebuild Trainer with new lr
   - Call `trainer.train(steps=n)`, collect loss from each step's `on_step` callback
   - Call `model.zero_masked_grads()` after training (belt-and-suspenders for connection masks)
   - Update step_count, last_loss
   - Return `{"losses": [...], "step": step_count, "last_loss": last_loss}`
2. `POST /iec/step` endpoint — upgrade from stub. Returns `{"losses": [...], "step": int, "last_loss": float, "reconstructions": {inputs, outputs}}` (include fresh reconstructions so UI updates in one round-trip).
3. Frontend updates:
   - `stepIec(n, lr)` action: calls `POST /iec/step`, updates `$iec.losses`, `$iec.step`, `$iec.lastLoss`, `$iec.reconstructions` from response
   - Step button enables. Loss displays as number. Step counter increments.
   - Loss history: simple inline sparkline or just the last N loss values as text (keep it minimal).

**Verification:** Open `/iec`. Click "Step x10" at lr=0.01. Loss displays a number (e.g. 0.35). Click "Step x10" again — loss is lower (e.g. 0.28). Click 5 more times — reconstructions visibly change from noise toward blurry digit shapes. Loss is below 0.20.

### Phase 3: Mutations End-to-End (M-IEC-3) --- COMPLETED

**Functionality:** I can add/remove channels, change activations, add/remove entire layers (encoder and decoder), and undo any mutation. Each layer shows its resolution (e.g. "28x28 -> 14x14"), channel pills are clickable to change activation, and the architecture panel shows clear data flow direction.

**Foundation:** Mutation functions as pure genome transforms (`add_channel`, `remove_channel`, `change_activation`, `add_connection`, `remove_connection`, `add_encoder_layer`, `remove_encoder_layer`, `add_decoder_layer`, `remove_decoder_layer`). `transfer_weights()` for preserving learned weights across structural changes. `_resize_mask_columns()` and `_sync_decoder_input_to_bottleneck()` for connection mask management during layer mutations. `ConvCPPN.resolution_info()` for per-layer resolution metadata. `IECSession.mutate()` and `undo()` with undo stack. `POST /iec/mutate` and `POST /iec/undo` endpoints. Store actions + UI controls wired.

Tasks:
1. `acc/models/conv_cppn.py` — mutation functions:
   - `add_channel(genome, layer_side, layer_idx, activation) -> genome`
   - `remove_channel(genome, layer_side, layer_idx, channel_idx) -> genome`
   - `add_connection(genome, layer_side, layer_idx, in_ch, out_ch) -> genome`
   - `remove_connection(genome, layer_side, layer_idx, in_ch, out_ch) -> genome`
   - `change_activation(genome, layer_side, layer_idx, channel_idx, new_activation) -> genome`
   - `add_encoder_layer(genome, position, activation, channels, stride) -> genome`
   - `remove_encoder_layer(genome, layer_idx) -> genome`
   - `add_decoder_layer(genome, position, activation, channels) -> genome`
   - `remove_decoder_layer(genome, layer_idx) -> genome`
   - `transfer_weights(old_model, new_model, old_genome, new_genome)` — copies compatible weight slices
   - `_resize_mask_columns()`, `_sync_decoder_input_to_bottleneck()` — mask management helpers
   - `ConvCPPN.resolution_info()` — returns per-layer input/output resolution
2. `acc/iec.py` — implement `mutate()` and `undo()`:
   - `mutate()`: push `(genome.to_dict(), model.state_dict())` to undo stack (cap at 20), apply mutation, build new model, transfer weights, rebuild trainer
   - Supports: `add_channel`, `remove_channel`, `change_activation`, `add_connection`, `remove_connection`, `add_layer`, `remove_layer`
   - `undo()`: pop undo stack, restore genome + weights, rebuild trainer
   - `get_state()` includes `resolutions` from `model.resolution_info()`
3. API endpoints — `POST /iec/mutate` and `POST /iec/undo` return full state (genome, step, loss, architecture, undo_depth, resolutions, reconstructions).
4. Frontend:
   - Encoder/decoder shown side by side with subtitles showing data flow ("image 28x28 -> bottleneck 3x3" / "bottleneck 3x3 -> output 28x28")
   - Per-layer resolution labels (e.g. "28x28 -> 14x14"), kernel size, stride, channel count
   - Channel pills with click-to-change-activation popover and x-to-remove
   - "Add Channel" per layer with activation selector
   - "Add Layer" button per side, "Remove Layer" button per non-output layer
   - Undo button with depth counter, Reset with confirmation
   - SVG loss line chart, normalize toggle on reconstructions
   - `IecLayerResolution`, `IecResolutions` types in `iec-types.ts`

**Verification:** Open `/iec`. Step x50 at lr=0.01 — loss around 0.20. Click "Add Channel" (encoder, layer 0, sin). Architecture updates to show 2 channels. Latent dim now 18. Step x50 more — loss decreases further. Click Undo — architecture reverts. Add an encoder layer — resolution labels update. Remove it — works. Add a decoder layer — decoder now has 4 layers. The human is driving both channel-level and layer-level architecture search.

### Phase 4: Checkpoints + Feature Maps (M-IEC-4)

**Functionality:** I can save named checkpoints, load them later, see per-channel feature map heatmaps for the current model, and compare architectures across saved checkpoints.

**Foundation:** `IECSession.save_checkpoint()` / `load_checkpoint()` via existing `CheckpointStore`. `get_feature_maps()` with forward hooks on `ConvCPPNLayer`. `POST /iec/checkpoint/save`, `POST /iec/checkpoint/load`, `GET /iec/checkpoints`, `GET /iec/features` endpoints. Checkpoint list + feature map display in UI.

Tasks:
1. `acc/iec.py` — implement checkpoint and feature map methods:
   - `save_checkpoint(tag, checkpoint_store)` — genome dict in metadata, model state_dict, step count
   - `load_checkpoint(checkpoint_id, checkpoint_store)` — rebuild model from stored genome, load weights
   - `list_checkpoints(checkpoint_store)` — filtered to IEC checkpoints
   - `get_feature_maps(image_idx=0)` — forward hooks on ConvCPPNLayers, capture per-channel activations, return as base64 heatmaps
2. API endpoints:
   - `POST /iec/checkpoint/save` — body: `{"tag": str}`
   - `POST /iec/checkpoint/load` — body: `{"id": str}`, returns full state + reconstructions
   - `GET /iec/checkpoints` — returns checkpoint list
   - `GET /iec/features?image_idx=0` — returns feature map heatmaps per layer per channel
3. Frontend:
   - Checkpoint controls: Save button (tag input), checkpoint list dropdown, Load button
   - Feature map panel: per-layer heatmap grid showing each channel's activation for a selected input image (reuse `InspectHeatmap` component)

**Verification:** Train a 1-channel model for 100 steps. Save as "1ch-baseline". Add sin + cos channels, train 100 more. Save as "3ch-mixed". Load "1ch-baseline" — architecture reverts, reconstructions match saved state. Load "3ch-mixed" — back to 3 channels. Feature maps panel shows 1 heatmap for 1-channel model, 3 heatmaps for 3-channel model. Each heatmap is visually distinct (sin channel shows periodic patterns, cos channel shows different patterns, tanh shows something else).

### Phase 5: Architecture Graph + Comparison (M-IEC-5)

**NOTE:** This phase is critical for usability. The genome JSON is unreadable — the human needs a topological graph to understand what they're building. Prioritize the DAG renderer before comparison features.

**Functionality:** I can see the CPPN topology as a visual DAG (channels as nodes, connections as edges), click any channel to see its feature map, and compare two checkpoints side-by-side.

**Foundation:** `IecArchGraph.tsx` canvas component (reusable DAG renderer). `POST /iec/compare` endpoint. Checkpoint tree visualization.

Tasks:
1. `IecArchGraph.tsx` — canvas-rendered DAG:
   - Layers as columns, channels as circles within each column
   - Color-coded by activation type (sin=purple, cos=blue, tanh=green, relu=red, etc.)
   - Connections as lines between circles. Disabled connections as dashed/faded.
   - Click channel → highlight and show its feature map heatmap below
2. `POST /iec/compare` endpoint — given two checkpoint IDs, return side-by-side reconstructions and per-channel feature maps for both
3. Checkpoint tree visualization in the sidebar (reuse pattern from main dashboard's checkpoint tree)

**Verification:** Graph renders the 4→1 encoder topology correctly. After adding 3 channels with different activations, graph shows 4 input → 3 hidden with connection lines. Clicking a channel shows its 28x28 activation heatmap for the current batch. Comparing two checkpoints shows their reconstructions side by side.

## Directory Structure (Anticipated)

```
acc/
  models/
    conv_cppn.py          # ConvCPPN, ConvCPPNLayer, HeterogeneousActivation,
                          # ConvCPPNGenome, ChannelDescriptor, LayerGenome,
                          # ACTIVATION_REGISTRY, mutations, transfer_weights
  iec.py                  # IECSession manager

dashboard/src/
  pages/
    iec.astro             # IEC page shell
  components/
    IecPanel.tsx          # Main IEC panel
    IecArchGraph.tsx      # Topology DAG visualization (Phase 5)
    IecReconGrid.tsx      # Reconstruction comparison grid
  lib/
    iec-store.ts          # Nanostore for IEC state
    iec-types.ts          # TypeScript types for IEC API
```

## What We Reuse (Not Rewrite)

| Existing | How IEC Uses It |
|----------|----------------|
| `Trainer` | IECSession creates a `Trainer(model, [recon_task], device, lr, batch_size=128)`. Calls `trainer.train(steps=n)` for SGD bursts. Trainer calls `model(batch[0])`, gets `dict[ModelOutput, Tensor]`, passes to task. |
| `ReconstructionTask` | Primary training task. L1 loss between `model_output[RECONSTRUCTION]` and `batch[0]`. Checks `model.has_decoder` in `check_compatible()`. |
| `CheckpointStore` | IEC checkpoints saved/loaded through existing store. Genome dict stored in checkpoint metadata. |
| `load_mnist()` | Same MNIST 28x28 dataset. |
| `TrainerAPI._is_model_busy()` | Extended with `self._iec is not None` check. Same 409 guard pattern. |
| `StepInspector` pattern | IEC page follows same lifecycle: `self._iec` field on TrainerAPI, mutually exclusive with inspector, setup/teardown endpoints, action-driven not polled. |
| `InspectHeatmap` | Reuse for feature map display. |
| `InspectWeightGrid` | Reuse for reconstruction grids. |

## What's New (Only New Things)

1. `ConvCPPN` + `ConvCPPNLayer` + `HeterogeneousActivation` — new nn.Modules (~250 lines)
2. `ConvCPPNGenome` + `ChannelDescriptor` + `LayerGenome` — new dataclasses (~80 lines)
3. `ACTIVATION_REGISTRY` + mutation functions + `transfer_weights` — ~150 lines
4. `IECSession` — new class (~250 lines)
5. `/iec/*` API endpoints — ~200 lines added to trainer_api.py
6. Frontend: iec page + panel + store + types — ~500 lines total

**Estimated total new code: ~1430 lines.** No existing code modified except: (a) adding `self._iec` field and `/iec/*` endpoint section to `trainer_api.py`, (b) adding `self._iec is not None` to `_is_model_busy()`.

## Phase Execution Order

Phase 1 → 2 → 3 → 4 → 5

**Every phase is end-to-end.** Phase 1 delivers a working `/iec` page in the browser with the model, API, and UI wired together (stubs for training/mutation). Each subsequent phase fills in real functionality where stubs were, and each is verified in the browser.

- After Phase 1: You can see the model and its reconstructions in the browser. Integration proven.
- After Phase 2: You can train the model and watch it learn. The core interactive loop works.
- After Phase 3: You can mutate the architecture and undo. The human IS the search algorithm.
- After Phase 4: You can save/load checkpoints and inspect feature maps. Persistence and diagnosis.
- After Phase 5: You can see the topology as a graph and compare checkpoints. Polish.

## Full Outcome

After all phases: The human opens `/iec`, sees a minimal 1-channel autoencoder. They train it for 100 steps, see blurry reconstructions. They add a sin channel and a cos channel, train more, see spatial frequency patterns emerge. They save checkpoints at each stage, fork to try different activation combinations, compare results. They grow the network to 8-12 channels across 2-3 layers, each hand-picked based on what they see in the feature maps and reconstructions. The internal representation is organized because the human organized it — each channel was added with intent, trained with purpose, and kept because it earned its place.
