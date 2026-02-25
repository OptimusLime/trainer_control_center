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

### Phase 4: Checkpoints + Feature Maps (M-IEC-4) --- COMPLETED

**Functionality:** I can save named checkpoints, load them later (even when architecture has changed), and see per-channel feature map heatmaps for every layer via canvas-rendered mini heatmaps.

**Foundation:** `CheckpointStore` extended with `load_metadata()` (reads genome without touching model) and `load_model_only()` (restores just model weights, fresh optimizer). `IECSession.save_checkpoint()` saves genome in `model_config` via `model.config()`, step count + loss in `metrics`. `IECSession.load_checkpoint()` rebuilds model from stored genome, then loads weights. `IECSession.get_feature_maps()` uses temporary forward hooks on all ConvCPPNLayers to capture per-channel activations as 2D arrays. Checkpoints filtered by `recipe_name="iec"`.

Tasks:
1. `acc/checkpoints.py` — extended with:
   - `load_metadata(checkpoint_id)` — reads .pt, returns Checkpoint with model_config (genome), without loading into model/trainer
   - `load_model_only(checkpoint_id, autoencoder, device)` — loads just model weights, skips optimizer/probes
2. `acc/iec.py` — checkpoint and feature map methods:
   - `save_checkpoint(tag, checkpoint_store)` — uses CheckpointStore.save(), genome stored in model_config, metrics has iec_step + iec_last_loss
   - `load_checkpoint(checkpoint_id, checkpoint_store)` — load_metadata() for genome, rebuild model, rebuild trainer, load_model_only() for weights
   - `list_checkpoints(checkpoint_store)` — filtered to recipe_name="iec"
   - `get_feature_maps(n=1)` — forward hooks on encoder_layers + decoder_layers, returns per-channel 2D arrays + input image
3. API endpoints:
   - `POST /iec/checkpoint/save` — body: `{"tag": str}`
   - `POST /iec/checkpoint/load` — body: `{"id": str}`, returns full state + reconstructions
   - `GET /iec/checkpoints` — returns filtered checkpoint list
   - `GET /iec/features` — returns per-layer per-channel activation arrays + input image
4. Frontend:
   - `IecCheckpoint`, `IecFeatureMaps`, `IecFeatureLayer`, `IecFeatureChannel` types
   - `saveCheckpoint()`, `loadCheckpoint()`, `fetchCheckpoints()`, `fetchFeatureMaps()` store actions
   - Checkpoint controls: tag input + Save button, checkpoint buttons for quick load
   - Feature map panel: Fetch/Refresh button, per-layer rows with canvas-rendered mini heatmaps (viridis colormap, auto-normalized)

**Verification:** Train 50 steps, save as "test-50". Add cos channel, train 50 more, save as "2ch-cos-100". Load "test-50" — architecture reverts to 1 channel, latent=9, step=50. Load "2ch-cos-100" — back to 2 channels. Feature maps show 1 heatmap per encoder channel (14x14) and 1 per decoder channel (7x7, 14x14, 28x28). Verified via API + browser.

### Phase 5: Architecture Graph + Feature Visualization (M-IEC-5) --- COMPLETED

**NOTE:** This phase is critical for usability. The genome JSON is unreadable — the human needs a topological graph to understand what they're building. Prioritize the DAG renderer before comparison features.

**Functionality:** I can see the CPPN topology as a visual DAG (channels as nodes, connections as edges), click any channel to see its feature map, gradient map, and per-input kernels. Horizontal feature map strip shows input (image + X + Y + Gauss), encoder, latent, decoder, and output columns. Output column shows reconstruction + per-pixel L1 error map + gradient. Loss (L1 and total) displayed in feature map header.

**Foundation:** `IecArchGraph.tsx` SVG DAG renderer (layers as columns, channels as colored circles, connections as lines, coord diamonds, click-to-select). SSIM loss implementation in `ReconstructionTask` with configurable `ssim_weight`. Gradient maps per channel via backward pass. Kernel visualization per channel. Feature maps auto-load on session start.

Tasks:
1. `IecArchGraph.tsx` — SVG DAG renderer:
   - Layers as columns, channels as circles within each column
   - Color-coded by activation type (sin=purple, cos=blue, tanh=green, relu=red, etc.)
   - Coordinate inputs shown as diamonds
   - Connections as lines between circles. Click channel → highlight.
2. Horizontal feature map strip below graph — input (img+X+Y+Gauss), encoder, latent, decoder, output columns
3. Gradient maps per channel (red/blue diverging heatmap via backward pass on loss)
4. Kernel visualization per channel (green/purple diverging heatmap, per-input KxK kernels)
5. Output column: reconstruction image + per-pixel L1 error map + gradient
6. Loss display in feature map header (both L1 and total)
7. SSIM loss implementation in `ReconstructionTask` with configurable `ssim_weight` parameter
8. SSIM weight slider in sticky top bar
9. Feature maps auto-load on session start, clicking channels only highlights (no re-fetch)
10. Sticky training controls bar

**Verification:** Graph renders the 4→1 encoder topology correctly. After adding 3 channels with different activations, graph shows 4 input → 3 hidden with connection lines. Clicking a channel highlights it and shows its feature map in the strip below. Gradient maps show red/blue heatmaps. Kernel visualization shows per-input KxK grids. Output column shows reconstruction + error map. SSIM weight slider adjusts the loss formula.

### Phase 6: Kernel Editing + Selective Freezing (M-IEC-6) --- COMPLETED

**NOTE:** This phase gives the human direct control over individual conv kernels — the lowest-level weights in the network. Combined with the architecture graph and feature maps, this makes the human a true weight-level optimizer: they can see what each kernel detects, decide to keep it, freeze it, or manually set it to a known pattern (edge detector, Gabor filter, identity, etc.).

**Functionality:** I can click any channel in the architecture graph or feature map view and see its per-input kernels. I can freeze any individual channel so SGD leaves it alone during training. I can manually edit a kernel's values via a clickable grid or select from preset patterns (identity, horizontal edge, vertical edge, Gabor, blur, sharpen). After editing, I train — frozen channels stay fixed while SGD optimizes the rest.

**Foundation:** `POST /iec/freeze_channel` and `POST /iec/unfreeze_channel` endpoints. `POST /iec/set_kernel` endpoint for manual kernel editing. `ChannelDescriptor.frozen` field already exists in the genome. `ConvCPPNLayer.zero_masked_grads()` already zeros frozen channel gradients. Frontend: `KernelEditor` component with clickable weight cells and preset patterns.

Tasks:
1. Backend — freeze/unfreeze:
   - `IECSession.freeze_channel(side, layer_idx, channel_idx)` — sets `genome.channel_descriptors[i].frozen = True`, rebuilds model (or toggles in-place since frozen is handled in `zero_masked_grads`)
   - `IECSession.unfreeze_channel(...)` — reverse
   - `POST /iec/freeze_channel` and `POST /iec/unfreeze_channel` endpoints
   - Verify: frozen channels have zero gradient after backward, weights unchanged after training steps
2. Backend — kernel editing:
   - `IECSession.set_kernel(side, layer_idx, out_ch, in_ch, values: list[list[float]])` — directly sets `conv.weight[out_ch, in_ch]` to provided 2D values
   - `POST /iec/set_kernel` endpoint with body `{ side, layer_idx, out_ch, in_ch, values: [[...]] }`
   - Preset kernel library: `KERNEL_PRESETS = { "identity": [[0,0,0],[0,1,0],[0,0,0]], "h_edge": [[-1,-1,-1],[0,0,0],[1,1,1]], "v_edge": [[-1,0,1],[-1,0,1],[-1,0,1]], "blur": [[1,1,1],[1,1,1],[1,1,1]] / 9, "sharpen": [[0,-1,0],[-1,5,-1],[0,-1,0]], "diagonal": [[1,0,0],[0,1,0],[0,0,1]] / 3 }` in `acc/models/conv_cppn.py`
   - `POST /iec/set_kernel_preset` endpoint with body `{ side, layer_idx, out_ch, in_ch, preset: str }`
3. Frontend — freeze toggle:
   - Freeze/unfreeze icon on each `ChannelPill` (snowflake icon or lock)
   - Frozen channels shown with a distinct visual (e.g. blue border, lock icon)
   - Frozen channels' kernels shown with a "frozen" overlay in the feature map strip
4. Frontend — kernel editor:
   - `KernelEditor.tsx` — clickable grid where each cell is an input field or drag-to-set
   - Preset buttons (identity, h_edge, v_edge, blur, sharpen)
   - Opens when clicking a kernel in the feature map strip
   - "Apply" button sends `POST /iec/set_kernel`, refreshes feature maps
5. Integration:
   - After setting a kernel manually, auto-freeze that channel (human intent should be preserved)
   - Undo stack captures pre-edit state (kernel values + frozen state)
   - Feature map display shows frozen channels with visual indicator

**Verification:** Open `/iec`. Train 50 steps. Click enc_0 ch0's first kernel in the feature map strip — kernel editor opens showing 3x3 grid. Click "h_edge" preset — kernel values update to horizontal edge detector. Feature map refreshes showing horizontal edge responses. Channel is auto-frozen (lock icon visible). Train 50 more steps — enc_0 ch0 weights unchanged (verify via feature map refresh), other channels trained normally. Unfreeze, train 50 more — enc_0 ch0 evolves away from edge detector. The human can sculpt individual kernels and protect them from SGD.

## Directory Structure (Anticipated)

```
acc/
  models/
    conv_cppn.py          # ConvCPPN, ConvCPPNLayer, HeterogeneousActivation,
                          # ConvCPPNGenome, ChannelDescriptor, LayerGenome,
                          # ACTIVATION_REGISTRY, mutations, transfer_weights
  iec.py                  # IECSession manager
  tasks/
    base.py               # Task, EvalOnlyTask (existing)
    reconstruction.py     # ReconstructionTask (existing, extended with loss_fn)
    classification.py     # ClassificationTask (existing)
    regression.py         # RegressionTask (existing)
    kl_divergence.py      # KLDivergenceTask (existing)
    effective_rank.py     # EffectiveRankTask (existing)
    weight_diversity.py   # WeightDiversityTask (existing)
    activation_sparsity.py # ActivationSparsityTask (existing)
    lifetime_sparsity.py  # LifetimeSparsityTask (Phase 7 — NEW)
    exclusivity.py        # WithinImageExclusivityTask (Phase 7 — NEW)
    center_of_mass.py     # CenterOfMassTask (Phase 7 — NEW)
    spatial_spread.py     # SpatialSpreadTask (Phase 7 — NEW)
    kernel_orthogonality.py # KernelOrthogonalityTask (Phase 7 — NEW)
    activation_overlap.py # ActivationOverlapDiagnostic (Phase 7 — NEW)

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
| `Trainer` | IECSession creates a `Trainer(model, tasks, device, lr, batch_size=128)`. `tasks` is a list of all configured Task subclasses (structural + reconstruction). Trainer does weighted random task sampling per step — one task, one loss, one backward. Multi-task learning emerges across steps. |
| `ReconstructionTask` | Reconstruction task. MSE/L1 + optional SSIM. Toggleable — disabled during structural-only training phases. |
| `Task` base class | All 5 new structural pressures are standard Task subclasses. They implement `check_compatible`, `_build_head` (returns None), `compute_loss`, `evaluate`. Zero Trainer changes needed. |
| `EvalOnlyTask` | Diagnostic tasks (ActivationOverlapDiagnostic) use the existing eval-only base class. `contributes_loss=False` so Trainer skips them during training. |
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
4. `IECSession` — new class (~250 lines, extended with task management in Phase 7)
5. `/iec/*` API endpoints — ~200 lines added to trainer_api.py (extended with `/iec/tasks` in Phase 7)
6. Frontend: iec page + panel + store + types — ~500 lines total (extended with task panel in Phase 7)
7. Structural pressure tasks — 5 new Task subclasses (~60-80 lines each, ~350 total)
8. Diagnostic task — 1 new EvalOnlyTask subclass (~80 lines)

**Estimated total new code: ~1860 lines.** No existing code modified except: (a) adding `self._iec` field and `/iec/*` endpoint section to `trainer_api.py`, (b) adding `self._iec is not None` to `_is_model_busy()`. **Zero changes to Trainer, Task base class, or any existing task.** The structural pressures are pure additions that plug into the existing multi-task system.

### Phase 7: Structural Pressure Tasks (M-IEC-7) --- COMPLETED

**NOTE:** This is the key theoretical contribution. Phases 1-6 give the human control over architecture and weights. Phase 7 gives the human control over *what the latent space means* — structural pressures that force UFR properties before reconstruction is even attempted. These implement the training protocol from the UFR brief (Kumar, Clune, Lehman, Stanley 2025): establish foundational regularities first, then build complex behavior on top.

**Design principle:** Each pressure is a standard `Task` subclass. The existing Trainer already does weighted random task sampling per step, tasks are independently toggleable via `task.enabled`, and `task.weight` controls loss scaling. The IEC session manages the task list — `_build_trainer()` passes all enabled tasks to the Trainer. The UI gets a task panel with toggle switches, weight sliders, and per-task parameter controls.

**Critical:** These tasks read `ModelOutput.SPATIAL` (encoder output pre-pool, `[B, C, H, W]`) and/or the pooled latent `[B, C, 3, 3]`. They do NOT need `RECONSTRUCTION`. This means they can run *without a decoder* — structural training can happen on the encoder alone, before the decoder even exists. The ConvCPPN forward already emits both `SPATIAL` and `LATENT`.

**Training protocol the human follows:**
1. Phase A — Structure: Enable structural tasks only (no reconstruction). Train until latent shows spatial correspondence and specificity.
2. Phase B — Reconstruction: Enable reconstruction task on top of structural pressures. Optionally freeze encoder weights so decoder learns to work with the structured latent.
3. Phase C — Evaluate: Run diagnostic tasks to measure UFR vs FER properties.

The human controls which phase they're in by toggling tasks on/off in the dashboard.

**Functionality:** I can toggle individual training pressures on/off from the dashboard. Each pressure has a weight slider and task-specific parameters (e.g. target_lifetime for sparsity). I can see per-task loss values in the training display. I can run structural-only training (no reconstruction), then bring reconstruction online later. The training protocol (structure → reconstruction) is a human choice, not hardcoded.

**Foundation:** 5 new Task subclasses in `acc/tasks/`, all following the existing Task protocol. `IECSession._build_trainer()` extended to manage the full task list. `POST /iec/tasks` endpoint to toggle/configure tasks. Frontend task panel with toggles. **Critically: zero changes to Trainer.** The Trainer already supports multi-task weighted sampling.

#### Structural Pressure Tasks (Category 1 — train first)

**Task: `LifetimeSparsityTask`** (`acc/tasks/lifetime_sparsity.py`)
- **What:** Across a batch, each latent spatial position should only activate significantly for a small fraction of images. Penalizes diffuse, always-on features.
- **Why:** SGD produces features that are orthogonal in weight space but overlap in activation space — each feature fires mildly for 60%+ of images. This is the core FER signature. Lifetime sparsity forces genuine specialization.
- **Reads:** `ModelOutput.SPATIAL` → pool to 3×3 → soft threshold → lifetime statistics.
- **Loss:** `mean((lifetime - target_lifetime)^2)` where `lifetime = mean_over_batch(sigmoid(sharpness * z))`.
- **Params:** `target_lifetime: float = 0.1`, `sharpness: float = 10.0`.
- **No probe head.** Operates directly on spatial activations.

**Task: `WithinImageExclusivityTask`** (`acc/tasks/exclusivity.py`)
- **What:** For each individual image, the spatial activation pattern should be concentrated — a few positions highly active, the rest quiet. Minimizes entropy of per-image spatial activation distribution.
- **Why:** Without this, every image lights up all 9 latent positions similarly. Exclusivity forces the network to commit: this "1" is described by *these* positions, this "0" by *those* positions.
- **Reads:** `ModelOutput.SPATIAL` → pool to 3×3 → flatten spatial → softmax → entropy.
- **Loss:** `mean(entropy(softmax(z_flat / temperature, dim=spatial)))`.
- **Params:** `temperature: float = 0.5`.
- **No probe head.**
- **Relationship:** Works with LifetimeSparsity. Lifetime says "don't fire for everything." Exclusivity says "when you fire, be decisive about where."

**Task: `CenterOfMassTask`** (`acc/tasks/center_of_mass.py`)
- **What:** The center of mass of the latent spatial activation must match the center of mass of the input image pixels. Gives the 3×3 latent spatial meaning.
- **Why:** Without this, the 3×3 latent has no reason to correspond to spatial location. With it, a digit in the top-left produces top-left latent activation. This is a "stepping stone" regularity.
- **Reads:** `ModelOutput.SPATIAL` → pool to 3×3, AND `batch[0]` (input image).
- **Loss:** `MSE(com_pred, com_gt)` for both x and y axes.
- **Params:** None (beyond standard weight).
- **No probe head.**

**Task: `SpatialSpreadTask`** (`acc/tasks/spatial_spread.py`)
- **What:** The second spatial moment (variance) of the latent activation should match the second moment of the input pixel distribution. Captures aspect ratio and spatial extent.
- **Why:** Position (CenterOfMass) tells you *where*. Spread tells you *how the mass is distributed*. A "1" is tall/narrow, a "0" is round. This encodes shape without explicitly defining it.
- **Reads:** `ModelOutput.SPATIAL` → pool to 3×3, AND `batch[0]` (input image).
- **Loss:** `MSE(var_pred, var_gt)` for both x and y axes.
- **Params:** None.
- **No probe head.**
- **Dependency:** Benefits from CenterOfMass (shares COM computation, but independent task).

**Task: `KernelOrthogonalityTask`** (`acc/tasks/kernel_orthogonality.py`)
- **What:** Decorrelates conv kernels across channels. Penalizes off-diagonal elements in the Gram matrix of normalized flattened kernels.
- **Why:** When channels > 1, ensures different channels learn different features rather than redundant ones. SGD does this somewhat naturally, but structural pressures can degrade it.
- **Reads:** Model parameters (encoder conv layer weights). Does NOT read model_output.
- **Loss:** `||W_norm @ W_norm.T - I||^2` where `W_norm` is row-normalized flattened kernels.
- **Params:** `layer_name: str` (which conv layer to target).
- **No probe head.** Accesses weights directly.
- **No-op when encoder has 1 channel.**

#### Diagnostic Tasks (Category 3 — measure, don't train)

These are `EvalOnlyTask` subclasses (existing base class) that measure UFR/FER properties without contributing gradient.

**Task: `ActivationOverlapDiagnostic`** (`acc/tasks/activation_overlap.py`)
- **What:** For each latent position, measures (a) lifetime (fraction of dataset that activates it), (b) pairwise cosine similarity of activation patterns across dataset, (c) co-activation fraction. Reports whether the latent exhibits UFR or FER signatures.
- **UFR target:** Low cosine similarity AND low co-activation. **FER signature:** Low cosine similarity BUT high co-activation.
- **Eval-only.** No gradient.

Tasks:
1. `acc/tasks/lifetime_sparsity.py` — `LifetimeSparsityTask(Task)`:
   - `__init__(name, dataset, target_lifetime=0.1, sharpness=10.0, **kwargs)`
   - `check_compatible`: model must emit SPATIAL
   - `_build_head`: returns None
   - `compute_loss`: pool SPATIAL to 3×3, soft threshold, compute lifetime, MSE vs target
   - `evaluate`: return actual lifetime statistics
2. `acc/tasks/exclusivity.py` — `WithinImageExclusivityTask(Task)`:
   - `__init__(name, dataset, temperature=0.5, **kwargs)`
   - `compute_loss`: pool SPATIAL to 3×3, flatten, softmax, entropy
   - `evaluate`: return mean entropy
3. `acc/tasks/center_of_mass.py` — `CenterOfMassTask(Task)`:
   - `compute_loss`: COM of SPATIAL (pooled to 3×3) vs COM of input image
   - `evaluate`: return mean COM error
4. `acc/tasks/spatial_spread.py` — `SpatialSpreadTask(Task)`:
   - `compute_loss`: second moment of SPATIAL vs second moment of input
   - `evaluate`: return mean spread error
5. `acc/tasks/kernel_orthogonality.py` — `KernelOrthogonalityTask(Task)`:
   - `compute_loss`: Gram matrix penalty on encoder kernels
   - `evaluate`: return off-diagonal Gram magnitude
6. `acc/tasks/activation_overlap.py` — `ActivationOverlapDiagnostic(EvalOnlyTask)`:
   - `evaluate`: lifetime, cosine sim, co-activation metrics
7. `acc/iec.py` — extend `_build_trainer()`:
   - Maintain a dict of task configs: `self.task_configs: dict[str, dict]`
   - Default: only `recon` task enabled
   - `set_task_config(task_name, enabled, weight, params)` method
   - `_build_trainer()` builds all configured tasks, passes to Trainer
8. `acc/trainer_api.py` — `POST /iec/tasks` endpoint:
   - Body: `{ "task_name": str, "enabled": bool, "weight": float, "params": dict }`
   - Returns updated task config list
   - `GET /iec/tasks` returns current task configs with per-task last loss
9. Frontend — task control panel in IecPanel.tsx:
   - Collapsible "Training Tasks" section
   - Per-task row: toggle switch, weight slider, task-specific param inputs
   - Per-task loss badge (from step callback)
   - Visual grouping: "Structural" vs "Reconstruction" vs "Diagnostic"

**Verification:** Open `/iec`. Enable LifetimeSparsity (weight=1.0) and CenterOfMass (weight=1.0), disable Reconstruction. Train 200 steps — structural losses decrease. Feature maps show spatially concentrated activations (not diffuse blobs). Enable Reconstruction — loss decreases, reconstructions improve. Disable structural tasks — run 200 more steps — observe whether latent structure degrades. Compare checkpoints: structural-first vs reconstruction-only. The human can see the difference.

## Phase Execution Order

Phase 1 → 2 → 3 → 4 → 5 → 6 → 7

**Every phase is end-to-end.** Phase 1 delivers a working `/iec` page in the browser with the model, API, and UI wired together (stubs for training/mutation). Each subsequent phase fills in real functionality where stubs were, and each is verified in the browser.

- After Phase 1: You can see the model and its reconstructions in the browser. Integration proven.
- After Phase 2: You can train the model and watch it learn. The core interactive loop works.
- After Phase 3: You can mutate the architecture and undo. The human IS the search algorithm.
- After Phase 4: You can save/load checkpoints and inspect feature maps. Persistence and diagnosis.
- After Phase 5: You can see the topology as a graph with feature maps, gradients, and kernels inline. Full diagnosis.
- After Phase 6: You can freeze channels and manually edit kernels. The human controls both topology AND weights.
- After Phase 7: You can apply structural pressures (sparsity, exclusivity, spatial correspondence) to the latent, train structure-first then reconstruction-second, and measure UFR vs FER properties. The human controls the training protocol.

## Full Outcome

After all phases: The human opens `/iec`, sees a minimal autoencoder. They first enable structural pressures — lifetime sparsity, exclusivity, center of mass — and train the encoder to develop spatially meaningful, specialized features *before asking it to reconstruct anything*. They inspect the latent: each 3×3 position activates only for digits in that spatial region. Features are concentrated, not diffuse. Then they bring reconstruction online, with the encoder optionally frozen. The decoder learns to reconstruct from a structured latent. They grow the network — adding channels with purpose, each one observed to specialize for a different class or property. They freeze good channels, sculpt kernels for edge detection, and let SGD optimize around their design. The internal representation is organized because (a) structural pressures forced specialization, (b) the human selected architecture based on observed quality, and (c) foundational regularities were established before complex behavior was built on top — exactly the Picbreeder principle.
