"""StepInspector — enum-keyed tensor capture for batch-by-batch training inspection.

The inspector sits between the recipe and the trainer. During each training
step, the recipe's training_metrics_fn calls inspector.capture(key, tensor)
for every tensor we want to inspect. After the step, the API calls
inspector.collect() to serialize the captured data, then inspector.commit()
to archive it to history.

The frontend and backend share a vocabulary via StepTensorKey. Adding a new
tensor to inspect:
  1. Add enum value to StepTensorKey
  2. Add inspector.capture(KEY, tensor) in the inspector's metrics_fn
  3. Add rendering in the frontend for that key

Usage:
    inspector = StepInspector()

    def metrics_fn(step):
        inspector.capture(StepTensorKey.BATCH_IMAGES, batch_images)
        inspector.capture(StepTensorKey.LOSS, loss_tensor)
        return None

    trainer.train(steps=1, training_metrics_fn=metrics_fn)
    data = inspector.collect()
    inspector.commit()
"""

import base64
import io
from enum import Enum
from typing import Any, Optional

import torch
import torchvision.utils as vutils
from PIL import Image


class StepTensorKey(str, Enum):
    """Every capturable tensor has an entry here.

    The string values are used as JSON keys in API responses and must match
    the TypeScript StepTensorKey enum in inspect-types.ts.
    """

    # -- Batch data --
    BATCH_IMAGES = "batch_images"  # [B, C, H, W] input images
    BATCH_LABELS = "batch_labels"  # [B] labels (if available)

    # -- Loss --
    LOSS = "loss"  # scalar

    # -- Encoder weights --
    ENCODER_WEIGHTS = "encoder_weights"  # [D, in_features] current weight matrix

    # -- BCL competition --
    RANK_SCORE = "rank_score"  # [B, D]
    STRENGTH = "strength"  # [B, D]
    FEATURE_NOVELTY = "feature_novelty"  # [D]
    IMAGE_COVERAGE = "image_coverage"  # [B]
    WIN_RATE = "win_rate"  # [D]

    # -- BCL neighborhoods --
    NEIGHBORS = "neighbors"  # [D, k]
    LOCAL_COVERAGE = "local_coverage"  # [B, D]
    LOCAL_NOVELTY = "local_novelty"  # [B, D]
    IN_NEIGHBORHOOD = "in_nbr"  # [B, D]

    # -- BCL blending weights --
    GRADIENT_WEIGHT = "gradient_weight"  # [D]
    CONTENDER_WEIGHT = "contender_weight"  # [D] (legacy, kept for compat)
    ATTRACTION_WEIGHT = "attraction_weight"  # [D] (legacy, kept for compat)
    SOM_WEIGHT_D = "som_weight_d"  # [D] per-feature SOM weight (1 - effective_win)

    # -- BCL forces --
    GRAD_MASK = "grad_mask"  # [B, D]
    LOCAL_TARGET = "local_target"  # [D, in_features]
    GLOBAL_TARGET = "global_target"  # [D, in_features] (legacy)
    SOM_TARGETS = "som_targets"  # [D, in_features]
    SOM_DELTA = "som_delta"  # [D, in_features]

    # -- BCL diagnostics --
    LOCAL_PULL_SUM = (
        "local_pull_sum"  # [D] raw pull signal per feature (pre-normalization)
    )

    # -- Rescue diagnostics --
    AFFINITY = "affinity"  # [B, D] cosine sim of each image to each feature
    IMAGE_NEED = "image_need"  # [B] 1/(image_coverage+1), how underserved
    WEIGHTED_AFFINITY = "weighted_affinity"  # [B, D] affinity * image_need
    RESCUE_PULL = "rescue_pull"  # [B, D] sparse normalized pull weights

    # -- Gradient --
    GRAD_MASKED = "grad_masked"  # [D, in_features]

    # -- Post-step --
    ENCODER_WEIGHTS_POST = "encoder_weights_post"  # [D, in_features]


# Keys whose full data is retained in every history entry (scalars + small vectors).
# Everything else is only in the ring buffer of full snapshots.
_HISTORY_RETAIN: set[StepTensorKey] = {
    StepTensorKey.LOSS,
    StepTensorKey.WIN_RATE,
    StepTensorKey.FEATURE_NOVELTY,
    StepTensorKey.GRADIENT_WEIGHT,
    StepTensorKey.CONTENDER_WEIGHT,
    StepTensorKey.ATTRACTION_WEIGHT,
    StepTensorKey.SOM_WEIGHT_D,
    StepTensorKey.IMAGE_COVERAGE,
    StepTensorKey.LOCAL_PULL_SUM,
}

# Keys that contain image data and should be serialized as base64 PNG grids.
_IMAGE_KEYS: set[StepTensorKey] = {
    StepTensorKey.BATCH_IMAGES,
}


def _serialize_tensor(key: StepTensorKey, tensor: torch.Tensor) -> Any:
    """Convert a tensor to JSON-serializable format based on its key and shape.

    - Image keys [B, C, H, W] -> base64 PNG grid
    - Scalars (0-dim or 1-element) -> float
    - 1D vectors [N] -> list of floats
    - 2D matrices [M, N] -> list of list of floats
    """
    if key in _IMAGE_KEYS:
        return _tensor_to_image_grid(tensor)

    if tensor.ndim == 0:
        return tensor.item()
    if tensor.ndim == 1 and tensor.numel() == 1:
        return tensor.item()
    if tensor.ndim == 1:
        return tensor.tolist()
    if tensor.ndim == 2:
        return tensor.tolist()

    # Fallback: flatten
    return tensor.flatten().tolist()


def _tensor_to_image_grid(tensor: torch.Tensor, nrow: int = 16) -> str:
    """Convert a batch of images [B, C, H, W] to a base64 PNG grid."""
    # Clamp to [0, 1] for display
    tensor = tensor.clamp(0, 1)
    grid = vutils.make_grid(tensor, nrow=nrow, padding=1, pad_value=0.5)
    # grid is [C, H, W] with C=3 or C=1
    if grid.shape[0] == 1:
        grid = grid.repeat(3, 1, 1)
    # Convert to PIL
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class StepInspector:
    """Captures tensors during training steps and accumulates history.

    Every step's captures are archived via commit(). Small tensors (scalars,
    [D]-vectors from _HISTORY_RETAIN) are stored in every history entry for
    timeline charts. Full tensor data is stored in a ring buffer of the last
    N steps for scrolling back to view batch images, matrices, etc.
    """

    def __init__(self, max_full_snapshots: int = 50):
        self._captures: dict[StepTensorKey, torch.Tensor] = {}
        self._step: int = 0
        self._history: list[dict] = []
        self._full_snapshots: dict[int, dict[str, Any]] = {}
        self._max_full_snapshots = max_full_snapshots

    def capture(self, key: StepTensorKey, tensor: torch.Tensor) -> None:
        """Store a tensor for this step. Clones to avoid mutation."""
        self._captures[key] = tensor.detach().clone().cpu()

    def capture_scalar(self, key: StepTensorKey, value: float) -> None:
        """Store a scalar value."""
        self._captures[key] = torch.tensor(value)

    def collect(self) -> dict[str, Any]:
        """Serialize all captured tensors for the current step.

        Returns dict keyed by StepTensorKey.value (string).
        """
        result: dict[str, Any] = {}
        for key, tensor in self._captures.items():
            result[key.value] = _serialize_tensor(key, tensor)
        result["_step"] = self._step
        result["_keys"] = [k.value for k in self._captures.keys()]
        return result

    def commit(self) -> None:
        """Archive current step to history, then clear for next step.

        Small tensors are stored in every history entry.
        Full data is stored in a ring buffer of the last N steps.
        """
        # Build history summary (small tensors only)
        summary: dict[str, Any] = {
            "step": self._step,
            "keys": [k.value for k in self._captures.keys()],
        }
        for key, tensor in self._captures.items():
            if key in _HISTORY_RETAIN:
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

    def get_step(self, step: int) -> Optional[dict[str, Any]]:
        """Return full tensor data for a specific step, if still in ring buffer."""
        return self._full_snapshots.get(step)

    @property
    def step(self) -> int:
        return self._step

    def clear(self) -> None:
        """Reset all state. Called on teardown."""
        self._captures.clear()
        self._history.clear()
        self._full_snapshots.clear()
        self._step = 0
