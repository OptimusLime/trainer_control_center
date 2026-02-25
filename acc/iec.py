"""IECSession — Interactive Evolutionary CPPN session manager.

Manages the lifecycle of a ConvCPPN model being interactively evolved:
setup, training steps, mutations, undo, checkpoints, and teardown.

Follows the same session pattern as StepInspector: one active session at a
time, mutually exclusive with other model-using features, action-driven.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import torch
import torch.nn as nn

from acc.models.conv_cppn import (
    ConvCPPN,
    ConvCPPNGenome,
    default_genome,
    ACTIVATION_NAMES,
    add_channel,
    remove_channel,
    change_activation,
    add_connection,
    remove_connection,
    add_encoder_layer,
    remove_encoder_layer,
    add_decoder_layer,
    remove_decoder_layer,
    toggle_coords,
    toggle_freeze,
    transfer_weights,
    KERNEL_PRESETS,
)
from acc.model_output import ModelOutput
from acc.trainer import Trainer
from acc.tasks.base import Task
from acc.tasks.reconstruction import ReconstructionTask
from acc.tasks.lifetime_sparsity import LifetimeSparsityTask
from acc.tasks.exclusivity import WithinImageExclusivityTask
from acc.tasks.center_of_mass import CenterOfMassTask
from acc.tasks.spatial_spread import SpatialSpreadTask
from acc.tasks.kernel_orthogonality import KernelOrthogonalityTask
from acc.tasks.activation_overlap import ActivationOverlapDiagnostic
from acc.dataset import AccDataset
from acc.checkpoints import CheckpointStore


# Registry of all available IEC tasks.
# Each entry: (display_name, category, builder_fn, default_params)
# builder_fn(name, dataset, **params) -> Task
_TASK_REGISTRY: dict[str, dict] = {
    "recon": {
        "display": "Reconstruction",
        "category": "reconstruction",
        "builder": lambda name, ds, **p: ReconstructionTask(
            name,
            ds,
            loss_fn=p.get("loss_fn", "mse"),
            ssim_weight=p.get("ssim_weight", 0.0),
        ),
        "default_params": {"loss_fn": "mse", "ssim_weight": 0.0},
    },
    "lifetime_sparsity": {
        "display": "Lifetime Sparsity",
        "category": "structural",
        "builder": lambda name, ds, **p: LifetimeSparsityTask(
            name,
            ds,
            target_lifetime=p.get("target_lifetime", 0.1),
            sharpness=p.get("sharpness", 10.0),
        ),
        "default_params": {"target_lifetime": 0.1, "sharpness": 10.0},
    },
    "exclusivity": {
        "display": "Within-Image Exclusivity",
        "category": "structural",
        "builder": lambda name, ds, **p: WithinImageExclusivityTask(
            name,
            ds,
            temperature=p.get("temperature", 0.5),
        ),
        "default_params": {"temperature": 0.5},
    },
    "center_of_mass": {
        "display": "Center of Mass",
        "category": "structural",
        "builder": lambda name, ds, **p: CenterOfMassTask(name, ds),
        "default_params": {},
    },
    "spatial_spread": {
        "display": "Spatial Spread",
        "category": "structural",
        "builder": lambda name, ds, **p: SpatialSpreadTask(name, ds),
        "default_params": {},
    },
    "kernel_orthogonality": {
        "display": "Kernel Orthogonality",
        "category": "structural",
        "builder": lambda name, ds, **p: KernelOrthogonalityTask(name, ds),
        "default_params": {},
    },
    "activation_overlap": {
        "display": "Activation Overlap (diagnostic)",
        "category": "diagnostic",
        "builder": lambda name, ds, **p: ActivationOverlapDiagnostic(
            name,
            ds,
            threshold=p.get("threshold", 0.1),
            n_eval_batches=p.get("n_eval_batches", 20),
        ),
        "default_params": {"threshold": 0.1, "n_eval_batches": 20},
    },
}


def _tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a [C, H, W] tensor to base64-encoded PNG."""
    import PIL.Image
    import numpy as np

    img = tensor.cpu().detach()
    if img.shape[0] == 1:
        img = img.squeeze(0)  # grayscale → (H, W)
    elif img.shape[0] == 3:
        img = img.permute(1, 2, 0)  # RGB → (H, W, 3)
    img_np = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = PIL.Image.fromarray(img_np)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class IECSession:
    """Manages an interactive ConvCPPN evolution session.

    Phase 1 (M-IEC-1): setup, get_state, get_reconstructions, teardown.
    Phase 2 (M-IEC-2): step (real training).
    Phase 3 (M-IEC-3): mutate, undo, transfer_weights.
    Phase 4 (M-IEC-4): save_checkpoint, load_checkpoint, get_feature_maps.
    """

    def __init__(self):
        self.model: Optional[ConvCPPN] = None
        self.genome: Optional[ConvCPPNGenome] = None
        self.trainer: Optional[Trainer] = None
        self.dataset: Optional[AccDataset] = None
        self.device: Optional[torch.device] = None
        self.step_count: int = 0
        self.last_loss: Optional[float] = None
        self.undo_stack: list[tuple[dict, dict]] = []  # (genome_dict, state_dict)
        self._lr: float = 1e-3
        self.ssim_weight: float = 0.0  # pixel_loss + ssim_weight * (1 - SSIM)
        self.loss_fn: str = "mse"  # 'mse' or 'l1'

        # Multi-task config: {task_name: {enabled, weight, params}}
        # Default: only reconstruction enabled.
        self.task_configs: dict[str, dict] = IECSession._default_task_configs()

    def setup(
        self,
        device: torch.device,
        mnist_dataset: AccDataset,
        genome_dict: Optional[dict] = None,
    ) -> dict:
        """Build model from genome (or default), create trainer.

        Returns the initial state dict.
        """
        self.device = device
        self.dataset = mnist_dataset

        # Build genome and model
        if genome_dict is not None:
            self.genome = ConvCPPNGenome.from_dict(genome_dict)
        else:
            self.genome = default_genome()

        self.model = ConvCPPN.from_genome(self.genome)
        self.model = self.model.to(device)

        # Create reconstruction task and trainer
        self._build_trainer()

        self.step_count = 0
        self.last_loss = None
        self.undo_stack = []

        return self.get_state()

    @staticmethod
    def _default_task_configs() -> dict[str, dict]:
        """Default task configuration: only reconstruction enabled."""
        configs: dict[str, dict] = {}
        for name, reg in _TASK_REGISTRY.items():
            configs[name] = {
                "enabled": name == "recon",  # only recon on by default
                "weight": 1.0,
                "params": dict(reg["default_params"]),
            }
        return configs

    def set_task_config(
        self,
        task_name: str,
        enabled: Optional[bool] = None,
        weight: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> list[dict]:
        """Update config for a single task and rebuild the trainer.

        Returns the updated task configs dict.
        """
        if task_name not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(_TASK_REGISTRY.keys())}"
            )
        cfg = self.task_configs.get(task_name)
        if cfg is None:
            reg = _TASK_REGISTRY[task_name]
            cfg = {
                "enabled": False,
                "weight": 1.0,
                "params": dict(reg["default_params"]),
            }
            self.task_configs[task_name] = cfg

        if enabled is not None:
            cfg["enabled"] = enabled
        if weight is not None:
            cfg["weight"] = weight
        if params is not None:
            cfg["params"].update(params)

        # Sync recon params from session-level ssim_weight / loss_fn
        if task_name == "recon":
            if "ssim_weight" in cfg["params"]:
                self.ssim_weight = cfg["params"]["ssim_weight"]
            if "loss_fn" in cfg["params"]:
                self.loss_fn = cfg["params"]["loss_fn"]

        if self.model is not None:
            self._build_trainer()

        return self.get_task_configs()

    def get_task_configs(self) -> list[dict]:
        """Return current task configs for the API.

        Each entry: {name, display, category, enabled, weight, params, default_params}
        """
        result = []
        for name, reg in _TASK_REGISTRY.items():
            cfg = self.task_configs.get(
                name,
                {
                    "enabled": False,
                    "weight": 1.0,
                    "params": dict(reg["default_params"]),
                },
            )
            result.append(
                {
                    "name": name,
                    "display": reg["display"],
                    "category": reg["category"],
                    "enabled": cfg["enabled"],
                    "weight": cfg["weight"],
                    "params": cfg["params"],
                    "default_params": reg["default_params"],
                }
            )
        return result

    def _build_trainer(self):
        """(Re)build the trainer with current model, dataset, and task configs.

        Builds all tasks from task_configs. Enabled tasks get task.enabled=True
        and their weight set. Disabled tasks are not included (Trainer only
        sees enabled tasks). Eval-only tasks are always included so they
        run during evaluate_all().
        """
        tasks: list[Task] = []

        for name, reg in _TASK_REGISTRY.items():
            cfg = self.task_configs.get(
                name,
                {
                    "enabled": False,
                    "weight": 1.0,
                    "params": dict(reg["default_params"]),
                },
            )

            # For recon task, inject session-level ssim_weight and loss_fn
            build_params = dict(cfg["params"])
            if name == "recon":
                build_params["ssim_weight"] = self.ssim_weight
                build_params["loss_fn"] = self.loss_fn

            try:
                task = reg["builder"](name, self.dataset, **build_params)
            except Exception as e:
                print(f"[IEC] Warning: failed to build task '{name}': {e}")
                continue

            task.enabled = cfg["enabled"]
            task.weight = cfg["weight"]

            try:
                task.attach(self.model)
            except Exception as e:
                print(f"[IEC] Warning: task '{name}' incompatible: {e}")
                continue

            tasks.append(task)

        self.trainer = Trainer(
            self.model,
            tasks,
            self.device,
            lr=self._lr,
            batch_size=128,
        )

    def set_ssim_weight(self, weight: float):
        """Update SSIM loss weight and rebuild trainer."""
        self.ssim_weight = weight
        # Keep recon task config in sync
        if "recon" in self.task_configs:
            self.task_configs["recon"]["params"]["ssim_weight"] = weight
        if self.model is not None:
            self._build_trainer()

    def get_state(self) -> dict:
        """Current session state for the API."""
        return {
            "active": self.model is not None,
            "genome": self.genome.to_dict() if self.genome else None,
            "step": self.step_count,
            "last_loss": self.last_loss,
            "latent_dim": self.model.latent_dim if self.model else 0,
            "architecture": self.model.architecture_summary() if self.model else "",
            "undo_depth": len(self.undo_stack),
            "activation_names": ACTIVATION_NAMES,
            "resolutions": self.model.resolution_info() if self.model else None,
            "ssim_weight": self.ssim_weight,
            "loss_fn": self.loss_fn,
            "tasks": self.get_task_configs(),
        }

    def get_reconstructions(self, n: int = 8, normalize: bool = False) -> dict:
        """Run inference, return base64 input/output image pairs.

        Args:
            n: Number of image pairs.
            normalize: If True, per-image min-max stretch to [0,1] so faint
                       patterns become visible. If False, raw clamp to [0,1].
        """
        if self.model is None or self.dataset is None:
            return {"inputs": [], "outputs": []}

        self.model.eval()
        with torch.no_grad():
            images = self.dataset.sample(n).to(self.device)
            model_out = self.model(images)
            recon = model_out[ModelOutput.RECONSTRUCTION]

            if normalize:
                # Per-image min-max normalization
                for i in range(recon.shape[0]):
                    rmin = recon[i].min()
                    rmax = recon[i].max()
                    if rmax - rmin > 1e-8:
                        recon[i] = (recon[i] - rmin) / (rmax - rmin)
                    else:
                        recon[i] = recon[i] * 0 + 0.5
            else:
                recon = recon.clamp(0, 1)

            inputs = [_tensor_to_base64(images[i]) for i in range(n)]
            outputs = [_tensor_to_base64(recon[i]) for i in range(n)]

        self.model.train()
        return {"inputs": inputs, "outputs": outputs}

    def step(self, n: int = 10, lr: Optional[float] = None) -> dict:
        """Train n steps. Returns loss history + step count.

        If lr is provided and differs from current, rebuilds the Trainer.
        """
        if self.model is None or self.trainer is None:
            raise RuntimeError("No active IEC session")

        # Rebuild trainer if lr changed
        if lr is not None and abs(lr - self._lr) > 1e-10:
            self._lr = lr
            self._build_trainer()

        # Collect losses via on_step callback
        losses: list[float] = []

        def on_step(step_info: dict):
            losses.append(step_info["task_loss"])

        self.model.train()
        self.trainer.train(steps=n, on_step=on_step)

        # Zero masked grads (belt-and-suspenders for connection masks)
        self.model.zero_masked_grads()

        self.step_count += n
        self.last_loss = losses[-1] if losses else self.last_loss

        return {
            "losses": losses,
            "step": self.step_count,
            "last_loss": self.last_loss,
        }

    # -- Mutations (M-IEC-3) --

    _UNDO_STACK_MAX = 20

    def mutate(self, mutation_type: str, **kwargs) -> dict:
        """Apply a genome mutation, rebuild model, transfer weights.

        Pushes current (genome, state_dict) to undo stack before mutating.

        Supported mutation types:
            add_channel:       side, layer_idx, activation
            remove_channel:    side, layer_idx, channel_idx
            change_activation: side, layer_idx, channel_idx, new_activation
            add_connection:    side, layer_idx, out_ch, in_ch
            remove_connection: side, layer_idx, out_ch, in_ch

        Returns:
            Full state dict (same shape as get_state()).
        """
        if self.model is None or self.genome is None:
            raise RuntimeError("No active IEC session")

        # Push current state to undo stack
        self.undo_stack.append(
            (
                self.genome.to_dict(),
                {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            )
        )
        if len(self.undo_stack) > self._UNDO_STACK_MAX:
            self.undo_stack.pop(0)

        # Apply mutation
        old_genome = self.genome
        old_model = self.model

        if mutation_type == "add_channel":
            self.genome = add_channel(
                old_genome,
                side=kwargs["side"],
                layer_idx=int(kwargs["layer_idx"]),
                activation=kwargs["activation"],
            )
        elif mutation_type == "remove_channel":
            self.genome = remove_channel(
                old_genome,
                side=kwargs["side"],
                layer_idx=int(kwargs["layer_idx"]),
                channel_idx=int(kwargs["channel_idx"]),
            )
        elif mutation_type == "change_activation":
            self.genome = change_activation(
                old_genome,
                side=kwargs["side"],
                layer_idx=int(kwargs["layer_idx"]),
                channel_idx=int(kwargs["channel_idx"]),
                new_activation=kwargs["new_activation"],
            )
        elif mutation_type == "add_connection":
            self.genome = add_connection(
                old_genome,
                side=kwargs["side"],
                layer_idx=int(kwargs["layer_idx"]),
                out_ch=int(kwargs["out_ch"]),
                in_ch=int(kwargs["in_ch"]),
            )
        elif mutation_type == "remove_connection":
            self.genome = remove_connection(
                old_genome,
                side=kwargs["side"],
                layer_idx=int(kwargs["layer_idx"]),
                out_ch=int(kwargs["out_ch"]),
                in_ch=int(kwargs["in_ch"]),
            )
        elif mutation_type == "add_layer":
            side = kwargs["side"]
            position = int(kwargs.get("position", -1))
            activation = kwargs.get(
                "activation", "identity" if side == "encoder" else "relu"
            )
            channels = int(kwargs.get("channels", 1))
            if side == "encoder":
                stride = int(kwargs.get("stride", 2))
                self.genome = add_encoder_layer(
                    old_genome,
                    position=position,
                    activation=activation,
                    channels=channels,
                    stride=stride,
                )
            else:
                self.genome = add_decoder_layer(
                    old_genome,
                    position=position,
                    activation=activation,
                    channels=channels,
                )
        elif mutation_type == "remove_layer":
            side = kwargs["side"]
            layer_idx = int(kwargs["layer_idx"])
            if side == "encoder":
                self.genome = remove_encoder_layer(old_genome, layer_idx)
            else:
                self.genome = remove_decoder_layer(old_genome, layer_idx)
        elif mutation_type == "toggle_coords":
            self.genome = toggle_coords(
                old_genome,
                side=kwargs["side"],
                layer_idx=int(kwargs["layer_idx"]),
            )
        elif mutation_type == "toggle_freeze":
            self.genome = toggle_freeze(
                old_genome,
                side=kwargs["side"],
                layer_idx=int(kwargs["layer_idx"]),
                channel_idx=int(kwargs["channel_idx"]),
            )
        else:
            # Roll back — invalid mutation type
            self.undo_stack.pop()
            raise ValueError(
                f"Unknown mutation type '{mutation_type}'. "
                f"Available: add_channel, remove_channel, change_activation, "
                f"add_connection, remove_connection, add_layer, remove_layer, "
                f"toggle_coords, toggle_freeze"
            )

        # Build new model and transfer weights
        self.model = ConvCPPN.from_genome(self.genome).to(self.device)
        transfer_weights(old_model, self.model, old_genome, self.genome)

        # Rebuild trainer with new model
        self._build_trainer()

        return self.get_state()

    def undo(self) -> dict:
        """Undo the last mutation, restoring genome + weights.

        Returns:
            Full state dict after undo.
        """
        if not self.undo_stack:
            raise ValueError("Nothing to undo")

        genome_dict, state_dict = self.undo_stack.pop()

        self.genome = ConvCPPNGenome.from_dict(genome_dict)
        self.model = ConvCPPN.from_genome(self.genome).to(self.device)
        self.model.load_state_dict(state_dict)

        # Rebuild trainer with restored model
        self._build_trainer()

        return self.get_state()

    # -- Kernel Editing (M-IEC-6) --

    def set_kernel(
        self,
        side: str,
        layer_idx: int,
        out_ch: int,
        in_ch: int,
        values: list[list[float]],
        auto_freeze: bool = True,
    ) -> dict:
        """Set a specific kernel's weights directly.

        This is a weight-level operation (not a genome mutation), but we
        push to the undo stack so the user can revert.

        Args:
            side: 'encoder' or 'decoder'.
            layer_idx: Which layer (0-indexed).
            out_ch: Output channel index.
            in_ch: Input channel index (relative to connection mask).
            values: 2D list of kernel weights (e.g. 3x3).
            auto_freeze: If True, freeze this channel after editing.

        Returns:
            Full state dict.
        """
        if self.model is None or self.genome is None:
            raise RuntimeError("No active IEC session")

        # Push current state to undo stack (captures weights + genome)
        self.undo_stack.append(
            (
                self.genome.to_dict(),
                {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            )
        )
        if len(self.undo_stack) > self._UNDO_STACK_MAX:
            self.undo_stack.pop(0)

        # Validate indices
        layers = (
            self.genome.encoder_layers
            if side == "encoder"
            else self.genome.decoder_layers
        )
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range for {side}")

        layer_genome = layers[layer_idx]
        if out_ch < 0 or out_ch >= len(layer_genome.channel_descriptors):
            raise ValueError(
                f"out_ch {out_ch} out of range "
                f"(layer has {len(layer_genome.channel_descriptors)} channels)"
            )

        # Get the model layer
        model_layers = (
            self.model.encoder_layers
            if side == "encoder"
            else self.model.decoder_layers
        )
        model_layer = model_layers[layer_idx]

        # Validate in_ch against actual weight shape
        weight = model_layer.conv.weight  # [C_out, C_in, K, K]
        if in_ch < 0 or in_ch >= weight.shape[1]:
            raise ValueError(
                f"in_ch {in_ch} out of range (weight has {weight.shape[1]} input channels)"
            )

        # Validate kernel dimensions
        import torch

        kernel_tensor = torch.tensor(values, dtype=weight.dtype, device=weight.device)
        if kernel_tensor.shape != weight.shape[2:]:
            raise ValueError(
                f"Kernel shape {list(kernel_tensor.shape)} doesn't match "
                f"weight kernel shape {list(weight.shape[2:])}"
            )

        # Set the kernel weights
        with torch.no_grad():
            weight[out_ch, in_ch] = kernel_tensor

        # Auto-freeze the channel so SGD doesn't overwrite the manual edit
        if auto_freeze:
            layer_genome.channel_descriptors[out_ch].frozen = True
            # Update frozen_channels list in the model layer
            if out_ch not in model_layer.frozen_channels:
                model_layer.frozen_channels.append(out_ch)

        return self.get_state()

    def get_kernel_presets(self) -> dict[str, list[list[float]]]:
        """Return available kernel presets."""
        return KERNEL_PRESETS

    # -- Checkpoints (M-IEC-4) --

    def save_checkpoint(self, tag: str, checkpoint_store: CheckpointStore) -> dict:
        """Save current genome + weights as a checkpoint.

        Uses CheckpointStore.save() — the genome is stored in model_config
        via model.config() automatically. step_count and last_loss go into metrics.

        Returns:
            Checkpoint metadata dict.
        """
        if self.model is None or self.trainer is None:
            raise RuntimeError("No active IEC session")

        metrics = {
            "iec_step": self.step_count,
            "iec_last_loss": self.last_loss,
        }
        cp = checkpoint_store.save(
            self.model,
            self.trainer,
            tag=tag,
            recipe_name="iec",
            description=f"IEC checkpoint at step {self.step_count}",
            metrics=metrics,
        )
        return cp.to_dict()

    def load_checkpoint(
        self, checkpoint_id: str, checkpoint_store: CheckpointStore
    ) -> dict:
        """Load a checkpoint: rebuild model from stored genome, restore weights.

        Flow:
        1. load_metadata() to get genome from model_config
        2. Rebuild ConvCPPN from genome
        3. Rebuild trainer
        4. load_model_only() to restore just the weights

        Returns:
            Full session state dict.
        """
        if self.device is None or self.dataset is None:
            raise RuntimeError("No active IEC session")

        # 1. Get metadata (genome lives in model_config)
        cp = checkpoint_store.load_metadata(checkpoint_id)
        genome_dict = cp.model_config.get("genome")
        if genome_dict is None:
            raise ValueError(
                f"Checkpoint {checkpoint_id} has no genome in model_config — "
                f"not an IEC checkpoint?"
            )

        # 2. Rebuild model from genome
        self.genome = ConvCPPNGenome.from_dict(genome_dict)
        self.model = ConvCPPN.from_genome(self.genome).to(self.device)

        # 3. Rebuild trainer (fresh optimizer — appropriate for architecture changes)
        self._build_trainer()

        # 4. Load just the model weights
        checkpoint_store.load_model_only(checkpoint_id, self.model, device=self.device)

        # Restore step count and loss from metrics
        self.step_count = cp.metrics.get("iec_step", 0)
        self.last_loss = cp.metrics.get("iec_last_loss", None)

        # Clear undo stack — loading a checkpoint is a fresh starting point
        self.undo_stack = []

        return self.get_state()

    def list_checkpoints(self, checkpoint_store: CheckpointStore) -> list[dict]:
        """List IEC checkpoints (filtered by recipe_name='iec').

        Returns:
            List of checkpoint metadata dicts.
        """
        return [
            cp.to_dict() for cp in checkpoint_store.tree() if cp.recipe_name == "iec"
        ]

    # -- Feature Maps (M-IEC-4) --

    def get_feature_maps(self, n: int = 1) -> dict:
        """Extract per-channel feature maps AND gradients from all layers.

        Runs a forward pass, computes L1 reconstruction loss, then backward
        pass to get gradients on each layer's output activations. The gradient
        maps show what SGD wants to change at each spatial location per channel.

        Args:
            n: Number of images to run (returns maps for the first one).

        Returns:
            {
                "input_image": "base64_png",
                "loss": float,
                "encoder": [ { "name", "resolution", "channels": [
                    { "activation", "data": [[...]], "grad": [[...]] }, ...
                ] } ],
                "latent": { same structure } | null,
                "decoder": [ same structure ],
            }
        """
        if self.model is None or self.dataset is None:
            raise RuntimeError("No active IEC session")

        # We need gradients on intermediate activations, so we use hooks that
        # retain_grad() on outputs instead of detaching them.
        activations: dict[str, torch.Tensor] = {}
        hooks = []

        def _make_hook(name: str):
            def hook_fn(mod, inp, out):
                out.retain_grad()
                activations[name] = out

            return hook_fn

        for i, layer in enumerate(self.model.encoder_layers):
            hooks.append(layer.register_forward_hook(_make_hook(f"enc_{i}")))

        for i, layer in enumerate(self.model.decoder_layers):
            hooks.append(layer.register_forward_hook(_make_hook(f"dec_{i}")))

        hooks.append(self.model.pool.register_forward_hook(_make_hook("latent")))

        # Forward pass WITH gradients
        self.model.eval()
        self.model.zero_grad()
        images = self.dataset.sample(n).to(self.device)
        output = self.model(images)

        # Compute same loss as training: pixel_loss + ssim_weight * (1 - SSIM)
        from acc.tasks.reconstruction import ssim_loss as _ssim_loss

        recon = output[ModelOutput.RECONSTRUCTION]
        if self.loss_fn == "mse":
            pixel_loss = nn.functional.mse_loss(recon, images)
        else:
            pixel_loss = nn.functional.l1_loss(recon, images)
        loss = pixel_loss
        if self.ssim_weight > 0:
            recon_clamped = recon.clamp(0, 1)
            loss = loss + self.ssim_weight * _ssim_loss(recon_clamped, images)
        loss_val = loss.item()
        l1_val = pixel_loss.item()

        # Backward to get gradients on all retained activations
        loss.backward()

        # Remove hooks
        for h in hooks:
            h.remove()

        self.model.train()

        # Build response — use first image
        input_img = _tensor_to_base64(images[0].detach())
        recon_img = _tensor_to_base64(recon[0].detach())

        # Per-pixel L1 error: |reconstruction - input| for the first image
        # This is [1, 28, 28] → squeeze to [28, 28]
        pixel_error = (recon[0] - images[0]).abs().detach().squeeze(0)
        error_map = pixel_error.cpu().numpy().tolist()

        # Collect conv layers for kernel extraction
        conv_layers: dict[str, nn.Module] = {}
        for i, layer in enumerate(self.model.encoder_layers):
            conv_layers[f"enc_{i}"] = layer
        for i, layer in enumerate(self.model.decoder_layers):
            conv_layers[f"dec_{i}"] = layer

        def _extract_kernels(
            layer_name: str, out_ch: int
        ) -> list[list[list[float]]] | None:
            """Extract masked kernel weights for a given output channel.

            Returns list of 2D arrays — one KxK kernel per input channel
            (only connected inputs, i.e. mask==1).
            """
            lyr = conv_layers.get(layer_name)
            if lyr is None:
                return None
            w = lyr.conv.weight.detach()  # [C_out, C_in, K, K] or transpose
            mask = lyr.conn_mask.detach()  # [C_out, C_in, 1, 1] or transpose
            if lyr.is_transpose:
                # Transpose conv: weight is [C_in, C_out, K, K], mask is [C_in, C_out, 1, 1]
                # For output channel out_ch, kernels are w[:, out_ch, :, :]
                kernels = []
                for in_ch in range(w.shape[0]):
                    if mask[in_ch, out_ch, 0, 0].item() > 0:
                        kernels.append(w[in_ch, out_ch].cpu().numpy().tolist())
                return kernels if kernels else None
            else:
                # Regular conv: weight is [C_out, C_in, K, K], mask is [C_out, C_in, 1, 1]
                kernels = []
                for in_ch in range(w.shape[1]):
                    if mask[out_ch, in_ch, 0, 0].item() > 0:
                        kernels.append(w[out_ch, in_ch].cpu().numpy().tolist())
                return kernels if kernels else None

        def _layer_maps(prefix: str, genome_layers, count: int) -> list:
            result = []
            for i in range(count):
                name = f"{prefix}_{i}"
                act_tensor = activations.get(name)
                if act_tensor is None:
                    continue
                per_image = act_tensor[0].detach()  # [C, H, W]
                grad_tensor = act_tensor.grad
                per_grad = grad_tensor[0].detach() if grad_tensor is not None else None
                C, H, W = per_image.shape
                channels = []
                for c in range(C):
                    ch_data = per_image[c].cpu().numpy().tolist()
                    ch_grad = (
                        per_grad[c].cpu().numpy().tolist()
                        if per_grad is not None
                        else None
                    )
                    act_name = (
                        genome_layers[i].channel_descriptors[c].activation
                        if c < len(genome_layers[i].channel_descriptors)
                        else "?"
                    )
                    is_frozen = (
                        genome_layers[i].channel_descriptors[c].frozen
                        if c < len(genome_layers[i].channel_descriptors)
                        else False
                    )
                    entry: dict = {"activation": act_name, "data": ch_data}
                    if ch_grad is not None:
                        entry["grad"] = ch_grad
                    kernels = _extract_kernels(name, c)
                    if kernels is not None:
                        entry["kernels"] = kernels
                    if is_frozen:
                        entry["frozen"] = True
                    channels.append(entry)
                result.append(
                    {
                        "name": name,
                        "resolution": [H, W],
                        "channels": channels,
                    }
                )
            return result

        # Build latent (post-pool) feature map + gradient
        latent_maps = None
        latent_tensor = activations.get("latent")
        if latent_tensor is not None:
            per_image = latent_tensor[0].detach()
            grad_tensor = latent_tensor.grad
            per_grad = grad_tensor[0].detach() if grad_tensor is not None else None
            C, H, W = per_image.shape
            channels = []
            last_enc = (
                self.genome.encoder_layers[-1] if self.genome.encoder_layers else None
            )
            for c in range(C):
                ch_data = per_image[c].cpu().numpy().tolist()
                ch_grad = (
                    per_grad[c].cpu().numpy().tolist() if per_grad is not None else None
                )
                act_name = (
                    last_enc.channel_descriptors[c].activation
                    if last_enc and c < len(last_enc.channel_descriptors)
                    else "?"
                )
                entry: dict = {"activation": act_name, "data": ch_data}
                if ch_grad is not None:
                    entry["grad"] = ch_grad
                channels.append(entry)
            latent_maps = {
                "name": "latent",
                "resolution": [H, W],
                "channels": channels,
            }

        return {
            "input_image": input_img,
            "recon_image": recon_img,
            "error_map": error_map,
            "loss": loss_val,
            "l1": l1_val,
            "encoder": _layer_maps(
                "enc", self.genome.encoder_layers, len(self.model.encoder_layers)
            ),
            "latent": latent_maps,
            "decoder": _layer_maps(
                "dec", self.genome.decoder_layers, len(self.model.decoder_layers)
            ),
        }

    def teardown(self):
        """Clean up all session state."""
        self.model = None
        self.genome = None
        self.trainer = None
        self.dataset = None
        self.device = None
        self.step_count = 0
        self.last_loss = None
        self.undo_stack = []
        self.task_configs = IECSession._default_task_configs()
