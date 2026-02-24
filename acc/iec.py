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
    transfer_weights,
)
from acc.model_output import ModelOutput
from acc.trainer import Trainer
from acc.tasks.reconstruction import ReconstructionTask
from acc.dataset import AccDataset
from acc.checkpoints import CheckpointStore


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

    def _build_trainer(self):
        """(Re)build the trainer with current model and dataset."""
        task = ReconstructionTask("recon", self.dataset)
        task.attach(self.model)
        self.trainer = Trainer(
            self.model,
            [task],
            self.device,
            lr=self._lr,
            batch_size=128,
        )

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
        else:
            # Roll back — invalid mutation type
            self.undo_stack.pop()
            raise ValueError(
                f"Unknown mutation type '{mutation_type}'. "
                f"Available: add_channel, remove_channel, change_activation, "
                f"add_connection, remove_connection, add_layer, remove_layer"
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
        """Extract per-channel feature maps from all layers via forward hooks.

        Runs inference on n images and captures each ConvCPPNLayer's output.
        Returns per-channel activations as 2D number arrays (for InspectHeatmap).

        Args:
            n: Number of images to run (returns feature maps for the first one).

        Returns:
            {
                "input_image": "base64_png",
                "encoder": [
                    {
                        "name": "enc_0",
                        "resolution": [H, W],
                        "channels": [
                            {"activation": "identity", "data": [[...], ...]},
                            ...
                        ]
                    },
                    ...
                ],
                "decoder": [ same structure ],
            }
        """
        if self.model is None or self.dataset is None:
            raise RuntimeError("No active IEC session")

        activations: dict[str, torch.Tensor] = {}
        hooks = []

        # Register forward hooks on all ConvCPPNLayers
        for i, layer in enumerate(self.model.encoder_layers):
            name = f"enc_{i}"

            def hook_fn(mod, inp, out, name=name):
                activations[name] = out.detach()

            hooks.append(layer.register_forward_hook(hook_fn))

        for i, layer in enumerate(self.model.decoder_layers):
            name = f"dec_{i}"

            def hook_fn(mod, inp, out, name=name):
                activations[name] = out.detach()

            hooks.append(layer.register_forward_hook(hook_fn))

        # Run forward pass
        self.model.eval()
        with torch.no_grad():
            images = self.dataset.sample(n).to(self.device)
            self.model(images)

        # Remove hooks
        for h in hooks:
            h.remove()

        self.model.train()

        # Build response — use first image
        input_img = _tensor_to_base64(images[0])

        def _layer_maps(prefix: str, genome_layers, count: int) -> list:
            result = []
            for i in range(count):
                name = f"{prefix}_{i}"
                act_tensor = activations.get(name)
                if act_tensor is None:
                    continue
                # act_tensor is [B, C, H, W] — take first image
                per_image = act_tensor[0]  # [C, H, W]
                C, H, W = per_image.shape
                channels = []
                for c in range(C):
                    ch_data = per_image[c].cpu().numpy().tolist()  # [[...], ...]
                    act_name = (
                        genome_layers[i].channel_descriptors[c].activation
                        if c < len(genome_layers[i].channel_descriptors)
                        else "?"
                    )
                    channels.append(
                        {
                            "activation": act_name,
                            "data": ch_data,
                        }
                    )
                result.append(
                    {
                        "name": name,
                        "resolution": [H, W],
                        "channels": channels,
                    }
                )
            return result

        return {
            "input_image": input_img,
            "encoder": _layer_maps(
                "enc", self.genome.encoder_layers, len(self.model.encoder_layers)
            ),
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
