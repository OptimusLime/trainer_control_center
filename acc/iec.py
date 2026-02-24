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
