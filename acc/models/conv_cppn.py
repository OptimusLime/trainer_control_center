"""ConvCPPN — Convolutional CPPN autoencoder with heterogeneous per-channel activations.

A CPPN node = a conv output channel. A CPPN connection = a weight slice in the
conv kernel. This module implements CPPN-style networks using standard PyTorch
conv layers with two additions: (1) per-channel heterogeneous activations via
grouped dispatch, (2) connection masking via a binary buffer.

Coordinate channels (X, Y, Gaussian) are concatenated to the input, giving the
network absolute positional awareness — the defining CPPN property.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acc.model_output import ModelOutput

# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

ACTIVATION_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "sin": torch.sin,
    "cos": torch.cos,
    "gaussian": lambda x: torch.exp(-x * x / 2.0),
    "abs": torch.abs,
    "softplus": lambda x: F.softplus(x),
}

ACTIVATION_NAMES: list[str] = list(ACTIVATION_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Genome dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChannelDescriptor:
    """Describes a single channel (CPPN node) in a layer."""

    activation: str  # key into ACTIVATION_REGISTRY
    is_passthrough: bool = False  # identity relay for skip connections
    passthrough_source: int = -1  # source channel idx in previous layer
    frozen: bool = False  # pass-through weights don't train

    def to_dict(self) -> dict:
        d: dict = {"activation": self.activation}
        if self.is_passthrough:
            d["is_passthrough"] = True
            d["passthrough_source"] = self.passthrough_source
            d["frozen"] = self.frozen
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelDescriptor":
        return cls(
            activation=d["activation"],
            is_passthrough=d.get("is_passthrough", False),
            passthrough_source=d.get("passthrough_source", -1),
            frozen=d.get("frozen", False),
        )


@dataclass
class LayerGenome:
    """Describes one conv layer's topology."""

    channel_descriptors: list[ChannelDescriptor]
    connection_mask: list[list[int]]  # [C_out, C_in] binary
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1

    @property
    def out_channels(self) -> int:
        return len(self.channel_descriptors)

    def to_dict(self) -> dict:
        return {
            "channel_descriptors": [cd.to_dict() for cd in self.channel_descriptors],
            "connection_mask": self.connection_mask,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayerGenome":
        return cls(
            channel_descriptors=[
                ChannelDescriptor.from_dict(cd) for cd in d["channel_descriptors"]
            ],
            connection_mask=d["connection_mask"],
            kernel_size=d.get("kernel_size", 3),
            stride=d.get("stride", 1),
            padding=d.get("padding", 1),
        )


@dataclass
class ConvCPPNGenome:
    """Full topology descriptor for a ConvCPPN autoencoder."""

    encoder_layers: list[LayerGenome]
    decoder_layers: list[LayerGenome]
    metadata: dict = field(default_factory=dict)

    @property
    def bottleneck_channels(self) -> int:
        """Number of channels at the encoder output (= bottleneck)."""
        if not self.encoder_layers:
            return 0
        return self.encoder_layers[-1].out_channels

    def to_dict(self) -> dict:
        return {
            "encoder_layers": [lg.to_dict() for lg in self.encoder_layers],
            "decoder_layers": [lg.to_dict() for lg in self.decoder_layers],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConvCPPNGenome":
        return cls(
            encoder_layers=[LayerGenome.from_dict(lg) for lg in d["encoder_layers"]],
            decoder_layers=[LayerGenome.from_dict(lg) for lg in d["decoder_layers"]],
            metadata=d.get("metadata", {}),
        )


def default_genome() -> ConvCPPNGenome:
    """Simplest possible ConvCPPN. 1 channel everywhere.

    Encoder: 4→1 identity, stride=2 (28→14). Pool to 3x3. Latent = 1*9 = 9.
    Decoder: 3 layers, 1 channel each, progressive upsample 3→7→14→28.
             The human grows it from here.
    """
    encoder_layer = LayerGenome(
        channel_descriptors=[ChannelDescriptor(activation="relu")],
        connection_mask=[[1, 1, 1, 1]],  # 1 output × 4 inputs
        kernel_size=3,
        stride=2,
        padding=1,
    )
    # Decoder layer 0: 3→7. Input = 1 bottleneck + 3 coords = 4
    dec_0 = LayerGenome(
        channel_descriptors=[ChannelDescriptor(activation="relu")],
        connection_mask=[[1, 1, 1, 1]],
        kernel_size=3,
        stride=1,
        padding=1,
    )
    # Decoder layer 1: 7→14. Input = 1 + 3 coords = 4
    dec_1 = LayerGenome(
        channel_descriptors=[ChannelDescriptor(activation="relu")],
        connection_mask=[[1, 1, 1, 1]],
        kernel_size=3,
        stride=1,
        padding=1,
    )
    # Decoder layer 2: 14→28. Input = 1 + 3 coords = 4. Output = 1ch relu.
    dec_2 = LayerGenome(
        channel_descriptors=[ChannelDescriptor(activation="relu")],
        connection_mask=[[1, 1, 1, 1]],
        kernel_size=3,
        stride=1,
        padding=1,
    )
    return ConvCPPNGenome(
        encoder_layers=[encoder_layer],
        decoder_layers=[dec_0, dec_1, dec_2],
    )


# ---------------------------------------------------------------------------
# Coordinate channels
# ---------------------------------------------------------------------------


def make_coord_channels(H: int, W: int) -> torch.Tensor:
    """Create [3, H, W] coordinate maps: X ∈ [-1,1], Y ∈ [-1,1], Gauss."""
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    gauss = torch.exp(-(grid_x**2 + grid_y**2) / 1.0)
    return torch.stack([grid_x, grid_y, gauss], dim=0)


# ---------------------------------------------------------------------------
# HeterogeneousActivation
# ---------------------------------------------------------------------------


class HeterogeneousActivation(nn.Module):
    """Per-channel heterogeneous activation via grouped dispatch.

    Groups channels by activation type, applies each group's function in one
    vectorized op. Cost: one kernel launch per unique activation type.
    """

    def __init__(self, activations: list[str]):
        super().__init__()
        self.activation_names = list(activations)

        # Group channels by activation type
        groups: dict[str, list[int]] = {}
        for i, act_name in enumerate(activations):
            if act_name not in ACTIVATION_REGISTRY:
                raise ValueError(
                    f"Unknown activation '{act_name}'. "
                    f"Available: {list(ACTIVATION_REGISTRY.keys())}"
                )
            groups.setdefault(act_name, []).append(i)

        # Register index buffers and store (name, fn) pairs
        self._group_names: list[str] = []
        for act_name, indices in groups.items():
            self.register_buffer(
                f"idx_{act_name}", torch.tensor(indices, dtype=torch.long)
            )
            self._group_names.append(act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for act_name in self._group_names:
            fn = ACTIVATION_REGISTRY[act_name]
            idx = getattr(self, f"idx_{act_name}")
            out[:, idx] = fn(x[:, idx])
        return out


# ---------------------------------------------------------------------------
# ConvCPPNLayer
# ---------------------------------------------------------------------------


class ConvCPPNLayer(nn.Module):
    """Single conv layer with connection masking and heterogeneous activation.

    The connection mask is a [C_out, C_in] binary buffer. Masked connections
    produce zero output regardless of optimizer state. Gradient zeroing for
    masked connections prevents Adam from accumulating state on dead connections.
    """

    def __init__(
        self,
        in_channels: int,
        channel_descriptors: list[ChannelDescriptor],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        connection_mask: Optional[torch.Tensor] = None,
        transpose: bool = False,
        output_padding: int = 0,
    ):
        super().__init__()
        out_channels = len(channel_descriptors)
        self.descriptors = list(channel_descriptors)
        self.is_transpose = transpose

        # Build conv
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=True,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )

        # Connection mask: for Conv2d weight is [C_out, C_in, K, K]
        # For ConvTranspose2d weight is [C_in, C_out, K, K] — mask must match
        if connection_mask is None:
            mask = torch.ones(out_channels, in_channels)
        else:
            mask = connection_mask.float()

        if transpose:
            # ConvTranspose2d weight: [C_in, C_out, K, K] → mask needs [C_in, C_out]
            mask_4d = mask.T.unsqueeze(-1).unsqueeze(-1)
        else:
            # Conv2d weight: [C_out, C_in, K, K] → mask is [C_out, C_in]
            mask_4d = mask.unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("conn_mask", mask_4d)

        # Heterogeneous activation
        act_names = [d.activation for d in channel_descriptors]
        self.activation = HeterogeneousActivation(act_names)

        # Frozen channel indices
        self.frozen_channels = [
            i for i, d in enumerate(channel_descriptors) if d.frozen
        ]

        # Initialize pass-through channels
        self._init_passthroughs()

        # Zero masked weights at init
        with torch.no_grad():
            self.conv.weight.data *= self.conn_mask

    def _init_passthroughs(self):
        """Set pass-through channels to identity kernel."""
        with torch.no_grad():
            for i, desc in enumerate(self.descriptors):
                if desc.is_passthrough and desc.passthrough_source >= 0:
                    self.conv.weight[i].zero_()
                    kh = self.conv.kernel_size[0] // 2
                    kw = self.conv.kernel_size[1] // 2
                    self.conv.weight[i, desc.passthrough_source, kh, kw] = 1.0
                    self.conv.bias.data[i] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply connection mask to weights
        weight = self.conv.weight * self.conn_mask

        if self.is_transpose:
            out = F.conv_transpose2d(
                x,
                weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                output_padding=self.conv.output_padding,
            )
        else:
            out = F.conv2d(
                x, weight, self.conv.bias, self.conv.stride, self.conv.padding
            )

        return self.activation(out)

    def zero_masked_grads(self):
        """Zero gradients for masked connections and frozen channels.

        Call after loss.backward(), before optimizer.step().
        """
        if self.conv.weight.grad is not None:
            self.conv.weight.grad *= self.conn_mask
            for i in self.frozen_channels:
                self.conv.weight.grad[i].zero_()
                if self.conv.bias.grad is not None:
                    self.conv.bias.grad[i] = 0.0


# ---------------------------------------------------------------------------
# ConvCPPN
# ---------------------------------------------------------------------------


class ConvCPPN(nn.Module):
    """Convolutional CPPN autoencoder.

    Encoder: image + coord channels → conv layers → AdaptiveAvgPool → latent
    Decoder: latent + coord channels → transposed conv layers → reconstruction

    Satisfies the model protocol: has_decoder, latent_dim, config(), forward()
    returns {LATENT, RECONSTRUCTION, SPATIAL}.
    """

    def __init__(self, genome: ConvCPPNGenome):
        super().__init__()
        self._genome = genome

        # Build coordinate buffers at each resolution we might need
        for res in [28, 14, 7, 3]:
            self.register_buffer(
                f"coords_{res}", make_coord_channels(res, res), persistent=False
            )

        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        enc_in = 4  # image (1) + coords (3)
        for lg in genome.encoder_layers:
            mask = torch.tensor(lg.connection_mask, dtype=torch.float32)
            layer = ConvCPPNLayer(
                in_channels=enc_in,
                channel_descriptors=lg.channel_descriptors,
                kernel_size=lg.kernel_size,
                stride=lg.stride,
                padding=lg.padding,
                connection_mask=mask,
            )
            self.encoder_layers.append(layer)
            enc_in = lg.out_channels

        # Bottleneck
        self.pool = nn.AdaptiveAvgPool2d(3)
        self._bottleneck_channels = genome.bottleneck_channels

        # Build decoder layers — using regular conv (not transpose) with interpolation
        # This is simpler and more controllable than transposed conv math
        self.decoder_layers = nn.ModuleList()
        # First decoder input: bottleneck channels + 3 coord channels
        dec_in = self._bottleneck_channels + 3
        for i, lg in enumerate(genome.decoder_layers):
            is_last = i == len(genome.decoder_layers) - 1
            mask = torch.tensor(lg.connection_mask, dtype=torch.float32)
            layer = ConvCPPNLayer(
                in_channels=dec_in,
                channel_descriptors=lg.channel_descriptors,
                kernel_size=lg.kernel_size,
                stride=1,  # always stride 1 — upsampling done via interpolate
                padding=lg.padding,
                connection_mask=mask,
                transpose=False,
            )
            self.decoder_layers.append(layer)
            if is_last:
                dec_in = lg.out_channels  # won't be used
            else:
                # Next decoder layer gets this output + 3 coord channels
                dec_in = lg.out_channels + 3

    # -- Model protocol --

    @property
    def has_decoder(self) -> bool:
        return True

    @property
    def latent_dim(self) -> int:
        return self._bottleneck_channels * 3 * 3

    def config(self) -> dict:
        return {
            "class": "ConvCPPN",
            "latent_dim": self.latent_dim,
            "bottleneck_channels": self._bottleneck_channels,
            "genome": self._genome.to_dict(),
        }

    # -- Build from genome --

    @classmethod
    def from_genome(cls, genome: ConvCPPNGenome) -> "ConvCPPN":
        """Build a ConvCPPN from a genome descriptor."""
        return cls(genome)

    def to_genome(self) -> ConvCPPNGenome:
        """Extract current topology as a genome (weights NOT included)."""
        return copy.deepcopy(self._genome)

    # -- Forward --

    def _get_coords(self, res: int, batch_size: int) -> torch.Tensor:
        """Get coordinate channels at given resolution, expanded to batch size."""
        coords = getattr(self, f"coords_{res}")  # [3, H, W]
        return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, 1, 28, 28] input images (raw MNIST).

        Returns:
            dict with LATENT [B, D], RECONSTRUCTION [B, 1, 28, 28], SPATIAL [B, C, H, W].
        """
        B = x.shape[0]

        # Encoder: concat coord channels to input
        coords_28 = self._get_coords(28, B)
        h = torch.cat([x, coords_28], dim=1)  # [B, 4, 28, 28]

        for layer in self.encoder_layers:
            h = layer(h)

        spatial = h  # encoder output before pooling

        # Bottleneck: pool to 3x3, flatten for latent
        pooled = self.pool(spatial)  # [B, C, 3, 3]
        latent = pooled.view(B, -1)  # [B, C*9]

        # Decoder: upsample then conv at each stage
        # Target resolutions for decoder: interpolate to this before conv
        # For the starter (1 decoder layer): 3x3 → upsample to 28x28 → conv → 28x28 out
        # For multi-layer: 3→7→14→28 etc.
        _DECODER_RESOLUTIONS = [28]  # single-layer decoder goes straight to 28
        if len(self.decoder_layers) == 2:
            _DECODER_RESOLUTIONS = [14, 28]
        elif len(self.decoder_layers) == 3:
            _DECODER_RESOLUTIONS = [7, 14, 28]
        elif len(self.decoder_layers) >= 4:
            _DECODER_RESOLUTIONS = [7, 14, 28] + [28] * (len(self.decoder_layers) - 3)

        h = pooled
        for i, layer in enumerate(self.decoder_layers):
            # Upsample to target resolution
            target_res = _DECODER_RESOLUTIONS[i]
            h = F.interpolate(h, size=target_res, mode="bilinear", align_corners=False)
            # Concatenate coord channels at this resolution
            coords = self._get_coords(target_res, B)
            h = torch.cat([h, coords], dim=1)
            h = layer(h)

        # Output is [B, 1, 28, 28] — already has sigmoid from last layer activation
        reconstruction = h

        return {
            ModelOutput.LATENT: latent,
            ModelOutput.RECONSTRUCTION: reconstruction,
            ModelOutput.SPATIAL: spatial,
        }

    def zero_masked_grads(self):
        """Zero gradients for all masked/frozen connections across all layers."""
        for layer in self.encoder_layers:
            layer.zero_masked_grads()
        for layer in self.decoder_layers:
            layer.zero_masked_grads()

    def resolution_info(self) -> dict:
        """Per-layer resolution info for the UI.

        Returns:
            {
                "encoder": [{"input_res": 28, "output_res": 14}, ...],
                "decoder": [{"input_res": 3, "output_res": 7}, ...],
                "bottleneck_res": 3,
            }
        """
        # Encoder: starts at 28, each stride-2 layer halves
        enc_info = []
        res = 28
        for lg in self._genome.encoder_layers:
            out_res = res
            if lg.stride == 2:
                out_res = res // 2
            enc_info.append({"input_res": res, "output_res": out_res})
            res = out_res

        # Decoder resolution map (same logic as forward())
        n_dec = len(self._genome.decoder_layers)
        if n_dec == 1:
            dec_targets = [28]
        elif n_dec == 2:
            dec_targets = [14, 28]
        elif n_dec == 3:
            dec_targets = [7, 14, 28]
        else:
            dec_targets = [7, 14, 28] + [28] * (n_dec - 3)

        dec_info = []
        prev_res = 3  # bottleneck is always 3x3
        for i, target in enumerate(dec_targets):
            dec_info.append({"input_res": prev_res, "output_res": target})
            prev_res = target

        return {
            "encoder": enc_info,
            "decoder": dec_info,
            "bottleneck_res": 3,
        }

    def architecture_summary(self) -> str:
        """Human-readable architecture summary."""
        parts = []

        # Encoder
        enc_parts = []
        for i, lg in enumerate(self._genome.encoder_layers):
            acts = ",".join(cd.activation for cd in lg.channel_descriptors)
            n_ch = lg.out_channels
            enc_parts.append(f"{n_ch}({acts})")
        parts.append(f"Encoder: 4→{'→'.join(enc_parts)}")

        # Bottleneck
        parts.append(f"Bottleneck: {self._bottleneck_channels}ch @ 3x3")

        # Decoder
        dec_parts = []
        for i, lg in enumerate(self._genome.decoder_layers):
            acts = ",".join(cd.activation for cd in lg.channel_descriptors)
            n_ch = lg.out_channels
            dec_parts.append(f"{n_ch}({acts})")
        parts.append(f"Decoder: {'→'.join(dec_parts)}")

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Genome mutation functions — pure transforms on ConvCPPNGenome
# ---------------------------------------------------------------------------


def add_channel(
    genome: ConvCPPNGenome,
    side: str,
    layer_idx: int,
    activation: str,
    connect_all: bool = True,
) -> ConvCPPNGenome:
    """Add a channel (CPPN node) to a layer.

    Args:
        genome: Source genome (not mutated).
        side: "encoder" or "decoder".
        layer_idx: Index into encoder_layers or decoder_layers.
        activation: Activation function name from ACTIVATION_REGISTRY.
        connect_all: If True, connect to all inputs. If False, no connections.

    Returns:
        New genome with the added channel.
    """
    if activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{activation}'. Available: {ACTIVATION_NAMES}"
        )

    genome = copy.deepcopy(genome)
    layers = genome.encoder_layers if side == "encoder" else genome.decoder_layers

    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(
            f"Layer index {layer_idx} out of range for {side} (have {len(layers)} layers)"
        )

    layer = layers[layer_idx]
    n_in = len(layer.connection_mask[0]) if layer.connection_mask else 0

    # Add the channel descriptor
    layer.channel_descriptors.append(ChannelDescriptor(activation=activation))

    # Add a row to the connection mask (new output channel connected to all/no inputs)
    new_row = [1] * n_in if connect_all else [0] * n_in
    layer.connection_mask.append(new_row)

    # Propagate: this layer's output changed, so the next layer's input must match.
    _propagate_output_change(genome, side, layer_idx)

    return genome


def remove_channel(
    genome: ConvCPPNGenome,
    side: str,
    layer_idx: int,
    channel_idx: int,
) -> ConvCPPNGenome:
    """Remove a channel from a layer.

    Args:
        genome: Source genome (not mutated).
        side: "encoder" or "decoder".
        layer_idx: Index into encoder_layers or decoder_layers.
        channel_idx: Which channel to remove.

    Returns:
        New genome with the channel removed.

    Raises:
        ValueError if removing would leave the layer with 0 channels, or if
        this is the last decoder layer (must keep 1 output channel).
    """
    genome = copy.deepcopy(genome)
    layers = genome.encoder_layers if side == "encoder" else genome.decoder_layers

    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Layer index {layer_idx} out of range for {side}")

    layer = layers[layer_idx]
    if layer.out_channels <= 1:
        raise ValueError("Cannot remove the last channel from a layer")

    if channel_idx < 0 or channel_idx >= layer.out_channels:
        raise ValueError(
            f"Channel index {channel_idx} out of range (have {layer.out_channels})"
        )

    # For the last decoder layer, we must keep exactly 1 output channel (the reconstruction)
    if side == "decoder" and layer_idx == len(genome.decoder_layers) - 1:
        raise ValueError(
            "Cannot remove channels from the final decoder layer (must output 1 channel)"
        )

    # Remove the channel descriptor and its mask row
    layer.channel_descriptors.pop(channel_idx)
    layer.connection_mask.pop(channel_idx)

    # Propagate: this layer's output changed, so the next layer's input must match.
    _propagate_output_change(genome, side, layer_idx)

    return genome


def change_activation(
    genome: ConvCPPNGenome,
    side: str,
    layer_idx: int,
    channel_idx: int,
    new_activation: str,
) -> ConvCPPNGenome:
    """Change a channel's activation function.

    Returns:
        New genome with the changed activation.
    """
    if new_activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{new_activation}'. Available: {ACTIVATION_NAMES}"
        )

    genome = copy.deepcopy(genome)
    layers = genome.encoder_layers if side == "encoder" else genome.decoder_layers

    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Layer index {layer_idx} out of range for {side}")

    layer = layers[layer_idx]
    if channel_idx < 0 or channel_idx >= layer.out_channels:
        raise ValueError(
            f"Channel index {channel_idx} out of range (have {layer.out_channels})"
        )

    layer.channel_descriptors[channel_idx].activation = new_activation
    return genome


def add_connection(
    genome: ConvCPPNGenome,
    side: str,
    layer_idx: int,
    out_ch: int,
    in_ch: int,
) -> ConvCPPNGenome:
    """Enable a connection between an input channel and output channel.

    Returns:
        New genome with the connection enabled.
    """
    genome = copy.deepcopy(genome)
    layers = genome.encoder_layers if side == "encoder" else genome.decoder_layers
    layer = layers[layer_idx]
    layer.connection_mask[out_ch][in_ch] = 1
    return genome


def remove_connection(
    genome: ConvCPPNGenome,
    side: str,
    layer_idx: int,
    out_ch: int,
    in_ch: int,
) -> ConvCPPNGenome:
    """Disable a connection between an input channel and output channel.

    Returns:
        New genome with the connection disabled.
    """
    genome = copy.deepcopy(genome)
    layers = genome.encoder_layers if side == "encoder" else genome.decoder_layers
    layer = layers[layer_idx]
    layer.connection_mask[out_ch][in_ch] = 0
    return genome


def add_encoder_layer(
    genome: ConvCPPNGenome,
    position: int = -1,
    activation: str = "identity",
    channels: int = 1,
    stride: int = 2,
) -> ConvCPPNGenome:
    """Add a layer to the encoder.

    Args:
        genome: Source genome (not mutated).
        position: Where to insert (-1 = end). 0-indexed.
        activation: Activation for all new channels.
        channels: Number of channels in the new layer.
        stride: Stride for the new layer (1 or 2).

    Returns:
        New genome with the added layer.
    """
    if activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{activation}'. Available: {ACTIVATION_NAMES}"
        )

    genome = copy.deepcopy(genome)
    n_enc = len(genome.encoder_layers)

    if position < 0:
        position = n_enc  # append at end

    if position < 0 or position > n_enc:
        raise ValueError(
            f"Position {position} out of range for encoder (have {n_enc} layers)"
        )

    # Determine input channels for the new layer
    if position == 0:
        in_ch = 4  # image + coords
    else:
        in_ch = genome.encoder_layers[position - 1].out_channels

    # Build the new layer
    descs = [ChannelDescriptor(activation=activation) for _ in range(channels)]
    mask = [[1] * in_ch for _ in range(channels)]
    padding = 1 if stride == 2 else 1
    new_layer = LayerGenome(
        channel_descriptors=descs,
        connection_mask=mask,
        kernel_size=3,
        stride=stride,
        padding=padding,
    )

    # Insert the new layer
    genome.encoder_layers.insert(position, new_layer)

    # If there's a layer AFTER the new one, update its connection mask width
    if position < len(genome.encoder_layers) - 1:
        next_layer = genome.encoder_layers[position + 1]
        _resize_mask_columns(next_layer, channels)

    # If the new layer is at the END (or we inserted before end and shifted things),
    # the bottleneck may have changed. Update first decoder layer input.
    # The bottleneck is always the last encoder layer's output.
    _sync_decoder_input_to_bottleneck(genome)

    return genome


def remove_encoder_layer(
    genome: ConvCPPNGenome,
    layer_idx: int,
) -> ConvCPPNGenome:
    """Remove a layer from the encoder.

    Args:
        genome: Source genome (not mutated).
        layer_idx: Which layer to remove (0-indexed).

    Returns:
        New genome with the layer removed.

    Raises:
        ValueError if encoder would be left with 0 layers.
    """
    genome = copy.deepcopy(genome)
    n_enc = len(genome.encoder_layers)

    if n_enc <= 1:
        raise ValueError("Cannot remove the last encoder layer")
    if layer_idx < 0 or layer_idx >= n_enc:
        raise ValueError(
            f"Layer index {layer_idx} out of range for encoder (have {n_enc} layers)"
        )

    # Determine what the next layer's input channels should become
    if layer_idx == 0:
        new_in_ch = 4  # image + coords
    else:
        new_in_ch = genome.encoder_layers[layer_idx - 1].out_channels

    # Remove the layer
    genome.encoder_layers.pop(layer_idx)

    # If there's a layer at layer_idx now (the one that was after), update its mask
    if layer_idx < len(genome.encoder_layers):
        next_layer = genome.encoder_layers[layer_idx]
        _resize_mask_columns(next_layer, new_in_ch)

    # Update first decoder layer input (bottleneck changed)
    _sync_decoder_input_to_bottleneck(genome)

    return genome


def add_decoder_layer(
    genome: ConvCPPNGenome,
    position: int = 0,
    activation: str = "relu",
    channels: int = 1,
) -> ConvCPPNGenome:
    """Add a layer to the decoder.

    Args:
        genome: Source genome (not mutated).
        position: Where to insert (0-indexed). -1 = before the last layer
                  (since last must output 1ch, you can't add after it meaningfully).
        activation: Activation for all new channels.
        channels: Number of channels in the new layer.

    Returns:
        New genome with the added layer.
    """
    if activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{activation}'. Available: {ACTIVATION_NAMES}"
        )

    genome = copy.deepcopy(genome)
    n_dec = len(genome.decoder_layers)

    if position < 0:
        position = n_dec - 1  # insert before the last (output) layer

    # Clamp: can't insert after the last layer
    if position > n_dec - 1:
        position = n_dec - 1

    if position < 0:
        position = 0

    # Determine input channels for the new layer.
    # Decoder inputs: layer 0 gets (bottleneck + 3 coords), layer i>0 gets (prev_out + 3 coords).
    if position == 0:
        in_ch = genome.bottleneck_channels + 3
    else:
        in_ch = genome.decoder_layers[position - 1].out_channels + 3

    # Build the new layer
    descs = [ChannelDescriptor(activation=activation) for _ in range(channels)]
    mask = [[1] * in_ch for _ in range(channels)]
    new_layer = LayerGenome(
        channel_descriptors=descs,
        connection_mask=mask,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    # Insert
    genome.decoder_layers.insert(position, new_layer)

    # Update the NEXT layer's connection mask (it now receives new_layer's output + 3 coords)
    if position + 1 < len(genome.decoder_layers):
        next_layer = genome.decoder_layers[position + 1]
        _resize_mask_columns(next_layer, channels + 3)  # output channels + 3 coords

    return genome


def remove_decoder_layer(
    genome: ConvCPPNGenome,
    layer_idx: int,
) -> ConvCPPNGenome:
    """Remove a layer from the decoder.

    Args:
        genome: Source genome (not mutated).
        layer_idx: Which layer to remove (0-indexed). Cannot remove the last layer.

    Returns:
        New genome with the layer removed.

    Raises:
        ValueError if decoder would be left with 0 layers, or if trying to
        remove the final output layer.
    """
    genome = copy.deepcopy(genome)
    n_dec = len(genome.decoder_layers)

    if n_dec <= 1:
        raise ValueError(
            "Cannot remove the last decoder layer (need at least one for output)"
        )
    if layer_idx < 0 or layer_idx >= n_dec:
        raise ValueError(
            f"Layer index {layer_idx} out of range for decoder (have {n_dec} layers)"
        )
    if layer_idx == n_dec - 1:
        raise ValueError(
            "Cannot remove the final decoder layer (it produces the 1-channel output). "
            "Remove an earlier layer instead."
        )

    # Determine what the next layer's input channels should become
    if layer_idx == 0:
        new_in_ch = genome.bottleneck_channels + 3  # bottleneck + coords
    else:
        new_in_ch = genome.decoder_layers[layer_idx - 1].out_channels + 3

    # Remove the layer
    genome.decoder_layers.pop(layer_idx)

    # Update the layer that was after the removed one
    if layer_idx < len(genome.decoder_layers):
        next_layer = genome.decoder_layers[layer_idx]
        _resize_mask_columns(next_layer, new_in_ch)

    return genome


def _resize_mask_columns(layer: LayerGenome, new_in_ch: int) -> None:
    """Resize connection mask columns to match new input channel count.

    If growing: new columns are 1 (connected). If shrinking: truncate.
    """
    for row in layer.connection_mask:
        current = len(row)
        if new_in_ch > current:
            row.extend([1] * (new_in_ch - current))
        elif new_in_ch < current:
            del row[new_in_ch:]


def _sync_decoder_input_to_bottleneck(genome: ConvCPPNGenome) -> None:
    """Sync first decoder layer's connection mask to current bottleneck channels.

    The first decoder layer receives [bottleneck_channels + 3 coords] as input.
    """
    if not genome.decoder_layers:
        return
    expected_in = genome.bottleneck_channels + 3
    _resize_mask_columns(genome.decoder_layers[0], expected_in)


def _propagate_output_change(genome: ConvCPPNGenome, side: str, layer_idx: int) -> None:
    """After a layer's output channel count changes, update the next layer's input.

    Handles all cases:
    - Encoder layer N changed → encoder layer N+1's mask must match N's output.
    - Last encoder layer changed → first decoder layer's mask = bottleneck + 3 coords.
    - Decoder layer N changed → decoder layer N+1's mask = N's output + 3 coords.
    """
    layers = genome.encoder_layers if side == "encoder" else genome.decoder_layers
    new_out = layers[layer_idx].out_channels

    if side == "encoder":
        if layer_idx < len(layers) - 1:
            # Next encoder layer: input = this layer's output (no coords)
            _resize_mask_columns(layers[layer_idx + 1], new_out)
        # Last encoder layer always affects first decoder layer
        if layer_idx == len(layers) - 1:
            _sync_decoder_input_to_bottleneck(genome)
    else:
        # Decoder: next decoder layer input = this layer's output + 3 coords
        if layer_idx < len(layers) - 1:
            _resize_mask_columns(layers[layer_idx + 1], new_out + 3)


def _grow_decoder_input(genome: ConvCPPNGenome) -> None:
    """After adding an encoder output channel, grow the first decoder layer's input.

    The first decoder layer sees [bottleneck_channels + 3 coords] as input.
    We add a column to its connection mask (all ones = connected to new channel).
    """
    if not genome.decoder_layers:
        return
    dec_layer = genome.decoder_layers[0]
    for row in dec_layer.connection_mask:
        # Insert before the last 3 entries (which are coord channels)
        # Actually — the mask is [C_out, C_in] where C_in = bottleneck + 3.
        # The new bottleneck channel is the last one before coords.
        # We insert a 1 at position -3 (before X, Y, Gauss).
        row.insert(len(row) - 3, 1)


def _shrink_decoder_input(genome: ConvCPPNGenome, removed_channel_idx: int) -> None:
    """After removing an encoder output channel, shrink the first decoder layer's input.

    Remove the corresponding column from the first decoder layer's connection mask.
    """
    if not genome.decoder_layers:
        return
    dec_layer = genome.decoder_layers[0]
    for row in dec_layer.connection_mask:
        # The input order is [bottleneck_ch_0, ..., bottleneck_ch_N, X, Y, Gauss].
        # We remove the column at removed_channel_idx (within the bottleneck range).
        if removed_channel_idx < len(row):
            row.pop(removed_channel_idx)


# ---------------------------------------------------------------------------
# Weight transfer — copy compatible weights from old model to new model
# ---------------------------------------------------------------------------


def transfer_weights(
    old_model: ConvCPPN,
    new_model: ConvCPPN,
    old_genome: ConvCPPNGenome,
    new_genome: ConvCPPNGenome,
) -> None:
    """Transfer learned weights from old_model to new_model where dimensions match.

    For add_channel: all existing channels keep their weights. The new channel
    gets random initialization (from ConvCPPNLayer's default init).

    For remove_channel: remaining channels keep their weights.

    For change_activation: all weights are preserved (only the activation changes).

    This operates in-place on new_model.
    """
    _transfer_layer_list(
        old_model.encoder_layers,
        new_model.encoder_layers,
        old_genome.encoder_layers,
        new_genome.encoder_layers,
    )
    _transfer_layer_list(
        old_model.decoder_layers,
        new_model.decoder_layers,
        old_genome.decoder_layers,
        new_genome.decoder_layers,
    )


def _transfer_layer_list(
    old_layers: nn.ModuleList,
    new_layers: nn.ModuleList,
    old_genomes: list[LayerGenome],
    new_genomes: list[LayerGenome],
) -> None:
    """Transfer weights layer-by-layer for a matched encoder or decoder stack."""
    for i in range(min(len(old_layers), len(new_layers))):
        old_layer: ConvCPPNLayer = old_layers[i]
        new_layer: ConvCPPNLayer = new_layers[i]
        old_w = old_layer.conv.weight.data  # [C_out, C_in, K, K]
        old_b = old_layer.conv.bias.data  # [C_out]
        new_w = new_layer.conv.weight.data
        new_b = new_layer.conv.bias.data

        # Copy the overlapping region
        c_out = min(old_w.shape[0], new_w.shape[0])
        c_in = min(old_w.shape[1], new_w.shape[1])
        with torch.no_grad():
            new_w[:c_out, :c_in] = old_w[:c_out, :c_in]
            new_b[:c_out] = old_b[:c_out]
