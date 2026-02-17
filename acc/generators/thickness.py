"""Thickness generator — lines and curves at controlled stroke widths.

Produces 32x32 grayscale images of white strokes on black backgrounds,
similar to MNIST visual style. Each image has a single stroke with a
controlled width. The target label is the normalized stroke width (0-1).

The encoder sees these through the same conv weights as MNIST digits.
When the thickness probe trains on these, it pushes z[4:7] to encode
stroke width. Those same filters fire on thick/thin strokes in MNIST.
"""

import math

import torch
import numpy as np
from PIL import Image, ImageDraw

from acc.dataset import AccDataset
from acc.generators.base import DatasetGenerator


class ThicknessGenerator(DatasetGenerator):
    """Generate synthetic thickness dataset with controlled stroke widths."""

    name = "thickness"
    description = "White strokes at controlled widths, target = normalized width (0-1)"
    parameters = {
        "n": {"type": "int", "default": 5000, "description": "Number of images"},
        "image_size": {"type": "int", "default": 32, "description": "Image size (square)"},
    }

    def generate(self, **params) -> AccDataset:
        n = int(params.get("n", 5000))
        image_size = int(params.get("image_size", 32))
        return generate_thickness(n=n, image_size=image_size)


def generate_thickness(n: int = 5000, image_size: int = 32) -> AccDataset:
    """Generate synthetic thickness dataset.

    Args:
        n: Number of images to generate.
        image_size: Output image size (square).

    Returns:
        AccDataset with float targets (normalized stroke width, 0 to 1).
    """
    images = []
    targets = []

    # Stroke widths from 1 to 8 pixels
    min_width = 1.0
    max_width = 8.0

    for _ in range(n):
        width = np.random.uniform(min_width, max_width)
        normalized_width = (width - min_width) / (max_width - min_width)

        img = _draw_stroke(image_size, width)
        images.append(img)
        targets.append(normalized_width)

    images_tensor = torch.stack(images)  # (N, 1, H, W)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)  # (N,)

    return AccDataset(images_tensor, targets_tensor, name="thickness_synth")


def _draw_stroke(image_size: int, width: float) -> torch.Tensor:
    """Draw a random stroke (line or curve) with given width.

    Returns (1, H, W) tensor with values in [0, 1].
    """
    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)

    # Random stroke type
    stroke_type = np.random.choice(["line", "arc", "polyline"])

    margin = int(image_size * 0.1)
    w = int(max(1, round(width)))

    if stroke_type == "line":
        x0 = np.random.randint(margin, image_size - margin)
        y0 = np.random.randint(margin, image_size - margin)
        x1 = np.random.randint(margin, image_size - margin)
        y1 = np.random.randint(margin, image_size - margin)
        draw.line([(x0, y0), (x1, y1)], fill=255, width=w)

    elif stroke_type == "arc":
        cx = np.random.randint(margin, image_size - margin)
        cy = np.random.randint(margin, image_size - margin)
        r = np.random.randint(4, image_size // 3)
        start_angle = np.random.uniform(0, 360)
        end_angle = start_angle + np.random.uniform(60, 270)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.arc(bbox, start_angle, end_angle, fill=255, width=w)

    elif stroke_type == "polyline":
        n_points = np.random.randint(3, 6)
        points = [
            (
                np.random.randint(margin, image_size - margin),
                np.random.randint(margin, image_size - margin),
            )
            for _ in range(n_points)
        ]
        draw.line(points, fill=255, width=w, joint="curve")

    # Convert to tensor
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
