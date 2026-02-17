"""Slant generator — strokes at controlled angles.

Produces 32x32 grayscale images of white strokes on black backgrounds,
similar to MNIST visual style. Each image has strokes at a controlled
angle. The target label is the normalized angle (0-1, mapping -45 to +45 degrees).

The encoder sees these through the same conv weights as MNIST digits.
When the slant probe trains on these, it pushes z[7:10] to encode
stroke angle. Those same filters fire on slanted strokes in MNIST.
"""

import math

import torch
import numpy as np
from PIL import Image, ImageDraw

from acc.dataset import AccDataset
from acc.generators.base import DatasetGenerator


class SlantGenerator(DatasetGenerator):
    """Generate synthetic slant dataset with controlled stroke angles."""

    name = "slant"
    description = "White strokes at controlled angles, target = normalized angle (0-1)"
    parameters = {
        "n": {"type": "int", "default": 5000, "description": "Number of images"},
        "image_size": {"type": "int", "default": 32, "description": "Image size (square)"},
    }

    def generate(self, **params) -> AccDataset:
        n = int(params.get("n", 5000))
        image_size = int(params.get("image_size", 32))
        return generate_slant(n=n, image_size=image_size)


def generate_slant(n: int = 5000, image_size: int = 32) -> AccDataset:
    """Generate synthetic slant dataset.

    Args:
        n: Number of images to generate.
        image_size: Output image size (square).

    Returns:
        AccDataset with float targets (normalized angle, 0 to 1).
    """
    images = []
    targets = []

    min_angle = -45.0  # degrees
    max_angle = 45.0

    for _ in range(n):
        angle = np.random.uniform(min_angle, max_angle)
        normalized_angle = (angle - min_angle) / (max_angle - min_angle)

        img = _draw_slanted_stroke(image_size, angle)
        images.append(img)
        targets.append(normalized_angle)

    images_tensor = torch.stack(images)  # (N, 1, H, W)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)  # (N,)

    return AccDataset(images_tensor, targets_tensor, name="slant_synth")


def _draw_slanted_stroke(image_size: int, angle_deg: float) -> torch.Tensor:
    """Draw strokes at a specific angle.

    Draws 1-3 parallel strokes at the given angle, varying position
    and width slightly for diversity.

    Returns (1, H, W) tensor with values in [0, 1].
    """
    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)

    cx = image_size / 2
    cy = image_size / 2
    angle_rad = math.radians(angle_deg)

    n_strokes = np.random.randint(1, 4)

    for _ in range(n_strokes):
        # Random offset from center
        offset_x = np.random.uniform(-image_size * 0.2, image_size * 0.2)
        offset_y = np.random.uniform(-image_size * 0.2, image_size * 0.2)

        # Stroke length
        length = np.random.uniform(image_size * 0.4, image_size * 0.8)
        half_len = length / 2

        # Endpoints along the angle
        dx = math.cos(angle_rad) * half_len
        dy = math.sin(angle_rad) * half_len

        x0 = cx + offset_x - dx
        y0 = cy + offset_y - dy
        x1 = cx + offset_x + dx
        y1 = cy + offset_y + dy

        width = np.random.randint(1, 4)
        draw.line([(x0, y0), (x1, y1)], fill=255, width=width)

    # Convert to tensor
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
