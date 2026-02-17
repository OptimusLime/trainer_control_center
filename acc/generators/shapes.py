"""Synthetic shapes generator with factor labels.

Generates images of simple shapes (circles, squares, triangles) with
known position, scale, and shape class. Used for testing factor-slot
architectures where ground-truth factor values are needed.
"""

import math

import torch

from acc.dataset import AccDataset
from acc.generators.base import DatasetGenerator


class ShapesGenerator(DatasetGenerator):
    """Generate synthetic shape images with factor labels."""

    name = "shapes"
    description = "Circles/squares/triangles with position, scale, and shape class targets"
    parameters = {
        "n": {"type": "int", "default": 5000, "description": "Number of images"},
        "image_size": {"type": "int", "default": 64, "description": "Image size (square)"},
        "num_shapes": {"type": "int", "default": 3, "description": "Number of shape types (up to 3)"},
    }

    def generate(self, **params) -> AccDataset:
        n = int(params.get("n", 5000))
        image_size = int(params.get("image_size", 64))
        num_shapes = int(params.get("num_shapes", 3))
        return generate_shapes(n=n, image_size=image_size, num_shapes=num_shapes)


def generate_shapes(
    n: int = 5000,
    image_size: int = 64,
    channels: int = 1,
    num_shapes: int = 3,
) -> AccDataset:
    """Generate synthetic shape images with factor labels.

    Each image contains one shape. Returns an AccDataset where targets
    is a float tensor of shape [N, 4]: [x_pos, y_pos, scale, shape_class].

    Args:
        n: Number of images to generate.
        image_size: Width/height of generated images.
        channels: Number of image channels (1=grayscale).
        num_shapes: Number of shape types (up to 3: circle, square, triangle).

    Returns:
        AccDataset with float targets [x_pos, y_pos, scale, shape_id].
    """
    images = torch.zeros(n, channels, image_size, image_size)
    targets = torch.zeros(n, 4, dtype=torch.float32)

    for i in range(n):
        # Random factors
        x_pos = torch.rand(1).item()  # [0, 1] normalized position
        y_pos = torch.rand(1).item()
        scale = 0.1 + torch.rand(1).item() * 0.3  # [0.1, 0.4] of image size
        shape_id = torch.randint(0, num_shapes, (1,)).item()

        # Store targets
        targets[i] = torch.tensor([x_pos, y_pos, scale, float(shape_id)])

        # Render shape
        cx = int(x_pos * (image_size - 1))
        cy = int(y_pos * (image_size - 1))
        radius = int(scale * image_size)

        img = torch.zeros(image_size, image_size)

        if shape_id == 0:  # circle
            yy, xx = torch.meshgrid(
                torch.arange(image_size), torch.arange(image_size), indexing="ij"
            )
            dist = ((xx - cx).float() ** 2 + (yy - cy).float() ** 2).sqrt()
            img[dist <= radius] = 1.0

        elif shape_id == 1:  # square
            y_min = max(0, cy - radius)
            y_max = min(image_size, cy + radius)
            x_min = max(0, cx - radius)
            x_max = min(image_size, cx + radius)
            img[y_min:y_max, x_min:x_max] = 1.0

        elif shape_id == 2:  # triangle
            for row in range(max(0, cy - radius), min(image_size, cy + radius)):
                # Width shrinks linearly from base to top
                progress = (row - (cy - radius)) / max(2 * radius, 1)
                half_width = int(radius * (1.0 - progress))
                x_min = max(0, cx - half_width)
                x_max = min(image_size, cx + half_width)
                img[row, x_min:x_max] = 1.0

        # Apply to all channels
        for c in range(channels):
            images[i, c] = img

    return AccDataset(images, targets, name="synthetic_shapes")
