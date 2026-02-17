"""AccDataset — thin wrapper over torch Dataset.

Adds describe() for dashboard display and sample() for thumbnails.
Has built-in train/eval split.
"""

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class AccDataset(Dataset):
    """Dataset for the Autoencoder Control Center.

    Wraps image tensors and optional targets with metadata for dashboard display.

    Args:
        images: Tensor of shape [N, C, H, W] with values in [0, 1].
        targets: Optional tensor of shape [N, ...] (int for classification, float for regression).
        name: Human-readable name for dashboard display.
        train_fraction: Fraction of data used for training (rest is eval).
    """

    def __init__(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        name: str = "unnamed",
        train_fraction: float = 0.85,
    ):
        assert images.ndim == 4, f"Expected [N, C, H, W], got shape {images.shape}"
        self.images = images
        self.targets = targets
        self.name = name

        # Train/eval split indices
        n = len(images)
        n_train = int(n * train_fraction)
        self._train_indices = list(range(n_train))
        self._eval_indices = list(range(n_train, n))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        img = self.images[idx]
        if self.targets is not None:
            return img, self.targets[idx]
        return (img,)

    @property
    def target_type(self) -> Optional[str]:
        """Returns 'classes' for int targets, 'float' for float targets, None if no targets."""
        if self.targets is None:
            return None
        if self.targets.dtype in (torch.int32, torch.int64, torch.long):
            return "classes"
        return "float"

    @property
    def num_classes(self) -> Optional[int]:
        """Number of unique classes if target_type is 'classes'."""
        if self.target_type != "classes":
            return None
        return int(self.targets.max().item()) + 1

    def describe(self) -> dict:
        """Metadata for dashboard display."""
        info = {
            "name": self.name,
            "size": len(self),
            "image_shape": list(self.images.shape[1:]),
            "target_type": self.target_type,
            "train_size": len(self._train_indices),
            "eval_size": len(self._eval_indices),
        }
        if self.num_classes is not None:
            info["num_classes"] = self.num_classes
        return info

    def sample(self, n: int) -> torch.Tensor:
        """Random sample of n images for dashboard thumbnails."""
        indices = torch.randperm(len(self))[:n]
        return self.images[indices]

    def train_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """DataLoader over the training split."""
        subset = Subset(self, self._train_indices)
        return DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )

    def eval_loader(self, batch_size: int) -> DataLoader:
        """DataLoader over the eval split."""
        subset = Subset(self, self._eval_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=False)


def load_mnist(image_size: int = 64, data_dir: str = "./acc/data") -> AccDataset:
    """Load MNIST dataset, resize to image_size, return as AccDataset.

    Downloads MNIST via torchvision if not already cached.
    """
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
        ]
    )

    mnist = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Materialize into tensors for fast access
    images = []
    targets = []
    loader = DataLoader(mnist, batch_size=1000, shuffle=False)
    for batch_imgs, batch_labels in loader:
        images.append(batch_imgs)
        targets.append(batch_labels)

    images = torch.cat(images, dim=0)
    targets = torch.cat(targets, dim=0).long()

    return AccDataset(images, targets, name="mnist")
