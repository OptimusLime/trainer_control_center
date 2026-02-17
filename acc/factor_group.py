"""FactorGroup — defines a named slice of the latent vector.

Factor groups partition the latent space. Each group owns a contiguous
range of dims. Tasks attach to specific groups via latent_slice.
"""

from dataclasses import dataclass


@dataclass
class FactorGroup:
    """A named contiguous slice of the latent vector.

    Args:
        name: Human-readable name (e.g., "position", "shape", "free").
        latent_start: Start index in z (inclusive).
        latent_end: End index in z (exclusive).
    """

    name: str
    latent_start: int
    latent_end: int

    @property
    def latent_dim(self) -> int:
        return self.latent_end - self.latent_start

    @property
    def latent_slice(self) -> tuple[int, int]:
        """Returns (start, end) tuple compatible with Task.latent_slice."""
        return (self.latent_start, self.latent_end)

    def __post_init__(self):
        if self.latent_start < 0:
            raise ValueError(f"latent_start must be >= 0, got {self.latent_start}")
        if self.latent_end <= self.latent_start:
            raise ValueError(
                f"latent_end must be > latent_start, got "
                f"start={self.latent_start}, end={self.latent_end}"
            )


def validate_factor_groups(groups: list[FactorGroup], total_latent_dim: int) -> None:
    """Validate that factor groups tile [0, total_latent_dim) without gaps or overlaps.

    Raises ValueError with clear message if invalid.
    """
    if not groups:
        return

    sorted_groups = sorted(groups, key=lambda g: g.latent_start)

    # Check no overlaps and no gaps
    expected_start = 0
    for g in sorted_groups:
        if g.latent_start != expected_start:
            if g.latent_start < expected_start:
                raise ValueError(
                    f"Factor groups overlap: '{g.name}' starts at {g.latent_start} "
                    f"but previous group ends at {expected_start}"
                )
            else:
                raise ValueError(
                    f"Gap in factor groups: nothing covers [{expected_start}, {g.latent_start})"
                )
        expected_start = g.latent_end

    if expected_start != total_latent_dim:
        raise ValueError(
            f"Factor groups end at {expected_start} but total_latent_dim={total_latent_dim}. "
            f"Groups must tile the entire latent range."
        )
