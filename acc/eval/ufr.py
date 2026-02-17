"""UFR (Unsupervised Factor Recovery) scoring.

Quantifies how well a factor-slot autoencoder disentangles factors.
Uses the concept x context transfer matrix approach:

For each factor group (concept):
  1. Encode a batch of images -> z (using MU for determinism)
  2. Measure how much variance each factor group captures
  3. Build a transfer matrix: how much does varying one factor
     affect each factor group's latent activations?

Metrics:
  - Disentanglement: each factor group captures only ONE concept
    (low entropy across rows of the transfer matrix)
  - Completeness: each concept is captured by only ONE factor group
    (low entropy across columns of the transfer matrix)
  - UFR: harmonic mean of disentanglement and completeness

All scores are in [0, 1] where 1 = perfect disentanglement.
"""

import torch
import torch.nn as nn
import numpy as np

from acc.eval_metric import EvalMetric
from acc.model_output import ModelOutput
from acc.dataset import AccDataset


def compute_ufr(
    model: nn.Module,
    datasets: dict[str, "AccDataset"],
    device: torch.device,
    n_samples: int = 500,
) -> dict[str, float]:
    """Compute UFR disentanglement metrics.

    Args:
        model: A FactorSlotAutoencoder with factor_groups.
        datasets: Available datasets (uses the first one).
        device: Torch device.
        n_samples: Number of images to encode for statistics.

    Returns:
        dict with EvalMetric.UFR, EvalMetric.DISENTANGLEMENT,
        EvalMetric.COMPLETENESS keys.
    """
    if not hasattr(model, "factor_groups"):
        raise ValueError("Model has no factor_groups — cannot compute UFR")

    factor_groups = model.factor_groups
    n_factors = len(factor_groups)

    if n_factors < 2:
        # With only one factor, disentanglement is trivially 1.0
        return {
            EvalMetric.UFR: 1.0,
            EvalMetric.DISENTANGLEMENT: 1.0,
            EvalMetric.COMPLETENESS: 1.0,
        }

    # Get images from first dataset
    ds = next(iter(datasets.values()), None)
    if ds is None:
        raise ValueError("No datasets available for UFR computation")

    n_encode = min(n_samples, len(ds))
    images = ds.sample(n_encode).to(device)

    model.eval()
    with torch.no_grad():
        model_out = model(images)
        z = model_out[ModelOutput.MU]  # (N, D) — use mean for determinism

    # Build variance matrix: how much variance does each factor group have?
    # V[i] = variance of factor group i's activations across the batch
    # This tells us how "active" each factor group is.
    var_per_group = []
    for fg in factor_groups:
        fg_z = z[:, fg.latent_start : fg.latent_end]  # (N, fg_dim)
        var_per_group.append(fg_z.var(dim=0).sum().item())

    # Build transfer matrix: T[i, j] = how much does the variance in
    # factor group j's inputs correlate with factor group i's latent?
    #
    # We use the correlation approach:
    # For each pair of factor groups (i, j), compute the average
    # absolute correlation between their latent dimensions.
    transfer = np.zeros((n_factors, n_factors))

    z_np = z.cpu().numpy()
    for i, fg_i in enumerate(factor_groups):
        zi = z_np[:, fg_i.latent_start : fg_i.latent_end]  # (N, di)
        for j, fg_j in enumerate(factor_groups):
            zj = z_np[:, fg_j.latent_start : fg_j.latent_end]  # (N, dj)
            # Average absolute correlation between all pairs of dims
            # Normalize each dim to zero mean, unit variance
            zi_norm = (zi - zi.mean(axis=0, keepdims=True))
            zj_norm = (zj - zj.mean(axis=0, keepdims=True))
            zi_std = zi_norm.std(axis=0, keepdims=True) + 1e-8
            zj_std = zj_norm.std(axis=0, keepdims=True) + 1e-8
            zi_norm = zi_norm / zi_std
            zj_norm = zj_norm / zj_std

            # Cross-correlation matrix (di x dj)
            corr = np.abs(zi_norm.T @ zj_norm) / n_encode
            transfer[i, j] = corr.mean()

    # Normalize rows to sum to 1 (each factor group's total correlation)
    row_sums = transfer.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    transfer_norm = transfer / row_sums

    # Disentanglement: each row should have low entropy
    # (each factor group should correlate with only ONE other group)
    # Perfect: each row is a one-hot vector -> entropy = 0
    disentanglement = _avg_one_minus_normalized_entropy(transfer_norm, axis=1)

    # Completeness: each column should have low entropy
    # (each concept should be captured by only ONE factor group)
    col_sums = transfer.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-8)
    transfer_col_norm = transfer / col_sums
    completeness = _avg_one_minus_normalized_entropy(transfer_col_norm, axis=0)

    # UFR: harmonic mean
    if disentanglement + completeness > 0:
        ufr = 2 * disentanglement * completeness / (disentanglement + completeness)
    else:
        ufr = 0.0

    return {
        EvalMetric.UFR: float(ufr),
        EvalMetric.DISENTANGLEMENT: float(disentanglement),
        EvalMetric.COMPLETENESS: float(completeness),
    }


def _avg_one_minus_normalized_entropy(matrix: np.ndarray, axis: int) -> float:
    """Average (1 - normalized_entropy) across rows or columns.

    Normalized entropy: H / log(K) where K is the number of elements.
    Returns value in [0, 1] where 1 = perfectly concentrated (one-hot).
    """
    K = matrix.shape[1 - axis]  # number of elements along the OTHER axis
    if K <= 1:
        return 1.0

    log_K = np.log(K)
    entropies = []

    if axis == 1:
        # Iterate over rows
        for i in range(matrix.shape[0]):
            row = matrix[i]
            row = row[row > 1e-10]  # avoid log(0)
            h = -np.sum(row * np.log(row))
            entropies.append(h / log_K)
    else:
        # Iterate over columns
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            col = col[col > 1e-10]
            h = -np.sum(col * np.log(col))
            entropies.append(h / log_K)

    avg_normalized_entropy = np.mean(entropies)
    return float(1.0 - avg_normalized_entropy)
