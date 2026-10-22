"""Deterministic day-level inference for the formal R4 protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairedDailyInference:
    n_days: int
    mean_difference: float
    median_difference: float
    daily_win_rate: float
    confidence_level: float
    lower: float
    upper: float
    one_sided_upper: float
    block_days: int
    replicates: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def circular_moving_block_indices(
    n_days: int,
    *,
    block_days: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Return ordered circular block resamples of day indices."""

    if n_days < 2:
        raise ValueError("day-level inference requires at least two dates")
    if not 1 <= block_days <= n_days:
        raise ValueError("block_days must lie in [1, n_days]")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(int(seed))
    blocks = int(np.ceil(n_days / block_days))
    starts = rng.integers(0, n_days, size=(replicates, blocks))
    offsets = np.arange(block_days, dtype=np.int64)
    indices = (starts[..., None] + offsets) % n_days
    return indices.reshape(replicates, -1)[:, :n_days]


def paired_daily_inference(
    differences: Iterable[float] | np.ndarray,
    *,
    block_days: int = 7,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20240813,
) -> PairedDailyInference:
    """Paired moving-block inference where one row is exactly one date.

    A negative difference means the learned method has lower loss/cost than
    its paired benchmark.  ``daily_win_rate`` therefore counts values below
    zero.  Origins are never accepted as rows by this API.
    """

    values = _daily_vector(differences)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    indices = circular_moving_block_indices(
        values.size,
        block_days=block_days,
        replicates=replicates,
        seed=seed,
    )
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return PairedDailyInference(
        n_days=int(values.size),
        mean_difference=float(values.mean()),
        median_difference=float(np.median(values)),
        daily_win_rate=float(np.mean(values < 0.0)),
        confidence_level=float(confidence),
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
        one_sided_upper=float(np.quantile(means, confidence)),
        block_days=int(block_days),
        replicates=int(replicates),
        seed=int(seed),
    )


def hierarchical_seed_date_bootstrap(
    differences_by_seed_day: np.ndarray,
    *,
    block_days: int = 7,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20240813,
) -> dict[str, float | int | list[float]]:
    """Outer seed resampling and inner circular date-block resampling.

    Input must be ``[training_seed, eligible_date]``.  This is the registered
    final three-seed inference; a single-seed matrix is rejected because it
    cannot represent training-seed uncertainty.
    """

    matrix = np.asarray(differences_by_seed_day, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("hierarchical inference requires [at least 2 seeds, days]")
    if not np.isfinite(matrix).all():
        raise ValueError("seed/day differences must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    n_seeds, n_days = matrix.shape
    if not 1 <= block_days <= n_days:
        raise ValueError("block_days must lie in [1, n_days]")
    rng = np.random.default_rng(int(seed))
    sampled_means = np.empty(replicates, dtype=np.float64)
    blocks = int(np.ceil(n_days / block_days))
    offsets = np.arange(block_days, dtype=np.int64)
    for replicate in range(replicates):
        selected_seeds = rng.integers(0, n_seeds, size=n_seeds)
        seed_means = np.empty(n_seeds, dtype=np.float64)
        for position, seed_index in enumerate(selected_seeds):
            starts = rng.integers(0, n_days, size=blocks)
            date_indices = (
                starts[:, None] + offsets[None, :]
            ).reshape(-1)[:n_days] % n_days
            seed_means[position] = matrix[seed_index, date_indices].mean()
        sampled_means[replicate] = seed_means.mean()
    tail = (1.0 - confidence) / 2.0
    per_seed_means = matrix.mean(axis=1)
    return {
        "method": "hierarchical_seed_outer_date_circular_moving_block_inner",
        "n_training_seeds": int(n_seeds),
        "n_days_per_seed": int(n_days),
        "block_days": int(block_days),
        "replicates": int(replicates),
        "confidence_level": float(confidence),
        "seed": int(seed),
        "mean_difference": float(matrix.mean()),
        "median_daily_difference": float(np.median(matrix.mean(axis=0))),
        "daily_win_rate": float(np.mean(matrix < 0.0)),
        "per_seed_means": [float(value) for value in per_seed_means],
        "three_seed_mean": float(per_seed_means.mean()),
        "three_seed_std": float(per_seed_means.std(ddof=1)),
        "lower": float(np.quantile(sampled_means, tail)),
        "upper": float(np.quantile(sampled_means, 1.0 - tail)),
        "one_sided_upper": float(np.quantile(sampled_means, confidence)),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjusted p-values in the original comparison order."""

    values = np.asarray(tuple(p_values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a nonempty vector")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("p_values must be finite values in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        (values.size - np.arange(values.size)) * sorted_values
    ).clip(max=1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _daily_vector(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(tuple(values) if not isinstance(values, np.ndarray) else values)
    array = array.astype(np.float64, copy=False)
    if array.ndim != 1:
        raise ValueError("paired inference accepts one value per day, not origins")
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("daily differences must contain at least two finite values")
    return array


__all__ = [
    "PairedDailyInference",
    "circular_moving_block_indices",
    "hierarchical_seed_date_bootstrap",
    "holm_adjust",
    "paired_daily_inference",
]
