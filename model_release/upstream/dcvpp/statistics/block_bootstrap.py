"""Deterministic paired moving-block bootstrap with common model indices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np


DEFAULT_BLOCK_LENGTHS = (7, 14, 28)


@dataclass(frozen=True)
class BootstrapEstimate:
    model: str
    requested_block_length: int
    effective_block_length: int
    observation_count: int
    sample_mean: float
    lower_confidence_bound: float
    central_interval_lower: float
    central_interval_upper: float
    confidence_level: float
    resamples: int
    index_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "requested_block_length": self.requested_block_length,
            "effective_block_length": self.effective_block_length,
            "observation_count": self.observation_count,
            "sample_mean": self.sample_mean,
            "lower_confidence_bound": self.lower_confidence_bound,
            "central_interval_lower": self.central_interval_lower,
            "central_interval_upper": self.central_interval_upper,
            "confidence_level": self.confidence_level,
            "resamples": self.resamples,
            "index_hash": self.index_hash,
        }


@dataclass(frozen=True)
class PairedBootstrapResult:
    estimates: Mapping[int, Mapping[str, BootstrapEstimate]]
    indices_by_block: Mapping[int, np.ndarray]
    common_valid_mask: np.ndarray
    confidence_level: float
    seed: int
    paired_indices: bool = True

    def lcb(self, model: str, block_length: int = 7) -> float:
        return self.estimates[int(block_length)][str(model)].lower_confidence_bound

    @property
    def lower_confidence_bounds(self) -> dict[int, dict[str, float]]:
        return {
            block: {
                model: estimate.lower_confidence_bound
                for model, estimate in model_estimates.items()
            }
            for block, model_estimates in self.estimates.items()
        }

    def as_dict(self, *, include_indices: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "confidence_level": self.confidence_level,
            "seed": self.seed,
            "paired_indices": self.paired_indices,
            "common_valid_count": int(np.count_nonzero(self.common_valid_mask)),
            "estimates": {
                str(block): {
                    model: estimate.as_dict()
                    for model, estimate in model_estimates.items()
                }
                for block, model_estimates in self.estimates.items()
            },
        }
        if include_indices:
            payload["indices_by_block"] = {
                str(block): indices.tolist()
                for block, indices in self.indices_by_block.items()
            }
        return payload


def _block_seed(seed: int, block_length: int) -> int:
    digest = hashlib.sha256(
        f"DART-ValueSelect|{int(seed)}|block={int(block_length)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def moving_block_indices(
    observation_count: int,
    block_length: int,
    *,
    resamples: int = 2000,
    seed: int = 20260809,
    circular: bool = True,
) -> np.ndarray:
    """Generate a deterministic ``(resamples, n)`` contiguous-block index map."""

    count = int(observation_count)
    requested = int(block_length)
    draws = int(resamples)
    if count <= 0:
        raise ValueError("observation_count must be positive")
    if requested <= 0:
        raise ValueError("block_length must be positive")
    if draws <= 0:
        raise ValueError("resamples must be positive")
    effective = min(requested, count)
    blocks_per_sample = int(math.ceil(count / effective))
    rng = np.random.default_rng(int(seed))
    if circular:
        starts = rng.integers(0, count, size=(draws, blocks_per_sample))
        offsets = np.arange(effective, dtype=int)
        indices = (starts[..., None] + offsets) % count
    else:
        maximum_start = count - effective
        starts = rng.integers(
            0, maximum_start + 1, size=(draws, blocks_per_sample)
        )
        indices = starts[..., None] + np.arange(effective, dtype=int)
    return indices.reshape(draws, -1)[:, :count]


def _validate_paired_series(
    values_by_model: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not values_by_model:
        raise ValueError("values_by_model must contain at least one model")
    converted: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for raw_name, raw_values in sorted(values_by_model.items()):
        name = str(raw_name)
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"daily values for {name} must be nonempty and 1-D")
        if expected_length is None:
            expected_length = values.size
        elif values.size != expected_length:
            raise ValueError("all models must share the same paired daily denominator")
        converted[name] = values
    stacked = np.vstack(list(converted.values()))
    common_valid_mask = np.all(np.isfinite(stacked), axis=0)
    if not bool(np.all(common_valid_mask)):
        invalid_positions = np.flatnonzero(~common_valid_mask)
        preview = invalid_positions[:10].astype(int).tolist()
        raise ValueError(
            "paired model series contain non-finite daily values; silent "
            f"day filtering is forbidden (count={invalid_positions.size}, "
            f"positions={preview})"
        )
    # Preserve the mask in the serialized audit for backward compatibility,
    # but never use it to shorten the preregistered operating-date panel.
    return converted, common_valid_mask


def paired_moving_block_bootstrap(
    values_by_model: Mapping[str, Sequence[float] | np.ndarray],
    *,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    primary_block_length: int = 7,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20260809,
    circular: bool = True,
) -> PairedBootstrapResult:
    """Bootstrap paired mean value using one common index map per block size.

    The 95% selection LCB is the 5th percentile (one-sided), while the returned
    central interval is the conventional 2.5th--97.5th percentile interval.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    blocks = tuple(int(block) for block in block_lengths)
    if not blocks or any(block <= 0 for block in blocks):
        raise ValueError("block_lengths must contain positive integers")
    if len(set(blocks)) != len(blocks):
        raise ValueError("block_lengths must be unique")
    if int(primary_block_length) not in blocks:
        raise ValueError("primary_block_length must appear in block_lengths")
    paired_values, common_mask = _validate_paired_series(values_by_model)
    count = next(iter(paired_values.values())).size
    estimate_map: dict[int, dict[str, BootstrapEstimate]] = {}
    index_map: dict[int, np.ndarray] = {}
    one_sided_tail = 1.0 - confidence_level
    central_tail = (1.0 - confidence_level) / 2.0
    for requested_block in blocks:
        effective = min(requested_block, count)
        indices = moving_block_indices(
            count,
            effective,
            resamples=resamples,
            seed=_block_seed(seed, requested_block),
            circular=circular,
        )
        indices.setflags(write=False)
        index_map[requested_block] = indices
        index_hash = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
        model_estimates: dict[str, BootstrapEstimate] = {}
        for model, values in paired_values.items():
            bootstrap_means = np.mean(values[indices], axis=1)
            model_estimates[model] = BootstrapEstimate(
                model=model,
                requested_block_length=requested_block,
                effective_block_length=effective,
                observation_count=count,
                sample_mean=float(np.mean(values)),
                lower_confidence_bound=float(
                    np.quantile(bootstrap_means, one_sided_tail)
                ),
                central_interval_lower=float(
                    np.quantile(bootstrap_means, central_tail)
                ),
                central_interval_upper=float(
                    np.quantile(bootstrap_means, 1.0 - central_tail)
                ),
                confidence_level=confidence_level,
                resamples=int(resamples),
                index_hash=index_hash,
            )
        estimate_map[requested_block] = model_estimates
    common_mask.setflags(write=False)
    return PairedBootstrapResult(
        estimates=estimate_map,
        indices_by_block=index_map,
        common_valid_mask=common_mask,
        confidence_level=confidence_level,
        seed=int(seed),
    )


def paired_bootstrap_lcbs(
    values_by_model: Mapping[str, Sequence[float] | np.ndarray],
    **kwargs: Any,
) -> dict[int, dict[str, float]]:
    return paired_moving_block_bootstrap(
        values_by_model, **kwargs
    ).lower_confidence_bounds


generate_moving_block_indices = moving_block_indices


__all__ = [
    "BootstrapEstimate",
    "DEFAULT_BLOCK_LENGTHS",
    "PairedBootstrapResult",
    "generate_moving_block_indices",
    "moving_block_indices",
    "paired_bootstrap_lcbs",
    "paired_moving_block_bootstrap",
]
