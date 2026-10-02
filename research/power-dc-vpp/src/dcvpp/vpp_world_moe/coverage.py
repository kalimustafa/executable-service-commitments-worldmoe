"""Finite-ensemble coverage mathematics for R5-V2.3 uncertainty gates.

Training remains a 20-path Monte Carlo calculation.  Release coverage uses a
separate 100-path ensemble and rank intervals whose exchangeable finite-sample
targets are explicit.  This module contains no runner or gate policy; it only
provides deterministic scientific primitives for those consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import torch
from torch import Tensor


PUBLIC_FAMILY_SLICES: Final = MappingProxyType(
    {
        "price": slice(0, 12),
        "workload": slice(12, 13),
        "pv": slice(13, 17),
        "pue": slice(17, 21),
    }
)
FORMAL_COVERAGE_PATHS: Final = 100
ANALYTIC_QUANTILE_LEVELS: Final = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)


class CoverageMathError(ValueError):
    """A tensor cannot support the registered finite-ensemble calculation."""


@dataclass(frozen=True)
class RankIntervalSpec:
    """One central order-statistic interval for a 100-path ensemble.

    Indices are zero based.  ``finite_sample_target`` is the exact probability
    that an exchangeable truth falls between the two one-based order
    statistics; endpoint ties have probability zero under continuous
    calibration.
    """

    nominal_percent: int
    lower_index: int
    upper_index: int
    finite_sample_target: float


@dataclass(frozen=True)
class ConditionalRankInterval:
    """Per-element interior interval after excluding structural atom paths."""

    lower: Tensor
    upper: Tensor
    finite_sample_target: Tensor
    valid_mask: Tensor
    interior_path_count: Tensor
    insufficient_interior_mask: Tensor


@dataclass(frozen=True)
class ConditionalQuantileInterval:
    """Path-free central interval of a conditional interior distribution."""

    lower: Tensor
    upper: Tensor
    finite_sample_target: Tensor
    valid_mask: Tensor


_RANK_INTERVALS_100: Final = MappingProxyType(
    {
        50: RankIntervalSpec(50, 24, 75, 51.0 / 101.0),
        80: RankIntervalSpec(80, 9, 90, 81.0 / 101.0),
        90: RankIntervalSpec(90, 4, 95, 91.0 / 101.0),
    }
)


def _nominal_percent(value: int | float) -> int:
    numeric = float(value)
    if 0.0 < numeric <= 1.0:
        numeric *= 100.0
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1.0e-12 or rounded not in _RANK_INTERVALS_100:
        raise CoverageMathError("coverage nominal must be one of 50, 80, or 90")
    return rounded


def rank_interval_spec_100(nominal: int | float) -> RankIntervalSpec:
    """Return the registered rank interval and its finite-sample target."""

    return _RANK_INTERVALS_100[_nominal_percent(nominal)]


def central_rank_interval_100(
    samples: Tensor,
    nominal: int | float,
    *,
    path_dim: int = 1,
) -> tuple[Tensor, Tensor, RankIntervalSpec]:
    """Select a central finite-rank interval from exactly 100 paths.

    Unlike linear ``torch.quantile`` interpolation, the returned order
    statistics have an auditable exchangeable target.  Scientific bounds are
    FP32 even if the neural rollout arrived in BF16.
    """

    if not isinstance(samples, Tensor) or samples.ndim < 1:
        raise CoverageMathError("coverage samples must be a non-scalar Tensor")
    axis = int(path_dim)
    if axis < 0:
        axis += samples.ndim
    if axis < 0 or axis >= samples.ndim:
        raise CoverageMathError("coverage path_dim is outside the sample tensor")
    if int(samples.shape[axis]) != FORMAL_COVERAGE_PATHS:
        raise CoverageMathError("formal coverage requires exactly 100 paths")
    values = samples.float()
    if not bool(torch.isfinite(values).all()):
        raise CoverageMathError("coverage samples contain non-finite values")
    spec = rank_interval_spec_100(nominal)
    ordered = values.sort(dim=axis).values
    lower = ordered.select(axis, spec.lower_index)
    upper = ordered.select(axis, spec.upper_index)
    return lower, upper, spec


def conditional_central_rank_interval_100(
    samples: Tensor,
    atom_mask: Tensor,
    truth_interior_mask: Tensor,
    nominal: int | float,
    *,
    path_dim: int = 1,
    minimum_interior_paths: int = 2,
) -> ConditionalRankInterval:
    """Rank an interior truth against only the non-atom members of 100 paths.

    Structural zeros and physical-boundary atoms are part of a mixed
    distribution and must be scored by their occurrence probabilities.  They
    cannot be inserted into a continuous central interval without destroying
    the exchangeable-rank target through ties.  The remaining path count may
    vary by element, so each valid element carries its own exact finite-sample
    target.
    """

    if not isinstance(samples, Tensor) or samples.ndim < 2:
        raise CoverageMathError("conditional coverage samples must be a Tensor")
    axis = int(path_dim)
    if axis < 0:
        axis += samples.ndim
    if axis < 0 or axis >= samples.ndim:
        raise CoverageMathError("conditional coverage path_dim is outside samples")
    if int(samples.shape[axis]) != FORMAL_COVERAGE_PATHS:
        raise CoverageMathError("conditional coverage requires exactly 100 paths")
    if atom_mask.shape != samples.shape or atom_mask.dtype is not torch.bool:
        raise CoverageMathError("atom_mask must be bool and match samples")
    expected_truth_shape = samples.shape[:axis] + samples.shape[axis + 1 :]
    if (
        truth_interior_mask.shape != expected_truth_shape
        or truth_interior_mask.dtype is not torch.bool
    ):
        raise CoverageMathError(
            "truth_interior_mask must be bool and omit only the path dimension"
        )
    minimum = int(minimum_interior_paths)
    if not 2 <= minimum <= FORMAL_COVERAGE_PATHS:
        raise CoverageMathError("minimum interior paths must lie in [2,100]")
    values = samples.float()
    if not bool(torch.isfinite(values).all()):
        raise CoverageMathError("conditional coverage samples are non-finite")
    nominal_fraction = float(_nominal_percent(nominal)) / 100.0
    interior_mask = ~atom_mask
    interior_count = interior_mask.sum(dim=axis)
    valid = truth_interior_mask & interior_count.ge(minimum)
    insufficient = truth_interior_mask & ~interior_count.ge(minimum)

    # Atom values sort to the end and are never selected because both order
    # indices are bounded by the element's interior count.
    ordered = values.masked_fill(atom_mask, torch.inf).sort(dim=axis).values
    count_plus_truth = interior_count + 1
    lower_order = torch.floor(
        (1.0 - nominal_fraction) * 0.5 * count_plus_truth.float()
    ).to(torch.long)
    lower_order = lower_order.clamp_min(1)
    lower_order = torch.minimum(lower_order, interior_count.clamp_min(1))
    upper_order = count_plus_truth - lower_order
    upper_order = torch.minimum(
        upper_order, interior_count.clamp_min(1)
    ).clamp_min(1)
    lower_index = (lower_order - 1).unsqueeze(axis)
    upper_index = (upper_order - 1).unsqueeze(axis)
    lower = torch.gather(ordered, axis, lower_index).squeeze(axis)
    upper = torch.gather(ordered, axis, upper_index).squeeze(axis)
    finite_target = (upper_order - lower_order).float() / count_plus_truth.float()
    zero = torch.zeros_like(finite_target)
    finite_target = torch.where(valid, finite_target, zero)
    return ConditionalRankInterval(
        lower=lower,
        upper=upper,
        finite_sample_target=finite_target,
        valid_mask=valid,
        interior_path_count=interior_count,
        insufficient_interior_mask=insufficient,
    )


def conditional_central_quantile_interval(
    quantiles: Tensor,
    truth_interior_mask: Tensor,
    nominal: int | float,
    *,
    levels: tuple[float, ...] | Tensor = ANALYTIC_QUANTILE_LEVELS,
    quantile_dim: int = -1,
) -> ConditionalQuantileInterval:
    """Select an analytic interval without filtering unconditional scenarios.

    The registered nine-level grid contains exact endpoints for 50%, 80%, and
    90% central intervals.  Consequently rare positive PV observations remain
    evaluable even when an unconditional finite ensemble happens to contain no
    draw from that component.  ``finite_sample_target`` keeps the historical
    field name for aggregation compatibility, but its value is the exact
    analytic nominal coverage rather than a rank correction.
    """

    if not isinstance(quantiles, Tensor) or quantiles.ndim < 2:
        raise CoverageMathError("conditional quantiles must be a Tensor")
    axis = int(quantile_dim)
    if axis < 0:
        axis += quantiles.ndim
    if axis < 0 or axis >= quantiles.ndim:
        raise CoverageMathError("conditional quantile_dim is outside quantiles")
    expected_truth_shape = quantiles.shape[:axis] + quantiles.shape[axis + 1 :]
    if (
        truth_interior_mask.shape != expected_truth_shape
        or truth_interior_mask.dtype is not torch.bool
    ):
        raise CoverageMathError(
            "truth_interior_mask must be bool and omit only the quantile dimension"
        )
    values = quantiles.float()
    if not bool(torch.isfinite(values).all()):
        raise CoverageMathError("conditional quantiles contain non-finite values")
    ordered_axis = values.movedim(axis, -1)
    if bool((ordered_axis[..., 1:] < ordered_axis[..., :-1] - 1.0e-6).any()):
        raise CoverageMathError("conditional quantiles are not monotone")

    supplied = torch.as_tensor(levels, dtype=torch.float32, device=values.device)
    if supplied.ndim != 1 or int(supplied.numel()) != int(values.shape[axis]):
        raise CoverageMathError("quantile levels do not match the quantile axis")
    if not bool(torch.isfinite(supplied).all()) or bool(
        ((supplied <= 0.0) | (supplied >= 1.0)).any()
    ):
        raise CoverageMathError("quantile levels must lie strictly inside (0,1)")
    if bool((supplied[1:] <= supplied[:-1]).any()):
        raise CoverageMathError("quantile levels must be strictly increasing")

    nominal_fraction = float(_nominal_percent(nominal)) / 100.0
    lower_level = 0.5 * (1.0 - nominal_fraction)
    upper_level = 1.0 - lower_level

    def exact_index(level: float) -> int:
        matches = torch.nonzero(
            torch.isclose(
                supplied,
                torch.tensor(level, device=supplied.device),
                rtol=0.0,
                atol=1.0e-7,
            ),
            as_tuple=False,
        ).flatten()
        if int(matches.numel()) != 1:
            raise CoverageMathError(
                f"conditional quantile grid lacks exact endpoint {level:.2f}"
            )
        return int(matches[0].item())

    lower = values.select(axis, exact_index(lower_level))
    upper = values.select(axis, exact_index(upper_level))
    target = torch.full_like(lower, nominal_fraction)
    target = torch.where(truth_interior_mask, target, torch.zeros_like(target))
    return ConditionalQuantileInterval(
        lower=lower,
        upper=upper,
        finite_sample_target=target,
        valid_mask=truth_interior_mask,
    )


def public_family_channel_weights(
    family: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return within-family channel weights that sum to one.

    Price preserves the registered DA/RT/DART masses 0.25/0.25/0.50 and
    distributes each mass uniformly across four nodes.  Other public families
    distribute their mass uniformly across their observed channels.
    """

    normalized = str(family).strip().lower()
    if normalized not in PUBLIC_FAMILY_SLICES:
        raise CoverageMathError(f"unknown public target family: {family!r}")
    if normalized == "price":
        values = [0.25 / 4.0] * 4 + [0.25 / 4.0] * 4 + [0.50 / 4.0] * 4
    else:
        channel_slice = PUBLIC_FAMILY_SLICES[normalized]
        channel_count = int(channel_slice.stop - channel_slice.start)
        values = [1.0 / float(channel_count)] * channel_count
    return torch.tensor(values, device=device, dtype=dtype)


def family_balanced_public_channel_weights(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return 21 channel weights with exactly one quarter mass per family."""

    return torch.cat(
        tuple(
            0.25
            * public_family_channel_weights(family, device=device, dtype=dtype)
            for family in PUBLIC_FAMILY_SLICES
        )
    )


__all__ = [
    "ANALYTIC_QUANTILE_LEVELS",
    "ConditionalQuantileInterval",
    "ConditionalRankInterval",
    "CoverageMathError",
    "FORMAL_COVERAGE_PATHS",
    "PUBLIC_FAMILY_SLICES",
    "RankIntervalSpec",
    "central_rank_interval_100",
    "conditional_central_rank_interval_100",
    "conditional_central_quantile_interval",
    "family_balanced_public_channel_weights",
    "public_family_channel_weights",
    "rank_interval_spec_100",
]
