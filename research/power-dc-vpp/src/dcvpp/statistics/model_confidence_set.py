"""Paired moving-block-bootstrap Model Confidence Set for daily losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .block_bootstrap import moving_block_indices


@dataclass(frozen=True)
class MCSResult:
    included_models: tuple[str, ...]
    excluded_models: tuple[str, ...]
    elimination_order: tuple[str, ...]
    mean_losses: Mapping[str, float]
    p_values: Mapping[str, float]
    confidence_level: float
    block_length: int
    resamples: int
    seed: int
    loss_semantics: str
    statistic: str = "centered_studentized_global_loss_range"
    step_history: tuple[Mapping[str, Any], ...] = ()
    status: str = "OK"

    @property
    def winner_set(self) -> tuple[str, ...]:
        return self.included_models

    def as_dict(self) -> dict[str, Any]:
        return {
            "included_models": list(self.included_models),
            "excluded_models": list(self.excluded_models),
            "elimination_order": list(self.elimination_order),
            "mean_losses": dict(self.mean_losses),
            "p_values": dict(self.p_values),
            "confidence_level": self.confidence_level,
            "block_length": self.block_length,
            "resamples": self.resamples,
            "seed": self.seed,
            "loss_semantics": self.loss_semantics,
            "statistic": self.statistic,
            "step_history": [dict(row) for row in self.step_history],
            "status": self.status,
        }


def _validate_losses(
    daily_losses: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not daily_losses:
        raise ValueError("daily_losses must contain at least one model")
    converted: dict[str, np.ndarray] = {}
    length: int | None = None
    for raw_name, raw_values in sorted(daily_losses.items()):
        name = str(raw_name)
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"daily losses for {name} must be nonempty and 1-D")
        if length is None:
            length = values.size
        elif values.size != length:
            raise ValueError("all MCS candidates must share the paired day panel")
        converted[name] = values
    stacked = np.vstack(list(converted.values()))
    common = np.all(np.isfinite(stacked), axis=0)
    if not np.all(common):
        missing = np.flatnonzero(~common)
        raise ValueError(
            "MCS requires the exact finite paired-day panel; "
            f"{missing.size} day(s) contain non-finite candidate losses"
        )
    return converted, common


def _seed_for_step(seed: int, members: Sequence[str]) -> int:
    key = f"MCS|{int(seed)}|{'|'.join(sorted(members))}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _standardize_pairwise(
    observed: np.ndarray,
    centered_bootstrap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Studentize pairwise means without hiding deterministic differences."""

    standard_error = np.std(centered_bootstrap, axis=0, ddof=1)
    observed_t = np.zeros_like(observed, dtype=float)
    bootstrap_t = np.zeros_like(centered_bootstrap, dtype=float)
    positive = standard_error > np.finfo(float).eps
    observed_t[positive] = observed[positive] / standard_error[positive]
    bootstrap_t[:, positive] = (
        centered_bootstrap[:, positive] / standard_error[positive]
    )
    deterministic_difference = (~positive) & (
        np.abs(observed) > np.finfo(float).eps
    )
    observed_t[deterministic_difference] = np.sign(
        observed[deterministic_difference]
    ) * np.inf
    return observed_t, bootstrap_t


def _range_step(
    losses: Mapping[str, np.ndarray],
    members: Sequence[str],
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the centered global equal-predictive-ability range null."""

    ordered = tuple(sorted(map(str, members)))
    matrix = np.column_stack([losses[model] for model in ordered])
    sample_count = matrix.shape[0]
    indices = moving_block_indices(
        sample_count,
        min(int(block_length), sample_count),
        resamples=int(resamples),
        seed=_seed_for_step(seed, ordered),
        circular=True,
    )
    means = np.mean(matrix, axis=0)
    bootstrap_means = np.mean(matrix[indices, :], axis=1)
    centered_means = bootstrap_means - means[None, :]
    observed_pairwise = means[:, None] - means[None, :]
    centered_pairwise = (
        centered_means[:, :, None] - centered_means[:, None, :]
    )
    observed_t, bootstrap_t = _standardize_pairwise(
        observed_pairwise,
        centered_pairwise,
    )
    off_diagonal = ~np.eye(len(ordered), dtype=bool)
    if len(ordered) == 1:
        observed_statistic = 0.0
        bootstrap_statistics = np.zeros(int(resamples), dtype=float)
    else:
        observed_statistic = float(np.max(np.abs(observed_t[off_diagonal])))
        bootstrap_statistics = np.max(
            np.abs(bootstrap_t[:, off_diagonal]), axis=1
        )
    p_value = float(
        (1 + np.count_nonzero(bootstrap_statistics >= observed_statistic))
        / (int(resamples) + 1)
    )
    # The range elimination rule removes the member with the largest
    # standardized loss relative to any surviving peer.  Mean loss and name
    # are deterministic tie breakers only for the elimination operation.
    worst_score = {
        model: float(np.max(observed_t[index, np.arange(len(ordered)) != index]))
        for index, model in enumerate(ordered)
    }
    worst = max(
        ordered,
        key=lambda model: (
            worst_score[model],
            float(means[ordered.index(model)]),
            model,
        ),
    )
    return {
        "members": list(ordered),
        "global_range_statistic": observed_statistic,
        "global_null_p_value": p_value,
        "elimination_candidate": worst,
        "elimination_scores": worst_score,
        "mean_losses": {
            model: float(means[index]) for index, model in enumerate(ordered)
        },
        "bootstrap_index_sha256": hashlib.sha256(
            np.ascontiguousarray(indices).tobytes()
        ).hexdigest(),
    }


def model_confidence_set(
    daily_losses: Mapping[str, Sequence[float] | np.ndarray],
    *,
    confidence_level: float = 0.95,
    block_length: int = 7,
    resamples: int = 2000,
    seed: int = 20260809,
    loss_semantics: str = "daily_realized_operating_cost_or_prescriptive_regret",
) -> MCSResult:
    """Return the standard centered range-statistic Model Confidence Set.

    At each step the equal-predictive-ability null is tested globally over all
    surviving pairwise loss differentials.  Moving-block bootstrap means are
    centered at the observed differentials before studentization.  A member is
    removed only after that global null is rejected; there are no uncorrected
    best-versus-worst pairwise tests.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if int(block_length) <= 0 or int(resamples) <= 0:
        raise ValueError("block_length and resamples must be positive")
    losses, _ = _validate_losses(daily_losses)
    means = {model: float(np.mean(values)) for model, values in losses.items()}
    current = sorted(losses)
    eliminated: list[str] = []
    p_values: dict[str, float] = {model: 1.0 for model in current}
    history: list[Mapping[str, Any]] = []
    alpha = 1.0 - confidence_level
    running_p_value = 0.0
    while len(current) > 1:
        step = _range_step(
            losses,
            current,
            block_length=int(block_length),
            resamples=int(resamples),
            seed=int(seed),
        )
        p_value = float(step["global_null_p_value"])
        running_p_value = max(running_p_value, p_value)
        rejected = p_value < alpha
        step = {
            **step,
            "centered_bootstrap": True,
            "global_null_rejected": rejected,
            "confidence_level": float(confidence_level),
        }
        history.append(step)
        if not rejected:
            break
        worst = str(step["elimination_candidate"])
        p_values[worst] = running_p_value
        current.remove(worst)
        eliminated.append(worst)
    included = tuple(sorted(current))
    excluded = tuple(eliminated)
    return MCSResult(
        included_models=included,
        excluded_models=excluded,
        elimination_order=tuple(eliminated),
        mean_losses=means,
        p_values=p_values,
        confidence_level=confidence_level,
        block_length=int(block_length),
        resamples=int(resamples),
        seed=int(seed),
        loss_semantics=str(loss_semantics),
        statistic="centered_studentized_global_loss_range",
        step_history=tuple(history),
        status="OK_RANGE_MCS",
    )


compute_model_confidence_set = model_confidence_set


__all__ = [
    "MCSResult",
    "compute_model_confidence_set",
    "model_confidence_set",
]
