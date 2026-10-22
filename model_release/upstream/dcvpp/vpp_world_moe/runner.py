"""Atomic runner for the independent VPP-WorldMoE training lineage."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    # Some Windows Python/OpenSSL builds intermittently fault when PyTorch and
    # pyarrow have already initialized their native runtimes.  CPython's
    # built-in SHA-256 implementation is byte-for-byte identical and avoids
    # that provider-lifetime failure.  Environments without the built-in
    # module retain the normal hashlib implementation.
    from _sha256 import sha256 as _sha256_factory
except ImportError:  # pragma: no cover - depends on the CPython build
    _sha256_factory = hashlib.sha256

import numpy as np
import torch
from torch import nn

from .c1_evaluator import C1DailyRecordStore, run_daily_closed_loop
from .config import (
    EXPERT_ASSIGNMENT_SCHEMA,
    R5_V23_SCHEMA_VERSION,
    R5_V24_SCHEMA_VERSION,
    R5_V25_SCHEMA_VERSION,
)
from .losses import ScoreTerm, aggregate_score_terms
from .gates import (
    GateResult,
    UnifiedGateResult,
    evaluate_closure_gate,
    evaluate_phase_release_gate,
    evaluate_planner_gate,
    evaluate_same_checkpoint_unified_gate,
    evaluate_stochastic_gate,
    evaluate_world_gate,
)
from .trainer import (
    CHECKPOINT_SCHEMA,
    PHASE_ORDER,
    AtomicCheckpointStore,
    PhaseSpec,
    ResumeState,
    WorldModelTrainer,
    _restore_rng_state,
    _rng_state,
    loss_weights_from_config,
)


RUN_STATE_SCHEMA = "dcvpp.vpp-world-moe.run-state.v1"
PHASE_RELEASE_SCHEMA = "dcvpp.vpp-world-moe.phase-release.v1"
SMOKE_SCHEMA = "dcvpp.vpp-world-moe.smoke.v1"
PARENT_CHECKPOINT_SCHEMA = "dcvpp.vpp-world-moe.parent-checkpoint.v1"
C1_HANDOFF_SCHEMA = "dcvpp.vpp-world-moe.c1-downstream-handoff.v1"
FAMILY_COVERAGE_SCHEMA = "dcvpp.vpp-world-moe.family-rank-coverage.v1"
MIXED_SUPPORT_COVERAGE_SCHEMA = (
    "dcvpp.vpp-world-moe.atom-aware-family-coverage.v2"
)
DISTRIBUTION_NATIVE_COVERAGE_SCHEMA = (
    "dcvpp.vpp-world-moe.distribution-native-mixed-coverage.v3"
)
UNCERTAINTY_FAMILIES = ("price", "workload", "pv", "pue")
COVERAGE_NOMINALS = (50, 80, 90)
GLOBAL_COVERAGE_ERROR_TOLERANCE = 0.02
FAMILY_COVERAGE_ERROR_TOLERANCE = 0.04
U0_FORMAL_SEED_OFFSETS = (0, 104_729, 209_759, 314_159, 419_431)
TAIL_ROUTE_BLOCKS = (1, 3, 5, 7, 9)
TAIL_ROUTE_AUPRC_BINS = 256
U0_PARETO_CHECKPOINT_CAP = 8
ALLOWED_PARENT_MODEL_UPGRADE_MISSING_KEYS = frozenset(
    {
        "dynamics.action_delta_projection.weight",
        "planner.dynamics.action_delta_projection.weight",
    }
)
R5_V24_SUPPORT_ATOM_STATE_PREFIXES = (
    "observation_queries.",
    "dynamics.observation_queries.",
    "planner.observation_queries.",
    "planner.dynamics.observation_queries.",
)
R5_V24_SUPPORT_ATOM_HEADS = (
    "workload_zero_query",
    "pv_atom_query",
    "pue_atom_query",
)
R5_V24_SUPPORT_ATOM_LEAVES = (
    "0.weight",
    "0.bias",
    "1.weight",
    "1.bias",
)
R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS = frozenset(
    f"{prefix}{head}.{leaf}"
    for prefix in R5_V24_SUPPORT_ATOM_STATE_PREFIXES
    for head in R5_V24_SUPPORT_ATOM_HEADS
    for leaf in R5_V24_SUPPORT_ATOM_LEAVES
)

DEFAULT_W0_STATE_THRESHOLDS = {
    "queue_occurrence_threshold_work_units": 1.0e-4,
    "queue_occurrence_probability_threshold": 0.50,
    "queue_positive_target_count_min": 1.0,
    "queue_paired_confidence": 0.95,
    "queue_occurrence_paired_upper_max": 0.0,
    "queue_severity_paired_upper_max": 0.0,
    "soc_paired_upper_max": 0.0,
}


class WorldRunnerError(RuntimeError):
    """Execution cannot preserve the registered lineage or data boundary."""


@dataclass(frozen=True)
class PhaseRunResult:
    phase: str
    status: str
    epochs_completed: int
    updates_completed: int
    best_validation: float
    best_checkpoint: str
    latest_checkpoint: str
    gate_path: str | None
    next_phase_released: bool
    locked_2025_opened: bool = False


@dataclass(frozen=True)
class ValidationSummary:
    """Exact global validation reduction plus phase-specific selection facts."""

    phase: str
    primary_score: float
    eligible: bool
    score_terms: Mapping[str, Mapping[str, float]]
    diagnostics: Mapping[str, Any]
    selection_metrics: Mapping[str, Any]
    batch_count: int


@dataclass(frozen=True)
class FormalSelectionDecision:
    selected: Mapping[str, Any] | None
    challenger_selected: bool
    candidate_signature: str
    relative_improvement: float


def _finite_probability(value: Any) -> float | None:
    """Return one finite probability without silently clipping bad evidence."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _legacy_family_rank_coverage_metrics(
    diagnostics: Mapping[str, Any],
    *,
    global_tolerance: float = GLOBAL_COVERAGE_ERROR_TOLERANCE,
    family_tolerance: float = FAMILY_COVERAGE_ERROR_TOLERANCE,
) -> dict[str, Any]:
    """Consume rank-corrected, family-specific coverage evidence fail-closed.

    The validation producer owns the finite-ensemble rank construction.  The
    runner deliberately consumes its explicit ``coverage_target_*`` values
    instead of replacing them with the nominal percentages.  Every family has
    one quarter of the gate mass, irrespective of its public-channel count.
    """

    targets: dict[str, float] = {}
    family_coverage: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    family_errors: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    targets_present = True
    fields_present = True
    formal_paths = float(diagnostics.get("formal_ensemble_paths", np.nan))
    formal_chunks = float(diagnostics.get("formal_ensemble_chunks", np.nan))
    formal_ensemble_ok = bool(formal_paths == 100.0 and formal_chunks == 5.0)
    for nominal in COVERAGE_NOMINALS:
        target = _finite_probability(diagnostics.get(f"coverage_target_{nominal}"))
        if target is None:
            targets_present = False
            continue
        targets[str(nominal)] = target
        for family in UNCERTAINTY_FAMILIES:
            observed = _finite_probability(
                diagnostics.get(f"coverage_{family}_{nominal}")
            )
            if observed is None:
                fields_present = False
                continue
            family_coverage[family][str(nominal)] = observed
            family_errors[family][str(nominal)] = abs(observed - target)

    complete = bool(
        targets_present
        and fields_present
        and len(targets) == len(COVERAGE_NOMINALS)
        and all(
            len(family_coverage[family]) == len(COVERAGE_NOMINALS)
            for family in UNCERTAINTY_FAMILIES
        )
    )
    daily_values: dict[str, dict[str, np.ndarray]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    daily_complete = True
    for nominal in COVERAGE_NOMINALS:
        for family in UNCERTAINTY_FAMILIES:
            raw_daily = diagnostics.get(f"_coverage_daily_{family}_{nominal}")
            try:
                values = np.asarray(raw_daily, dtype=np.float64)
            except (TypeError, ValueError):
                daily_complete = False
                continue
            if (
                values.shape != (364,)
                or not np.isfinite(values).all()
                or bool(((values < 0.0) | (values > 1.0)).any())
            ):
                daily_complete = False
                continue
            daily_values[family][str(nominal)] = values
    balanced: dict[str, float] = {}
    balanced_errors: dict[str, float] = {}
    if complete:
        balanced = {
            str(nominal): float(
                np.mean(
                    [
                        family_coverage[family][str(nominal)]
                        for family in UNCERTAINTY_FAMILIES
                    ]
                )
            )
            for nominal in COVERAGE_NOMINALS
        }
        balanced_errors = {
            str(nominal): abs(balanced[str(nominal)] - targets[str(nominal)])
            for nominal in COVERAGE_NOMINALS
        }
        worst_error = max(
            family_errors[family][str(nominal)]
            for family in UNCERTAINTY_FAMILIES
            for nominal in COVERAGE_NOMINALS
        )
        worst_balanced_error = max(balanced_errors.values())
        nested = bool(
            all(
                family_coverage[family]["50"]
                <= family_coverage[family]["80"]
                <= family_coverage[family]["90"]
                for family in UNCERTAINTY_FAMILIES
            )
            and balanced["50"] <= balanced["80"] <= balanced["90"]
        )
    else:
        worst_error = np.inf
        worst_balanced_error = np.inf
        nested = False

    def bootstrap_abs_error_ucb(
        values: np.ndarray, target: float, *, seed: int
    ) -> float:
        generator = np.random.default_rng(int(seed))
        indices = generator.integers(
            0, values.size, size=(2000, values.size), endpoint=False
        )
        errors = np.abs(values[indices].mean(axis=1) - float(target))
        return float(np.quantile(errors, 0.95, method="higher"))

    family_bootstrap_ucb: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    global_bootstrap_ucb: dict[str, float] = {}
    if complete and daily_complete:
        for nominal in COVERAGE_NOMINALS:
            key = str(nominal)
            for family_index, family in enumerate(UNCERTAINTY_FAMILIES):
                family_bootstrap_ucb[family][key] = bootstrap_abs_error_ucb(
                    daily_values[family][key],
                    targets[key],
                    seed=20240813 + 1000 * nominal + family_index,
                )
            global_daily = np.mean(
                np.stack(
                    [daily_values[family][key] for family in UNCERTAINTY_FAMILIES],
                    axis=0,
                ),
                axis=0,
            )
            global_bootstrap_ucb[key] = bootstrap_abs_error_ucb(
                global_daily,
                targets[key],
                seed=20240813 + 1000 * nominal + 99,
            )
    day_block_passed = bool(
        complete
        and daily_complete
        and max(global_bootstrap_ucb.values(), default=np.inf) <= 0.03
        and max(
            (
                value
                for family in UNCERTAINTY_FAMILIES
                for value in family_bootstrap_ucb[family].values()
            ),
            default=np.inf,
        )
        <= 0.05
    )
    passed = bool(
        complete
        and formal_ensemble_ok
        and nested
        and worst_error <= float(family_tolerance)
        and worst_balanced_error <= float(global_tolerance)
        and day_block_passed
    )
    return {
        "coverage_schema": FAMILY_COVERAGE_SCHEMA,
        "coverage_targets": targets,
        "rank_corrected_coverage_targets_present": bool(targets_present),
        "family_coverage_fields_present": bool(fields_present and complete),
        "formal_ensemble_paths": formal_paths,
        "formal_ensemble_chunks": formal_chunks,
        "formal_100_path_coverage_present": formal_ensemble_ok,
        "family_coverage": family_coverage,
        "family_coverage_errors": family_errors,
        "family_balanced_coverage": balanced,
        "family_balanced_coverage_errors": balanced_errors,
        "family_coverage_weights": {
            family: 1.0 / len(UNCERTAINTY_FAMILIES)
            for family in UNCERTAINTY_FAMILIES
        },
        "worst_family_coverage_error": float(worst_error),
        "worst_balanced_coverage_error": float(worst_balanced_error),
        "global_coverage_error_tolerance": float(global_tolerance),
        "family_coverage_error_tolerance": float(family_tolerance),
        "coverage_intervals_nested": nested,
        "coverage_validation_days": 364 if daily_complete else 0,
        "day_block_coverage_present": bool(daily_complete),
        "family_coverage_bootstrap_ucb95": family_bootstrap_ucb,
        "global_coverage_bootstrap_ucb95": global_bootstrap_ucb,
        "global_bootstrap_ucb_tolerance": 0.03,
        "family_bootstrap_ucb_tolerance": 0.05,
        "day_block_coverage_passed": day_block_passed,
        "family_coverage_passed": passed,
        "coverage_passed": passed,
    }


_ATOM_EVENTS: Mapping[str, tuple[str, ...]] = {
    "workload": ("zero", "positive"),
    "pv": ("zero", "capacity", "interior"),
    "pue": ("min", "max", "interior"),
}


def _finite_nonnegative(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result < 0.0:
        return None
    return result


def _validation_daily_array(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        result.shape != (364,)
        or not np.isfinite(result).all()
        or bool((result < 0.0).any())
    ):
        return None
    return result


def _circular_moving_block_indices(
    *,
    length: int,
    block_days: int,
    resamples: int,
    seed: int,
) -> np.ndarray:
    """Draw circular day blocks while preserving serial dependence."""

    generator = np.random.default_rng(int(seed))
    block_count = int(np.ceil(float(length) / float(block_days)))
    starts = generator.integers(
        0,
        int(length),
        size=(int(resamples), block_count, 1),
        endpoint=False,
    )
    offsets = np.arange(int(block_days), dtype=np.int64).reshape(1, 1, -1)
    return ((starts + offsets) % int(length)).reshape(int(resamples), -1)[
        :, : int(length)
    ]


def _bootstrap_ratio_error(
    numerator: np.ndarray,
    target: np.ndarray,
    count: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray | None:
    sampled_count = count[indices].sum(axis=1)
    if bool((sampled_count <= 0.0).any()):
        return None
    sampled_observed = numerator[indices].sum(axis=1) / sampled_count
    sampled_target = target[indices].sum(axis=1) / sampled_count
    return sampled_observed - sampled_target


def _mixed_support_family_coverage_metrics(
    diagnostics: Mapping[str, Any],
    *,
    atom_brier: Mapping[str, float] | None,
    global_tolerance: float,
    family_tolerance: float,
) -> dict[str, Any]:
    """Validate atoms and the conditional continuous component fail-closed."""

    support_present = diagnostics.get("atom_support_fields_present") == 1.0
    analytic_coverage = diagnostics.get("analytic_distribution_coverage") == 1.0
    formal_paths = float(diagnostics.get("formal_ensemble_paths", np.nan))
    formal_chunks = float(diagnostics.get("formal_ensemble_chunks", np.nan))
    formal_ensemble_ok = bool(formal_paths == 100.0 and formal_chunks == 5.0)

    family_coverage: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    family_targets: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    family_errors: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    lower_miss: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    upper_miss: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    interior_counts: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    daily_coverage: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {family: {} for family in UNCERTAINTY_FAMILIES}
    coverage_fields_present = bool(support_present)
    interior_support_present = True
    daily_coverage_present = True
    for family in UNCERTAINTY_FAMILIES:
        for nominal in COVERAGE_NOMINALS:
            key = str(nominal)
            prefix = f"coverage_{family}_{nominal}"
            hit_sum = _finite_nonnegative(diagnostics.get(f"{prefix}_hit_sum"))
            target_sum = _finite_nonnegative(
                diagnostics.get(f"{prefix}_target_sum")
            )
            valid_count = _finite_nonnegative(
                diagnostics.get(f"{prefix}_valid_count")
            )
            insufficient_count = _finite_nonnegative(
                diagnostics.get(f"{prefix}_insufficient_count")
            )
            lower_sum = _finite_nonnegative(
                diagnostics.get(f"{prefix}_lower_miss_sum")
            )
            upper_sum = _finite_nonnegative(
                diagnostics.get(f"{prefix}_upper_miss_sum")
            )
            values = (
                hit_sum,
                target_sum,
                valid_count,
                insufficient_count,
                lower_sum,
                upper_sum,
            )
            if any(value is None for value in values) or not valid_count:
                coverage_fields_present = False
                interior_support_present = False
            else:
                assert hit_sum is not None
                assert target_sum is not None
                assert insufficient_count is not None
                assert lower_sum is not None
                assert upper_sum is not None
                tolerance = 2.0e-5 * max(1.0, valid_count)
                if (
                    hit_sum > valid_count + tolerance
                    or target_sum > valid_count + tolerance
                    or abs(hit_sum + lower_sum + upper_sum - valid_count)
                    > tolerance
                ):
                    coverage_fields_present = False
                observed = hit_sum / valid_count
                target = target_sum / valid_count
                if not (0.0 <= observed <= 1.0 and 0.0 <= target <= 1.0):
                    coverage_fields_present = False
                family_coverage[family][key] = observed
                family_targets[family][key] = target
                family_errors[family][key] = abs(observed - target)
                lower_miss[family][key] = lower_sum / valid_count
                upper_miss[family][key] = upper_sum / valid_count
                interior_counts[family][key] = valid_count
                if insufficient_count > 0.0:
                    interior_support_present = False

            daily_prefix = f"_coverage_daily_{family}_{nominal}"
            daily_hit = _validation_daily_array(
                diagnostics.get(f"{daily_prefix}_hit_sum")
            )
            daily_target = _validation_daily_array(
                diagnostics.get(f"{daily_prefix}_target_sum")
            )
            daily_count = _validation_daily_array(
                diagnostics.get(f"{daily_prefix}_valid_count")
            )
            if daily_hit is None or daily_target is None or daily_count is None:
                daily_coverage_present = False
            else:
                daily_coverage[family][key] = (
                    daily_hit,
                    daily_target,
                    daily_count,
                )
                if (
                    hit_sum is None
                    or target_sum is None
                    or valid_count is None
                    or not np.isclose(daily_hit.sum(), hit_sum, rtol=1.0e-6, atol=1.0e-5)
                    or not np.isclose(
                        daily_target.sum(), target_sum, rtol=1.0e-6, atol=1.0e-5
                    )
                    or not np.isclose(
                        daily_count.sum(), valid_count, rtol=1.0e-6, atol=1.0e-5
                    )
                ):
                    daily_coverage_present = False

    coverage_complete = bool(
        coverage_fields_present
        and interior_support_present
        and all(
            len(family_coverage[family]) == len(COVERAGE_NOMINALS)
            for family in UNCERTAINTY_FAMILIES
        )
    )
    balanced_coverage: dict[str, float] = {}
    balanced_targets: dict[str, float] = {}
    balanced_errors: dict[str, float] = {}
    if coverage_complete:
        for nominal in COVERAGE_NOMINALS:
            key = str(nominal)
            balanced_coverage[key] = float(
                np.mean(
                    [family_coverage[family][key] for family in UNCERTAINTY_FAMILIES]
                )
            )
            balanced_targets[key] = float(
                np.mean(
                    [family_targets[family][key] for family in UNCERTAINTY_FAMILIES]
                )
            )
            balanced_errors[key] = abs(
                balanced_coverage[key] - balanced_targets[key]
            )
        worst_family_error = max(
            family_errors[family][str(nominal)]
            for family in UNCERTAINTY_FAMILIES
            for nominal in COVERAGE_NOMINALS
        )
        worst_balanced_error = max(balanced_errors.values())
        nested = bool(
            all(
                family_coverage[family]["50"]
                <= family_coverage[family]["80"]
                <= family_coverage[family]["90"]
                for family in UNCERTAINTY_FAMILIES
            )
            and balanced_coverage["50"]
            <= balanced_coverage["80"]
            <= balanced_coverage["90"]
        )
    else:
        worst_family_error = np.inf
        worst_balanced_error = np.inf
        nested = False

    atom_forecasts: dict[str, dict[str, float]] = {
        family: {} for family in _ATOM_EVENTS
    }
    atom_prevalence: dict[str, dict[str, float]] = {
        family: {} for family in _ATOM_EVENTS
    }
    atom_reliability: dict[str, dict[str, float]] = {
        family: {} for family in _ATOM_EVENTS
    }
    daily_atoms: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {family: {} for family in _ATOM_EVENTS}
    atom_fields_present = bool(support_present)
    atom_daily_present = True
    for family, events in _ATOM_EVENTS.items():
        for event in events:
            prefix = f"atom_{family}_{event}"
            forecast_sum = _finite_nonnegative(
                diagnostics.get(f"{prefix}_forecast_sum")
            )
            observed_sum = _finite_nonnegative(
                diagnostics.get(f"{prefix}_observed_sum")
            )
            valid_count = _finite_nonnegative(
                diagnostics.get(f"{prefix}_valid_count")
            )
            if (
                forecast_sum is None
                or observed_sum is None
                or not valid_count
                or forecast_sum > valid_count + 1.0e-5
                or observed_sum > valid_count + 1.0e-5
            ):
                atom_fields_present = False
            else:
                atom_forecasts[family][event] = forecast_sum / valid_count
                atom_prevalence[family][event] = observed_sum / valid_count
                atom_reliability[family][event] = abs(
                    forecast_sum - observed_sum
                ) / valid_count

            daily_forecast = _validation_daily_array(
                diagnostics.get(f"_atom_daily_{family}_{event}_forecast_sum")
            )
            daily_observed = _validation_daily_array(
                diagnostics.get(f"_atom_daily_{family}_{event}_observed_sum")
            )
            daily_count = _validation_daily_array(
                diagnostics.get(f"_atom_daily_{family}_{event}_valid_count")
            )
            if (
                daily_forecast is None
                or daily_observed is None
                or daily_count is None
                or bool((daily_forecast > daily_count + 1.0e-5).any())
                or bool((daily_observed > daily_count + 1.0e-5).any())
            ):
                atom_daily_present = False
            else:
                daily_atoms[family][event] = (
                    daily_forecast,
                    daily_observed,
                    daily_count,
                )
                if (
                    forecast_sum is None
                    or observed_sum is None
                    or valid_count is None
                    or not np.isclose(
                        daily_forecast.sum(), forecast_sum, rtol=1.0e-6, atol=1.0e-5
                    )
                    or not np.isclose(
                        daily_observed.sum(), observed_sum, rtol=1.0e-6, atol=1.0e-5
                    )
                    or not np.isclose(
                        daily_count.sum(), valid_count, rtol=1.0e-6, atol=1.0e-5
                    )
                ):
                    atom_daily_present = False

    atom_brier_values: dict[str, float] = {}
    if atom_brier is not None:
        for family in _ATOM_EVENTS:
            value = _finite_nonnegative(atom_brier.get(family))
            if value is not None and value <= 1.0:
                atom_brier_values[family] = value
    atom_brier_present = len(atom_brier_values) == len(_ATOM_EVENTS)
    atom_complete = bool(
        atom_fields_present
        and atom_brier_present
        and all(
            len(atom_reliability[family]) == len(events)
            for family, events in _ATOM_EVENTS.items()
        )
    )
    if atom_complete:
        atom_family_errors = {
            family: float(np.mean(tuple(atom_reliability[family].values())))
            for family in _ATOM_EVENTS
        }
        worst_atom_error = max(
            error
            for family in _ATOM_EVENTS
            for error in atom_reliability[family].values()
        )
        balanced_atom_error = float(np.mean(tuple(atom_family_errors.values())))
        atom_point_passed = bool(
            worst_atom_error <= float(family_tolerance)
            and balanced_atom_error <= float(global_tolerance)
        )
        balanced_atom_brier = float(np.mean(tuple(atom_brier_values.values())))
    else:
        atom_family_errors = {}
        worst_atom_error = np.inf
        balanced_atom_error = np.inf
        atom_point_passed = False
        balanced_atom_brier = np.inf

    moving_block_days = 14
    bootstrap_resamples = 2000
    family_bootstrap_ucb: dict[str, dict[str, float]] = {
        family: {} for family in UNCERTAINTY_FAMILIES
    }
    global_bootstrap_ucb: dict[str, float] = {}
    atom_bootstrap_ucb: dict[str, dict[str, float]] = {
        family: {} for family in _ATOM_EVENTS
    }
    global_atom_bootstrap_ucb = np.inf
    coverage_bootstrap_ok = bool(coverage_complete and daily_coverage_present)
    if coverage_bootstrap_ok:
        for nominal in COVERAGE_NOMINALS:
            key = str(nominal)
            indices = _circular_moving_block_indices(
                length=364,
                block_days=moving_block_days,
                resamples=bootstrap_resamples,
                seed=20240813 + 1000 * nominal,
            )
            signed_errors: list[np.ndarray] = []
            for family in UNCERTAINTY_FAMILIES:
                values = daily_coverage[family].get(key)
                if values is None:
                    coverage_bootstrap_ok = False
                    break
                error = _bootstrap_ratio_error(*values, indices)
                if error is None:
                    coverage_bootstrap_ok = False
                    break
                family_bootstrap_ucb[family][key] = float(
                    np.quantile(np.abs(error), 0.95, method="higher")
                )
                signed_errors.append(error)
            if not coverage_bootstrap_ok:
                break
            global_error = np.abs(np.mean(np.stack(signed_errors), axis=0))
            global_bootstrap_ucb[key] = float(
                np.quantile(global_error, 0.95, method="higher")
            )

    atom_bootstrap_ok = bool(atom_complete and atom_daily_present)
    if atom_bootstrap_ok:
        indices = _circular_moving_block_indices(
            length=364,
            block_days=moving_block_days,
            resamples=bootstrap_resamples,
            seed=20240813 + 777_777,
        )
        family_bootstrap_errors: list[np.ndarray] = []
        for family, events in _ATOM_EVENTS.items():
            event_errors: list[np.ndarray] = []
            for event in events:
                values = daily_atoms[family].get(event)
                if values is None:
                    atom_bootstrap_ok = False
                    break
                error = _bootstrap_ratio_error(*values, indices)
                if error is None:
                    atom_bootstrap_ok = False
                    break
                absolute = np.abs(error)
                atom_bootstrap_ucb[family][event] = float(
                    np.quantile(absolute, 0.95, method="higher")
                )
                event_errors.append(absolute)
            if not atom_bootstrap_ok:
                break
            family_bootstrap_errors.append(np.mean(np.stack(event_errors), axis=0))
        if atom_bootstrap_ok:
            global_atom_bootstrap_ucb = float(
                np.quantile(
                    np.mean(np.stack(family_bootstrap_errors), axis=0),
                    0.95,
                    method="higher",
                )
            )

    coverage_block_passed = bool(
        coverage_bootstrap_ok
        and max(global_bootstrap_ucb.values(), default=np.inf) <= 0.03
        and max(
            (
                value
                for family in UNCERTAINTY_FAMILIES
                for value in family_bootstrap_ucb[family].values()
            ),
            default=np.inf,
        )
        <= 0.05
    )
    atom_block_passed = bool(
        atom_bootstrap_ok
        and global_atom_bootstrap_ucb <= 0.03
        and max(
            (
                value
                for family in _ATOM_EVENTS
                for value in atom_bootstrap_ucb[family].values()
            ),
            default=np.inf,
        )
        <= 0.05
    )
    coverage_point_passed = bool(
        coverage_complete
        and nested
        and worst_family_error <= float(family_tolerance)
        and worst_balanced_error <= float(global_tolerance)
    )
    passed = bool(
        support_present
        and formal_ensemble_ok
        and coverage_point_passed
        and atom_point_passed
        and coverage_block_passed
        and atom_block_passed
    )
    return {
        "coverage_schema": (
            DISTRIBUTION_NATIVE_COVERAGE_SCHEMA
            if analytic_coverage
            else MIXED_SUPPORT_COVERAGE_SCHEMA
        ),
        "price_rank_method": (
            "analytic_monotone_conditional_quantiles"
            if analytic_coverage
            else "continuous_100_path_order_statistics"
        ),
        "atom_occurrence_calibration": True,
        "interior_conditional_coverage": True,
        "moving_block_days": moving_block_days,
        "bootstrap_method": "14_day_circular_moving_block",
        "bootstrap_resamples": bootstrap_resamples,
        "atom_support_fields_present": bool(support_present),
        "formal_ensemble_paths": formal_paths,
        "formal_ensemble_chunks": formal_chunks,
        "formal_100_path_coverage_present": formal_ensemble_ok,
        "rank_corrected_coverage_targets_present": bool(
            coverage_fields_present and not analytic_coverage
        ),
        "analytic_coverage_targets_present": bool(
            coverage_fields_present and analytic_coverage
        ),
        "family_coverage_fields_present": bool(coverage_complete),
        "interior_support_present": bool(interior_support_present),
        "family_coverage": family_coverage,
        "family_coverage_targets": family_targets,
        "family_coverage_errors": family_errors,
        "family_coverage_lower_miss": lower_miss,
        "family_coverage_upper_miss": upper_miss,
        "family_interior_valid_counts": interior_counts,
        "family_balanced_coverage": balanced_coverage,
        "family_balanced_coverage_targets": balanced_targets,
        "family_balanced_coverage_errors": balanced_errors,
        "family_coverage_weights": {
            family: 1.0 / len(UNCERTAINTY_FAMILIES)
            for family in UNCERTAINTY_FAMILIES
        },
        "worst_family_coverage_error": float(worst_family_error),
        "worst_balanced_coverage_error": float(worst_balanced_error),
        "global_coverage_error_tolerance": float(global_tolerance),
        "family_coverage_error_tolerance": float(family_tolerance),
        "coverage_intervals_nested": nested,
        "coverage_validation_days": 364 if daily_coverage_present else 0,
        "atom_forecast_probabilities": atom_forecasts,
        "atom_observed_prevalence": atom_prevalence,
        "atom_reliability_errors": atom_reliability,
        "atom_family_reliability_errors": atom_family_errors,
        "worst_atom_reliability_error": float(worst_atom_error),
        "family_balanced_atom_reliability_error": float(balanced_atom_error),
        "atom_brier": atom_brier_values,
        "family_balanced_atom_brier": float(balanced_atom_brier),
        "atom_brier_fields_present": bool(atom_brier_present),
        "atom_calibration_passed": bool(atom_point_passed),
        "day_block_coverage_present": bool(coverage_bootstrap_ok),
        "family_coverage_bootstrap_ucb95": family_bootstrap_ucb,
        "global_coverage_bootstrap_ucb95": global_bootstrap_ucb,
        "atom_reliability_bootstrap_ucb95": atom_bootstrap_ucb,
        "global_atom_reliability_bootstrap_ucb95": float(
            global_atom_bootstrap_ucb
        ),
        "global_bootstrap_ucb_tolerance": 0.03,
        "family_bootstrap_ucb_tolerance": 0.05,
        "coverage_moving_block_passed": coverage_block_passed,
        "atom_moving_block_passed": atom_block_passed,
        "day_block_coverage_passed": bool(
            coverage_block_passed and atom_block_passed
        ),
        "family_coverage_passed": passed,
        "coverage_passed": passed,
    }


def _family_rank_coverage_metrics(
    diagnostics: Mapping[str, Any],
    *,
    atom_brier: Mapping[str, float] | None = None,
    global_tolerance: float = GLOBAL_COVERAGE_ERROR_TOLERANCE,
    family_tolerance: float = FAMILY_COVERAGE_ERROR_TOLERANCE,
) -> dict[str, Any]:
    """Route old lineages and atom-aware R5-V2.4 evidence explicitly."""

    if "atom_support_fields_present" not in diagnostics:
        return _legacy_family_rank_coverage_metrics(
            diagnostics,
            global_tolerance=global_tolerance,
            family_tolerance=family_tolerance,
        )
    return _mixed_support_family_coverage_metrics(
        diagnostics,
        atom_brier=atom_brier,
        global_tolerance=global_tolerance,
        family_tolerance=family_tolerance,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl_append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        existing + json.dumps(dict(payload), sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl_append_unique(
    path: Path,
    payload: Mapping[str, Any],
    *,
    identity_fields: Sequence[str],
) -> None:
    """Append one evidence record exactly once across interrupted resumes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_text = path.read_text(encoding="utf-8") if path.is_file() else ""
    identity = tuple(payload.get(name) for name in identity_fields)
    for line in existing_text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorldRunnerError(f"append-only manifest is malformed: {path}") from exc
        if not isinstance(record, Mapping):
            raise WorldRunnerError(f"append-only manifest is malformed: {path}")
        if tuple(record.get(name) for name in identity_fields) != identity:
            continue
        old_core = {key: value for key, value in record.items() if key != "recorded_at_utc"}
        new_core = {
            key: value for key, value in payload.items() if key != "recorded_at_utc"
        }
        if old_core != new_core:
            raise WorldRunnerError(
                f"append-only manifest identity collision for {identity}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        existing_text + json.dumps(dict(payload), sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _call_supported(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(callback)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = (
        kwargs
        if accepts_kwargs
        else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
    )
    return callback(*args, **selected)


def _configured_weight_kwargs(
    config: Mapping[str, Any], *, synthetic_loss_builder: bool
) -> dict[str, Any]:
    """Require registered weights for real runs while preserving tiny test losses."""

    training = config.get("training")
    objectives = training.get("phase_objectives") if isinstance(training, Mapping) else None
    if isinstance(objectives, Mapping):
        return {"weights": loss_weights_from_config(config)}
    if synthetic_loss_builder:
        return {}
    raise WorldRunnerError(
        "formal WorldMoE execution requires training.phase_objectives weights"
    )


def _u0_failure_plateau_policy(
    config: Mapping[str, Any],
) -> tuple[bool, int, float, str]:
    """Read the registered terminal U0 failure policy; old lineages opt out."""

    if config.get("schema_version") not in {
        R5_V24_SCHEMA_VERSION,
        R5_V25_SCHEMA_VERSION,
    }:
        return False, 0, 0.0, ""
    training = config.get("training")
    selection = (
        training.get("checkpoint_selection")
        if isinstance(training, Mapping)
        else None
    )
    early = selection.get("early_stop") if isinstance(selection, Mapping) else None
    failure = early.get("failure_plateau") if isinstance(early, Mapping) else None
    if not isinstance(failure, Mapping):
        raise WorldRunnerError("R5-V2.4+ requires checkpoint failure_plateau policy")
    evaluations = int(failure.get("formal_evaluations", 0))
    threshold = float(failure.get("relative_improvement_below", np.nan))
    terminal = str(failure.get("terminal_status", ""))
    if (
        evaluations < 1
        or not np.isfinite(threshold)
        or not 0.0 < threshold < 1.0
        or terminal != "FAIL_NO_RELEASE"
    ):
        raise WorldRunnerError("R5-V2.4+ failure_plateau policy is invalid")
    return True, evaluations, threshold, terminal


def _u0_failure_gate_gap(summary: ValidationSummary) -> float:
    """Return a finite normalized distance from all U0 release requirements."""

    metrics = summary.selection_metrics
    gap = 0.0
    required_flags = (
        "dart_identity_passed",
        "formal_100_path_coverage_present",
        "atom_support_fields_present",
        "atom_brier_fields_present",
        "family_coverage_fields_present",
        "interior_support_present",
        "coverage_intervals_nested",
        "coverage_moving_block_passed",
        "atom_calibration_passed",
        "atom_moving_block_passed",
        "tail_routing_passed",
        "duplicate_passed",
        "diversity_passed",
    )
    gap += float(sum(metrics.get(name) is not True for name in required_flags))

    def excess(name: str, limit: float) -> float:
        value = float(metrics.get(name, np.inf))
        if not np.isfinite(value):
            return 10.0
        return max(0.0, value / float(limit) - 1.0)

    gap += excess("worst_family_coverage_error", 0.04)
    gap += excess("worst_balanced_coverage_error", 0.02)
    gap += excess("worst_atom_reliability_error", 0.04)
    gap += excess("family_balanced_atom_reliability_error", 0.02)
    global_coverage = metrics.get("global_coverage_bootstrap_ucb95")
    if isinstance(global_coverage, Mapping) and global_coverage:
        gap += max(
            excess_value / 0.03 - 1.0
            for excess_value in (
                max(0.03, float(value)) for value in global_coverage.values()
            )
        )
    else:
        gap += 10.0
    gap += excess("global_atom_reliability_bootstrap_ucb95", 0.03)

    minimum_separation = float(
        metrics.get("tail_route_min_block_separation", -np.inf)
    )
    minimum_lift = float(
        metrics.get("tail_route_min_block_hit_utilization_lift", -np.inf)
    )
    minimum_aucpr_margin = float(metrics.get("tail_route_min_aucpr_margin", -np.inf))
    gap += (
        10.0
        if not np.isfinite(minimum_separation)
        else max(0.0, -minimum_separation)
    )
    gap += 10.0 if not np.isfinite(minimum_lift) else max(0.0, 1.0 - minimum_lift)
    gap += (
        10.0
        if not np.isfinite(minimum_aucpr_margin)
        else max(0.0, -minimum_aucpr_margin)
    )
    if not np.isfinite(gap) or gap < 0.0:
        raise WorldRunnerError("U0 failure gate gap is invalid")
    return float(gap)


def _update_u0_failure_plateau(
    state: ResumeState,
    summary: ValidationSummary,
    *,
    relative_improvement: float,
) -> tuple[int, float, float, float, bool]:
    """Advance the resumable ineligible-stream plateau monitor."""

    gate_gap = _u0_failure_gate_gap(summary)
    proper_score = float(summary.primary_score)
    if not np.isfinite(proper_score) or proper_score < 0.0:
        raise WorldRunnerError("U0 failure proper score must be finite and nonnegative")

    def gain(best: float, current: float) -> float:
        if not np.isfinite(best):
            return np.inf
        return (best - current) / max(abs(best), 1.0e-12)

    gate_gain = gain(float(state.failure_best_gate_gap), gate_gap)
    proper_gain = gain(float(state.failure_best_proper_score), proper_score)
    improved = bool(
        gate_gain >= float(relative_improvement)
        or proper_gain >= float(relative_improvement)
    )
    stable = 0 if improved else state.failure_stable_formal_evaluations + 1
    return (
        stable,
        min(float(state.failure_best_gate_gap), gate_gap),
        min(float(state.failure_best_proper_score), proper_score),
        gate_gap,
        improved,
    )


def _u0_failure_plateau_should_stop(
    state: ResumeState,
    *,
    enabled: bool,
    required_evaluations: int,
    minimum_epoch: int,
) -> bool:
    """Stop only on a recoverable full-epoch checkpoint with no release model."""

    return bool(
        enabled
        and state.phase == "u0"
        and not state.selected_record_json
        and state.failure_stable_formal_evaluations >= int(required_evaluations)
        and state.epoch >= int(minimum_epoch)
        and state.epoch % state.short_epoch_factor == 0
    )


def _first_mapping(value: Any, *names: str) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    for name in names:
        candidate = value.get(name)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _w0_state_batch_diagnostics(
    output: Any,
    batch: Any,
    *,
    occurrence_threshold: float,
    occurrence_probability_threshold: float,
) -> dict[str, float]:
    """Measure W0 state occurrence/severity in raw physical target space."""

    predictions = _first_mapping(
        output, "state_predictions", "state_reconstructions", "reconstructions"
    )
    targets = _first_mapping(
        batch, "state_targets", "reconstruction_targets", "state_reconstruction_targets"
    )
    masks = _first_mapping(
        batch, "state_masks", "reconstruction_masks", "state_reconstruction_masks"
    )
    if predictions is None or targets is None:
        raise WorldRunnerError("formal W0 validation omitted state predictions/targets")
    for name in ("queue", "soc"):
        if not isinstance(predictions.get(name), torch.Tensor) or not isinstance(
            targets.get(name), torch.Tensor
        ):
            raise WorldRunnerError(f"formal W0 validation omitted {name} tensors")

    queue_prediction = predictions["queue"].detach().float()
    queue_target = targets["queue"].detach().float()
    soc_prediction = predictions["soc"].detach().float()
    soc_target = targets["soc"].detach().float()
    if (
        queue_prediction.shape != queue_target.shape
        or soc_prediction.shape != soc_target.shape
    ):
        raise WorldRunnerError("formal W0 state prediction/target shapes differ")
    if bool((queue_prediction < 0).any()) or bool((queue_target < 0).any()):
        raise WorldRunnerError("formal W0 queue values must be nonnegative")

    queue_mask = torch.isfinite(queue_prediction) & torch.isfinite(queue_target)
    soc_mask = torch.isfinite(soc_prediction) & torch.isfinite(soc_target)
    if masks is not None:
        raw_queue_mask = masks.get("queue")
        raw_soc_mask = masks.get("soc")
        if isinstance(raw_queue_mask, torch.Tensor):
            queue_mask &= raw_queue_mask.to(device=queue_mask.device, dtype=torch.bool)
        if isinstance(raw_soc_mask, torch.Tensor):
            soc_mask &= raw_soc_mask.to(device=soc_mask.device, dtype=torch.bool)
    if not bool(queue_mask.any()) or not bool(soc_mask.any()):
        raise WorldRunnerError("formal W0 state masks contain no valid values")

    queue_distribution = _first_mapping(output, "queue_distribution")
    if queue_distribution is None:
        raise WorldRunnerError("formal W0 validation omitted queue_distribution")
    probability = queue_distribution.get("occurrence_probability")
    if not isinstance(probability, torch.Tensor):
        logits = queue_distribution.get("occurrence_logits")
        if not isinstance(logits, torch.Tensor):
            raise WorldRunnerError(
                "formal W0 queue distribution omitted occurrence probability/logits"
            )
        probability = torch.sigmoid(logits.detach().float())
    else:
        probability = probability.detach().float()
    severity_prediction = queue_distribution.get("positive_severity")
    if not isinstance(severity_prediction, torch.Tensor):
        raise WorldRunnerError(
            "formal W0 queue distribution omitted conditional positive severity"
        )
    severity_prediction = severity_prediction.detach().float()
    if (
        probability.shape != queue_target.shape
        or severity_prediction.shape != queue_target.shape
    ):
        raise WorldRunnerError("formal W0 queue distribution shapes differ from target")
    if bool((severity_prediction < 0).any()) or not bool(
        torch.isfinite(probability).all() & torch.isfinite(severity_prediction).all()
    ):
        raise WorldRunnerError("formal W0 queue distribution is invalid")

    threshold = float(occurrence_threshold)
    true_occurrence = (queue_target > threshold) & queue_mask
    predicted_occurrence = (
        probability >= float(occurrence_probability_threshold)
    ) & queue_mask
    negative_target = (~true_occurrence) & queue_mask
    true_positive = true_occurrence & predicted_occurrence
    false_positive = negative_target & predicted_occurrence
    positive_log_error = (
        torch.log1p(severity_prediction) - torch.log1p(queue_target)
    ).abs()
    soc_error = (soc_prediction - soc_target).abs()
    return {
        "queue_positive_target_count": float(true_occurrence.sum().cpu()),
        "queue_negative_target_count": float(negative_target.sum().cpu()),
        "queue_predicted_positive_count": float(predicted_occurrence.sum().cpu()),
        "queue_true_positive_count": float(true_positive.sum().cpu()),
        "queue_false_positive_count": float(false_positive.sum().cpu()),
        "queue_positive_log1p_error_sum": float(
            positive_log_error[true_occurrence].sum().cpu()
        ),
        "soc_abs_error_sum": float(soc_error[soc_mask].sum().cpu()),
        "soc_valid_count": float(soc_mask.sum().cpu()),
        "soc_max_abs": float(soc_error[soc_mask].amax().cpu()),
    }


def _finalize_w0_state_diagnostics(values: Mapping[str, float]) -> dict[str, float]:
    result = dict(values)
    positive = float(result.get("queue_positive_target_count", 0.0))
    negative = float(result.get("queue_negative_target_count", 0.0))
    predicted = float(result.get("queue_predicted_positive_count", 0.0))
    soc_count = float(result.get("soc_valid_count", 0.0))
    result["queue_occurrence_recall"] = (
        float(result.get("queue_true_positive_count", 0.0)) / positive
        if positive > 0.0
        else 0.0
    )
    result["queue_occurrence_precision"] = (
        float(result.get("queue_true_positive_count", 0.0)) / predicted
        if predicted > 0.0
        else 0.0
    )
    result["queue_false_positive_rate"] = (
        float(result.get("queue_false_positive_count", 0.0)) / negative
        if negative > 0.0
        else 1.0
    )
    result["queue_positive_log1p_mae"] = (
        float(result.get("queue_positive_log1p_error_sum", 0.0)) / positive
        if positive > 0.0
        else 1.0e30
    )
    result["soc_mae"] = (
        float(result.get("soc_abs_error_sum", 0.0)) / soc_count
        if soc_count > 0.0
        else 1.0e30
    )
    return result


def _tail_time_target(
    batch: Any, *, occurrence_threshold: float
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return aggregate queue-event truth for route diagnostics only.

    The teacher event is deliberately built outside the model forward. It may
    score the causal router, but it can never activate the conditional expert.
    """

    if isinstance(batch, Mapping):
        direct_target = batch.get("queue_risk_event_target")
        direct_mask = batch.get("queue_risk_event_mask")
        if isinstance(direct_target, torch.Tensor) and isinstance(
            direct_mask, torch.Tensor
        ):
            target = direct_target.detach().to(dtype=torch.bool)
            valid = direct_mask.detach().to(device=target.device, dtype=torch.bool)
            if target.ndim >= 2 and target.shape == valid.shape:
                return (
                    target.reshape(target.shape[0], target.shape[1], -1).any(dim=-1),
                    valid.reshape(valid.shape[0], valid.shape[1], -1).any(dim=-1),
                )

    targets = _first_mapping(
        batch, "state_targets", "reconstruction_targets", "state_reconstruction_targets"
    )
    masks = _first_mapping(
        batch, "state_masks", "reconstruction_masks", "state_reconstruction_masks"
    )
    if targets is None or not isinstance(targets.get("queue"), torch.Tensor):
        return None
    queue = targets["queue"].detach().float()
    valid = torch.isfinite(queue)
    if masks is not None and isinstance(masks.get("queue"), torch.Tensor):
        valid &= masks["queue"].to(device=queue.device, dtype=torch.bool)
    if queue.ndim < 2:
        return None
    queue = queue.reshape(queue.shape[0], queue.shape[1], -1)
    valid = valid.reshape(valid.shape[0], valid.shape[1], -1)
    time_valid = valid.any(dim=-1)
    event = ((queue > float(occurrence_threshold)) & valid).any(dim=-1)
    return event, time_valid


def _tail_time_tensor(value: Any, *, boolean: bool = False) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        return None
    tensor = value.detach()
    if boolean:
        tensor = tensor.to(dtype=torch.bool)
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1).any(dim=-1)
    tensor = tensor.float()
    # A model may expose either an already aggregated [B,T,1] probability or
    # slot-level [B,T,S] values. Amax answers whether any risk-relevant slot at
    # that time is requesting the tail residual and avoids dilution by 40 slots.
    return tensor.reshape(tensor.shape[0], tensor.shape[1], -1).amax(dim=-1)


def _tail_routing_batch_diagnostics(
    output: Any,
    batch: Any,
    *,
    occurrence_threshold: float,
) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        return {}
    raw_blocks = output.get("moe_tail_diagnostics")
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)):
        return {}
    truth = _tail_time_target(batch, occurrence_threshold=occurrence_threshold)
    if truth is None:
        return {}
    event, valid = truth
    result: dict[str, Any] = {}
    for position, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping):
            raise WorldRunnerError("moe_tail_diagnostics entries must be mappings")
        block_index = int(raw.get("block_index", position))
        prefix = f"tail_block_{block_index}"
        probability = _tail_time_tensor(raw.get("tail_route_probabilities"))
        active = _tail_time_tensor(raw.get("tail_active_mask"), boolean=True)
        if probability is None or active is None:
            raise WorldRunnerError(
                f"{prefix} omitted token-level probability or causal active mask"
            )
        if probability.shape != event.shape or active.shape != event.shape:
            raise WorldRunnerError(
                f"{prefix} route diagnostics differ from queue time axis"
            )
        finite = valid & torch.isfinite(probability)
        positive = finite & event
        negative = finite & (~event)
        probability_bin = torch.clamp(
            (probability.clamp(0.0, 1.0) * TAIL_ROUTE_AUPRC_BINS).long(),
            max=TAIL_ROUTE_AUPRC_BINS - 1,
        )
        positive_histogram = torch.bincount(
            probability_bin[positive], minlength=TAIL_ROUTE_AUPRC_BINS
        )
        negative_histogram = torch.bincount(
            probability_bin[negative], minlength=TAIL_ROUTE_AUPRC_BINS
        )
        selected = raw.get("tail_selected_count", float(active[finite].sum().cpu()))
        capacity = raw.get("tail_capacity", float(finite.sum().cpu()))
        selected_value = float(torch.as_tensor(selected).detach().float().sum().cpu())
        capacity_value = float(torch.as_tensor(capacity).detach().float().sum().cpu())
        result.update(
            {
                f"{prefix}_predicted_activation_sum": float(active[finite].sum().cpu()),
                f"{prefix}_valid_time_count": float(finite.sum().cpu()),
                f"{prefix}_tail_selected_count": selected_value,
                f"{prefix}_tail_capacity_count": capacity_value,
                f"{prefix}_teacher_positive_hit_count": float(
                    (active & positive).sum().cpu()
                ),
                f"{prefix}_teacher_positive_count": float(positive.sum().cpu()),
                f"{prefix}_teacher_negative_activation_count": float(
                    (active & negative).sum().cpu()
                ),
                f"{prefix}_teacher_negative_count": float(negative.sum().cpu()),
                f"{prefix}_positive_probability_sum": float(
                    probability[positive].sum().cpu()
                ),
                f"{prefix}_positive_probability_count": float(positive.sum().cpu()),
                f"{prefix}_negative_probability_sum": float(
                    probability[negative].sum().cpu()
                ),
                f"{prefix}_negative_probability_count": float(negative.sum().cpu()),
                f"_tail_hist_positive_{block_index}": tuple(
                    float(value) for value in positive_histogram.cpu().tolist()
                ),
                f"_tail_hist_negative_{block_index}": tuple(
                    float(value) for value in negative_histogram.cpu().tolist()
                ),
            }
        )
    return result


def _finalize_tail_routing_diagnostics(
    values: Mapping[str, float],
) -> dict[str, float]:
    result = dict(values)
    prefixes = sorted(
        {
            name.rsplit("_", 3)[0]
            for name in result
            if name.startswith("tail_block_") and name.endswith("_valid_time_count")
        }
    )
    for prefix in prefixes:
        valid = float(result.get(f"{prefix}_valid_time_count", 0.0))
        selected = float(result.get(f"{prefix}_tail_selected_count", 0.0))
        capacity = float(result.get(f"{prefix}_tail_capacity_count", 0.0))
        positives = float(result.get(f"{prefix}_teacher_positive_count", 0.0))
        negatives = float(result.get(f"{prefix}_teacher_negative_count", 0.0))
        positive_probability = (
            float(result.get(f"{prefix}_positive_probability_sum", 0.0)) / positives
            if positives > 0.0
            else 0.0
        )
        negative_probability = (
            float(result.get(f"{prefix}_negative_probability_sum", 0.0)) / negatives
            if negatives > 0.0
            else 0.0
        )
        result.update(
            {
                f"{prefix}_predicted_activation_rate": (
                    float(result.get(f"{prefix}_predicted_activation_sum", 0.0)) / valid
                    if valid > 0.0
                    else 0.0
                ),
                f"{prefix}_tail_utilization": (
                    selected / capacity if capacity > 0.0 else 0.0
                ),
                f"{prefix}_teacher_event_hit_rate": (
                    float(result.get(f"{prefix}_teacher_positive_hit_count", 0.0))
                    / positives
                    if positives > 0.0
                    else 0.0
                ),
                f"{prefix}_teacher_negative_activation_rate": (
                    float(
                        result.get(f"{prefix}_teacher_negative_activation_count", 0.0)
                    )
                    / negatives
                    if negatives > 0.0
                    else 0.0
                ),
                f"{prefix}_positive_route_probability": positive_probability,
                f"{prefix}_negative_route_probability": negative_probability,
                f"{prefix}_routing_separation": (
                    positive_probability - negative_probability
                ),
            }
        )
    return result


def _tail_routing_gate_metrics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate the five tail routers into a minimum non-random capability gate."""

    positive_count = 0.0
    negative_count = 0.0
    positive_probability_sum = 0.0
    negative_probability_sum = 0.0
    positive_hit_count = 0.0
    activation_sum = 0.0
    valid_count = 0.0
    support_present = True
    per_block_separation: dict[str, float] = {}
    per_block_lift: dict[str, float] = {}
    per_block_aucpr: dict[str, float] = {}
    per_block_prevalence: dict[str, float] = {}

    def histogram_aucpr(positive_histogram: np.ndarray, negative_histogram: np.ndarray) -> float:
        positives_desc = positive_histogram[::-1]
        negatives_desc = negative_histogram[::-1]
        positive_total = float(positives_desc.sum())
        if positive_total <= 0.0:
            return 0.0
        true_positive = np.cumsum(positives_desc)
        false_positive = np.cumsum(negatives_desc)
        precision = np.divide(
            true_positive,
            true_positive + false_positive,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=(true_positive + false_positive) > 0.0,
        )
        recall_increment = positives_desc / positive_total
        return float(np.dot(precision, recall_increment))

    for block in TAIL_ROUTE_BLOCKS:
        prefix = f"tail_block_{block}"
        positive = float(diagnostics.get(f"{prefix}_teacher_positive_count", 0.0))
        negative = float(diagnostics.get(f"{prefix}_teacher_negative_count", 0.0))
        if positive <= 0.0 or negative <= 0.0:
            support_present = False
        positive_count += positive
        negative_count += negative
        positive_probability_sum += float(
            diagnostics.get(f"{prefix}_positive_probability_sum", 0.0)
        )
        negative_probability_sum += float(
            diagnostics.get(f"{prefix}_negative_probability_sum", 0.0)
        )
        positive_hit_count += float(
            diagnostics.get(f"{prefix}_teacher_positive_hit_count", 0.0)
        )
        activation_sum += float(
            diagnostics.get(f"{prefix}_predicted_activation_sum", 0.0)
        )
        block_activation = float(
            diagnostics.get(f"{prefix}_predicted_activation_sum", 0.0)
        )
        block_valid = float(diagnostics.get(f"{prefix}_valid_time_count", 0.0))
        valid_count += block_valid
        block_positive_probability = (
            float(diagnostics.get(f"{prefix}_positive_probability_sum", 0.0))
            / positive
            if positive > 0.0
            else 0.0
        )
        block_negative_probability = (
            float(diagnostics.get(f"{prefix}_negative_probability_sum", 0.0))
            / negative
            if negative > 0.0
            else 0.0
        )
        block_hit_rate = (
            float(diagnostics.get(f"{prefix}_teacher_positive_hit_count", 0.0))
            / positive
            if positive > 0.0
            else 0.0
        )
        block_utilization = block_activation / block_valid if block_valid > 0.0 else 0.0
        per_block_separation[str(block)] = (
            block_positive_probability - block_negative_probability
        )
        per_block_lift[str(block)] = (
            block_hit_rate / block_utilization if block_utilization > 0.0 else 0.0
        )
        try:
            positive_histogram = np.asarray(
                diagnostics.get(f"_tail_hist_positive_{block}"), dtype=np.float64
            )
            negative_histogram = np.asarray(
                diagnostics.get(f"_tail_hist_negative_{block}"), dtype=np.float64
            )
        except (TypeError, ValueError):
            positive_histogram = np.asarray((), dtype=np.float64)
            negative_histogram = np.asarray((), dtype=np.float64)
        histogram_valid = bool(
            positive_histogram.shape == (TAIL_ROUTE_AUPRC_BINS,)
            and negative_histogram.shape == (TAIL_ROUTE_AUPRC_BINS,)
            and np.isfinite(positive_histogram).all()
            and np.isfinite(negative_histogram).all()
            and bool((positive_histogram >= 0.0).all())
            and bool((negative_histogram >= 0.0).all())
        )
        if not histogram_valid:
            support_present = False
            per_block_aucpr[str(block)] = 0.0
            per_block_prevalence[str(block)] = 1.0
        else:
            histogram_total = float(
                positive_histogram.sum() + negative_histogram.sum()
            )
            per_block_aucpr[str(block)] = histogram_aucpr(
                positive_histogram, negative_histogram
            )
            per_block_prevalence[str(block)] = (
                float(positive_histogram.sum()) / histogram_total
                if histogram_total > 0.0
                else 1.0
            )
    positive_probability = (
        positive_probability_sum / positive_count if positive_count > 0.0 else 0.0
    )
    negative_probability = (
        negative_probability_sum / negative_count if negative_count > 0.0 else 0.0
    )
    separation = positive_probability - negative_probability
    hit_rate = positive_hit_count / positive_count if positive_count > 0.0 else 0.0
    utilization = activation_sum / valid_count if valid_count > 0.0 else 0.0
    lift = hit_rate / utilization if utilization > 0.0 else 0.0
    finite = bool(
        np.isfinite(
            (
                positive_probability,
                negative_probability,
                separation,
                hit_rate,
                utilization,
                lift,
            )
        ).all()
    )
    block_values_finite = bool(
        np.isfinite(tuple(per_block_separation.values())).all()
        and np.isfinite(tuple(per_block_lift.values())).all()
    )
    separation_ok = bool(
        finite
        and block_values_finite
        and separation > 0.0
        and min(per_block_separation.values(), default=-np.inf) > 0.0
    )
    lift_ok = bool(
        finite
        and block_values_finite
        and lift > 1.0
        and min(per_block_lift.values(), default=-np.inf) > 1.0
    )
    aucpr_ok = bool(
        support_present
        and all(
            per_block_aucpr[str(block)] > per_block_prevalence[str(block)]
            for block in TAIL_ROUTE_BLOCKS
        )
    )
    passed = bool(support_present and separation_ok and lift_ok and aucpr_ok)
    return {
        "tail_route_support_present": bool(support_present),
        "tail_route_positive_count": float(positive_count),
        "tail_route_negative_count": float(negative_count),
        "tail_route_positive_probability": float(positive_probability),
        "tail_route_negative_probability": float(negative_probability),
        "tail_route_routing_separation": float(separation),
        "tail_route_block_separation": per_block_separation,
        "tail_route_min_block_separation": float(
            min(per_block_separation.values(), default=-np.inf)
        ),
        "tail_route_separation_passed": separation_ok,
        "tail_route_teacher_event_hit_rate": float(hit_rate),
        "tail_route_predicted_utilization": float(utilization),
        "tail_route_hit_utilization_lift": float(lift),
        "tail_route_block_hit_utilization_lift": per_block_lift,
        "tail_route_min_block_hit_utilization_lift": float(
            min(per_block_lift.values(), default=-np.inf)
        ),
        "tail_route_lift_passed": lift_ok,
        "tail_route_block_aucpr": per_block_aucpr,
        "tail_route_block_event_prevalence": per_block_prevalence,
        "tail_route_min_aucpr_margin": float(
            min(
                (
                    per_block_aucpr[str(block)]
                    - per_block_prevalence[str(block)]
                    for block in TAIL_ROUTE_BLOCKS
                ),
                default=-np.inf,
            )
        ),
        "tail_route_aucpr_passed": aucpr_ok,
        "tail_routing_passed": passed,
    }


def _tail_log_payload(
    result: float | ValidationSummary | None,
) -> dict[str, float] | None:
    if not isinstance(result, ValidationSummary):
        return None
    payload = {
        name: float(value)
        for name, value in result.diagnostics.items()
        if name.startswith("tail_block_")
        and name.endswith(
            (
                "_predicted_activation_rate",
                "_tail_utilization",
                "_teacher_event_hit_rate",
                "_teacher_negative_activation_rate",
                "_positive_route_probability",
                "_negative_route_probability",
                "_routing_separation",
            )
        )
    }
    return payload or None


def _sha256(path: Path) -> str:
    digest = _sha256_factory()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_sha256(model: nn.Module) -> str:
    """Hash C1 model tensors without retaining a second full state copy."""

    digest = _sha256_factory()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    digest = _sha256_factory()
    digest.update(payload)
    return digest.hexdigest()


def _atomic_checkpoint_alias(source: Path, target: Path) -> Path:
    """Point the legacy ``best.pt`` name at an explicit release checkpoint."""

    source_path = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.alias-{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source_path, temporary)
    except OSError:
        # Some Windows volumes disable hard links.  The copy fallback preserves
        # atomic visibility, at the cost of one compatibility copy.
        shutil.copy2(source_path, temporary)
    os.replace(temporary, target)
    return target


@dataclass(frozen=True)
class EfficientPhaseSchedule:
    """Opt-in W0/W1 cadence; epoch means the existing phase's short epoch."""

    enabled: bool = False
    first_formal_epoch: int = 1
    recovery_interval_seconds: float = 1800.0
    progress_interval_seconds: float = 60.0
    fixed_budget: bool = False
    early_stop_start_epoch: int = 64
    early_stop_evaluations: int = 4
    relative_improvement: float = 0.001
    defer_w1_reference: bool = True
    selection_policy: str = "capability_gated"
    checkpoint_policy: str = "milestones"

    def formal_due(self, epoch: int, spec: PhaseSpec) -> bool:
        return epoch >= self.first_formal_epoch and (
            (epoch - self.first_formal_epoch) % spec.formal_validation_interval == 0
            or epoch == spec.maximum_epochs
        )

    def stale_count(self, epoch: int, previous: int, improvement: float) -> int:
        if epoch < self.early_stop_start_epoch:
            return 0
        unchanged = (
            improvement <= 0.0 if self.relative_improvement == 0.0
            else improvement < self.relative_improvement
        )
        return previous + 1 if unchanged else 0


def efficient_phase_schedule(
    config: Mapping[str, Any], phase: str, spec: PhaseSpec
) -> EfficientPhaseSchedule:
    raw = _phase_mapping(config, phase).get("efficient_schedule", {})
    if not isinstance(raw, Mapping):
        raise WorldRunnerError("efficient_schedule must be a mapping")
    schedule = EfficientPhaseSchedule(**raw)
    if not schedule.enabled:
        return schedule
    if phase not in {"w0", "w1"}:
        raise WorldRunnerError("efficient_schedule currently supports W0/W1 only")
    if (
        not 1 <= schedule.first_formal_epoch <= spec.maximum_epochs
        or schedule.first_formal_epoch % spec.short_epoch_factor
        or schedule.recovery_interval_seconds <= 0
        or schedule.progress_interval_seconds <= 0
        or schedule.early_stop_evaluations < 1
        or schedule.relative_improvement < 0
        or schedule.selection_policy not in {"capability_gated", "validation_score_only"}
        or schedule.checkpoint_policy not in {"milestones", "best_only"}
    ):
        raise WorldRunnerError("invalid efficient phase schedule")
    if schedule.checkpoint_policy == "best_only" and schedule.selection_policy != "validation_score_only":
        raise WorldRunnerError("best_only checkpoint policy requires validation_score_only selection")
    return schedule


def replay_formal_score_patience(
    records: Iterable[Mapping[str, Any]],
    schedule: EfficientPhaseSchedule,
    *,
    through_epoch: int,
    through_update: int,
) -> dict[str, Any]:
    """Rebuild revised score-only patience from completed, checkpoint-bound vals."""
    formal_rows: dict[int, Mapping[str, Any]] = {}
    for row in records:
        if row.get("formal_validation_loss") is None:
            continue
        epoch = int(row["epoch"])
        if epoch <= through_epoch and int(row["updates_completed"]) <= through_update:
            formal_rows[epoch] = row
    best = float("inf")
    stale = 0
    evidence_epoch = None
    evidence_update = None
    trace = []
    for epoch, row in sorted(formal_rows.items()):
        if epoch < schedule.first_formal_epoch:
            continue
        summary = row.get("formal_selection")
        score = float(
            summary["primary_score"] if isinstance(summary, Mapping)
            else row["formal_validation_loss"]
        )
        if not np.isfinite(score):
            raise WorldRunnerError("cannot rebuild patience from a nonfinite formal score")
        improvement = (
            max(0.0, (best - min(best, score)) / max(abs(best), 1e-12))
            if np.isfinite(best) else 1.0
        )
        best = min(best, score)
        stale = schedule.stale_count(epoch, stale, improvement)
        if stale < schedule.early_stop_evaluations:
            evidence_epoch = evidence_update = None
        elif evidence_epoch is None:
            evidence_epoch, evidence_update = epoch, int(row["updates_completed"])
        if epoch >= schedule.early_stop_start_epoch:
            trace.append({
                "epoch": epoch, "primary_score": score,
                "relative_improvement": improvement, "stale_evaluations": stale,
                "updates_completed": int(row["updates_completed"]),
            })
    return {
        "stale_evaluations": stale,
        "stop_evidence_epoch": evidence_epoch,
        "stop_evidence_update": evidence_update,
        "formal_history_count": len(formal_rows),
        "monitored_formal_history_count": sum(epoch >= schedule.first_formal_epoch for epoch in formal_rows),
        "trace": trace,
    }


def _cpu_model_snapshot(model: nn.Module) -> dict[str, Any]:
    return {
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "rng_state": _rng_state(
            capture_cuda=any(parameter.is_cuda for parameter in model.parameters())
        ),
    }


@contextmanager
def _temporary_model_snapshot(model: nn.Module, snapshot: Mapping[str, Any]):
    """Evaluate frozen initial weights without changing training weights or RNG."""
    current = _cpu_model_snapshot(model)
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        model.load_state_dict(snapshot["model_state"])
        _restore_rng_state(snapshot["rng_state"])
        yield
    finally:
        model.load_state_dict(current["model_state"])
        _restore_rng_state(current["rng_state"])
        for name, module in model.named_modules():
            module.training = modes[name]


def _phase_mapping(config: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise WorldRunnerError("configuration.training must be a mapping")
    phases = training.get("phases")
    if (
        not isinstance(phases, Mapping)
        or tuple(str(key).lower() for key in phases) != PHASE_ORDER
    ):
        raise WorldRunnerError(
            f"training phases must be ordered as {list(PHASE_ORDER)}"
        )
    value = phases.get(phase)
    if not isinstance(value, Mapping):
        raise WorldRunnerError(f"training phase {phase} must be a mapping")
    return value


def phase_spec_from_config(config: Mapping[str, Any], phase: str) -> PhaseSpec:
    normalized = str(phase).lower()
    value = _phase_mapping(config, normalized)
    is_c1 = normalized == "c1"
    training = config.get("training")
    checkpointing = (
        training.get("checkpointing", {}) if isinstance(training, Mapping) else {}
    )
    uncertainty_validation = (
        training.get("uncertainty_validation", {})
        if isinstance(training, Mapping)
        else {}
    )
    if not isinstance(checkpointing, Mapping):
        raise WorldRunnerError("configuration.training.checkpointing must be a mapping")
    if not isinstance(uncertainty_validation, Mapping):
        raise WorldRunnerError(
            "configuration.training.uncertainty_validation must be a mapping"
        )
    short_epoch_factor = int(value.get("short_epoch_factor", 1))
    batch_size = int(value.get("batch_size", 32))
    raw_budget = value.get("target_update_budgets", {})
    if not isinstance(raw_budget, Mapping):
        raise WorldRunnerError(
            f"training.phases.{normalized}.target_update_budgets must be a mapping"
        )
    budget_values = {
        "minimum": int(raw_budget.get("minimum", 0)),
        "maximum": int(raw_budget.get("maximum", 0)),
        "patience": int(raw_budget.get("patience_equivalent", 0)),
    }
    train_examples = int(raw_budget.get("train_examples", 0))
    expected_full_updates = (
        (train_examples + batch_size - 1) // batch_size if train_examples else 0
    )
    if raw_budget:
        reference_batch = int(raw_budget.get("reference_batch_size", 0))
        reference_updates = int(raw_budget.get("reference_full_epoch_updates", 0))
        if (
            train_examples <= 0
            or reference_batch <= 0
            or reference_updates
            != (train_examples + reference_batch - 1) // reference_batch
            or not all(budget_values.values())
        ):
            raise WorldRunnerError(
                f"training.phases.{normalized}.target_update_budgets is incomplete"
            )
        configured_limits = {
            "minimum": int(value.get("min_epochs", 0)),
            "maximum": int(value.get("max_epochs", 0)),
            "patience": int(value.get("patience", 0)),
        }
        derived_limits = {
            name: short_epoch_limit_from_update_budget(
                target,
                updates_per_full_epoch=expected_full_updates,
                short_epoch_factor=short_epoch_factor,
            )
            for name, target in budget_values.items()
        }
        if configured_limits != derived_limits:
            raise WorldRunnerError(
                f"training.phases.{normalized} short-epoch limits {configured_limits} "
                f"do not preserve target update budgets {derived_limits}"
            )
    return PhaseSpec(
        name=normalized,
        minimum_epochs=int(value.get("min_epochs", 0 if is_c1 else 1)),
        maximum_epochs=int(value.get("max_epochs", 0 if is_c1 else -1)),
        learning_rate=float(value.get("learning_rate", 0.0 if is_c1 else -1.0)),
        weight_decay=float(value.get("weight_decay", 1.0e-4)),
        gradient_clip_norm=float(value.get("gradient_clip_norm", 1.0)),
        patience=int(value.get("patience", 20)),
        batch_size=batch_size,
        validation_batch_size=(
            None
            if value.get("validation_batch_size") is None
            else int(value["validation_batch_size"])
        ),
        sample_count=int(value.get("sample_count", 20)),
        formal_sample_count=(
            None
            if normalized != "u0"
            else int(
                value.get(
                    "formal_sample_count",
                    uncertainty_validation.get("formal_paths", 100),
                )
            )
        ),
        horizon=int(value.get("horizon", 96)),
        cvar_alpha=float(value.get("cvar_alpha", 0.9)),
        short_epoch_factor=short_epoch_factor,
        formal_validation_interval=int(
            value.get("formal_validation_interval_short_epochs", short_epoch_factor)
        ),
        sentinel_validation_batches=int(
            checkpointing.get("sentinel_validation_batches", 12)
        ),
        checkpoint_interval_updates=int(checkpointing.get("fixed_update_interval", 4)),
        checkpoint_max_seconds=float(
            checkpointing.get("max_wall_clock_seconds", 180.0)
        ),
        expected_updates_per_full_epoch=expected_full_updates,
        target_minimum_updates=budget_values["minimum"],
        target_maximum_updates=budget_values["maximum"],
        target_patience_updates=budget_values["patience"],
        head_learning_rate=(
            None
            if value.get("head_learning_rate") is None
            else float(value["head_learning_rate"])
        ),
        router_learning_rate=(
            None
            if value.get("router_learning_rate") is None
            else float(value["router_learning_rate"])
        ),
        tail_learning_rate=(
            None
            if value.get("tail_learning_rate") is None
            else float(value["tail_learning_rate"])
        ),
        matrix_weight_decay=(
            None
            if value.get("matrix_weight_decay") is None
            else float(value["matrix_weight_decay"])
        ),
        warmup_updates=int(value.get("warmup_updates", 0)),
        cosine_decay_updates=int(value.get("cosine_decay_updates", 0)),
        minimum_learning_rate_ratio=float(
            value.get("minimum_learning_rate_ratio", 1.0)
        ),
        adamw_beta2=float(value.get("adamw_beta2", 0.999)),
        w0_pcgrad=bool(value.get("w0_pcgrad", False)),
    )


def formal_validation_schedule(
    config: Mapping[str, Any], phase: str, spec: PhaseSpec
) -> tuple[int, int]:
    """Return the early formal cadence and the last early short epoch.

    ``PhaseSpec.formal_validation_interval`` remains the full-data cadence used
    after the early window.  W0 defaults to every four short epochs through 32;
    both values can be frozen in the phase config without changing the resume
    epoch coordinate contract.
    """

    normalized = str(phase).lower()
    value = _phase_mapping(config, normalized)
    training = config.get("training")
    checkpointing = (
        training.get("checkpointing", {}) if isinstance(training, Mapping) else {}
    )
    if not isinstance(checkpointing, Mapping):
        raise WorldRunnerError("configuration.training.checkpointing must be a mapping")
    default_interval = (
        min(4, spec.formal_validation_interval)
        if normalized == "w0" and spec.short_epoch_factor > 1
        else spec.formal_validation_interval
    )
    default_until = 32 if normalized == "w0" and spec.short_epoch_factor > 1 else 0
    early_interval = int(
        value.get(
            "early_formal_interval_short_epochs",
            value.get(
                "early_formal_validation_interval_short_epochs",
                checkpointing.get(
                    "early_formal_validation_interval_short_epochs",
                    default_interval,
                ),
            ),
        )
    )
    early_until = int(
        value.get(
            "early_formal_until_short_epoch",
            value.get(
                "early_formal_validation_until_short_epoch",
                checkpointing.get(
                    "early_formal_validation_until_short_epoch", default_until
                ),
            ),
        )
    )
    if (
        early_interval <= 0
        or spec.formal_validation_interval % early_interval
        or early_until < 0
        or (early_until and early_until % early_interval)
    ):
        raise WorldRunnerError(
            "early formal validation requires a positive divisor of the regular "
            "interval and an aligned nonnegative end short epoch"
        )
    return early_interval, early_until


def is_formal_validation_epoch(
    epoch: int,
    *,
    regular_interval: int,
    early_interval: int,
    early_until: int,
) -> bool:
    value = int(epoch)
    if value <= int(early_until):
        return value % int(early_interval) == 0
    return value % int(regular_interval) == 0


def short_epoch_limit_from_update_budget(
    target_updates: int,
    *,
    updates_per_full_epoch: int,
    short_epoch_factor: int,
) -> int:
    """Round an optimizer-update target up to a complete full-data epoch."""

    target = int(target_updates)
    updates = int(updates_per_full_epoch)
    factor = int(short_epoch_factor)
    if target <= 0 or updates <= 0 or factor <= 0 or factor & (factor - 1):
        raise WorldRunnerError(
            "update-budget conversion requires positive values and power-of-two factor"
        )
    return ((target + updates - 1) // updates) * factor


def short_epoch_coordinates(short_epoch: int, factor: int) -> tuple[int, int]:
    """Map a one-based short epoch to its full-epoch shuffle and shard indices."""

    epoch = int(short_epoch)
    count = int(factor)
    if epoch <= 0 or count <= 0 or count & (count - 1):
        raise WorldRunnerError(
            "short epoch coordinates require a positive power of two"
        )
    return (epoch - 1) // count + 1, (epoch - 1) % count


def stratified_validation_ordinals(
    total_batches: int, requested: int
) -> tuple[int, ...]:
    """Choose deterministic stratum midpoints spanning the complete validation year."""

    total = int(total_batches)
    count = min(total, int(requested))
    if total <= 0 or count <= 0:
        raise WorldRunnerError("validation must expose at least one sentinel batch")
    # Integer midpoint arithmetic is stable across NumPy/Python versions and picks
    # one batch from every equally sized chronological stratum.
    return tuple(((2 * index + 1) * total) // (2 * count) for index in range(count))


def validation_ordinal_hash(
    ordinals: Sequence[int],
    *,
    total_batches: int,
    batch_size: int,
    phase: str,
) -> str:
    payload = json.dumps(
        {
            "total_batches": int(total_batches),
            "batch_size": int(batch_size),
            "phase": str(phase).lower(),
            "split": "validation",
            "batch_ordinals": [int(value) for value in ordinals],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    # FNV-1a is sufficient for this small, non-security lineage checksum and
    # avoids depending on a platform crypto provider in the hot runner path.
    digest = 0xCBF29CE484222325
    for byte in payload:
        digest ^= byte
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{digest:016x}"


def _mean_terms(
    values: Mapping[str, float], names: Sequence[str], *, label: str
) -> float:
    missing = [name for name in names if name not in values]
    if missing:
        raise WorldRunnerError(f"{label} is missing exact terms {missing}")
    result = float(np.mean([float(values[name]) for name in names]))
    if not np.isfinite(result):
        raise WorldRunnerError(f"{label} is non-finite")
    return result


def _confidence_half_width(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return 0.0
    return float(1.96 * array.std(ddof=1) / np.sqrt(array.size))


def _paired_one_sided_upper(
    model_values: Sequence[float],
    baseline_values: Sequence[float],
    model_counts: Sequence[float],
    baseline_counts: Sequence[float],
    *,
    confidence: float,
) -> tuple[float, int]:
    """Normal paired upper bound, excluding batches empty for either score."""

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (model_values, baseline_values, model_counts, baseline_counts)
    ]
    if not arrays or len({array.shape for array in arrays}) != 1:
        raise WorldRunnerError("paired queue score arrays have inconsistent shapes")
    valid = (
        np.isfinite(arrays[0])
        & np.isfinite(arrays[1])
        & (arrays[2] > 0.0)
        & (arrays[3] > 0.0)
    )
    if not np.allclose(arrays[2][valid], arrays[3][valid], rtol=1.0e-7, atol=1.0e-7):
        raise WorldRunnerError("paired model/baseline valid counts differ")
    differences = arrays[0][valid] - arrays[1][valid]
    if differences.size == 0:
        return float("inf"), 0
    level = float(confidence)
    if abs(level - 0.95) > 1.0e-12:
        raise WorldRunnerError("W0 queue paired gate requires 95% confidence")
    weights = arrays[2][valid]
    weight_sum = float(weights.sum())
    weighted_mean = float(np.dot(weights, differences) / weight_sum)
    weight_square_sum = float(np.square(weights).sum())
    effective_batches = weight_sum * weight_sum / weight_square_sum
    variance_denominator = weight_sum - weight_square_sum / weight_sum
    weighted_variance = (
        0.0
        if differences.size <= 1 or variance_denominator <= 0.0
        else float(
            np.dot(weights, np.square(differences - weighted_mean))
            / variance_denominator
        )
    )
    standard_error = float(np.sqrt(weighted_variance / effective_batches))
    return (
        float(weighted_mean + 1.6448536269514722 * standard_error),
        int(differences.size),
    )


def summarize_validation(
    phase: str,
    terms: Mapping[str, ScoreTerm],
    *,
    batch_term_values: Mapping[str, Sequence[float]],
    batch_term_counts: Mapping[str, Sequence[float]] | None = None,
    diagnostics: Mapping[str, Any],
    batch_count: int,
    w0_state_thresholds: Mapping[str, float] | None = None,
) -> ValidationSummary:
    """Build the registered formal selector without MAE/RMSE."""

    normalized = str(phase).lower()
    scalars = {
        name: float(term.scalar.detach().float().cpu()) for name, term in terms.items()
    }
    forbidden = [name for name in scalars if name.lower() in {"mae", "rmse"}]
    if forbidden:
        raise WorldRunnerError("MAE/RMSE are prohibited from checkpoint selection")
    identity_ok = float(diagnostics.get("dart_identity_max_abs", 0.0)) <= 1.0e-6
    eligible = identity_ok
    metrics: dict[str, Any] = {"dart_identity_passed": identity_ok}

    if normalized == "w0":
        families = [f"quantile/{name}" for name in ("price", "workload", "pv", "pue")]
        # Twice integrated pinball is the quantile representation of CRPS.
        group_crps = 2.0 * _mean_terms(scalars, families, label="W0 group CRPS")
        queue_terms = ("state/queue_occurrence", "state/queue_severity")
        state = _mean_terms(scalars, ("state/soc", *queue_terms), label="W0 state gate")
        thresholds = dict(DEFAULT_W0_STATE_THRESHOLDS)
        if w0_state_thresholds is not None:
            thresholds.update(
                {str(key): float(value) for key, value in w0_state_thresholds.items()}
            )
        counts = {} if batch_term_counts is None else batch_term_counts
        occurrence_upper, occurrence_batches = _paired_one_sided_upper(
            batch_term_values.get("state/queue_occurrence", ()),
            batch_term_values.get("state/queue_occurrence_zero_baseline", ()),
            counts.get("state/queue_occurrence", ()),
            counts.get("state/queue_occurrence_zero_baseline", ()),
            confidence=thresholds["queue_paired_confidence"],
        )
        severity_upper, severity_batches = _paired_one_sided_upper(
            batch_term_values.get("state/queue_severity", ()),
            batch_term_values.get("state/queue_severity_zero_baseline", ()),
            counts.get("state/queue_severity", ()),
            counts.get("state/queue_severity_zero_baseline", ()),
            confidence=thresholds["queue_paired_confidence"],
        )
        soc_upper, soc_batches = _paired_one_sided_upper(
            batch_term_values.get("state/soc", ()),
            batch_term_values.get("state/soc_persistence_baseline", ()),
            counts.get("state/soc", ()),
            counts.get("state/soc_persistence_baseline", ()),
            confidence=thresholds["queue_paired_confidence"],
        )
        positive_support = (
            float(diagnostics.get("queue_positive_target_count", 0.0))
            >= thresholds["queue_positive_target_count_min"]
        )
        occurrence_ok = (
            positive_support
            and occurrence_batches > 0
            and occurrence_upper < thresholds["queue_occurrence_paired_upper_max"]
        )
        severity_ok = (
            positive_support
            and severity_batches > 0
            and severity_upper < thresholds["queue_severity_paired_upper_max"]
        )
        soc_ok = soc_batches > 0 and soc_upper < thresholds["soc_paired_upper_max"]
        state_ok = bool(np.isfinite(state) and occurrence_ok and severity_ok and soc_ok)
        eligible = eligible and state_ok
        primary = group_crps
        metrics.update(
            group_balanced_crps=group_crps,
            state_score=state,
            state_gate_passed=state_ok,
            queue_occurrence_passed=occurrence_ok,
            queue_severity_passed=severity_ok,
            soc_passed=soc_ok,
            queue_positive_target_count=float(
                diagnostics.get("queue_positive_target_count", 0.0)
            ),
            queue_occurrence_paired_upper=occurrence_upper,
            queue_occurrence_paired_batches=occurrence_batches,
            queue_severity_paired_upper=severity_upper,
            queue_severity_paired_batches=severity_batches,
            soc_paired_upper=soc_upper,
            soc_paired_batches=soc_batches,
            queue_occurrence_recall=float(
                diagnostics.get("queue_occurrence_recall", 0.0)
            ),
            queue_false_positive_rate=float(
                diagnostics.get("queue_false_positive_rate", 1.0)
            ),
            queue_positive_log1p_mae=float(
                diagnostics.get("queue_positive_log1p_mae", 1.0e30)
            ),
            soc_mae=float(diagnostics.get("soc_mae", 1.0e30)),
            soc_max_abs=float(diagnostics.get("soc_max_abs", 1.0e30)),
            state_thresholds=thresholds,
        )
    elif normalized == "w1":
        horizon_scores: dict[str, float] = {}
        for horizon in (1, 4, 16, 96):
            names = [
                f"rollout/h{horizon}/{family}"
                for family in ("price", "workload", "pv", "pue")
            ]
            horizon_scores[f"h{horizon}"] = 2.0 * _mean_terms(
                scalars, names, label=f"W1 h{horizon} proper score"
            )
        raw_kl: dict[str, float] = {}
        floor_fraction: dict[str, float] = {}
        queue_valid_counts: dict[str, dict[str, float]] = {}
        for horizon in (1, 4, 16, 96):
            raw_name = f"latent/raw_h{horizon}"
            floor_name = f"latent/free_bit_floor_fraction_h{horizon}"
            if raw_name not in terms or floor_name not in terms:
                raise WorldRunnerError(
                    f"W1 h{horizon} omitted raw-KL/free-bit diagnostics"
                )
            raw_kl[f"h{horizon}"] = scalars[raw_name]
            floor_fraction[f"h{horizon}"] = scalars[floor_name]
            queue_valid_counts[f"h{horizon}"] = {}
            for family in ("queue_occurrence", "queue_severity"):
                name = f"state/h{horizon}/{family}"
                if name not in terms:
                    raise WorldRunnerError(f"W1 omitted queue term {name}")
                queue_valid_counts[f"h{horizon}"][family] = float(
                    terms[name].valid_count.detach().float().cpu()
                )
        latent_non_degenerate = bool(
            all(
                np.isfinite(raw_kl[f"h{horizon}"])
                and raw_kl[f"h{horizon}"] > 1.0e-6
                and float(
                    terms[f"latent/raw_h{horizon}"]
                    .valid_count.detach()
                    .float()
                    .cpu()
                )
                > 0.0
                for horizon in (1, 4, 16, 96)
            )
        )
        occurrence_support = all(
            values["queue_occurrence"] > 0.0
            for values in queue_valid_counts.values()
        )
        severity_support = {
            horizon: values["queue_severity"] > 0.0
            for horizon, values in queue_valid_counts.items()
        }
        # A horizon with no genuinely positive queue target has no conditional
        # severity task.  It is not a failure and must never be populated with
        # fabricated positives; at least one horizon in the full validation
        # set must still exercise severity.
        queue_term_support = occurrence_support and any(severity_support.values())
        queue_non_degenerate = bool(
            queue_term_support
            and float(diagnostics.get("queue_rollout_non_degenerate", 0.0))
            > 0.0
        )
        eligible = eligible and latent_non_degenerate and queue_non_degenerate
        primary = max(horizon_scores.values())
        metrics.update(
            horizon_scores=horizon_scores,
            worst_horizon_score=primary,
            latent_raw_kl=raw_kl,
            latent_free_bit_floor_fraction=floor_fraction,
            latent_kl_non_degenerate=latent_non_degenerate,
            queue_rollout_valid_counts=queue_valid_counts,
            queue_severity_applicable=severity_support,
            queue_prediction_non_degenerate_rate=float(
                diagnostics.get("queue_rollout_non_degenerate", 0.0)
            ),
            queue_rollout_non_degenerate=queue_non_degenerate,
        )
    elif normalized == "u0":
        crps = _mean_terms(
            scalars,
            tuple(f"crps/{name}" for name in ("price", "workload", "pv", "pue")),
            label="U0 CRPS",
        )
        tail = _mean_terms(
            scalars,
            tuple(f"tail/{name}" for name in ("price", "workload", "pv", "pue")),
            label="U0 tail CRPS",
        )
        if "energy" not in scalars:
            raise WorldRunnerError("U0 selection is missing Energy Score")
        energy = scalars["energy"]
        energy_half = _confidence_half_width(batch_term_values.get("energy", ()))
        atom_brier_scores = {
            family: scalars[f"atom/{family}"]
            for family in ("workload", "pv", "pue")
            if f"atom/{family}" in scalars
        }
        coverage_metrics = _family_rank_coverage_metrics(
            diagnostics,
            atom_brier=atom_brier_scores,
        )
        balanced_atom_brier = (
            float(np.mean(tuple(atom_brier_scores.values())))
            if len(atom_brier_scores) == 3
            else np.inf
        )
        coverage_metrics = dict(coverage_metrics)
        recorded_atom_brier = coverage_metrics.setdefault(
            "atom_brier", atom_brier_scores
        )
        recorded_balanced_atom_brier = coverage_metrics.setdefault(
            "family_balanced_atom_brier", balanced_atom_brier
        )
        if dict(recorded_atom_brier) != atom_brier_scores or not np.isclose(
            float(recorded_balanced_atom_brier),
            float(balanced_atom_brier),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise WorldRunnerError(
                "U0 atom Brier evidence disagrees between proper scores and "
                "mixed-support coverage"
            )
        coverage_ok = bool(coverage_metrics["family_coverage_passed"])
        tail_route_metrics = _tail_routing_gate_metrics(diagnostics)
        tail_route_ok = bool(tail_route_metrics["tail_routing_passed"])
        duplicate_ok = float(diagnostics.get("scenario_duplicate_rate", np.inf)) <= 0.01
        diversity_ok = float(diagnostics.get("scenario_diversity", -np.inf)) >= 1.0e-4
        eligible = (
            eligible
            and coverage_ok
            and tail_route_ok
            and duplicate_ok
            and diversity_ok
        )
        primary = energy
        metrics.update(
            energy=energy,
            energy_ci_lower=energy - energy_half,
            energy_ci_upper=energy + energy_half,
            crps=crps,
            tail_crps=tail,
            duplicate_passed=duplicate_ok,
            diversity_passed=diversity_ok,
            mixture_nll=float(scalars.get("mixture_nll", np.inf)),
            **coverage_metrics,
            **tail_route_metrics,
        )
    elif normalized in {"p1", "j1"}:
        prefix = "policy/" if normalized == "j1" else ""
        cost_name, cvar_name = f"{prefix}expected_cost", f"{prefix}cvar"
        if cost_name not in scalars or cvar_name not in scalars:
            raise WorldRunnerError(f"{normalized.upper()} selection lacks cost/CVaR")
        cost, cvar = scalars[cost_name], scalars[cvar_name]
        cost_half = _confidence_half_width(batch_term_values.get(cost_name, ()))
        cvar_half = _confidence_half_width(batch_term_values.get(cvar_name, ()))
        feasible = (
            float(diagnostics.get("constraint_violation_max_abs", np.inf)) <= 1.0e-5
        )
        sla = float(diagnostics.get("sla_expired_work_max", np.inf)) <= 1.0e-8
        if normalized == "j1":
            coverage_ok = all(
                abs(
                    float(diagnostics.get(f"coverage_{nominal}", np.inf))
                    - nominal / 100.0
                )
                <= 0.03
                for nominal in (50, 80, 90)
            )
            duplicate_ok = (
                float(diagnostics.get("scenario_duplicate_rate", np.inf)) <= 0.01
            )
            diversity_ok = (
                float(diagnostics.get("scenario_diversity", -np.inf)) >= 1.0e-4
            )
            uncertainty_ok = coverage_ok and duplicate_ok and diversity_ok
            eligible = eligible and uncertainty_ok
            metrics.update(
                uncertainty_gate_passed=uncertainty_ok,
                coverage_passed=coverage_ok,
                duplicate_passed=duplicate_ok,
                diversity_passed=diversity_ok,
            )
            world_horizons: dict[str, float] = {}
            latent_valid = True
            occurrence_valid = True
            severity_support = False
            for horizon in (1, 4, 16, 96):
                names = tuple(
                    f"world/rollout/h{horizon}/{family}"
                    for family in ("price", "workload", "pv", "pue")
                )
                world_horizons[f"h{horizon}"] = 2.0 * _mean_terms(
                    scalars, names, label=f"J1 world h{horizon} proper score"
                )
                latent_name = f"world/latent/raw_h{horizon}"
                occurrence_name = f"world/state/h{horizon}/queue_occurrence"
                severity_name = f"world/state/h{horizon}/queue_severity"
                if any(name not in terms for name in (latent_name, occurrence_name, severity_name)):
                    raise WorldRunnerError(
                        f"J1 world structure omitted h{horizon} latent/queue evidence"
                    )
                latent_valid = bool(
                    latent_valid
                    and np.isfinite(scalars[latent_name])
                    and scalars[latent_name] > 1.0e-6
                    and float(terms[latent_name].valid_count.detach().float().cpu()) > 0.0
                )
                occurrence_valid = bool(
                    occurrence_valid
                    and float(
                        terms[occurrence_name].valid_count.detach().float().cpu()
                    )
                    > 0.0
                )
                severity_support = bool(
                    severity_support
                    or float(terms[severity_name].valid_count.detach().float().cpu())
                    > 0.0
                )
            horizons_finite = all(np.isfinite(tuple(world_horizons.values())))
            queue_support = occurrence_valid and severity_support
            world_structure_ok = horizons_finite and latent_valid and queue_support
            eligible = eligible and world_structure_ok
            metrics.update(
                world_horizon_scores=world_horizons,
                world_horizons_finite=horizons_finite,
                latent_kl_non_degenerate=latent_valid,
                queue_rollout_has_support=queue_support,
                world_structure_passed=world_structure_ok,
            )
        eligible = eligible and feasible and sla
        primary = cost + 0.25 * cvar
        metrics.update(
            expected_cost=cost,
            cvar95=cvar,
            expected_cost_upper=cost + cost_half,
            cvar95_upper=cvar + cvar_half,
            conservative_value=cost + cost_half + 0.25 * (cvar + cvar_half),
            feasibility_passed=feasible,
            sla_passed=sla,
        )
    else:
        raise WorldRunnerError(f"no formal selector for phase {phase}")
    detached = {name: term.detached() for name, term in terms.items()}
    return ValidationSummary(
        phase=normalized,
        primary_score=float(primary),
        eligible=bool(eligible),
        score_terms=detached,
        diagnostics=dict(diagnostics),
        selection_metrics=metrics,
        batch_count=int(batch_count),
    )


def formal_candidate_record(
    summary: ValidationSummary, *, epoch: int
) -> dict[str, Any]:
    return {
        "phase": summary.phase,
        "epoch": int(epoch),
        "eligible": bool(summary.eligible),
        "primary_score": float(summary.primary_score),
        "selection_metrics": dict(summary.selection_metrics),
    }


def _primary_challenger_selected(
    incumbent: Mapping[str, Any] | None, challenger: Mapping[str, Any]
) -> bool:
    if incumbent is None:
        return True
    return (
        float(challenger["primary_score"]),
        int(challenger["epoch"]),
    ) < (
        float(incumbent["primary_score"]),
        int(incumbent["epoch"]),
    )


def _u0_challenger_selected(
    incumbent: Mapping[str, Any], challenger: Mapping[str, Any]
) -> bool:
    """Apply the registered Energy-CI/CRPS/tail/epoch lexicography."""

    new_metrics = challenger.get("selection_metrics")
    old_metrics = incumbent.get("selection_metrics")
    if not isinstance(new_metrics, Mapping) or not isinstance(old_metrics, Mapping):
        raise WorldRunnerError("U0 candidates require selection_metrics")
    required = ("energy_ci_lower", "energy_ci_upper", "crps", "tail_crps")
    if any(name not in new_metrics or name not in old_metrics for name in required):
        raise WorldRunnerError("U0 candidates omitted the registered lexicography")
    new_interval = (
        float(new_metrics["energy_ci_lower"]),
        float(new_metrics["energy_ci_upper"]),
    )
    old_interval = (
        float(old_metrics["energy_ci_lower"]),
        float(old_metrics["energy_ci_upper"]),
    )
    if not np.isfinite((*new_interval, *old_interval)).all():
        raise WorldRunnerError("U0 Energy Score confidence interval is non-finite")
    if new_interval[1] < old_interval[0]:
        return True
    if old_interval[1] < new_interval[0]:
        return False
    return (
        float(new_metrics["crps"]),
        float(new_metrics["tail_crps"]),
        int(challenger["epoch"]),
    ) < (
        float(old_metrics["crps"]),
        float(old_metrics["tail_crps"]),
        int(incumbent["epoch"]),
    )


def select_formal_score_candidate(
    phase: str,
    incumbent: Mapping[str, Any] | None,
    challenger: Mapping[str, Any],
) -> FormalSelectionDecision:
    """Select the most useful formal score candidate irrespective of gates.

    This diagnostic selector never authorizes release.  In U0 it deliberately
    uses the complete registered lexicography, so a raw Energy minimum cannot
    overwrite a confidence-set candidate with better CRPS and tail score.
    """

    normalized = str(phase).lower()
    current = dict(challenger)
    if incumbent is None:
        return FormalSelectionDecision(current, True, str(current["epoch"]), 1.0)
    previous = dict(incumbent)
    if normalized == "u0":
        current_metrics = current.get("selection_metrics")
        previous_metrics = previous.get("selection_metrics")
        required = {"energy_ci_lower", "energy_ci_upper", "crps", "tail_crps"}
        has_registered_metrics = (
            isinstance(current_metrics, Mapping)
            and isinstance(previous_metrics, Mapping)
            and required.issubset(current_metrics)
            and required.issubset(previous_metrics)
        )
        # Synthetic legacy unit losses have no ScoreTerms or U0 selection
        # metrics.  They retain scalar ordering for compatibility; every formal
        # R5-V2.3 candidate enters the complete branch above.
        choose = (
            _u0_challenger_selected(previous, current)
            if has_registered_metrics
            else _primary_challenger_selected(previous, current)
        )
    else:
        choose = _primary_challenger_selected(previous, current)
    selected = current if choose else previous
    previous_score = float(previous["primary_score"])
    selected_score = float(selected["primary_score"])
    return FormalSelectionDecision(
        selected=selected,
        challenger_selected=choose,
        candidate_signature=str(selected["epoch"]),
        relative_improvement=float(
            max(
                0.0,
                (previous_score - selected_score)
                / max(abs(previous_score), 1.0e-12),
            )
        ),
    )


def _w0_pareto_vector(record: Mapping[str, Any]) -> tuple[float, float] | None:
    metrics = record.get("selection_metrics")
    if not isinstance(metrics, Mapping):
        return None
    vector = (
        float(record.get("primary_score", np.inf)),
        float(metrics.get("queue_occurrence_paired_upper", np.inf)),
    )
    return vector if np.isfinite(vector).all() else None


def update_w0_pareto_frontier(
    frontier: Sequence[Mapping[str, Any]], challenger: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    """Update the proper-score/queue-upper frontier irrespective of eligibility."""

    candidate = dict(challenger)
    vector = _w0_pareto_vector(candidate)
    if vector is None:
        return [dict(item) for item in frontier], False
    retained: list[dict[str, Any]] = []
    for raw in frontier:
        item = dict(raw)
        incumbent = _w0_pareto_vector(item)
        if incumbent is None:
            continue
        incumbent_dominates = all(
            left <= right for left, right in zip(incumbent, vector)
        )
        if incumbent_dominates:
            # Equality is deliberately stale: rerunning an identical candidate
            # must not keep resetting W0 early stopping.
            return [dict(existing) for existing in frontier], False
        candidate_dominates = all(
            left <= right for left, right in zip(vector, incumbent)
        )
        if not candidate_dominates:
            retained.append(item)
    retained.append(candidate)
    retained.sort(
        key=lambda item: (
            *(_w0_pareto_vector(item) or (np.inf, np.inf)),
            int(item["epoch"]),
        )
    )
    return retained, True


def _u0_pareto_vector(record: Mapping[str, Any]) -> tuple[float, ...] | None:
    metrics = record.get("selection_metrics")
    if not isinstance(metrics, Mapping):
        return None
    vector = (
        float(metrics.get("energy", np.inf)),
        float(metrics.get("crps", np.inf)),
        float(metrics.get("tail_crps", np.inf)),
        float(metrics.get("worst_family_coverage_error", np.inf)),
    )
    return vector if np.isfinite(vector).all() else None


def update_u0_pareto_frontier(
    frontier: Sequence[Mapping[str, Any]], challenger: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    """Keep non-dominated U0 score/calibration candidates, including failures."""

    candidate = dict(challenger)
    vector = _u0_pareto_vector(candidate)
    if vector is None:
        return [dict(item) for item in frontier], False
    retained: list[dict[str, Any]] = []
    for raw in frontier:
        item = dict(raw)
        incumbent = _u0_pareto_vector(item)
        if incumbent is None:
            continue
        if all(left <= right for left, right in zip(incumbent, vector, strict=True)):
            return [dict(existing) for existing in frontier], False
        if not all(left <= right for left, right in zip(vector, incumbent, strict=True)):
            retained.append(item)
    retained.append(candidate)
    if len(retained) > U0_PARETO_CHECKPOINT_CAP:
        vectors = np.asarray(
            [_u0_pareto_vector(item) for item in retained], dtype=np.float64
        )
        selected_indices = {int(np.argmin(vectors[:, axis])) for axis in range(4)}
        minimum = vectors.min(axis=0)
        span = vectors.max(axis=0) - minimum
        normalized = (vectors - minimum) / np.where(span > 0.0, span, 1.0)
        knee_order = sorted(
            range(len(retained)),
            key=lambda index: (
                float(np.linalg.vector_norm(normalized[index])),
                int(retained[index]["epoch"]),
            ),
        )
        for index in knee_order:
            if len(selected_indices) >= U0_PARETO_CHECKPOINT_CAP:
                break
            selected_indices.add(index)
        retained = [retained[index] for index in sorted(selected_indices)]
    retained.sort(
        key=lambda item: (*(_u0_pareto_vector(item) or (np.inf,) * 4), int(item["epoch"]))
    )
    candidate_survived = any(
        int(item.get("epoch", -1)) == int(candidate.get("epoch", -2))
        for item in retained
    )
    return retained, candidate_survived


def select_formal_candidate(
    phase: str,
    incumbent: Mapping[str, Any] | None,
    challenger: Mapping[str, Any],
    *,
    w1_reference: Mapping[str, float] | None = None,
) -> FormalSelectionDecision:
    """Incremental phase selector implementing the registered lexicography."""

    normalized = str(phase).lower()
    current = dict(challenger)
    if not bool(current.get("eligible")):
        selected = None if incumbent is None else dict(incumbent)
        signature = "NONE" if selected is None else str(selected["epoch"])
        return FormalSelectionDecision(selected, False, signature, 0.0)
    if incumbent is None:
        return FormalSelectionDecision(current, True, str(current["epoch"]), 1.0)
    previous = dict(incumbent)
    previous_score = float(previous["primary_score"])
    choose = False
    if normalized == "w1":
        if not w1_reference:
            raise WorldRunnerError("W1 selector requires frozen horizon references")

        def worst_relative(record: Mapping[str, Any]) -> float:
            horizons = record["selection_metrics"]["horizon_scores"]
            return max(
                float(horizons[key]) / max(float(w1_reference[key]), 1.0e-12)
                for key in ("h1", "h4", "h16", "h96")
            )

        current_score, previous_rank = worst_relative(current), worst_relative(previous)
        current["relative_worst_horizon_score"] = current_score
        choose = (current_score, int(current["epoch"])) < (
            previous_rank,
            int(previous["epoch"]),
        )
    elif normalized == "u0":
        choose = _u0_challenger_selected(previous, current)
    elif normalized in {"p1", "j1"}:
        choose = (
            float(current["selection_metrics"]["conservative_value"]),
            int(current["epoch"]),
        ) < (
            float(previous["selection_metrics"]["conservative_value"]),
            int(previous["epoch"]),
        )
    else:
        choose = (float(current["primary_score"]), int(current["epoch"])) < (
            previous_score,
            int(previous["epoch"]),
        )
    selected = current if choose else previous
    selected_score = float(selected["primary_score"])
    improvement = max(
        0.0, (previous_score - selected_score) / max(abs(previous_score), 1.0e-12)
    )
    return FormalSelectionDecision(
        selected=selected,
        challenger_selected=choose,
        candidate_signature=str(selected["epoch"]),
        relative_improvement=float(improvement),
    )


def _validation_primary(value: float | ValidationSummary) -> float:
    return (
        float(value.primary_score)
        if isinstance(value, ValidationSummary)
        else float(value)
    )


def _assert_development_date(value: Any, *, label: str) -> None:
    try:
        parsed = date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise WorldRunnerError(f"{label} contains an invalid ISO date") from exc
    if parsed.year == 2025:
        raise WorldRunnerError(f"{label} attempted to open the sealed 2025 test year")
    if parsed.year not in {2022, 2023, 2024}:
        raise WorldRunnerError(f"{label} lies outside the development protocol")


def assert_development_batch(batch: Any, *, label: str = "batch") -> None:
    if not isinstance(batch, Mapping):
        return
    for key in ("date", "operating_date", "issue_date"):
        if key in batch:
            value = batch[key]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    _assert_development_date(item, label=f"{label}.{key}")
            else:
                _assert_development_date(value, label=f"{label}.{key}")
    for key in ("dates", "operating_dates", "issue_dates"):
        value = batch.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                _assert_development_date(item, label=f"{label}.{key}")
    if batch.get("locked_2025_opened") not in (None, False):
        raise WorldRunnerError(f"{label} reports locked_2025_opened=true")


def assert_w1_h96_target(batch: Any) -> None:
    if not isinstance(batch, Mapping):
        raise WorldRunnerError("W1 batch must expose a true next-day target packet")
    if not any(
        key in batch
        for key in (
            "target_packet",
            "next_day_packet",
            "future_input_packet",
            "next_day_batch",
            "target_batch",
        )
    ):
        raise WorldRunnerError("W1 h96 target packet is missing")
    horizons = batch.get("latent_target_indices", (1, 4, 16, 96))
    if tuple(int(value) for value in horizons) != (1, 4, 16, 96):
        raise WorldRunnerError("W1 target indices must be exactly 1/4/16/96")
    available = batch.get("h96_target_available", True)
    if isinstance(available, torch.Tensor):
        if available.dtype is not torch.bool or not bool(available.all()):
            raise WorldRunnerError("W1 batch contains unavailable h96 targets")
    elif available is not True:
        raise WorldRunnerError("W1 batch contains unavailable h96 targets")


class WorldMoERunner:
    """Train one registered phase at a time and release only measured passes."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        model: nn.Module,
        corpus: Any,
        output_root: str | Path,
        device: str | torch.device,
        run_id: str,
        seed: int = 20240813,
        loss_builder: Callable[..., Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.model = model.to(device)
        self.corpus = corpus
        self.output_root = Path(output_root).resolve() / f"run={run_id}"
        self.device = torch.device(device)
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.loss_builder = loss_builder
        self.checkpoints = AtomicCheckpointStore(self.output_root, run_id=self.run_id)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

    def _phase_root(self, phase: str) -> Path:
        return self.output_root / f"phase={phase}"

    def _expert_assignment_contract(self) -> dict[str, Any]:
        model = self.config.get("model")
        raw = model if isinstance(model, Mapping) else {}
        return {
            "schema": str(
                raw.get("expert_assignment_schema", EXPERT_ASSIGNMENT_SCHEMA)
            ),
            "general_experts": int(raw.get("general_experts", 7)),
            "general_expert_indices": list(range(int(raw.get("general_experts", 7)))),
            "regular_top_k": int(raw.get("regular_top_k", 2)),
            "conditional_tail_expert": bool(raw.get("conditional_tail_expert", True)),
            "tail_expert_index": int(raw.get("general_experts", 7)),
            "tail_capacity_fraction": float(raw.get("tail_capacity_fraction", 0.05)),
            "tail_capacity_max_fraction": float(
                raw.get("tail_capacity_max_fraction", 1.0)
            ),
            "tail_min_tokens": int(raw.get("tail_min_tokens", 32)),
            "tail_risk_threshold_work_units": float(
                raw.get("tail_risk_threshold_work_units", 1.0e-4)
            ),
            "tail_activation_probability_threshold": float(
                raw.get("tail_activation_probability_threshold", 0.50)
            ),
            "tail_route_aux_weight": float(raw.get("tail_route_aux_weight", 0.05)),
            "tail_route_aux_temperature": float(
                raw.get("tail_route_aux_temperature", 1.0)
            ),
            "tail_gate_input": str(
                raw.get("tail_gate_input", "causal_issue_time_only")
            ),
            "tail_teacher_role": str(
                raw.get(
                    "tail_teacher_role",
                    "auxiliary_supervision_and_diagnostics_only",
                )
            ),
            "teacher_truth_can_activate_tail": False,
        }

    def _checkpoint_contract_extra(self) -> dict[str, Any]:
        assignment = self._expert_assignment_contract()
        if assignment["schema"] != EXPERT_ASSIGNMENT_SCHEMA:
            raise WorldRunnerError("active expert assignment schema is not registered")
        return {
            "expert_assignment_schema": EXPERT_ASSIGNMENT_SCHEMA,
            "expert_assignment": assignment,
        }

    def _assert_checkpoint_expert_assignment(
        self,
        path: Path,
        *,
        allow_v23_parent_upgrade: bool = False,
    ) -> Mapping[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise WorldRunnerError("checkpoint payload is malformed")
        extra = payload.get("extra")
        expected = self._checkpoint_contract_extra()
        exact = bool(
            isinstance(extra, Mapping)
            and all(extra.get(name) == value for name, value in expected.items())
        )
        registered_v23_upgrade = False
        if not exact and allow_v23_parent_upgrade and isinstance(extra, Mapping):
            actual_assignment = extra.get("expert_assignment")
            expected_assignment = expected["expert_assignment"]
            if isinstance(actual_assignment, Mapping) and isinstance(
                expected_assignment, Mapping
            ):
                changed = {
                    name
                    for name in set(actual_assignment) | set(expected_assignment)
                    if actual_assignment.get(name) != expected_assignment.get(name)
                }
                registered_v23_upgrade = bool(
                    extra.get("expert_assignment_schema") == EXPERT_ASSIGNMENT_SCHEMA
                    and changed == {"tail_capacity_fraction", "tail_min_tokens"}
                    and float(actual_assignment.get("tail_capacity_fraction", -1.0))
                    == 0.05
                    and int(actual_assignment.get("tail_min_tokens", -1)) == 32
                    and float(expected_assignment["tail_capacity_fraction"])
                    == 0.019967682659626007
                    and int(expected_assignment["tail_min_tokens"]) == 11
                )
        if not exact and not registered_v23_upgrade:
            raise WorldRunnerError(
                "checkpoint expert assignment is incompatible; start a new run "
                "instead of resuming a pre-tail-expert checkpoint"
            )
        return payload

    def _parent_phase(self, phase: str) -> str | None:
        normalized = str(phase).lower()
        index = PHASE_ORDER.index(normalized)
        return None if index == 0 else PHASE_ORDER[index - 1]

    @staticmethod
    def _derived_gate_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.resolve().parent.parent / "gate.json"

    def _validate_parent_gate(
        self,
        *,
        gate_path: Path,
        parent_phase: str,
        checkpoint_id: str,
        required: bool,
    ) -> dict[str, Any]:
        if not gate_path.is_file():
            if required:
                raise WorldRunnerError(
                    f"{parent_phase} parent gate is required but missing: {gate_path}"
                )
            return {
                "path": str(gate_path),
                "present": False,
                "release_verified": False,
            }
        raw = json.loads(gate_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise WorldRunnerError("parent gate payload must be a mapping")
        if str(raw.get("phase", "")).lower() != parent_phase:
            raise WorldRunnerError("parent gate phase does not match the checkpoint")
        gate_checkpoint_id = str(raw.get("checkpoint_id", ""))
        if gate_checkpoint_id and gate_checkpoint_id != checkpoint_id:
            raise WorldRunnerError(
                "parent gate selects a different checkpoint than the requested parent"
            )
        released = bool(
            raw.get("schema") == PHASE_RELEASE_SCHEMA
            and raw.get("status") == "PASS"
            and raw.get("passed") is True
            and raw.get("next_phase_released") is True
            and raw.get("locked_2025_opened") is False
            and gate_checkpoint_id == checkpoint_id
        )
        if parent_phase == "u0":
            released = bool(
                released
                and raw.get("checkpoint_selection_role")
                == "release_best_all_gates_eligible"
                and raw.get("release_candidate_existed") is True
            )
        if required and not released:
            raise WorldRunnerError(f"{parent_phase} parent gate did not release the child")
        return {
            "path": str(gate_path.resolve()),
            "present": True,
            "release_verified": released,
            "status": str(raw.get("status", "")),
            "checkpoint_id": gate_checkpoint_id or None,
            "checkpoint_selection_role": raw.get("checkpoint_selection_role"),
        }

    def _local_selected_parent_path(self, child_phase: str) -> Path:
        parent_phase = self._parent_phase(child_phase)
        if parent_phase is None:
            raise WorldRunnerError("W0 has no parent checkpoint")
        state_path = self._phase_root(parent_phase) / "state.json"
        if not state_path.is_file():
            raise WorldRunnerError(
                f"{child_phase} requires a selected {parent_phase} checkpoint; "
                "for a new cross-run lineage pass --parent-checkpoint"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != RUN_STATE_SCHEMA
            or str(state.get("phase", "")).lower() != parent_phase
        ):
            raise WorldRunnerError("parent phase state is malformed")
        raw_path = str(state.get("best_checkpoint", "")).strip()
        if not raw_path:
            raise WorldRunnerError("parent phase state omitted its selected checkpoint")
        checkpoint_root = (self._phase_root(parent_phase) / "checkpoints").resolve()
        selected = Path(raw_path)
        if not selected.is_file():
            selected = checkpoint_root / Path(raw_path).name
        resolved = selected.resolve()
        if resolved.parent != checkpoint_root or not resolved.is_file():
            raise WorldRunnerError(
                "parent selected checkpoint is missing or outside its phase checkpoint root"
            )
        return resolved

    def _initialize_from_parent(
        self,
        child_phase: str,
        *,
        checkpoint_path: str | Path | None,
        gate_path: str | Path | None,
        require_release: bool,
    ) -> dict[str, Any]:
        """Load only parent model weights and return immutable lineage evidence."""

        parent_phase = self._parent_phase(child_phase)
        if parent_phase is None:
            raise WorldRunnerError("W0 cannot be initialized from a parent checkpoint")
        selected = (
            self._local_selected_parent_path(child_phase)
            if checkpoint_path is None
            else Path(checkpoint_path).expanduser().resolve()
        )
        if not selected.is_file():
            raise WorldRunnerError(f"parent checkpoint is missing: {selected}")
        checkpoint_id = _sha256(selected)
        allow_v23_parent_upgrade = bool(
            str(child_phase).lower() == "u0"
            and self.config.get("schema_version")
            in {
                R5_V23_SCHEMA_VERSION,
                R5_V24_SCHEMA_VERSION,
                R5_V25_SCHEMA_VERSION,
            }
        )
        allow_v24_support_upgrade = bool(
            str(child_phase).lower() == "u0"
            and self.config.get("schema_version")
            in {R5_V24_SCHEMA_VERSION, R5_V25_SCHEMA_VERSION}
        )
        allow_v25_distribution_upgrade = bool(
            str(child_phase).lower() == "u0"
            and self.config.get("schema_version") == R5_V25_SCHEMA_VERSION
        )
        payload = self._assert_checkpoint_expert_assignment(
            selected,
            allow_v23_parent_upgrade=allow_v23_parent_upgrade,
        )
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise WorldRunnerError("parent checkpoint schema mismatch")
        if str(payload.get("phase", "")).lower() != parent_phase:
            raise WorldRunnerError(
                f"{child_phase} requires a {parent_phase} checkpoint, got "
                f"{payload.get('phase')!r}"
            )
        if payload.get("locked_2025_opened") is not False:
            raise WorldRunnerError("parent checkpoint does not preserve the 2025 lock")
        if allow_v24_support_upgrade:
            training = self.config.get("training")
            registered_parent = (
                training.get("parent_checkpoint")
                if isinstance(training, Mapping)
                else None
            )
            source_state = payload.get("resume_state")
            if not isinstance(registered_parent, Mapping) or not isinstance(
                source_state, Mapping
            ):
                raise WorldRunnerError(
                "R5-V2.4+ requires a registered W1 parent identity"
                )
            parent_identity_matches = bool(
                checkpoint_id == str(registered_parent.get("sha256", ""))
                and str(payload.get("run_id", ""))
                == str(registered_parent.get("run_id", ""))
                and str(payload.get("phase", "")).lower()
                == str(registered_parent.get("phase", "")).lower()
                and selected.stem == str(registered_parent.get("role", ""))
                and int(source_state.get("epoch", -1))
                == int(registered_parent.get("epoch", -2))
                and int(source_state.get("update", -1))
                == int(registered_parent.get("update", -2))
            )
            if not parent_identity_matches:
                raise WorldRunnerError(
                    "registered parent differs from the PASS W1 epoch-56 "
                    "lineage"
                )
        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise WorldRunnerError("parent checkpoint omitted model_state")
        resolved_gate = (
            self._derived_gate_path(selected)
            if gate_path is None
            else Path(gate_path).expanduser().resolve()
        )
        gate_identity = self._validate_parent_gate(
            gate_path=resolved_gate,
            parent_phase=parent_phase,
            checkpoint_id=checkpoint_id,
            required=require_release,
        )
        # Cross-phase initialization intentionally excludes optimizer and RNG.
        # R5-V2.2 adds one zero-initialized action-delta projection, exposed
        # under two shared-module state-dict paths.  Permit only that exact
        # forward-compatible upgrade from the frozen V2.1 W0 parent.
        initialized_state = self.model.state_dict()
        v25_allowed_prefixes: tuple[str, ...] = ()
        if allow_v25_distribution_upgrade:
            training = self.config.get("training")
            upgrade = (
                training.get("parent_model_upgrade")
                if isinstance(training, Mapping)
                else None
            )
            raw_prefixes = (
                upgrade.get("allowed_missing_prefixes")
                if isinstance(upgrade, Mapping)
                else None
            )
            if (
                not isinstance(raw_prefixes, Sequence)
                or isinstance(raw_prefixes, (str, bytes))
                or not raw_prefixes
            ):
                raise WorldRunnerError(
                    "R5-V2.5 requires registered parent-upgrade prefixes"
                )
            v25_allowed_prefixes = tuple(str(value) for value in raw_prefixes)
            if any(not value or value.strip() != value for value in v25_allowed_prefixes):
                raise WorldRunnerError("R5-V2.5 parent-upgrade prefixes are invalid")
        initialized_v25_state = {
            name: value.detach().clone()
            for name, value in initialized_state.items()
            if any(name.startswith(prefix) for prefix in v25_allowed_prefixes)
        }
        if allow_v25_distribution_upgrade and not initialized_v25_state:
            raise WorldRunnerError(
                "R5-V2.5 model exposes none of its registered new parameters"
            )
        initialized_support_state = {
            name: initialized_state[name].detach().clone()
            for name in R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS
            if name in initialized_state
        }
        if allow_v24_support_upgrade and set(initialized_support_state) != set(
            R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS
        ):
            absent = sorted(
                set(R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS)
                - set(initialized_support_state)
            )
            raise WorldRunnerError(
                "R5-V2.4 model omitted registered support-atom parameters: "
                f"{absent}"
            )
        incompatible = self.model.load_state_dict(model_state, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        allowed_missing = set(ALLOWED_PARENT_MODEL_UPGRADE_MISSING_KEYS)
        if allow_v24_support_upgrade:
            allowed_missing.update(R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS)
        if allow_v25_distribution_upgrade:
            allowed_missing.update(initialized_v25_state)
        if unexpected or not missing.issubset(allowed_missing):
            raise WorldRunnerError(
                "parent model upgrade is incompatible; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if allow_v24_support_upgrade and not set(
            R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS
        ).issubset(missing):
            raise WorldRunnerError(
                "R5-V2.4 formal W1 parent unexpectedly contains only part of the "
                "registered support-atom upgrade"
            )
        nonzero_new = [
            name
            for name in missing & set(ALLOWED_PARENT_MODEL_UPGRADE_MISSING_KEYS)
            if name not in initialized_state
            or bool(torch.count_nonzero(initialized_state[name].detach()))
            or bool(torch.count_nonzero(self.model.state_dict()[name].detach()))
        ]
        if nonzero_new:
            raise WorldRunnerError(
                "new parent-upgrade parameters must retain exact zero initialization: "
                f"{sorted(nonzero_new)}"
            )
        changed_support = [
            name
            for name in missing & set(R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS)
            if not torch.equal(
                initialized_support_state[name],
                self.model.state_dict()[name].detach(),
            )
        ]
        if changed_support:
            raise WorldRunnerError(
                "R5-V2.4 support-atom parameters changed while loading the W1 "
                f"parent: {sorted(changed_support)}"
            )
        changed_v25 = [
            name
            for name in missing & set(initialized_v25_state)
            if not torch.equal(
                initialized_v25_state[name],
                self.model.state_dict()[name].detach(),
            )
        ]
        if changed_v25:
            raise WorldRunnerError(
                "R5-V2.5 new distribution parameters changed while loading the "
                f"W1 parent: {sorted(changed_v25)}"
            )
        self.model.to(self.device)
        source_state = payload.get("resume_state")
        return {
            "schema": PARENT_CHECKPOINT_SCHEMA,
            "child_phase": str(child_phase).lower(),
            "source_run_id": str(payload.get("run_id", "")),
            "source_phase": parent_phase,
            "source_role": selected.stem,
            "source_selection_role": gate_identity.get(
                "checkpoint_selection_role"
            ),
            "path": str(selected),
            "sha256": checkpoint_id,
            "source_epoch": (
                int(source_state.get("epoch", 0))
                if isinstance(source_state, Mapping)
                else None
            ),
            "source_update": (
                int(source_state.get("update", 0))
                if isinstance(source_state, Mapping)
                else None
            ),
            "optimizer_loaded": False,
            "rng_loaded": False,
            "model_upgrade": {
                "missing_keys": sorted(missing),
                "unexpected_keys": sorted(unexpected),
                "missing_keys_zero_initialized": not nonzero_new,
                "support_atom_missing_keys": sorted(
                    missing & set(R5_V24_SUPPORT_ATOM_UPGRADE_MISSING_KEYS)
                ),
                "support_atom_train_prior_initialization_preserved": not (
                    changed_support
                ),
                "distribution_native_missing_keys": sorted(
                    missing & set(initialized_v25_state)
                ),
                "distribution_native_initialization_preserved": not changed_v25,
            },
            "expert_assignment_upgrade": (
                {
                    "registered": True,
                    "reason": (
                        "R5-V2.3+ train-prevalence tail-capacity "
                        "calibration"
                    ),
                    "source_tail_capacity_fraction": 0.05,
                    "source_tail_min_tokens": 32,
                    "target_tail_capacity_fraction": 0.019967682659626007,
                    "target_tail_min_tokens": 11,
                    "parameter_keys_changed": False,
                }
                if allow_v23_parent_upgrade
                else None
            ),
            "gate": gate_identity,
            "loaded_at_utc": _utc_now(),
        }

    def _resume_parent_identity(
        self,
        child_phase: str,
        raw: Any,
        *,
        require_release: bool,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise WorldRunnerError(
                f"{child_phase} resume checkpoint predates explicit parent lineage; "
                "start a new run from --parent-checkpoint"
            )
        identity = dict(raw)
        expected_parent = self._parent_phase(child_phase)
        gate = identity.get("gate")
        if (
            identity.get("schema") != PARENT_CHECKPOINT_SCHEMA
            or str(identity.get("child_phase", "")).lower() != child_phase
            or str(identity.get("source_phase", "")).lower() != expected_parent
            or not str(identity.get("sha256", "")).strip()
            or identity.get("optimizer_loaded") is not False
            or identity.get("rng_loaded") is not False
            or not isinstance(gate, Mapping)
        ):
            raise WorldRunnerError("resume parent-checkpoint lineage is malformed")
        if require_release and gate.get("release_verified") is not True:
            raise WorldRunnerError("resume parent checkpoint was not release-gate verified")
        return identity

    def _w0_state_thresholds(self) -> dict[str, float]:
        capability = self.config.get("capability_gates")
        world_state = (
            capability.get("world_state", {}) if isinstance(capability, Mapping) else {}
        )
        if not isinstance(world_state, Mapping):
            raise WorldRunnerError("capability_gates.world_state must be a mapping")
        thresholds = dict(DEFAULT_W0_STATE_THRESHOLDS)
        for name in thresholds:
            if name not in world_state:
                raise WorldRunnerError(f"capability_gates.world_state omitted {name}")
            thresholds[name] = float(world_state[name])
        return thresholds

    def _batches(
        self,
        split: str,
        spec: PhaseSpec,
        *,
        epoch: int,
        shard_index: int = 0,
        shard_count: int = 1,
        start_batch: int = 0,
        batch_ordinals: Sequence[int] | None = None,
    ) -> Iterable[Any]:
        if split in {"final_test", "test", "2025"}:
            raise WorldRunnerError("the development runner cannot request 2025")
        callback = getattr(self.corpus, "batches", None)
        if not callable(callback):
            raise WorldRunnerError("WorldMoECorpus must expose batches()")
        batch_size = (
            spec.batch_size
            if split == "train"
            else self._validation_batch_size(spec)
        )
        result = _call_supported(
            callback,
            split,
            batch_size=batch_size,
            device=self.device,
            phase=spec.name,
            shuffle=split == "train",
            seed=self.seed + epoch,
            shuffle_seed=(self.seed + epoch if split == "train" else None),
            include_next_day=spec.name in {"w1", "u0", "j1"},
            require_h96=spec.name in {"w1", "u0", "j1"},
            shard_index=shard_index if split == "train" else 0,
            shard_count=shard_count if split == "train" else 1,
            start_batch=start_batch if split == "train" else 0,
            batch_ordinals=batch_ordinals if split == "validation" else None,
        )
        if not isinstance(result, Iterable):
            raise WorldRunnerError("corpus.batches() must return an iterable")
        return result

    def _validation_batch_size(self, spec: PhaseSpec) -> int:
        value = int(spec.effective_validation_batch_size)
        if value <= 0:
            raise WorldRunnerError("validation_batch_size must be positive")
        return value

    def _prepare_batch(self, batch: Any, *, phase: str) -> Any:
        assert_development_batch(batch, label=f"{phase} batch")
        if phase in {"w1", "j1"}:
            assert_w1_h96_target(batch)
        elif isinstance(batch, Mapping):
            # A target packet triggers a second backbone encode.  It is a W1-only
            # supervision input and must not consume W0/U0/P1/J1 memory or compute.
            batch = {
                key: value
                for key, value in batch.items()
                if key
                not in ("target_packet", "next_day_packet", "future_input_packet")
            }
        packetize = getattr(self.corpus, "world_input_packet", None)
        if not callable(packetize) or not isinstance(batch, Mapping):
            return batch
        if any(key in batch for key in ("world_input", "input_packet", "packet")):
            return batch
        packet = _call_supported(packetize, batch)
        result = dict(batch)
        result["world_input"] = packet
        if phase in {"w1", "j1"}:
            target_raw = batch.get("next_day_batch", batch.get("target_batch"))
            if target_raw is not None and "target_packet" not in result:
                result["target_packet"] = _call_supported(packetize, target_raw)
        return result

    def _validation_batch_count(self, spec: PhaseSpec) -> int:
        callback = getattr(self.corpus, "batch_count", None)
        if not callable(callback):
            raise WorldRunnerError(
                "WorldMoECorpus must expose packet-free batch_count() for sentinel validation"
            )
        result = _call_supported(
            callback,
            "validation",
            batch_size=self._validation_batch_size(spec),
            require_h96=spec.name in {"w1", "u0", "j1"},
        )
        count = int(result)
        if count <= 0:
            raise WorldRunnerError("validation batch_count() must be positive")
        return count

    def _validation_loss(
        self,
        trainer: WorldModelTrainer,
        *,
        batch_ordinals: Sequence[int] | None = None,
    ) -> float | ValidationSummary:
        values: list[float] = []
        origin_counts: list[float] = []
        score_batches: list[Mapping[str, ScoreTerm]] = []
        batch_term_values: dict[str, list[float]] = {}
        batch_term_counts: dict[str, list[float]] = {}
        diagnostic_batches: list[Mapping[str, Any]] = []
        for index, raw_batch in enumerate(
            self._batches(
                "validation",
                trainer.spec,
                epoch=0,
                batch_ordinals=batch_ordinals,
            )
        ):
            if batch_ordinals is not None and index >= len(batch_ordinals):
                raise WorldRunnerError(
                    "corpus ignored the fixed sentinel batch ordinal selection"
                )
            batch = self._prepare_batch(raw_batch, phase=trainer.spec.name)
            validation_ordinal = (
                index if batch_ordinals is None else int(batch_ordinals[index])
            )
            evaluate_diagnostics = getattr(
                trainer, "evaluate_loss_and_diagnostics", None
            )
            if callable(evaluate_diagnostics):
                captured_output: list[Any] = []
                capture = trainer.model.register_forward_hook(
                    lambda _module, _args, value: captured_output.append(value)
                )
                try:
                    result, diagnostics = _call_supported(
                        evaluate_diagnostics,
                        batch,
                        seed=self.seed + 10_000 + validation_ordinal,
                        formal_seed_offsets=(
                            U0_FORMAL_SEED_OFFSETS
                            if trainer.spec.name == "u0" and batch_ordinals is None
                            else None
                        ),
                    )
                finally:
                    capture.remove()
                if captured_output:
                    diagnostics = dict(diagnostics)
                    diagnostics.update(
                        _tail_routing_batch_diagnostics(
                            captured_output[-1],
                            batch,
                            occurrence_threshold=self._expert_assignment_contract()[
                                "tail_risk_threshold_work_units"
                            ],
                        )
                    )
                if trainer.spec.name == "w0" and result.score_terms:
                    if len(captured_output) != 1:
                        raise WorldRunnerError(
                            "formal W0 validation must capture exactly one model output"
                        )
                    diagnostics = dict(diagnostics)
                    diagnostics.update(
                        _w0_state_batch_diagnostics(
                            captured_output[0],
                            batch,
                            occurrence_threshold=self._w0_state_thresholds()[
                                "queue_occurrence_threshold_work_units"
                            ],
                            occurrence_probability_threshold=(
                                self._w0_state_thresholds()[
                                    "queue_occurrence_probability_threshold"
                                ]
                            ),
                        )
                    )
            else:
                result = trainer.evaluate_loss(
                    batch, seed=self.seed + 10_000 + validation_ordinal
                )
                diagnostics = {}
            values.append(float(result.total.detach().float().cpu()))
            count = float(diagnostics.get("_origin_count", 1.0))
            if count <= 0.0 or not np.isfinite(count):
                raise WorldRunnerError("validation origin count is invalid")
            origin_counts.append(count)
            diagnostic_batches.append(dict(diagnostics))
            if result.score_terms:
                score_batches.append(result.score_terms)
                for name, term in result.score_terms.items():
                    batch_term_values.setdefault(name, []).append(
                        float(term.scalar.detach().float().cpu())
                    )
                    batch_term_counts.setdefault(name, []).append(
                        float(term.valid_count.detach().float().cpu())
                    )
        if batch_ordinals is not None and len(values) != len(batch_ordinals):
            raise WorldRunnerError(
                "corpus did not return every fixed sentinel validation batch"
            )
        if not values or not np.isfinite(values).all():
            raise WorldRunnerError("validation produced no finite phase loss")
        if not score_batches:
            # Compatibility path for synthetic unit-test loss builders only.
            # Formal R5-V2 losses always expose exact ScoreTerms.
            return float(np.average(values, weights=origin_counts))
        if len(score_batches) != len(values):
            raise WorldRunnerError(
                "formal validation omitted ScoreTerms in some batches"
            )
        expected_names = set(score_batches[0])
        if any(set(batch) != expected_names for batch in score_batches[1:]):
            raise WorldRunnerError("formal ScoreTerm keys changed across batches")
        aggregate = aggregate_score_terms(score_batches)
        diagnostics: dict[str, Any] = {}
        diagnostic_names = set().union(*(batch.keys() for batch in diagnostic_batches))
        for name in sorted(diagnostic_names - {"_origin_count"}):
            if name.startswith("_tail_hist_"):
                raw_histograms = [batch.get(name) for batch in diagnostic_batches]
                arrays = []
                for value in raw_histograms:
                    try:
                        array = np.asarray(value, dtype=np.float64)
                    except (TypeError, ValueError) as exc:
                        raise WorldRunnerError(
                            f"validation diagnostic {name} has an invalid histogram"
                        ) from exc
                    if (
                        array.shape != (TAIL_ROUTE_AUPRC_BINS,)
                        or not np.isfinite(array).all()
                        or bool((array < 0.0).any())
                    ):
                        raise WorldRunnerError(
                            f"validation diagnostic {name} has an invalid histogram"
                        )
                    arrays.append(array)
                diagnostics[name] = tuple(np.stack(arrays).sum(axis=0).tolist())
                continue
            if name.startswith(("_coverage_daily_", "_atom_daily_")):
                raw_daily = [batch.get(name) for batch in diagnostic_batches]
                if any(
                    not isinstance(value, Sequence)
                    or isinstance(value, (str, bytes))
                    for value in raw_daily
                ):
                    raise WorldRunnerError(
                        f"validation diagnostic {name} omitted daily coverage values"
                    )
                flattened = tuple(
                    float(item) for value in raw_daily for item in value
                )
                if not flattened or not np.isfinite(flattened).all():
                    raise WorldRunnerError(
                        f"validation diagnostic {name} has invalid daily coverage"
                    )
                diagnostics[name] = flattened
                continue
            pairs = [
                (float(batch[name]), count)
                for batch, count in zip(diagnostic_batches, origin_counts, strict=True)
                if name in batch
            ]
            if len(pairs) != len(diagnostic_batches):
                raise WorldRunnerError(
                    f"validation diagnostic {name} is missing from some batches"
                )
            raw = np.asarray([item[0] for item in pairs], dtype=np.float64)
            if not np.isfinite(raw).all():
                raise WorldRunnerError(f"validation diagnostic {name} is non-finite")
            if name.endswith("_count") or name.endswith("_sum"):
                diagnostics[name] = float(raw.sum())
            elif name.endswith("_max") or name.endswith("_max_abs"):
                diagnostics[name] = float(raw.max())
            else:
                diagnostics[name] = float(
                    np.average(raw, weights=[item[1] for item in pairs])
                )
        if trainer.spec.name == "w0":
            diagnostics = _finalize_w0_state_diagnostics(diagnostics)
        diagnostics = _finalize_tail_routing_diagnostics(diagnostics)
        return summarize_validation(
            trainer.spec.name,
            aggregate,
            batch_term_values=batch_term_values,
            batch_term_counts=batch_term_counts,
            diagnostics=diagnostics,
            batch_count=len(values),
            w0_state_thresholds=(
                self._w0_state_thresholds() if trainer.spec.name == "w0" else None
            ),
        )

    def _previous_phase_release(self, phase: str) -> None:
        index = PHASE_ORDER.index(phase)
        if index == 0:
            return
        previous = PHASE_ORDER[index - 1]
        gate_path = self._phase_root(previous) / "gate.json"
        if not gate_path.is_file():
            raise WorldRunnerError(f"{phase} is blocked until {previous} gate passes")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("schema") != PHASE_RELEASE_SCHEMA
            or gate.get("status") != "PASS"
            or gate.get("passed") is not True
            or gate.get("next_phase_released") is not True
            or gate.get("locked_2025_opened") is not False
        ):
            raise WorldRunnerError(f"{phase} is blocked by {previous} gate")
        if str(phase).lower() == "p1":
            release_path = self.checkpoints.path("u0", "release_best")
            if (
                gate.get("checkpoint_selection_role")
                != "release_best_all_gates_eligible"
                or gate.get("release_candidate_existed") is not True
                or not release_path.is_file()
                or gate.get("checkpoint_id") != _sha256(release_path)
            ):
                raise WorldRunnerError(
                    "p1 requires the SHA-bound eligible U0 release_best checkpoint"
                )

    def registered_phase_gate_callback(
        self,
        *,
        phase: str,
        checkpoint_path: str | Path,
        checkpoint_id: str,
        **_: Any,
    ) -> GateResult:
        """Turn the selected formal summary into the native phase release gate."""

        resolved_checkpoint = Path(checkpoint_path)
        payload = self._assert_checkpoint_expert_assignment(resolved_checkpoint)
        if not isinstance(payload, Mapping):
            raise WorldRunnerError("selected checkpoint payload is malformed")
        extra = payload.get("extra")
        formal = extra.get("formal_selection") if isinstance(extra, Mapping) else None
        if not isinstance(formal, Mapping):
            raise WorldRunnerError(
                "selected checkpoint omitted its formal validation summary"
            )
        return evaluate_phase_release_gate(
            phase=str(phase),
            checkpoint_id=str(checkpoint_id),
            formal_selection=formal,
        )

    def run_phase(
        self,
        phase: str,
        **kwargs: Any,
    ) -> PhaseRunResult:
        self._active_recovery_callback = None
        try:
            return self._run_phase(phase, **kwargs)
        except BaseException as exc:
            recover = self._active_recovery_callback
            if recover is not None:
                try:
                    recover(exc)
                except BaseException as recovery_error:
                    exc.add_note(f"Emergency checkpoint failed: {recovery_error}")
            raise
        finally:
            self._active_recovery_callback = None

    def _run_phase(
        self,
        phase: str,
        *,
        resume: bool = False,
        parent_checkpoint: str | Path | None = None,
        parent_gate: str | Path | None = None,
        gate_callback: Callable[..., GateResult | Mapping[str, Any]] | None = None,
        smoke: bool = False,
        ignore_phase_gate: bool = False,
        ignore_formal_gates: bool = False,
    ) -> PhaseRunResult:
        normalized = str(phase).lower()
        if normalized == "c1":
            raise WorldRunnerError("use run_c1() for closed-loop validation")
        if normalized not in PHASE_ORDER:
            raise WorldRunnerError(f"unknown phase {phase!r}")
        if normalized == "p1" and ignore_phase_gate:
            raise WorldRunnerError("P1 cannot bypass the eligible U0 release gate")
        if resume and (parent_checkpoint is not None or parent_gate is not None):
            raise WorldRunnerError(
                "resume uses the parent identity embedded in latest.pt; "
                "do not pass a new parent checkpoint"
            )
        if parent_gate is not None and parent_checkpoint is None:
            raise WorldRunnerError("parent_gate requires an explicit parent_checkpoint")
        if normalized == "w0" and (
            parent_checkpoint is not None or parent_gate is not None
        ):
            raise WorldRunnerError("W0 cannot receive a parent checkpoint")
        spec = phase_spec_from_config(self.config, normalized)
        schedule = efficient_phase_schedule(self.config, normalized, spec)
        score_only = schedule.enabled and schedule.selection_policy == "validation_score_only"
        best_only = schedule.enabled and schedule.checkpoint_policy == "best_only"
        parent_identity: dict[str, Any] | None = None
        if normalized != "w0" and not resume and not smoke:
            parent_identity = self._initialize_from_parent(
                normalized,
                checkpoint_path=parent_checkpoint,
                gate_path=parent_gate,
                require_release=not ignore_phase_gate and not score_only,
            )
        trainer_kwargs = (
            {} if self.loss_builder is None else {"loss_builder": self.loss_builder}
        )
        trainer_kwargs.update(
            _configured_weight_kwargs(
                self.config, synthetic_loss_builder=self.loss_builder is not None
            )
        )
        trainer = WorldModelTrainer(
            self.model,
            spec,
            device=self.device,
            **trainer_kwargs,
        )
        state = ResumeState(
            phase=normalized, short_epoch_factor=spec.short_epoch_factor
        )
        resume_payload: Mapping[str, Any] | None = None
        if resume:
            resume_payload = self._assert_checkpoint_expert_assignment(
                self.checkpoints.path(normalized, "latest")
            )
            state = self.checkpoints.load(
                normalized,
                model=self.model,
                optimizer=trainer.optimizer,
                map_location="cpu",
            )
            self.model.to(self.device)
        if state.short_epoch_factor != spec.short_epoch_factor:
            raise WorldRunnerError(
                "checkpoint short_epoch_factor differs from the active configuration"
            )
        if state.batch_in_short_epoch and state.epoch >= spec.maximum_epochs:
            raise WorldRunnerError("partial checkpoint lies beyond maximum short epoch")
        phase_root = self._phase_root(normalized)
        state_path = phase_root / "state.json"
        log_path = phase_root / "train.jsonl"
        early_formal_interval, early_formal_until = formal_validation_schedule(
            self.config, normalized, spec
        )
        (
            failure_plateau_enabled,
            failure_plateau_evaluations,
            failure_relative_improvement,
            failure_terminal_status,
        ) = _u0_failure_plateau_policy(self.config)
        validation_batches = self._validation_batch_count(spec)
        sentinel_ordinals = (
            () if schedule.enabled else stratified_validation_ordinals(
                validation_batches, spec.sentinel_validation_batches
            )
        )
        sentinel_hash = validation_ordinal_hash(
            sentinel_ordinals,
            total_batches=validation_batches,
            batch_size=self._validation_batch_size(spec),
            phase=normalized,
        )
        raw_resume_extra = (
            resume_payload.get("extra") if isinstance(resume_payload, Mapping) else None
        )
        timing = {
            "train_step_seconds": 0.0, "train_step_count": 0,
            "validation_seconds": 0.0, "validation_count": 0,
            "checkpoint_seconds": 0.0, "checkpoint_write_count": 0,
            "checkpoint_alias_seconds": 0.0, "checkpoint_alias_count": 0,
        }
        checkpoint_timing = (
            raw_resume_extra.get("timing", {}) if isinstance(raw_resume_extra, Mapping) else {}
        )
        if isinstance(checkpoint_timing, Mapping):
            timing.update({k: checkpoint_timing[k] for k in timing if k in checkpoint_timing})
        timing_path = phase_root / "timing.json"
        if resume and schedule.enabled and timing_path.is_file():
            persisted_timing = json.loads(timing_path.read_text(encoding="utf-8"))
            identity = {k: getattr(state, k) for k in ("epoch", "batch_in_short_epoch", "update")}
            if (
                persisted_timing.get("run_id") == self.run_id
                and persisted_timing.get("checkpoint_state") == identity
            ):
                values = persisted_timing.get("timing", {})
                timing.update({k: values[k] for k in timing if k in values})

        def timed_validation(*args: Any, **kwargs: Any) -> float | ValidationSummary:
            started = time.perf_counter()
            try:
                return self._validation_loss(*args, **kwargs)
            finally:
                timing["validation_seconds"] += time.perf_counter() - started
                timing["validation_count"] += 1

        if resume and normalized != "w0":
            parent_identity = self._resume_parent_identity(
                normalized,
                (
                    raw_resume_extra.get("parent_checkpoint")
                    if isinstance(raw_resume_extra, Mapping)
                    else None
                ),
                require_release=not ignore_phase_gate and not score_only,
            )
        zero_update_reference: dict[str, Any] | None = None
        deferred_initial_snapshot: Mapping[str, Any] | None = None
        is_v23 = self.config.get("schema_version") == R5_V23_SCHEMA_VERSION

        def register_w1_reference(zero_result: float | ValidationSummary) -> None:
            nonlocal state, zero_update_reference
            if not isinstance(zero_result, ValidationSummary):
                raise WorldRunnerError(
                    "formal W1 zero-update validation must expose proper-score horizons"
                )
            raw_horizons = zero_result.selection_metrics.get("horizon_scores")
            if not isinstance(raw_horizons, Mapping) or set(raw_horizons) != {
                "h1",
                "h4",
                "h16",
                "h96",
            }:
                raise WorldRunnerError(
                    "W1 zero-update validation omitted exact h1/h4/h16/h96 scores"
                )
            horizon_reference = {
                name: float(raw_horizons[name])
                for name in ("h1", "h4", "h16", "h96")
            }
            if not all(
                np.isfinite(value) and value > 0.0
                for value in horizon_reference.values()
            ):
                raise WorldRunnerError("W1 zero-update horizon scores must be finite and positive")
            state = replace(
                state,
                selection_reference_json=json.dumps(
                    horizon_reference, sort_keys=True, separators=(",", ":")
                ),
            )
            zero_update_reference = {
                "schema": "dcvpp.vpp-world-moe.w1-zero-update-reference.v1",
                "phase": "w1",
                "updates_completed": 0,
                "before_any_optimizer_update": True,
                "evaluated_from_frozen_initial_state": deferred_initial_snapshot is not None,
                "evaluation_deferred_until_epoch": (
                    schedule.first_formal_epoch if deferred_initial_snapshot is not None else 0
                ),
                "parent_checkpoint": parent_identity,
                "horizon_scores": horizon_reference,
                "formal_summary": asdict(zero_result),
                "measured_at_utc": _utc_now(),
            }
            _atomic_json(
                self._phase_root("w1") / "zero_update_reference.json",
                zero_update_reference,
            )

        if resume and (normalized == "w1" or (normalized == "u0" and is_v23)):
            raw_zero_reference = (
                raw_resume_extra.get("zero_update_reference")
                if isinstance(raw_resume_extra, Mapping) else None
            )
            raw_deferred = (
                raw_resume_extra.get("deferred_w1_initial_snapshot")
                if isinstance(raw_resume_extra, Mapping) else None
            )
            if isinstance(raw_zero_reference, Mapping):
                zero_update_reference = dict(raw_zero_reference)
            elif schedule.enabled and normalized == "w1" and isinstance(raw_deferred, Mapping):
                deferred_initial_snapshot = raw_deferred
            else:
                raise WorldRunnerError(
                    f"{normalized.upper()} resume checkpoint has neither zero-update "
                    "reference nor frozen initial snapshot"
                )
        elif normalized == "w1" and not smoke:
            if schedule.enabled and schedule.defer_w1_reference:
                deferred_initial_snapshot = _cpu_model_snapshot(self.model)
            else:
                register_w1_reference(timed_validation(trainer))
        elif normalized == "u0" and is_v23 and not smoke:
            zero_result = timed_validation(trainer)
            if not isinstance(zero_result, ValidationSummary):
                raise WorldRunnerError(
                    "R5-V2.3 U0 zero-update validation must expose formal proper scores"
                )
            data_config = self.config.get("data")
            data_mapping = data_config if isinstance(data_config, Mapping) else {}
            normalization = data_mapping.get("normalization")
            normalization_mapping = (
                normalization if isinstance(normalization, Mapping) else {}
            )
            zero_update_reference = {
                "schema": "dcvpp.vpp-world-moe.u0-zero-update-reference.v1",
                "phase": "u0",
                "updates_completed": 0,
                "before_any_optimizer_update": True,
                "parent_checkpoint": parent_identity,
                "model_state_sha256": _model_state_sha256(self.model),
                "configuration_sha256": _mapping_sha256(self.config),
                "data_lineage_path": data_mapping.get("lineage_path"),
                "normalization_artifact_path": normalization_mapping.get(
                    "artifact_path"
                ),
                "formal_paths": 100,
                "formal_seed_offsets": list(U0_FORMAL_SEED_OFFSETS),
                "formal_summary": asdict(zero_result),
                "locked_2025_opened": False,
                "measured_at_utc": _utc_now(),
            }
            _atomic_json(
                self._phase_root("u0") / "zero_update_reference.json",
                zero_update_reference,
            )
        raw_training_control = (
            raw_resume_extra.get("training_control")
            if isinstance(raw_resume_extra, Mapping)
            else None
        )
        efficient_stale_evaluations = int(
            raw_training_control.get("efficient_stale_evaluations", 0)
            if isinstance(raw_training_control, Mapping) else 0
        )
        formal_primary_record: dict[str, Any] | None = None
        pareto_frontier: list[dict[str, Any]] = []
        pareto_stale_evaluations = 0
        w1_dynamics_gradient_nonzero = False
        w1_dynamics_parameters_updated = False
        if isinstance(raw_training_control, Mapping):
            raw_primary = raw_training_control.get("formal_primary_record")
            if isinstance(raw_primary, Mapping):
                formal_primary_record = dict(raw_primary)
            raw_frontier = raw_training_control.get("state_pareto_frontier", ())
            if isinstance(raw_frontier, Sequence) and not isinstance(
                raw_frontier, (str, bytes)
            ):
                pareto_frontier = [
                    dict(item) for item in raw_frontier if isinstance(item, Mapping)
                ]
            pareto_stale_evaluations = int(
                raw_training_control.get("pareto_stale_evaluations", 0)
            )
            w1_dynamics_gradient_nonzero = bool(
                raw_training_control.get("w1_dynamics_gradient_nonzero", False)
            )
            w1_dynamics_parameters_updated = bool(
                raw_training_control.get("w1_dynamics_parameters_updated", False)
            )
        if formal_primary_record is None and state.selected_record_json:
            formal_primary_record = dict(json.loads(state.selected_record_json))
        if score_only and formal_primary_record is not None:
            # A user-authorized policy change on resume must inherit the existing
            # formal score winner without replacing current optimizer/RNG/weights.
            state = replace(
                state,
                best_validation=float(formal_primary_record["primary_score"]),
                selected_record_json=json.dumps(formal_primary_record, sort_keys=True, separators=(",", ":")),
            )
            pareto_frontier = []
            pareto_stale_evaluations = 0
        if not score_only and normalized in {"w0", "u0"} and not pareto_frontier and formal_primary_record:
            vector = (
                _w0_pareto_vector(formal_primary_record)
                if normalized == "w0"
                else _u0_pareto_vector(formal_primary_record)
            )
            if vector is not None:
                pareto_frontier = [dict(formal_primary_record)]
        stop_details: dict[str, Any] = {"stopped_early": False}
        stop_on_resume = False
        prior_schedule = (
            raw_training_control.get("efficient_schedule", {})
            if isinstance(raw_training_control, Mapping) else {}
        )
        policy_keys = ("first_formal_epoch", "early_stop_start_epoch", "early_stop_evaluations", "relative_improvement")
        if (
            resume and score_only and normalized == "w1" and not schedule.fixed_budget
            and isinstance(prior_schedule, Mapping)
            and any(prior_schedule.get(key) != getattr(schedule, key) for key in policy_keys)
        ):
            if state.formal_evaluations and not log_path.is_file():
                raise WorldRunnerError("revised early stopping requires the existing formal history")
            history = (
                [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if log_path.is_file() else []
            )
            replay = replay_formal_score_patience(
                history, schedule, through_epoch=state.last_full_validation_epoch,
                through_update=state.update,
            )
            if replay["formal_history_count"] != state.formal_evaluations:
                raise WorldRunnerError("formal history does not cover the saved validation count")
            efficient_stale_evaluations = replay["stale_evaluations"]
            stop_on_resume = (
                replay["stop_evidence_epoch"] is not None and formal_primary_record is not None
            )
            stop_details = {
                **replay, "stopped_early": stop_on_resume,
                "reason": "revised_policy_satisfied_by_saved_validation_history" if stop_on_resume else "revised_policy_history_replayed",
                "policy_revision": {
                    "previous": {key: prior_schedule.get(key) for key in policy_keys},
                    "current": {key: getattr(schedule, key) for key in policy_keys},
                    "history_path": str(log_path),
                    "resume_epoch": state.epoch,
                    "resume_batch_in_short_epoch": state.batch_in_short_epoch,
                    "resume_update": state.update,
                    "recorded_at_utc": _utc_now(),
                },
                "new_training_updates_after_resume": 0,
            }
        w1_action_parameters = (
            tuple(trainer.groups.parameters["action_dynamics"])
            if normalized == "w1"
            else ()
        )
        initial_parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        w1_action_initial = tuple(
            deferred_initial_snapshot["model_state"][initial_parameter_names[id(parameter)]].clone()
            if deferred_initial_snapshot is not None else parameter.detach().cpu().clone()
            for parameter in w1_action_parameters
        )

        def training_control_payload(
            release_record_json: str | None = None,
        ) -> dict[str, Any]:
            active_release_json = (
                state.selected_record_json
                if release_record_json is None
                else release_record_json
            )
            release_record = (
                json.loads(active_release_json) if active_release_json else None
            )
            return {
                "parent_checkpoint": parent_identity,
                "zero_update_reference": zero_update_reference,
                "formal_primary_record": formal_primary_record,
                "state_pareto_frontier": list(pareto_frontier),
                "pareto_stale_evaluations": pareto_stale_evaluations,
                "w1_dynamics_gradient_nonzero": w1_dynamics_gradient_nonzero,
                "w1_dynamics_parameters_updated": w1_dynamics_parameters_updated,
                "release_record": release_record,
                "selection_policy": schedule.selection_policy if schedule.enabled else "capability_gated",
                "efficient_schedule": asdict(schedule),
                "efficient_stale_evaluations": efficient_stale_evaluations,
                "stop_details": stop_details,
                "early_stop_rule": (
                    "fixed_budget" if schedule.enabled and schedule.fixed_budget
                    else "delayed_relative_improvement_patience" if schedule.enabled
                    else "two_formal_evaluations_without_w0_pareto_expansion"
                    if normalized == "w0"
                    else (
                        "eligible_selector_stability_with_ineligible_u0_pareto"
                        if normalized == "u0" and not failure_plateau_enabled
                        else (
                            "eligible_stability_or_resumable_failure_plateau"
                            if normalized == "u0"
                            else "registered_phase_selector_stability"
                        )
                    )
                ),
            }

        def phase_checkpoint_extra() -> dict[str, Any]:
            return {
                **self._checkpoint_contract_extra(),
                "parent_checkpoint": parent_identity,
                "zero_update_reference": zero_update_reference,
                "deferred_w1_initial_snapshot": deferred_initial_snapshot,
                "timing": dict(timing),
                "selection_policy": schedule.selection_policy if schedule.enabled else "capability_gated",
            }

        _atomic_json(
            state_path,
            {
                "schema": RUN_STATE_SCHEMA,
                "run_id": self.run_id,
                "phase": normalized,
                "status": "TRAINING",
                "expert_assignment": self._expert_assignment_contract(),
                "parent_checkpoint": parent_identity,
                "zero_update_reference": zero_update_reference,
                "resume_state": asdict(state),
                "training_control": training_control_payload(),
                "short_epoch_policy": {
                    "factor": spec.short_epoch_factor,
                    "formal_validation_interval_short_epochs": (
                        spec.formal_validation_interval
                    ),
                    "early_formal_validation_interval_short_epochs": (
                        early_formal_interval
                    ),
                    "early_formal_validation_until_short_epoch": (early_formal_until),
                    "sentinel_batch_ordinals": list(sentinel_ordinals),
                    "sentinel_ordinal_hash": sentinel_hash,
                    "validation_batch_count": validation_batches,
                    "validation_batch_size": self._validation_batch_size(spec),
                    "sentinel_strategy": "equal_chronological_strata_midpoints",
                    "expected_updates_per_full_epoch": (
                        spec.expected_updates_per_full_epoch
                    ),
                    "target_update_budgets": {
                        "minimum": spec.target_minimum_updates,
                        "maximum": spec.target_maximum_updates,
                        "patience_equivalent": spec.target_patience_updates,
                    },
                },
                "locked_2025_opened": False,
                "updated_at_utc": _utc_now(),
            },
        )
        best_path = self.checkpoints.path(normalized, "best")
        release_best_path = self.checkpoints.path(normalized, "release_best")
        formal_primary_best_path = self.checkpoints.path(
            normalized, "formal_primary_best"
        )
        score_best_path = self.checkpoints.path(normalized, "score_best")
        state_pareto_best_path = self.checkpoints.path(normalized, "state_pareto_best")
        candidate_best_path = self.checkpoints.path(normalized, "candidate_best")
        candidate_selection_role = (
            "diagnostic_sentinel_only_not_formal_selection"
            if (
                spec.short_epoch_factor > 1
                or spec.formal_validation_interval > 1
            )
            else "formal_full_validation_alias_no_separate_candidate"
        )
        latest_path = self.checkpoints.path(normalized, "latest")

        def write_u0_gate_current(
            *,
            status: str,
            checkpoint_path: Path | None,
            checkpoint_role: str,
            formal_epoch: int | None,
            formal_summary: ValidationSummary | None,
            reason: str,
        ) -> Path | None:
            if normalized != "u0":
                return None
            path = phase_root / "gate.json"
            checkpoint_id = (
                _sha256(checkpoint_path)
                if checkpoint_path is not None and checkpoint_path.is_file()
                else None
            )
            _atomic_json(
                path,
                {
                    "schema": PHASE_RELEASE_SCHEMA,
                    "run_id": self.run_id,
                    "phase": normalized,
                    "status": str(status),
                    "passed": False,
                    "next_phase_released": False,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_selection_role": checkpoint_role,
                    "release_candidate_existed": release_best_path.is_file(),
                    "current_formal_epoch": formal_epoch,
                    "current_formal_eligible": (
                        None if formal_summary is None else bool(formal_summary.eligible)
                    ),
                    "current_formal_summary": (
                        None if formal_summary is None else asdict(formal_summary)
                    ),
                    "reasons": [reason],
                    "locked_2025_opened": False,
                    "evaluated_at_utc": _utc_now(),
                },
            )
            return path

        if normalized == "u0":
            existing_score = (
                score_best_path
                if score_best_path.is_file()
                else (
                    formal_primary_best_path
                    if formal_primary_best_path.is_file()
                    else None
                )
            )
            write_u0_gate_current(
                status="TRAINING",
                checkpoint_path=existing_score,
                checkpoint_role=(
                    "formal_score_best_pending_release"
                    if existing_score is not None
                    else "no_formal_score_candidate"
                ),
                formal_epoch=(
                    int(formal_primary_record["epoch"])
                    if formal_primary_record is not None
                    else None
                ),
                formal_summary=None,
                reason="U0 training is active; P1 remains blocked",
            )
        last_latest_save = time.perf_counter()
        last_progress_write = 0.0
        progress_state = state
        serialized_key: tuple[int, int, int] | None = None
        serialized_path: Path | None = None

        def persist_timing(saved_state: ResumeState) -> None:
            if schedule.enabled:
                _atomic_json(timing_path, {
                    "run_id": self.run_id, "phase": normalized,
                    "checkpoint_state": {
                        k: getattr(saved_state, k) for k in ("epoch", "batch_in_short_epoch", "update")
                    },
                    "timing": dict(timing), "updated_at_utc": _utc_now(),
                })

        def checkpoint_alias(source: Path, target: Path) -> Path:
            started = time.perf_counter()
            alias = _atomic_checkpoint_alias(source, target)
            timing["checkpoint_alias_seconds"] += time.perf_counter() - started
            timing["checkpoint_alias_count"] += 1
            persist_timing(progress_state)
            return alias

        def save_checkpoint(**kwargs: Any) -> Path:
            nonlocal last_latest_save, serialized_key, serialized_path
            save_state = kwargs["state"]
            key = (save_state.epoch, save_state.batch_in_short_epoch, save_state.update)
            role = kwargs.get("role", "latest")
            if (
                schedule.enabled and role != "latest"
                and serialized_key == key and serialized_path is not None
            ):
                alias = checkpoint_alias(
                    serialized_path, self.checkpoints.path(normalized, role)
                )
                persist_timing(save_state)
                return alias
            started = time.perf_counter()
            path = self.checkpoints.save(**kwargs)
            timing["checkpoint_seconds"] += time.perf_counter() - started
            timing["checkpoint_write_count"] += 1
            if schedule.enabled:
                serialized_key, serialized_path = key, path
            if role == "latest":
                last_latest_save = time.perf_counter()
            persist_timing(save_state)
            return path

        def write_progress(*, force: bool = False, status: str = "TRAINING") -> None:
            nonlocal last_progress_write
            if not schedule.enabled:
                return
            now = time.perf_counter()
            if not force and now - last_progress_write < schedule.progress_interval_seconds:
                return
            current_epoch = progress_state.epoch + bool(progress_state.batch_in_short_epoch)
            _atomic_json(phase_root / "progress.json", {
                "schema": "dcvpp.vpp-world-moe.progress.v1",
                "run_id": self.run_id,
                "phase": normalized,
                "status": status,
                "epoch": current_epoch,
                "completed_epoch": progress_state.epoch,
                "batch_in_short_epoch": progress_state.batch_in_short_epoch,
                "update": progress_state.update,
                "maximum_epochs": spec.maximum_epochs,
                "first_formal_epoch": schedule.first_formal_epoch,
                "selection_policy": schedule.selection_policy if schedule.enabled else "capability_gated",
                "checkpoint_policy": schedule.checkpoint_policy if schedule.enabled else "milestones",
                "last_full_validation_epoch": progress_state.last_full_validation_epoch,
                "efficient_stale_evaluations": efficient_stale_evaluations,
                "stop_details": stop_details,
                "latest_checkpoint_exists": latest_path.is_file(),
                "timing": dict(timing),
                "updated_at_utc": _utc_now(),
            })
            last_progress_write = now

        def recover_interruption(exc: BaseException) -> None:
            save_checkpoint(
                state=progress_state, model=self.model, optimizer=trainer.optimizer,
                role="latest", extra={
                    **phase_checkpoint_extra(),
                    "checkpoint_reason": "interruption",
                    "interruption_type": type(exc).__name__,
                    "training_control": training_control_payload(progress_state.selected_record_json),
                },
            )
            write_progress(force=True, status="PAUSED" if isinstance(exc, KeyboardInterrupt) else "FAILED")

        if schedule.enabled:
            self._active_recovery_callback = recover_interruption
            if score_only and formal_primary_record is not None:
                if not formal_primary_best_path.is_file():
                    raise WorldRunnerError("score-only resume requires the existing formal-primary checkpoint")
                checkpoint_alias(formal_primary_best_path, best_path)
            write_progress(force=True)
        failure_plateau_stopped = _u0_failure_plateau_should_stop(
            state,
            enabled=failure_plateau_enabled,
            required_evaluations=failure_plateau_evaluations,
            minimum_epoch=spec.minimum_epochs,
        )
        epoch_range: Iterable[int] = (
            ()
            if failure_plateau_stopped or stop_on_resume
            else range(state.epoch + 1, spec.maximum_epochs + 1)
        )
        for epoch in epoch_range:
            epoch_started = time.perf_counter()
            full_epoch, shard_index = short_epoch_coordinates(
                epoch, spec.short_epoch_factor
            )
            resume_batch = state.batch_in_short_epoch
            short_loss_sum = state.short_epoch_loss_sum
            short_updates = state.short_epoch_updates
            updates = state.update
            global_updates = state.global_update
            progress_state = state
            for relative_index, raw_batch in enumerate(
                self._batches(
                    "train",
                    spec,
                    epoch=full_epoch,
                    shard_index=shard_index,
                    shard_count=spec.short_epoch_factor,
                    start_batch=resume_batch,
                )
            ):
                batch_index = resume_batch + relative_index
                if schedule.enabled and (phase_root / "PAUSE").exists():
                    raise KeyboardInterrupt("phase PAUSE file requested a recoverable stop")
                train_started = time.perf_counter()
                batch = self._prepare_batch(raw_batch, phase=normalized)
                step = trainer.train_step(
                    batch, seed=self.seed + epoch * 1_000_003 + batch_index
                )
                timing["train_step_seconds"] += time.perf_counter() - train_started
                timing["train_step_count"] += 1
                if normalized == "w1" and not w1_dynamics_gradient_nonzero:
                    w1_dynamics_gradient_nonzero = any(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all())
                        and bool(torch.count_nonzero(parameter.grad.detach()))
                        for parameter in w1_action_parameters
                    )
                short_loss_sum += step.loss
                short_updates += 1
                updates += 1
                global_updates += 1
                progress_state = ResumeState(
                    phase=normalized,
                    epoch=epoch - 1,
                    batch_in_short_epoch=batch_index + 1,
                    short_epoch_factor=spec.short_epoch_factor,
                    update=updates,
                    global_update=global_updates,
                    best_validation=state.best_validation,
                    best_sentinel_validation=state.best_sentinel_validation,
                    formal_stale_short_epochs=(state.formal_stale_short_epochs),
                    short_epoch_loss_sum=short_loss_sum,
                    short_epoch_updates=short_updates,
                    last_full_validation_epoch=state.last_full_validation_epoch,
                    formal_evaluations=state.formal_evaluations,
                    stable_formal_evaluations=state.stable_formal_evaluations,
                    failure_stable_formal_evaluations=(
                        state.failure_stable_formal_evaluations
                    ),
                    failure_best_gate_gap=state.failure_best_gate_gap,
                    failure_best_proper_score=state.failure_best_proper_score,
                    selected_record_json=state.selected_record_json,
                    selection_reference_json=state.selection_reference_json,
                )
                now = time.perf_counter()
                write_progress()
                fixed_update_due = (
                    not schedule.enabled and updates % spec.checkpoint_interval_updates == 0
                )
                wall_clock_due = not best_only and now - last_latest_save >= (
                    schedule.recovery_interval_seconds if schedule.enabled
                    else spec.checkpoint_max_seconds
                )
                if fixed_update_due or wall_clock_due:
                    reason = (
                        "fixed_update_boundary"
                        if fixed_update_due
                        else "wall_clock_emergency"
                    )
                    latest_path = save_checkpoint(
                        state=progress_state,
                        model=self.model,
                        optimizer=trainer.optimizer,
                        role="latest",
                        extra={
                            **phase_checkpoint_extra(),
                            "checkpoint_reason": reason,
                            "short_epoch": epoch,
                            "full_epoch": full_epoch,
                            "shard_index": shard_index,
                            "batch_in_short_epoch": batch_index + 1,
                            "sentinel_ordinal_hash": sentinel_hash,
                            "locked_2025_opened": False,
                            "training_control": training_control_payload(),
                        },
                    )
                    last_latest_save = time.perf_counter()
            if short_updates == 0:
                raise WorldRunnerError(
                    f"{normalized} corpus returned no training batches"
                )
            if schedule.enabled and (phase_root / "PAUSE").exists():
                raise KeyboardInterrupt("phase PAUSE file requested a recoverable stop")
            if schedule.enabled:
                formal_result = None
                sentinel_result = None
                if schedule.formal_due(epoch, spec) or smoke:
                    if deferred_initial_snapshot is not None:
                        with _temporary_model_snapshot(self.model, deferred_initial_snapshot):
                            zero_result = timed_validation(trainer)
                        register_w1_reference(zero_result)
                        progress_state = replace(
                            progress_state, selection_reference_json=state.selection_reference_json
                        )
                        deferred_initial_snapshot = None
                    formal_result = timed_validation(trainer)
            elif (
                spec.short_epoch_factor == 1
                and spec.formal_validation_interval == 1
            ):
                formal_result: float | ValidationSummary | None = timed_validation(
                    trainer
                )
                sentinel_result = formal_result
            else:
                sentinel_result = timed_validation(
                    trainer, batch_ordinals=sentinel_ordinals
                )
                formal_result = (
                    timed_validation(trainer)
                    if is_formal_validation_epoch(
                        epoch,
                        regular_interval=spec.formal_validation_interval,
                        early_interval=early_formal_interval,
                        early_until=early_formal_until,
                    )
                    or smoke
                    else None
                )
            sentinel_validation = (
                None if sentinel_result is None else _validation_primary(sentinel_result)
            )
            formal_validation = (
                None if formal_result is None else _validation_primary(formal_result)
            )
            candidate_improved = (
                sentinel_validation is not None
                and sentinel_validation < state.best_sentinel_validation
            )
            selected_record_json = state.selected_record_json
            selection_reference_json = state.selection_reference_json
            formal_evaluations = state.formal_evaluations
            stable_formal_evaluations = state.stable_formal_evaluations
            failure_stable_formal_evaluations = (
                state.failure_stable_formal_evaluations
            )
            failure_best_gate_gap = state.failure_best_gate_gap
            failure_best_proper_score = state.failure_best_proper_score
            failure_gate_gap: float | None = None
            failure_stream_improved = False
            selection_decision: FormalSelectionDecision | None = None
            score_selection_decision: FormalSelectionDecision | None = None
            formal_improved = False
            primary_improved = False
            pareto_improved = False
            pruned_u0_pareto_roles: set[str] = set()
            selected_validation = state.best_validation
            if isinstance(formal_result, ValidationSummary):
                effective_summary = formal_result
                if ignore_formal_gates and normalized != "u0" and not score_only:
                    effective_summary = replace(
                        formal_result,
                        eligible=True,
                        selection_metrics=dict(formal_result.selection_metrics)
                        | {"gate_thresholds_ignored": True},
                        score_terms=dict(formal_result.score_terms),
                        diagnostics=dict(formal_result.diagnostics),
                    )
                if normalized == "w1":
                    reference = (
                        json.loads(state.selection_reference_json)
                        if state.selection_reference_json
                        else None
                    )
                    if not isinstance(reference, Mapping):
                        raise WorldRunnerError(
                            "W1 formal evaluation has no zero-update reference"
                        )
                    horizons = effective_summary.selection_metrics.get(
                        "horizon_scores"
                    )
                    if not isinstance(horizons, Mapping):
                        raise WorldRunnerError("W1 formal evaluation omitted horizons")
                    pre_reference_passed = all(
                        float(horizons[name]) <= float(reference[name])
                        for name in ("h1", "h4", "h16", "h96")
                    )
                    if not w1_dynamics_parameters_updated:
                        w1_dynamics_parameters_updated = any(
                            not torch.equal(
                                initial,
                                parameter.detach().cpu(),
                            )
                            for initial, parameter in zip(
                                w1_action_initial,
                                w1_action_parameters,
                                strict=True,
                            )
                        )
                    latent_non_degenerate = bool(
                        effective_summary.selection_metrics.get(
                            "latent_kl_non_degenerate", False
                        )
                    )
                    queue_non_degenerate = bool(
                        effective_summary.selection_metrics.get(
                            "queue_rollout_non_degenerate", False
                        )
                    )
                    metrics = dict(effective_summary.selection_metrics) | {
                        "pre_w1_reference_passed": pre_reference_passed,
                        "zero_update_horizon_scores": dict(reference),
                        "dynamics_gradient_nonzero": (
                            w1_dynamics_gradient_nonzero
                        ),
                        "dynamics_parameters_updated": (
                            w1_dynamics_parameters_updated
                        ),
                        "latent_kl_non_degenerate": latent_non_degenerate,
                        "queue_rollout_non_degenerate": queue_non_degenerate,
                    }
                    capability_passed = all(
                        metrics.get(name) is True
                        for name in (
                            "pre_w1_reference_passed",
                            "dart_identity_passed",
                            "dynamics_gradient_nonzero",
                            "dynamics_parameters_updated",
                            "latent_kl_non_degenerate",
                            "queue_rollout_non_degenerate",
                        )
                    )
                    effective_summary = replace(
                        effective_summary,
                        eligible=(
                            True
                            if ignore_formal_gates and not score_only
                            else bool(effective_summary.eligible and capability_passed)
                        ),
                        selection_metrics=metrics,
                    )
                    formal_result = effective_summary
                if failure_plateau_enabled and normalized == "u0":
                    if effective_summary.eligible:
                        failure_stable_formal_evaluations = 0
                    else:
                        (
                            failure_stable_formal_evaluations,
                            failure_best_gate_gap,
                            failure_best_proper_score,
                            failure_gate_gap,
                            failure_stream_improved,
                        ) = _update_u0_failure_plateau(
                            state,
                            effective_summary,
                            relative_improvement=failure_relative_improvement,
                        )
                challenger = formal_candidate_record(effective_summary, epoch=epoch)
                score_selection_decision = select_formal_score_candidate(
                    normalized, formal_primary_record, challenger
                )
                primary_improved = score_selection_decision.challenger_selected
                if primary_improved:
                    assert score_selection_decision.selected is not None
                    formal_primary_record = dict(score_selection_decision.selected)
                if normalized in {"w0", "u0"} and not score_only:
                    prior_u0_pareto_roles = (
                        {
                            str(item["checkpoint_role"])
                            for item in pareto_frontier
                            if str(item.get("checkpoint_role", "")).startswith(
                                "pareto_epoch_"
                            )
                        }
                        if normalized == "u0"
                        else set()
                    )
                    update_frontier = (
                        update_w0_pareto_frontier
                        if normalized == "w0"
                        else update_u0_pareto_frontier
                    )
                    pareto_frontier, pareto_improved = update_frontier(
                        pareto_frontier, challenger
                    )
                    pareto_stale_evaluations = (
                        0 if pareto_improved else pareto_stale_evaluations + 1
                    )
                    if normalized == "u0":
                        retained_roles = {
                            str(item["checkpoint_role"])
                            for item in pareto_frontier
                            if str(item.get("checkpoint_role", "")).startswith(
                                "pareto_epoch_"
                            )
                        }
                        pruned_u0_pareto_roles = (
                            prior_u0_pareto_roles - retained_roles
                        )
                    if normalized == "u0" and pareto_improved:
                        pareto_role = f"pareto_epoch_{epoch:04d}"
                        intended_path = self.checkpoints.path(normalized, pareto_role)
                        for item in pareto_frontier:
                            if int(item.get("epoch", -1)) == int(epoch):
                                item["checkpoint_role"] = pareto_role
                                item["checkpoint_path"] = str(intended_path)
                incumbent = (
                    json.loads(state.selected_record_json)
                    if state.selected_record_json
                    else None
                )
                reference = (
                    json.loads(state.selection_reference_json)
                    if state.selection_reference_json
                    else None
                )
                if normalized == "w1" and reference is None:
                    raise WorldRunnerError(
                        "W1 selection requires the persisted zero-update reference; "
                        "epoch 1 cannot define its own baseline"
                    )
                selection_decision = score_selection_decision if score_only else select_formal_candidate(
                    normalized,
                    incumbent,
                    challenger,
                    w1_reference=reference,
                )
                formal_improved = selection_decision.challenger_selected
                formal_evaluations += 1
                previous_signature = (
                    str(json.loads(state.selected_record_json)["epoch"])
                    if state.selected_record_json
                    else "NONE"
                )
                unchanged_and_small = (
                    incumbent is not None
                    and selection_decision.selected is not None
                    and selection_decision.candidate_signature == previous_signature
                    and selection_decision.relative_improvement < 0.001
                )
                stable_formal_evaluations = (
                    state.stable_formal_evaluations + 1 if unchanged_and_small else 0
                )
                if selection_decision.selected is not None:
                    selected_record_json = json.dumps(
                        selection_decision.selected,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    selected_validation = float(
                        selection_decision.selected["primary_score"]
                    )
            elif formal_validation is not None:
                # Synthetic/legacy unit-test losses retain their historical
                # scalar selector; formal R5-V2 never enters this branch.
                challenger = {
                    "phase": normalized,
                    "epoch": int(epoch),
                    "eligible": True,
                    "primary_score": float(formal_validation),
                    "selection_metrics": {},
                }
                primary_improved = _primary_challenger_selected(
                    formal_primary_record, challenger
                )
                if primary_improved:
                    formal_primary_record = dict(challenger)
                formal_improved = formal_validation < state.best_validation
                selected_validation = min(state.best_validation, formal_validation)
            if schedule.enabled and formal_validation is not None:
                relative_improvement = (
                    selection_decision.relative_improvement
                    if selection_decision is not None
                    else (
                        (state.best_validation - formal_validation) / max(abs(state.best_validation), 1e-12)
                        if np.isfinite(state.best_validation) else 1.0
                    )
                )
                efficient_stale_evaluations = schedule.stale_count(
                    epoch, efficient_stale_evaluations, relative_improvement
                )
            if formal_validation is None:
                formal_stale = state.formal_stale_short_epochs
            elif formal_improved:
                formal_stale = 0
            else:
                formal_stale = (
                    state.formal_stale_short_epochs + spec.formal_validation_interval
                )
            next_state = ResumeState(
                phase=normalized,
                epoch=epoch,
                short_epoch_factor=spec.short_epoch_factor,
                update=updates,
                global_update=global_updates,
                best_validation=selected_validation,
                best_sentinel_validation=(
                    state.best_sentinel_validation if sentinel_validation is None else
                    min(state.best_sentinel_validation, sentinel_validation)
                ),
                formal_stale_short_epochs=formal_stale,
                last_full_validation_epoch=(
                    epoch
                    if formal_validation is not None
                    else state.last_full_validation_epoch
                ),
                formal_evaluations=formal_evaluations,
                stable_formal_evaluations=stable_formal_evaluations,
                failure_stable_formal_evaluations=(
                    failure_stable_formal_evaluations
                ),
                failure_best_gate_gap=failure_best_gate_gap,
                failure_best_proper_score=failure_best_proper_score,
                selected_record_json=selected_record_json,
                selection_reference_json=selection_reference_json,
            )
            progress_state = next_state
            if not schedule.enabled or (
                formal_validation is not None and (not best_only or primary_improved)
            ):
                latest_path = save_checkpoint(
                    state=next_state,
                    model=self.model,
                    optimizer=trainer.optimizer,
                    role="latest",
                    extra={
                        **phase_checkpoint_extra(),
                        "checkpoint_reason": "short_epoch_complete",
                        "selection_role": (
                            "validation_score_only" if score_only
                            else "release_best_all_gates_eligible" if formal_improved
                            else "formal_primary_best_even_if_ineligible" if primary_improved
                            else "formal_node_recovery"
                        ),
                        "formal_primary_record": formal_primary_record,
                        "state_pareto_frontier": list(pareto_frontier),
                        "formal_selection_decision": (
                            asdict(selection_decision) if selection_decision is not None else None
                        ),
                        "sentinel_validation_loss": sentinel_validation,
                        "formal_validation_loss": formal_validation,
                        "sentinel_tail_routing": _tail_log_payload(sentinel_result),
                        "formal_tail_routing": _tail_log_payload(formal_result),
                        "formal_selection": (
                            asdict(formal_result)
                            if isinstance(formal_result, ValidationSummary)
                            else None
                        ),
                        "sentinel_batch_ordinals": list(sentinel_ordinals),
                        "sentinel_ordinal_hash": sentinel_hash,
                        "training_control": training_control_payload(selected_record_json),
                    },
                )
            if normalized == "u0" and isinstance(formal_result, ValidationSummary):
                _atomic_jsonl_append_unique(
                    phase_root / "formal_candidates.jsonl",
                    {
                        "schema": "dcvpp.vpp-world-moe.u0-formal-candidate.v1",
                        "run_id": self.run_id,
                        "phase": "u0",
                        "epoch": int(epoch),
                        "update": int(updates),
                        "global_update": int(global_updates),
                        "candidate": challenger,
                        "checkpoint_role": "latest_recovery_snapshot_same_epoch",
                        "checkpoint_path": str(latest_path),
                        "parent_checkpoint": parent_identity,
                        "formal_seed_offsets": list(U0_FORMAL_SEED_OFFSETS),
                        "formal_paths": 100,
                        "validation_days": 364,
                        "locked_2025_opened": False,
                        "recorded_at_utc": _utc_now(),
                    },
                    identity_fields=("run_id", "phase", "epoch", "update"),
                )
            if candidate_improved and (
                spec.short_epoch_factor > 1
                or spec.formal_validation_interval > 1
            ):
                candidate_best_path = save_checkpoint(
                    state=next_state,
                    model=self.model,
                    optimizer=trainer.optimizer,
                    role="candidate_best",
                    extra={
                        **phase_checkpoint_extra(),
                        "selection_role": candidate_selection_role,
                        "sentinel_validation_loss": sentinel_validation,
                        "sentinel_batch_ordinals": list(sentinel_ordinals),
                        "sentinel_ordinal_hash": sentinel_hash,
                    },
                )
            if primary_improved:
                primary_role = "score_best" if normalized == "u0" else "formal_primary_best"
                primary_path = save_checkpoint(
                    state=next_state,
                    model=self.model,
                    optimizer=None if normalized == "u0" else trainer.optimizer,
                    role=primary_role,
                    extra={
                        **phase_checkpoint_extra(),
                        "selection_role": (
                            "formal_score_best_even_if_ineligible"
                            if normalized == "u0"
                            else "formal_primary_best_even_if_ineligible"
                        ),
                        "formal_validation_loss": formal_validation,
                        "formal_selection": (
                            asdict(formal_result)
                            if isinstance(formal_result, ValidationSummary)
                            else None
                        ),
                        "formal_primary_record": formal_primary_record,
                        "formal_score_selection_decision": (
                            asdict(score_selection_decision)
                            if score_selection_decision is not None
                            else None
                        ),
                        "validation_batch_count": validation_batches,
                        "sentinel_ordinal_hash": sentinel_hash,
                        "training_control": training_control_payload(
                            selected_record_json
                        ),
                    },
                )
                if normalized == "u0":
                    score_best_path = primary_path
                    formal_primary_best_path = checkpoint_alias(
                        score_best_path, formal_primary_best_path
                    )
                else:
                    formal_primary_best_path = primary_path
                if score_only:
                    best_path = checkpoint_alias(formal_primary_best_path, best_path)
            if pareto_improved and normalized == "w0":
                state_pareto_best_path = save_checkpoint(
                    state=next_state,
                    model=self.model,
                    optimizer=trainer.optimizer,
                    role="state_pareto_best",
                    extra={
                        **phase_checkpoint_extra(),
                        "selection_role": "state_pareto_frontier_candidate",
                        "formal_validation_loss": formal_validation,
                        "formal_selection": asdict(formal_result),
                        "state_pareto_frontier": list(pareto_frontier),
                        "pareto_stale_evaluations": pareto_stale_evaluations,
                        "validation_batch_count": validation_batches,
                        "sentinel_ordinal_hash": sentinel_hash,
                        "training_control": training_control_payload(
                            selected_record_json
                        ),
                    },
                )
            elif pareto_improved and normalized == "u0":
                pareto_role = f"pareto_epoch_{epoch:04d}"
                pareto_path = self.checkpoints.path(normalized, pareto_role)
                for item in pareto_frontier:
                    if int(item.get("epoch", -1)) == int(epoch):
                        item["checkpoint_role"] = pareto_role
                        item["checkpoint_path"] = str(pareto_path)
                pareto_path = save_checkpoint(
                    state=next_state,
                    model=self.model,
                    optimizer=None,
                    role=pareto_role,
                    extra={
                        **phase_checkpoint_extra(),
                        "selection_role": "u0_pareto_candidate_even_if_ineligible",
                        "formal_validation_loss": formal_validation,
                        "formal_selection": asdict(formal_result),
                        "state_pareto_frontier": list(pareto_frontier),
                        "pareto_stale_evaluations": pareto_stale_evaluations,
                        "validation_batch_count": validation_batches,
                        "sentinel_ordinal_hash": sentinel_hash,
                        "training_control": training_control_payload(
                            selected_record_json
                        ),
                    },
                )
                state_pareto_best_path = checkpoint_alias(
                    pareto_path, state_pareto_best_path
                )
                for pruned_role in sorted(pruned_u0_pareto_roles):
                    suffix = pruned_role.removeprefix("pareto_epoch_")
                    if not suffix.isdigit():
                        raise WorldRunnerError(
                            f"invalid U0 Pareto checkpoint role: {pruned_role}"
                        )
                    pruned_path = self.checkpoints.path("u0", pruned_role)
                    expected_parent = self.checkpoints.path(
                        "u0", "latest"
                    ).parent.resolve()
                    if pruned_path.parent.resolve() != expected_parent:
                        raise WorldRunnerError(
                            "U0 Pareto pruning escaped the checkpoint directory"
                        )
                    pruned_path.unlink(missing_ok=True)
            if formal_improved and not score_only:
                release_best_path = save_checkpoint(
                    state=next_state,
                    model=self.model,
                    optimizer=trainer.optimizer,
                    role="release_best",
                    extra={
                        **phase_checkpoint_extra(),
                        "selection_role": "release_best_all_gates_eligible",
                        "formal_validation_loss": formal_validation,
                        "formal_selection": (
                            asdict(formal_result)
                            if isinstance(formal_result, ValidationSummary)
                            else None
                        ),
                        "formal_selection_decision": (
                            asdict(selection_decision)
                            if selection_decision is not None
                            else None
                        ),
                        "validation_batch_count": validation_batches,
                        "sentinel_ordinal_hash": sentinel_hash,
                        "training_control": training_control_payload(
                            selected_record_json
                        ),
                    },
                )
                best_path = checkpoint_alias(release_best_path, best_path)
                if (
                    spec.short_epoch_factor == 1
                    and spec.formal_validation_interval == 1
                ):
                    candidate_best_path = best_path
            if normalized == "u0" and isinstance(formal_result, ValidationSummary):
                current_score_path = (
                    release_best_path
                    if release_best_path.is_file()
                    else (
                        score_best_path
                        if score_best_path.is_file()
                        else (
                            formal_primary_best_path
                            if formal_primary_best_path.is_file()
                            else latest_path
                        )
                    )
                )
                write_u0_gate_current(
                    status="TRAINING",
                    checkpoint_path=current_score_path,
                    checkpoint_role=(
                        "release_candidate_pending_final_gate"
                        if release_best_path.is_file()
                        else "formal_score_best_ineligible_no_release"
                    ),
                    formal_epoch=epoch,
                    formal_summary=formal_result,
                    reason=(
                        "eligible U0 candidate awaits the final phase gate"
                        if release_best_path.is_file()
                        else "no eligible U0 candidate exists; P1 remains blocked"
                    ),
                )
            _atomic_jsonl_append(
                log_path,
                {
                    "phase": normalized,
                    "epoch": epoch,
                    "short_epoch": epoch,
                    "full_epoch": full_epoch,
                    "shard_index": shard_index,
                    "short_epoch_factor": spec.short_epoch_factor,
                    "updates_completed": updates,
                    "short_epoch_updates": short_updates,
                    "train_loss_mean": short_loss_sum / short_updates,
                    "timing": dict(timing),
                    "sentinel_validation_loss": sentinel_validation,
                    "sentinel_batch_ordinals": list(sentinel_ordinals),
                    "sentinel_ordinal_hash": sentinel_hash,
                    "formal_validation_loss": formal_validation,
                    "sentinel_tail_routing": _tail_log_payload(sentinel_result),
                    "formal_tail_routing": _tail_log_payload(formal_result),
                    "best_formal_validation": next_state.best_validation,
                    "best_sentinel_validation": (next_state.best_sentinel_validation),
                    "candidate_improved": candidate_improved,
                    "formal_primary_improved": primary_improved,
                    "state_pareto_improved": pareto_improved,
                    "pareto_stale_evaluations": pareto_stale_evaluations,
                    "formal_improved": formal_improved,
                    "formal_stale_short_epochs": formal_stale,
                    "formal_evaluations": formal_evaluations,
                    "stable_formal_evaluations": stable_formal_evaluations,
                    "failure_stable_formal_evaluations": (
                        failure_stable_formal_evaluations
                    ),
                    "failure_gate_gap": failure_gate_gap,
                    "failure_best_gate_gap": failure_best_gate_gap,
                    "failure_best_proper_score": failure_best_proper_score,
                    "failure_stream_improved": failure_stream_improved,
                    "formal_selection": (
                        asdict(formal_result)
                        if isinstance(formal_result, ValidationSummary)
                        else None
                    ),
                    "formal_selection_decision": (
                        asdict(selection_decision)
                        if selection_decision is not None
                        else None
                    ),
                    "formal_score_selection_decision": (
                        asdict(score_selection_decision)
                        if score_selection_decision is not None
                        else None
                    ),
                    "seconds": time.perf_counter() - epoch_started,
                    "locked_2025_opened": False,
                },
            )
            state = next_state
            _atomic_json(
                state_path,
                {
                    "schema": RUN_STATE_SCHEMA,
                    "run_id": self.run_id,
                    "phase": normalized,
                    "status": "TRAINING",
                    "expert_assignment": self._expert_assignment_contract(),
                    "parent_checkpoint": parent_identity,
                    "zero_update_reference": zero_update_reference,
                    "resume_state": asdict(state),
                    "training_control": training_control_payload(),
                    "formal_primary_best_checkpoint": (
                        str(formal_primary_best_path)
                        if formal_primary_best_path.is_file()
                        else None
                    ),
                    "formal_primary_best_checkpoint_exists": (
                        formal_primary_best_path.is_file()
                    ),
                    "score_best_checkpoint": (
                        str(score_best_path) if score_best_path.is_file() else None
                    ),
                    "score_best_checkpoint_exists": score_best_path.is_file(),
                    "state_pareto_best_checkpoint": (
                        str(state_pareto_best_path)
                        if state_pareto_best_path.is_file()
                        else None
                    ),
                    "state_pareto_best_checkpoint_exists": (
                        state_pareto_best_path.is_file()
                    ),
                    "u0_pareto_candidate_checkpoints": (
                        [
                            str(item["checkpoint_path"])
                            for item in pareto_frontier
                            if isinstance(item.get("checkpoint_path"), str)
                        ]
                        if normalized == "u0"
                        else []
                    ),
                    "release_best_checkpoint": (
                        str(release_best_path) if release_best_path.is_file() else None
                    ),
                    "release_best_checkpoint_exists": release_best_path.is_file(),
                    "best_checkpoint": str(best_path) if best_path.is_file() else None,
                    "best_checkpoint_exists": best_path.is_file(),
                    "best_checkpoint_selection_role": (
                        "validation_score_only" if score_only else "release_best_all_gates_eligible"
                    ),
                    "candidate_best_checkpoint": (
                        str(candidate_best_path)
                        if candidate_best_path.is_file()
                        else None
                    ),
                    "candidate_best_checkpoint_exists": (candidate_best_path.is_file()),
                    "candidate_best_selection_role": candidate_selection_role,
                    "latest_checkpoint": str(latest_path),
                    "current_gate": (
                        str(phase_root / "gate.json")
                        if normalized == "u0"
                        else None
                    ),
                    "sentinel_batch_ordinals": list(sentinel_ordinals),
                    "sentinel_ordinal_hash": sentinel_hash,
                    "locked_2025_opened": False,
                    "updated_at_utc": _utc_now(),
                },
            )
            w0_pareto_early_stop = (
                normalized == "w0"
                and state.formal_evaluations >= 3
                and pareto_stale_evaluations >= 2
            )
            proper_early_stop = (
                normalized != "w0"
                and state.formal_evaluations > 0
                # An ineligible stream has no selected model to declare
                # stable.  In particular, steadily improving U0 coverage must
                # be allowed to continue instead of treating NONE == NONE as
                # a converged candidate.
                and bool(state.selected_record_json)
                and state.stable_formal_evaluations >= 3
            )
            legacy_early_stop = (
                state.formal_evaluations == 0 and formal_stale >= spec.patience
            )
            failure_plateau_early_stop = _u0_failure_plateau_should_stop(
                state,
                enabled=failure_plateau_enabled,
                required_evaluations=failure_plateau_evaluations,
                minimum_epoch=spec.minimum_epochs,
            )
            progress_state = state
            write_progress(force=True)
            if schedule.enabled:
                w0_pareto_early_stop = False
                legacy_early_stop = False
                failure_plateau_early_stop = False
                proper_early_stop = (
                    not schedule.fixed_budget
                    and epoch >= schedule.early_stop_start_epoch
                    and efficient_stale_evaluations >= schedule.early_stop_evaluations
                    and bool(state.selected_record_json or self.loss_builder is not None)
                )
            if (
                epoch >= spec.minimum_epochs
                and (
                    w0_pareto_early_stop
                    or proper_early_stop
                    or legacy_early_stop
                    or failure_plateau_early_stop
                )
                and epoch % spec.short_epoch_factor == 0
            ):
                failure_plateau_stopped = failure_plateau_early_stop
                if schedule.enabled and proper_early_stop:
                    stop_details = {
                        **stop_details, "stopped_early": True,
                        "reason": "formal_validation_plateau",
                        "stop_evidence_epoch": state.last_full_validation_epoch,
                        "stop_evidence_update": state.update,
                        "stale_evaluations": efficient_stale_evaluations,
                    }
                break
            if smoke:
                break
        if schedule.enabled:
            stop_details.update(
                actual_completed_epoch=state.epoch,
                actual_training_epoch=state.epoch + bool(state.batch_in_short_epoch),
                actual_batch_in_short_epoch=state.batch_in_short_epoch,
                actual_update=state.update,
                maximum_epoch_budget=spec.maximum_epochs,
            )
            if stop_details.get("policy_revision"):
                stop_details["new_training_updates_after_resume"] = (
                    state.update - stop_details["policy_revision"]["resume_update"]
                )
            if stop_details.get("stopped_early"):
                _atomic_json(phase_root / "early_stop.json", stop_details)
            # The final boundary is already a formal node; retain it without
            # serializing the model/optimizer again before loading the best.
            if not score_only:
                checkpoint_alias(latest_path, self.checkpoints.path(normalized, "final"))
            write_progress(force=True, status="TRAINING_COMPLETE")
            self._active_recovery_callback = None
        release_available = not score_only and release_best_path.is_file()
        if score_only:
            if formal_primary_record is None or not formal_primary_best_path.is_file():
                raise WorldRunnerError("score-only selection produced no formal checkpoint")
            if not np.isfinite(float(formal_primary_record["primary_score"])):
                raise WorldRunnerError("score-only selection cannot release a nonfinite validation score")
            if not best_path.is_file() or not best_path.samefile(formal_primary_best_path):
                checkpoint_alias(formal_primary_best_path, best_path)
            selected_checkpoint_path = formal_primary_best_path
            selected_checkpoint_role = "formal_primary_best"
            selected_checkpoint_label = "validation_score_only"
        elif release_available:
            selected_checkpoint_path = best_path
            selected_checkpoint_role = "best"
            selected_checkpoint_label = "release_best_all_gates_eligible"
        elif normalized == "u0" and score_best_path.is_file():
            selected_checkpoint_path = score_best_path
            selected_checkpoint_role = "score_best"
            selected_checkpoint_label = "formal_score_best_ineligible_no_release"
        elif formal_primary_best_path.is_file():
            selected_checkpoint_path = formal_primary_best_path
            selected_checkpoint_role = "formal_primary_best"
            selected_checkpoint_label = "formal_primary_best_ineligible_fallback"
        else:
            raise WorldRunnerError(
                f"{normalized} produced no full-validation checkpoint"
            )
        self.checkpoints.load(
            normalized,
            model=self.model,
            optimizer=(
                None if selected_checkpoint_role == "score_best" else trainer.optimizer
            ),
            role=selected_checkpoint_role,
            map_location="cpu",
        )
        self.model.to(self.device)
        gate_path: Path | None = None
        released = False
        status = (
            "TRAINED_AWAITING_GATE"
            if release_available
            else "TRAINED_NO_RELEASE_CANDIDATE"
        )
        if failure_plateau_stopped and not release_available:
            status = failure_terminal_status
        if smoke:
            status = "SMOKE_COMPLETE"
        elif gate_callback is not None or score_only:
            result = _call_supported(
                gate_callback or self.registered_phase_gate_callback,
                phase=normalized,
                model=self.model,
                corpus=self.corpus,
                checkpoint_path=selected_checkpoint_path,
                checkpoint_id=_sha256(selected_checkpoint_path),
                device=self.device,
            )
            gate_payload = (
                result.as_dict() if isinstance(result, GateResult) else dict(result)
            )
            passed = (
                (release_available or score_only)
                and gate_payload.get("status") == "PASS"
                and gate_payload.get("passed") is True
            )
            released = True if score_only else bool(passed)
            status = "VALIDATION_SCORE_SELECTED" if score_only else (
                "CAPABILITY_GATE_PASSED" if passed else "CAPABILITY_GATE_FAILED"
            )
            gate_path = phase_root / "gate.json"
            _atomic_json(
                gate_path,
                {
                    "schema": PHASE_RELEASE_SCHEMA,
                    "run_id": self.run_id,
                    "phase": normalized,
                    "status": "PASS" if passed else "FAIL",
                    "passed": passed,
                    "next_phase_released": released,
                    "selection_policy": schedule.selection_policy if schedule.enabled else "capability_gated",
                    "stage_transition_allowed": released,
                    "checkpoint_id": _sha256(selected_checkpoint_path),
                    "checkpoint_selection_role": selected_checkpoint_label,
                    "release_candidate_existed": release_available,
                    "native_evidence": gate_payload,
                    "locked_2025_opened": False,
                    "evaluated_at_utc": _utc_now(),
                },
            )
        elif not smoke and normalized == "u0" and not release_available:
            gate_path = write_u0_gate_current(
                status="FAIL",
                checkpoint_path=selected_checkpoint_path,
                checkpoint_role=selected_checkpoint_label,
                formal_epoch=(
                    int(formal_primary_record["epoch"])
                    if formal_primary_record is not None
                    else None
                ),
                formal_summary=None,
                reason=(
                    "U0 ineligible stream reached the registered failure plateau"
                    if failure_plateau_stopped
                    else "U0 completed without an eligible release checkpoint"
                ),
            )
        result = PhaseRunResult(
            phase=normalized,
            status=status,
            epochs_completed=state.epoch,
            updates_completed=state.update,
            best_validation=(
                state.best_validation
                if release_available
                else float(formal_primary_record["primary_score"])
            ),
            best_checkpoint=str(selected_checkpoint_path),
            latest_checkpoint=str(latest_path),
            gate_path=None if gate_path is None else str(gate_path),
            next_phase_released=released,
        )
        if score_only and not smoke:
            checkpoint_root = formal_primary_best_path.parent.resolve()
            for obsolete_role in ("candidate_best", "release_best", "state_pareto_best", "final"):
                obsolete = self.checkpoints.path(normalized, obsolete_role)
                if obsolete.parent.resolve() != checkpoint_root:
                    raise WorldRunnerError("obsolete checkpoint cleanup escaped the phase directory")
                obsolete.unlink(missing_ok=True)
        _atomic_json(
            state_path,
            {
                "schema": RUN_STATE_SCHEMA,
                "run_id": self.run_id,
                **asdict(result),
                "resume_state": asdict(state),
                "stop_details": stop_details,
                "selection_policy": schedule.selection_policy if schedule.enabled else "capability_gated",
                "stage_transition_allowed": released,
                "expert_assignment": self._expert_assignment_contract(),
                "parent_checkpoint": parent_identity,
                "zero_update_reference": zero_update_reference,
                "training_control": training_control_payload(),
                "best_checkpoint_selection_role": selected_checkpoint_label,
                "formal_primary_best_checkpoint": (
                    str(formal_primary_best_path)
                    if formal_primary_best_path.is_file()
                    else None
                ),
                "score_best_checkpoint": (
                    str(score_best_path) if score_best_path.is_file() else None
                ),
                "state_pareto_best_checkpoint": (
                    str(state_pareto_best_path)
                    if state_pareto_best_path.is_file()
                    else None
                ),
                "u0_pareto_candidate_checkpoints": (
                    [
                        str(item["checkpoint_path"])
                        for item in pareto_frontier
                        if isinstance(item.get("checkpoint_path"), str)
                    ]
                    if normalized == "u0"
                    else []
                ),
                "release_best_checkpoint": (
                    str(release_best_path) if release_available else None
                ),
                "candidate_best_checkpoint": (
                    str(candidate_best_path) if candidate_best_path.is_file() else None
                ),
                "candidate_best_selection_role": candidate_selection_role,
                "short_epoch_policy": {
                    "factor": spec.short_epoch_factor,
                    "formal_validation_interval_short_epochs": (
                        spec.formal_validation_interval
                    ),
                    "early_formal_validation_interval_short_epochs": (
                        early_formal_interval
                    ),
                    "early_formal_validation_until_short_epoch": (early_formal_until),
                    "sentinel_batch_ordinals": list(sentinel_ordinals),
                    "sentinel_ordinal_hash": sentinel_hash,
                    "validation_batch_count": validation_batches,
                    "validation_batch_size": self._validation_batch_size(spec),
                    "sentinel_strategy": "equal_chronological_strata_midpoints",
                    "expected_updates_per_full_epoch": (
                        spec.expected_updates_per_full_epoch
                    ),
                    "target_update_budgets": {
                        "minimum": spec.target_minimum_updates,
                        "maximum": spec.target_maximum_updates,
                        "patience_equivalent": spec.target_patience_updates,
                    },
                },
                "updated_at_utc": _utc_now(),
            },
        )
        return result

    def run_c1(
        self,
        evaluator: Callable[..., Sequence[Mapping[str, Any]]],
        *,
        ignore_phase_gate: bool = False,
        dates: Sequence[str] | None = None,
        resume_evaluation: bool = False,
        day_batch_size: int = 1,
        max_workers: int = 1,
        parallel_safe: bool = False,
    ) -> UnifiedGateResult:
        """Run the complete 364-day validation on the immutable J1 checkpoint."""

        if not ignore_phase_gate:
            self._previous_phase_release("c1")
        checkpoint_path = self.checkpoints.path("j1", "best")
        if not checkpoint_path.is_file():
            raise WorldRunnerError("C1 requires the selected J1 checkpoint")
        payload = self._assert_checkpoint_expert_assignment(checkpoint_path)
        checkpoint_id = _sha256(checkpoint_path)
        if payload.get("locked_2025_opened") is not False:
            raise WorldRunnerError("J1 checkpoint does not preserve the 2025 lock")
        j1_gate_path = self._phase_root("j1") / "gate.json"
        if not j1_gate_path.is_file():
            raise WorldRunnerError("C1 requires the immutable J1 release gate")
        j1_gate = json.loads(j1_gate_path.read_text(encoding="utf-8"))
        if (
            not isinstance(j1_gate, Mapping)
            or j1_gate.get("schema") != PHASE_RELEASE_SCHEMA
            or j1_gate.get("phase") != "j1"
            or j1_gate.get("status") != "PASS"
            or j1_gate.get("passed") is not True
            or j1_gate.get("next_phase_released") is not True
            or j1_gate.get("locked_2025_opened") is not False
            or str(j1_gate.get("checkpoint_id", "")) != checkpoint_id
        ):
            raise WorldRunnerError("J1 gate and best.pt are not bound to the same SHA")
        gate_checkpoint_path = str(j1_gate.get("checkpoint_path", "")).strip()
        if gate_checkpoint_path and Path(gate_checkpoint_path).resolve() != checkpoint_path.resolve():
            raise WorldRunnerError("J1 gate selects a path other than J1 best.pt")
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        phase_root = self._phase_root("c1")
        record_path = phase_root / "daily_evidence.jsonl"
        progress_path = phase_root / "progress.json"
        invalidation_path = phase_root / "INVALIDATED.json"
        if invalidation_path.is_file():
            raise WorldRunnerError(
                "C1 evidence was previously invalidated; start a new evaluation lineage"
            )
        original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        model_state_before = _model_state_sha256(self.model)
        appended_days = 0
        try:
            if dates is not None:
                daily = run_daily_closed_loop(
                    dates=dates,
                    checkpoint_id=checkpoint_id,
                    record_path=record_path,
                    progress_path=progress_path,
                    evaluate_day=evaluator,
                    model=self.model,
                    corpus=self.corpus,
                    resume=resume_evaluation,
                    day_batch_size=day_batch_size,
                    max_workers=max_workers,
                    parallel_safe=parallel_safe,
                )
                records = tuple(daily.records)
                appended_days = int(daily.appended_days)
            else:
                with torch.inference_mode():
                    evaluated = tuple(
                        _call_supported(
                            evaluator,
                            model=self.model,
                            corpus=self.corpus,
                            checkpoint_id=checkpoint_id,
                            split="validation",
                            device=self.device,
                            expected_days=364,
                        )
                    )
                schedule = tuple(sorted(str(row.get("date", ""))[:10] for row in evaluated))
                store = C1DailyRecordStore(
                    record_path,
                    checkpoint_id=checkpoint_id,
                    dates=schedule,
                    resume=resume_evaluation,
                )
                existing_dates = {str(row["date"]) for row in store.records()}
                for row in sorted(evaluated, key=lambda item: str(item.get("date", ""))):
                    if str(row.get("date", ""))[:10] not in existing_dates:
                        store.append(row)
                        appended_days += 1
                records = tuple(store.records())
        finally:
            model_state_after = _model_state_sha256(self.model)
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(original_requires_grad[name])
        if model_state_after != model_state_before:
            _atomic_json(
                invalidation_path,
                {
                    "schema": "dcvpp.vpp-world-moe.c1-invalidation.v1",
                    "checkpoint_id": checkpoint_id,
                    "reason": "evaluator_mutated_immutable_j1_model_state",
                    "model_state_sha256_before": model_state_before,
                    "model_state_sha256_after": model_state_after,
                    "locked_2025_opened": False,
                    "invalidated_at_utc": _utc_now(),
                },
            )
            raise WorldRunnerError("C1 evaluator mutated the immutable J1 model state")
        if len(records) != 364 or any(
            str(record.get("checkpoint_id", "")) != checkpoint_id
            for record in records
        ):
            raise WorldRunnerError(
                "C1 requires 364 complete daily records bound to J1 best.pt SHA"
            )
        gates = {
            "world": evaluate_world_gate(records),
            "stochastic": evaluate_stochastic_gate(records),
            "planner": evaluate_planner_gate(records),
            "closure": evaluate_closure_gate(records),
        }
        unified = evaluate_same_checkpoint_unified_gate(gates)
        _atomic_json(
            phase_root / "daily_evidence.json",
            {
                "checkpoint_id": checkpoint_id,
                "append_only_source": str(record_path),
                "records": records,
            },
        )
        c1_gate_path = phase_root / "gate.json"
        _atomic_json(c1_gate_path, unified.as_dict())
        handoff_path = phase_root / "downstream_handoff.json"
        _atomic_json(
            handoff_path,
            {
                "schema": C1_HANDOFF_SCHEMA,
                "run_id": self.run_id,
                "status": "RELEASED" if unified.passed else "BLOCKED",
                "final_test_released": bool(unified.passed),
                "final_test_executed": False,
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "c1_gate_path": str(c1_gate_path.resolve()),
                "c1_gate_sha256": _sha256(c1_gate_path),
                "append_only_record_path": str(record_path),
                "append_only_record_sha256": _sha256(record_path),
                "validation_days": len(records),
                "locked_2025_opened": False,
                "created_at_utc": _utc_now(),
            },
        )
        _atomic_json(
            phase_root / "state.json",
            {
                "schema": RUN_STATE_SCHEMA,
                "run_id": self.run_id,
                "phase": "c1",
                "expert_assignment": self._expert_assignment_contract(),
                "status": "UNIFIED_GATE_PASSED"
                if unified.passed
                else "UNIFIED_GATE_FAILED",
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "j1_gate_path": str(j1_gate_path.resolve()),
                "validation_days": len(records),
                "append_only_record_path": str(record_path),
                "downstream_handoff_path": str(handoff_path),
                "appended_days_this_invocation": appended_days,
                "model_state_sha256_before": model_state_before,
                "model_state_sha256_after": model_state_after,
                "evaluation_only": True,
                "optimizer_created": False,
                "locked_2025_opened": False,
                "updated_at_utc": _utc_now(),
            },
        )
        return unified

    def smoke(self) -> Mapping[str, Any]:
        """One development batch per trainable phase; never reads 2025."""

        original = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        results: dict[str, Any] = {}
        try:
            for phase in PHASE_ORDER[:-1]:
                spec = replace(phase_spec_from_config(self.config, phase), batch_size=1)
                trainer_kwargs = _configured_weight_kwargs(
                    self.config,
                    synthetic_loss_builder=self.loss_builder is not None,
                )
                if self.loss_builder is not None:
                    trainer_kwargs["loss_builder"] = self.loss_builder
                trainer = WorldModelTrainer(
                    self.model,
                    spec,
                    device=self.device,
                    **trainer_kwargs,
                )
                raw_batch = next(iter(self._batches("train", spec, epoch=0)))
                batch = self._prepare_batch(raw_batch, phase=phase)
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                started = time.perf_counter()
                step = trainer.train_step(batch, seed=self.seed)
                results[phase] = {
                    **asdict(step),
                    "seconds": time.perf_counter() - started,
                    "max_cuda_allocated_mb": (
                        torch.cuda.max_memory_allocated(self.device) / 1024**2
                        if self.device.type == "cuda"
                        else 0.0
                    ),
                    "max_cuda_reserved_mb": (
                        torch.cuda.max_memory_reserved(self.device) / 1024**2
                        if self.device.type == "cuda"
                        else 0.0
                    ),
                }
            c1_spec = replace(phase_spec_from_config(self.config, "c1"), batch_size=1)
            c1_trainer_kwargs = _configured_weight_kwargs(
                self.config,
                synthetic_loss_builder=self.loss_builder is not None,
            )
            if self.loss_builder is not None:
                c1_trainer_kwargs["loss_builder"] = self.loss_builder
            c1_trainer = WorldModelTrainer(
                self.model,
                c1_spec,
                device=self.device,
                **c1_trainer_kwargs,
            )
            raw_batch = next(iter(self._batches("validation", c1_spec, epoch=0)))
            batch = self._prepare_batch(raw_batch, phase="c1")
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            with torch.no_grad():
                output = c1_trainer.forward(batch, seed=self.seed)
            results["c1"] = {
                "forward_completed": output is not None,
                "evaluation_only": True,
                "seconds": time.perf_counter() - started,
                "max_cuda_allocated_mb": (
                    torch.cuda.max_memory_allocated(self.device) / 1024**2
                    if self.device.type == "cuda"
                    else 0.0
                ),
                "max_cuda_reserved_mb": (
                    torch.cuda.max_memory_reserved(self.device) / 1024**2
                    if self.device.type == "cuda"
                    else 0.0
                ),
            }
        finally:
            self.model.load_state_dict(original)
        payload = {
            "schema": SMOKE_SCHEMA,
            "run_id": self.run_id,
            "phase_order": list(PHASE_ORDER),
            "results": results,
            "locked_2025_opened": False,
            "completed_at_utc": _utc_now(),
        }
        _atomic_json(self.output_root / "smoke.json", payload)
        return payload


__all__ = [
    "C1_HANDOFF_SCHEMA",
    "PHASE_RELEASE_SCHEMA",
    "RUN_STATE_SCHEMA",
    "SMOKE_SCHEMA",
    "PhaseRunResult",
    "WorldMoERunner",
    "WorldRunnerError",
    "assert_development_batch",
    "assert_w1_h96_target",
    "phase_spec_from_config",
]
