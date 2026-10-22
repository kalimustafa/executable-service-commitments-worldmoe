"""Scientific gates for VPP-WorldMoE.

These gates have their own schemas and evidence contract.  They never import,
translate, or inherit a gate decision from an earlier architecture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np


EXPECTED_VALIDATION_DAYS = 364
VALIDATION_START = date(2024, 1, 1)
VALIDATION_END = date(2024, 12, 31)
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
IDENTITY_TOLERANCE = 1.0e-6
PHYSICS_TOLERANCE = 1.0e-5
COVERAGE_ERROR_TOLERANCE = 0.03
SLA_INCREASE_TOLERANCE = 0.001
MINIMUM_DIVERSITY = 1.0e-4
MAXIMUM_DUPLICATE_RATE = 0.01
U0_GLOBAL_COVERAGE_ERROR_TOLERANCE = 0.02
U0_FAMILY_COVERAGE_ERROR_TOLERANCE = 0.04

GATE_SCHEMA = "dcvpp.vpp-world-moe.scientific-gate.v1"
PHASE_GATE_SCHEMA = "dcvpp.vpp-world-moe.phase-capability-gate.r5.v2"
UNIFIED_SCHEMA = "dcvpp.vpp-world-moe.same-checkpoint-unified-gate.v1"
U0_LEGACY_COVERAGE_SCHEMA = "dcvpp.vpp-world-moe.family-rank-coverage.v1"
U0_MIXED_SUPPORT_COVERAGE_SCHEMA = (
    "dcvpp.vpp-world-moe.atom-aware-family-coverage.v2"
)
U0_DISTRIBUTION_NATIVE_COVERAGE_SCHEMA = (
    "dcvpp.vpp-world-moe.distribution-native-mixed-coverage.v3"
)

_U0_REQUIRED_CAPABILITIES = (
    "dart_identity_passed",
    "family_coverage_fields_present",
    "formal_100_path_coverage_present",
    "coverage_intervals_nested",
    "day_block_coverage_present",
    "day_block_coverage_passed",
    "family_coverage_passed",
    "coverage_passed",
    "tail_route_support_present",
    "tail_route_separation_passed",
    "tail_route_lift_passed",
    "tail_route_aucpr_passed",
    "tail_routing_passed",
    "duplicate_passed",
    "diversity_passed",
)

_U0_MIXED_SUPPORT_REQUIRED_CAPABILITIES = (
    "atom_support_fields_present",
    "atom_brier_fields_present",
    "atom_calibration_passed",
    "interior_support_present",
    "coverage_moving_block_passed",
    "atom_moving_block_passed",
)


class WorldGateError(ValueError):
    """Gate evidence is incomplete, incomparable, or violates the test lock."""


@dataclass(frozen=True)
class GateResult:
    schema: str
    gate: str
    status: str
    passed: bool
    checkpoint_id: str
    dates: tuple[str, ...]
    metrics: Mapping[str, Any]
    reasons: tuple[str, ...]
    method: Mapping[str, Any]
    locked_2025_opened: bool = False

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class UnifiedGateResult:
    schema: str
    status: str
    passed: bool
    checkpoint_id: str | None
    dates: tuple[str, ...]
    validation_days: int
    gates: Mapping[str, GateResult]
    reasons: tuple[str, ...]
    locked_2025_opened: bool = False

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_phase_release_gate(
    *,
    phase: str,
    checkpoint_id: str,
    formal_selection: Mapping[str, Any],
) -> GateResult:
    """Fail-closed release from the same selected formal-validation summary."""

    normalized = str(phase).lower()
    if normalized not in {"w0", "w1", "u0", "p1", "j1"}:
        raise WorldGateError(f"unsupported trainable phase gate: {phase!r}")
    checkpoint = str(checkpoint_id).strip()
    if not checkpoint:
        raise WorldGateError("phase release requires an immutable checkpoint id")
    metrics = formal_selection.get("selection_metrics")
    if not isinstance(metrics, Mapping):
        raise WorldGateError("formal selection omitted selection_metrics")
    reasons: list[str] = []
    if str(formal_selection.get("phase", "")).lower() != normalized:
        reasons.append("formal selection phase does not match release phase")
    if formal_selection.get("eligible") is not True:
        reasons.append("selected formal checkpoint is not eligible")
    if normalized == "w0":
        required = (
            "dart_identity_passed",
            "state_gate_passed",
            "queue_occurrence_passed",
            "queue_severity_passed",
            "soc_passed",
        )
        for name in required:
            if metrics.get(name) is not True:
                reasons.append(f"W0 capability condition failed: {name}")
    elif normalized == "w1":
        # W1 releases learned action-conditioned world dynamics, not merely a
        # checkpoint with a finite validation loss.  Every condition is
        # fail-closed and is measured on the selected checkpoint/its training
        # lineage.  No arbitrary 1% superiority margin is imposed.
        required = (
            "pre_w1_reference_passed",
            "dart_identity_passed",
            "dynamics_gradient_nonzero",
            "dynamics_parameters_updated",
            "latent_kl_non_degenerate",
            "queue_rollout_non_degenerate",
        )
        for name in required:
            if metrics.get(name) is not True:
                reasons.append(f"W1 capability condition failed: {name}")
    elif normalized == "u0":
        coverage_schema = metrics.get("coverage_schema")
        required = _U0_REQUIRED_CAPABILITIES
        if coverage_schema in {
            U0_MIXED_SUPPORT_COVERAGE_SCHEMA,
            U0_DISTRIBUTION_NATIVE_COVERAGE_SCHEMA,
        }:
            required += _U0_MIXED_SUPPORT_REQUIRED_CAPABILITIES
        elif coverage_schema != U0_LEGACY_COVERAGE_SCHEMA:
            reasons.append("U0 capability condition failed: coverage_schema")
        target_contract = (
            "analytic_coverage_targets_present"
            if coverage_schema == U0_DISTRIBUTION_NATIVE_COVERAGE_SCHEMA
            else "rank_corrected_coverage_targets_present"
        )
        if metrics.get(target_contract) is not True:
            reasons.append(f"U0 capability condition failed: {target_contract}")
        for name in required:
            if metrics.get(name) is not True:
                reasons.append(f"U0 capability condition failed: {name}")
        if coverage_schema in {
            U0_MIXED_SUPPORT_COVERAGE_SCHEMA,
            U0_DISTRIBUTION_NATIVE_COVERAGE_SCHEMA,
        }:
            tolerance_limits = {
                "global_coverage_error_tolerance": (
                    U0_GLOBAL_COVERAGE_ERROR_TOLERANCE
                ),
                "family_coverage_error_tolerance": (
                    U0_FAMILY_COVERAGE_ERROR_TOLERANCE
                ),
            }
            for name, upper_limit in tolerance_limits.items():
                try:
                    value = float(metrics[name])
                except (KeyError, TypeError, ValueError):
                    reasons.append(f"U0 capability condition failed: {name}")
                    continue
                if not np.isfinite(value) or value > upper_limit:
                    reasons.append(
                        f"U0 capability condition failed: {name} exceeds "
                        f"{upper_limit:g}"
                    )
    elif normalized == "j1":
        required = (
            "dart_identity_passed",
            "world_horizons_finite",
            "latent_kl_non_degenerate",
            "queue_rollout_has_support",
            "world_structure_passed",
            "uncertainty_gate_passed",
            "feasibility_passed",
            "sla_passed",
        )
        for name in required:
            if metrics.get(name) is not True:
                reasons.append(f"J1 capability condition failed: {name}")
    passed = not reasons
    return GateResult(
        schema=PHASE_GATE_SCHEMA,
        gate=f"{normalized}_release",
        status="PASS" if passed else "FAIL",
        passed=passed,
        checkpoint_id=checkpoint,
        dates=(),
        metrics={
            "formal_primary_score": float(formal_selection.get("primary_score", np.inf)),
            "formal_selection_metrics": dict(metrics),
        },
        reasons=tuple(reasons),
        method={
            "evidence": "same_selected_full_2024_validation_summary",
            "release_rule": "explicit_phase_capability_conditions",
            "mae_rmse_checkpoint_selection": False,
        },
    )


def _checkpoint(row: Mapping[str, Any]) -> str:
    for name in ("checkpoint_id", "checkpoint_sha256", "checkpoint"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    raise WorldGateError("each C1 row requires one immutable checkpoint identifier")


def _ordered_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    gate: str,
    expected_days: int,
    validation_start: str | date,
    validation_end: str | date,
) -> tuple[tuple[dict[str, Any], ...], str, tuple[str, ...]]:
    rows = tuple(dict(row) for row in records)
    if len(rows) != int(expected_days):
        raise WorldGateError(f"{gate} requires {expected_days} days; got {len(rows)}")
    try:
        start = date.fromisoformat(str(validation_start)[:10])
        end = date.fromisoformat(str(validation_end)[:10])
    except ValueError as exc:
        raise WorldGateError("validation bounds must be ISO dates") from exc
    if start > end or start.year != 2024 or end.year != 2024:
        raise WorldGateError("C1 development validation must be bounded within 2024")
    parsed: list[date] = []
    checkpoints: set[str] = set()
    for row in rows:
        try:
            current = date.fromisoformat(str(row["date"])[:10])
        except (KeyError, ValueError) as exc:
            raise WorldGateError(f"{gate} row requires an ISO date") from exc
        if current.year == 2025 or not start <= current <= end:
            raise WorldGateError(f"{gate} received a non-development date: {current}")
        if row.get("locked_2025_opened") is not False:
            raise WorldGateError(
                "C1 row must explicitly prove locked_2025_opened=false"
            )
        parsed.append(current)
        checkpoints.add(_checkpoint(row))
    if len(set(parsed)) != len(parsed):
        raise WorldGateError(f"{gate} dates must be unique")
    if len(checkpoints) != 1:
        raise WorldGateError(f"{gate} rows must come from exactly one checkpoint")
    order = np.argsort(np.asarray(parsed, dtype="datetime64[D]"), kind="stable")
    ordered = tuple(rows[int(index)] for index in order)
    dates = tuple(str(row["date"])[:10] for row in ordered)
    return ordered, next(iter(checkpoints)), dates


def _scalar(row: Mapping[str, Any], names: Sequence[str], *, label: str) -> float:
    for name in names:
        if name in row:
            try:
                result = float(row[name])
            except (TypeError, ValueError) as exc:
                raise WorldGateError(f"invalid C1 metric {name!r}") from exc
            if not np.isfinite(result):
                raise WorldGateError(f"C1 metric {name!r} is non-finite")
            return result
    raise WorldGateError(f"missing C1 metric {label!r}; accepted={list(names)}")


def _nested_scalar(
    row: Mapping[str, Any],
    container_names: Sequence[str],
    key: str | int,
    *,
    aliases: Sequence[str] = (),
    label: str,
) -> float:
    for container_name in container_names:
        container = row.get(container_name)
        if isinstance(container, Mapping):
            for candidate in (key, str(key), f"h{key}"):
                if candidate in container:
                    try:
                        result = float(container[candidate])
                    except (TypeError, ValueError) as exc:
                        raise WorldGateError(f"invalid C1 metric {label!r}") from exc
                    if not np.isfinite(result):
                        raise WorldGateError(f"C1 metric {label!r} is non-finite")
                    return result
    return _scalar(row, aliases, label=label)


def _vector(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str], *, label: str
) -> np.ndarray:
    result = np.asarray(
        [_scalar(row, names, label=label) for row in rows], dtype=np.float64
    )
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise WorldGateError(f"C1 metric {label!r} must contain finite daily scalars")
    return result


def _paired_inference(
    differences: np.ndarray,
    *,
    block_days: int,
    replications: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise WorldGateError("paired differences must be a finite daily vector")
    block = int(block_days)
    repeats = int(replications)
    level = float(confidence)
    if block < 1 or repeats < 100 or not 0.5 < level < 1.0:
        raise WorldGateError("invalid circular block-bootstrap settings")
    rng = np.random.default_rng(int(seed))
    block_count = int(np.ceil(values.size / block))
    offsets = np.arange(block, dtype=np.int64)
    means = np.empty(repeats, dtype=np.float64)
    # Vectorize bounded chunks rather than running 10,000 Python iterations or
    # materializing every metric's complete bootstrap tensor at once.
    chunk_size = min(1024, repeats)
    for start in range(0, repeats, chunk_size):
        stop = min(start + chunk_size, repeats)
        starts = rng.integers(
            0, values.size, size=(stop - start, block_count), dtype=np.int64
        )
        sampled = (starts[:, :, None] + offsets[None, None, :]) % values.size
        sampled = sampled.reshape(stop - start, -1)[:, : values.size]
        means[start:stop] = values[sampled].mean(axis=1)
    return {
        "observations": int(values.size),
        "mean_difference": float(values.mean()),
        "one_sided_upper": float(np.quantile(means, level)),
        "two_sided_lower": float(np.quantile(means, (1.0 - level) / 2.0)),
        "two_sided_upper": float(np.quantile(means, 1.0 - (1.0 - level) / 2.0)),
    }


def _method(
    *, block_days: int, replications: int, confidence: float, seed: int
) -> dict[str, Any]:
    return {
        "paired_by": "closed_loop_validation_day",
        "bootstrap": "circular_moving_block",
        "block_days": int(block_days),
        "replications": int(replications),
        "confidence": float(confidence),
        "seed": int(seed),
        "difference_direction": "model_minus_registered_causal_baseline",
    }


def _paired_metric(
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    label: str,
    metrics: dict[str, Any],
    reasons: list[str],
    block_days: int,
    replications: int,
    confidence: float,
    seed: int,
    upper_limit: float = 0.0,
) -> None:
    inference = _paired_inference(
        model - baseline,
        block_days=block_days,
        replications=replications,
        confidence=confidence,
        seed=seed,
    )
    metrics[f"{label}_model_mean"] = float(model.mean())
    metrics[f"{label}_baseline_mean"] = float(baseline.mean())
    metrics[f"{label}_paired_delta"] = inference
    if float(inference["one_sided_upper"]) >= float(upper_limit):
        reasons.append(f"{label} paired one-sided upper is not below {upper_limit:g}")


def evaluate_world_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    reconstruction_outputs: Sequence[str] = ("public_21", "queue", "soc"),
    dynamics_horizons: Sequence[int] = (1, 4, 16, 96),
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    validation_start: str | date = VALIDATION_START,
    validation_end: str | date = VALIDATION_END,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 20240813,
) -> GateResult:
    """Require non-inferior reconstruction and every dynamics horizon."""

    rows, checkpoint, dates = _ordered_rows(
        records,
        gate="world",
        expected_days=expected_days,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    if tuple(int(value) for value in dynamics_horizons) != (1, 4, 16, 96):
        raise WorldGateError("world dynamics gate requires horizons 1/4/16/96")
    if not reconstruction_outputs:
        raise WorldGateError("world reconstruction gate requires named outputs")
    metrics: dict[str, Any] = {"validation_days": len(rows)}
    reasons: list[str] = []
    offset = 0
    for output in reconstruction_outputs:
        model = np.asarray(
            [
                _nested_scalar(
                    row,
                    ("model_reconstruction_losses",),
                    str(output),
                    aliases=(f"model_reconstruction_{output}",),
                    label=f"model reconstruction {output}",
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [
                _nested_scalar(
                    row,
                    ("baseline_reconstruction_losses",),
                    str(output),
                    aliases=(f"baseline_reconstruction_{output}",),
                    label=f"baseline reconstruction {output}",
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        _paired_metric(
            model,
            baseline,
            label=f"reconstruction_{output}",
            metrics=metrics,
            reasons=reasons,
            block_days=block_days,
            replications=replications,
            confidence=confidence,
            seed=seed + offset,
        )
        offset += 1
    for horizon in dynamics_horizons:
        model = np.asarray(
            [
                _nested_scalar(
                    row,
                    ("model_dynamics_losses",),
                    int(horizon),
                    aliases=(f"model_dynamics_h{horizon}",),
                    label=f"model dynamics h{horizon}",
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [
                _nested_scalar(
                    row,
                    ("baseline_dynamics_losses",),
                    int(horizon),
                    aliases=(f"baseline_dynamics_h{horizon}",),
                    label=f"baseline dynamics h{horizon}",
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        _paired_metric(
            model,
            baseline,
            label=f"dynamics_h{horizon}",
            metrics=metrics,
            reasons=reasons,
            block_days=block_days,
            replications=replications,
            confidence=confidence,
            seed=seed + offset,
        )
        offset += 1
    return GateResult(
        schema=GATE_SCHEMA,
        gate="world",
        status="PASS" if not reasons else "FAIL",
        passed=not reasons,
        checkpoint_id=checkpoint,
        dates=dates,
        metrics=metrics,
        reasons=tuple(reasons),
        method=_method(
            block_days=block_days,
            replications=replications,
            confidence=confidence,
            seed=seed,
        ),
    )


def evaluate_stochastic_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    validation_start: str | date = VALIDATION_START,
    validation_end: str | date = VALIDATION_END,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 20240813,
    coverage_tolerance: float = COVERAGE_ERROR_TOLERANCE,
    minimum_diversity: float = MINIMUM_DIVERSITY,
    maximum_duplicate_rate: float = MAXIMUM_DUPLICATE_RATE,
) -> GateResult:
    """Require all proper scores, coverage, tails, and non-collapse."""

    rows, checkpoint, dates = _ordered_rows(
        records,
        gate="stochastic",
        expected_days=expected_days,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    metrics: dict[str, Any] = {"validation_days": len(rows)}
    reasons: list[str] = []
    for offset, score in enumerate(("nll", "crps", "energy", "variogram", "tail")):
        model = _vector(rows, (f"model_{score}",), label=f"model {score}")
        baseline = _vector(rows, (f"baseline_{score}",), label=f"baseline {score}")
        _paired_metric(
            model,
            baseline,
            label=score,
            metrics=metrics,
            reasons=reasons,
            block_days=block_days,
            replications=replications,
            confidence=confidence,
            seed=seed + offset,
        )
    coverage_metrics: dict[str, Any] = {}
    for nominal in (50, 80, 90):
        coverage = _vector(
            rows,
            (f"coverage_{nominal}", f"empirical_coverage_{nominal}"),
            label=f"coverage {nominal}",
        )
        error = abs(float(coverage.mean()) - nominal / 100.0)
        coverage_metrics[str(nominal)] = {
            "empirical_mean": float(coverage.mean()),
            "absolute_error": error,
        }
        if error > float(coverage_tolerance):
            reasons.append(f"coverage {nominal} error exceeds {coverage_tolerance:g}")
    metrics["coverage"] = coverage_metrics
    metrics["coverage_error_tolerance"] = float(coverage_tolerance)
    diversity = _vector(
        rows,
        ("scenario_diversity", "mean_pairwise_scenario_distance"),
        label="scenario diversity",
    )
    duplicate_rate = _vector(
        rows, ("scenario_duplicate_rate",), label="scenario duplicate rate"
    )
    metrics["scenario_diversity_mean"] = float(diversity.mean())
    metrics["scenario_duplicate_rate_mean"] = float(duplicate_rate.mean())
    if float(diversity.mean()) < float(minimum_diversity):
        reasons.append("scenario diversity is below the registered minimum")
    if float(duplicate_rate.mean()) > float(maximum_duplicate_rate):
        reasons.append("scenario duplicate rate exceeds the registered maximum")
    return GateResult(
        schema=GATE_SCHEMA,
        gate="stochastic",
        status="PASS" if not reasons else "FAIL",
        passed=not reasons,
        checkpoint_id=checkpoint,
        dates=dates,
        metrics=metrics,
        reasons=tuple(reasons),
        method=_method(
            block_days=block_days,
            replications=replications,
            confidence=confidence,
            seed=seed,
        ),
    )


def evaluate_planner_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    validation_start: str | date = VALIDATION_START,
    validation_end: str | date = VALIDATION_END,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 20240813,
    feasibility_tolerance: float = PHYSICS_TOLERANCE,
    sla_increase_tolerance: float = SLA_INCREASE_TOLERANCE,
) -> GateResult:
    """Require planner value, tail value, feasibility, and SLA protection."""

    rows, checkpoint, dates = _ordered_rows(
        records,
        gate="planner",
        expected_days=expected_days,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    metrics: dict[str, Any] = {"validation_days": len(rows)}
    reasons: list[str] = []
    for offset, score in enumerate(("closed_loop_cost", "closed_loop_cvar")):
        model = _vector(rows, (f"model_{score}",), label=f"model {score}")
        baseline = _vector(rows, (f"baseline_{score}",), label=f"baseline {score}")
        _paired_metric(
            model,
            baseline,
            label=score,
            metrics=metrics,
            reasons=reasons,
            block_days=block_days,
            replications=replications,
            confidence=confidence,
            seed=seed + offset,
        )
    violation = _vector(
        rows,
        ("constraint_violation_max_abs", "planner_constraint_violation_max_abs"),
        label="planner constraint violation",
    )
    violation_max = float(violation.max())
    metrics["constraint_violation_max_abs"] = violation_max
    metrics["feasibility_tolerance"] = float(feasibility_tolerance)
    if violation_max > float(feasibility_tolerance):
        reasons.append("planner constraint violation exceeds tolerance")
    model_sla = _vector(rows, ("model_sla_violation_rate",), label="model SLA")
    baseline_sla = _vector(rows, ("baseline_sla_violation_rate",), label="baseline SLA")
    _paired_metric(
        model_sla,
        baseline_sla,
        label="sla_violation_rate",
        metrics=metrics,
        reasons=reasons,
        block_days=block_days,
        replications=replications,
        confidence=confidence,
        seed=seed + 2,
        upper_limit=float(sla_increase_tolerance),
    )
    metrics["sla_increase_tolerance"] = float(sla_increase_tolerance)
    return GateResult(
        schema=GATE_SCHEMA,
        gate="planner",
        status="PASS" if not reasons else "FAIL",
        passed=not reasons,
        checkpoint_id=checkpoint,
        dates=dates,
        metrics=metrics,
        reasons=tuple(reasons),
        method=_method(
            block_days=block_days,
            replications=replications,
            confidence=confidence,
            seed=seed,
        ),
    )


def evaluate_closure_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    validation_start: str | date = VALIDATION_START,
    validation_end: str | date = VALIDATION_END,
    identity_tolerance: float = IDENTITY_TOLERANCE,
    physics_tolerance: float = PHYSICS_TOLERANCE,
) -> GateResult:
    """Require exact DART construction and deterministic physical closures."""

    rows, checkpoint, dates = _ordered_rows(
        records,
        gate="closure",
        expected_days=expected_days,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    reasons: list[str] = []
    metrics: dict[str, Any] = {"validation_days": len(rows)}
    closures = {
        "dart_identity": (
            ("dart_identity_max_abs", "da_rt_dart_identity_max_abs"),
            float(identity_tolerance),
        ),
        "power_balance": (
            ("power_balance_residual_max_abs",),
            float(physics_tolerance),
        ),
        "soc_balance": (("soc_balance_residual_max_abs",), float(physics_tolerance)),
        "queue_balance": (
            ("queue_balance_residual_max_abs",),
            float(physics_tolerance),
        ),
        "action_bounds": (
            ("action_bound_violation_max_abs",),
            float(physics_tolerance),
        ),
        "shield": (("shield_residual_max_abs",), float(physics_tolerance)),
    }
    for label, (aliases, tolerance) in closures.items():
        values = _vector(rows, aliases, label=label)
        maximum = float(np.abs(values).max())
        metrics[f"{label}_max_abs"] = maximum
        metrics[f"{label}_tolerance"] = tolerance
        if maximum > tolerance:
            reasons.append(f"{label} exceeds tolerance {tolerance:g}")
    return GateResult(
        schema=GATE_SCHEMA,
        gate="closure",
        status="PASS" if not reasons else "FAIL",
        passed=not reasons,
        checkpoint_id=checkpoint,
        dates=dates,
        metrics=metrics,
        reasons=tuple(reasons),
        method={
            "reduction": "maximum_absolute_daily_residual",
            "dart_identity": "constructed_exactly_as_DA_minus_RT",
            "physics": "deterministic_rollout_followed_by_analytic_shield",
        },
    )


def evaluate_same_checkpoint_unified_gate(
    gates: Mapping[str, GateResult],
) -> UnifiedGateResult:
    """Release a model only when all C1 gates pass on one checkpoint."""

    required = ("world", "stochastic", "planner", "closure")
    if tuple(gates) != required:
        raise WorldGateError(f"unified gate requires ordered gates {list(required)}")
    for name, gate in gates.items():
        if not isinstance(gate, GateResult) or gate.schema != GATE_SCHEMA:
            raise WorldGateError(f"{name} is not native VPP-WorldMoE gate evidence")
        if gate.gate != name:
            raise WorldGateError(
                f"gate mapping key {name!r} does not match its payload"
            )
        if gate.locked_2025_opened:
            raise WorldGateError("a component gate opened locked 2025")
    checkpoints = {gate.checkpoint_id for gate in gates.values()}
    dates = {gate.dates for gate in gates.values()}
    reasons: list[str] = []
    if len(checkpoints) != 1:
        reasons.append("component gates do not use the same checkpoint")
    if len(dates) != 1:
        reasons.append("component gates do not use the same C1 date axis")
    for name, gate in gates.items():
        if not gate.passed:
            reasons.append(f"{name} gate failed")
    common_dates = next(iter(dates)) if len(dates) == 1 else ()
    if len(common_dates) != EXPECTED_VALIDATION_DAYS:
        reasons.append(f"C1 does not contain {EXPECTED_VALIDATION_DAYS} days")
    passed = not reasons
    return UnifiedGateResult(
        schema=UNIFIED_SCHEMA,
        status="PASS" if passed else "FAIL",
        passed=passed,
        checkpoint_id=next(iter(checkpoints)) if len(checkpoints) == 1 else None,
        dates=common_dates,
        validation_days=len(common_dates),
        gates=dict(gates),
        reasons=tuple(reasons),
    )
