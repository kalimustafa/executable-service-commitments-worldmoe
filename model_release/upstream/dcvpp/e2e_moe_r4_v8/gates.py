"""Unified paper-method scientific gates for R4-V8.

All primary comparisons are paired by validation day.  Representation and the
Distribution Energy Score use the registered one-sided 95% seven-day circular
moving-block bootstrap.  Point MAE is neither read nor used by any gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

from dcvpp.e2e_moe_r4.statistics import paired_daily_inference


EXPECTED_VALIDATION_DAYS = 364
VALIDATION_START = date(2024, 1, 1)
VALIDATION_END = date(2024, 12, 31)
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
SECONDARY_RATIO_LIMIT = 1.01
COVERAGE_ERROR_LIMIT = 0.03
IDENTITY_TOLERANCE = 1.0e-6
SLA_INCREASE_LIMIT = 0.001

GATE_SCHEMA = "dcvpp.r4.v8.scientific-gate.v1"
UNIFIED_GATE_SCHEMA = "dcvpp.r4.v8.unified-scientific-gate.v1"


class ScientificGateError(ValueError):
    """Raised when daily evidence is incomplete, incomparable, or locked."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ScientificGateResult:
    schema: str
    gate: str
    status: str
    passed: bool
    checkpoint_id: str
    dates: tuple[str, ...]
    metrics: Mapping[str, Any]
    reasons: tuple[str, ...]
    method: Mapping[str, Any]
    mae_used: bool = False
    locked_2025_opened: bool = False

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class UnifiedScientificGateResult:
    schema: str
    status: str
    passed: bool
    checkpoint_id: str | None
    dates: tuple[str, ...]
    validation_days: int
    gates: Mapping[str, ScientificGateResult]
    reasons: tuple[str, ...]
    mae_used: bool = False
    locked_2025_opened: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "passed": self.passed,
            "checkpoint_id": self.checkpoint_id,
            "dates": list(self.dates),
            "validation_days": self.validation_days,
            "gates": {name: gate.as_dict() for name, gate in self.gates.items()},
            "reasons": list(self.reasons),
            "mae_used": self.mae_used,
            "locked_2025_opened": self.locked_2025_opened,
        }


def _checkpoint_id(row: Mapping[str, Any]) -> str:
    for name in ("checkpoint_sha256", "checkpoint_id", "checkpoint"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    raise ScientificGateError("every daily metric row requires a checkpoint identifier")


def _ordered_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    gate: str,
    expected_days: int,
    validation_start: str | date,
    validation_end: str | date,
) -> tuple[tuple[dict[str, Any], ...], str, tuple[str, ...]]:
    rows = tuple(dict(row) for row in records)
    if int(expected_days) < 1:
        raise ScientificGateError("expected_days must be positive")
    if len(rows) != int(expected_days):
        raise ScientificGateError(
            f"{gate} requires exactly {expected_days} daily rows; got {len(rows)}"
        )
    try:
        start = date.fromisoformat(str(validation_start)[:10])
        end = date.fromisoformat(str(validation_end)[:10])
    except ValueError as exc:
        raise ScientificGateError("validation bounds must be ISO dates") from exc
    if start > end:
        raise ScientificGateError("validation_start must not exceed validation_end")
    if start <= date(2025, 12, 31) and end >= date(2025, 1, 1):
        raise ScientificGateError("locked 2025 cannot enter a V8 scientific gate")

    parsed: list[date] = []
    for row in rows:
        try:
            current = date.fromisoformat(str(row["date"])[:10])
        except (KeyError, ValueError) as exc:
            raise ScientificGateError(f"{gate} row requires one ISO date") from exc
        if current.year == 2025:
            raise ScientificGateError("locked 2025 cannot enter a V8 scientific gate")
        if not start <= current <= end:
            raise ScientificGateError(
                f"{gate} date {current} lies outside {start} through {end}"
            )
        if row.get("locked_2025_opened") is True or row.get(
            "locked_test_2025_opened"
        ) is True:
            raise ScientificGateError("daily evidence reports that locked 2025 was opened")
        parsed.append(current)
    if len(set(parsed)) != len(parsed):
        raise ScientificGateError(f"{gate} daily dates must be unique")
    checkpoint_ids = {_checkpoint_id(row) for row in rows}
    if len(checkpoint_ids) != 1:
        raise ScientificGateError(
            f"{gate} daily rows must use one immutable checkpoint"
        )
    order = np.argsort(np.asarray(parsed, dtype="datetime64[D]"), kind="stable")
    ordered = tuple(rows[int(index)] for index in order)
    dates = tuple(str(row["date"])[:10] for row in ordered)
    return ordered, next(iter(checkpoint_ids)), dates


def _value(row: Mapping[str, Any], aliases: Sequence[str], *, label: str) -> float:
    for name in aliases:
        if name in row:
            try:
                result = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ScientificGateError(f"invalid daily metric {name!r}") from exc
            if not np.isfinite(result):
                raise ScientificGateError(f"daily metric {name!r} is non-finite")
            return result
    raise ScientificGateError(
        f"missing daily metric {label!r}; accepted keys={list(aliases)}"
    )


def _vector(
    rows: Sequence[Mapping[str, Any]], aliases: Sequence[str], *, label: str
) -> np.ndarray:
    values = np.asarray(
        [_value(row, aliases, label=label) for row in rows], dtype=np.float64
    )
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ScientificGateError(f"daily metric {label!r} must be finite scalars")
    return values


def _nested_loss(row: Mapping[str, Any], side: str, output: str) -> float | None:
    nested = row.get(f"{side}_losses")
    if not isinstance(nested, Mapping) or output not in nested:
        return None
    try:
        value = float(nested[output])
    except (TypeError, ValueError) as exc:
        raise ScientificGateError(
            f"invalid {side}_losses.{output} daily metric"
        ) from exc
    if not np.isfinite(value):
        raise ScientificGateError(f"{side}_losses.{output} is non-finite")
    return value


def _representation_vector(
    rows: Sequence[Mapping[str, Any]], side: str, output: str
) -> np.ndarray:
    aliases = (
        f"{side}_{output}_loss",
        f"{side}_{output}_normalized_loss",
    )
    values: list[float] = []
    for row in rows:
        nested = _nested_loss(row, side, output)
        values.append(
            nested
            if nested is not None
            else _value(row, aliases, label=f"{side}_{output}_loss")
        )
    return np.asarray(values, dtype=np.float64)


def _inference(
    differences: np.ndarray,
    *,
    seed: int,
    block_days: int,
    replications: int,
    confidence: float,
) -> dict[str, float | int]:
    result = paired_daily_inference(
        differences,
        block_days=int(block_days),
        replicates=int(replications),
        confidence=float(confidence),
        seed=int(seed),
    )
    return {
        key: value
        for key, value in result.as_dict().items()
        if isinstance(value, (int, float))
    }


def _method(
    *, block_days: int, replications: int, confidence: float, seed: int
) -> dict[str, Any]:
    return {
        "paired_by": "validation_day",
        "bootstrap": "circular_moving_block",
        "block_days": int(block_days),
        "replications": int(replications),
        "confidence": float(confidence),
        "seed": int(seed),
        "difference_direction": "model_minus_baseline",
        "mae_used": False,
    }


def evaluate_representation_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    outputs: Sequence[str] = ("dart", "da", "rt", "workload", "pv", "pue"),
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    validation_start: str | date = VALIDATION_START,
    validation_end: str | date = VALIDATION_END,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 20240813,
    identity_tolerance: float = IDENTITY_TOLERANCE,
) -> ScientificGateResult:
    rows, checkpoint, dates = _ordered_rows(
        records,
        gate="representation",
        expected_days=expected_days,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    if not outputs or len(set(outputs)) != len(tuple(outputs)):
        raise ScientificGateError("Representation outputs must be unique and nonempty")
    reasons: list[str] = []
    metrics: dict[str, Any] = {"validation_days": len(rows)}
    for offset, output in enumerate(outputs):
        model = _representation_vector(rows, "model", str(output))
        baseline = _representation_vector(rows, "baseline", str(output))
        inference = _inference(
            model - baseline,
            seed=seed + offset,
            block_days=block_days,
            replications=replications,
            confidence=confidence,
        )
        metrics[f"{output}_model_loss_mean"] = float(model.mean())
        metrics[f"{output}_baseline_loss_mean"] = float(baseline.mean())
        metrics[f"{output}_paired_delta"] = inference
        metrics[f"{output}_paired_delta_one_sided_upper"] = float(
            inference["one_sided_upper"]
        )
        if float(inference["one_sided_upper"]) >= 0.0:
            reasons.append(
                f"{output} paired loss model-minus-baseline one-sided upper is not below zero"
            )
    identity = _vector(
        rows,
        (
            "identity_residual",
            "model_identity_residual",
            "da_rt_dart_identity_max_abs",
            "model_da_rt_dart_identity_max_abs",
        ),
        label="DA-RT-DART identity residual",
    )
    identity_max = float(identity.max())
    metrics["identity_residual_max_abs"] = identity_max
    metrics["identity_tolerance"] = float(identity_tolerance)
    if identity_max > float(identity_tolerance):
        reasons.append(f"identity residual exceeds {identity_tolerance:g}")
    return ScientificGateResult(
        schema=GATE_SCHEMA,
        gate="representation",
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


_DISTRIBUTION_ALIASES = {
    "energy_score": ("energy_score",),
    "marginal_crps": ("marginal_crps", "crps"),
    "interval_score_90": ("interval_score_90", "interval90"),
    "interval_score_95": ("interval_score_95", "interval95"),
    "tail_score": ("tail_score", "tail"),
    "dependence_score": ("dependence_score", "variogram_score", "dependence"),
}


def _score_vector(
    rows: Sequence[Mapping[str, Any]], side: str, score: str
) -> np.ndarray:
    aliases = tuple(
        f"{side}_{name}" for name in _DISTRIBUTION_ALIASES[score]
    )
    return _vector(rows, aliases, label=f"{side}_{score}")


def evaluate_distribution_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    validation_start: str | date = VALIDATION_START,
    validation_end: str | date = VALIDATION_END,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 20240813,
    secondary_ratio_limit: float = SECONDARY_RATIO_LIMIT,
    coverage_error_limit: float = COVERAGE_ERROR_LIMIT,
    identity_tolerance: float = IDENTITY_TOLERANCE,
) -> ScientificGateResult:
    rows, checkpoint, dates = _ordered_rows(
        records,
        gate="distribution",
        expected_days=expected_days,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    reasons: list[str] = []
    metrics: dict[str, Any] = {"validation_days": len(rows)}
    model_energy = _score_vector(rows, "model", "energy_score")
    baseline_energy = _score_vector(rows, "baseline", "energy_score")
    energy = _inference(
        model_energy - baseline_energy,
        seed=seed,
        block_days=block_days,
        replications=replications,
        confidence=confidence,
    )
    metrics["energy_score_paired_delta"] = energy
    metrics["energy_score_paired_delta_one_sided_upper"] = float(
        energy["one_sided_upper"]
    )
    if float(energy["one_sided_upper"]) >= 0.0:
        reasons.append("paired Energy Score one-sided upper is not below zero")

    for score in (
        "marginal_crps",
        "interval_score_90",
        "interval_score_95",
        "tail_score",
        "dependence_score",
    ):
        model = _score_vector(rows, "model", score)
        baseline = _score_vector(rows, "baseline", score)
        baseline_mean = float(baseline.mean())
        if baseline_mean <= 0.0:
            raise ScientificGateError(f"baseline {score} mean must be positive")
        ratio = float(model.mean() / baseline_mean)
        if np.isclose(
            ratio, float(secondary_ratio_limit), rtol=0.0, atol=1.0e-12
        ):
            ratio = float(secondary_ratio_limit)
        metrics[f"model_{score}"] = float(model.mean())
        metrics[f"baseline_{score}"] = baseline_mean
        metrics[f"{score}_ratio"] = ratio
        if ratio > float(secondary_ratio_limit):
            reasons.append(f"{score} ratio exceeds {secondary_ratio_limit:g}")

    for level in (90, 95):
        observed = float(
            _vector(
                rows,
                (f"model_coverage_{level}", f"coverage_{level}"),
                label=f"model_coverage_{level}",
            ).mean()
        )
        error = abs(observed - level / 100.0)
        if np.isclose(
            error, float(coverage_error_limit), rtol=0.0, atol=1.0e-12
        ):
            error = float(coverage_error_limit)
        metrics[f"model_coverage_{level}"] = observed
        metrics[f"absolute_coverage_error_{level}"] = error
        if error > float(coverage_error_limit):
            reasons.append(f"coverage {level} error exceeds {coverage_error_limit:g}")

    identity = _vector(
        rows,
        (
            "identity_residual",
            "model_identity_residual",
            "da_rt_dart_identity_max_abs",
            "model_da_rt_dart_identity_max_abs",
        ),
        label="DA-RT-DART identity residual",
    )
    identity_max = float(identity.max())
    if np.isclose(
        identity_max, float(identity_tolerance), rtol=0.0, atol=1.0e-12
    ):
        identity_max = float(identity_tolerance)
    metrics["identity_residual_max_abs"] = identity_max
    if identity_max > float(identity_tolerance):
        reasons.append(f"identity residual exceeds {identity_tolerance:g}")
    metrics["secondary_ratio_limit"] = float(secondary_ratio_limit)
    metrics["coverage_error_limit"] = float(coverage_error_limit)
    metrics["identity_tolerance"] = float(identity_tolerance)
    return ScientificGateResult(
        schema=GATE_SCHEMA,
        gate="distribution",
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


def _policy_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    direct_aliases: Sequence[str],
    model_aliases: Sequence[str],
    baseline_aliases: Sequence[str],
    label: str,
) -> np.ndarray:
    if all(any(name in row for name in direct_aliases) for row in rows):
        return _vector(rows, direct_aliases, label=label)
    return _vector(rows, model_aliases, label=f"model_{label}") - _vector(
        rows, baseline_aliases, label=f"baseline_{label}"
    )


def evaluate_policy_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    validation_start: str | date = VALIDATION_START,
    validation_end: str | date = VALIDATION_END,
    block_days: int = BOOTSTRAP_BLOCK_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 20240813,
    sla_increase_limit: float = SLA_INCREASE_LIMIT,
) -> ScientificGateResult:
    rows, checkpoint, dates = _ordered_rows(
        records,
        gate="policy",
        expected_days=expected_days,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    reasons: list[str] = []
    cost_delta = _policy_delta(
        rows,
        direct_aliases=("cost_delta", "cost_delta_usd", "daily_cost_delta"),
        model_aliases=("model_daily_cost", "operating_cost_usd", "model_cost"),
        baseline_aliases=(
            "baseline_daily_cost",
            "benchmark_operating_cost_usd",
            "baseline_cost",
        ),
        label="daily_cost",
    )
    cost = _inference(
        cost_delta,
        seed=seed,
        block_days=block_days,
        replications=replications,
        confidence=confidence,
    )
    cost_upper = float(cost["one_sided_upper"])
    if np.isclose(cost_upper, 0.0, rtol=0.0, atol=1.0e-12):
        cost_upper = 0.0
        cost["one_sided_upper"] = 0.0
    if cost_upper > 0.0:
        reasons.append("paired daily cost one-sided upper exceeds zero")
    sla_delta = _policy_delta(
        rows,
        direct_aliases=("sla_delta", "normalized_sla_increase"),
        model_aliases=(
            "model_normalized_sla",
            "normalized_sla_shortfall",
            "model_sla",
        ),
        baseline_aliases=(
            "baseline_normalized_sla",
            "benchmark_normalized_sla_shortfall",
            "baseline_sla",
        ),
        label="normalized_sla",
    )
    aggregate_sla = float(sla_delta.mean())
    at_sla_boundary = np.isclose(
        aggregate_sla,
        float(sla_increase_limit),
        rtol=0.0,
        atol=1.0e-12,
    )
    if at_sla_boundary:
        aggregate_sla = float(sla_increase_limit)
    if aggregate_sla > float(sla_increase_limit):
        reasons.append(f"normalized SLA increase exceeds {sla_increase_limit:g}")
    feasibility = _vector(
        rows,
        ("post_shield_feasibility", "feasibility", "feasibility_fraction"),
        label="post_shield_feasibility",
    )
    feasibility_fraction = float(feasibility.mean())
    if not bool(np.all(feasibility == 1.0)):
        reasons.append("post-shield feasibility is not exactly one on every day")
    rejected = _vector(
        rows,
        (
            "non_exogenous_rejected_origins",
            "rejected_origins",
            "rejected_non_exogenous_origins",
        ),
        label="non_exogenous_rejected_origins",
    )
    rejected_total = float(rejected.sum())
    if rejected_total != 0.0:
        reasons.append("non-exogenous rejected origins are nonzero")
    metrics: dict[str, Any] = {
        "validation_days": len(rows),
        "cost_paired_delta": cost,
        "cost_paired_delta_one_sided_upper": cost_upper,
        "aggregate_normalized_sla_increase": aggregate_sla,
        "post_shield_feasibility": feasibility_fraction,
        "non_exogenous_rejected_origins": rejected_total,
        "sla_increase_limit": float(sla_increase_limit),
    }
    return ScientificGateResult(
        schema=GATE_SCHEMA,
        gate="policy",
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


def evaluate_unified_gate(
    representation: ScientificGateResult,
    distribution: ScientificGateResult,
    policy: ScientificGateResult,
    *,
    expected_days: int = EXPECTED_VALIDATION_DAYS,
) -> UnifiedScientificGateResult:
    gates = {
        "representation": representation,
        "distribution": distribution,
        "policy": policy,
    }
    for name, gate in gates.items():
        if not isinstance(gate, ScientificGateResult) or gate.gate != name:
            raise ScientificGateError(f"unified gate requires the {name} gate result")
    reasons: list[str] = []
    checkpoint_ids = {gate.checkpoint_id for gate in gates.values()}
    checkpoint = next(iter(checkpoint_ids)) if len(checkpoint_ids) == 1 else None
    if len(checkpoint_ids) != 1:
        reasons.append("three gates do not refer to the same checkpoint")
    date_panels = {gate.dates for gate in gates.values()}
    dates = representation.dates if len(date_panels) == 1 else ()
    if len(date_panels) != 1:
        reasons.append("three gates do not use the same daily validation panel")
    if any(len(gate.dates) != int(expected_days) for gate in gates.values()):
        reasons.append(f"every gate must contain exactly {expected_days} validation days")
    for name, gate in gates.items():
        if not gate.passed or gate.status != "PASS":
            reasons.append(f"{name} scientific gate did not pass")
        if gate.mae_used:
            reasons.append(f"{name} gate improperly used MAE")
        if gate.locked_2025_opened or any(value.startswith("2025-") for value in gate.dates):
            reasons.append(f"{name} gate opened locked 2025")
    return UnifiedScientificGateResult(
        schema=UNIFIED_GATE_SCHEMA,
        status="PASS" if not reasons else "FAIL",
        passed=not reasons,
        checkpoint_id=checkpoint,
        dates=dates,
        validation_days=len(dates),
        gates=gates,
        reasons=tuple(reasons),
    )


def evaluate_all_scientific_gates(
    representation_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = EXPECTED_VALIDATION_DAYS,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = 20240813,
) -> UnifiedScientificGateResult:
    representation = evaluate_representation_gate(
        representation_rows,
        expected_days=expected_days,
        replications=replications,
        seed=seed,
    )
    distribution = evaluate_distribution_gate(
        distribution_rows,
        expected_days=expected_days,
        replications=replications,
        seed=seed + 100,
    )
    policy = evaluate_policy_gate(
        policy_rows,
        expected_days=expected_days,
        replications=replications,
        seed=seed + 200,
    )
    return evaluate_unified_gate(
        representation, distribution, policy, expected_days=expected_days
    )


__all__ = [
    "BOOTSTRAP_BLOCK_DAYS",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_REPLICATIONS",
    "COVERAGE_ERROR_LIMIT",
    "EXPECTED_VALIDATION_DAYS",
    "GATE_SCHEMA",
    "IDENTITY_TOLERANCE",
    "SECONDARY_RATIO_LIMIT",
    "SLA_INCREASE_LIMIT",
    "ScientificGateError",
    "ScientificGateResult",
    "UNIFIED_GATE_SCHEMA",
    "UnifiedScientificGateResult",
    "evaluate_all_scientific_gates",
    "evaluate_distribution_gate",
    "evaluate_policy_gate",
    "evaluate_representation_gate",
    "evaluate_unified_gate",
]
