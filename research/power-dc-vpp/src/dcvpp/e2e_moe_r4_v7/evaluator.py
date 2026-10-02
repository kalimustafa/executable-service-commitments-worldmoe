"""Shared-path N=512 post-shield evaluation for R4-V7.

The evaluator freezes one exogenous path panel per operating day and passes the
same object to the learned policy and the Price-aware EDF benchmark.  Risk is
computed from executed, post-shield costs; a policy's own predictive law never
scores that policy.  Small panels are supported only for implementation parity
tests, while formal daily rows require exactly 512 paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
import math
from typing import Any

import numpy as np

from dcvpp.e2e_moe_r4.hashing import sha256_bytes

from .runtime import (
    HORIZON,
    SITES,
    Locked2025Error,
    V7Runtime,
    V7RuntimeError,
)


FORMAL_PATHS = 512
FORMAL_ALPHA = 0.95
DEFAULT_CAPABILITY_DAYS = 179
EVALUATION_PATH_SCHEMA = "dcvpp.r4.v7.frozen-evaluation-paths.v1"
DAILY_ROW_SCHEMA = "dcvpp.r4.v7.policy-capability-daily-row.v1"
CAPABILITY_GATE_SCHEMA = "dcvpp.r4.v7.policy-capability-gate.v1"


class V7EvaluationError(V7RuntimeError):
    """Raised when formal evaluation evidence is incomplete or incomparable."""


def _locked_date(value: Any) -> str:
    text = str(value)[:10]
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise V7EvaluationError(f"invalid operating date: {value!r}") from exc
    if parsed.year == 2025:
        raise Locked2025Error("locked test year is not available to V7 evaluation")
    return parsed.isoformat()


def _finite_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int | None, ...] | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if shape is not None:
        if array.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape, strict=True)
        ):
            raise V7EvaluationError(f"{name} has invalid shape {array.shape}; expected {shape}")
    if not np.isfinite(array).all():
        raise V7EvaluationError(f"{name} contains non-finite values")
    if nonnegative and np.any(array < 0.0):
        raise V7EvaluationError(f"{name} must be nonnegative")
    return array


def _hash_array(payload: bytearray, name: str, value: np.ndarray) -> None:
    canonical = np.ascontiguousarray(value.astype("<f8", copy=False))
    payload.extend(name.encode("utf-8"))
    payload.extend(str(canonical.shape).encode("ascii"))
    payload.extend(canonical.tobytes(order="C"))


def _immutable_array(value: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(value)
    return np.frombuffer(canonical.tobytes(order="C"), dtype=canonical.dtype).reshape(
        canonical.shape
    )


@dataclass(frozen=True)
class FrozenEvaluationPaths:
    """Immutable exogenous panel shared by every policy in one paired test."""

    operating_date: str
    rt_price: np.ndarray
    da_price: np.ndarray
    arrival: np.ndarray
    pv_mw: np.ndarray
    physical_multiplier: np.ndarray
    initial_soc_mwh: np.ndarray
    weights: np.ndarray
    seed: int
    source_contract: str
    formal: bool = True
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        operating_date = _locked_date(self.operating_date)
        rt = _finite_array(
            self.rt_price, name="rt_price", shape=(None, HORIZON, SITES)
        )
        count = rt.shape[0]
        if count < 1:
            raise V7EvaluationError("evaluation panel must contain at least one path")
        da = _finite_array(self.da_price, name="da_price")
        if da.shape == (HORIZON, SITES):
            da = np.broadcast_to(da, (count, HORIZON, SITES)).copy()
        if da.shape != rt.shape:
            raise V7EvaluationError("da_price must be [96,4] or [N,96,4]")
        arrival = _finite_array(
            self.arrival,
            name="arrival",
            shape=(count, HORIZON),
            nonnegative=True,
        )
        pv = _finite_array(
            self.pv_mw,
            name="pv_mw",
            shape=(count, HORIZON, SITES),
            nonnegative=True,
        )
        multiplier = _finite_array(self.physical_multiplier, name="physical_multiplier")
        if multiplier.shape == (SITES,):
            multiplier = np.broadcast_to(multiplier, (count, HORIZON, SITES)).copy()
        elif multiplier.shape == (count, SITES):
            multiplier = np.broadcast_to(
                multiplier[:, None, :], (count, HORIZON, SITES)
            ).copy()
        if multiplier.shape != (count, HORIZON, SITES) or np.any(multiplier <= 0.0):
            raise V7EvaluationError(
                "physical_multiplier must be positive [4], [N,4], or [N,96,4]"
            )
        initial_soc = _finite_array(self.initial_soc_mwh, name="initial_soc_mwh")
        if initial_soc.shape == (SITES,):
            initial_soc = np.broadcast_to(initial_soc, (count, SITES)).copy()
        if initial_soc.shape != (count, SITES) or np.any(initial_soc < 0.0):
            raise V7EvaluationError("initial_soc_mwh must be nonnegative [4] or [N,4]")
        weights = _finite_array(
            self.weights, name="weights", shape=(count,), nonnegative=True
        )
        total = float(weights.sum())
        if total <= 0.0:
            raise V7EvaluationError("evaluation weights must have positive mass")
        weights /= total
        if self.formal and count != FORMAL_PATHS:
            raise V7EvaluationError(
                f"formal post-shield evaluation requires exactly N={FORMAL_PATHS}"
            )
        if not str(self.source_contract).strip():
            raise V7EvaluationError("source_contract must identify the frozen path generator")
        object.__setattr__(self, "operating_date", operating_date)
        for name, value in (
            ("rt_price", rt),
            ("da_price", da),
            ("arrival", arrival),
            ("pv_mw", pv),
            ("physical_multiplier", multiplier),
            ("initial_soc_mwh", initial_soc),
            ("weights", weights),
        ):
            object.__setattr__(self, name, _immutable_array(value))
        payload = bytearray()
        payload.extend(EVALUATION_PATH_SCHEMA.encode("utf-8"))
        payload.extend(operating_date.encode("ascii"))
        payload.extend(str(int(self.seed)).encode("ascii"))
        payload.extend(str(self.source_contract).encode("utf-8"))
        for name in (
            "rt_price",
            "da_price",
            "arrival",
            "pv_mw",
            "physical_multiplier",
            "initial_soc_mwh",
            "weights",
        ):
            _hash_array(payload, name, getattr(self, name))
        # The repository's isolated helper avoids the Windows OpenSSL/long-lived
        # Torch process collision; the digest is computed once, not per replay.
        object.__setattr__(self, "_sha256", sha256_bytes(bytes(payload)))

    @property
    def sample_count(self) -> int:
        return int(self.rt_price.shape[0])

    @property
    def sha256(self) -> str:
        return self._sha256

    def subset(self, index: int) -> "FrozenEvaluationPaths":
        if not 0 <= int(index) < self.sample_count:
            raise IndexError(index)
        location = slice(int(index), int(index) + 1)
        return FrozenEvaluationPaths(
            operating_date=self.operating_date,
            rt_price=self.rt_price[location],
            da_price=self.da_price[location],
            arrival=self.arrival[location],
            pv_mw=self.pv_mw[location],
            physical_multiplier=self.physical_multiplier[location],
            initial_soc_mwh=self.initial_soc_mwh[location],
            weights=np.ones(1, dtype=np.float64),
            seed=self.seed,
            source_contract=self.source_contract,
            formal=False,
        )


def freeze_evaluation_paths(
    *,
    operating_date: str,
    rt_price: Any,
    da_price: Any,
    arrival: Any,
    pv_mw: Any,
    physical_multiplier: Any,
    initial_soc_mwh: Any,
    weights: Any | None = None,
    seed: int,
    source_contract: str,
    formal: bool = True,
) -> FrozenEvaluationPaths:
    count = int(np.asarray(rt_price).shape[0])
    return FrozenEvaluationPaths(
        operating_date=operating_date,
        rt_price=rt_price,
        da_price=da_price,
        arrival=arrival,
        pv_mw=pv_mw,
        physical_multiplier=physical_multiplier,
        initial_soc_mwh=initial_soc_mwh,
        weights=np.full(count, 1.0 / count) if weights is None else weights,
        seed=int(seed),
        source_contract=str(source_contract),
        formal=bool(formal),
    )


def paths_from_distribution_samples(
    samples_zonal: Any,
    *,
    operating_date: str,
    initial_soc_mwh: Any,
    physical_multiplier: Any,
    weights: Any | None,
    seed: int,
    source_contract: str,
    formal: bool = True,
) -> FrozenEvaluationPaths:
    """Convert canonical ``[N,96,21]`` samples into a frozen replay panel."""

    samples = np.asarray(samples_zonal, dtype=np.float64)
    if samples.ndim == 4 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim != 3 or samples.shape[1:] != (HORIZON, 21):
        raise V7EvaluationError("distribution samples must be [N,96,21]")
    return freeze_evaluation_paths(
        operating_date=operating_date,
        rt_price=samples[:, :, 4:8],
        da_price=samples[:, :, 0:4],
        arrival=np.maximum(samples[:, :, 12], 0.0),
        pv_mw=np.maximum(samples[:, :, 13:17], 0.0),
        physical_multiplier=physical_multiplier,
        initial_soc_mwh=initial_soc_mwh,
        weights=weights,
        seed=seed,
        source_contract=source_contract,
        formal=formal,
    )


def exact_weighted_upper_cvar(
    values: Any, weights: Any, *, alpha: float = FORMAL_ALPHA
) -> tuple[float, float]:
    """Return exact weighted VaR/CVaR, including a fractional VaR atom."""

    observed = _finite_array(values, name="values")
    probability = _finite_array(weights, name="weights", nonnegative=True)
    if observed.ndim != 1 or probability.shape != observed.shape:
        raise V7EvaluationError("values and weights must be equal one-dimensional arrays")
    if not 0.0 < float(alpha) < 1.0:
        raise V7EvaluationError("alpha must lie in (0,1)")
    total = float(probability.sum())
    if total <= 0.0:
        raise V7EvaluationError("weights must have positive mass")
    probability = probability / total
    order = np.argsort(observed, kind="stable")
    sorted_values = observed[order]
    sorted_weights = probability[order]
    cumulative = np.cumsum(sorted_weights)
    index = min(int(np.searchsorted(cumulative, alpha, side="left")), len(observed) - 1)
    var = float(sorted_values[index])
    tail_mass = 1.0 - float(alpha)
    mass_above = float(sorted_weights[index + 1 :].sum())
    weighted_above = float(
        np.dot(sorted_values[index + 1 :], sorted_weights[index + 1 :])
    )
    boundary_mass = max(tail_mass - mass_above, 0.0)
    cvar = (weighted_above + boundary_mass * var) / tail_mass
    return var, float(cvar)


def _weighted_mean(value: Any, weights: np.ndarray) -> float:
    array = _finite_array(value, name="weighted value", shape=(len(weights),))
    return float(np.dot(array, weights))


def batched_post_shield_replay(
    runtime: V7Runtime,
    model: Any,
    evaluation_paths: FrozenEvaluationPaths,
    issue_batch: Mapping[str, Any],
    *,
    require_formal: bool = True,
    keep_origin_records: bool = False,
) -> dict[str, Any]:
    """Run all N paths in one batch per origin and attest path immutability."""

    if require_formal and (
        not evaluation_paths.formal or evaluation_paths.sample_count != FORMAL_PATHS
    ):
        raise V7EvaluationError(f"formal replay requires a frozen N={FORMAL_PATHS} panel")
    before = evaluation_paths.sha256
    result = runtime.run_day(
        model,
        issue_batch,
        evaluation_paths=evaluation_paths,
        issue_batch=issue_batch,
        keep_origin_records=keep_origin_records,
    )
    after = evaluation_paths.sha256
    if before != after:
        raise V7EvaluationError("evaluation paths changed during replay")
    count = evaluation_paths.sample_count
    for name in (
        "realized_cost",
        "terminal_soc_gap",
        "terminal_backlog",
        "sla_violation",
        "shield_intervention",
        "shield_intervention_rate",
        "feasible",
    ):
        if np.asarray(result.get(name)).shape != (count,):
            raise V7EvaluationError(f"batched replay field {name} is not [N]")
    if int(result.get("completed_origins", -1)) != HORIZON:
        raise V7EvaluationError("batched replay did not execute all 96 origins")
    return {
        **result,
        "evaluation_paths_sha256": before,
        "formal_post_shield_sample_count": count,
        "post_shield": True,
    }


def summarize_replay(
    replay: Mapping[str, Any],
    paths: FrozenEvaluationPaths,
    *,
    alpha: float = FORMAL_ALPHA,
) -> dict[str, float | bool]:
    if replay.get("evaluation_paths_sha256") != paths.sha256:
        raise V7EvaluationError("replay is not bound to the supplied evaluation paths")
    costs = _finite_array(
        replay.get("realized_cost"), name="realized_cost", shape=(paths.sample_count,)
    )
    var, cvar = exact_weighted_upper_cvar(costs, paths.weights, alpha=alpha)
    feasibility = np.asarray(replay.get("feasible"), dtype=bool)
    return {
        "expected_cost_usd": _weighted_mean(costs, paths.weights),
        "var95_usd": var,
        "cvar95_usd": cvar,
        "normalized_sla_shortfall": _weighted_mean(
            replay.get("sla_violation"), paths.weights
        )
        / max(_weighted_mean(paths.arrival.sum(axis=1), paths.weights), 1.0e-12),
        "shield_intervention_mean": _weighted_mean(
            replay.get("shield_intervention"), paths.weights
        ),
        "shield_intervention_rate": _weighted_mean(
            replay.get("shield_intervention_rate"), paths.weights
        ),
        "feasibility_fraction": _weighted_mean(feasibility.astype(float), paths.weights),
        "terminal_soc_gap_mean": _weighted_mean(
            replay.get("terminal_soc_gap"), paths.weights
        ),
        "terminal_backlog_mean": _weighted_mean(
            replay.get("terminal_backlog"), paths.weights
        ),
        "all_values_finite": True,
    }


def paired_post_shield_replay(
    runtime: V7Runtime,
    model: Any,
    benchmark: Any,
    evaluation_paths: FrozenEvaluationPaths,
    issue_batch: Mapping[str, Any],
    *,
    require_formal: bool = True,
) -> dict[str, Any]:
    """Evaluate learned and benchmark policies on the exact same frozen paths."""

    source_hash = evaluation_paths.sha256
    model_replay = batched_post_shield_replay(
        runtime,
        model,
        evaluation_paths,
        issue_batch,
        require_formal=require_formal,
    )
    benchmark_replay = batched_post_shield_replay(
        runtime,
        benchmark,
        evaluation_paths,
        issue_batch,
        require_formal=require_formal,
    )
    if (
        model_replay["evaluation_paths_sha256"]
        != benchmark_replay["evaluation_paths_sha256"]
        or model_replay["evaluation_paths_sha256"] != source_hash
    ):
        raise V7EvaluationError("model and benchmark did not share one frozen path panel")
    model_summary = summarize_replay(model_replay, evaluation_paths)
    benchmark_summary = summarize_replay(benchmark_replay, evaluation_paths)
    row = capability_daily_row(
        evaluation_paths,
        model_replay,
        benchmark_replay,
        model_summary=model_summary,
        benchmark_summary=benchmark_summary,
    )
    return {
        "schema_version": "dcvpp.r4.v7.paired-post-shield-replay.v1",
        "operating_date": evaluation_paths.operating_date,
        "evaluation_paths_sha256": source_hash,
        "common_frozen_evaluation_paths": True,
        "formal_post_shield_sample_count": evaluation_paths.sample_count,
        "model_replay": model_replay,
        "benchmark_replay": benchmark_replay,
        "model_summary": model_summary,
        "benchmark_summary": benchmark_summary,
        "daily_row": row,
        "locked_test_2025_opened": False,
    }


def capability_daily_row(
    paths: FrozenEvaluationPaths,
    model_replay: Mapping[str, Any],
    benchmark_replay: Mapping[str, Any],
    *,
    model_summary: Mapping[str, Any] | None = None,
    benchmark_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one-row-per-day evidence consumed by the policy gate."""

    _locked_date(paths.operating_date)
    model = dict(model_summary or summarize_replay(model_replay, paths))
    base = dict(benchmark_summary or summarize_replay(benchmark_replay, paths))
    if model_replay.get("evaluation_paths_sha256") != paths.sha256:
        raise V7EvaluationError("model replay uses a different evaluation panel")
    if benchmark_replay.get("evaluation_paths_sha256") != paths.sha256:
        raise V7EvaluationError("benchmark replay uses a different evaluation panel")
    row = {
        "schema_version": DAILY_ROW_SCHEMA,
        "date": paths.operating_date,
        "evaluation_paths_sha256": paths.sha256,
        "formal_post_shield_sample_count": paths.sample_count,
        "completed_origins": HORIZON,
        "post_shield": True,
        "common_frozen_evaluation_paths": True,
        "operating_cost_usd": float(model["expected_cost_usd"]),
        "benchmark_operating_cost_usd": float(base["expected_cost_usd"]),
        "cost_delta_usd": float(model["expected_cost_usd"] - base["expected_cost_usd"]),
        "weighted_cvar95_usd": float(model["cvar95_usd"]),
        "benchmark_weighted_cvar95_usd": float(base["cvar95_usd"]),
        "cvar95_delta_usd": float(model["cvar95_usd"] - base["cvar95_usd"]),
        "normalized_sla_shortfall": float(model["normalized_sla_shortfall"]),
        "benchmark_normalized_sla_shortfall": float(base["normalized_sla_shortfall"]),
        "sla_delta": float(
            model["normalized_sla_shortfall"] - base["normalized_sla_shortfall"]
        ),
        "corrective_intervention_rate": float(model["shield_intervention_rate"]),
        "benchmark_corrective_intervention_rate": float(base["shield_intervention_rate"]),
        "corrective_intervention_mean": float(model["shield_intervention_mean"]),
        "benchmark_corrective_intervention_mean": float(base["shield_intervention_mean"]),
        "post_shield_feasibility": float(model["feasibility_fraction"]),
        "benchmark_post_shield_feasibility": float(base["feasibility_fraction"]),
        "terminal_soc_gap_mean": float(model["terminal_soc_gap_mean"]),
        "terminal_backlog_mean": float(model["terminal_backlog_mean"]),
        "all_values_finite": bool(model["all_values_finite"] and base["all_values_finite"]),
        "locked_test_2025_opened": False,
    }
    numeric = [
        value
        for key, value in row.items()
        if key not in {"schema_version", "date", "evaluation_paths_sha256"}
        and isinstance(value, (int, float, np.integer, np.floating))
    ]
    row["all_values_finite"] = bool(
        row["all_values_finite"] and np.isfinite(np.asarray(numeric, dtype=float)).all()
    )
    return row


@dataclass(frozen=True)
class OneSidedBootstrap:
    estimate: float
    upper: float
    confidence: float
    replications: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "upper": self.upper,
            "confidence": self.confidence,
            "replications": self.replications,
            "seed": self.seed,
        }


def one_sided_paired_bootstrap(
    differences: Any,
    *,
    confidence: float = 0.95,
    replications: int = 10_000,
    seed: int = 20240813,
) -> OneSidedBootstrap:
    """Bootstrap the paired daily mean; positive values favor the benchmark."""

    delta = _finite_array(differences, name="paired differences")
    if delta.ndim != 1 or delta.size < 1:
        raise V7EvaluationError("paired bootstrap requires one or more daily differences")
    if not 0.5 < float(confidence) < 1.0 or int(replications) < 1_000:
        raise V7EvaluationError("invalid confidence or bootstrap replication count")
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(replications), dtype=np.float64)
    chunk = 1_000
    for start in range(0, int(replications), chunk):
        stop = min(start + chunk, int(replications))
        indices = generator.integers(0, delta.size, size=(stop - start, delta.size))
        means[start:stop] = delta[indices].mean(axis=1)
    upper = float(np.quantile(means, confidence, method="linear"))
    return OneSidedBootstrap(
        estimate=float(delta.mean()),
        upper=upper,
        confidence=float(confidence),
        replications=int(replications),
        seed=int(seed),
    )


def evaluate_policy_capability(
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    expected_days: int = DEFAULT_CAPABILITY_DAYS,
    validation_start: str | date | None = None,
    validation_end: str | date | None = None,
    sla_margin: float = 0.001,
    require_cvar_noninferiority: bool = True,
    replications: int = 10_000,
    seed: int = 20240813,
) -> dict[str, Any]:
    """Apply the unchanged cost/SLA/feasibility gate plus formal CVaR evidence."""

    rows = tuple(dict(row) for row in daily_rows)
    reasons: list[str] = []
    if int(expected_days) < 1:
        raise V7EvaluationError("expected_days must be positive")
    if len(rows) != int(expected_days):
        reasons.append(f"expected {expected_days} paired validation days; got {len(rows)}")
    dates = [_locked_date(row.get("date", "")) for row in rows]
    if len(dates) != len(set(dates)):
        reasons.append("paired capability rows contain duplicate dates")
    if (validation_start is None) != (validation_end is None):
        raise V7EvaluationError(
            "validation_start and validation_end must be provided together"
        )
    requested_start: str | None = None
    requested_end: str | None = None
    if validation_start is not None and validation_end is not None:
        requested_start = _locked_date(validation_start)
        requested_end = _locked_date(validation_end)
        if requested_start > requested_end:
            raise V7EvaluationError(
                "validation_start must not exceed validation_end"
            )
        if requested_start <= "2025-12-31" and requested_end >= "2025-01-01":
            raise Locked2025Error(
                "locked test year cannot enter the development validation window"
            )
        outside = [
            operating_date
            for operating_date in dates
            if not requested_start <= operating_date <= requested_end
        ]
        if outside:
            raise V7EvaluationError(
                "policy capability date falls outside the explicit development "
                f"validation window: {outside[0]}"
            )
    validation_window = {
        "requested_start": requested_start,
        "requested_end": requested_end,
        "expected_days": int(expected_days),
        "actual_start": min(dates) if dates else None,
        "actual_end": max(dates) if dates else None,
        "actual_days": len(rows),
    }
    required = (
        "cost_delta_usd",
        "cvar95_delta_usd",
        "sla_delta",
        "corrective_intervention_rate",
        "corrective_intervention_mean",
        "post_shield_feasibility",
    )
    vectors: dict[str, np.ndarray] = {}
    for name in required:
        try:
            vectors[name] = _finite_array(
                [row[name] for row in rows], name=name, shape=(len(rows),)
            )
        except (KeyError, V7EvaluationError) as exc:
            reasons.append(f"invalid daily field {name}: {exc}")
            vectors[name] = np.full(len(rows), math.inf)
    if any(int(row.get("completed_origins", -1)) != HORIZON for row in rows):
        reasons.append("one or more days did not execute all 96 origins")
    if any(int(row.get("formal_post_shield_sample_count", -1)) != FORMAL_PATHS for row in rows):
        reasons.append(f"one or more days lack N={FORMAL_PATHS} formal replay")
    if any(row.get("common_frozen_evaluation_paths") is not True for row in rows):
        reasons.append("model and benchmark do not share frozen evaluation paths")
    if any(row.get("post_shield") is not True for row in rows):
        reasons.append("one or more daily metrics are not post-shield")
    if any(row.get("locked_test_2025_opened") is not False for row in rows):
        reasons.append("locked-test status is not closed")
    if any(row.get("all_values_finite") is not True for row in rows):
        reasons.append("one or more daily rows contain non-finite evidence")

    cost = one_sided_paired_bootstrap(
        vectors["cost_delta_usd"], replications=replications, seed=seed
    ) if rows else None
    cvar = one_sided_paired_bootstrap(
        vectors["cvar95_delta_usd"], replications=replications, seed=seed + 1
    ) if rows else None
    if cost is None or cost.upper > 0.0:
        reasons.append("paired daily cost one-sided 95% bootstrap upper bound exceeds zero")
    if require_cvar_noninferiority and (cvar is None or cvar.upper > 0.0):
        reasons.append("paired daily CVaR one-sided 95% bootstrap upper bound exceeds zero")
    sla_increase = float(vectors["sla_delta"].mean()) if rows else math.inf
    if sla_increase > float(sla_margin):
        reasons.append(f"normalized SLA increase exceeds {sla_margin:g}")
    feasibility = float(vectors["post_shield_feasibility"].mean()) if rows else 0.0
    if not np.all(vectors["post_shield_feasibility"] >= 1.0 - 1.0e-12):
        reasons.append("post-shield feasibility is not 100%")
    metrics = {
        "validation_days": len(rows),
        "validation_origins": int(sum(int(row.get("completed_origins", 0)) for row in rows)),
        "formal_post_shield_sample_count": FORMAL_PATHS,
        "cost_delta_bootstrap": None if cost is None else cost.as_dict(),
        "cvar95_delta_bootstrap": None if cvar is None else cvar.as_dict(),
        "aggregate_normalized_sla_increase": sla_increase,
        "corrective_intervention_rate": float(
            vectors["corrective_intervention_rate"].mean()
        ) if rows else math.inf,
        "corrective_intervention_mean": float(
            vectors["corrective_intervention_mean"].mean()
        ) if rows else math.inf,
        "post_shield_feasibility": feasibility,
    }
    return {
        "schema_version": CAPABILITY_GATE_SCHEMA,
        "status": "PASS" if not reasons else "FAIL",
        "passed": not reasons,
        "dates": tuple(dates),
        "validation_window": validation_window,
        "metrics": metrics,
        "reasons": tuple(reasons),
        "locked_test_2025_opened": False,
    }


def batched_scalar_parity(
    runtime: V7Runtime,
    evaluation_paths: FrozenEvaluationPaths,
    policy_factory: Callable[[], Any],
    issue_batch: Mapping[str, Any],
    *,
    atol: float = 1.0e-9,
) -> dict[str, Any]:
    """Compare one batched replay with N isolated scalar replays."""

    batched = batched_post_shield_replay(
        runtime,
        policy_factory(),
        evaluation_paths,
        issue_batch,
        require_formal=False,
    )
    fields = (
        "realized_cost",
        "terminal_soc_gap",
        "terminal_backlog",
        "sla_violation",
        "shield_intervention",
        "shield_intervention_rate",
    )
    scalar: dict[str, list[float]] = {name: [] for name in fields}
    scalar_feasible: list[bool] = []
    for index in range(evaluation_paths.sample_count):
        result = batched_post_shield_replay(
            runtime,
            policy_factory(),
            evaluation_paths.subset(index),
            issue_batch,
            require_formal=False,
        )
        for name in fields:
            scalar[name].append(float(np.asarray(result[name])[0]))
        scalar_feasible.append(bool(np.asarray(result["feasible"])[0]))
    max_abs = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(batched[name], dtype=np.float64)
                    - np.asarray(scalar[name], dtype=np.float64)
                ),
                initial=0.0,
            )
        )
        for name in fields
    }
    feasible_equal = bool(
        np.array_equal(np.asarray(batched["feasible"], dtype=bool), scalar_feasible)
    )
    passed = feasible_equal and max(max_abs.values(), default=0.0) <= float(atol)
    return {
        "passed": passed,
        "atol": float(atol),
        "max_abs_by_field": max_abs,
        "feasible_equal": feasible_equal,
        "sample_count": evaluation_paths.sample_count,
    }


__all__ = [
    "CAPABILITY_GATE_SCHEMA",
    "DAILY_ROW_SCHEMA",
    "DEFAULT_CAPABILITY_DAYS",
    "EVALUATION_PATH_SCHEMA",
    "FORMAL_ALPHA",
    "FORMAL_PATHS",
    "FrozenEvaluationPaths",
    "OneSidedBootstrap",
    "V7EvaluationError",
    "batched_post_shield_replay",
    "batched_scalar_parity",
    "capability_daily_row",
    "evaluate_policy_capability",
    "exact_weighted_upper_cvar",
    "freeze_evaluation_paths",
    "one_sided_paired_bootstrap",
    "paired_post_shield_replay",
    "paths_from_distribution_samples",
    "summarize_replay",
]
