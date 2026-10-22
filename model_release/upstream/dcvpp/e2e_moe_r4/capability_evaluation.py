"""Concrete, fail-closed phase-end capability evaluation for formal R4.

These evaluators consume one already-materialized row per 2023-H2 date.  They
never infer capability from an optimization loss, and they reject incomplete,
duplicated, non-finite, or out-of-window evidence.  Expensive model/runtime
execution is supplied by the phase runner; this module owns the registered
statistics and acceptance rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import numpy as np

from .statistics import paired_daily_inference


VALIDATION_START = date(2023, 7, 1)
VALIDATION_END = date(2023, 12, 31)
EXPECTED_VALIDATION_DAYS = 179


class CapabilityEvidenceError(ValueError):
    """Phase-end evidence is absent, malformed, or outside 2023-H2."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class GateDecision:
    phase: str
    status: str
    passed: bool
    dates: tuple[str, ...]
    metrics: Mapping[str, float]
    reasons: tuple[str, ...]
    horizon: int | None = None
    hard_shield: bool = False

    def as_evaluator_payload(self) -> dict[str, Any]:
        return asdict(self)


def validation_dates() -> tuple[str, ...]:
    days = (VALIDATION_END - VALIDATION_START).days + 1
    dates = tuple(
        (VALIDATION_START + timedelta(days=index)).isoformat()
        for index in range(days)
    )
    # The formal eligible set excludes five missing/invalid source dates.  A
    # phase runner passes the authoritative 179-day list; this helper exposes
    # only the calendar envelope and must not be mistaken for eligibility.
    return dates


def _rows(records: Sequence[Mapping[str, Any]], *, phase: str) -> tuple[dict[str, Any], ...]:
    rows = tuple(dict(item) for item in records)
    if len(rows) != EXPECTED_VALIDATION_DAYS:
        raise CapabilityEvidenceError(
            f"{phase} requires exactly 179 paired validation-date rows; got {len(rows)}"
        )
    parsed: list[date] = []
    for row in rows:
        if "date" not in row:
            raise CapabilityEvidenceError(f"{phase} row is missing date")
        try:
            day = date.fromisoformat(str(row["date"])[:10])
        except ValueError as exc:
            raise CapabilityEvidenceError(f"invalid {phase} date {row['date']!r}") from exc
        if not VALIDATION_START <= day <= VALIDATION_END:
            raise CapabilityEvidenceError(f"{phase} date is outside 2023-H2: {day}")
        parsed.append(day)
    if len(set(parsed)) != len(parsed):
        raise CapabilityEvidenceError(f"{phase} dates must be unique")
    evidence_hashes: list[str] = []
    for row in rows:
        digest = str(row.get("evidence_sha256", "")).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise CapabilityEvidenceError(
                f"{phase} daily row {row.get('date')} requires evidence_sha256"
            )
        evidence_hashes.append(digest)
    if len(set(evidence_hashes)) != len(evidence_hashes):
        raise CapabilityEvidenceError(f"{phase} daily evidence hashes must be unique")
    order = np.argsort(np.asarray(parsed, dtype="datetime64[D]"), kind="stable")
    return tuple(rows[int(index)] for index in order)


def _vector(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    try:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise CapabilityEvidenceError(f"missing or invalid daily metric {key!r}") from exc
    if values.shape != (EXPECTED_VALIDATION_DAYS,) or not np.isfinite(values).all():
        raise CapabilityEvidenceError(f"daily metric {key!r} must be 179 finite scalars")
    return values


def _flag(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    try:
        values = np.asarray([row[key] for row in rows])
    except KeyError as exc:
        raise CapabilityEvidenceError(f"missing daily audit flag {key!r}") from exc
    if values.shape != (EXPECTED_VALIDATION_DAYS,) or values.dtype.kind not in "b":
        # Refuse truthy strings or numbers; formal flags must be actual bools.
        raise CapabilityEvidenceError(f"daily audit flag {key!r} must contain booleans")
    return values.astype(bool)


def _ci(values: np.ndarray, seed: int) -> dict[str, float]:
    result = paired_daily_inference(
        values, block_days=7, replicates=10_000, confidence=0.95, seed=seed
    )
    return {key: float(value) for key, value in result.as_dict().items() if isinstance(value, (int, float))}


def evaluate_representation(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    seed: int = 20240813,
    noninferiority_margin_fraction: float = 0.0,
    identity_tolerance: float = 1e-6,
) -> GateDecision:
    """Evaluate raw-unit representation predictions against one frozen baseline.

    Required row keys for each head are ``model_<head>_mae``,
    ``baseline_<head>_mae``, ``model_<head>_rmse``, and corresponding NMAE/NRMSE.
    The stronger baseline must already have been frozen from the train split and
    its name is recorded in every row as ``baseline_name``.
    """

    rows = _rows(records, phase="representation")
    reasons: list[str] = []
    names = {str(row.get("baseline_name", "")) for row in rows}
    if names != {str(baseline)} or baseline not in {"causal_persistence", "seasonal_naive"}:
        reasons.append("one registered frozen representation baseline was not used for every date")
    baseline_hashes = {str(row.get("baseline_sha256", "")).lower() for row in rows}
    if len(baseline_hashes) != 1 or not _is_sha256(next(iter(baseline_hashes), "")):
        reasons.append("representation baseline artifact hash is absent or not frozen")
    for name in ("causal_persistence", "seasonal_naive"):
        hashes = {str(row.get(f"{name}_sha256", "")).lower() for row in rows}
        if len(hashes) != 1 or not _is_sha256(next(iter(hashes), "")):
            reasons.append(f"{name} baseline was not independently frozen and hashed")
    heads = ("dart", "da", "rt", "workload", "pv", "pue")
    metrics: dict[str, float] = {}
    for head in heads:
        for metric in ("mae", "rmse", "nmae", "nrmse"):
            model = _vector(rows, f"model_{head}_{metric}")
            benchmark = _vector(rows, f"baseline_{head}_{metric}")
            metrics[f"model_{head}_{metric}"] = float(model.mean())
            metrics[f"baseline_{head}_{metric}"] = float(benchmark.mean())
    dart_delta = _vector(rows, "model_dart_nmae") - _vector(rows, "baseline_dart_nmae")
    if noninferiority_margin_fraction != 0.0:
        raise CapabilityEvidenceError(
            "formal representation uses the paper's zero-margin paired NMAE rule"
        )
    dart_ci = _ci(dart_delta, seed)
    metrics.update({f"dart_nmae_delta_{key}": value for key, value in dart_ci.items()})
    if dart_ci["one_sided_upper"] >= 0.0:
        reasons.append(
            "DART paired NMAE model-minus-baseline one-sided 95% upper bound is not below zero"
        )
    for head in ("da", "rt", "workload", "pv", "pue"):
        model = _vector(rows, f"model_{head}_nmae")
        benchmark = _vector(rows, f"baseline_{head}_nmae")
        delta = model - benchmark
        inference = _ci(delta, seed + heads.index(head))
        metrics[f"{head}_nmae_delta_one_sided_upper"] = inference["one_sided_upper"]
        if inference["one_sided_upper"] >= 0.0:
            reasons.append(
                f"{head} paired NMAE model-minus-baseline one-sided 95% upper bound is not below zero"
            )
    identity = _vector(rows, "da_rt_dart_identity_max_abs")
    metrics["da_rt_dart_identity_max_abs"] = float(identity.max())
    if identity.max() > identity_tolerance:
        reasons.append("DA-RT-DART identity exceeds tolerance")
    leakage = _flag(rows, "causal_no_future_leakage_passed")
    leakage_residual = _vector(rows, "future_leakage_forecast_invariance_max_abs")
    metrics["future_leakage_audited_days"] = float(leakage.sum())
    metrics["future_leakage_forecast_invariance_max_abs"] = float(
        leakage_residual.max()
    )
    if not leakage.all() or leakage_residual.max() > identity_tolerance:
        reasons.append("future-realization perturbation changed an issue-time forecast")
    dates = tuple(str(row["date"])[:10] for row in rows)
    return GateDecision(
        "representation", "PASS" if not reasons else "FAIL", not reasons,
        dates, metrics, tuple(reasons), None, False,
    )


def evaluate_scenario(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20240813,
    noninferiority_margin_fraction: float = 0.01,
) -> GateDecision:
    """Evaluate learned 20-particle trajectories vs frozen residual sampling."""

    rows = _rows(records, phase="scenario")
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    if {str(row.get("baseline_name", "")) for row in rows} != {"frozen_training_residual_sampling"}:
        reasons.append("frozen training residual-sampling baseline was not used for every date")
    baseline_hashes = {str(row.get("baseline_sha256", "")).lower() for row in rows}
    if len(baseline_hashes) != 1 or not _is_sha256(next(iter(baseline_hashes), "")):
        reasons.append("scenario baseline artifact hash is absent or not frozen")
    for name in (
        "energy_score", "marginal_crps", "variogram_score", "tail_score",
        "coverage_90", "coverage_95", "interval_width_90", "interval_width_95",
        "effective_sample_size", "average_pairwise_particle_distance", "weight_entropy",
        "revealed_prefix_identity_max_abs", "da_rt_dart_identity_max_abs",
    ):
        model = _vector(rows, f"model_{name}")
        metrics[f"model_{name}"] = float(model.mean())
        if name in {"energy_score", "marginal_crps", "variogram_score", "tail_score"}:
            benchmark = _vector(rows, f"baseline_{name}")
            metrics[f"baseline_{name}"] = float(benchmark.mean())
    energy_delta = _vector(rows, "model_energy_score") - _vector(rows, "baseline_energy_score")
    energy_ci = _ci(energy_delta, seed)
    metrics.update({f"energy_score_delta_{key}": value for key, value in energy_ci.items()})
    if energy_ci["upper"] >= 0.0:
        reasons.append("Energy Score paired 95% upper confidence bound is not below zero")
    for offset, name in enumerate(("variogram_score", "tail_score"), start=1):
        model = _vector(rows, f"model_{name}")
        benchmark = _vector(rows, f"baseline_{name}")
        adjusted = model - benchmark - noninferiority_margin_fraction * benchmark
        upper = _ci(adjusted, seed + offset)["one_sided_upper"]
        metrics[f"{name}_noninferiority_one_sided_upper"] = upper
        if upper > 0.0:
            reasons.append(f"{name} is not non-inferior within the 1% margin")
    for level in (90, 95):
        observed = float(_vector(rows, f"model_coverage_{level}").mean())
        target = level / 100.0
        metrics[f"absolute_coverage_error_{level}"] = abs(observed - target)
        if abs(observed - target) > 0.03:
            reasons.append(f"{level}% coverage error exceeds 0.03")
    ess = _vector(rows, "model_effective_sample_size")
    pairwise = _vector(rows, "model_average_pairwise_particle_distance")
    metrics["minimum_effective_sample_size"] = float(ess.min())
    metrics["minimum_average_pairwise_particle_distance"] = float(pairwise.min())
    if ess.min() < 10.0:
        reasons.append("effective sample size falls below 10/20")
    if pairwise.min() <= 0.0:
        reasons.append("learned particles are degenerate")
    for flag, description in (
        ("finite_weighted_joint_exact20", "particles are not finite weighted joint exact-20"),
        ("revealed_prefix_identity_passed", "revealed-prefix identity failed"),
    ):
        if not _flag(rows, flag).all():
            reasons.append(description)
    leakage_status = np.asarray(
        [str(row.get("future_leakage_audit_status", "")) == "PASSED" for row in rows],
        dtype=bool,
    )
    metrics["future_leakage_audited_days"] = float(leakage_status.sum())
    audited_origins = {
        int(row.get("future_leakage_audit_origin", -1)) for row in rows
    }
    metrics["future_leakage_audited_unique_origins"] = float(
        len(audited_origins.intersection(range(96)))
    )
    if not leakage_status.all() or set(range(96)) - audited_origins:
        reasons.append(
            "future-realization invariance did not pass on all dates and all 96 origins"
        )
    for field, description in (
        ("future_action_invariance_max_abs", "future perturbation changed a current action"),
        (
            "future_revealed_prefix_particle_invariance_max_abs",
            "future perturbation changed a revealed-prefix particle",
        ),
    ):
        values = _vector(rows, field)
        metrics[field] = float(values.max())
        if values.max() > 1e-6:
            reasons.append(description)
    if _vector(rows, "model_revealed_prefix_identity_max_abs").max() > 1e-6:
        reasons.append("revealed-prefix identity residual exceeds tolerance")
    if _vector(rows, "model_da_rt_dart_identity_max_abs").max() > 1e-6:
        reasons.append("scenario DA-RT-DART identity exceeds tolerance")
    dates = tuple(str(row["date"])[:10] for row in rows)
    return GateDecision(
        "scenario", "PASS" if not reasons else "FAIL", not reasons,
        dates, metrics, tuple(reasons), None, False,
    )


def evaluate_distribution(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20240813,
    noninferiority_margin_fraction: float = 0.01,
    formal_probability_samples: int = 256,
    formal_empirical_risk_samples: int = 512,
    minimum_effective_sample_fraction: float = 0.95,
    identity_tolerance: float = 1.0e-6,
) -> GateDecision:
    """Gate the V6 continuous Distribution head on all 179 validation dates.

    Learned and frozen residual distributions must be materialized with the
    same N=256 contract.  N=512 is frozen here only as the later empirical-risk
    budget; post-shield replay belongs to the Joint Policy/A1/A2 gates and is
    intentionally not repeated for every Distribution candidate.
    """

    rows = _rows(records, phase="distribution")
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    if {
        str(row.get("baseline_name", "")) for row in rows
    } != {"frozen_training_residual_sampling_variable_n"}:
        reasons.append(
            "one frozen train-only variable-N residual baseline was not used for every date"
        )
    for key, description in (
        ("baseline_sha256", "Distribution baseline artifact hash is absent or not frozen"),
        (
            "formal_sampling_contract_sha256",
            "formal sampling-budget contract hash is absent or not frozen",
        ),
        (
            "distribution_training_contract_sha256",
            "Distribution training contract hash is absent or not frozen",
        ),
        (
            "distribution_support_contract_sha256",
            "Distribution support/score-scale contract hash is absent or not frozen",
        ),
        ("checkpoint_sha256", "one immutable candidate checkpoint was not used"),
    ):
        values = {str(row.get(key, "")).lower() for row in rows}
        if len(values) != 1 or not _is_sha256(next(iter(values), "")):
            reasons.append(description)

    probability_counts = _vector(rows, "formal_probability_sample_count")
    empirical_risk_counts = _vector(rows, "formal_empirical_risk_sample_count")
    metrics["formal_probability_sample_count"] = float(probability_counts[0])
    metrics["formal_empirical_risk_sample_count"] = float(empirical_risk_counts[0])
    if not np.all(probability_counts == int(formal_probability_samples)):
        reasons.append(
            f"formal probability gate did not use N={int(formal_probability_samples)} on every date"
        )
    if not np.all(empirical_risk_counts == int(formal_empirical_risk_samples)):
        reasons.append(
            f"formal empirical-risk budget is not frozen at N={int(formal_empirical_risk_samples)}"
        )
    if {
        str(row.get("empirical_risk_replay_status", "")) for row in rows
    } != {"NOT_RUN_DISTRIBUTION_STAGE"}:
        reasons.append(
            "Distribution candidates must not run or self-certify post-shield empirical risk"
        )

    model_metric_names = (
        "energy_score",
        "marginal_crps",
        "variogram_score",
        "tail_score",
        "coverage_90",
        "coverage_95",
        "interval_width_90",
        "interval_width_95",
        "interval_score_90",
        "interval_score_95",
        "effective_sample_size",
        "average_pairwise_sample_distance",
        "minimum_feature_std",
        "weight_entropy",
        "revealed_prefix_identity_max_abs",
        "da_rt_dart_identity_max_abs",
        "support_excess_max_abs",
        "positive_support_violation_max_abs",
    )
    for name in model_metric_names:
        model = _vector(rows, f"model_{name}")
        metrics[f"model_{name}"] = float(model.mean())
        if name in {
            "energy_score",
            "marginal_crps",
            "variogram_score",
            "tail_score",
            "interval_score_90",
            "interval_score_95",
        }:
            baseline = _vector(rows, f"baseline_{name}")
            metrics[f"baseline_{name}"] = float(baseline.mean())

    energy_delta = _vector(rows, "model_energy_score") - _vector(
        rows, "baseline_energy_score"
    )
    energy_ci = _ci(energy_delta, seed)
    metrics.update(
        {f"energy_score_delta_{key}": value for key, value in energy_ci.items()}
    )
    # Candidate selection and the phase gate use the same pre-registered
    # one-sided 95% upper bound.  Keep the two-sided interval in ``metrics`` as
    # descriptive evidence, but never substitute its upper endpoint for the
    # formal decision statistic.
    if energy_ci["one_sided_upper"] >= 0.0:
        reasons.append(
            "Energy Score paired 95% upper confidence bound is not below zero"
        )

    # Proper-score/dependence/tail/interval guards prevent an Energy-only
    # candidate from winning by destroying calibration elsewhere.
    secondary = (
        "marginal_crps",
        "variogram_score",
        "tail_score",
        "interval_score_90",
        "interval_score_95",
    )
    for offset, name in enumerate(secondary, start=1):
        model = _vector(rows, f"model_{name}")
        baseline = _vector(rows, f"baseline_{name}")
        adjusted = model - baseline - noninferiority_margin_fraction * np.abs(baseline)
        upper = _ci(adjusted, seed + offset)["one_sided_upper"]
        metrics[f"{name}_noninferiority_one_sided_upper"] = upper
        if upper > 0.0:
            reasons.append(f"{name} is not non-inferior within the 1% margin")

    for level in (90, 95):
        observed = float(_vector(rows, f"model_coverage_{level}").mean())
        error = abs(observed - level / 100.0)
        metrics[f"absolute_coverage_error_{level}"] = error
        if error > 0.03:
            reasons.append(f"{level}% coverage error exceeds 0.03")

    ess = _vector(rows, "model_effective_sample_size")
    pairwise = _vector(rows, "model_average_pairwise_sample_distance")
    feature_std = _vector(rows, "model_minimum_feature_std")
    metrics["minimum_effective_sample_size"] = float(ess.min())
    metrics["minimum_average_pairwise_sample_distance"] = float(pairwise.min())
    metrics["minimum_feature_std"] = float(feature_std.min())
    fraction = float(minimum_effective_sample_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise CapabilityEvidenceError(
            "minimum_effective_sample_fraction must be in (0,1]"
        )
    metrics["minimum_effective_sample_fraction"] = fraction
    minimum_ess = fraction * float(formal_probability_samples)
    if ess.min() < minimum_ess:
        reasons.append(
            f"effective sample size falls below {minimum_ess:g}/{formal_probability_samples}"
        )
    if pairwise.min() <= 1.0e-6 or feature_std.min() <= 1.0e-6:
        reasons.append("continuous Distribution samples are collapsed or non-diverse")

    for flag, description in (
        ("fixed_rng_reproducible", "fixed RNG did not reproduce identical samples"),
        (
            "finite_distribution_parameters",
            "Distribution parameters or world latent are non-finite",
        ),
        ("finite_joint_samples", "Distribution samples are non-finite"),
        ("support_bounds_passed", "Distribution support bounds were violated"),
        (
            "baseline_support_contract_same",
            "learned and residual baseline did not use the same canonical support",
        ),
        ("sample_diversity_passed", "Distribution sample-collapse audit failed"),
        ("revealed_prefix_identity_passed", "revealed-prefix identity failed"),
    ):
        if not _flag(rows, flag).all():
            reasons.append(description)

    leakage_status = np.asarray(
        [str(row.get("future_leakage_audit_status", "")) == "PASSED" for row in rows],
        dtype=bool,
    )
    audited_origins = {int(row.get("future_leakage_audit_origin", -1)) for row in rows}
    metrics["future_leakage_audited_days"] = float(leakage_status.sum())
    metrics["future_leakage_audited_unique_origins"] = float(
        len(audited_origins.intersection(range(96)))
    )
    if not leakage_status.all() or set(range(96)) - audited_origins:
        reasons.append(
            "future-realization invariance did not pass on all dates and all 96 origins"
        )
    for field, description in (
        ("future_action_invariance_max_abs", "future perturbation changed current action"),
        (
            "future_distribution_latent_invariance_max_abs",
            "future perturbation changed Distributional World Latent",
        ),
        (
            "future_revealed_prefix_sample_invariance_max_abs",
            "future perturbation changed a revealed-prefix sample",
        ),
        (
            "future_complete_distribution_sample_invariance_max_abs",
            "future perturbation changed the conditional sample under fixed RNG",
        ),
    ):
        values = _vector(rows, field)
        metrics[field] = float(values.max())
        if values.max() > identity_tolerance:
            reasons.append(description)

    for field, description in (
        ("model_revealed_prefix_identity_max_abs", "revealed-prefix identity exceeds tolerance"),
        ("model_da_rt_dart_identity_max_abs", "DA-RT-DART identity exceeds tolerance"),
        ("model_support_excess_max_abs", "declared absolute support was exceeded"),
        (
            "model_residual_cap_excess_max_abs",
            "train-frozen model-centred residual cap was exceeded",
        ),
        (
            "model_positive_support_violation_max_abs",
            "workload/PV/PUE canonical support was violated",
        ),
        (
            "baseline_support_excess_max_abs",
            "frozen residual baseline exceeded the shared absolute support",
        ),
        (
            "baseline_da_rt_dart_identity_max_abs",
            "frozen residual baseline violated DA-RT-DART identity",
        ),
    ):
        if _vector(rows, field).max() > identity_tolerance:
            reasons.append(description)

    for field in (
        "formal_selection_2024_opened",
        "locked_test_2025_opened",
        "active_pyomo_loaded",
        "active_gurobipy_loaded",
    ):
        if _flag(rows, field).any():
            reasons.append(f"forbidden formal data/solver flag is true: {field}")

    dates = tuple(str(row["date"])[:10] for row in rows)
    return GateDecision(
        "distribution",
        "PASS" if not reasons else "FAIL",
        not reasons,
        dates,
        metrics,
        tuple(reasons),
        None,
        False,
    )


def evaluate_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    phase: str = "policy",
    seed: int = 20240813,
) -> GateDecision:
    """Hard-gate a K96 policy/A1/A2 candidate on all 17,184 origins.

    One row is one day.  Per-origin hard checks must already have been reduced
    into exact counts by :class:`R4Runtime` plus ``validate_record``.
    """

    phase_name = str(phase).lower()
    if phase_name not in {"policy", "a1", "a2"}:
        raise CapabilityEvidenceError(f"invalid decision capability phase {phase!r}")
    rows = _rows(records, phase=phase_name)
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    checkpoint_hashes = {str(row.get("checkpoint_sha256", "")).lower() for row in rows}
    benchmark_hashes = {str(row.get("benchmark_sha256", "")).lower() for row in rows}
    if len(checkpoint_hashes) != 1 or not _is_sha256(next(iter(checkpoint_hashes), "")):
        reasons.append("one frozen candidate checkpoint hash was not used for all dates")
    if len(benchmark_hashes) != 1 or not _is_sha256(next(iter(benchmark_hashes), "")):
        reasons.append("one frozen Price-aware EDF benchmark hash was not used for all dates")
    completed = _vector(rows, "completed_origins")
    validated = _vector(rows, "independently_validated_origins")
    rejected = _vector(rows, "non_exogenous_rejected_origins")
    feasible = _vector(rows, "post_shield_feasible_origins")
    total_origins = int(completed.sum())
    metrics.update(
        {
            "validation_days": float(len(rows)),
            "validation_origins": float(total_origins),
            "independently_validated_origins": float(validated.sum()),
            "post_shield_feasibility": float(feasible.sum() / max(total_origins, 1)),
            "non_exogenous_rejected_origins": float(rejected.sum()),
        }
    )
    if total_origins != 17_184 or not np.all(completed == 96):
        reasons.append("policy hard gate did not execute all 17,184 K96 origins")
    if validated.sum() != 17_184 or feasible.sum() != 17_184:
        reasons.append("independent post-shield hard feasibility is not 100%")
    if rejected.sum() != 0:
        reasons.append("non-exogenous origins were rejected")
    learned_sla = _vector(rows, "normalized_sla_shortfall")
    benchmark_sla = _vector(rows, "benchmark_normalized_sla_shortfall")
    sla_increase = float(np.mean(learned_sla - benchmark_sla))
    metrics["aggregate_normalized_sla_increase"] = sla_increase
    if sla_increase > 0.001:
        reasons.append("normalized SLA shortfall is more than 0.001 worse than Price-aware EDF")
    learned_cost = _vector(rows, "operating_cost_usd")
    benchmark_cost = _vector(rows, "benchmark_operating_cost_usd")
    inference = _ci(learned_cost - benchmark_cost, seed)
    metrics.update({f"cost_delta_{key}": value for key, value in inference.items()})
    if inference["upper"] > 0.0:
        reasons.append("paired daily operating-cost 95% upper confidence bound exceeds zero")
    for flag, description in (
        ("all_values_finite", "a cost/CVaR/latency/intervention/checkpoint value is non-finite"),
        ("day_ahead_frozen", "day-ahead schedule changed after origin zero"),
        ("checkpoint_lineage_valid", "checkpoint lineage is invalid"),
        ("checkpoint_resume_exact", "checkpoint does not resume exactly"),
    ):
        if not _flag(rows, flag).all():
            reasons.append(description)
    if phase_name == "a1":
        for flag, description in (
            ("all_allowed_gradients_finite_nonzero", "A1 allowed gradients are missing, zero, or non-finite"),
            ("frozen_parameters_unchanged", "A1 changed a frozen parameter"),
            ("dart_capability_gate_retained", "A1 did not retain the DART capability gate"),
        ):
            if not _flag(rows, flag).all():
                reasons.append(description)
    if phase_name == "a2" and not _flag(rows, "physical_twin_core_frozen").all():
        reasons.append("A2 changed or unfroze the physical-twin core")
    for name in (
        "weighted_cvar95_usd", "inference_latency_ms", "corrective_intervention_rate",
        "corrective_intervention_mean", "corrective_intervention_p95",
    ):
        metrics[name] = float(_vector(rows, name).mean())
    dates = tuple(str(row["date"])[:10] for row in rows)
    return GateDecision(
        phase_name, "PASS" if not reasons else "FAIL", not reasons,
        dates, metrics, tuple(reasons), 96, True,
    )


class CapabilityEvaluator:
    """Adapter usable directly by ``trainer.run_capability_gate``."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        baseline: str | None = None,
        seed: int = 20240813,
    ) -> None:
        self.records = tuple(dict(row) for row in records)
        self.baseline = baseline
        self.seed = int(seed)

    def __call__(self, *, phase: str, horizon: int | None) -> Mapping[str, Any]:
        if phase == "representation":
            if self.baseline is None:
                raise CapabilityEvidenceError("representation evaluator requires a frozen baseline name")
            result = evaluate_representation(self.records, baseline=self.baseline, seed=self.seed)
        elif phase == "scenario":
            result = evaluate_scenario(self.records, seed=self.seed)
        elif phase == "distribution":
            result = evaluate_distribution(self.records, seed=self.seed)
        else:
            result = evaluate_policy(self.records, phase=phase, seed=self.seed)
        if result.horizon != horizon:
            if phase in {"policy", "a1", "a2"}:
                raise CapabilityEvidenceError(
                    f"requested K={horizon}, measured decision evidence is K={result.horizon}"
                )
        return result.as_evaluator_payload()


def build_capability_evaluator(
    phase: str,
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str | None = None,
    seed: int = 20240813,
) -> CapabilityEvaluator:
    """Phase-runner factory; validates evidence immediately, before gate use."""

    evaluator = CapabilityEvaluator(daily_rows, baseline=baseline, seed=seed)
    horizon = 96 if str(phase).lower() in {"policy", "a1", "a2"} else None
    # Eager validation prevents a malformed adapter from being registered and
    # only failing after training has already been labelled complete.
    evaluator(phase=str(phase).lower(), horizon=horizon)
    return evaluator


__all__ = [
    "CapabilityEvaluator",
    "CapabilityEvidenceError",
    "EXPECTED_VALIDATION_DAYS",
    "GateDecision",
    "build_capability_evaluator",
    "evaluate_distribution",
    "evaluate_policy",
    "evaluate_representation",
    "evaluate_scenario",
]
