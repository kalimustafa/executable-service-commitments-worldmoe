from __future__ import annotations

"""Full-date causal validation for the solver-free formal R3 program.

The benchmark is calibrated independently of the learned model.  Every method
uses the same :class:`NeuralConstrainedRuntime`, constraint shield, independent
validator, realized settlement, and 20-particle issue-time contract.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constrained_runtime import (
    INTERVALS,
    SCENARIOS,
    FrozenPhysicalArtifact,
    NeuralConstrainedRuntime,
    PassiveEDFShieldPolicy,
    PriceAwareEDFShieldPolicy,
    R3DataError,
    SelectionDay,
    validate_record,
)

VALIDATION_START = "2023-07-01"
VALIDATION_END = "2023-12-31"


class FormalValidationError(ValueError):
    """The formal validation panel or metric contract is invalid."""


def _date_of(value: Any) -> str:
    date = str(value)[:10]
    if not VALIDATION_START <= date <= VALIDATION_END:
        raise FormalValidationError(
            f"formal validator accepts only {VALIDATION_START}..{VALIDATION_END}; "
            f"rejected {date!r}"
        )
    return date


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _physical(value: Any) -> FrozenPhysicalArtifact:
    if isinstance(value, FrozenPhysicalArtifact):
        return value
    if isinstance(value, Mapping):
        return FrozenPhysicalArtifact.from_mapping(value)
    required = (
        "calibration_id",
        "twin_ids",
        "transfer_regime",
        "idle_mw",
        "work_mw_per_unit",
        "capacity_work_units",
        "grid_import_limit_mw",
        "battery_power_mw",
        "battery_energy_mwh",
        "physical_multiplier_pool",
    )
    if not all(hasattr(value, name) for name in required):
        raise FormalValidationError("physical artifact has an unsupported type")
    return FrozenPhysicalArtifact(
        calibration_id=str(value.calibration_id),
        twin_ids=tuple(map(str, value.twin_ids)),
        transfer_regime=str(value.transfer_regime),
        idle_mw=value.idle_mw,
        work_mw_per_unit=value.work_mw_per_unit,
        capacity_work_units=value.capacity_work_units,
        grid_import_limit_mw=value.grid_import_limit_mw,
        battery_power_mw=value.battery_power_mw,
        battery_energy_mwh=value.battery_energy_mwh,
        physical_multiplier_pool=value.physical_multiplier_pool,
        zones=tuple(map(str, getattr(value, "zones", ("WEST", "CAPITL", "N.Y.C.", "LONGIL")))),
        source_hashes=dict(getattr(value, "source_hashes", {})),
        frozen=bool(getattr(value, "frozen", True)),
    )


@dataclass(frozen=True)
class _DaySample:
    day: SelectionDay
    inputs: Mapping[str, torch.Tensor]


def _day_from_sample(sample: Mapping[str, Any]) -> _DaySample:
    date = _date_of(sample.get("operating_date"))
    decision = sample.get("decision")
    if not isinstance(decision, Mapping):
        raise FormalValidationError(f"{date}: dataset sample has no decision mapping")
    market = sample.get("market")
    workload = sample.get("workload")
    physical_input = sample.get("physical")
    state = sample.get("state")
    if not all(isinstance(item, torch.Tensor) for item in (market, workload, physical_input, state)):
        raise FormalValidationError(f"{date}: model input tensors are missing")
    da = _to_numpy(decision["da_price"]).astype(float, copy=True)
    rt = _to_numpy(decision["rt_price"]).astype(float, copy=True)
    realized_workload = _to_numpy(decision["workload"]).astype(float, copy=True)
    realized_pv = _to_numpy(decision["pv"]).astype(float, copy=True)
    multiplier = _to_numpy(decision["physical_multiplier"]).astype(float, copy=True)
    initial_soc = _to_numpy(decision["initial_soc"]).astype(float, copy=True)
    scenario = _to_numpy(sample["scenario_target"]).astype(float, copy=True)
    if scenario.ndim != 3 or scenario.shape[1:] != (INTERVALS, 6):
        raise FormalValidationError(f"{date}: scenario_target must be [S,96,6]")
    if scenario.shape[0] < 100:
        raise FormalValidationError(f"{date}: scenario_target requires 100 issue-time rows")
    rt_scalar = scenario[:100, :, 1]
    pv_scalar = np.maximum(scenario[:100, :, 4], 0.0)
    rt_pool = np.repeat(rt_scalar[:, None, :], 4, axis=1)
    pv_pool = np.repeat(pv_scalar[:, None, :], 4, axis=1)
    workload_pool = np.maximum(scenario[:100, :, 3], 0.0)
    day = SelectionDay(
        operating_date=date,
        timestamps_utc=np.asarray(
            [f"{date}T{origin // 4:02d}:{15 * (origin % 4):02d}:00Z" for origin in range(INTERVALS)]
        ),
        da_price=da,
        rt_price_pool=rt_pool,
        workload_pool=workload_pool,
        pv_pool=pv_pool,
        realized_workload=realized_workload,
        realized_pv=realized_pv,
        physical_multiplier=multiplier,
        initial_soc_mwh=initial_soc,
        source_hashes={"dataset_bundle": str(sample.get("bundle_path", date))},
        locked_test=False,
        realized_da_price=da,
        realized_rt_price=rt,
    )
    day.validate()
    return _DaySample(
        day=day,
        inputs={
            "market": market.detach().clone(),
            "workload": workload.detach().clone(),
            "physical": physical_input.detach().clone(),
            "state": state.detach().clone(),
        },
    )


def _materialize_days(dataset: Any) -> list[_DaySample]:
    if dataset is None or not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise FormalValidationError("dataset must be an indexable complete-date panel")
    days = [_day_from_sample(dataset[index]) for index in range(len(dataset))]
    if not days:
        raise FormalValidationError("validation dataset is empty")
    dates = [item.day.operating_date for item in days]
    if len(set(dates)) != len(dates):
        raise FormalValidationError("validation dataset contains duplicate operating dates")
    if dates != sorted(dates):
        days.sort(key=lambda item: item.day.operating_date)
    return days


def _summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("summary")
    if not isinstance(value, Mapping):
        raise FormalValidationError("runtime result has no summary mapping")
    return value


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_summary(result) for result in results]
    records = [record for result in results for record in result.get("records", [])]
    completed_origins = sum(int(result.get("completed_origins", 0)) for result in results)
    expected_origins = len(results) * INTERVALS
    total_arrivals = sum(
        float(record.get("workload_arrival_units", 0.0))
        for record in records
        if not bool(record.get("rejected", False))
    )
    sla_shortfall = sum(float(item.get("sla_shortfall_work_units", 0.0)) for item in summaries)
    corrective = [
        float(record.get("corrective_intervention_magnitude", 0.0))
        for record in records
        if not bool(record.get("rejected", False))
    ]
    rejected_non_exogenous = sum(
        bool(record.get("rejected", False)) and not bool(record.get("exogenous_infeasible", False))
        for record in records
    )
    feasible = sum(
        bool(record.get("passed", False))
        for record in records
        if not bool(record.get("rejected", False))
    )
    raw_category_counts: dict[str, int] = {}
    for record in records:
        categories = record.get("raw_violation_categories", [])
        if isinstance(categories, Sequence) and not isinstance(categories, (str, bytes)):
            for category in categories:
                key = str(category)
                raw_category_counts[key] = raw_category_counts.get(key, 0) + 1
    return {
        "day_count": len(results),
        "completed_origin_count": completed_origins,
        "expected_origin_count": expected_origins,
        "feasibility_rate": feasible / max(completed_origins, 1),
        "post_shield_feasibility_fraction": feasible / max(completed_origins, 1),
        "rejected_origins": sum(bool(record.get("rejected", False)) for record in records),
        "rejected_non_exogenous_origins": int(rejected_non_exogenous),
        "exogenous_infeasible_origins": sum(bool(record.get("exogenous_infeasible", False)) for record in records),
        "total_cost_usd": float(sum(float(item.get("total_cost_usd", 0.0)) for item in summaries)),
        "operating_cost_usd": float(sum(float(item.get("total_cost_usd", 0.0)) for item in summaries)),
        "cvar_95_usd": float(sum(float(item.get("weighted_particle_cvar_95_usd", 0.0)) for item in summaries)),
        "sla_shortfall_work_units": sla_shortfall,
        "normalized_sla_shortfall": sla_shortfall / max(total_arrivals, 1e-12),
        "corrective_magnitude": float(sum(corrective)),
        "corrective_intervention_rate": (
            float(np.mean(np.asarray(corrective) > 0.0)) if corrective else 0.0
        ),
        "correction_rate": (
            float(np.mean(np.asarray(corrective) > 0.0)) if corrective else 0.0
        ),
        "corrective_magnitude_mean": float(np.mean(corrective)) if corrective else 0.0,
        "corrective_magnitude_p95": float(np.quantile(corrective, 0.95)) if corrective else 0.0,
        "policy_inference_time_seconds": float(
            sum(float(record.get("policy_inference_seconds", 0.0)) for record in records)
        ),
        "constraint_shield_time_seconds": float(
            sum(float(record.get("shield_seconds", 0.0)) for record in records)
        ),
        "raw_violation_category_counts": raw_category_counts,
        "all_days_complete": all(
            result.get("status") == "COMPLETE"
            and int(result.get("completed_origins", 0)) == INTERVALS
            for result in results
        ),
    }


def _independently_audit_day(
    result: Mapping[str, Any],
    *,
    physical: FrozenPhysicalArtifact,
    day: SelectionDay,
    tolerance: float,
) -> dict[str, Any]:
    """Recompute every emitted hard residual outside the shield itself."""

    accepted = [
        record
        for record in result.get("records", [])
        if not bool(record.get("rejected", False))
    ]
    audits: list[dict[str, Any]] = []
    for record in accepted:
        audit = validate_record(
            record,
            physical=physical,
            day=day,
            tolerance=tolerance,
        )
        audits.append(dict(audit))
        if isinstance(record, dict):
            record["independent_validation"] = dict(audit)
    residuals = [
        float(value)
        for audit in audits
        for key, value in audit.items()
        if key != "passed"
        and isinstance(value, (int, float, np.integer, np.floating))
    ]
    return {
        "accepted_origin_count": len(accepted),
        "audited_origin_count": len(audits),
        "passed_origin_count": sum(audit.get("passed") is True for audit in audits),
        "all_passed": bool(audits) and all(audit.get("passed") is True for audit in audits),
        "max_residual": max(residuals, default=math.inf),
    }


def _row_records(result: Mapping[str, Any], *, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in result.get("records", []):
        row = dict(value)
        row["method"] = method
        rows.append(row)
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _candidate_grid(calibration: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw = calibration.get("battery_candidate_grid")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [
            (
                float(item["low_price_charge_fraction"]),
                float(item["high_price_discharge_fraction"]),
            )
            for item in raw
            if isinstance(item, Mapping)
        ]
    else:
        values = [(charge, discharge) for charge in (0.0, 0.5, 1.0) for discharge in (0.0, 0.5, 1.0)]
    required = {(charge, discharge) for charge in (0.0, 0.5, 1.0) for discharge in (0.0, 0.5, 1.0)}
    if set(values) != required or len(values) != 9:
        raise FormalValidationError("battery candidate grid must be exactly {0,.5,1}^2")
    return sorted(values)


def _baseline_runtime(
    day: SelectionDay,
    physical: FrozenPhysicalArtifact,
    runtime_config: Mapping[str, Any],
    policy: Any,
    *,
    formal: bool,
    max_origins: int,
    logfile: Path | str | None,
) -> dict[str, Any]:
    return NeuralConstrainedRuntime(physical, runtime_config, logfile=logfile).run_day(
        day,
        policy_callback=policy,
        max_origins=max_origins,
        formal=formal,
    )


def calibrate_price_aware_battery_rules(
    dataset: Any,
    physical: Any,
    *,
    runtime_config: Mapping[str, Any] | None,
    calibration: Mapping[str, Any],
    formal: bool = True,
    max_origins: int = INTERVALS,
    logfile: Path | str | None = None,
    max_normalized_sla_shortfall_increase: float = 0.001,
) -> dict[str, Any]:
    """Select one preregistered battery pair on every validation date.

    This function never accepts a model and therefore cannot adapt the primary
    benchmark to a learned checkpoint.  Dataset date gating occurs before any
    runtime execution and excludes 2024/2025.
    """

    if int(max_origins) != INTERVALS:
        raise FormalValidationError("formal benchmark calibration requires 96 origins")
    runtime_cfg = dict(runtime_config or {})
    physical_artifact = _physical(physical)
    days = _materialize_days(dataset)
    passive_results = [
        _baseline_runtime(
            item.day,
            physical_artifact,
            runtime_cfg,
            PassiveEDFShieldPolicy(),
            formal=formal,
            max_origins=INTERVALS,
            logfile=logfile,
        )
        for item in days
    ]
    passive = _aggregate(passive_results)
    candidates: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for charge_fraction, discharge_fraction in _candidate_grid(calibration):
        metadata = dict(calibration)
        metadata.update(
            low_price_charge_fraction=charge_fraction,
            high_price_discharge_fraction=discharge_fraction,
        )
        policy = PriceAwareEDFShieldPolicy.from_calibration(metadata)
        results = [
            _baseline_runtime(
                item.day,
                physical_artifact,
                runtime_cfg,
                policy,
                formal=formal,
                max_origins=INTERVALS,
                logfile=logfile,
            )
            for item in days
        ]
        aggregate = _aggregate(results)
        sla_increase = float(aggregate["normalized_sla_shortfall"]) - float(
            passive["normalized_sla_shortfall"]
        )
        eligible = bool(
            aggregate["all_days_complete"]
            and aggregate["post_shield_feasibility_fraction"] == 1.0
            and aggregate["rejected_non_exogenous_origins"] == 0
            and sla_increase <= float(max_normalized_sla_shortfall_increase)
        )
        candidate = {
            "low_price_charge_fraction": charge_fraction,
            "high_price_discharge_fraction": discharge_fraction,
            **aggregate,
            "normalized_sla_shortfall_increase_vs_passive": sla_increase,
            "eligible": eligible,
        }
        candidates.append(candidate)
        method = f"price_aware_edf_shield_charge={charge_fraction:g}_discharge={discharge_fraction:g}"
        for result in results:
            candidate_records.extend(_row_records(result, method=method))
    eligible_candidates = [item for item in candidates if item["eligible"]]
    if not eligible_candidates:
        raise FormalValidationError("no preregistered Price-aware EDF battery candidate passed validation")
    selected = min(
        eligible_candidates,
        key=lambda item: (
            float(item["total_cost_usd"]),
            float(item["low_price_charge_fraction"]),
            float(item["high_price_discharge_fraction"]),
        ),
    )
    frozen = dict(calibration)
    frozen.update(
        low_price_charge_fraction=float(selected["low_price_charge_fraction"]),
        high_price_discharge_fraction=float(selected["high_price_discharge_fraction"]),
        battery_candidate_results=candidates,
        battery_fractions_frozen=True,
        battery_candidate_selection_split="validation",
        battery_candidate_selection_order=["feasibility", "sla_noninferiority", "lowest_cost"],
        formal_2024_payload_opened=False,
        locked_2025_opened=False,
    )
    passive_rows = [row for result in passive_results for row in _row_records(result, method="passive_edf_shield")]
    return {
        "status": "COMPLETE",
        "completed_day_count": len(days),
        "completed_origin_count": len(days) * INTERVALS * 10,
        "passive_aggregate": passive,
        "candidate_results": candidates,
        "selected_benchmark_parameters": {
            "low_price_charge_fraction": selected["low_price_charge_fraction"],
            "high_price_discharge_fraction": selected["high_price_discharge_fraction"],
        },
        "frozen_calibration": frozen,
        "records": passive_rows + candidate_records,
        "formal_2024_opened": False,
        "locked_2025_opened": False,
    }


def _learned_policy_adapter(
    model: torch.nn.Module,
    sample: _DaySample,
    physical: FrozenPhysicalArtifact,
    device: torch.device,
) -> Any:
    """Build one causal model callback; issue particles stay fixed per day."""

    inputs = {
        name: value.unsqueeze(0).to(device=device)
        for name, value in sample.inputs.items()
    }
    energy = np.maximum(np.asarray(physical.battery_energy_mwh, dtype=np.float32), 1e-6)
    grid_limit = np.asarray(physical.grid_import_limit_mw, dtype=np.float32)
    battery_limit = np.asarray(physical.battery_power_mw, dtype=np.float32)
    work_capacity = np.asarray(physical.capacity_work_units, dtype=np.float32)
    periods = torch.arange(INTERVALS, device=device).unsqueeze(0)
    state_history = np.zeros((INTERVALS, 4), dtype=np.float32)
    prepared_context = (
        model.prepare_origin_context(inputs)  # type: ignore[attr-defined]
        if hasattr(model, "prepare_origin_context")
        else None
    )

    def callback(context: Mapping[str, Any]) -> Mapping[str, Any]:
        origin = int(context["origin"])
        soc = np.clip(np.asarray(context["soc_mwh"], dtype=np.float32), 0.0, energy)
        queue = np.asarray(context["queue_age_buckets"], dtype=np.float32)
        previous = context.get("previous_action")
        if isinstance(previous, Mapping):
            previous_action = {
                "grid": torch.as_tensor(previous.get("grid", np.zeros(4)), dtype=torch.float32, device=device).unsqueeze(0),
                "battery": torch.as_tensor(previous.get("battery", np.zeros(4)), dtype=torch.float32, device=device).unsqueeze(0),
                "compute": torch.as_tensor(previous.get("compute", np.zeros(4)), dtype=torch.float32, device=device).unsqueeze(0),
                "migration": torch.zeros((1, 4, 4), dtype=torch.float32, device=device),
            }
        else:
            previous_action = {
                "grid": torch.zeros((1, 4), device=device),
                "battery": torch.zeros((1, 4), device=device),
                "compute": torch.zeros((1, 4), device=device),
                "migration": torch.zeros((1, 4, 4), device=device),
            }
        state_history[origin] = soc / energy
        call_inputs = {
            **inputs,
            "state": torch.from_numpy(state_history.copy()).unsqueeze(0).to(device),
        }
        kwargs: dict[str, Any] = {
            "origin": origin,
            "current_soc": torch.from_numpy(soc).unsqueeze(0).to(device),
            "backlog": torch.full((1, 4), float(queue.sum()) / 4.0, device=device),
            "deadline_pressure": torch.full(
                (1, 4), (float(queue[0]) if queue.size else 0.0) / 4.0, device=device
            ),
            "previous_action": previous_action,
            "revealed_mask": periods.le(origin),
        }
        if prepared_context is not None:
            kwargs["prepared_context"] = prepared_context
        with torch.inference_mode():
            output = model.forward_origin(call_inputs, **kwargs)  # type: ignore[attr-defined]
        actions = output.get("actions", output.get("policy"))
        if not isinstance(actions, Mapping):
            raise FormalValidationError("forward_origin omitted actions")
        raw: dict[str, Any] = {
            "grid": _to_numpy(actions["grid"])[0] * grid_limit,
            "battery": _to_numpy(actions["battery"])[0] * battery_limit,
            "compute": _to_numpy(actions["compute"])[0] * work_capacity,
            # Selection contract is known-zero.  Raw migration is not exposed
            # as an executable degree of freedom in formal validation.
            "migration": np.zeros((4, 4), dtype=float),
            "policy_mode": "e2e_moe_solver_free_r3",
        }
        feasibility = _to_numpy(output.get("feasibility", np.ones(1))).reshape(-1)
        fallback = _to_numpy(output.get("fallback_confidence", np.ones(1))).reshape(-1)
        raw["fallback"] = bool(feasibility[0] < 0.5 or fallback[0] < 0.5)
        if origin == 0:
            raw["day_ahead"] = _to_numpy(actions["day_ahead"])[0] * grid_limit[:, None]
        return raw

    return callback


def _particles_to_joint(
    particles: np.ndarray,
    day: SelectionDay,
) -> dict[str, np.ndarray]:
    """Lift the learned six-channel particles into the zonal runtime seam."""

    if particles.shape != (SCENARIOS, INTERVALS, 6):
        raise FormalValidationError("learned scenarios must have shape [20,96,6]")
    rt_base = np.asarray(day.rt_price_pool, dtype=float)[:SCENARIOS]
    rt_center = rt_base.mean(axis=1, keepdims=True)
    rt_scale = np.divide(
        particles[:, None, :, 1],
        np.where(np.abs(rt_center) > 1e-6, rt_center, 1.0),
    )
    pv_base = np.asarray(day.pv_pool, dtype=float)[:SCENARIOS]
    pv_center = pv_base.mean(axis=1, keepdims=True)
    pv_scale = np.divide(
        np.maximum(particles[:, None, :, 4], 0.0),
        np.where(pv_center > 1e-6, pv_center, 1.0),
    )
    pue = np.maximum(particles[:, :, 5].mean(axis=1), 1e-6)
    pue /= max(float(pue.mean()), 1e-6)
    return {
        "rt": rt_base * rt_scale,
        "workload": np.maximum(particles[:, :, 3], 0.0),
        "pv": np.maximum(pv_base * pv_scale, 0.0),
        "physical": pue[:, None]
        * np.asarray(day.physical_multiplier, dtype=float)[None, :],
    }


def _learned_distribution_and_callback(
    model: torch.nn.Module,
    sample: _DaySample,
    physical: FrozenPhysicalArtifact,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, Any]:
    """Issue the origin-zero distribution once and reuse its exact action."""

    callback = _learned_policy_adapter(model, sample, physical, device)
    captured: dict[str, Any] = {}

    # The regular callback intentionally exposes actions only.  Evaluate the
    # identical causal origin-zero inputs once here to obtain learned particles
    # and weights, then cache that action so runtime still sees one policy call
    # per origin semantically.
    inputs = {
        name: value.unsqueeze(0).to(device=device)
        for name, value in sample.inputs.items()
    }
    energy = np.maximum(np.asarray(physical.battery_energy_mwh, dtype=np.float32), 1e-6)
    current_soc = np.asarray(sample.day.initial_soc_mwh, dtype=np.float32)
    prepared = (
        model.prepare_origin_context(inputs)  # type: ignore[attr-defined]
        if hasattr(model, "prepare_origin_context")
        else None
    )
    zero_previous = {
        "grid": torch.zeros((1, 4), device=device),
        "battery": torch.zeros((1, 4), device=device),
        "compute": torch.zeros((1, 4), device=device),
        "migration": torch.zeros((1, 4, 4), device=device),
    }
    call_inputs = dict(inputs)
    state = torch.zeros((1, INTERVALS, 4), dtype=torch.float32, device=device)
    state[:, 0] = torch.from_numpy(current_soc / energy).to(device)
    call_inputs["state"] = state
    kwargs: dict[str, Any] = {
        "origin": 0,
        "current_soc": torch.from_numpy(current_soc).unsqueeze(0).to(device),
        "backlog": torch.full(
            (1, 4), float(sample.day.realized_workload[0]) / 4.0, device=device
        ),
        "deadline_pressure": torch.zeros((1, 4), device=device),
        "previous_action": zero_previous,
        "revealed_mask": torch.arange(INTERVALS, device=device).unsqueeze(0).le(0),
    }
    if prepared is not None:
        kwargs["prepared_context"] = prepared
    with torch.inference_mode():
        output = model.forward_origin(call_inputs, **kwargs)  # type: ignore[attr-defined]
    scenario = output.get("scenario")
    if not isinstance(scenario, Mapping):
        raise FormalValidationError("forward_origin omitted learned scenario mapping")
    particles = _to_numpy(scenario["particles"])[0]
    weights = _to_numpy(scenario["weights"])[0].astype(float, copy=True)
    if (
        particles.shape != (SCENARIOS, INTERVALS, 6)
        or weights.shape != (SCENARIOS,)
        or not np.isfinite(particles).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or float(weights.sum()) <= 0.0
    ):
        raise FormalValidationError("learned 20-particle distribution is invalid")
    weights /= float(weights.sum())

    origin_zero_context = {
        "origin": 0,
        "soc_mwh": current_soc,
        "queue_age_buckets": np.asarray(
            [0.0, 0.0, 0.0, 0.0, float(sample.day.realized_workload[0])],
            dtype=np.float32,
        ),
        "previous_action": None,
    }
    captured["origin_zero"] = callback(origin_zero_context)

    def cached_callback(context: Mapping[str, Any]) -> Mapping[str, Any]:
        if int(context["origin"]) == 0 and "origin_zero" in captured:
            return captured.pop("origin_zero")
        return callback(context)

    return _particles_to_joint(particles, sample.day), weights, cached_callback


def _paired_day_result(
    date: str,
    learned_result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
) -> dict[str, Any]:
    learned = _summary(learned_result)
    baseline = _summary(baseline_result)
    learned_cost = float(learned.get("total_cost_usd", math.nan))
    baseline_cost = float(baseline.get("total_cost_usd", math.nan))
    learned_sla = float(learned.get("normalized_sla_shortfall", math.nan))
    baseline_sla = float(baseline.get("normalized_sla_shortfall", math.nan))
    return {
        "operating_date": date,
        "learned_status": learned_result.get("status"),
        "baseline_status": baseline_result.get("status"),
        "learned_completed_origins": int(learned_result.get("completed_origins", 0)),
        "baseline_completed_origins": int(baseline_result.get("completed_origins", 0)),
        "learned_total_cost_usd": learned_cost,
        "baseline_total_cost_usd": baseline_cost,
        "cost_delta_usd": learned_cost - baseline_cost,
        "cost_delta_learned_minus_baseline_usd": learned_cost - baseline_cost,
        "learned_cvar_95_usd": float(learned.get("weighted_particle_cvar_95_usd", math.nan)),
        "baseline_cvar_95_usd": float(baseline.get("weighted_particle_cvar_95_usd", math.nan)),
        "learned_sla_shortfall_work_units": float(learned.get("sla_shortfall_work_units", 0.0)),
        "baseline_sla_shortfall_work_units": float(baseline.get("sla_shortfall_work_units", 0.0)),
        "learned_normalized_sla_shortfall": learned_sla,
        "baseline_normalized_sla_shortfall": baseline_sla,
        "normalized_sla_shortfall_delta": learned_sla - baseline_sla,
        "learned_feasibility_rate": float(learned.get("feasibility_rate", 0.0)),
        "baseline_feasibility_rate": float(baseline.get("feasibility_rate", 0.0)),
        "learned_rejected_origins": int(learned.get("rejected_origins", 0)),
        "baseline_rejected_origins": int(baseline.get("rejected_origins", 0)),
        "learned_rejected_non_exogenous_origins": int(
            sum(
                bool(row.get("rejected", False))
                and not bool(row.get("exogenous_infeasible", False))
                for row in learned_result.get("records", [])
            )
        ),
        "baseline_rejected_non_exogenous_origins": int(
            sum(
                bool(row.get("rejected", False))
                and not bool(row.get("exogenous_infeasible", False))
                for row in baseline_result.get("records", [])
            )
        ),
        "learned_exogenous_infeasible_origins": int(learned.get("exogenous_infeasible_origins", 0)),
        "baseline_exogenous_infeasible_origins": int(baseline.get("exogenous_infeasible_origins", 0)),
        "learned_corrective_magnitude": float(
            sum(float(row.get("corrective_intervention_magnitude", 0.0)) for row in learned_result.get("records", []))
        ),
        "baseline_corrective_magnitude": float(
            sum(float(row.get("corrective_intervention_magnitude", 0.0)) for row in baseline_result.get("records", []))
        ),
    }


class FormalValidationEvaluator:
    """Trainer-compatible full-panel causal validation callback."""

    def __init__(
        self,
        *,
        runtime_config: Mapping[str, Any] | None,
        calibration: Mapping[str, Any],
        formal: bool = True,
        max_origins: int = INTERVALS,
        logfile: Path | str | None = None,
        repository_root: Path | str | None = None,
        scenario_count: int = SCENARIOS,
        calibration_path: Path | str | None = None,
        max_normalized_sla_shortfall_increase: float = 0.001,
    ) -> None:
        if int(max_origins) != INTERVALS:
            raise FormalValidationError("formal evaluator requires 96 origins")
        if int(scenario_count) != SCENARIOS:
            raise FormalValidationError("formal evaluator requires 20 scenarios")
        self.runtime_config = dict(runtime_config or {})
        self.calibration = dict(calibration)
        self.formal = bool(formal)
        self.max_origins = INTERVALS
        self.logfile = None if logfile is None else Path(logfile)
        self.repository_root = None if repository_root is None else Path(repository_root)
        self.calibration_path = None if calibration_path is None else Path(calibration_path)
        self.max_normalized_sla_shortfall_increase = float(max_normalized_sla_shortfall_increase)

    def resolve_benchmark(self, dataset: Any, physical: Any) -> dict[str, Any]:
        """Freeze the model-independent benchmark; safe to call pre-training."""

        if self.calibration_path is not None and self.calibration_path.exists():
            try:
                persisted = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                persisted = None
            if isinstance(persisted, Mapping) and persisted.get("battery_fractions_frozen") is True:
                self.calibration = dict(persisted)
        if self.calibration.get("battery_fractions_frozen") is True:
            return {
                "status": "COMPLETE",
                "candidate_results": list(self.calibration.get("battery_candidate_results", [])),
                "selected_benchmark_parameters": {
                    "low_price_charge_fraction": float(self.calibration["low_price_charge_fraction"]),
                    "high_price_discharge_fraction": float(self.calibration["high_price_discharge_fraction"]),
                },
                "frozen_calibration": dict(self.calibration),
                "reused_frozen_calibration": True,
            }
        resolution = calibrate_price_aware_battery_rules(
            dataset,
            physical,
            runtime_config=self.runtime_config,
            calibration=self.calibration,
            formal=self.formal,
            max_origins=self.max_origins,
            logfile=self.logfile,
            max_normalized_sla_shortfall_increase=self.max_normalized_sla_shortfall_increase,
        )
        self.calibration = dict(resolution["frozen_calibration"])
        if self.calibration_path is not None:
            _atomic_json(self.calibration_path, self.calibration)
        return resolution

    calibrate_benchmark = resolve_benchmark

    def __call__(
        self,
        *,
        model: torch.nn.Module,
        dataset: Any,
        physical: Any,
        phase: str,
        device: str | torch.device,
    ) -> Mapping[str, Any]:
        return self.evaluate(model, dataset, physical, phase=phase, device=device)

    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: Any,
        physical: Any,
        *,
        phase: str = "a2",
        device: str | torch.device = "cpu",
    ) -> Mapping[str, Any]:
        if str(phase).lower() not in {"policy", "a1", "a2", "evaluation", "validation"}:
            raise FormalValidationError(f"unsupported decision validation phase {phase!r}")
        physical_artifact = _physical(physical)
        benchmark_resolution = self.resolve_benchmark(dataset, physical_artifact)
        days = _materialize_days(dataset)
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise FormalValidationError("requested CUDA validation device is unavailable")
        prior_training = bool(model.training)
        model.to(resolved_device).eval()
        baseline_policy = PriceAwareEDFShieldPolicy.from_calibration(self.calibration)
        learned_results: list[dict[str, Any]] = []
        baseline_results: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        paired: list[dict[str, Any]] = []
        learned_audits: list[dict[str, Any]] = []
        baseline_audits: list[dict[str, Any]] = []
        tolerance = float(self.runtime_config.get("validation_tolerance", 1.0e-5))
        try:
            for item in days:
                # Reuse the deployment adapter: it makes one causal origin-zero
                # forward, caches that exact action, and returns the learned
                # 20-particle distribution and normalized weights. Both
                # methods below consume the identical distribution.
                from .cli import _learned_policy_adapter as deployment_adapter

                joint_scenarios, scenario_weights, policy_callback = deployment_adapter(
                    model, item.day, physical_artifact
                )
                learned = NeuralConstrainedRuntime(
                    physical_artifact, self.runtime_config, logfile=self.logfile
                ).run_day(
                    item.day,
                    policy_callback=policy_callback,
                    joint_scenarios=joint_scenarios,
                    scenario_weights=scenario_weights,
                    max_origins=INTERVALS,
                    formal=self.formal,
                )
                baseline = NeuralConstrainedRuntime(
                    physical_artifact, self.runtime_config, logfile=self.logfile
                ).run_day(
                    item.day,
                    policy_callback=baseline_policy,
                    joint_scenarios=joint_scenarios,
                    scenario_weights=scenario_weights,
                    max_origins=INTERVALS,
                    formal=self.formal,
                )
                learned_audits.append(
                    _independently_audit_day(
                        learned,
                        physical=physical_artifact,
                        day=item.day,
                        tolerance=tolerance,
                    )
                )
                baseline_audits.append(
                    _independently_audit_day(
                        baseline,
                        physical=physical_artifact,
                        day=item.day,
                        tolerance=tolerance,
                    )
                )
                learned_results.append(learned)
                baseline_results.append(baseline)
                rows.extend(_row_records(learned, method="e2e_moe_solver_free_r3"))
                rows.extend(_row_records(baseline, method=PriceAwareEDFShieldPolicy.name))
                paired.append(_paired_day_result(item.day.operating_date, learned, baseline))
        finally:
            if prior_training:
                model.train()
        learned_aggregate = _aggregate(learned_results)
        baseline_aggregate = _aggregate(baseline_results)
        sla_increase = float(learned_aggregate["normalized_sla_shortfall"]) - float(
            baseline_aggregate["normalized_sla_shortfall"]
        )
        sla_passed = sla_increase <= self.max_normalized_sla_shortfall_increase
        completed_days = sum(
            learned.get("status") == "COMPLETE"
            and baseline.get("status") == "COMPLETE"
            and int(learned.get("completed_origins", 0)) == INTERVALS
            and int(baseline.get("completed_origins", 0)) == INTERVALS
            for learned, baseline in zip(learned_results, baseline_results, strict=True)
        )
        independently_valid = bool(
            len(learned_audits) == len(days)
            and len(baseline_audits) == len(days)
            and all(audit["all_passed"] for audit in learned_audits)
            and all(audit["all_passed"] for audit in baseline_audits)
            and all(audit["audited_origin_count"] == INTERVALS for audit in learned_audits)
            and all(audit["audited_origin_count"] == INTERVALS for audit in baseline_audits)
        )
        learned_independent_max = max(
            (float(audit["max_residual"]) for audit in learned_audits),
            default=math.inf,
        )
        baseline_independent_max = max(
            (float(audit["max_residual"]) for audit in baseline_audits),
            default=math.inf,
        )
        learned_independent_passed = sum(
            int(audit["passed_origin_count"]) for audit in learned_audits
        )
        baseline_independent_passed = sum(
            int(audit["passed_origin_count"]) for audit in baseline_audits
        )
        learned_independent_fraction = learned_independent_passed / max(
            len(days) * INTERVALS, 1
        )
        baseline_independent_fraction = baseline_independent_passed / max(
            len(days) * INTERVALS, 1
        )
        # Checkpoint eligibility consumes the aggregate/top-level feasibility
        # field, so it must be the independently recomputed result rather than
        # the shield's self-reported record flag.
        learned_aggregate["feasibility_rate"] = learned_independent_fraction
        learned_aggregate[
            "post_shield_feasibility_fraction"
        ] = learned_independent_fraction
        baseline_aggregate["feasibility_rate"] = baseline_independent_fraction
        baseline_aggregate[
            "post_shield_feasibility_fraction"
        ] = baseline_independent_fraction
        finite = all(
            math.isfinite(float(value))
            for value in (
                learned_aggregate["operating_cost_usd"],
                learned_aggregate["cvar_95_usd"],
                learned_aggregate["corrective_magnitude"],
                baseline_aggregate["operating_cost_usd"],
            )
        )
        status = (
            "COMPLETE"
            if completed_days == len(days) and finite and independently_valid
            else "PARTIAL"
        )
        result = {
            "status": status,
            "phase": str(phase),
            "validation_start": days[0].day.operating_date,
            "validation_end": days[-1].day.operating_date,
            "validation_day_count": len(days),
            "completed_day_count": completed_days,
            "completed_origin_count": int(learned_aggregate["completed_origin_count"] + baseline_aggregate["completed_origin_count"]),
            "expected_origin_count": len(days) * INTERVALS * 2,
            "scenario_contract": "same_model_generated_20_particles_and_normalized_weights",
            "independent_validation": {
                "all_passed": independently_valid,
                "tolerance": tolerance,
                "learned_audited_origin_count": sum(
                    int(audit["audited_origin_count"]) for audit in learned_audits
                ),
                "baseline_audited_origin_count": sum(
                    int(audit["audited_origin_count"]) for audit in baseline_audits
                ),
                "learned_max_residual": learned_independent_max,
                "baseline_max_residual": baseline_independent_max,
            },
            "learned": learned_aggregate,
            "learned_aggregate": learned_aggregate,
            "baseline": baseline_aggregate,
            "baseline_aggregate": baseline_aggregate,
            "aggregates": {
                "e2e_moe_solver_free_r3": learned_aggregate,
                PriceAwareEDFShieldPolicy.name: baseline_aggregate,
            },
            "paired_day_results": paired,
            "records": rows,
            "rows": rows,
            "candidate_results": benchmark_resolution.get(
                "candidate_results", self.calibration.get("battery_candidate_results", [])
            ),
            "selected_benchmark_parameters": benchmark_resolution["selected_benchmark_parameters"],
            "frozen_calibration": dict(self.calibration),
            "post_shield_feasibility_fraction": learned_independent_fraction,
            "sla_noninferiority_passed": bool(sla_passed),
            "normalized_sla_shortfall_increase": sla_increase,
            "operating_cost_usd": float(learned_aggregate["operating_cost_usd"]),
            "cvar_95_usd": float(learned_aggregate["cvar_95_usd"]),
            "corrective_magnitude": float(learned_aggregate["corrective_magnitude"]),
            "formal_2024_opened": False,
            "locked_2025_opened": False,
        }
        return result


__all__ = [
    "FormalValidationError",
    "FormalValidationEvaluator",
    "calibrate_price_aware_battery_rules",
]
