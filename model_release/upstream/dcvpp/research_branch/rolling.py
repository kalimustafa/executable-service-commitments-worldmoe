from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import FrozenPhysicalArtifact, ScenarioBundle
from .optimization import OptimizationFailure, solve_policy


ROLLING_INTERVAL_HOURS = 0.25
ROLLING_SCHEMA_VERSION = "rolling_counterfactual_day_v1"


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _update_array_hash(digest: Any, name: str, value: Any) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject:
        array = np.asarray(value, dtype="U")
    array = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _input_fingerprint(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    policy_id: str,
    realized_scenario_index: int,
    settings: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "operating_date": bundle.operating_date,
                "regime": bundle.regime,
                "kappa": bundle.kappa,
                "seed": bundle.seed,
                "policy_id": policy_id,
                "realized_scenario_index": realized_scenario_index,
                "calibration_id": physical.calibration_id,
                "twin_ids": list(physical.twin_ids),
                "settings": _plain(settings),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    for name, value in (
        ("timestamps", bundle.timestamps_utc),
        ("da_price", bundle.da_price),
        ("rt_price", bundle.rt_price),
        ("workload", bundle.workload),
        ("pv", bundle.pv),
        ("physical_multiplier", bundle.physical_multiplier),
        ("probabilities", bundle.probabilities),
        ("idle_mw", physical.idle_mw),
        ("work_mw_per_unit", physical.work_mw_per_unit),
        ("capacity_work_units", physical.capacity_work_units),
        ("grid_limit_mw", physical.grid_import_limit_mw),
        ("battery_power_mw", physical.battery_power_mw),
        ("battery_energy_mwh", physical.battery_energy_mwh),
    ):
        _update_array_hash(digest, name, value)
    return digest.hexdigest()


def _validate_quarter_hour_grid(timestamps: np.ndarray) -> None:
    if len(timestamps) <= 1:
        return
    parsed = [
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        for value in timestamps
    ]
    differences = [
        (later - earlier).total_seconds() for earlier, later in zip(parsed, parsed[1:])
    ]
    if not all(np.isclose(seconds, 15 * 60) for seconds in differences):
        raise ValueError("rolling operation requires a contiguous 15-minute time grid")


def _weighted_tail_mean(
    values: Sequence[float], probabilities: Sequence[float], alpha: float
) -> float:
    costs = np.asarray(values, dtype=float)
    weights = np.asarray(probabilities, dtype=float)
    order = np.argsort(costs)[::-1]
    remaining = 1.0 - alpha
    total = 0.0
    mass = 0.0
    for index in order:
        take = min(float(weights[index]), remaining - mass)
        if take > 0:
            total += take * float(costs[index])
            mass += take
        if mass >= remaining - 1e-15:
            break
    return float(total / mass) if mass > 0 else float(np.max(costs))


def _recompute_weighted_cvar(
    values: Sequence[float], probabilities: Sequence[float], alpha: float
) -> float:
    costs = np.asarray(values, dtype=float)
    weights = np.asarray(probabilities, dtype=float)
    order = np.argsort(costs)
    costs = costs[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    var = costs[
        min(int(np.searchsorted(cumulative, alpha, side="left")), len(costs) - 1)
    ]
    return float(
        var + np.sum(weights * np.maximum(costs - var, 0.0)) / (1.0 - alpha)
    )


def _pending_total(pending: Sequence[Mapping[str, Any]]) -> float:
    return float(sum(float(item["amount_work_units"]) for item in pending))


def _consume_fifo(
    pending: list[dict[str, Any]], amount: float, interval: int
) -> tuple[list[dict[str, float]], float]:
    remaining = max(float(amount), 0.0)
    completions: list[dict[str, float]] = []
    consumed = 0.0
    while remaining > 1e-9 and pending:
        item = pending[0]
        take = min(remaining, float(item["amount_work_units"]))
        if take > 0:
            completions.append(
                {
                    "delay_steps": float(
                        max(interval - int(item["release_interval"]), 0)
                    ),
                    "amount_work_units": take,
                }
            )
            item["amount_work_units"] = float(item["amount_work_units"]) - take
            remaining -= take
            consumed += take
        if float(item["amount_work_units"]) <= 1e-9:
            pending.pop(0)
    return completions, consumed


def _expire_due(
    pending: list[dict[str, Any]], interval: int
) -> tuple[float, float]:
    missed = 0.0
    critical_missed = 0.0
    retained: list[dict[str, Any]] = []
    for item in pending:
        if int(item["deadline_interval"]) <= interval:
            amount = float(item["amount_work_units"])
            missed += amount
            critical_missed += float(item.get("critical_fraction", 0.0)) * amount
        else:
            retained.append(item)
    pending[:] = retained
    return missed, critical_missed


def _rolling_suffix(
    bundle: ScenarioBundle,
    *,
    interval: int,
    realized_scenario_index: int,
    pending: Sequence[Mapping[str, Any]],
    deadline_steps: int,
) -> tuple[ScenarioBundle, np.ndarray, np.ndarray]:
    workload = np.asarray(bundle.workload[:, interval:], dtype=float).copy()
    rt_price = np.asarray(bundle.rt_price[:, :, interval:], dtype=float).copy()
    pv = np.asarray(bundle.pv[:, :, interval:], dtype=float).copy()
    physical_multiplier = np.asarray(bundle.physical_multiplier, dtype=float).copy()

    # Only the current interval is replaced by the designated replay outcome.
    # The future remains the issue-time scenario ensemble.  The physical
    # multiplier is day-level in this contract, so its realized value becomes
    # known for the remaining day once revealed.
    pending_work = _pending_total(pending)
    workload[:, 0] = pending_work
    rt_price[:, :, 0] = bundle.rt_price[realized_scenario_index, :, interval]
    pv[:, :, 0] = bundle.pv[realized_scenario_index, :, interval]
    physical_multiplier[:, :] = bundle.physical_multiplier[
        realized_scenario_index, :
    ]

    cumulative_release = np.cumsum(workload, axis=1)
    cumulative_due = np.zeros_like(workload)
    horizon = workload.shape[1]
    for scenario in range(bundle.reduced_count):
        for relative_period in range(horizon):
            absolute_period = interval + relative_period
            due = sum(
                float(item["amount_work_units"])
                for item in pending
                if int(item["deadline_interval"]) <= absolute_period
            )
            for future_period in range(interval + 1, absolute_period + 1):
                if future_period + deadline_steps <= absolute_period:
                    due += float(bundle.workload[scenario, future_period])
            cumulative_due[scenario, relative_period] = due

    suffix = replace(
        bundle,
        timestamps_utc=np.asarray(bundle.timestamps_utc[interval:]).copy(),
        da_price=np.asarray(bundle.da_price[:, interval:], dtype=float).copy(),
        rt_price=rt_price,
        workload=workload,
        pv=pv,
        physical_multiplier=physical_multiplier,
    )
    suffix.validate()
    return suffix, cumulative_release, cumulative_due


def _safe_fallback_dispatch(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    interval: int,
    realized_scenario_index: int,
    soc_mwh: np.ndarray,
    da_schedule_mw: np.ndarray,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    multiplier = bundle.physical_multiplier[realized_scenario_index]
    load = multiplier * np.asarray(physical.idle_mw, dtype=float)
    available_pv = bundle.pv[realized_scenario_index, :, interval]
    pv_used = np.minimum(load, available_pv)
    grid = np.maximum(load - pv_used, 0.0)
    p_da = da_schedule_mw[:, interval]
    dev = grid - p_da
    return {
        "work_work_units": np.zeros(len(bundle.zones)),
        "shortfall_work_units": np.zeros(len(bundle.zones)),
        "reassignment_work_units": np.zeros(len(bundle.zones)),
        "grid_import_mw": grid,
        "pv_used_mw": pv_used,
        "pv_curtailment_mw": np.maximum(available_pv - pv_used, 0.0),
        "battery_charge_mw": np.zeros(len(bundle.zones)),
        "battery_discharge_mw": np.zeros(len(bundle.zones)),
        "soc_start_mwh": soc_mwh.copy(),
        "soc_end_mwh": soc_mwh.copy(),
        "day_ahead_schedule_mw": p_da,
        "rt_deviation_purchase_mw": np.maximum(dev, 0.0),
        "rt_deviation_sale_mw": np.maximum(-dev, 0.0),
        "ramp_slack_mw": 0.0,
        "concentration_slack_mw": np.zeros(len(bundle.zones)),
        "solver": {
            "solve_time_seconds": 0.0,
            "mip_gap": None,
            "time_limit_reached": False,
            "number_of_variables": 0,
            "number_of_constraints": 0,
            "termination_condition": "fallback_after_noncritical_failure",
            "sensitivity_setting_keys_supported": [],
            "sensitivity_settings_consumed": {},
            "sensitivity_variant": None,
            "sensitivity_effective_inputs": {},
            "fallback": True,
        },
    }


def _first_interval_dispatch(result: Mapping[str, Any]) -> dict[str, Any]:
    details = result["dispatch_details"]

    def first(name: str) -> np.ndarray:
        return np.asarray(details[name], dtype=float)[0, :, 0]

    soc = np.asarray(details["soc_mwh"], dtype=float)[0]
    return {
        "work_work_units": first("work_work_units"),
        "shortfall_work_units": first("shortfall_work_units"),
        "reassignment_work_units": first("reassignment_work_units"),
        "grid_import_mw": first("grid_import_mw"),
        "pv_used_mw": first("pv_used_mw"),
        "pv_curtailment_mw": first("pv_curtailment_mw"),
        "battery_charge_mw": first("battery_charge_mw"),
        "battery_discharge_mw": first("battery_discharge_mw"),
        "soc_start_mwh": soc[:, 0],
        "soc_end_mwh": soc[:, 1],
        "day_ahead_schedule_mw": first("day_ahead_schedule_mw"),
        "rt_deviation_purchase_mw": first("rt_deviation_purchase_mw"),
        "rt_deviation_sale_mw": first("rt_deviation_sale_mw"),
        "ramp_slack_mw": float(
            np.asarray(details["ramp_slack_mw"], dtype=float)[0, 0]
        ),
        "concentration_slack_mw": first("concentration_slack_mw"),
        "solver": {
            "solve_time_seconds": float(result["solve_time_seconds"]),
            "mip_gap": result["mip_gap"],
            "time_limit_reached": bool(result["time_limit_reached"]),
            "number_of_variables": int(result["number_of_variables"]),
            "number_of_constraints": int(result["number_of_constraints"]),
            "termination_condition": result["termination_condition"],
            "sensitivity_setting_keys_supported": result[
                "sensitivity_setting_keys_supported"
            ],
            "sensitivity_settings_consumed": result[
                "sensitivity_settings_consumed"
            ],
            "sensitivity_variant": result["sensitivity_variant"],
            "sensitivity_effective_inputs": result[
                "sensitivity_effective_inputs"
            ],
            "fallback": False,
        },
    }


def _interval_validation(
    dispatch: Mapping[str, Any],
    *,
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    interval: int,
    realized_scenario_index: int,
    modeled_result: Mapping[str, Any] | None,
    missed_work_units: float,
    tolerance: float,
    interval_hours: float,
    pue_slope_multiplier: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    checkpoint_granularity_work_units: float,
) -> dict[str, Any]:
    work = np.asarray(dispatch["work_work_units"], dtype=float)
    grid = np.asarray(dispatch["grid_import_mw"], dtype=float)
    pv_used = np.asarray(dispatch["pv_used_mw"], dtype=float)
    pv_curtail = np.asarray(dispatch["pv_curtailment_mw"], dtype=float)
    charge = np.asarray(dispatch["battery_charge_mw"], dtype=float)
    discharge = np.asarray(dispatch["battery_discharge_mw"], dtype=float)
    soc_start = np.asarray(dispatch["soc_start_mwh"], dtype=float)
    soc_end = np.asarray(dispatch["soc_end_mwh"], dtype=float)
    p_da = np.asarray(dispatch["day_ahead_schedule_mw"], dtype=float)
    dev_plus = np.asarray(dispatch["rt_deviation_purchase_mw"], dtype=float)
    dev_minus = np.asarray(dispatch["rt_deviation_sale_mw"], dtype=float)
    multiplier = bundle.physical_multiplier[realized_scenario_index]
    load = multiplier * (
        physical.idle_mw
        + pue_slope_multiplier * physical.work_mw_per_unit * work
    )
    available_pv = bundle.pv[realized_scenario_index, :, interval]
    facility = grid + pv_used + discharge - load - charge
    battery = soc_end - soc_start - interval_hours * (
        charge_efficiency * charge - discharge / discharge_efficiency
    )
    settlement = dev_plus - dev_minus - (grid - p_da)
    modeled_short = float(np.asarray(dispatch["shortfall_work_units"]).sum())
    checks: dict[str, Any] = {
        "facility_energy_balance_max_abs_mw": float(
            np.max(np.abs(facility), initial=0.0)
        ),
        "pv_balance_max_abs_mw": float(
            np.max(np.abs(pv_used + pv_curtail - available_pv), initial=0.0)
        ),
        "battery_state_max_abs_mwh": float(
            np.max(np.abs(battery), initial=0.0)
        ),
        "market_deviation_max_abs_mw": float(
            np.max(np.abs(settlement), initial=0.0)
        ),
        "grid_bound_max_violation_mw": float(
            np.max(grid - physical.grid_import_limit_mw, initial=0.0)
        ),
        "capacity_max_violation_work_units": float(
            np.max(work - physical.capacity_work_units, initial=0.0)
        ),
        "integrality_max_abs_work_units": float(
            checkpoint_granularity_work_units
            * np.max(
                np.abs(
                    work / checkpoint_granularity_work_units
                    - np.rint(work / checkpoint_granularity_work_units)
                ),
                initial=0.0,
            )
        ),
        "simultaneous_charge_discharge_max_product_mw2": float(
            np.max(charge * discharge, initial=0.0)
        ),
        "simultaneous_deviation_max_product_mw2": float(
            np.max(dev_plus * dev_minus, initial=0.0)
        ),
        "realized_sla_accounting_model_difference_work_units": float(
            abs(modeled_short - missed_work_units)
        ),
    }
    if modeled_result is not None:
        components = sum(
            float(modeled_result[key][0])
            for key in (
                "scenario_day_ahead_costs_usd",
                "scenario_rt_imbalance_costs_usd",
                "scenario_sla_penalty_costs_usd",
                "scenario_migration_costs_usd",
                "scenario_migration_energy_costs_usd",
                "scenario_battery_degradation_costs_usd",
                "scenario_curtailment_costs_usd",
                "scenario_constraint_slack_costs_usd",
            )
        )
        checks["objective_decomposition_max_abs_usd"] = abs(
            float(modeled_result["scenario_costs_usd"][0]) - components
        )
        recomputed_cvar = _recompute_weighted_cvar(
            modeled_result["scenario_costs_usd"],
            modeled_result["scenario_probabilities"],
            float(modeled_result["cvar_alpha"]),
        )
        checks["empirical_cvar_recomputed_usd"] = recomputed_cvar
        checks["empirical_cvar_max_abs_usd"] = abs(
            recomputed_cvar - float(modeled_result["cvar_total_cost_usd"])
        )
        checks["solver_validation_passed"] = bool(
            modeled_result["validations"]["passed"]
        )
    else:
        checks["objective_decomposition_max_abs_usd"] = 0.0
        checks["empirical_cvar_recomputed_usd"] = 0.0
        checks["empirical_cvar_max_abs_usd"] = 0.0
        checks["solver_validation_passed"] = False
    required = [
        float(value)
        for key, value in checks.items()
        if key
        not in {
            "realized_sla_accounting_model_difference_work_units",
            "empirical_cvar_recomputed_usd",
            "solver_validation_passed",
        }
        and isinstance(value, (int, float))
    ]
    checks["passed"] = bool(
        checks["solver_validation_passed"]
        and max(required, default=0.0) <= tolerance
    )
    checks["sla_accounting_note"] = (
        "realized FIFO deadline accounting is authoritative; optimizer shortfall is "
        "reported separately because it is a horizon feasibility slack"
    )
    return checks


def _record_costs(
    dispatch: Mapping[str, Any],
    *,
    bundle: ScenarioBundle,
    interval: int,
    realized_scenario_index: int,
    missed_work_units: float,
    settings: Mapping[str, Any],
) -> dict[str, float]:
    dt = ROLLING_INTERVAL_HOURS
    p_da = np.asarray(dispatch["day_ahead_schedule_mw"], dtype=float)
    dev_plus = np.asarray(dispatch["rt_deviation_purchase_mw"], dtype=float)
    dev_minus = np.asarray(dispatch["rt_deviation_sale_mw"], dtype=float)
    rt_buy = bundle.rt_price[realized_scenario_index, :, interval]
    haircut = float(
        settings.get(
            "rt_sell_back_haircut", settings.get("rt_sell_haircut_fraction", 0.0)
        )
    )
    rt_sell = rt_buy - haircut * np.abs(rt_buy)
    migrated = 0.5 * float(
        np.asarray(dispatch["reassignment_work_units"], dtype=float).sum()
    )
    migration_energy = float(
        settings.get("transfer_energy_mwh_per_work_unit", 0.0)
    ) * migrated
    day_ahead = dt * float(np.dot(p_da, bundle.da_price[:, interval]))
    rt_imbalance = dt * float(np.dot(dev_plus, rt_buy) - np.dot(dev_minus, rt_sell))
    migration = float(settings.get("migration_penalty_usd_per_work_unit", 0.05)) * (
        1.0 + float(settings.get("checkpoint_overhead_fraction", 0.0))
    ) * migrated
    migration_energy_cost = float(
        settings.get(
            "migration_energy_price_usd_per_mwh",
            max(float(np.mean(bundle.da_price)), 0.0),
        )
    ) * migration_energy
    battery_throughput = dt * float(
        np.asarray(dispatch["battery_charge_mw"]).sum()
        + np.asarray(dispatch["battery_discharge_mw"]).sum()
    )
    battery = float(
        settings.get("battery_degradation_usd_per_mwh", 1.0)
    ) * battery_throughput
    curtailed = dt * float(np.asarray(dispatch["pv_curtailment_mw"]).sum())
    curtailment = float(
        settings.get("pv_curtailment_usd_per_mwh", 0.01)
    ) * curtailed
    constraint_slack = float(
        settings.get("constraint_slack_penalty_usd", 100000.0)
    ) * (
        float(dispatch.get("ramp_slack_mw", 0.0))
        + float(
            np.asarray(dispatch.get("concentration_slack_mw", 0.0), dtype=float).sum()
        )
    )
    sla = float(settings.get("sla_penalty_usd_per_work_unit", 10000.0)) * float(
        missed_work_units
    )
    total = (
        day_ahead
        + rt_imbalance
        + migration
        + migration_energy_cost
        + battery
        + curtailment
        + sla
        + constraint_slack
    )
    return {
        "simulated_total_cost_usd": total,
        "day_ahead_cost_usd": day_ahead,
        "rt_imbalance_cost_usd": rt_imbalance,
        "migration_cost_usd": migration,
        "migration_energy_mwh": migration_energy,
        "migration_energy_cost_usd": migration_energy_cost,
        "battery_degradation_cost_usd": battery,
        "battery_throughput_mwh": battery_throughput,
        "curtailment_cost_usd": curtailment,
        "pv_curtailment_mwh": curtailed,
        "sla_penalty_cost_usd": sla,
        "constraint_slack_cost_usd": constraint_slack,
    }


def _summarize(checkpoint: Mapping[str, Any], physical: FrozenPhysicalArtifact) -> dict[str, Any]:
    intervals = list(checkpoint["intervals"])
    costs = [record["costs"] for record in intervals]
    state = checkpoint["state"]
    ex_ante = checkpoint["ex_ante_risk"]
    scenario_costs = np.asarray(ex_ante["scenario_costs_usd"], dtype=float)
    probabilities = np.asarray(ex_ante["scenario_probabilities"], dtype=float)
    mean_cost = float(np.dot(probabilities, scenario_costs))
    downside = float(
        np.sqrt(np.dot(probabilities, np.maximum(scenario_costs - mean_cost, 0.0) ** 2))
    )
    arrivals = float(sum(record["service"]["arrivals_work_units"] for record in intervals))
    misses = float(sum(record["service"]["deadline_missed_work_units"] for record in intervals))
    critical_misses = float(
        sum(record["service"]["critical_shortfall_work_units"] for record in intervals)
    )
    delay_values: list[float] = []
    delay_weights: list[float] = []
    for record in intervals:
        for sample in record["service"]["completion_delay_samples"]:
            delay_values.append(float(sample["delay_steps"]) * ROLLING_INTERVAL_HOURS)
            delay_weights.append(float(sample["amount_work_units"]))
    if delay_weights and sum(delay_weights) > 0:
        mean_delay = float(np.average(delay_values, weights=delay_weights))
        order = np.argsort(delay_values)
        cumulative = np.cumsum(np.asarray(delay_weights)[order])
        threshold = 0.95 * cumulative[-1]
        p95_delay = float(np.asarray(delay_values)[order][np.searchsorted(cumulative, threshold)])
    else:
        mean_delay = 0.0
        p95_delay = 0.0
    grid_profiles = [
        np.asarray(record["dispatch"]["grid_import_mw"], dtype=float)
        for record in intervals
    ]
    system_grid = np.asarray([profile.sum() for profile in grid_profiles], dtype=float)
    ramps = np.abs(np.diff(system_grid)) if len(system_grid) > 1 else np.asarray([])
    peak = float(np.max(system_grid, initial=0.0))
    load_factor = float(system_grid.mean() / peak) if peak > 0 else 0.0
    concentration = 0.0
    for profile in grid_profiles:
        total = float(profile.sum())
        if total > 0:
            concentration = max(concentration, float(profile.max() / total))
    pv_used_mwh = ROLLING_INTERVAL_HOURS * sum(
        float(np.asarray(record["dispatch"]["pv_used_mw"]).sum()) for record in intervals
    )
    pv_available_mwh = ROLLING_INTERVAL_HOURS * sum(
        float(record["revealed"]["pv_available_mw_total"]) for record in intervals
    )
    battery_throughput = sum(cost["battery_throughput_mwh"] for cost in costs)
    total_battery_energy = float(np.asarray(physical.battery_energy_mwh).sum())
    battery_cycles = (
        battery_throughput / (2.0 * total_battery_energy)
        if total_battery_energy > 0
        else 0.0
    )
    realized_prices = np.asarray(
        [record["revealed"]["mean_rt_price_usd_per_mwh"] for record in intervals],
        dtype=float,
    )
    rebound = 0.0
    if len(system_grid) > 1:
        low_threshold = float(np.quantile(realized_prices, 0.10))
        for index in range(len(system_grid) - 1):
            if realized_prices[index] <= low_threshold:
                rebound = max(rebound, float(system_grid[index + 1] - system_grid[index]))
    solve_records = [checkpoint["day_ahead_solve"]] + [
        record["solver"] for record in intervals
    ]
    gaps = [float(item["mip_gap"]) for item in solve_records if item.get("mip_gap") is not None]
    failures = int(sum(bool(record.get("failure")) for record in intervals))
    validation_max = 0.0
    for record in intervals:
        for key, value in record["validation"].items():
            if isinstance(value, (int, float)) and (
                "max_" in key or key.endswith("_max_abs_usd")
            ):
                validation_max = max(validation_max, abs(float(value)))
    completed_delayed = sum(
        float(sample["amount_work_units"])
        for record in intervals
        for sample in record["service"]["completion_delay_samples"]
        if float(sample["delay_steps"]) > 0
    )
    migrated = 0.5 * sum(
        float(np.asarray(record["dispatch"]["reassignment_work_units"]).sum())
        for record in intervals
    )
    total_cost = float(sum(cost["simulated_total_cost_usd"] for cost in costs))
    return {
        "economic": {
            "simulated_total_cost_usd": total_cost,
            "cost_difference_vs_B0_usd": None,
            "market_value_vs_B0_usd": None,
            "cost_difference_vs_B4_usd": None,
            "market_value_vs_B4_usd": None,
            "day_ahead_cost_usd": float(sum(cost["day_ahead_cost_usd"] for cost in costs)),
            "rt_imbalance_cost_usd": float(sum(cost["rt_imbalance_cost_usd"] for cost in costs)),
            "migration_cost_usd": float(sum(cost["migration_cost_usd"] for cost in costs)),
            "migration_energy_mwh": float(sum(cost["migration_energy_mwh"] for cost in costs)),
            "migration_energy_cost_usd": float(sum(cost["migration_energy_cost_usd"] for cost in costs)),
            "battery_degradation_cost_usd": float(sum(cost["battery_degradation_cost_usd"] for cost in costs)),
            "curtailment_cost_usd": float(sum(cost["curtailment_cost_usd"] for cost in costs)),
            "sla_penalty_cost_usd": float(sum(cost["sla_penalty_cost_usd"] for cost in costs)),
            "constraint_slack_cost_usd": float(
                sum(cost["constraint_slack_cost_usd"] for cost in costs)
            ),
        },
        "risk": {
            "empirical_cvar_95_usd": _weighted_tail_mean(scenario_costs, probabilities, 0.95),
            "worst_5_percent_mean_daily_cost_usd": _weighted_tail_mean(scenario_costs, probabilities, 0.95),
            "maximum_daily_cost_usd": float(np.max(scenario_costs)),
            "downside_deviation_usd": downside,
            "risk_source": "day_ahead_issue_time_scenario_distribution",
        },
        "service": {
            "weighted_sla_satisfaction": 1.0 - misses / arrivals if arrivals > 0 else 1.0,
            "deadline_miss_rate": misses / arrivals if arrivals > 0 else 0.0,
            "critical_task_shortfall_work_units": critical_misses,
            "mean_delay_hours": mean_delay,
            "p95_delay_hours": p95_delay,
            "unfinished_work_units": _pending_total(state["pending_work"]),
            "arrivals_work_units": arrivals,
            "completed_work_units": float(
                sum(record["service"]["completed_work_units"] for record in intervals)
            ),
            "deadline_missed_work_units": misses,
        },
        "flexibility_and_grid": {
            "deferred_work_units": float(completed_delayed),
            "migrated_work_units": migrated,
            "migrated_gb": migrated * float(checkpoint["settings"].get("migration_gb_per_work_unit", 0.0)),
            "pv_self_consumption_mwh": pv_used_mwh,
            "pv_self_consumption_share": pv_used_mwh / pv_available_mwh if pv_available_mwh > 0 else 0.0,
            "battery_equivalent_full_cycles": battery_cycles,
            "peak_import_mw": peak,
            "maximum_15_minute_ramp_mw": float(np.max(ramps, initial=0.0)),
            "load_factor": load_factor,
            "maximum_zone_concentration_share": concentration,
            "rebound_after_low_price_intervals_mw": rebound,
        },
        "computation": {
            "solve_time_seconds": float(sum(float(item.get("solve_time_seconds", 0.0)) for item in solve_records)),
            "mean_mip_gap": float(np.mean(gaps)) if gaps else None,
            "maximum_mip_gap": float(np.max(gaps)) if gaps else None,
            "timeout_count": int(sum(bool(item.get("time_limit_reached")) for item in solve_records)),
            "timeout_rate": float(sum(bool(item.get("time_limit_reached")) for item in solve_records) / len(solve_records)),
            "infeasibility_or_failure_count": failures,
            "infeasibility_or_failure_rate": failures / max(len(intervals), 1),
            "variables_solved_total": int(sum(int(item.get("number_of_variables", 0)) for item in solve_records)),
            "constraints_solved_total": int(sum(int(item.get("number_of_constraints", 0)) for item in solve_records)),
            "maximum_validation_residual": validation_max,
            "all_interval_validations_passed": bool(
                intervals and all(record["validation"]["passed"] for record in intervals)
            ),
        },
    }


def compare_rolling_policy_results(
    results_by_policy: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Compute paired B0/B4 values without inventing an unavailable baseline."""

    totals = {
        policy_id: float(result["metrics"]["economic"]["simulated_total_cost_usd"])
        for policy_id, result in results_by_policy.items()
    }
    output: dict[str, dict[str, float | None]] = {}
    for policy_id, total in totals.items():
        b0 = totals.get("B0")
        b4 = totals.get("B4")
        output[policy_id] = {
            "cost_difference_vs_B0_usd": None if b0 is None else total - b0,
            "market_value_vs_B0_usd": None if b0 is None else b0 - total,
            "cost_difference_vs_B4_usd": None if b4 is None else total - b4,
            "market_value_vs_B4_usd": None if b4 is None else b4 - total,
        }
    return output


def run_rolling_day(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    policy_id: str,
    config: Mapping[str, Any] | None = None,
    *,
    realized_scenario_index: int = 0,
    checkpoint_path: Path | None = None,
    max_intervals: int | None = None,
) -> dict[str, Any]:
    """Run or resume one causally revealed 15-minute counterfactual day.

    The day-ahead schedule is solved exactly once.  Every recourse solve sees
    only the current revealed interval plus the original future forecast
    ensemble, executes only its non-anticipative first decision, and never
    reopens a recorded past decision.
    """

    bundle.validate()
    _validate_quarter_hour_grid(bundle.timestamps_utc)
    settings = dict(config or {})
    interval_hours = float(settings.get("interval_hours", ROLLING_INTERVAL_HOURS))
    if not np.isclose(interval_hours, ROLLING_INTERVAL_HOURS):
        raise ValueError("rolling operation is fixed at 15-minute intervals")
    settings["interval_hours"] = ROLLING_INTERVAL_HOURS
    if not 0 <= realized_scenario_index < bundle.reduced_count:
        raise ValueError("realized_scenario_index is outside the reduced bundle")
    if int(settings.get("migration_duration_steps", 1)) != 1:
        raise ValueError(
            "the current cohort transfer contract supports one-interval active transfers only"
        )
    if policy_id == "B8":
        raise ValueError(
            "B8 is an oracle lower bound and is excluded from the causal rolling interface"
        )
    if max_intervals is not None and max_intervals < 0:
        raise ValueError("max_intervals must be nonnegative or null")

    fingerprint = _input_fingerprint(
        bundle, physical, policy_id, realized_scenario_index, settings
    )
    horizon = bundle.rt_price.shape[2]
    deadline_steps = max(0, int(settings.get("deadline_steps", horizon - 1)))
    critical_fraction = float(settings.get("critical_workload_share", 0.0))
    if not 0.0 <= critical_fraction <= 1.0:
        raise ValueError("critical_workload_share must lie in [0, 1]")
    validation_tolerance = float(settings.get("validation_tolerance", 1e-5))

    checkpoint: dict[str, Any]
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("schema_version") != ROLLING_SCHEMA_VERSION:
            raise ValueError("rolling checkpoint schema does not match")
        if checkpoint.get("input_sha256") != fingerprint:
            raise ValueError("rolling checkpoint input fingerprint does not match")
        if checkpoint.get("status") == "COMPLETE":
            return checkpoint
    else:
        initial_soc = np.asarray(
            settings.get(
                "initial_soc_mwh", 0.5 * np.asarray(physical.battery_energy_mwh)
            ),
            dtype=float,
        )
        terminal_target = np.asarray(
            settings.get("terminal_soc_target_mwh", initial_soc), dtype=float
        )
        day_ahead_settings = dict(settings)
        for key in (
            "fixed_day_ahead_schedule_mw",
            "previous_grid_import_mw",
            "rolling_cumulative_release_work_units",
            "rolling_cumulative_due_work_units",
        ):
            day_ahead_settings.pop(key, None)
        day_ahead_settings.update(
            {
                "initial_soc_mwh": initial_soc,
                "terminal_soc_target_mwh": terminal_target,
                "enforce_terminal_soc": True,
                "recourse_nonanticipative_prefix_steps": 0,
                "return_dispatch_details": False,
            }
        )
        day_ahead = solve_policy(
            bundle, physical, policy_id, day_ahead_settings
        )
        schedules = np.asarray(day_ahead["day_ahead_schedule_mw"], dtype=float)
        schedule_index = realized_scenario_index if policy_id == "B8" else 0
        day_ahead_schedule = schedules[schedule_index]
        previous_grid_setting = settings.get("initial_previous_grid_import_mw")
        checkpoint = {
            "schema_version": ROLLING_SCHEMA_VERSION,
            "status": "IN_PROGRESS",
            "operating_date": bundle.operating_date,
            "policy_id": policy_id,
            "input_sha256": fingerprint,
            "realized_scenario_index": realized_scenario_index,
            "interval_hours": ROLLING_INTERVAL_HOURS,
            "interval_count": horizon,
            "settings": _plain(settings),
            "day_ahead_schedule_mw": day_ahead_schedule.tolist(),
            "day_ahead_solve": {
                "solve_time_seconds": day_ahead["solve_time_seconds"],
                "mip_gap": day_ahead["mip_gap"],
                "time_limit_reached": day_ahead["time_limit_reached"],
                "number_of_variables": day_ahead["number_of_variables"],
                "number_of_constraints": day_ahead["number_of_constraints"],
                "termination_condition": day_ahead["termination_condition"],
                "sensitivity_setting_keys_supported": day_ahead[
                    "sensitivity_setting_keys_supported"
                ],
                "sensitivity_settings_consumed": day_ahead[
                    "sensitivity_settings_consumed"
                ],
                "sensitivity_variant": day_ahead["sensitivity_variant"],
                "sensitivity_effective_inputs": day_ahead[
                    "sensitivity_effective_inputs"
                ],
            },
            "ex_ante_risk": {
                "scenario_costs_usd": day_ahead["scenario_costs_usd"],
                "scenario_probabilities": day_ahead["scenario_probabilities"],
                "empirical_cvar_usd": day_ahead["cvar_total_cost_usd"],
                "cvar_alpha": day_ahead["cvar_alpha"],
            },
            "state": {
                "next_interval": 0,
                "soc_mwh": initial_soc.tolist(),
                "previous_grid_import_mw": (
                    None
                    if previous_grid_setting is None
                    else np.asarray(previous_grid_setting, dtype=float).tolist()
                ),
                "active_transfers_work_units_by_zone": np.zeros(
                    len(bundle.zones)
                ).tolist(),
                "pending_work": [],
            },
            "intervals": [],
            "failures": [],
            "evidence_role": "counterfactual",
            "measurement_relationship": "not_measured_by_this_study",
            "future_realized_targets_used": False,
            "calibration_id": physical.calibration_id,
            "twin_ids": list(physical.twin_ids),
        }
        checkpoint["metrics"] = _summarize(checkpoint, physical)
        if checkpoint_path is not None:
            _atomic_json(checkpoint_path, checkpoint)

    day_ahead_schedule = np.asarray(
        checkpoint["day_ahead_schedule_mw"], dtype=float
    )
    processed_this_call = 0
    while int(checkpoint["state"]["next_interval"]) < horizon:
        if max_intervals is not None and processed_this_call >= max_intervals:
            break
        interval = int(checkpoint["state"]["next_interval"])
        state = checkpoint["state"]
        pending = [dict(item) for item in state["pending_work"]]
        arrival = float(bundle.workload[realized_scenario_index, interval])
        if arrival > 0:
            pending.append(
                {
                    "release_interval": interval,
                    "deadline_interval": interval + deadline_steps,
                    "amount_work_units": arrival,
                    "critical_fraction": critical_fraction,
                }
            )
        state_before = {
            "unfinished_work_units": _pending_total(pending),
            "soc_mwh": list(state["soc_mwh"]),
            "previous_grid_import_mw": state["previous_grid_import_mw"],
            "active_transfers_work_units_by_zone": list(
                state["active_transfers_work_units_by_zone"]
            ),
        }
        suffix, cumulative_release, cumulative_due = _rolling_suffix(
            bundle,
            interval=interval,
            realized_scenario_index=realized_scenario_index,
            pending=pending,
            deadline_steps=deadline_steps,
        )
        recourse_settings = dict(settings)
        recourse_settings.update(
            {
                "initial_soc_mwh": state["soc_mwh"],
                "terminal_soc_target_mwh": settings.get(
                    "terminal_soc_target_mwh",
                    0.5 * np.asarray(physical.battery_energy_mwh),
                ),
                "enforce_terminal_soc": True,
                "fixed_day_ahead_schedule_mw": day_ahead_schedule[:, interval:],
                "recourse_nonanticipative_prefix_steps": 1,
                "return_dispatch_details": True,
                "rolling_cumulative_release_work_units": cumulative_release,
                "rolling_cumulative_due_work_units": cumulative_due,
            }
        )
        if state["previous_grid_import_mw"] is not None:
            recourse_settings["previous_grid_import_mw"] = state[
                "previous_grid_import_mw"
            ]
        modeled_result: Mapping[str, Any] | None = None
        failure: dict[str, Any] | None = None
        try:
            modeled_result = solve_policy(
                suffix, physical, policy_id, recourse_settings
            )
            dispatch = _first_interval_dispatch(modeled_result)
        except OptimizationFailure as exc:
            critical_policies = set(
                map(str, settings.get("critical_policy_ids", ("B0", "B4")))
            )
            if policy_id in critical_policies or not bool(
                settings.get("continue_on_noncritical_failure", True)
            ):
                raise
            failure = {
                "interval": interval,
                "type": type(exc).__name__,
                "message": str(exc),
                "diagnostics": _plain(exc.diagnostics),
            }
            checkpoint["failures"].append(failure)
            dispatch = _safe_fallback_dispatch(
                bundle,
                physical,
                interval,
                realized_scenario_index,
                np.asarray(state["soc_mwh"], dtype=float),
                day_ahead_schedule,
                settings,
            )

        requested_execution = float(
            np.asarray(dispatch["work_work_units"], dtype=float).sum()
        )
        completion_samples, completed = _consume_fifo(
            pending, requested_execution, interval
        )
        execution_overdraw = max(requested_execution - completed, 0.0)
        missed, critical_missed = _expire_due(pending, interval)
        costs = _record_costs(
            dispatch,
            bundle=bundle,
            interval=interval,
            realized_scenario_index=realized_scenario_index,
            missed_work_units=missed,
            settings=settings,
        )
        validation = _interval_validation(
            dispatch,
            bundle=bundle,
            physical=physical,
            interval=interval,
            realized_scenario_index=realized_scenario_index,
            modeled_result=modeled_result,
            missed_work_units=missed,
            tolerance=validation_tolerance,
            interval_hours=ROLLING_INTERVAL_HOURS,
            pue_slope_multiplier=float(settings.get("pue_slope_multiplier", 1.0)),
            charge_efficiency=float(settings.get("battery_charge_efficiency", 0.95)),
            discharge_efficiency=float(
                settings.get("battery_discharge_efficiency", 0.95)
            ),
            checkpoint_granularity_work_units=float(
                settings.get("checkpoint_granularity_work_units", 1.0)
            ),
        )
        validation["workload_execution_overdraw_work_units"] = execution_overdraw
        queue_conservation = abs(
            float(state_before["unfinished_work_units"])
            - completed
            - missed
            - _pending_total(pending)
        )
        validation["task_conservation_max_abs_work_units"] = queue_conservation
        if modeled_result is not None:
            validation["migration_conservation_max_abs"] = float(
                modeled_result["validations"][
                    "assignment_migration_conservation_max_abs"
                ]
            )
        else:
            validation["migration_conservation_max_abs"] = float("inf")
        if (
            execution_overdraw > validation_tolerance
            or queue_conservation > validation_tolerance
            or validation["migration_conservation_max_abs"] > validation_tolerance
        ):
            validation["passed"] = False
        record = {
            "interval": interval,
            "timestamp_utc": str(bundle.timestamps_utc[interval]),
            "revealed_through_interval": interval,
            "past_decisions_fixed_count": interval,
            "future_realized_targets_used": False,
            "revealed": {
                "rt_price_usd_per_mwh_by_zone": bundle.rt_price[
                    realized_scenario_index, :, interval
                ].tolist(),
                "mean_rt_price_usd_per_mwh": float(
                    bundle.rt_price[realized_scenario_index, :, interval].mean()
                ),
                "workload_arrival_work_units": arrival,
                "pv_available_mw_total": float(
                    bundle.pv[realized_scenario_index, :, interval].sum()
                ),
                "physical_multiplier_by_zone": bundle.physical_multiplier[
                    realized_scenario_index
                ].tolist(),
            },
            "state_before": state_before,
            "dispatch": _plain(dispatch),
            "service": {
                "arrivals_work_units": arrival,
                "completed_work_units": completed,
                "deadline_missed_work_units": missed,
                "critical_shortfall_work_units": critical_missed,
                "unfinished_work_units": _pending_total(pending),
                "completion_delay_samples": completion_samples,
                "modeled_horizon_shortfall_work_units": float(
                    np.asarray(dispatch["shortfall_work_units"]).sum()
                ),
            },
            "costs": costs,
            "solver": _plain(dispatch["solver"]),
            "validation": validation,
            "failure": failure,
        }
        checkpoint["intervals"].append(record)
        state["next_interval"] = interval + 1
        state["soc_mwh"] = np.asarray(dispatch["soc_end_mwh"], dtype=float).tolist()
        state["previous_grid_import_mw"] = np.asarray(
            dispatch["grid_import_mw"], dtype=float
        ).tolist()
        state["active_transfers_work_units_by_zone"] = np.asarray(
            dispatch["reassignment_work_units"], dtype=float
        ).tolist()
        state["pending_work"] = _plain(pending)
        checkpoint["metrics"] = _summarize(checkpoint, physical)
        processed_this_call += 1
        if checkpoint_path is not None:
            _atomic_json(checkpoint_path, checkpoint)

    if int(checkpoint["state"]["next_interval"]) == horizon:
        checkpoint["status"] = (
            "COMPLETE"
            if not checkpoint["failures"]
            and checkpoint["metrics"]["computation"][
                "all_interval_validations_passed"
            ]
            else "INVALID"
        )
    else:
        checkpoint["status"] = "IN_PROGRESS"
    checkpoint["completed_interval_count"] = len(checkpoint["intervals"])
    checkpoint["metrics"] = _summarize(checkpoint, physical)
    if checkpoint_path is not None:
        _atomic_json(checkpoint_path, checkpoint)
    return checkpoint
