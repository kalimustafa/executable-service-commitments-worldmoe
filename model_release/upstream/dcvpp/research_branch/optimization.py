from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pyomo.environ as pyo

from .contracts import FrozenPhysicalArtifact, NYISO_ZONES, ScenarioBundle, XcelTariffDefinition


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    name: str
    temporal_flexibility: bool
    spatial_flexibility: bool
    storage: bool
    stochastic: bool
    cvar: bool
    stress_aware: bool = False
    oracle: bool = False


POLICIES: dict[str, PolicySpec] = {
    "B0": PolicySpec("B0", "rigid", False, False, False, False, False),
    "B1": PolicySpec("B1", "time_only", True, False, False, False, False),
    "B2": PolicySpec("B2", "space_only", False, True, False, False, False),
    "B3": PolicySpec("B3", "time_and_space_no_storage", True, True, False, False, False),
    "B4": PolicySpec("B4", "deterministic_full_flex_with_storage", True, True, True, False, False),
    "B5": PolicySpec("B5", "stochastic_expected_cost", True, True, True, True, False),
    "B6": PolicySpec("B6", "stochastic_nominal_cvar", True, True, True, True, True),
    "B7": PolicySpec("B7", "stress_aware_mixture_cvar", True, True, True, True, True, True),
    "B8": PolicySpec("B8", "perfect_information_lower_bound", True, True, True, True, False, False, True),
}

SENSITIVITY_SETTING_KEYS = (
    "flexible_share",
    "deadline_steps",
    "migration_eligible",
    "migration_bandwidth_multiplier",
    "migration_bandwidth_base_work_units_per_interval",
    "transfer_energy_mwh_per_work_unit",
    "checkpoint_overhead_fraction",
    "workload_scale",
    "battery_duration_hours",
    "pv_capacity_multiplier",
    "pue_slope_multiplier",
    "physical_ensemble_selector",
    "physical_multiplier_scale",
    "support_distance_threshold",
    "out_of_support_capacity_derating",
    "site_capacity_multiplier",
    "hardware_transfer_multiplier",
    "rt_volatility_scale",
    "negative_price_floor_usd_per_mwh",
    "rt_sell_back_haircut",
    "cvar_alpha",
    "risk_weight",
    "stress_mass",
    "ramp_limit_mw_per_step",
    "concentration_share_limit",
)


class OptimizationFailure(RuntimeError):
    def __init__(self, message: str, diagnostics: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def _home_chunks(demand_chunks: np.ndarray, zones: int) -> np.ndarray:
    scenarios, horizon = demand_chunks.shape
    home = np.zeros((scenarios, zones, horizon), dtype=int)
    for scenario in range(scenarios):
        for period in range(horizon):
            quotient, remainder = divmod(int(demand_chunks[scenario, period]), zones)
            home[scenario, :, period] = quotient
            home[scenario, :remainder, period] += 1
    return home


def _safe_value(value: Any) -> float:
    result = pyo.value(value, exception=False)
    return float(result) if result is not None else float("nan")


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(nested) for nested in value]
    return value


def _constraint_violation_summary(
    model: pyo.ConcreteModel,
    *,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Measure primal constraint violations at the model's current values."""

    maximum = 0.0
    violating: dict[str, float] = {}
    for component in model.component_objects(pyo.Constraint, active=True):
        component_maximum = 0.0
        for constraint in component.values():
            body = pyo.value(constraint.body, exception=False)
            if body is None or not np.isfinite(float(body)):
                component_maximum = float("inf")
                break
            violation = 0.0
            if constraint.lower is not None:
                lower = pyo.value(constraint.lower, exception=False)
                if lower is None or not np.isfinite(float(lower)):
                    violation = float("inf")
                else:
                    violation = max(violation, float(lower) - float(body))
            if constraint.upper is not None:
                upper = pyo.value(constraint.upper, exception=False)
                if upper is None or not np.isfinite(float(upper)):
                    violation = float("inf")
                else:
                    violation = max(violation, float(body) - float(upper))
            component_maximum = max(component_maximum, violation)
        maximum = max(maximum, component_maximum)
        if component_maximum > tolerance:
            violating[str(component.name)] = float(component_maximum)
    return {
        "feasible": bool(maximum <= tolerance),
        "maximum_constraint_violation": float(maximum),
        "violating_constraint_components": violating,
        "tolerance": float(tolerance),
    }


def _seed_all_short_mip_start(
    model: pyo.ConcreteModel,
    *,
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    policy: PolicySpec,
    home: np.ndarray,
    effective_physical_multiplier: np.ndarray,
    initial_soc_mwh: np.ndarray,
    terminal_soc_target_mwh: np.ndarray | None,
    interval_hours: float,
    eta_charge: float,
    eta_discharge: float,
    pue_slope_multiplier: float,
    granularity: float,
    previous_grid_import_mw: np.ndarray | None,
    ramp_limit: float | None,
    concentration_limit: float | None,
    tolerance: float,
) -> dict[str, Any]:
    """Install a conservative feasible MIP start without relaxing constraints.

    Every released workload chunk remains at its home site and is represented
    as explicit SLA shortfall.  Migration and storage are zero.  This is an
    expensive operational policy, but it is a scientifically valid incumbent
    whenever the frozen hard constraints admit it.
    """

    variables = tuple(model.component_data_objects(pyo.Var, descend_into=True))
    original_values = [(variable, variable.value) for variable in variables]
    w_count, z_count, horizon = bundle.rt_price.shape

    def assign(variable: Any, value: float) -> None:
        variable.set_value(value, skip_validation=True)

    try:
        grid_totals = np.zeros((w_count, horizon), dtype=float)
        for w in range(w_count):
            for z in range(z_count):
                initial_soc = float(initial_soc_mwh[z])
                terminal_soc = (
                    initial_soc
                    if terminal_soc_target_mwh is None
                    else float(terminal_soc_target_mwh[z])
                )
                load = float(effective_physical_multiplier[w, z]) * float(
                    physical.idle_mw[z]
                )
                charge_profile = np.zeros(horizon, dtype=float)
                discharge_profile = np.zeros(horizon, dtype=float)
                soc_delta = terminal_soc - initial_soc
                if soc_delta > tolerance:
                    required_charge = soc_delta / (interval_hours * eta_charge)
                    charge_capacity = np.minimum(
                        float(physical.battery_power_mw[z]),
                        np.maximum(
                            float(physical.grid_import_limit_mw[z])
                            + np.asarray(bundle.pv[w, z], dtype=float)
                            - load,
                            0.0,
                        ),
                    )
                    for t in range(horizon):
                        charge_profile[t] = min(required_charge, charge_capacity[t])
                        required_charge -= charge_profile[t]
                    if required_charge > tolerance:
                        raise ValueError(
                            "all-short MIP start cannot reach the terminal SOC by charging"
                        )
                elif soc_delta < -tolerance:
                    required_discharge = -soc_delta * eta_discharge / interval_hours
                    discharge_capacity = min(
                        float(physical.battery_power_mw[z]), load
                    )
                    for t in range(horizon):
                        discharge_profile[t] = min(
                            required_discharge, discharge_capacity
                        )
                        required_discharge -= discharge_profile[t]
                    if required_discharge > tolerance:
                        raise ValueError(
                            "all-short MIP start cannot reach the terminal SOC by discharging"
                        )
                soc = initial_soc
                assign(model.soc[z, 0, w], soc)
                for t in range(horizon):
                    assign(model.work[z, t, w], 0.0)
                    assign(model.short[z, t, w], int(home[w, z, t]))
                    assign(model.reassignment[z, t, w], 0.0)
                    charge = float(charge_profile[t])
                    discharge = float(discharge_profile[t])
                    assign(model.charge[z, t, w], charge)
                    assign(model.discharge[z, t, w], discharge)
                    assign(model.battery_mode[z, t, w], 1.0 if charge > 0 else 0.0)
                    soc += interval_hours * (
                        eta_charge * charge - discharge / eta_discharge
                    )
                    assign(model.soc[z, t + 1, w], soc)

                    available_pv = float(bundle.pv[w, z, t])
                    net_load = load + charge - discharge
                    pv_used = min(net_load, available_pv)
                    grid = max(net_load - pv_used, 0.0)
                    assign(model.pv_used[z, t, w], pv_used)
                    assign(model.pv_curtail[z, t, w], available_pv - pv_used)
                    assign(model.grid[z, t, w], grid)
                    grid_totals[w, t] += grid

                    p_da = (
                        float(model.p_da[z, t, w].value)
                        if model.p_da[z, t, w].fixed
                        and model.p_da[z, t, w].value is not None
                        else 0.0
                    )
                    assign(model.p_da[z, t, w], p_da)
                    deviation = grid - p_da
                    assign(model.dev_plus[z, t, w], max(deviation, 0.0))
                    assign(model.dev_minus[z, t, w], max(-deviation, 0.0))

        for w in range(w_count):
            for t in range(horizon):
                if ramp_limit is None:
                    ramp_slack = 0.0
                elif t == 0 and previous_grid_import_mw is not None:
                    ramp_slack = max(
                        abs(grid_totals[w, t] - float(previous_grid_import_mw.sum()))
                        - float(ramp_limit),
                        0.0,
                    )
                elif t > 0:
                    ramp_slack = max(
                        abs(grid_totals[w, t] - grid_totals[w, t - 1])
                        - float(ramp_limit),
                        0.0,
                    )
                else:
                    ramp_slack = 0.0
                assign(model.ramp_slack[t, w], ramp_slack)
                for z in range(z_count):
                    concentration_slack = (
                        0.0
                        if concentration_limit is None
                        else max(
                            float(model.grid[z, t, w].value)
                            - float(concentration_limit) * grid_totals[w, t],
                            0.0,
                        )
                    )
                    assign(model.concentration_slack[z, t, w], concentration_slack)

        if policy.cvar:
            scenario_costs = np.asarray(
                [pyo.value(model.scenario_cost[w]) for w in range(w_count)],
                dtype=float,
            )
            eta = float(np.max(scenario_costs))
            assign(model.eta, eta)
            for w, scenario_cost in enumerate(scenario_costs):
                assign(model.excess[w], max(float(scenario_cost) - eta, 0.0))

        summary = _constraint_violation_summary(model, tolerance=tolerance)
        summary.update(
            {
                "attempted": True,
                "strategy": "all_short_home_no_migration_zero_storage_v1",
                "all_hard_constraints_retained": True,
                "sla_shortfall_is_penalized_not_relaxed": True,
            }
        )
        if summary["feasible"]:
            return summary
    except Exception as exc:
        summary = {
            "attempted": True,
            "feasible": False,
            "strategy": "all_short_home_no_migration_zero_storage_v1",
            "all_hard_constraints_retained": True,
            "sla_shortfall_is_penalized_not_relaxed": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    for variable, value in original_values:
        variable.set_value(value, skip_validation=True)
    return summary


def _highs_solve_outcome(
    solver: Any,
    solve_result: Any,
) -> dict[str, Any]:
    """Classify HiGHS termination without conflating timeout and infeasibility."""

    termination = str(solve_result.solver.termination_condition)
    normalized = "".join(character for character in termination.lower() if character.isalnum())
    status = str(solve_result.solver.status)
    problem = solve_result.problem
    lower_bound = _optional_float(getattr(problem, "lower_bound", None))
    upper_bound = _optional_float(getattr(problem, "upper_bound", None))
    native = getattr(solver, "_last_results_object", None)
    incumbent = None
    objective_bound = None
    if native is not None:
        for name in ("incumbent_objective", "best_feasible_objective"):
            incumbent = _optional_float(getattr(native, name, None))
            if incumbent is not None:
                break
        for name in ("objective_bound", "best_objective_bound"):
            objective_bound = _optional_float(getattr(native, name, None))
            if objective_bound is not None:
                break
    if incumbent is None:
        incumbent = upper_bound
    if objective_bound is None:
        objective_bound = lower_bound
    solution = getattr(solver, "_sol", None)
    has_incumbent = bool(
        incumbent is not None
        or getattr(solution, "value_valid", False)
        or normalized
        in {
            "optimal",
            "globallyoptimal",
            "locallyoptimal",
            "convergencecriteriasatisfied",
        }
    )
    timed_out = normalized in {"maxtimelimit", "timelimit"}
    if normalized in {
        "infeasible",
        "proveninfeasible",
        "infeasibleorunbounded",
        "unboundedorinfeasible",
    }:
        classification = "INFEASIBLE"
        has_incumbent = False
    elif normalized in {
        "optimal",
        "globallyoptimal",
        "locallyoptimal",
        "convergencecriteriasatisfied",
    }:
        classification = "OPTIMAL"
    elif timed_out and has_incumbent:
        classification = "FEASIBLE_TIMEOUT"
    elif timed_out:
        classification = "NO_INCUMBENT_TIMEOUT"
    elif has_incumbent:
        classification = "FEASIBLE_LIMIT"
    else:
        classification = "NO_FEASIBLE_SOLUTION"
    return {
        "classification": classification,
        "solver_status": status,
        "termination_condition": termination,
        "has_incumbent": has_incumbent,
        "timed_out": timed_out,
        "incumbent_objective": incumbent,
        "objective_bound": objective_bound,
        "lower_bound": objective_bound,
        "upper_bound": incumbent,
    }


def _load_highs_incumbent(solver: Any) -> None:
    """Load a previously verified HiGHS incumbent into the Pyomo model."""

    if hasattr(solver, "load_vars"):
        solver.load_vars()
        return
    if hasattr(solver, "_load_vars"):
        solver._load_vars()
        return
    native = getattr(solver, "_last_results_object", None)
    loader = getattr(native, "solution_loader", None)
    if loader is not None and hasattr(loader, "load_solution"):
        loader.load_solution()
        return
    raise RuntimeError("HiGHS interface did not expose an incumbent loader")


def _effective_probabilities(
    bundle: ScenarioBundle,
    stress_mass_setting: Any,
) -> tuple[np.ndarray, float]:
    if stress_mass_setting is None:
        return np.asarray(bundle.probabilities, dtype=float).copy(), float(
            bundle.mixture_stress_mass
        )
    stress_mass = float(stress_mass_setting)
    if not 0.0 <= stress_mass < 1.0:
        raise ValueError("stress_mass must be in [0, 1)")
    nominal_count = int(bundle.nominal_component_count)
    stress_count = int(bundle.stress_component_count)
    if nominal_count + stress_count != bundle.reduced_count or nominal_count <= 0:
        if stress_mass != float(bundle.mixture_stress_mass):
            raise ValueError("stress_mass can only reweight a nominal/stress mixture bundle")
        return np.asarray(bundle.probabilities, dtype=float).copy(), stress_mass
    nominal = np.asarray(bundle.probabilities[:nominal_count], dtype=float)
    stress = np.asarray(bundle.probabilities[nominal_count:], dtype=float)
    probabilities = np.zeros(bundle.reduced_count, dtype=float)
    probabilities[:nominal_count] = (1.0 - stress_mass) * nominal / nominal.sum()
    if stress_count:
        if stress_mass > 0 and stress.sum() <= 0:
            raise ValueError("stress mixture has no positive stress probability to reweight")
        if stress.sum() > 0:
            probabilities[nominal_count:] = stress_mass * stress / stress.sum()
    elif stress_mass > 0:
        raise ValueError("positive stress_mass requires stress scenarios")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("effective scenario probabilities do not sum to one")
    return probabilities, stress_mass


def _weighted_cvar(values: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    order = np.argsort(values)
    values = values[order]
    probabilities = probabilities[order]
    cumulative = np.cumsum(probabilities)
    var = values[min(int(np.searchsorted(cumulative, alpha, side="left")), len(values) - 1)]
    return float(var + np.sum(probabilities * np.maximum(values - var, 0.0)) / (1.0 - alpha))


def _validate_solution(
    model: pyo.ConcreteModel,
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    policy: PolicySpec,
    demand_chunks: np.ndarray,
    home: np.ndarray,
    rigid_home: np.ndarray,
    non_migratable_home_totals: np.ndarray,
    migration_bandwidth_chunks: float | None,
    granularity: float,
    interval_hours: float,
    pue_slope_multiplier: float,
    tolerance: float,
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    w_count, z_count, horizon = bundle.rt_price.shape
    effective_physical_multiplier = np.asarray(
        effective["physical_multiplier"], dtype=float
    )
    effective_pv = np.asarray(effective["pv"], dtype=float)
    effective_da_price = np.asarray(effective["da_price"], dtype=float)
    effective_rt_buy_price = np.asarray(effective["rt_buy_price"], dtype=float)
    effective_rt_sell_price = np.asarray(effective["rt_sell_price"], dtype=float)
    effective_capacity = np.asarray(effective["capacity_work_units"], dtype=float)
    hardware_transfer_bounds = np.asarray(
        effective["hardware_transfer_bounds_work_units"], dtype=float
    )
    negative_price_floor = effective["negative_price_floor_usd_per_mwh"]
    workload_residuals: list[float] = []
    rigid_floor_violations: list[float] = []
    migration_eligibility_violations: list[float] = []
    migration_bandwidth_violations: list[float] = []
    if not policy.temporal_flexibility and not policy.spatial_flexibility:
        for w in range(w_count):
            for z in range(z_count):
                for t in range(horizon):
                    workload_residuals.append(
                        _safe_value(model.work[z, t, w] + model.short[z, t, w]) - home[w, z, t]
                    )
    elif policy.temporal_flexibility and not policy.spatial_flexibility:
        for w in range(w_count):
            for z in range(z_count):
                lhs = sum(_safe_value(model.work[z, t, w] + model.short[z, t, w]) for t in range(horizon))
                workload_residuals.append(lhs - float(home[w, z].sum()))
    elif not policy.temporal_flexibility:
        for w in range(w_count):
            for t in range(horizon):
                lhs = sum(_safe_value(model.work[z, t, w] + model.short[z, t, w]) for z in range(z_count))
                workload_residuals.append(lhs - float(demand_chunks[w, t]))
    else:
        for w in range(w_count):
            lhs = sum(
                _safe_value(model.work[z, t, w] + model.short[z, t, w])
                for z in range(z_count)
                for t in range(horizon)
            )
            workload_residuals.append(lhs - float(demand_chunks[w].sum()))

    for w in range(w_count):
        for z in range(z_count):
            zone_total = sum(
                _safe_value(model.work[z, t, w] + model.short[z, t, w])
                for t in range(horizon)
            )
            migration_eligibility_violations.append(
                max(float(non_migratable_home_totals[w, z]) - zone_total, 0.0)
            )
            for t in range(horizon):
                rigid_floor_violations.append(
                    max(
                        float(rigid_home[w, z, t])
                        - _safe_value(model.work[z, t, w] + model.short[z, t, w]),
                        0.0,
                    )
                )
        if migration_bandwidth_chunks is not None:
            for t in range(horizon):
                migrated = 0.5 * sum(
                    _safe_value(model.reassignment[z, t, w])
                    for z in range(z_count)
                )
                migration_bandwidth_violations.append(
                    max(migrated - migration_bandwidth_chunks, 0.0)
                )

    facility: list[float] = []
    battery: list[float] = []
    initial_soc: list[float] = []
    terminal_soc: list[float] = []
    settlement: list[float] = []
    settlement_cost: list[float] = []
    pv_balance: list[float] = []
    capacity_violations: list[float] = []
    hardware_transfer_violations: list[float] = []
    simultaneous_deviation: list[float] = []
    deviation_purchase_bound_violations: list[float] = []
    deviation_sale_bound_violations: list[float] = []
    day_ahead_bound_violations: list[float] = []
    integrality: list[float] = []
    terminal_soc_target = effective.get("terminal_soc_target_mwh")
    recourse_prefix = int(effective.get("recourse_nonanticipative_prefix_steps", 0))
    fixed_da = effective.get("fixed_day_ahead_schedule_mw")
    previous_grid = effective.get("previous_grid_import_mw")
    cumulative_release_chunks = np.asarray(
        effective["cumulative_release_chunks"], dtype=float
    )
    cumulative_due_chunks = np.asarray(
        effective["cumulative_due_chunks"], dtype=float
    )
    release_violations: list[float] = []
    deadline_violations: list[float] = []
    for w in range(w_count):
        if policy.temporal_flexibility:
            for t in range(horizon):
                completed = sum(
                    _safe_value(model.work[z, tau, w])
                    for z in range(z_count)
                    for tau in range(t + 1)
                )
                accounted = completed + sum(
                    _safe_value(model.short[z, tau, w])
                    for z in range(z_count)
                    for tau in range(t + 1)
                )
                release_violations.append(
                    max(completed - cumulative_release_chunks[w, t], 0.0)
                )
                deadline_violations.append(
                    max(cumulative_due_chunks[w, t] - accounted, 0.0)
                )
        for z in range(z_count):
            initial_soc.append(
                _safe_value(model.soc[z, 0, w])
                - float(np.asarray(effective["initial_soc_mwh"], dtype=float)[z])
            )
            for t in range(horizon):
                load = effective_physical_multiplier[w, z] * (
                    physical.idle_mw[z]
                    + pue_slope_multiplier
                    * physical.work_mw_per_unit[z]
                    * granularity
                    * _safe_value(model.work[z, t, w])
                )
                facility.append(
                    _safe_value(model.grid[z, t, w] + model.pv_used[z, t, w] + model.discharge[z, t, w])
                    - load
                    - _safe_value(model.charge[z, t, w])
                )
                pv_balance.append(
                    _safe_value(model.pv_used[z, t, w] + model.pv_curtail[z, t, w])
                    - float(effective_pv[w, z, t])
                )
                capacity_violations.append(
                    max(
                        granularity * _safe_value(model.work[z, t, w])
                        - float(effective_capacity[z]),
                        0.0,
                    )
                )
                hardware_transfer_violations.append(
                    max(
                        granularity * _safe_value(model.reassignment[z, t, w])
                        - float(hardware_transfer_bounds[z]),
                        0.0,
                    )
                )
                battery.append(
                    _safe_value(model.soc[z, t + 1, w] - model.soc[z, t, w])
                    - interval_hours
                    * (
                        _safe_value(model.charge[z, t, w]) * _safe_value(model.eta_charge)
                        - _safe_value(model.discharge[z, t, w]) / _safe_value(model.eta_discharge)
                    )
                )
                settlement.append(
                    _safe_value(model.dev_plus[z, t, w] - model.dev_minus[z, t, w])
                    - _safe_value(model.grid[z, t, w] - model.p_da[z, t, w])
                )
                simultaneous_deviation.append(
                    _safe_value(model.dev_plus[z, t, w]) * _safe_value(model.dev_minus[z, t, w])
                )
                deviation_bound = float(
                    physical.grid_import_limit_mw[z]
                    + physical.battery_power_mw[z]
                )
                deviation_purchase_bound_violations.append(
                    max(_safe_value(model.dev_plus[z, t, w]) - deviation_bound, 0.0)
                )
                deviation_sale_bound_violations.append(
                    max(_safe_value(model.dev_minus[z, t, w]) - deviation_bound, 0.0)
                )
                day_ahead_bound_violations.append(
                    max(_safe_value(model.p_da[z, t, w]) - deviation_bound, 0.0)
                )
                for variable in (model.work[z, t, w], model.short[z, t, w]):
                    value = _safe_value(variable)
                    integrality.append(value - round(value))
            if terminal_soc_target is not None:
                terminal_soc.append(
                    _safe_value(model.soc[z, horizon, w])
                    - float(np.asarray(terminal_soc_target, dtype=float)[z])
                )
        manual_market_cost = interval_hours * sum(
            _safe_value(model.p_da[z, t, w]) * float(effective_da_price[z, t])
            + _safe_value(model.dev_plus[z, t, w])
            * float(effective_rt_buy_price[w, z, t])
            - _safe_value(model.dev_minus[z, t, w])
            * float(effective_rt_sell_price[w, z, t])
            for z in range(z_count)
            for t in range(horizon)
        )
        settlement_cost.append(manual_market_cost - _safe_value(model.market_cost[w]))

    nonanticipativity = 0.0
    if w_count > 1:
        for z in range(z_count):
            for t in range(horizon):
                values = [_safe_value(model.p_da[z, t, w]) for w in range(w_count)]
                nonanticipativity = max(nonanticipativity, max(values) - min(values))
    recourse_nonanticipativity = 0.0
    if w_count > 1 and recourse_prefix > 0:
        for z in range(z_count):
            for t in range(min(recourse_prefix, horizon)):
                for variable in (
                    model.work,
                    model.short,
                    model.reassignment,
                    model.grid,
                    model.pv_used,
                    model.pv_curtail,
                    model.charge,
                    model.discharge,
                    model.dev_plus,
                    model.dev_minus,
                ):
                    values = [_safe_value(variable[z, t, w]) for w in range(w_count)]
                    recourse_nonanticipativity = max(
                        recourse_nonanticipativity, max(values) - min(values)
                    )
                soc_values = [
                    _safe_value(model.soc[z, t + 1, w]) for w in range(w_count)
                ]
                recourse_nonanticipativity = max(
                    recourse_nonanticipativity, max(soc_values) - min(soc_values)
                )
    fixed_da_residual = 0.0
    if fixed_da is not None:
        fixed_da_array = np.asarray(fixed_da, dtype=float)
        for w in range(w_count):
            for z in range(z_count):
                for t in range(horizon):
                    target = (
                        fixed_da_array[z, t]
                        if fixed_da_array.ndim == 2
                        else fixed_da_array[w, z, t]
                    )
                    fixed_da_residual = max(
                        fixed_da_residual,
                        abs(_safe_value(model.p_da[z, t, w]) - float(target)),
                    )
    ramp_boundary_violation = 0.0
    if previous_grid is not None and effective.get("ramp_limit_mw_per_step") is not None:
        previous_total = float(np.asarray(previous_grid, dtype=float).sum())
        ramp_limit = float(effective["ramp_limit_mw_per_step"])
        for w in range(w_count):
            first_total = sum(_safe_value(model.grid[z, 0, w]) for z in range(z_count))
            boundary_slack = _safe_value(model.ramp_slack[0, w])
            ramp_boundary_violation = max(
                ramp_boundary_violation,
                abs(first_total - previous_total) - ramp_limit - boundary_slack,
                0.0,
            )
    checks = {
        "workload_conservation_max_abs": float(np.max(np.abs(workload_residuals), initial=0.0)),
        "assignment_migration_conservation_max_abs": float(np.max(np.abs(workload_residuals), initial=0.0)),
        "workload_release_max_violation_chunks": float(
            np.max(release_violations, initial=0.0)
        ),
        "deadline_accounting_max_violation_chunks": float(
            np.max(deadline_violations, initial=0.0)
        ),
        "rigid_share_floor_max_violation_chunks": float(
            np.max(rigid_floor_violations, initial=0.0)
        ),
        "migration_eligibility_max_violation_chunks": float(
            np.max(migration_eligibility_violations, initial=0.0)
        ),
        "migration_bandwidth_max_violation_chunks": float(
            np.max(migration_bandwidth_violations, initial=0.0)
        ),
        "facility_energy_balance_max_abs_mw": float(np.max(np.abs(facility), initial=0.0)),
        "pv_balance_max_abs_mw": float(np.max(np.abs(pv_balance), initial=0.0)),
        "capacity_max_violation_work_units": float(
            np.max(capacity_violations, initial=0.0)
        ),
        "hardware_transfer_max_violation_work_units": float(
            np.max(hardware_transfer_violations, initial=0.0)
        ),
        "battery_balance_max_abs_mwh": float(np.max(np.abs(battery), initial=0.0)),
        "battery_initial_soc_max_abs_mwh": float(
            np.max(np.abs(initial_soc), initial=0.0)
        ),
        "battery_terminal_soc_max_abs_mwh": float(np.max(np.abs(terminal_soc), initial=0.0)),
        "battery_terminal_soc_status": (
            "enforced" if terminal_soc_target is not None else "rolling_horizon_not_enforced"
        ),
        "two_settlement_identity_max_abs_mw": float(np.max(np.abs(settlement), initial=0.0)),
        "two_settlement_cost_max_abs_usd": float(np.max(np.abs(settlement_cost), initial=0.0)),
        "simultaneous_up_down_deviation_max_product_mw2": float(
            np.max(np.abs(simultaneous_deviation), initial=0.0)
        ),
        "simultaneous_up_down_deviation_status": (
            "diagnostic_only_price_order_makes_continuous_formulation_exact"
        ),
        "rt_deviation_purchase_bound_max_violation_mw": float(
            np.max(deviation_purchase_bound_violations, initial=0.0)
        ),
        "rt_deviation_sale_bound_max_violation_mw": float(
            np.max(deviation_sale_bound_violations, initial=0.0)
        ),
        "day_ahead_schedule_bound_max_violation_mw": float(
            np.max(day_ahead_bound_violations, initial=0.0)
        ),
        "deviation_sell_above_buy_max_violation_usd_per_mwh": float(
            np.max(
                np.maximum(effective_rt_sell_price - effective_rt_buy_price, 0.0),
                initial=0.0,
            )
        ),
        "checkpoint_integrality_max_abs": float(np.max(np.abs(integrality), initial=0.0)),
        "nonanticipativity_max_abs_mw": float(nonanticipativity),
        "nonanticipativity_status": "oracle_exempt" if policy.oracle else "enforced",
        "recourse_nonanticipativity_prefix_steps": recourse_prefix,
        "recourse_nonanticipativity_max_abs": float(recourse_nonanticipativity),
        "fixed_day_ahead_schedule_max_abs_mw": float(fixed_da_residual),
        "previous_grid_ramp_boundary_max_violation_mw": float(
            ramp_boundary_violation
        ),
        "original_negative_rt_price_count": int((bundle.rt_price < 0).sum()),
        "original_negative_da_price_count": int((bundle.da_price < 0).sum()),
        "negative_rt_price_count": int((effective_rt_buy_price < 0).sum()),
        "negative_da_price_count": int((effective_da_price < 0).sum()),
        "negative_prices_retained_unclipped": negative_price_floor is None,
    }
    required = [
        checks["workload_conservation_max_abs"],
        checks["workload_release_max_violation_chunks"],
        checks["deadline_accounting_max_violation_chunks"],
        checks["rigid_share_floor_max_violation_chunks"],
        checks["migration_eligibility_max_violation_chunks"],
        checks["migration_bandwidth_max_violation_chunks"],
        checks["facility_energy_balance_max_abs_mw"],
        checks["pv_balance_max_abs_mw"],
        checks["capacity_max_violation_work_units"],
        checks["hardware_transfer_max_violation_work_units"],
        checks["battery_balance_max_abs_mwh"],
        checks["battery_initial_soc_max_abs_mwh"],
        checks["battery_terminal_soc_max_abs_mwh"],
        checks["two_settlement_identity_max_abs_mw"],
        checks["two_settlement_cost_max_abs_usd"],
        checks["rt_deviation_purchase_bound_max_violation_mw"],
        checks["rt_deviation_sale_bound_max_violation_mw"],
        checks["day_ahead_schedule_bound_max_violation_mw"],
        checks["deviation_sell_above_buy_max_violation_usd_per_mwh"],
        checks["checkpoint_integrality_max_abs"],
        checks["recourse_nonanticipativity_max_abs"],
        checks["fixed_day_ahead_schedule_max_abs_mw"],
        checks["previous_grid_ramp_boundary_max_violation_mw"],
    ]
    if not policy.oracle:
        required.append(checks["nonanticipativity_max_abs_mw"])
    checks["passed"] = bool(max(required, default=0.0) <= tolerance)
    return checks


def solve_policy(
    bundle: ScenarioBundle,
    physical: FrozenPhysicalArtifact,
    policy_id: str,
    config: Mapping[str, Any] | None = None,
    *,
    xcel_tariff: XcelTariffDefinition | None = None,
) -> dict[str, Any]:
    """Solve one two-stage, four-zone counterfactual policy with HiGHS."""

    bundle.validate()
    settings = dict(config or {})
    if policy_id not in POLICIES:
        raise ValueError(f"unknown policy {policy_id}; expected one of {tuple(POLICIES)}")
    policy = POLICIES[policy_id]
    if policy.stress_aware and bundle.kappa <= 0:
        raise ValueError("B7 requires a kappa=0.3 or kappa=0.6 stress bundle")

    w_count, z_count, horizon = bundle.rt_price.shape
    scenario_probabilities, effective_stress_mass = _effective_probabilities(
        bundle, settings.get("stress_mass")
    )
    physical_multiplier_scale = float(settings.get("physical_multiplier_scale", 1.0))
    if not np.isfinite(physical_multiplier_scale) or physical_multiplier_scale <= 0:
        raise ValueError("physical_multiplier_scale must be finite and positive")
    effective_physical_multiplier = (
        np.asarray(bundle.physical_multiplier, dtype=float) * physical_multiplier_scale
    )
    if not np.isfinite(effective_physical_multiplier).all() or (
        effective_physical_multiplier <= 0
    ).any():
        raise ValueError("effective physical multipliers must be finite and positive")
    granularity = float(settings.get("checkpoint_granularity_work_units", 1.0))
    if granularity <= 0:
        raise ValueError("checkpoint granularity must be positive")
    exact_chunks = bundle.workload / granularity
    demand_chunks = np.rint(exact_chunks).astype(int)
    rounding_error = float(np.max(np.abs(exact_chunks - demand_chunks), initial=0.0))
    if rounding_error > float(settings.get("checkpoint_rounding_tolerance_chunks", 0.5000001)):
        raise ValueError(
            f"workload is not representable at checkpoint granularity {granularity}; max chunk error={rounding_error}"
        )
    home = _home_chunks(demand_chunks, z_count)
    interval_hours = float(settings.get("interval_hours", 1.0))
    deadline_steps = max(0, int(settings.get("deadline_steps", horizon - 1)))

    def cumulative_chunk_matrix(value: Any, *, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape == (horizon,):
            array = np.broadcast_to(array, (w_count, horizon)).copy()
        if array.shape != (w_count, horizon) or not np.isfinite(array).all():
            raise ValueError(
                f"{name} must have shape ({horizon},) or ({w_count}, {horizon})"
            )
        exact = array / granularity
        chunks = np.rint(exact).astype(int)
        if np.max(np.abs(exact - chunks), initial=0.0) > float(
            settings.get("checkpoint_rounding_tolerance_chunks", 0.5000001)
        ):
            raise ValueError(f"{name} is not representable at checkpoint granularity")
        if (chunks < 0).any() or (np.diff(chunks, axis=1) < 0).any():
            raise ValueError(f"{name} must be nonnegative and cumulative nondecreasing")
        return chunks

    default_cumulative_release_chunks = np.cumsum(demand_chunks, axis=1)
    cumulative_release_chunks = (
        default_cumulative_release_chunks
        if settings.get("rolling_cumulative_release_work_units") is None
        else cumulative_chunk_matrix(
            settings["rolling_cumulative_release_work_units"],
            name="rolling_cumulative_release_work_units",
        )
    )
    default_cumulative_due_chunks = np.zeros_like(demand_chunks)
    for period in range(horizon):
        due_period = period - deadline_steps
        if due_period >= 0:
            default_cumulative_due_chunks[:, period] = demand_chunks[
                :, : due_period + 1
            ].sum(axis=1)
    cumulative_due_chunks = (
        default_cumulative_due_chunks
        if settings.get("rolling_cumulative_due_work_units") is None
        else cumulative_chunk_matrix(
            settings["rolling_cumulative_due_work_units"],
            name="rolling_cumulative_due_work_units",
        )
    )
    demand_totals = demand_chunks.sum(axis=1)
    if not np.array_equal(cumulative_release_chunks[:, -1], demand_totals):
        raise ValueError("cumulative release must end at total workload")
    if (cumulative_due_chunks > cumulative_release_chunks).any() or (
        cumulative_due_chunks[:, -1] > demand_totals
    ).any():
        raise ValueError("cumulative due workload cannot exceed released workload")
    alpha = float(settings.get("cvar_alpha", 0.95))
    risk_weight = float(settings.get("risk_weight", 0.2 if policy.cvar else 0.0))
    flexible_share = float(settings.get("flexible_share", 1.0))
    migration_eligible_share = float(
        settings.get("migration_eligible", settings.get("migration_eligible_share", 1.0))
    )
    migration_bandwidth = settings.get(
        "migration_bandwidth_base_work_units_per_interval",
        settings.get("migration_bandwidth_work_units_per_step"),
    )
    migration_bandwidth_multiplier = float(
        settings.get("migration_bandwidth_multiplier", 1.0)
    )
    transfer_energy_mwh_per_work_unit = float(
        settings.get("transfer_energy_mwh_per_work_unit", 0.0)
    )
    checkpoint_overhead_fraction = float(
        settings.get("checkpoint_overhead_fraction", 0.0)
    )
    migration_energy_price = float(
        settings.get(
            "migration_energy_price_usd_per_mwh",
            max(float(np.mean(bundle.da_price)), 0.0),
        )
    )
    pue_slope_multiplier = float(settings.get("pue_slope_multiplier", 1.0))
    rt_sell_haircut_fraction = float(
        settings.get(
            "rt_sell_back_haircut",
            settings.get("rt_sell_haircut_fraction", 0.0),
        )
    )
    effective_rt_sell_price = np.asarray(bundle.rt_price, dtype=float) - (
        rt_sell_haircut_fraction * np.abs(np.asarray(bundle.rt_price, dtype=float))
    )
    sell_above_buy_max = float(
        np.max(effective_rt_sell_price - np.asarray(bundle.rt_price, dtype=float))
    )
    if sell_above_buy_max > 1e-12:
        raise ValueError(
            "rt sell price cannot exceed rt buy price for the exact continuous "
            "deviation formulation"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("cvar_alpha must be strictly between zero and one")
    if risk_weight < 0:
        raise ValueError("risk_weight must be nonnegative")
    if not 0.0 <= flexible_share <= 1.0:
        raise ValueError("flexible_share must be between zero and one")
    if not 0.0 <= migration_eligible_share <= 1.0:
        raise ValueError("migration_eligible_share must be between zero and one")
    if migration_bandwidth is not None and float(migration_bandwidth) < 0:
        raise ValueError("migration bandwidth must be nonnegative")
    if migration_bandwidth_multiplier < 0:
        raise ValueError("migration bandwidth multiplier must be nonnegative")
    if min(
        transfer_energy_mwh_per_work_unit,
        checkpoint_overhead_fraction,
        migration_energy_price,
        pue_slope_multiplier,
        rt_sell_haircut_fraction,
    ) < 0:
        raise ValueError("sensitivity coefficients must be nonnegative")
    if rt_sell_haircut_fraction > 1:
        raise ValueError("rt_sell_haircut_fraction cannot exceed one")
    for key in ("workload_scale", "pv_capacity_multiplier", "rt_volatility_scale"):
        if key in settings and (
            not np.isfinite(float(settings[key])) or float(settings[key]) < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if "workload_scale" in settings and float(settings["workload_scale"]) <= 0:
        raise ValueError("workload_scale must be positive")
    if "battery_duration_hours" in settings:
        battery_duration = float(settings["battery_duration_hours"])
        if not np.isfinite(battery_duration) or battery_duration < 0:
            raise ValueError("battery_duration_hours must be finite and nonnegative")
        if not np.allclose(
            physical.battery_energy_mwh,
            physical.battery_power_mw * battery_duration,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError(
                "battery_duration_hours metadata disagrees with the transformed physical artifact"
            )
    negative_price_floor = settings.get("negative_price_floor_usd_per_mwh")
    if negative_price_floor is not None:
        negative_price_floor = float(negative_price_floor)
        if not np.isfinite(negative_price_floor):
            raise ValueError("negative_price_floor_usd_per_mwh must be finite or null")
        if (
            np.min(bundle.da_price) < negative_price_floor - 1e-9
            or np.min(bundle.rt_price) < negative_price_floor - 1e-9
        ):
            raise ValueError(
                "negative-price floor metadata was not consumed by the transformed scenario bundle"
            )
    if policy_id == "B5":
        risk_weight = 0.0
    rigid_home = np.floor((1.0 - flexible_share) * home + 1e-12).astype(int)
    non_migratable_home_totals = np.ceil(
        (1.0 - migration_eligible_share) * home.sum(axis=2) - 1e-12
    ).astype(int)
    migration_bandwidth_chunks = (
        None
        if migration_bandwidth is None
        else float(migration_bandwidth)
        * migration_bandwidth_multiplier
        / granularity
    )
    hardware_transfer_multiplier = float(
        settings.get("hardware_transfer_multiplier", 1.0)
    )
    if not np.isfinite(hardware_transfer_multiplier) or hardware_transfer_multiplier < 0:
        raise ValueError("hardware_transfer_multiplier must be finite and nonnegative")
    hardware_transfer_bounds_work_units = hardware_transfer_multiplier * np.maximum(
        np.max(home, axis=(0, 2)).astype(float) * granularity,
        np.asarray(physical.capacity_work_units, dtype=float),
    )
    eta_charge = float(settings.get("battery_charge_efficiency", 0.95))
    eta_discharge = float(settings.get("battery_discharge_efficiency", 0.95))
    sla_penalty = float(settings.get("sla_penalty_usd_per_work_unit", 10000.0))
    migration_penalty = float(settings.get("migration_penalty_usd_per_work_unit", 0.05))
    battery_penalty = float(settings.get("battery_degradation_usd_per_mwh", 1.0))
    curtailment_penalty = float(settings.get("pv_curtailment_usd_per_mwh", 0.01))
    slack_penalty = float(settings.get("constraint_slack_penalty_usd", 100000.0))
    ramp_limit = settings.get("ramp_limit_mw_per_step")
    concentration_limit = settings.get("concentration_share_limit")
    default_initial_soc = 0.5 * np.asarray(physical.battery_energy_mwh, dtype=float)

    def zone_vector(value: Any, *, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != (z_count,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite vector with shape ({z_count},)")
        return array

    initial_soc_mwh = zone_vector(
        settings.get("initial_soc_mwh", default_initial_soc),
        name="initial_soc_mwh",
    )
    if (initial_soc_mwh < 0).any() or (
        initial_soc_mwh > np.asarray(physical.battery_energy_mwh) + 1e-9
    ).any():
        raise ValueError("initial_soc_mwh must lie within battery energy bounds")
    enforce_terminal_soc = bool(settings.get("enforce_terminal_soc", True))
    terminal_setting = settings.get("terminal_soc_target_mwh", initial_soc_mwh)
    terminal_soc_target_mwh = (
        None
        if not enforce_terminal_soc or terminal_setting is None
        else zone_vector(terminal_setting, name="terminal_soc_target_mwh")
    )
    if terminal_soc_target_mwh is not None and (
        (terminal_soc_target_mwh < 0).any()
        or (
            terminal_soc_target_mwh
            > np.asarray(physical.battery_energy_mwh) + 1e-9
        ).any()
    ):
        raise ValueError("terminal_soc_target_mwh must lie within battery energy bounds")
    previous_grid_import_mw = (
        None
        if settings.get("previous_grid_import_mw") is None
        else zone_vector(
            settings["previous_grid_import_mw"], name="previous_grid_import_mw"
        )
    )
    if previous_grid_import_mw is not None and (previous_grid_import_mw < 0).any():
        raise ValueError("previous_grid_import_mw must be nonnegative")
    recourse_nonanticipative_prefix_steps = int(
        settings.get("recourse_nonanticipative_prefix_steps", 0)
    )
    if not 0 <= recourse_nonanticipative_prefix_steps <= horizon:
        raise ValueError(
            "recourse_nonanticipative_prefix_steps must lie within the solve horizon"
        )
    fixed_day_ahead_schedule_mw = settings.get("fixed_day_ahead_schedule_mw")
    if fixed_day_ahead_schedule_mw is not None:
        fixed_day_ahead_schedule_mw = np.asarray(
            fixed_day_ahead_schedule_mw, dtype=float
        )
        if fixed_day_ahead_schedule_mw.shape not in {
            (z_count, horizon),
            (w_count, z_count, horizon),
        }:
            raise ValueError(
                "fixed_day_ahead_schedule_mw must have shape (zones, horizon) "
                "or (scenarios, zones, horizon)"
            )
        if not np.isfinite(fixed_day_ahead_schedule_mw).all() or (
            fixed_day_ahead_schedule_mw < 0
        ).any():
            raise ValueError("fixed_day_ahead_schedule_mw must be finite and nonnegative")

    model = pyo.ConcreteModel(name=f"dcvpp_{policy_id}_{bundle.operating_date}_{bundle.regime}")
    model.Z = pyo.RangeSet(0, z_count - 1)
    model.T = pyo.RangeSet(0, horizon - 1)
    model.TSOC = pyo.RangeSet(0, horizon)
    model.W = pyo.RangeSet(0, w_count - 1)
    model.eta_charge = pyo.Param(initialize=eta_charge)
    model.eta_discharge = pyo.Param(initialize=eta_discharge)
    model.work = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeIntegers)
    model.short = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeIntegers)
    model.reassignment = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.grid = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.pv_used = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.pv_curtail = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.Z, model.TSOC, model.W, domain=pyo.NonNegativeReals)
    model.battery_mode = pyo.Var(model.Z, model.T, model.W, domain=pyo.Binary)
    model.p_da = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.dev_plus = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.dev_minus = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)
    model.ramp_slack = pyo.Var(model.T, model.W, domain=pyo.NonNegativeReals)
    model.concentration_slack = pyo.Var(model.Z, model.T, model.W, domain=pyo.NonNegativeReals)

    def capacity_rule(m: pyo.ConcreteModel, z: int, t: int, w: int) -> Any:
        return granularity * m.work[z, t, w] <= float(physical.capacity_work_units[z])

    model.capacity = pyo.Constraint(model.Z, model.T, model.W, rule=capacity_rule)

    if not policy.temporal_flexibility and not policy.spatial_flexibility:
        model.workload_conservation = pyo.Constraint(
            model.Z,
            model.T,
            model.W,
            rule=lambda m, z, t, w: m.work[z, t, w] + m.short[z, t, w] == int(home[w, z, t]),
        )
    elif policy.temporal_flexibility and not policy.spatial_flexibility:
        model.workload_conservation = pyo.Constraint(
            model.Z,
            model.W,
            rule=lambda m, z, w: sum(m.work[z, t, w] + m.short[z, t, w] for t in m.T)
            == int(home[w, z].sum()),
        )
    elif not policy.temporal_flexibility:
        model.workload_conservation = pyo.Constraint(
            model.T,
            model.W,
            rule=lambda m, t, w: sum(m.work[z, t, w] + m.short[z, t, w] for z in m.Z)
            == int(demand_chunks[w, t]),
        )
    else:
        model.workload_conservation = pyo.Constraint(
            model.W,
            rule=lambda m, w: sum(m.work[z, t, w] + m.short[z, t, w] for z in m.Z for t in m.T)
            == int(demand_chunks[w].sum()),
        )

    if flexible_share < 1.0:
        model.rigid_work_floor = pyo.Constraint(
            model.Z,
            model.T,
            model.W,
            rule=lambda m, z, t, w: m.work[z, t, w] + m.short[z, t, w]
            >= int(rigid_home[w, z, t]),
        )
    if migration_eligible_share < 1.0:
        model.non_migratable_home_floor = pyo.Constraint(
            model.Z,
            model.W,
            rule=lambda m, z, w: sum(
                m.work[z, t, w] + m.short[z, t, w] for t in m.T
            )
            >= int(non_migratable_home_totals[w, z]),
        )

    if policy.temporal_flexibility:
        def release_rule(m: pyo.ConcreteModel, t: int, w: int) -> Any:
            return sum(
                m.work[z, tau, w] for z in m.Z for tau in range(t + 1)
            ) <= int(cumulative_release_chunks[w, t])

        def deadline_rule(m: pyo.ConcreteModel, t: int, w: int) -> Any:
            due_chunks = int(cumulative_due_chunks[w, t])
            if due_chunks <= 0:
                return pyo.Constraint.Skip
            return sum(
                m.work[z, tau, w] + m.short[z, tau, w] for z in m.Z for tau in range(t + 1)
            ) >= due_chunks

        model.release = pyo.Constraint(model.T, model.W, rule=release_rule)
        model.deadline = pyo.Constraint(model.T, model.W, rule=deadline_rule)

    model.reassignment_up = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        # ``short`` is unserved work retained at its home site, not a spatial
        # transfer.  Comparing executed work alone with the home allocation
        # incorrectly classifies capacity-driven SLA shortfall as migration
        # and can make an otherwise feasible cell violate the migration-bandwidth
        # limit.  Accounted home demand is execution plus explicit shortfall;
        # only a change in that site assignment is reassignment.
        rule=lambda m, z, t, w: m.reassignment[z, t, w]
        >= m.work[z, t, w] + m.short[z, t, w] - int(home[w, z, t]),
    )
    model.reassignment_down = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.reassignment[z, t, w]
        >= int(home[w, z, t]) - m.work[z, t, w] - m.short[z, t, w],
    )
    if migration_bandwidth is not None:
        model.migration_bandwidth = pyo.Constraint(
            model.T,
            model.W,
            rule=lambda m, t, w: 0.5
            * sum(m.reassignment[z, t, w] for z in m.Z)
            <= migration_bandwidth_chunks,
        )
    model.hardware_transfer_bound = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: granularity * m.reassignment[z, t, w]
        <= float(hardware_transfer_bounds_work_units[z]),
    )

    def facility_rule(m: pyo.ConcreteModel, z: int, t: int, w: int) -> Any:
        load = float(effective_physical_multiplier[w, z]) * (
            float(physical.idle_mw[z])
            + pue_slope_multiplier
            * float(physical.work_mw_per_unit[z])
            * granularity
            * m.work[z, t, w]
        )
        return m.grid[z, t, w] + m.pv_used[z, t, w] + m.discharge[z, t, w] == load + m.charge[z, t, w]

    model.facility_balance = pyo.Constraint(model.Z, model.T, model.W, rule=facility_rule)
    model.pv_balance = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.pv_used[z, t, w] + m.pv_curtail[z, t, w]
        == float(bundle.pv[w, z, t]),
    )
    model.grid_limit = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.grid[z, t, w] <= float(physical.grid_import_limit_mw[z]),
    )

    def soc_rule(m: pyo.ConcreteModel, z: int, t: int, w: int) -> Any:
        return m.soc[z, t + 1, w] == m.soc[z, t, w] + interval_hours * (
            eta_charge * m.charge[z, t, w] - m.discharge[z, t, w] / eta_discharge
        )

    model.soc_balance = pyo.Constraint(model.Z, model.T, model.W, rule=soc_rule)
    model.soc_limit = pyo.Constraint(
        model.Z,
        model.TSOC,
        model.W,
        rule=lambda m, z, t, w: m.soc[z, t, w] <= float(physical.battery_energy_mwh[z]),
    )
    model.soc_initial = pyo.Constraint(
        model.Z,
        model.W,
        rule=lambda m, z, w: m.soc[z, 0, w] == float(initial_soc_mwh[z]),
    )
    if terminal_soc_target_mwh is not None:
        model.soc_terminal = pyo.Constraint(
            model.Z,
            model.W,
            rule=lambda m, z, w: m.soc[z, horizon, w]
            == float(terminal_soc_target_mwh[z]),
        )
    model.charge_limit = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.charge[z, t, w]
        <= float(physical.battery_power_mw[z]) * m.battery_mode[z, t, w],
    )
    model.discharge_limit = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.discharge[z, t, w]
        <= float(physical.battery_power_mw[z]) * (1 - m.battery_mode[z, t, w]),
    )
    if not policy.storage:
        for z in range(z_count):
            for t in range(horizon):
                for w in range(w_count):
                    model.charge[z, t, w].fix(0.0)
                    model.discharge[z, t, w].fix(0.0)

    model.settlement_identity = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.dev_plus[z, t, w] - m.dev_minus[z, t, w]
        == m.grid[z, t, w] - m.p_da[z, t, w],
    )
    # This is an exact continuous formulation because rt_sell <= rt_buy.  If
    # both deviations are positive, subtracting their common minimum preserves
    # the settlement identity and changes the minimization objective by
    # -(rt_buy - rt_sell) * minimum, which never makes it worse.  Thus an
    # optimal canonical solution always exists with at most one positive side;
    # the former deviation-mode binary restricted only equivalent optima.
    deviation_bound = np.asarray(physical.grid_import_limit_mw) + np.asarray(physical.battery_power_mw)
    model.dev_plus_bound = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.dev_plus[z, t, w]
        <= float(deviation_bound[z]),
    )
    model.dev_minus_bound = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.dev_minus[z, t, w]
        <= float(deviation_bound[z]),
    )
    model.pda_bound = pyo.Constraint(
        model.Z,
        model.T,
        model.W,
        rule=lambda m, z, t, w: m.p_da[z, t, w] <= float(deviation_bound[z]),
    )
    if not policy.oracle:
        model.nonanticipativity = pyo.Constraint(
            model.Z,
            model.T,
            pyo.RangeSet(1, w_count - 1),
            rule=lambda m, z, t, w: m.p_da[z, t, w] == m.p_da[z, t, 0],
        )

    if fixed_day_ahead_schedule_mw is not None:
        for z in range(z_count):
            for t in range(horizon):
                for w in range(w_count):
                    fixed_value = (
                        fixed_day_ahead_schedule_mw[z, t]
                        if fixed_day_ahead_schedule_mw.ndim == 2
                        else fixed_day_ahead_schedule_mw[w, z, t]
                    )
                    if fixed_value > deviation_bound[z] + 1e-9:
                        raise ValueError(
                            "fixed_day_ahead_schedule_mw exceeds the admissible schedule bound"
                        )
                    model.p_da[z, t, w].fix(float(fixed_value))
    elif not policy.stochastic:
        expected_work = np.tensordot(scenario_probabilities, bundle.workload, axes=(0, 0))
        expected_pv = np.tensordot(scenario_probabilities, bundle.pv, axes=(0, 0))
        for z in range(z_count):
            for t in range(horizon):
                point_load = physical.idle_mw[z] + physical.work_mw_per_unit[z] * expected_work[t] / z_count
                point_import = float(np.clip(point_load - expected_pv[z, t], 0.0, deviation_bound[z]))
                for w in range(w_count):
                    model.p_da[z, t, w].fix(point_import)

    model.recourse_nonanticipativity = pyo.ConstraintList()
    if recourse_nonanticipative_prefix_steps > 0 and w_count > 1:
        recourse_variables = (
            model.work,
            model.short,
            model.reassignment,
            model.grid,
            model.pv_used,
            model.pv_curtail,
            model.charge,
            model.discharge,
            model.battery_mode,
            model.dev_plus,
            model.dev_minus,
        )
        for t in range(recourse_nonanticipative_prefix_steps):
            for z in range(z_count):
                for w in range(1, w_count):
                    for variable in recourse_variables:
                        model.recourse_nonanticipativity.add(
                            variable[z, t, w] == variable[z, t, 0]
                        )
                    model.recourse_nonanticipativity.add(
                        model.soc[z, t + 1, w] == model.soc[z, t + 1, 0]
                    )

    if ramp_limit is None:
        for t in range(horizon):
            for w in range(w_count):
                model.ramp_slack[t, w].fix(0.0)
    else:
        model.ramp_up = pyo.Constraint(
            pyo.RangeSet(1, horizon - 1),
            model.W,
            rule=lambda m, t, w: sum(m.grid[z, t, w] - m.grid[z, t - 1, w] for z in m.Z)
            <= float(ramp_limit) + m.ramp_slack[t, w],
        )
        model.ramp_down = pyo.Constraint(
            pyo.RangeSet(1, horizon - 1),
            model.W,
            rule=lambda m, t, w: sum(m.grid[z, t - 1, w] - m.grid[z, t, w] for z in m.Z)
            <= float(ramp_limit) + m.ramp_slack[t, w],
        )
        if previous_grid_import_mw is None:
            for w in range(w_count):
                model.ramp_slack[0, w].fix(0.0)
        else:
            previous_grid_total_mw = float(previous_grid_import_mw.sum())
            model.ramp_boundary_up = pyo.Constraint(
                model.W,
                rule=lambda m, w: sum(m.grid[z, 0, w] for z in m.Z)
                - previous_grid_total_mw
                <= float(ramp_limit) + m.ramp_slack[0, w],
            )
            model.ramp_boundary_down = pyo.Constraint(
                model.W,
                rule=lambda m, w: previous_grid_total_mw
                - sum(m.grid[z, 0, w] for z in m.Z)
                <= float(ramp_limit) + m.ramp_slack[0, w],
            )

    if concentration_limit is None:
        for z in range(z_count):
            for t in range(horizon):
                for w in range(w_count):
                    model.concentration_slack[z, t, w].fix(0.0)
    else:
        model.concentration = pyo.Constraint(
            model.Z,
            model.T,
            model.W,
            rule=lambda m, z, t, w: m.grid[z, t, w]
            <= float(concentration_limit) * sum(m.grid[j, t, w] for j in m.Z)
            + m.concentration_slack[z, t, w],
        )

    def market_cost(m: pyo.ConcreteModel, w: int) -> Any:
        return interval_hours * sum(
            m.p_da[z, t, w] * float(bundle.da_price[z, t])
            + m.dev_plus[z, t, w] * float(bundle.rt_price[w, z, t])
            - m.dev_minus[z, t, w] * float(effective_rt_sell_price[w, z, t])
            for z in m.Z
            for t in m.T
        )

    def operating_cost(m: pyo.ConcreteModel, w: int) -> Any:
        migrated_work = 0.5 * granularity * sum(
            m.reassignment[z, t, w] for z in m.Z for t in m.T
        )
        return (
            sla_penalty * granularity * sum(m.short[z, t, w] for z in m.Z for t in m.T)
            + migration_penalty * (1.0 + checkpoint_overhead_fraction) * migrated_work
            + migration_energy_price
            * transfer_energy_mwh_per_work_unit
            * migrated_work
            + battery_penalty
            * interval_hours
            * sum(m.charge[z, t, w] + m.discharge[z, t, w] for z in m.Z for t in m.T)
            + curtailment_penalty * interval_hours * sum(m.pv_curtail[z, t, w] for z in m.Z for t in m.T)
            + slack_penalty
            * (
                sum(m.ramp_slack[t, w] for t in m.T)
                + sum(m.concentration_slack[z, t, w] for z in m.Z for t in m.T)
            )
        )

    model.market_cost = pyo.Expression(model.W, rule=market_cost)
    model.operating_cost = pyo.Expression(model.W, rule=operating_cost)
    model.scenario_cost = pyo.Expression(model.W, rule=lambda m, w: m.market_cost[w] + m.operating_cost[w])
    model.eta = pyo.Var(domain=pyo.Reals)
    model.excess = pyo.Var(model.W, domain=pyo.NonNegativeReals)
    if policy.cvar:
        model.tail = pyo.Constraint(model.W, rule=lambda m, w: m.excess[w] >= m.scenario_cost[w] - m.eta)
        cvar_expression = model.eta + sum(
            float(scenario_probabilities[w]) * model.excess[w] / (1.0 - alpha)
            for w in range(w_count)
        )
    else:
        model.eta.fix(0.0)
        for w in range(w_count):
            model.excess[w].fix(0.0)
        cvar_expression = 0.0
    expected_expression = sum(
        float(scenario_probabilities[w]) * model.scenario_cost[w]
        for w in range(w_count)
    )
    model.objective = pyo.Objective(expr=expected_expression + risk_weight * cvar_expression, sense=pyo.minimize)

    solver_name = str(settings.get("solver", "appsi_highs"))
    if solver_name not in {"highs", "appsi_highs"}:
        raise ValueError("the research branch is restricted to HiGHS-compatible solvers")
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise OptimizationFailure(f"solver {solver_name} is unavailable", {"solver": solver_name})
    solver.options["time_limit"] = float(settings.get("time_limit_seconds", 120.0))
    solver.options["mip_rel_gap"] = float(settings.get("mip_relative_gap", 1e-4))
    solver_threads = int(settings.get("solver_threads", 1))
    if solver_threads < 1:
        raise ValueError("solver_threads must be a positive integer")
    solver.options["threads"] = solver_threads
    mip_start = _seed_all_short_mip_start(
        model,
        bundle=bundle,
        physical=physical,
        policy=policy,
        home=home,
        effective_physical_multiplier=effective_physical_multiplier,
        initial_soc_mwh=initial_soc_mwh,
        terminal_soc_target_mwh=terminal_soc_target_mwh,
        interval_hours=interval_hours,
        eta_charge=eta_charge,
        eta_discharge=eta_discharge,
        pue_slope_multiplier=pue_slope_multiplier,
        granularity=granularity,
        previous_grid_import_mw=previous_grid_import_mw,
        ramp_limit=None if ramp_limit is None else float(ramp_limit),
        concentration_limit=(
            None if concentration_limit is None else float(concentration_limit)
        ),
        tolerance=float(settings.get("mip_start_feasibility_tolerance", 1e-7)),
    )
    warmstart = bool(solver_name == "appsi_highs" and mip_start.get("feasible"))
    mip_start["solver_backend_supports_warmstart"] = solver_name == "appsi_highs"
    mip_start["submitted_to_solver"] = warmstart
    solve_started = perf_counter()
    try:
        solve_kwargs = {
            "tee": bool(settings.get("tee", False)),
            "load_solutions": False,
        }
        if solver_name == "appsi_highs":
            solve_kwargs["warmstart"] = warmstart
        solve_result = solver.solve(model, **solve_kwargs)
    except Exception as exc:
        raise OptimizationFailure(
            f"{policy_id} solve raised {type(exc).__name__}: {exc}",
            {
                "policy_id": policy_id,
                "operating_date": bundle.operating_date,
                "regime": bundle.regime,
                "mip_start": mip_start,
            },
        ) from exc
    solve_time_seconds = perf_counter() - solve_started
    solver_outcome = _highs_solve_outcome(solver, solve_result)
    termination = str(solver_outcome["termination_condition"])
    if not solver_outcome["has_incumbent"]:
        raise OptimizationFailure(
            f"{policy_id} solve produced {solver_outcome['classification']}: {termination}",
            {
                "policy_id": policy_id,
                "operating_date": bundle.operating_date,
                "regime": bundle.regime,
                **solver_outcome,
                "mip_start": mip_start,
            },
        )
    try:
        _load_highs_incumbent(solver)
    except Exception as exc:
        raise OptimizationFailure(
            f"{policy_id} incumbent load raised {type(exc).__name__}: {exc}",
            {
                "policy_id": policy_id,
                "operating_date": bundle.operating_date,
                "regime": bundle.regime,
                **solver_outcome,
                "mip_start": mip_start,
            },
        ) from exc

    solver_gap = _optional_float(getattr(solve_result.solver, "gap", None))
    lower_bound = _optional_float(solver_outcome["lower_bound"])
    upper_bound = _optional_float(solver_outcome["upper_bound"])
    if solver_gap is None and lower_bound is not None and upper_bound is not None:
        solver_gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-12)
    if solver_gap is None and solver_outcome["classification"] == "OPTIMAL":
        solver_gap = 0.0
    timed_out = bool(solver_outcome["timed_out"])
    objective_value = _safe_value(model.objective)

    scenario_costs = np.asarray([_safe_value(model.scenario_cost[w]) for w in range(w_count)])
    expected_cost = float(np.dot(scenario_probabilities, scenario_costs))
    empirical_cvar = _weighted_cvar(scenario_costs, scenario_probabilities, alpha)
    day_ahead_costs = np.asarray(
        [
            interval_hours
            * sum(
                _safe_value(model.p_da[z, t, w]) * float(bundle.da_price[z, t])
                for z in range(z_count)
                for t in range(horizon)
            )
            for w in range(w_count)
        ]
    )
    rt_imbalance_costs = np.asarray(
        [
            interval_hours
            * sum(
                _safe_value(model.dev_plus[z, t, w])
                * float(bundle.rt_price[w, z, t])
                - _safe_value(model.dev_minus[z, t, w])
                * float(effective_rt_sell_price[w, z, t])
                for z in range(z_count)
                for t in range(horizon)
            )
            for w in range(w_count)
        ]
    )
    total_shortfall = np.asarray(
        [
            sum(_safe_value(model.short[z, t, w]) for z in range(z_count) for t in range(horizon)) * granularity
            for w in range(w_count)
        ]
    )
    total_grid = np.asarray(
        [
            sum(_safe_value(model.grid[z, t, w]) for z in range(z_count) for t in range(horizon)) * interval_hours
            for w in range(w_count)
        ]
    )
    migrated_work = np.asarray(
        [
            0.5
            * granularity
            * sum(
                _safe_value(model.reassignment[z, t, w])
                for z in range(z_count)
                for t in range(horizon)
            )
            for w in range(w_count)
        ]
    )
    migration_energy = transfer_energy_mwh_per_work_unit * migrated_work
    battery_throughput = np.asarray(
        [
            interval_hours
            * sum(
                _safe_value(model.charge[z, t, w])
                + _safe_value(model.discharge[z, t, w])
                for z in range(z_count)
                for t in range(horizon)
            )
            for w in range(w_count)
        ]
    )
    pv_curtailment = np.asarray(
        [
            interval_hours
            * sum(
                _safe_value(model.pv_curtail[z, t, w])
                for z in range(z_count)
                for t in range(horizon)
            )
            for w in range(w_count)
        ]
    )
    sla_penalty_costs = sla_penalty * total_shortfall
    migration_costs = (
        migration_penalty * (1.0 + checkpoint_overhead_fraction) * migrated_work
    )
    migration_energy_costs = migration_energy_price * migration_energy
    battery_degradation_costs = battery_penalty * battery_throughput
    curtailment_costs = curtailment_penalty * pv_curtailment
    slack_costs = np.asarray(
        [
            slack_penalty
            * (
                sum(_safe_value(model.ramp_slack[t, w]) for t in range(horizon))
                + sum(
                    _safe_value(model.concentration_slack[z, t, w])
                    for z in range(z_count)
                    for t in range(horizon)
                )
            )
            for w in range(w_count)
        ]
    )
    grid_profiles = np.asarray(
        [
            [
                [_safe_value(model.grid[z, t, w]) for t in range(horizon)]
                for z in range(z_count)
            ]
            for w in range(w_count)
        ]
    )
    system_profiles = grid_profiles.sum(axis=1)
    scenario_rebound_peaks = np.max(system_profiles, axis=1)
    denominators = np.maximum(system_profiles[:, None, :], 1e-12)
    scenario_concentrations = np.max(grid_profiles / denominators, axis=(1, 2))
    validations = _validate_solution(
        model,
        bundle,
        physical,
        policy,
        demand_chunks,
        home,
        rigid_home,
        non_migratable_home_totals,
        migration_bandwidth_chunks,
        granularity,
        interval_hours,
        pue_slope_multiplier,
        float(settings.get("validation_tolerance", 1e-5)),
        {
            "physical_multiplier": effective_physical_multiplier,
            "pv": bundle.pv,
            "da_price": bundle.da_price,
            "rt_buy_price": bundle.rt_price,
            "rt_sell_price": effective_rt_sell_price,
            "capacity_work_units": physical.capacity_work_units,
            "hardware_transfer_bounds_work_units": hardware_transfer_bounds_work_units,
            "negative_price_floor_usd_per_mwh": negative_price_floor,
            "initial_soc_mwh": initial_soc_mwh,
            "terminal_soc_target_mwh": terminal_soc_target_mwh,
            "recourse_nonanticipative_prefix_steps": recourse_nonanticipative_prefix_steps,
            "fixed_day_ahead_schedule_mw": fixed_day_ahead_schedule_mw,
            "previous_grid_import_mw": previous_grid_import_mw,
            "ramp_limit_mw_per_step": ramp_limit,
            "cumulative_release_chunks": cumulative_release_chunks,
            "cumulative_due_chunks": cumulative_due_chunks,
        },
    )
    if not validations["passed"]:
        raise OptimizationFailure("post-solve conservation validation failed", validations)

    xcel_summary: dict[str, Any] | None = None
    if xcel_tariff is not None:
        if len(xcel_tariff.cpp_event_mask) != horizon:
            raise ValueError("Xcel CPP event mask length must match the optimization horizon")
        z = NYISO_ZONES.index(xcel_tariff.zone)
        tariff_costs = []
        for w in range(w_count):
            hourly = np.asarray([_safe_value(model.grid[z, t, w]) for t in range(horizon)])
            cost = (
                xcel_tariff.energy_usd_per_mwh * interval_hours * hourly.sum()
                + xcel_tariff.demand_usd_per_mw_day * hourly.max(initial=0.0)
                + xcel_tariff.cpp_adder_usd_per_mwh
                * interval_hours
                * hourly[xcel_tariff.cpp_event_mask].sum()
            )
            tariff_costs.append(cost)
        xcel_summary = {
            "expected_tariff_cost_usd": float(np.dot(scenario_probabilities, tariff_costs)),
            "evidence_role": "counterfactual",
            "source_relationship": "tariff_defined_counterfactual",
            "measurement_relationship": "not_measured_by_this_study",
            "observed_event_calls": False,
            "source_sha256": xcel_tariff.source_sha256,
        }

    pda_schedule = [
        [
            [_safe_value(model.p_da[z, t, w]) for t in range(horizon)]
            for z in range(z_count)
        ]
        for w in range(w_count)
    ]
    sensitivity_settings_consumed = {
        key: _plain_value(settings[key])
        for key in SENSITIVITY_SETTING_KEYS
        if key in settings
    }
    effective_battery_duration = [
        (
            float(physical.battery_energy_mwh[z] / physical.battery_power_mw[z])
            if physical.battery_power_mw[z] > 0
            else 0.0
        )
        for z in range(z_count)
    ]
    effective_input_diagnostics = {
        "expected_workload_total_work_units": float(
            np.dot(scenario_probabilities, bundle.workload.sum(axis=1))
        ),
        "expected_pv_available_mwh": float(
            interval_hours
            * np.dot(scenario_probabilities, bundle.pv.sum(axis=(1, 2)))
        ),
        "rt_price_min_usd_per_mwh": float(np.min(bundle.rt_price)),
        "rt_price_max_usd_per_mwh": float(np.max(bundle.rt_price)),
        "rt_price_std_usd_per_mwh": float(np.std(bundle.rt_price)),
        "da_price_min_usd_per_mwh": float(np.min(bundle.da_price)),
        "effective_rt_sell_price_min_usd_per_mwh": float(
            np.min(effective_rt_sell_price)
        ),
        "physical_multiplier_mean": float(np.mean(effective_physical_multiplier)),
        "physical_multiplier_min": float(np.min(effective_physical_multiplier)),
        "physical_multiplier_max": float(np.max(effective_physical_multiplier)),
        "capacity_work_units": physical.capacity_work_units.tolist(),
        "grid_import_limit_mw": physical.grid_import_limit_mw.tolist(),
        "battery_power_mw": physical.battery_power_mw.tolist(),
        "battery_energy_mwh": physical.battery_energy_mwh.tolist(),
        "battery_duration_hours_by_zone": effective_battery_duration,
        "migration_bandwidth_effective_work_units_per_interval": (
            None
            if migration_bandwidth_chunks is None
            else float(migration_bandwidth_chunks * granularity)
        ),
        "hardware_transfer_bounds_work_units": hardware_transfer_bounds_work_units.tolist(),
        "rigid_home_chunk_count": int(rigid_home.sum()),
        "non_migratable_home_chunk_count": int(non_migratable_home_totals.sum()),
        "effective_stress_mass": effective_stress_mass,
        "nominal_probability_mass": (
            float(scenario_probabilities[: bundle.nominal_component_count].sum())
            if bundle.nominal_component_count
            else None
        ),
        "stress_probability_mass": (
            float(scenario_probabilities[bundle.nominal_component_count :].sum())
            if bundle.stress_component_count
            else None
        ),
        "initial_soc_mwh": initial_soc_mwh.tolist(),
        "terminal_soc_target_mwh": (
            None
            if terminal_soc_target_mwh is None
            else terminal_soc_target_mwh.tolist()
        ),
        "fixed_day_ahead_schedule": fixed_day_ahead_schedule_mw is not None,
        "recourse_nonanticipative_prefix_steps": recourse_nonanticipative_prefix_steps,
    }
    result = {
        "schema_version": "research_optimization_result_v1",
        "operating_date": bundle.operating_date,
        "policy_id": policy_id,
        "policy": asdict(policy),
        "regime": bundle.regime,
        "kappa": bundle.kappa,
        "initial_scenario_count": bundle.initial_count,
        "reduced_scenario_count": bundle.reduced_count,
        "scenario_seed": bundle.seed,
        "scenario_mixture_stress_mass": effective_stress_mass,
        "stress_mass": effective_stress_mass,
        "scenario_nominal_component_count": bundle.nominal_component_count,
        "scenario_stress_component_count": bundle.stress_component_count,
        "solver": solver_name,
        "solver_status": str(solve_result.solver.status),
        "termination_condition": termination,
        "solver_outcome": solver_outcome["classification"],
        "solver_has_incumbent": bool(solver_outcome["has_incumbent"]),
        "time_limit_reached": timed_out,
        "optimality_proven": solver_outcome["classification"] == "OPTIMAL",
        "has_incumbent": bool(solver_outcome["has_incumbent"]),
        "objective_value": objective_value,
        "objective_lower_bound": lower_bound,
        "objective_upper_bound": upper_bound,
        "relative_mip_gap": solver_gap,
        "solver_objective_value": objective_value,
        "solver_objective_lower_bound": lower_bound,
        "solver_objective_upper_bound": upper_bound,
        "solver_lower_bound_usd": lower_bound,
        "solver_upper_bound_usd": upper_bound,
        "mip_gap": solver_gap,
        "mip_gap_target": float(settings.get("mip_relative_gap", 1e-4)),
        "time_limit_seconds": float(settings.get("time_limit_seconds", 120.0)),
        "solver_threads": solver_threads,
        "mip_start": mip_start,
        "solve_time_seconds": float(solve_time_seconds),
        "number_of_variables": int(model.nvariables()),
        "number_of_constraints": int(model.nconstraints()),
        "objective_usd": objective_value,
        "expected_total_cost_usd": expected_cost,
        "cvar_total_cost_usd": empirical_cvar,
        "cvar_alpha": alpha,
        "risk_weight": risk_weight,
        "interval_hours": interval_hours,
        "deadline_steps": deadline_steps,
        "flexible_share": flexible_share,
        "migration_eligible": migration_eligible_share,
        "migration_bandwidth_base_work_units_per_interval": (
            None if migration_bandwidth is None else float(migration_bandwidth)
        ),
        "migration_bandwidth_multiplier": migration_bandwidth_multiplier,
        "transfer_energy_mwh_per_work_unit": transfer_energy_mwh_per_work_unit,
        "migration_energy_price_usd_per_mwh": migration_energy_price,
        "checkpoint_overhead_fraction": checkpoint_overhead_fraction,
        "pue_slope_multiplier": pue_slope_multiplier,
        "physical_multiplier_scale": physical_multiplier_scale,
        "workload_scale": float(settings.get("workload_scale", 1.0)),
        "battery_duration_hours": settings.get("battery_duration_hours"),
        "pv_capacity_multiplier": float(settings.get("pv_capacity_multiplier", 1.0)),
        "physical_ensemble_selector": settings.get(
            "physical_ensemble_selector", "scenario"
        ),
        "support_distance_threshold": settings.get("support_distance_threshold"),
        "out_of_support_capacity_derating": settings.get(
            "out_of_support_capacity_derating"
        ),
        "site_capacity_multiplier": settings.get("site_capacity_multiplier", 1.0),
        "hardware_transfer_multiplier": hardware_transfer_multiplier,
        "rt_volatility_scale": float(settings.get("rt_volatility_scale", 1.0)),
        "negative_price_floor_usd_per_mwh": negative_price_floor,
        "rt_sell_back_haircut": rt_sell_haircut_fraction,
        "expected_sla_shortfall_work_units": float(np.dot(scenario_probabilities, total_shortfall)),
        "expected_grid_energy_mwh": float(np.dot(scenario_probabilities, total_grid)),
        "expected_migrated_work_units": float(
            np.dot(scenario_probabilities, migrated_work)
        ),
        "expected_migration_energy_mwh": float(
            np.dot(scenario_probabilities, migration_energy)
        ),
        "expected_checkpoint_overhead_work_units": float(
            checkpoint_overhead_fraction
            * np.dot(scenario_probabilities, migrated_work)
        ),
        "expected_battery_throughput_mwh": float(
            np.dot(scenario_probabilities, battery_throughput)
        ),
        "expected_pv_curtailment_mwh": float(
            np.dot(scenario_probabilities, pv_curtailment)
        ),
        "rebound_peak_mw": float(np.dot(scenario_probabilities, scenario_rebound_peaks)),
        "maximum_zone_concentration_share": float(np.dot(scenario_probabilities, scenario_concentrations)),
        "checkpoint_granularity_work_units": granularity,
        "deviation_formulation": "exact_continuous_rt_sell_not_above_rt_buy_v1",
        "deviation_binary_variables": 0,
        "checkpoint_rounding_max_chunks": rounding_error,
        "checkpoint_rounding_max_work_units": rounding_error * granularity,
        "constraint_variant": str(settings.get("constraint_variant", "base")),
        "ramp_limit_mw_per_step": None if ramp_limit is None else float(ramp_limit),
        "concentration_share_limit": None if concentration_limit is None else float(concentration_limit),
        "scenario_costs_usd": scenario_costs.tolist(),
        "scenario_day_ahead_costs_usd": day_ahead_costs.tolist(),
        "scenario_rt_imbalance_costs_usd": rt_imbalance_costs.tolist(),
        "scenario_sla_penalty_costs_usd": sla_penalty_costs.tolist(),
        "scenario_migration_costs_usd": migration_costs.tolist(),
        "scenario_migration_energy_costs_usd": migration_energy_costs.tolist(),
        "scenario_battery_degradation_costs_usd": battery_degradation_costs.tolist(),
        "scenario_curtailment_costs_usd": curtailment_costs.tolist(),
        "scenario_constraint_slack_costs_usd": slack_costs.tolist(),
        "scenario_probabilities": scenario_probabilities.tolist(),
        "sensitivity_setting_keys_supported": list(SENSITIVITY_SETTING_KEYS),
        "sensitivity_settings_consumed": sensitivity_settings_consumed,
        "sensitivity_variant": _plain_value(settings.get("sensitivity_variant")),
        "sensitivity_effective_inputs": effective_input_diagnostics,
        "day_ahead_schedule_mw": pda_schedule,
        "validations": validations,
        "counterfactual": True,
        "evidence_layer": "counterfactual",
        "calibration_id": physical.calibration_id,
        "twin_ids": list(physical.twin_ids),
        "transfer_regime": physical.transfer_regime,
        "zones": list(bundle.zones),
        "nyiso_evidence_role": "counterfactual",
        "nyiso_source_relationship": "counterfactual_using_third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
        "xcel_tariff_counterfactual": xcel_summary,
    }
    if bool(settings.get("return_dispatch_details", False)):
        def extract_zt(variable: Any, *, scale: float = 1.0) -> list[list[list[float]]]:
            return [
                [
                    [scale * _safe_value(variable[z, t, w]) for t in range(horizon)]
                    for z in range(z_count)
                ]
                for w in range(w_count)
            ]

        result["dispatch_details"] = {
            "work_work_units": extract_zt(model.work, scale=granularity),
            "shortfall_work_units": extract_zt(model.short, scale=granularity),
            "reassignment_work_units": extract_zt(
                model.reassignment, scale=granularity
            ),
            "grid_import_mw": grid_profiles.tolist(),
            "pv_used_mw": extract_zt(model.pv_used),
            "pv_curtailment_mw": extract_zt(model.pv_curtail),
            "battery_charge_mw": extract_zt(model.charge),
            "battery_discharge_mw": extract_zt(model.discharge),
            "day_ahead_schedule_mw": pda_schedule,
            "rt_deviation_purchase_mw": extract_zt(model.dev_plus),
            "rt_deviation_sale_mw": extract_zt(model.dev_minus),
            "ramp_slack_mw": [
                [_safe_value(model.ramp_slack[t, w]) for t in range(horizon)]
                for w in range(w_count)
            ],
            "concentration_slack_mw": extract_zt(model.concentration_slack),
            "soc_mwh": [
                [
                    [_safe_value(model.soc[z, t, w]) for t in range(horizon + 1)]
                    for z in range(z_count)
                ]
                for w in range(w_count)
            ],
        }
    return result


def save_optimization_result(result: Mapping[str, Any], path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(result), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path
