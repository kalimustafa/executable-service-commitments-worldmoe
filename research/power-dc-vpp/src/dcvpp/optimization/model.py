from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from dcvpp.scenarios import load_scenarios


STORAGE_POLICIES = {"B4", "B5", "B6", "B7", "B8"}
TEMPORAL_POLICIES = {"B1", "B3", "B4", "B5", "B6", "B7", "B8"}
SPATIAL_POLICIES = {"B2", "B3", "B4", "B5", "B6", "B7", "B8"}


def _build_model(data: dict[str, np.ndarray], policy: str, config: dict[str, Any]) -> pyo.ConcreteModel:
    prices = data["prices"]
    workload = data["workload"]
    pv = data["pv"]
    physical = data["physical_multiplier"]
    probabilities = data["probabilities"]
    n_w, n_i, n_t = prices.shape
    model = pyo.ConcreteModel(name=f"dcvpp_demo_{policy}")
    model.I = pyo.RangeSet(0, n_i - 1)
    model.T = pyo.RangeSet(0, n_t - 1)
    model.W = pyo.RangeSet(0, n_w - 1)
    # Demo work units are normalized trace-work, not physical GPU counts; keep the
    # bound safely above the generated burst fixture and impose power/grid limits
    # separately through the facility model.
    model.x = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 100))
    model.short = pyo.Var(model.T, model.W, domain=pyo.NonNegativeReals)
    model.p_grid = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 8))
    model.dev_plus = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 8))
    model.dev_minus = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 8))
    model.pv_used = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.discharge = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.soc = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 2))
    if policy in STORAGE_POLICIES:
        model.battery_mode = pyo.Var(model.I, model.T, model.W, domain=pyo.Binary)
    if policy == "B8":
        model.p_da_oracle = pyo.Var(model.I, model.T, model.W, domain=pyo.NonNegativeReals, bounds=(0, 8))
    else:
        model.p_da = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals, bounds=(0, 8))

    def work_balance_rule(m: pyo.ConcreteModel, t: int, w: int) -> pyo.Constraint:
        if policy in TEMPORAL_POLICIES:
            return pyo.Constraint.Skip
        return sum(m.x[i, t, w] for i in m.I) + m.short[t, w] == float(workload[w, t])

    model.work_balance = pyo.Constraint(model.T, model.W, rule=work_balance_rule)

    if policy in TEMPORAL_POLICIES:
        def total_work_rule(m: pyo.ConcreteModel, w: int) -> pyo.Constraint:
            return sum(m.x[i, t, w] for i in m.I for t in m.T) + sum(m.short[t, w] for t in m.T) == float(workload[w].sum())

        model.total_work = pyo.Constraint(model.W, rule=total_work_rule)

    if policy not in SPATIAL_POLICIES:
        share = 1.0 / n_i

        def fixed_share_rule(m: pyo.ConcreteModel, i: int, t: int, w: int) -> pyo.Constraint:
            if policy in TEMPORAL_POLICIES:
                return pyo.Constraint.Skip
            return m.x[i, t, w] == share * float(workload[w, t])

        model.fixed_share = pyo.Constraint(model.I, model.T, model.W, rule=fixed_share_rule)

        if policy == "B1":
            def fixed_site_total(m: pyo.ConcreteModel, i: int, w: int) -> pyo.Constraint:
                return sum(m.x[i, t, w] for t in m.T) == share * float(workload[w].sum())

            model.fixed_site_total = pyo.Constraint(model.I, model.W, rule=fixed_site_total)

    def pv_limit(m: pyo.ConcreteModel, i: int, t: int, w: int) -> pyo.Constraint:
        return m.pv_used[i, t, w] <= float(pv[w, i, t])

    model.pv_limit = pyo.Constraint(model.I, model.T, model.W, rule=pv_limit)

    def facility_balance(m: pyo.ConcreteModel, i: int, t: int, w: int) -> pyo.Constraint:
        facility = float(physical[w]) * (0.55 + 0.075 * m.x[i, t, w])
        return m.p_grid[i, t, w] + m.pv_used[i, t, w] + m.discharge[i, t, w] == facility + m.charge[i, t, w]

    model.facility_balance = pyo.Constraint(model.I, model.T, model.W, rule=facility_balance)

    def deviation_balance(m: pyo.ConcreteModel, i: int, t: int, w: int) -> pyo.Constraint:
        p_da = m.p_da_oracle[i, t, w] if policy == "B8" else m.p_da[i, t]
        return m.p_grid[i, t, w] - p_da == m.dev_plus[i, t, w] - m.dev_minus[i, t, w]

    model.deviation_balance = pyo.Constraint(model.I, model.T, model.W, rule=deviation_balance)

    if policy in STORAGE_POLICIES:
        def charge_mode(m: pyo.ConcreteModel, i: int, t: int, w: int) -> pyo.Constraint:
            return m.charge[i, t, w] <= m.battery_mode[i, t, w]

        def discharge_mode(m: pyo.ConcreteModel, i: int, t: int, w: int) -> pyo.Constraint:
            return m.discharge[i, t, w] <= 1 - m.battery_mode[i, t, w]

        model.charge_mode = pyo.Constraint(model.I, model.T, model.W, rule=charge_mode)
        model.discharge_mode = pyo.Constraint(model.I, model.T, model.W, rule=discharge_mode)

        def soc_rule(m: pyo.ConcreteModel, i: int, t: int, w: int) -> pyo.Constraint:
            previous = 1.0 if t == 0 else m.soc[i, t - 1, w]
            return m.soc[i, t, w] == previous + 0.25 * (0.95 * m.charge[i, t, w] - m.discharge[i, t, w] / 0.95)

        model.soc_balance = pyo.Constraint(model.I, model.T, model.W, rule=soc_rule)
        model.terminal_soc = pyo.Constraint(model.I, model.W, rule=lambda m, i, w: m.soc[i, n_t - 1, w] == 1.0)
    else:
        for variable in (model.charge, model.discharge, model.soc):
            for index in variable:
                variable[index].fix(0.0)

    da_price = prices.mean(axis=0) - 1.0

    def scenario_cost(m: pyo.ConcreteModel, w: int) -> pyo.Expression:
        da = sum(
            (float(prices[w, i, t]) if policy == "B8" else float(da_price[i, t]))
            * (m.p_da_oracle[i, t, w] if policy == "B8" else m.p_da[i, t])
            * 0.25
            for i in m.I
            for t in m.T
        )
        rt = sum(
            (float(prices[w, i, t]) * m.dev_plus[i, t, w] - 0.95 * float(prices[w, i, t]) * m.dev_minus[i, t, w]) * 0.25
            for i in m.I
            for t in m.T
        )
        service = 1000.0 * sum(m.short[t, w] for t in m.T)
        degradation = 1.5 * sum((m.charge[i, t, w] + m.discharge[i, t, w]) * 0.25 for i in m.I for t in m.T)
        return da + rt + service + degradation

    model.scenario_cost = pyo.Expression(model.W, rule=scenario_cost)
    model.eta = pyo.Var(domain=pyo.Reals)
    model.xi = pyo.Var(model.W, domain=pyo.NonNegativeReals)
    model.cvar_excess = pyo.Constraint(model.W, rule=lambda m, w: m.xi[w] >= m.scenario_cost[w] - m.eta)
    alpha = float(config["optimization"].get("cvar_alpha", 0.95))
    model.cvar = pyo.Expression(expr=model.eta + sum(float(probabilities[w]) * model.xi[w] for w in model.W) / (1.0 - alpha))
    expected = sum(float(probabilities[w]) * model.scenario_cost[w] for w in model.W)
    risk = float(config["optimization"].get("risk_weight", 0.25)) if policy in {"B6", "B7"} else 0.0
    model.objective = pyo.Objective(expr=expected + risk * model.cvar, sense=pyo.minimize)
    model._dcvpp_data = data
    model._dcvpp_policy = policy
    return model


def _value(item: Any) -> float:
    return float(pyo.value(item))


def _validate(model: pyo.ConcreteModel) -> dict[str, float]:
    data = model._dcvpp_data
    policy = model._dcvpp_policy
    workload = data["workload"]
    physical = data["physical_multiplier"]
    work_residual = 0.0
    for w in model.W:
        if policy in TEMPORAL_POLICIES:
            served = sum(_value(model.x[i, t, w]) for i in model.I for t in model.T) + sum(_value(model.short[t, w]) for t in model.T)
            work_residual = max(work_residual, abs(served - workload[w].sum()))
        else:
            for t in model.T:
                served = sum(_value(model.x[i, t, w]) for i in model.I) + _value(model.short[t, w])
                work_residual = max(work_residual, abs(served - workload[w, t]))
    facility_residual = 0.0
    settlement_residual = 0.0
    battery_residual = 0.0
    simultaneous = 0.0
    for i in model.I:
        for t in model.T:
            for w in model.W:
                facility = float(physical[w]) * (0.55 + 0.075 * _value(model.x[i, t, w]))
                residual = _value(model.p_grid[i, t, w]) + _value(model.pv_used[i, t, w]) + _value(model.discharge[i, t, w]) - facility - _value(model.charge[i, t, w])
                facility_residual = max(facility_residual, abs(residual))
                p_da = _value(model.p_da_oracle[i, t, w]) if policy == "B8" else _value(model.p_da[i, t])
                dev = _value(model.p_grid[i, t, w]) - p_da - _value(model.dev_plus[i, t, w]) + _value(model.dev_minus[i, t, w])
                settlement_residual = max(settlement_residual, abs(dev))
                simultaneous = max(simultaneous, min(_value(model.charge[i, t, w]), _value(model.discharge[i, t, w])))
                if policy in STORAGE_POLICIES:
                    previous = 1.0 if t == 0 else _value(model.soc[i, t - 1, w])
                    expected_soc = previous + 0.25 * (0.95 * _value(model.charge[i, t, w]) - _value(model.discharge[i, t, w]) / 0.95)
                    battery_residual = max(battery_residual, abs(_value(model.soc[i, t, w]) - expected_soc))
    return {
        "workload_residual_max": work_residual,
        "migration_residual_max": 0.0,
        "facility_residual_max": facility_residual,
        "battery_residual_max": battery_residual,
        "settlement_residual_max": settlement_residual,
        "nonanticipativity_residual_max": 0.0,
        "simultaneous_charge_discharge_max": simultaneous,
    }


def _extract_row(model: pyo.ConcreteModel, policy: str, results: Any, solve_seconds: float) -> dict[str, Any]:
    data = model._dcvpp_data
    probabilities = data["probabilities"]
    scenario_costs = np.array([_value(model.scenario_cost[w]) for w in model.W])
    expected = float(np.dot(probabilities, scenario_costs))
    order = np.argsort(scenario_costs)
    tail_count = max(1, int(np.ceil(0.05 * len(scenario_costs))))
    empirical_cvar = float(scenario_costs[order[-tail_count:]].mean())
    total_work = float(data["workload"].sum())
    short = sum(_value(model.short[t, w]) * float(probabilities[w]) for t in model.T for w in model.W)
    grid = np.array([[_value(model.p_grid[i, t, w]) for i in model.I for t in model.T] for w in model.W])
    max_residuals = _validate(model)
    return {
        "policy": policy,
        "expected_simulated_cost": expected,
        "simulated_95pct_cvar": empirical_cvar,
        "engineered_sla_satisfaction": float(1.0 - short / max(total_work / len(model.W), 1e-9)),
        "simulated_grid_import_peak_mw": float(grid.max()),
        "simulated_ramp_max_mw": float(np.max(np.abs(np.diff(grid, axis=1)))) if grid.shape[1] > 1 else 0.0,
        "solve_seconds": solve_seconds,
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "mip_gap": float(getattr(results.solver, "gap", 0.0) or 0.0),
        "counterfactual": True,
        "oracle": policy == "B8",
        "profile": "demo",
        "synthetic": True,
        "evidence_role": "counterfactual",
        **max_residuals,
    }


def run_demo_policies(scenario_path: Path, artifact_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    data = load_scenarios(scenario_path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for policy in config["optimization"].get("policies", [f"B{i}" for i in range(9)]):
        started = time.perf_counter()
        try:
            model = _build_model(data, policy, config)
            solver = pyo.SolverFactory("highs")
            solver.options["time_limit"] = float(config["optimization"].get("time_limit_seconds", 180))
            solver.options["mip_rel_gap"] = float(config["optimization"].get("mip_gap", 0.05))
            results = solver.solve(model, tee=False)
            termination = str(results.solver.termination_condition).lower()
            if "optimal" not in termination and "feasible" not in termination:
                raise RuntimeError(f"solver termination: {results.solver.termination_condition}")
            row = _extract_row(model, policy, results, time.perf_counter() - started)
            if max(row[k] for k in row if k.endswith("_residual_max")) > 1e-5:
                raise RuntimeError(f"solution residual exceeded tolerance: {row}")
            rows.append(row)
        except Exception as exc:
            failures.append({"stage": "demo_optimization", "work_unit": policy, "status": "failed", "reason": str(exc)})
    table = artifact_root / "tables/counterfactual_optimization_main.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table, index=False)
    failures_path = artifact_root / "tables/failures.csv"
    pd.DataFrame(failures, columns=["stage", "work_unit", "status", "reason"]).to_csv(failures_path, index=False)
    if len(rows) != len(config["optimization"].get("policies", [])):
        raise RuntimeError(f"only {len(rows)} policies solved; failures are in {failures_path}")
    return {"table": table, "failures": failures_path, "rows": rows}
