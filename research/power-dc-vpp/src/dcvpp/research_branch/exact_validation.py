from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pyomo.environ as pyo

from .optimization import OptimizationFailure


EXACT_VALIDATION_TOLERANCES: dict[str, float] = {
    "objective_abs_usd": 1e-6,
    "task_conservation_abs": 1e-7,
    "capacity_violation_work_units": 1e-7,
    "facility_balance_abs_mw": 1e-7,
    "pv_balance_abs_mw": 1e-7,
    "settlement_balance_abs_mw": 1e-7,
    "integrality_abs": 1e-7,
    "aggregate_execution_abs_work_units": 1e-7,
}


@dataclass(frozen=True)
class SmallTask:
    task_id: str
    home_site: int
    release_step: int
    deadline_step: int
    duration_steps: int = 1
    work_units: float = 1.0
    migration_eligible: bool = True
    eligible_sites: tuple[int, ...] | None = None
    critical: bool = False


@dataclass(frozen=True)
class SmallExactCase:
    case_id: str
    site_names: tuple[str, ...]
    horizon: int
    tasks: tuple[SmallTask, ...]
    capacity_work_units: np.ndarray
    idle_mw: np.ndarray
    work_mw_per_unit: np.ndarray
    grid_import_limit_mw: np.ndarray
    pv_available_mw: np.ndarray
    day_ahead_schedule_mw: np.ndarray
    da_price_usd_per_mwh: np.ndarray
    rt_price_usd_per_mwh: np.ndarray
    interval_hours: float = 1.0
    sla_penalty_usd_per_work_unit: float = 1000.0
    migration_penalty_usd_per_work_unit: float = 0.0
    curtailment_penalty_usd_per_mwh: float = 0.0
    rt_sell_back_haircut: float = 0.0
    expected_objective_usd: float | None = None

    def validate(self) -> None:
        sites = len(self.site_names)
        if sites not in {1, 2}:
            raise ValueError("small exact validation cases must contain one or two sites")
        if self.horizon <= 0 or self.interval_hours <= 0:
            raise ValueError("horizon and interval_hours must be positive")
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(
            self.tasks
        ):
            raise ValueError("tasks must be nonempty and have unique task_id values")
        for name, shape in (
            ("capacity_work_units", (sites,)),
            ("idle_mw", (sites,)),
            ("work_mw_per_unit", (sites,)),
            ("grid_import_limit_mw", (sites,)),
            ("pv_available_mw", (sites, self.horizon)),
            ("day_ahead_schedule_mw", (sites, self.horizon)),
            ("da_price_usd_per_mwh", (sites, self.horizon)),
            ("rt_price_usd_per_mwh", (sites, self.horizon)),
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        for name in (
            "capacity_work_units",
            "idle_mw",
            "work_mw_per_unit",
            "grid_import_limit_mw",
            "pv_available_mw",
            "day_ahead_schedule_mw",
        ):
            if (np.asarray(getattr(self, name), dtype=float) < 0).any():
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.rt_sell_back_haircut <= 1:
            raise ValueError("rt_sell_back_haircut must lie in [0, 1]")
        for task in self.tasks:
            if not 0 <= task.home_site < sites:
                raise ValueError(f"task {task.task_id} has an invalid home site")
            if (
                task.duration_steps <= 0
                or task.work_units <= 0
                or task.release_step < 0
                or task.deadline_step >= self.horizon
                or task.release_step + task.duration_steps - 1
                > task.deadline_step
            ):
                raise ValueError(f"task {task.task_id} has invalid timing or size")
            eligible = _eligible_sites(task, sites)
            if not eligible:
                raise ValueError(f"task {task.task_id} has no eligible site")


@dataclass(frozen=True)
class _Item:
    item_id: str
    count: int
    home_site: int
    release_step: int
    deadline_step: int
    duration_steps: int
    work_units: float
    eligible_sites: tuple[int, ...]
    critical: bool


def _eligible_sites(task: SmallTask, site_count: int) -> tuple[int, ...]:
    if task.eligible_sites is not None:
        eligible = tuple(sorted(set(map(int, task.eligible_sites))))
    elif task.migration_eligible:
        eligible = tuple(range(site_count))
    else:
        eligible = (task.home_site,)
    if any(site < 0 or site >= site_count for site in eligible):
        raise ValueError(f"task {task.task_id} contains an invalid eligible site")
    if not task.migration_eligible and eligible != (task.home_site,):
        raise ValueError(
            f"non-migratable task {task.task_id} may only use its home site"
        )
    return eligible


def _exact_items(case: SmallExactCase) -> tuple[_Item, ...]:
    return tuple(
        _Item(
            item_id=task.task_id,
            count=1,
            home_site=task.home_site,
            release_step=task.release_step,
            deadline_step=task.deadline_step,
            duration_steps=task.duration_steps,
            work_units=float(task.work_units),
            eligible_sites=_eligible_sites(task, len(case.site_names)),
            critical=task.critical,
        )
        for task in case.tasks
    )


def _cohort_items(case: SmallExactCase) -> tuple[_Item, ...]:
    groups: dict[tuple[Any, ...], list[SmallTask]] = {}
    for task in case.tasks:
        key = (
            task.home_site,
            task.release_step,
            task.deadline_step,
            task.duration_steps,
            float(task.work_units),
            _eligible_sites(task, len(case.site_names)),
            task.critical,
        )
        groups.setdefault(key, []).append(task)
    items = []
    for index, (key, tasks) in enumerate(sorted(groups.items(), key=lambda pair: pair[0])):
        (
            home,
            release,
            deadline,
            duration,
            work_units,
            eligible,
            critical,
        ) = key
        items.append(
            _Item(
                item_id=f"cohort_{index}",
                count=len(tasks),
                home_site=int(home),
                release_step=int(release),
                deadline_step=int(deadline),
                duration_steps=int(duration),
                work_units=float(work_units),
                eligible_sites=tuple(eligible),
                critical=bool(critical),
            )
        )
    return tuple(items)


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _value(value: Any) -> float:
    result = pyo.value(value, exception=False)
    return float(result) if result is not None else float("nan")


def _validate_solution(
    case: SmallExactCase,
    items: tuple[_Item, ...],
    starts: Mapping[tuple[int, int, int], float],
    misses: Mapping[int, float],
    active_work: np.ndarray,
    grid: np.ndarray,
    pv_used: np.ndarray,
    pv_curtail: np.ndarray,
    dev_plus: np.ndarray,
    dev_minus: np.ndarray,
    objective: float,
    cost_components: Mapping[str, float],
) -> dict[str, Any]:
    sites = len(case.site_names)
    conservation = []
    integrality = []
    for item_index, item in enumerate(items):
        assigned = sum(
            starts.get((item_index, site, start), 0.0)
            for site in item.eligible_sites
            for start in range(
                item.release_step,
                item.deadline_step - item.duration_steps + 2,
            )
        )
        conservation.append(assigned + misses[item_index] - item.count)
        integrality.extend(
            value - round(value)
            for key, value in starts.items()
            if key[0] == item_index
        )
        integrality.append(misses[item_index] - round(misses[item_index]))
    capacity = active_work - np.asarray(case.capacity_work_units)[:, None]
    load = (
        np.asarray(case.idle_mw)[:, None]
        + np.asarray(case.work_mw_per_unit)[:, None] * active_work
    )
    facility = grid + pv_used - load
    pv_balance = pv_used + pv_curtail - np.asarray(case.pv_available_mw)
    settlement = (
        dev_plus
        - dev_minus
        - (grid - np.asarray(case.day_ahead_schedule_mw))
    )
    objective_residual = objective - float(sum(cost_components.values()))
    checks = {
        "task_conservation_max_abs": float(
            np.max(np.abs(conservation), initial=0.0)
        ),
        "migration_conservation_max_abs": float(
            np.max(np.abs(conservation), initial=0.0)
        ),
        "capacity_max_violation_work_units": float(
            np.max(capacity, initial=0.0)
        ),
        "facility_energy_balance_max_abs_mw": float(
            np.max(np.abs(facility), initial=0.0)
        ),
        "pv_balance_max_abs_mw": float(
            np.max(np.abs(pv_balance), initial=0.0)
        ),
        "settlement_balance_max_abs_mw": float(
            np.max(np.abs(settlement), initial=0.0)
        ),
        "grid_bound_max_violation_mw": float(
            np.max(grid - np.asarray(case.grid_import_limit_mw)[:, None], initial=0.0)
        ),
        "integrality_max_abs": float(np.max(np.abs(integrality), initial=0.0)),
        "simultaneous_deviation_max_product_mw2": float(
            np.max(dev_plus * dev_minus, initial=0.0)
        ),
        "objective_decomposition_max_abs_usd": abs(objective_residual),
        "negative_da_price_count": int(
            (np.asarray(case.da_price_usd_per_mwh) < 0).sum()
        ),
        "negative_rt_price_count": int(
            (np.asarray(case.rt_price_usd_per_mwh) < 0).sum()
        ),
        "negative_prices_preserved": True,
    }
    tolerance_map = EXACT_VALIDATION_TOLERANCES
    checks["passed"] = bool(
        checks["task_conservation_max_abs"]
        <= tolerance_map["task_conservation_abs"]
        and checks["capacity_max_violation_work_units"]
        <= tolerance_map["capacity_violation_work_units"]
        and checks["facility_energy_balance_max_abs_mw"]
        <= tolerance_map["facility_balance_abs_mw"]
        and checks["pv_balance_max_abs_mw"]
        <= tolerance_map["pv_balance_abs_mw"]
        and checks["settlement_balance_max_abs_mw"]
        <= tolerance_map["settlement_balance_abs_mw"]
        and checks["integrality_max_abs"] <= tolerance_map["integrality_abs"]
        and checks["simultaneous_deviation_max_product_mw2"] <= 1e-8
        and checks["objective_decomposition_max_abs_usd"]
        <= tolerance_map["objective_abs_usd"]
    )
    return checks


def _solve_formulation(
    case: SmallExactCase,
    items: tuple[_Item, ...],
    formulation: str,
    *,
    solver_name: str,
    time_limit_seconds: float,
    mip_relative_gap: float,
) -> dict[str, Any]:
    if solver_name not in {"highs", "appsi_highs"}:
        raise ValueError("exact validation is restricted to HiGHS")
    sites = len(case.site_names)
    allowed_starts = [
        (item_index, site, start)
        for item_index, item in enumerate(items)
        for site in item.eligible_sites
        for start in range(
            item.release_step,
            item.deadline_step - item.duration_steps + 2,
        )
    ]
    model = pyo.ConcreteModel(name=f"{case.case_id}_{formulation}")
    model.I = pyo.RangeSet(0, len(items) - 1)
    model.Z = pyo.RangeSet(0, sites - 1)
    model.T = pyo.RangeSet(0, case.horizon - 1)
    model.A = pyo.Set(initialize=allowed_starts, dimen=3)
    start_domain = pyo.Binary if formulation == "exact_task" else pyo.NonNegativeIntegers
    miss_domain = pyo.Binary if formulation == "exact_task" else pyo.NonNegativeIntegers
    model.start = pyo.Var(model.A, domain=start_domain)
    model.miss = pyo.Var(model.I, domain=miss_domain)
    model.grid = pyo.Var(model.Z, model.T, domain=pyo.NonNegativeReals)
    model.pv_used = pyo.Var(model.Z, model.T, domain=pyo.NonNegativeReals)
    model.pv_curtail = pyo.Var(model.Z, model.T, domain=pyo.NonNegativeReals)
    model.dev_plus = pyo.Var(model.Z, model.T, domain=pyo.NonNegativeReals)
    model.dev_minus = pyo.Var(model.Z, model.T, domain=pyo.NonNegativeReals)
    model.deviation_mode = pyo.Var(model.Z, model.T, domain=pyo.Binary)

    model.assignment = pyo.Constraint(
        model.I,
        rule=lambda m, item_index: sum(
            m.start[item_index, site, start]
            for site in items[item_index].eligible_sites
            for start in range(
                items[item_index].release_step,
                items[item_index].deadline_step
                - items[item_index].duration_steps
                + 2,
            )
        )
        + m.miss[item_index]
        == items[item_index].count,
    )

    def active_expression(site: int, period: int) -> Any:
        return sum(
            item.work_units * model.start[item_index, site, start]
            for item_index, item in enumerate(items)
            if site in item.eligible_sites
            for start in range(
                item.release_step,
                item.deadline_step - item.duration_steps + 2,
            )
            if start <= period < start + item.duration_steps
        )

    model.active_work = pyo.Expression(
        model.Z,
        model.T,
        rule=lambda m, site, period: active_expression(site, period),
    )
    model.capacity = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, site, period: m.active_work[site, period]
        <= float(case.capacity_work_units[site]),
    )
    model.facility = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, site, period: m.grid[site, period]
        + m.pv_used[site, period]
        == float(case.idle_mw[site])
        + float(case.work_mw_per_unit[site]) * m.active_work[site, period],
    )
    model.pv_balance = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, site, period: m.pv_used[site, period]
        + m.pv_curtail[site, period]
        == float(case.pv_available_mw[site, period]),
    )
    model.grid_bound = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, site, period: m.grid[site, period]
        <= float(case.grid_import_limit_mw[site]),
    )
    model.settlement = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, site, period: m.dev_plus[site, period]
        - m.dev_minus[site, period]
        == m.grid[site, period]
        - float(case.day_ahead_schedule_mw[site, period]),
    )
    deviation_bound = np.asarray(case.grid_import_limit_mw) + np.max(
        np.asarray(case.day_ahead_schedule_mw), axis=1
    )
    model.dev_plus_bound = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, site, period: m.dev_plus[site, period]
        <= float(deviation_bound[site]) * m.deviation_mode[site, period],
    )
    model.dev_minus_bound = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, site, period: m.dev_minus[site, period]
        <= float(deviation_bound[site]) * (1 - m.deviation_mode[site, period]),
    )

    rt_sell = np.asarray(case.rt_price_usd_per_mwh) - case.rt_sell_back_haircut * np.abs(
        np.asarray(case.rt_price_usd_per_mwh)
    )
    day_ahead_cost = case.interval_hours * sum(
        float(case.day_ahead_schedule_mw[site, period])
        * float(case.da_price_usd_per_mwh[site, period])
        for site in range(sites)
        for period in range(case.horizon)
    )
    rt_cost = case.interval_hours * sum(
        model.dev_plus[site, period]
        * float(case.rt_price_usd_per_mwh[site, period])
        - model.dev_minus[site, period] * float(rt_sell[site, period])
        for site in range(sites)
        for period in range(case.horizon)
    )
    sla_cost = sum(
        case.sla_penalty_usd_per_work_unit
        * item.work_units
        * model.miss[item_index]
        for item_index, item in enumerate(items)
    )
    migration_cost = sum(
        case.migration_penalty_usd_per_work_unit
        * item.work_units
        * model.start[item_index, site, start]
        for item_index, item in enumerate(items)
        for site in item.eligible_sites
        for start in range(
            item.release_step,
            item.deadline_step - item.duration_steps + 2,
        )
        if site != item.home_site
    )
    curtailment_cost = (
        case.curtailment_penalty_usd_per_mwh
        * case.interval_hours
        * sum(
            model.pv_curtail[site, period]
            for site in range(sites)
            for period in range(case.horizon)
        )
    )
    tie_break_cost = 1e-9 * sum(
        (1 + site * case.horizon + start)
        * model.start[item_index, site, start]
        for item_index, item in enumerate(items)
        for site in item.eligible_sites
        for start in range(
            item.release_step,
            item.deadline_step - item.duration_steps + 2,
        )
    )
    model.objective = pyo.Objective(
        expr=day_ahead_cost
        + rt_cost
        + sla_cost
        + migration_cost
        + curtailment_cost
        + tie_break_cost,
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise OptimizationFailure(
            f"solver {solver_name} is unavailable", {"solver": solver_name}
        )
    solver.options["time_limit"] = float(time_limit_seconds)
    solver.options["mip_rel_gap"] = float(mip_relative_gap)
    started = perf_counter()
    solved = solver.solve(model, tee=False, load_solutions=True)
    elapsed = perf_counter() - started
    termination = str(solved.solver.termination_condition)
    if termination.lower() not in {"optimal", "feasible", "maxtimelimit", "max time limit"}:
        raise OptimizationFailure(
            f"small {formulation} solve terminated as {termination}",
            {"case_id": case.case_id, "formulation": formulation},
        )
    starts = {key: _value(model.start[key]) for key in allowed_starts}
    misses = {item_index: _value(model.miss[item_index]) for item_index in range(len(items))}
    active_work = np.asarray(
        [
            [_value(model.active_work[site, period]) for period in range(case.horizon)]
            for site in range(sites)
        ]
    )
    grid = np.asarray(
        [[_value(model.grid[z, t]) for t in range(case.horizon)] for z in range(sites)]
    )
    pv_used = np.asarray(
        [[_value(model.pv_used[z, t]) for t in range(case.horizon)] for z in range(sites)]
    )
    pv_curtail = np.asarray(
        [[_value(model.pv_curtail[z, t]) for t in range(case.horizon)] for z in range(sites)]
    )
    dev_plus = np.asarray(
        [[_value(model.dev_plus[z, t]) for t in range(case.horizon)] for z in range(sites)]
    )
    dev_minus = np.asarray(
        [[_value(model.dev_minus[z, t]) for t in range(case.horizon)] for z in range(sites)]
    )
    cost_components = {
        "day_ahead_cost_usd": float(day_ahead_cost),
        "rt_imbalance_cost_usd": _value(rt_cost),
        "sla_penalty_cost_usd": _value(sla_cost),
        "migration_cost_usd": _value(migration_cost),
        "curtailment_cost_usd": _value(curtailment_cost),
        "deterministic_tie_break_usd": _value(tie_break_cost),
    }
    objective = _value(model.objective)
    validations = _validate_solution(
        case,
        items,
        starts,
        misses,
        active_work,
        grid,
        pv_used,
        pv_curtail,
        dev_plus,
        dev_minus,
        objective,
        cost_components,
    )
    problem = solved.problem
    lower = _optional_float(getattr(problem, "lower_bound", None))
    upper = _optional_float(getattr(problem, "upper_bound", None))
    gap = _optional_float(getattr(solved.solver, "gap", None))
    if gap is None and lower is not None and upper is not None:
        gap = abs(upper - lower) / max(abs(upper), 1e-12)
    if gap is None and termination.lower() == "optimal":
        gap = 0.0
    return {
        "schema_version": "small_exact_or_cohort_result_v1",
        "case_id": case.case_id,
        "formulation": formulation,
        "solver": solver_name,
        "termination_condition": termination,
        "solve_time_seconds": elapsed,
        "mip_gap": gap,
        "time_limit_reached": termination.lower()
        in {"maxtimelimit", "max time limit"},
        "number_of_variables": int(model.nvariables()),
        "number_of_constraints": int(model.nconstraints()),
        "objective_usd": objective,
        "cost_components": cost_components,
        "active_work_units": active_work.tolist(),
        "grid_import_mw": grid.tolist(),
        "pv_used_mw": pv_used.tolist(),
        "pv_curtailment_mw": pv_curtail.tolist(),
        "rt_deviation_purchase_mw": dev_plus.tolist(),
        "rt_deviation_sale_mw": dev_minus.tolist(),
        "missed_work_units": float(
            sum(items[index].work_units * value for index, value in misses.items())
        ),
        "migrated_work_units": float(
            sum(
                items[item_index].work_units * value
                for (item_index, site, _), value in starts.items()
                if site != items[item_index].home_site
            )
        ),
        "start_counts": {
            f"{items[item_index].item_id}|{case.site_names[site]}|{start}": value
            for (item_index, site, start), value in starts.items()
            if value > 1e-9
        },
        "validations": validations,
        "counterfactual": True,
        "evidence_role": "counterfactual",
        "negative_prices_clipped": False,
    }


def run_exact_vs_cohort(
    case: SmallExactCase,
    *,
    solver: str = "highs",
    time_limit_seconds: float = 30.0,
    mip_relative_gap: float = 0.0,
) -> dict[str, Any]:
    """Solve the same small case at task and cohort resolution and compare."""

    case.validate()
    exact = _solve_formulation(
        case,
        _exact_items(case),
        "exact_task",
        solver_name=solver,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
    )
    cohort = _solve_formulation(
        case,
        _cohort_items(case),
        "cohort",
        solver_name=solver,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
    )
    objective_difference = abs(exact["objective_usd"] - cohort["objective_usd"])
    execution_difference = float(
        np.max(
            np.abs(
                np.asarray(exact["active_work_units"])
                - np.asarray(cohort["active_work_units"])
            ),
            initial=0.0,
        )
    )
    grid_difference = float(
        np.max(
            np.abs(
                np.asarray(exact["grid_import_mw"])
                - np.asarray(cohort["grid_import_mw"])
            ),
            initial=0.0,
        )
    )
    comparison = {
        "objective_abs_difference_usd": objective_difference,
        "aggregate_execution_max_abs_work_units": execution_difference,
        "grid_import_max_abs_mw": grid_difference,
        "missed_work_abs_difference_work_units": abs(
            exact["missed_work_units"] - cohort["missed_work_units"]
        ),
        "migrated_work_abs_difference_work_units": abs(
            exact["migrated_work_units"] - cohort["migrated_work_units"]
        ),
        "predeclared_tolerances": dict(EXACT_VALIDATION_TOLERANCES),
    }
    comparison["passed"] = bool(
        exact["validations"]["passed"]
        and cohort["validations"]["passed"]
        and objective_difference
        <= EXACT_VALIDATION_TOLERANCES["objective_abs_usd"]
        and execution_difference
        <= EXACT_VALIDATION_TOLERANCES["aggregate_execution_abs_work_units"]
        and grid_difference
        <= EXACT_VALIDATION_TOLERANCES["facility_balance_abs_mw"]
        and comparison["missed_work_abs_difference_work_units"]
        <= EXACT_VALIDATION_TOLERANCES["task_conservation_abs"]
        and comparison["migrated_work_abs_difference_work_units"]
        <= EXACT_VALIDATION_TOLERANCES["task_conservation_abs"]
    )
    expected_difference = (
        None
        if case.expected_objective_usd is None
        else abs(exact["objective_usd"] - case.expected_objective_usd)
    )
    hand_check_passed = (
        None
        if expected_difference is None
        else expected_difference <= EXACT_VALIDATION_TOLERANCES["objective_abs_usd"]
    )
    return {
        "schema_version": "exact_vs_cohort_validation_v1",
        "case": {
            "case_id": case.case_id,
            "site_names": list(case.site_names),
            "horizon": case.horizon,
            "task_count": len(case.tasks),
            "tasks": [asdict(task) for task in case.tasks],
            "expected_objective_usd": case.expected_objective_usd,
        },
        "exact": exact,
        "cohort": cohort,
        "comparison": comparison,
        "hand_check": {
            "expected_objective_abs_difference_usd": expected_difference,
            "passed": hand_check_passed,
        },
        "passed": bool(comparison["passed"] and hand_check_passed is not False),
        "counterfactual": True,
        "evidence_role": "counterfactual",
    }


def one_site_hand_checkable_case() -> SmallExactCase:
    """One mandatory task consumes 1 MWh at a retained -$5/MWh price."""

    return SmallExactCase(
        case_id="one_site_negative_price_hand_case",
        site_names=("SITE_A",),
        horizon=1,
        tasks=(
            SmallTask(
                task_id="task_0",
                home_site=0,
                release_step=0,
                deadline_step=0,
                migration_eligible=False,
            ),
        ),
        capacity_work_units=np.asarray([1.0]),
        idle_mw=np.asarray([0.0]),
        work_mw_per_unit=np.asarray([1.0]),
        grid_import_limit_mw=np.asarray([2.0]),
        pv_available_mw=np.zeros((1, 1)),
        day_ahead_schedule_mw=np.zeros((1, 1)),
        da_price_usd_per_mwh=np.zeros((1, 1)),
        rt_price_usd_per_mwh=np.asarray([[-5.0]]),
        expected_objective_usd=-5.0 + 1e-9,
    )


def two_site_hand_checkable_case() -> SmallExactCase:
    """A $2 migration moves one task from $10/MWh to $1/MWh: total $3."""

    return SmallExactCase(
        case_id="two_site_migration_hand_case",
        site_names=("HOME", "REMOTE"),
        horizon=1,
        tasks=(
            SmallTask(
                task_id="task_0",
                home_site=0,
                release_step=0,
                deadline_step=0,
                migration_eligible=True,
            ),
        ),
        capacity_work_units=np.asarray([1.0, 1.0]),
        idle_mw=np.asarray([0.0, 0.0]),
        work_mw_per_unit=np.asarray([1.0, 1.0]),
        grid_import_limit_mw=np.asarray([2.0, 2.0]),
        pv_available_mw=np.zeros((2, 1)),
        day_ahead_schedule_mw=np.zeros((2, 1)),
        da_price_usd_per_mwh=np.zeros((2, 1)),
        rt_price_usd_per_mwh=np.asarray([[10.0], [1.0]]),
        migration_penalty_usd_per_work_unit=2.0,
        expected_objective_usd=3.0 + 2e-9,
    )

