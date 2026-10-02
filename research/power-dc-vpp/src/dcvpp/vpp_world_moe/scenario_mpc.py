"""Persistent 20-scenario CVaR MPC adapted to the current R physical task.

W1-SMPC-20 restores the scenario/CVaR/first-action rolling paradigm. It does
not reproduce the historical R3 scenario model or its different physics.
The caller supplies frozen-W1 joint Monte Carlo paths and reveals realized
observations only after applying the returned first action.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping

import highspy
import numpy as np
from scipy.sparse import coo_matrix


SCENARIOS, PERIODS, NODES = 20, 96, 4


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


class _Topology:
    def __init__(self):
        self.ncols = 0
        self.integer: list[int] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.values: list[float] = []
        self.dynamic: dict[str, list[tuple[int, int]]] = {}

    def variables(self, shape, *, integer=False):
        size = int(np.prod(shape))
        result = np.arange(self.ncols, self.ncols + size, dtype=np.int32).reshape(shape)
        self.ncols += size
        self.integer.extend([int(integer)] * size)
        return result

    def row(self, terms=(), *, lower=-np.inf, upper=np.inf):
        row = len(self.lb)
        self.lb.append(lower)
        self.ub.append(upper)
        for column, value in terms:
            self.entry(row, column, value)
        return row

    def entry(self, row, column, value):
        self.rows.append(int(row)); self.cols.append(int(column)); self.values.append(float(value))

    def parameter(self, row, column, name, index):
        self.dynamic.setdefault(name, []).append((len(self.values), int(index)))
        self.entry(row, column, 0.0)


class ScenarioMPC20:
    """One native HiGHS model; parameter updates preserve its 96x20 topology."""

    name = "W1-SMPC-20"
    cvar_alpha = 0.95
    risk_weight = 0.25

    def __init__(self, time_limit_seconds=600.0, mip_gap=0.0, threads=1,
                 accepted_timeout_gap=0.05, battery_degradation_weight=0.01,
                 sla_weight=100.0, curtailment_weight=1.0, deadline_intervals=4,
                 action_residual_fraction=0.25):
        started = time.perf_counter()
        if time_limit_seconds <= 0 or not math.isfinite(time_limit_seconds):
            raise ValueError("time_limit_seconds must be positive and finite")
        if mip_gap != 0.0:
            raise ValueError("W1-SMPC-20 requires target relative gap zero")
        if threads != 1:
            raise ValueError("The registered persistent solver uses one thread")
        if not 0 <= accepted_timeout_gap <= 0.05:
            raise ValueError("Timeout acceptance may not exceed the registered 5% gap")
        if not 1 <= deadline_intervals <= PERIODS:
            raise ValueError("Invalid deadline interval count")
        for value in (battery_degradation_weight, sla_weight, curtailment_weight, action_residual_fraction):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Physical cost coefficients must be finite and nonnegative")
        self.time_limit_seconds = float(time_limit_seconds)
        self.accepted_timeout_gap = float(accepted_timeout_gap)
        self.degradation = float(battery_degradation_weight)
        self.sla_weight = float(sla_weight)
        self.curtailment = float(curtailment_weight)
        self.deadline = int(deadline_intervals)
        self.action_fraction = float(action_residual_fraction)
        self.highs = highspy.Highs()
        for name, value in {"output_flag": False, "threads": 1,
                "time_limit": self.time_limit_seconds, "mip_rel_gap": 0.0,
                "mip_abs_gap": 1.0e-10}.items():
            self._check(self.highs.setOptionValue(name, value), f"set {name}")
        self.effective_options = {name: self.highs.getOptionValue(name)[1]
                                 for name in ("threads", "time_limit", "mip_rel_gap", "mip_abs_gap")}
        self.topology = None
        self.model_build_count = 0
        self.pass_model_count = 0
        self.structure_rebuild_count = 0
        self.solve_count = 0
        self.last_solverstats: dict[str, Any] = {}
        self.initialization_seconds = time.perf_counter() - started

    @staticmethod
    def _check(status, operation):
        if status == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS failed to {operation}")

    def _build_topology(self):
        m = _Topology()
        shape = (SCENARIOS, PERIODS, NODES)
        v = {name: m.variables(shape, integer=name in {"charging_mode", "import_saturated"})
             for name in ("compute", "charge", "discharge", "pv_used", "grid", "unserved",
                          "soc", "queue", "charging_mode", "import_saturated")}
        v.update({"cumulative_service": m.variables((SCENARIOS, PERIODS)),
                  "cumulative_expiry": m.variables((SCENARIOS, PERIODS)),
                  "day_ahead": m.variables((PERIODS, NODES)),
                  "pv_fraction": m.variables((PERIODS, NODES)),
                  "scenario_cost": m.variables((SCENARIOS,)), "eta": m.variables((1,)),
                  "excess": m.variables((SCENARIOS,))})
        r = {name: np.empty(shape, dtype=np.int32) for name in
             ("charge_cap", "discharge_cap", "unserved_cap", "import_sat", "balance", "soc", "queue", "pv_fraction")}
        r.update({name: np.empty((SCENARIOS, PERIODS), dtype=np.int32) for name in
                  ("cumulative_service", "expiry_monotonic", "expiry_due")})
        r.update({name: np.empty((SCENARIOS, NODES), dtype=np.int32) for name in ("charge_budget", "discharge_budget")})
        r.update({name: np.empty((SCENARIOS - 1, PERIODS, NODES), dtype=np.int32)
                  for name in ("na_compute", "na_charge", "na_discharge")})
        r["scenario_cost"] = np.empty(SCENARIOS, dtype=np.int32)
        for s in range(SCENARIOS):
            cost_row = m.row(((v["scenario_cost"][s], 1),
                              (v["cumulative_expiry"][s, -1], -self.sla_weight)), lower=0, upper=0)
            r["scenario_cost"][s] = cost_row
            for t in range(PERIODS):
                for z in range(NODES):
                    k = (s * PERIODS + t) * NODES + z
                    x, c, d, p, g, u, soc, q, mode, sat = (int(v[name][s, t, z]) for name in
                        ("compute", "charge", "discharge", "pv_used", "grid", "unserved",
                         "soc", "queue", "charging_mode", "import_saturated"))
                    row = m.row(((c, 1),), upper=0); r["charge_cap"][s, t, z] = row
                    m.parameter(row, mode, "negative_power", k)
                    row = m.row(((d, 1),), upper=0); r["discharge_cap"][s, t, z] = row
                    m.parameter(row, mode, "power", k)
                    row = m.row(((u, 1),), upper=0); r["unserved_cap"][s, t, z] = row
                    m.parameter(row, sat, "negative_max_unserved", k)
                    row = m.row(((g, 1),), lower=0); r["import_sat"][s, t, z] = row
                    m.parameter(row, sat, "negative_grid_range", k)
                    row = m.row(((g, 1), (u, 1), (c, -1), (d, 1), (p, 1)), lower=0, upper=0)
                    r["balance"][s, t, z] = row
                    m.parameter(row, x, "negative_work_pue", k)
                    row = m.row(((soc, 1),), lower=0, upper=0); r["soc"][s, t, z] = row
                    if t: m.entry(row, v["soc"][s, t - 1, z], -1)
                    m.parameter(row, c, "negative_dt_eta", k); m.parameter(row, d, "dt_over_eta", k)
                    row = m.row(((q, 1), (x, 1)), lower=0, upper=0); r["queue"][s, t, z] = row
                    if t: m.entry(row, v["queue"][s, t - 1, z], -1)
                    row = m.row(((p, 1),)); r["pv_fraction"][s, t, z] = row
                    m.parameter(row, v["pv_fraction"][t, z], "negative_pv", k)
                    if s:
                        for name in ("compute", "charge", "discharge"):
                            r[f"na_{name}"][s - 1, t, z] = m.row(((v[name][s, t, z], 1), (v[name][0, t, z], -1)))
                    for column, parameter in ((g, "cost_grid"), (v["day_ahead"][t, z], "cost_da"),
                            (c, "cost_degradation"), (d, "cost_degradation"),
                            (p, "cost_used_pv"), (u, "cost_unserved")):
                        m.parameter(cost_row, column, parameter, k)
                cumulative = v["cumulative_service"][s, t]
                row = m.row(((cumulative, 1),), lower=0, upper=0)
                r["cumulative_service"][s, t] = row
                if t: m.entry(row, v["cumulative_service"][s, t - 1], -1)
                for z in range(NODES): m.entry(row, v["compute"][s, t, z], -1)
                expiry = v["cumulative_expiry"][s, t]
                row = m.row(((expiry, 1),), lower=0); r["expiry_monotonic"][s, t] = row
                if t: m.entry(row, v["cumulative_expiry"][s, t - 1], -1)
                r["expiry_due"][s, t] = m.row(((expiry, 1), (cumulative, 1)), lower=0)
            for z in range(NODES):
                cr = m.row(upper=0); dr = m.row(upper=0)
                r["charge_budget"][s, z], r["discharge_budget"][s, z] = cr, dr
                for t in range(PERIODS):
                    k = (s * PERIODS + t) * NODES + z
                    m.parameter(cr, v["charge"][s, t, z], "dt_eta", k)
                    m.parameter(dr, v["discharge"][s, t, z], "dt_over_eta", k)
            m.row(((v["excess"][s], 1), (v["scenario_cost"][s], -1), (v["eta"][0], 1)), lower=0)
        self.topology, self.variables, self.row_groups = m, v, r
        self._rows = np.asarray(m.rows, dtype=np.int32)
        self._cols = np.asarray(m.cols, dtype=np.int32)
        self._static_values = np.asarray(m.values, dtype=np.float64)
        self._dynamic = {name: np.asarray(items, dtype=np.int64).T for name, items in m.dynamic.items()}
        self._col_indices = np.arange(m.ncols, dtype=np.int32)
        self._row_indices = np.arange(len(m.lb), dtype=np.int32)
        self.model_build_count += 1

    def _validate(self, scenarios, weights, constraints, previous_da, origin, stage):
        if set(scenarios) != {"da", "rt", "arrivals", "pv", "pue"}:
            raise ValueError("Provide only the five causal joint-scenario arrays; no realized targets")
        data = {name: np.asarray(value, dtype=np.float64).copy() for name, value in scenarios.items()}
        if any(value.shape != (SCENARIOS, PERIODS, NODES) or not np.isfinite(value).all() for value in data.values()):
            raise ValueError("All scenario arrays must be finite [20,96,4]; topology is fixed")
        if (data["arrivals"] < 0).any() or (data["pv"] < 0).any() or (data["pue"] < 1).any():
            raise ValueError("Joint scenarios violate physical support")
        probability = np.asarray(weights, dtype=np.float64).copy()
        if probability.shape != (SCENARIOS,) or not np.isfinite(probability).all() or (probability < 0).any() or not np.isclose(probability.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError("Twenty nonnegative scenario probabilities must sum to one")
        probability /= probability.sum()
        names = ("battery_power_limit", "battery_energy_capacity", "initial_soc", "compute_capacity",
                 "initial_queue", "grid_import_limit", "grid_export_limit", "idle_power", "work_power_per_unit",
                 "charge_efficiency", "discharge_efficiency")
        physical = {name: np.asarray(constraints[name], dtype=np.float64).copy() for name in names}
        if any(value.shape != (NODES,) or not np.isfinite(value).all() or (value < 0).any() for value in physical.values()):
            raise ValueError("Physical constraints must be finite nonnegative four-node arrays")
        dt = float(constraints.get("dt", constraints.get("dt_hours", math.nan)))
        if not math.isfinite(dt) or dt <= 0 or (physical["charge_efficiency"] <= 0).any() or (physical["discharge_efficiency"] <= 0).any():
            raise ValueError("Invalid time step or storage efficiency")
        physical["initial_soc"] = np.minimum(physical["initial_soc"], physical["battery_energy_capacity"])
        physical["dt"] = dt
        prior = np.asarray(previous_da, dtype=np.float64).copy()
        if prior.shape != (PERIODS, NODES) or not np.isfinite(prior).all():
            raise ValueError("A finite known published DA baseline [96,4] is required")
        if isinstance(origin, bool) or int(origin) != origin or not 0 <= origin < PERIODS or stage not in (0, 1):
            raise ValueError("Invalid stage or rolling origin")
        if stage == 0 and origin != 0:
            raise ValueError("The registered DA decision is at origin zero")
        return data, probability, physical, prior

    def _parameters(self, data, weights, c, previous, origin, stage):
        m, v, r = self.topology, self.variables, self.row_groups
        shape = (SCENARIOS, PERIODS, NODES)
        active = np.arange(PERIODS) >= origin
        current = np.arange(PERIODS) == origin
        a = active[None, :, None]
        first = current[None, :, None]
        bcast = lambda value: np.broadcast_to(value, shape)
        power, energy = c["battery_power_limit"], c["battery_energy_capacity"]
        imports, exports = c["grid_import_limit"], c["grid_export_limit"]
        initial = c["initial_soc"]
        dt = c["dt"]
        max_unserved = np.maximum((c["idle_power"] + c["work_power_per_unit"] * c["compute_capacity"]) * data["pue"] + power - imports, 0)
        lower = np.zeros(m.ncols); upper = np.full(m.ncols, np.inf); cost = np.zeros(m.ncols)
        for name, bound in {"compute": c["compute_capacity"], "charge": power, "discharge": power,
                "pv_used": data["pv"], "grid": imports, "unserved": max_unserved,
                "soc": energy, "queue": np.inf, "charging_mode": (power > 0).astype(float),
                "import_saturated": (max_unserved > 0).astype(float)}.items():
            upper[v[name]] = np.where(a, bound, 0)
        lower[v["grid"]] = np.where(a, -exports, 0)
        for name in ("cumulative_service", "cumulative_expiry"):
            upper[v[name]] = np.where(active[None, :], np.inf, 0)
        lower[v["scenario_cost"]] = -np.inf
        lower[v["eta"]] = -np.inf
        if stage == 1:
            lower[v["day_ahead"]] = previous; upper[v["day_ahead"]] = previous
        else:
            lower[v["day_ahead"]] = np.maximum(previous - self.action_fraction * imports, -exports)
            upper[v["day_ahead"]] = np.minimum(previous + self.action_fraction * imports, imports)
        if np.any(lower > upper):
            raise ValueError("Published DA baseline has no reachable residual envelope")
        upper[v["pv_fraction"]] = current[:, None]
        no_pv = np.all(data["pv"][:, origin] == 0, axis=0)
        lower[v["pv_fraction"][origin, no_pv]] = 1.0
        cost[v["scenario_cost"]] = weights
        cost[v["eta"]] = self.risk_weight
        cost[v["excess"]] = self.risk_weight * weights / (1 - self.cvar_alpha)
        rl, ru = np.asarray(m.lb, dtype=float).copy(), np.asarray(m.ub, dtype=float).copy()
        ru[r["discharge_cap"]] = bcast(power)
        rl[r["import_sat"]] = bcast(-exports)
        for name, rhs in {"balance": data["pue"] * c["idle_power"] * a,
                "soc": bcast(initial) * first,
                "queue": data["arrivals"] * a + bcast(c["initial_queue"]) * first}.items():
            rl[r[name]] = rhs; ru[r[name]] = rhs
        arrivals = np.sum(data["arrivals"], axis=2) * active
        arrivals[:, origin] += c["initial_queue"].sum()
        cumulative = np.cumsum(arrivals, axis=1)
        due = np.zeros_like(cumulative)
        due[:, self.deadline - 1:] = cumulative[:, :PERIODS - self.deadline + 1]
        rl[r["expiry_due"]] = due
        ru[r["charge_budget"]] = np.broadcast_to(energy - initial, (SCENARIOS, NODES))
        ru[r["discharge_budget"]] = np.broadcast_to(initial, (SCENARIOS, NODES))
        for name in ("na_compute", "na_charge", "na_discharge", "pv_fraction"):
            indices = r[name]
            rl[indices] = np.where(first, 0, -np.inf)
            ru[indices] = np.where(first, 0, np.inf)
        pv_constant = self.curtailment * dt * np.sum(data["pv"] * a, axis=(1, 2))
        rl[r["scenario_cost"]] = pv_constant; ru[r["scenario_cost"]] = pv_constant
        groups = {"negative_power": -bcast(power), "power": bcast(power),
            "negative_max_unserved": -max_unserved, "negative_grid_range": -bcast(imports + exports),
            "negative_work_pue": -data["pue"] * c["work_power_per_unit"],
            "negative_dt_eta": -bcast(dt * c["charge_efficiency"]),
            "dt_eta": bcast(dt * c["charge_efficiency"]), "dt_over_eta": bcast(dt / c["discharge_efficiency"]),
            "negative_pv": -data["pv"], "cost_grid": -dt * data["rt"],
            "cost_da": -dt * (data["da"] - data["rt"]) * a,
            "cost_degradation": bcast(-self.degradation * dt),
            "cost_used_pv": bcast(self.curtailment * dt), "cost_unserved": bcast(-self.sla_weight * dt)}
        coefficients = self._static_values.copy()
        for name, (positions, source) in self._dynamic.items():
            coefficients[positions] = groups[name].ravel()[source]
        return cost, lower, upper, rl, ru, coefficients

    def _pass_or_update(self, parameters):
        cost, lower, upper, rl, ru, coefficients = parameters
        if self.pass_model_count == 0:
            matrix = coo_matrix((coefficients, (self._rows, self._cols)),
                shape=(len(rl), len(cost))).tocsc()
            lp = highspy.HighsLp()
            lp.num_col_, lp.num_row_ = len(cost), len(rl)
            lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lower, upper
            lp.row_lower_, lp.row_upper_ = rl, ru
            lp.integrality_ = [highspy.HighsVarType.kInteger if value else highspy.HighsVarType.kContinuous for value in self.topology.integer]
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
            lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
            lp.a_matrix_.value_ = matrix.data
            self._check(self.highs.passModel(lp), "pass fixed model")
            self.pass_model_count += 1
            changed = len(coefficients)
        else:
            self._check(self.highs.changeColsCost(len(cost), self._col_indices, cost), "update objective")
            self._check(self.highs.changeColsBounds(len(cost), self._col_indices, lower, upper), "update column bounds")
            self._check(self.highs.changeRowsBounds(len(rl), self._row_indices, rl, ru), "update row bounds")
            modified = np.flatnonzero(coefficients != self._previous_coefficients)
            for index in modified:
                self._check(self.highs.changeCoeff(int(self._rows[index]), int(self._cols[index]), float(coefficients[index])), "update coefficient")
            changed = len(modified)
        self._previous_coefficients = coefficients
        return changed

    @staticmethod
    def _weighted_cvar(costs, weights, alpha):
        remaining, total = 1 - alpha, 0.0
        for index in np.argsort(costs)[::-1]:
            amount = min(float(weights[index]), remaining)
            total += amount * float(costs[index]); remaining -= amount
            if remaining <= 1e-14: break
        return total / (1 - alpha)

    def solve(self, scenarios: Mapping[str, Any], weights, constraints, previous_da, origin, stage=1):
        started = time.perf_counter()
        data, probability, physical, previous = self._validate(scenarios, weights, constraints, previous_da, origin, stage)
        build_seconds = 0.0
        first_build = self.topology is None
        if first_build:
            before = time.perf_counter(); self._build_topology(); build_seconds += time.perf_counter() - before
        before = time.perf_counter()
        parameters = self._parameters(data, probability, physical, previous, origin, stage)
        update_seconds = time.perf_counter() - before
        before = time.perf_counter(); changed = self._pass_or_update(parameters)
        if first_build: build_seconds += time.perf_counter() - before
        else: update_seconds += time.perf_counter() - before
        # HiGHS clocks accumulate across run() calls. highspy 1.15.1 does not
        # bind zeroAllClocks, so use an absolute deadline on that same clock.
        # See https://ergo-code.github.io/HiGHS/dev/interfaces/c_api/
        # This preserves a fresh per-call budget without clearing solver state.
        native_before = float(self.highs.getRunTime())
        native_deadline = native_before + self.time_limit_seconds
        self._check(self.highs.setOptionValue("time_limit", native_deadline), "set per-call time deadline")
        before = time.perf_counter()
        self._check(self.highs.run(), "solve")
        solve_seconds = time.perf_counter() - before
        native_after = float(self.highs.getRunTime())
        self.solve_count += 1
        extraction_start = time.perf_counter()
        info, solution = self.highs.getInfo(), self.highs.getSolution()
        status = self.highs.getModelStatus()
        feasible = (solution.value_valid and info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible)
        values = np.asarray(solution.col_value, dtype=float)
        incumbent = bool(feasible and values.shape == (self.topology.ncols,) and np.isfinite(values).all())
        actual_gap = _finite(info.mip_gap)
        optimal = bool(status == highspy.HighsModelStatus.kOptimal and incumbent)
        timed_acceptance = bool(status == highspy.HighsModelStatus.kTimeLimit and incumbent
                                and actual_gap is not None and actual_gap <= self.accepted_timeout_gap)
        accepted = optimal or timed_acceptance
        v = self.variables
        plan0 = None
        scenario_costs = None
        stats = {"baseline": self.name, "solver": "native_HiGHS", "highs_version": self.highs.version(),
            "scenario_count": SCENARIOS, "scenario_probabilities": probability.tolist(),
            "input_probability_sum": float(np.asarray(weights, dtype=float).sum()),
            "scenario_weight_source": "caller_supplied_joint_probabilities_not_old_R3_learned_weights",
            "horizon_topology": PERIODS, "origin": int(origin), "active_horizon": PERIODS - int(origin), "stage": int(stage),
            "variables": self.topology.ncols, "constraints": len(self.topology.lb),
            "integer_variables": int(sum(self.topology.integer)),
            "active_integer_variables": int(np.sum(np.asarray(self.topology.integer, dtype=bool) & (parameters[2] > parameters[1]))),
            "status": self.highs.modelStatusToString(status), "has_incumbent": incumbent,
            "optimal": optimal, "accepted": accepted, "accepted_not_optimal": timed_acceptance,
            "fallback": not accepted, "actual_gap": actual_gap,
            "objective": _finite(info.objective_function_value) if incumbent else None,
            "objective_bound": _finite(info.mip_dual_bound), "mip_node_count": int(info.mip_node_count),
            "target_relative_gap": self.effective_options["mip_rel_gap"],
            "target_absolute_gap": self.effective_options["mip_abs_gap"],
            "time_limit_seconds": self.effective_options["time_limit"], "threads": self.effective_options["threads"],
            "effective_native_time_limit": native_deadline, "highs_runtime_before": native_before,
            "highs_runtime_after": native_after, "highs_runtime_increment": native_after - native_before,
            "time_budget_mode": "per_call_via_cumulative_timer_deadline",
            "accepted_timeout_relative_gap": self.accepted_timeout_gap,
            "model_build_count": self.model_build_count, "pass_model_count": self.pass_model_count,
            "structure_rebuild_count": self.structure_rebuild_count, "solve_count": self.solve_count,
            "updated_coefficient_count": changed, "build_seconds": build_seconds,
            "update_seconds": update_seconds, "solve_seconds": solve_seconds,
            "controller_initialization_seconds": self.initialization_seconds,
            "user_mip_start_supplied": False, "realized_control_inputs_used": False,
            "first_action_only": True, "cvar_alpha": self.cvar_alpha, "cvar_weight": self.risk_weight}
        if accepted:
            plan0 = {name: values[indices[0]].copy() for name, indices in v.items()
                     if name in {"compute", "charge", "discharge", "pv_used", "grid", "unserved", "soc", "queue", "cumulative_service", "cumulative_expiry"}}
            plan0["battery"] = plan0["discharge"] - plan0["charge"]
            day_ahead = values[v["day_ahead"]].copy()
            plan0["day_ahead"] = day_ahead
            first_action = {"battery": plan0["battery"][origin].copy(), "compute": plan0["compute"][origin].copy(),
                "pv_fraction": values[v["pv_fraction"][origin]].copy(), "day_ahead": day_ahead[origin].copy(),
                "target_soc": plan0["soc"][origin].copy()}
            scenario_costs = values[v["scenario_cost"]].copy()
            eta, excess = float(values[v["eta"][0]]), values[v["excess"]]
            cvar = eta + float(probability @ excess) / (1 - self.cvar_alpha)
            na_error = max(float(np.max(np.abs(values[v[name][:, origin]] - values[v[name][0, origin]])))
                           for name in ("compute", "charge", "discharge"))
            pv_error = np.max(np.abs(values[v["pv_used"][:, origin]] - data["pv"][:, origin] * first_action["pv_fraction"]))
            stats.update(expected_scenario_cost=float(probability @ scenario_costs), cvar_eta=eta,
                scenario_cvar95=float(cvar), exact_weighted_scenario_cvar95=self._weighted_cvar(scenario_costs, probability, self.cvar_alpha),
                first_control_nonanticipativity_max_abs=na_error, first_pv_fraction_consistency_max_abs=float(pv_error),
                first_grid_cross_scenario_range=float(np.ptp(values[v["grid"][:, origin]], axis=0).max()))
        else:
            first_action = {"battery": np.zeros(NODES), "compute": 0.5 * physical["compute_capacity"],
                "pv_fraction": np.full(NODES, 1 / (1 + math.exp(-2))), "day_ahead": previous[origin].copy(),
                "target_soc": 0.5 * physical["battery_energy_capacity"]}
            day_ahead = previous.copy()
            stats["fallback_policy"] = "registered_zero_raw_safe_tensor"
        stats["extract_seconds"] = time.perf_counter() - extraction_start
        stats["decision_wall_seconds"] = time.perf_counter() - started
        self.last_solverstats = stats
        return {"first_action": first_action, "day_ahead_schedule": day_ahead,
                "plan0": plan0, "scenario_costs": scenario_costs, "solverstats": stats}


__all__ = ["ScenarioMPC20", "SCENARIOS", "PERIODS", "NODES"]
