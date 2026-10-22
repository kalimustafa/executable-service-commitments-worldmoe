"""Deterministic frozen-W1 horizon optimization on the registered R action set.

One independent suffix is optimized at each registered issue origin. This is
an MPC-style finite-horizon baseline, not an additional closed-loop rollout.
Only the causal expert packet enters optimization; realized targets enter the
existing replay after this controller has returned its fixed raw actions.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time
from typing import Any
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
import torch

from .r_ecmoe import RECMoEOutput, RiskConditionPacket, W1ExpertPacket
from .r_ecmoe_evaluation import _complete_output, _safe_raw


class MPCInputError(ValueError):
    """The supplied packet cannot support a causal matched MPC decision."""


def _array(tensor, index: int) -> np.ndarray:
    return tensor[index].detach().to(device="cpu", dtype=torch.float64).numpy().copy()


def _finite_number(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class _MILP:
    def __init__(self):
        self.cost: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integer: list[int] = []
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []

    def variable(self, shape, *, lower=0.0, upper=np.inf, cost=0.0, integer=False):
        count = int(np.prod(shape))
        indices = np.arange(len(self.cost), len(self.cost) + count).reshape(shape)
        for destination, value in ((self.cost, cost), (self.lower, lower), (self.upper, upper)):
            destination.extend(np.broadcast_to(value, shape).ravel().tolist())
        self.integer.extend([int(integer)] * count)
        return indices

    def constraint(self, coefficients, *, lower=-np.inf, upper=np.inf):
        row = len(self.row_lower)
        for column, value in coefficients.items():
            if value:
                self.rows.append(row)
                self.columns.append(int(column))
                self.values.append(float(value))
        self.row_lower.append(float(lower))
        self.row_upper.append(float(upper))

    def matrix(self):
        return coo_matrix((self.values, (self.rows, self.columns)),
                          shape=(len(self.row_lower), len(self.cost))).tocsc()


class W1MPCController:
    """MILP decisions with the same conservative assembler-reachable bounds.

    Gurobi's installed license rejects 2001 variables (error 10010), so the
    executable backend is SciPy/HiGHS. All incumbent/status/fallback evidence
    is exposed in ``last_solver_records``; ``last_plans`` is diagnostic only.
    """

    backend = "scipy_highs_milp"
    inverse_epsilon = 1.0e-6

    def __init__(self, model, time_limit_seconds=2.0, mip_gap=1.0e-4, threads=1):
        if not math.isfinite(float(time_limit_seconds)) or time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be finite and positive")
        if not math.isfinite(float(mip_gap)) or mip_gap < 0:
            raise ValueError("mip_gap must be finite and nonnegative")
        if isinstance(threads, bool) or int(threads) != threads or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.model = model
        self.time_limit_seconds = float(time_limit_seconds)
        self.mip_gap = float(mip_gap)
        self.threads = int(threads)
        self.last_solver_records: list[dict[str, Any]] = []
        self.last_plans: list[dict[str, np.ndarray] | None] = []

    def _inputs(self, packet, risk, index):
        timeline = packet.state_latent.shape[1]
        origin = int(packet.origin_index[index].detach().cpu())
        horizon = int(risk.horizon[index].detach().cpu())
        stage = int(risk.stage[index].detach().cpu())
        action_range = float(risk.action_range[index].detach().cpu())
        if stage not in (0, 1) or origin < 0 or horizon < 1 or origin + horizon > timeline:
            raise MPCInputError("Invalid registered stage/origin/horizon")
        if not math.isfinite(action_range) or action_range <= 0:
            raise MPCInputError("MPC requires a positive inference action range")
        stop = origin + horizon
        known = _array(packet.valid_action_masks[..., 0], index).astype(bool)
        if not known[origin:stop].all() or (stage == 1 and not known.all()):
            raise MPCInputError("MPC requires the causal published DA baseline; no realized fallback is allowed")
        c = packet.constraints
        data = {name: _array(getattr(c, name), index) for name in (
            "battery_power_limit", "battery_energy_capacity", "initial_soc", "compute_capacity",
            "initial_queue", "grid_import_limit", "grid_export_limit", "idle_power",
            "work_power_per_unit", "charge_efficiency", "discharge_efficiency")}
        data.update({name: _array(getattr(packet.reference, name), index)[origin:stop]
                     for name in ("da", "rt", "arrivals", "pv", "pue")})
        data["previous_da"] = _array(packet.previous_actions[..., 0], index)[origin:stop]
        data["dt"] = float(c.dt_hours[index].detach().cpu())
        if not all(np.isfinite(value).all() for value in data.values()):
            raise MPCInputError("Nonfinite causal forecast or physical input")
        if data["dt"] <= 0 or (data["charge_efficiency"] <= 0).any() or (data["discharge_efficiency"] <= 0).any():
            raise MPCInputError("Invalid physical time step or efficiency")
        for name in ("battery_power_limit", "battery_energy_capacity", "compute_capacity", "initial_queue",
                     "grid_import_limit", "grid_export_limit", "idle_power", "work_power_per_unit"):
            if (data[name] < 0).any():
                raise MPCInputError(f"Negative physical quantity {name}")
        data["initial_soc"] = np.clip(data["initial_soc"], 0, data["battery_energy_capacity"])
        data["arrivals"] = np.maximum(data["arrivals"], 0)
        data["pv"] = np.maximum(data["pv"], 0)
        if (data["pue"] < 1).any():
            raise MPCInputError("Forecast PUE must be at least one")
        return data, origin, horizon, stage, action_range

    def _build(self, data, horizon, stage):
        m = _MILP()
        shape = (horizon, 4)
        dt = data["dt"]
        cfg = self.model.world_model.config
        degradation = float(cfg.battery_degradation_weight)
        sla_weight = float(cfg.unserved_power_weight)
        power = data["battery_power_limit"]
        energy = data["battery_energy_capacity"]
        initial = data["initial_soc"]
        imports, exports = data["grid_import_limit"], data["grid_export_limit"]
        facility_max = (data["idle_power"] + data["work_power_per_unit"] * data["compute_capacity"]) * data["pue"]
        max_unserved = np.maximum(facility_max + power - imports, 0.0)
        amount = float(self.model.config.action_residual_fraction) * imports
        da_lower = np.maximum(data["previous_da"] - amount, -exports) if stage == 0 else data["previous_da"]
        da_upper = np.minimum(data["previous_da"] + amount, imports) if stage == 0 else data["previous_da"]
        if (da_lower > da_upper).any():
            raise MPCInputError("Published DA baseline has no reachable DA residual envelope")
        v = {
            "compute": m.variable(shape, upper=data["compute_capacity"]),
            "charge": m.variable(shape, upper=power, cost=degradation * dt),
            "discharge": m.variable(shape, upper=power, cost=degradation * dt),
            "pv_used": m.variable(shape, upper=data["pv"], cost=-dt),
            "grid": m.variable(shape, lower=-exports, upper=imports, cost=data["rt"] * dt),
            "unserved": m.variable(shape, upper=max_unserved, cost=sla_weight * dt),
            "day_ahead": m.variable(shape, lower=da_lower, upper=da_upper, cost=(data["da"] - data["rt"]) * dt),
            "soc": m.variable(shape, upper=energy),
            "queue": m.variable(shape),
            "charging_mode": m.variable(shape, upper=np.broadcast_to(power > 0, shape).astype(float), integer=True),
            "import_saturated": m.variable(shape, upper=(max_unserved > 0).astype(float), integer=True),
            "cumulative_expiry": m.variable((horizon,), upper=float(data["initial_queue"].sum() + data["arrivals"].sum())),
        }
        m.cost[int(v["cumulative_expiry"][-1])] = sla_weight
        for t in range(horizon):
            for z in range(4):
                x, c, d, pv, g, u = (int(v[name][t, z]) for name in ("compute", "charge", "discharge", "pv_used", "grid", "unserved"))
                mode, saturated = int(v["charging_mode"][t, z]), int(v["import_saturated"][t, z])
                m.constraint({c: 1, mode: -power[z]}, upper=0)
                m.constraint({d: 1, mode: power[z]}, upper=power[z])
                m.constraint({u: 1, saturated: -max_unserved[t, z]}, upper=0)
                m.constraint({g: 1, saturated: -(imports[z] + exports[z])}, lower=-exports[z])
                rhs = data["idle_power"][z] * data["pue"][t, z]
                m.constraint({g: 1, u: 1, x: -data["work_power_per_unit"][z] * data["pue"][t, z],
                              c: -1, d: 1, pv: 1}, lower=rhs, upper=rhs)
                soc_row = {int(v["soc"][t, z]): 1, c: -dt * data["charge_efficiency"][z],
                           d: dt / data["discharge_efficiency"][z]}
                queue_row = {int(v["queue"][t, z]): 1, x: 1}
                soc_rhs = initial[z] if t == 0 else 0.0
                queue_rhs = data["arrivals"][t, z] + (data["initial_queue"][z] if t == 0 else 0.0)
                if t:
                    soc_row[int(v["soc"][t - 1, z])] = -1
                    queue_row[int(v["queue"][t - 1, z])] = -1
                m.constraint(soc_row, lower=soc_rhs, upper=soc_rhs)
                m.constraint(queue_row, lower=queue_rhs, upper=queue_rhs)
            e = int(v["cumulative_expiry"][t])
            if t:
                m.constraint({e: 1, int(v["cumulative_expiry"][t - 1]): -1}, lower=0)
            due_index = t - int(self.model.config.deadline_intervals) + 1
            due = (float(data["initial_queue"].sum() + data["arrivals"][:due_index + 1].sum())
                   if due_index >= 0 else 0.0)
            row = {int(index): 1.0 for index in v["compute"][:t + 1].ravel()}
            row[e] = 1.0
            m.constraint(row, lower=due)
        for z in range(4):
            m.constraint({int(v["discharge"][t, z]): dt / data["discharge_efficiency"][z] for t in range(horizon)},
                         upper=initial[z])
            m.constraint({int(v["charge"][t, z]): dt * data["charge_efficiency"][z] for t in range(horizon)},
                         upper=energy[z] - initial[z])
        # Curtailment cost = available PV - used PV. No terminal-SOC penalty,
        # extra reserve, risk scenario, or realized trajectory enters this LP.
        return m, v, float(data["pv"].sum() * dt)

    def _solve(self, model):
        # SciPy 1.17 forwards threads verbatim to HiGHS; suppress only the
        # corresponding wrapper notice, not solver failure diagnostics.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads.*", category=RuntimeWarning)
            return milp(np.asarray(model.cost), integrality=np.asarray(model.integer),
                        bounds=Bounds(model.lower, model.upper),
                        constraints=LinearConstraint(model.matrix(), model.row_lower, model.row_upper),
                        options={"disp": False, "presolve": True, "time_limit": self.time_limit_seconds,
                                 "mip_rel_gap": self.mip_gap, "threads": self.threads})

    def _raw_plan(self, plan, data, action_range):
        eps = self.inverse_epsilon
        def ratio(value, scale):
            return np.divide(value, scale, out=np.zeros_like(value), where=np.asarray(scale) > 0)
        def logit(value):
            value = np.clip(value, eps, 1 - eps)
            return np.log(value / (1 - value))
        battery = plan["discharge"] - plan["charge"]
        raw = np.zeros((battery.shape[0], 36), dtype=np.float32)
        cap = float(self.model.config.action_residual_fraction) * data["grid_import_limit"]
        raw[:, :4] = np.arctanh(np.clip(ratio(plan["day_ahead"] - data["previous_da"], cap), -1 + eps, 1 - eps)) / action_range
        raw[:, 4:8] = np.arctanh(np.clip(ratio(battery, data["battery_power_limit"]), -1 + eps, 1 - eps)) / action_range
        raw[:, 8:12] = logit(ratio(plan["compute"], data["compute_capacity"])) / action_range
        pv_fraction = np.divide(plan["pv_used"], data["pv"], out=np.ones_like(plan["pv_used"]), where=data["pv"] > 0)
        raw[:, 28:32] = (logit(pv_fraction) - 2.0) / action_range
        raw[:, 32:36] = logit(ratio(plan["soc"], data["battery_energy_capacity"])) / action_range
        return raw

    def decide(self, packet: W1ExpertPacket, risk: RiskConditionPacket) -> RECMoEOutput:
        batch, timeline = packet.state_latent.shape[:2]
        risk.validate(batch, timeline)
        raw = _safe_raw(packet)
        self.last_solver_records, self.last_plans = [], []
        for index in range(batch):
            start = time.perf_counter()
            try:
                data, origin, horizon, stage, action_range = self._inputs(packet, risk, index)
                formulation, variables, offset = self._build(data, horizon, stage)
            except Exception as exc:
                self.last_solver_records.append({"sample_index": index, "solver": self.backend,
                    "status": "invalid_causal_input", "error": str(exc), "fallback_used": False})
                raise
            solver_start = time.perf_counter()
            error = None
            try:
                result = self._solve(formulation)
            except Exception as exc:
                result, error = None, f"{type(exc).__name__}: {exc}"
            solve_seconds = time.perf_counter() - solver_start
            solution = None if result is None else getattr(result, "x", None)
            has_incumbent = bool(solution is not None and np.isfinite(solution).all())
            record = {"sample_index": index, "solver": self.backend, "stage_id": stage,
                "decision_origin": origin, "horizon": horizon, "status_code": None if result is None else int(result.status),
                "status": error if result is None else str(result.message),
                "optimal": bool(result is not None and result.status == 0),
                "has_incumbent": has_incumbent, "fallback_used": not has_incumbent,
                "solve_seconds": solve_seconds, "time_limit_seconds": self.time_limit_seconds,
                "requested_mip_gap": self.mip_gap, "threads": self.threads,
                "variables": len(formulation.cost), "integer_variables": sum(formulation.integer),
                "constraints": len(formulation.row_lower), "forecast_source": "frozen_W1_causal_reference",
                "realized_control_inputs_used": False, "terminal_soc_reserve_enforced": False,
                "gurobi_capacity_probe": "2001 variables rejected: size_limited_license_error_10010"}
            plan = None
            if has_incumbent:
                plan = {name: np.asarray(solution)[ids].copy() for name, ids in variables.items()}
                raw["joined"][index, origin:origin + horizon] = torch.as_tensor(
                    self._raw_plan(plan, data, action_range), device=packet.state_latent.device)
                activity = formulation.matrix() @ solution
                violation = max(float(np.maximum(np.asarray(formulation.row_lower) - activity, 0).max(initial=0)),
                                float(np.maximum(activity - np.asarray(formulation.row_upper), 0).max(initial=0)))
                record.update(objective=_finite_number(result.fun + offset),
                    objective_bound=_finite_number(None if getattr(result, "mip_dual_bound", None) is None else result.mip_dual_bound + offset),
                    relative_gap=_finite_number(getattr(result, "mip_gap", None)),
                    mip_node_count=_finite_number(getattr(result, "mip_node_count", None)),
                    plan_constraint_max_violation=violation)
            else:
                record.update(objective=None, objective_bound=None, relative_gap=None,
                              fallback_policy="registered_zero_raw_safe_tensor")
            record["decision_wall_seconds"] = time.perf_counter() - start
            self.last_solver_records.append(record)
            self.last_plans.append(plan)
        assembly_start = time.perf_counter()
        assembled = self.model.assembler(raw, packet, risk)
        output = _complete_output(self.model, packet, risk, assembled, None, None, None,
                                  torch.zeros_like(assembled.decision_active_mask), None)
        if packet.state_latent.device.type == "cuda":
            torch.cuda.synchronize(packet.state_latent.device)
        assembly_seconds = time.perf_counter() - assembly_start
        for index, (record, plan) in enumerate(zip(self.last_solver_records, self.last_plans)):
            record["assembly_seconds_per_sample"] = assembly_seconds / batch
            record["decision_wall_seconds"] += assembly_seconds / batch
            if plan is None:
                continue
            origin, stop = record["decision_origin"], record["decision_origin"] + record["horizon"]
            pairs = {"battery": (output.actions.battery, plan["discharge"] - plan["charge"]),
                     "compute": (output.actions.compute, plan["compute"]),
                     "pv_used": (output.actions.pv_used, plan["pv_used"]),
                     "grid": (output.actions.grid, plan["grid"]),
                     "unserved": (output.actions.unserved_power, plan["unserved"]),
                     "day_ahead": (output.day_ahead_commitment, plan["day_ahead"])}
            for name, (actual, predicted) in pairs.items():
                record[f"plan_assembly_max_abs_{name}"] = float(np.max(np.abs(_array(actual, index)[origin:stop] - predicted)))
        return replace(output, diagnostics={**output.diagnostics, "controller": "deterministic_frozen_W1_MPC",
            "solver": self.backend, "external_solver": True, "candidate_search": False,
            "independent_suffix_replay": True, "closed_loop_rollout": False,
            "predicted_risk_values_available": False, "predicted_cost_values_available": False,
            "fallback_sample_count": sum(record["fallback_used"] for record in self.last_solver_records)})


__all__ = ["MPCInputError", "W1MPCController"]
