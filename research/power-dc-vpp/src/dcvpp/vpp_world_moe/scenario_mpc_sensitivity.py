"""Finite-window SMPC20 with an external expected grid-sensitivity penalty.

The frozen ScenarioMPC20 implementation is unchanged. This subclass keeps its
persistent 96-period topology, scenario-cost/CVaR definition and solver budget,
but limits flows/costs/deadlines to one planning window. Historical cumulative
due/service/expiry and the preceding two actual grid vectors are explicit inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from .scenario_mpc import NODES, PERIODS, SCENARIOS, ScenarioMPC20


GRID_WEIGHTS = (0.0, 0.1, 1.0, 10.0)
FORECAST_NAMES = {"da", "rt", "arrivals", "pv", "pue"}


def _vector(value, label):
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (NODES,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite four-vector")
    return result


def grid_penalty_terms(active_grid, previous_grid, previous_previous_grid, grid_range):
    """Compute the declared ramp and positive recovery-curvature proxies.

    Input is [S,H,4] or [H,4], with only the active planned trajectory. This
    curvature proxy is not the historical V-shaped rebound metric.
    """
    grid = np.asarray(active_grid, dtype=np.float64)
    if grid.ndim == 2:
        grid = grid[None]
    if grid.ndim != 3 or grid.shape[-1] != NODES or grid.shape[1] < 1 or not np.isfinite(grid).all():
        raise ValueError("active_grid must be finite [S,H,4] or [H,4]")
    prior = _vector(previous_grid, "previous_grid")
    prior2 = _vector(previous_previous_grid, "previous_previous_grid")
    scale = _vector(grid_range, "grid_range")
    if (scale <= 0).any():
        raise ValueError("grid normalization requires positive import-plus-export range at every node")
    history1 = np.broadcast_to(prior, (grid.shape[0], 1, NODES))
    history2 = np.broadcast_to(prior2, (grid.shape[0], 1, NODES))
    joined = np.concatenate((history2, history1, grid), axis=1)
    ramps = np.abs(np.diff(joined[:, 1:], axis=1)) / scale[None, None]
    aggregate = joined.sum(axis=-1)
    curvature = np.maximum(aggregate[:, 2:] - 2 * aggregate[:, 1:-1] + aggregate[:, :-2], 0)
    ramp_mean = ramps.mean(axis=(1, 2))
    curvature_mean = curvature.mean(axis=1) / float(scale.sum())
    return {"ramp_by_scenario": ramp_mean, "recovery_curvature_by_scenario": curvature_mean,
            "total_by_scenario": ramp_mean + curvature_mean}


class ScenarioMPCSensitivity(ScenarioMPC20):
    """H16 by default, with lambda in the four prespecified grid weights."""

    name = "W1-SMPC-20-grid-sensitivity"

    def __init__(self, *, train_cost_scale, grid_weight=0.0, planning_horizon=16, **kwargs):
        if not math.isfinite(float(train_cost_scale)) or float(train_cost_scale) <= 0:
            raise ValueError("train_cost_scale must be a positive train-frozen raw daily cost scale")
        if float(grid_weight) not in GRID_WEIGHTS:
            raise ValueError("grid_weight must be one of the prespecified 0, 0.1, 1, 10")
        if isinstance(planning_horizon, bool) or int(planning_horizon) != planning_horizon or not 1 <= planning_horizon <= PERIODS:
            raise ValueError("planning_horizon must be an integer in [1,96]")
        if float(kwargs.get("time_limit_seconds", 600.0)) != 600.0:
            raise ValueError("the sensitivity study retains the 600-second per-call solver budget")
        super().__init__(**kwargs)
        self.train_cost_scale = float(train_cost_scale)
        self.grid_weight = float(grid_weight)
        self.planning_horizon = int(planning_horizon)
        self._window: dict[str, Any] | None = None

    def _build_topology(self):
        super()._build_topology()
        m, v, r = self.topology, self.variables, self.row_groups
        v["absolute_grid_ramp"] = m.variables((SCENARIOS, PERIODS, NODES))
        v["positive_recovery_curvature"] = m.variables((SCENARIOS, PERIODS))
        r["grid_ramp_positive"] = np.empty((SCENARIOS, PERIODS, NODES), dtype=np.int32)
        r["grid_ramp_negative"] = np.empty((SCENARIOS, PERIODS, NODES), dtype=np.int32)
        r["grid_recovery_curvature"] = np.empty((SCENARIOS, PERIODS), dtype=np.int32)
        for s in range(SCENARIOS):
            for t in range(PERIODS):
                curvature_terms = [(v["positive_recovery_curvature"][s, t], 1)]
                for z in range(NODES):
                    ramp, grid = v["absolute_grid_ramp"][s, t, z], v["grid"][s, t, z]
                    positive = [(ramp, 1), (grid, -1)]
                    negative = [(ramp, 1), (grid, 1)]
                    if t:
                        positive.append((v["grid"][s, t - 1, z], 1))
                        negative.append((v["grid"][s, t - 1, z], -1))
                    r["grid_ramp_positive"][s, t, z] = m.row(positive)
                    r["grid_ramp_negative"][s, t, z] = m.row(negative)
                    curvature_terms.append((grid, -1))
                    if t:
                        curvature_terms.append((v["grid"][s, t - 1, z], 2))
                    if t > 1:
                        curvature_terms.append((v["grid"][s, t - 2, z], -1))
                r["grid_recovery_curvature"][s, t] = m.row(curvature_terms)
        # Append once, before the parent's first passModel; subsequent calls
        # update coefficients/bounds in the same native model instance.
        self._rows = np.asarray(m.rows, dtype=np.int32)
        self._cols = np.asarray(m.cols, dtype=np.int32)
        self._static_values = np.asarray(m.values, dtype=np.float64)
        self._dynamic = {name: np.asarray(items, dtype=np.int64).T for name, items in m.dynamic.items()}
        self._col_indices = np.arange(m.ncols, dtype=np.int32)
        self._row_indices = np.arange(len(m.lb), dtype=np.int32)

    def _parameters(self, data, weights, c, previous, origin, stage):
        parameters = super()._parameters(data, weights, c, previous, origin, stage)
        cost, lower, upper, rl, ru, coefficients = parameters
        v, r, window = self.variables, self.row_groups, self._window
        if window is None:
            raise RuntimeError("solve must establish the finite planning window")
        stop, horizon = window["stop"], window["horizon"]
        active = (np.arange(PERIODS) >= origin) & (np.arange(PERIODS) < stop)
        inactive = ~active
        # Flows stop, while states remain free to persist after stop. Their
        # recurrences then keep SOC/queue/service flat without terminal targets.
        for name in ("compute", "charge", "discharge", "pv_used", "grid", "unserved", "charging_mode", "import_saturated"):
            lower[v[name][:, inactive]] = 0
            upper[v[name][:, inactive]] = 0
        rl[r["balance"][:, inactive]] = 0
        ru[r["balance"][:, inactive]] = 0
        lower[v["day_ahead"][inactive]] = previous[inactive]
        upper[v["day_ahead"][inactive]] = previous[inactive]
        # No deadlines accrue after the planning window. Cumulative new expiry
        # is explicitly held flat, not forced to zero by post-stop bounds.
        rl[r["expiry_due"][:, inactive]] = 0
        ru[r["expiry_monotonic"][:, stop:]] = 0
        if window["known_due"] is not None:
            arrivals = data["arrivals"].sum(axis=2)
            cumulative = np.cumsum(arrivals, axis=1)
            future_due = np.zeros_like(cumulative)
            future_due[:, self.deadline - 1:] = cumulative[:, :PERIODS - self.deadline + 1]
            due_rhs = window["known_due"][None] + future_due - window["served_before"] - window["expired_before"]
            rl[r["expiry_due"][:, active]] = due_rhs[:, active]
        window["due_rhs"] = rl[r["expiry_due"]].copy()
        window["normalized_cost_scale"] = self.train_cost_scale * horizon / PERIODS
        grid_range = c["grid_import_limit"] + c["grid_export_limit"]
        if (grid_range <= 0).any():
            raise ValueError("grid normalization requires positive import-plus-export range at every node")
        for name in ("absolute_grid_ramp", "positive_recovery_curvature"):
            upper[v[name]] = 0
        if self.grid_weight > 0:
            upper[v["absolute_grid_ramp"][:, active]] = np.inf
            upper[v["positive_recovery_curvature"][:, active]] = np.inf
            for name in ("grid_ramp_positive", "grid_ramp_negative", "grid_recovery_curvature"):
                rl[r[name][:, active]] = 0
            prior, prior2 = window["previous_grid"], window["previous_previous_grid"]
            rl[r["grid_ramp_positive"][:, origin]] = -prior
            rl[r["grid_ramp_negative"][:, origin]] = prior
            rl[r["grid_recovery_curvature"][:, origin]] = -2 * prior.sum() + prior2.sum()
            if origin + 1 < stop:
                rl[r["grid_recovery_curvature"][:, origin + 1]] = prior.sum()
            factor = self.grid_weight * window["normalized_cost_scale"]
            cost[v["absolute_grid_ramp"][:, active]] = (
                factor * weights[:, None, None] / (horizon * NODES * grid_range[None, None])
            )
            cost[v["positive_recovery_curvature"][:, active]] = factor * weights[:, None] / (horizon * grid_range.sum())
        window["parameter_coefficients"] = coefficients
        return parameters

    def solve(self, scenarios, weights, constraints, previous_da, origin, stage=1, *, horizon=None,
              previous_grid=None, previous_previous_grid=None, initial_due_profile=None,
              known_cumulative_due_profile=None, served_before=0.0, expired_before=0.0):
        """Optimize one finite window and return only the first executable control.

        known_cumulative_due_profile[t] is cumulative ORIGINAL workload from
        observed cohorts due by absolute slot t, including the original initial
        backlog. It excludes forecast arrivals and has not subtracted service or
        expired workload. served_before and expired_before share that ledger's
        origin and contain only completed earlier executions. initial_due_profile
        is a compatibility alias for the same cumulative (not incremental) array.
        """
        started = time.perf_counter()
        if isinstance(origin, bool) or int(origin) != origin or not 0 <= origin < PERIODS:
            raise ValueError("origin must be an integer in [0,95]")
        requested = self.planning_horizon if horizon is None else horizon
        if isinstance(requested, bool) or int(requested) != requested or not 1 <= requested <= PERIODS:
            raise ValueError("horizon must be an integer in [1,96]")
        origin = int(origin)
        stop = min(PERIODS, origin + int(requested))
        if set(scenarios) != FORECAST_NAMES:
            raise ValueError("provide only causal joint scenarios; no current or future realized target fields")
        # Discard unplanned time slots BEFORE numeric/support validation. Even
        # a poison value beyond stop cannot enter balances, costs or deadlines.
        planned = {}
        for name in FORECAST_NAMES:
            raw = np.asarray(scenarios[name], dtype=np.float64)
            if raw.shape != (SCENARIOS, PERIODS, NODES):
                raise ValueError("joint scenario topology remains [20,96,4]")
            value = np.ones_like(raw) if name == "pue" else np.zeros_like(raw)
            value[:, origin:stop] = raw[:, origin:stop]
            planned[name] = value
        if initial_due_profile is not None and known_cumulative_due_profile is not None:
            raise ValueError("provide one cumulative due profile, not both alias names")
        raw_due = known_cumulative_due_profile if known_cumulative_due_profile is not None else initial_due_profile
        due = None if raw_due is None else np.asarray(raw_due, dtype=np.float64).copy()
        if due is not None and (due.shape != (PERIODS,) or not np.isfinite(due).all() or (due < 0).any() or (np.diff(due) < -1e-10).any()):
            raise ValueError("known due profile must be finite nonnegative cumulative [96]")
        served, expired = float(served_before), float(expired_before)
        if not all(math.isfinite(value) and value >= 0 for value in (served, expired)):
            raise ValueError("historical service and peak expiry must be nonnegative finite scalars")
        if due is None and (served != 0 or expired != 0):
            raise ValueError("nonzero historical ledgers require a known cumulative due profile")
        if self.grid_weight > 0 and (previous_grid is None or previous_previous_grid is None):
            raise ValueError("positive grid weight requires the preceding two observed grid vectors")
        prior = np.zeros(NODES) if previous_grid is None else _vector(previous_grid, "previous_grid")
        prior2 = np.zeros(NODES) if previous_previous_grid is None else _vector(previous_previous_grid, "previous_previous_grid")
        self._window = {"origin": origin, "stop": stop, "horizon": stop - origin,
                        "previous_grid": prior, "previous_previous_grid": prior2,
                        "known_due": due, "served_before": served, "expired_before": expired}
        result = super().solve(planned, weights, constraints, previous_da, origin, stage)
        stats = result["solverstats"]
        window = self._window
        stats.update(active_horizon=stop - origin, planning_horizon_requested=int(requested), planning_stop_exclusive=stop,
                     grid_weight=self.grid_weight, train_frozen_raw_daily_cost_scale=self.train_cost_scale,
                     active_window_normalization_denominator=window["normalized_cost_scale"],
                     grid_penalty_outside_cost_cvar=True, grid_recovery_metric="positive_aggregate_second_difference_not_V_shaped_rebound",
                     state_continuation_after_stop="flat_SOC_queue_service_and_incremental_expiry; no_added_terminal_target",
                     no_cost_or_additional_deadline_after_stop=True,
                     due_history_mode="caller_observed_cumulative_cohort_ledger" if due is not None else "legacy_initial_queue_treated_as_new_at_origin_for_regression_only",
                     historical_served_before=served, historical_peak_expired_before=expired,
                     previous_actual_grid=prior.tolist(), previous_previous_actual_grid=prior2.tolist(),
                     known_historical_inputs_used=True, current_or_future_realized_outcomes_used=False,
                     cumulative_expiry_represents="new_peak_proxy_exceedance_beyond_expired_before")
        digest = hashlib.sha256()
        for name in sorted(planned):
            digest.update(name.encode()); digest.update(planned[name][:, origin:stop].tobytes())
        digest.update(json.dumps({"origin": origin, "stop": stop, "grid_weight": self.grid_weight,
                                 "served_before": served, "expired_before": expired}, sort_keys=True).encode())
        digest.update(prior.tobytes()); digest.update(prior2.tobytes())
        if due is not None:
            digest.update(due[origin:stop].tobytes())
        stats["active_forecast_and_history_sha256"] = digest.hexdigest()
        if stats["accepted"]:
            values = np.asarray(self.highs.getSolution().col_value, dtype=np.float64)
            grid = values[self.variables["grid"]][:, origin:stop]
            grid_range = np.asarray(constraints["grid_import_limit"]) + np.asarray(constraints["grid_export_limit"])
            components = grid_penalty_terms(grid, prior, prior2, grid_range)
            probability = np.asarray(stats["scenario_probabilities"], dtype=np.float64)
            expected_grid = float(probability @ components["total_by_scenario"])
            grid_cost = self.grid_weight * window["normalized_cost_scale"] * expected_grid
            economic = stats["expected_scenario_cost"] + self.risk_weight * stats["exact_weighted_scenario_cvar95"]
            stats.update(expected_grid_penalty=expected_grid,
                         expected_normalized_absolute_ramp=float(probability @ components["ramp_by_scenario"]),
                         expected_normalized_recovery_curvature=float(probability @ components["recovery_curvature_by_scenario"]),
                         grid_penalty_raw_objective=grid_cost, economic_cost_cvar_objective=economic,
                         combined_objective_recomputed=economic + grid_cost,
                         combined_objective_abs_error=abs(stats["objective"] - economic - grid_cost),
                         predicted_new_expiry=float(probability @ values[self.variables["cumulative_expiry"]][:, stop - 1]))
            if self.grid_weight > 0:
                ramp = values[self.variables["absolute_grid_ramp"]][:, origin:stop]
                curvature = values[self.variables["positive_recovery_curvature"]][:, origin:stop]
                epigraph = (ramp / grid_range[None, None]).mean(axis=(1, 2)) + curvature.mean(axis=1) / grid_range.sum()
                stats["expected_grid_penalty_from_epigraph"] = float(probability @ epigraph)
                stats["grid_epigraph_vs_direct_abs_error"] = abs(float(probability @ epigraph) - expected_grid)
            else:
                stats["expected_grid_penalty_from_epigraph"] = None
            result["grid_penalty_by_scenario"] = components
            result["plan0"]["predicted_new_expiry"] = result["plan0"]["cumulative_expiry"].copy()
        stats["sensitivity_total_wall_seconds"] = time.perf_counter() - started
        self.last_solverstats = stats
        return result


__all__ = ["ScenarioMPCSensitivity", "grid_penalty_terms", "GRID_WEIGHTS"]
