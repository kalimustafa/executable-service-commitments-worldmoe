from __future__ import annotations

"""Persistent Pyomo + APPSI Gurobi verification for E2E-MoE R3.

The algebraic model is deliberately fixed at 96 quarter-hours and 20 joint
scenarios.  A runtime owns exactly one APPSI Gurobi object, calls
``set_instance`` exactly once, and subsequently changes mutable parameters and
executed-state variable bounds only.  No user MIP start is accepted.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Gurobi

from dcvpp.optimization.replay_engine import settle_realized_components
from dcvpp.research_branch.contracts import FrozenPhysicalArtifact, NYISO_ZONES


INTERVALS = 96
SCENARIOS = 20
INTERVAL_HOURS = 0.25
ALLOWED_YEARS = {2022, 2023, 2024}


class GurobiBlockedError(RuntimeError):
    """Raised when Gurobi import, installation, or licensing is unavailable."""


class R3DataError(ValueError):
    """Raised when a selection input violates the locked-period contract."""


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _year(date_text: str) -> int:
    try:
        value = datetime.fromisoformat(str(date_text)[:10]).year
    except ValueError as exc:
        raise R3DataError(f"invalid operating date: {date_text}") from exc
    if value not in ALLOWED_YEARS:
        raise R3DataError(
            f"R3 selection inputs are closed outside 2022-2024; rejected {date_text}"
        )
    return value


@dataclass(frozen=True)
class SelectionDay:
    operating_date: str
    timestamps_utc: np.ndarray
    da_price: np.ndarray
    rt_price_pool: np.ndarray
    workload_pool: np.ndarray
    pv_pool: np.ndarray
    realized_workload: np.ndarray
    realized_pv: np.ndarray
    physical_multiplier: np.ndarray
    initial_soc_mwh: np.ndarray
    source_hashes: Mapping[str, str]
    locked_test: bool
    migration_actions_enabled: bool = False
    realized_da_price: np.ndarray | None = None
    realized_rt_price: np.ndarray | None = None

    def validate(self) -> None:
        _year(self.operating_date)
        if self.locked_test:
            raise R3DataError("2025 locked-test input is forbidden in R3 selection work")
        if self.migration_actions_enabled:
            raise R3DataError(
                "this R3 selection adapter requires the audited battery-only "
                "bundle contract; migration-enabled inputs need migration algebra"
            )
        shapes = {
            "timestamps_utc": (INTERVALS,),
            "da_price": (len(NYISO_ZONES), INTERVALS),
            "rt_price_pool": (100, len(NYISO_ZONES), INTERVALS),
            "workload_pool": (100, INTERVALS),
            "pv_pool": (100, len(NYISO_ZONES), INTERVALS),
            "realized_workload": (INTERVALS,),
            "realized_pv": (len(NYISO_ZONES), INTERVALS),
            "physical_multiplier": (len(NYISO_ZONES),),
            "initial_soc_mwh": (len(NYISO_ZONES),),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise R3DataError(f"{name} must have shape {shape}, got {value.shape}")
            if name != "timestamps_utc" and not np.isfinite(value.astype(float)).all():
                raise R3DataError(f"{name} contains non-finite values")
        for name in ("realized_da_price", "realized_rt_price"):
            realized_price = getattr(self, name)
            if realized_price is None:
                continue
            realized = np.asarray(realized_price, dtype=float)
            expected_rt = (len(NYISO_ZONES), INTERVALS)
            if realized.shape != expected_rt or not np.isfinite(realized).all():
                raise R3DataError(
                    f"{name} must be finite with shape {expected_rt}"
                )


def load_selection_day(
    path: Path | str,
    *,
    realized_settlement_path: Path | str | None = None,
) -> SelectionDay:
    """Load the existing counterfactual-input seam without materializing 2025."""

    path = Path(path)
    # Reject by path before opening the file.  This is part of the 2025 access gate.
    for part in path.parts:
        if part == "2025" or part.startswith("2025-"):
            raise R3DataError(f"refusing to open locked 2025 path: {path}")
    with np.load(path, allow_pickle=False) as data:
        operating_date = str(data["operating_date"].item())
        _year(operating_date)
        locked = bool(data["locked_test"].item())
        if locked:
            raise R3DataError(f"refusing to read locked-test bundle: {path}")
        source_hashes = json.loads(str(data["source_hashes_json"].item()))
        actionability = (
            json.loads(str(data["actionability_policy_json"].item()))
            if "actionability_policy_json" in data.files
            else {}
        )
        realized_da_price: np.ndarray | None = None
        realized_rt_price: np.ndarray | None = None
        if realized_settlement_path is not None:
            settlement_path = Path(realized_settlement_path)
            for part in settlement_path.parts:
                if part == "2025" or part.startswith("2025-"):
                    raise R3DataError(
                        f"refusing to open locked 2025 settlement path: {settlement_path}"
                    )
            with np.load(settlement_path, allow_pickle=False) as settlement:
                if not np.array_equal(
                    np.asarray(settlement["timestamps_utc"]),
                    np.asarray(data["timestamps_utc"]),
                ):
                    raise R3DataError("settlement timestamps do not match selection day")
                realized_da_price = np.asarray(
                    settlement["da_lbmp"], dtype=float
                ).copy()
                realized_rt_price = np.asarray(
                    settlement["rt_lbmp"], dtype=float
                ).copy()
        day = SelectionDay(
            operating_date=operating_date,
            timestamps_utc=np.asarray(data["timestamps_utc"]).copy(),
            da_price=np.asarray(data["da_price"], dtype=float).copy(),
            rt_price_pool=np.asarray(data["rt_price_pool"], dtype=float).copy(),
            workload_pool=np.asarray(data["workload_pool"], dtype=float).copy(),
            pv_pool=np.asarray(data["pv_pool"], dtype=float).copy(),
            realized_workload=np.asarray(data["workload"], dtype=float).copy(),
            realized_pv=np.asarray(data["pv"], dtype=float).copy(),
            physical_multiplier=np.asarray(data["physical_multiplier"], dtype=float).copy(),
            initial_soc_mwh=np.asarray(data["initial_soc_mwh_by_zone"], dtype=float).copy(),
            source_hashes=source_hashes,
            locked_test=locked,
            migration_actions_enabled=bool(
                actionability.get("migration_actions_enabled", False)
            ),
            realized_da_price=realized_da_price,
            realized_rt_price=realized_rt_price,
        )
    day.validate()
    return day


def load_frozen_physical(path: Path | str) -> FrozenPhysicalArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # The canonical artifact has a boolean field named ``frozen``; wrapping it
    # avoids the legacy loader interpreting that flag as the artifact payload.
    return FrozenPhysicalArtifact.from_mapping({"frozen_artifact": payload})


def _tiny_model() -> pyo.ConcreteModel:
    model = pyo.ConcreteModel(name="r3_gurobi_preflight")
    model.x = pyo.Var(domain=pyo.Binary)
    model.y = pyo.Var(bounds=(0.0, None))
    model.cover = pyo.Constraint(expr=model.y >= 2.0 - model.x)
    model.objective = pyo.Objective(expr=model.y + 0.1 * model.x)
    return model


def preflight_gurobi(*, logfile: Path | str | None = None) -> dict[str, Any]:
    """Test the mandated APPSI Gurobi backend with one tiny MIP."""

    started = time.perf_counter()
    solver = Gurobi()
    availability = str(solver.available())
    result: dict[str, Any] = {
        "backend": "pyomo.contrib.appsi.solvers.Gurobi",
        "availability": availability,
        "license_valid": False,
        "tiny_model_solved": False,
        "set_instance_calls": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if availability.lower() not in {"fulllicense", "limitedlicense"}:
        result.update(
            status="BLOCKED",
            reason=f"APPSI Gurobi unavailable: {availability}",
            elapsed_seconds=time.perf_counter() - started,
        )
        return result
    try:
        model = _tiny_model()
        # Gurobi 13 on Windows cannot open a log path containing non-ASCII
        # workspace characters.  Keep the required run log ourselves instead
        # of letting an otherwise valid solve fail while opening that path.
        solver.config.time_limit = 30.0
        solver.config.mip_gap = 0.0
        solver.gurobi_options["Threads"] = 1
        solver.set_instance(model)
        result["set_instance_calls"] = 1
        solved = solver.solve(model)
        result.update(
            status="COMPLETE",
            license_valid=True,
            tiny_model_solved=solved.termination_condition
            == TerminationCondition.optimal,
            termination_condition=str(solved.termination_condition),
            objective=float(pyo.value(model.objective)),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # licensing errors surface while Model is constructed
        result.update(
            status="BLOCKED",
            reason=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
    if logfile:
        Path(logfile).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _build_fixed_model(physical: FrozenPhysicalArtifact) -> pyo.ConcreteModel:
    z_count = len(NYISO_ZONES)
    model = pyo.ConcreteModel(name="e2e_moe_r3_96x20")
    model.T = pyo.RangeSet(0, INTERVALS - 1)
    model.S = pyo.RangeSet(0, SCENARIOS - 1)
    model.Z = pyo.RangeSet(0, z_count - 1)
    model.origin = pyo.Param(initialize=0, mutable=True, within=pyo.NonNegativeIntegers)
    model.mode_guided = pyo.Param(initialize=1.0, mutable=True)
    model.deviation_weight = pyo.Param(initialize=1.0, mutable=True)
    model.scenario_weight = pyo.Param(
        model.S, initialize=1.0 / SCENARIOS, mutable=True, within=pyo.NonNegativeReals
    )
    model.da_price = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.rt_price = pyo.Param(model.S, model.Z, model.T, initialize=0.0, mutable=True)
    model.pv = pyo.Param(model.S, model.Z, model.T, initialize=0.0, mutable=True)
    model.workload_release = pyo.Param(model.S, model.T, initialize=0.0, mutable=True)
    model.cumulative_workload_due = pyo.Param(
        model.S, model.T, initialize=0.0, mutable=True
    )
    model.physical_multiplier = pyo.Param(model.S, model.Z, initialize=1.0, mutable=True)
    model.initial_soc = pyo.Param(model.Z, initialize=0.0, mutable=True)
    model.active_period = pyo.Param(
        model.T, initialize=1.0, mutable=True, within=pyo.UnitInterval
    )
    model.past_grid = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.past_work = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.current_origin = pyo.Param(
        model.T, initialize=0.0, mutable=True, within=pyo.UnitInterval
    )
    model.da_frozen = pyo.Param(initialize=0.0, mutable=True, within=pyo.UnitInterval)
    model.frozen_da = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.neural_grid = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.neural_da = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.neural_battery = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.neural_compute = pyo.Param(model.Z, model.T, initialize=0.0, mutable=True)
    model.da_schedule = pyo.Var(
        model.Z, model.T, bounds=lambda m, z, t: (0.0, float(physical.grid_import_limit_mw[z]))
    )
    model.grid = pyo.Var(
        model.S,
        model.Z,
        model.T,
        bounds=lambda m, s, z, t: (0.0, float(physical.grid_import_limit_mw[z])),
    )
    model.pv_used = pyo.Var(model.S, model.Z, model.T, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(
        model.S,
        model.Z,
        model.T,
        bounds=lambda m, s, z, t: (0.0, float(physical.battery_power_mw[z])),
    )
    model.discharge = pyo.Var(
        model.S,
        model.Z,
        model.T,
        bounds=lambda m, s, z, t: (0.0, float(physical.battery_power_mw[z])),
    )
    # One binary mode per site, scenario and interval formally excludes
    # simultaneous charging and discharging.  The variable topology remains
    # fixed for the complete 96x20 rolling sequence; ``active_period`` merely
    # fixes already-executed modes and dispatch to zero.
    model.battery_charge_mode = pyo.Var(
        model.S, model.Z, model.T, domain=pyo.Binary
    )
    model.battery_mode_active = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.battery_charge_mode[s, z, t]
        <= m.active_period[t],
    )
    model.battery_charge_limit = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.charge[s, z, t]
        <= float(physical.battery_power_mw[z])
        * m.battery_charge_mode[s, z, t],
    )
    model.battery_discharge_limit = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.discharge[s, z, t]
        <= float(physical.battery_power_mw[z])
        * (m.active_period[t] - m.battery_charge_mode[s, z, t]),
    )
    model.work = pyo.Var(
        model.S,
        model.Z,
        model.T,
        bounds=lambda m, s, z, t: (0.0, float(physical.capacity_work_units[z])),
    )
    model.unserved = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(
        model.S,
        model.Z,
        pyo.RangeSet(0, INTERVALS),
        bounds=lambda m, s, z, t: (0.0, float(physical.battery_energy_mwh[z])),
    )
    model.dev_pos = pyo.Var(model.S, model.Z, model.T, domain=pyo.NonNegativeReals)
    model.dev_neg = pyo.Var(model.S, model.Z, model.T, domain=pyo.NonNegativeReals)
    model.grid_abs = pyo.Var(model.S, model.Z, model.T, domain=pyo.NonNegativeReals)
    model.battery_abs = pyo.Var(model.S, model.Z, model.T, domain=pyo.NonNegativeReals)
    model.compute_abs = pyo.Var(model.S, model.Z, model.T, domain=pyo.NonNegativeReals)
    model.da_abs = pyo.Var(model.Z, model.T, domain=pyo.NonNegativeReals)

    def energy_rule(m: pyo.ConcreteModel, s: int, z: int, t: int) -> Any:
        load = m.physical_multiplier[s, z] * (
            float(physical.idle_mw[z]) + float(physical.work_mw_per_unit[z]) * m.work[s, z, t]
        )
        # For executed periods ``active=0`` this becomes a harmless fixed-state
        # identity, while the persistent algebraic structure stays unchanged.
        return m.active_period[t] * (
            m.grid[s, z, t] + m.pv_used[s, z, t] + m.discharge[s, z, t]
            - load
            - m.charge[s, z, t]
        ) + (1.0 - m.active_period[t]) * (
            m.grid[s, z, t] - m.past_grid[z, t]
        ) == 0.0

    model.energy_balance = pyo.Constraint(model.S, model.Z, model.T, rule=energy_rule)
    model.pv_bound = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.pv_used[s, z, t]
        <= m.active_period[t] * m.pv[s, z, t],
    )
    model.soc_initial = pyo.Constraint(
        model.S, model.Z, rule=lambda m, s, z: m.soc[s, z, 0] == m.initial_soc[z]
    )
    model.soc_transition = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.soc[s, z, t + 1]
        == m.soc[s, z, t]
        + m.active_period[t]
        * INTERVAL_HOURS
        * (0.95 * m.charge[s, z, t] - m.discharge[s, z, t] / 0.95),
    )
    # Keep an economically defensible reserve without forcing the suffix back
    # to its rolling initial SOC (which becomes impossible after a nonzero
    # executed action).  Half capacity matches the audited bundle initial SOC.
    model.terminal_soc = pyo.Constraint(
        model.S,
        model.Z,
        rule=lambda m, s, z: m.soc[s, z, INTERVALS]
        >= 0.5 * float(physical.battery_energy_mwh[z]),
    )
    model.work_capacity = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: sum(m.work[s, z, t] for z in m.Z)
        <= m.active_period[t] * sum(
            m.workload_release[s, tau] for tau in range(t + 1)
        )
        + (1.0 - m.active_period[t]) * sum(m.past_work[z, t] for z in m.Z),
    )
    model.work_release_limit = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: sum(
            m.past_work[z, tau] for z in m.Z for tau in range(t + 1)
        )
        + sum(
            m.active_period[tau] * m.work[s, z, tau]
            for z in m.Z
            for tau in range(t + 1)
        )
        <= sum(m.workload_release[s, tau] for tau in range(t + 1)),
    )
    model.work_deadline = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: sum(
            m.past_work[z, tau] for z in m.Z for tau in range(t + 1)
        )
        + sum(
            m.active_period[tau] * m.work[s, z, tau]
            for z in m.Z
            for tau in range(t + 1)
        )
        + m.unserved[s, t]
        >= m.cumulative_workload_due[s, t],
    )
    model.unserved_non_decreasing = pyo.Constraint(
        model.S,
        pyo.RangeSet(1, INTERVALS - 1),
        rule=lambda m, s, t: m.unserved[s, t] >= m.unserved[s, t - 1],
    )
    model.settlement = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.active_period[t]
        * (
            m.grid[s, z, t]
            - m.da_schedule[z, t]
            - m.dev_pos[s, z, t]
            + m.dev_neg[s, z, t]
        )
        + (1.0 - m.active_period[t]) * m.grid[s, z, t]
        == (1.0 - m.active_period[t]) * m.past_grid[z, t],
    )
    # Fixed-topology nonanticipativity: ``current_origin`` is a one-hot mutable
    # parameter, so the executed action at every rolling origin (not merely
    # t=0) is shared across all 20 scenarios.
    model.grid_nonanticipative = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: pyo.Constraint.Skip
        if s == 0
        else m.current_origin[t]
        * (m.grid[s, z, t] - m.grid[0, z, t])
        == 0.0,
    )
    model.work_nonanticipative = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: pyo.Constraint.Skip
        if s == 0
        else m.current_origin[t]
        * (m.work[s, z, t] - m.work[0, z, t])
        == 0.0,
    )
    model.charge_nonanticipative = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: pyo.Constraint.Skip
        if s == 0
        else m.current_origin[t]
        * (m.charge[s, z, t] - m.charge[0, z, t])
        == 0.0,
    )
    model.discharge_nonanticipative = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: pyo.Constraint.Skip
        if s == 0
        else m.current_origin[t]
        * (m.discharge[s, z, t] - m.discharge[0, z, t])
        == 0.0,
    )
    model.battery_mode_nonanticipative = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: pyo.Constraint.Skip
        if s == 0
        else m.current_origin[t]
        * (
            m.battery_charge_mode[s, z, t]
            - m.battery_charge_mode[0, z, t]
        )
        == 0.0,
    )
    model.da_freeze = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, z, t: m.da_frozen
        * (m.da_schedule[z, t] - m.frozen_da[z, t])
        == 0.0,
    )
    for name, variable, target in (
        ("grid", model.grid, model.neural_grid),
        ("compute", model.work, model.neural_compute),
    ):
        absolute = getattr(model, f"{name}_abs")
        setattr(
            model,
            f"{name}_dev_pos",
            pyo.Constraint(
                model.S,
                model.Z,
                model.T,
                rule=lambda m, s, z, t, v=variable, a=absolute, q=target: a[s, z, t]
                >= v[s, z, t] - q[z, t],
            ),
        )
        setattr(
            model,
            f"{name}_dev_neg",
            pyo.Constraint(
                model.S,
                model.Z,
                model.T,
                rule=lambda m, s, z, t, v=variable, a=absolute, q=target: a[s, z, t]
                >= q[z, t] - v[s, z, t],
            ),
        )
    # The neural battery action is signed net charging power: positive means
    # charge and negative means discharge.  It is a mutable repair target, not
    # a user MIP start.
    model.battery_dev_pos = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.battery_abs[s, z, t]
        >= m.charge[s, z, t]
        - m.discharge[s, z, t]
        - m.neural_battery[z, t],
    )
    model.battery_dev_neg = pyo.Constraint(
        model.S,
        model.Z,
        model.T,
        rule=lambda m, s, z, t: m.battery_abs[s, z, t]
        >= m.neural_battery[z, t]
        - m.charge[s, z, t]
        + m.discharge[s, z, t],
    )
    model.da_dev_pos = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, z, t: m.da_abs[z, t]
        >= m.da_schedule[z, t] - m.neural_da[z, t],
    )
    model.da_dev_neg = pyo.Constraint(
        model.Z,
        model.T,
        rule=lambda m, z, t: m.da_abs[z, t]
        >= m.neural_da[z, t] - m.da_schedule[z, t],
    )
    def scenario_total_cost_rule(m: pyo.ConcreteModel, s: int) -> Any:
        return sum(
            m.active_period[t]
            * (
                INTERVAL_HOURS
                * sum(
                    m.da_schedule[z, t] * m.da_price[z, t]
                    + m.dev_pos[s, z, t] * m.rt_price[s, z, t]
                    - m.dev_neg[s, z, t] * m.rt_price[s, z, t]
                    for z in m.Z
                )
                + (
                    1000.0 * m.unserved[s, t]
                    if t == INTERVALS - 1
                    else 0.0
                )
                + INTERVAL_HOURS
                * sum(
                    m.charge[s, z, t] + m.discharge[s, z, t]
                    for z in m.Z
                )
            )
            for t in m.T
        )

    model.scenario_total_cost = pyo.Expression(
        model.S, rule=scenario_total_cost_rule
    )
    economic = sum(
        model.scenario_weight[s] * model.scenario_total_cost[s]
        for s in model.S
    )
    model.cvar_eta = pyo.Var(domain=pyo.Reals)
    model.cvar_excess = pyo.Var(model.S, domain=pyo.NonNegativeReals)
    model.cvar_excess_bound = pyo.Constraint(
        model.S,
        rule=lambda m, s: m.cvar_excess[s]
        >= m.scenario_total_cost[s] - m.cvar_eta,
    )
    cvar_alpha = 0.95
    model.formal_scenario_cvar = pyo.Expression(
        expr=model.cvar_eta
        + sum(
            model.scenario_weight[s]
            * model.cvar_excess[s]
            / (1.0 - cvar_alpha)
            for s in model.S
        )
    )
    guided = model.mode_guided * model.deviation_weight * (
        sum(
            model.scenario_weight[s]
            * model.active_period[t]
            * (
                model.grid_abs[s, z, t]
                + model.battery_abs[s, z, t]
                + model.compute_abs[s, z, t]
            )
            for s in model.S
            for z in model.Z
            for t in model.T
        )
        + sum(
            model.active_period[t] * model.da_abs[z, t]
            for z in model.Z
            for t in model.T
        )
    )
    model.objective = pyo.Objective(
        expr=economic + 0.25 * model.formal_scenario_cvar + guided
    )
    return model


def _select_joint_scenarios(day: SelectionDay, indices: np.ndarray) -> dict[str, np.ndarray]:
    if indices.shape != (SCENARIOS,):
        raise ValueError(f"scenario indices must have shape {(SCENARIOS,)}")
    return {
        "rt": day.rt_price_pool[indices],
        "workload": day.workload_pool[indices],
        "pv": day.pv_pool[indices],
        "physical": np.repeat(day.physical_multiplier[None, :], SCENARIOS, axis=0),
    }


def _validate_joint_scenarios(
    day: SelectionDay,
    joint_scenarios: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate solver-facing learned particles without changing their order.

    The neural adapter is responsible for mapping the model's six joint
    variables into the four physical solver arrays below.  This seam preserves
    the learned particle pairing and does not re-sample or re-weight it.
    """

    required = {"rt", "workload", "pv"}
    optional = {"physical"}
    missing = required.difference(joint_scenarios)
    unknown = set(joint_scenarios).difference(required | optional)
    if missing:
        raise ValueError(f"learned joint scenarios are missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown learned joint scenario keys: {sorted(unknown)}")
    z_count = len(NYISO_ZONES)
    expected_shapes = {
        "rt": (SCENARIOS, z_count, INTERVALS),
        "workload": (SCENARIOS, INTERVALS),
        "pv": (SCENARIOS, z_count, INTERVALS),
        "physical": (SCENARIOS, z_count),
    }
    values: dict[str, np.ndarray] = {}
    for name in required:
        value = np.asarray(joint_scenarios[name], dtype=float)
        if value.shape != expected_shapes[name]:
            raise ValueError(
                f"learned joint scenario {name!r} must have shape "
                f"{expected_shapes[name]}, got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"learned joint scenario {name!r} contains non-finite values")
        values[name] = value.copy()
    physical = np.asarray(
        joint_scenarios.get(
            "physical",
            np.repeat(day.physical_multiplier[None, :], SCENARIOS, axis=0),
        ),
        dtype=float,
    )
    if physical.shape != expected_shapes["physical"]:
        raise ValueError(
            "learned joint scenario 'physical' must have shape "
            f"{expected_shapes['physical']}, got {physical.shape}"
        )
    if not np.isfinite(physical).all() or (physical <= 0.0).any():
        raise ValueError("learned physical multipliers must be finite and positive")
    if (values["workload"] < 0.0).any() or (values["pv"] < 0.0).any():
        raise ValueError("learned workload and PV scenarios must be nonnegative")
    values["physical"] = physical.copy()
    return values


class PersistentGurobiVPP:
    """One fixed model and one active Gurobi object for a rolling sequence."""

    def __init__(
        self,
        physical: FrozenPhysicalArtifact,
        solver_config: Mapping[str, Any],
        *,
        logfile: Path | str | None = None,
    ) -> None:
        self.physical = physical
        self.config = dict(solver_config)
        self.logfile = None if logfile is None else Path(logfile)
        self.model: pyo.ConcreteModel | None = None
        self.solver: Gurobi | None = None
        self.model_build_count = 0
        self.set_instance_count = 0
        self.structure_rebuilds: list[dict[str, Any]] = []
        self._executed_grid = np.zeros((len(NYISO_ZONES), INTERVALS), dtype=float)
        self._executed_work = np.zeros((len(NYISO_ZONES), INTERVALS), dtype=float)

    def _write_log(self, message: str) -> None:
        if self.logfile is None:
            return
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        with self.logfile.open("a", encoding="utf-8") as stream:
            stream.write(message.rstrip() + "\n")

    @property
    def available(self) -> bool:
        return str(Gurobi().available()).lower() in {"fulllicense", "limitedlicense"}

    def build(self) -> None:
        if self.model is not None:
            raise RuntimeError("persistent rolling model may be built only once")
        if not self.available:
            raise GurobiBlockedError(f"APPSI Gurobi unavailable: {Gurobi().available()}")
        self.model = _build_fixed_model(self.physical)
        self.model_build_count += 1
        self.solver = Gurobi()
        self.solver.config.time_limit = float(self.config.get("formal_time_limit_seconds", 600))
        self.solver.config.mip_gap = float(self.config.get("formal_target_relative_gap", 0.0))
        if self.logfile is not None:
            self.logfile.parent.mkdir(parents=True, exist_ok=True)
            self.logfile.write_text(
                "APPSI Gurobi persistent runtime initialized; solver-native log "
                "disabled because the Windows workspace path contains Unicode.\n",
                encoding="utf-8",
            )
        self.solver.gurobi_options["Threads"] = int(self.config.get("threads", 1))
        self.solver.set_instance(self.model)
        self.set_instance_count += 1
        self._write_log(
            "model_built=1 set_instance_count=1 intervals=96 scenarios=20 "
            f"variables={sum(1 for _ in self.model.component_data_objects(pyo.Var))} "
            f"constraints={sum(1 for _ in self.model.component_data_objects(pyo.Constraint))}"
        )

    def _update(
        self,
        day: SelectionDay,
        origin: int,
        scenarios: Mapping[str, np.ndarray],
        weights: np.ndarray,
        neural_action: Mapping[str, np.ndarray] | None,
        soc: np.ndarray,
        mode: str,
        frozen_da: np.ndarray | None,
    ) -> None:
        assert self.model is not None
        m = self.model
        if mode not in {"guided_repair", "full_optimize"}:
            raise ValueError(f"unknown Gurobi mode: {mode}")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (SCENARIOS,) or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("exactly 20 nonnegative scenario weights must sum to one")
        m.origin.set_value(origin)
        m.mode_guided.set_value(1.0 if mode == "guided_repair" else 0.0)
        m.da_frozen.set_value(0.0 if frozen_da is None else 1.0)
        guided_cfg = self.config.get("modes", {}).get("guided_repair", {})
        m.deviation_weight.set_value(float(guided_cfg.get("deviation_penalty", 1.0)))
        action = neural_action or {}
        targets = {
            "da": np.asarray(
                action.get("da", np.zeros((len(NYISO_ZONES), INTERVALS))),
                dtype=float,
            ),
            "grid": np.asarray(action.get("grid", np.zeros((len(NYISO_ZONES), INTERVALS))), dtype=float),
            "battery": np.asarray(action.get("battery", np.zeros((len(NYISO_ZONES), INTERVALS))), dtype=float),
            "compute": np.asarray(action.get("compute", np.zeros((len(NYISO_ZONES), INTERVALS))), dtype=float),
        }
        expected_target_shapes = {
            "da": (len(NYISO_ZONES), INTERVALS),
            "grid": (len(NYISO_ZONES), INTERVALS),
            "battery": (len(NYISO_ZONES), INTERVALS),
            "compute": (len(NYISO_ZONES), INTERVALS),
        }
        for name, expected_shape in expected_target_shapes.items():
            if targets[name].shape != expected_shape:
                raise ValueError(
                    f"neural {name} target must have shape {expected_shape}, "
                    f"got {targets[name].shape}"
                )
        for s in range(SCENARIOS):
            m.scenario_weight[s] = float(weights[s])
            for z in range(len(NYISO_ZONES)):
                m.physical_multiplier[s, z] = float(scenarios["physical"][s, z])
                for t in range(INTERVALS):
                    # Past parameter values remain harmless because all past variables are fixed.
                    m.rt_price[s, z, t] = float(scenarios["rt"][s, z, t])
                    m.pv[s, z, t] = float(max(scenarios["pv"][s, z, t], 0.0))
                    if z == 0:
                        release = np.maximum(scenarios["workload"][s], 0.0)
                        m.workload_release[s, t] = float(release[t])
                        deadline_steps = 4
                        due_index = t - deadline_steps
                        m.cumulative_workload_due[s, t] = float(
                            0.0 if due_index < 0 else release[: due_index + 1].sum()
                        )
        for z in range(len(NYISO_ZONES)):
            m.initial_soc[z] = float(soc[z])
            for t in range(INTERVALS):
                m.active_period[t] = 0.0 if t < origin else 1.0
                m.current_origin[t] = 1.0 if t == origin else 0.0
                m.past_grid[z, t] = float(self._executed_grid[z, t])
                m.past_work[z, t] = float(self._executed_work[z, t])
                m.frozen_da[z, t] = (
                    0.0 if frozen_da is None else float(frozen_da[z, t])
                )
                m.da_price[z, t] = float(day.da_price[z, t])
                m.neural_da[z, t] = float(targets["da"][z, t])
                m.neural_grid[z, t] = float(targets["grid"][z, t])
                m.neural_battery[z, t] = float(targets["battery"][z, t])
                m.neural_compute[z, t] = float(max(targets["compute"][z, t], 0.0))
        assert self.solver is not None
        self.solver.update_params()

    def _accept(self, result: Any) -> tuple[bool, float | None, bool]:
        assert self.solver is not None
        termination = result.termination_condition
        incumbent = bool(result.best_feasible_objective is not None)
        gap: float | None = None
        if incumbent and result.best_objective_bound is not None:
            denom = max(abs(float(result.best_feasible_objective)), 1e-9)
            gap = abs(float(result.best_feasible_objective) - float(result.best_objective_bound)) / denom
        accepted = incumbent and (
            termination == TerminationCondition.optimal
            or (
                termination == TerminationCondition.maxTimeLimit
                and gap is not None
                and gap
                <= float(self.config.get("accepted_timeout_relative_gap", 0.05))
            )
        )
        return accepted, gap, incumbent

    def run_day(
        self,
        day: SelectionDay,
        *,
        scenario_indices: np.ndarray | None = None,
        joint_scenarios: Mapping[str, np.ndarray] | None = None,
        scenario_weights: np.ndarray | None = None,
        neural_actions: Mapping[str, np.ndarray] | None = None,
        max_origins: int = INTERVALS,
        formal: bool = False,
    ) -> dict[str, Any]:
        day.validate()
        if self.model is None:
            self.build()
        assert self.model is not None and self.solver is not None
        if joint_scenarios is not None and scenario_indices is not None:
            raise ValueError(
                "provide learned joint_scenarios or legacy scenario_indices, not both"
            )
        if joint_scenarios is None:
            indices = np.asarray(
                np.arange(SCENARIOS) if scenario_indices is None else scenario_indices,
                dtype=int,
            )
            scenarios = _select_joint_scenarios(day, indices)
        else:
            scenarios = _validate_joint_scenarios(day, joint_scenarios)
        # Realized values are revealed only at their origin.
        weights = np.asarray(
            np.full(SCENARIOS, 1.0 / SCENARIOS) if scenario_weights is None else scenario_weights,
            dtype=float,
        )
        if formal and day.realized_rt_price is None:
            raise R3DataError(
                "formal evaluation requires target-day official realized RT prices"
            )
        realized_rt = (
            np.asarray(day.realized_rt_price, dtype=float)
            if day.realized_rt_price is not None
            else np.asarray(day.rt_price_pool[0], dtype=float)
        )
        realized_da = (
            np.asarray(day.realized_da_price, dtype=float)
            if day.realized_da_price is not None
            else np.asarray(day.da_price, dtype=float)
        )
        soc = np.asarray(day.initial_soc_mwh, dtype=float).copy()
        records: list[dict[str, Any]] = []
        da_schedule = np.zeros((len(NYISO_ZONES), INTERVALS), dtype=float)
        frozen_da: np.ndarray | None = None
        total_started = time.perf_counter()
        for origin in range(min(int(max_origins), INTERVALS)):
            scenarios["rt"][:, :, origin] = realized_rt[:, origin]
            scenarios["workload"][:, origin] = day.realized_workload[origin]
            scenarios["pv"][:, :, origin] = day.realized_pv[:, origin]
            scenario_action: dict[str, np.ndarray] = {}
            for key in ("da", "grid", "battery", "compute"):
                raw = None if neural_actions is None else neural_actions.get(key)
                if raw is not None:
                    value = np.asarray(raw, dtype=float)
                    scenario_action[key] = value[origin] if value.ndim == 3 else value
            risk_trigger = bool(
                neural_actions is not None
                and np.asarray(neural_actions.get("fallback", np.zeros(INTERVALS)))[origin] >= 0.5
            )
            # Guided repair is meaningful only when it has an explicit neural
            # target.  Missing neural actions must not silently create a
            # zero-action repair target.
            mode = (
                "full_optimize"
                if neural_actions is None or risk_trigger
                else "guided_repair"
            )
            self._update(
                day,
                origin,
                scenarios,
                weights,
                scenario_action,
                soc,
                mode,
                frozen_da,
            )
            if formal or mode == "full_optimize":
                self.solver.config.time_limit = float(self.config.get("formal_time_limit_seconds", 600))
            else:
                self.solver.config.time_limit = float(
                    self.config.get("modes", {}).get("guided_repair", {}).get("time_limit_seconds", 30)
                )
            started = time.perf_counter()
            try:
                result = self.solver.solve(self.model)
            except Exception as exc:
                message = str(exc)
                if "size-limited license" in message.lower() or "license" in message.lower():
                    self._write_log(
                        f"origin={origin} status=BLOCKED {type(exc).__name__}: {message}"
                    )
                    raise GurobiBlockedError(
                        "Gurobi license cannot solve the mandatory fixed 96-period, "
                        f"20-scenario model: {type(exc).__name__}: {message}"
                    ) from exc
                raise
            solve_seconds = time.perf_counter() - started
            accepted, gap, incumbent = self._accept(result)
            if not accepted:
                raise RuntimeError(
                    f"origin {origin} rejected: termination={result.termination_condition}, "
                    f"incumbent={incumbent}, relative_gap={gap}"
                )
            result.solution_loader.load_vars()
            formal_scenario_costs = np.asarray(
                [
                    pyo.value(self.model.scenario_total_cost[s])
                    for s in range(SCENARIOS)
                ],
                dtype=float,
            )
            formal_scenario_cvar = float(
                pyo.value(self.model.formal_scenario_cvar)
            )
            if frozen_da is None:
                frozen_da = np.asarray(
                    [
                        [pyo.value(self.model.da_schedule[z, t]) for t in range(INTERVALS)]
                        for z in range(len(NYISO_ZONES))
                    ],
                    dtype=float,
                )
                da_schedule[:, :] = frozen_da
            grid = np.asarray([pyo.value(self.model.grid[0, z, origin]) for z in range(len(NYISO_ZONES))])
            work = np.asarray([pyo.value(self.model.work[0, z, origin]) for z in range(len(NYISO_ZONES))])
            charge = np.asarray([pyo.value(self.model.charge[0, z, origin]) for z in range(len(NYISO_ZONES))])
            discharge = np.asarray([pyo.value(self.model.discharge[0, z, origin]) for z in range(len(NYISO_ZONES))])
            battery_mode = np.asarray(
                [
                    pyo.value(self.model.battery_charge_mode[0, z, origin])
                    for z in range(len(NYISO_ZONES))
                ],
                dtype=float,
            )
            pv_used = np.asarray([pyo.value(self.model.pv_used[0, z, origin]) for z in range(len(NYISO_ZONES))])
            q_da = np.asarray([pyo.value(self.model.da_schedule[z, origin]) for z in range(len(NYISO_ZONES))])
            soc_end = soc + INTERVAL_HOURS * (0.95 * charge - discharge / 0.95)
            load = day.physical_multiplier * (
                np.asarray(self.physical.idle_mw) + np.asarray(self.physical.work_mw_per_unit) * work
            )
            energy_residual = grid + pv_used + discharge - load - charge
            soc_violation = max(float(np.max(-soc_end)), float(np.max(soc_end - self.physical.battery_energy_mwh)), 0.0)
            cumulative_released = float(day.realized_workload[: origin + 1].sum())
            cumulative_executed = float(
                self._executed_work[:, :origin].sum() + work.sum()
            )
            queue_violation = max(cumulative_executed - cumulative_released, 0.0)
            due_index = origin - 4
            cumulative_due = float(
                0.0
                if due_index < 0
                else day.realized_workload[: due_index + 1].sum()
            )
            service_residual = max(
                cumulative_due
                - cumulative_executed
                - float(
                    pyo.value(self.model.unserved[0, INTERVALS - 1])
                ),
                0.0,
                queue_violation,
            )
            settlement = settle_realized_components(
                q_da,
                grid,
                realized_da[:, origin],
                realized_rt[:, origin],
                interval_hours=INTERVAL_HOURS,
                component_costs_usd={
                    "battery_degradation_cost_usd": float(INTERVAL_HOURS * np.sum(charge + discharge)),
                    "sla_penalty_cost_usd": float(
                        1000.0
                        * (
                            pyo.value(self.model.unserved[0, INTERVALS - 1])
                            if origin == INTERVALS - 1
                            else 0.0
                        )
                    ),
                },
            )
            validations = {
                "energy_balance_max_abs_mw": float(np.max(np.abs(energy_residual))),
                "soc_bound_max_violation_mwh": soc_violation,
                "migration_max_violation_work_units": 0.0,
                "sla_balance_max_abs_work_units": service_residual,
                "integrality_max_abs": float(
                    np.max(np.abs(battery_mode - np.rint(battery_mode)), initial=0.0)
                ),
                "battery_simultaneous_charge_discharge_max_mw": float(
                    np.max(np.minimum(charge, discharge), initial=0.0)
                ),
                "settlement_identity_residual_mw": float(settlement["settlement_identity_residual_mw"]),
            }
            validations["passed"] = max(validations.values()) <= 1e-5
            if not validations["passed"]:
                raise RuntimeError(f"origin {origin} failed independent validation: {validations}")
            self._executed_grid[:, origin] = grid
            self._executed_work[:, origin] = work
            da_schedule[:, origin] = q_da
            soc = soc_end
            records.append(
                {
                    "operating_date": day.operating_date,
                    "origin": origin,
                    "timestamp_utc": str(day.timestamps_utc[origin]),
                    "mode": mode,
                    "termination_condition": str(result.termination_condition),
                    "relative_gap": gap,
                    "solve_seconds": solve_seconds,
                    "grid_import_mw": float(grid.sum()),
                    "work_completed_units": float(work.sum()),
                    "unserved_work_units": float(
                        pyo.value(self.model.unserved[0, INTERVALS - 1])
                        if origin == INTERVALS - 1
                        else 0.0
                    ),
                    "soc_mwh": float(soc.sum()),
                    "total_cost_usd": float(settlement["simulated_total_cost_usd"]),
                    "formal_expected_scenario_cost_usd": float(
                        np.dot(weights, formal_scenario_costs)
                    ),
                    "formal_cvar_eta_usd": float(pyo.value(self.model.cvar_eta)),
                    "formal_scenario_cvar_95_usd": formal_scenario_cvar,
                    "valid": True,
                    **validations,
                }
            )
            self._write_log(
                f"origin={origin} status=accepted mode={mode} "
                f"termination={result.termination_condition} relative_gap={gap} "
                f"solve_seconds={solve_seconds:.6f} validations_passed=true"
            )
        costs = np.asarray([r["total_cost_usd"] for r in records], dtype=float)
        formal_cvars = np.asarray(
            [r["formal_scenario_cvar_95_usd"] for r in records], dtype=float
        )
        service = np.asarray([r["unserved_work_units"] for r in records], dtype=float)
        arrivals = np.asarray(day.realized_workload[: len(records)], dtype=float)
        return {
            "status": "COMPLETE" if len(records) == INTERVALS else "PARTIAL",
            "operating_date": day.operating_date,
            "completed_origins": len(records),
            "model_build_count": self.model_build_count,
            "set_instance_count": self.set_instance_count,
            "structure_rebuild_count": len(self.structure_rebuilds),
                "scenario_count": SCENARIOS,
                "migration_contract": (
                    "known-zero; source bundle migration_actions_enabled=false"
                ),
                "records": records,
            "summary": {
                "feasibility_rate": float(np.mean([r["valid"] for r in records])) if records else 0.0,
                "total_cost_usd": float(costs.sum()),
                # This is the exact weighted 20-particle CVaR from the final
                # accepted formal objective at each rolling origin.  It is not
                # an empirical tail over the 96 realized interval costs.
                "formal_scenario_cvar_95_usd": float(formal_cvars[0])
                if records
                else None,
                "mean_rolling_formal_scenario_cvar_95_usd": float(
                    formal_cvars.mean()
                )
                if records
                else None,
                # Backward-compatible field name retained for existing result
                # readers, now explicitly populated by the formal scenario
                # measure rather than a mislabeled interval-cost statistic.
                "empirical_cvar_95_usd": float(formal_cvars[0])
                if records
                else None,
                "sla_satisfaction": 1.0 - float(service.sum() / max(arrivals.sum(), 1e-12)),
                "solve_time_seconds": float(sum(r["solve_seconds"] for r in records)),
                "wall_time_seconds": time.perf_counter() - total_started,
            },
        }


def records_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(result.get("records", [])))
