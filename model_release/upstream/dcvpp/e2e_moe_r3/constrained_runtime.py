from __future__ import annotations

"""Solver-free rolling runtime with a hard, analytic constraint shield.

The neural model proposes actions; it is not entrusted with feasibility.  At
each rolling origin :class:`CertifiedConstraintShield` maps the proposal to an
action satisfying the represented battery, power, compute, queue and selection
migration constraints.  The executed action is then recomputed by an
independent validator.  This module deliberately imports neither Pyomo nor a
mathematical-programming solver.

The formal selection contract represented here is the audited battery-only
contract: migration is known zero.  Deadline misses are explicit SLA slack,
not silently relabelled as hard feasibility.  A physically impossible state
(for example, idle demand exceeding grid, PV and admissible discharge) is
reported as ``INFEASIBLE`` and no fabricated action is executed.
"""

from dataclasses import dataclass, field
from datetime import datetime
import inspect
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

INTERVALS = 96
SCENARIOS = 20
INTERVAL_HOURS = 0.25
ALLOWED_YEARS = frozenset({2022, 2023, 2024})
NYISO_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
_ZONES = len(NYISO_ZONES)


class R3DataError(ValueError):
    """A selection input violates the 2022--2024 solver-free contract."""


class InfeasibleStateError(RuntimeError):
    """The exogenous state admits no action satisfying the hard constraints."""

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.details = dict(details or {})


def _year(date_text: str) -> int:
    try:
        year = datetime.fromisoformat(str(date_text)[:10]).year
    except ValueError as exc:
        raise R3DataError(f"invalid operating date: {date_text}") from exc
    if year not in ALLOWED_YEARS:
        raise R3DataError(
            f"R3 selection inputs are closed outside 2022-2024; rejected {date_text}"
        )
    return year


def _reject_locked_path(path: Path | str) -> Path:
    candidate = Path(path)
    # This check intentionally precedes every file open.  It catches both a
    # YYYY directory and the usual YYYY-MM-DD filename convention.
    for part in candidate.parts:
        token = str(part).lower()
        if token == "2025" or token.startswith("2025-") or token.startswith("2025_"):
            raise R3DataError(f"refusing to open locked 2025 path: {candidate}")
    return candidate


def _finite_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    copy: bool = True,
) -> np.ndarray:
    array = np.array(value, dtype=float, copy=copy)
    if shape is not None and array.shape != shape:
        raise R3DataError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise R3DataError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class FrozenPhysicalArtifact:
    """Solver-free immutable view of the calibrated four-zone physical twin."""

    calibration_id: str
    twin_ids: tuple[str, ...]
    transfer_regime: str
    idle_mw: np.ndarray
    work_mw_per_unit: np.ndarray
    capacity_work_units: np.ndarray
    grid_import_limit_mw: np.ndarray
    battery_power_mw: np.ndarray
    battery_energy_mwh: np.ndarray
    physical_multiplier_pool: np.ndarray
    zones: tuple[str, ...] = NYISO_ZONES
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.frozen:
            raise R3DataError("physical artifact must be frozen")
        if tuple(self.zones) != NYISO_ZONES:
            raise R3DataError(f"zones must be exactly {NYISO_ZONES}")
        if len(self.twin_ids) != _ZONES or len(set(self.twin_ids)) != _ZONES:
            raise R3DataError("one unique twin_id is required for each NYISO zone")
        for name in (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
        ):
            object.__setattr__(
                self,
                name,
                _finite_array(getattr(self, name), name=name, shape=(_ZONES,)),
            )
        pool = _finite_array(
            self.physical_multiplier_pool,
            name="physical_multiplier_pool",
        )
        if pool.ndim != 2 or pool.shape[1] != _ZONES or pool.shape[0] < 100:
            raise R3DataError(
                "physical_multiplier_pool must have shape (at least 100, 4)"
            )
        if (pool <= 0).any():
            raise R3DataError("physical multipliers must be strictly positive")
        object.__setattr__(self, "physical_multiplier_pool", pool)
        for name in (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
        ):
            if (getattr(self, name) < 0).any():
                raise R3DataError(f"{name} must be nonnegative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenPhysicalArtifact":
        frozen = payload.get("frozen_artifact", payload.get("frozen", payload))
        if isinstance(frozen, cls):
            return frozen
        if not isinstance(frozen, Mapping):
            raise R3DataError("physical artifact must be a mapping")
        return cls(
            calibration_id=str(frozen["calibration_id"]),
            twin_ids=tuple(map(str, frozen["twin_ids"])),
            transfer_regime=str(
                frozen.get("transfer_regime", "frozen_transfer_ensemble")
            ),
            idle_mw=frozen["idle_mw"],
            work_mw_per_unit=frozen["work_mw_per_unit"],
            capacity_work_units=frozen["capacity_work_units"],
            grid_import_limit_mw=frozen["grid_import_limit_mw"],
            battery_power_mw=frozen.get("battery_power_mw", np.zeros(_ZONES)),
            battery_energy_mwh=frozen.get("battery_energy_mwh", np.zeros(_ZONES)),
            physical_multiplier_pool=frozen["physical_multiplier_pool"],
            zones=tuple(map(str, frozen.get("zones", NYISO_ZONES))),
            source_hashes=dict(frozen.get("source_hashes", {})),
            frozen=bool(frozen.get("frozen", True)),
        )


@dataclass(frozen=True)
class SelectionDay:
    """Audited issue-time bundle plus optional official realized settlement."""

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
        if bool(self.locked_test):
            raise R3DataError("2025 locked-test input is forbidden in R3 selection work")
        if bool(self.migration_actions_enabled):
            raise R3DataError(
                "solver-free R3 selection uses the audited known-zero migration "
                "contract; migration-enabled input is outside this model"
            )
        timestamps = np.asarray(self.timestamps_utc)
        if timestamps.shape != (INTERVALS,):
            raise R3DataError(
                f"timestamps_utc must have shape {(INTERVALS,)}, got {timestamps.shape}"
            )
        if len(np.unique(timestamps)) != INTERVALS:
            raise R3DataError("timestamps_utc must contain 96 unique origins")
        shapes = {
            "da_price": (_ZONES, INTERVALS),
            "rt_price_pool": (100, _ZONES, INTERVALS),
            "workload_pool": (100, INTERVALS),
            "pv_pool": (100, _ZONES, INTERVALS),
            "realized_workload": (INTERVALS,),
            "realized_pv": (_ZONES, INTERVALS),
            "physical_multiplier": (_ZONES,),
            "initial_soc_mwh": (_ZONES,),
        }
        for name, shape in shapes.items():
            _finite_array(getattr(self, name), name=name, shape=shape, copy=False)
        if (np.asarray(self.realized_workload, dtype=float) < 0.0).any():
            raise R3DataError("realized_workload cannot be negative")
        if (np.asarray(self.realized_pv, dtype=float) < 0.0).any():
            raise R3DataError("realized_pv cannot be negative")
        if (np.asarray(self.physical_multiplier, dtype=float) <= 0.0).any():
            raise R3DataError("physical_multiplier must be strictly positive")
        for name in ("realized_da_price", "realized_rt_price"):
            value = getattr(self, name)
            if value is not None:
                _finite_array(
                    value,
                    name=name,
                    shape=(_ZONES, INTERVALS),
                    copy=False,
                )


def load_selection_day(
    path: Path | str,
    realized_settlement_path: Path | str | None = None,
) -> SelectionDay:
    """Load a selection day while rejecting a 2025 path before opening it."""

    bundle_path = _reject_locked_path(path)
    with np.load(bundle_path, allow_pickle=False) as data:
        operating_date = str(data["operating_date"].item())
        _year(operating_date)
        locked = bool(data["locked_test"].item())
        if locked:
            raise R3DataError(f"refusing to read locked-test bundle: {bundle_path}")
        source_hashes = json.loads(str(data["source_hashes_json"].item()))
        actionability = (
            json.loads(str(data["actionability_policy_json"].item()))
            if "actionability_policy_json" in data.files
            else {}
        )
        timestamps = np.asarray(data["timestamps_utc"]).copy()
        realized_da: np.ndarray | None = None
        realized_rt: np.ndarray | None = None
        if realized_settlement_path is not None:
            settlement_path = _reject_locked_path(realized_settlement_path)
            with np.load(settlement_path, allow_pickle=False) as settlement:
                settlement_timestamps = np.asarray(settlement["timestamps_utc"])
                if not np.array_equal(settlement_timestamps, timestamps):
                    raise R3DataError("settlement timestamps do not match selection day")
                realized_da = np.asarray(settlement["da_lbmp"], dtype=float).copy()
                realized_rt = np.asarray(settlement["rt_lbmp"], dtype=float).copy()
        day = SelectionDay(
            operating_date=operating_date,
            timestamps_utc=timestamps,
            da_price=np.asarray(data["da_price"], dtype=float).copy(),
            rt_price_pool=np.asarray(data["rt_price_pool"], dtype=float).copy(),
            workload_pool=np.asarray(data["workload_pool"], dtype=float).copy(),
            pv_pool=np.asarray(data["pv_pool"], dtype=float).copy(),
            realized_workload=np.asarray(data["workload"], dtype=float).copy(),
            realized_pv=np.asarray(data["pv"], dtype=float).copy(),
            physical_multiplier=np.asarray(
                data["physical_multiplier"], dtype=float
            ).copy(),
            initial_soc_mwh=np.asarray(
                data["initial_soc_mwh_by_zone"], dtype=float
            ).copy(),
            source_hashes=dict(source_hashes),
            locked_test=locked,
            migration_actions_enabled=bool(
                actionability.get("migration_actions_enabled", False)
            ),
            realized_da_price=realized_da,
            realized_rt_price=realized_rt,
        )
    day.validate()
    return day


def load_frozen_physical(path: Path | str) -> FrozenPhysicalArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenPhysicalArtifact.from_mapping({"frozen_artifact": payload})


def _normalized_weights(weights: Any) -> np.ndarray:
    value = _finite_array(weights, name="scenario_weights", shape=(SCENARIOS,))
    if (value < 0.0).any() or float(value.sum()) <= 0.0:
        raise R3DataError("scenario_weights must be nonnegative with positive sum")
    return value / float(value.sum())


def weighted_cvar(costs: Any, weights: Any, alpha: float = 0.95) -> float | np.ndarray:
    """Exact upper-tail CVaR of a discrete weighted 20-particle distribution.

    The final dimension is the particle dimension.  The returned value is the
    Rockafellar--Uryasev expression evaluated at an exact weighted empirical
    VaR.  This is an accounting utility, not an optimization claim.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    values = np.asarray(costs, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if values.shape[-1:] != (SCENARIOS,):
        raise ValueError("costs must end in exactly 20 particle values")
    if probabilities.shape == (SCENARIOS,):
        probabilities = np.broadcast_to(probabilities, values.shape)
    if probabilities.shape != values.shape:
        raise ValueError("weights must be [20] or have the same shape as costs")
    if not np.isfinite(values).all() or not np.isfinite(probabilities).all():
        raise ValueError("costs and weights must be finite")
    if (probabilities < 0.0).any():
        raise ValueError("weights cannot be negative")
    totals = probabilities.sum(axis=-1, keepdims=True)
    if (totals <= 0.0).any():
        raise ValueError("weights must have positive sum")
    probabilities = probabilities / totals
    order = np.argsort(values, axis=-1, kind="stable")
    sorted_costs = np.take_along_axis(values, order, axis=-1)
    sorted_weights = np.take_along_axis(probabilities, order, axis=-1)
    cumulative = np.cumsum(sorted_weights, axis=-1)
    index = np.argmax(cumulative >= float(alpha), axis=-1)
    eta = np.take_along_axis(sorted_costs, np.expand_dims(index, -1), axis=-1)[..., 0]
    excess = np.maximum(values - np.expand_dims(eta, -1), 0.0)
    result = eta + np.sum(probabilities * excess, axis=-1) / (1.0 - float(alpha))
    return float(result) if np.ndim(result) == 0 else result


exact_weighted_cvar = weighted_cvar


def _violation_lower(value: np.ndarray, lower: float | np.ndarray = 0.0) -> float:
    return float(np.max(np.maximum(np.asarray(lower) - value, 0.0), initial=0.0))


def _violation_upper(value: np.ndarray, upper: float | np.ndarray) -> float:
    return float(np.max(np.maximum(value - np.asarray(upper), 0.0), initial=0.0))


def validate_executed_action(
    *,
    physical: FrozenPhysicalArtifact,
    physical_multiplier: Any,
    soc_start_mwh: Any,
    soc_end_mwh: Any,
    work: Any,
    grid: Any,
    pv_available: Any,
    pv_used: Any,
    charge: Any,
    discharge: Any,
    migration: Any | None = None,
    queue_before: Any | None = None,
    queue_after: Any | None = None,
    arrivals: float = 0.0,
    served: float | None = None,
    sla_shortfall: float = 0.0,
    day_ahead: Any | None = None,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
    interval_hours: float = INTERVAL_HOURS,
    tolerance: float = 1e-6,
) -> dict[str, float | bool]:
    """Independently recompute all represented hard-action residuals."""

    if not 0.0 < charge_efficiency <= 1.0:
        raise ValueError("charge_efficiency must lie in (0, 1]")
    if not 0.0 < discharge_efficiency <= 1.0:
        raise ValueError("discharge_efficiency must lie in (0, 1]")
    if interval_hours <= 0.0 or tolerance < 0.0:
        raise ValueError("interval_hours must be positive and tolerance nonnegative")
    arrays = {
        "physical_multiplier": physical_multiplier,
        "soc_start_mwh": soc_start_mwh,
        "soc_end_mwh": soc_end_mwh,
        "work": work,
        "grid": grid,
        "pv_available": pv_available,
        "pv_used": pv_used,
        "charge": charge,
        "discharge": discharge,
    }
    parsed = {
        name: _finite_array(value, name=name, shape=(_ZONES,))
        for name, value in arrays.items()
    }
    multiplier = parsed["physical_multiplier"]
    soc_start = parsed["soc_start_mwh"]
    soc_end = parsed["soc_end_mwh"]
    work_array = parsed["work"]
    grid_array = parsed["grid"]
    pv_available_array = parsed["pv_available"]
    pv_used_array = parsed["pv_used"]
    charge_array = parsed["charge"]
    discharge_array = parsed["discharge"]
    load = multiplier * (
        np.asarray(physical.idle_mw, dtype=float)
        + np.asarray(physical.work_mw_per_unit, dtype=float) * work_array
    )
    energy_residual = grid_array + pv_used_array + discharge_array - load - charge_array
    expected_soc = soc_start + interval_hours * (
        charge_efficiency * charge_array
        - discharge_array / discharge_efficiency
    )
    migration_array = (
        np.zeros((_ZONES, _ZONES), dtype=float)
        if migration is None
        else _finite_array(
            migration, name="migration", shape=(_ZONES, _ZONES)
        )
    )
    queue_residual = 0.0
    if queue_before is not None or queue_after is not None:
        if queue_before is None or queue_after is None:
            raise ValueError("queue_before and queue_after must be supplied together")
        before = np.asarray(queue_before, dtype=float)
        after = np.asarray(queue_after, dtype=float)
        if before.ndim != 1 or after.shape != before.shape:
            raise ValueError("queue arrays must be one-dimensional with matching shapes")
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("queue arrays must be finite")
        served_value = float(work_array.sum()) if served is None else float(served)
        queue_residual = abs(
            float(before.sum())
            + float(arrivals)
            - served_value
            - float(sla_shortfall)
            - float(after.sum())
        )
    settlement_residual = 0.0
    if day_ahead is not None:
        q_da = _finite_array(day_ahead, name="day_ahead", shape=(_ZONES,))
        deviation = grid_array - q_da
        settlement_residual = float(
            np.max(
                np.abs(
                    deviation
                    - (np.maximum(deviation, 0.0) - np.maximum(-deviation, 0.0))
                ),
                initial=0.0,
            )
        )
    charge_mode = (charge_array > tolerance).astype(float)
    discharge_mode = (discharge_array > tolerance).astype(float)
    result: dict[str, float | bool] = {
        "energy_balance_max_abs_mw": float(
            np.max(np.abs(energy_residual), initial=0.0)
        ),
        "soc_transition_max_abs_mwh": float(
            np.max(np.abs(soc_end - expected_soc), initial=0.0)
        ),
        "soc_bound_max_violation_mwh": max(
            _violation_lower(soc_end),
            _violation_upper(soc_end, np.asarray(physical.battery_energy_mwh)),
        ),
        "grid_bound_max_violation_mw": max(
            _violation_lower(grid_array),
            _violation_upper(grid_array, np.asarray(physical.grid_import_limit_mw)),
        ),
        "pv_bound_max_violation_mw": max(
            _violation_lower(pv_used_array),
            _violation_upper(pv_used_array, pv_available_array),
        ),
        "compute_bound_max_violation_work_units": max(
            _violation_lower(work_array),
            _violation_upper(work_array, np.asarray(physical.capacity_work_units)),
        ),
        "battery_power_max_violation_mw": max(
            _violation_lower(charge_array),
            _violation_lower(discharge_array),
            _violation_upper(charge_array, np.asarray(physical.battery_power_mw)),
            _violation_upper(discharge_array, np.asarray(physical.battery_power_mw)),
        ),
        "migration_max_violation_work_units": float(
            np.max(np.abs(migration_array), initial=0.0)
        ),
        "sla_balance_max_abs_work_units": float(queue_residual),
        "integrality_max_abs": float(
            np.max(
                np.abs(charge_mode + discharge_mode - np.rint(charge_mode + discharge_mode)),
                initial=0.0,
            )
        ),
        "battery_simultaneous_charge_discharge_max_mw": float(
            np.max(np.minimum(charge_array, discharge_array), initial=0.0)
        ),
        "settlement_identity_residual_mw": settlement_residual,
    }
    residual_names = tuple(result)
    result["passed"] = bool(
        max(float(result[name]) for name in residual_names) <= float(tolerance)
    )
    return result


def validate_record(
    record: Mapping[str, Any],
    *,
    physical: FrozenPhysicalArtifact,
    day: SelectionDay,
    tolerance: float = 1e-6,
) -> dict[str, float | bool]:
    """Revalidate one emitted record without relying on stored residuals."""

    origin = int(record["origin"])
    stored_da = np.asarray(record["day_ahead_schedule_mw"], dtype=float)
    if stored_da.shape == (_ZONES, INTERVALS):
        stored_da = stored_da[:, origin]
    return validate_executed_action(
        physical=physical,
        physical_multiplier=day.physical_multiplier,
        soc_start_mwh=record["soc_start_by_zone_mwh"],
        soc_end_mwh=record["soc_end_by_zone_mwh"],
        work=record["work_by_zone"],
        grid=record["grid_import_by_zone_mw"],
        pv_available=day.realized_pv[:, origin],
        pv_used=record["pv_used_by_zone_mw"],
        charge=record["charge_by_zone_mw"],
        discharge=record["discharge_by_zone_mw"],
        migration=record["migration_by_route"],
        queue_before=record["queue_age_buckets_before"],
        queue_after=record["queue_age_buckets_after"],
        arrivals=float(record["workload_arrival_units"]),
        served=float(record["work_completed_units"]),
        sla_shortfall=float(record["sla_shortfall_work_units"]),
        day_ahead=stored_da,
        tolerance=tolerance,
    )


def _project_capped_simplex(raw: np.ndarray, caps: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto ``0 <= x <= caps, sum(x) == total``."""

    raw = np.asarray(raw, dtype=float)
    caps = np.asarray(caps, dtype=float)
    feasible_total = float(np.clip(total, 0.0, caps.sum()))
    if feasible_total <= 0.0:
        return np.zeros_like(raw)
    if feasible_total >= float(caps.sum()) - 1e-12:
        return caps.copy()
    lower = float(np.min(raw - caps))
    upper = float(np.max(raw))
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        candidate = np.clip(raw - midpoint, 0.0, caps)
        if float(candidate.sum()) > feasible_total:
            lower = midpoint
        else:
            upper = midpoint
    result = np.clip(raw - upper, 0.0, caps)
    residual = feasible_total - float(result.sum())
    if residual > 0.0:
        for index in np.argsort(-(caps - result), kind="stable"):
            amount = min(residual, float(caps[index] - result[index]))
            result[index] += amount
            residual -= amount
            if residual <= 1e-12:
                break
    elif residual < 0.0:
        for index in np.argsort(-result, kind="stable"):
            amount = min(-residual, float(result[index]))
            result[index] -= amount
            residual += amount
            if residual >= -1e-12:
                break
    return result


def _action_vector(value: Any, *, name: str, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(_ZONES, float(default), dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(_ZONES, float(array), dtype=float)
    if array.shape != (_ZONES,) or not np.isfinite(array).all():
        raise R3DataError(f"policy {name} must be finite with shape {(_ZONES,)}")
    return array.copy()


def _day_ahead_matrix(value: Any, default: np.ndarray) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float).copy()
    array = np.asarray(value, dtype=float)
    if array.shape == (INTERVALS, _ZONES):
        array = array.T
    elif array.shape == (_ZONES,):
        array = np.repeat(array[:, None], INTERVALS, axis=1)
    if array.shape != (_ZONES, INTERVALS) or not np.isfinite(array).all():
        raise R3DataError(
            f"policy day_ahead must be finite with shape {(_ZONES, INTERVALS)}"
        )
    return array.copy()


class CertifiedConstraintShield:
    """Analytic hard projection for one solver-free rolling action."""

    def __init__(
        self,
        physical: FrozenPhysicalArtifact,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.physical = physical
        outer = dict(config or {})
        nested = outer.get("constraint_shield", outer.get("shield", outer))
        self.config = dict(nested) if isinstance(nested, Mapping) else outer
        battery_config = outer.get("battery", {})
        if not isinstance(battery_config, Mapping):
            battery_config = {}
        workload_config = outer.get("workload", {})
        if not isinstance(workload_config, Mapping):
            workload_config = {}
        self.charge_efficiency = float(
            self.config.get(
                "charge_efficiency",
                battery_config.get("charge_efficiency", 0.95),
            )
        )
        self.discharge_efficiency = float(
            self.config.get(
                "discharge_efficiency",
                battery_config.get("discharge_efficiency", 0.95),
            )
        )
        self.terminal_soc_fraction = float(
            self.config.get(
                "terminal_soc_fraction",
                outer.get("terminal_soc_reserve_fraction", 0.5),
            )
        )
        self.deadline_steps = int(
            self.config.get(
                "deadline_steps", workload_config.get("deadline_intervals", 4)
            )
        )
        self.tolerance = float(
            self.config.get(
                "validation_tolerance", outer.get("validation_tolerance", 1e-6)
            )
        )
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must lie in (0, 1]")
        if not 0.0 < self.discharge_efficiency <= 1.0:
            raise ValueError("discharge_efficiency must lie in (0, 1]")
        if not 0.0 <= self.terminal_soc_fraction <= 1.0:
            raise ValueError("terminal_soc_fraction must lie in [0, 1]")
        if self.deadline_steps < 0:
            raise ValueError("deadline_steps must be nonnegative")
        self.call_count = 0

    @property
    def queue_size(self) -> int:
        return self.deadline_steps + 1

    def _terminal_lower_next(
        self, origin: int, physical_multiplier: np.ndarray
    ) -> np.ndarray:
        energy = np.asarray(self.physical.battery_energy_mwh, dtype=float)
        power = np.asarray(self.physical.battery_power_mw, dtype=float)
        grid_limit = np.asarray(self.physical.grid_import_limit_mw, dtype=float)
        idle_load = np.asarray(physical_multiplier, dtype=float) * np.asarray(
            self.physical.idle_mw, dtype=float
        )
        # The reachability envelope may rely only on charging power that is
        # guaranteed after fixed idle demand.  Workload can be deferred (and
        # any unavoidable miss is reported as SLA slack); uncertain PV is not
        # assumed available for the terminal reserve.
        guaranteed_charge = np.minimum(
            power, np.maximum(grid_limit - idle_load, 0.0)
        )
        reserve = self.terminal_soc_fraction * energy
        intervals_after = INTERVALS - origin - 1
        recoverable = (
            intervals_after
            * INTERVAL_HOURS
            * self.charge_efficiency
            * guaranteed_charge
        )
        return np.maximum(0.0, reserve - recoverable)

    def project(
        self,
        *,
        origin: int,
        raw_action: Mapping[str, Any],
        soc_mwh: Any,
        queue_before: Any,
        arrival: float,
        pv_available: Any,
        physical_multiplier: Any,
        frozen_day_ahead: Any | None = None,
        fallback: bool = False,
    ) -> dict[str, Any]:
        """Project one proposal and return an independently validated action."""

        self.call_count += 1
        if not 0 <= int(origin) < INTERVALS:
            raise ValueError(f"origin must lie in [0, {INTERVALS - 1}]")
        if not isinstance(raw_action, Mapping):
            raise R3DataError("policy callback must return a mapping")
        soc = _finite_array(soc_mwh, name="soc_mwh", shape=(_ZONES,))
        queue_prior = np.asarray(queue_before, dtype=float)
        if queue_prior.shape != (self.queue_size,) or not np.isfinite(queue_prior).all():
            raise R3DataError(
                f"queue_before must be finite with shape {(self.queue_size,)}"
            )
        if (queue_prior < -self.tolerance).any() or not np.isfinite(float(arrival)):
            raise R3DataError("queue and arrival must be finite and nonnegative")
        arrival_value = max(float(arrival), 0.0)
        pv = _finite_array(pv_available, name="pv_available", shape=(_ZONES,))
        pv = np.maximum(pv, 0.0)
        multiplier = _finite_array(
            physical_multiplier, name="physical_multiplier", shape=(_ZONES,)
        )
        if (multiplier <= 0.0).any():
            raise R3DataError("physical_multiplier must be strictly positive")
        energy = np.asarray(self.physical.battery_energy_mwh, dtype=float)
        power = np.asarray(self.physical.battery_power_mw, dtype=float)
        grid_limit = np.asarray(self.physical.grid_import_limit_mw, dtype=float)
        if (
            (soc < -self.tolerance).any()
            or (soc > energy + self.tolerance).any()
        ):
            raise InfeasibleStateError(
                "SOC is outside the frozen battery bounds",
                details={"origin": int(origin), "soc_mwh": soc.tolist()},
            )

        structural_mapping: list[str] = []
        corrective_interventions: list[str] = []
        raw_violations: dict[str, float] = {}
        corrective_magnitude = 0.0

        def map_structure(name: str) -> None:
            structural_mapping.append(str(name))

        def record_correction(
            category: str, intervention_name: str, magnitude: float
        ) -> None:
            nonlocal corrective_magnitude
            distance = max(float(magnitude), 0.0)
            if distance <= self.tolerance:
                return
            corrective_interventions.append(str(intervention_name))
            raw_violations[str(category)] = (
                raw_violations.get(str(category), 0.0) + distance
            )
            corrective_magnitude += distance

        idle = np.asarray(self.physical.idle_mw, dtype=float)
        work_power = np.asarray(self.physical.work_mw_per_unit, dtype=float)
        work_capacity = np.asarray(self.physical.capacity_work_units, dtype=float)
        default_da_interval = np.clip(multiplier * idle, 0.0, grid_limit)
        default_da = np.repeat(default_da_interval[:, None], INTERVALS, axis=1)
        raw_da_value = raw_action.get("day_ahead", raw_action.get("da"))
        if origin == 0:
            proposed_da = _day_ahead_matrix(raw_da_value, default_da)
            day_ahead = np.clip(proposed_da, 0.0, grid_limit[:, None])
            da_delta = float(np.abs(day_ahead - proposed_da).sum())
            if raw_da_value is None:
                map_structure("day_ahead_safe_default")
            else:
                record_correction(
                    "day_ahead_envelope", "day_ahead_projection", da_delta
                )
        else:
            if frozen_day_ahead is None:
                raise RuntimeError("day-ahead schedule must be frozen after origin zero")
            day_ahead = _finite_array(
                frozen_day_ahead,
                name="frozen_day_ahead",
                shape=(_ZONES, INTERVALS),
            )
            if raw_da_value is not None:
                proposed_da = _day_ahead_matrix(raw_da_value, day_ahead)
                da_delta = float(np.abs(proposed_da - day_ahead).sum())
                record_correction(
                    "day_ahead_freeze", "day_ahead_freeze", da_delta
                )

        lower_next = self._terminal_lower_next(int(origin), multiplier)
        required_terminal_charge = np.minimum(
            power,
            np.maximum(
                (lower_next - soc)
                / (INTERVAL_HOURS * self.charge_efficiency),
                0.0,
            ),
        )
        max_discharge = np.minimum(
            power,
            np.maximum(
                self.discharge_efficiency * (soc - lower_next) / INTERVAL_HOURS,
                0.0,
            ),
        )
        # Terminal reserve is a hard battery constraint and therefore receives
        # power budget before flexible compute.  EDF uses the remainder; any
        # impossible deadline demand is exposed as SLA slack below.
        available_for_load = (
            grid_limit + pv + max_discharge - required_terminal_charge
        )
        idle_load = multiplier * idle
        if (idle_load > available_for_load + self.tolerance).any():
            raise InfeasibleStateError(
                "idle demand exceeds grid, PV and admissible battery discharge",
                details={
                    "origin": int(origin),
                    "idle_load_mw": idle_load.tolist(),
                    "available_power_mw": available_for_load.tolist(),
                },
            )
        compute_caps = work_capacity.copy()
        positive_slope = work_power > self.tolerance
        compute_caps[positive_slope] = np.minimum(
            compute_caps[positive_slope],
            np.maximum(
                (
                    available_for_load[positive_slope]
                    / multiplier[positive_slope]
                    - idle[positive_slope]
                )
                / work_power[positive_slope],
                0.0,
            ),
        )
        queue_available = queue_prior.copy()
        queue_available[-1] += arrival_value
        total_available_work = float(queue_available.sum())
        max_service = min(total_available_work, float(compute_caps.sum()))

        # Necessary current service for every already-released deadline prefix.
        # Future capacity deliberately excludes uncertain PV and battery energy.
        guaranteed_caps = work_capacity.copy()
        guaranteed_power = grid_limit
        infeasible_idle_without_flex = idle_load > guaranteed_power + self.tolerance
        guaranteed_caps[infeasible_idle_without_flex] = 0.0
        safe_site = positive_slope & ~infeasible_idle_without_flex
        guaranteed_caps[safe_site] = np.minimum(
            guaranteed_caps[safe_site],
            np.maximum(
                (
                    guaranteed_power[safe_site] / multiplier[safe_site]
                    - idle[safe_site]
                )
                / work_power[safe_site],
                0.0,
            ),
        )
        guaranteed_future = float(guaranteed_caps.sum())
        required_now = 0.0
        for remaining in range(self.queue_size):
            due_prefix = float(queue_available[: remaining + 1].sum())
            required_now = max(
                required_now, due_prefix - remaining * guaranteed_future
            )
        required_now = max(required_now, 0.0)
        anticipated_shortfall = max(required_now - max_service, 0.0)
        compute_proposed = "compute" in raw_action or "work" in raw_action
        proposed_compute = _action_vector(
            raw_action.get("compute", raw_action.get("work")), name="compute"
        )
        if not compute_proposed:
            map_structure("compute_safe_default")
        raw_compute = np.maximum(proposed_compute, 0.0)
        map_structure("edf_queue_accounting")
        if fallback:
            desired_service = max_service
            map_structure("certified_safe_policy_mapping")
        else:
            desired_service = float(
                np.clip(
                    raw_compute.sum(),
                    min(required_now, max_service),
                    max_service,
                )
            )
        work = _project_capped_simplex(raw_compute, compute_caps, desired_service)
        if compute_proposed:
            compute_delta = float(np.abs(work - proposed_compute).sum())
            record_correction(
                "compute_envelope", "edf_compute_projection", compute_delta
            )

        load = multiplier * (idle + work_power * work)
        lo_soc = np.maximum(
            -power,
            -self.discharge_efficiency * (soc - lower_next) / INTERVAL_HOURS,
        )
        hi_soc = np.minimum(
            power,
            (energy - soc) / (INTERVAL_HOURS * self.charge_efficiency),
        )
        lo_energy = -load
        hi_energy = grid_limit + pv - load
        battery_lower = np.maximum(lo_soc, lo_energy)
        battery_upper = np.minimum(hi_soc, hi_energy)
        if (battery_lower > battery_upper + self.tolerance).any():
            raise InfeasibleStateError(
                "battery/SOC and power-balance intervals do not intersect",
                details={
                    "origin": int(origin),
                    "battery_lower_mw": battery_lower.tolist(),
                    "battery_upper_mw": battery_upper.tolist(),
                },
            )
        battery_proposed = (
            "battery" in raw_action or "signed_battery" in raw_action
        )
        proposed_battery = _action_vector(
            raw_action.get("battery", raw_action.get("signed_battery")),
            name="battery",
        )
        if not battery_proposed:
            map_structure("battery_safe_default")
        raw_battery = proposed_battery
        if fallback:
            raw_battery = np.zeros(_ZONES, dtype=float)
        signed_battery = np.clip(raw_battery, battery_lower, battery_upper)
        if battery_proposed:
            battery_delta = float(
                np.abs(signed_battery - proposed_battery).sum()
            )
            record_correction(
                "battery_envelope", "signed_battery_projection", battery_delta
            )
        charge = np.maximum(signed_battery, 0.0)
        discharge = np.maximum(-signed_battery, 0.0)
        map_structure("signed_battery_charge_discharge_split")
        soc_end = soc + INTERVAL_HOURS * (
            self.charge_efficiency * charge
            - discharge / self.discharge_efficiency
        )
        # Bounded arithmetic can leave sub-ULP values such as -1e-17 at an
        # exact zero boundary.  Canonicalize the rolling state while the
        # independent transition validator below still audits the resulting
        # (tolerance-scale) closure residual.
        soc_end = np.clip(soc_end, 0.0, energy)

        net_import_requirement = load + signed_battery
        grid_lower = np.maximum(0.0, net_import_requirement - pv)
        grid_upper = np.minimum(grid_limit, net_import_requirement)
        if (grid_lower > grid_upper + self.tolerance).any():
            raise InfeasibleStateError(
                "no grid/PV split closes the power balance",
                details={"origin": int(origin)},
            )
        raw_grid_value = raw_action.get(
            "grid", raw_action.get("real_time", raw_action.get("rt_grid"))
        )
        raw_pv_value = raw_action.get("pv_used", raw_action.get("pv"))
        proposed_grid = (
            None
            if raw_grid_value is None
            else _action_vector(raw_grid_value, name="grid")
        )
        proposed_pv = (
            None
            if raw_pv_value is None
            else _action_vector(raw_pv_value, name="pv_used")
        )
        map_structure("grid_pv_energy_identity")
        if proposed_grid is None and proposed_pv is None:
            # Choosing one point on the analytically feasible grid/PV identity
            # is a structural map, not a corrective intervention.
            grid = grid_lower.copy()
        elif proposed_grid is None:
            assert proposed_pv is not None
            pv_lower = net_import_requirement - grid_upper
            pv_upper = net_import_requirement - grid_lower
            projected_pv = np.clip(proposed_pv, pv_lower, pv_upper)
            grid = net_import_requirement - projected_pv
            record_correction(
                "grid_pv_envelope",
                "grid_pv_envelope_projection",
                float(np.abs(projected_pv - proposed_pv).sum()),
            )
        else:
            grid = np.clip(proposed_grid, grid_lower, grid_upper)
            grid_pv_delta = float(np.abs(grid - proposed_grid).sum())
            if proposed_pv is not None:
                derived_pv = net_import_requirement - grid
                grid_pv_delta += float(np.abs(derived_pv - proposed_pv).sum())
            record_correction(
                "grid_pv_envelope",
                "grid_pv_envelope_projection",
                grid_pv_delta,
            )
        pv_used = net_import_requirement - grid

        raw_migration = raw_action.get("migration")
        if raw_migration is None:
            proposed_migration = np.zeros((_ZONES, _ZONES), dtype=float)
        else:
            proposed_migration = _finite_array(
                raw_migration,
                name="policy migration",
                shape=(_ZONES, _ZONES),
            )
        migration = np.zeros((_ZONES, _ZONES), dtype=float)
        migration_delta = float(np.abs(proposed_migration).sum())
        map_structure("selection_migration_known_zero")
        record_correction(
            "migration_fixed_zero",
            "selection_migration_fixed_zero",
            migration_delta,
        )

        queue_after = queue_available.copy()
        service_left = float(work.sum())
        for bucket in range(self.queue_size):
            served_here = min(service_left, float(queue_after[bucket]))
            queue_after[bucket] -= served_here
            service_left -= served_here
            if service_left <= self.tolerance:
                break
        current_shortfall = max(float(queue_after[0]), 0.0)
        queue_after[0] = 0.0
        exogenous_infeasible = bool(
            anticipated_shortfall > self.tolerance
            or current_shortfall > self.tolerance
        )
        if exogenous_infeasible:
            map_structure("explicit_sla_slack_accounting")

        validation = validate_executed_action(
            physical=self.physical,
            physical_multiplier=multiplier,
            soc_start_mwh=soc,
            soc_end_mwh=soc_end,
            work=work,
            grid=grid,
            pv_available=pv,
            pv_used=pv_used,
            charge=charge,
            discharge=discharge,
            migration=migration,
            queue_before=queue_prior,
            queue_after=queue_after,
            arrivals=arrival_value,
            served=float(work.sum()),
            sla_shortfall=current_shortfall,
            day_ahead=day_ahead[:, int(origin)],
            charge_efficiency=self.charge_efficiency,
            discharge_efficiency=self.discharge_efficiency,
            tolerance=self.tolerance,
        )
        if not bool(validation["passed"]):
            raise InfeasibleStateError(
                "internal constraint shield failed independent validation",
                details={"origin": int(origin), "validation": validation},
            )
        unique_structural = tuple(dict.fromkeys(structural_mapping))
        unique_corrections = tuple(dict.fromkeys(corrective_interventions))
        violation_categories = tuple(sorted(raw_violations))
        violation_values = tuple(raw_violations.values())
        return {
            "day_ahead": day_ahead,
            "grid": grid,
            "pv_used": pv_used,
            "signed_battery": signed_battery,
            "charge": charge,
            "discharge": discharge,
            "soc_end": soc_end,
            "work": work,
            "migration": migration,
            "queue_available": queue_available,
            "queue_after": queue_after,
            "sla_shortfall": current_shortfall,
            "anticipated_sla_shortfall": anticipated_shortfall,
            "required_service_now": required_now,
            "compute_caps": compute_caps,
            "terminal_soc_lower_next": lower_next,
            "structural_mapping_type": "+".join(unique_structural) or "none",
            "structural_mapping": bool(unique_structural),
            "structural_mapping_categories": list(unique_structural),
            "structural_mapping_count": len(unique_structural),
            # Structural transformations have no proposal-to-execution
            # correction distance by definition.
            "structural_mapping_magnitude": 0.0,
            "corrective_intervention": bool(unique_corrections),
            "corrective_intervention_type": "+".join(unique_corrections) or "none",
            "corrective_intervention_categories": list(unique_corrections),
            "corrective_intervention_magnitude": corrective_magnitude,
            "correction_magnitude": corrective_magnitude,
            "raw_violation_categories": list(violation_categories),
            "raw_violation_count": len(violation_categories),
            "raw_violation_max": max(violation_values, default=0.0),
            "raw_violation_sum": float(sum(violation_values)),
            "raw_day_ahead_violation": float(
                raw_violations.get("day_ahead_envelope", 0.0)
                + raw_violations.get("day_ahead_freeze", 0.0)
            ),
            "raw_compute_violation": float(
                raw_violations.get("compute_envelope", 0.0)
            ),
            "raw_battery_violation": float(
                raw_violations.get("battery_envelope", 0.0)
            ),
            "raw_grid_pv_violation": float(
                raw_violations.get("grid_pv_envelope", 0.0)
            ),
            "raw_migration_violation": float(
                raw_violations.get("migration_fixed_zero", 0.0)
            ),
            # Backward-compatible aliases now mean corrective intervention
            # only.  Structural identity mappings never enter these values.
            "intervention_type": "+".join(unique_corrections) or "none",
            "intervention_magnitude": corrective_magnitude,
            "exogenous_infeasible": exogenous_infeasible,
            "rejected": False,
            "validation": validation,
        }


@dataclass(frozen=True)
class _PolicyDayView:
    """Issue-time-only day view used by the legacy three-argument callback."""

    operating_date: str
    timestamps_utc: np.ndarray
    da_price: np.ndarray
    scenario_suffix: Mapping[str, np.ndarray]
    scenario_weights: np.ndarray
    revealed_history: Mapping[str, np.ndarray]


def _readonly(value: Any) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _context_fingerprint(context: Mapping[str, Any]) -> str:
    # Pure-Python FNV-1a avoids importing OpenSSL-backed hashlib in the same
    # Windows process as Torch/NumPy native runtimes.  This is an audit key, not
    # a cryptographic provenance digest.
    digest = 0xCBF29CE484222325

    def update(raw: bytes) -> None:
        nonlocal digest
        for byte in raw:
            digest ^= byte
            digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    for key in (
        "origin",
        "soc_mwh",
        "queue_age_buckets",
        "current_rt_price",
        "current_workload",
        "current_pv",
        "scenario_weights",
    ):
        update(str(key).encode("utf-8"))
        value = context.get(key)
        if isinstance(value, np.ndarray):
            update(str(value.shape).encode("ascii"))
            update(np.ascontiguousarray(value).tobytes())
        else:
            update(repr(value).encode("utf-8"))
    return f"{digest:016x}"


def _scenario_bundle(
    day: SelectionDay,
    joint_scenarios: Mapping[str, Any] | None,
    scenario_weights: Any | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if joint_scenarios is None:
        scenarios = {
            "rt": np.asarray(day.rt_price_pool[:SCENARIOS], dtype=float).copy(),
            "workload": np.asarray(
                day.workload_pool[:SCENARIOS], dtype=float
            ).copy(),
            "pv": np.asarray(day.pv_pool[:SCENARIOS], dtype=float).copy(),
            "physical": np.repeat(
                np.asarray(day.physical_multiplier, dtype=float)[None, :],
                SCENARIOS,
                axis=0,
            ),
        }
    else:
        scenarios = {
            "rt": _finite_array(
                joint_scenarios["rt"],
                name="joint_scenarios.rt",
                shape=(SCENARIOS, _ZONES, INTERVALS),
            ),
            "workload": _finite_array(
                joint_scenarios["workload"],
                name="joint_scenarios.workload",
                shape=(SCENARIOS, INTERVALS),
            ),
            "pv": _finite_array(
                joint_scenarios["pv"],
                name="joint_scenarios.pv",
                shape=(SCENARIOS, _ZONES, INTERVALS),
            ),
            "physical": _finite_array(
                joint_scenarios.get(
                    "physical",
                    np.repeat(
                        np.asarray(day.physical_multiplier)[None, :],
                        SCENARIOS,
                        axis=0,
                    ),
                ),
                name="joint_scenarios.physical",
                shape=(SCENARIOS, _ZONES),
            ),
        }
    if (scenarios["workload"] < -1e-8).any() or (scenarios["pv"] < -1e-8).any():
        raise R3DataError("joint workload and PV particles must be nonnegative")
    scenarios["workload"] = np.maximum(scenarios["workload"], 0.0)
    scenarios["pv"] = np.maximum(scenarios["pv"], 0.0)
    if (scenarios["physical"] <= 0.0).any():
        raise R3DataError("joint physical multipliers must be positive")
    weights = _normalized_weights(
        np.full(SCENARIOS, 1.0 / SCENARIOS)
        if scenario_weights is None
        else scenario_weights
    )
    return scenarios, weights


def _slice_static_action(
    actions: Mapping[str, Any], origin: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, raw in actions.items():
        value = np.asarray(raw)
        normalized_key = {
            "da": "day_ahead",
            "work": "compute",
            "signed_battery": "battery",
            "rt_grid": "grid",
        }.get(str(key), str(key))
        if normalized_key == "day_ahead":
            output[normalized_key] = value
        elif normalized_key == "migration":
            if value.shape == (INTERVALS, _ZONES, _ZONES):
                output[normalized_key] = value[origin]
            elif value.shape == (_ZONES, _ZONES, INTERVALS):
                output[normalized_key] = value[:, :, origin]
            else:
                output[normalized_key] = value
        elif normalized_key == "fallback":
            output[normalized_key] = (
                value[origin] if value.shape == (INTERVALS,) else value
            )
        else:
            if value.shape == (_ZONES, INTERVALS):
                output[normalized_key] = value[:, origin]
            elif value.shape == (INTERVALS, _ZONES):
                output[normalized_key] = value[origin]
            elif value.shape == (INTERVALS,):
                output[normalized_key] = value[origin]
            else:
                output[normalized_key] = value
    return output


PolicyCallback = Callable[..., Mapping[str, Any]]


class PassiveEDFShieldPolicy:
    """Deterministic work-conserving EDF proposal evaluated by the same shield."""

    name = "passive_edf_shield"

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        queue = _finite_array(
            context["queue_age_buckets"], name="queue_age_buckets"
        )
        target = max(float(queue.sum()), 0.0)
        return {
            "compute": np.full(_ZONES, target / _ZONES, dtype=float),
            "policy_mode": self.name,
        }


class PriceAwareEDFShieldPolicy:
    """Causal calibrated price-aware EDF proposal using the common shield.

    Threshold fitting is deliberately restricted to training or validation
    prices.  Selection/test prices may be consumed causally at inference, but
    can never update the calibration state.
    """

    name = "price_aware_edf_shield"
    _CALIBRATION_SPLITS = frozenset({"train", "training", "validation", "val"})

    def __init__(
        self,
        *,
        low_quantile: float = 0.35,
        high_quantile: float = 0.65,
        low_price_charge_fraction: float = 1.0,
        high_price_discharge_fraction: float = 1.0,
    ) -> None:
        if not 0.0 <= low_quantile < high_quantile <= 1.0:
            raise ValueError("price quantiles must satisfy 0 <= low < high <= 1")
        self.low_quantile = float(low_quantile)
        self.high_quantile = float(high_quantile)
        if not 0.0 <= low_price_charge_fraction <= 1.0:
            raise ValueError("low-price charge fraction must lie in [0,1]")
        if not 0.0 <= high_price_discharge_fraction <= 1.0:
            raise ValueError("high-price discharge fraction must lie in [0,1]")
        self.low_price_charge_fraction = float(low_price_charge_fraction)
        self.high_price_discharge_fraction = float(high_price_discharge_fraction)
        self.low_price: float | None = None
        self.high_price: float | None = None
        self.calibration_split: str | None = None
        self.calibration_observations = 0
        self.calibration_source_hash: str | None = None
        self.calibration_source_label: str | None = None

    @property
    def calibrated(self) -> bool:
        return self.low_price is not None and self.high_price is not None

    def calibrate(
        self,
        price_samples: Any,
        *,
        split: str,
        source_hash: str | None = None,
        source_label: str | None = None,
    ) -> "PriceAwareEDFShieldPolicy":
        normalized_split = str(split).strip().lower()
        if normalized_split not in self._CALIBRATION_SPLITS:
            raise R3DataError(
                "price-aware EDF calibration is restricted to train/validation"
            )
        samples = np.asarray(price_samples, dtype=float).reshape(-1)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise R3DataError("calibration prices must be nonempty and finite")
        self.low_price = float(np.quantile(samples, self.low_quantile))
        self.high_price = float(np.quantile(samples, self.high_quantile))
        self.calibration_split = normalized_split
        self.calibration_observations = int(samples.size)
        self.calibration_source_hash = (
            None if source_hash is None else str(source_hash)
        )
        self.calibration_source_label = (
            None if source_label is None else str(source_label)
        )
        return self

    @classmethod
    def from_calibration(
        cls, metadata: Mapping[str, Any]
    ) -> "PriceAwareEDFShieldPolicy":
        """Reconstruct frozen train/validation thresholds without refitting."""

        if not isinstance(metadata, Mapping):
            raise R3DataError("price calibration metadata must be a mapping")
        policy = cls(
            low_quantile=float(metadata.get("low_quantile", 0.35)),
            high_quantile=float(metadata.get("high_quantile", 0.65)),
            low_price_charge_fraction=float(
                metadata.get("low_price_charge_fraction", 1.0)
            ),
            high_price_discharge_fraction=float(
                metadata.get("high_price_discharge_fraction", 1.0)
            ),
        )
        split = str(metadata.get("calibration_split", "")).strip().lower()
        if split not in cls._CALIBRATION_SPLITS:
            raise R3DataError(
                "persisted price calibration must originate from train/validation"
            )
        low_price = float(metadata["low_price"])
        high_price = float(metadata["high_price"])
        observations = int(metadata.get("calibration_observations", 0))
        if (
            not np.isfinite([low_price, high_price]).all()
            or low_price > high_price
            or observations <= 0
        ):
            raise R3DataError("persisted price calibration metadata is invalid")
        policy.low_price = low_price
        policy.high_price = high_price
        policy.calibration_split = split
        policy.calibration_observations = observations
        policy.calibration_source_hash = (
            None
            if metadata.get("calibration_source_hash") is None
            else str(metadata["calibration_source_hash"])
        )
        policy.calibration_source_label = (
            None
            if metadata.get("calibration_source_label") is None
            else str(metadata["calibration_source_label"])
        )
        return policy

    def calibration_metadata(self) -> dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "calibration_split": self.calibration_split,
            "calibration_observations": self.calibration_observations,
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
            "low_price": self.low_price,
            "high_price": self.high_price,
            "low_price_charge_fraction": self.low_price_charge_fraction,
            "high_price_discharge_fraction": self.high_price_discharge_fraction,
            "calibration_source_hash": self.calibration_source_hash,
            "calibration_source_label": self.calibration_source_label,
        }

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.calibrated:
            raise R3DataError(
                "PriceAwareEDFShieldPolicy must be calibrated on train/validation"
            )
        assert self.low_price is not None and self.high_price is not None
        queue = _finite_array(
            context["queue_age_buckets"], name="queue_age_buckets"
        )
        total = max(float(queue.sum()), 0.0)
        due_now = max(float(queue[0]), 0.0) if queue.size else 0.0
        current_price = float(np.mean(np.asarray(context["current_rt_price"])))
        if current_price <= self.low_price:
            fraction = 1.0
            battery_fraction = self.low_price_charge_fraction
        elif current_price >= self.high_price:
            fraction = 0.0
            battery_fraction = -self.high_price_discharge_fraction
        else:
            width = max(self.high_price - self.low_price, 1e-12)
            fraction = (self.high_price - current_price) / width
            battery_fraction = 0.0
        # Keep price shifting causal without sacrificing the shield's EDF
        # viability envelope.  Each bucket i must receive enough service now
        # that its released prefix remains executable over the next i origins
        # using conservative no-PV/no-battery compute capacity.
        physical_multiplier = _finite_array(
            context["physical_multiplier"],
            name="physical_multiplier",
            shape=(_ZONES,),
        )
        grid_limit = _finite_array(
            context["grid_import_limit_mw"],
            name="grid_import_limit_mw",
            shape=(_ZONES,),
        )
        idle = _finite_array(
            context["idle_mw"], name="idle_mw", shape=(_ZONES,)
        )
        work_power = _finite_array(
            context["work_mw_per_unit"],
            name="work_mw_per_unit",
            shape=(_ZONES,),
        )
        work_capacity = _finite_array(
            context["capacity_work_units"],
            name="capacity_work_units",
            shape=(_ZONES,),
        )
        guaranteed_caps = work_capacity.copy()
        positive_slope = work_power > 1e-12
        safe_site = physical_multiplier * idle <= grid_limit + 1e-12
        guaranteed_caps[~safe_site] = 0.0
        active = positive_slope & safe_site
        guaranteed_caps[active] = np.minimum(
            guaranteed_caps[active],
            np.maximum(
                (
                    grid_limit[active] / physical_multiplier[active]
                    - idle[active]
                )
                / work_power[active],
                0.0,
            ),
        )
        guaranteed_future = float(guaranteed_caps.sum())
        required_now = 0.0
        for remaining in range(queue.size):
            required_now = max(
                required_now,
                float(queue[: remaining + 1].sum())
                - remaining * guaranteed_future,
            )
        # EDF remains work-conserving exactly like the passive comparator; the
        # calibrated price awareness is isolated to the battery rule. This
        # keeps workload/SLA service identical unless a battery envelope is
        # genuinely capacity-binding, rather than confounding price response
        # with discretionary workload deferral.
        target = total
        return {
            "compute": np.full(_ZONES, target / _ZONES, dtype=float),
            # Fractions are converted to physical MW using the current frozen
            # physical artifact supplied in the causal runtime context.
            "battery": battery_fraction
            * _finite_array(
                context["battery_power_mw"],
                name="battery_power_mw",
                shape=(_ZONES,),
            ),
            "policy_mode": self.name,
        }


class CertifiedSafePolicy:
    """Explicit by-construction fallback benchmark, not a learned policy."""

    name = "certified_safe_policy"

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        del context
        return {"fallback": True, "policy_mode": self.name}


class NeuralConstrainedRuntime:
    """Run 96 state-conditioned neural decisions through the hard shield."""

    def __init__(
        self,
        physical: FrozenPhysicalArtifact,
        runtime_config: Mapping[str, Any] | None = None,
        logfile: Path | str | None = None,
    ) -> None:
        self.physical = physical
        self.config = dict(runtime_config or {})
        self.logfile = None if logfile is None else Path(logfile)
        self.shield = CertifiedConstraintShield(physical, self.config)
        self.runtime_build_count = 1
        self.policy_call_count = 0
        self.policy_inference_count = 0
        self.shield_call_count = 0
        self.validation_count = 0
        self.structure_rebuild_count = 0

    def _write_log(self, payload: Mapping[str, Any]) -> None:
        if self.logfile is None:
            return
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        with self.logfile.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    @staticmethod
    def _invoke_callback(
        callback: PolicyCallback,
        context: Mapping[str, Any],
        state: Mapping[str, Any],
        day_view: _PolicyDayView,
    ) -> Mapping[str, Any]:
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(context)
        parameters = tuple(signature.parameters.values())
        positional = tuple(
            item
            for item in parameters
            if item.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters) and not positional:
            return callback(
                origin=int(context["origin"]),
                state=state,
                day=day_view,
                context=context,
            )
        if len(positional) >= 3:
            return callback(int(context["origin"]), state, day_view)
        if len(positional) == 2:
            return callback(int(context["origin"]), state)
        return callback(context)

    def _policy_context(
        self,
        *,
        day: SelectionDay,
        origin: int,
        soc: np.ndarray,
        queue_available: np.ndarray,
        previous_action: Mapping[str, Any] | None,
        frozen_da: np.ndarray | None,
        rolling_scenarios: Mapping[str, np.ndarray],
        weights: np.ndarray,
        revealed_rt: np.ndarray,
        revealed_workload: np.ndarray,
        revealed_pv: np.ndarray,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], _PolicyDayView]:
        scenario_suffix = MappingProxyType(
            {
                "rt": _readonly(rolling_scenarios["rt"][:, :, origin:]),
                "workload": _readonly(
                    rolling_scenarios["workload"][:, origin:]
                ),
                "pv": _readonly(rolling_scenarios["pv"][:, :, origin:]),
                "physical": _readonly(rolling_scenarios["physical"]),
            }
        )
        history = MappingProxyType(
            {
                "rt_price": _readonly(revealed_rt[:, : origin + 1]),
                "workload": _readonly(revealed_workload[: origin + 1]),
                "pv": _readonly(revealed_pv[:, : origin + 1]),
            }
        )
        state = MappingProxyType(
            {
                "origin": int(origin),
                "soc_mwh": _readonly(soc),
                "queue_age_buckets": _readonly(queue_available),
                "previous_action": previous_action,
                "frozen_day_ahead": None
                if frozen_da is None
                else _readonly(frozen_da),
                "scenario_suffix": scenario_suffix,
                "scenario_weights": _readonly(weights),
                "revealed_history": history,
            }
        )
        context_dict: dict[str, Any] = {
            **dict(state),
            "frozen_da": None if frozen_da is None else _readonly(frozen_da),
            "operating_date": day.operating_date,
            "timestamp_utc": str(day.timestamps_utc[origin]),
            "current_rt_price": _readonly(revealed_rt[:, origin]),
            "current_workload": float(revealed_workload[origin]),
            "current_pv": _readonly(revealed_pv[:, origin]),
            "day_ahead_price": _readonly(day.da_price),
            "battery_power_mw": _readonly(self.physical.battery_power_mw),
            "battery_energy_mwh": _readonly(self.physical.battery_energy_mwh),
            "grid_import_limit_mw": _readonly(
                self.physical.grid_import_limit_mw
            ),
            "idle_mw": _readonly(self.physical.idle_mw),
            "work_mw_per_unit": _readonly(
                self.physical.work_mw_per_unit
            ),
            "capacity_work_units": _readonly(
                self.physical.capacity_work_units
            ),
            "physical_multiplier": _readonly(day.physical_multiplier),
        }
        context = MappingProxyType(context_dict)
        day_view = _PolicyDayView(
            operating_date=day.operating_date,
            timestamps_utc=_readonly(day.timestamps_utc),
            da_price=_readonly(day.da_price),
            scenario_suffix=scenario_suffix,
            scenario_weights=_readonly(weights),
            revealed_history=history,
        )
        return context, state, day_view

    @staticmethod
    def _settlement(
        q_da: np.ndarray,
        grid: np.ndarray,
        da_price: np.ndarray,
        rt_price: np.ndarray,
        *,
        battery_cost: float,
        sla_cost: float,
    ) -> dict[str, float]:
        deviation = grid - q_da
        purchase = np.maximum(deviation, 0.0)
        sale = np.maximum(-deviation, 0.0)
        da_cost = INTERVAL_HOURS * float(np.sum(q_da * da_price))
        rt_cost = INTERVAL_HOURS * float(
            np.sum(purchase * rt_price - sale * rt_price)
        )
        return {
            "day_ahead_cost_usd": da_cost,
            "rt_imbalance_cost_usd": rt_cost,
            "battery_degradation_cost_usd": float(battery_cost),
            "sla_penalty_cost_usd": float(sla_cost),
            "simulated_total_cost_usd": float(
                da_cost + rt_cost + battery_cost + sla_cost
            ),
            "settlement_identity_residual_mw": float(
                np.max(
                    np.abs(deviation - (purchase - sale)), initial=0.0
                )
            ),
        }

    def run_day(
        self,
        day: SelectionDay,
        policy_callback: PolicyCallback | None = None,
        *,
        joint_scenarios: Mapping[str, Any] | None = None,
        scenario_weights: Any | None = None,
        neural_actions: Mapping[str, Any] | None = None,
        max_origins: int = INTERVALS,
        formal: bool = False,
    ) -> dict[str, Any]:
        day.validate()
        if policy_callback is not None and neural_actions is not None:
            raise ValueError("provide policy_callback or neural_actions, not both")
        if formal and (
            day.realized_da_price is None or day.realized_rt_price is None
        ):
            raise R3DataError(
                "formal evaluation requires official realized DA and RT prices"
            )
        scenarios, weights = _scenario_bundle(
            day, joint_scenarios, scenario_weights
        )
        if not 1 <= int(max_origins) <= INTERVALS:
            raise ValueError(f"max_origins must lie in [1, {INTERVALS}]")
        risk_scenarios = {name: value.copy() for name, value in scenarios.items()}
        rolling_scenarios = {
            name: value.copy() for name, value in scenarios.items()
        }
        realized_da = (
            np.asarray(day.realized_da_price, dtype=float)
            if day.realized_da_price is not None
            else np.asarray(day.da_price, dtype=float)
        )
        realized_rt = (
            np.asarray(day.realized_rt_price, dtype=float)
            if day.realized_rt_price is not None
            else np.asarray(day.rt_price_pool[0], dtype=float)
        )
        energy = np.asarray(self.physical.battery_energy_mwh, dtype=float)
        soc = np.asarray(day.initial_soc_mwh, dtype=float).copy()
        if (
            (soc < -self.shield.tolerance).any()
            or (soc > energy + self.shield.tolerance).any()
        ):
            return self._infeasible_result(
                day,
                reason="initial SOC is outside frozen battery bounds",
                origin=0,
                records=[],
            )

        # Counters are per rolling execution even when one runtime object is reused.
        self.policy_call_count = 0
        self.policy_inference_count = 0
        self.shield_call_count = 0
        self.validation_count = 0
        self.shield.call_count = 0
        records: list[dict[str, Any]] = []
        queue_prior = np.zeros(self.shield.queue_size, dtype=float)
        frozen_da: np.ndarray | None = None
        previous_action: Mapping[str, Any] | None = None
        executed_grid = np.zeros((_ZONES, INTERVALS), dtype=float)
        executed_da = np.zeros((_ZONES, INTERVALS), dtype=float)
        cumulative_battery_cost = 0.0
        cumulative_sla_cost = 0.0
        cumulative_shortfall = 0.0
        revealed_rt = np.full((_ZONES, INTERVALS), np.nan, dtype=float)
        revealed_workload = np.full(INTERVALS, np.nan, dtype=float)
        revealed_pv = np.full((_ZONES, INTERVALS), np.nan, dtype=float)
        total_started = time.perf_counter()
        infeasible_reason: str | None = None
        infeasible_origin: int | None = None

        for origin in range(int(max_origins)):
            revealed_rt[:, origin] = realized_rt[:, origin]
            revealed_workload[origin] = float(day.realized_workload[origin])
            revealed_pv[:, origin] = day.realized_pv[:, origin]
            # Only the current realization is injected.  Future suffix values
            # remain issue-time particles and no future realization enters the
            # callback context.
            rolling_scenarios["rt"][:, :, origin] = realized_rt[:, origin]
            rolling_scenarios["workload"][:, origin] = day.realized_workload[origin]
            rolling_scenarios["pv"][:, :, origin] = day.realized_pv[:, origin]
            queue_available = queue_prior.copy()
            queue_available[-1] += float(day.realized_workload[origin])
            context, state, day_view = self._policy_context(
                day=day,
                origin=origin,
                soc=soc,
                queue_available=queue_available,
                previous_action=previous_action,
                frozen_da=frozen_da,
                rolling_scenarios=rolling_scenarios,
                weights=weights,
                revealed_rt=revealed_rt,
                revealed_workload=revealed_workload,
                revealed_pv=revealed_pv,
            )
            fingerprint = _context_fingerprint(context)
            inference_started = time.perf_counter()
            if policy_callback is not None:
                raw_action = self._invoke_callback(
                    policy_callback, context, state, day_view
                )
            elif neural_actions is not None:
                raw_action = _slice_static_action(neural_actions, origin)
            else:
                # Explicit deterministic safe-policy baseline, never described
                # as a learned policy.  The hard shield still validates it.
                raw_action = CertifiedSafePolicy()({})
            inference_seconds = time.perf_counter() - inference_started
            self.policy_call_count += 1
            self.policy_inference_count += 1
            if not isinstance(raw_action, Mapping):
                raise R3DataError("policy callback must return a mapping")
            fallback = bool(np.asarray(raw_action.get("fallback", False)).item())
            explicit_policy_mode = raw_action.get("policy_mode")
            policy_mode = str(
                explicit_policy_mode
                if explicit_policy_mode is not None
                else "certified_safe_policy" if fallback else "learned_policy"
            )
            execution_mode = str(
                explicit_policy_mode
                if explicit_policy_mode is not None
                else "certified_safe_fallback"
                if fallback
                else "neural_constraint_shield"
            )
            shield_started = time.perf_counter()
            self.shield_call_count += 1
            try:
                action = self.shield.project(
                    origin=origin,
                    raw_action=raw_action,
                    soc_mwh=soc,
                    queue_before=queue_prior,
                    arrival=float(day.realized_workload[origin]),
                    pv_available=day.realized_pv[:, origin],
                    physical_multiplier=day.physical_multiplier,
                    frozen_day_ahead=frozen_da,
                    fallback=fallback,
                )
            except InfeasibleStateError as exc:
                infeasible_reason = exc.reason
                infeasible_origin = origin
                shield_seconds = time.perf_counter() - shield_started
                # Audit the rejection without emitting any field that could be
                # mistaken for an executed action.
                records.append(
                    {
                        "operating_date": day.operating_date,
                        "origin": origin,
                        "timestamp_utc": str(day.timestamps_utc[origin]),
                        "mode": "rejected_infeasible_state",
                        "policy_mode": policy_mode,
                        "policy_context_fingerprint": fingerprint,
                        "policy_inference_seconds": inference_seconds,
                        "shield_seconds": shield_seconds,
                        "intervention_type": "hard_infeasible_state",
                        "intervention_magnitude": 0.0,
                        "structural_mapping_type": "none",
                        "structural_mapping": False,
                        "structural_mapping_categories": [],
                        "structural_mapping_count": 0,
                        "structural_mapping_magnitude": 0.0,
                        "corrective_intervention": False,
                        "corrective_intervention_type": "none",
                        "corrective_intervention_categories": [],
                        "corrective_intervention_magnitude": 0.0,
                        "correction_magnitude": 0.0,
                        "raw_violation_categories": [],
                        "raw_violation_count": 0,
                        "raw_violation_max": 0.0,
                        "raw_violation_sum": 0.0,
                        "exogenous_infeasible": True,
                        "rejected": True,
                        "valid": False,
                        "passed": False,
                        "reason": exc.reason,
                        "infeasible_details": exc.details,
                    }
                )
                self._write_log(
                    {
                        "origin": origin,
                        "status": "INFEASIBLE",
                        "reason": exc.reason,
                        "details": exc.details,
                    }
                )
                break
            shield_seconds = time.perf_counter() - shield_started
            self.validation_count += 1
            if frozen_da is None:
                frozen_da = np.asarray(action["day_ahead"], dtype=float).copy()
            q_da = frozen_da[:, origin]
            charge = np.asarray(action["charge"], dtype=float)
            discharge = np.asarray(action["discharge"], dtype=float)
            work = np.asarray(action["work"], dtype=float)
            grid = np.asarray(action["grid"], dtype=float)
            battery_cost = float(
                INTERVAL_HOURS
                * np.sum(charge + discharge)
                * float(self.config.get("battery_degradation_usd_per_mwh", 1.0))
            )
            sla_cost = float(
                action["sla_shortfall"]
                * float(self.config.get("sla_penalty_usd_per_work_unit", 1000.0))
            )
            settlement = self._settlement(
                q_da,
                grid,
                realized_da[:, origin],
                realized_rt[:, origin],
                battery_cost=battery_cost,
                sla_cost=sla_cost,
            )
            cumulative_battery_cost += battery_cost
            cumulative_sla_cost += sla_cost
            cumulative_shortfall += float(action["sla_shortfall"])
            validation = dict(action["validation"])
            validation["settlement_identity_residual_mw"] = settlement[
                "settlement_identity_residual_mw"
            ]
            particle_interval_cost = INTERVAL_HOURS * (
                float(np.sum(q_da * day.da_price[:, origin]))
                + np.sum(
                    (grid - q_da)[None, :]
                    * risk_scenarios["rt"][:, :, origin],
                    axis=1,
                )
            ) + battery_cost + sla_cost
            record = {
                "operating_date": day.operating_date,
                "origin": origin,
                "timestamp_utc": str(day.timestamps_utc[origin]),
                "mode": execution_mode,
                "policy_mode": policy_mode,
                "policy_context_fingerprint": fingerprint,
                "policy_inference_seconds": inference_seconds,
                "shield_seconds": shield_seconds,
                "intervention_type": action["intervention_type"],
                "intervention_magnitude": float(action["intervention_magnitude"]),
                "structural_mapping_type": action["structural_mapping_type"],
                "structural_mapping": bool(action["structural_mapping"]),
                "structural_mapping_categories": action[
                    "structural_mapping_categories"
                ],
                "structural_mapping_count": int(action["structural_mapping_count"]),
                "structural_mapping_magnitude": float(
                    action["structural_mapping_magnitude"]
                ),
                "corrective_intervention": bool(
                    action["corrective_intervention"]
                ),
                "corrective_intervention_type": action[
                    "corrective_intervention_type"
                ],
                "corrective_intervention_categories": action[
                    "corrective_intervention_categories"
                ],
                "corrective_intervention_magnitude": float(
                    action["corrective_intervention_magnitude"]
                ),
                "correction_magnitude": float(action["correction_magnitude"]),
                "raw_violation_categories": action["raw_violation_categories"],
                "raw_violation_count": int(action["raw_violation_count"]),
                "raw_violation_max": float(action["raw_violation_max"]),
                "raw_violation_sum": float(action["raw_violation_sum"]),
                "raw_day_ahead_violation": float(
                    action["raw_day_ahead_violation"]
                ),
                "raw_compute_violation": float(action["raw_compute_violation"]),
                "raw_battery_violation": float(action["raw_battery_violation"]),
                "raw_grid_pv_violation": float(action["raw_grid_pv_violation"]),
                "raw_migration_violation": float(
                    action["raw_migration_violation"]
                ),
                "exogenous_infeasible": bool(action["exogenous_infeasible"]),
                "rejected": False,
                "day_ahead_schedule_mw": frozen_da.tolist(),
                "grid_import_by_zone_mw": grid.tolist(),
                "pv_used_by_zone_mw": np.asarray(action["pv_used"]).tolist(),
                "charge_by_zone_mw": charge.tolist(),
                "discharge_by_zone_mw": discharge.tolist(),
                "signed_battery_by_zone_mw": np.asarray(
                    action["signed_battery"]
                ).tolist(),
                "work_by_zone": work.tolist(),
                "migration_by_route": np.asarray(action["migration"]).tolist(),
                "soc_start_by_zone_mwh": soc.tolist(),
                "soc_end_by_zone_mwh": np.asarray(action["soc_end"]).tolist(),
                "queue_age_buckets_before": queue_prior.tolist(),
                "queue_age_buckets_available": np.asarray(
                    action["queue_available"]
                ).tolist(),
                "queue_age_buckets_after": np.asarray(
                    action["queue_after"]
                ).tolist(),
                "workload_arrival_units": float(day.realized_workload[origin]),
                "work_completed_units": float(work.sum()),
                "sla_shortfall_work_units": float(action["sla_shortfall"]),
                "anticipated_sla_shortfall_work_units": float(
                    action["anticipated_sla_shortfall"]
                ),
                "cumulative_sla_shortfall_work_units": cumulative_shortfall,
                "required_service_now_work_units": float(
                    action["required_service_now"]
                ),
                "terminal_soc_lower_next_mwh": np.asarray(
                    action["terminal_soc_lower_next"]
                ).tolist(),
                "grid_import_mw": float(grid.sum()),
                "soc_mwh": float(np.asarray(action["soc_end"]).sum()),
                "total_cost_usd": settlement["simulated_total_cost_usd"],
                "weighted_particle_interval_cvar_95_usd": float(
                    weighted_cvar(particle_interval_cost, weights, 0.95)
                ),
                "valid": True,
                **settlement,
                **validation,
            }
            records.append(record)
            executed_grid[:, origin] = grid
            executed_da[:, origin] = q_da
            soc = np.asarray(action["soc_end"], dtype=float).copy()
            queue_after = np.asarray(action["queue_after"], dtype=float)
            queue_prior = np.zeros_like(queue_after)
            if self.shield.deadline_steps > 0:
                queue_prior[:-1] = queue_after[1:]
            previous_action = MappingProxyType(
                {
                    "grid": _readonly(grid),
                    "battery": _readonly(action["signed_battery"]),
                    "compute": _readonly(work),
                    "soc_end_mwh": _readonly(soc),
                }
            )
            self._write_log(
                {
                    "origin": origin,
                    "status": "accepted",
                    "mode": record["mode"],
                    "validations_passed": bool(validation["passed"]),
                    "policy_inference_seconds": inference_seconds,
                    "shield_seconds": shield_seconds,
                }
            )

        completed_records = [item for item in records if not item.get("rejected", False)]
        completed = len(completed_records)
        if infeasible_reason is not None:
            status = "INFEASIBLE"
        elif completed == INTERVALS:
            status = "COMPLETE"
        else:
            status = "PARTIAL"
        scenario_costs = np.zeros(SCENARIOS, dtype=float)
        if completed:
            horizon = completed
            da_cost = INTERVAL_HOURS * float(
                np.sum(
                    executed_da[:, :horizon]
                    * day.da_price[:, :horizon]
                )
            )
            scenario_imbalance = INTERVAL_HOURS * np.sum(
                (
                    executed_grid[:, :horizon]
                    - executed_da[:, :horizon]
                )[None, :, :]
                * risk_scenarios["rt"][:, :, :horizon],
                axis=(1, 2),
            )
            scenario_costs = (
                da_cost
                + scenario_imbalance
                + cumulative_battery_cost
                + cumulative_sla_cost
            )
        total_arrivals = float(day.realized_workload[:completed].sum())
        valid_flags = [bool(item.get("passed", False)) for item in records]
        corrective_values = np.asarray(
            [
                float(item.get("corrective_intervention_magnitude", 0.0))
                for item in completed_records
            ],
            dtype=float,
        )
        structural_flags = [
            item.get("structural_mapping_type", "none") != "none"
            for item in completed_records
        ]
        summary = {
            "feasibility_rate": float(np.mean(valid_flags)) if valid_flags else 0.0,
            "total_cost_usd": float(
                sum(float(item.get("total_cost_usd", 0.0)) for item in completed_records)
            ),
            "weighted_particle_cvar_95_usd": float(
                weighted_cvar(scenario_costs, weights, 0.95)
            ),
            "weighted_particle_expected_cost_usd": float(
                np.dot(weights, scenario_costs)
            ),
            "weighted_particle_cost_contract": (
                "frozen_issue_time_rt_particles_with_executed_grid_path"
            ),
            "sla_satisfaction": 1.0
            - cumulative_shortfall / max(total_arrivals, 1e-12),
            "sla_shortfall_work_units": cumulative_shortfall,
            "normalized_sla_shortfall": (
                cumulative_shortfall / max(total_arrivals, 1e-12)
            ),
            "policy_inference_time_seconds": float(
                sum(float(item.get("policy_inference_seconds", 0.0)) for item in records)
            ),
            "constraint_shield_time_seconds": float(
                sum(float(item.get("shield_seconds", 0.0)) for item in records)
            ),
            "wall_time_seconds": time.perf_counter() - total_started,
            "structural_mapping_rate": (
                float(np.mean(structural_flags)) if structural_flags else 0.0
            ),
            "corrective_intervention_rate": (
                float(np.mean(corrective_values > self.shield.tolerance))
                if corrective_values.size
                else 0.0
            ),
            "corrective_intervention_mean": (
                float(np.mean(corrective_values)) if corrective_values.size else 0.0
            ),
            "corrective_intervention_p95": (
                float(np.quantile(corrective_values, 0.95))
                if corrective_values.size
                else 0.0
            ),
            "correction_rate": (
                float(np.mean(corrective_values > self.shield.tolerance))
                if corrective_values.size
                else 0.0
            ),
            "correction_magnitude_mean": (
                float(np.mean(corrective_values)) if corrective_values.size else 0.0
            ),
            "correction_magnitude_p95": (
                float(np.quantile(corrective_values, 0.95))
                if corrective_values.size
                else 0.0
            ),
            "raw_violation_origin_rate": (
                float(
                    np.mean(
                        [
                            int(item.get("raw_violation_count", 0)) > 0
                            for item in completed_records
                        ]
                    )
                )
                if completed_records
                else 0.0
            ),
            # Kept for downstream artifact compatibility; its semantics are
            # explicitly corrective-only from this runtime version onward.
            "intervention_rate": (
                float(np.mean(corrective_values > self.shield.tolerance))
                if corrective_values.size
                else 0.0
            ),
            "rejected_origins": int(
                sum(bool(item.get("rejected", False)) for item in records)
            ),
            "exogenous_infeasible_origins": int(
                sum(bool(item.get("exogenous_infeasible", False)) for item in records)
            ),
        }
        result: dict[str, Any] = {
            "status": status,
            "operating_date": day.operating_date,
            "completed_origins": completed,
            "scenario_count": SCENARIOS,
            "runtime_build_count": self.runtime_build_count,
            "policy_call_count": self.policy_call_count,
            "policy_inference_count": self.policy_inference_count,
            "shield_call_count": self.shield_call_count,
            "validation_count": self.validation_count,
            "structure_rebuild_count": self.structure_rebuild_count,
            "migration_contract": (
                "known-zero; source bundle migration_actions_enabled=false"
            ),
            "constraint_contract": (
                "feasible-by-construction hard shield; SLA shortfall explicit"
            ),
            "intervention_accounting_contract": (
                "structural_mapping_separate_from_raw_envelope_correction"
            ),
            "records": records,
            "summary": summary,
        }
        if infeasible_reason is not None:
            result["reason"] = infeasible_reason
            result["infeasible_origin"] = infeasible_origin
        return result

    def _infeasible_result(
        self,
        day: SelectionDay,
        *,
        reason: str,
        origin: int,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "status": "INFEASIBLE",
            "operating_date": day.operating_date,
            "reason": str(reason),
            "infeasible_origin": int(origin),
            "completed_origins": 0,
            "scenario_count": SCENARIOS,
            "runtime_build_count": self.runtime_build_count,
            "policy_call_count": 0,
            "policy_inference_count": 0,
            "shield_call_count": 0,
            "validation_count": 0,
            "structure_rebuild_count": self.structure_rebuild_count,
            "records": records,
            "summary": {
                "feasibility_rate": 0.0,
                "total_cost_usd": 0.0,
                "weighted_particle_cvar_95_usd": 0.0,
                "sla_satisfaction": 0.0,
                "sla_shortfall_work_units": 0.0,
                "normalized_sla_shortfall": 0.0,
                "structural_mapping_rate": 0.0,
                "corrective_intervention_rate": 0.0,
                "corrective_intervention_mean": 0.0,
                "corrective_intervention_p95": 0.0,
                "correction_rate": 0.0,
                "correction_magnitude_mean": 0.0,
                "correction_magnitude_p95": 0.0,
                "raw_violation_origin_rate": 0.0,
                "intervention_rate": 0.0,
                "rejected_origins": len(records),
                "exogenous_infeasible_origins": len(records),
            },
        }


# Practical compatibility names for the CLI/tests while the method name is
# finalized.  They are aliases, not separate implementations.
ConstrainedNeuralVPP = NeuralConstrainedRuntime
ConstraintShieldRuntime = NeuralConstrainedRuntime


def run_policy_benchmarks(
    physical: FrozenPhysicalArtifact,
    runtime_config: Mapping[str, Any] | None,
    day: SelectionDay,
    *,
    policies: Mapping[str, PolicyCallback] | None = None,
    joint_scenarios: Mapping[str, Any] | None = None,
    scenario_weights: Any | None = None,
    max_origins: int = INTERVALS,
    formal: bool = False,
    logfile: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run deterministic policies with identical data, scenarios and accounting.

    When learned ``joint_scenarios`` and ``scenario_weights`` are supplied,
    every policy receives that same frozen set of 20 particles.  Otherwise all
    policies use the same issue-time particle pool embedded in ``day``.
    """

    selected: Mapping[str, PolicyCallback] = policies or {
        PassiveEDFShieldPolicy.name: PassiveEDFShieldPolicy(),
        CertifiedSafePolicy.name: CertifiedSafePolicy(),
    }
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError("policies must be a nonempty mapping")
    outputs: dict[str, dict[str, Any]] = {}
    base_log = None if logfile is None else Path(logfile)
    for name, policy in selected.items():
        policy_name = str(name)
        if not policy_name or not callable(policy):
            raise ValueError("each benchmark policy must have a name and be callable")
        runtime = NeuralConstrainedRuntime(
            physical, runtime_config, logfile=base_log
        )
        result = runtime.run_day(
            day,
            policy_callback=policy,
            joint_scenarios=joint_scenarios,
            scenario_weights=scenario_weights,
            max_origins=max_origins,
            formal=formal,
        )
        result["benchmark_policy"] = policy_name
        if isinstance(policy, PriceAwareEDFShieldPolicy):
            result["calibration"] = policy.calibration_metadata()
        outputs[policy_name] = result
    return outputs


def records_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(result.get("records", [])))


__all__ = [
    "ALLOWED_YEARS",
    "CertifiedConstraintShield",
    "CertifiedSafePolicy",
    "ConstrainedNeuralVPP",
    "ConstraintShieldRuntime",
    "FrozenPhysicalArtifact",
    "INTERVALS",
    "INTERVAL_HOURS",
    "InfeasibleStateError",
    "NeuralConstrainedRuntime",
    "PassiveEDFShieldPolicy",
    "PriceAwareEDFShieldPolicy",
    "R3DataError",
    "SCENARIOS",
    "SelectionDay",
    "exact_weighted_cvar",
    "load_frozen_physical",
    "load_selection_day",
    "records_frame",
    "run_policy_benchmarks",
    "validate_executed_action",
    "validate_record",
    "weighted_cvar",
]
