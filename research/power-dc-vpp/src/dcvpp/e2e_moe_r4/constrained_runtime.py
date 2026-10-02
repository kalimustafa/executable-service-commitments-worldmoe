"""Solver-free R4 rolling runtime and independent physical validator.

No optimization package is imported here.  A neural callback proposes one
action at each of 96 origins; an analytic shield maps it onto the represented
power, battery, workload and terminal-reserve contract.  The mapped action is
then independently recomputed before it may become rolling state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .hashing import sha256_bytes
import pandas as pd

from .data import (
    INTERVALS,
    NYISO_ZONES,
    R4DataError,
    R4SelectionDay,
    SCENARIOS,
)

INTERVAL_HOURS = 0.25
_ZONES = len(NYISO_ZONES)


class InfeasibleStateError(RuntimeError):
    """The current exogenous state admits no represented feasible action."""

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.details = dict(details or {})


def _finite(
    value: Any, *, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    if shape is not None and result.shape != shape:
        raise R4DataError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise R4DataError(f"{name} contains non-finite values")
    return result


@dataclass(frozen=True)
class PhysicalLimits:
    """Immutable four-zone physical limits used by the analytic shield."""

    idle_mw: np.ndarray
    work_mw_per_unit: np.ndarray
    capacity_work_units: np.ndarray
    grid_import_limit_mw: np.ndarray
    battery_power_mw: np.ndarray
    battery_energy_mwh: np.ndarray
    zones: tuple[str, ...] = NYISO_ZONES
    calibration_id: str = "unknown"
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.frozen:
            raise R4DataError("PhysicalLimits must be frozen")
        if tuple(self.zones) != NYISO_ZONES:
            raise R4DataError(f"zones must be exactly {NYISO_ZONES}")
        for name in (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
        ):
            value = _finite(getattr(self, name), name=name, shape=(_ZONES,))
            if (value < 0).any():
                raise R4DataError(f"{name} must be nonnegative")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PhysicalLimits":
        value = payload.get("frozen_artifact", payload.get("physical_limits", payload))
        if not isinstance(value, Mapping):
            raise R4DataError("physical limits must be a mapping")
        return cls(
            idle_mw=value["idle_mw"],
            work_mw_per_unit=value["work_mw_per_unit"],
            capacity_work_units=value["capacity_work_units"],
            grid_import_limit_mw=value["grid_import_limit_mw"],
            battery_power_mw=value.get("battery_power_mw", np.zeros(_ZONES)),
            battery_energy_mwh=value.get("battery_energy_mwh", np.zeros(_ZONES)),
            zones=tuple(map(str, value.get("zones", NYISO_ZONES))),
            calibration_id=str(value.get("calibration_id", "unknown")),
            source_hashes=dict(value.get("source_hashes", {})),
            frozen=bool(value.get("frozen", True)),
        )


def load_physical_limits(path: Path | str) -> PhysicalLimits:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhysicalLimits.from_mapping(payload)


def _core_array(core: Any, name: str) -> np.ndarray:
    value = getattr(core, name, None)
    if value is None:
        raise R4DataError(f"frozen physical core lacks required buffer {name}")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=float).copy()
    if not np.isfinite(result).all():
        raise R4DataError(f"frozen physical core buffer {name} is non-finite")
    return result


class _PhysicalPlant:
    """NumPy execution twin of the frozen artifact-backed Torch core.

    The formal runtime must use the same calibrated piecewise facility curve as
    representation training.  Reading the immutable buffers avoids a second
    fitted model while keeping the analytic shield solver-free.
    """

    def __init__(self, physical: PhysicalLimits, core: Any | None) -> None:
        self.physical = physical
        self.core = core
        self.mode = "legacy_linear_limits"
        self.parity_valid = False
        self.calibration_id = str(physical.calibration_id)
        self.core_buffer_sha256 = ""
        if core is None:
            return
        if getattr(core, "formal_artifact_backed", False) is not True:
            raise R4DataError("runtime physical core is not the formal artifact-backed core")
        if bool(getattr(core, "training", True)):
            raise R4DataError("runtime physical core must be frozen in eval mode")
        parameters = tuple(core.parameters()) if hasattr(core, "parameters") else ()
        if any(bool(getattr(parameter, "requires_grad", False)) for parameter in parameters):
            raise R4DataError("runtime physical core contains trainable parameters")

        bound_names = (
            "idle_mw",
            "work_mw_per_unit",
            "capacity_work_units",
            "grid_import_limit_mw",
            "battery_power_mw",
            "battery_energy_mwh",
        )
        for name in bound_names:
            observed = _core_array(core, name)
            expected = np.asarray(getattr(physical, name), dtype=float)
            if observed.shape != expected.shape or not np.allclose(
                observed, expected, atol=1.0e-12, rtol=0.0
            ):
                raise R4DataError(
                    f"runtime physical limits differ from frozen core buffer {name}"
                )
        self.knots_x = _core_array(core, "it_power_knots_kw")
        self.knots_y = _core_array(core, "facility_power_knots_kw")
        self.zone_scale = _core_array(core, "zone_scale")
        if (
            self.knots_x.ndim != 1
            or self.knots_y.shape != self.knots_x.shape
            or self.knots_x.size < 2
            or np.any(np.diff(self.knots_x) <= 0.0)
            or self.zone_scale.shape != (_ZONES,)
        ):
            raise R4DataError("frozen physical core piecewise buffers are malformed")
        self.slopes = np.diff(self.knots_y) / np.diff(self.knots_x)
        metadata = getattr(core, "artifact_metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        core_calibration_value = metadata.get("calibration_id")
        if not core_calibration_value:
            try:
                core_calibration_value = getattr(core, "calibration_id", "")
            except (KeyError, TypeError, ValueError):
                core_calibration_value = ""
        core_calibration = str(core_calibration_value)
        if not core_calibration:
            raise R4DataError("frozen physical core lacks calibration lineage")
        if physical.calibration_id not in {"", "unknown", core_calibration}:
            raise R4DataError("physical limit/core calibration identifiers differ")
        self.calibration_id = core_calibration
        digest_payload = bytearray()
        for name in (
            *bound_names,
            "it_power_knots_kw",
            "facility_power_knots_kw",
            "zone_scale",
        ):
            array = _core_array(core, name)
            digest_payload.extend(name.encode("utf-8"))
            digest_payload.extend(str(array.shape).encode("ascii"))
            digest_payload.extend(array.astype("<f8", copy=False).tobytes(order="C"))
        digest_payload.extend(
            json.dumps(
                dict(metadata), sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        )
        self.core_buffer_sha256 = sha256_bytes(bytes(digest_payload))
        self.mode = "artifact_piecewise_core_exact"
        self.parity_valid = True

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "artifact_piecewise_core_exact": bool(self.parity_valid),
            "calibration_id": self.calibration_id,
            "core_buffer_sha256": self.core_buffer_sha256,
        }

    def facility_load(self, multiplier: Any, work: Any) -> np.ndarray:
        multiplier_value = _finite(
            multiplier, name="physical_multiplier", shape=(_ZONES,)
        )
        work_value = _finite(work, name="work", shape=(_ZONES,))
        if self.core is None:
            return multiplier_value * (
                self.physical.idle_mw
                + self.physical.work_mw_per_unit * work_value
            )
        utilization = work_value / self.physical.capacity_work_units
        it_reference = self.knots_x[0] + utilization * (
            self.knots_x[-1] - self.knots_x[0]
        )
        if np.any(it_reference <= 0.0):
            raise R4DataError("runtime utilization implies non-positive IT power")
        segment = np.searchsorted(self.knots_x[1:-1], it_reference, side="left")
        prediction = self.knots_y[segment] + self.slopes[segment] * (
            it_reference - self.knots_x[segment]
        )
        facility_reference = np.maximum(prediction, it_reference)
        return facility_reference * self.zone_scale * multiplier_value / 1000.0

    def max_work_for_power(self, power_mw: Any, multiplier: Any) -> np.ndarray:
        available = _finite(power_mw, name="available_power", shape=(_ZONES,))
        multiplier_value = _finite(
            multiplier, name="physical_multiplier", shape=(_ZONES,)
        )
        if self.core is None:
            positive = self.physical.work_mw_per_unit > 1.0e-12
            result = np.zeros(_ZONES, dtype=float)
            result[positive] = np.maximum(
                available[positive] / multiplier_value[positive]
                - self.physical.idle_mw[positive],
                0.0,
            ) / self.physical.work_mw_per_unit[positive]
            result[~positive] = self.physical.capacity_work_units[~positive]
            return np.minimum(result, self.physical.capacity_work_units)

        low = np.zeros(_ZONES, dtype=float)
        high = np.asarray(self.physical.capacity_work_units, dtype=float).copy()
        idle = self.facility_load(multiplier_value, low)
        feasible = available >= idle
        for _ in range(52):
            middle = (low + high) / 2.0
            within = self.facility_load(multiplier_value, middle) <= available
            low = np.where(within, middle, low)
            high = np.where(within, high, middle)
        return np.where(feasible, low, 0.0)


def _lower_violation(value: np.ndarray, lower: Any = 0.0) -> float:
    return float(np.max(np.maximum(np.asarray(lower) - value, 0.0), initial=0.0))


def _upper_violation(value: np.ndarray, upper: Any) -> float:
    return float(np.max(np.maximum(value - np.asarray(upper), 0.0), initial=0.0))


def validate_executed_action(
    *,
    physical: PhysicalLimits,
    physical_multiplier: Any,
    soc_start_mwh: Any,
    soc_end_mwh: Any,
    work: Any,
    grid: Any,
    pv_available: Any,
    pv_used: Any,
    charge: Any,
    discharge: Any,
    migration: Any,
    queue_before: Any,
    queue_after: Any,
    arrivals: float,
    served: float,
    sla_shortfall: float,
    day_ahead: Any,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
    tolerance: float = 1e-6,
    physical_twin_core: Any | None = None,
    _physical_plant: _PhysicalPlant | None = None,
) -> dict[str, float | bool]:
    """Independently recompute every represented hard residual."""

    arrays = {
        name: _finite(value, name=name, shape=(_ZONES,))
        for name, value in {
            "physical_multiplier": physical_multiplier,
            "soc_start_mwh": soc_start_mwh,
            "soc_end_mwh": soc_end_mwh,
            "work": work,
            "grid": grid,
            "pv_available": pv_available,
            "pv_used": pv_used,
            "charge": charge,
            "discharge": discharge,
            "day_ahead": day_ahead,
        }.items()
    }
    migration_value = _finite(
        migration, name="migration", shape=(_ZONES, _ZONES)
    )
    before = _finite(queue_before, name="queue_before")
    after = _finite(queue_after, name="queue_after")
    if before.ndim != 1 or after.shape != before.shape:
        raise R4DataError("queue arrays must be one-dimensional with equal shape")
    multiplier = arrays["physical_multiplier"]
    plant = _physical_plant or _PhysicalPlant(physical, physical_twin_core)
    load = plant.facility_load(multiplier, arrays["work"])
    energy = (
        arrays["grid"]
        + arrays["pv_used"]
        + arrays["discharge"]
        - load
        - arrays["charge"]
    )
    expected_soc = arrays["soc_start_mwh"] + INTERVAL_HOURS * (
        charge_efficiency * arrays["charge"]
        - arrays["discharge"] / discharge_efficiency
    )
    deviation = arrays["grid"] - arrays["day_ahead"]
    queue_residual = abs(
        float(before.sum())
        + float(arrivals)
        - float(served)
        - float(sla_shortfall)
        - float(after.sum())
    )
    result: dict[str, float | bool] = {
        "energy_balance_max_abs_mw": float(np.max(np.abs(energy), initial=0.0)),
        "soc_transition_max_abs_mwh": float(
            np.max(np.abs(arrays["soc_end_mwh"] - expected_soc), initial=0.0)
        ),
        "soc_bound_max_violation_mwh": max(
            _lower_violation(arrays["soc_end_mwh"]),
            _upper_violation(arrays["soc_end_mwh"], physical.battery_energy_mwh),
        ),
        "grid_bound_max_violation_mw": max(
            _lower_violation(arrays["grid"]),
            _upper_violation(arrays["grid"], physical.grid_import_limit_mw),
        ),
        "pv_bound_max_violation_mw": max(
            _lower_violation(arrays["pv_used"]),
            _upper_violation(arrays["pv_used"], arrays["pv_available"]),
        ),
        "compute_bound_max_violation_work_units": max(
            _lower_violation(arrays["work"]),
            _upper_violation(arrays["work"], physical.capacity_work_units),
        ),
        "battery_power_max_violation_mw": max(
            _lower_violation(arrays["charge"]),
            _lower_violation(arrays["discharge"]),
            _upper_violation(arrays["charge"], physical.battery_power_mw),
            _upper_violation(arrays["discharge"], physical.battery_power_mw),
        ),
        "battery_simultaneous_charge_discharge_max_mw": float(
            np.max(np.minimum(arrays["charge"], arrays["discharge"]), initial=0.0)
        ),
        "migration_max_violation_work_units": float(
            np.max(np.abs(migration_value), initial=0.0)
        ),
        "sla_balance_max_abs_work_units": queue_residual,
        "settlement_identity_residual_mw": float(
            np.max(
                np.abs(
                    deviation
                    - (np.maximum(deviation, 0.0) - np.maximum(-deviation, 0.0))
                ),
                initial=0.0,
            )
        ),
    }
    keys = tuple(result)
    result["passed"] = bool(
        max(float(result[name]) for name in keys) <= float(tolerance)
    )
    return result


def _capped_simplex(raw: np.ndarray, caps: np.ndarray, total: float) -> np.ndarray:
    target = float(np.clip(total, 0.0, caps.sum()))
    if target <= 0:
        return np.zeros_like(raw)
    if target >= float(caps.sum()) - 1e-12:
        return caps.copy()
    low, high = float(np.min(raw - caps)), float(np.max(raw))
    for _ in range(64):
        middle = (low + high) / 2.0
        if float(np.clip(raw - middle, 0.0, caps).sum()) > target:
            low = middle
        else:
            high = middle
    result = np.clip(raw - high, 0.0, caps)
    residual = target - float(result.sum())
    for index in np.argsort(-(caps - result), kind="stable"):
        if residual <= 1e-10:
            break
        amount = min(residual, float(caps[index] - result[index]))
        result[index] += amount
        residual -= amount
    return result


def _vector(value: Any, *, name: str, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(_ZONES, float(default))
    result = _to_numpy(value)
    if result.shape == (1, _ZONES):
        result = result[0]
    if result.ndim == 0:
        result = np.full(_ZONES, float(result))
    return _finite(result, name=name, shape=(_ZONES,))


def _day_ahead(value: Any, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default.copy()
    result = _to_numpy(value)
    if result.shape == (1, _ZONES, INTERVALS):
        result = result[0]
    if result.shape == (INTERVALS, _ZONES):
        result = result.T
    if result.shape == (_ZONES,):
        result = np.repeat(result[:, None], INTERVALS, axis=1)
    return _finite(result, name="day_ahead", shape=(_ZONES, INTERVALS))


class AnalyticConstraintShield:
    """One-origin EDF/SOC/terminal-reserve projection without a solver."""

    def __init__(
        self,
        physical: PhysicalLimits,
        config: Mapping[str, Any] | None = None,
        *,
        physical_twin_core: Any | None = None,
    ) -> None:
        self.physical = physical
        self.config = dict(config or {})
        self.plant = _PhysicalPlant(physical, physical_twin_core)
        self.charge_efficiency = float(self.config.get("charge_efficiency", 0.95))
        self.discharge_efficiency = float(
            self.config.get("discharge_efficiency", 0.95)
        )
        self.terminal_soc_fraction = float(
            self.config.get("terminal_soc_fraction", 0.5)
        )
        self.deadline_steps = int(self.config.get("deadline_steps", 4))
        self.tolerance = float(self.config.get("validation_tolerance", 1e-6))
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("battery efficiencies must lie in (0,1]")
        if not 0 <= self.terminal_soc_fraction <= 1 or self.deadline_steps < 0:
            raise ValueError("invalid terminal reserve or deadline")

    @property
    def queue_size(self) -> int:
        return self.deadline_steps + 1

    def _terminal_lower(self, origin: int, multiplier: np.ndarray) -> np.ndarray:
        idle = self.plant.facility_load(multiplier, np.zeros(_ZONES, dtype=float))
        guaranteed_charge = np.minimum(
            self.physical.battery_power_mw,
            np.maximum(self.physical.grid_import_limit_mw - idle, 0.0),
        )
        recoverable = (
            (INTERVALS - origin - 1)
            * INTERVAL_HOURS
            * self.charge_efficiency
            * guaranteed_charge
        )
        reserve = self.terminal_soc_fraction * self.physical.battery_energy_mwh
        return np.maximum(reserve - recoverable, 0.0)

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
        frozen_day_ahead: Any | None,
    ) -> dict[str, Any]:
        if not 0 <= int(origin) < INTERVALS:
            raise ValueError("origin must lie in [0,95]")
        soc = _finite(soc_mwh, name="soc_mwh", shape=(_ZONES,))
        queue_prior = _finite(queue_before, name="queue_before")
        if queue_prior.shape != (self.queue_size,) or (queue_prior < 0).any():
            raise R4DataError(f"queue_before must be nonnegative [{self.queue_size}]")
        pv = np.maximum(_finite(pv_available, name="pv", shape=(_ZONES,)), 0.0)
        multiplier = _finite(
            physical_multiplier, name="physical_multiplier", shape=(_ZONES,)
        )
        if (multiplier <= 0).any():
            raise R4DataError("physical_multiplier must be positive")
        if (soc < -self.tolerance).any() or (
            soc > self.physical.battery_energy_mwh + self.tolerance
        ).any():
            raise InfeasibleStateError("SOC is outside battery bounds")

        structural: list[str] = []
        corrective: list[str] = []
        raw_violations: dict[str, float] = {}

        def structure(name: str) -> None:
            structural.append(name)

        def correction(name: str, amount: float) -> None:
            value = max(float(amount), 0.0)
            if value > self.tolerance:
                corrective.append(name)
                raw_violations[name] = raw_violations.get(name, 0.0) + value

        default_interval = np.clip(
            self.plant.facility_load(multiplier, np.zeros(_ZONES, dtype=float)),
            0.0,
            self.physical.grid_import_limit_mw,
        )
        default_da = np.repeat(default_interval[:, None], INTERVALS, axis=1)
        raw_da = raw_action.get("day_ahead", raw_action.get("da"))
        if origin == 0:
            proposed_da = _day_ahead(raw_da, default_da)
            day_ahead = np.clip(
                proposed_da, 0.0, self.physical.grid_import_limit_mw[:, None]
            )
            if raw_da is None:
                structure("day_ahead_safe_default")
            else:
                correction("day_ahead_envelope", np.abs(day_ahead - proposed_da).sum())
        else:
            if frozen_day_ahead is None:
                raise RuntimeError("day-ahead schedule must be frozen after origin zero")
            day_ahead = _day_ahead(frozen_day_ahead, default_da)
            if raw_da is not None:
                proposed_da = _day_ahead(raw_da, day_ahead)
                correction("day_ahead_freeze", np.abs(proposed_da - day_ahead).sum())

        lower_next = self._terminal_lower(origin, multiplier)
        required_charge = np.minimum(
            self.physical.battery_power_mw,
            np.maximum(
                (lower_next - soc) / (INTERVAL_HOURS * self.charge_efficiency),
                0.0,
            ),
        )
        max_discharge = np.minimum(
            self.physical.battery_power_mw,
            np.maximum(
                self.discharge_efficiency * (soc - lower_next) / INTERVAL_HOURS,
                0.0,
            ),
        )
        idle_load = self.plant.facility_load(
            multiplier, np.zeros(_ZONES, dtype=float)
        )
        available = self.physical.grid_import_limit_mw + pv + max_discharge - required_charge
        if (idle_load > available + self.tolerance).any():
            raise InfeasibleStateError("idle demand exceeds admissible supply")
        caps = self.plant.max_work_for_power(available, multiplier)
        queue_available = queue_prior.copy()
        queue_available[-1] += max(float(arrival), 0.0)
        max_service = min(float(queue_available.sum()), float(caps.sum()))
        guaranteed_caps = self.plant.max_work_for_power(
            self.physical.grid_import_limit_mw, multiplier
        )
        guaranteed_future = float(guaranteed_caps.sum())
        required_now = max(
            [
                float(queue_available[: remaining + 1].sum())
                - remaining * guaranteed_future
                for remaining in range(self.queue_size)
            ]
            + [0.0]
        )
        proposed_compute_present = "compute" in raw_action or "work" in raw_action
        proposed_compute = _vector(
            raw_action.get("compute", raw_action.get("work")), name="compute"
        )
        if not proposed_compute_present:
            structure("compute_safe_default")
        desired = float(
            np.clip(np.maximum(proposed_compute, 0.0).sum(), min(required_now, max_service), max_service)
        )
        work = _capped_simplex(np.maximum(proposed_compute, 0.0), caps, desired)
        structure("edf_queue_accounting")
        if proposed_compute_present:
            correction("edf_compute_projection", np.abs(work - proposed_compute).sum())

        load = self.plant.facility_load(multiplier, work)
        lo_soc = np.maximum(
            -self.physical.battery_power_mw,
            -self.discharge_efficiency * (soc - lower_next) / INTERVAL_HOURS,
        )
        hi_soc = np.minimum(
            self.physical.battery_power_mw,
            (self.physical.battery_energy_mwh - soc)
            / (INTERVAL_HOURS * self.charge_efficiency),
        )
        battery_lower = np.maximum(lo_soc, -load)
        battery_upper = np.minimum(
            hi_soc, self.physical.grid_import_limit_mw + pv - load
        )
        if (battery_lower > battery_upper + self.tolerance).any():
            raise InfeasibleStateError("battery and power envelopes do not intersect")
        battery_present = "battery" in raw_action or "signed_battery" in raw_action
        proposed_battery = _vector(
            raw_action.get("battery", raw_action.get("signed_battery")),
            name="battery",
        )
        if not battery_present:
            structure("battery_safe_default")
        signed_battery = np.clip(proposed_battery, battery_lower, battery_upper)
        if battery_present:
            correction("signed_battery_projection", np.abs(signed_battery - proposed_battery).sum())
        charge = np.maximum(signed_battery, 0.0)
        discharge = np.maximum(-signed_battery, 0.0)
        structure("signed_battery_charge_discharge_split")
        soc_end = np.clip(
            soc
            + INTERVAL_HOURS
            * (
                self.charge_efficiency * charge
                - discharge / self.discharge_efficiency
            ),
            0.0,
            self.physical.battery_energy_mwh,
        )

        net = load + signed_battery
        grid_lower = np.maximum(0.0, net - pv)
        grid_upper = np.minimum(self.physical.grid_import_limit_mw, net)
        if (grid_lower > grid_upper + self.tolerance).any():
            raise InfeasibleStateError("no grid/PV split closes energy balance")
        structural_diagnostics = raw_action.get("_structural_diagnostics", {})
        if not isinstance(structural_diagnostics, Mapping):
            raise R4DataError("structural action diagnostics must be a mapping")
        raw_grid = structural_diagnostics.get(
            "grid", structural_diagnostics.get("real_time")
        )
        raw_pv = structural_diagnostics.get("pv_used")
        proposed_grid = None if raw_grid is None else _vector(raw_grid, name="grid")
        proposed_pv = None if raw_pv is None else _vector(raw_pv, name="pv_used")
        structure("grid_pv_energy_identity")
        # Grid and PV are not free policy actions.  They are derived exactly
        # from load, battery and PV availability.  Any model-emitted values are
        # retained only as diagnostics and can never create a "correction".
        grid = grid_lower.copy()
        pv_used = net - grid
        if proposed_grid is not None or proposed_pv is not None:
            structure("grid_pv_model_outputs_diagnostic_only")
        grid_diagnostic_l1 = (
            0.0 if proposed_grid is None else float(np.abs(grid - proposed_grid).sum())
        )
        pv_diagnostic_l1 = (
            0.0 if proposed_pv is None else float(np.abs(pv_used - proposed_pv).sum())
        )

        raw_migration = raw_action.get("migration")
        proposed_migration = (
            np.zeros((_ZONES, _ZONES))
            if raw_migration is None
            else _finite(
                np.squeeze(_to_numpy(raw_migration), axis=0)
                if _to_numpy(raw_migration).shape == (1, _ZONES, _ZONES)
                else _to_numpy(raw_migration),
                name="migration",
                shape=(_ZONES, _ZONES),
            )
        )
        migration = np.zeros((_ZONES, _ZONES))
        structure("selection_migration_known_zero")
        correction("migration_fixed_zero", np.abs(proposed_migration).sum())

        queue_after = queue_available.copy()
        service_left = float(work.sum())
        for bucket in range(self.queue_size):
            served_here = min(service_left, float(queue_after[bucket]))
            queue_after[bucket] -= served_here
            service_left -= served_here
        sla_shortfall = max(float(queue_after[0]), 0.0)
        queue_after[0] = 0.0
        if sla_shortfall > self.tolerance:
            structure("explicit_sla_slack_accounting")

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
            arrivals=float(arrival),
            served=float(work.sum()),
            sla_shortfall=sla_shortfall,
            day_ahead=day_ahead[:, origin],
            charge_efficiency=self.charge_efficiency,
            discharge_efficiency=self.discharge_efficiency,
            tolerance=self.tolerance,
            _physical_plant=self.plant,
        )
        if not validation["passed"]:
            raise InfeasibleStateError(
                "independent action validation failed", details=validation
            )
        structural_unique = tuple(dict.fromkeys(structural))
        corrective_unique = tuple(dict.fromkeys(corrective))
        amounts = tuple(raw_violations.values())
        return {
            "day_ahead": day_ahead,
            "grid": grid,
            "pv_used": pv_used,
            "signed_battery": signed_battery,
            "charge": charge,
            "discharge": discharge,
            "work": work,
            "migration": migration,
            "soc_end": soc_end,
            "queue_available": queue_available,
            "queue_after": queue_after,
            "sla_shortfall": sla_shortfall,
            "required_service_now": required_now,
            "terminal_soc_lower_next": lower_next,
            "structural_mapping": bool(structural_unique),
            "structural_mapping_categories": list(structural_unique),
            "structural_mapping_type": "+".join(structural_unique) or "none",
            "structural_mapping_magnitude": 0.0,
            "structural_diagnostics": {
                "model_grid_present": proposed_grid is not None,
                "model_pv_used_present": proposed_pv is not None,
                "model_grid_l1_to_derived": grid_diagnostic_l1,
                "model_pv_l1_to_derived": pv_diagnostic_l1,
            },
            "corrective_intervention": bool(corrective_unique),
            "corrective_intervention_categories": list(corrective_unique),
            "corrective_intervention_type": "+".join(corrective_unique) or "none",
            "corrective_intervention_magnitude": float(sum(amounts)),
            "raw_violation_count": len(corrective_unique),
            "raw_violation_max": max(amounts, default=0.0),
            "raw_violation_sum": float(sum(amounts)),
            "validation": validation,
        }


def _to_numpy(value: Any) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


def _readonly(value: Any) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _normalize_weights(value: Any | None) -> np.ndarray:
    weights = (
        np.full(SCENARIOS, 1.0 / SCENARIOS)
        if value is None
        else np.squeeze(_to_numpy(value))
    )
    weights = _finite(weights, name="scenario_weights", shape=(SCENARIOS,))
    if (weights < 0).any() or float(weights.sum()) <= 0:
        raise R4DataError("20 scenario weights must be nonnegative with positive sum")
    return weights / float(weights.sum())


def _validate_particles(value: Any) -> np.ndarray:
    particles = _to_numpy(value)
    if particles.ndim in (4, 5) and particles.shape[0] == 1:
        particles = particles[0]
    if (
        particles.ndim not in (3, 4)
        or particles.shape[0] != SCENARIOS
        or particles.shape[1] != INTERVALS
    ):
        raise R4DataError(
            "scenario particles must have shape [20,96,variables] or "
            "[20,96,zones,variables] after removing batch"
        )
    if not np.isfinite(particles).all():
        raise R4DataError("scenario particles contain non-finite values")
    return particles


def _scenario_contract(
    day: R4SelectionDay,
    joint_scenarios: Any | None,
    scenario_weights: Any | None,
) -> tuple[Any, np.ndarray]:
    if joint_scenarios is None:
        scenarios: Any = MappingProxyType(
            {
                "rt": _readonly(day.rt_price_pool[:SCENARIOS]),
                "workload": _readonly(day.workload_pool[:SCENARIOS]),
                "pv": _readonly(day.pv_pool[:SCENARIOS]),
            }
        )
    elif isinstance(joint_scenarios, Mapping):
        if "particles_zonal" in joint_scenarios or "particles" in joint_scenarios:
            scenarios = _validate_particles(
                joint_scenarios.get("particles_zonal", joint_scenarios.get("particles"))
            )
            if scenario_weights is None:
                scenario_weights = joint_scenarios.get("weights")
        else:
            scenarios = MappingProxyType(
                {
                    "rt": _finite(
                        joint_scenarios["rt"],
                        name="joint_scenarios.rt",
                        shape=(SCENARIOS, _ZONES, INTERVALS),
                    ),
                    "workload": _finite(
                        joint_scenarios["workload"],
                        name="joint_scenarios.workload",
                        shape=(SCENARIOS, INTERVALS),
                    ),
                    "pv": _finite(
                        joint_scenarios["pv"],
                        name="joint_scenarios.pv",
                        shape=(SCENARIOS, _ZONES, INTERVALS),
                    ),
                }
            )
    else:
        scenarios = _validate_particles(joint_scenarios)
    return scenarios, _normalize_weights(scenario_weights)


def _extract_actions(output: Any) -> tuple[dict[str, Any], Any, Any, Any]:
    if not isinstance(output, Mapping):
        raise R4DataError("policy callback must return a mapping")
    action_value = output.get("actions", output.get("policy", output))
    if not isinstance(action_value, Mapping):
        raise R4DataError("callback actions/policy must be a mapping")
    actions = dict(action_value)
    structural_diagnostics = {
        name: actions.pop(name)
        for name in ("grid", "real_time", "pv_used")
        if name in actions
    }
    if structural_diagnostics:
        actions["_structural_diagnostics"] = structural_diagnostics
    policy_memory = actions.pop(
        "policy_memory_next",
        output.get("policy_memory_next"),
    )
    liquid_state = output.get("liquid_state_next")
    scenario_value = output.get("scenarios", output.get("scenario"))
    return actions, policy_memory, liquid_state, scenario_value


PolicyCallback = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class R4Runtime:
    """Execute one R4 day through 96 causal model/shield/state transitions."""

    def __init__(
        self,
        physical: PhysicalLimits,
        config: Mapping[str, Any] | None = None,
        *,
        physical_twin_core: Any | None = None,
    ) -> None:
        self.physical = physical
        self.config = dict(config or {})
        self.shield = AnalyticConstraintShield(
            physical,
            self.config,
            physical_twin_core=physical_twin_core,
        )

    @staticmethod
    def _torch_module_reference(callback: Any) -> tuple[Any, Any] | None:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is an R4 dependency
            return None
        if not isinstance(callback, torch.nn.Module):
            return None
        reference = next(callback.parameters(), None)
        if reference is None:
            return torch.device("cpu"), torch.float32
        return reference.device, reference.dtype

    @classmethod
    def _model_input(cls, value: Any, reference: tuple[Any, Any] | None) -> Any:
        """Move numeric arrays to a torch model without touching audit strings."""

        if reference is None:
            return value
        import torch

        device, dtype = reference
        if isinstance(value, Mapping):
            return {
                key: cls._model_input(item, reference) for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return type(value)(cls._model_input(item, reference) for item in value)
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "OUSM":
                return value
            # Runtime audit arrays are intentionally read-only.  A private
            # copy avoids sharing non-writable NumPy storage with the model.
            tensor = torch.as_tensor(np.array(value, copy=True), device=device)
            return tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
        return value

    def _inference_call(self, target: Any, value: Mapping[str, Any]) -> Any:
        if not bool(self.config.get("inference_mode", True)):
            return target(value)
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is an R4 dependency
            return target(value)
        with torch.inference_mode():
            return target(value)

    def _invoke(self, callback: Any, context: Mapping[str, Any]) -> Any:
        target = callback.forward_session if hasattr(callback, "forward_session") else callback
        reference = self._torch_module_reference(callback)
        return self._inference_call(target, self._model_input(context, reference))

    def run_day(
        self,
        day: R4SelectionDay,
        policy_callback: PolicyCallback,
        joint_scenarios: Any | None = None,
        scenario_weights: Any | None = None,
    ) -> dict[str, Any]:
        if day.sealed and not bool(self.config.get("allow_sealed_selection", False)):
            raise R4DataError(
                "R4Runtime requires allow_sealed_selection=true to execute 2024"
            )
        scenarios, weights = _scenario_contract(
            day, joint_scenarios, scenario_weights
        )
        soc = np.asarray(day.initial_soc_mwh, dtype=float).copy()
        queue = np.zeros(self.shield.queue_size)
        frozen_da: np.ndarray | None = None
        previous_action: Mapping[str, Any] = MappingProxyType(
            {
                "grid": _readonly(np.zeros((1, _ZONES))),
                "battery": _readonly(np.zeros((1, _ZONES))),
                "compute": _readonly(np.zeros((1, _ZONES))),
                "migration": _readonly(np.zeros((1, _ZONES, _ZONES))),
            }
        )
        policy_memory: Any = None
        liquid_state: Any = None
        records: list[dict[str, Any]] = []
        revealed_rt = np.zeros((INTERVALS, _ZONES))
        revealed_workload = np.zeros((INTERVALS, 1))
        revealed_pv = np.zeros((INTERVALS, _ZONES))
        revealed_zonal = np.zeros((INTERVALS, 21))
        scenario_source = "provided" if joint_scenarios is not None else "issue_time_empirical"
        issue_context: Any = None
        if hasattr(policy_callback, "prepare_issue_context"):
            issue_batch = {
                "market": _readonly(day.market[None, :, :]),
                "workload": _readonly(day.workload[None, :, :]),
                "physical_static": _readonly(day.physical_static[None, :]),
                "physical_dynamic": _readonly(day.physical_dynamic[None, :, :]),
            }
            reference = self._torch_module_reference(policy_callback)
            issue_context = self._inference_call(
                policy_callback.prepare_issue_context,
                self._model_input(issue_batch, reference),
            )

        for origin in range(INTERVALS):
            revealed_mask = np.arange(INTERVALS) <= origin
            revealed_rt[origin] = day.realized_rt_price[:, origin]
            revealed_workload[origin, 0] = day.realized_workload[origin]
            revealed_pv[origin] = day.realized_pv[:, origin]
            revealed_zonal[origin, 0:4] = day.realized_da_price[:, origin]
            revealed_zonal[origin, 4:8] = day.realized_rt_price[:, origin]
            revealed_zonal[origin, 8:12] = (
                day.realized_da_price[:, origin]
                - day.realized_rt_price[:, origin]
            )
            revealed_zonal[origin, 12] = day.realized_workload[origin]
            revealed_zonal[origin, 13:17] = day.realized_pv[:, origin]
            revealed_zonal[origin, 17:21] = day.realized_pue[origin]
            queue_available = queue.copy()
            queue_available[-1] += float(day.realized_workload[origin])
            backlog_scalar = float(queue_available.sum())
            deadline_scalar = (
                float(queue_available[0] / max(backlog_scalar, 1e-8))
                if backlog_scalar > 0
                else 0.0
            )
            current_soc = soc[None, :]
            backlog = np.full((1, _ZONES), backlog_scalar / _ZONES)
            deadline = np.full((1, _ZONES), deadline_scalar)
            state = MappingProxyType(
                {
                    "current_soc": _readonly(current_soc),
                    "backlog": _readonly(backlog),
                    "deadline_pressure": _readonly(deadline),
                    "queue_age_buckets": _readonly(queue_available[None, :]),
                    "revealed_rt": _readonly(revealed_rt[None, :, :]),
                    "revealed_workload": _readonly(revealed_workload[None, :, :]),
                    "revealed_pv": _readonly(revealed_pv[None, :, :]),
                    "revealed_mask": _readonly(revealed_mask[None, :]),
                    "policy_memory": policy_memory,
                    "liquid_state": liquid_state,
                }
            )
            context = MappingProxyType(
                {
                    "origin": origin,
                    "operating_date": day.operating_date,
                    "issue_context": issue_context,
                    "market": _readonly(day.market[None, :, :]),
                    "workload": _readonly(day.workload[None, :, :]),
                    "physical_static": _readonly(day.physical_static[None, :]),
                    "physical_dynamic": _readonly(day.physical_dynamic[None, :, :]),
                    "state": state,
                    "current_soc": state["current_soc"],
                    "backlog": state["backlog"],
                    "deadline_pressure": state["deadline_pressure"],
                    "previous_action": previous_action,
                    "revealed_mask": state["revealed_mask"],
                    "revealed_zonal": _readonly(revealed_zonal[None, :, :]),
                    "revealed": MappingProxyType(
                        {
                            "zonal": _readonly(revealed_zonal[None, :, :]),
                            "mask": state["revealed_mask"],
                        }
                    ),
                    "frozen_day_ahead": None
                    if frozen_da is None
                    else _readonly(frozen_da[None, :, :]),
                    "policy_memory": policy_memory,
                    "liquid_state": liquid_state,
                    "joint_scenarios": scenarios,
                    "scenario_weights": _readonly(weights[None, :]),
                    "feature_provenance": day.feature_provenance,
                }
            )
            output = self._invoke(policy_callback, context)
            raw_action, next_policy_memory, next_liquid_state, model_scenario = (
                _extract_actions(output)
            )
            if model_scenario is not None:
                if not isinstance(model_scenario, Mapping):
                    raise R4DataError("model scenario output must be a mapping")
                particles = model_scenario.get(
                    "particles_zonal", model_scenario.get("particles")
                )
                if particles is not None:
                    _validate_particles(particles)
                model_weights = model_scenario.get("weights")
                if model_weights is not None:
                    _normalize_weights(model_weights)
            try:
                action = self.shield.project(
                    origin=origin,
                    raw_action=raw_action,
                    soc_mwh=soc,
                    queue_before=queue,
                    arrival=float(day.realized_workload[origin]),
                    pv_available=day.realized_pv[:, origin],
                    physical_multiplier=day.physical_multiplier,
                    frozen_day_ahead=frozen_da,
                )
            except InfeasibleStateError as exc:
                return {
                    "status": "INFEASIBLE",
                    "operating_date": day.operating_date,
                    "failed_origin": origin,
                    "reason": exc.reason,
                    "details": exc.details,
                    "records": records,
                    "completed_origins": len(records),
                    "scenario_count": SCENARIOS,
                    "scenario_source": scenario_source,
                }
            if frozen_da is None:
                frozen_da = np.asarray(action["day_ahead"]).copy()
            q_da = frozen_da[:, origin]
            grid = np.asarray(action["grid"])
            deviation = grid - q_da
            da_cost = INTERVAL_HOURS * float(
                np.sum(q_da * day.realized_da_price[:, origin])
            )
            rt_cost = INTERVAL_HOURS * float(
                np.sum(deviation * day.realized_rt_price[:, origin])
            )
            validation = dict(action["validation"])
            record = {
                "operating_date": day.operating_date,
                "origin": origin,
                "timestamp_utc": str(day.timestamps_utc[origin]),
                "day_ahead_schedule_mw": frozen_da.tolist(),
                "grid_import_by_zone_mw": grid.tolist(),
                "pv_used_by_zone_mw": np.asarray(action["pv_used"]).tolist(),
                "signed_battery_by_zone_mw": np.asarray(action["signed_battery"]).tolist(),
                "charge_by_zone_mw": np.asarray(action["charge"]).tolist(),
                "discharge_by_zone_mw": np.asarray(action["discharge"]).tolist(),
                "work_by_zone": np.asarray(action["work"]).tolist(),
                "migration_by_route": np.asarray(action["migration"]).tolist(),
                "soc_start_by_zone_mwh": soc.tolist(),
                "soc_end_by_zone_mwh": np.asarray(action["soc_end"]).tolist(),
                "queue_age_buckets_before": queue.tolist(),
                "queue_age_buckets_after": np.asarray(action["queue_after"]).tolist(),
                "workload_arrival_units": float(day.realized_workload[origin]),
                "work_completed_units": float(np.asarray(action["work"]).sum()),
                "sla_shortfall_work_units": float(action["sla_shortfall"]),
                "terminal_soc_lower_next_mwh": np.asarray(
                    action["terminal_soc_lower_next"]
                ).tolist(),
                "structural_mapping": bool(action["structural_mapping"]),
                "structural_mapping_type": action["structural_mapping_type"],
                "structural_mapping_categories": action["structural_mapping_categories"],
                "structural_mapping_magnitude": 0.0,
                "structural_diagnostics": dict(action["structural_diagnostics"]),
                "corrective_intervention": bool(action["corrective_intervention"]),
                "corrective_intervention_type": action["corrective_intervention_type"],
                "corrective_intervention_categories": action[
                    "corrective_intervention_categories"
                ],
                "corrective_intervention_magnitude": float(
                    action["corrective_intervention_magnitude"]
                ),
                "day_ahead_cost_usd": da_cost,
                "rt_imbalance_cost_usd": rt_cost,
                "total_cost_usd": da_cost + rt_cost,
                "valid": bool(validation["passed"]),
                **validation,
            }
            records.append(record)
            soc = np.asarray(action["soc_end"]).copy()
            queue_after = np.asarray(action["queue_after"])
            queue = np.zeros_like(queue_after)
            if self.shield.deadline_steps:
                queue[:-1] = queue_after[1:]
            previous_action = MappingProxyType(
                {
                    "grid": _readonly(grid[None, :]),
                    "battery": _readonly(np.asarray(action["signed_battery"])[None, :]),
                    "compute": _readonly(np.asarray(action["work"])[None, :]),
                    "migration": _readonly(np.asarray(action["migration"])[None, :, :]),
                }
            )
            policy_memory = next_policy_memory
            liquid_state = next_liquid_state

        return {
            "status": "COMPLETE",
            "operating_date": day.operating_date,
            "completed_origins": INTERVALS,
            "records": records,
            "records_frame": records_frame(records),
            "scenario_count": SCENARIOS,
            "scenario_source": scenario_source,
            "all_actions_independently_validated": all(
                bool(record["passed"]) for record in records
            ),
            "day_ahead_frozen_after_origin_zero": all(
                np.array_equal(
                    np.asarray(record["day_ahead_schedule_mw"]), frozen_da
                )
                for record in records
            ),
            "physical_plant_contract": self.shield.plant.contract,
            "final_policy_memory": policy_memory,
            "final_liquid_state": liquid_state,
        }


def validate_record(
    record: Mapping[str, Any],
    *,
    physical: PhysicalLimits,
    day: R4SelectionDay,
    tolerance: float = 1e-6,
    physical_twin_core: Any | None = None,
    _physical_plant: _PhysicalPlant | None = None,
) -> dict[str, float | bool]:
    origin = int(record["origin"])
    schedule = np.asarray(record["day_ahead_schedule_mw"], dtype=float)
    if schedule.shape == (_ZONES, INTERVALS):
        schedule = schedule[:, origin]
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
        day_ahead=schedule,
        tolerance=tolerance,
        physical_twin_core=physical_twin_core,
        _physical_plant=_physical_plant,
    )


def records_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a stable one-row-per-origin audit frame."""

    frame = pd.DataFrame([dict(record) for record in records])
    if not frame.empty:
        frame = frame.sort_values("origin", kind="stable").reset_index(drop=True)
    return frame


__all__ = [
    "AnalyticConstraintShield",
    "INTERVAL_HOURS",
    "InfeasibleStateError",
    "PhysicalLimits",
    "PolicyCallback",
    "R4Runtime",
    "load_physical_limits",
    "records_frame",
    "validate_executed_action",
    "validate_record",
]
