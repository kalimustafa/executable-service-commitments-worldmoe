"""Causal, solver-free runtime for the R4-V7 joint world-policy model.

The module deliberately has one narrow execution contract:

* the complete day-ahead schedule is issued from issue-time inputs before an
  outcome from the operating day is read, and is immutable afterwards;
* one current battery/compute action is executed at each of the 96 origins;
* training and formal replay use the same EDF/SOC/power-envelope equations;
* only day-ahead, signed battery and compute are free actions.

The Torch and NumPy shield implementations below are intentionally kept next
to each other.  ``SHIELD_ALGEBRA_VERSION`` identifies their common equations,
and the parity tests exercise the two paths on the same inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
import inspect
from typing import Any

import numpy as np
import torch
from torch import Tensor


HORIZON = 96
SITES = 4
INTERVAL_HOURS = 0.25
SHIELD_ALGEBRA_VERSION = "dcvpp.r4.v7.shared-edf-soc-envelope.v1"
RUNTIME_SCHEMA_VERSION = "dcvpp.r4.v7.runtime-result.v1"


class V7RuntimeError(RuntimeError):
    """Raised when a V7 causal or action contract is violated."""


class Locked2025Error(V7RuntimeError):
    """Raised before a locked-2025 object or path is evaluated."""


class ActionSchemaError(V7RuntimeError):
    """Raised when a policy exposes an action outside the V7 action space."""


def _contains_forbidden_action(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "migration" in str(key).lower():
                return True
            if _contains_forbidden_action(item):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_forbidden_action(item) for item in value)
    return False


def reject_removed_actions(value: Any) -> None:
    """Fail closed if an input attempts to reintroduce a removed action."""

    if _contains_forbidden_action(value):
        raise ActionSchemaError("the supplied action is outside the V7 action schema")


def _date_text(value: Any) -> str:
    text = str(value)[:10]
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise V7RuntimeError(f"invalid operating date: {value!r}") from exc
    if parsed.year == 2025:
        raise Locked2025Error("locked test year is not available to V7 selection/evaluation")
    return parsed.isoformat()


def guard_unlocked_path(path: Any) -> None:
    """Reject a path containing a 2025 date component before any file open."""

    from pathlib import Path

    for part in Path(path).parts:
        lowered = str(part).lower()
        if lowered == "2025" or lowered.startswith("2025-") or "locked_2025" in lowered:
            raise Locked2025Error("locked test path is not available to V7")


def _np_vector(value: Any, *, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array[None], SITES)
    if array.shape != (SITES,) or not np.isfinite(array).all():
        raise V7RuntimeError(f"{name} must be four finite values")
    if positive and np.any(array <= 0.0):
        raise V7RuntimeError(f"{name} must be strictly positive")
    if not positive and np.any(array < 0.0):
        raise V7RuntimeError(f"{name} must be nonnegative")
    return array.copy()


def _immutable_numpy(value: np.ndarray) -> np.ndarray:
    """Return an array backed by immutable bytes, not a resettable write flag."""

    canonical = np.ascontiguousarray(value)
    return np.frombuffer(canonical.tobytes(order="C"), dtype=canonical.dtype).reshape(
        canonical.shape
    )


@dataclass(frozen=True)
class ShieldLimits:
    """Frozen physical scales used by both policy scaling and the shield."""

    grid_mw: np.ndarray
    battery_mw: np.ndarray
    battery_mwh: np.ndarray
    compute_work: np.ndarray
    idle_mw: np.ndarray
    work_mw_per_unit: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "grid_mw",
            "battery_mw",
            "battery_mwh",
            "compute_work",
            "idle_mw",
            "work_mw_per_unit",
        ):
            value = _np_vector(
                getattr(self, name),
                name=name,
                positive=name in {"grid_mw", "compute_work", "work_mw_per_unit"},
            )
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ShieldLimits") -> "ShieldLimits":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise V7RuntimeError("limits must be a ShieldLimits or mapping")

        def required(*names: str) -> Any:
            for name in names:
                if name in value:
                    return value[name]
            raise V7RuntimeError(f"limits are missing {names[0]}")

        return cls(
            grid_mw=required("grid_mw", "grid_import_limit_mw"),
            battery_mw=required("battery_mw", "battery_power_mw"),
            battery_mwh=required("battery_mwh", "battery_energy_mwh"),
            compute_work=required("compute_work", "capacity_work_units"),
            idle_mw=required("idle_mw"),
            work_mw_per_unit=required("work_mw_per_unit"),
        )

    def numpy(self, batch: int | None = None) -> dict[str, np.ndarray]:
        result = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in (
                "grid_mw",
                "battery_mw",
                "battery_mwh",
                "compute_work",
                "idle_mw",
                "work_mw_per_unit",
            )
        }
        if batch is not None:
            result = {name: np.broadcast_to(item, (int(batch), SITES)) for name, item in result.items()}
        return result

    def torch(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        batch: int | None = None,
    ) -> dict[str, Tensor]:
        result = {
            name: torch.as_tensor(np.array(getattr(self, name), copy=True), device=device, dtype=dtype)
            for name in (
                "grid_mw",
                "battery_mw",
                "battery_mwh",
                "compute_work",
                "idle_mw",
                "work_mw_per_unit",
            )
        }
        if batch is not None:
            result = {name: item.unsqueeze(0).expand(int(batch), -1) for name, item in result.items()}
        return result


@dataclass(frozen=True)
class ShieldConfig:
    deadline_steps: int = 4
    terminal_soc_fraction: float = 0.5
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    interval_hours: float = INTERVAL_HOURS
    sla_penalty_per_work: float = 0.0
    battery_degradation_per_mwh: float = 0.0
    tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        if int(self.deadline_steps) < 0:
            raise V7RuntimeError("deadline_steps must be nonnegative")
        if not 0.0 <= float(self.terminal_soc_fraction) <= 1.0:
            raise V7RuntimeError("terminal_soc_fraction must lie in [0,1]")
        if not 0.0 < float(self.charge_efficiency) <= 1.0:
            raise V7RuntimeError("charge_efficiency must lie in (0,1]")
        if not 0.0 < float(self.discharge_efficiency) <= 1.0:
            raise V7RuntimeError("discharge_efficiency must lie in (0,1]")
        if float(self.interval_hours) <= 0.0 or float(self.tolerance) < 0.0:
            raise V7RuntimeError("interval duration/tolerance is invalid")

    @property
    def queue_size(self) -> int:
        return int(self.deadline_steps) + 1


def _np_batch(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if width is not None and result.shape == (width,):
        result = result[None, :]
    if result.ndim == 0:
        result = result.reshape(1)
    if width is not None and (result.ndim != 2 or result.shape[1] != width):
        raise V7RuntimeError(f"{name} must be [batch,{width}]")
    if not np.isfinite(result).all():
        raise V7RuntimeError(f"{name} contains non-finite values")
    return result


def _torch_batch(
    value: Any,
    *,
    name: str,
    reference: Tensor | None = None,
    width: int | None = None,
) -> Tensor:
    if isinstance(value, Tensor):
        result = value
    else:
        kwargs = {} if reference is None else {"device": reference.device, "dtype": reference.dtype}
        result = torch.as_tensor(value, **kwargs)
    if not result.is_floating_point():
        result = result.to(dtype=torch.float32 if reference is None else reference.dtype)
    if width is not None and result.shape == (width,):
        result = result.unsqueeze(0)
    if result.ndim == 0:
        result = result.reshape(1)
    if width is not None and (result.ndim != 2 or result.shape[1] != width):
        raise V7RuntimeError(f"{name} must be [batch,{width}]")
    if not bool(torch.isfinite(result).all()):
        raise V7RuntimeError(f"{name} contains non-finite values")
    return result


def _expand_np(value: np.ndarray, batch: int, *, name: str) -> np.ndarray:
    if value.shape == (SITES,):
        return np.broadcast_to(value, (batch, SITES))
    if value.shape == (1, SITES):
        return np.broadcast_to(value, (batch, SITES))
    if value.shape != (batch, SITES):
        raise V7RuntimeError(f"{name} cannot broadcast to [batch,4]")
    return value


def _expand_torch(value: Tensor, batch: int, *, name: str) -> Tensor:
    if value.shape == (SITES,):
        return value.unsqueeze(0).expand(batch, -1)
    if value.shape == (1, SITES):
        return value.expand(batch, -1)
    if value.shape != (batch, SITES):
        raise V7RuntimeError(f"{name} cannot broadcast to [batch,4]")
    return value


def scale_normalized_action_numpy(
    action: Mapping[str, Any], limits: Mapping[str, Any] | ShieldLimits
) -> dict[str, np.ndarray]:
    """Scale bounded policy fractions with the real device limits."""

    reject_removed_actions(action)
    source = action.get("free_actions", action)
    if not isinstance(source, Mapping):
        raise ActionSchemaError("free_actions must be a mapping")
    allowed = {"day_ahead", "battery", "compute"}
    unknown = set(map(str, source)).difference(allowed)
    if unknown:
        raise ActionSchemaError(f"unknown V7 free action(s): {sorted(unknown)}")
    frozen = ShieldLimits.from_mapping(limits)
    result: dict[str, np.ndarray] = {}
    if "day_ahead" in source:
        day_ahead = np.asarray(source["day_ahead"], dtype=np.float64)
        if day_ahead.ndim == 2:
            day_ahead = day_ahead[None]
        if day_ahead.shape[-2:] == (SITES, HORIZON):
            day_ahead = np.swapaxes(day_ahead, -1, -2)
        if day_ahead.ndim != 3 or day_ahead.shape[-2:] != (HORIZON, SITES):
            raise ActionSchemaError("day_ahead must be [batch,96,4]")
        result["day_ahead"] = np.clip(day_ahead, 0.0, 1.0) * frozen.grid_mw[None, None, :]
    if "battery" in source:
        battery = _np_batch(source["battery"], name="battery", width=SITES)
        result["battery"] = np.clip(battery, -1.0, 1.0) * frozen.battery_mw[None, :]
    if "compute" in source:
        compute = _np_batch(source["compute"], name="compute", width=SITES)
        result["compute"] = np.clip(compute, 0.0, 1.0) * frozen.compute_work[None, :]
    return result


def scale_normalized_action_torch(
    action: Mapping[str, Any], limits: Mapping[str, Any] | ShieldLimits
) -> dict[str, Tensor]:
    reject_removed_actions(action)
    source = action.get("free_actions", action)
    if not isinstance(source, Mapping):
        raise ActionSchemaError("free_actions must be a mapping")
    allowed = {"day_ahead", "battery", "compute"}
    unknown = set(map(str, source)).difference(allowed)
    if unknown:
        raise ActionSchemaError(f"unknown V7 free action(s): {sorted(unknown)}")
    reference = next((item for item in source.values() if isinstance(item, Tensor)), None)
    if reference is None:
        reference = torch.zeros((), dtype=torch.float32)
    frozen = ShieldLimits.from_mapping(limits)
    scales = frozen.torch(device=reference.device, dtype=reference.dtype)
    result: dict[str, Tensor] = {}
    if "day_ahead" in source:
        day_ahead = torch.as_tensor(source["day_ahead"], device=reference.device, dtype=reference.dtype)
        if day_ahead.ndim == 2:
            day_ahead = day_ahead.unsqueeze(0)
        if day_ahead.shape[-2:] == (SITES, HORIZON):
            day_ahead = day_ahead.transpose(-1, -2)
        if day_ahead.ndim != 3 or day_ahead.shape[-2:] != (HORIZON, SITES):
            raise ActionSchemaError("day_ahead must be [batch,96,4]")
        result["day_ahead"] = day_ahead.clamp(0.0, 1.0) * scales["grid_mw"][None, None]
    if "battery" in source:
        battery = _torch_batch(source["battery"], name="battery", reference=reference, width=SITES)
        result["battery"] = battery.clamp(-1.0, 1.0) * scales["battery_mw"].unsqueeze(0)
    if "compute" in source:
        compute = _torch_batch(source["compute"], name="compute", reference=reference, width=SITES)
        result["compute"] = compute.clamp(0.0, 1.0) * scales["compute_work"].unsqueeze(0)
    return result


def _capped_simplex_numpy(raw: np.ndarray, caps: np.ndarray, total: np.ndarray) -> np.ndarray:
    target = np.clip(total, 0.0, caps.sum(axis=1))
    low = np.min(raw - caps, axis=1)
    high = np.max(raw, axis=1)
    for _ in range(64):
        middle = (low + high) / 2.0
        current = np.clip(raw - middle[:, None], 0.0, caps).sum(axis=1)
        low = np.where(current > target, middle, low)
        high = np.where(current > target, high, middle)
    projected = np.clip(raw - high[:, None], 0.0, caps)
    projected = np.where((target <= 0.0)[:, None], 0.0, projected)
    projected = np.where((target >= caps.sum(axis=1) - 1.0e-12)[:, None], caps, projected)
    return projected


def _capped_simplex_torch(raw: Tensor, caps: Tensor, total: Tensor) -> Tensor:
    target = torch.minimum(total.clamp_min(0.0), caps.sum(dim=1))
    low = (raw - caps).amin(dim=1)
    high = raw.amax(dim=1)
    for _ in range(64):
        middle = (low + high) / 2.0
        current = torch.clamp(raw - middle[:, None], min=0.0)
        current = torch.minimum(current, caps).sum(dim=1)
        more = current > target
        low = torch.where(more, middle, low)
        high = torch.where(more, high, middle)
    projected = torch.clamp(raw - high[:, None], min=0.0)
    projected = torch.minimum(projected, caps)
    projected = torch.where((target <= 0.0)[:, None], torch.zeros_like(projected), projected)
    projected = torch.where(
        (target >= caps.sum(dim=1) - 1.0e-12)[:, None], caps, projected
    )
    return projected


def _np_required_now(queue: np.ndarray, guaranteed_future: np.ndarray) -> np.ndarray:
    candidates = [np.zeros(queue.shape[0], dtype=queue.dtype)]
    for remaining in range(queue.shape[1]):
        candidates.append(queue[:, : remaining + 1].sum(axis=1) - remaining * guaranteed_future)
    return np.maximum(np.stack(candidates, axis=1).max(axis=1), 0.0)


def _torch_required_now(queue: Tensor, guaranteed_future: Tensor) -> Tensor:
    candidates = [torch.zeros(queue.shape[0], device=queue.device, dtype=queue.dtype)]
    for remaining in range(queue.shape[1]):
        candidates.append(queue[:, : remaining + 1].sum(dim=1) - remaining * guaranteed_future)
    return torch.stack(candidates, dim=1).amax(dim=1).clamp_min(0.0)


def _np_serve_edf(queue: np.ndarray, served: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    after = queue.copy()
    left = served.copy()
    for bucket in range(queue.shape[1]):
        amount = np.minimum(left, after[:, bucket])
        after[:, bucket] -= amount
        left -= amount
    sla = np.maximum(after[:, 0], 0.0)
    after[:, 0] = 0.0
    next_queue = np.zeros_like(after)
    if queue.shape[1] > 1:
        next_queue[:, :-1] = after[:, 1:]
    return after, next_queue, sla


def _torch_serve_edf(queue: Tensor, served: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    buckets: list[Tensor] = []
    left = served
    for bucket in range(queue.shape[1]):
        amount = torch.minimum(left, queue[:, bucket])
        buckets.append(queue[:, bucket] - amount)
        left = left - amount
    after = torch.stack(buckets, dim=1)
    sla = after[:, 0].clamp_min(0.0)
    after = torch.cat((torch.zeros_like(after[:, :1]), after[:, 1:]), dim=1)
    next_queue = torch.cat((after[:, 1:], torch.zeros_like(after[:, :1])), dim=1)
    return after, next_queue, sla


def _origin_numpy(value: Any, batch: int) -> np.ndarray:
    origin = np.asarray(value)
    if origin.ndim == 0:
        origin = np.repeat(origin.reshape(1), batch)
    if origin.shape != (batch,) or not np.equal(origin, np.round(origin)).all():
        raise V7RuntimeError("origin must be scalar or [batch] integer indices")
    origin = origin.astype(np.int64)
    if np.any((origin < 0) | (origin >= HORIZON)):
        raise V7RuntimeError("origin must lie in [0,95]")
    return origin


def _origin_torch(value: Any, batch: int, reference: Tensor) -> Tensor:
    origin = torch.as_tensor(value, device=reference.device)
    if origin.ndim == 0:
        origin = origin.expand(batch)
    if origin.shape != (batch,):
        raise V7RuntimeError("origin must be scalar or [batch] integer indices")
    if origin.dtype.is_floating_point and not bool(torch.equal(origin, origin.round())):
        raise V7RuntimeError("origin must contain integer indices")
    origin = origin.long()
    if bool(((origin < 0) | (origin >= HORIZON)).any()):
        raise V7RuntimeError("origin must lie in [0,95]")
    return origin


def numpy_shield_step(
    action: Mapping[str, Any],
    state: Mapping[str, Any],
    exogenous: Mapping[str, Any],
    limits: Mapping[str, Any] | ShieldLimits,
    config: ShieldConfig | None = None,
    *,
    action_is_normalized: bool = False,
) -> dict[str, np.ndarray]:
    """Project a NumPy batch using the formal V7 shield equations."""

    reject_removed_actions(action)
    cfg = config or ShieldConfig()
    frozen = ShieldLimits.from_mapping(limits)
    source = action.get("free_actions", action)
    if not isinstance(source, Mapping):
        raise ActionSchemaError("free_actions must be a mapping")
    if action_is_normalized:
        source = scale_normalized_action_numpy(source, frozen)
    unknown = set(map(str, source)).difference({"day_ahead", "battery", "compute"})
    if unknown:
        raise ActionSchemaError(f"unknown V7 free action(s): {sorted(unknown)}")
    if "battery" not in source or "compute" not in source:
        raise ActionSchemaError("current action requires battery and compute")

    soc = _np_batch(state.get("soc_mwh", state.get("current_soc")), name="soc_mwh", width=SITES)
    queue = _np_batch(state.get("queue"), name="queue")
    batch = soc.shape[0]
    if queue.shape != (batch, cfg.queue_size):
        raise V7RuntimeError(f"queue must be [batch,{cfg.queue_size}]")
    if np.any(queue < 0.0):
        raise V7RuntimeError("queue must be nonnegative")
    origin = _origin_numpy(exogenous.get("origin"), batch)
    arrival = _np_batch(exogenous.get("arrival"), name="arrival")
    if arrival.shape != (batch,) or np.any(arrival < 0.0):
        raise V7RuntimeError("arrival must be nonnegative [batch]")
    pv = _np_batch(exogenous.get("pv_mw"), name="pv_mw", width=SITES)
    multiplier = _np_batch(
        exogenous.get("physical_multiplier", np.ones((batch, SITES))),
        name="physical_multiplier",
        width=SITES,
    )
    if np.any(pv < 0.0) or np.any(multiplier <= 0.0):
        raise V7RuntimeError("PV must be nonnegative and physical multiplier positive")
    arrays = frozen.numpy(batch)
    if np.any(soc < -cfg.tolerance) or np.any(soc > arrays["battery_mwh"] + cfg.tolerance):
        raise V7RuntimeError("SOC is outside the physical battery envelope")
    queue_available = queue.copy()
    queue_available[:, -1] += arrival
    idle = multiplier * arrays["idle_mw"]
    future_charge = np.minimum(arrays["battery_mw"], np.maximum(arrays["grid_mw"] - idle, 0.0))
    remaining = (HORIZON - origin - 1).astype(np.float64)[:, None]
    reserve = cfg.terminal_soc_fraction * arrays["battery_mwh"]
    terminal_lower = np.maximum(
        reserve - remaining * cfg.interval_hours * cfg.charge_efficiency * future_charge,
        0.0,
    )
    required_charge = np.minimum(
        arrays["battery_mw"],
        np.maximum((terminal_lower - soc) / (cfg.interval_hours * cfg.charge_efficiency), 0.0),
    )
    max_discharge = np.minimum(
        arrays["battery_mw"],
        np.maximum(cfg.discharge_efficiency * (soc - terminal_lower) / cfg.interval_hours, 0.0),
    )
    available = arrays["grid_mw"] + pv + max_discharge - required_charge
    caps = np.maximum((available / multiplier - arrays["idle_mw"]) / arrays["work_mw_per_unit"], 0.0)
    caps = np.minimum(caps, arrays["compute_work"])
    guaranteed_caps = np.maximum(
        (arrays["grid_mw"] / multiplier - arrays["idle_mw"]) / arrays["work_mw_per_unit"],
        0.0,
    )
    guaranteed_future = np.minimum(guaranteed_caps, arrays["compute_work"]).sum(axis=1)
    required_now = _np_required_now(queue_available, guaranteed_future)
    max_service = np.minimum(queue_available.sum(axis=1), caps.sum(axis=1))
    raw_compute = _expand_np(_np_batch(source["compute"], name="compute", width=SITES), batch, name="compute")
    raw_battery = _expand_np(_np_batch(source["battery"], name="battery", width=SITES), batch, name="battery")
    desired = np.clip(np.maximum(raw_compute, 0.0).sum(axis=1), np.minimum(required_now, max_service), max_service)
    compute = _capped_simplex_numpy(np.maximum(raw_compute, 0.0), caps, desired)
    load = multiplier * (arrays["idle_mw"] + arrays["work_mw_per_unit"] * compute)
    battery_lower = np.maximum(
        np.maximum(-arrays["battery_mw"], -cfg.discharge_efficiency * (soc - terminal_lower) / cfg.interval_hours),
        -load,
    )
    battery_upper = np.minimum(
        np.minimum(
            arrays["battery_mw"],
            (arrays["battery_mwh"] - soc) / (cfg.interval_hours * cfg.charge_efficiency),
        ),
        arrays["grid_mw"] + pv - load,
    )
    envelope_ok = battery_lower <= battery_upper + cfg.tolerance
    safe_upper = np.maximum(battery_upper, battery_lower)
    signed_battery = np.minimum(np.maximum(raw_battery, battery_lower), safe_upper)
    charge = np.maximum(signed_battery, 0.0)
    discharge = np.maximum(-signed_battery, 0.0)
    soc_next = soc + cfg.interval_hours * (
        cfg.charge_efficiency * charge - discharge / cfg.discharge_efficiency
    )
    net = load + signed_battery
    grid = np.maximum(net - pv, 0.0)
    pv_used = net - grid
    queue_after, queue_next, sla = _np_serve_edf(queue_available, compute.sum(axis=1))
    compute_scale = np.maximum(arrays["compute_work"].sum(axis=1), 1.0e-12)
    battery_scale = np.maximum(arrays["battery_mw"].sum(axis=1), 1.0e-12)
    intervention = (
        np.abs(compute - raw_compute).sum(axis=1) / compute_scale
        + np.abs(signed_battery - raw_battery).sum(axis=1) / battery_scale
    )
    energy_residual = grid + pv_used + discharge - load - charge
    queue_residual = (
        queue.sum(axis=1) + arrival - compute.sum(axis=1) - sla - queue_after.sum(axis=1)
    )
    feasible = (
        envelope_ok.all(axis=1)
        & (np.max(np.abs(energy_residual), axis=1) <= cfg.tolerance)
        & (np.abs(queue_residual) <= cfg.tolerance)
        & (grid >= -cfg.tolerance).all(axis=1)
        & (grid <= arrays["grid_mw"] + cfg.tolerance).all(axis=1)
        & (pv_used >= -cfg.tolerance).all(axis=1)
        & (pv_used <= pv + cfg.tolerance).all(axis=1)
        & (soc_next >= -cfg.tolerance).all(axis=1)
        & (soc_next <= arrays["battery_mwh"] + cfg.tolerance).all(axis=1)
        & (soc_next >= terminal_lower - cfg.tolerance).all(axis=1)
    )
    return {
        "schema_version": np.asarray(SHIELD_ALGEBRA_VERSION),
        "compute": compute,
        "signed_battery": signed_battery,
        "charge": charge,
        "discharge": discharge,
        "load_mw": load,
        "grid_mw": grid,
        "pv_used_mw": pv_used,
        "soc_next_mwh": soc_next,
        "queue_available": queue_available,
        "queue_after_service": queue_after,
        "queue_next": queue_next,
        "required_now": required_now,
        "terminal_soc_lower_next_mwh": terminal_lower,
        "sla_violation": sla,
        "shield_intervention": intervention,
        "feasible": feasible,
        "energy_balance_max_abs_mw": np.max(np.abs(energy_residual), axis=1),
        "queue_balance_abs_work": np.abs(queue_residual),
    }


def torch_shield_step(
    action: Mapping[str, Any],
    state: Mapping[str, Any],
    exogenous: Mapping[str, Any],
    limits: Mapping[str, Any] | ShieldLimits,
    config: ShieldConfig | None = None,
    *,
    action_is_normalized: bool = False,
) -> dict[str, Tensor]:
    """Differentiable projection using the exact formal-shield algebra."""

    reject_removed_actions(action)
    cfg = config or ShieldConfig()
    frozen = ShieldLimits.from_mapping(limits)
    source = action.get("free_actions", action)
    if not isinstance(source, Mapping):
        raise ActionSchemaError("free_actions must be a mapping")
    if action_is_normalized:
        source = scale_normalized_action_torch(source, frozen)
    unknown = set(map(str, source)).difference({"day_ahead", "battery", "compute"})
    if unknown:
        raise ActionSchemaError(f"unknown V7 free action(s): {sorted(unknown)}")
    if "battery" not in source or "compute" not in source:
        raise ActionSchemaError("current action requires battery and compute")

    reference = source["battery"] if isinstance(source["battery"], Tensor) else None
    soc = _torch_batch(state.get("soc_mwh", state.get("current_soc")), name="soc_mwh", reference=reference, width=SITES)
    queue = _torch_batch(state.get("queue"), name="queue", reference=soc)
    batch = soc.shape[0]
    if queue.shape != (batch, cfg.queue_size):
        raise V7RuntimeError(f"queue must be [batch,{cfg.queue_size}]")
    if bool((queue < 0.0).any()):
        raise V7RuntimeError("queue must be nonnegative")
    origin = _origin_torch(exogenous.get("origin"), batch, soc)
    arrival = _torch_batch(exogenous.get("arrival"), name="arrival", reference=soc)
    if arrival.shape != (batch,) or bool((arrival < 0.0).any()):
        raise V7RuntimeError("arrival must be nonnegative [batch]")
    pv = _torch_batch(exogenous.get("pv_mw"), name="pv_mw", reference=soc, width=SITES)
    multiplier = _torch_batch(
        exogenous.get("physical_multiplier", torch.ones_like(soc)),
        name="physical_multiplier",
        reference=soc,
        width=SITES,
    )
    if bool((pv < 0.0).any()) or bool((multiplier <= 0.0).any()):
        raise V7RuntimeError("PV must be nonnegative and physical multiplier positive")
    arrays = frozen.torch(device=soc.device, dtype=soc.dtype, batch=batch)
    if bool((soc < -cfg.tolerance).any()) or bool((soc > arrays["battery_mwh"] + cfg.tolerance).any()):
        raise V7RuntimeError("SOC is outside the physical battery envelope")
    queue_available = torch.cat((queue[:, :-1], queue[:, -1:] + arrival[:, None]), dim=1)
    idle = multiplier * arrays["idle_mw"]
    future_charge = torch.minimum(arrays["battery_mw"], (arrays["grid_mw"] - idle).clamp_min(0.0))
    remaining = (HORIZON - origin - 1).to(dtype=soc.dtype)[:, None]
    reserve = cfg.terminal_soc_fraction * arrays["battery_mwh"]
    terminal_lower = (
        reserve - remaining * cfg.interval_hours * cfg.charge_efficiency * future_charge
    ).clamp_min(0.0)
    required_charge = torch.minimum(
        arrays["battery_mw"],
        ((terminal_lower - soc) / (cfg.interval_hours * cfg.charge_efficiency)).clamp_min(0.0),
    )
    max_discharge = torch.minimum(
        arrays["battery_mw"],
        (cfg.discharge_efficiency * (soc - terminal_lower) / cfg.interval_hours).clamp_min(0.0),
    )
    available = arrays["grid_mw"] + pv + max_discharge - required_charge
    caps = ((available / multiplier - arrays["idle_mw"]) / arrays["work_mw_per_unit"]).clamp_min(0.0)
    caps = torch.minimum(caps, arrays["compute_work"])
    guaranteed_caps = (
        (arrays["grid_mw"] / multiplier - arrays["idle_mw"]) / arrays["work_mw_per_unit"]
    ).clamp_min(0.0)
    guaranteed_future = torch.minimum(guaranteed_caps, arrays["compute_work"]).sum(dim=1)
    required_now = _torch_required_now(queue_available, guaranteed_future)
    max_service = torch.minimum(queue_available.sum(dim=1), caps.sum(dim=1))
    raw_compute = _expand_torch(
        _torch_batch(source["compute"], name="compute", reference=soc, width=SITES), batch, name="compute"
    )
    raw_battery = _expand_torch(
        _torch_batch(source["battery"], name="battery", reference=soc, width=SITES), batch, name="battery"
    )
    desired = torch.maximum(raw_compute.clamp_min(0.0).sum(dim=1), torch.minimum(required_now, max_service))
    desired = torch.minimum(desired, max_service)
    compute = _capped_simplex_torch(raw_compute.clamp_min(0.0), caps, desired)
    load = multiplier * (arrays["idle_mw"] + arrays["work_mw_per_unit"] * compute)
    battery_lower = torch.maximum(
        torch.maximum(
            -arrays["battery_mw"],
            -cfg.discharge_efficiency * (soc - terminal_lower) / cfg.interval_hours,
        ),
        -load,
    )
    battery_upper = torch.minimum(
        torch.minimum(
            arrays["battery_mw"],
            (arrays["battery_mwh"] - soc) / (cfg.interval_hours * cfg.charge_efficiency),
        ),
        arrays["grid_mw"] + pv - load,
    )
    envelope_ok = battery_lower <= battery_upper + cfg.tolerance
    safe_upper = torch.maximum(battery_upper, battery_lower)
    signed_battery = torch.minimum(torch.maximum(raw_battery, battery_lower), safe_upper)
    charge = signed_battery.clamp_min(0.0)
    discharge = (-signed_battery).clamp_min(0.0)
    soc_next = soc + cfg.interval_hours * (
        cfg.charge_efficiency * charge - discharge / cfg.discharge_efficiency
    )
    net = load + signed_battery
    grid = (net - pv).clamp_min(0.0)
    pv_used = net - grid
    queue_after, queue_next, sla = _torch_serve_edf(queue_available, compute.sum(dim=1))
    compute_scale = arrays["compute_work"].sum(dim=1).clamp_min(1.0e-12)
    battery_scale = arrays["battery_mw"].sum(dim=1).clamp_min(1.0e-12)
    intervention = (
        (compute - raw_compute).abs().sum(dim=1) / compute_scale
        + (signed_battery - raw_battery).abs().sum(dim=1) / battery_scale
    )
    energy_residual = grid + pv_used + discharge - load - charge
    queue_residual = queue.sum(dim=1) + arrival - compute.sum(dim=1) - sla - queue_after.sum(dim=1)
    feasible = (
        envelope_ok.all(dim=1)
        & (energy_residual.abs().amax(dim=1) <= cfg.tolerance)
        & (queue_residual.abs() <= cfg.tolerance)
        & (grid >= -cfg.tolerance).all(dim=1)
        & (grid <= arrays["grid_mw"] + cfg.tolerance).all(dim=1)
        & (pv_used >= -cfg.tolerance).all(dim=1)
        & (pv_used <= pv + cfg.tolerance).all(dim=1)
        & (soc_next >= -cfg.tolerance).all(dim=1)
        & (soc_next <= arrays["battery_mwh"] + cfg.tolerance).all(dim=1)
        & (soc_next >= terminal_lower - cfg.tolerance).all(dim=1)
    )
    return {
        "compute": compute,
        "signed_battery": signed_battery,
        "charge": charge,
        "discharge": discharge,
        "load_mw": load,
        "grid_mw": grid,
        "pv_used_mw": pv_used,
        "soc_next_mwh": soc_next,
        "queue_available": queue_available,
        "queue_after_service": queue_after,
        "queue_next": queue_next,
        "required_now": required_now,
        "terminal_soc_lower_next_mwh": terminal_lower,
        "sla_violation": sla,
        "shield_intervention": intervention,
        "feasible": feasible,
        "energy_balance_max_abs_mw": energy_residual.abs().amax(dim=1),
        "queue_balance_abs_work": queue_residual.abs(),
    }


def freeze_day_ahead_numpy(
    value: Any,
    limits: Mapping[str, Any] | ShieldLimits,
    *,
    normalized: bool = False,
    batch: int | None = None,
) -> np.ndarray:
    """Validate, scale if requested, and return an immutable ``[B,96,4]`` DA."""

    reject_removed_actions(value)
    frozen = ShieldLimits.from_mapping(limits)
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 2:
        result = result[None]
    if result.shape[-2:] == (SITES, HORIZON):
        result = np.swapaxes(result, -1, -2)
    if result.ndim != 3 or result.shape[-2:] != (HORIZON, SITES):
        raise ActionSchemaError("issued day_ahead must be [batch,96,4]")
    if batch is not None:
        if result.shape[0] == 1:
            result = np.broadcast_to(result, (int(batch), HORIZON, SITES)).copy()
        elif result.shape[0] != int(batch):
            raise ActionSchemaError("day_ahead batch does not match rollout batch")
    scale = frozen.grid_mw[None, None, :]
    if normalized:
        result = np.clip(result, 0.0, 1.0) * scale
    elif np.any(result < 0.0) or np.any(result > scale + 1.0e-9):
        raise ActionSchemaError("issued day_ahead exceeds the real grid envelope")
    if not np.isfinite(result).all():
        raise ActionSchemaError("issued day_ahead contains non-finite values")
    return _immutable_numpy(np.asarray(result))


def _call_supported(target: Callable[..., Any], **kwargs: Any) -> Any:
    """Call model methods without hiding errors raised inside the model."""

    signature = inspect.signature(target)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return target(**kwargs)
    selected = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return target(**selected)


def _module_reference(policy: Any) -> Tensor | None:
    wrapped = getattr(policy, "base", None)
    if wrapped is not None and wrapped is not policy:
        reference = _module_reference(wrapped)
        if reference is not None:
            return reference
    if not isinstance(policy, torch.nn.Module):
        return None
    return next(policy.parameters(), next(policy.buffers(), None))


def _to_model(value: Any, reference: Tensor | None) -> Any:
    if reference is None:
        return value
    if isinstance(value, Mapping):
        return {key: _to_model(item, reference) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.dtype.kind not in "OUSM":
        result = torch.as_tensor(np.array(value, copy=True), device=reference.device)
        return result.to(dtype=reference.dtype) if result.is_floating_point() else result
    return value


def _broadcast_issue_batch(value: Any, batch: int) -> Any:
    """Repeat one issue-time day across evaluation worlds, never outcomes."""

    if isinstance(value, Mapping):
        return {key: _broadcast_issue_batch(item, batch) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[0] == 1:
        return np.broadcast_to(value, (int(batch), *value.shape[1:])).copy()
    if isinstance(value, Tensor) and value.ndim >= 2 and value.shape[0] == 1:
        return value.expand(int(batch), *value.shape[1:])
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _policy_section(output: Any) -> Mapping[str, Any]:
    reject_removed_actions(output)
    if not isinstance(output, Mapping):
        raise ActionSchemaError("policy output must be a mapping")
    policy = output.get("policy", output)
    if not isinstance(policy, Mapping):
        raise ActionSchemaError("policy section must be a mapping")
    return policy


def _issue_schedule(output: Any) -> Any:
    policy = _policy_section(output)
    issue_mask = policy.get("day_ahead_issue_mask")
    if issue_mask is not None and not bool(np.asarray(_to_numpy(issue_mask)).all()):
        raise ActionSchemaError("the issue call did not authorize every DA schedule")
    free = policy.get("free_actions", policy)
    if not isinstance(free, Mapping) or "day_ahead" not in free:
        raise ActionSchemaError("forward_issue must emit policy.day_ahead")
    return free["day_ahead"]


def _current_action(output: Any, frozen_da: np.ndarray) -> dict[str, Any]:
    policy = _policy_section(output)
    free = policy.get("free_actions", policy.get("actions", policy))
    if not isinstance(free, Mapping):
        raise ActionSchemaError("origin free_actions must be a mapping")
    unknown = set(map(str, free)).difference(
        {"day_ahead", "battery", "compute", "fallback_confidence", "policy_confidence"}
    )
    if unknown:
        raise ActionSchemaError(f"unknown V7 free action(s): {sorted(unknown)}")
    if "day_ahead" in free:
        observed = np.asarray(_to_numpy(free["day_ahead"]), dtype=np.float64)
        if observed.ndim == 2:
            observed = observed[None]
        if observed.shape[-2:] == (SITES, HORIZON):
            observed = np.swapaxes(observed, -1, -2)
        if observed.shape != frozen_da.shape or not np.allclose(
            observed, frozen_da, atol=1.0e-6, rtol=0.0
        ):
            raise ActionSchemaError("day_ahead is immutable after the independent issue call")
    if "battery" not in free or "compute" not in free:
        raise ActionSchemaError("origin output requires current battery and compute")
    return {"battery": _to_numpy(free["battery"]), "compute": _to_numpy(free["compute"])}


class V7Runtime:
    """Execute actual days or frozen evaluation worlds with the V7 contract."""

    def __init__(
        self,
        limits: Mapping[str, Any] | ShieldLimits,
        config: ShieldConfig | None = None,
        *,
        policy_outputs_normalized: bool = False,
    ) -> None:
        self.limits = ShieldLimits.from_mapping(limits)
        self.config = config or ShieldConfig()
        self.policy_outputs_normalized = bool(policy_outputs_normalized)

    def _issue(
        self,
        policy: Any,
        issue_batch: Mapping[str, Any],
        *,
        batch: int,
    ) -> tuple[np.ndarray, Any]:
        reject_removed_actions(issue_batch)
        forbidden = {key for key in issue_batch if str(key).lower().startswith(("realized", "revealed"))}
        if forbidden:
            raise V7RuntimeError(f"issue batch contains operating-day outcomes: {sorted(forbidden)}")
        reference = _module_reference(policy)
        model_batch = _to_model(_broadcast_issue_batch(issue_batch, batch), reference)
        model_limits = (
            self.limits.torch(
                device=reference.device, dtype=reference.dtype, batch=batch
            )
            if reference is not None
            else self.limits.numpy(batch)
        )
        target = getattr(policy, "forward_issue", None) or getattr(policy, "issue_day_ahead", None)
        if target is None:
            if not callable(policy):
                raise V7RuntimeError("policy has no forward_issue/issue_day_ahead method")
            output = policy({"mode": "issue", "batch": model_batch, "limits": model_limits})
        else:
            with torch.inference_mode() if reference is not None else _nullcontext():
                output = _call_supported(target, batch=model_batch, issue_batch=model_batch, limits=model_limits)
        schedule = freeze_day_ahead_numpy(
            _to_numpy(_issue_schedule(output)),
            self.limits,
            normalized=self.policy_outputs_normalized,
            batch=batch,
        )
        issue_context = output.get("issue_context") if isinstance(output, Mapping) else None
        return schedule, issue_context

    def rollout_paths(
        self,
        paths: Any,
        policy: Any,
        issue_batch: Mapping[str, Any],
        *,
        keep_origin_records: bool = False,
    ) -> dict[str, Any]:
        """Replay a policy on ``N`` worlds, batching all worlds at each origin."""

        operating_value = getattr(paths, "operating_date", None)
        if operating_value is None and isinstance(paths, Mapping):
            operating_value = paths.get("operating_date")
        if operating_value is None:
            raise V7RuntimeError("evaluation paths are missing operating_date")
        operating_date = _date_text(operating_value)

        def field(name: str) -> np.ndarray:
            value = getattr(paths, name, None)
            if value is None and isinstance(paths, Mapping):
                value = paths.get(name)
            if value is None:
                raise V7RuntimeError(f"evaluation paths are missing {name}")
            result = np.asarray(value, dtype=np.float64)
            if not np.isfinite(result).all():
                raise V7RuntimeError(f"evaluation paths {name} is non-finite")
            return result

        rt_price = field("rt_price")
        if rt_price.ndim != 3 or rt_price.shape[1:] != (HORIZON, SITES):
            raise V7RuntimeError("rt_price must be [N,96,4]")
        batch = rt_price.shape[0]
        da_price = field("da_price")
        arrival = field("arrival")
        pv = field("pv_mw")
        multiplier = field("physical_multiplier")
        if da_price.shape == (HORIZON, SITES):
            da_price = np.broadcast_to(da_price, (batch, HORIZON, SITES))
        if arrival.shape != (batch, HORIZON) or pv.shape != (batch, HORIZON, SITES):
            raise V7RuntimeError("arrival/PV evaluation-path shapes are invalid")
        if multiplier.shape == (batch, SITES):
            multiplier = np.broadcast_to(multiplier[:, None, :], (batch, HORIZON, SITES))
        if multiplier.shape != (batch, HORIZON, SITES) or da_price.shape != rt_price.shape:
            raise V7RuntimeError("physical_multiplier/DA evaluation-path shapes are invalid")
        initial_soc = field("initial_soc_mwh")
        if initial_soc.shape == (SITES,):
            initial_soc = np.broadcast_to(initial_soc, (batch, SITES)).copy()
        if initial_soc.shape != (batch, SITES):
            raise V7RuntimeError("initial_soc_mwh must be [N,4]")

        frozen_da, issue_context = self._issue(policy, issue_batch, batch=batch)
        soc = initial_soc.copy()
        queue = np.zeros((batch, self.config.queue_size), dtype=np.float64)
        total_cost = np.zeros(batch, dtype=np.float64)
        total_sla = np.zeros(batch, dtype=np.float64)
        total_intervention = np.zeros(batch, dtype=np.float64)
        intervention_origins = np.zeros(batch, dtype=np.float64)
        all_feasible = np.ones(batch, dtype=bool)
        records: list[dict[str, Any]] = []
        reference = _module_reference(policy)
        model_limits = (
            self.limits.torch(device=reference.device, dtype=reference.dtype, batch=batch)
            if reference is not None
            else self.limits.numpy(batch)
        )
        policy_memory: Any = None
        previous_grid = np.zeros((batch, SITES), dtype=np.float64)
        previous_battery = np.zeros((batch, SITES), dtype=np.float64)
        previous_compute = np.zeros((batch, SITES), dtype=np.float64)
        revealed_rt = np.zeros_like(rt_price)
        revealed_arrival = np.zeros_like(arrival)
        revealed_pv = np.zeros_like(pv)
        for origin in range(HORIZON):
            revealed_rt[:, origin] = rt_price[:, origin]
            revealed_arrival[:, origin] = arrival[:, origin]
            revealed_pv[:, origin] = pv[:, origin]
            backlog_total = queue.sum(axis=1)
            backlog_by_site = np.repeat((backlog_total / SITES)[:, None], SITES, axis=1)
            pressure = np.divide(
                queue[:, 0],
                np.maximum(backlog_total, 1.0e-12),
                out=np.zeros_like(backlog_total),
                where=backlog_total > 0.0,
            )
            rolling_state = {
                "soc_mwh": soc,
                "current_soc": soc,
                "queue": queue,
                "backlog": backlog_by_site,
                "deadline_pressure": np.repeat(pressure[:, None], SITES, axis=1),
                "previous_grid": previous_grid,
                "previous_battery": previous_battery,
                "previous_compute": previous_compute,
                "revealed_rt": revealed_rt,
                "revealed_workload": revealed_arrival[..., None],
                "revealed_pv": revealed_pv,
                "revealed_mask": np.arange(HORIZON)[None, :] <= origin,
                "policy_memory": policy_memory,
            }
            target = getattr(policy, "forward_origin", None) or getattr(policy, "act", None)
            model_state = _to_model(rolling_state, reference)
            issued_for_model = _to_model(frozen_da, reference)
            if target is None:
                if not callable(policy):
                    raise V7RuntimeError("policy has no forward_origin/act method")
                output = policy(
                    {
                        "mode": "origin",
                        "origin": origin,
                        "issue_context": issue_context,
                        "rolling_state": model_state,
                        "limits": model_limits,
                        "issued_day_ahead": issued_for_model,
                    }
                )
            else:
                with torch.inference_mode() if reference is not None else _nullcontext():
                    output = _call_supported(
                        target,
                        issue_context=issue_context,
                        rolling_state=model_state,
                        state=model_state,
                        origin=origin,
                        limits=model_limits,
                        issued_day_ahead=issued_for_model,
                    )
            raw_action = _current_action(output, frozen_da)
            step = numpy_shield_step(
                raw_action,
                {"soc_mwh": soc, "queue": queue},
                {
                    "origin": origin,
                    "arrival": arrival[:, origin],
                    "pv_mw": pv[:, origin],
                    "physical_multiplier": multiplier[:, origin],
                },
                self.limits,
                self.config,
                action_is_normalized=self.policy_outputs_normalized,
            )
            da_now = frozen_da[:, origin]
            energy_cost = self.config.interval_hours * (
                (da_now * da_price[:, origin]).sum(axis=1)
                + ((step["grid_mw"] - da_now) * rt_price[:, origin]).sum(axis=1)
            )
            degradation = (
                self.config.interval_hours
                * self.config.battery_degradation_per_mwh
                * (step["charge"] + step["discharge"]).sum(axis=1)
            )
            sla_cost = self.config.sla_penalty_per_work * step["sla_violation"]
            total_cost += energy_cost + degradation + sla_cost
            total_sla += step["sla_violation"]
            total_intervention += step["shield_intervention"]
            intervention_origins += step["shield_intervention"] > self.config.tolerance
            all_feasible &= step["feasible"]
            soc = step["soc_next_mwh"]
            queue = step["queue_next"]
            previous_grid = step["grid_mw"]
            previous_battery = step["signed_battery"]
            previous_compute = step["compute"]
            policy_memory = output.get("policy_memory_next") if isinstance(output, Mapping) else None
            if policy_memory is None and isinstance(output, Mapping):
                policy_memory = _policy_section(output).get("policy_memory_next")
            if keep_origin_records:
                records.append(
                    {
                        "origin": origin,
                        "grid_mw": step["grid_mw"].copy(),
                        "pv_used_mw": step["pv_used_mw"].copy(),
                        "signed_battery_mw": step["signed_battery"].copy(),
                        "compute_work": step["compute"].copy(),
                        "soc_next_mwh": soc.copy(),
                        "sla_violation": step["sla_violation"].copy(),
                        "shield_intervention": step["shield_intervention"].copy(),
                        "feasible": step["feasible"].copy(),
                    }
                )
        terminal_reserve = self.config.terminal_soc_fraction * self.limits.battery_mwh[None, :]
        terminal_soc_gap = np.maximum(terminal_reserve - soc, 0.0).sum(axis=1)
        terminal_backlog = queue.sum(axis=1)
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "shield_algebra_version": SHIELD_ALGEBRA_VERSION,
            "operating_date": operating_date,
            "completed_origins": HORIZON,
            "batch_size": batch,
            "issued_day_ahead_mw": frozen_da,
            "day_ahead_issued_before_realized_prefix": True,
            "day_ahead_frozen": True,
            "current_actions_executed": HORIZON,
            "suffix_actions_executed": 0,
            "realized_cost": total_cost,
            "terminal_soc_gap": terminal_soc_gap,
            "terminal_backlog": terminal_backlog,
            "sla_violation": total_sla,
            "shield_intervention": total_intervention / HORIZON,
            "shield_intervention_rate": intervention_origins / HORIZON,
            "feasible": all_feasible & (terminal_soc_gap <= self.config.tolerance),
            "origin_records": tuple(records),
        }

    def run_day(
        self,
        model: Any,
        day_input: Any,
        evaluation_paths: Any | None = None,
        *,
        issue_batch: Mapping[str, Any] | None = None,
        keep_origin_records: bool = True,
    ) -> dict[str, Any]:
        """Run one realized day, issuing DA before reading any realized field."""

        if evaluation_paths is not None:
            if issue_batch is None:
                if not isinstance(day_input, Mapping):
                    raise V7RuntimeError(
                        "issue_batch is required when evaluation_paths are supplied"
                    )
                issue_batch = day_input
            return self.rollout_paths(
                evaluation_paths,
                model,
                issue_batch,
                keep_origin_records=keep_origin_records,
            )

        day = day_input
        policy = model
        operating_date = _date_text(getattr(day, "operating_date"))
        # Keep this construction above every access to a realized/revealed field.
        issue_batch = {
            "operating_date": operating_date,
            "issue_time_utc": getattr(day, "issue_time_utc"),
            "market": np.asarray(getattr(day, "market"))[None],
            "workload": np.asarray(getattr(day, "workload"))[None],
            "physical_static": np.asarray(getattr(day, "physical_static"))[None],
            "physical_dynamic": np.asarray(getattr(day, "physical_dynamic"))[None],
            "initial_soc_mwh": np.asarray(getattr(day, "initial_soc_mwh"))[None],
        }
        frozen_da, issue_context = self._issue(policy, issue_batch, batch=1)
        # Only now may the operating-day outcome be materialized.
        rt_price = np.asarray(getattr(day, "realized_rt_price"), dtype=np.float64).T[None]
        da_price = np.asarray(getattr(day, "realized_da_price"), dtype=np.float64).T[None]
        arrival = np.asarray(getattr(day, "realized_workload"), dtype=np.float64)[None]
        pv = np.asarray(getattr(day, "realized_pv"), dtype=np.float64).T[None]
        multiplier = np.asarray(getattr(day, "physical_multiplier"), dtype=np.float64)[None]

        class _IssuedPolicy:
            def __init__(self, outer: "V7Runtime", base: Any, schedule: np.ndarray, context: Any) -> None:
                self.outer = outer
                self.base = base
                self.schedule = schedule
                self.context = context

            def forward_issue(self, batch: Any, limits: Any = None) -> Mapping[str, Any]:
                return {
                    "issue_context": self.context,
                    "policy": {
                        "day_ahead": self.schedule,
                        "day_ahead_issue_mask": np.ones(self.schedule.shape[0], dtype=bool),
                    },
                }

            def forward_origin(self, **kwargs: Any) -> Any:
                target = getattr(self.base, "forward_origin", None) or getattr(self.base, "act", None)
                if target is None:
                    return self.base({"mode": "origin", **kwargs})
                return _call_supported(target, **kwargs)

        issued_policy = _IssuedPolicy(self, policy, frozen_da, issue_context)
        paths = {
            "operating_date": operating_date,
            "rt_price": rt_price,
            "da_price": da_price,
            "arrival": arrival,
            "pv_mw": pv,
            "physical_multiplier": multiplier,
            "initial_soc_mwh": issue_batch["initial_soc_mwh"],
        }
        return self.rollout_paths(
            paths,
            issued_policy,
            issue_batch,
            keep_origin_records=keep_origin_records,
        )


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


__all__ = [
    "ActionSchemaError",
    "HORIZON",
    "INTERVAL_HOURS",
    "Locked2025Error",
    "RUNTIME_SCHEMA_VERSION",
    "SHIELD_ALGEBRA_VERSION",
    "SITES",
    "ShieldConfig",
    "ShieldLimits",
    "V7Runtime",
    "V7RuntimeError",
    "freeze_day_ahead_numpy",
    "guard_unlocked_path",
    "numpy_shield_step",
    "reject_removed_actions",
    "scale_normalized_action_numpy",
    "scale_normalized_action_torch",
    "torch_shield_step",
]
