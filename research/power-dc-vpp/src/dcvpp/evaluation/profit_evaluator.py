"""Counterfactual realized/replayed operational-value accounting.

The conventional PCR name is retained for comparability, but every value in
this module is counterfactual operational value (or equivalent cost reduction),
never a claim of realized market revenue or commercial profit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .action_weighted_crps import NullableMetric, OK


PCR_DENOMINATOR_NONPOSITIVE = "PCR_DENOMINATOR_NONPOSITIVE_OR_ZERO"


@dataclass(frozen=True)
class OperationalValueResult:
    """Frozen-decision cost components and their value-equivalent."""

    realized_operational_value: float
    total_operating_cost: float
    task_value: float
    cost_components: Mapping[str, float]
    counterfactual: bool = True
    evidence_role: str = "counterfactual_replayed"
    value_semantics: str = (
        "counterfactual_net_operational_value_not_commercial_profit"
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "realized_operational_value": self.realized_operational_value,
            "total_operating_cost": self.total_operating_cost,
            "task_value": self.task_value,
            "cost_components": dict(self.cost_components),
            "counterfactual": self.counterfactual,
            "evidence_role": self.evidence_role,
            "value_semantics": self.value_semantics,
        }


def _float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _finite_sum(value: Any, *, name: str) -> float:
    array = _float_array(value, name=name)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return float(np.sum(array))


def realized_operational_value(
    da_lbmp: Any,
    rt_lbmp: Any,
    q_da: Any,
    q_grid: Any,
    *,
    interval_hours: Any = 1.0,
    task_value: float = 0.0,
    battery_degradation_cost: Any = 0.0,
    migration_cost: Any = 0.0,
    sla_penalty: Any = 0.0,
    transaction_cost: Any = 0.0,
    other_cost: Any = 0.0,
) -> OperationalValueResult:
    """Replay frozen quantities against realized DA and RT market records.

    ``q_da`` and ``q_grid`` are power quantities.  ``interval_hours`` performs
    the explicit MWh conversion and may be scalar or broadcastable by interval.
    Negative prices are preserved.
    """

    da, rt, day_ahead_quantity, grid_quantity, hours = np.broadcast_arrays(
        _float_array(da_lbmp, name="da_lbmp"),
        _float_array(rt_lbmp, name="rt_lbmp"),
        _float_array(q_da, name="q_da"),
        _float_array(q_grid, name="q_grid"),
        _float_array(interval_hours, name="interval_hours"),
    )
    for name, array in (
        ("da_lbmp", da),
        ("rt_lbmp", rt),
        ("q_da", day_ahead_quantity),
        ("q_grid", grid_quantity),
        ("interval_hours", hours),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite for realized replay")
    if np.any(hours <= 0.0):
        raise ValueError("interval_hours must be positive")
    if not np.isfinite(task_value):
        raise ValueError("task_value must be finite")

    day_ahead_cost = float(np.sum(da * day_ahead_quantity * hours))
    rt_deviation_cost = float(
        np.sum(rt * (grid_quantity - day_ahead_quantity) * hours)
    )
    components = {
        "day_ahead_settlement_cost": day_ahead_cost,
        "real_time_deviation_settlement_cost": rt_deviation_cost,
        "battery_degradation_cost": _finite_sum(
            battery_degradation_cost, name="battery_degradation_cost"
        ),
        "migration_cost": _finite_sum(migration_cost, name="migration_cost"),
        "sla_penalty": _finite_sum(sla_penalty, name="sla_penalty"),
        "transaction_cost": _finite_sum(
            transaction_cost, name="transaction_cost"
        ),
        "other_cost": _finite_sum(other_cost, name="other_cost"),
    }
    total_cost = float(sum(components.values()))
    return OperationalValueResult(
        realized_operational_value=float(task_value - total_cost),
        total_operating_cost=total_cost,
        task_value=float(task_value),
        cost_components=components,
    )


def incremental_operational_value(
    model_values: Any,
    baseline_values: Any,
) -> np.ndarray:
    """Return paired ``model - baseline`` counterfactual operational value."""

    model, baseline = np.broadcast_arrays(
        _float_array(model_values, name="model_values"),
        _float_array(baseline_values, name="baseline_values"),
    )
    return model - baseline


def prescriptive_regret(
    perfect_information_values: Any,
    model_values: Any,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Return ``PI - model`` with PI treated only as an unattainable bound."""

    perfect, model = np.broadcast_arrays(
        _float_array(perfect_information_values, name="perfect_information_values"),
        _float_array(model_values, name="model_values"),
    )
    regret = perfect - model
    if np.any(regret < -abs(float(tolerance))):
        raise ValueError(
            "perfect-information value must be an upper bound on model value"
        )
    return np.maximum(regret, 0.0)


def profit_capture_ratio(
    model_values: Any,
    baseline_values: Any,
    perfect_information_values: Any,
    *,
    zero_tolerance: float = 1e-12,
) -> NullableMetric:
    """Compute required PCR orientation ``(model-B)/(PI-B)``.

    A nonpositive or numerically zero perfect-information opportunity emits an
    explicit null rather than a misleading infinity or sign reversal.
    """

    model, baseline, perfect = np.broadcast_arrays(
        _float_array(model_values, name="model_values"),
        _float_array(baseline_values, name="baseline_values"),
        _float_array(perfect_information_values, name="perfect_information_values"),
    )
    valid = np.isfinite(model) & np.isfinite(baseline) & np.isfinite(perfect)
    if not np.any(valid):
        return NullableMetric(
            value=None,
            status=PCR_DENOMINATOR_NONPOSITIVE,
            numerator=0.0,
            denominator=0.0,
            count=0,
        )
    numerator = float(np.sum(model[valid] - baseline[valid]))
    denominator = float(np.sum(perfect[valid] - baseline[valid]))
    tolerance = max(abs(float(zero_tolerance)), np.finfo(float).eps)
    if denominator <= tolerance:
        return NullableMetric(
            value=None,
            status=PCR_DENOMINATOR_NONPOSITIVE,
            numerator=numerator,
            denominator=denominator,
            count=int(np.count_nonzero(valid)),
        )
    return NullableMetric(
        value=numerator / denominator,
        status=OK,
        numerator=numerator,
        denominator=denominator,
        count=int(np.count_nonzero(valid)),
    )


def profit_capture_ratio_value(*args: Any, **kwargs: Any) -> float | None:
    """Convenience scalar/null view of :func:`profit_capture_ratio`."""

    return profit_capture_ratio(*args, **kwargs).value


def empirical_cvar(losses: Any, *, alpha: float = 0.95) -> float | None:
    """Empirical upper-tail CVaR with fractional boundary mass."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    values = _float_array(losses, name="losses").ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    descending = np.sort(values)[::-1]
    tail_mass = (1.0 - alpha) * descending.size
    full_count = int(np.floor(tail_mass))
    fractional = tail_mass - full_count
    total = float(np.sum(descending[:full_count])) if full_count else 0.0
    if fractional > 0.0:
        total += fractional * float(descending[full_count])
    elif full_count == 0:
        total = float(descending[0])
        tail_mass = 1.0
    return total / tail_mass


def maximum_drawdown(incremental_values: Any) -> float | None:
    """Maximum peak-to-trough drop of cumulative incremental value."""

    values = _float_array(incremental_values, name="incremental_values").ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    running_peak = np.maximum.accumulate(cumulative)
    return float(np.max(running_peak - cumulative))


def longest_loss_run(incremental_values: Any) -> int:
    values = _float_array(incremental_values, name="incremental_values").ravel()
    longest = 0
    current = 0
    for value in values:
        if np.isfinite(value) and value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def decision_value_summary(
    model_values: Any,
    baseline_values: Any,
    perfect_information_values: Any,
    *,
    risk_alpha: float = 0.95,
    feasible_flexible_energy_mwh: Any | None = None,
) -> dict[str, Any]:
    """Summarize paired value, regret, PCR, downside risk, and drawdown."""

    model, baseline, perfect = np.broadcast_arrays(
        _float_array(model_values, name="model_values"),
        _float_array(baseline_values, name="baseline_values"),
        _float_array(perfect_information_values, name="perfect_information_values"),
    )
    valid = np.isfinite(model) & np.isfinite(baseline) & np.isfinite(perfect)
    delta = model[valid] - baseline[valid]
    regrets = prescriptive_regret(perfect[valid], model[valid])
    losses = -delta
    pcr = profit_capture_ratio(model[valid], baseline[valid], perfect[valid])
    if delta.size == 0:
        return {
            "status": "NO_VALID_PAIRED_DAYS",
            "paired_day_count": 0,
            "profit_capture_ratio": pcr.as_dict(),
            "counterfactual": True,
        }
    worst_tail = empirical_cvar(losses, alpha=risk_alpha)
    feasible_total: float | None = None
    value_per_mwh: float | None = None
    if feasible_flexible_energy_mwh is not None:
        feasible = np.broadcast_to(
            _float_array(
                feasible_flexible_energy_mwh,
                name="feasible_flexible_energy_mwh",
            ),
            model.shape,
        )[valid]
        if np.any(~np.isfinite(feasible)) or np.any(feasible < 0.0):
            raise ValueError("feasible flexible energy must be finite and nonnegative")
        feasible_total = float(np.sum(feasible))
        if feasible_total > 0.0:
            value_per_mwh = float(np.sum(delta) / feasible_total)
    return {
        "status": "OK",
        "paired_day_count": int(delta.size),
        "cumulative_incremental_operational_value": float(np.sum(delta)),
        "mean_daily_incremental_operational_value": float(np.mean(delta)),
        "mean_prescriptive_regret": float(np.mean(regrets)),
        "cumulative_prescriptive_regret": float(np.sum(regrets)),
        "profit_capture_ratio": pcr.as_dict(),
        "probability_below_baseline": float(np.mean(delta < 0.0)),
        "cvar_loss": worst_tail,
        "worst_tail_mean_loss": worst_tail,
        "maximum_drawdown": maximum_drawdown(delta),
        "maximum_daily_loss": float(max(0.0, np.max(losses))),
        "longest_loss_run_days": longest_loss_run(delta),
        "positive_day_share": float(np.mean(delta > 0.0)),
        "feasible_flexible_energy_mwh": feasible_total,
        "value_per_mwh_feasible_flexibility": value_per_mwh,
        "risk_alpha": risk_alpha,
        "counterfactual": True,
        "value_semantics": (
            "counterfactual_incremental_operational_value_not_realized_revenue"
        ),
        "perfect_information_semantics": "UNATTAINABLE_UPPER_BOUND_ONLY",
    }


compute_realized_operational_value = realized_operational_value
compute_profit_capture_ratio = profit_capture_ratio


__all__ = [
    "OperationalValueResult",
    "PCR_DENOMINATOR_NONPOSITIVE",
    "compute_profit_capture_ratio",
    "compute_realized_operational_value",
    "decision_value_summary",
    "empirical_cvar",
    "incremental_operational_value",
    "longest_loss_run",
    "maximum_drawdown",
    "prescriptive_regret",
    "profit_capture_ratio",
    "profit_capture_ratio_value",
    "realized_operational_value",
]
