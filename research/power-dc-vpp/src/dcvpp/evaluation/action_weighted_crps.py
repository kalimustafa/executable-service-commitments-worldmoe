"""Gate-time actionability metrics for canonical DA-minus-RT DART."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


ACTION_CLASSES = (-1, 0, 1)
"""Ordered classes: decrease DA, no action, increase DA procurement."""

NO_FEASIBLE_FLEXIBILITY = "NO_FEASIBLE_FLEXIBILITY"
NO_FEASIBLE_OPPORTUNITY = "NO_FEASIBLE_OPPORTUNITY"
OK = "OK"

_FORBIDDEN_GATE_FIELDS = {
    "future_price",
    "future_realized_price",
    "realized_future_price",
    "future_workload",
    "future_realized_workload",
    "realized_future_workload",
    "future_pv",
    "future_realized_pv",
    "realized_future_pv",
    "locked_test_outcome",
    "locked_test_value",
    "test_decision_value",
}


@dataclass(frozen=True)
class NullableMetric:
    """A scalar metric whose undefined state is explicit and serializable."""

    value: float | None
    status: str
    numerator: float
    denominator: float
    count: int

    @property
    def is_null(self) -> bool:
        return self.value is None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "value": self.value,
            "status": self.status,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "count": self.count,
        }

    def __getitem__(self, key: str) -> float | int | str | None:
        return self.as_dict()[key]


def _float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _normalize_field_name(field: str) -> str:
    return str(field).strip().lower().replace("-", "_").replace(" ", "_")


def _field_is_future_leakage(field: str) -> bool:
    normalized = _normalize_field_name(field)
    if normalized in _FORBIDDEN_GATE_FIELDS:
        return True
    if "locked_test" in normalized or "test_outcome" in normalized:
        return True
    sensitive = any(token in normalized for token in ("price", "workload", "pv"))
    future_or_unavailable = any(
        token in normalized
        for token in ("future_realized", "realized_future", "future_outcome")
    ) or normalized.startswith("future_")
    return sensitive and future_or_unavailable


def validate_gate_time_flexibility(
    feasible_flexible_energy: Any,
    *,
    source_fields: Iterable[str] = (),
    available_at_gate: Any | None = None,
) -> np.ndarray:
    """Validate that ``F[d,z,t]`` is nonnegative and gate-time available.

    The numeric values cannot prove causal availability by themselves, so the
    interface also accepts the exact source-field lineage and/or a boolean
    availability mask.  Any explicitly future realized price, workload, PV,
    locked-test outcome, or unavailable row is a hard failure.
    """

    weights = _float_array(
        feasible_flexible_energy, name="feasible_flexible_energy"
    )
    if not np.all(np.isfinite(weights)):
        raise ValueError("feasible flexible energy must be finite")
    if np.any(weights < 0.0):
        raise ValueError("feasible flexible energy must be nonnegative")
    forbidden = sorted(
        str(field) for field in source_fields if _field_is_future_leakage(str(field))
    )
    if forbidden:
        raise ValueError(
            "gate-time feasible flexibility uses forbidden future/test fields: "
            + ", ".join(forbidden)
        )
    if available_at_gate is not None:
        availability = np.asarray(available_at_gate, dtype=bool)
        try:
            availability = np.broadcast_to(availability, weights.shape)
        except ValueError as error:
            raise ValueError(
                "available_at_gate must broadcast to feasible flexibility"
            ) from error
        if not np.all(availability):
            raise ValueError(
                "feasible flexibility contains information unavailable at gate time"
            )
    return weights


def validate_normalized_feasibility_weight(
    feasibility_weight: Any,
    *,
    source_fields: Iterable[str] = (),
    available_at_gate: Any | None = None,
) -> np.ndarray:
    """Validate the candidate-independent normalized weight ``F in [0, 1]``."""

    weights = validate_gate_time_flexibility(
        feasibility_weight,
        source_fields=source_fields,
        available_at_gate=available_at_gate,
    )
    if np.any(weights > 1.0):
        raise ValueError("normalized feasibility weight F must lie in [0, 1]")
    return weights


def action_weighted_crps(
    crps_values: Any,
    feasible_flexible_energy: Any,
    *,
    source_fields: Iterable[str] = (),
    available_at_gate: Any | None = None,
) -> NullableMetric:
    """Compute AWCRPS and emit an explicit null when ``sum(F) == 0``."""

    scores, weights = np.broadcast_arrays(
        _float_array(crps_values, name="crps_values"),
        validate_gate_time_flexibility(
            feasible_flexible_energy,
            source_fields=source_fields,
            available_at_gate=available_at_gate,
        ),
    )
    valid = np.isfinite(scores)
    if not np.all(valid | (weights == 0.0)):
        raise ValueError("CRPS must be finite wherever feasible flexibility is positive")
    effective_weights = np.where(valid, weights, 0.0)
    denominator = float(np.sum(effective_weights))
    numerator = float(np.sum(np.where(valid, scores, 0.0) * effective_weights))
    count = int(np.count_nonzero(effective_weights > 0.0))
    if denominator == 0.0:
        return NullableMetric(
            value=None,
            status=NO_FEASIBLE_FLEXIBILITY,
            numerator=0.0,
            denominator=0.0,
            count=0,
        )
    return NullableMetric(
        value=numerator / denominator,
        status=OK,
        numerator=numerator,
        denominator=denominator,
        count=count,
    )


def action_weighted_crps_value(*args: Any, **kwargs: Any) -> float | None:
    """Convenience scalar/null view of :func:`action_weighted_crps`."""

    return action_weighted_crps(*args, **kwargs).value


def feasibility_conditioned_threshold_weighted_crps(
    observations: Any,
    ensemble: Any,
    feasibility_weight: Any,
    *,
    lower_threshold: Any,
    upper_threshold: Any,
    scenario_axis: int = -1,
    source_fields: Iterable[str] = (),
    available_at_gate: Any | None = None,
) -> NullableMetric:
    """Compute the proper tail score and then weight it by frozen flexibility.

    The threshold score is proper because its threshold functions are frozen
    from pre-origin training data.  Multiplication and aggregation use only a
    candidate-independent, gate-time-available ``F``.  Consequently this
    metric audits tail credibility where the VPP can act without turning the
    final selector into an arbitrary weighted composite score.
    """

    from .dart_metrics import threshold_weighted_crps

    per_event = threshold_weighted_crps(
        observations,
        ensemble,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        region="two_sided",
        scenario_axis=scenario_axis,
        reduction="none",
    )
    weights = validate_normalized_feasibility_weight(
        feasibility_weight,
        source_fields=source_fields,
        available_at_gate=available_at_gate,
    )
    return action_weighted_crps(
        per_event,
        weights,
        source_fields=source_fields,
        available_at_gate=available_at_gate,
    )


def _validate_no_action_band(
    lower_no_action_bound: Any,
    upper_no_action_bound: Any,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.broadcast_to(
        _float_array(lower_no_action_bound, name="lower_no_action_bound"), shape
    )
    upper = np.broadcast_to(
        _float_array(upper_no_action_bound, name="upper_no_action_bound"), shape
    )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("no-action bounds must be finite")
    if np.any(lower > 0.0) or np.any(upper < 0.0) or np.any(lower > upper):
        raise ValueError(
            "no-action band must satisfy lower <= 0 <= upper at every row"
        )
    return lower, upper


def economic_action_labels(
    dart_da_minus_rt: Any,
    lower_no_action_bound: Any,
    upper_no_action_bound: Any,
) -> np.ndarray:
    """Map canonical DART to ``{-1, 0, +1}`` demand-side actions.

    Positive DART (DA more expensive) maps to ``-1``: decrease DA
    procurement.  Negative DART maps to ``+1``: increase DA procurement.
    Threshold equality remains in the no-action band.
    """

    dart = _float_array(dart_da_minus_rt, name="dart_da_minus_rt")
    lower, upper = _validate_no_action_band(
        lower_no_action_bound, upper_no_action_bound, dart.shape
    )
    actions = np.zeros(dart.shape, dtype=np.int8)
    actions[dart > upper] = -1
    actions[dart < lower] = 1
    actions[~np.isfinite(dart)] = 0
    return actions


def economic_net_gain_signal(
    dart_da_minus_rt: Any,
    lower_no_action_bound: Any,
    upper_no_action_bound: Any,
) -> np.ndarray:
    """Return DART net of the frozen, time-varying no-action band.

    Positive values support decreasing DA procurement; negative values support
    increasing it.  Values inside the band are exactly zero.  The band is
    required to include transaction, degradation, migration/bandwidth, minimum
    dispatch, settlement asymmetry, and engineered SLA-risk costs upstream.
    """

    dart = _float_array(dart_da_minus_rt, name="dart_da_minus_rt")
    lower, upper = _validate_no_action_band(
        lower_no_action_bound, upper_no_action_bound, dart.shape
    )
    signal = np.zeros(dart.shape, dtype=float)
    above = dart > upper
    below = dart < lower
    signal[above] = dart[above] - upper[above]
    signal[below] = dart[below] - lower[below]
    signal[~np.isfinite(dart)] = np.nan
    return signal


def economic_action_probabilities(
    dart_scenarios: Any,
    lower_no_action_bound: Any,
    upper_no_action_bound: Any,
    *,
    scenario_axis: int = -1,
    scenario_probabilities: Any | None = None,
) -> np.ndarray:
    """Return probabilities for classes ordered as ``(-1, 0, +1)``."""

    scenarios = np.moveaxis(
        _float_array(dart_scenarios, name="dart_scenarios"), scenario_axis, -1
    )
    if scenarios.shape[-1] == 0:
        raise ValueError("at least one DART scenario is required")
    lower, upper = _validate_no_action_band(
        lower_no_action_bound, upper_no_action_bound, scenarios.shape[:-1]
    )
    if scenario_probabilities is None:
        probabilities = np.full(
            scenarios.shape[-1], 1.0 / scenarios.shape[-1], dtype=float
        )
    else:
        probabilities = _float_array(
            scenario_probabilities, name="scenario_probabilities"
        )
        if probabilities.ndim != 1 or probabilities.size != scenarios.shape[-1]:
            raise ValueError(
                "scenario_probabilities must be one-dimensional and match scenarios"
            )
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("scenario probabilities must be finite and nonnegative")
        total = float(np.sum(probabilities))
        if not np.isclose(total, 1.0):
            raise ValueError("scenario probabilities must sum to one")
        probabilities = probabilities / total
    finite = np.isfinite(scenarios)
    finite_mass = np.sum(finite * probabilities, axis=-1)
    if np.any(finite_mass == 0.0):
        raise ValueError("every forecast row must contain a finite scenario")
    decrease_da = scenarios > upper[..., None]
    increase_da = scenarios < lower[..., None]
    no_action = finite & ~(decrease_da | increase_da)
    output = np.stack(
        [
            np.sum(decrease_da * probabilities, axis=-1),
            np.sum(no_action * probabilities, axis=-1),
            np.sum(increase_da * probabilities, axis=-1),
        ],
        axis=-1,
    )
    output = output / finite_mass[..., None]
    if not np.allclose(np.sum(output, axis=-1), 1.0):
        raise AssertionError("economic action probabilities do not sum to one")
    return output


def economic_action_brier(
    action_probabilities: Any,
    realized_actions: Any,
    *,
    sample_weight: Any | None = None,
) -> float | None:
    """Compute the multiclass Brier score with optional F weighting."""

    probabilities = _float_array(
        action_probabilities, name="action_probabilities"
    )
    if probabilities.shape[-1] != len(ACTION_CLASSES):
        raise ValueError("the last probability axis must contain (-1, 0, +1)")
    actions = np.broadcast_to(
        np.asarray(realized_actions, dtype=int), probabilities.shape[:-1]
    )
    if np.any(~np.isin(actions, ACTION_CLASSES)):
        raise ValueError("realized actions must be in {-1, 0, +1}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("action probabilities must be finite and nonnegative")
    if not np.allclose(np.sum(probabilities, axis=-1), 1.0):
        raise ValueError("action probabilities must sum to one")
    weights = (
        np.ones(actions.shape, dtype=float)
        if sample_weight is None
        else validate_gate_time_flexibility(sample_weight)
    )
    weights = np.broadcast_to(weights, actions.shape)
    denominator = float(np.sum(weights))
    if denominator == 0.0:
        return None
    class_array = np.asarray(ACTION_CLASSES, dtype=int)
    one_hot = actions[..., None] == class_array
    row_score = np.sum((probabilities - one_hot.astype(float)) ** 2, axis=-1)
    return float(np.sum(weights * row_score) / denominator)


def _weighted_classification(
    realized: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, dict[str, float | None]], float | None, float | None, float | None]:
    per_class: dict[str, dict[str, float | None]] = {}
    support_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    for action in ACTION_CLASSES:
        true_class = realized == action
        predicted_class = predicted == action
        true_positive = float(np.sum(weights[true_class & predicted_class]))
        false_positive = float(np.sum(weights[~true_class & predicted_class]))
        false_negative = float(np.sum(weights[true_class & ~predicted_class]))
        support = float(np.sum(weights[true_class]))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0.0
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0.0
            else None
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall > 0.0
            else None
        )
        per_class[str(action)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_flexible_mwh": support,
        }
        if support > 0.0:
            support_total += support
            precision_total += support * (precision if precision is not None else 0.0)
            recall_total += support * (recall if recall is not None else 0.0)
            f1_total += support * (f1 if f1 is not None else 0.0)
    if support_total == 0.0:
        return per_class, None, None, None
    return (
        per_class,
        precision_total / support_total,
        recall_total / support_total,
        f1_total / support_total,
    )


def economic_action_metrics(
    action_probabilities: Any,
    realized_actions: Any,
    feasible_flexible_energy: Any,
    *,
    opportunity_magnitude: Any | None = None,
    source_fields: Iterable[str] = (),
    available_at_gate: Any | None = None,
) -> dict[str, Any]:
    """Brier, F-weighted P/R/F1, wrong-action, and missed-opportunity loss."""

    probabilities = _float_array(
        action_probabilities, name="action_probabilities"
    )
    if probabilities.shape[-1] != 3:
        raise ValueError("action_probabilities must have three final-axis classes")
    realized = np.broadcast_to(
        np.asarray(realized_actions, dtype=int), probabilities.shape[:-1]
    )
    if np.any(~np.isin(realized, ACTION_CLASSES)):
        raise ValueError("realized actions must be in {-1, 0, +1}")
    weights = np.broadcast_to(
        validate_gate_time_flexibility(
            feasible_flexible_energy,
            source_fields=source_fields,
            available_at_gate=available_at_gate,
        ),
        realized.shape,
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("action probabilities must be finite and nonnegative")
    if not np.allclose(np.sum(probabilities, axis=-1), 1.0):
        raise ValueError("action probabilities must sum to one")
    predicted = np.asarray(ACTION_CLASSES, dtype=int)[np.argmax(probabilities, axis=-1)]
    brier = economic_action_brier(
        probabilities, realized, sample_weight=weights
    )
    per_class, precision, recall, f1 = _weighted_classification(
        realized, predicted, weights
    )
    magnitude = (
        np.ones(realized.shape, dtype=float)
        if opportunity_magnitude is None
        else np.broadcast_to(
            np.abs(_float_array(opportunity_magnitude, name="opportunity_magnitude")),
            realized.shape,
        )
    )
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("opportunity magnitude must be finite")
    value_weights = weights * magnitude
    loss_denominator = float(np.sum(value_weights))
    wrong_mask = (predicted != realized) & (predicted != 0)
    missed_mask = (realized != 0) & (predicted == 0)
    predicted_action_denominator = float(np.sum(weights[predicted != 0]))
    realized_opportunity_denominator = float(np.sum(weights[realized != 0]))
    wrong_action_weight = float(np.sum(weights[wrong_mask]))
    missed_opportunity_weight = float(np.sum(weights[missed_mask]))
    wrong_total = float(np.sum(value_weights[wrong_mask]))
    missed_total = float(np.sum(value_weights[missed_mask]))
    status = OK if float(np.sum(weights)) > 0.0 else NO_FEASIBLE_FLEXIBILITY
    return {
        "status": status,
        "class_order": list(ACTION_CLASSES),
        "economic_action_brier": brier,
        "flexibility_weighted_precision": precision,
        "flexibility_weighted_recall": recall,
        "flexibility_weighted_f1": f1,
        "per_class": per_class,
        "wrong_action_loss": (
            wrong_total / loss_denominator if loss_denominator > 0.0 else None
        ),
        "wrong_action_loss_total": wrong_total,
        "wrong_action_rate": (
            wrong_action_weight / predicted_action_denominator
            if predicted_action_denominator > 0.0
            else None
        ),
        "wrong_action_rate_numerator": wrong_action_weight,
        "wrong_action_rate_denominator": predicted_action_denominator,
        "missed_opportunity_loss": (
            missed_total / loss_denominator if loss_denominator > 0.0 else None
        ),
        "missed_opportunity_loss_total": missed_total,
        "missed_opportunity_rate": (
            missed_opportunity_weight / realized_opportunity_denominator
            if realized_opportunity_denominator > 0.0
            else None
        ),
        "missed_opportunity_rate_numerator": missed_opportunity_weight,
        "missed_opportunity_rate_denominator": realized_opportunity_denominator,
        "opportunity_weight_denominator": loss_denominator,
        "predicted_actions": predicted,
    }


def feasible_opportunity_ranking_loss(
    feasible_transferable_energy: Any,
    realized_net_gain: Any,
    predicted_net_gain: Any,
) -> NullableMetric:
    """Compute the exact preregistered Feasible Opportunity Ranking Loss."""

    energy, realized, predicted = np.broadcast_arrays(
        validate_gate_time_flexibility(feasible_transferable_energy),
        _float_array(realized_net_gain, name="realized_net_gain"),
        _float_array(predicted_net_gain, name="predicted_net_gain"),
    )
    valid = np.isfinite(realized) & np.isfinite(predicted)
    if not np.all(valid | (energy == 0.0)):
        raise ValueError("gains must be finite for every feasible opportunity")
    opportunity_weight = np.where(valid, energy * np.abs(realized), 0.0)
    denominator = float(np.sum(opportunity_weight))
    wrong_order = valid & ((predicted * realized) < 0.0)
    numerator = float(np.sum(opportunity_weight[wrong_order]))
    count = int(np.count_nonzero(opportunity_weight > 0.0))
    if denominator == 0.0:
        return NullableMetric(
            value=None,
            status=NO_FEASIBLE_OPPORTUNITY,
            numerator=0.0,
            denominator=0.0,
            count=0,
        )
    return NullableMetric(
        value=numerator / denominator,
        status=OK,
        numerator=numerator,
        denominator=denominator,
        count=count,
    )


def feasible_opportunity_ranking_loss_value(
    *args: Any, **kwargs: Any
) -> float | None:
    return feasible_opportunity_ranking_loss(*args, **kwargs).value


compute_action_weighted_crps = action_weighted_crps
compute_forl = feasible_opportunity_ranking_loss


__all__ = [
    "ACTION_CLASSES",
    "NO_FEASIBLE_FLEXIBILITY",
    "NO_FEASIBLE_OPPORTUNITY",
    "NullableMetric",
    "action_weighted_crps",
    "action_weighted_crps_value",
    "compute_action_weighted_crps",
    "compute_forl",
    "economic_action_brier",
    "economic_action_labels",
    "economic_action_metrics",
    "economic_net_gain_signal",
    "economic_action_probabilities",
    "feasibility_conditioned_threshold_weighted_crps",
    "feasible_opportunity_ranking_loss",
    "feasible_opportunity_ranking_loss_value",
    "validate_gate_time_flexibility",
    "validate_normalized_feasibility_weight",
]
