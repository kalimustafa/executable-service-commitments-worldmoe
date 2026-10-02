"""Decision-oriented evaluation for canonical NYISO DART forecasts.

The package is deliberately independent of the pipeline.  It exposes pure,
deterministic functions that can be connected to the optimization/replay
branch only after those interfaces are frozen.
"""

from .action_weighted_crps import (
    ACTION_CLASSES,
    NO_FEASIBLE_FLEXIBILITY,
    NO_FEASIBLE_OPPORTUNITY,
    NullableMetric,
    action_weighted_crps,
    economic_action_labels,
    economic_action_metrics,
    economic_action_probabilities,
    feasible_opportunity_ranking_loss,
    validate_gate_time_flexibility,
)
from .dart_metrics import (
    CANONICAL_DART_FIELD,
    INVERSE_SPREAD_FIELD,
    assert_canonical_dart,
    assert_no_ambiguous_spread,
    crps_from_quantiles,
    derive_dart_fields,
    ensemble_crps,
    fit_tail_thresholds,
    interval_calibration,
    joint_path_diagnostics,
    pinball_loss,
    tail_weighted_crps,
)
from .dart_value_select import (
    ContextualSelectorStatus,
    FeasibilityCriteria,
    SelectionResult,
    contextual_selector_status,
    evaluate_feasibility,
    select_dart_value,
)
from .profit_evaluator import (
    OperationalValueResult,
    decision_value_summary,
    incremental_operational_value,
    prescriptive_regret,
    profit_capture_ratio,
    realized_operational_value,
)

__all__ = [
    "ACTION_CLASSES",
    "CANONICAL_DART_FIELD",
    "ContextualSelectorStatus",
    "FeasibilityCriteria",
    "INVERSE_SPREAD_FIELD",
    "NO_FEASIBLE_FLEXIBILITY",
    "NO_FEASIBLE_OPPORTUNITY",
    "NullableMetric",
    "OperationalValueResult",
    "SelectionResult",
    "action_weighted_crps",
    "assert_canonical_dart",
    "assert_no_ambiguous_spread",
    "contextual_selector_status",
    "crps_from_quantiles",
    "decision_value_summary",
    "derive_dart_fields",
    "economic_action_labels",
    "economic_action_metrics",
    "economic_action_probabilities",
    "ensemble_crps",
    "evaluate_feasibility",
    "feasible_opportunity_ranking_loss",
    "fit_tail_thresholds",
    "incremental_operational_value",
    "interval_calibration",
    "joint_path_diagnostics",
    "pinball_loss",
    "prescriptive_regret",
    "profit_capture_ratio",
    "realized_operational_value",
    "select_dart_value",
    "tail_weighted_crps",
    "validate_gate_time_flexibility",
]
