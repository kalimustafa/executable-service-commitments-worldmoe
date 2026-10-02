"""Frozen DART-ValueSelect feasibility, MCS, LCB, and tie protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dcvpp.core import sha256_json
from dcvpp.statistics.block_bootstrap import (
    DEFAULT_BLOCK_LENGTHS,
    PairedBootstrapResult,
    paired_moving_block_bootstrap,
)
from dcvpp.statistics.model_confidence_set import MCSResult, model_confidence_set

from .profit_evaluator import empirical_cvar


SKIPPED_OPTIONAL_EXTENSION = "SKIPPED_OPTIONAL_EXTENSION"
A1_NONINFERIORITY_FRACTION = 0.005
ECONOMIC_WINNER_TOLERANCE_FRACTION = 0.005
MAXIMUM_ACCEPTED_MIP_GAP = 0.05


@dataclass(frozen=True)
class FeasibilityCriteria:
    minimum_sla_satisfaction: float = 0.99
    cvar95_loss_budget_fraction_of_baseline_cost: float = 0.05
    maximum_probability_below_baseline: float = 0.50
    market_gate_budget_seconds: float = 3600.0
    require_zero_infeasible_days: bool = True
    require_zero_invalid_days: bool = True
    maximum_solver_relative_mip_gap: float = MAXIMUM_ACCEPTED_MIP_GAP
    require_zero_solver_gap_over_cap_days: bool = True
    require_zero_solver_missing_gap_days: bool = True
    require_zero_solver_no_incumbent_days: bool = True
    solver_timeout_day_field: str = "solver_timeout_days"
    solver_nonzero_gap_day_field: str = "solver_nonzero_gap_days"
    solver_gap_over_cap_day_field: str = "solver_gap_over_cap_days"
    solver_missing_gap_day_field: str = "solver_missing_gap_days"
    solver_no_incumbent_day_field: str = "solver_no_incumbent_days"
    allow_legacy_solver_field_aliases: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_sla_satisfaction <= 1.0:
            raise ValueError("minimum_sla_satisfaction must be in [0, 1]")
        if self.cvar95_loss_budget_fraction_of_baseline_cost < 0.0:
            raise ValueError("CVaR budget fraction must be nonnegative")
        if not 0.0 <= self.maximum_probability_below_baseline <= 1.0:
            raise ValueError("maximum probability below baseline must be in [0, 1]")
        if self.market_gate_budget_seconds <= 0.0:
            raise ValueError("market gate budget must be positive")
        if not 0.0 <= self.maximum_solver_relative_mip_gap <= 1.0:
            raise ValueError("maximum solver relative MIP gap must be in [0, 1]")
        if self.solver_timeout_day_field != "solver_timeout_days":
            raise ValueError("paper-mode timeout field must be solver_timeout_days")
        if self.solver_nonzero_gap_day_field != "solver_nonzero_gap_days":
            raise ValueError(
                "paper-mode nonzero-gap field must be solver_nonzero_gap_days"
            )
        if self.solver_gap_over_cap_day_field != "solver_gap_over_cap_days":
            raise ValueError("paper-mode over-cap field must be solver_gap_over_cap_days")
        if self.solver_missing_gap_day_field != "solver_missing_gap_days":
            raise ValueError("paper-mode missing-gap field must be solver_missing_gap_days")
        if self.solver_no_incumbent_day_field != "solver_no_incumbent_days":
            raise ValueError(
                "paper-mode no-incumbent field must be solver_no_incumbent_days"
            )


@dataclass(frozen=True)
class ContextualSelectorStatus:
    enabled: bool
    status: str
    selected_model: str | None = None
    winner_set: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "selected_model": self.selected_model,
            "winner_set": list(self.winner_set),
        }


@dataclass(frozen=True)
class SelectionResult:
    status: str
    selected_model: str | None
    winner_set: tuple[str, ...]
    feasible_models: tuple[str, ...]
    excluded_by_feasibility: Mapping[str, tuple[str, ...]]
    mcs_set: tuple[str, ...]
    lcb_by_model: Mapping[str, float]
    robustness_lcb_by_block: Mapping[int, Mapping[str, float]]
    tie_break_trace: tuple[Mapping[str, Any], ...]
    bootstrap: PairedBootstrapResult | None
    mcs: MCSResult | None
    contextual_selector: ContextualSelectorStatus
    ranking_pool: tuple[str, ...] = ()
    excluded_by_preregistered_gate: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    preregistered_gate: Mapping[str, Any] | None = None
    diagnostic_representative: str | None = None
    mean_baseline_daily_cost: float | None = None
    economic_tolerance_fraction: float = ECONOMIC_WINNER_TOLERANCE_FRACTION
    economic_tolerance_usd: float | None = None
    operating_date_panel_sha256: str | None = None
    operating_date_count: int | None = None
    exact_loss_value_date_panel_verified: bool = False
    selection_metric: str = (
        "95pct_lower_confidence_bound_of_counterfactual_incremental_operational_value"
    )

    def as_dict(self, *, include_bootstrap_indices: bool = False) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_model": self.selected_model,
            "winner_set": list(self.winner_set),
            "feasible_models": list(self.feasible_models),
            "excluded_by_feasibility": {
                model: list(reasons)
                for model, reasons in self.excluded_by_feasibility.items()
            },
            "mcs_set": list(self.mcs_set),
            "lcb_by_model": dict(self.lcb_by_model),
            "robustness_lcb_by_block": {
                str(block): dict(values)
                for block, values in self.robustness_lcb_by_block.items()
            },
            "tie_break_trace": [dict(row) for row in self.tie_break_trace],
            "bootstrap": (
                self.bootstrap.as_dict(include_indices=include_bootstrap_indices)
                if self.bootstrap is not None
                else None
            ),
            "mcs": self.mcs.as_dict() if self.mcs is not None else None,
            "contextual_selector": self.contextual_selector.as_dict(),
            "ranking_pool": list(self.ranking_pool),
            "excluded_by_preregistered_gate": {
                model: list(reasons)
                for model, reasons in self.excluded_by_preregistered_gate.items()
            },
            "preregistered_gate": (
                dict(self.preregistered_gate)
                if self.preregistered_gate is not None
                else None
            ),
            "diagnostic_representative": self.diagnostic_representative,
            "diagnostic_representative_is_scientific_winner": bool(
                self.diagnostic_representative is not None
                and len(self.winner_set) == 1
                and self.diagnostic_representative == self.winner_set[0]
            ),
            "mean_baseline_daily_cost": self.mean_baseline_daily_cost,
            "economic_tolerance_fraction": self.economic_tolerance_fraction,
            "economic_tolerance_usd": self.economic_tolerance_usd,
            "operating_date_panel_sha256": self.operating_date_panel_sha256,
            "operating_date_count": self.operating_date_count,
            "exact_loss_value_date_panel_verified": (
                self.exact_loss_value_date_panel_verified
            ),
            "selection_metric": self.selection_metric,
            "perfect_information_semantics": "UNATTAINABLE_UPPER_BOUND_ONLY",
            "value_semantics": (
                "counterfactual_incremental_operational_value_not_commercial_profit"
            ),
        }


def contextual_selector_status(
    *,
    enabled: bool = False,
    selected_model: str | None = None,
    winner_set: Sequence[str] = (),
) -> ContextualSelectorStatus:
    """Return the required explicit default skip for the optional extension."""

    if not enabled:
        if selected_model is not None or winner_set:
            raise ValueError("a disabled contextual selector cannot select a model")
        return ContextualSelectorStatus(
            enabled=False,
            status=SKIPPED_OPTIONAL_EXTENSION,
        )
    frozen_set = tuple(sorted({str(model) for model in winner_set}))
    return ContextualSelectorStatus(
        enabled=True,
        status="COMPLETE" if selected_model or frozen_set else "NOT_RUN",
        selected_model=str(selected_model) if selected_model is not None else None,
        winner_set=frozen_set,
    )


def _finite_metric(
    summary: Mapping[str, Any],
    key: str,
    aliases: Sequence[str] = (),
) -> float | None:
    for candidate in (key, *aliases):
        value = summary.get(candidate)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric if np.isfinite(numeric) else None
    return None


def evaluate_feasibility(
    model_summaries: Mapping[str, Mapping[str, Any]],
    *,
    criteria: FeasibilityCriteria | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Apply every frozen operational/risk gate without dropping failures."""

    rule = criteria or FeasibilityCriteria()
    feasible: list[str] = []
    exclusions: dict[str, tuple[str, ...]] = {}
    for raw_model, summary in sorted(model_summaries.items()):
        model = str(raw_model)
        reasons: list[str] = []
        sla = _finite_metric(
            summary,
            "weighted_sla_satisfaction",
            aliases=("sla_satisfaction",),
        )
        if sla is None:
            reasons.append("MISSING_WEIGHTED_SLA_SATISFACTION")
        elif sla < rule.minimum_sla_satisfaction:
            reasons.append("SLA_BELOW_THRESHOLD")

        cvar_loss = _finite_metric(
            summary, "cvar95_loss", aliases=("cvar_loss", "cvar_0_95_loss")
        )
        baseline_cost = _finite_metric(
            summary,
            "baseline_operating_cost",
            aliases=("baseline_cost", "baseline_total_cost"),
        )
        if cvar_loss is None:
            reasons.append("MISSING_CVAR95_LOSS")
        if baseline_cost is None or baseline_cost <= 0.0:
            reasons.append("MISSING_OR_NONPOSITIVE_BASELINE_COST")
        if cvar_loss is not None and baseline_cost is not None and baseline_cost > 0.0:
            budget = (
                rule.cvar95_loss_budget_fraction_of_baseline_cost * baseline_cost
            )
            if cvar_loss > budget:
                reasons.append("CVAR95_LOSS_BUDGET_EXCEEDED")

        probability_below = _finite_metric(
            summary,
            "probability_below_baseline",
            aliases=("loss_probability",),
        )
        if probability_below is None:
            reasons.append("MISSING_PROBABILITY_BELOW_BASELINE")
        elif probability_below > rule.maximum_probability_below_baseline:
            reasons.append("PROBABILITY_BELOW_BASELINE_EXCEEDED")

        latency = _finite_metric(
            summary,
            "forecast_plus_solve_latency_seconds",
            aliases=("total_latency_seconds", "latency_seconds"),
        )
        if latency is None:
            reasons.append("MISSING_FORECAST_PLUS_SOLVE_LATENCY")
        elif latency > rule.market_gate_budget_seconds:
            reasons.append("MARKET_GATE_LATENCY_EXCEEDED")

        infeasible = _finite_metric(
            summary,
            "infeasible_days",
            aliases=("infeasible_count", "number_infeasible_days"),
        )
        if infeasible is None:
            reasons.append("MISSING_INFEASIBLE_DAY_COUNT")
        elif rule.require_zero_infeasible_days and infeasible != 0.0:
            reasons.append("NONZERO_INFEASIBLE_DAYS")

        invalid = _finite_metric(
            summary,
            "invalid_days",
            aliases=("invalid_count", "number_invalid_days"),
        )
        # An absent invalid count is conservatively interpreted as zero only
        # when the explicit all-days-valid flag is true.
        if invalid is None and summary.get("all_days_valid") is True:
            invalid = 0.0
        if invalid is None:
            reasons.append("MISSING_INVALID_DAY_COUNT")
        elif rule.require_zero_invalid_days and invalid != 0.0:
            reasons.append("NONZERO_INVALID_DAYS")

        timeout_aliases: tuple[str, ...] = ()
        gap_aliases: tuple[str, ...] = ()
        if rule.allow_legacy_solver_field_aliases:
            timeout_aliases = (
                "solver_time_limit_days",
                "time_limit_days",
                "timeout_days",
                "solver_any_time_limit_reached_days",
            )
            gap_aliases = (
                "nonzero_gap_days",
                "solver_gap_days",
                "solver_any_nonzero_gap_days",
            )
        # Timeout and any nonzero gap remain mandatory disclosures, but neither
        # is itself an exclusion under the bounded-gap amendment.  A timeout is
        # admissible only when the solver returned an incumbent and a finite
        # relative gap at or below the preregistered cap; the three failure
        # counts below enforce that condition without hiding timeout days.
        solver_timeout_days = _finite_metric(
            summary,
            rule.solver_timeout_day_field,
            aliases=timeout_aliases,
        )
        if solver_timeout_days is None:
            reasons.append("MISSING_SOLVER_TIMEOUT_DAY_COUNT")

        solver_nonzero_gap_days = _finite_metric(
            summary,
            rule.solver_nonzero_gap_day_field,
            aliases=gap_aliases,
        )
        if solver_nonzero_gap_days is None:
            reasons.append("MISSING_SOLVER_NONZERO_GAP_DAY_COUNT")

        bounded_gap_counts = (
            (
                rule.solver_gap_over_cap_day_field,
                "MISSING_SOLVER_GAP_OVER_CAP_DAY_COUNT",
                "NONZERO_SOLVER_GAP_OVER_CAP_DAYS",
                rule.require_zero_solver_gap_over_cap_days,
            ),
            (
                rule.solver_missing_gap_day_field,
                "MISSING_SOLVER_MISSING_GAP_DAY_COUNT",
                "NONZERO_SOLVER_MISSING_GAP_DAYS",
                rule.require_zero_solver_missing_gap_days,
            ),
            (
                rule.solver_no_incumbent_day_field,
                "MISSING_SOLVER_NO_INCUMBENT_DAY_COUNT",
                "NONZERO_SOLVER_NO_INCUMBENT_DAYS",
                rule.require_zero_solver_no_incumbent_days,
            ),
        )
        for field_name, missing_reason, nonzero_reason, require_zero in bounded_gap_counts:
            count = _finite_metric(summary, field_name)
            # Clean legacy zero-gap summaries remain unambiguous: when the
            # disclosed nonzero-gap, timeout, and infeasible counts are all
            # exactly zero, each bounded-gap failure count is necessarily zero.
            if (
                count is None
                and solver_nonzero_gap_days == 0.0
                and solver_timeout_days == 0.0
                and infeasible == 0.0
            ):
                count = 0.0
            if count is None:
                reasons.append(missing_reason)
            elif require_zero and count != 0.0:
                reasons.append(nonzero_reason)

        if reasons:
            exclusions[model] = tuple(reasons)
        else:
            feasible.append(model)
    return tuple(feasible), exclusions


def _derive_summary_from_values(
    values: np.ndarray,
    supplied: Mapping[str, Any],
    *,
    risk_alpha: float,
) -> dict[str, Any]:
    result = dict(supplied)
    finite_mask = np.isfinite(values)
    computed_invalid = int(np.count_nonzero(~finite_mask))
    for key in ("invalid_days", "invalid_count", "number_invalid_days"):
        if key not in supplied or supplied[key] is None:
            continue
        try:
            declared = float(supplied[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a numeric invalid-day count") from exc
        if (
            not np.isfinite(declared)
            or declared < 0.0
            or not declared.is_integer()
            or int(declared) != computed_invalid
        ):
            raise ValueError(
                f"{key} contradicts the computed invalid-day count "
                f"({declared!r} != {computed_invalid})"
            )
    if "all_days_valid" in supplied and supplied["all_days_valid"] is not None:
        declared_valid = supplied["all_days_valid"]
        if not isinstance(declared_valid, (bool, np.bool_)):
            raise ValueError("all_days_valid must be Boolean when supplied")
        if bool(declared_valid) != (computed_invalid == 0):
            raise ValueError("all_days_valid contradicts the daily value series")
    result["invalid_days"] = computed_invalid
    result["all_days_valid"] = computed_invalid == 0
    if computed_invalid == 0:
        result.setdefault(
            "probability_below_baseline", float(np.mean(values < 0.0))
        )
        result.setdefault("cvar95_loss", empirical_cvar(-values, alpha=risk_alpha))
    return result


def _validated_operating_date_panel(
    values: Sequence[str] | None,
    *,
    expected_count: int,
    label: str,
) -> tuple[tuple[str, ...] | None, str | None]:
    if values is None:
        return None, None
    dates = tuple(str(value) for value in values)
    if len(dates) != expected_count:
        raise ValueError(
            f"{label} length differs from the daily value denominator "
            f"({len(dates)} != {expected_count})"
        )
    if not dates or len(set(dates)) != len(dates):
        raise ValueError(f"{label} must be nonempty and unique")
    return dates, sha256_json(list(dates))


_TIE_BREAK_FIELDS = (
    ("mean_prescriptive_regret", "lower_prescriptive_regret"),
    ("tail_calibration_error", "better_tail_calibration"),
    ("forecast_plus_solve_latency_seconds", "lower_latency"),
    ("model_complexity", "lower_model_complexity"),
)


def _within_tolerance(
    value: float,
    best: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    tolerance = absolute_tolerance + relative_tolerance * max(
        abs(value), abs(best), 1.0
    )
    return abs(value - best) <= tolerance


def _normalize_preregistered_gate(
    gate: Mapping[str, Any] | None,
    candidate_ids: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], Mapping[str, Any] | None]:
    candidates = tuple(sorted(map(str, candidate_ids)))
    if gate is None:
        return candidates, {}, None
    normalized = dict(gate)
    raw_eligible = normalized.get("eligible_models")
    if not isinstance(raw_eligible, Sequence) or isinstance(raw_eligible, (str, bytes)):
        raise ValueError("preregistered_gate.eligible_models must be a sequence")
    eligible = tuple(sorted(set(map(str, raw_eligible))))
    unknown = set(eligible).difference(candidates)
    if unknown:
        raise ValueError(
            f"preregistered gate names unknown candidates: {sorted(unknown)}"
        )
    raw_exclusions = normalized.get("excluded_by_model", {})
    if not isinstance(raw_exclusions, Mapping):
        raise ValueError("preregistered_gate.excluded_by_model must be a mapping")
    exclusions: dict[str, tuple[str, ...]] = {}
    for model in candidates:
        if model in eligible:
            continue
        raw_reasons = raw_exclusions.get(model, ("PREREGISTERED_GATE_FAILED",))
        if not isinstance(raw_reasons, Sequence) or isinstance(raw_reasons, (str, bytes)):
            raise ValueError(
                f"preregistered gate reasons must be a sequence for {model}"
            )
        reasons = tuple(map(str, raw_reasons))
        if not reasons:
            reasons = ("PREREGISTERED_GATE_FAILED",)
        exclusions[model] = reasons
    normalized["eligible_models"] = list(eligible)
    normalized["excluded_by_model"] = {
        model: list(reasons) for model, reasons in exclusions.items()
    }
    return eligible, exclusions, normalized


def _apply_tie_breaks(
    candidates: Sequence[str],
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    remaining = tuple(sorted(candidates))
    trace: list[Mapping[str, Any]] = []
    for tie_field, label in _TIE_BREAK_FIELDS:
        if len(remaining) <= 1:
            break
        values: dict[str, float] = {}
        for model in remaining:
            raw = diagnostics.get(model, {}).get(tie_field)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                values = {}
                break
            if not np.isfinite(value):
                values = {}
                break
            values[model] = value
        if not values:
            trace.append(
                {
                    "criterion": label,
                    "status": "SKIPPED_INCOMPLETE_DIAGNOSTIC",
                    "remaining": list(remaining),
                }
            )
            continue
        best = min(values.values())
        retained = tuple(
            model
            for model in remaining
            if _within_tolerance(
                values[model],
                best,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        )
        trace.append(
            {
                "criterion": label,
                "status": "APPLIED",
                "values": values,
                "remaining": list(retained),
            }
        )
        remaining = retained
    return remaining, tuple(trace)


def select_dart_value(
    daily_incremental_values: Mapping[str, Sequence[float] | np.ndarray],
    model_summaries: Mapping[str, Mapping[str, Any]],
    *,
    daily_losses: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    operating_dates: Sequence[str] | None = None,
    daily_loss_operating_dates: Sequence[str] | None = None,
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    preregistered_gate: Mapping[str, Any] | None = None,
    feasibility_criteria: FeasibilityCriteria | None = None,
    risk_alpha: float = 0.95,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    primary_block_length: int = 7,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
    mcs_resamples: int = 2000,
    seed: int = 20260809,
    lcb_absolute_tolerance: float = 1e-9,
    lcb_relative_tolerance: float = 1e-9,
    economic_tolerance_fraction: float = ECONOMIC_WINNER_TOLERANCE_FRACTION,
    contextual_enabled: bool = False,
) -> SelectionResult:
    """Select by feasible/MCS-retained primary LCB, never by point error."""

    if not daily_incremental_values:
        raise ValueError("at least one candidate daily value series is required")
    if (
        not np.isfinite(float(economic_tolerance_fraction))
        or float(economic_tolerance_fraction) < 0.0
    ):
        raise ValueError("economic_tolerance_fraction must be finite and nonnegative")
    values = {
        str(model): np.asarray(series, dtype=float)
        for model, series in daily_incremental_values.items()
    }
    normalized_summaries = {
        str(model): dict(summary) for model, summary in model_summaries.items()
    }
    if any(series.ndim != 1 or series.size == 0 for series in values.values()):
        raise ValueError("every daily incremental value series must be nonempty and 1-D")
    if set(values) != set(normalized_summaries):
        raise ValueError(
            "daily values and model summaries must contain the identical candidate set"
        )
    lengths = {series.size for series in values.values()}
    if len(lengths) != 1:
        raise ValueError("all candidates must share the identical daily denominator")
    observation_count = int(next(iter(lengths)))
    normalized_loss_input: dict[str, np.ndarray] | None = None
    if daily_losses is not None:
        normalized_loss_input = {
            str(model): np.asarray(series, dtype=float)
            for model, series in daily_losses.items()
        }
        if set(normalized_loss_input) != set(values):
            raise ValueError(
                "daily_losses and incremental values must contain the identical "
                "candidate set"
            )
        for model, loss_series in normalized_loss_input.items():
            if loss_series.ndim != 1 or loss_series.shape != values[model].shape:
                raise ValueError(
                    f"daily loss/value denominator differs for {model}: "
                    f"{loss_series.shape} != {values[model].shape}"
                )
            if not bool(np.all(np.isfinite(loss_series))):
                raise ValueError(
                    f"daily losses for {model} contain non-finite values; silent "
                    "day filtering is forbidden"
                )
    value_dates, value_panel_sha = _validated_operating_date_panel(
        operating_dates,
        expected_count=observation_count,
        label="operating_dates",
    )
    loss_dates: tuple[str, ...] | None = None
    loss_panel_sha: str | None = None
    if daily_losses is not None and (
        operating_dates is not None or daily_loss_operating_dates is not None
    ):
        if operating_dates is None or daily_loss_operating_dates is None:
            raise ValueError(
                "both value and loss operating-date panels are required when either "
                "panel is supplied"
            )
        loss_dates, loss_panel_sha = _validated_operating_date_panel(
            daily_loss_operating_dates,
            expected_count=observation_count,
            label="daily_loss_operating_dates",
        )
        if loss_dates != value_dates or loss_panel_sha != value_panel_sha:
            raise ValueError(
                "daily_losses and incremental values do not share the exact ordered "
                "operating-date panel"
            )
    panel_verified = bool(
        value_dates is not None
        and (daily_losses is None or loss_dates is not None)
    )
    supplied_diagnostics = {
        str(model): dict(row) for model, row in (diagnostics or {}).items()
    }
    derived_summaries = {
        model: _derive_summary_from_values(
            series,
            normalized_summaries[model],
            risk_alpha=risk_alpha,
        )
        for model, series in values.items()
    }
    feasible, exclusions = evaluate_feasibility(
        derived_summaries, criteria=feasibility_criteria
    )
    gate_eligible, gate_exclusions, normalized_gate = _normalize_preregistered_gate(
        preregistered_gate, tuple(values)
    )
    contextual = contextual_selector_status(enabled=contextual_enabled)
    if not feasible:
        return SelectionResult(
            status="NO_FEASIBLE_MODEL",
            selected_model=None,
            winner_set=(),
            feasible_models=(),
            excluded_by_feasibility=exclusions,
            mcs_set=(),
            lcb_by_model={},
            robustness_lcb_by_block={},
            tie_break_trace=(),
            bootstrap=None,
            mcs=None,
            contextual_selector=contextual,
            ranking_pool=(),
            excluded_by_preregistered_gate=gate_exclusions,
            preregistered_gate=normalized_gate,
            economic_tolerance_fraction=float(economic_tolerance_fraction),
            operating_date_panel_sha256=value_panel_sha,
            operating_date_count=(observation_count if value_dates is not None else None),
            exact_loss_value_date_panel_verified=panel_verified,
        )

    ranking_pool = tuple(model for model in feasible if model in set(gate_eligible))
    if not ranking_pool:
        return SelectionResult(
            status="NO_FEASIBLE_PREREGISTERED_GATE_MODEL",
            selected_model=None,
            winner_set=(),
            feasible_models=feasible,
            excluded_by_feasibility=exclusions,
            mcs_set=(),
            lcb_by_model={},
            robustness_lcb_by_block={},
            tie_break_trace=(),
            bootstrap=None,
            mcs=None,
            contextual_selector=contextual,
            ranking_pool=(),
            excluded_by_preregistered_gate=gate_exclusions,
            preregistered_gate=normalized_gate,
            economic_tolerance_fraction=float(economic_tolerance_fraction),
            operating_date_panel_sha256=value_panel_sha,
            operating_date_count=(observation_count if value_dates is not None else None),
            exact_loss_value_date_panel_verified=panel_verified,
        )

    feasible_values = {model: values[model] for model in ranking_pool}
    bootstrap = paired_moving_block_bootstrap(
        feasible_values,
        block_lengths=block_lengths,
        primary_block_length=primary_block_length,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    if daily_losses is None:
        loss_map = {model: -values[model] for model in ranking_pool}
    else:
        assert normalized_loss_input is not None
        loss_map = {
            model: normalized_loss_input[model] for model in ranking_pool
        }
    mcs = model_confidence_set(
        loss_map,
        confidence_level=confidence_level,
        block_length=primary_block_length,
        resamples=mcs_resamples,
        seed=seed,
    )
    retained = tuple(model for model in ranking_pool if model in mcs.included_models)
    if not retained:
        return SelectionResult(
            status="NO_FEASIBLE_MCS_MEMBER",
            selected_model=None,
            winner_set=(),
            feasible_models=feasible,
            excluded_by_feasibility=exclusions,
            mcs_set=mcs.included_models,
            lcb_by_model={},
            robustness_lcb_by_block=bootstrap.lower_confidence_bounds,
            tie_break_trace=(),
            bootstrap=bootstrap,
            mcs=mcs,
            contextual_selector=contextual,
            ranking_pool=ranking_pool,
            excluded_by_preregistered_gate=gate_exclusions,
            preregistered_gate=normalized_gate,
            economic_tolerance_fraction=float(economic_tolerance_fraction),
            operating_date_panel_sha256=value_panel_sha,
            operating_date_count=(observation_count if value_dates is not None else None),
            exact_loss_value_date_panel_verified=panel_verified,
        )
    primary_lcbs = {
        model: bootstrap.lcb(model, primary_block_length) for model in retained
    }
    maximum_lcb = max(primary_lcbs.values())
    baseline_costs = [
        _finite_metric(derived_summaries[model], "baseline_operating_cost")
        for model in ranking_pool
    ]
    if any(value is None or value <= 0.0 for value in baseline_costs):
        raise ValueError(
            "economic winner tolerance requires positive baseline daily cost"
        )
    mean_baseline_daily_cost = float(
        np.mean(np.asarray(baseline_costs, dtype=float))
    )
    economic_tolerance_usd = (
        float(economic_tolerance_fraction) * mean_baseline_daily_cost
    )
    lcb_tied = tuple(
        sorted(
            model
            for model, value in primary_lcbs.items()
            if maximum_lcb - value <= economic_tolerance_usd + 1e-12
        )
    )
    trace: list[Mapping[str, Any]] = [
        {
            "criterion": "maximum_primary_7_day_lcb_with_economic_tolerance",
            "status": "APPLIED",
            "values": primary_lcbs,
            "mean_baseline_daily_cost": mean_baseline_daily_cost,
            "economic_tolerance_fraction": float(economic_tolerance_fraction),
            "economic_tolerance_usd": economic_tolerance_usd,
            "remaining": list(lcb_tied),
        }
    ]
    finalists, tie_trace = _apply_tie_breaks(
        lcb_tied,
        supplied_diagnostics,
        absolute_tolerance=lcb_absolute_tolerance,
        relative_tolerance=lcb_relative_tolerance,
    )
    trace.extend(tie_trace)
    diagnostic_representative = finalists[0] if len(finalists) == 1 else None
    selected = lcb_tied[0] if len(lcb_tied) == 1 else None
    # ``winner_set`` is the preregistered statistical LCB-tied set.  The
    # deterministic tie-break protocol may nominate a deployment/reporting
    # representative, but it cannot erase statistical indistinguishability.
    status = (
        "SELECTED"
        if len(lcb_tied) == 1
        else "STATISTICALLY_INDISTINGUISHABLE_WINNER_SET"
    )
    return SelectionResult(
        status=status,
        selected_model=selected,
        winner_set=lcb_tied,
        feasible_models=feasible,
        excluded_by_feasibility=exclusions,
        mcs_set=mcs.included_models,
        lcb_by_model=primary_lcbs,
        robustness_lcb_by_block=bootstrap.lower_confidence_bounds,
        tie_break_trace=tuple(trace),
        bootstrap=bootstrap,
        mcs=mcs,
        contextual_selector=contextual,
        ranking_pool=ranking_pool,
        excluded_by_preregistered_gate=gate_exclusions,
        preregistered_gate=normalized_gate,
        diagnostic_representative=diagnostic_representative,
        mean_baseline_daily_cost=mean_baseline_daily_cost,
        economic_tolerance_fraction=float(economic_tolerance_fraction),
        economic_tolerance_usd=economic_tolerance_usd,
        operating_date_panel_sha256=value_panel_sha,
        operating_date_count=(observation_count if value_dates is not None else None),
        exact_loss_value_date_panel_verified=panel_verified,
    )


dart_value_select = select_dart_value
filter_feasible_models = evaluate_feasibility


__all__ = [
    "ContextualSelectorStatus",
    "A1_NONINFERIORITY_FRACTION",
    "ECONOMIC_WINNER_TOLERANCE_FRACTION",
    "MAXIMUM_ACCEPTED_MIP_GAP",
    "FeasibilityCriteria",
    "SKIPPED_OPTIONAL_EXTENSION",
    "SelectionResult",
    "contextual_selector_status",
    "dart_value_select",
    "evaluate_feasibility",
    "filter_feasible_models",
    "select_dart_value",
]
