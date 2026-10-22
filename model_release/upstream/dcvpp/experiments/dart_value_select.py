"""Evidence-gated top-level orchestration for DART-ValueSelect.

The module deliberately does not train a forecast model or solve an optimizer.
It accepts completed, hash-addressed forecast/action/replay evidence; verifies
the exact paired panel; performs the frozen statistical selection; freezes all
selection choices before 2025 is opened; and applies those choices to the
once-only locked test.  Every reported number is either copied from supplied
evidence or deterministically derived from it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import csv
from dataclasses import dataclass, field
from datetime import date
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dcvpp.core import atomic_json, atomic_text, load_yaml, sha256_file, sha256_json, utc_now
from dcvpp.evaluation.dart_value_select import (
    A1_NONINFERIORITY_FRACTION,
    ECONOMIC_WINNER_TOLERANCE_FRACTION,
    MAXIMUM_ACCEPTED_MIP_GAP,
    FeasibilityCriteria,
    SKIPPED_OPTIONAL_EXTENSION,
    evaluate_feasibility,
    select_dart_value,
)
from dcvpp.evaluation.profit_evaluator import decision_value_summary
from dcvpp.experiments.dart_forecast_oos import CANDIDATE_POOL, FREEZE_FINGERPRINT_KEYS
from dcvpp.experiments.dart_partial_candidate import (
    PARTIAL_CANDIDATE_STATUS,
    PARTIAL_CANDIDATE_TERMINAL_CAP,
    validate_partial_candidate_pool_contract,
)
from dcvpp.experiments.dart_replay import (
    DART_VALUE_SELECT_SEEDS,
    EXPERIMENT_IDS,
    CandidateEvidence,
    build_common_replay_panel,
    decision_value_metrics,
    derive_a3_metric_contradictions,
)
from dcvpp.statistics.block_bootstrap import (
    moving_block_indices,
    paired_moving_block_bootstrap,
)


SCHEMA_VERSION = "dart_value_select_experiment_v4"
PHASES = ("selection", "locked_test")
STAGE_ORDER = (
    "selection_evidence",
    "selection_analysis",
    "selection_freeze",
    "locked_test_open",
    "locked_test_evidence",
    "finalization",
)
REGIMES = (
    "ordinary",
    "positive_dart_spike",
    "negative_dart_spike",
    "high_workload",
    "low_pv",
    "high_congestion",
)

REQUIRED_METRIC_FIELDS = (
    "mae",
    "rmse",
    "crps",
    "pinball_loss",
    "tail_crps",
    "threshold_weighted_crps",
    "emphasized_tail_threshold_weighted_crps",
    "q05_q95_threshold_weighted_crps",
    "q01_q99_extreme_threshold_weighted_crps",
    "tail_calibration_error",
    "interval_coverage_80",
    "interval_width_80",
    "interval_calibration_error_80",
    "interval_coverage_90",
    "interval_width_90",
    "interval_calibration_error_90",
    "energy_score",
    "variogram_score",
)
NULLABLE_METRIC_FIELDS = (
    "awcrps",
    "feasibility_conditioned_tail_crps",
    "feasibility_conditioned_q05_q95_tail_crps",
    "feasibility_conditioned_q01_q99_extreme_crps",
    "forl",
    "economic_action_brier",
    "action_precision",
    "action_recall",
    "action_f1",
    "wrong_action_loss",
    "missed_opportunity_loss",
    "wrong_action_rate",
    "missed_opportunity_rate",
    "wrong_action_economic_loss_usd",
    "missed_opportunity_economic_loss_usd",
    "q99_event_brier",
    "q99_pr_auc",
    "q99_precision",
    "q99_recall",
    "q99_f1",
    "q99_miss_rate",
    "spike_precision",
    "spike_recall",
    "spike_f1",
    "spike_pr_auc",
)
RATIO_OF_SUMS_METRIC_FIELDS = (
    "awcrps",
    "feasibility_conditioned_tail_crps",
    "feasibility_conditioned_q05_q95_tail_crps",
    "feasibility_conditioned_q01_q99_extreme_crps",
    "forl",
    "economic_action_brier",
    "action_precision",
    "action_recall",
    "action_f1",
    "wrong_action_loss",
    "missed_opportunity_loss",
    "wrong_action_rate",
    "missed_opportunity_rate",
    "wrong_action_economic_loss_usd",
    "missed_opportunity_economic_loss_usd",
)
POOLED_Q99_METRIC_FIELDS = (
    "q99_event_brier",
    "q99_pr_auc",
    "q99_precision",
    "q99_recall",
    "q99_f1",
    "q99_miss_rate",
    "spike_precision",
    "spike_recall",
    "spike_f1",
    "spike_pr_auc",
)
POOLED_Q99_RATIO_STAT_FIELDS = frozenset(
    set(POOLED_Q99_METRIC_FIELDS).difference({"q99_pr_auc", "spike_pr_auc"})
)
POOLED_Q99_AGGREGATION_SCOPE = (
    "candidate_pooled_date_event_panel_after_seed_probability_mean"
)
REQUIRED_METRIC_HASH_FIELDS = (
    "tail_threshold_contract_sha256",
    "normalized_feasibility_weight_sha256",
    "physical_flexible_mwh_sha256",
)
REQUIRED_METRIC_CONTRACT_NUMERIC_FIELDS = (
    "normalized_feasibility_weight_min",
    "normalized_feasibility_weight_max",
    "normalized_feasibility_weight_sum",
    "physical_flexible_mwh_total",
)
UNIT_INTERVAL_METRIC_FIELDS = frozenset(
    {
        "tail_calibration_error",
        "interval_coverage_80",
        "interval_calibration_error_80",
        "interval_coverage_90",
        "interval_calibration_error_90",
        "action_precision",
        "action_recall",
        "action_f1",
        "wrong_action_rate",
        "missed_opportunity_rate",
        "q99_event_brier",
        "q99_pr_auc",
        "q99_precision",
        "q99_recall",
        "q99_f1",
        "q99_miss_rate",
        "spike_f1",
        "spike_pr_auc",
    }
)
REQUIRED_OPERATIONAL_FIELDS = (
    "weighted_sla_satisfaction",
    "deadline_violations",
    "infeasible_day",
    "invalid_day",
    "solver_any_time_limit_reached",
    "solver_any_nonzero_gap",
    "solver_any_infeasible_under_frozen_rule",
    "first_stage_solver_audit",
    "recourse_solver_audit",
    "solver_feasibility_rule",
    "forecast_time_seconds",
    "solve_time_seconds",
    "model_complexity",
    "gate_time_feasible_flexible_energy_mwh",
    "source_evidence_sha256",
)


class DARTExperimentContractError(RuntimeError):
    """Base class for orchestration contract failures."""


class EvidenceContractError(DARTExperimentContractError):
    """Raised when supplied evidence is incomplete or internally inconsistent."""


class StageTransitionError(DARTExperimentContractError):
    """Raised when a stage is entered out of order or with changed inputs."""


class LockedTestGuardError(DARTExperimentContractError):
    """Raised when the once-only 2025 boundary would be violated."""


def _valid_sha256(value: Any, *, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise EvidenceContractError(f"{label} is not a SHA-256 digest")
    return digest


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceContractError(f"{label} must be numeric") from exc
    if not np.isfinite(result):
        raise EvidenceContractError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise EvidenceContractError(f"{label} must be at least {minimum}")
    return result


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))
    except (TypeError, ValueError):
        return False


def _date_tuple(values: Sequence[str], *, phase: str) -> tuple[str, ...]:
    normalized = tuple(sorted(str(value) for value in values))
    if not normalized or len(set(normalized)) != len(normalized):
        raise EvidenceContractError(f"{phase} operating dates must be nonempty and unique")
    parsed: list[date] = []
    for value in normalized:
        try:
            current = date.fromisoformat(value)
        except ValueError as exc:
            raise EvidenceContractError(
                f"{phase} operating date must use YYYY-MM-DD: {value}"
            ) from exc
        if current.isoformat() != value:
            raise EvidenceContractError(f"noncanonical operating date: {value}")
        parsed.append(current)
    expected_years = {2022, 2023, 2024} if phase == "selection" else {2025}
    observed_years = {item.year for item in parsed}
    if phase == "selection" and observed_years != expected_years:
        raise EvidenceContractError(
            "selection evidence must contain rolling OOS dates in each of 2022, 2023, and 2024"
        )
    if phase == "locked_test" and observed_years != expected_years:
        raise EvidenceContractError("locked-test evidence may contain 2025 dates only")
    return normalized


def _candidate_to_dict(candidate: CandidateEvidence) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "experiment_id": candidate.experiment_id,
        "hashes": dict(candidate.hashes),
        "evidence_sha256": candidate.evidence_sha256,
    }


def _candidate_from_dict(value: Mapping[str, Any]) -> CandidateEvidence:
    candidate = CandidateEvidence(
        candidate_id=str(value["candidate_id"]),
        experiment_id=str(value["experiment_id"]),
        hashes=dict(value["hashes"]),
    )
    supplied = value.get("evidence_sha256")
    if supplied is not None and supplied != candidate.evidence_sha256:
        raise EvidenceContractError("serialized candidate evidence hash changed")
    return candidate


@dataclass(frozen=True)
class DARTPhaseEvidence:
    """Completed evidence supplied for selection or the locked test.

    ``metric_records`` must contain one row per date/candidate/study-seed.
    ``replay_results`` must contain the exact date/candidate/A1-A2/three-seed
    panel emitted by :func:`run_dart_replay_panel`.
    """

    phase: Literal["selection", "locked_test"]
    operating_dates: tuple[str, ...]
    candidates_by_experiment: Mapping[str, tuple[CandidateEvidence, ...]]
    replay_results: tuple[Mapping[str, Any], ...]
    metric_records: tuple[Mapping[str, Any], ...]
    regime_labels_by_date: Mapping[str, tuple[str, ...]]
    source_evidence_sha256: str
    frozen_dependency_fingerprints: Mapping[str, str] = field(default_factory=dict)
    contextual_selector: Mapping[str, Any] | None = None
    candidate_pool_contract: Mapping[str, Any] | None = None
    fixture: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": self.phase,
            "operating_dates": list(self.operating_dates),
            "candidates_by_experiment": {
                experiment: [_candidate_to_dict(candidate) for candidate in candidates]
                for experiment, candidates in sorted(self.candidates_by_experiment.items())
            },
            "replay_results": [dict(row) for row in self.replay_results],
            "metric_records": [dict(row) for row in self.metric_records],
            "regime_labels_by_date": {
                day: list(labels)
                for day, labels in sorted(self.regime_labels_by_date.items())
            },
            "source_evidence_sha256": self.source_evidence_sha256,
            "frozen_dependency_fingerprints": dict(
                sorted(self.frozen_dependency_fingerprints.items())
            ),
            "contextual_selector": (
                dict(self.contextual_selector)
                if self.contextual_selector is not None
                else None
            ),
            "candidate_pool_contract": (
                dict(self.candidate_pool_contract)
                if self.candidate_pool_contract is not None
                else None
            ),
            "fixture": self.fixture,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DARTPhaseEvidence":
        return cls(
            phase=str(value["phase"]),  # type: ignore[arg-type]
            operating_dates=tuple(map(str, value["operating_dates"])),
            candidates_by_experiment={
                str(experiment): tuple(
                    _candidate_from_dict(candidate) for candidate in candidates
                )
                for experiment, candidates in value["candidates_by_experiment"].items()
            },
            replay_results=tuple(dict(row) for row in value["replay_results"]),
            metric_records=tuple(dict(row) for row in value["metric_records"]),
            regime_labels_by_date={
                str(day): tuple(map(str, labels))
                for day, labels in value["regime_labels_by_date"].items()
            },
            source_evidence_sha256=str(value["source_evidence_sha256"]),
            frozen_dependency_fingerprints=dict(
                value.get("frozen_dependency_fingerprints", {})
            ),
            contextual_selector=(
                dict(value["contextual_selector"])
                if value.get("contextual_selector") is not None
                else None
            ),
            candidate_pool_contract=(
                dict(value["candidate_pool_contract"])
                if value.get("candidate_pool_contract") is not None
                else None
            ),
            fixture=bool(value.get("fixture", False)),
        )

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


def _mean(values: Sequence[Any]) -> float | None:
    if not values:
        return None
    numeric: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            raise EvidenceContractError(
                f"mean input contains an undeclared null at position {index}; "
                "silent row filtering is forbidden"
            )
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise EvidenceContractError(
                f"mean input is nonnumeric at position {index}"
            ) from exc
        if not np.isfinite(converted):
            raise EvidenceContractError(
                f"mean input is non-finite at position {index}; silent row "
                "filtering is forbidden"
            )
        numeric.append(converted)
    return float(np.mean(np.asarray(numeric, dtype=float)))


def _average_ranks(values: Mapping[str, float], *, higher_is_better: bool) -> dict[str, float]:
    ordered = sorted(
        values,
        key=lambda model: ((-values[model]) if higher_is_better else values[model], model),
    )
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and np.isclose(
            values[ordered[end]], values[ordered[index]], rtol=1e-12, atol=1e-12
        ):
            end += 1
        rank = (index + 1 + end) / 2.0
        for model in ordered[index:end]:
            ranks[model] = rank
        index = end
    return ranks


def _spearman(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    models = tuple(sorted(set(left).intersection(right)))
    if len(models) < 3:
        return None
    left_values = np.asarray([left[model] for model in models], dtype=float)
    right_values = np.asarray([right[model] for model in models], dtype=float)
    if np.std(left_values) <= 1e-15 or np.std(right_values) <= 1e-15:
        return None
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _top_rank_set(ranks: Mapping[str, float]) -> tuple[str, ...]:
    if not ranks:
        return ()
    best = min(float(value) for value in ranks.values())
    return tuple(
        sorted(
            model
            for model, value in ranks.items()
            if np.isclose(float(value), best, rtol=1e-12, atol=1e-12)
        )
    )


def _normalized_kendall_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> tuple[float, int, int]:
    """Count only strictly discordant candidate pairs, with ties retained."""

    models = tuple(sorted(set(left).intersection(right)))
    total = len(models) * (len(models) - 1) // 2
    if total == 0:
        return 0.0, 0, 0
    discordant = 0
    for first_index, first in enumerate(models[:-1]):
        for second in models[first_index + 1 :]:
            left_difference = float(left[first]) - float(left[second])
            right_difference = float(right[first]) - float(right[second])
            if left_difference * right_difference < 0.0:
                discordant += 1
    return float(discordant / total), discordant, total


def _linear_least_squares_predict(
    training_features: np.ndarray,
    training_response: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    """Fit one transparent standardized linear model without future data."""

    train_x = np.asarray(training_features, dtype=float)
    train_y = np.asarray(training_response, dtype=float)
    test_x = np.asarray(test_features, dtype=float)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_y.ndim != 1:
        raise ValueError("H4c regression arrays have invalid dimensions")
    if train_x.shape[0] != train_y.size or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("H4c regression arrays have incompatible shapes")
    means = np.mean(train_x, axis=0)
    scales = np.std(train_x, axis=0)
    scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
    train_design = np.column_stack(
        [np.ones(train_x.shape[0]), (train_x - means) / scales]
    )
    test_design = np.column_stack(
        [np.ones(test_x.shape[0]), (test_x - means) / scales]
    )
    coefficients, *_ = np.linalg.lstsq(train_design, train_y, rcond=None)
    return np.asarray(test_design @ coefficients, dtype=float)


class DARTValueSelectExperiment:
    """Persistent state machine for the complete DART-ValueSelect experiment."""

    def __init__(
        self,
        output_root: Path,
        config: Path | Mapping[str, Any],
        *,
        contract_profile: Literal["paper", "test_fixture"] = "paper",
        candidate_pool_contract: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.contract_profile = contract_profile
        if isinstance(config, Path):
            self.config_path = config.resolve()
            self.config = load_yaml(self.config_path)
            self.config_sha256 = sha256_file(self.config_path)
        else:
            self.config_path = None
            self.config = deepcopy(dict(config))
            self.config_sha256 = sha256_json(self.config)
        self._validate_config()
        self.candidate_pool_contract = (
            validate_partial_candidate_pool_contract(candidate_pool_contract)
            if candidate_pool_contract is not None
            else None
        )
        self.candidate_ids = tuple(
            map(
                str,
                (
                    self.candidate_pool_contract["eligible_candidates"]
                    if self.candidate_pool_contract is not None
                    else self.config["candidate_contract"]["frozen_deployable_pool"]
                ),
            )
        )
        self.experiment_fingerprint = sha256_json(
            {
                "schema_version": SCHEMA_VERSION,
                "config_sha256": self.config_sha256,
                "contract_profile": self.contract_profile,
                "candidate_ids": list(self.candidate_ids),
                "candidate_pool_contract_sha256": (
                    self.candidate_pool_contract.get("contract_sha256")
                    if self.candidate_pool_contract is not None
                    else None
                ),
                "study_seeds": list(DART_VALUE_SELECT_SEEDS),
            }
        )

    @property
    def state_path(self) -> Path:
        return self.output_root / "checkpoints" / "dart_value_select_state.json"

    @property
    def selection_evidence_path(self) -> Path:
        return self.output_root / "checkpoints" / "selection_evidence.json"

    @property
    def selection_analysis_path(self) -> Path:
        return self.output_root / "checkpoints" / "selection_analysis.json"

    @property
    def selection_freeze_path(self) -> Path:
        return self.output_root / "checkpoints" / "selection_freeze_manifest.json"

    @property
    def locked_ledger_path(self) -> Path:
        return self.output_root / "checkpoints" / "locked_test_once.json"

    @property
    def locked_open_contract_path(self) -> Path:
        return self.output_root / "checkpoints" / "locked_test_open_contract.json"

    @property
    def locked_evidence_path(self) -> Path:
        return self.output_root / "checkpoints" / "locked_test_evidence.json"

    @property
    def final_manifest_path(self) -> Path:
        return self.output_root / "manifests" / "dart_value_select_manifest.json"

    @property
    def partial_candidate_state_amendment_path(self) -> Path:
        return (
            self.output_root
            / "checkpoints"
            / "partial_candidate_pre_result_state_amendment.json"
        )

    def _validate_config(self) -> None:
        required = {
            "candidate_contract",
            "randomization_contract",
            "chronology",
            "decision_value",
            "feasibility",
            "bootstrap",
            "model_confidence_set",
            "selection",
            "experiments",
            "contextual_selector",
            "sign_contract",
            "aggregation_contract",
            "hypothesis_contract",
        }
        missing = required.difference(self.config)
        if missing:
            raise DARTExperimentContractError(
                f"DART config is missing sections: {sorted(missing)}"
            )
        candidates = tuple(
            map(str, self.config["candidate_contract"]["frozen_deployable_pool"])
        )
        if not candidates or len(set(candidates)) != len(candidates):
            raise DARTExperimentContractError("candidate pool must be nonempty and unique")
        paper_pool = tuple(candidate.name for candidate in CANDIDATE_POOL)
        if self.contract_profile == "paper" and candidates != paper_pool:
            raise DARTExperimentContractError(
                "paper mode requires the exact frozen nine-candidate pool"
            )
        seeds = tuple(self.config["randomization_contract"]["study_seeds"])
        if seeds != DART_VALUE_SELECT_SEEDS:
            raise DARTExperimentContractError(
                f"study seeds must be exactly {DART_VALUE_SELECT_SEEDS}"
            )
        chronology = self.config["chronology"]
        if tuple(chronology["selection_years"]) != (2022, 2023, 2024):
            raise DARTExperimentContractError("selection years must be 2022-2024")
        if chronology["locked_test_year"] != 2025 or chronology["locked_test_runs"] != 1:
            raise DARTExperimentContractError("the locked test must be 2025 and once-only")
        blocks = tuple(self.config["bootstrap"]["blocks_days"])
        if blocks != (7, 14, 28) or self.config["bootstrap"]["primary_block_days"] != 7:
            raise DARTExperimentContractError("bootstrap blocks must be primary 7 plus 14/28")
        if self.config["decision_value"]["profit_capture_ratio_formula"] != (
            "sum(Pi_real_model - Pi_real_baseline) / "
            "sum(Pi_perfect_information - Pi_real_baseline)"
        ):
            raise DARTExperimentContractError("PCR orientation is not canonical")
        sign = self.config["sign_contract"]
        if sign["canonical_dart_formula"] != "da_lbmp - rt_lbmp":
            raise DARTExperimentContractError("canonical DART must be DA minus RT")
        for experiment in ("A1_dart_only_replacement", "A2_full_deployment_chain", "A3_metric_contradiction"):
            if self.config["experiments"][experiment]["required"] is not True:
                raise DARTExperimentContractError(f"{experiment} must remain required")
        hypotheses = self.config["hypothesis_contract"]
        required_hypotheses = {
            "H4a_material_rank_disagreement",
            "H4b_conservative_set_valued_dominance",
            "H4c_incremental_explanatory_test",
            "A3_date_level_contradiction",
        }
        if not required_hypotheses.issubset(hypotheses):
            raise DARTExperimentContractError(
                "DART config lacks one or more frozen H4/A3 decision contracts"
            )
        h4c = hypotheses["H4c_incremental_explanatory_test"]
        if tuple(h4c["point_error_features"]) != ("mae", "rmse"):
            raise DARTExperimentContractError("H4c point features must remain MAE and RMSE")
        if tuple(h4c["years"]) != (2022, 2023, 2024):
            raise DARTExperimentContractError("H4c must use selection OOS years 2022-2024")

    def initialize(self) -> dict[str, Any]:
        if self.state_path.is_file():
            existing = json.loads(self.state_path.read_text(encoding="utf-8"))
            if existing.get("experiment_fingerprint") == self.experiment_fingerprint:
                return existing
            if self.candidate_pool_contract is None:
                raise StageTransitionError("existing state belongs to a different contract")
            full_candidate_ids = tuple(
                map(
                    str,
                    self.config["candidate_contract"]["frozen_deployable_pool"],
                )
            )
            result_paths = (
                self.selection_evidence_path,
                self.selection_analysis_path,
                self.selection_freeze_path,
                self.locked_open_contract_path,
                self.locked_ledger_path,
                self.locked_evidence_path,
                self.final_manifest_path,
            )
            if (
                existing.get("status") != "INITIALIZED"
                or existing.get("config_sha256") != self.config_sha256
                or tuple(map(str, existing.get("candidate_ids", ())))
                != full_candidate_ids
                or existing.get("candidate_pool_contract") is not None
                or existing.get("stages") not in ({}, None)
                or int(existing.get("locked_test_run_count", -1)) != 0
                or any(path.exists() for path in result_paths)
            ):
                raise StageTransitionError(
                    "existing selector state has results and cannot adopt a partial pool"
                )
            existing_file_sha = sha256_file(self.state_path)
            amendment = {
                "schema_version": "dart_partial_candidate_pre_result_state_v1",
                "status": "AMENDED_BEFORE_SELECTION_EVIDENCE",
                "source_experiment_fingerprint": existing.get(
                    "experiment_fingerprint"
                ),
                "source_state_file_sha256": existing_file_sha,
                "source_candidate_ids": list(full_candidate_ids),
                "source_stage_count": 0,
                "source_locked_test_run_count": 0,
                "target_experiment_fingerprint": self.experiment_fingerprint,
                "candidate_pool_contract": self.candidate_pool_contract,
                "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
                "selection_or_locked_results_consumed": False,
                "amended_at_utc": utc_now(),
            }
            amendment["amendment_sha256"] = sha256_json(amendment)
            atomic_json(self.partial_candidate_state_amendment_path, amendment)
        state = {
            "schema_version": SCHEMA_VERSION,
            "status": "INITIALIZED",
            "experiment_fingerprint": self.experiment_fingerprint,
            "config_sha256": self.config_sha256,
            "contract_profile": self.contract_profile,
            "candidate_ids": list(self.candidate_ids),
            "candidate_pool_contract": self.candidate_pool_contract,
            "study_seeds": list(DART_VALUE_SELECT_SEEDS),
            "stages": {},
            "locked_test_run_count": 0,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
        }
        if self.partial_candidate_state_amendment_path.is_file():
            state["pre_result_candidate_pool_amendment"] = {
                "path": self.partial_candidate_state_amendment_path.as_posix(),
                "sha256": sha256_file(self.partial_candidate_state_amendment_path),
            }
        atomic_json(self.state_path, state)
        return state

    def _load_state(self) -> dict[str, Any]:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("experiment_fingerprint") != self.experiment_fingerprint:
            raise StageTransitionError("existing state belongs to a different contract")
        return state

    @staticmethod
    def _verify_outputs(record: Mapping[str, Any]) -> None:
        for output in record.get("outputs", []):
            path = Path(output["path"])
            if not path.is_file() or sha256_file(path) != output["sha256"]:
                raise StageTransitionError(f"completed stage output changed: {path}")

    def _require_stage(self, name: str) -> None:
        self.initialize()
        state = self._load_state()
        record = state.get("stages", {}).get(name)
        if not record or record.get("status") != "COMPLETE":
            raise StageTransitionError(f"required stage is not complete: {name}")
        self._verify_outputs(record)

    def _begin_stage(
        self,
        name: str,
        input_sha256: str,
        *,
        previous: str | None,
    ) -> tuple[dict[str, Any], bool]:
        self.initialize()
        if previous is not None:
            self._require_stage(previous)
        state = self._load_state()
        existing = state["stages"].get(name)
        if existing is not None:
            if existing.get("input_sha256") != input_sha256:
                raise StageTransitionError(f"{name} input changed after checkpointing")
            if existing.get("status") == "COMPLETE":
                self._verify_outputs(existing)
                return existing, True
        record = {
            "status": "STARTED",
            "input_sha256": input_sha256,
            "started_at_utc": (existing or {}).get("started_at_utc", utc_now()),
            "last_resumed_at_utc": utc_now(),
            "attempt_count": int((existing or {}).get("attempt_count", 0)) + 1,
        }
        state["stages"][name] = record
        state["status"] = f"{name.upper()}_RUNNING"
        state["updated_at_utc"] = utc_now()
        atomic_json(self.state_path, state)
        return record, False

    def _finish_stage(
        self,
        name: str,
        input_sha256: str,
        outputs: Sequence[Path],
        *,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._load_state()
        current = state["stages"].get(name)
        if current is None or current.get("input_sha256") != input_sha256:
            raise StageTransitionError(f"{name} was not started with this input")
        output_records = []
        for path in outputs:
            resolved = path.resolve()
            if not resolved.is_file():
                raise StageTransitionError(f"stage output is absent: {resolved}")
            output_records.append(
                {"path": resolved.as_posix(), "sha256": sha256_file(resolved)}
            )
        record = {
            **current,
            "status": "COMPLETE",
            "completed_at_utc": utc_now(),
            "outputs": output_records,
            "details": dict(details or {}),
        }
        state["stages"][name] = record
        state["status"] = f"{name.upper()}_COMPLETE"
        state["updated_at_utc"] = utc_now()
        atomic_json(self.state_path, state)
        return record

    def _validate_phase_evidence(self, evidence: DARTPhaseEvidence) -> dict[str, Any]:
        if evidence.phase not in PHASES:
            raise EvidenceContractError(f"unknown phase: {evidence.phase}")
        dates = _date_tuple(evidence.operating_dates, phase=evidence.phase)
        if self.contract_profile == "paper" and evidence.fixture:
            raise EvidenceContractError("fixture evidence is forbidden in paper mode")
        source_sha = _valid_sha256(
            evidence.source_evidence_sha256, label="phase source evidence"
        )
        if self.candidate_pool_contract is None:
            if evidence.candidate_pool_contract is not None:
                raise EvidenceContractError(
                    "full-pool experiment received a partial candidate contract"
                )
        else:
            try:
                evidence_pool = validate_partial_candidate_pool_contract(
                    evidence.candidate_pool_contract
                )
            except ValueError as exc:
                raise EvidenceContractError(str(exc)) from exc
            if (
                evidence_pool.get("contract_sha256")
                != self.candidate_pool_contract.get("contract_sha256")
            ):
                raise EvidenceContractError(
                    "phase candidate pool differs from the admitted selection contract"
                )
        if set(evidence.candidates_by_experiment) != set(EXPERIMENT_IDS):
            raise EvidenceContractError(f"candidate mapping must contain {EXPERIMENT_IDS}")
        for experiment in EXPERIMENT_IDS:
            observed = tuple(
                sorted(candidate.candidate_id for candidate in evidence.candidates_by_experiment[experiment])
            )
            if observed != tuple(sorted(self.candidate_ids)):
                raise EvidenceContractError(
                    f"{experiment} candidate set differs from the frozen pool"
                )
        panel = build_common_replay_panel(dates, evidence.candidates_by_experiment)
        expected = {
            (
                cell.operating_date,
                cell.candidate_id,
                cell.experiment_id,
                cell.seed,
            ): cell
            for cell in panel["cells"]
        }
        observed: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
        baseline_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
        pi_hashes: dict[str, set[str]] = defaultdict(set)
        for row in evidence.replay_results:
            key = (
                str(row.get("operating_date")),
                str(row.get("candidate_id")),
                str(row.get("experiment_id")),
                int(row.get("seed")),
            )
            if key in observed:
                raise EvidenceContractError(f"duplicate replay cell: {key}")
            if key not in expected:
                raise EvidenceContractError(f"unexpected replay cell: {key}")
            cell = expected[key]
            if row.get("cell_sha256") != cell.cell_sha256:
                raise EvidenceContractError(f"replay cell hash mismatch: {key}")
            if row.get("identity_sha256") != cell.identity_sha256:
                raise EvidenceContractError(f"replay identity hash mismatch: {key}")
            if dict(row.get("evidence_hashes", {})) != dict(cell.evidence_hashes):
                raise EvidenceContractError(f"replay evidence hashes changed: {key}")
            candidate = row.get("candidate_result")
            baseline = row.get("baseline_result")
            perfect = row.get("perfect_information_result")
            supplied_value = row.get("decision_value")
            if not all(isinstance(item, Mapping) for item in (candidate, baseline, perfect, supplied_value)):
                raise EvidenceContractError(f"replay cell lacks cost/value mappings: {key}")
            computed = decision_value_metrics(candidate, baseline, perfect)
            for field_name in (
                "candidate_simulated_total_cost_usd",
                "baseline_simulated_total_cost_usd",
                "perfect_information_cost_lower_bound_usd",
                "incremental_operational_value_usd",
                "prescriptive_regret_usd",
                "profit_capture_ratio",
            ):
                if not _same_number(supplied_value.get(field_name), computed.get(field_name)):
                    raise EvidenceContractError(
                        f"replay decision metric {field_name} failed recomputation: {key}"
                    )
            if supplied_value.get("counterfactual") is not True:
                raise EvidenceContractError("replay decision value must be counterfactual")
            if supplied_value.get("realized_commercial_profit_claim") is not False:
                raise EvidenceContractError("commercial-profit claims are forbidden")
            self._validate_operational_result(candidate, key=key, role="candidate")
            self._validate_operational_result(baseline, key=key, role="baseline")
            self._perfect_information_solver_status(perfect, key=key)
            baseline_hashes[(key[0], key[3])].add(sha256_json(baseline))
            pi_hashes[key[0]].add(sha256_json(perfect))
            observed[key] = row
        missing = set(expected).difference(observed)
        if missing or len(observed) != len(expected):
            preview = sorted(missing)[:3]
            raise EvidenceContractError(
                f"replay panel is incomplete: missing={len(missing)} preview={preview}"
            )
        if any(len(values) != 1 for values in baseline_hashes.values()):
            raise EvidenceContractError("baseline changed across candidates")
        if any(len(values) != 1 for values in pi_hashes.values()):
            raise EvidenceContractError("perfect-information bound changed within a date")

        expected_metrics = {
            (day, candidate, seed)
            for day in dates
            for candidate in self.candidate_ids
            for seed in DART_VALUE_SELECT_SEEDS
        }
        metric_keys: set[tuple[str, str, int]] = set()
        metric_hashes_by_date: dict[str, dict[str, set[str]]] = {
            field_name: defaultdict(set) for field_name in REQUIRED_METRIC_HASH_FIELDS
        }
        metric_contract_values_by_date: dict[str, dict[str, list[float]]] = {
            field_name: defaultdict(list)
            for field_name in REQUIRED_METRIC_CONTRACT_NUMERIC_FIELDS
        }
        pooled_q99_contract_by_candidate: dict[
            str, dict[str, list[tuple[Any, Any, Any]]]
        ] = {
            candidate: {field_name: [] for field_name in POOLED_Q99_METRIC_FIELDS}
            for candidate in self.candidate_ids
        }
        for row in evidence.metric_records:
            key = (
                str(row.get("operating_date")),
                str(row.get("candidate_id")),
                int(row.get("seed")),
            )
            if key in metric_keys:
                raise EvidenceContractError(f"duplicate metric record: {key}")
            if key not in expected_metrics:
                raise EvidenceContractError(f"unexpected metric record: {key}")
            for field_name in REQUIRED_METRIC_FIELDS:
                metric_value = _finite(
                    row.get(field_name),
                    label=f"metric {field_name} at {key}",
                    minimum=0.0,
                )
                if field_name in UNIT_INTERVAL_METRIC_FIELDS and metric_value > 1.0:
                    raise EvidenceContractError(
                        f"metric {field_name} at {key} must not exceed one"
                    )
            for field_name in NULLABLE_METRIC_FIELDS:
                value = row.get(field_name)
                status = row.get(f"{field_name}_status")
                if value is None:
                    if not status:
                        raise EvidenceContractError(
                            f"null {field_name} requires an explicit status at {key}"
                        )
                else:
                    metric_value = _finite(
                        value,
                        label=f"metric {field_name} at {key}",
                        minimum=0.0,
                    )
                    if (
                        field_name in UNIT_INTERVAL_METRIC_FIELDS
                        and metric_value > 1.0
                    ):
                        raise EvidenceContractError(
                            f"metric {field_name} at {key} must not exceed one"
                        )
                if field_name in RATIO_OF_SUMS_METRIC_FIELDS:
                    sufficient = row.get("metric_sufficient_statistics", {}).get(
                        field_name
                    )
                    if not isinstance(sufficient, Mapping):
                        raise EvidenceContractError(
                            f"metric {field_name} lacks ratio sufficient statistics at {key}"
                        )
                    numerator = _finite(
                        sufficient.get("numerator"),
                        label=f"metric {field_name} numerator at {key}",
                        minimum=0.0,
                    )
                    denominator = _finite(
                        sufficient.get("denominator"),
                        label=f"metric {field_name} denominator at {key}",
                        minimum=0.0,
                    )
                    _finite(
                        sufficient.get("count"),
                        label=f"metric {field_name} count at {key}",
                        minimum=0.0,
                    )
                    del numerator
                    if denominator == 0.0 and not (
                        sufficient.get("undefined_reason") or status
                    ):
                        raise EvidenceContractError(
                            f"zero-denominator {field_name} lacks an explicit reason at {key}"
                        )
            for field_name in POOLED_Q99_RATIO_STAT_FIELDS:
                sufficient = row.get("metric_sufficient_statistics", {}).get(
                    field_name
                )
                if not isinstance(sufficient, Mapping):
                    raise EvidenceContractError(
                        f"pooled q99 metric {field_name} lacks sufficient statistics "
                        f"at {key}"
                    )
                if sufficient.get("aggregation_scope") != POOLED_Q99_AGGREGATION_SCOPE:
                    raise EvidenceContractError(
                        f"pooled q99 metric {field_name} has the wrong panel scope "
                        f"at {key}"
                    )
                for statistic_name in ("numerator", "denominator", "count"):
                    _finite(
                        sufficient.get(statistic_name),
                        label=(
                            f"pooled q99 metric {field_name} {statistic_name} at {key}"
                        ),
                        minimum=0.0,
                    )
                if not _same_number(sufficient.get("value"), row.get(field_name)):
                    raise EvidenceContractError(
                        f"pooled q99 metric {field_name} value changed at {key}"
                    )
                if sufficient.get("status") != row.get(f"{field_name}_status"):
                    raise EvidenceContractError(
                        f"pooled q99 metric {field_name} status changed at {key}"
                    )
            proper_tail_sum = float(row["q05_q95_threshold_weighted_crps"]) + float(
                row["q01_q99_extreme_threshold_weighted_crps"]
            )
            for alias in (
                "tail_crps",
                "threshold_weighted_crps",
                "emphasized_tail_threshold_weighted_crps",
            ):
                if not _same_number(row[alias], proper_tail_sum):
                    raise EvidenceContractError(
                        f"{alias} is not the q05/q95 plus q01/q99 proper score at {key}"
                    )
            for canonical, legacy in (
                ("q99_precision", "spike_precision"),
                ("q99_recall", "spike_recall"),
                ("q99_f1", "spike_f1"),
                ("q99_pr_auc", "spike_pr_auc"),
            ):
                if not _same_number(row.get(canonical), row.get(legacy)):
                    raise EvidenceContractError(
                        f"metric aliases {canonical} and {legacy} disagree at {key}"
                    )
                if row.get(f"{canonical}_status") != row.get(f"{legacy}_status"):
                    raise EvidenceContractError(
                        f"metric alias statuses {canonical} and {legacy} disagree at {key}"
                    )
                if row.get(f"{canonical}_undefined_reason") != row.get(
                    f"{legacy}_undefined_reason"
                ):
                    raise EvidenceContractError(
                        f"metric alias reasons {canonical} and {legacy} disagree at {key}"
                    )
            for field_name in POOLED_Q99_METRIC_FIELDS:
                pooled_q99_contract_by_candidate[key[1]][field_name].append(
                    (
                        row.get(field_name),
                        row.get(f"{field_name}_status"),
                        row.get(f"{field_name}_undefined_reason"),
                    )
                )
            for field_name in REQUIRED_METRIC_HASH_FIELDS:
                digest = _valid_sha256(
                    row.get(field_name), label=f"metric {field_name} at {key}"
                )
                metric_hashes_by_date[field_name][key[0]].add(digest)
            normalized_min = _finite(
                row.get("normalized_feasibility_weight_min"),
                label=f"normalized feasibility weight minimum at {key}",
                minimum=0.0,
            )
            normalized_max = _finite(
                row.get("normalized_feasibility_weight_max"),
                label=f"normalized feasibility weight maximum at {key}",
                minimum=0.0,
            )
            normalized_sum = _finite(
                row.get("normalized_feasibility_weight_sum"),
                label=f"normalized feasibility weight sum at {key}",
                minimum=0.0,
            )
            physical_total = _finite(
                row.get("physical_flexible_mwh_total"),
                label=f"physical flexible MWh total at {key}",
                minimum=0.0,
            )
            if normalized_min > 1.0 or normalized_max > 1.0:
                raise EvidenceContractError(
                    f"normalized feasibility weight range exceeds [0, 1] at {key}"
                )
            if normalized_min > normalized_max + 1e-12:
                raise EvidenceContractError(
                    f"normalized feasibility weight minimum exceeds maximum at {key}"
                )
            if normalized_sum + 1e-12 < normalized_max:
                raise EvidenceContractError(
                    f"normalized feasibility weight sum is smaller than maximum at {key}"
                )
            if normalized_max == 0.0 and normalized_sum > 1e-12:
                raise EvidenceContractError(
                    f"zero normalized feasibility weights have a nonzero sum at {key}"
                )
            metric_contract_values = {
                "normalized_feasibility_weight_min": normalized_min,
                "normalized_feasibility_weight_max": normalized_max,
                "normalized_feasibility_weight_sum": normalized_sum,
                "physical_flexible_mwh_total": physical_total,
            }
            for field_name, metric_value in metric_contract_values.items():
                metric_contract_values_by_date[field_name][key[0]].append(metric_value)
            _valid_sha256(
                row.get("source_evidence_sha256"),
                label=f"metric source evidence at {key}",
            )
            if row.get("future_realized_inputs_used") is not False:
                raise EvidenceContractError("metric evidence used future realized inputs")
            if row.get("locked_test_outcome_used_for_gate_features") is not False:
                raise EvidenceContractError("metric evidence used a locked-test outcome")
            metric_keys.add(key)
        if metric_keys != expected_metrics:
            raise EvidenceContractError(
                f"metric panel is incomplete: missing={len(expected_metrics - metric_keys)}"
            )
        candidate_independent_metric_contract: dict[str, Any] = {}
        for field_name, by_date in metric_hashes_by_date.items():
            candidate_independent_metric_contract[field_name] = {}
            for operating_date in dates:
                values = by_date[operating_date]
                if len(values) != 1:
                    raise EvidenceContractError(
                        f"metric {field_name} is not candidate-independent within date "
                        f"{operating_date}"
                    )
                candidate_independent_metric_contract[field_name][operating_date] = next(
                    iter(values)
                )
        for field_name, by_date in metric_contract_values_by_date.items():
            candidate_independent_metric_contract[field_name] = {}
            for operating_date in dates:
                values = by_date[operating_date]
                reference = values[0]
                if any(not _same_number(reference, value) for value in values[1:]):
                    raise EvidenceContractError(
                        f"metric {field_name} is not candidate-independent within date "
                        f"{operating_date}"
                    )
                candidate_independent_metric_contract[field_name][operating_date] = (
                    reference
                )
        for candidate, metric_contract in pooled_q99_contract_by_candidate.items():
            for field_name, values in metric_contract.items():
                reference_value, reference_status, reference_reason = values[0]
                if any(
                    not _same_number(reference_value, value)
                    or status != reference_status
                    or reason != reference_reason
                    for value, status, reason in values[1:]
                ):
                    raise EvidenceContractError(
                        f"pooled q99 metric {field_name} is not unique within "
                        f"candidate {candidate}"
                    )
        regimes = {
            str(day): tuple(map(str, labels))
            for day, labels in evidence.regime_labels_by_date.items()
        }
        if set(regimes) != set(dates):
            raise EvidenceContractError("regime labels must cover the exact operating-date panel")
        for day, labels in regimes.items():
            if not labels:
                raise EvidenceContractError(f"date {day} requires at least one supplied regime label")
            if len(set(labels)) != len(labels) or not set(labels).issubset(REGIMES):
                raise EvidenceContractError(f"date {day} has invalid regime labels")
        return {
            "phase": evidence.phase,
            "operating_dates": list(dates),
            "operating_date_count": len(dates),
            "candidate_ids": list(self.candidate_ids),
            "candidate_pool_contract": self.candidate_pool_contract,
            "experiments": list(EXPERIMENT_IDS),
            "seeds": list(DART_VALUE_SELECT_SEEDS),
            "panel_sha256": panel["panel_sha256"],
            "panel_cell_count": panel["cell_count"],
            "metric_record_count": len(metric_keys),
            "candidate_independent_metric_contract_sha256": sha256_json(
                candidate_independent_metric_contract
            ),
            "identity_audits": panel["identity_audits"],
            "source_evidence_sha256": source_sha,
            "evidence_fingerprint": evidence.fingerprint,
            "counterfactual": True,
            "realized_commercial_profit_claim": False,
        }

    @staticmethod
    def _validate_operational_result(
        result: Mapping[str, Any],
        *,
        key: tuple[str, str, str, int],
        role: str,
    ) -> None:
        missing = set(REQUIRED_OPERATIONAL_FIELDS).difference(result)
        if missing:
            raise EvidenceContractError(
                f"{role} result {key} lacks operational fields: {sorted(missing)}"
            )
        sla = _finite(
            result["weighted_sla_satisfaction"],
            label=f"{role} weighted SLA at {key}",
        )
        if not 0.0 <= sla <= 1.0:
            raise EvidenceContractError(
                f"{role} weighted SLA must be in [0,1] at {key}"
            )
        for field_name in (
            "deadline_violations",
            "forecast_time_seconds",
            "solve_time_seconds",
            "model_complexity",
            "gate_time_feasible_flexible_energy_mwh",
        ):
            _finite(
                result[field_name],
                label=f"{role} {field_name} at {key}",
                minimum=0.0,
            )
        for field_name in (
            "infeasible_day",
            "invalid_day",
            "solver_any_time_limit_reached",
            "solver_any_nonzero_gap",
            "solver_any_infeasible_under_frozen_rule",
        ):
            if not isinstance(result[field_name], bool):
                raise EvidenceContractError(
                    f"{role} {field_name} must be Boolean at {key}"
                )
        solver_rule = result.get("solver_feasibility_rule")
        if solver_rule not in {
            "incumbent_required_and_finite_relative_mip_gap_le_0_05",
            # Read-only compatibility for pre-amendment exact-optimal fixtures.
            "timeout_or_nonzero_gap_or_nonfeasible_termination_is_infeasible",
        }:
            raise EvidenceContractError(f"{role} solver rule changed at {key}")
        first_stage = result.get("first_stage_solver_audit")
        recourse = result.get("recourse_solver_audit")
        if not isinstance(first_stage, Mapping) or not isinstance(recourse, Mapping):
            raise EvidenceContractError(f"{role} solver audit is absent at {key}")
        for field_name in (
            "mip_gap_reported",
            "time_limit_reached",
            "nonzero_gap",
            "termination_infeasible",
            "infeasible_under_frozen_rule",
        ):
            if not isinstance(first_stage.get(field_name), bool):
                raise EvidenceContractError(
                    f"{role} first-stage {field_name} must be Boolean at {key}"
                )
        first_gap_reported = first_stage["mip_gap_reported"]
        raw_first_gap = first_stage.get("mip_gap")
        if first_gap_reported:
            first_gap = _finite(
                raw_first_gap,
                label=f"{role} first-stage MIP gap at {key}",
                minimum=0.0,
            )
        else:
            if raw_first_gap is not None:
                raise EvidenceContractError(
                    f"{role} first-stage unreported MIP gap has a value at {key}"
                )
            first_gap = None
        first_timeout = first_stage["time_limit_reached"]
        first_nonzero = first_stage["nonzero_gap"]
        expected_first_nonzero = bool(first_gap is not None and first_gap > 0.0)
        if first_nonzero is not expected_first_nonzero:
            raise EvidenceContractError(
                f"{role} first-stage nonzero-gap flag contradicts MIP gap at {key}"
            )
        expected_first_over_cap = bool(
            first_gap is not None and first_gap > MAXIMUM_ACCEPTED_MIP_GAP
        )
        if first_stage.get("gap_over_cap", expected_first_over_cap) is not expected_first_over_cap:
            raise EvidenceContractError(
                f"{role} first-stage over-cap flag contradicts MIP gap at {key}"
            )
        first_has_incumbent = first_stage.get("has_incumbent")
        if first_has_incumbent is None:
            first_has_incumbent = not bool(first_stage["termination_infeasible"])
        if not isinstance(first_has_incumbent, bool):
            raise EvidenceContractError(
                f"{role} first-stage has_incumbent must be Boolean at {key}"
            )
        if solver_rule == (
            "timeout_or_nonzero_gap_or_nonfeasible_termination_is_infeasible"
        ):
            expected_first_infeasible = bool(
                not first_gap_reported
                or first_timeout
                or first_nonzero
                or first_stage["termination_infeasible"]
            )
        else:
            expected_first_infeasible = bool(
                not first_has_incumbent
                or not first_gap_reported
                or expected_first_over_cap
                or first_stage["termination_infeasible"]
            )
        first_infeasible = first_stage["infeasible_under_frozen_rule"]
        rejected_flag = first_stage.get("rejected_under_bounded_gap_rule")
        if first_infeasible is not expected_first_infeasible or (
            rejected_flag is not None
            and rejected_flag is not expected_first_infeasible
        ):
            raise EvidenceContractError(
                f"{role} first-stage infeasibility contradicts its audit at {key}"
            )
        for field_name in (
            "timeout_count",
            "nonzero_gap_count",
            "infeasible_under_frozen_rule_count",
            "mip_gap_missing_count",
        ):
            raw = recourse.get(field_name)
            if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
                raise EvidenceContractError(
                    f"{role} recourse {field_name} is invalid at {key}"
                )
        recourse_timeout = int(recourse["timeout_count"]) > 0
        recourse_nonzero = int(recourse["nonzero_gap_count"]) > 0
        recourse_over_cap_count = recourse.get("gap_over_cap_count", 0)
        recourse_no_incumbent_count = recourse.get("no_incumbent_count", 0)
        for field_name, raw in (
            ("gap_over_cap_count", recourse_over_cap_count),
            ("no_incumbent_count", recourse_no_incumbent_count),
        ):
            if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
                raise EvidenceContractError(
                    f"{role} recourse {field_name} is invalid at {key}"
                )
        recourse_over_cap = int(recourse_over_cap_count) > 0
        recourse_no_incumbent = int(recourse_no_incumbent_count) > 0
        recourse_missing = int(recourse["mip_gap_missing_count"]) > 0
        recourse_infeasible = (
            int(recourse["infeasible_under_frozen_rule_count"]) > 0
        )
        if int(recourse["infeasible_under_frozen_rule_count"]) < max(
            int(recourse_over_cap_count),
            int(recourse_no_incumbent_count),
            int(recourse["mip_gap_missing_count"]),
        ):
            raise EvidenceContractError(
                f"{role} recourse failure counts contradict infeasibility at {key}"
            )
        expected_timeout = first_timeout or recourse_timeout
        expected_nonzero = first_nonzero or recourse_nonzero
        expected_infeasible = (
            first_infeasible
            or not first_gap_reported
            or recourse_infeasible
            or recourse_over_cap
            or recourse_no_incumbent
            or recourse_missing
        )
        if result["solver_any_time_limit_reached"] is not expected_timeout:
            raise EvidenceContractError(
                f"{role} timeout aggregate contradicts solver audits at {key}"
            )
        if result["solver_any_nonzero_gap"] is not expected_nonzero:
            raise EvidenceContractError(
                f"{role} nonzero-gap aggregate contradicts solver audits at {key}"
            )
        if result.get("solver_any_gap_over_cap", expected_first_over_cap or recourse_over_cap) is not (
            expected_first_over_cap or recourse_over_cap
        ):
            raise EvidenceContractError(
                f"{role} over-cap aggregate contradicts solver audits at {key}"
            )
        if result.get("solver_any_missing_gap", not first_gap_reported or recourse_missing) is not (
            not first_gap_reported or recourse_missing
        ):
            raise EvidenceContractError(
                f"{role} missing-gap aggregate contradicts solver audits at {key}"
            )
        if result.get("solver_any_no_incumbent", not first_has_incumbent or recourse_no_incumbent) is not (
            not first_has_incumbent or recourse_no_incumbent
        ):
            raise EvidenceContractError(
                f"{role} no-incumbent aggregate contradicts solver audits at {key}"
            )
        if (
            result["solver_any_infeasible_under_frozen_rule"]
            is not expected_infeasible
            or result["infeasible_day"] is not expected_infeasible
        ):
            raise EvidenceContractError(
                f"{role} infeasibility aggregate contradicts solver audits at {key}"
            )
        _valid_sha256(
            result["source_evidence_sha256"],
            label=f"{role} source evidence at {key}",
        )

    @staticmethod
    def _perfect_information_solver_status(
        result: Mapping[str, Any],
        *,
        key: tuple[str, str, str, int],
    ) -> tuple[bool, str | None]:
        audit = result.get("solver_audit")
        if not isinstance(audit, Mapping):
            valid = False
            reason = "MISSING_PI_SOLVER_AUDIT"
        elif audit.get("mip_gap_reported") is not True or audit.get("mip_gap") is None:
            valid = False
            reason = "MISSING_PI_MIP_GAP"
        else:
            try:
                gap = float(audit["mip_gap"])
            except (TypeError, ValueError):
                gap = float("nan")
            if not np.isfinite(gap) or gap < 0.0:
                valid = False
                reason = "INVALID_PI_MIP_GAP"
            elif audit.get("nonzero_gap") is not (gap > 0.0):
                raise EvidenceContractError(
                    f"perfect-information gap flag contradicts MIP gap at {key}"
                )
            elif audit.get("has_incumbent", True) is not True:
                valid = False
                reason = "PI_SOLVER_NO_INCUMBENT"
            elif gap > MAXIMUM_ACCEPTED_MIP_GAP:
                valid = False
                reason = "PI_SOLVER_GAP_OVER_CAP"
            elif audit.get("termination_infeasible") is not False:
                valid = False
                reason = "PI_SOLVER_NONFEASIBLE_TERMINATION"
            elif audit.get("rejected_under_bounded_gap_rule", False) is not False:
                valid = False
                reason = "PI_SOLVER_REJECTED_UNDER_BOUNDED_GAP_RULE"
            else:
                valid = True
                reason = None
        declared = result.get("pi_metrics_valid")
        if not isinstance(declared, bool) or declared is not valid:
            raise EvidenceContractError(
                "perfect-information solver-validity flag contradicts its audit "
                f"at {key}"
            )
        declared_reason = result.get("pi_metrics_invalid_reason")
        if valid and declared_reason not in (None, ""):
            raise EvidenceContractError(
                f"valid perfect-information evidence has an invalid reason at {key}"
            )
        if not valid and str(declared_reason or "") != str(reason):
            raise EvidenceContractError(
                "perfect-information invalid reason contradicts its solver audit "
                f"at {key}"
            )
        return valid, reason

    def ingest_selection_evidence(self, evidence: DARTPhaseEvidence) -> dict[str, Any]:
        if evidence.phase != "selection":
            raise EvidenceContractError("selection ingestion requires selection evidence")
        validation = self._validate_phase_evidence(evidence)
        input_sha = evidence.fingerprint
        record, resumed = self._begin_stage(
            "selection_evidence", input_sha, previous=None
        )
        if resumed:
            return dict(record["details"])
        payload = {**evidence.to_dict(), "validation": validation}
        atomic_json(self.selection_evidence_path, payload)
        self._finish_stage(
            "selection_evidence",
            input_sha,
            (self.selection_evidence_path,),
            details=validation,
        )
        return validation

    def _read_evidence(self, path: Path) -> DARTPhaseEvidence:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence = DARTPhaseEvidence.from_dict(payload)
        if payload.get("validation", {}).get("evidence_fingerprint") != evidence.fingerprint:
            raise EvidenceContractError(f"evidence snapshot changed: {path}")
        return evidence

    @staticmethod
    def _operational_value_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
        if not rows:
            raise EvidenceContractError("operational value aggregation requires rows")

        def finite_cost(field_name: str) -> float:
            values = np.asarray(
                [row["decision_value"][field_name] for row in rows], dtype=float
            )
            if values.ndim != 1 or values.size != len(rows) or not np.all(np.isfinite(values)):
                raise EvidenceContractError(
                    f"{field_name} must be finite on the exact seed panel"
                )
            return float(np.mean(values))

        candidate_cost = finite_cost("candidate_simulated_total_cost_usd")
        baseline_cost = finite_cost("baseline_simulated_total_cost_usd")
        pi_cost = finite_cost("perfect_information_cost_lower_bound_usd")
        return -float(candidate_cost), -float(baseline_cost), -float(pi_cost)

    def _daily_and_decision_summaries(
        self, evidence: DARTPhaseEvidence
    ) -> tuple[
        dict[str, dict[str, list[float]]],
        dict[str, dict[str, list[float]]],
        dict[str, dict[str, dict[str, Any]]],
        dict[str, dict[str, dict[str, Any]]],
    ]:
        grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in evidence.replay_results:
            grouped[
                (
                    str(row["experiment_id"]),
                    str(row["candidate_id"]),
                    str(row["operating_date"]),
                )
            ].append(row)
        values: dict[str, dict[str, list[float]]] = {
            experiment: {candidate: [] for candidate in self.candidate_ids}
            for experiment in EXPERIMENT_IDS
        }
        losses: dict[str, dict[str, list[float]]] = {
            experiment: {candidate: [] for candidate in self.candidate_ids}
            for experiment in EXPERIMENT_IDS
        }
        summaries: dict[str, dict[str, dict[str, Any]]] = {
            experiment: {} for experiment in EXPERIMENT_IDS
        }
        daily_details: dict[str, dict[str, dict[str, Any]]] = {
            experiment: {} for experiment in EXPERIMENT_IDS
        }
        for experiment in EXPERIMENT_IDS:
            for candidate in self.candidate_ids:
                model_values: list[float] = []
                baseline_values: list[float] = []
                pi_values: list[float] = []
                feasible_energy: list[float] = []
                daily_sla: list[float] = []
                daily_deadline: list[float] = []
                daily_infeasible: list[bool] = []
                daily_invalid: list[bool] = []
                daily_solver_timeout: list[bool] = []
                daily_solver_nonzero_gap: list[bool] = []
                daily_solver_gap_over_cap: list[bool] = []
                daily_solver_missing_gap: list[bool] = []
                daily_solver_no_incumbent: list[bool] = []
                daily_latency: list[float] = []
                complexities: list[float] = []
                source_hashes: list[str] = []
                pi_invalid_days: list[str] = []
                pi_invalid_reasons: set[str] = set()
                for day in sorted(evidence.operating_dates):
                    records = grouped[(experiment, candidate, day)]
                    if len(records) != len(DART_VALUE_SELECT_SEEDS):
                        raise EvidenceContractError(
                            f"daily seed denominator changed for {(experiment, candidate, day)}"
                        )
                    model, baseline, supplied_perfect = self._operational_value_arrays(
                        records
                    )
                    pi_statuses = []
                    for row in records:
                        pi_key = (
                            str(row["operating_date"]),
                            str(row["candidate_id"]),
                            str(row["experiment_id"]),
                            int(row["seed"]),
                        )
                        pi_statuses.append(
                            self._perfect_information_solver_status(
                                row["perfect_information_result"], key=pi_key
                            )
                        )
                    pi_valid = all(status[0] for status in pi_statuses)
                    if not pi_valid:
                        pi_invalid_days.append(day)
                        pi_invalid_reasons.update(
                            str(status[1]) for status in pi_statuses if status[1]
                        )
                    # PI is not needed for incremental value.  Use a finite
                    # placeholder only to compute PI-independent summaries,
                    # then explicitly null every PI-dependent field below.
                    perfect = supplied_perfect if pi_valid else model
                    model_values.append(model)
                    baseline_values.append(baseline)
                    pi_values.append(perfect)
                    candidate_operations = [
                        record["candidate_result"] for record in records
                    ]
                    baseline_operations = [
                        record["baseline_result"] for record in records
                    ]
                    replay_operations = candidate_operations + baseline_operations
                    feasible_energy.append(
                        float(
                            np.mean(
                                [
                                    op["gate_time_feasible_flexible_energy_mwh"]
                                    for op in candidate_operations
                                ]
                            )
                        )
                    )
                    daily_sla.append(
                        float(
                            np.mean(
                                [
                                    op["weighted_sla_satisfaction"]
                                    for op in candidate_operations
                                ]
                            )
                        )
                    )
                    daily_deadline.append(
                        float(
                            np.mean(
                                [
                                    op["deadline_violations"]
                                    for op in candidate_operations
                                ]
                            )
                        )
                    )
                    daily_infeasible.append(
                        any(op["infeasible_day"] for op in replay_operations)
                    )
                    daily_invalid.append(
                        any(op["invalid_day"] for op in replay_operations)
                        or not pi_valid
                    )
                    daily_solver_timeout.append(
                        any(
                            op["solver_any_time_limit_reached"]
                            for op in replay_operations
                        )
                    )
                    daily_solver_nonzero_gap.append(
                        any(op["solver_any_nonzero_gap"] for op in replay_operations)
                    )
                    daily_solver_gap_over_cap.append(
                        any(
                            bool(op.get("solver_any_gap_over_cap", False))
                            for op in replay_operations
                        )
                    )
                    daily_solver_missing_gap.append(
                        any(
                            bool(op.get("solver_any_missing_gap", False))
                            for op in replay_operations
                        )
                    )
                    daily_solver_no_incumbent.append(
                        any(
                            bool(op.get("solver_any_no_incumbent", False))
                            for op in replay_operations
                        )
                    )
                    daily_latency.append(
                        max(
                            float(op["forecast_time_seconds"]) + float(op["solve_time_seconds"])
                            for op in replay_operations
                        )
                    )
                    complexities.extend(
                        float(op["model_complexity"])
                        for op in candidate_operations
                    )
                    source_hashes.extend(
                        str(op["source_evidence_sha256"])
                        for op in replay_operations
                    )
                    daily_details[experiment].setdefault(candidate, {})[day] = {
                        "model_value": model,
                        "baseline_value": baseline,
                        "perfect_information_value": (
                            supplied_perfect if pi_valid else None
                        ),
                        "incremental_value": model - baseline,
                        "prescriptive_regret": (
                            supplied_perfect - model if pi_valid else None
                        ),
                        "perfect_information_metric_status": (
                            "OK" if pi_valid else "INVALID_PI_SOLVER_EVIDENCE"
                        ),
                        "perfect_information_invalid_reason": (
                            None
                            if pi_valid
                            else ";".join(
                                sorted(
                                    str(status[1])
                                    for status in pi_statuses
                                    if status[1]
                                )
                            )
                        ),
                        "feasible_flexible_energy_mwh": feasible_energy[-1],
                    }
                values[experiment][candidate] = [
                    model - baseline
                    for model, baseline in zip(model_values, baseline_values, strict=True)
                ]
                losses[experiment][candidate] = [-model for model in model_values]
                summary = decision_value_summary(
                    model_values,
                    baseline_values,
                    pi_values,
                    risk_alpha=float(self.config["decision_value"]["risk_alpha"]),
                    feasible_flexible_energy_mwh=feasible_energy,
                )
                if pi_invalid_days:
                    pi_reason = ";".join(sorted(pi_invalid_reasons))
                    summary.update(
                        {
                            "mean_prescriptive_regret": None,
                            "cumulative_prescriptive_regret": None,
                            "profit_capture_ratio": {
                                "value": None,
                                "status": "INVALID_PI_SOLVER_EVIDENCE",
                                "undefined_reason": pi_reason,
                                "numerator": None,
                                "denominator": None,
                                "count": 0,
                            },
                            "perfect_information_metric_status": (
                                "INVALID_PI_SOLVER_EVIDENCE"
                            ),
                            "perfect_information_invalid_reason": pi_reason,
                            "perfect_information_invalid_day_count": len(
                                pi_invalid_days
                            ),
                            "perfect_information_invalid_operating_dates": list(
                                pi_invalid_days
                            ),
                        }
                    )
                else:
                    summary.update(
                        {
                            "perfect_information_metric_status": "OK",
                            "perfect_information_invalid_reason": None,
                            "perfect_information_invalid_day_count": 0,
                            "perfect_information_invalid_operating_dates": [],
                        }
                    )
                baseline_cost_total = float(np.sum(np.negative(baseline_values)))
                summary.update(
                    {
                        "weighted_sla_satisfaction": float(np.mean(daily_sla)),
                        "deadline_violations": float(np.sum(daily_deadline)),
                        "infeasible_days": int(np.count_nonzero(daily_infeasible)),
                        "invalid_days": int(np.count_nonzero(daily_invalid)),
                        "solver_timeout_days": int(
                            np.count_nonzero(daily_solver_timeout)
                        ),
                        "solver_nonzero_gap_days": int(
                            np.count_nonzero(daily_solver_nonzero_gap)
                        ),
                        "solver_gap_over_cap_days": int(
                            np.count_nonzero(daily_solver_gap_over_cap)
                        ),
                        "solver_missing_gap_days": int(
                            np.count_nonzero(daily_solver_missing_gap)
                        ),
                        "solver_no_incumbent_days": int(
                            np.count_nonzero(daily_solver_no_incumbent)
                        ),
                        "all_days_valid": not any(daily_invalid),
                        "forecast_plus_solve_latency_seconds": max(daily_latency),
                        "model_complexity": float(np.max(complexities)),
                        "baseline_operating_cost": float(
                            np.mean(np.negative(baseline_values))
                        ),
                        "equivalent_operating_cost_reduction": summary.get(
                            "cumulative_incremental_operational_value"
                        ),
                        "operating_cost_reduction_rate": (
                            summary["cumulative_incremental_operational_value"]
                            / baseline_cost_total
                            if baseline_cost_total > 0.0
                            else None
                        ),
                        "cvar95_loss": summary.get("cvar_loss"),
                        "source_record_count": len(source_hashes),
                        "source_records_sha256": sha256_json(sorted(source_hashes)),
                        "experiment_id": experiment,
                        "candidate_id": candidate,
                    }
                )
                summaries[experiment][candidate] = summary
        return values, losses, summaries, daily_details

    @staticmethod
    def _aggregate_ratio_records(
        rows: Sequence[Mapping[str, Any]],
        field_name: str,
    ) -> dict[str, Any]:
        """Average three-seed sufficient stats within day, then pool days."""

        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["operating_date"])].append(row)
        day_rows: list[dict[str, Any]] = []
        numerator = 0.0
        denominator = 0.0
        count = 0.0
        reasons: set[str] = set()
        for operating_date in sorted(grouped):
            day_records = grouped[operating_date]
            if len(day_records) != len(DART_VALUE_SELECT_SEEDS) or {
                int(row["seed"]) for row in day_records
            } != set(DART_VALUE_SELECT_SEEDS):
                raise EvidenceContractError(
                    f"{field_name} lacks the exact three-seed panel on {operating_date}"
                )
            statistics = [
                row["metric_sufficient_statistics"][field_name]
                for row in day_records
            ]
            day_numerator = float(
                np.mean([float(item["numerator"]) for item in statistics])
            )
            day_denominator = float(
                np.mean([float(item["denominator"]) for item in statistics])
            )
            day_count = float(
                np.mean([float(item.get("count", 0.0)) for item in statistics])
            )
            day_reasons = sorted(
                {
                    str(item.get("undefined_reason"))
                    for item in statistics
                    if item.get("undefined_reason")
                }
            )
            reasons.update(day_reasons)
            numerator += day_numerator
            denominator += day_denominator
            count += day_count
            day_rows.append(
                {
                    "operating_date": operating_date,
                    "seed_count": len(day_records),
                    "seed_average_numerator": day_numerator,
                    "seed_average_denominator": day_denominator,
                    "seed_average_count": day_count,
                    "undefined_reasons": day_reasons,
                }
            )
        no_positive_spikes = field_name == "spike_f1" and count <= 0.0
        if denominator <= 0.0 or no_positive_spikes:
            reason = ";".join(sorted(reasons)) or (
                "NO_POSITIVE_SPIKES"
                if no_positive_spikes
                else "ZERO_AGGREGATE_DENOMINATOR"
            )
            value: float | None = None
            status = reason
        else:
            value = float(numerator / denominator)
            status = "OK"
            reason = None
        return {
            "value": value,
            "status": status,
            "undefined_reason": reason,
            "numerator": numerator,
            "denominator": denominator,
            "count": int(round(count)),
            "aggregation_order": (
                "average_sufficient_statistics_across_exact_three_seeds_within_date;"
                "sum_across_common_dates;divide_once"
            ),
            "per_day_seed_averaged_sufficient_statistics": day_rows,
        }

    @staticmethod
    def _metric_contract_summary(
        rows: Sequence[Mapping[str, Any]],
        *,
        candidate: str,
    ) -> dict[str, Any]:
        """Preserve date-scoped v3 metric contracts without inventing one hash."""

        ordered = sorted(
            rows,
            key=lambda row: (str(row["operating_date"]), int(row["seed"])),
        )
        per_record_hash_ledger = [
            {
                "candidate_id": candidate,
                "operating_date": str(row["operating_date"]),
                "seed": int(row["seed"]),
                **{
                    field_name: str(row[field_name])
                    for field_name in REQUIRED_METRIC_HASH_FIELDS
                },
            }
            for row in ordered
        ]
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in ordered:
            grouped[str(row["operating_date"])].append(row)
        per_date_contract: list[dict[str, Any]] = []
        for operating_date in sorted(grouped):
            date_rows = grouped[operating_date]
            per_date_contract.append(
                {
                    "operating_date": operating_date,
                    "seed_count": len(date_rows),
                    **{
                        field_name: str(date_rows[0][field_name])
                        for field_name in REQUIRED_METRIC_HASH_FIELDS
                    },
                    **{
                        field_name: float(
                            np.mean([float(row[field_name]) for row in date_rows])
                        )
                        for field_name in REQUIRED_METRIC_CONTRACT_NUMERIC_FIELDS
                    },
                }
            )
        summary: dict[str, Any] = {
            "metric_contract_operating_date_count": len(per_date_contract),
            "metric_contract_by_operating_date_sha256": sha256_json(per_date_contract),
            "metric_contract_per_record_hash_ledger_sha256": sha256_json(
                per_record_hash_ledger
            ),
            "metric_contract_aggregation_order": (
                "validate_candidate_independence_within_date;average_exact_three_"
                "seed_numeric_contract_within_date;preserve_cross_date_hash_ledger"
            ),
        }
        for field_name in REQUIRED_METRIC_HASH_FIELDS:
            unique_values = sorted(
                {str(row[field_name]) for row in per_date_contract}
            )
            field_ledger = [
                {
                    "candidate_id": row["candidate_id"],
                    "operating_date": row["operating_date"],
                    "seed": row["seed"],
                    field_name: row[field_name],
                }
                for row in per_record_hash_ledger
            ]
            summary[field_name] = (
                unique_values[0] if len(unique_values) == 1 else None
            )
            summary[f"{field_name}_status"] = (
                "COMMON_ACROSS_ALL_RECORDS"
                if len(unique_values) == 1
                else "VARIES_BY_OPERATING_DATE"
            )
            summary[f"{field_name}_unique_value_count"] = len(unique_values)
            summary[f"{field_name}_per_record_ledger_sha256"] = sha256_json(
                field_ledger
            )
        summary["normalized_feasibility_weight_min"] = min(
            float(row["normalized_feasibility_weight_min"])
            for row in per_date_contract
        )
        summary["normalized_feasibility_weight_max"] = max(
            float(row["normalized_feasibility_weight_max"])
            for row in per_date_contract
        )
        summary["normalized_feasibility_weight_sum"] = float(
            np.sum(
                [
                    float(row["normalized_feasibility_weight_sum"])
                    for row in per_date_contract
                ]
            )
        )
        summary["physical_flexible_mwh_total"] = float(
            np.sum(
                [float(row["physical_flexible_mwh_total"]) for row in per_date_contract]
            )
        )
        summary["normalized_feasibility_weight_range_aggregation"] = (
            "min_and_max_across_date_level_exact_three_seed_means"
        )
        summary["normalized_feasibility_weight_sum_aggregation"] = (
            "sum_across_dates_after_exact_three_seed_mean_within_date"
        )
        summary["physical_flexible_mwh_total_aggregation"] = (
            "sum_across_dates_after_exact_three_seed_mean_within_date"
        )
        return summary

    def _metric_summaries(
        self, evidence: DARTPhaseEvidence
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in evidence.metric_records:
            grouped[str(row["candidate_id"])].append(row)
        summaries: dict[str, dict[str, Any]] = {}
        for candidate in self.candidate_ids:
            rows = grouped[candidate]
            summary: dict[str, Any] = {}
            for field_name in REQUIRED_METRIC_FIELDS:
                summary[field_name] = _mean([row[field_name] for row in rows])
            for field_name in NULLABLE_METRIC_FIELDS:
                if field_name in POOLED_Q99_METRIC_FIELDS:
                    first = rows[0]
                    summary[field_name] = first.get(field_name)
                    summary[f"{field_name}_status"] = first.get(
                        f"{field_name}_status"
                    )
                    summary[f"{field_name}_undefined_reason"] = first.get(
                        f"{field_name}_undefined_reason"
                    )
                    summary[f"{field_name}_aggregation"] = {
                        "aggregation_scope": POOLED_Q99_AGGREGATION_SCOPE,
                        "selector_operation": (
                            "validate_unique_producer_pooled_value_no_reaggregation"
                        ),
                        "source_record_count": len(rows),
                    }
                    continue
                if field_name in RATIO_OF_SUMS_METRIC_FIELDS:
                    aggregate = self._aggregate_ratio_records(rows, field_name)
                    summary[field_name] = aggregate["value"]
                    summary[f"{field_name}_status"] = aggregate["status"]
                    summary[f"{field_name}_undefined_reason"] = aggregate[
                        "undefined_reason"
                    ]
                    summary[f"{field_name}_aggregation"] = aggregate
                    continue
                null_rows = [row for row in rows if row.get(field_name) is None]
                if null_rows:
                    statuses = sorted(
                        {
                            str(row.get(f"{field_name}_status"))
                            for row in null_rows
                            if row.get(f"{field_name}_status")
                        }
                    )
                    summary[field_name] = None
                    summary[f"{field_name}_status"] = (
                        ";".join(statuses) or "UNAVAILABLE_ON_COMMON_PANEL"
                    )
                    summary[f"{field_name}_undefined_reason"] = summary[
                        f"{field_name}_status"
                    ]
                else:
                    summary[field_name] = _mean([row[field_name] for row in rows])
                    summary[f"{field_name}_status"] = "OK"
                    summary[f"{field_name}_undefined_reason"] = None
            summary.update(self._metric_contract_summary(rows, candidate=candidate))
            hashes = sorted(str(row["source_evidence_sha256"]) for row in rows)
            summary.update(
                {
                    "candidate_id": candidate,
                    "source_record_count": len(rows),
                    "source_records_sha256": sha256_json(hashes),
                }
            )
            summaries[candidate] = summary
        return summaries

    def _criteria(self) -> FeasibilityCriteria:
        config = self.config["feasibility"]
        return FeasibilityCriteria(
            minimum_sla_satisfaction=float(config["minimum_sla_satisfaction"]),
            cvar95_loss_budget_fraction_of_baseline_cost=float(
                config["cvar95_loss_budget_fraction_of_baseline_cost"]
            ),
            maximum_probability_below_baseline=float(
                config["maximum_probability_below_baseline"]
            ),
            market_gate_budget_seconds=float(config["market_gate_budget_seconds"]),
            require_zero_infeasible_days=bool(config["require_zero_infeasible_days"]),
            require_zero_invalid_days=bool(config["require_zero_invalid_days"]),
            maximum_solver_relative_mip_gap=float(
                config.get("maximum_solver_relative_mip_gap", 0.05)
            ),
            require_zero_solver_gap_over_cap_days=bool(
                config.get("require_zero_solver_gap_over_cap_days", True)
            ),
            require_zero_solver_missing_gap_days=bool(
                config.get("require_zero_solver_missing_gap_days", True)
            ),
            require_zero_solver_no_incumbent_days=bool(
                config.get("require_zero_solver_no_incumbent_days", True)
            ),
            solver_timeout_day_field=str(config["solver_timeout_day_field"]),
            solver_nonzero_gap_day_field=str(
                config["solver_nonzero_gap_day_field"]
            ),
            solver_gap_over_cap_day_field=str(
                config.get("solver_gap_over_cap_day_field", "solver_gap_over_cap_days")
            ),
            solver_missing_gap_day_field=str(
                config.get("solver_missing_gap_day_field", "solver_missing_gap_days")
            ),
            solver_no_incumbent_day_field=str(
                config.get("solver_no_incumbent_day_field", "solver_no_incumbent_days")
            ),
            allow_legacy_solver_field_aliases=False,
        )

    def _a1_noninferiority_gate(
        self,
        a1_result: Mapping[str, Any],
        a1_decisions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        primary_block = int(self.config["bootstrap"]["primary_block_days"])
        robustness = a1_result.get("robustness_lcb_by_block", {})
        lcb_by_model = (
            robustness.get(str(primary_block), {})
            if isinstance(robustness, Mapping)
            else {}
        )
        if not isinstance(lcb_by_model, Mapping):
            raise EvidenceContractError("A1 noninferiority LCB map is absent")
        baseline_costs = {
            model: _finite(
                a1_decisions[model].get("baseline_operating_cost"),
                label=f"A1 mean baseline daily cost for {model}",
                minimum=0.0,
            )
            for model in self.candidate_ids
        }
        if any(value <= 0.0 for value in baseline_costs.values()):
            raise EvidenceContractError(
                "A1 noninferiority requires positive mean baseline daily cost"
            )
        mean_baseline = float(np.mean(list(baseline_costs.values())))
        if not all(
            np.isclose(value, mean_baseline, rtol=1e-8, atol=1e-8)
            for value in baseline_costs.values()
        ):
            raise EvidenceContractError(
                "A1 noninferiority requires a candidate-independent baseline cost"
            )
        epsilon = A1_NONINFERIORITY_FRACTION * mean_baseline
        physically_feasible = set(map(str, a1_result.get("feasible_models", [])))
        eligible: list[str] = []
        excluded: dict[str, list[str]] = {}
        normalized_lcbs: dict[str, float | None] = {}
        for model in self.candidate_ids:
            raw = lcb_by_model.get(model)
            lcb = None if raw is None else _finite(raw, label=f"A1 LCB for {model}")
            normalized_lcbs[model] = lcb
            reasons: list[str] = []
            if model not in physically_feasible:
                reasons.append("A1_FEASIBILITY_GATE_FAILED")
            if lcb is None:
                reasons.append("A1_7_DAY_LCB_UNAVAILABLE")
            elif lcb < -epsilon:
                reasons.append("A1_NONINFERIORITY_LCB_BELOW_MINUS_EPSILON")
            if reasons:
                excluded[model] = reasons
            else:
                eligible.append(model)
        gate = {
            "gate_id": "A1_DART_NONINFERIORITY",
            "status": "PASSED" if eligible else "NO_ELIGIBLE_MODEL",
            "source_experiment_id": "A1",
            "target_experiment_id": "A2",
            "primary_block_days": primary_block,
            "epsilon_fraction_of_mean_baseline_daily_cost": (
                A1_NONINFERIORITY_FRACTION
            ),
            "mean_baseline_daily_cost_usd": mean_baseline,
            "epsilon_usd": epsilon,
            "pass_rule": "A1_7_day_LCB_greater_than_or_equal_to_minus_epsilon",
            "lcb_by_model": normalized_lcbs,
            "eligible_models": sorted(eligible),
            "excluded_by_model": excluded,
            "within_date_seed_aggregation": (
                "mean_across_exact_three_common_seeds_before_date_block_inference"
            ),
            "selection_updated_from_2025": False,
        }
        gate["gate_sha256"] = sha256_json(gate)
        return gate

    def _derive_a3(
        self,
        evidence: DARTPhaseEvidence,
    ) -> dict[str, Any]:
        metrics = {
            (
                str(row["operating_date"]),
                str(row["candidate_id"]),
                int(row["seed"]),
            ): row
            for row in evidence.metric_records
        }
        augmented: list[dict[str, Any]] = []
        source_hashes: list[str] = []
        missing_q99_tail_records: list[dict[str, Any]] = []
        for record in evidence.replay_results:
            row = dict(record)
            candidate = dict(row["candidate_result"])
            key = (
                str(row["operating_date"]),
                str(row["candidate_id"]),
                int(row["seed"]),
            )
            metric = metrics[key]
            accuracy_metrics = {
                "mae": float(metric["mae"]),
                "crps": float(metric["crps"]),
            }
            q99_tail_f1 = metric.get("q99_f1")
            if q99_tail_f1 is None:
                missing_q99_tail_records.append(
                    {
                        "operating_date": key[0],
                        "candidate_id": key[1],
                        "seed": key[2],
                        "reason": metric.get("q99_f1_undefined_reason")
                        or metric.get("q99_f1_status")
                        or "Q99_TAIL_F1_UNAVAILABLE",
                    }
                )
            else:
                value = _finite(
                    q99_tail_f1,
                    label=f"q99-tail-F1 for {key}",
                    minimum=0.0,
                )
                if value > 1.0:
                    raise EvidenceContractError("q99-tail-F1 must not exceed one")
                # The shared A3 helper ranks lower scores as better.  Ranking
                # loss preserves the correct higher-is-better F1 ordering;
                # the public result is converted back to F1 below.
                accuracy_metrics["q99_tail_f1_ranking_loss"] = 1.0 - value
            candidate["accuracy_metrics"] = accuracy_metrics
            row["candidate_result"] = candidate
            augmented.append(row)
            source_hashes.append(sha256_json(record))
        def q99_tail_result(source_experiment: str) -> dict[str, Any]:
            if missing_q99_tail_records:
                return {
                    "schema_version": "dart_metric_contradiction_a3_v2",
                    "experiment_id": "A3",
                    "source_experiment_id": source_experiment,
                    "accuracy_metric": "q99_tail_f1",
                    "accuracy_orientation": "higher_is_better",
                    "status": "NOT_APPLICABLE_NULL_COMMON_PANEL_METRIC",
                    "derived_without_solves": True,
                    "candidate_count": len(self.candidate_ids),
                    "operating_date_count": len(evidence.operating_dates),
                    "any_aggregate_rank_contradiction": False,
                    "any_date_level_contradiction": False,
                    "any_rank_contradiction": False,
                    "rows": [],
                    "date_findings": [],
                    "date_candidate_rows": [],
                    "missing_record_count": len(missing_q99_tail_records),
                    "missing_records_sha256": sha256_json(missing_q99_tail_records),
                    "null_finding_retained": True,
                }
            raw = derive_a3_metric_contradictions(
                augmented,
                accuracy_metric="q99_tail_f1_ranking_loss",
                experiment_id=source_experiment,
                regime_labels_by_date=evidence.regime_labels_by_date,
            )
            result = deepcopy(raw)
            result["accuracy_metric"] = "q99_tail_f1"
            result["accuracy_orientation"] = "higher_is_better"
            result["ranking_transform"] = "one_minus_q99_tail_f1"
            for row in result.get("rows", []):
                loss = float(row.pop("mean_q99_tail_f1_ranking_loss"))
                row["mean_q99_tail_f1"] = 1.0 - loss
            for finding in result.get("date_findings", []):
                finding["accuracy_metric"] = "q99_tail_f1"
                finding["best_accuracy_winner_set"] = finding.pop(
                    "lowest_accuracy_winner_set"
                )
            for row in result.get("date_candidate_rows", []):
                row["accuracy_metric"] = "q99_tail_f1"
                row["mean_accuracy"] = 1.0 - float(row["mean_accuracy"])
                row["is_best_accuracy_member"] = row.pop(
                    "is_lowest_accuracy_member"
                )
            return result

        by_source_experiment: dict[str, dict[str, Any]] = {}
        for source_experiment in EXPERIMENT_IDS:
            by_source_experiment[source_experiment] = {
                metric_name: derive_a3_metric_contradictions(
                    augmented,
                    accuracy_metric=metric_name,
                    experiment_id=source_experiment,
                    regime_labels_by_date=evidence.regime_labels_by_date,
                )
                for metric_name in ("mae", "crps")
            }
            by_source_experiment[source_experiment]["q99_tail_f1"] = (
                q99_tail_result(source_experiment)
            )
        # Keep top-level aliases while making the full-deployment A2 chain the
        # primary ranking comparison.
        # adapter while exposing the complete operating-date/candidate/
        # experiment grain under ``by_source_experiment``.
        mae = by_source_experiment["A2"]["mae"]
        crps = by_source_experiment["A2"]["crps"]
        q99_tail_f1 = by_source_experiment["A2"]["q99_tail_f1"]
        any_metric_contradiction = any(
            bool(result.get("any_rank_contradiction"))
            for experiment_results in by_source_experiment.values()
            for result in experiment_results.values()
        )
        return {
            "experiment_id": "A3",
            "derived_without_new_solves": True,
            "source_replay_result_count": len(augmented),
            "source_replay_results_sha256": sha256_json(sorted(source_hashes)),
            "mae": mae,
            "crps": crps,
            "q99_tail_f1": q99_tail_f1,
            "by_source_experiment": by_source_experiment,
            "source_experiments": list(EXPERIMENT_IDS),
            "primary_source_experiment": "A2",
            "any_metric_contradiction": bool(any_metric_contradiction),
            "date_level_required_grain": "operating_date_candidate_experiment",
            "explicit_null_findings_retained": True,
        }

    def _h4a(
        self,
        metrics: Mapping[str, Mapping[str, Any]],
        decision: Mapping[str, Mapping[str, Any]],
        *,
        phase: str,
    ) -> dict[str, Any]:
        contract = self.config["hypothesis_contract"][
            "H4a_material_rank_disagreement"
        ]
        decision_value = {
            model: float(decision[model]["mean_daily_incremental_operational_value"])
            for model in self.candidate_ids
        }
        ranking_specs = {
            "mae": False,
            "crps": False,
            "q99_f1": True,
        }
        decision_ranks = _average_ranks(decision_value, higher_is_better=True)
        decision_top_set = _top_rank_set(decision_ranks)
        baseline_costs = {
            model: float(decision[model]["baseline_operating_cost"])
            for model in self.candidate_ids
        }
        positive_baseline = all(value > 0.0 for value in baseline_costs.values())
        common_baseline = float(np.mean(list(baseline_costs.values())))
        baseline_consistent = all(
            np.isclose(value, common_baseline, rtol=1e-8, atol=1e-8)
            for value in baseline_costs.values()
        )
        evaluations: dict[str, Any] = {}
        materially_different: list[str] = []
        for field_name, higher_is_better in ranking_specs.items():
            raw_values = {model: metrics[model].get(field_name) for model in self.candidate_ids}
            missing = sorted(model for model, value in raw_values.items() if value is None)
            if missing:
                evaluations[field_name] = {
                    "applicable": False,
                    "status": "NOT_APPLICABLE_NULL_COMMON_PANEL_METRIC",
                    "missing_models": missing,
                    "missing_reasons": {
                        model: metrics[model].get(f"{field_name}_undefined_reason")
                        or metrics[model].get(f"{field_name}_status")
                        for model in missing
                    },
                    "materially_different": False,
                }
                continue
            values = {model: float(raw_values[model]) for model in self.candidate_ids}
            ranks = _average_ranks(values, higher_is_better=higher_is_better)
            traditional_top_set = _top_rank_set(ranks)
            distance, discordant, total_pairs = _normalized_kendall_distance(
                ranks, decision_ranks
            )
            top_choice_disagreement = set(traditional_top_set).isdisjoint(
                decision_top_set
            )
            cross_set_gaps = [
                abs(decision_value[decision_model] - decision_value[traditional_model])
                for decision_model in decision_top_set
                for traditional_model in traditional_top_set
            ]
            absolute_gap = min(cross_set_gaps) if cross_set_gaps else 0.0
            value_gap_fraction = (
                absolute_gap / common_baseline
                if positive_baseline and baseline_consistent
                else None
            )
            conditions = {
                "top_choice_disagreement": top_choice_disagreement,
                "normalized_kendall_distance_at_least_minimum": bool(
                    distance
                    >= float(contract["minimum_normalized_kendall_distance"])
                ),
                "counterfactual_value_gap_fraction_at_least_minimum": bool(
                    value_gap_fraction is not None
                    and value_gap_fraction
                    >= float(contract["minimum_counterfactual_value_gap_fraction"])
                ),
            }
            material = all(conditions.values())
            if material:
                materially_different.append(field_name)
            evaluations[field_name] = {
                "applicable": True,
                "metric_values": values,
                "rank_map": ranks,
                "rank_order": sorted(ranks, key=lambda model: (ranks[model], model)),
                "traditional_top_set": list(traditional_top_set),
                "decision_value_top_set": list(decision_top_set),
                "top_choice_disagreement_is_set_disjoint": top_choice_disagreement,
                "discordant_candidate_pairs": discordant,
                "total_candidate_pairs": total_pairs,
                "normalized_kendall_distance": distance,
                "minimum_normalized_kendall_distance": float(
                    contract["minimum_normalized_kendall_distance"]
                ),
                "absolute_mean_daily_value_gap": absolute_gap,
                "positive_baseline_operating_cost": (
                    common_baseline
                    if positive_baseline and baseline_consistent
                    else None
                ),
                "counterfactual_value_gap_fraction": value_gap_fraction,
                "minimum_counterfactual_value_gap_fraction": float(
                    contract["minimum_counterfactual_value_gap_fraction"]
                ),
                "conditions": conditions,
                "materially_different": material,
            }
        return {
            "hypothesis": "H4a",
            "phase": phase,
            "status": "SUPPORTED" if materially_different else "NOT_SUPPORTED",
            "evaluation": "frozen_material_rank_disagreement",
            "support_aggregation": (
                "at_least_one_applicable_traditional_metric_meets_all_three_"
                "frozen_materiality_conditions"
            ),
            "all_three_conditions_required_per_metric": bool(
                contract["all_three_conditions_required"]
            ),
            "decision_value_rank_map": decision_ranks,
            "decision_value_top_set": list(decision_top_set),
            "baseline_costs_by_model": baseline_costs,
            "baseline_operating_cost_positive": positive_baseline,
            "baseline_operating_cost_common": baseline_consistent,
            "metric_evaluations": evaluations,
            "materially_different_rankings": materially_different,
            "statistical_significance_claimed": False,
        }

    def _h4c_candidate_day_rows(
        self,
        evidence: DARTPhaseEvidence,
        daily: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        contract = self.config["hypothesis_contract"][
            "H4c_incremental_explanatory_test"
        ]
        features = tuple(contract["point_error_features"]) + tuple(
            contract["incremental_features"]
        )
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in evidence.metric_records:
            grouped[(str(row["operating_date"]), str(row["candidate_id"]))].append(row)
        output: list[dict[str, Any]] = []
        for operating_date in sorted(evidence.operating_dates):
            for candidate in self.candidate_ids:
                records = grouped[(operating_date, candidate)]
                if len(records) != len(DART_VALUE_SELECT_SEEDS) or {
                    int(row["seed"]) for row in records
                } != set(DART_VALUE_SELECT_SEEDS):
                    raise EvidenceContractError(
                        f"H4c candidate-day lacks three seeds: {(operating_date, candidate)}"
                    )
                values: dict[str, float | None] = {}
                null_reasons: dict[str, str] = {}
                for field_name in features:
                    if field_name in RATIO_OF_SUMS_METRIC_FIELDS:
                        statistics = [
                            row["metric_sufficient_statistics"][field_name]
                            for row in records
                        ]
                        numerator = float(
                            np.mean([float(item["numerator"]) for item in statistics])
                        )
                        denominator = float(
                            np.mean([float(item["denominator"]) for item in statistics])
                        )
                        if denominator <= 0.0:
                            values[field_name] = None
                            null_reasons[field_name] = ";".join(
                                sorted(
                                    {
                                        str(item.get("undefined_reason"))
                                        for item in statistics
                                        if item.get("undefined_reason")
                                    }
                                )
                            ) or "ZERO_DAILY_AGGREGATE_DENOMINATOR"
                        else:
                            values[field_name] = numerator / denominator
                    else:
                        raw = [row.get(field_name) for row in records]
                        if any(value is None for value in raw):
                            values[field_name] = None
                            null_reasons[field_name] = ";".join(
                                sorted(
                                    {
                                        str(row.get(f"{field_name}_status"))
                                        for row in records
                                        if row.get(field_name) is None
                                        and row.get(f"{field_name}_status")
                                    }
                                )
                            ) or "NULL_ON_ONE_OR_MORE_STUDY_SEEDS"
                        else:
                            values[field_name] = float(np.mean(np.asarray(raw, dtype=float)))
                output.append(
                    {
                        "operating_date": operating_date,
                        "candidate_id": candidate,
                        **values,
                        "response": float(daily[candidate][operating_date]["incremental_value"]),
                        "applicable": not null_reasons,
                        "null_reasons": null_reasons,
                        "seed_count": len(records),
                    }
                )
        return output

    def _h4c(
        self,
        evidence: DARTPhaseEvidence,
        daily: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        contract = self.config["hypothesis_contract"][
            "H4c_incremental_explanatory_test"
        ]
        rows = self._h4c_candidate_day_rows(evidence, daily)
        point_fields = tuple(map(str, contract["point_error_features"]))
        incremental_fields = tuple(map(str, contract["incremental_features"]))
        all_fields = point_fields + incremental_fields
        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_date[str(row["operating_date"])].append(row)
        excluded_rows = [
            {
                "operating_date": row["operating_date"],
                "candidate_id": row["candidate_id"],
                "null_reasons": row["null_reasons"],
            }
            for row in rows
            if not row["applicable"]
        ]
        applicable_dates = tuple(
            day
            for day in sorted(by_date)
            if len(by_date[day]) == len(self.candidate_ids)
            and all(row["applicable"] for row in by_date[day])
        )
        excluded_dates = tuple(sorted(set(by_date).difference(applicable_dates)))
        base_result: dict[str, Any] = {
            "hypothesis": "H4c",
            "phase": "selection_2022_2024",
            "evaluation": "rolling_forward_blocked_incremental_explanatory_test",
            "unit": "selection_oos_candidate_day",
            "response": contract["response"],
            "point_error_features": list(point_fields),
            "incremental_features": list(incremental_fields),
            "no_imputation": True,
            "exact_candidate_pool_per_applicable_date": True,
            "total_operating_date_count": len(by_date),
            "applicable_operating_date_count": len(applicable_dates),
            "excluded_operating_dates": list(excluded_dates),
            "excluded_candidate_day_count": len(excluded_rows),
            "excluded_candidate_days": excluded_rows,
            "candidate_day_denominator_audit_sha256": sha256_json(rows),
        }
        folds = int(contract["folds"])
        repeats = int(contract["repeats"])
        repeat_seeds = tuple(map(int, contract["repeat_seeds"]))
        if repeats != len(repeat_seeds):
            raise DARTExperimentContractError("H4c repeat count and seeds differ")
        if len(applicable_dates) < folds + 1:
            return {
                **base_result,
                "status": "NOT_SUPPORTED",
                "support": False,
                "reason": "INSUFFICIENT_COMMON_APPLICABLE_DATES_FOR_FROZEN_FOLDS",
                "minimum_required_operating_dates": folds + 1,
            }
        blocks = [tuple(map(str, block.tolist())) for block in np.array_split(
            np.asarray(applicable_dates, dtype=object), folds + 1
        )]
        if any(not block for block in blocks):
            return {
                **base_result,
                "status": "NOT_SUPPORTED",
                "support": False,
                "reason": "EMPTY_ROLLING_FORWARD_BLOCK",
            }
        squared_errors: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        fold_records: list[dict[str, Any]] = []
        for repeat_index, repeat_seed in enumerate(repeat_seeds):
            for fold_index in range(1, folds + 1):
                training_dates = tuple(
                    day for block in blocks[:fold_index] for day in block
                )
                test_dates = blocks[fold_index]
                seed_digest = hashlib.sha256(
                    f"H4c|{repeat_seed}|fold={fold_index}".encode("utf-8")
                ).digest()
                training_indices = moving_block_indices(
                    len(training_dates),
                    min(int(contract["block_days"]), len(training_dates)),
                    resamples=1,
                    seed=int.from_bytes(seed_digest[:8], "big", signed=False),
                    circular=True,
                )[0]
                sampled_training_dates = [training_dates[index] for index in training_indices]
                training_rows = [
                    row
                    for day in sampled_training_dates
                    for row in by_date[day]
                ]
                test_rows = [row for day in test_dates for row in by_date[day]]
                point_train = np.asarray(
                    [[float(row[field]) for field in point_fields] for row in training_rows],
                    dtype=float,
                )
                augmented_train = np.asarray(
                    [[float(row[field]) for field in all_fields] for row in training_rows],
                    dtype=float,
                )
                response_train = np.asarray(
                    [float(row["response"]) for row in training_rows], dtype=float
                )
                point_test = np.asarray(
                    [[float(row[field]) for field in point_fields] for row in test_rows],
                    dtype=float,
                )
                augmented_test = np.asarray(
                    [[float(row[field]) for field in all_fields] for row in test_rows],
                    dtype=float,
                )
                point_prediction = _linear_least_squares_predict(
                    point_train, response_train, point_test
                )
                augmented_prediction = _linear_least_squares_predict(
                    augmented_train, response_train, augmented_test
                )
                for row, point_value, augmented_value in zip(
                    test_rows, point_prediction, augmented_prediction, strict=True
                ):
                    actual = float(row["response"])
                    squared_errors[(str(row["operating_date"]), str(row["candidate_id"]))].append(
                        (
                            float((actual - point_value) ** 2),
                            float((actual - augmented_value) ** 2),
                        )
                    )
                fold_records.append(
                    {
                        "repeat_index": repeat_index,
                        "repeat_seed": repeat_seed,
                        "fold_index": fold_index,
                        "training_date_count": len(training_dates),
                        "test_date_count": len(test_dates),
                        "training_date_start": min(training_dates),
                        "training_date_end": max(training_dates),
                        "test_date_start": min(test_dates),
                        "test_date_end": max(test_dates),
                        "training_block_resample_sha256": hashlib.sha256(
                            np.ascontiguousarray(training_indices).tobytes()
                        ).hexdigest(),
                    }
                )
        evaluation_dates = tuple(day for block in blocks[1:] for day in block)
        candidate_day_records: list[dict[str, Any]] = []
        for operating_date in evaluation_dates:
            for candidate in self.candidate_ids:
                errors = squared_errors[(operating_date, candidate)]
                if len(errors) != repeats:
                    raise EvidenceContractError(
                        "H4c repeated CV did not score every candidate-day exactly once per repeat"
                    )
                point_mse = float(np.mean([item[0] for item in errors]))
                augmented_mse = float(np.mean([item[1] for item in errors]))
                candidate_day_records.append(
                    {
                        "operating_date": operating_date,
                        "candidate_id": candidate,
                        "point_error_only_squared_error": point_mse,
                        "augmented_squared_error": augmented_mse,
                        "improvement": point_mse - augmented_mse,
                        "repeat_count": len(errors),
                    }
                )
        daily_improvement: list[dict[str, Any]] = []
        for operating_date in evaluation_dates:
            day_rows = [
                row
                for row in candidate_day_records
                if row["operating_date"] == operating_date
            ]
            daily_improvement.append(
                {
                    "operating_date": operating_date,
                    "candidate_count": len(day_rows),
                    "mean_candidate_day_improvement": float(
                        np.mean([row["improvement"] for row in day_rows])
                    ),
                }
            )
        bootstrap = paired_moving_block_bootstrap(
            {
                "point_error_only_minus_augmented_squared_error": [
                    row["mean_candidate_day_improvement"] for row in daily_improvement
                ]
            },
            block_lengths=(int(contract["block_days"]),),
            primary_block_length=int(contract["block_days"]),
            resamples=int(contract["resamples"]),
            confidence_level=float(contract["confidence_level"]),
            seed=int(contract["seed"]),
        )
        estimate = bootstrap.estimates[int(contract["block_days"])][
            "point_error_only_minus_augmented_squared_error"
        ]
        lcb = float(estimate.lower_confidence_bound)
        return {
            **base_result,
            "status": "SUPPORTED" if lcb > 0.0 else "NOT_SUPPORTED",
            "support": lcb > 0.0,
            "validation": contract["validation"],
            "folds": folds,
            "repeats": repeats,
            "repeat_seeds": list(repeat_seeds),
            "training_resampling": "paired_7_day_moving_blocks_within_past_dates_only",
            "linear_explanatory_model": "standardized_ordinary_least_squares_with_intercept",
            "initial_training_dates_excluded_from_oos_scoring": list(blocks[0]),
            "evaluated_candidate_day_count": len(candidate_day_records),
            "point_error_only_oos_mse": float(
                np.mean(
                    [row["point_error_only_squared_error"] for row in candidate_day_records]
                )
            ),
            "augmented_oos_mse": float(
                np.mean([row["augmented_squared_error"] for row in candidate_day_records])
            ),
            "mean_oos_squared_error_improvement": float(estimate.sample_mean),
            "paired_7_day_block_bootstrap_lcb": lcb,
            "support_rule": contract["support_rule"],
            "strictly_above_zero": lcb > 0.0,
            "bootstrap": bootstrap.as_dict(include_indices=False),
            "fold_records": fold_records,
            "candidate_day_oos_records_sha256": sha256_json(candidate_day_records),
            "daily_clustered_improvement": daily_improvement,
            "bootstrap_cluster": "operating_date_with_all_candidates_preserved",
            "evidence_scope": "predictive_incremental_explanation_not_causal",
        }

    def _frozen_selectors(
        self,
        metrics: Mapping[str, Mapping[str, Any]],
        values: Mapping[str, Mapping[str, Sequence[float]]],
        selection_results: Mapping[str, Mapping[str, Any]],
        contextual: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        tolerance = self.config["selection"]["tie_tolerance"]
        absolute = float(tolerance["absolute"])
        relative = float(tolerance["relative"])

        def choose(field_name: str, *, higher: bool) -> dict[str, Any]:
            missing = tuple(
                sorted(
                    model
                    for model in self.candidate_ids
                    if metrics[model].get(field_name) is None
                )
            )
            if missing:
                return {
                    "status": "NOT_APPLICABLE_NULL_COMMON_PANEL_METRIC",
                    "selected_model": None,
                    "winner_set": [],
                    "selection_basis": field_name,
                    "selection_values": {
                        model: metrics[model].get(field_name)
                        for model in self.candidate_ids
                    },
                    "missing_models": list(missing),
                    "missing_reasons": {
                        model: metrics[model].get(f"{field_name}_undefined_reason")
                        or metrics[model].get(f"{field_name}_status")
                        for model in missing
                    },
                }
            raw = {
                model: float(metrics[model][field_name])
                for model in self.candidate_ids
            }
            best = max(raw.values()) if higher else min(raw.values())
            winners = tuple(
                sorted(
                    model
                    for model, value in raw.items()
                    if abs(value - best)
                    <= absolute + relative * max(abs(value), abs(best), 1.0)
                )
            )
            return {
                "status": "SELECTED" if len(winners) == 1 else "TIED_WINNER_SET",
                "selected_model": winners[0] if len(winners) == 1 else None,
                "winner_set": list(winners),
                "selection_basis": field_name,
                "selection_values": raw,
            }

        mae_selector = choose("mae", higher=False)
        mae_tie_set = tuple(map(str, mae_selector.get("winner_set", [])))
        mae_selector["primary_metric_tie_set"] = list(mae_tie_set)
        mae_selector["selection_basis"] = "minimum_mae"
        if len(mae_tie_set) > 1:
            crps_values = {
                model: metrics[model].get("crps") for model in mae_tie_set
            }
            if all(value is not None for value in crps_values.values()):
                best_crps = min(float(value) for value in crps_values.values())
                retained = tuple(
                    sorted(
                        model
                        for model, value in crps_values.items()
                        if abs(float(value) - best_crps)
                        <= absolute
                        + relative * max(abs(float(value)), abs(best_crps), 1.0)
                    )
                )
                mae_selector.update(
                    {
                        "status": (
                            "SELECTED" if len(retained) == 1 else "TIED_WINNER_SET"
                        ),
                        "selected_model": retained[0] if len(retained) == 1 else None,
                        "winner_set": list(retained),
                        "selection_basis": "minimum_mae_then_minimum_crps_tie_break",
                        "deterministic_tie_break": {
                            "metric": "crps",
                            "values": crps_values,
                            "scientific_claim": False,
                        },
                    }
                )
        selectors = {
            "MAE-Select": mae_selector,
            "CRPS-Select": choose("crps", higher=False),
            "q99-tail-F1-Select": choose("q99_f1", higher=True),
        }
        mean_values = {
            model: float(np.mean(values["A2"][model])) for model in self.candidate_ids
        }
        best_mean = max(mean_values.values())
        primary = selection_results["A2"]
        primary_gate = primary.get("preregistered_gate", {})
        primary_gate = dict(primary_gate) if isinstance(primary_gate, Mapping) else {}
        baseline_for_tolerance = primary.get("mean_baseline_daily_cost")
        if baseline_for_tolerance is None:
            baseline_for_tolerance = primary_gate.get(
                "mean_baseline_daily_cost_usd"
            )
        mean_economic_tolerance = float(
            baseline_for_tolerance
        ) * ECONOMIC_WINNER_TOLERANCE_FRACTION
        mean_winners = tuple(
            sorted(
                model
                for model, value in mean_values.items()
                if best_mean - value <= mean_economic_tolerance + 1e-12
            )
        )
        selectors["mean-validation-value-Select"] = {
            "status": "SELECTED" if len(mean_winners) == 1 else "TIED_WINNER_SET",
            "selected_model": mean_winners[0] if len(mean_winners) == 1 else None,
            "winner_set": list(mean_winners),
            "selection_basis": "mean_2022_2024_A2_incremental_operational_value",
            "selection_values": mean_values,
            "economic_tolerance_fraction": ECONOMIC_WINNER_TOLERANCE_FRACTION,
            "economic_tolerance_usd": mean_economic_tolerance,
        }
        selectors["DART-ValueSelect"] = {
            "status": primary["status"],
            "selected_model": primary["selected_model"],
            "winner_set": primary["winner_set"],
            "diagnostic_representative": primary.get("diagnostic_representative"),
            "selection_basis": (
                "A1_noninferior_then_A2_feasible_MCS_maximum_7_day_LCB_"
                "with_0.5pct_baseline_economic_winner_set"
            ),
            "selection_values": primary["lcb_by_model"],
            "primary_experiment_id": "A2",
            "a1_noninferiority_gate": primary.get("preregistered_gate"),
            "economic_tolerance_fraction": primary.get(
                "economic_tolerance_fraction"
            ),
            "economic_tolerance_usd": primary.get("economic_tolerance_usd"),
        }
        contextual_config = self.config["contextual_selector"]
        if not contextual_config["enabled"]:
            selectors["contextual_DART-ValueSelect"] = {
                "status": SKIPPED_OPTIONAL_EXTENSION,
                "selected_model": None,
                "winner_set": [],
                "selection_basis": None,
            }
        else:
            if contextual is None:
                raise EvidenceContractError(
                    "enabled contextual selector requires supplied 2022-2024 evidence"
                )
            source = _valid_sha256(
                contextual.get("source_evidence_sha256"),
                label="contextual selector source evidence",
            )
            selected = contextual.get("selected_model")
            winners = tuple(sorted(map(str, contextual.get("winner_set", []))))
            if selected is not None:
                selected = str(selected)
                winners = (selected,)
            if not winners or not set(winners).issubset(self.candidate_ids):
                raise EvidenceContractError("contextual selector has an invalid winner set")
            selectors["contextual_DART-ValueSelect"] = {
                "status": "SELECTED" if selected is not None else "TIED_WINNER_SET",
                "selected_model": selected,
                "winner_set": list(winners),
                "selection_basis": "supplied_frozen_contextual_selector",
                "source_evidence_sha256": source,
            }
        return selectors

    def analyze_selection(self) -> dict[str, Any]:
        self._require_stage("selection_evidence")
        input_sha = sha256_file(self.selection_evidence_path)
        record, resumed = self._begin_stage(
            "selection_analysis", input_sha, previous="selection_evidence"
        )
        if resumed:
            return json.loads(self.selection_analysis_path.read_text(encoding="utf-8"))
        evidence = self._read_evidence(self.selection_evidence_path)
        values, losses, decisions, daily = self._daily_and_decision_summaries(evidence)
        metrics = self._metric_summaries(evidence)
        bootstrap = self.config["bootstrap"]
        mcs = self.config["model_confidence_set"]
        def run_selector(
            experiment: str,
            preregistered_gate: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            diagnostics = {
                model: {
                    "mean_prescriptive_regret": decisions[experiment][model][
                        "mean_prescriptive_regret"
                    ],
                    "tail_calibration_error": metrics[model]["tail_calibration_error"],
                    "forecast_plus_solve_latency_seconds": decisions[experiment][model][
                        "forecast_plus_solve_latency_seconds"
                    ],
                    "model_complexity": decisions[experiment][model]["model_complexity"],
                }
                for model in self.candidate_ids
            }
            result = select_dart_value(
                values[experiment],
                decisions[experiment],
                daily_losses=losses[experiment],
                operating_dates=tuple(sorted(evidence.operating_dates)),
                daily_loss_operating_dates=tuple(sorted(evidence.operating_dates)),
                diagnostics=diagnostics,
                preregistered_gate=preregistered_gate,
                feasibility_criteria=self._criteria(),
                risk_alpha=float(self.config["decision_value"]["risk_alpha"]),
                block_lengths=tuple(bootstrap["blocks_days"]),
                primary_block_length=int(bootstrap["primary_block_days"]),
                confidence_level=float(bootstrap["confidence_level"]),
                bootstrap_resamples=int(bootstrap["resamples"]),
                mcs_resamples=int(mcs["resamples"]),
                seed=int(bootstrap["seed"]),
                lcb_absolute_tolerance=float(
                    self.config["selection"]["lcb_absolute_tolerance"]
                ),
                lcb_relative_tolerance=float(
                    self.config["selection"]["lcb_relative_tolerance"]
                ),
                economic_tolerance_fraction=ECONOMIC_WINNER_TOLERANCE_FRACTION,
                contextual_enabled=False,
            )
            return result.as_dict()

        selection_results: dict[str, dict[str, Any]] = {}
        selection_results["A1"] = run_selector("A1")
        a1_gate = self._a1_noninferiority_gate(
            selection_results["A1"], decisions["A1"]
        )
        selection_results["A2"] = run_selector("A2", a1_gate)
        selectors = self._frozen_selectors(
            metrics,
            values,
            selection_results,
            evidence.contextual_selector,
        )
        hypotheses = {
            "H4a": self._h4a(metrics, decisions["A2"], phase="selection"),
            "H4c": self._h4c(evidence, daily["A2"]),
        }
        analysis = {
            "schema_version": SCHEMA_VERSION,
            "status": "SELECTION_ANALYZED",
            "selection_years": [2022, 2023, 2024],
            "locked_test_opened": False,
            "source_evidence_sha256": evidence.source_evidence_sha256,
            "source_evidence_snapshot_sha256": input_sha,
            "three_layers": {
                "probabilistic_credibility": metrics,
                "actionability": metrics,
                "decision_value": decisions,
                "primary_ranking_layer": "decision_value",
                "point_metrics_select_final_winner_alone": False,
            },
            "selection_results_by_experiment": selection_results,
            "frozen_selector_candidates": selectors,
            "A3": self._derive_a3(evidence),
            "hypotheses_before_locked_test": {
                **hypotheses,
                "H4b": {"status": "PENDING_LOCKED_TEST"},
                "H4d": {
                    "status": (
                        "PENDING_LOCKED_TEST"
                        if self.config["contextual_selector"]["enabled"]
                        else SKIPPED_OPTIONAL_EXTENSION
                    )
                },
            },
            "created_at_utc": utc_now(),
        }
        analysis["analysis_sha256"] = sha256_json(analysis)
        atomic_json(self.selection_analysis_path, analysis)
        self._finish_stage(
            "selection_analysis",
            input_sha,
            (self.selection_analysis_path,),
            details={
                "analysis_sha256": analysis["analysis_sha256"],
                "A1_status": selection_results["A1"]["status"],
                "A2_status": selection_results["A2"]["status"],
            },
        )
        return analysis

    @staticmethod
    def _dependency_fingerprints(values: Mapping[str, str]) -> dict[str, str]:
        missing = set(FREEZE_FINGERPRINT_KEYS).difference(values)
        if missing:
            raise EvidenceContractError(
                f"freeze dependencies are missing: {sorted(missing)}"
            )
        return {
            str(key): _valid_sha256(value, label=f"freeze dependency {key}")
            for key, value in sorted(values.items())
        }

    def freeze_selection(self, dependency_fingerprints: Mapping[str, str]) -> dict[str, Any]:
        self._require_stage("selection_analysis")
        dependencies = self._dependency_fingerprints(dependency_fingerprints)
        contract = {
            "schema_version": SCHEMA_VERSION,
            "experiment_fingerprint": self.experiment_fingerprint,
            "config_sha256": self.config_sha256,
            "selection_evidence_sha256": sha256_file(self.selection_evidence_path),
            "selection_analysis_sha256": sha256_file(self.selection_analysis_path),
            "dependency_fingerprints": dependencies,
            "selection_years": [2022, 2023, 2024],
            "locked_test_year": 2025,
            "locked_test_runs_allowed": 1,
            "locked_test_may_update_selector": False,
            "candidate_pool_contract": self.candidate_pool_contract,
            "terminal_status_cap": (
                PARTIAL_CANDIDATE_TERMINAL_CAP
                if self.candidate_pool_contract is not None
                else None
            ),
        }
        input_sha = sha256_json(contract)
        record, resumed = self._begin_stage(
            "selection_freeze", input_sha, previous="selection_analysis"
        )
        if resumed:
            return json.loads(self.selection_freeze_path.read_text(encoding="utf-8"))
        if self.locked_ledger_path.exists() or self.locked_evidence_path.exists():
            raise LockedTestGuardError("2025 evidence exists before the selection freeze")
        analysis = json.loads(self.selection_analysis_path.read_text(encoding="utf-8"))
        manifest = {
            "status": "FROZEN_BEFORE_2025",
            "frozen_contract": contract,
            "frozen_selectors": analysis["frozen_selector_candidates"],
            "selection_results_by_experiment": analysis[
                "selection_results_by_experiment"
            ],
            "freeze_fingerprint": input_sha,
            "frozen_at_utc": utc_now(),
        }
        atomic_json(self.selection_freeze_path, manifest)
        self._finish_stage(
            "selection_freeze",
            input_sha,
            (self.selection_freeze_path,),
            details={"freeze_fingerprint": input_sha},
        )
        return manifest

    def begin_locked_test(
        self,
        operating_dates: Sequence[str],
        *,
        dependency_fingerprints: Mapping[str, str],
        resume: bool = True,
    ) -> dict[str, Any]:
        self._require_stage("selection_freeze")
        dates = _date_tuple(operating_dates, phase="locked_test")
        dependencies = self._dependency_fingerprints(dependency_fingerprints)
        freeze = json.loads(self.selection_freeze_path.read_text(encoding="utf-8"))
        if dependencies != freeze["frozen_contract"]["dependency_fingerprints"]:
            raise LockedTestGuardError("locked-test dependencies differ from the freeze")
        contract = {
            "schema_version": SCHEMA_VERSION,
            "freeze_fingerprint": freeze["freeze_fingerprint"],
            "freeze_manifest_sha256": sha256_file(self.selection_freeze_path),
            "dependency_fingerprints": dependencies,
            "operating_dates": list(dates),
            "candidate_ids": list(self.candidate_ids),
            "experiments": list(EXPERIMENT_IDS),
            "seeds": list(DART_VALUE_SELECT_SEEDS),
            "expected_cell_count": (
                len(dates)
                * len(self.candidate_ids)
                * len(EXPERIMENT_IDS)
                * len(DART_VALUE_SELECT_SEEDS)
            ),
            "locked_test_results_may_modify_selection": False,
        }
        input_sha = sha256_json(contract)
        if self.locked_ledger_path.is_file():
            ledger = json.loads(self.locked_ledger_path.read_text(encoding="utf-8"))
            if ledger.get("locked_test_fingerprint") != input_sha:
                raise LockedTestGuardError(
                    "a different contract already opened the locked test"
                )
            if not resume:
                raise LockedTestGuardError("the once-only locked test is already open")
        record, resumed = self._begin_stage(
            "locked_test_open", input_sha, previous="selection_freeze"
        )
        if resumed:
            return json.loads(self.locked_ledger_path.read_text(encoding="utf-8"))
        state = self._load_state()
        if state.get("locked_test_run_count", 0) != 0:
            raise LockedTestGuardError("locked-test run count is already nonzero")
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "status": "OPEN",
            "locked_test_fingerprint": input_sha,
            "locked_contract": contract,
            "run_count": 1,
            "opened_at_utc": utc_now(),
            "selection_updated_from_2025": False,
            "candidate_pool_contract": self.candidate_pool_contract,
            "terminal_status_cap": (
                PARTIAL_CANDIDATE_TERMINAL_CAP
                if self.candidate_pool_contract is not None
                else None
            ),
        }
        open_contract = {
            "schema_version": SCHEMA_VERSION,
            "status": "LOCKED_TEST_OPEN_CONTRACT_FROZEN",
            "locked_test_fingerprint": input_sha,
            "locked_contract": contract,
        }
        atomic_json(self.locked_open_contract_path, open_contract)
        atomic_json(self.locked_ledger_path, ledger)
        state = self._load_state()
        state["locked_test_run_count"] = 1
        state["updated_at_utc"] = utc_now()
        atomic_json(self.state_path, state)
        self._finish_stage(
            "locked_test_open",
            input_sha,
            (self.locked_open_contract_path,),
            details={"locked_test_fingerprint": input_sha, "run_count": 1},
        )
        return ledger

    def ingest_locked_test_evidence(self, evidence: DARTPhaseEvidence) -> dict[str, Any]:
        self._require_stage("locked_test_open")
        if evidence.phase != "locked_test":
            raise EvidenceContractError("locked ingestion requires locked-test evidence")
        ledger = json.loads(self.locked_ledger_path.read_text(encoding="utf-8"))
        if tuple(sorted(evidence.operating_dates)) != tuple(
            ledger["locked_contract"]["operating_dates"]
        ):
            raise LockedTestGuardError("locked-test operating-date panel changed")
        dependencies = self._dependency_fingerprints(
            evidence.frozen_dependency_fingerprints
        )
        if dependencies != ledger["locked_contract"]["dependency_fingerprints"]:
            raise LockedTestGuardError("locked evidence used changed frozen dependencies")
        validation = self._validate_phase_evidence(evidence)
        input_sha = evidence.fingerprint
        record, resumed = self._begin_stage(
            "locked_test_evidence", input_sha, previous="locked_test_open"
        )
        if resumed:
            return dict(record["details"])
        payload = {**evidence.to_dict(), "validation": validation}
        atomic_json(self.locked_evidence_path, payload)
        self._finish_stage(
            "locked_test_evidence",
            input_sha,
            (self.locked_evidence_path,),
            details=validation,
        )
        return validation

    def _locked_selector_rows(
        self,
        frozen_selectors: Mapping[str, Mapping[str, Any]],
        locked_decisions: Mapping[str, Mapping[str, Mapping[str, Any]]],
        locked_metrics: Mapping[str, Mapping[str, Any]],
        *,
        selection_freeze_sha256: str,
        locked_evidence_sha256: str,
        locked_replay_results_sha256: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        global_a1_gate = frozen_selectors.get("DART-ValueSelect", {}).get(
            "a1_noninferiority_gate", {}
        )
        global_a1_gate = (
            dict(global_a1_gate) if isinstance(global_a1_gate, Mapping) else {}
        )
        a1_lcbs = global_a1_gate.get("lcb_by_model", {})
        a1_lcbs = dict(a1_lcbs) if isinstance(a1_lcbs, Mapping) else {}
        a1_eligible = set(map(str, global_a1_gate.get("eligible_models", [])))
        for selector, frozen in frozen_selectors.items():
            selected = frozen.get("selected_model")
            winner_set = tuple(sorted(map(str, frozen.get("winner_set", []))))
            if not winner_set and selected is not None:
                winner_set = (str(selected),)
            members: tuple[str | None, ...] = winner_set or (None,)
            comparison_role = (
                "PRIMARY_MAE_SELECT_VS_DART_VALUE_SELECT"
                if selector in {"MAE-Select", "DART-ValueSelect"}
                else "SUPPLEMENTARY_SELECTOR_COMPARISON"
            )
            for member in members:
                a2_summary = (
                    locked_decisions.get("A2", {}).get(member, {})
                    if member is not None
                    else {}
                )
                a1_summary = (
                    locked_decisions.get("A1", {}).get(member, {})
                    if member is not None
                    else {}
                )
                metric = locked_metrics.get(member, {}) if member is not None else {}
                a2_pcr = a2_summary.get("profit_capture_ratio", {})
                a1_pcr = a1_summary.get("profit_capture_ratio", {})
                selection_values = frozen.get("selection_values", {})
                selection_value = (
                    selection_values.get(member)
                    if member is not None and isinstance(selection_values, Mapping)
                    else None
                )
                rows.append(
                    {
                        "selector": selector,
                        "comparison_role": comparison_role,
                        "status": frozen["status"],
                        "selected_model": selected,
                        "candidate_id": member,
                        "winner_set_member": member,
                        "winner_set": json.dumps(list(winner_set)),
                        "winner_set_size": len(winner_set),
                        "member_is_selected_model": bool(
                            member is not None and member == selected
                        ),
                        "diagnostic_representative": frozen.get(
                            "diagnostic_representative"
                        ),
                        "member_is_diagnostic_representative": bool(
                            member is not None
                            and member == frozen.get("diagnostic_representative")
                        ),
                        "scientific_winner_set_collapsed": False,
                        "selection_basis": frozen.get("selection_basis"),
                        "selection_value": selection_value,
                        "primary_selection_experiment": "A2",
                        "a1_role": "PREREGISTERED_NONINFERIORITY_GATE",
                        "a1_noninferiority_epsilon_fraction": global_a1_gate.get(
                            "epsilon_fraction_of_mean_baseline_daily_cost"
                        ),
                        "a1_noninferiority_epsilon_usd": global_a1_gate.get(
                            "epsilon_usd"
                        ),
                        "a1_noninferiority_lcb": a1_lcbs.get(member),
                        "a1_noninferiority_passed": (
                            member in a1_eligible if member is not None else None
                        ),
                        "locked_test_mae": metric.get("mae"),
                        "locked_test_crps": metric.get("crps"),
                        "locked_test_q99_tail_f1": metric.get("q99_f1"),
                        "locked_test_cumulative_incremental_operational_value": (
                            a2_summary.get("cumulative_incremental_operational_value")
                        ),
                        "locked_test_mean_daily_incremental_operational_value": (
                            a2_summary.get("mean_daily_incremental_operational_value")
                        ),
                        "profit_capture_ratio": a2_pcr.get("value"),
                        "profit_capture_ratio_status": a2_pcr.get("status"),
                        "mean_prescriptive_regret": a2_summary.get(
                            "mean_prescriptive_regret"
                        ),
                        "probability_below_baseline": a2_summary.get(
                            "probability_below_baseline"
                        ),
                        "cvar95_loss": a2_summary.get("cvar95_loss"),
                        "weighted_sla_satisfaction": a2_summary.get(
                            "weighted_sla_satisfaction"
                        ),
                        "forecast_plus_solve_latency_seconds": a2_summary.get(
                            "forecast_plus_solve_latency_seconds"
                        ),
                        "locked_test_A2_cumulative_incremental_operational_value": (
                            a2_summary.get("cumulative_incremental_operational_value")
                        ),
                        "locked_test_A2_mean_daily_incremental_operational_value": (
                            a2_summary.get("mean_daily_incremental_operational_value")
                        ),
                        "locked_test_A2_mean_prescriptive_regret": a2_summary.get(
                            "mean_prescriptive_regret"
                        ),
                        "locked_test_A2_probability_below_baseline": a2_summary.get(
                            "probability_below_baseline"
                        ),
                        "locked_test_A2_cvar95_loss": a2_summary.get("cvar95_loss"),
                        "locked_test_A2_weighted_sla_satisfaction": a2_summary.get(
                            "weighted_sla_satisfaction"
                        ),
                        "locked_test_A2_profit_capture_ratio": a2_pcr.get("value"),
                        "locked_test_A1_cumulative_incremental_operational_value": (
                            a1_summary.get("cumulative_incremental_operational_value")
                        ),
                        "locked_test_A1_mean_daily_incremental_operational_value": (
                            a1_summary.get("mean_daily_incremental_operational_value")
                        ),
                        "locked_test_A1_mean_prescriptive_regret": a1_summary.get(
                            "mean_prescriptive_regret"
                        ),
                        "locked_test_A1_profit_capture_ratio": a1_pcr.get("value"),
                        "derived_from_same_locked_A2_replay": True,
                        "derived_without_new_solves": True,
                        "selection_oos_years": "2022-2024",
                        "locked_test_year": 2025,
                        "test_used_for_selection": False,
                        "selection_updated_from_2025": False,
                        "selection_freeze_sha256": selection_freeze_sha256,
                        "locked_evidence_sha256": locked_evidence_sha256,
                        "locked_replay_results_sha256": locked_replay_results_sha256,
                        "A2_source_records_sha256": a2_summary.get(
                            "source_records_sha256"
                        ),
                        "A1_source_records_sha256": a1_summary.get(
                            "source_records_sha256"
                        ),
                        "metric_source_records_sha256": metric.get(
                            "source_records_sha256"
                        ),
                        "counterfactual": True,
                        "denominator": (
                            "locked_2025_operating_date_after_within_date_exact_"
                            "three_seed_mean"
                        ),
                    }
                )
        return rows

    def _h4b(
        self,
        frozen_selectors: Mapping[str, Mapping[str, Any]],
        locked_decisions: Mapping[str, Mapping[str, Any]],
        locked_daily: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        contract = self.config["hypothesis_contract"][
            "H4b_conservative_set_valued_dominance"
        ]
        target = frozen_selectors["DART-ValueSelect"]
        comparators = (
            "MAE-Select",
            "CRPS-Select",
            "q99-tail-F1-Select",
            "mean-validation-value-Select",
        )
        feasible, exclusions = evaluate_feasibility(
            locked_decisions, criteria=self._criteria()
        )
        feasible_set = set(feasible)
        target_members = tuple(sorted(map(str, target.get("winner_set", []))))
        target_all_feasible = bool(target_members) and set(target_members).issubset(
            feasible_set
        )
        target_infeasible = tuple(
            sorted(set(target_members).difference(feasible_set))
        )
        comparison_audit: dict[str, Any] = {}
        series: dict[str, list[float]] = {}
        series_metadata: dict[str, dict[str, Any]] = {}
        all_comparator_selectors_applicable = True
        dates = tuple(
            sorted(next(iter(locked_daily.values())).keys())
            if locked_daily
            else ()
        )
        for name in comparators:
            comparator = frozen_selectors[name]
            comparator_members = tuple(
                sorted(map(str, comparator.get("winner_set", [])))
            )
            comparator_feasible = tuple(
                member for member in comparator_members if member in feasible_set
            )
            comparator_infeasible = tuple(
                member for member in comparator_members if member not in feasible_set
            )
            applicable = bool(comparator_feasible)
            all_comparator_selectors_applicable = (
                all_comparator_selectors_applicable and applicable
            )
            comparison_audit[name] = {
                "frozen_winner_set": list(comparator_members),
                "feasible_comparator_members": list(comparator_feasible),
                "infeasible_comparator_members_disclosed_not_counted": list(
                    comparator_infeasible
                ),
                "comparison_applicable": applicable,
                "selector_status": comparator.get("status"),
                "pairwise_comparisons": [],
            }
            for target_member in target_members:
                for comparator_member in comparator_feasible:
                    if target_member not in locked_daily or comparator_member not in locked_daily:
                        raise EvidenceContractError(
                            "H4b winner set names a model absent from locked daily evidence"
                        )
                    target_dates = tuple(sorted(locked_daily[target_member]))
                    comparator_dates = tuple(sorted(locked_daily[comparator_member]))
                    if target_dates != dates or comparator_dates != dates:
                        raise EvidenceContractError(
                            "H4b pair does not share the exact locked operating-date panel"
                        )
                    pair_id = f"{name}|{target_member}|{comparator_member}"
                    value_key = f"{pair_id}|target_minus_comparator_value"
                    regret_key = f"{pair_id}|comparator_minus_target_regret"
                    series[value_key] = [
                        float(locked_daily[target_member][day]["incremental_value"])
                        - float(locked_daily[comparator_member][day]["incremental_value"])
                        for day in dates
                    ]
                    series[regret_key] = [
                        float(locked_daily[comparator_member][day]["prescriptive_regret"])
                        - float(locked_daily[target_member][day]["prescriptive_regret"])
                        for day in dates
                    ]
                    metadata = {
                        "selector": name,
                        "target_member": target_member,
                        "comparator_member": comparator_member,
                        "value_series_key": value_key,
                        "regret_series_key": regret_key,
                    }
                    comparison_audit[name]["pairwise_comparisons"].append(metadata)
                    series_metadata[pair_id] = metadata
        if not series:
            return {
                "hypothesis": "H4b",
                "status": "NOT_SUPPORTED",
                "support": False,
                "reason": "NO_FEASIBLE_SET_VALUED_COMPARATOR_PAIR",
                "target_policy": contract["target_policy"],
                "DART_ValueSelect_winner_set": list(target_members),
                "all_target_members_feasible": target_all_feasible,
                "infeasible_target_members": list(target_infeasible),
                "comparison_audit": comparison_audit,
                "selection_updated_from_2025": False,
                "counterfactual": True,
                "cssa": {
                    "status": "NOT_APPLICABLE_NO_FEASIBLE_SET_VALUED_PAIR",
                    "set_valued": True,
                    "derived_without_new_solves": True,
                    "block_days": list(self.config["bootstrap"]["blocks_days"]),
                },
            }
        cssa_blocks = tuple(map(int, self.config["bootstrap"]["blocks_days"]))
        primary_block = int(
            contract.get("block_days_primary", contract.get("block_days", 7))
        )
        bootstrap = paired_moving_block_bootstrap(
            series,
            block_lengths=cssa_blocks,
            primary_block_length=primary_block,
            resamples=int(contract["resamples"]),
            confidence_level=float(contract["confidence_level"]),
            seed=int(contract["seed"]),
        )
        estimates = bootstrap.estimates[primary_block]
        one_sided_tail = 1.0 - float(contract["confidence_level"])
        cssa_by_comparator: dict[str, dict[str, Any]] = {}
        every_cssa_lcb_positive = True
        for name in comparators:
            selector_pairs = [
                metadata
                for metadata in series_metadata.values()
                if metadata["selector"] == name
            ]
            if not selector_pairs:
                cssa_by_comparator[name] = {
                    "status": "NOT_APPLICABLE_NO_FEASIBLE_PAIR",
                    "pair_count": 0,
                }
                every_cssa_lcb_positive = False
                continue
            value_lcb_by_block: dict[str, float] = {}
            regret_lcb_by_block: dict[str, float] = {}
            index_hash_by_block: dict[str, str] = {}
            for block in cssa_blocks:
                indices = bootstrap.indices_by_block[block]
                value_draws = np.vstack(
                    [
                        np.mean(
                            np.asarray(series[metadata["value_series_key"]], dtype=float)[
                                indices
                            ],
                            axis=1,
                        )
                        for metadata in selector_pairs
                    ]
                )
                regret_draws = np.vstack(
                    [
                        np.mean(
                            np.asarray(series[metadata["regret_series_key"]], dtype=float)[
                                indices
                            ],
                            axis=1,
                        )
                        for metadata in selector_pairs
                    ]
                )
                simultaneous_value_draw = np.min(value_draws, axis=0)
                simultaneous_regret_draw = np.min(regret_draws, axis=0)
                value_lcb_by_block[str(block)] = float(
                    np.quantile(simultaneous_value_draw, one_sided_tail)
                )
                regret_lcb_by_block[str(block)] = float(
                    np.quantile(simultaneous_regret_draw, one_sided_tail)
                )
                first_key = selector_pairs[0]["value_series_key"]
                index_hash_by_block[str(block)] = bootstrap.estimates[block][
                    first_key
                ].index_hash
            primary_value_lcb = value_lcb_by_block[str(primary_block)]
            primary_regret_lcb = regret_lcb_by_block[str(primary_block)]
            cssa_positive = primary_value_lcb > 0.0 and primary_regret_lcb > 0.0
            every_cssa_lcb_positive = every_cssa_lcb_positive and cssa_positive
            cssa_by_comparator[name] = {
                "status": "COMPLETE",
                "pair_count": len(selector_pairs),
                "target_minus_comparator_sample_CSSA": min(
                    float(np.mean(series[metadata["value_series_key"]]))
                    for metadata in selector_pairs
                ),
                "comparator_minus_target_regret_sample_CSSA": min(
                    float(np.mean(series[metadata["regret_series_key"]]))
                    for metadata in selector_pairs
                ),
                "target_minus_comparator_value_CSSA_lcb_by_block": (
                    value_lcb_by_block
                ),
                "comparator_minus_target_regret_CSSA_lcb_by_block": (
                    regret_lcb_by_block
                ),
                "primary_value_CSSA_lcb_strictly_above_zero": (
                    primary_value_lcb > 0.0
                ),
                "primary_regret_CSSA_lcb_strictly_above_zero": (
                    primary_regret_lcb > 0.0
                ),
                "common_bootstrap_index_sha256_by_block": index_hash_by_block,
            }
            comparison_audit[name]["cssa"] = cssa_by_comparator[name]

        every_pairwise_lcb_positive = True
        pair_results: list[dict[str, Any]] = []
        for pair_id, metadata in sorted(series_metadata.items()):
            value_estimate = estimates[metadata["value_series_key"]]
            regret_estimate = estimates[metadata["regret_series_key"]]
            value_positive = value_estimate.lower_confidence_bound > 0.0
            regret_positive = regret_estimate.lower_confidence_bound > 0.0
            every_pairwise_lcb_positive = (
                every_pairwise_lcb_positive and value_positive and regret_positive
            )
            result = {
                "pair_id": pair_id,
                "selector": metadata["selector"],
                "target_member": metadata["target_member"],
                "comparator_member": metadata["comparator_member"],
                "target_minus_comparator_mean_value": value_estimate.sample_mean,
                "target_minus_comparator_value_7_day_lcb": (
                    value_estimate.lower_confidence_bound
                ),
                "comparator_minus_target_mean_regret": regret_estimate.sample_mean,
                "comparator_minus_target_regret_7_day_lcb": (
                    regret_estimate.lower_confidence_bound
                ),
                "value_lcb_strictly_above_zero": value_positive,
                "regret_lcb_strictly_above_zero": regret_positive,
                "common_bootstrap_index_sha256": value_estimate.index_hash,
                "target_minus_comparator_value_lcb_by_block": {
                    str(block): bootstrap.estimates[block][
                        metadata["value_series_key"]
                    ].lower_confidence_bound
                    for block in cssa_blocks
                },
                "comparator_minus_target_regret_lcb_by_block": {
                    str(block): bootstrap.estimates[block][
                        metadata["regret_series_key"]
                    ].lower_confidence_bound
                    for block in cssa_blocks
                },
                "common_bootstrap_index_sha256_by_block": {
                    str(block): bootstrap.estimates[block][
                        metadata["value_series_key"]
                    ].index_hash
                    for block in cssa_blocks
                },
            }
            pair_results.append(result)
            comparison_audit[metadata["selector"]]["pairwise_results"] = [
                row
                for row in pair_results
                if row["selector"] == metadata["selector"]
            ]
        primary_comparator = str(contract.get("primary_comparator", "MAE-Select"))
        primary_cssa = cssa_by_comparator.get(primary_comparator, {})
        primary_comparator_applicable = bool(
            comparison_audit.get(primary_comparator, {}).get(
                "comparison_applicable", False
            )
        )
        primary_cssa_positive = bool(
            primary_cssa.get("primary_value_CSSA_lcb_strictly_above_zero") is True
            and primary_cssa.get("primary_regret_CSSA_lcb_strictly_above_zero")
            is True
        )
        supported = bool(
            target_all_feasible
            and primary_comparator_applicable
            and primary_cssa_positive
        )
        return {
            "hypothesis": "H4b",
            "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
            "support": supported,
            "target_policy": contract["target_policy"],
            "comparison_policy": contract["comparison_policy"],
            "DART_ValueSelect_winner_set": list(target_members),
            "deployment_representative_not_used_to_force_uniqueness": target.get(
                "diagnostic_representative"
            ),
            "all_target_members_feasible": target_all_feasible,
            "infeasible_target_members": list(target_infeasible),
            "all_comparator_selectors_have_feasible_members": (
                all_comparator_selectors_applicable
            ),
            "primary_comparator": primary_comparator,
            "primary_comparator_has_feasible_members": (
                primary_comparator_applicable
            ),
            "primary_CSSA_value_and_regret_lcb_strictly_above_zero": (
                primary_cssa_positive
            ),
            "every_paired_value_and_regret_lcb_strictly_above_zero": (
                every_pairwise_lcb_positive
            ),
            "every_simultaneous_CSSA_value_and_regret_lcb_strictly_above_zero": (
                every_cssa_lcb_positive
            ),
            "locked_feasible_models": list(feasible),
            "locked_feasibility_exclusions": {
                model: list(reasons) for model, reasons in exclusions.items()
            },
            "comparison_audit": comparison_audit,
            "pairwise_results": pair_results,
            "block_days": primary_block,
            "block_days_sensitivity": list(cssa_blocks),
            "confidence_level": float(contract["confidence_level"]),
            "resamples": int(contract["resamples"]),
            "seed": int(contract["seed"]),
            "paired_common_bootstrap_indices": bootstrap.paired_indices,
            "bootstrap": bootstrap.as_dict(include_indices=False),
            "cssa": {
                "status": "COMPLETE",
                "name": "conservative_set_valued_selection_advantage",
                "definition": (
                    "Q_0.05_star_of_min_over_DART_winner_i_and_comparator_"
                    "winner_j_of_common_block_bootstrap_mean_daily_DeltaV_i_minus_j"
                ),
                "formula": (
                    "Q_0.05^*(min_{i in W_DV,j in W_k} "
                    "mean_d^*(DeltaV_i,d-DeltaV_j,d))"
                ),
                "set_valued": True,
                "primary_comparison": "MAE-Select_vs_DART-ValueSelect",
                "primary_support_uses_only_MAE_comparison": True,
                "supplementary_comparators": [
                    "CRPS-Select",
                    "q99-tail-F1-Select",
                    "mean-validation-value-Select",
                ],
                "block_days": list(cssa_blocks),
                "primary_block_days": primary_block,
                "paired_common_bootstrap_indices": bootstrap.paired_indices,
                "derived_from_same_locked_A2_replay": True,
                "derived_without_new_solves": True,
                "by_comparator_selector": cssa_by_comparator,
                "pairwise_results": pair_results,
            },
            "selection_updated_from_2025": False,
            "counterfactual": True,
        }

    def _h4d(self, selector_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not self.config["contextual_selector"]["enabled"]:
            return {
                "hypothesis": "H4d",
                "status": SKIPPED_OPTIONAL_EXTENSION,
            }
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selector_rows:
            grouped[str(row["selector"])].append(row)

        def representative(name: str) -> Mapping[str, Any]:
            candidates = grouped[name]
            selected = candidates[0].get("selected_model") if candidates else None
            if selected is None:
                return candidates[0]
            return next(
                (
                    row
                    for row in candidates
                    if row.get("winner_set_member") == selected
                ),
                candidates[0],
            )

        contextual = representative("contextual_DART-ValueSelect")
        static = representative("DART-ValueSelect")
        if contextual.get("selected_model") is None or static.get("selected_model") is None:
            return {"hypothesis": "H4d", "status": "INCONCLUSIVE_FROZEN_WINNER_SET"}
        value_gain = float(
            contextual["locked_test_mean_daily_incremental_operational_value"]
        ) - float(static["locked_test_mean_daily_incremental_operational_value"])
        no_sla_worsening = float(contextual["weighted_sla_satisfaction"]) >= float(
            static["weighted_sla_satisfaction"]
        )
        no_tail_worsening = float(contextual["cvar95_loss"]) <= float(
            static["cvar95_loss"]
        )
        return {
            "hypothesis": "H4d",
            "status": (
                "SUPPORTED"
                if value_gain > 0.0 and no_sla_worsening and no_tail_worsening
                else "NOT_SUPPORTED"
            ),
            "mean_incremental_value_gain": value_gain,
            "no_sla_worsening": no_sla_worsening,
            "no_tail_loss_worsening": no_tail_worsening,
        }

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
        atomic_text(path, buffer.getvalue())

    def _table_rows(
        self,
        selection_evidence: DARTPhaseEvidence,
        locked_evidence: DARTPhaseEvidence,
        selection_metrics: Mapping[str, Mapping[str, Any]],
        locked_metrics: Mapping[str, Mapping[str, Any]],
        selection_decisions: Mapping[str, Mapping[str, Mapping[str, Any]]],
        locked_decisions: Mapping[str, Mapping[str, Mapping[str, Any]]],
        selection_daily: Mapping[str, Mapping[str, Mapping[str, Any]]],
        locked_daily: Mapping[str, Mapping[str, Mapping[str, Any]]],
        selector_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        probability_rows: list[dict[str, Any]] = []
        decision_rows: list[dict[str, Any]] = []
        regime_rows: list[dict[str, Any]] = []
        for phase, evidence, metric_map, decision_map, daily_map in (
            (
                "selection_2022_2024",
                selection_evidence,
                selection_metrics,
                selection_decisions,
                selection_daily,
            ),
            (
                "locked_test_2025",
                locked_evidence,
                locked_metrics,
                locked_decisions,
                locked_daily,
            ),
        ):
            for candidate in self.candidate_ids:
                probability_rows.append(
                    {
                        "phase": phase,
                        "model": candidate,
                        **metric_map[candidate],
                        "evidence_role": "derived_from_supplied_oos_metric_records",
                        "denominator": "operating_date_x_exact_three_seed_panel",
                    }
                )
            for experiment in EXPERIMENT_IDS:
                for candidate in self.candidate_ids:
                    summary = decision_map[experiment][candidate]
                    pcr = summary["profit_capture_ratio"]
                    decision_rows.append(
                        {
                            "phase": phase,
                            "model": candidate,
                            "experiment_id": experiment,
                            "cumulative_incremental_operational_value": summary[
                                "cumulative_incremental_operational_value"
                            ],
                            "mean_daily_incremental_operational_value": summary[
                                "mean_daily_incremental_operational_value"
                            ],
                            "equivalent_operating_cost_reduction": summary[
                                "equivalent_operating_cost_reduction"
                            ],
                            "operating_cost_reduction_rate": summary[
                                "operating_cost_reduction_rate"
                            ],
                            "profit_capture_ratio": pcr["value"],
                            "profit_capture_ratio_status": pcr["status"],
                            "mean_prescriptive_regret": summary[
                                "mean_prescriptive_regret"
                            ],
                            "probability_below_baseline": summary[
                                "probability_below_baseline"
                            ],
                            "cvar95_loss": summary["cvar95_loss"],
                            "maximum_drawdown": summary["maximum_drawdown"],
                            "weighted_sla_satisfaction": summary[
                                "weighted_sla_satisfaction"
                            ],
                            "deadline_violations": summary["deadline_violations"],
                            "infeasible_days": summary["infeasible_days"],
                            "invalid_days": summary["invalid_days"],
                            "solver_timeout_days": summary["solver_timeout_days"],
                            "solver_nonzero_gap_days": summary[
                                "solver_nonzero_gap_days"
                            ],
                            "solver_gap_over_cap_days": summary[
                                "solver_gap_over_cap_days"
                            ],
                            "solver_missing_gap_days": summary[
                                "solver_missing_gap_days"
                            ],
                            "solver_no_incumbent_days": summary[
                                "solver_no_incumbent_days"
                            ],
                            "forecast_plus_solve_latency_seconds": summary[
                                "forecast_plus_solve_latency_seconds"
                            ],
                            "counterfactual": True,
                            "denominator": "daily_mean_across_exact_three_seed_panel",
                            "source_records_sha256": summary[
                                "source_records_sha256"
                            ],
                        }
                    )
                    for regime in REGIMES:
                        dates = [
                            day
                            for day, labels in evidence.regime_labels_by_date.items()
                            if regime in labels
                        ]
                        values = [
                            daily_map[experiment][candidate][day]["incremental_value"]
                            for day in dates
                        ]
                        regime_rows.append(
                            {
                                "phase": phase,
                                "model": candidate,
                                "experiment_id": experiment,
                                "regime": regime,
                                "day_count": len(values),
                                "mean_incremental_operational_value": (
                                    float(np.mean(values)) if values else None
                                ),
                                "status": "OK" if values else "NO_DAYS_IN_SUPPLIED_EVIDENCE",
                                "counterfactual": True,
                                "denominator": "regime_days_mean_across_exact_three_seed_panel",
                            }
                        )
        return probability_rows, decision_rows, regime_rows

    def finalize(self) -> dict[str, Any]:
        self._require_stage("locked_test_evidence")
        input_sha = sha256_json(
            {
                "selection_freeze_sha256": sha256_file(self.selection_freeze_path),
                "selection_evidence_sha256": sha256_file(self.selection_evidence_path),
                "locked_evidence_sha256": sha256_file(self.locked_evidence_path),
                "locked_ledger_fingerprint": json.loads(
                    self.locked_ledger_path.read_text(encoding="utf-8")
                )["locked_test_fingerprint"],
            }
        )
        record, resumed = self._begin_stage(
            "finalization", input_sha, previous="locked_test_evidence"
        )
        if resumed:
            return json.loads(self.final_manifest_path.read_text(encoding="utf-8"))
        selection_evidence = self._read_evidence(self.selection_evidence_path)
        locked_evidence = self._read_evidence(self.locked_evidence_path)
        selection_values, _, selection_decisions, selection_daily = (
            self._daily_and_decision_summaries(selection_evidence)
        )
        locked_values, _, locked_decisions, locked_daily = (
            self._daily_and_decision_summaries(locked_evidence)
        )
        del selection_values, locked_values
        selection_metrics = self._metric_summaries(selection_evidence)
        locked_metrics = self._metric_summaries(locked_evidence)
        freeze = json.loads(self.selection_freeze_path.read_text(encoding="utf-8"))
        frozen_selectors = freeze["frozen_selectors"]
        selector_rows = self._locked_selector_rows(
            frozen_selectors,
            locked_decisions,
            locked_metrics,
            selection_freeze_sha256=sha256_file(self.selection_freeze_path),
            locked_evidence_sha256=sha256_file(self.locked_evidence_path),
            locked_replay_results_sha256=sha256_json(
                [sha256_json(row) for row in locked_evidence.replay_results]
            ),
        )
        selection_analysis = json.loads(
            self.selection_analysis_path.read_text(encoding="utf-8")
        )
        hypotheses = {
            "H4a": self._h4a(
                locked_metrics, locked_decisions["A2"], phase="locked_test"
            ),
            "H4c": selection_analysis["hypotheses_before_locked_test"]["H4c"],
        }
        hypotheses["H4b"] = self._h4b(
            frozen_selectors,
            locked_decisions["A2"],
            locked_daily["A2"],
        )
        cssa_by_comparator = (
            hypotheses["H4b"].get("cssa", {}).get("by_comparator_selector", {})
            if isinstance(hypotheses["H4b"], Mapping)
            else {}
        )
        cssa_by_comparator = (
            cssa_by_comparator
            if isinstance(cssa_by_comparator, Mapping)
            else {}
        )
        for row in selector_rows:
            selector = str(row.get("selector", ""))
            comparator = "MAE-Select" if selector == "DART-ValueSelect" else selector
            cssa = cssa_by_comparator.get(comparator, {})
            cssa = cssa if isinstance(cssa, Mapping) else {}
            value_lcbs = cssa.get(
                "target_minus_comparator_value_CSSA_lcb_by_block", {}
            )
            regret_lcbs = cssa.get(
                "comparator_minus_target_regret_CSSA_lcb_by_block", {}
            )
            value_lcbs = value_lcbs if isinstance(value_lcbs, Mapping) else {}
            regret_lcbs = regret_lcbs if isinstance(regret_lcbs, Mapping) else {}
            row.update(
                {
                    "cssa_comparison": (
                        f"DART-ValueSelect_minus_{comparator}"
                        if comparator
                        else None
                    ),
                    "cssa_7_day_value_lcb": value_lcbs.get("7"),
                    "cssa_14_day_value_lcb": value_lcbs.get("14"),
                    "cssa_28_day_value_lcb": value_lcbs.get("28"),
                    "cssa_7_day_regret_lcb": regret_lcbs.get("7"),
                    "cssa_primary_supported": (
                        cssa.get(
                            "primary_value_CSSA_lcb_strictly_above_zero"
                        )
                        is True
                        and cssa.get(
                            "primary_regret_CSSA_lcb_strictly_above_zero"
                        )
                        is True
                    ),
                    "cssa_simultaneous_set_valued": True,
                    "cssa_derived_without_new_solves": True,
                }
            )
        hypotheses["H4d"] = self._h4d(selector_rows)
        probability_rows, decision_rows, regime_rows = self._table_rows(
            selection_evidence,
            locked_evidence,
            selection_metrics,
            locked_metrics,
            selection_decisions,
            locked_decisions,
            selection_daily,
            locked_daily,
            selector_rows,
        )
        tables = self.output_root / "tables"
        probability_path = tables / "dart_probability_quality.csv"
        decision_path = tables / "dart_decision_value.csv"
        methods_path = tables / "dart_selection_methods.csv"
        regime_path = tables / "dart_regime_value.csv"
        probability_fields = (
            "phase",
            "model",
            "mae",
            "rmse",
            "crps",
            "pinball_loss",
            "tail_crps",
            "threshold_weighted_crps",
            "emphasized_tail_threshold_weighted_crps",
            "q05_q95_threshold_weighted_crps",
            "q01_q99_extreme_threshold_weighted_crps",
            "tail_calibration_error",
            "awcrps",
            "awcrps_status",
            "awcrps_undefined_reason",
            "feasibility_conditioned_tail_crps",
            "feasibility_conditioned_tail_crps_status",
            "feasibility_conditioned_tail_crps_undefined_reason",
            "feasibility_conditioned_q05_q95_tail_crps",
            "feasibility_conditioned_q05_q95_tail_crps_status",
            "feasibility_conditioned_q05_q95_tail_crps_undefined_reason",
            "feasibility_conditioned_q01_q99_extreme_crps",
            "feasibility_conditioned_q01_q99_extreme_crps_status",
            "feasibility_conditioned_q01_q99_extreme_crps_undefined_reason",
            "economic_action_brier",
            "economic_action_brier_status",
            "economic_action_brier_undefined_reason",
            "action_precision",
            "action_precision_status",
            "action_precision_undefined_reason",
            "action_recall",
            "action_recall_status",
            "action_recall_undefined_reason",
            "action_f1",
            "action_f1_status",
            "action_f1_undefined_reason",
            "wrong_action_loss",
            "wrong_action_loss_status",
            "wrong_action_loss_undefined_reason",
            "missed_opportunity_loss",
            "missed_opportunity_loss_status",
            "missed_opportunity_loss_undefined_reason",
            "wrong_action_rate",
            "wrong_action_rate_status",
            "wrong_action_rate_undefined_reason",
            "missed_opportunity_rate",
            "missed_opportunity_rate_status",
            "missed_opportunity_rate_undefined_reason",
            "wrong_action_economic_loss_usd",
            "wrong_action_economic_loss_usd_status",
            "wrong_action_economic_loss_usd_undefined_reason",
            "missed_opportunity_economic_loss_usd",
            "missed_opportunity_economic_loss_usd_status",
            "missed_opportunity_economic_loss_usd_undefined_reason",
            "q99_event_brier",
            "q99_event_brier_status",
            "q99_event_brier_undefined_reason",
            "q99_pr_auc",
            "q99_pr_auc_status",
            "q99_pr_auc_undefined_reason",
            "q99_precision",
            "q99_precision_status",
            "q99_precision_undefined_reason",
            "q99_recall",
            "q99_recall_status",
            "q99_recall_undefined_reason",
            "q99_f1",
            "q99_f1_status",
            "q99_f1_undefined_reason",
            "q99_miss_rate",
            "q99_miss_rate_status",
            "q99_miss_rate_undefined_reason",
            "spike_f1",
            "spike_f1_status",
            "spike_f1_undefined_reason",
            "spike_pr_auc",
            "spike_pr_auc_status",
            "spike_pr_auc_undefined_reason",
            "interval_coverage_80",
            "interval_width_80",
            "interval_calibration_error_80",
            "interval_coverage_90",
            "interval_width_90",
            "interval_calibration_error_90",
            "energy_score",
            "variogram_score",
            "forl",
            "forl_status",
            "forl_undefined_reason",
            "tail_threshold_contract_sha256",
            "tail_threshold_contract_sha256_status",
            "tail_threshold_contract_sha256_unique_value_count",
            "tail_threshold_contract_sha256_per_record_ledger_sha256",
            "normalized_feasibility_weight_sha256",
            "normalized_feasibility_weight_sha256_status",
            "normalized_feasibility_weight_sha256_unique_value_count",
            "normalized_feasibility_weight_sha256_per_record_ledger_sha256",
            "normalized_feasibility_weight_min",
            "normalized_feasibility_weight_max",
            "normalized_feasibility_weight_sum",
            "physical_flexible_mwh_sha256",
            "physical_flexible_mwh_sha256_status",
            "physical_flexible_mwh_sha256_unique_value_count",
            "physical_flexible_mwh_sha256_per_record_ledger_sha256",
            "physical_flexible_mwh_total",
            "metric_contract_operating_date_count",
            "metric_contract_by_operating_date_sha256",
            "metric_contract_per_record_hash_ledger_sha256",
            "metric_contract_aggregation_order",
            "source_record_count",
            "source_records_sha256",
            "evidence_role",
            "denominator",
        )
        decision_fields = (
            "phase",
            "model",
            "experiment_id",
            "cumulative_incremental_operational_value",
            "mean_daily_incremental_operational_value",
            "equivalent_operating_cost_reduction",
            "operating_cost_reduction_rate",
            "profit_capture_ratio",
            "profit_capture_ratio_status",
            "mean_prescriptive_regret",
            "probability_below_baseline",
            "cvar95_loss",
            "maximum_drawdown",
            "weighted_sla_satisfaction",
            "deadline_violations",
            "infeasible_days",
            "invalid_days",
            "solver_timeout_days",
            "solver_nonzero_gap_days",
            "solver_gap_over_cap_days",
            "solver_missing_gap_days",
            "solver_no_incumbent_days",
            "forecast_plus_solve_latency_seconds",
            "counterfactual",
            "denominator",
            "source_records_sha256",
        )
        method_fields = tuple(selector_rows[0]) if selector_rows else ("selector",)
        regime_fields = (
            "phase",
            "model",
            "experiment_id",
            "regime",
            "day_count",
            "mean_incremental_operational_value",
            "status",
            "counterfactual",
            "denominator",
        )
        self._write_csv(probability_path, probability_rows, probability_fields)
        self._write_csv(decision_path, decision_rows, decision_fields)
        self._write_csv(methods_path, selector_rows, method_fields)
        self._write_csv(regime_path, regime_rows, regime_fields)

        manifests = self.output_root / "manifests"
        replay_manifest_path = manifests / "dart_replay_manifest.json"
        statistics_manifest_path = manifests / "dart_statistical_selection_manifest.json"
        selection_validation = json.loads(
            self.selection_evidence_path.read_text(encoding="utf-8")
        )["validation"]
        locked_validation = json.loads(
            self.locked_evidence_path.read_text(encoding="utf-8")
        )["validation"]
        replay_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "selection_panel": selection_validation,
            "locked_test_panel": locked_validation,
            "experiments": {
                "A1": "DART_ONLY_REPLACEMENT",
                "A2": "FULL_DEPLOYMENT_CHAIN",
                "A3": "DERIVED_FROM_COMPLETED_REPLAY_WITHOUT_NEW_SOLVES",
            },
            "common_exact_three_seed_panel": True,
            "counterfactual": True,
            "realized_commercial_profit_claim": False,
            "candidate_pool_contract": self.candidate_pool_contract,
            "terminal_status_cap": (
                PARTIAL_CANDIDATE_TERMINAL_CAP
                if self.candidate_pool_contract is not None
                else None
            ),
            "created_at_utc": utc_now(),
        }
        atomic_json(replay_manifest_path, replay_manifest)
        statistics_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "selection_years": [2022, 2023, 2024],
            "locked_test_year": 2025,
            "selection_results_by_experiment": freeze[
                "selection_results_by_experiment"
            ],
            "frozen_selectors": frozen_selectors,
            "locked_test_selector_rows": selector_rows,
            "selector_comparison_contract": {
                "primary_selection_experiment": "A2",
                "preregistered_noninferiority_experiment": "A1",
                "a1_noninferiority_epsilon_fraction": (
                    A1_NONINFERIORITY_FRACTION
                ),
                "economic_winner_tolerance_fraction": (
                    ECONOMIC_WINNER_TOLERANCE_FRACTION
                ),
                "primary_comparison": ["MAE-Select", "DART-ValueSelect"],
                "supplementary_comparisons": [
                    "CRPS-Select",
                    "q99-tail-F1-Select",
                    "mean-validation-value-Select",
                ],
                "winner_set_member_rows_preserved": True,
                "diagnostic_tie_break_collapses_scientific_winner_set": False,
                "same_locked_A2_replay": True,
                "derived_without_new_solves": True,
                "selection_updated_from_2025": False,
                "locked_test_selector_rows_sha256": sha256_json(selector_rows),
            },
            "bootstrap_blocks_days": [7, 14, 28],
            "primary_block_days": 7,
            "MCS_before_maximum_LCB": True,
            "no_forced_winner": True,
            "A3_selection": selection_analysis["A3"],
            "A3_locked_test": self._derive_a3(locked_evidence),
            "hypotheses": hypotheses,
            "H4d_optional_boundary": (
                "ENABLED_WITH_SUPPLIED_FROZEN_EVIDENCE"
                if self.config["contextual_selector"]["enabled"]
                else SKIPPED_OPTIONAL_EXTENSION
            ),
            "selection_updated_from_2025": False,
            "candidate_pool_contract": self.candidate_pool_contract,
            "terminal_status_cap": (
                PARTIAL_CANDIDATE_TERMINAL_CAP
                if self.candidate_pool_contract is not None
                else None
            ),
            "created_at_utc": utc_now(),
        }
        atomic_json(statistics_manifest_path, statistics_manifest)
        table_paths = (probability_path, decision_path, methods_path, regime_path)
        final_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                PARTIAL_CANDIDATE_STATUS
                if self.candidate_pool_contract is not None
                else "COMPLETE"
            ),
            "completed": True,
            "downstream_ready": True,
            "experiment_fingerprint": self.experiment_fingerprint,
            "config_sha256": self.config_sha256,
            "chronology": {
                "selection_years": [2022, 2023, 2024],
                "selection_frozen_before_locked_test": True,
                "locked_test_year": 2025,
                "locked_test_run_count": 1,
                "selection_updated_from_2025": False,
            },
            "three_layers": {
                "probabilistic_credibility": "dart_probability_quality.csv",
                "actionability": "dart_probability_quality.csv",
                "decision_value": "dart_decision_value.csv",
                "primary_ranking_layer": "decision_value",
                "primary_selection_experiment": "A2",
                "A1_role": "PREREGISTERED_NONINFERIORITY_GATE",
            },
            "frozen_selectors": frozen_selectors,
            "hypotheses": hypotheses,
            "A3_derived_without_new_solves": True,
            "perfect_information_semantics": "UNATTAINABLE_UPPER_BOUND_ONLY",
            "counterfactual": True,
            "realized_commercial_profit_claim": False,
            "candidate_pool_contract": self.candidate_pool_contract,
            "terminal_status_cap": (
                PARTIAL_CANDIDATE_TERMINAL_CAP
                if self.candidate_pool_contract is not None
                else None
            ),
            "outputs": [
                {"path": path.resolve().as_posix(), "sha256": sha256_file(path)}
                for path in (*table_paths, replay_manifest_path, statistics_manifest_path)
            ],
            "created_at_utc": utc_now(),
        }
        atomic_json(self.final_manifest_path, final_manifest)
        ledger = json.loads(self.locked_ledger_path.read_text(encoding="utf-8"))
        ledger.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": utc_now(),
                "run_count": 1,
                "selection_updated_from_2025": False,
                "locked_evidence_sha256": sha256_file(self.locked_evidence_path),
                "final_manifest_sha256": sha256_file(self.final_manifest_path),
            }
        )
        atomic_json(self.locked_ledger_path, ledger)
        output_paths = (
            *table_paths,
            replay_manifest_path,
            statistics_manifest_path,
            self.final_manifest_path,
        )
        self._finish_stage(
            "finalization",
            input_sha,
            output_paths,
            details={
                "status": "COMPLETE",
                "locked_test_run_count": 1,
                "selection_updated_from_2025": False,
            },
        )
        state = self._load_state()
        state["status"] = "COMPLETE"
        state["updated_at_utc"] = utc_now()
        atomic_json(self.state_path, state)
        return final_manifest

    def complete_locked_test(self, evidence: DARTPhaseEvidence) -> dict[str, Any]:
        """Ingest supplied locked evidence and finalize without reselection."""

        self.ingest_locked_test_evidence(evidence)
        return self.finalize()


__all__ = [
    "DARTExperimentContractError",
    "DARTPhaseEvidence",
    "DARTValueSelectExperiment",
    "EvidenceContractError",
    "LockedTestGuardError",
    "REGIMES",
    "SCHEMA_VERSION",
    "StageTransitionError",
]
