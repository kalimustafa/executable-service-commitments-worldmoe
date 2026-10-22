from __future__ import annotations

import csv
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from dcvpp.core import atomic_json, load_yaml, sha256_file, sha256_json, utc_now


SCHEMA_VERSION = "dcvpp.v1_dart_readiness.v12"
POLICY_SCHEMA_VERSION = "dcvpp.v1_dart_testing_selection_policy.v12"
AMENDMENT_SCHEMA_VERSION = "dcvpp.v1_dart_testing_selection_amendment.v12"
ANCHOR_SNAPSHOT_SCHEMA_VERSION = "dcvpp.dart_v3_pre_oos_anchor_snapshot.v1"
ANCHOR_AMENDMENT_STATUS = (
    "AMENDED_PRE_OOS_ANCHOR_ONLY_CHECKPOINT_SELECTION_AFTER_FAIL_CLOSED_REPLAY_DEFECT"
)
RECOVERY_MANIFEST_SCHEMA_VERSION = "paper_forecast_v1_recovery_manifest/v1"
STUDY_SEEDS = (20260809, 20260819, 20260829)
LEGACY_EXCLUDED_SEEDS = (20260908, 20260918)
CANDIDATES = (
    "seasonal_naive",
    "persistence",
    "lightgbm_quantile",
    "lstm",
    "dlinear",
    "patchtst",
    "itransformer",
    "static_graph_patch_transformer",
    "dgpt_e",
)
FORECAST_TRAINING_SEED_ROLES = {
    "deterministic_one_fit": ("seasonal_naive", "persistence"),
    "statistical_fixed_one_fit": ("lightgbm_quantile",),
    "neural_three_fit": (
        "lstm",
        "dlinear",
        "patchtst",
        "itransformer",
        "static_graph_patch_transformer",
        "dgpt_e",
    ),
}
REQUIRED_IMPLEMENTATION_CODE_PATHS = (
    "src/dcvpp/statistics/block_bootstrap.py",
    "src/dcvpp/statistics/model_confidence_set.py",
    "src/dcvpp/evaluation/dart_metrics.py",
    "src/dcvpp/evaluation/action_weighted_crps.py",
    "src/dcvpp/evaluation/profit_evaluator.py",
    "src/dcvpp/evaluation/dart_value_select.py",
    "src/dcvpp/optimization/replay_engine.py",
    "src/dcvpp/optimization/stochastic_milp.py",
    "src/dcvpp/experiments/dart_checkpoint_selection.py",
    "src/dcvpp/experiments/dart_checkpoint_training.py",
    "src/dcvpp/experiments/dart_checkpoint_replay_compat.py",
    "src/dcvpp/experiments/dart_training.py",
    "src/dcvpp/experiments/dart_forecast_oos.py",
    "src/dcvpp/experiments/paper_training.py",
    "src/dcvpp/models/energy/dgpt_e.py",
    "src/dcvpp/experiments/dart_replay.py",
    "src/dcvpp/experiments/dart_real_evidence.py",
    "src/dcvpp/experiments/dart_value_select.py",
    "src/dcvpp/experiments/dart_pipeline.py",
    "src/dcvpp/paper_tables_reports.py",
    "src/dcvpp/paper_artifacts.py",
    "scripts/run_dart_primary_selection.py",
)
CANDIDATE_TRAINING_ROLE = {
    candidate: role
    for role, candidates in FORECAST_TRAINING_SEED_ROLES.items()
    for candidate in candidates
}
REQUIRED_COMPLETION_GATES = (
    "v1_manifest_immutable",
    "v1_output_hashes_verified",
    "v1_recovery_incident_and_lineage_verified",
    "v1_recovery_locked_metric_nonconsumption_verified",
    "forecast_training_seed_roles_verified",
    "all_candidate_three_replay_seed_contract_frozen",
    "validation_only_projection",
    "required_metric_denominator_matrix_verified",
    "audited_dgpt_native_issue_chronology_verified",
    "audited_dgpt_cutoff_issue_target_order_verified",
    "audited_dgpt_timezone_and_issue_profile_verified",
    "audited_dgpt_exact_da_rt_key_panel_verified",
    "testing_selection_contract_frozen_v12",
    "hierarchical_checkpoint_selection_contract_frozen",
    "pre_oos_anchor_import_contract_frozen",
    "v4_full_epoch_checkpoint_contract_frozen",
    "formal_execution_order_contract_frozen",
    "testing_selection_implementation_hashes_verified",
    "centered_studentized_global_mcs_contract_frozen",
    "solver_timeout_and_nonzero_gap_contract_frozen",
    "native_rt_cross_head_auxiliary_only",
    "native_rt_excluded_from_dart_selection",
    "common_da_gate_retraining_required",
    "locked_metric_loader_nonconsumption_verified",
)
VALIDATION_METRIC_FIELDS = (
    "mae",
    "rmse",
    "crps_quantile_approximation",
    "mean_pinball",
    "mean_weighted_pinball",
    "absolute_calibration_error",
    "coverage_80",
    "coverage_90",
    "mean_width_80",
    "mean_width_90",
    "spike_precision",
    "spike_recall",
    "spike_f1",
    "quantile_crossing_count",
    "quantile_crossing_rate",
    "wape",
    "n",
)
LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "crps_quantile_approximation",
    "mean_pinball",
    "mean_weighted_pinball",
    "absolute_calibration_error",
    "mean_width_80",
    "mean_width_90",
    "quantile_crossing_count",
    "quantile_crossing_rate",
    "wape",
}
HIGHER_IS_BETTER = {"spike_precision", "spike_recall", "spike_f1"}
POLICY_SECTIONS = (
    "protocol_amendment",
    "sign_contract",
    "candidate_contract",
    "execution_order",
    "randomization_contract",
    "forecast_issue_contract",
    "chronology",
    "checkpoint_selection",
    "probabilistic_credibility",
    "actionability",
    "decision_value",
    "identical_replay",
    "decision_pipeline_contract",
    "feasibility",
    "bootstrap",
    "model_confidence_set",
    "selection",
    "experiments",
    "selector_comparators",
    "contextual_selector",
    "claim_boundaries",
    "implementation_freeze",
    "pre_dart_v1_readiness",
    "aggregation_contract",
    "selection_freeze_contract",
    "locked_test_exposure_disclosure",
    "hypothesis_contract",
)
EXPOSURE_DISCLOSURE = (
    "DART_2025_is_algorithmically_once_locked_after_the_protocol_amendment_"
    "but_prior_auxiliary_v1_2025_results_existed"
)
AMENDMENT_REASONS = (
    "validation_only_loader_must_not_materialize_locked_test_metrics",
    "require_feature_cutoff_le_issue_le_target",
    "require_declared_native_issue_profile_and_market_timezone",
    "require_complete_metric_denominator_matrix",
    "forbid_silent_da_rt_inner_join",
    "separate_forecast_fit_seed_roles_from_common_replay_seed_slots",
    "scope_native_chronology_claim_to_audited_dgpt_artifacts",
    "freeze_h4_and_a3_testing_rules_before_results",
    "align_model_confidence_set_with_centered_studentized_global_range_implementation",
    "require_explicit_zero_solver_timeout_and_nonzero_gap_day_counts",
    "require_hash_verified_mixed_lineage_recovery_after_the_disclosed_v1_incident",
    "bind_executable_testing_selection_and_replay_code_hashes",
    "forbid_silent_nonfinite_day_filtering_in_model_confidence_set",
    "narrow_locked_test_claim_to_algorithmic_loader_nonconsumption",
    "replace_prior_confirmatory_panels_with_three_seed_panel_before_formal_dart_training",
    "replace_impossible_native_cross_task_denominator_equality_with_an_explicit_complete_day_common_panel_ledger",
    "define_the_96_step_operating_date_as_a_canonical_UTC_day_while_retaining_America_New_York_market_time_metadata",
    "store_every_readiness_evidence_path_repository_relative_for_containment_audit",
    "make_A2_full_deployment_value_primary_and_A1_a_noninferiority_gate",
    "replace_outcome_reweighted_tail_diagnostic_with_proper_zone_q01_q99_threshold_weighted_CRPS",
    "normalize_candidate_independent_actionability_weight_to_zero_one",
    "retain_economic_tolerance_winner_sets_and_report_set_valued_CSSA",
    "compare_minimum_MAE_and_DART_ValueSelect_as_the_two_primary_locked_rules",
    "replace_single_seed_minimum_validation_pinball_checkpoint_with_three_seed_hierarchical_epoch_confidence_set",
    "freeze_112_day_internal_holdout_and_synchronized_three_seed_group_early_stopping",
    "replace_strict_cross_model_5_epoch_rotation_with_hybrid_40_epoch_warmup_and_10_epoch_active_group_rounds",
    "defer_ablations_and_oracle_variants_until_full_date_nine_candidate_training_completes",
    "restrict_neural_epoch_candidates_to_exact_common_hash_verified_persisted_five_epoch_anchors",
    "reuse_v3_training_anchors_read_only_without_copy_hardlink_rewrite_or_mutable_latest_alias",
    "record_fail_closed_cuda_replay_nondeterminism_before_any_oos_materialization_without_using_performance_evidence",
)
EXPECTED_V10_FREEZE_SHA256 = (
    "ca7838b0600a378cfe4fe2c1551da49c5441e19f73d6e8c0ee71d64fbedcb9a9"
)
EXPECTED_V10_AMENDMENT_SHA256 = (
    "60e3077a42e281c9404de90c8ee398d8fba44a6b360a5cad9a9cf71482e591ee"
)
EXPECTED_V10_FREEZE_BYTE_COUNT = 50355
EXPECTED_V10_AMENDMENT_BYTE_COUNT = 3920
EXPECTED_PRE_V5_FREEZE_SHA256 = (
    "339077e754e1616b3e33f85bca06a53ddd38929331a248aa6be1234fff9cd0f8"
)
EXPECTED_PRE_V5_AMENDMENT_SHA256 = (
    "817e2d6a8b1b0502859c908ffef3c8c86d0e396ed5622b662a40f3cabaf1849e"
)
EXPECTED_V11_FREEZE_SHA256 = (
    "92ca0de71db4f2cd59115f3741771cc71befb16fbc7f7e9dbf0149feb4131d21"
)
EXPECTED_V11_FROZEN_POLICY_SHA256 = (
    "f993b46f367c31d41987029cb86104e999ff2b646c739ccec84881f369c28d5f"
)
EXPECTED_V11_AMENDMENT_SHA256 = (
    "47082bf4a040889f1241ab2257b61591d1a104e0ab252b76f947af7295812510"
)
EXPECTED_V11_READINESS_SHA256 = (
    "e8ad0633d6b88b55294adc30d00d03e0ab6769f0986ffb981e0efb6e4745c2b5"
)
EXPECTED_V11_FREEZE_BYTE_COUNT = 61467
EXPECTED_V11_AMENDMENT_BYTE_COUNT = 5002
EXPECTED_V11_READINESS_BYTE_COUNT = 12247
_EXPLICIT_ZONE_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$", re.IGNORECASE)


class V1DARTReadinessError(RuntimeError):
    pass


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved_root = root.resolve()
    resolved = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise V1DARTReadinessError(
            f"readiness evidence path escapes the repository root: {value}"
        ) from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise V1DARTReadinessError(f"required JSON is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V1DARTReadinessError(f"JSON root must be an object: {path}")
    return value


def _json_string_end(text: str, start: int) -> int:
    if start >= len(text) or text[start] != '"':
        raise V1DARTReadinessError("JSON string scanner received a non-string token")
    index = start + 1
    escaped = False
    while index < len(text):
        character = text[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            return index + 1
        index += 1
    raise V1DARTReadinessError("unterminated JSON string")


def _json_value_end(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        raise V1DARTReadinessError("JSON object key has no value")
    if text[index] == '"':
        return _json_string_end(text, index)
    if text[index] in "[{":
        stack: list[str] = []
        scan = index
        while scan < len(text):
            character = text[scan]
            if character == '"':
                scan = _json_string_end(text, scan)
                continue
            if character == "{":
                stack.append("}")
            elif character == "[":
                stack.append("]")
            elif character in "}]":
                if not stack or character != stack[-1]:
                    raise V1DARTReadinessError("mismatched JSON composite delimiter")
                stack.pop()
                if not stack:
                    return scan + 1
            scan += 1
        raise V1DARTReadinessError("unterminated JSON composite value")
    while index < len(text) and text[index] not in ",}]":
        index += 1
    return index


def _redact_json_key_values(text: str, forbidden_key: str) -> str:
    """Replace a forbidden key's raw value with null before JSON decoding.

    The locked-test metric token is located lexically and its value span is
    skipped without passing that span to ``json.loads``.  Thus only validation
    fields are ever materialized as Python values.
    """

    pieces: list[str] = []
    cursor = 0
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        string_end = _json_string_end(text, index)
        after = string_end
        while after < len(text) and text[after].isspace():
            after += 1
        if after >= len(text) or text[after] != ":":
            index = string_end
            continue
        key = json.loads(text[index:string_end])
        if key != forbidden_key:
            index = string_end
            continue
        value_start = after + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        value_end = _json_value_end(text, value_start)
        pieces.extend((text[cursor:value_start], "null"))
        cursor = value_end
        index = value_end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _purge_key(value: Any, forbidden_key: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _purge_key(member, forbidden_key)
            for key, member in value.items()
            if str(key) != forbidden_key
        }
    if isinstance(value, list):
        return [_purge_key(member, forbidden_key) for member in value]
    return value


def _read_validation_only_leaderboard(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise V1DARTReadinessError(f"required validation leaderboard is absent: {path}")
    raw = path.read_text(encoding="utf-8")
    sanitized = _redact_json_key_values(raw, "locked_test_metrics")
    value = json.loads(sanitized)
    if not isinstance(value, dict):
        raise V1DARTReadinessError(f"leaderboard root must be an object: {path}")
    validation_only = _purge_key(value, "locked_test_metrics")
    if "locked_test_metrics" in json.dumps(validation_only, sort_keys=True):
        raise V1DARTReadinessError("locked-test metric field survived validation-only loading")
    return dict(validation_only), sha256_json(validation_only)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    temporary.replace(path)


def _code_sha256() -> str:
    return sha256_file(Path(__file__))


def _checkpoint_selection_contract_is_frozen(config: Mapping[str, Any]) -> bool:
    checkpoint = config.get("checkpoint_selection")
    if not isinstance(checkpoint, Mapping):
        return False
    training = checkpoint.get("neural_training")
    holdout = checkpoint.get("pre_cutoff_internal_holdout")
    thresholds = checkpoint.get("tail_thresholds")
    health = checkpoint.get("epoch_health_gate")
    coverage = checkpoint.get("simultaneous_coverage_gate")
    scores = checkpoint.get("score_confidence_sets")
    inference = checkpoint.get("inference")
    outer = checkpoint.get("outer_selection_contract")
    candidate_grid = checkpoint.get("candidate_epoch_grid")
    if not all(
        isinstance(value, Mapping)
        for value in (
            training,
            holdout,
            thresholds,
            health,
            coverage,
            scores,
            inference,
            outer,
            candidate_grid,
        )
    ):
        return False
    scheduler = training.get("cross_model_scheduler")
    if not isinstance(scheduler, Mapping):
        return False
    return bool(
        checkpoint.get("schema_version")
        == "dcvpp.dart_checkpoint_selection_policy.v12"
        and checkpoint.get("scope") == "within_model_neural_epoch_selection_only"
        and tuple(checkpoint.get("applies_to", ()))
        == FORECAST_TRAINING_SEED_ROLES["neural_three_fit"]
        and tuple(checkpoint.get("applies_separately_by", ()))
        == ("annual_origin", "target_head")
        and tuple(checkpoint.get("study_seeds", ())) == STUDY_SEEDS
        and candidate_grid.get("source") == "exact_common_persisted_training_anchors"
        and tuple(candidate_grid.get("applies_to", ()))
        == ("provisional_assessment", "final_selection")
        and int(candidate_grid.get("anchor_interval_epochs", 0)) == 5
        and candidate_grid.get("positive_epoch_required") is True
        and candidate_grid.get("exact_anchor_required_for_all_three_seeds") is True
        and candidate_grid.get("eligible_epoch_formula")
        == "E_A_intersection_of_positive_epochs_divisible_by_5_with_hash_verified_persisted_anchor_for_each_study_seed"
        and candidate_grid.get("non_anchor_epoch_records_role")
        == "audit_only_not_selection_candidates"
        and candidate_grid.get("selected_checkpoint_source")
        == "exact_persisted_anchor_payload_no_replay_training"
        and candidate_grid.get("between_anchor_replay_forbidden") is True
        and candidate_grid.get("earliest_surviving_anchor_tie_rule") is True
        and int(training.get("maximum_epoch_count", 0)) == 100
        and int(training.get("minimum_epoch_count", 0)) == 40
        and training.get("every_completed_epoch_recorded") is True
        and training.get("v4_every_completed_epoch_full_immutable_checkpoint_required")
        is True
        and int(training.get("v4_full_epoch_checkpoint_interval_epochs", 0)) == 1
        and tuple(training.get("v4_full_epoch_checkpoint_required_state", ()))
        == (
            "model_state",
            "optimizer_state",
            "gradient_scaler_state",
            "python_numpy_torch_and_cuda_rng_state",
            "dataloader_generator_and_sampler_state",
            "training_health_record",
            "validation_epoch_ledger",
        )
        and training.get(
            "v4_full_epoch_checkpoint_must_be_hash_pinned_and_append_only"
        )
        is True
        and training.get("v4_non_five_epoch_checkpoint_selection_role")
        == "audit_and_resume_only_not_formal_selection_candidate"
        and training.get("v3_source_checkpoint_cadence_unchanged") is True
        and training.get("individual_seed_early_stopping_selects_final_checkpoint")
        is False
        and training.get("common_epoch_required_across_exact_three_seeds") is True
        and training.get("synchronized_three_seed_group_early_stopping") is True
        and int(training.get("provisional_assessment_interval_epochs", 0)) == 5
        and int(training.get("provisional_bootstrap_resamples", 0)) == 200
        and int(training.get("required_consecutive_stable_assessments", 0)) == 3
        and int(training.get("minimum_stability_span_epochs", 0)) == 15
        and tuple(training.get("stable_assessment_requires_unchanged", ()))
        == (
            "provisional_selected_epoch",
            "global_crps_confidence_set",
            "proper_tail_twcrps_confidence_set",
            "selected_epoch_health_status",
            "selected_epoch_simultaneous_80_90_coverage_status",
        )
        and float(
            training.get(
                "maximum_best_mean_global_crps_improvement_for_stability", -1.0
            )
        )
        == 1.0e-8
        and float(
            training.get(
                "maximum_best_mean_proper_tail_twcrps_improvement_for_stability",
                -1.0,
            )
        )
        == 1.0e-8
        and training.get("stop_all_three_seeds_together") is True
        and training.get("final_selection_uses_exact_common_persisted_anchor_panel")
        is True
        and training.get("provisional_selection_uses_exact_common_persisted_anchor_panel")
        is True
        and scheduler.get("strategy")
        == "warmup_each_group_to_40_then_global_assessment_then_active_groups_in_10_epoch_rounds"
        and scheduler.get("scheduling_basis")
        == "efficiency_not_candidate_performance"
        and scheduler.get("group_definition")
        == "candidate_by_annual_origin_by_target_head_exact_three_seed_group"
        and scheduler.get("strict_cross_model_rotation_every_5_epochs") is False
        and int(scheduler.get("warmup_target_epoch", 0)) == 40
        and scheduler.get("every_group_reaches_warmup_target_before_first_global_assessment")
        is True
        and int(scheduler.get("active_group_round_increment_epochs", 0)) == 10
        and scheduler.get("active_group_definition")
        == "group_not_stopped_by_frozen_synchronized_three_seed_rule"
        and scheduler.get("all_active_groups_receive_one_increment_before_next_global_assessment")
        is True
        and int(scheduler.get("within_group_ledger_interval_epochs", 0)) == 5
        and int(
            scheduler.get("within_group_provisional_assessment_interval_epochs", 0)
        )
        == 5
        and tuple(scheduler.get("global_assessment_and_summary_epochs", ()))
        == (40, 50, 60, 70, 80, 90, 100)
        and scheduler.get("stop_decisions_materialized_only_at_global_round_boundaries")
        is True
        and scheduler.get("stable_groups_skipped_in_later_rounds") is True
        and int(scheduler.get("maximum_epoch_count", 0)) == 100
        and scheduler.get("individual_seed_scheduling_or_stopping_forbidden")
        is True
        and int(holdout.get("total_days", 0)) == 112
        and int(holdout.get("checkpoint_calibration_days", 0)) == 56
        and int(holdout.get("checkpoint_selection_days", 0)) == 28
        and int(holdout.get("final_calibration_and_coupling_days", 0)) == 28
        and tuple(holdout.get("chronological_order", ()))
        == (
            "core_training",
            "checkpoint_calibration",
            "checkpoint_selection",
            "final_calibration_and_coupling",
            "annual_cutoff",
        )
        and holdout.get("every_panel_strictly_before_annual_cutoff") is True
        and holdout.get("panel_overlap_forbidden") is True
        and tuple(thresholds.get("quantiles", ())) == (0.01, 0.05, 0.95, 0.99)
        and thresholds.get("source") == "core_training_history_only"
        and thresholds.get("contract_sha256_required") is True
        and health.get("finite_training_loss_required") is True
        and health.get("finite_gradient_diagnostics_required") is True
        and health.get("finite_predictions_required") is True
        and int(health.get("nonfinite_gradient_step_count_required", -1)) == 0
        and float(health.get("maximum_raw_quantile_crossing_rate", -1.0)) == 0.05
        and int(health.get("post_calibration_quantile_crossing_count_required", -1))
        == 0
        and tuple(coverage.get("nominal_levels", ())) == (0.80, 0.90)
        and float(coverage.get("maximum_absolute_coverage_error", -1.0)) == 0.05
        and coverage.get("simultaneous_across_levels") is True
        and tuple(scores.get("ordered_scores", ()))
        == ("global_crps", "proper_q01_q05_q95_q99_threshold_weighted_crps")
        and scores.get("arbitrary_weighted_composite_forbidden") is True
        and inference.get("seed_aggregation")
        == "within_operating_date_mean_across_exact_three_seeds"
        and inference.get("seed_values_are_not_independent_date_samples") is True
        and float(inference.get("confidence_level", 0.0)) == 0.95
        and inference.get("bootstrap") == "circular_moving_block"
        and int(inference.get("block_length_days", 0)) == 7
        and int(inference.get("resamples", 0)) == 1000
        and int(inference.get("seed", 0)) == 20260809
        and inference.get("simultaneous_one_sided_confidence_sets") is True
        and inference.get("exact_common_epoch_and_date_panel_required") is True
        and tuple(checkpoint.get("selection_order", ()))
        == (
            "epoch_health_gate",
            "simultaneous_80_90_coverage_gate",
            "global_crps_epoch_confidence_set",
            "proper_q01_q05_q95_q99_twcrps_epoch_confidence_set",
            "earliest_surviving_epoch",
        )
        and checkpoint.get("tie_rule") == "earliest_epoch"
        and checkpoint.get("empty_set_policy")
        == "fail_annual_origin_target_candidate_cell_closed"
        and tuple(checkpoint.get("forbidden_inputs", ()))
        == (
            "action_metrics",
            "MILP_outputs",
            "operating_cost",
            "profit",
            "regret",
            "CVaR",
            "rolling_evaluation_year_OOS_outcomes",
            "locked_2025_targets_predictions_or_metrics",
        )
        and outer.get("unchanged") is True
        and outer.get("minimum_MAE_comparator_unchanged") is True
        and outer.get("DART_ValueSelect_A2_primary_A1_noninferiority_unchanged")
        is True
    )


def _execution_order_contract_is_frozen(config: Mapping[str, Any]) -> bool:
    execution = config.get("execution_order")
    return bool(
        isinstance(execution, Mapping)
        and execution.get("schema_version") == "dcvpp.dart_execution_order.v1"
        and execution.get("frozen_before_formal_dart_training") is True
        and execution.get("decision_basis")
        == "user_specified_resource_sequence_not_candidate_performance"
        and execution.get("immediate_phase")
        == "formal_full_date_main_and_control_candidate_training"
        and tuple(execution.get("immediate_candidate_pool", ())) == CANDIDATES
        and execution.get("exact_frozen_nine_candidate_pool_required") is True
        and execution.get("all_rolling_origins_and_dates_required") is True
        and execution.get("ablations_deferred_until_immediate_phase_complete") is True
        and execution.get("oracle_variants_deferred_until_immediate_phase_complete")
        is True
        and tuple(execution.get("deferred_families", ()))
        == ("ablations", "ORACLE_variants")
        and execution.get("deferred_families_excluded_from_immediate_run") is True
        and execution.get("oracle_variants_deployable") is False
        and execution.get("performance_informed_reordering_forbidden") is True
    )


def _anchor_only_import_contract(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    readiness = config.get("pre_dart_v1_readiness")
    if not isinstance(readiness, Mapping):
        return None
    amendment = readiness.get("anchor_only_checkpoint_amendment")
    return amendment if isinstance(amendment, Mapping) else None


def _anchor_only_import_contract_is_frozen(config: Mapping[str, Any]) -> bool:
    amendment = _anchor_only_import_contract(config)
    protocol = config.get("protocol_amendment")
    protocol_amendment = (
        protocol.get("anchor_only_checkpoint_amendment")
        if isinstance(protocol, Mapping)
        else None
    )
    if not isinstance(amendment, Mapping) or not isinstance(
        protocol_amendment, Mapping
    ):
        return False
    return bool(
        amendment.get("enabled") is True
        and amendment.get("amendment_role")
        == "performance_independent_recovery_after_fail_closed_pre_oos_cuda_replay_defect"
        and amendment.get("evidence_trigger")
        == "selected_epoch_replay_changed_its_validation_prediction_or_health_record"
        and amendment.get("performance_independent") is True
        and amendment.get("model_selection_performance_reviewed_for_this_amendment")
        is False
        and amendment.get("locked_2025_dart_metrics_consumed") is False
        and amendment.get("rolling_2022_2024_oos_predictions_or_metrics_materialized")
        is False
        and amendment.get("market_replay_milp_action_or_value_results_consumed")
        is False
        and amendment.get("formal_dart_training_anchors_existed") is True
        and int(amendment.get("formal_dart_completed_oos_units", -1)) == 0
        and amendment.get("formal_dart_replay_started") is False
        and amendment.get("source_run_id") == "paper-dart-forecast-v3"
        and amendment.get("destination_run_id") == "paper-dart-forecast-v4"
        and amendment.get("source_reuse_mode")
        == "immutable_hash_pinned_read_only_anchor_import_without_copy_hardlink_or_rewrite"
        and int(amendment.get("source_snapshot_manifest_byte_count", -1))
        == 1_867_213
        and amendment.get("source_snapshot_manifest_file_sha256")
        == "02f767744454d1cef0e0a007106d939a2a312a29e557217242997d5175a81c61"
        and amendment.get("source_snapshot_manifest_sha256")
        == "4a8686597bf9facca86c738220986c7503ee808d8017c060af5b5b57fa848d42"
        and int(amendment.get("source_anchor_record_count", -1)) == 1266
        and int(amendment.get("source_anchor_total_bytes", -1)) == 6_726_342_704
        and amendment.get("source_anchor_record_core_schema")
        == "repository_relative_path_sha256_byte_count_epoch_seed"
        and amendment.get("source_anchor_record_core_order")
        == "ascending_repository_relative_posix_path"
        and amendment.get("source_anchor_record_core_hash_algorithm")
        == "sha256_of_utf8_canonical_json_sort_keys_compact_separators"
        and amendment.get("source_anchor_records_sha256")
        == "43030db511b7aa9591216753e3b76d2c73e7937f99b7d0183fd88c94c5d76a36"
        and amendment.get("source_anchor_expanded_records_sha256")
        == "e250dde398183695acd498c4c38ca4440ae566024e1d751022c28623d9bebcaf"
        and amendment.get("source_training_module_sha256")
        == "8e43fc4f3c61be2aca7f8ec67ffc821a909aef50b23626955ca9caa2ae2d10a7"
        and amendment.get("source_config_sha256")
        == "8f31c15cfa8c2a273223b78b658150a45e9db25ab1beee61d77dea815fe640a9"
        and int(amendment.get("source_selection_stage_manifest_byte_count", -1))
        == 4705
        and amendment.get("source_selection_stage_manifest_sha256")
        == "c02aeb8483890fe7d4d742552d3eb083b30037153a40edd30b1d11a4c786d42c"
        and int(amendment.get("source_scheduler_state_byte_count", -1)) == 2_988_675
        and amendment.get("source_scheduler_state_sha256")
        == "e8f68a6f91f201f191be7d9e6f33db83dc9760837a265d4f5fbaea307c0ec656"
        and amendment.get("prior_run_id") == "paper-dart-v1-readiness-v7"
        and amendment.get("prior_policy_schema_version")
        == "dcvpp.v1_dart_testing_selection_policy.v11"
        and int(amendment.get("prior_freeze_byte_count", -1))
        == EXPECTED_V11_FREEZE_BYTE_COUNT
        and amendment.get("prior_freeze_sha256") == EXPECTED_V11_FREEZE_SHA256
        and amendment.get("prior_frozen_policy_sha256")
        == EXPECTED_V11_FROZEN_POLICY_SHA256
        and amendment.get("prior_amendment_schema_version")
        == "dcvpp.v1_dart_testing_selection_amendment.v11"
        and amendment.get("prior_amendment_status")
        == "AMENDED_PRE_FORMAL_DART_CHECKPOINT_SELECTION"
        and int(amendment.get("prior_amendment_byte_count", -1))
        == EXPECTED_V11_AMENDMENT_BYTE_COUNT
        and amendment.get("prior_amendment_sha256")
        == EXPECTED_V11_AMENDMENT_SHA256
        and int(amendment.get("prior_readiness_manifest_byte_count", -1))
        == EXPECTED_V11_READINESS_BYTE_COUNT
        and amendment.get("prior_readiness_manifest_sha256")
        == EXPECTED_V11_READINESS_SHA256
        and amendment.get("v4_full_immutable_checkpoint_every_completed_epoch_required")
        is True
        and int(amendment.get("v4_full_epoch_checkpoint_interval_epochs", 0)) == 1
        and amendment.get("v4_full_epoch_checkpoint_selection_role")
        == "exact_positive_five_epoch_grid_only"
        and amendment.get("v4_non_five_epoch_checkpoints_are_audit_and_resume_only")
        is True
        and amendment.get("v3_checkpoint_cadence_or_bytes_changed") is False
        and protocol_amendment.get("performance_independent") is True
        and int(
            protocol_amendment.get(
                "formal_dart_completed_oos_units_before_amendment", -1
            )
        )
        == 0
        and protocol_amendment.get("locked_2025_dart_results_read_before_amendment")
        is False
        and protocol_amendment.get("market_value_milp_or_action_metrics_consumed")
        is False
        and protocol_amendment.get(
            "model_architecture_loss_hyperparameters_data_splits_and_seeds_changed"
        )
        is False
        and protocol_amendment.get("outer_DART_ValueSelect_changed") is False
    )


def _verify_hash_pinned_file(
    root: Path,
    *,
    path_value: Any,
    sha_value: Any,
    byte_value: Any | None = None,
    role: str,
) -> dict[str, Any]:
    path = _resolve(root, str(path_value or ""))
    expected_sha = str(sha_value or "")
    if not path.is_file() or len(expected_sha) != 64 or sha256_file(path) != expected_sha:
        raise V1DARTReadinessError(f"hash-pinned {role} changed or is absent")
    if byte_value is not None and path.stat().st_size != int(byte_value):
        raise V1DARTReadinessError(f"hash-pinned {role} byte count changed")
    return {
        "role": role,
        "path": _relative(root, path),
        "sha256": expected_sha,
        "byte_count": path.stat().st_size,
    }


def _verify_anchor_snapshot_and_predecessor(
    root: Path,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    prior_freeze = _verify_hash_pinned_file(
        root,
        path_value=amendment.get("prior_freeze_path"),
        sha_value=amendment.get("prior_freeze_sha256"),
        byte_value=amendment.get("prior_freeze_byte_count"),
        role="v11_policy_freeze",
    )
    prior_amendment = _verify_hash_pinned_file(
        root,
        path_value=amendment.get("prior_amendment_path"),
        sha_value=amendment.get("prior_amendment_sha256"),
        byte_value=amendment.get("prior_amendment_byte_count"),
        role="v11_policy_amendment",
    )
    prior_readiness = _verify_hash_pinned_file(
        root,
        path_value=amendment.get("prior_readiness_manifest_path"),
        sha_value=amendment.get("prior_readiness_manifest_sha256"),
        byte_value=amendment.get("prior_readiness_manifest_byte_count"),
        role="v11_readiness_manifest",
    )
    stage = _verify_hash_pinned_file(
        root,
        path_value=amendment.get("source_selection_stage_manifest_path"),
        sha_value=amendment.get("source_selection_stage_manifest_sha256"),
        byte_value=amendment.get("source_selection_stage_manifest_byte_count"),
        role="v3_pre_oos_selection_stage_manifest",
    )
    scheduler = _verify_hash_pinned_file(
        root,
        path_value=amendment.get("source_scheduler_state_path"),
        sha_value=amendment.get("source_scheduler_state_sha256"),
        byte_value=amendment.get("source_scheduler_state_byte_count"),
        role="v3_checkpoint_scheduler_state",
    )
    prior = _read_json(_resolve(root, str(amendment["prior_freeze_path"])))
    if (
        prior.get("contract", {}).get("schema_version")
        != amendment.get("prior_policy_schema_version")
        or prior.get("frozen_policy_sha256")
        != amendment.get("prior_frozen_policy_sha256")
        or prior.get("contract", {}).get("config_sha256")
        != amendment.get("source_config_sha256")
    ):
        raise V1DARTReadinessError("v11 predecessor policy identity changed")
    prior_ledger = _verify_ledger_fingerprint(
        _resolve(root, str(amendment["prior_amendment_path"])),
        str(amendment["prior_amendment_schema_version"]),
        expected_statuses=(str(amendment["prior_amendment_status"]),),
    )
    if prior_ledger.get("replacement", {}).get("file_sha256") != prior_freeze[
        "sha256"
    ]:
        raise V1DARTReadinessError("v11 predecessor amendment link changed")
    prior_readiness_json = _read_json(
        _resolve(root, str(amendment["prior_readiness_manifest_path"]))
    )
    if (
        prior_readiness_json.get("status") != "COMPLETE"
        or prior_readiness_json.get("run_id") != amendment.get("prior_run_id")
        or prior_readiness_json.get("policy_freeze_file_sha256")
        != prior_freeze["sha256"]
        or prior_readiness_json.get("amendment_ledger_sha256")
        != prior_amendment["sha256"]
    ):
        raise V1DARTReadinessError("v11 predecessor readiness links changed")
    stage_json = _read_json(
        _resolve(root, str(amendment["source_selection_stage_manifest_path"]))
    )
    if (
        stage_json.get("run_id") != amendment.get("source_run_id")
        or stage_json.get("status") != "PARTIAL"
        or int(stage_json.get("completed_unit_count", -1)) != 0
        or int(stage_json.get("failed_unit_count", -1)) != 1
        or stage_json.get("input_fingerprints", {}).get("dart_value_select_config")
        != amendment.get("source_config_sha256")
        or stage_json.get("input_fingerprints", {}).get("v1_readiness_manifest")
        != prior_readiness["sha256"]
    ):
        raise V1DARTReadinessError("v3 fail-closed pre-OOS stage identity changed")
    failures = stage_json.get("failed_units")
    if (
        not isinstance(failures, list)
        or len(failures) != 1
        or failures[0].get("error")
        != "selected epoch replay changed its validation prediction/health record"
    ):
        raise V1DARTReadinessError("v3 fail-closed replay-defect evidence changed")
    snapshot_path = _resolve(root, str(amendment.get("source_snapshot_manifest_path", "")))
    snapshot_record = _verify_hash_pinned_file(
        root,
        path_value=amendment.get("source_snapshot_manifest_path"),
        sha_value=amendment.get("source_snapshot_manifest_file_sha256"),
        byte_value=amendment.get("source_snapshot_manifest_byte_count"),
        role="v3_positive_anchor_snapshot_manifest",
    )
    snapshot = _read_json(snapshot_path)
    try:
        from dcvpp.experiments.dart_checkpoint_replay_compat import (
            validate_v3_anchor_snapshot,
        )

        validated_snapshot = validate_v3_anchor_snapshot(
            snapshot_path,
            repository_root=root,
            strict_paper_counts=True,
        )
    except Exception as exc:
        raise V1DARTReadinessError(
            "v3 anchor snapshot or one of its 1266 source files failed validation"
        ) from exc
    supplied_manifest_sha = snapshot.get("manifest_sha256")
    unsigned = dict(snapshot)
    unsigned.pop("manifest_sha256", None)
    if (
        snapshot.get("schema_version") != ANCHOR_SNAPSHOT_SCHEMA_VERSION
        or snapshot.get("status") != "COMPLETE_HASH_PINNED_READ_ONLY_SOURCE"
        or snapshot.get("source_run_id") != amendment.get("source_run_id")
        or snapshot.get("destination_run_id") != amendment.get("destination_run_id")
        or int(snapshot.get("record_count", -1))
        != int(amendment.get("source_anchor_record_count", -2))
        or int(snapshot.get("total_bytes", -1))
        != int(amendment.get("source_anchor_total_bytes", -2))
        or snapshot.get("records_sha256")
        != amendment.get("source_anchor_records_sha256")
        or snapshot.get("expanded_records_sha256")
        != amendment.get("source_anchor_expanded_records_sha256")
        or validated_snapshot.get("record_core_sha256")
        != amendment.get("source_anchor_records_sha256")
        or int(validated_snapshot.get("record_count", -1))
        != int(amendment.get("source_anchor_record_count", -2))
        or int(validated_snapshot.get("total_bytes", -1))
        != int(amendment.get("source_anchor_total_bytes", -2))
        or int(snapshot.get("completed_oos_unit_count", -1)) != 0
        or snapshot.get("copy_hardlink_or_source_rewrite_performed") is not False
        or not isinstance(supplied_manifest_sha, str)
        or supplied_manifest_sha != amendment.get("source_snapshot_manifest_sha256")
        or sha256_json(unsigned) != supplied_manifest_sha
    ):
        raise V1DARTReadinessError("v3 anchor snapshot scientific contract changed")
    if (
        snapshot.get("source_selection_stage_manifest", {}).get("sha256")
        != stage["sha256"]
        or snapshot.get("source_scheduler_state", {}).get("sha256")
        != scheduler["sha256"]
        or snapshot.get("source_training_module_sha256")
        != amendment.get("source_training_module_sha256")
        or snapshot.get("source_config_sha256") != amendment.get("source_config_sha256")
        or snapshot.get("source_readiness_manifest_sha256")
        != prior_readiness["sha256"]
    ):
        raise V1DARTReadinessError("v3 anchor snapshot predecessor links changed")
    return {
        "status": "VERIFIED_COMPLETE_HASH_PINNED_READ_ONLY_SOURCE",
        "source_run_id": amendment["source_run_id"],
        "destination_run_id": amendment["destination_run_id"],
        "snapshot_manifest_path": _relative(root, snapshot_path),
        "snapshot_manifest_file_sha256": snapshot_record["sha256"],
        "snapshot_manifest_sha256": supplied_manifest_sha,
        "record_count": int(snapshot["record_count"]),
        "total_bytes": int(snapshot["total_bytes"]),
        "records_sha256": snapshot["records_sha256"],
        "every_source_anchor_file_rehashed": True,
        "completed_oos_unit_count": 0,
        "copy_hardlink_or_source_rewrite_performed": False,
        "predecessor_records": [
            prior_freeze,
            prior_amendment,
            prior_readiness,
            stage,
            scheduler,
        ],
    }


def _policy_contract(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_version") != "dcvpp.dart_value_select.v12":
        raise V1DARTReadinessError("DART policy schema is not v12")
    missing = [section for section in POLICY_SECTIONS if section not in config]
    if missing:
        raise V1DARTReadinessError(f"DART policy lacks sections: {missing}")
    readiness = config["pre_dart_v1_readiness"]
    if readiness.get("schema_version") != "dcvpp.pre_dart_v1_readiness.v12":
        raise V1DARTReadinessError("pre-DART readiness schema is not v12")
    configured_gates = tuple(readiness.get("required_completion_gates", ()))
    if configured_gates != REQUIRED_COMPLETION_GATES:
        raise V1DARTReadinessError("pre-DART readiness completion gates changed")
    if (
        readiness.get("formal_run_id") != "paper-dart-v1-readiness-v8"
        or readiness.get("formal_output_root")
        != "artifacts/dart_value_select/v1_readiness/run_id=paper-dart-v1-readiness-v8"
    ):
        raise V1DARTReadinessError("formal v8 readiness destination changed")
    if tuple(config["candidate_contract"]["frozen_deployable_pool"]) != CANDIDATES:
        raise V1DARTReadinessError("pre-DART candidate pool differs from the fixed nine")
    randomization = config["randomization_contract"]
    if tuple(randomization["study_seeds"]) != STUDY_SEEDS:
        raise V1DARTReadinessError("pre-DART seed contract changed")
    if not _checkpoint_selection_contract_is_frozen(config):
        raise V1DARTReadinessError("hierarchical checkpoint-selection contract changed")
    if not _anchor_only_import_contract_is_frozen(config):
        raise V1DARTReadinessError("pre-OOS anchor-only import contract changed")
    if not _execution_order_contract_is_frozen(config):
        raise V1DARTReadinessError("formal DART execution-order contract changed")
    configured_roles = {
        str(role): tuple(map(str, candidates))
        for role, candidates in randomization["forecast_training_seed_roles"].items()
    }
    if configured_roles != FORECAST_TRAINING_SEED_ROLES:
        raise V1DARTReadinessError("forecast training seed roles changed")
    if tuple(readiness.get("accepted_v1_manifest_schema_versions", ())) != (
        RECOVERY_MANIFEST_SCHEMA_VERSION,
    ):
        raise V1DARTReadinessError("v1 recovery manifest schema contract changed")
    if readiness.get("current_run_requires_hash_verified_recovery_lineage") is not True:
        raise V1DARTReadinessError("current v1 run must use hash-verified recovery lineage")
    common_panel = readiness.get("native_validation_common_panel")
    if (
        not isinstance(common_panel, Mapping)
        or common_panel.get("policy")
        != "intersection_after_explicit_complete_operating_day_exclusion_ledger"
        or common_panel.get(
            "exclusions_must_be_candidate_independent_complete_target_days"
        )
        is not True
        or int(common_panel.get("required_zones", 0)) != 4
        or int(common_panel.get("required_intervals_per_day", 0)) != 96
        or common_panel.get("operating_date_timezone") != "UTC"
        or common_panel.get("imputation_forbidden") is not True
        or common_panel.get("cross_task_metric_ranking_forbidden") is not True
        or common_panel.get("task_specific_metrics_remain_within_task_only")
        is not True
    ):
        raise V1DARTReadinessError("native validation common-panel contract changed")
    amendment = readiness.get("post_recovery_structural_missingness_amendment")
    if (
        not isinstance(amendment, Mapping)
        or amendment.get("enabled") is not True
        or amendment.get("three_seed_amendment_enabled") is not True
        or amendment.get("performance_independent") is not True
        or amendment.get("prior_run_id") != "paper-dart-v1-readiness-v6"
        or amendment.get("prior_policy_schema_version")
        != "dcvpp.v1_dart_testing_selection_policy.v10"
        or int(amendment.get("prior_freeze_byte_count", -1))
        != EXPECTED_V10_FREEZE_BYTE_COUNT
        or amendment.get("prior_freeze_sha256") != EXPECTED_V10_FREEZE_SHA256
        or amendment.get("prior_amendment_schema_version")
        != "dcvpp.v1_dart_testing_selection_amendment.v10"
        or amendment.get("prior_amendment_status")
        != "AMENDED_POST_RESULT_STRUCTURAL_MISSINGNESS_AND_THREE_SEED_PANEL"
        or int(amendment.get("prior_amendment_byte_count", -1))
        != EXPECTED_V10_AMENDMENT_BYTE_COUNT
        or amendment.get("prior_amendment_sha256")
        != EXPECTED_V10_AMENDMENT_SHA256
    ):
        raise V1DARTReadinessError("post-recovery three-seed amendment changed")
    method_amendment = readiness.get("validation_informed_selection_method_amendment")
    if (
        not isinstance(method_amendment, Mapping)
        or method_amendment.get("enabled") is not True
        or method_amendment.get("auxiliary_validation_metrics_were_reviewed") is not True
        or method_amendment.get("locked_2025_dart_metrics_consumed") is not False
        or method_amendment.get("formal_dart_training_started") is not False
        or method_amendment.get("formal_dart_replay_started") is not False
        or method_amendment.get("formal_dart_selection_results_existed") is not False
        or method_amendment.get("requires_new_dart_forecast_replay_value_select_run_ids")
        is not True
    ):
        raise V1DARTReadinessError("validation-informed selection amendment changed")
    checkpoint_amendment = config["protocol_amendment"].get(
        "checkpoint_selection_amendment"
    )
    if (
        not isinstance(checkpoint_amendment, Mapping)
        or checkpoint_amendment.get("formal_dart_training_started_before_amendment")
        is not False
        or checkpoint_amendment.get("formal_dart_replay_started_before_amendment")
        is not False
        or checkpoint_amendment.get("locked_2025_dart_results_read_before_amendment")
        is not False
        or checkpoint_amendment.get("outer_DART_ValueSelect_changed") is not False
    ):
        raise V1DARTReadinessError("checkpoint-selection amendment boundary changed")
    anchor_protocol_amendment = config["protocol_amendment"].get(
        "anchor_only_checkpoint_amendment"
    )
    if (
        not isinstance(anchor_protocol_amendment, Mapping)
        or anchor_protocol_amendment.get("evidence_trigger")
        != "fail_closed_cuda_selected_epoch_replay_nondeterminism_before_any_oos_materialization"
        or anchor_protocol_amendment.get("performance_independent") is not True
        or int(
            anchor_protocol_amendment.get(
                "formal_dart_completed_oos_units_before_amendment", -1
            )
        )
        != 0
        or anchor_protocol_amendment.get(
            "rolling_2022_2024_oos_outcomes_consumed_before_amendment", False
        )
        is not False
        or anchor_protocol_amendment.get("formal_dart_replay_started_before_amendment")
        is not False
        or anchor_protocol_amendment.get(
            "locked_2025_dart_results_read_before_amendment"
        )
        is not False
        or anchor_protocol_amendment.get(
            "market_value_milp_or_action_metrics_consumed"
        )
        is not False
        or anchor_protocol_amendment.get(
            "model_architecture_loss_hyperparameters_data_splits_and_seeds_changed"
        )
        is not False
        or anchor_protocol_amendment.get("outer_DART_ValueSelect_changed") is not False
    ):
        raise V1DARTReadinessError("anchor-only protocol amendment boundary changed")
    if (
        readiness.get("recovery_run_kind") != "paper_forecast_v1_integrity_recovery"
        or readiness.get("recovery_measurement_relationship")
        != "not_measured_by_this_study"
    ):
        raise V1DARTReadinessError("v1 recovery semantic contract changed")
    mcs = config["model_confidence_set"]
    if (
        mcs.get("statistic") != "centered_studentized_global_loss_range"
        or mcs.get("bootstrap_centering")
        != "observed_pairwise_mean_loss_differentials"
        or mcs.get("studentization")
        != "pairwise_moving_block_bootstrap_standard_error"
        or mcs.get("sequential_step_down") is not True
        or mcs.get("uncorrected_pairwise_tests_forbidden") is not True
    ):
        raise V1DARTReadinessError("centered global MCS contract changed")
    feasibility = config["feasibility"]
    if (
        feasibility.get("require_zero_solver_timeout_days") is not True
        or feasibility.get("require_zero_solver_nonzero_gap_days") is not True
        or feasibility.get("solver_timeout_day_field") != "solver_timeout_days"
        or feasibility.get("solver_nonzero_gap_day_field")
        != "solver_nonzero_gap_days"
        or feasibility.get("solver_feasibility_rule")
        != "timeout_or_nonzero_gap_or_nonfeasible_termination_is_infeasible"
    ):
        raise V1DARTReadinessError("solver timeout/nonzero-gap contract changed")
    implementation = config["implementation_freeze"]
    if (
        implementation.get("schema_version")
        != "dcvpp.dart_testing_selection_implementation_freeze.v4"
        or implementation.get("require_hash_match_before_reading_selection_or_locked_results")
        is not True
        or implementation.get("exact_paired_day_panel_required") is not True
        or implementation.get("silent_nonfinite_day_filtering_forbidden") is not True
    ):
        raise V1DARTReadinessError("testing/selection implementation freeze changed")
    if tuple(implementation.get("required_code_paths", ())) != (
        REQUIRED_IMPLEMENTATION_CODE_PATHS
    ):
        raise V1DARTReadinessError("implementation hash path set changed")
    probabilistic = config["probabilistic_credibility"]
    if (
        tuple(probabilistic.get("tail_quantiles", ())) != (0.01, 0.05, 0.95, 0.99)
        or probabilistic.get("tail_threshold_scope")
        != "per_evaluation_year_and_nyiso_zone_common_across_candidates"
        or probabilistic.get("threshold_weighted_crps_definition")
        != "integral_w_u_times_F_u_minus_indicator_y_le_u_squared_du"
    ):
        raise V1DARTReadinessError("proper q01/q05/q95/q99 tail-score contract changed")
    actionability = config["actionability"]
    if (
        tuple(actionability.get("feasible_flexibility_range", ())) != (0.0, 1.0)
        or actionability.get("candidate_independent_feasibility_hash_required")
        is not True
    ):
        raise V1DARTReadinessError("normalized actionability contract changed")
    selection = config["selection"]
    if (
        selection.get("primary_experiment") != "A2_full_deployment_chain"
        or float(
            selection.get("A1_noninferiority", {}).get(
                "epsilon_fraction_of_mean_baseline_daily_cost", -1.0
            )
        )
        != 0.005
        or float(
            selection.get(
                "winner_economic_tolerance_fraction_of_mean_baseline_daily_cost",
                -1.0,
            )
        )
        != 0.005
        or selection.get("no_forced_winner_when_indistinguishable") is not True
    ):
        raise V1DARTReadinessError("A2/A1 economic winner-set contract changed")
    main_comparators = config["selector_comparators"].get("main_paper")
    if tuple(main_comparators or ()) != (
        "minimum_mae_with_crps_as_deterministic_tie_break_only",
        "DART-ValueSelect",
    ):
        raise V1DARTReadinessError("two-method main selector comparison changed")
    implementation_hashes: dict[str, str] = {}
    for raw_path in implementation.get("required_code_paths", ()):
        relative_path = Path(str(raw_path)).as_posix()
        code_path = config_path.resolve().parents[1] / relative_path
        if not code_path.is_file():
            raise V1DARTReadinessError(
                f"testing/selection implementation file is absent: {relative_path}"
            )
        implementation_hashes[relative_path] = sha256_file(code_path)
    replay = randomization["decision_replay_seed_contract"]
    if (
        tuple(replay.get("study_seeds", ())) != STUDY_SEEDS
        or replay.get("every_candidate_uses_all_three_seed_slots") is not True
        or tuple(replay.get("candidate_pool", ())) != CANDIDATES
    ):
        raise V1DARTReadinessError("all-candidate three-slot replay contract changed")
    native = readiness["native_issue_contract"]
    for task, minimum_lead in (("nyiso_da", 86400), ("nyiso_rt", 300)):
        profile = native.get(task)
        if not isinstance(profile, Mapping):
            raise V1DARTReadinessError(f"native issue profile is absent: {task}")
        if (
            profile.get("audited_candidate") != "dgpt_e"
            or profile.get("artifact") != f"{task}_dgpt_e_quantiles.parquet"
            or profile.get("storage_timezone") != "UTC"
            or profile.get("market_timezone") != "America/New_York"
            or int(profile.get("minimum_issue_lead_seconds", -1)) != minimum_lead
        ):
            raise V1DARTReadinessError(f"native issue profile changed: {task}")
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "purpose": "freeze_validation_informed_testing_and_selection_rules_before_formal_dart_training_replay_or_locked_test",
        "config_path": config_path.resolve().relative_to(
            config_path.resolve().parents[1]
        ).as_posix(),
        "config_sha256": sha256_file(config_path),
        "evaluator_code_sha256": _code_sha256(),
        "policy_sections": {
            section: deepcopy(config[section]) for section in POLICY_SECTIONS
        },
        "implementation_code_hashes": implementation_hashes,
        "evaluation_split": "validation",
        "evaluation_calendar_years": [2024],
        "evaluation_year_timezone": "UTC",
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "audited_chronology_scope": {
            "candidate": "dgpt_e",
            "tasks": ["nyiso_da", "nyiso_rt"],
            "claim": "audited_dgpt_native_issue_artifacts_only",
        },
        "forecast_training_seed_roles": {
            role: list(candidates)
            for role, candidates in FORECAST_TRAINING_SEED_ROLES.items()
        },
        "all_candidate_replay_seed_slots": list(STUDY_SEEDS),
        "native_rt_cross_head_label": "AUXILIARY_NOT_DA_GATE_ELIGIBLE",
        "native_rt_selection_eligible": False,
        "formal_dart_requires_common_da_gate_retraining": True,
        "performance_threshold_blocks_dart_training": False,
        "integrity_or_chronology_failure_blocks_dart_training": True,
        "locked_test_exposure_disclosure": EXPOSURE_DISCLOSURE,
        "post_result_structural_missingness_and_three_seed_amendment": True,
        "validation_informed_selection_method_amendment": True,
        "hierarchical_checkpoint_selection_amendment": True,
        "anchor_only_checkpoint_amendment": True,
        "anchor_only_checkpoint_amendment_performance_independent": True,
        "anchor_only_checkpoint_amendment_completed_oos_units": 0,
        "v4_full_epoch_checkpoint_contract_frozen": True,
        "formal_execution_order_contract_frozen": True,
        "auxiliary_validation_metrics_were_reviewed": True,
        "locked_2025_dart_metrics_consumed_for_amendment": False,
        "formal_dart_training_anchors_existed_at_anchor_amendment": True,
        "formal_dart_completed_oos_units_at_anchor_amendment": 0,
        "formal_dart_replay_existed_at_anchor_amendment": False,
        "amendment_performance_independent": bool(
            readiness.get("post_recovery_structural_missingness_amendment", {}).get(
                "performance_independent"
            )
        ),
    }


def _verify_ledger_fingerprint(
    path: Path,
    expected_schema: str,
    *,
    expected_statuses: Sequence[str] = ("AMENDED_PRE_RESULT",),
) -> dict[str, Any]:
    ledger = _read_json(path)
    if (
        ledger.get("schema_version") != expected_schema
        or ledger.get("status") not in tuple(expected_statuses)
    ):
        raise V1DARTReadinessError(f"invalid policy amendment ledger: {path}")
    expected = ledger.get("ledger_fingerprint")
    unsigned = dict(ledger)
    unsigned.pop("ledger_fingerprint", None)
    if not isinstance(expected, str) or sha256_json(unsigned) != expected:
        raise V1DARTReadinessError(f"policy amendment ledger fingerprint changed: {path}")
    return ledger


def _verify_pre_v5_freeze_history(root: Path, destination: Path) -> list[dict[str, Any]]:
    """Verify the preserved v1 -> v4 chain before later seed-panel amendments."""

    if sha256_file(destination) != EXPECTED_PRE_V5_FREEZE_SHA256:
        raise V1DARTReadinessError("the expected pre-v5 v4 freeze changed")
    existing = _read_json(destination)
    if existing.get("contract", {}).get("schema_version") != (
        "dcvpp.v1_dart_testing_selection_policy.v4"
    ):
        raise V1DARTReadinessError("the expected pre-v5 freeze is not policy v4")
    verified = _verify_frozen_chain_records(
        root, existing.get("transitive_superseded_chain")
    )
    v4_ledger_path = destination.parent / "testing_selection_policy_amendment_v4.json"
    if (
        not v4_ledger_path.is_file()
        or sha256_file(v4_ledger_path) != EXPECTED_PRE_V5_AMENDMENT_SHA256
    ):
        raise V1DARTReadinessError("the pre-v5 v4 amendment ledger changed")
    v4 = _verify_ledger_fingerprint(
        v4_ledger_path, "dcvpp.v1_dart_testing_selection_amendment.v4"
    )
    if (
        v4.get("replacement", {}).get("file_sha256")
        != EXPECTED_PRE_V5_FREEZE_SHA256
        or v4.get("replacement", {}).get("schema_version")
        != "dcvpp.v1_dart_testing_selection_policy.v4"
    ):
        raise V1DARTReadinessError("the pre-v5 v4 amendment links changed")
    verified.append(
        {
            "role": "v4_amendment_ledger",
            "path": _relative(root, v4_ledger_path),
            "sha256": EXPECTED_PRE_V5_AMENDMENT_SHA256,
        }
    )
    return verified


def _verify_frozen_chain_records(
    root: Path,
    records: Any,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise V1DARTReadinessError("amended freeze lacks transitive history")
    verified: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise V1DARTReadinessError("freeze history contains a non-object record")
        path = _resolve(root, str(record.get("path", "")))
        expected = str(record.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected:
            raise V1DARTReadinessError("a transitive freeze-history artifact changed")
        if record.get("byte_count") is not None and path.stat().st_size != int(
            record["byte_count"]
        ):
            raise V1DARTReadinessError(
                "a transitive freeze-history artifact byte count changed"
            )
        verified.append(dict(record))
    return verified


def freeze_v1_dart_testing_policy(
    repository_root: Path,
    config_path: Path,
    output_root: Path,
    *,
    require_before_v1_complete: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    destination = Path(output_root).resolve() / "testing_selection_policy_freeze.json"
    contract = _policy_contract(config_file)
    amendment_name = str(
        contract["policy_sections"]["pre_dart_v1_readiness"]["readiness_outputs"][
            "amendment_ledger"
        ]
    )
    amendment_ledger = Path(output_root).resolve() / amendment_name
    contract_sha = sha256_json(contract)
    v1_manifest = root / "artifacts/real_forecasting/paper-forecast-v1/run_manifest.json"
    complete_exists = False
    if v1_manifest.is_file():
        try:
            complete_exists = str(_read_json(v1_manifest).get("status", "")).upper() == "COMPLETE"
        except (OSError, ValueError, V1DARTReadinessError):
            complete_exists = False
    readiness_contract = contract["policy_sections"]["pre_dart_v1_readiness"]
    post_amendment = readiness_contract.get(
        "post_recovery_structural_missingness_amendment", {}
    )
    post_amendment_enabled = (
        isinstance(post_amendment, Mapping)
        and post_amendment.get("enabled") is True
        and post_amendment.get("three_seed_amendment_enabled") is True
        and post_amendment.get("performance_independent") is True
    )
    method_amendment = readiness_contract.get(
        "validation_informed_selection_method_amendment", {}
    )
    validation_informed_method_amendment = (
        isinstance(method_amendment, Mapping)
        and method_amendment.get("enabled") is True
        and method_amendment.get("auxiliary_validation_metrics_were_reviewed") is True
        and method_amendment.get("locked_2025_dart_metrics_consumed") is False
        and method_amendment.get("formal_dart_training_started") is False
        and method_amendment.get("formal_dart_replay_started") is False
        and method_amendment.get("formal_dart_selection_results_existed") is False
    )
    anchor_amendment = readiness_contract.get("anchor_only_checkpoint_amendment", {})
    anchor_only_amendment = (
        isinstance(anchor_amendment, Mapping)
        and _anchor_only_import_contract_is_frozen(
            {"pre_dart_v1_readiness": readiness_contract, "protocol_amendment": contract["policy_sections"]["protocol_amendment"]}
        )
    )
    recovery_manifest_path = _resolve(
        root, readiness_contract["recovery_manifest_path"]
    )
    recovery_complete = (
        recovery_manifest_path.is_file()
        and str(_read_json(recovery_manifest_path).get("status", "")).upper()
        == "COMPLETE"
    )
    complete_exists = complete_exists or recovery_complete
    post_result_amendment = post_amendment_enabled and complete_exists
    if (
        require_before_v1_complete
        and complete_exists
        and not destination.is_file()
        and not post_amendment_enabled
    ):
        raise V1DARTReadinessError(
            "testing policy was not frozen before the v1 COMPLETE manifest appeared"
        )
    superseded: dict[str, Any] | None = None
    transitive_history: list[dict[str, Any]] = []
    anchor_import_evidence: dict[str, Any] | None = None
    if destination.is_file():
        existing = _read_json(destination)
        if (
            existing.get("frozen_policy_sha256") == contract_sha
            and sha256_json(existing.get("contract")) == contract_sha
        ):
            if existing.get("pre_result_amendment") is True:
                superseded_record = existing.get("superseded_freeze")
                if (
                    not isinstance(superseded_record, Mapping)
                    or superseded_record.get("schema_version")
                    != "dcvpp.v1_dart_testing_selection_policy.v4"
                    or superseded_record.get("file_sha256")
                    != EXPECTED_PRE_V5_FREEZE_SHA256
                ):
                    raise V1DARTReadinessError("v5 superseded-freeze metadata changed")
                transitive_history = _verify_frozen_chain_records(
                    root, existing.get("transitive_superseded_chain")
                )
                if not amendment_ledger.is_file():
                    raise V1DARTReadinessError("v5 amendment ledger is absent")
                current_ledger = _verify_ledger_fingerprint(
                    amendment_ledger,
                    AMENDMENT_SCHEMA_VERSION,
                )
                if (
                    current_ledger.get("superseded", {}).get("file_sha256")
                    != EXPECTED_PRE_V5_FREEZE_SHA256
                    or current_ledger.get("replacement", {}).get("file_sha256")
                    != sha256_file(destination)
                ):
                    raise V1DARTReadinessError("v5 amendment ledger links changed")
            elif existing.get("anchor_only_checkpoint_amendment") is True:
                transitive_history = _verify_frozen_chain_records(
                    root, existing.get("transitive_superseded_chain")
                )
                if not amendment_ledger.is_file():
                    raise V1DARTReadinessError("v12 anchor-only amendment ledger is absent")
                current_ledger = _verify_ledger_fingerprint(
                    amendment_ledger,
                    AMENDMENT_SCHEMA_VERSION,
                    expected_statuses=(ANCHOR_AMENDMENT_STATUS,),
                )
                anchor_import_evidence = _verify_anchor_snapshot_and_predecessor(
                    root, anchor_amendment
                )
                if (
                    current_ledger.get("replacement", {}).get("file_sha256")
                    != sha256_file(destination)
                    or current_ledger.get("replacement", {}).get("schema_version")
                    != POLICY_SCHEMA_VERSION
                    or current_ledger.get("anchor_import_evidence", {}).get(
                        "snapshot_manifest_file_sha256"
                    )
                    != anchor_import_evidence["snapshot_manifest_file_sha256"]
                ):
                    raise V1DARTReadinessError(
                        "v12 anchor-only amendment ledger links changed"
                    )
            elif existing.get("hierarchical_checkpoint_selection_amendment") is True:
                transitive_history = _verify_frozen_chain_records(
                    root, existing.get("transitive_superseded_chain")
                )
                if not amendment_ledger.is_file():
                    raise V1DARTReadinessError("v11 checkpoint amendment ledger is absent")
                current_ledger = _verify_ledger_fingerprint(
                    amendment_ledger,
                    AMENDMENT_SCHEMA_VERSION,
                    expected_statuses=(
                        "AMENDED_PRE_FORMAL_DART_CHECKPOINT_SELECTION",
                    ),
                )
                if (
                    current_ledger.get("replacement", {}).get("file_sha256")
                    != sha256_file(destination)
                    or current_ledger.get("replacement", {}).get("schema_version")
                    != POLICY_SCHEMA_VERSION
                ):
                    raise V1DARTReadinessError(
                        "v11 checkpoint amendment ledger links changed"
                    )
            result = {
                **existing,
                "policy_freeze_path": _relative(root, destination),
                "policy_freeze_file_sha256": sha256_file(destination),
            }
            if amendment_ledger.is_file():
                result.update(
                    {
                        "amendment_ledger_path": _relative(root, amendment_ledger),
                        "amendment_ledger_sha256": sha256_file(amendment_ledger),
                    }
                )
            return result
        if complete_exists:
            raise V1DARTReadinessError(
                "a readiness amendment cannot replace a policy freeze after v1 COMPLETE"
            )
        existing_sha = sha256_file(destination)
        prior_schema = existing.get("contract", {}).get("schema_version")
        if prior_schema not in {
            "dcvpp.v1_dart_testing_selection_policy.v4",
        }:
            raise V1DARTReadinessError(
                "only the documented pre-result v4 freeze may be amended to v5"
            )
        if prior_schema == "dcvpp.v1_dart_testing_selection_policy.v4":
            transitive_history = _verify_pre_v5_freeze_history(root, destination)
        archives = sorted(
            destination.parent.glob(
                "testing_selection_policy_freeze.*.superseded_pre_result_amendment.json"
            )
        )
        matching_archives = [path for path in archives if sha256_file(path) == existing_sha]
        if not matching_archives:
            schema_tag = re.sub(r"[^A-Za-z0-9._-]", "_", str(prior_schema))
            archive = destination.parent / (
                f"testing_selection_policy_freeze.{schema_tag}.{existing_sha[:12]}."
                "superseded_pre_result_amendment.json"
            )
            temporary = archive.with_suffix(archive.suffix + ".tmp")
            temporary.write_bytes(destination.read_bytes())
            temporary.replace(archive)
            matching_archives = [archive]
        if len(matching_archives) != 1:
            raise V1DARTReadinessError(
                "the superseded freeze was not preserved byte-for-byte exactly once"
            )
        superseded = {
            "path": _relative(root, matching_archives[0]),
            "file_sha256": existing_sha,
            "frozen_policy_sha256": existing.get("frozen_policy_sha256"),
            "schema_version": existing.get("contract", {}).get("schema_version"),
            "frozen_at_utc": existing.get("frozen_at_utc"),
        }
        if prior_schema == "dcvpp.v1_dart_testing_selection_policy.v4":
            transitive_history.append(
                {
                    "role": "v4_policy_archive",
                    "path": _relative(root, matching_archives[0]),
                    "sha256": existing_sha,
                }
            )
    elif complete_exists and anchor_only_amendment:
        anchor_import_evidence = _verify_anchor_snapshot_and_predecessor(
            root, anchor_amendment
        )
        prior_freeze = _resolve(root, str(anchor_amendment["prior_freeze_path"]))
        prior = _read_json(prior_freeze)
        superseded = {
            "path": _relative(root, prior_freeze),
            "byte_count": prior_freeze.stat().st_size,
            "file_sha256": sha256_file(prior_freeze),
            "frozen_policy_sha256": prior.get("frozen_policy_sha256"),
            "schema_version": prior.get("contract", {}).get("schema_version"),
            "frozen_at_utc": prior.get("frozen_at_utc"),
        }
        prior_history = prior.get("transitive_superseded_chain")
        if isinstance(prior_history, list) and prior_history:
            transitive_history = _verify_frozen_chain_records(root, prior_history)
        transitive_history.extend(anchor_import_evidence["predecessor_records"])
        transitive_history.append(
            {
                "role": "v3_positive_anchor_snapshot_manifest",
                "path": anchor_import_evidence["snapshot_manifest_path"],
                "byte_count": _resolve(
                    root, anchor_import_evidence["snapshot_manifest_path"]
                ).stat().st_size,
                "sha256": anchor_import_evidence[
                    "snapshot_manifest_file_sha256"
                ],
            }
        )
    elif post_result_amendment:
        prior_freeze = _resolve(root, str(post_amendment["prior_freeze_path"]))
        prior_amendment = _resolve(
            root, str(post_amendment["prior_amendment_path"])
        )
        if (
            not prior_freeze.is_file()
            or prior_freeze.stat().st_size
            != int(post_amendment["prior_freeze_byte_count"])
            or sha256_file(prior_freeze)
            != str(post_amendment["prior_freeze_sha256"])
            or not prior_amendment.is_file()
            or prior_amendment.stat().st_size
            != int(post_amendment["prior_amendment_byte_count"])
            or sha256_file(prior_amendment)
            != str(post_amendment["prior_amendment_sha256"])
        ):
            raise V1DARTReadinessError(
                "the declared v10 readiness freeze/amendment history changed"
            )
        prior = _read_json(prior_freeze)
        if (
            prior.get("contract", {}).get("schema_version")
            != str(post_amendment["prior_policy_schema_version"])
        ):
            raise V1DARTReadinessError("the declared v10 policy schema changed")
        prior_ledger = _verify_ledger_fingerprint(
            prior_amendment,
            str(post_amendment["prior_amendment_schema_version"]),
            expected_statuses=(str(post_amendment["prior_amendment_status"]),),
        )
        if (
            prior_ledger.get("replacement", {}).get("file_sha256")
            != sha256_file(prior_freeze)
            or prior_ledger.get("replacement", {}).get("schema_version")
            != str(post_amendment["prior_policy_schema_version"])
        ):
            raise V1DARTReadinessError("the declared v10 amendment links changed")
        superseded = {
            "path": _relative(root, prior_freeze),
            "byte_count": prior_freeze.stat().st_size,
            "file_sha256": sha256_file(prior_freeze),
            "frozen_policy_sha256": prior.get("frozen_policy_sha256"),
            "schema_version": prior.get("contract", {}).get("schema_version"),
            "frozen_at_utc": prior.get("frozen_at_utc"),
        }
        transitive_history = _verify_frozen_chain_records(
            root, prior.get("transitive_superseded_chain")
        )
        transitive_history.extend([
            {
                "role": "v10_policy_freeze",
                "path": _relative(root, prior_freeze),
                "byte_count": prior_freeze.stat().st_size,
                "sha256": sha256_file(prior_freeze),
            },
            {
                "role": "v10_policy_amendment",
                "path": _relative(root, prior_amendment),
                "byte_count": prior_amendment.stat().st_size,
                "sha256": sha256_file(prior_amendment),
            },
        ])
    payload = {
        "status": "FROZEN",
        "terminal": False,
        "frozen_policy_sha256": contract_sha,
        "frozen_before_v1_complete_manifest": not complete_exists,
        "contract": contract,
        "pre_result_amendment": (
            superseded is not None
            and not post_result_amendment
            and not (complete_exists and anchor_only_amendment)
        ),
        "post_result_structural_missingness_and_three_seed_amendment": (
            superseded is not None and post_result_amendment
        ),
        "validation_informed_selection_method_amendment": (
            superseded is not None and validation_informed_method_amendment
        ),
        "hierarchical_checkpoint_selection_amendment": (
            superseded is not None and validation_informed_method_amendment
        ),
        "anchor_only_checkpoint_amendment": (
            superseded is not None and complete_exists and anchor_only_amendment
        ),
        "anchor_import_evidence": (
            anchor_import_evidence
            if superseded is not None and complete_exists and anchor_only_amendment
            else None
        ),
        "superseded_freeze": superseded,
        "transitive_superseded_chain": transitive_history,
        "amendment_reasons": list(AMENDMENT_REASONS) if superseded is not None else [],
        "frozen_at_utc": utc_now(),
    }
    atomic_json(destination, payload)
    result = {
        **payload,
        "policy_freeze_path": _relative(root, destination),
        "policy_freeze_file_sha256": sha256_file(destination),
    }
    if superseded is not None:
        ledger = {
            "schema_version": AMENDMENT_SCHEMA_VERSION,
            "status": (
                ANCHOR_AMENDMENT_STATUS
                if complete_exists and anchor_only_amendment
                else
                "AMENDED_PRE_FORMAL_DART_CHECKPOINT_SELECTION"
                if post_result_amendment and validation_informed_method_amendment
                else "AMENDED_POST_RESULT_STRUCTURAL_MISSINGNESS_AND_THREE_SEED_PANEL"
                if post_result_amendment
                else "AMENDED_PRE_RESULT"
            ),
            "terminal": False,
            "amended_before_v1_complete_manifest": not complete_exists,
            "performance_independent": bool(
                (complete_exists and anchor_only_amendment)
                or (post_result_amendment and not validation_informed_method_amendment)
            ),
            "selection_method_amendment_validation_informed": bool(
                validation_informed_method_amendment
                and not (complete_exists and anchor_only_amendment)
            ),
            "hierarchical_checkpoint_selection_validation_informed": bool(
                validation_informed_method_amendment
                and not (complete_exists and anchor_only_amendment)
            ),
            "locked_2025_dart_metrics_consumed": False,
            "formal_dart_training_anchors_existed": bool(
                complete_exists and anchor_only_amendment
            ),
            "formal_dart_completed_oos_units": 0,
            "formal_dart_replay_existed": False,
            "formal_dart_replay_started": False,
            "rolling_2022_2024_oos_predictions_or_metrics_consumed": False,
            "market_replay_milp_action_or_value_results_consumed": False,
            "model_selection_performance_reviewed_for_anchor_amendment": False,
            "confirmatory_study_seeds": list(STUDY_SEEDS),
            "legacy_excluded_seeds": list(LEGACY_EXCLUDED_SEEDS),
            "superseded": superseded,
            "replacement": {
                "path": _relative(root, destination),
                "file_sha256": result["policy_freeze_file_sha256"],
                "frozen_policy_sha256": contract_sha,
                "schema_version": POLICY_SCHEMA_VERSION,
            },
            "reasons": list(AMENDMENT_REASONS),
            "transitive_superseded_chain": transitive_history,
            "anchor_import_evidence": (
                anchor_import_evidence
                if complete_exists and anchor_only_amendment
                else None
            ),
            "amended_at_utc": utc_now(),
        }
        ledger["ledger_fingerprint"] = sha256_json(ledger)
        atomic_json(amendment_ledger, ledger)
        result.update(
            {
                "amendment_ledger_path": _relative(root, amendment_ledger),
                "amendment_ledger_sha256": sha256_file(amendment_ledger),
            }
        )
    return result


def _verify_v1_outputs(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise V1DARTReadinessError("v1 manifest has no output ledger")
    checked = 0
    total_bytes = 0
    paths: dict[str, dict[str, Any]] = {}
    for record in outputs:
        if not isinstance(record, Mapping):
            raise V1DARTReadinessError("v1 output ledger contains a non-object record")
        raw_path = record.get("path")
        expected_sha = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise V1DARTReadinessError("v1 output record lacks path/SHA-256")
        path = _resolve(root, raw_path)
        if not path.is_file():
            raise V1DARTReadinessError(f"v1 output is absent: {raw_path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise V1DARTReadinessError(f"v1 output hash mismatch: {raw_path}")
        actual_bytes = path.stat().st_size
        if record.get("byte_count") is not None and int(record["byte_count"]) != actual_bytes:
            raise V1DARTReadinessError(f"v1 output byte count mismatch: {raw_path}")
        paths[Path(raw_path).name] = {
            "path": path,
            "sha256": actual_sha,
            "byte_count": actual_bytes,
        }
        checked += 1
        total_bytes += actual_bytes
    return {"checked": checked, "total_bytes": total_bytes, "by_name": paths}


def _verify_v1_recovery_manifest(
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    readiness_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the disclosed mixed-lineage v1 recovery before reading metrics."""

    accepted = tuple(readiness_config.get("accepted_v1_manifest_schema_versions", ()))
    if accepted != (RECOVERY_MANIFEST_SCHEMA_VERSION,):
        raise V1DARTReadinessError("recovery schema allowlist changed")
    if manifest.get("schema_version") != RECOVERY_MANIFEST_SCHEMA_VERSION:
        raise V1DARTReadinessError("v1 readiness requires the recovery manifest schema")
    expected_run_id = str(readiness_config.get("recovery_run_id", ""))
    if manifest.get("run_id") != expected_run_id:
        raise V1DARTReadinessError("v1 recovery run id differs from the frozen policy")
    if (
        manifest.get("run_kind") != readiness_config.get("recovery_run_kind")
        or manifest.get("measurement_relationship")
        != readiness_config.get("recovery_measurement_relationship")
        or manifest.get("source_relationship")
        != "third_party_referenced_public_data"
        or manifest.get("evidence_role")
        != "referenced_public_counterfactual_input"
    ):
        raise V1DARTReadinessError("v1 recovery evidence semantics differ from policy")
    if (
        manifest.get("terminal") is not True
        or manifest.get("formal_readiness_allowed") is not True
        or int(manifest.get("actionable_pending_count", -1)) != 0
        or manifest.get("integrity_failures") not in ([], None)
        or manifest.get("task_failures") not in ([], None)
        or manifest.get("started_from_real_referenced_data") is not True
        or manifest.get("synthetic_data_used") is not False
    ):
        raise V1DARTReadinessError("v1 recovery terminal/integrity contract failed")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or (
        int(counts.get("tasks_requested", -1)) != 3
        or int(counts.get("tasks_completed", -1)) != 3
        or int(counts.get("tasks_failed", -1)) != 0
        or int(counts.get("failed_seeds", -1)) != 0
    ):
        raise V1DARTReadinessError("v1 recovery task counts failed")

    supplied_fingerprint = manifest.get("manifest_fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("manifest_fingerprint", None)
    if (
        not isinstance(supplied_fingerprint, str)
        or sha256_json(unsigned) != supplied_fingerprint
    ):
        raise V1DARTReadinessError("v1 recovery manifest fingerprint mismatch")

    incident = manifest.get("incident")
    ledger_ref = incident.get("corruption_ledger") if isinstance(incident, Mapping) else None
    if not isinstance(ledger_ref, Mapping):
        raise V1DARTReadinessError("v1 recovery lacks the corruption-ledger reference")
    ledger_path = _resolve(root, str(ledger_ref.get("path", "")))
    ledger_sha = str(ledger_ref.get("sha256", ""))
    if not ledger_path.is_file() or sha256_file(ledger_path) != ledger_sha:
        raise V1DARTReadinessError("v1 corruption ledger hash verification failed")
    ledger = _read_json(ledger_path)
    records = ledger.get("records")
    record_count = int(ledger_ref.get("record_count", -1))
    if (
        record_count != 28
        or int(ledger.get("record_count", -1)) != 28
        or not isinstance(records, list)
        or len(records) != 28
    ):
        raise V1DARTReadinessError("v1 corruption ledger is not the exact 28-file incident")
    for record in records:
        if not isinstance(record, Mapping) or record.get("excluded_from_canonical_outputs") is not True:
            raise V1DARTReadinessError("a corrupt v1 output is not explicitly excluded")
        snapshot_path = _resolve(root, str(record.get("preserved_snapshot_path", "")))
        snapshot_sha = str(record.get("preserved_snapshot_sha256", ""))
        if not snapshot_path.is_file() or sha256_file(snapshot_path) != snapshot_sha:
            raise V1DARTReadinessError("a preserved corrupt-output snapshot changed")

    physical = manifest.get("physical_recovery_contract")
    requirements = readiness_config.get("recovery_requirements")
    if not isinstance(physical, Mapping) or not isinstance(requirements, Mapping):
        raise V1DARTReadinessError("v1 recovery lacks the physical recovery contract")
    physical_path = _resolve(root, str(physical.get("path", "")))
    if (
        physical.get("status") != "VERIFIED_COMPLETED"
        or not physical_path.is_file()
        or sha256_file(physical_path) != physical.get("sha256")
        or physical.get("workload_events_contract_sha256")
        != requirements.get("workload_events_contract_sha256")
        or physical.get("workload_events_artifact_sha256")
        != requirements.get("workload_events_artifact_sha256")
    ):
        raise V1DARTReadinessError("physical recovery lineage/hash verification failed")

    lineage = manifest.get("recovery_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("composition_mode") != (
        "hash_verified_mixed_original_and_recovered_evidence"
    ):
        raise V1DARTReadinessError("v1 mixed recovery lineage is absent")
    locked = lineage.get("locked_2025_policy")
    locked_nonconsumption = bool(
        manifest.get("locked_2025_metrics_consumed") is False
        and manifest.get("locked_2025_metrics_materialized") is False
        and manifest.get("locked_2025_policy_decisions_consumed") is False
        and isinstance(locked, Mapping)
        and locked.get("metrics_consumed") is False
        and locked.get("metrics_materialized") is False
        and locked.get("strategy_decisions_used") is False
    )
    if not locked_nonconsumption:
        raise V1DARTReadinessError("v1 recovery consumed or materialized locked metrics")

    def _records_by_path(key: str) -> dict[str, Mapping[str, Any]]:
        values = manifest.get(key)
        if not isinstance(values, list):
            raise V1DARTReadinessError(f"v1 recovery {key} ledger is absent")
        result: dict[str, Mapping[str, Any]] = {}
        for value in values:
            if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
                raise V1DARTReadinessError(f"v1 recovery {key} record is invalid")
            path_key = str(value["path"])
            if path_key in result:
                raise V1DARTReadinessError(f"v1 recovery {key} contains duplicate paths")
            result[path_key] = value
        return result

    canonical = _records_by_path("outputs")
    recovered = _records_by_path("recovered_outputs")
    intact = _records_by_path("intact_outputs")
    corrupt = _records_by_path("corrupt_outputs")
    if set(canonical) != set(recovered) | set(intact) or set(recovered) & set(intact):
        raise V1DARTReadinessError("v1 canonical recovery outputs do not equal recovered plus intact")
    if set(canonical) & set(corrupt):
        raise V1DARTReadinessError("a corrupt v1 output entered the canonical output ledger")
    for path_key, record in canonical.items():
        source = recovered.get(path_key, intact.get(path_key))
        if source is None or source.get("sha256") != record.get("sha256"):
            raise V1DARTReadinessError("v1 canonical recovery output lineage mismatch")

    return {
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_fingerprint": supplied_fingerprint,
        "corruption_ledger_path": _relative(root, ledger_path),
        "corruption_ledger_sha256": ledger_sha,
        "corruption_record_count": record_count,
        "physical_recovery_path": _relative(root, physical_path),
        "physical_recovery_sha256": physical.get("sha256"),
        "canonical_output_count": len(canonical),
        "recovered_output_count": len(recovered),
        "intact_output_count": len(intact),
        "corrupt_output_count": len(corrupt),
        "incident_and_lineage_verified": True,
        "locked_metric_nonconsumption_verified": True,
    }


def _candidate_records(leaderboard: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    baselines = leaderboard.get("deployable_baselines")
    if not isinstance(baselines, list):
        raise V1DARTReadinessError("energy leaderboard lacks deployable baselines")
    records: dict[str, Mapping[str, Any]] = {}
    for value in baselines:
        if not isinstance(value, Mapping) or not isinstance(value.get("model"), str):
            raise V1DARTReadinessError("invalid deployable baseline record")
        model = str(value["model"])
        if model in records:
            raise V1DARTReadinessError(f"duplicate deployable baseline record: {model}")
        records[model] = value
    dgpt = leaderboard.get("dgpt_e")
    if not isinstance(dgpt, Mapping):
        raise V1DARTReadinessError("energy leaderboard lacks DGPT-E")
    records["dgpt_e"] = dgpt
    if tuple(sorted(records)) != tuple(sorted(CANDIDATES)):
        raise V1DARTReadinessError(
            f"v1 candidate set differs from fixed DART pool: {sorted(records)}"
        )
    return records


def _finite_metric(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise V1DARTReadinessError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric):
        raise V1DARTReadinessError(f"{label} is non-finite")
    return numeric


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["metric"])), []).append(row)
    for (_, metric), members in groups.items():
        if metric not in LOWER_IS_BETTER | HIGHER_IS_BETTER:
            for member in members:
                member["native_validation_rank"] = None
            continue
        reverse = metric in HIGHER_IS_BETTER
        ordered = sorted(members, key=lambda item: float(item["value"]), reverse=reverse)
        prior: float | None = None
        prior_rank = 0
        for position, member in enumerate(ordered, start=1):
            value = float(member["value"])
            rank = prior_rank if prior is not None and np.isclose(value, prior) else position
            member["native_validation_rank"] = rank
            prior = value
            prior_rank = rank


def _validation_projection(
    root: Path,
    v1_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    projection_tasks: dict[str, Any] = {}
    forecast_seed_roles_verified = True
    denominator_matrix: dict[str, dict[str, int]] = {}
    for task in ("nyiso_da", "nyiso_rt"):
        path = v1_root / "metrics" / f"{task}_paper_energy_leaderboard.json"
        leaderboard, validation_content_sha = _read_validation_only_leaderboard(path)
        if leaderboard.get("task") != task:
            raise V1DARTReadinessError(f"leaderboard task mismatch: {path}")
        if leaderboard.get("locked_test_used_for_selection") is not False:
            raise V1DARTReadinessError(f"locked test entered v1 selection: {task}")
        records = _candidate_records(leaderboard)
        projected_candidates: dict[str, Any] = {}
        denominator_matrix[task] = {}
        for candidate in CANDIDATES:
            record = records[candidate]
            validation = record.get("validation_metrics")
            if not isinstance(validation, Mapping):
                raise V1DARTReadinessError(
                    f"{task}/{candidate} lacks validation metrics"
                )
            missing_metrics = set(VALIDATION_METRIC_FIELDS).difference(validation)
            if missing_metrics:
                raise V1DARTReadinessError(
                    f"{task}/{candidate} lacks required validation metrics: "
                    f"{sorted(missing_metrics)}"
                )
            projected_metrics: dict[str, Any] = {}
            for metric in VALIDATION_METRIC_FIELDS:
                numeric = _finite_metric(
                    validation[metric], label=f"{task}/{candidate}/{metric}"
                )
                if metric == "n":
                    if numeric <= 0 or not float(numeric).is_integer():
                        raise V1DARTReadinessError(
                            f"{task}/{candidate} has an invalid metric denominator"
                        )
                    denominator_matrix[task][candidate] = int(numeric)
                projected_metrics[metric] = numeric
                metrics_rows.append(
                    {
                        "task": task,
                        "candidate": candidate,
                        "metric": metric,
                        "value": numeric,
                        "native_validation_rank": None,
                        "evaluation_split": "validation",
                        "evaluation_calendar_year": 2024,
                        "issue_profile": (
                            "native_day_ahead_issue_profile"
                            if task == "nyiso_da"
                            else "native_intraday_real_time_issue_profile"
                        ),
                        "evidence_role": "auxiliary_native_issue_validation",
                        "dart_selection_eligible": False,
                        "ineligibility_reason": "not_common_da_gate_cross_head_evidence",
                    }
                )
            training_role = CANDIDATE_TRAINING_ROLE[candidate]
            seed_trials = record.get("seed_trials")
            legacy_excluded_trials: list[Mapping[str, Any]] = []
            if training_role == "neural_three_fit":
                if not isinstance(seed_trials, list):
                    forecast_seed_roles_verified = False
                    seeds: tuple[int, ...] = ()
                    statuses: tuple[str, ...] = ()
                else:
                    unexpected = [
                        value
                        for value in seed_trials
                        if int(value["seed"]) not in {*STUDY_SEEDS, *LEGACY_EXCLUDED_SEEDS}
                    ]
                    legacy_excluded_trials = [
                        value
                        for value in seed_trials
                        if int(value["seed"]) in LEGACY_EXCLUDED_SEEDS
                    ]
                    amended_trials = [
                        value
                        for value in seed_trials
                        if int(value["seed"]) in STUDY_SEEDS
                    ]
                    seeds = tuple(sorted(int(value["seed"]) for value in amended_trials))
                    statuses = tuple(
                        str(value.get("status", "")).lower() for value in amended_trials
                    )
                    if unexpected:
                        forecast_seed_roles_verified = False
                if (
                    record.get("deterministic") is not False
                    or seeds != tuple(sorted(STUDY_SEEDS))
                    or any(status != "completed" for status in statuses)
                ):
                    forecast_seed_roles_verified = False
            else:
                if record.get("deterministic") is not True or seed_trials not in (None, []):
                    forecast_seed_roles_verified = False
                seeds = ()
                statuses = ()
            projected_candidates[candidate] = {
                "deterministic": bool(record.get("deterministic", False)),
                "deployable": bool(record.get("deployable", True)),
                "forecast_training_seed_role": training_role,
                "validation_metrics": projected_metrics,
                "forecast_fit_seed_contract_verified": (
                    training_role != "neural_three_fit"
                    or (
                        seeds == tuple(sorted(STUDY_SEEDS))
                        and all(status == "completed" for status in statuses)
                    )
                ),
                "decision_replay_seed_slots": list(STUDY_SEEDS),
                "legacy_excluded_seeds": list(LEGACY_EXCLUDED_SEEDS),
                "legacy_pre_amendment_trial_count": len(legacy_excluded_trials),
                "legacy_pre_amendment_policy": "excluded_from_confirmatory_three_seed_panel",
                "locked_test_fields_projected": False,
            }
        task_denominators = set(denominator_matrix[task].values())
        if len(task_denominators) != 1:
            raise V1DARTReadinessError(
                f"{task} candidates do not share one validation denominator"
            )
        projection_tasks[task] = {
            "leaderboard_path": _relative(root, path),
            "validation_only_content_sha256": validation_content_sha,
            "candidates": projected_candidates,
        }
    da_denominator = next(iter(set(denominator_matrix["nyiso_da"].values())))
    rt_denominator = next(iter(set(denominator_matrix["nyiso_rt"].values())))
    _rank_rows(metrics_rows)
    projection = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "evaluation_split": "validation",
        "evaluation_calendar_years": [2024],
        "evaluation_year_timezone": "UTC",
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "selection_or_policy_updated_from_v1_results": False,
        "forecast_training_seed_roles": {
            role: list(candidates)
            for role, candidates in FORECAST_TRAINING_SEED_ROLES.items()
        },
        "decision_replay_seed_slots_for_every_candidate": list(STUDY_SEEDS),
        "metric_denominator_matrix": denominator_matrix,
        "cross_task_denominator_disclosure": {
            "nyiso_da": da_denominator,
            "nyiso_rt": rt_denominator,
            "equal": da_denominator == rt_denominator,
            "cross_task_metric_ranking_performed": False,
            "reason": (
                "native issue task metrics remain within-task; cross-head DART "
                "selection requires new common-DA-gate retraining"
            ),
        },
        "tasks": projection_tasks,
    }
    projection["projection_sha256"] = sha256_json(projection)
    projection_path = output_root / "v1_validation_only_projection.json"
    atomic_json(projection_path, projection)
    return projection, metrics_rows, {
        "forecast_training_seed_roles_verified": forecast_seed_roles_verified,
        "required_metric_denominator_matrix_verified": bool(denominator_matrix),
        "projection_path": projection_path,
        "projection_file_sha256": sha256_file(projection_path),
    }


def _aware_utc(values: pd.Series, *, label: str) -> pd.Series:
    dtype_timezone = getattr(values.dtype, "tz", None)
    if dtype_timezone is None:
        for raw in values:
            if isinstance(raw, pd.Timestamp):
                if raw.tzinfo is None:
                    raise V1DARTReadinessError(f"{label} contains a timezone-naive value")
            elif not _EXPLICIT_ZONE_SUFFIX.search(str(raw)):
                raise V1DARTReadinessError(f"{label} lacks an explicit UTC offset")
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise V1DARTReadinessError(f"{label} contains a null timestamp")
    return parsed


def _validation_timing_audit(
    v1_root: Path,
    readiness_config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    market_timezone = "America/New_York"
    evaluation_timezone = str(readiness_config["evaluation_year_timezone"])
    if evaluation_timezone != "UTC":
        raise V1DARTReadinessError("native validation operating-date timezone is not UTC")
    for task in ("nyiso_da", "nyiso_rt"):
        profile = readiness_config["native_issue_contract"][task]
        path = v1_root / "predictions" / str(profile["artifact"])
        frame = pd.read_parquet(
            path,
            filters=[("split", "==", "validation")],
            columns=[
                "split",
                "timestamp_utc",
                "issue_time_utc",
                "feature_data_cutoff_utc",
                "zone",
                "observed_target",
                "q05",
                "q10",
                "q50",
                "q90",
                "q95",
            ],
        )
        if frame.empty or set(frame["split"].astype(str)) != {"validation"}:
            raise V1DARTReadinessError(f"{task} validation projection is empty or mixed")
        timestamp = _aware_utc(frame["timestamp_utc"], label=f"{task}/timestamp_utc")
        issue = _aware_utc(frame["issue_time_utc"], label=f"{task}/issue_time_utc")
        cutoff = _aware_utc(
            frame["feature_data_cutoff_utc"],
            label=f"{task}/feature_data_cutoff_utc",
        )
        years = sorted(set(timestamp.dt.tz_convert(evaluation_timezone).dt.year.astype(int)))
        if years != [2024]:
            raise V1DARTReadinessError(
                f"{task} UTC operating-date validation years are not exactly 2024: {years}"
            )
        if not bool((cutoff <= issue).all() and (issue <= timestamp).all()):
            raise V1DARTReadinessError(
                f"{task} violates feature_cutoff <= issue_time <= target_time"
            )
        lead_seconds = (timestamp - issue).dt.total_seconds()
        minimum_lead = int(profile["minimum_issue_lead_seconds"])
        if not bool((lead_seconds >= minimum_lead).all()):
            raise V1DARTReadinessError(
                f"{task} violates its declared minimum issue lead of {minimum_lead} seconds"
            )
        observed = frame["observed_target"].to_numpy(float)
        if not np.isfinite(observed).all():
            raise V1DARTReadinessError(f"{task} validation targets are non-finite")
        quantiles = frame[["q05", "q10", "q50", "q90", "q95"]].to_numpy(float)
        if not np.isfinite(quantiles).all() or bool((np.diff(quantiles, axis=1) < -1e-10).any()):
            raise V1DARTReadinessError(f"{task} validation quantiles are invalid")
        normalized = frame.copy()
        normalized["timestamp_utc"] = timestamp
        normalized["issue_time_utc"] = issue
        normalized["feature_data_cutoff_utc"] = cutoff
        key_frame = normalized[["timestamp_utc", "zone"]]
        if key_frame.duplicated().any():
            raise V1DARTReadinessError(f"{task} contains duplicate timestamp/zone keys")
        normalized["_key"] = [
            f"{value.isoformat()}|{zone}"
            for value, zone in zip(timestamp, normalized["zone"].astype(str), strict=True)
        ]
        frames[task] = normalized
        records[task] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "row_count": int(len(frame)),
            "years": years,
            "evaluation_year_timezone": evaluation_timezone,
            "market_timezone": market_timezone,
            "storage_timezone": "UTC",
            "timestamps_explicitly_zoned": True,
            "audited_candidate": str(profile["audited_candidate"]),
            "issue_profile_id": str(profile["profile_id"]),
            "minimum_required_issue_lead_seconds": minimum_lead,
            "minimum_observed_issue_lead_seconds": float(lead_seconds.min()),
            "minimum_target_utc": timestamp.min().isoformat(),
            "maximum_target_utc": timestamp.max().isoformat(),
            "issue_precedes_target": True,
            "feature_cutoff_precedes_or_equals_issue": True,
            "quantiles_monotone_and_finite": True,
        }
    da_keys = set(frames["nyiso_da"]["_key"].astype(str))
    rt_keys = set(frames["nyiso_rt"]["_key"].astype(str))
    common_keys = da_keys & rt_keys
    if not common_keys:
        raise V1DARTReadinessError("v1 DA/RT validation common panel is empty")
    required_zones = int(
        readiness_config["native_validation_common_panel"]["required_zones"]
    )
    required_intervals = int(
        readiness_config["native_validation_common_panel"]
        ["required_intervals_per_day"]
    )

    def excluded_complete_days(task: str, excluded: set[str]) -> list[dict[str, Any]]:
        if not excluded:
            return []
        subset = frames[task].loc[frames[task]["_key"].isin(excluded)].copy()
        subset["operating_date"] = (
            subset["timestamp_utc"].dt.tz_convert(evaluation_timezone).dt.date.astype(str)
        )
        output: list[dict[str, Any]] = []
        for operating_date, group in subset.groupby("operating_date", sort=True):
            zone_counts = {
                str(zone): int(count)
                for zone, count in group.groupby("zone").size().items()
            }
            if (
                len(zone_counts) != required_zones
                or set(zone_counts.values()) != {required_intervals}
            ):
                raise V1DARTReadinessError(
                    f"{task} exclusion is not a complete operating day: "
                    f"{operating_date}/{zone_counts}"
                )
            output.append(
                {
                    "task": task,
                    "operating_date": str(operating_date),
                    "row_count": int(len(group)),
                    "zone_counts": zone_counts,
                    "reason": "referenced_realized_target_day_not_complete_in_both_native_issue_tasks",
                }
            )
        if sum(record["row_count"] for record in output) != len(excluded):
            raise V1DARTReadinessError(
                f"{task} excluded-key ledger does not cover every excluded key"
            )
        return output

    da_only = da_keys - rt_keys
    rt_only = rt_keys - da_keys
    exclusions = excluded_complete_days("nyiso_da", da_only) + excluded_complete_days(
        "nyiso_rt", rt_only
    )
    exclusion_name = str(
        readiness_config["readiness_outputs"]["native_common_panel_exclusions"]
    )
    exclusion_path = output_root / exclusion_name
    exclusion_payload = {
        "schema_version": "dcvpp.v1_native_validation_common_panel_exclusions.v1",
        "status": "COMPLETE",
        "policy": "candidate_independent_complete_operating_day_exclusions",
        "imputation_used": False,
        "silent_inner_join_used": False,
        "da_original_key_count": len(da_keys),
        "rt_original_key_count": len(rt_keys),
        "common_key_count": len(common_keys),
        "da_only_key_count": len(da_only),
        "rt_only_key_count": len(rt_only),
        "excluded_operating_days": exclusions,
        "common_key_set_sha256": sha256_json(sorted(common_keys)),
        "da_only_key_set_sha256": sha256_json(sorted(da_only)),
        "rt_only_key_set_sha256": sha256_json(sorted(rt_only)),
    }
    exclusion_payload["ledger_fingerprint"] = sha256_json(exclusion_payload)
    atomic_json(exclusion_path, exclusion_payload)
    da = frames["nyiso_da"].rename(
        columns={
            "observed_target": "da_observed",
            "q50": "da_q50",
            "issue_time_utc": "da_issue_time_utc",
        }
    )
    rt = frames["nyiso_rt"].rename(
        columns={
            "observed_target": "rt_observed",
            "q50": "rt_q50",
            "issue_time_utc": "rt_issue_time_utc",
        }
    )
    da = da.loc[da["_key"].isin(common_keys)].copy()
    rt = rt.loc[rt["_key"].isin(common_keys)].copy()
    aligned = da[
        ["_key", "timestamp_utc", "zone", "da_observed", "da_q50", "da_issue_time_utc"]
    ].merge(
        rt[["_key", "rt_observed", "rt_q50", "rt_issue_time_utc"]],
        on="_key",
        how="inner",
        validate="one_to_one",
        indicator=True,
    )
    if (
        aligned.empty
        or set(aligned["_merge"].astype(str)) != {"both"}
        or len(aligned) != len(common_keys)
    ):
        raise V1DARTReadinessError("v1 DA/RT validation projections do not align exactly")
    actual_dart = aligned["da_observed"].to_numpy(float) - aligned["rt_observed"].to_numpy(float)
    actual_inverse = aligned["rt_observed"].to_numpy(float) - aligned["da_observed"].to_numpy(float)
    predicted_diagnostic = aligned["da_q50"].to_numpy(float) - aligned["rt_q50"].to_numpy(float)
    if not np.allclose(actual_dart, -actual_inverse, equal_nan=True):
        raise V1DARTReadinessError("v1 validation actual DART sign identity failed")
    records["cross_head_diagnostic"] = {
        "aligned_row_count": int(len(aligned)),
        "exact_da_rt_key_set_equal": True,
        "original_da_rt_key_set_equal": da_keys == rt_keys,
        "declared_common_panel_used": True,
        "silent_inner_join_used": False,
        "common_panel_exclusion_ledger": {
            "path": exclusion_path.as_posix(),
            "sha256": sha256_file(exclusion_path),
            "excluded_operating_day_count": len(exclusions),
        },
        "actual_dart_inverse_gate_passed": True,
        "native_issue_point_diagnostic_mae": float(
            np.mean(np.abs(predicted_diagnostic - actual_dart))
        ),
        "label": "AUXILIARY_NOT_DA_GATE_ELIGIBLE",
        "selection_eligible": False,
        "reason": "DA and native intraday RT do not share the common DA-gate information set",
        "joint_probabilistic_dart_claimed": False,
        "chronology_claim_scope": "audited_dgpt_native_issue_artifacts_only",
    }
    return records


def _eligibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in ("nyiso_da", "nyiso_rt", "nlr_workload"):
        rows.append(
            {
                "task": task,
                "layer": "probabilistic_credibility",
                "status": "AUXILIARY_NATIVE_ISSUE_ELIGIBLE",
                "numeric_evaluation_required": task in {"nyiso_da", "nyiso_rt"},
                "dart_selection_eligible": False,
                "reason": "native-issue validation only",
            }
        )
    for layer in ("actionability", "decision_value"):
        rows.append(
            {
                "task": "nyiso_da_x_native_rt",
                "layer": layer,
                "status": "AUXILIARY_NOT_DA_GATE_ELIGIBLE",
                "numeric_evaluation_required": False,
                "dart_selection_eligible": False,
                "reason": "native RT uses information unavailable at the common DA gate",
            }
        )
    return rows


def _load_completed_manifest_result(
    root: Path,
    manifest_path: Path,
    policy_sha: str,
    v1_manifest_sha: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    payload = _read_json(manifest_path)
    if payload.get("status") != "COMPLETE":
        return None
    if payload.get("frozen_policy_sha256") != policy_sha:
        raise V1DARTReadinessError("completed readiness manifest uses a different frozen policy")
    if payload.get("v1_forecast_manifest_sha256") != v1_manifest_sha:
        raise V1DARTReadinessError("completed readiness manifest uses a different v1 manifest")
    for record in payload.get("outputs", []):
        path = _resolve(root, str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise V1DARTReadinessError("completed readiness output changed")
    return {
        **payload,
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "resumed": True,
    }


def run_v1_dart_readiness_audit(
    repository_root: Path,
    v1_manifest_path: Path,
    config_path: Path | None = None,
    output_root: Path | None = None,
    *,
    dart_config_path: Path | None = None,
    run_id: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    v1_manifest_file = Path(v1_manifest_path).resolve()
    selected_config = config_path if config_path is not None else dart_config_path
    if selected_config is None:
        raise ValueError("config_path or dart_config_path is required")
    if output_root is None:
        raise ValueError("output_root is required")
    config_file = Path(selected_config).resolve()
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    freeze = freeze_v1_dart_testing_policy(
        root, config_file, destination, require_before_v1_complete=False
    )
    if (
        freeze.get("frozen_before_v1_complete_manifest") is not True
        and freeze.get("post_result_structural_missingness_and_three_seed_amendment")
        is not True
        and freeze.get("anchor_only_checkpoint_amendment") is not True
    ):
        raise V1DARTReadinessError(
            "testing policy is neither pre-result frozen nor a declared structural/anchor amendment"
        )
    v1_manifest_sha_before = sha256_file(v1_manifest_file)
    manifest_path = destination / "v1_dart_readiness_manifest.json"
    if resume:
        completed = _load_completed_manifest_result(
            root,
            manifest_path,
            str(freeze["frozen_policy_sha256"]),
            v1_manifest_sha_before,
        )
        if completed is not None:
            return completed
    v1_manifest = _read_json(v1_manifest_file)
    if str(v1_manifest.get("status", "")).upper() != "COMPLETE":
        raise V1DARTReadinessError("paper-forecast-v1 is not COMPLETE")
    counts = v1_manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise V1DARTReadinessError("v1 manifest lacks counts")
    if (
        int(counts.get("tasks_requested", -1)) != 3
        or int(counts.get("tasks_completed", -1)) != 3
        or int(counts.get("tasks_failed", -1)) != 0
        or int(counts.get("failed_seeds", -1)) != 0
        or v1_manifest.get("task_failures") not in ([], None)
    ):
        raise V1DARTReadinessError("v1 task/seed completion contract failed")
    dart_config = load_yaml(config_file)
    readiness_config = dart_config["pre_dart_v1_readiness"]
    recovery_audit = _verify_v1_recovery_manifest(
        root,
        v1_manifest_file,
        v1_manifest,
        readiness_config,
    )
    output_audit = _verify_v1_outputs(root, v1_manifest)
    v1_root = v1_manifest_file.parent
    projection, metric_rows, projection_audit = _validation_projection(
        root, v1_root, destination
    )
    timing_audit = _validation_timing_audit(
        v1_root, readiness_config, destination
    )
    eligibility_rows = _eligibility_rows()
    metrics_path = destination / "v1_native_validation_metrics.csv"
    eligibility_path = destination / "v1_dart_eligibility_matrix.csv"
    _atomic_csv(
        metrics_path,
        metric_rows,
        (
            "task",
            "candidate",
            "metric",
            "value",
            "native_validation_rank",
            "evaluation_split",
            "evaluation_calendar_year",
            "issue_profile",
            "evidence_role",
            "dart_selection_eligible",
            "ineligibility_reason",
        ),
    )
    _atomic_csv(
        eligibility_path,
        eligibility_rows,
        (
            "task",
            "layer",
            "status",
            "numeric_evaluation_required",
            "dart_selection_eligible",
            "reason",
        ),
    )
    v1_manifest_sha_after = sha256_file(v1_manifest_file)
    gates = {
        "v1_manifest_immutable": v1_manifest_sha_before == v1_manifest_sha_after,
        "v1_output_hashes_verified": output_audit["checked"] > 0,
        "v1_recovery_incident_and_lineage_verified": bool(
            recovery_audit["incident_and_lineage_verified"]
        ),
        "v1_recovery_locked_metric_nonconsumption_verified": bool(
            recovery_audit["locked_metric_nonconsumption_verified"]
        ),
        "forecast_training_seed_roles_verified": bool(
            projection_audit["forecast_training_seed_roles_verified"]
        ),
        "all_candidate_three_replay_seed_contract_frozen": all(
            candidate["decision_replay_seed_slots"] == list(STUDY_SEEDS)
            for task in projection["tasks"].values()
            for candidate in task["candidates"].values()
        ),
        "validation_only_projection": (
            projection.get("evaluation_split") == "validation"
            and projection.get("evaluation_calendar_years") == [2024]
            and projection.get("evaluation_year_timezone") == "UTC"
            and projection.get("locked_2025_metrics_materialized") is False
        ),
        "required_metric_denominator_matrix_verified": bool(
            projection_audit["required_metric_denominator_matrix_verified"]
        ),
        "audited_dgpt_native_issue_chronology_verified": all(
            timing_audit[task]["years"] == [2024]
            and timing_audit[task]["audited_candidate"] == "dgpt_e"
            for task in ("nyiso_da", "nyiso_rt")
        ),
        "audited_dgpt_cutoff_issue_target_order_verified": all(
            timing_audit[task]["issue_precedes_target"]
            and timing_audit[task]["feature_cutoff_precedes_or_equals_issue"]
            for task in ("nyiso_da", "nyiso_rt")
        ),
        "audited_dgpt_timezone_and_issue_profile_verified": all(
            timing_audit[task]["timestamps_explicitly_zoned"]
            and timing_audit[task]["storage_timezone"] == "UTC"
            and timing_audit[task]["evaluation_year_timezone"]
            == "UTC"
            and timing_audit[task]["minimum_observed_issue_lead_seconds"]
            >= timing_audit[task]["minimum_required_issue_lead_seconds"]
            for task in ("nyiso_da", "nyiso_rt")
        ),
        "audited_dgpt_exact_da_rt_key_panel_verified": (
            timing_audit["cross_head_diagnostic"]["exact_da_rt_key_set_equal"] is True
            and timing_audit["cross_head_diagnostic"]["silent_inner_join_used"] is False
        ),
        "testing_selection_contract_frozen_v12": (
            freeze.get("contract", {}).get("schema_version") == POLICY_SCHEMA_VERSION
            and bool(freeze.get("frozen_policy_sha256"))
        ),
        "hierarchical_checkpoint_selection_contract_frozen": (
            _checkpoint_selection_contract_is_frozen(dart_config)
            and freeze.get("contract", {})
            .get("policy_sections", {})
            .get("checkpoint_selection")
            == dart_config.get("checkpoint_selection")
        ),
        "pre_oos_anchor_import_contract_frozen": (
            _anchor_only_import_contract_is_frozen(dart_config)
            and freeze.get("contract", {})
            .get("policy_sections", {})
            .get("pre_dart_v1_readiness", {})
            .get("anchor_only_checkpoint_amendment")
            == dart_config.get("pre_dart_v1_readiness", {}).get(
                "anchor_only_checkpoint_amendment"
            )
            and (
                freeze.get("frozen_before_v1_complete_manifest") is True
                or (
                    freeze.get("anchor_only_checkpoint_amendment") is True
                    and freeze.get("anchor_import_evidence", {}).get("status")
                    == "VERIFIED_COMPLETE_HASH_PINNED_READ_ONLY_SOURCE"
                    and freeze.get("anchor_import_evidence", {}).get(
                        "every_source_anchor_file_rehashed"
                    )
                    is True
                )
            )
        ),
        "v4_full_epoch_checkpoint_contract_frozen": (
            dart_config.get("checkpoint_selection", {})
            .get("neural_training", {})
            .get("v4_every_completed_epoch_full_immutable_checkpoint_required")
            is True
            and int(
                dart_config.get("checkpoint_selection", {})
                .get("neural_training", {})
                .get("v4_full_epoch_checkpoint_interval_epochs", 0)
            )
            == 1
            and dart_config.get("checkpoint_selection", {})
            .get("neural_training", {})
            .get("v4_full_epoch_checkpoint_must_be_hash_pinned_and_append_only")
            is True
            and dart_config.get("checkpoint_selection", {})
            .get("neural_training", {})
            .get("v4_non_five_epoch_checkpoint_selection_role")
            == "audit_and_resume_only_not_formal_selection_candidate"
        ),
        "formal_execution_order_contract_frozen": (
            _execution_order_contract_is_frozen(dart_config)
            and freeze.get("contract", {})
            .get("policy_sections", {})
            .get("execution_order")
            == dart_config.get("execution_order")
        ),
        "testing_selection_implementation_hashes_verified": (
            isinstance(
                freeze.get("contract", {}).get("implementation_code_hashes"),
                Mapping,
            )
            and bool(freeze["contract"]["implementation_code_hashes"])
            and all(
                (config_file.resolve().parents[1] / relative_path).is_file()
                and sha256_file(config_file.resolve().parents[1] / relative_path)
                == expected_sha
                for relative_path, expected_sha in freeze["contract"][
                    "implementation_code_hashes"
                ].items()
            )
        ),
        "centered_studentized_global_mcs_contract_frozen": (
            dart_config["model_confidence_set"].get("statistic")
            == "centered_studentized_global_loss_range"
            and dart_config["model_confidence_set"].get("bootstrap_centering")
            == "observed_pairwise_mean_loss_differentials"
            and dart_config["model_confidence_set"].get("studentization")
            == "pairwise_moving_block_bootstrap_standard_error"
            and dart_config["model_confidence_set"].get("sequential_step_down") is True
        ),
        "solver_timeout_and_nonzero_gap_contract_frozen": (
            dart_config["feasibility"].get("require_zero_solver_timeout_days") is True
            and dart_config["feasibility"].get("require_zero_solver_nonzero_gap_days")
            is True
            and dart_config["feasibility"].get("solver_timeout_day_field")
            == "solver_timeout_days"
            and dart_config["feasibility"].get("solver_nonzero_gap_day_field")
            == "solver_nonzero_gap_days"
        ),
        "native_rt_cross_head_auxiliary_only": (
            timing_audit["cross_head_diagnostic"]["label"]
            == "AUXILIARY_NOT_DA_GATE_ELIGIBLE"
        ),
        "native_rt_excluded_from_dart_selection": (
            timing_audit["cross_head_diagnostic"]["selection_eligible"] is False
        ),
        "common_da_gate_retraining_required": True,
        "locked_metric_loader_nonconsumption_verified": (
            projection.get("locked_2025_metrics_consumed") is False
            and projection.get("locked_2025_metrics_materialized") is False
            and projection.get("selection_or_policy_updated_from_v1_results") is False
            and recovery_audit["locked_metric_nonconsumption_verified"] is True
        ),
    }
    if tuple(gates) != REQUIRED_COMPLETION_GATES:
        raise V1DARTReadinessError("readiness gate names/order differ from frozen policy")
    completed = all(gates.values())
    output_records = [
        {
            "path": _relative(root, _resolve(root, str(freeze["policy_freeze_path"]))),
            "sha256": str(freeze["policy_freeze_file_sha256"]),
        },
        {
            "path": _relative(root, Path(projection_audit["projection_path"])),
            "sha256": str(projection_audit["projection_file_sha256"]),
        },
        {"path": _relative(root, metrics_path), "sha256": sha256_file(metrics_path)},
        {"path": _relative(root, eligibility_path), "sha256": sha256_file(eligibility_path)},
        {
            "path": recovery_audit["corruption_ledger_path"],
            "sha256": recovery_audit["corruption_ledger_sha256"],
        },
        {
            "path": recovery_audit["physical_recovery_path"],
            "sha256": recovery_audit["physical_recovery_sha256"],
        },
    ]
    common_panel_ledger = timing_audit["cross_head_diagnostic"][
        "common_panel_exclusion_ledger"
    ]
    output_records.append(
        {
            "path": _relative(root, Path(common_panel_ledger["path"])),
            "sha256": common_panel_ledger["sha256"],
        }
    )
    if freeze.get("amendment_ledger_path"):
        output_records.append(
            {
                "path": str(freeze["amendment_ledger_path"]),
                "sha256": str(freeze["amendment_ledger_sha256"]),
            }
        )
    for record in freeze.get("transitive_superseded_chain", []):
        output_records.append(
            {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "COMPLETE" if completed else "BLOCKED",
        "terminal": False,
        "formal_dart_training_allowed": completed,
        "evaluation_role": "auxiliary_native_issue_pipeline_readiness",
        "evaluation_split": "validation",
        "evaluation_calendar_years": [2024],
        "evaluation_year_timezone": "UTC",
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "algorithmic_locked_metric_loader_nonconsumption_verified": True,
        "no_prior_human_exposure_claimed": False,
        "locked_test_exposure_disclosure": EXPOSURE_DISCLOSURE,
        "audited_chronology_scope": "audited_dgpt_native_issue_artifacts_only",
        "forecast_training_seed_roles": {
            role: list(candidates)
            for role, candidates in FORECAST_TRAINING_SEED_ROLES.items()
        },
        "all_candidate_replay_seed_slots": list(STUDY_SEEDS),
        "native_rt_cross_head_label": "AUXILIARY_NOT_DA_GATE_ELIGIBLE",
        "native_rt_selection_eligible": False,
        "common_da_gate_retraining_required": True,
        "v1_forecast_manifest_path": _relative(root, v1_manifest_file),
        "v1_forecast_manifest_sha256": v1_manifest_sha_before,
        "frozen_policy_sha256": freeze["frozen_policy_sha256"],
        "policy_freeze_path": freeze["policy_freeze_path"],
        "policy_freeze_file_sha256": freeze["policy_freeze_file_sha256"],
        "amendment_ledger_path": freeze.get("amendment_ledger_path"),
        "amendment_ledger_sha256": freeze.get("amendment_ledger_sha256"),
        "transitive_superseded_chain": freeze.get(
            "transitive_superseded_chain", []
        ),
        "anchor_import_evidence": freeze.get("anchor_import_evidence"),
        "completion_gates": gates,
        "performance_gate_applied": False,
        "performance_results_may_update_policy": False,
        "projection": {
            "path": _relative(root, Path(projection_audit["projection_path"])),
            "sha256": projection_audit["projection_file_sha256"],
            "metric_row_count": len(metric_rows),
        },
        "timing_and_sign_audit": timing_audit,
        "v1_output_audit": {
            "checked_output_count": output_audit["checked"],
            "checked_output_bytes": output_audit["total_bytes"],
        },
        "v1_recovery_audit": recovery_audit,
        "outputs": output_records,
        "completed_at_utc": utc_now(),
    }
    payload["manifest_fingerprint"] = sha256_json(payload)
    atomic_json(manifest_path, payload)
    return {
        **payload,
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "resumed": False,
    }


__all__ = [
    "CANDIDATES",
    "EXPOSURE_DISCLOSURE",
    "FORECAST_TRAINING_SEED_ROLES",
    "POLICY_SCHEMA_VERSION",
    "RECOVERY_MANIFEST_SCHEMA_VERSION",
    "REQUIRED_COMPLETION_GATES",
    "SCHEMA_VERSION",
    "STUDY_SEEDS",
    "V1DARTReadinessError",
    "freeze_v1_dart_testing_policy",
    "run_v1_dart_readiness_audit",
]
