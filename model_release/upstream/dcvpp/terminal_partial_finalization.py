from __future__ import annotations

"""Fail-closed terminalization for the current evidence-limited paper run.

This module deliberately does not call :func:`paper_finalization.finalize_paper_run`.
That general finalizer expects a complete-calendar DART chain, whereas this run is
scientifically limited to the pre-registered common observable-settlement panel.
The dedicated path hash-pins that conditional protocol and its causal input bundle,
inventories only source-backed artifacts, and persists an unconditional
``PARTIAL_PAPER`` terminal cap.  It never opens locked 2025 or writes numeric paper
tables or figures itself.
"""

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dcvpp.core import atomic_json, atomic_text, sha256_file, sha256_json, utc_now


PARTIAL_PAPER = "PARTIAL_PAPER"
DEFAULT_RUN_ID = "paper-terminal-partial-v1"
CANDIDATE_POOL_CONTRACT_SHA256 = (
    "0455f6f55a136413820b9f9b778b7679738e301349de1544e653aa342d91c223"
)

DEFAULT_PATHS = {
    "selection": "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos/selection_stage_manifest.json",
    "official_rt_recovery": "artifacts/dart_value_select/real_input_preparation/v1/official_rt_recovery/official_rt_recovery_manifest.json",
    "conditional_settlement_protocol": "artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1/conditional_settlement_protocol_manifest.json",
    "counterfactual_inputs": "artifacts/dart_value_select/counterfactual_inputs/v1/materialization_manifest.json",
    "selection_replay_terminal": "artifacts/dart_value_select/run_control/conditional_selection_replay_v1/terminal_manifest.json",
    "terminal_path_execution": "artifacts/run_control/terminal_partial_path/execution_manifest.json",
    "extension": "artifacts/real_forecasting_extensions/run_id=paper-forecast-extensions-v1/extension_run_manifest.json",
    "extension_pause": "artifacts/manifests/forecast_extension_round1_efficiency_amendment.json",
    "forecast": "artifacts/real_forecasting_recovery/run_id=paper-forecast-v1-recovery-v1/run_manifest.json",
    "physical": "artifacts/research/physical/run_id=paper-physical-v1/frozen_physical_artifact.json",
    "optimization_execution": "artifacts/optimization_branch/paper-optimization-v1/direct_execution_manifest.json",
}


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _assert_no_synthetic_flag(value: Any, trail: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in {
                "synthetic",
                "synthetic_data_used",
                "synthetic_fallback",
                "synthetic_fallback_used",
            } and nested is True:
                raise RuntimeError(f"synthetic evidence is forbidden ({trail}.{key}=true)")
            _assert_no_synthetic_flag(nested, f"{trail}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_no_synthetic_flag(nested, f"{trail}[{index}]")


def _validate_self_fingerprint(payload: Mapping[str, Any], label: str) -> None:
    recorded = payload.get("manifest_fingerprint")
    if recorded in (None, ""):
        return
    body = dict(payload)
    body.pop("manifest_fingerprint", None)
    if recorded != sha256_json(body):
        raise RuntimeError(f"{label} self-fingerprint is invalid")


def _validate_dart_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    contract = selection.get("candidate_pool_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("DART selection lacks candidate_pool_contract")
    expected = {
        "status": "PARTIAL_WITH_FAILED_CANDIDATE",
        "completed_unit_count": 123,
        "failed_unit_count": 3,
        "eligible_completed_unit_count": 108,
        "date_drop_count": 0,
        "terminal_status_cap": PARTIAL_PAPER,
    }
    for key, wanted in expected.items():
        if selection.get(key) != wanted:
            raise RuntimeError(
                f"DART selection invariant changed: {key}={selection.get(key)!r}, expected {wanted!r}"
            )
    exclusions = contract.get("globally_excluded_candidates")
    if not isinstance(exclusions, list) or len(exclusions) != 1:
        raise RuntimeError("DART selection must contain one global candidate exclusion")
    exclusion = exclusions[0]
    if not isinstance(exclusion, Mapping) or {
        "candidate_name": exclusion.get("candidate_name"),
        "failed_unit_count": exclusion.get("failed_unit_count"),
        "reason_code": exclusion.get("reason_code"),
    } != {
        "candidate_name": "itransformer",
        "failed_unit_count": 3,
        "reason_code": "NO_HEALTHY_EPOCH",
    }:
        raise RuntimeError("the sole DART exclusion is not the measured iTransformer 3-unit failure")
    failed = selection.get("failed_units")
    failed_seeds = sorted(
        int(row.get("seed"))
        for row in failed or []
        if isinstance(row, Mapping) and row.get("candidate") == "itransformer"
    )
    if failed_seeds != [20260809, 20260819, 20260829]:
        raise RuntimeError(f"iTransformer failed-seed lineage changed: {failed_seeds}")
    if contract.get("contract_sha256") != CANDIDATE_POOL_CONTRACT_SHA256:
        raise RuntimeError("DART candidate-pool contract hash changed")
    if int(contract.get("eligible_candidate_count", -1)) != 8:
        raise RuntimeError("DART candidate-pool eligible count is not eight")
    return {
        "status": selection["status"],
        "completed_unit_count": 123,
        "failed_unit_count": 3,
        "eligible_completed_unit_count": 108,
        "date_drop_count": 0,
        "global_exclusion": dict(exclusion),
        "failed_seeds": failed_seeds,
        "eligible_candidate_count": 8,
        "candidate_pool_contract_sha256": CANDIDATE_POOL_CONTRACT_SHA256,
    }


def _validate_official_rt_recovery(recovery: Mapping[str, Any]) -> dict[str, Any]:
    if recovery.get("schema_version") != "dcvpp.nyiso_official_rt_recovery.v1":
        raise RuntimeError("unexpected official RT recovery schema")
    if recovery.get("status") != "PARTIAL_OFFICIAL_SOURCE_RECOVERY":
        raise RuntimeError("official RT recovery is not the expected partial outcome")
    if recovery.get("locked_2025_model_results_opened") is not False:
        raise RuntimeError("official recovery opened locked 2025 model results")
    if recovery.get("formal_full_panel_a1_a2_mcs_lcb_winner_defined") is not False:
        raise RuntimeError("official recovery cannot define a full-panel DART winner")
    source_policy = recovery.get("source_policy")
    forbidden = (
        "forecast_price_used",
        "forward_fill_used",
        "interpolation_used",
        "neighbor_zone_substitution_used",
    )
    if not isinstance(source_policy, Mapping) or any(source_policy.get(key) is not False for key in forbidden):
        raise RuntimeError("official RT recovery used a forbidden settlement substitute")
    if source_policy.get("official_nyiso_only") is not True:
        raise RuntimeError("official RT recovery is not NYISO-only")

    phases = recovery.get("phase_summaries")
    coverage = recovery.get("post_recovery_coverage")
    replay = recovery.get("nyiso_replay_by_date")
    if not all(isinstance(value, Mapping) for value in (phases, coverage, replay)):
        raise RuntimeError("official RT recovery lacks phase, coverage, or replay records")
    expected = {
        "selection": (1096, 1086, 10, 116, 90, 26),
        "locked_test": (365, 362, 3, 55, 34, 21),
    }
    summary: dict[str, Any] = {}
    for phase, values in expected.items():
        record = phases.get(phase) if isinstance(phases, Mapping) else None
        post = coverage.get(phase) if isinstance(coverage, Mapping) else None
        replay_record = replay.get(phase) if isinstance(replay, Mapping) else None
        if not all(isinstance(value, Mapping) for value in (record, post, replay_record)):
            raise RuntimeError(f"official RT recovery lacks {phase} records")
        observed = (
            post.get("settlement_evaluable_date_count"),
            post.get("settlement_invalid_date_count"),
            record.get("original_missing_15m_block_count"),
            record.get("recovered_15m_block_count"),
            record.get("unresolved_15m_block_count"),
            replay_record.get("day_count"),
        )
        wanted = (values[1], values[2], values[3], values[4], values[5], values[1])
        if observed != wanted or post.get("date_drop_count") != 0:
            raise RuntimeError(
                f"post-recovery settlement {phase} changed: counts={observed}, "
                f"date_drop_count={post.get('date_drop_count')}"
            )
        summary[phase] = {
            "expected_operating_date_count": values[0],
            "settlement_evaluable_date_count": values[1],
            "settlement_invalid_date_count": values[2],
            "original_missing_settlement_interval_count": values[3],
            "recovered_settlement_interval_count": values[4],
            "missing_settlement_interval_count": values[5],
            "date_drop_count": 0,
        }
    return summary


def _validate_conditional_protocol(
    protocol: Mapping[str, Any], *, root: Path, recovery_path: Path
) -> dict[str, Any]:
    if protocol.get("schema_version") != "dcvpp.dart_conditional_settlement_protocol.v1":
        raise RuntimeError("unexpected conditional settlement protocol schema")
    if protocol.get("status") != "FROZEN_PRE_RESULT_CONDITIONAL_PROTOCOL":
        raise RuntimeError("conditional settlement protocol was not frozen pre-result")
    if protocol.get("terminal_paper_cap") != PARTIAL_PAPER:
        raise RuntimeError("conditional settlement protocol lacks PARTIAL_PAPER cap")
    if protocol.get("claim_scope") != "conditional_decision_value_on_fully_observed_settlement_days":
        raise RuntimeError("conditional settlement claim scope changed")
    if protocol.get("formal_complete_calendar_winner_claim_allowed") is not False:
        raise RuntimeError("complete-calendar DART claims must remain forbidden")
    timing = protocol.get("protocol_timing")
    if not isinstance(timing, Mapping) or {
        "DART_value_result_consumed_by_this_step": timing.get("DART_value_result_consumed_by_this_step"),
        "frozen_before_any_DART_value_result": timing.get("frozen_before_any_DART_value_result"),
        "locked_2025_result_opened_by_this_step": timing.get("locked_2025_result_opened_by_this_step"),
        "solver_launched_by_this_step": timing.get("solver_launched_by_this_step"),
    } != {
        "DART_value_result_consumed_by_this_step": False,
        "frozen_before_any_DART_value_result": True,
        "locked_2025_result_opened_by_this_step": False,
        "solver_launched_by_this_step": False,
    }:
        raise RuntimeError("conditional settlement protocol timing boundary changed")
    contract = protocol.get("candidate_contract")
    if not isinstance(contract, Mapping) or contract.get("candidate_pool_contract_sha256") != CANDIDATE_POOL_CONTRACT_SHA256:
        raise RuntimeError("conditional protocol candidate contract changed")
    phases = protocol.get("phases")
    expected = {"selection": (1096, 1086, 10, 26), "locked_test": (365, 362, 3, 21)}
    summary: dict[str, Any] = {}
    for phase, wanted in expected.items():
        record = phases.get(phase) if isinstance(phases, Mapping) else None
        if not isinstance(record, Mapping):
            raise RuntimeError(f"conditional protocol lacks {phase}")
        observed = (
            record.get("expected_operating_date_count"),
            record.get("settlement_evaluable_date_count"),
            record.get("settlement_invalid_date_count"),
            record.get("missing_settlement_interval_count"),
        )
        if observed != wanted or record.get("retained_operating_date_count") != wanted[0]:
            raise RuntimeError(f"conditional protocol {phase} counts changed: {observed}")
        ledger_path = _resolve(root, str(record.get("ledger_path", "")))
        if not ledger_path.is_file() or sha256_file(ledger_path) != record.get("ledger_sha256"):
            raise RuntimeError(f"conditional protocol {phase} ledger hash changed")
        summary[phase] = {**dict(record), "ledger": _record(root, ledger_path)}
    source = protocol.get("sources", {}).get("official_rt_recovery_manifest", {})
    if source.get("sha256") != sha256_file(recovery_path):
        raise RuntimeError("conditional protocol does not hash-pin the official recovery manifest")
    return {
        "status": protocol["status"],
        "claim_scope": protocol["claim_scope"],
        "terminal_paper_cap": PARTIAL_PAPER,
        "formal_complete_calendar_winner_claim_allowed": False,
        "candidate_pool_contract_sha256": CANDIDATE_POOL_CONTRACT_SHA256,
        "phases": summary,
    }


def _validate_counterfactual_inputs(
    materialization: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if materialization.get("schema_version") != "dcvpp.dart_counterfactual_input_materialization.v1":
        raise RuntimeError("unexpected counterfactual input schema")
    if materialization.get("status") != "SELECTION_INPUTS_COMPLETE_LOCKED_TEST_CLOSED":
        raise RuntimeError("counterfactual selection input materialization is incomplete")
    selection = materialization.get("selection")
    locked = materialization.get("locked_test")
    causal = materialization.get("causal_audit")
    if not all(isinstance(value, Mapping) for value in (selection, locked, causal)):
        raise RuntimeError("counterfactual materialization lacks selection, locked, or causal records")
    if {
        "calendar": selection.get("calendar_day_count"),
        "materialized": selection.get("materialized_day_count"),
        "bundles": selection.get("shared_daily_bundle_count"),
        "training": selection.get("model_training_runs_launched"),
        "prediction": selection.get("model_prediction_runs_launched"),
        "solver": selection.get("solver_runs_launched"),
    } != {"calendar": 1096, "materialized": 1086, "bundles": 1086, "training": 0, "prediction": 0, "solver": 0}:
        raise RuntimeError("counterfactual selection materialization counts changed")
    if selection.get("candidate_independent") is not True:
        raise RuntimeError("counterfactual selection inputs are candidate-dependent")
    if locked.get("status") != "CLOSED_NOT_OPENED" or locked.get("materialized_day_count") != 0:
        raise RuntimeError("locked 2025 counterfactual inputs were opened")
    if locked.get("locked_predictions_or_targets_opened") is not False:
        raise RuntimeError("locked 2025 predictions or targets were opened")
    required_true = (
        "all_common_price_sources_strictly_before_issue",
        "all_issue_pv_sources_deployable_by_issue",
        "all_issue_workload_sources_strictly_before_issue",
        "target_day_pv_only_in_replay_forcing",
    )
    if any(causal.get(key) is not True for key in required_true):
        raise RuntimeError("counterfactual causal input audit failed")
    if causal.get("future_realized_workload_or_pv_used_at_issue") is not False or causal.get("locked_2025_opened") is not False:
        raise RuntimeError("counterfactual causal or locked boundary failed")
    selection_ledger_sha = protocol["phases"]["selection"]["ledger_sha256"]
    if materialization.get("source_hashes", {}).get("settlement_coverage_snapshot") != selection_ledger_sha:
        raise RuntimeError("counterfactual inputs do not hash-pin the conditional selection ledger")
    return {
        "status": materialization["status"],
        "selection_calendar_day_count": 1096,
        "selection_materialized_day_count": 1086,
        "candidate_independent": True,
        "locked_test_status": "CLOSED_NOT_OPENED",
        "locked_2025_opened": False,
        "migration_actions_enabled": materialization.get("claim_boundaries", {}).get("migration_actions_enabled"),
        "materialization_contract_sha256": materialization.get("materialization_contract_sha256"),
    }


def _validate_selection_replay_terminal(replay: Mapping[str, Any]) -> dict[str, Any]:
    if replay.get("schema_version") != "dcvpp.conditional_dart_selection_replay_terminal.v1":
        raise RuntimeError("unexpected conditional selection replay terminal schema")
    if replay.get("terminal") is not True:
        raise RuntimeError("conditional selection replay still has actionable work")
    if replay.get("locked_2025_opened") is not False or replay.get("main_optimization_started") is not False:
        raise RuntimeError("conditional replay crossed a locked or optimization boundary")
    if int(replay.get("actionable_pending_count", -1)) != 0:
        raise RuntimeError("conditional selection replay retains actionable pending work")
    panel = replay.get("panel")
    if not isinstance(panel, Mapping) or {
        "dates": panel.get("selection_valid_date_count"),
        "candidates": panel.get("candidate_count"),
        "experiments": panel.get("experiment_count"),
        "seeds": panel.get("seed_count"),
        "candidate_cells": panel.get("formal_candidate_cell_count"),
    } != {"dates": 1086, "candidates": 8, "experiments": 2, "seeds": 3, "candidate_cells": 52128}:
        raise RuntimeError("conditional selection replay panel changed")
    return {
        "status": replay.get("status"),
        "terminal": True,
        "locked_2025_opened": False,
        "main_optimization_started": False,
        "formal_cells_started": int(replay.get("formal_cells_started", 0) or 0),
        "actionable_pending_count": 0,
        "panel": dict(panel),
        "probe": dict(replay.get("probe", {})) if isinstance(replay.get("probe"), Mapping) else {},
        "blocker_reasons": list(replay.get("blocker_reasons", [])),
    }


def _validate_extension_pause(
    terminal_execution: Mapping[str, Any],
    extension: Mapping[str, Any],
    pause: Mapping[str, Any],
) -> dict[str, Any]:
    policy = terminal_execution.get("forecast_extension")
    if not isinstance(policy, Mapping) or policy.get("status") != "SKIPPED_USER_PAUSED":
        raise RuntimeError("forecast extension is not recorded as user-paused")
    if int(policy.get("seeds_resumed_by_this_path", -1)) != 0:
        raise RuntimeError("terminal path resumed forecast-extension seeds")
    if extension.get("run_id") != "paper-forecast-extensions-v1":
        raise RuntimeError("unexpected forecast-extension run")
    remaining = pause.get("remaining_seeds")
    if remaining != [20260819, 20260829]:
        raise RuntimeError(f"forecast-extension paused-seed lineage changed: {remaining}")
    if int(pause.get("completed_round_seed", -1)) != 20260809:
        raise RuntimeError("forecast-extension completed round is not seed 20260809")
    return {
        "status": "SKIPPED_USER_PAUSED",
        "underlying_manifest_status": extension.get("status"),
        "underlying_manifest_terminal": extension.get("terminal"),
        "completed_round_seed": 20260809,
        "remaining_paused_seeds": [20260819, 20260829],
        "seeds_resumed_by_terminal_path": 0,
    }


def _validate_optimization(execution: Mapping[str, Any]) -> dict[str, Any]:
    if execution.get("terminal") is not True:
        raise RuntimeError(
            "optimization direct execution is not terminal; finalization is intentionally deferred"
        )
    if execution.get("status") != "BLOCKED":
        raise RuntimeError("optimization direct execution is not the expected BLOCKED outcome")
    result = execution.get("optimization_result")
    if not isinstance(result, Mapping):
        raise RuntimeError("optimization direct execution lacks optimization_result")
    pending = int(result.get("actionable_pending_count", result.get("pending_count", -1)) or 0)
    if pending != 0:
        raise RuntimeError(f"optimization retains {pending} actionable pending cells")
    status = str(result.get("status", "")).strip()
    if not status or status.upper() in {"READY", "RUNNING", "PLANNED", "PENDING"}:
        raise RuntimeError(f"optimization result is not terminal: {status!r}")
    if status != "NOT_RUN_PROBE_BLOCKED" or int(result.get("formal_cells_started", -1)) != 0:
        raise RuntimeError("optimization is not the measured zero-formal-cell probe blocker")
    probe = execution.get("probe")
    if not isinstance(probe, Mapping) or probe.get("error_type") != "NoFeasibleSolutionError":
        raise RuntimeError("optimization lacks the measured infeasible B6 probe")
    return {
        "execution_status": execution.get("status"),
        "status": status,
        "completed_count": int(result.get("completed_count", 0) or 0),
        "failure_count": int(result.get("failure_count", 0) or 0),
        "pending_count": int(result.get("pending_count", 0) or 0),
        "actionable_pending_count": 0,
        "formal_cells_started": 0,
        "probe_error_type": "NoFeasibleSolutionError",
        "paths": dict(result.get("paths", {})) if isinstance(result.get("paths"), Mapping) else {},
    }


def _evidence_rows(
    selection: Mapping[str, Any],
    settlement: Mapping[str, Any],
    protocol: Mapping[str, Any],
    inputs: Mapping[str, Any],
    replay: Mapping[str, Any],
    extension: Mapping[str, Any],
    optimization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "stage": "dart_forecast_oos",
            "work_unit": "itransformer",
            "status": "EXCLUDED_NO_HEALTHY_EPOCH",
            "reason": "3/3 formal seeds failed the quantile-crossing health gate; 123 units completed and 3 failed",
            "event_count": 3,
        },
        {
            "stage": "dart_settlement_selection",
            "work_unit": "conditional_2022_2024_panel",
            "status": "CONDITIONAL_VALID_PANEL_WITH_INVALID_LEDGER_DAYS",
            "reason": "official recovery restored 90/116 intervals; 1086 valid days used uniformly and 10 invalid dates remain in the full ledger",
            "event_count": 10,
        },
        {
            "stage": "dart_settlement_locked",
            "work_unit": "sealed_2025_panel",
            "status": "INVALID_BUT_UNOPENED",
            "reason": "official recovery restored 34/55 intervals; 3 invalid dates remain and locked 2025 was not opened",
            "event_count": 3,
        },
        {
            "stage": "dart_conditional_protocol",
            "work_unit": "pre_result_freeze",
            "status": str(protocol.get("status")),
            "reason": "conditional claims only; complete-calendar winner claims remain forbidden",
            "event_count": 1,
        },
        {
            "stage": "dart_counterfactual_inputs",
            "work_unit": "selection_common_panel",
            "status": str(inputs.get("status")),
            "reason": "1086 candidate-independent daily bundles materialized; locked 2025 remains closed",
            "event_count": 1086,
        },
        {
            "stage": "dart_conditional_replay",
            "work_unit": "selection_a1_a2",
            "status": str(replay.get("status")),
            "reason": json.dumps(replay.get("blocker_reasons", []), sort_keys=True, separators=(",", ":")),
            "event_count": int(replay.get("formal_cells_started", 0) or 0),
        },
        {
            "stage": "forecast_extension",
            "work_unit": "seeds_20260819_20260829",
            "status": "SKIPPED_USER_PAUSED_NOT_RESUMED",
            "reason": "round 1 seed 20260809 is retained; remaining two seeds were explicitly paused",
            "event_count": 2,
        },
        {
            "stage": "optimization",
            "work_unit": "paper_optimization_v1",
            "status": str(optimization.get("status")),
            "reason": (
                f"terminal optimization outcome; completed={optimization.get('completed_count', 0)}, "
                f"failed={optimization.get('failure_count', 0)}, actionable_pending=0"
            ),
            "event_count": int(optimization.get("failure_count", 0) or 0),
        },
    ]
    return rows


def _registry_summary(
    root: Path, inventory: Mapping[str, Any], expected_registry_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = inventory.get("artifacts")
    if not isinstance(records, list):
        raise RuntimeError("artifact inventory has no registry records")
    canonical = [
        dict(record)
        for record in records
        if isinstance(record, Mapping) and record.get("requirement") == "mandatory"
    ]
    reported = int(inventory.get("canonical_registry_count", -1))
    if reported != expected_registry_count or len(canonical) != expected_registry_count:
        raise RuntimeError(
            f"canonical artifact registry count changed during inventory: reported={reported}, "
            f"observed={len(canonical)}, expected={expected_registry_count}"
        )
    missing = [record for record in canonical if record.get("status") != "MATERIALIZED"]
    if int(inventory.get("missing_mandatory_count", -1)) != len(missing):
        raise RuntimeError("artifact inventory missing count is inconsistent with its registry")
    paths: dict[str, Any] = {}
    for key in ("inventory_path", "pending_and_skips_path"):
        value = inventory.get(key)
        if value:
            path = _resolve(root, str(value))
            if path.is_file():
                paths[key] = _record(root, path)
    return (
        {
            "status": inventory.get("status"),
            "completed": inventory.get("completed") is True,
            "canonical_registry_count": expected_registry_count,
            "mandatory_count": len(canonical),
            "materialized_mandatory_count": len(canonical) - len(missing),
            "missing_mandatory_count": len(missing),
            "optional_registry_count": int(inventory.get("optional_registry_count", 0) or 0),
            "optional_skipped_count": int(inventory.get("optional_skipped_count", 0) or 0),
            "paths": paths,
        },
        missing,
    )


def finalize_terminal_partial_paper(
    root: Path | str,
    *,
    run_id: str = DEFAULT_RUN_ID,
    paths: Mapping[str, str | Path] | None = None,
    inventory_finalizer: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the terminal ``PARTIAL_PAPER`` record after optimization terminates.

    Both the optimization direct-execution manifest and the optional conditional
    replay terminal manifest are timing gates.  A missing/nonterminal replay
    manifest is allowed only until it appears; once present it must be terminal
    with no actionable work before this finalizer may write terminal state.
    """

    project_root = Path(root).resolve()
    configured = {**DEFAULT_PATHS, **dict(paths or {})}
    resolved = {key: _resolve(project_root, value) for key, value in configured.items()}

    selection_raw = _load_mapping(resolved["selection"], "DART selection manifest")
    recovery_raw = _load_mapping(resolved["official_rt_recovery"], "official RT recovery manifest")
    protocol_raw = _load_mapping(
        resolved["conditional_settlement_protocol"], "conditional settlement protocol"
    )
    inputs_raw = _load_mapping(resolved["counterfactual_inputs"], "counterfactual inputs")
    replay_path = resolved["selection_replay_terminal"]
    replay_raw = (
        _load_mapping(replay_path, "conditional selection replay terminal")
        if replay_path.is_file()
        else {
            "status": "NOT_RECORDED",
            "terminal": True,
            "locked_2025_opened": False,
            "main_optimization_started": False,
            "formal_cells_started": 0,
            "actionable_pending_count": 0,
            "panel": {
                "selection_valid_date_count": 1086,
                "candidate_count": 8,
                "experiment_count": 2,
                "seed_count": 3,
                "formal_candidate_cell_count": 52128,
            },
            "probe": {},
            "blocker_reasons": ["conditional selection replay terminal manifest not yet recorded"],
        }
    )
    terminal_execution_raw = _load_mapping(
        resolved["terminal_path_execution"], "terminal partial execution manifest"
    )
    extension_raw = _load_mapping(resolved["extension"], "forecast extension manifest")
    extension_pause_raw = _load_mapping(resolved["extension_pause"], "forecast extension pause evidence")
    forecast_raw = _load_mapping(resolved["forecast"], "forecast recovery manifest")
    physical_raw = _load_mapping(resolved["physical"], "frozen physical artifact")
    optimization_execution_raw = _load_mapping(
        resolved["optimization_execution"], "optimization direct execution manifest"
    )

    for label, payload in (
        ("selection", selection_raw),
        ("official_rt_recovery", recovery_raw),
        ("conditional_settlement_protocol", protocol_raw),
        ("counterfactual_inputs", inputs_raw),
        ("selection_replay_terminal", replay_raw),
        ("terminal_execution", terminal_execution_raw),
        ("extension", extension_raw),
        ("forecast", forecast_raw),
        ("physical", physical_raw),
        ("optimization", optimization_execution_raw),
    ):
        _assert_no_synthetic_flag(payload, label)
        _validate_self_fingerprint(payload, label)

    selection = _validate_dart_selection(selection_raw)
    settlement = _validate_official_rt_recovery(recovery_raw)
    protocol = _validate_conditional_protocol(
        protocol_raw, root=project_root, recovery_path=resolved["official_rt_recovery"]
    )
    inputs = _validate_counterfactual_inputs(inputs_raw, protocol=protocol)
    replay = (
        _validate_selection_replay_terminal(replay_raw)
        if replay_path.is_file()
        else {
            "status": "NOT_RECORDED",
            "terminal": True,
            "locked_2025_opened": False,
            "main_optimization_started": False,
            "formal_cells_started": 0,
            "actionable_pending_count": 0,
            "panel": dict(replay_raw["panel"]),
            "probe": {},
            "blocker_reasons": list(replay_raw["blocker_reasons"]),
        }
    )
    extension = _validate_extension_pause(
        terminal_execution_raw, extension_raw, extension_pause_raw
    )
    optimization = _validate_optimization(optimization_execution_raw)

    output_root = project_root / "artifacts/paper_finalization" / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_rows = _evidence_rows(
        selection, settlement, protocol, inputs, replay, extension, optimization
    )
    evidence_csv = output_root / "terminal_evidence_ledger.csv"
    _atomic_csv(
        evidence_csv,
        evidence_rows,
        ("stage", "work_unit", "status", "reason", "event_count"),
    )

    # The canonical inventory is the only authority allowed to decide whether
    # a table/figure is materialized.  The dedicated finalizer itself creates
    # no table/figure except the factual failure/evidence ledger above.
    if inventory_finalizer is None:
        from dcvpp.paper_artifacts import (
            CANONICAL_ARTIFACT_REGISTRY,
            finalize_artifact_inventory,
        )

        expected_registry_count = len(CANONICAL_ARTIFACT_REGISTRY)
        inventory_finalizer = finalize_artifact_inventory
    else:
        from dcvpp.paper_artifacts import CANONICAL_ARTIFACT_REGISTRY

        expected_registry_count = len(CANONICAL_ARTIFACT_REGISTRY)

    inventory_inputs = {
        "physical": physical_raw,
        "forecast": forecast_raw,
        "dart": replay_raw,
        "optimization": optimization_execution_raw.get("optimization_result", {}),
        "terminal_partial_evidence": {
            "selection_manifest": _relative(project_root, resolved["selection"]),
            "official_rt_recovery_manifest": _relative(project_root, resolved["official_rt_recovery"]),
            "conditional_settlement_protocol": _relative(project_root, resolved["conditional_settlement_protocol"]),
            "counterfactual_inputs": _relative(project_root, resolved["counterfactual_inputs"]),
            "selection_replay_terminal": (
                _relative(project_root, replay_path) if replay_path.is_file() else None
            ),
            "extension_pause_manifest": _relative(project_root, resolved["extension_pause"]),
            "optimization_execution_manifest": _relative(
                project_root, resolved["optimization_execution"]
            ),
        },
        "paper_table_inputs": {
            "failures.csv": {
                "source": _relative(project_root, evidence_csv),
                "sha256": sha256_file(evidence_csv),
            }
        },
    }
    inventory = dict(inventory_finalizer(project_root, inventory_inputs))
    registry, missing_registry = _registry_summary(
        project_root, inventory, expected_registry_count
    )

    missing_csv = output_root / "canonical_registry_missing.csv"
    _atomic_csv(
        missing_csv,
        [
            {
                "artifact_id": row.get("artifact_id"),
                "category": row.get("category"),
                "relative_path": row.get("relative_path"),
                "status": row.get("status"),
                "reason": row.get("reason"),
            }
            for row in missing_registry
        ],
        ("artifact_id", "category", "relative_path", "status", "reason"),
    )

    source_evidence = {
        key: _record(project_root, path)
        for key, path in resolved.items()
        if path.is_file()
    }
    source_evidence["terminal_evidence_ledger"] = _record(project_root, evidence_csv)
    source_evidence["canonical_registry_missing"] = _record(project_root, missing_csv)

    justification = (
        "Terminal PARTIAL_PAPER: real physical and forecast evidence is retained, "
        "but iTransformer failed 3/3 formal seeds, official recovery leaves 10 selection "
        "and 3 sealed-2025 dates invalid, the registered claim is conditional on 1086 "
        "fully observed selection dates, locked evaluation was not opened, and paused forecast-extension "
        "seeds were not resumed. Missing canonical artifacts remain explicitly missing."
    )
    terminal_manifest = {
        "schema_version": "dcvpp.dedicated_terminal_partial_finalization.v1",
        "run_id": run_id,
        "status": PARTIAL_PAPER,
        "terminal_status": PARTIAL_PAPER,
        "terminal": True,
        "completed": True,
        "paper_results_available": True,
        "synthetic_fallback": False,
        "synthetic_fallback_used": False,
        "materialization_policy": {
            "numeric_table_or_figure_fabrication_allowed": False,
            "missing_registry_items_retained": True,
            "canonical_inventory_is_single_materialization_authority": True,
        },
        "dart_selection": selection,
        "settlement_coverage": settlement,
        "conditional_settlement_protocol": protocol,
        "counterfactual_inputs": inputs,
        "dart_terminal_gate": replay,
        "locked_2025": {
            "opened": False,
            "training_started": False,
            "replay_started": False,
            "winner_defined": False,
        },
        "forecast_extension": extension,
        "optimization": optimization,
        "canonical_artifact_registry": registry,
        "missing_mandatory_registry": [
            {
                "artifact_id": row.get("artifact_id"),
                "category": row.get("category"),
                "relative_path": row.get("relative_path"),
                "status": row.get("status"),
                "reason": row.get("reason"),
            }
            for row in missing_registry
        ],
        "failed_evidence": [
            {"evidence_id": "conditional_selection_replay", "status": replay["status"], "blockers": replay["blocker_reasons"]}
        ],
        "skipped_evidence": [
            {
                "evidence_id": "forecast_extension_remaining_seeds",
                "status": "SKIPPED_USER_PAUSED_NOT_RESUMED",
                "seeds": [20260819, 20260829],
            },
        ],
        "pending_evidence": [],
        "source_evidence": source_evidence,
        "justification": justification,
        "completed_at_utc": utc_now(),
    }
    terminal_manifest["manifest_fingerprint"] = sha256_json(terminal_manifest)
    terminal_manifest_path = output_root / "terminal_partial_manifest.json"
    atomic_json(terminal_manifest_path, terminal_manifest)

    branch_results = {
        "physical": {"status": physical_raw.get("status", "frozen")},
        "forecast": {"status": forecast_raw.get("status")},
        "forecast_extension": extension,
        "dart": {
            "status": replay["status"],
            "locked_2025_opened": False,
            "formal_winner_defined": False,
            "claim_scope": protocol["claim_scope"],
        },
        "optimization": optimization,
    }
    execution_manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "status": PARTIAL_PAPER,
        "terminal_status": PARTIAL_PAPER,
        "terminal": True,
        "paper_results_available": True,
        "synthetic_fallback": False,
        "branch_results": branch_results,
        "terminal_partial_manifest": _record(project_root, terminal_manifest_path),
        "canonical_artifact_registry": registry,
        "updated_at_utc": utc_now(),
    }
    atomic_json(project_root / "artifacts/manifests/paper_execution_manifest.json", execution_manifest)

    run_manifest = {
        "schema_version": 3,
        "profile": "paper",
        "run_id": run_id,
        "status": PARTIAL_PAPER,
        "terminal_status": PARTIAL_PAPER,
        "terminal": True,
        "paper_training_started": True,
        "paper_results_available": True,
        "synthetic_fallback": False,
        "justification": justification,
        "branch_results": branch_results,
        "canonical_artifact_inventory": registry,
        "terminal_partial_manifest": _record(project_root, terminal_manifest_path),
        "completed_at_utc": utc_now(),
    }
    run_manifest_path = project_root / "artifacts/manifests/run_manifest.json"
    atomic_json(run_manifest_path, run_manifest)

    state_path = project_root / "artifacts/state/pipeline_state.json"
    state = _load_mapping(state_path, "pipeline state") if state_path.is_file() else {}
    state.update(
        {
            "run_id": run_id,
            "profile": "paper",
            "terminal_status": PARTIAL_PAPER,
            "paper": {
                "run_status": PARTIAL_PAPER,
                "terminal": True,
                "terminal_status": PARTIAL_PAPER,
                "paper_training_started": True,
                "paper_results_available": True,
                "synthetic_fallback": False,
                "branch_results": branch_results,
            },
            "updated_at": utc_now(),
        }
    )
    state.setdefault("stages", {})["paper_execution"] = {
        "name": "paper_execution",
        "status": "completed",
        "terminal": True,
        "terminal_status": PARTIAL_PAPER,
        "finished_at": utc_now(),
        "outputs": [
            _relative(project_root, terminal_manifest_path),
            _relative(project_root, run_manifest_path),
            _relative(project_root, missing_csv),
        ],
        "message": justification,
    }
    atomic_json(state_path, state)
    atomic_text(
        project_root / "artifacts/status/CURRENT_STAGE_STATUS.md",
        "\n".join(
            [
                "# Current Stage Status",
                "",
                f"- Run ID: `{run_id}`",
                "- Stage: `paper_execution`",
                "- Terminal status: `PARTIAL_PAPER`",
                "- Terminal: `true`",
                "- Synthetic fallback: `false`",
                "- DART forecast units: `123 completed / 3 failed`",
                "- Globally excluded candidate: `iTransformer (NO_HEALTHY_EPOCH)`",
                "- Conditional settlement-valid dates: `1086 selection / 362 locked`",
                "- Settlement-invalid ledger dates: `10 selection / 3 locked`",
                f"- Conditional selection replay: `{replay['status']}`",
                "- Locked 2025 opened: `false`",
                "- Forecast extension: `paused; seeds 20260819 and 20260829 not resumed`",
                f"- Optimization: `{optimization['status']}`",
                f"- Canonical registry: `{registry['materialized_mandatory_count']}/{registry['mandatory_count']} materialized; {registry['missing_mandatory_count']} missing`",
                f"- Terminal manifest: `{_relative(project_root, terminal_manifest_path)}`",
                f"- Updated: `{state['updated_at']}`",
                "",
            ]
        ),
    )

    return {
        "status": PARTIAL_PAPER,
        "terminal_status": PARTIAL_PAPER,
        "terminal": True,
        "run_id": run_id,
        "terminal_partial_manifest": _relative(project_root, terminal_manifest_path),
        "run_manifest": _relative(project_root, run_manifest_path),
        "canonical_registry_count": registry["canonical_registry_count"],
        "materialized_mandatory_count": registry["materialized_mandatory_count"],
        "missing_mandatory_count": registry["missing_mandatory_count"],
        "optimization_status": optimization["status"],
        "locked_2025_opened": False,
    }


__all__ = [
    "DEFAULT_PATHS",
    "DEFAULT_RUN_ID",
    "PARTIAL_PAPER",
    "finalize_terminal_partial_paper",
]
