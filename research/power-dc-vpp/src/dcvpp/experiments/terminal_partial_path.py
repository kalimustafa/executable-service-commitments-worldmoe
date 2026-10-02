from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dcvpp.core import atomic_json, load_yaml, sha256_file, sha256_json, utc_now


TERMINAL_PARTIAL_RUN_ID = "paper-terminal-partial-v1"
OPTIMIZATION_RUN_ID = "paper-optimization-v1"
TERMINAL_ROOT = Path("artifacts/run_control/terminal_partial_path")
DART_RESULT_PATH = TERMINAL_ROOT / "dart_blocked_result.json"
EXECUTION_MANIFEST_PATH = TERMINAL_ROOT / "execution_manifest.json"
PARTIAL_PAPER = "PARTIAL_PAPER"

_SELECTION_STAGE = Path(
    "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/"
    "forecast_oos/selection_stage_manifest.json"
)
_REAL_INPUT_PREPARATION = Path(
    "artifacts/dart_value_select/real_input_preparation/v1/"
    "dart_real_evidence_input_preparation_manifest.json"
)
_COUNTERFACTUAL_INPUT_DESIGN = Path(
    "artifacts/dart_value_select/real_input_preparation/v1/"
    "counterfactual_input_design_manifest.json"
)


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required evidence is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"required evidence is not a JSON mapping: {path}")
    return dict(value)


def _record(root: Path, relative: Path) -> dict[str, Any]:
    path = (root / relative).resolve()
    path.relative_to(root)
    if not path.is_file():
        raise ValueError(f"required evidence is absent: {relative.as_posix()}")
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _phase_input(phase: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    inputs = phase.get("inputs")
    value = inputs.get(name) if isinstance(inputs, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"real-input phase has no {name} evidence")
    return value


def build_dart_blocked_result(root: Path) -> dict[str, Any]:
    """Freeze an honest terminal DART failure without opening locked 2025.

    The existing forecast checkpoint pool remains scientific evidence, but it
    is not admitted as a DART winner because the real replay inputs do not form
    the required unchanged date panels.  No missing item is reported as
    pending: every downstream item is skipped behind the hard evidence gate.
    """

    root = Path(root).resolve()
    selection = _mapping(root / _SELECTION_STAGE)
    preparation = _mapping(root / _REAL_INPUT_PREPARATION)
    design = _mapping(root / _COUNTERFACTUAL_INPUT_DESIGN)

    from dcvpp.experiments.dart_partial_candidate import (
        validate_partial_candidate_pool_contract,
    )

    if str(selection.get("status", "")).upper() != "PARTIAL_WITH_FAILED_CANDIDATE":
        raise ValueError("DART forecast selection is not the frozen partial pool")
    pool = validate_partial_candidate_pool_contract(
        selection.get("candidate_pool_contract")
    )
    if selection.get("downstream_ready") is not False:
        raise ValueError("DART selection unexpectedly declares downstream readiness")
    if str(preparation.get("status", "")).upper() != "BLOCKED":
        raise ValueError("real DART input preparation is not fail-closed BLOCKED")
    if str(design.get("status", "")).upper() != "BLOCKED_BY_SETTLEMENT_COVERAGE":
        raise ValueError("counterfactual input design no longer has the audited blocker")
    if design.get("locked_2025_opened") is not False:
        raise ValueError("locked 2025 must remain unopened")
    if design.get("paper_terminal_cap") != PARTIAL_PAPER:
        raise ValueError("counterfactual input design does not cap the paper at partial")

    phases = preparation.get("phases")
    if not isinstance(phases, Mapping):
        raise ValueError("real DART input preparation has no phase evidence")
    selection_phase = phases.get("selection")
    locked_phase = phases.get("locked_test")
    if not isinstance(selection_phase, Mapping) or not isinstance(locked_phase, Mapping):
        raise ValueError("real DART input preparation has incomplete phase evidence")
    if str(selection_phase.get("status", "")).upper() != "BLOCKED":
        raise ValueError("selection real-input panel is not blocked")
    if str(locked_phase.get("status", "")).upper() != "BLOCKED":
        raise ValueError("locked real-input panel is not blocked")

    selection_issue = _phase_input(selection_phase, "issue_time")
    selection_action = _phase_input(selection_phase, "actionability")
    selection_forcing = _phase_input(selection_phase, "forcing")
    locked_issue = _phase_input(locked_phase, "issue_time")
    locked_action = _phase_input(locked_phase, "actionability")
    locked_forcing = _phase_input(locked_phase, "forcing")
    if any(
        int(value.get("missing_date_count", -1)) != 1096
        for value in (selection_issue, selection_action, selection_forcing)
    ):
        raise ValueError("selection missing-date evidence differs from the frozen audit")
    if int(locked_issue.get("missing_date_count", -1)) != 0:
        raise ValueError("locked issue-time forecast index is not complete")
    if any(
        int(value.get("missing_date_count", -1)) != 365
        for value in (locked_action, locked_forcing)
    ):
        raise ValueError("locked actionability/forcing evidence differs from the audit")

    coverage = design.get("settlement_coverage")
    selection_coverage = coverage.get("selection") if isinstance(coverage, Mapping) else None
    locked_coverage = coverage.get("locked_test") if isinstance(coverage, Mapping) else None
    if not isinstance(selection_coverage, Mapping) or not isinstance(locked_coverage, Mapping):
        raise ValueError("settlement coverage evidence is incomplete")

    failed_evidence = [
        {
            "evidence_id": "dart_selection_exact_settlement_panel",
            "status": "FAILED_HARD_GATE",
            "invalid_date_count": int(selection_coverage.get("settlement_invalid_date_count", -1)),
            "missing_interval_count": int(selection_coverage.get("missing_settlement_interval_count", -1)),
            "date_drop_count": int(selection_coverage.get("date_drop_count", -1)),
        },
        {
            "evidence_id": "dart_selection_issue_time_panel",
            "status": "FAILED_HARD_GATE",
            "missing_date_count": int(selection_issue.get("missing_date_count", -1)),
        },
        {
            "evidence_id": "dart_selection_actionability_panel",
            "status": "FAILED_HARD_GATE",
            "missing_date_count": int(selection_action.get("missing_date_count", -1)),
        },
        {
            "evidence_id": "dart_selection_replay_forcing_panel",
            "status": "FAILED_HARD_GATE",
            "missing_date_count": int(selection_forcing.get("missing_date_count", -1)),
        },
        {
            "evidence_id": "dart_locked_exact_settlement_panel",
            "status": "FAILED_BUT_UNOPENED",
            "invalid_date_count": int(locked_coverage.get("settlement_invalid_date_count", -1)),
            "missing_interval_count": int(locked_coverage.get("missing_settlement_interval_count", -1)),
            "date_drop_count": int(locked_coverage.get("date_drop_count", -1)),
        },
    ]
    skipped_evidence = [
        {"evidence_id": key, "status": "SKIPPED_UPSTREAM_HARD_GATE"}
        for key in (
            "dart_a1_replay",
            "dart_a2_replay",
            "dart_a3_ranking_contradiction",
            "dart_mcs_lcb_selection",
            "dart_static_rule_freeze",
            "locked_2025_training",
            "locked_2025_replay",
        )
    ]

    payload: dict[str, Any] = {
        "schema_version": "dcvpp.terminal_dart_blocked_result.v1",
        "run_id": TERMINAL_PARTIAL_RUN_ID,
        "created_at_utc": utc_now(),
        "status": "BLOCKED_REAL_EVIDENCE_INPUTS",
        "terminal": True,
        "completed": False,
        "downstream_ready": False,
        "actionable_pending_count": 0,
        "paper_results_available": True,
        "terminal_status_cap": PARTIAL_PAPER,
        "candidate_pool_contract": pool,
        "locked_2025_opened": False,
        "locked_2025_training_started": False,
        "locked_2025_replay_started": False,
        "formal_a1_a2_mcs_lcb_winner_defined": False,
        "synthetic_fallback_used": False,
        "selected_scenario_as_realized": False,
        "failed_evidence": failed_evidence,
        "skipped_evidence": skipped_evidence,
        "pending_evidence": [],
        "source_evidence": {
            "selection_stage": _record(root, _SELECTION_STAGE),
            "real_input_preparation": _record(root, _REAL_INPUT_PREPARATION),
            "counterfactual_input_design": _record(root, _COUNTERFACTUAL_INPUT_DESIGN),
        },
    }
    payload["manifest_fingerprint"] = sha256_json(payload)
    output = root / DART_RESULT_PATH
    atomic_json(output, payload)
    result = dict(payload)
    result["manifest_path"] = DART_RESULT_PATH.as_posix()
    result["manifest_file_sha256"] = sha256_file(output)
    return result


def optimization_config_for_terminal_partial(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact three-seed, 96-day-only optimization configuration."""

    value = copy.deepcopy(dict(config))
    seeds = tuple(map(int, value.get("seed_list", ())))
    if seeds != (20260809, 20260819, 20260829):
        raise ValueError("terminal partial optimization requires the frozen three-seed panel")
    optimization = dict(value.get("optimization", {}))
    optimization["full_year_extension"] = False
    value["optimization"] = optimization
    branch = dict(value.get("optimization_branch", {}))
    branch["full_year_extension"] = False
    branch["run_ofat_sensitivities"] = True
    branch["seed_list"] = list(seeds)
    value["optimization_branch"] = branch
    return value


def expected_main96_cell_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Expose the preregistered workload before any solver process is started."""

    value = optimization_config_for_terminal_partial(config)
    seeds = len(value["seed_list"])
    from dcvpp.research_branch.runner import (
        _default_cells,
        _resolve_regimes,
        build_ofat_variants,
    )

    reference_per_day = len(
        _default_cells(
            ["2000-01-01"],
            seeds=value["seed_list"],
            counts=value["optimization_branch"].get(
                "scenario_counts", (10, 20, 40, 80)
            ),
            regimes=_resolve_regimes(value["optimization_branch"].get("regimes")),
            phase="primary_96_day",
            include_h6_variant=isinstance(
                value["optimization_branch"].get("h6_constraints"), Mapping
            ),
        )
    )
    primary_reference = 96 * reference_per_day
    # build_ofat_variants is deterministic and performs no experiment work.
    ofat_variants = len(build_ofat_variants(value["optimization_branch"], value))
    ofat_cells = 96 * ofat_variants
    timezone_auxiliary = 96 * seeds * 2
    return {
        "reference_per_day": reference_per_day,
        "primary_reference_cells": primary_reference,
        "ofat_variant_count": ofat_variants,
        "ofat_cells": ofat_cells,
        "timezone_auxiliary_cells": timezone_auxiliary,
        "total_cells": primary_reference + ofat_cells + timezone_auxiliary,
        "full_year_extension_cells": 0,
    }


def write_execution_manifest(
    root: Path,
    *,
    dart_result: Mapping[str, Any],
    workload_normalization: Mapping[str, Any] | None = None,
    optimization_result: Mapping[str, Any] | None = None,
    terminal_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    normalization = dict(workload_normalization or {})
    optimization = dict(optimization_result or {})
    terminal = dict(terminal_result or {})
    payload: dict[str, Any] = {
        "schema_version": "dcvpp.terminal_partial_execution.v1",
        "run_id": TERMINAL_PARTIAL_RUN_ID,
        "updated_at_utc": utc_now(),
        "required_order": [
            "frozen_physical",
            "frozen_forecast_v1",
            "forecast_extension_user_pause",
            "dart_selection_real_evidence_gate",
            "workload_unit_normalization_gate",
            "main96_optimization",
            "statistics_and_artifacts",
            "terminal_finalization",
        ],
        "dart_result": {
            "status": dart_result.get("status"),
            "manifest_path": dart_result.get("manifest_path"),
            "manifest_file_sha256": dart_result.get("manifest_file_sha256"),
        },
        "forecast_extension": {
            "status": "SKIPPED_USER_PAUSED",
            "seeds_resumed_by_this_path": 0,
            "actionable_pending_count": 0,
        },
        "workload_normalization": normalization,
        "optimization": optimization,
        "terminal_result": terminal,
        "failed_evidence": list(dart_result.get("failed_evidence", [])),
        "skipped_evidence": [
            *list(dart_result.get("skipped_evidence", [])),
            {
                "evidence_id": "forecast_extension_remaining_seeds",
                "status": "SKIPPED_USER_PAUSED_NOT_RESUMED",
            },
        ],
        "pending_evidence": [],
        "locked_2025_opened": False,
        "paper_terminal_status_cap": PARTIAL_PAPER,
    }
    optimization_pending = int(
        optimization.get("actionable_pending_count", optimization.get("pending_count", 0))
        or 0
    )
    if optimization_pending > 0:
        payload["status"] = "MAIN96_OPTIMIZATION_RESUMABLE"
        payload["terminal"] = False
        payload["pending_evidence"] = [
            {
                "evidence_id": "main96_optimization_cells",
                "status": "PENDING_ACTIONABLE",
                "count": optimization_pending,
            }
        ]
    elif str(terminal.get("status", "")).upper() == PARTIAL_PAPER:
        payload["status"] = PARTIAL_PAPER
        payload["terminal"] = True
    elif optimization:
        payload["status"] = "READY_FOR_TERMINAL_FINALIZATION"
        payload["terminal"] = False
    else:
        payload["status"] = "PREPARED_NO_EXPERIMENT_STARTED"
        payload["terminal"] = False
    payload["manifest_fingerprint"] = sha256_json(payload)
    path = root / EXECUTION_MANIFEST_PATH
    atomic_json(path, payload)
    result = dict(payload)
    result["manifest_path"] = EXECUTION_MANIFEST_PATH.as_posix()
    result["manifest_file_sha256"] = sha256_file(path)
    return result


def prepare_terminal_partial_path(
    root: Path,
    config_path: Path = Path("configs/paper.yaml"),
) -> dict[str, Any]:
    """Write only control evidence; never start training or optimization."""

    root = Path(root).resolve()
    path = config_path if config_path.is_absolute() else root / config_path
    config = load_yaml(path)
    dart = build_dart_blocked_result(root)
    manifest = write_execution_manifest(root, dart_result=dart)
    return {
        "status": manifest["status"],
        "terminal": False,
        "optimization_started": False,
        "dart_result": dart,
        "execution_manifest": manifest,
        "expected_cells": expected_main96_cell_counts(config),
        "optimization_run_id": OPTIMIZATION_RUN_ID,
    }


def blocked_optimization_result(
    *, reason: str,
    admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Represent a non-actionable scientific input blocker for finalization."""

    return {
        "status": "BLOCKED_INPUT_UNIT_CONTRACT",
        "terminal": True,
        "completed_count": 0,
        "failed_count": 0,
        "pending_count": 0,
        "actionable_pending_count": 0,
        "paper_results_available": False,
        "reason": reason,
        "workload_normalization_admission": dict(admission or {}),
        "failed_evidence": [
            {
                "evidence_id": "workload_raw_to_normalized_unit_contract",
                "status": "FAILED_HARD_GATE",
                "reason": reason,
            }
        ],
        "pending_evidence": [],
    }


def _preflight_result(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifacts/manifests/run_manifest.json"
    manifest = _mapping(manifest_path)
    if str(manifest.get("status", "")).upper() != "PAPER_PREFLIGHT_READY":
        raise ValueError("frozen paper preflight is not ready")
    source = manifest.get("source_manifest")
    timezone = manifest.get("timezone_manifest")
    if not isinstance(source, Mapping) or not isinstance(timezone, Mapping):
        raise ValueError("frozen paper preflight has no source/timezone records")
    for label, record in (("source", source), ("timezone", timezone)):
        path = root / str(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"frozen {label} manifest hash changed")
    return {
        "status": "PAPER_PREFLIGHT_READY",
        "source_manifest": str(source["path"]),
        "timezone_manifest": str(timezone["path"]),
        "timezone_gate": manifest.get("timezone_gate", {}),
        "run_manifest": "artifacts/manifests/run_manifest.json",
        "run_manifest_sha256": sha256_file(manifest_path),
    }


def _existing_branch_results(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify and load the already-complete physical, v1 and readiness evidence."""

    from dcvpp.experiments.dart_pipeline import (
        audit_v1_dart_readiness_evidence,
    )
    from dcvpp.paper_execution import (
        _forecast_config,
        _run_or_resume_forecasting_branch,
        _run_or_resume_physical_branch,
    )

    execution = dict(config.get("paper_execution", {}))

    def forbidden_runner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("terminal partial path may only consume frozen completed evidence")

    physical = _run_or_resume_physical_branch(
        root,
        config,
        str(execution.get("physical_run_id", "paper-physical-v1")),
        forbidden_runner,
    )
    forecast = _run_or_resume_forecasting_branch(
        root,
        _forecast_config(config, physical),
        str(execution.get("forecast_run_id", "paper-forecast-v1")),
        execution,
        forbidden_runner,
    )
    readiness_path = (
        root
        / "artifacts/dart_value_select/v1_readiness"
        / f"run_id={execution.get('dart_v1_readiness_run_id', 'paper-dart-v1-readiness-v8')}"
        / "v1_dart_readiness_manifest.json"
    )
    readiness = {
        "status": "COMPLETE",
        "manifest_path": readiness_path.relative_to(root).as_posix(),
        "manifest_file_sha256": sha256_file(readiness_path),
        "downstream_ready": True,
        "actionable_pending_count": 0,
    }
    readiness_audit = audit_v1_dart_readiness_evidence(
        root,
        readiness,
        expected_v1_manifest=root / str(forecast["run_manifest"]),
    )
    readiness["readiness_audit"] = readiness_audit
    if readiness_audit.get("passed") is not True:
        raise ValueError("frozen v1 DART-readiness evidence failed revalidation")
    return physical, forecast, readiness


def run_or_resume_terminal_partial_path(
    root: Path,
    config_path: Path = Path("configs/paper.yaml"),
    *,
    normalization_preparer: Callable[..., Mapping[str, Any]] | None = None,
    optimization_runner: Callable[..., dict[str, Any]] | None = None,
    finalizer: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resume the independent main96 branch, then truthfully finalize partial.

    No DART, locked-2025, or forecast-extension runner is reachable here.  The
    optimization runner is reachable only after its raw-workload unit bridge
    returns a hash-pinned ``ADMITTED`` forecast payload.
    """

    root = Path(root).resolve()
    path = config_path if config_path.is_absolute() else root / config_path
    config = optimization_config_for_terminal_partial(load_yaml(path))
    dart = build_dart_blocked_result(root)
    preflight = _preflight_result(root)
    physical, forecast_v1, readiness = _existing_branch_results(root, config)

    if normalization_preparer is None:
        try:
            from dcvpp.research_branch.workload_unit_bridge import (
                prepare_normalized_optimization_forecasts,
            )
        except ImportError:
            normalization = {
                "status": "BLOCKED",
                "admitted": False,
                "reason": "workload normalization bridge is not installed",
            }
        else:
            normalization = dict(
                prepare_normalized_optimization_forecasts(
                    root, run_id=OPTIMIZATION_RUN_ID
                )
            )
    else:
        normalization = dict(
            normalization_preparer(root, run_id=OPTIMIZATION_RUN_ID)
        )

    admitted = (
        str(normalization.get("status", "")).upper() == "ADMITTED"
        and normalization.get("admitted") is True
    )
    normalized_forecast = normalization.get("forecast_result")
    if admitted and not isinstance(normalized_forecast, Mapping):
        index_path = normalization.get("index_path")
        index_sha = normalization.get("index_sha256")
        normalized_forecast = {
            **forecast_v1,
            "status": "COMPLETE",
            "optimization_allowed": True,
            "forecast_day_index": {
                "index_path": index_path,
                "index_sha256": index_sha,
            },
            "frozen_day_index": {
                "index_path": index_path,
                "index_sha256": index_sha,
            },
            "workload_normalization_admission": normalization,
        }
    if admitted:
        if optimization_runner is None:
            from dcvpp.research_branch import run_optimization_branch

            optimization_runner = run_optimization_branch
        optimization = optimization_runner(
            root,
            config,
            OPTIMIZATION_RUN_ID,
            physical,
            dict(normalized_forecast),
        )
    else:
        optimization = blocked_optimization_result(
            reason=str(
                normalization.get(
                    "reason", "workload normalization was not scientifically admitted"
                )
            ),
            admission=normalization,
        )

    pending = int(
        optimization.get("actionable_pending_count", optimization.get("pending_count", 0))
        or 0
    )
    if pending > 0:
        execution = write_execution_manifest(
            root,
            dart_result=dart,
            workload_normalization=normalization,
            optimization_result=optimization,
        )
        return {
            "status": "MAIN96_OPTIMIZATION_RESUMABLE",
            "terminal": False,
            "actionable_pending_count": pending,
            "optimization_result": optimization,
            "execution_manifest": execution,
        }

    if finalizer is None:
        # The dedicated finalizer consumes this stable direct-execution
        # handoff, inventories the real partial evidence, and never requires
        # a completed DART replay chain.
        direct_path = (
            root
            / "artifacts/optimization_branch"
            / OPTIMIZATION_RUN_ID
            / "direct_execution_manifest.json"
        )
        direct: dict[str, Any] = {
            "schema_version": "dcvpp.direct_optimization_execution.v1",
            "run_id": OPTIMIZATION_RUN_ID,
            "created_at_utc": utc_now(),
            "status": str(optimization.get("status", "")),
            "terminal": True,
            "optimization_result": optimization,
            "workload_normalization_admission": normalization,
        }
        direct["manifest_fingerprint"] = sha256_json(direct)
        atomic_json(direct_path, direct)
        from dcvpp.terminal_partial_finalization import (
            finalize_terminal_partial_paper,
        )

        terminal = finalize_terminal_partial_paper(
            root,
            run_id=TERMINAL_PARTIAL_RUN_ID,
            paths={"optimization_execution": direct_path},
        )
    else:
        terminal = finalizer(
            root,
            config,
            TERMINAL_PARTIAL_RUN_ID,
            preflight,
            physical,
            dict(normalized_forecast) if admitted else forecast_v1,
            optimization,
            dart_result=dart,
            dart_v1_readiness_result=readiness,
            state=None,
        )
    if str(terminal.get("status", "")).upper() != PARTIAL_PAPER:
        raise RuntimeError("strict terminal path may only finalize as PARTIAL_PAPER")
    execution = write_execution_manifest(
        root,
        dart_result=dart,
        workload_normalization=normalization,
        optimization_result=optimization,
        terminal_result=terminal,
    )
    return {**terminal, "execution_manifest": execution}


__all__ = [
    "DART_RESULT_PATH",
    "EXECUTION_MANIFEST_PATH",
    "OPTIMIZATION_RUN_ID",
    "TERMINAL_PARTIAL_RUN_ID",
    "blocked_optimization_result",
    "build_dart_blocked_result",
    "expected_main96_cell_counts",
    "optimization_config_for_terminal_partial",
    "prepare_terminal_partial_path",
    "run_or_resume_terminal_partial_path",
    "write_execution_manifest",
]
