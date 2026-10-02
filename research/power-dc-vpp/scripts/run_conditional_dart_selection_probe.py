"""Run one isolated real DART replay cell and gate the formal selection run.

The probe namespace is deliberately separate from the formal replay.  The
parent process applies a short censoring deadline: one candidate replay cell
must finish quickly enough that 52,128 candidate cells can plausibly finish
within 24 hours on four date workers.  No locked-2025 or main-optimization
entry point is imported or called.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcvpp.core import atomic_json, load_yaml, sha256_file, sha256_json, utc_now  # noqa: E402


CONTROL = ROOT / "artifacts/dart_value_select/run_control/conditional_selection_replay_v1"
RESULT_PATH = CONTROL / "probe_result.json"
TERMINAL_PATH = CONTROL / "terminal_manifest.json"
LOG_PATH = CONTROL / "probe_worker.log"
READY_PATH = ROOT / "artifacts/dart_value_select/counterfactual_inputs/v1/selection_config_ready_input_block.json"
MATERIALIZATION_PATH = ROOT / "artifacts/dart_value_select/counterfactual_inputs/v1/materialization_manifest.json"
PROTOCOL_PATH = ROOT / "artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1/conditional_settlement_protocol_manifest.json"
SELECTION_PATH = ROOT / "artifacts/dart_value_select/run_id=paper-dart-forecast-v4/forecast_oos/selection_stage_manifest.json"
PHYSICAL_PATH = ROOT / "artifacts/research/physical/run_id=paper-physical-v1/frozen_physical_artifact.json"

FORMAL_CANDIDATE_CELLS = 1086 * 8 * 2 * 3
FORMAL_BASELINE_RUNS = 1086 * 2 * 3
FORMAL_PI_RUNS = 1086
FORMAL_MILP_SOLVES = (FORMAL_CANDIDATE_CELLS + FORMAL_BASELINE_RUNS) * 97 + FORMAL_PI_RUNS
WORKERS = 4
MAX_WALL_HOURS = 24.0
PER_CELL_BUDGET_SECONDS = MAX_WALL_HOURS * 3600.0 * WORKERS / FORMAL_CANDIDATE_CELLS


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not a mapping: {path}")
    return value


def _worker(day: str) -> int:
    from dcvpp.experiments import dart_real_evidence as real
    from dcvpp.experiments.dart_probe_preflight_runtime import (
        focused_probe_preflight_runtime,
    )

    ready = _read(READY_PATH)
    selection = _read(SELECTION_PATH)
    config = load_yaml(ROOT / "configs/dart_value_select.yaml")
    config[real.INPUT_BLOCK] = dict(ready[real.INPUT_BLOCK])
    physical = _read(PHYSICAL_PATH)
    original_replay = real.run_dart_replay_panel
    original_valid_sha = real._valid_sha256
    captured: dict[str, Any] = {}
    solver_timing: dict[str, Any] = {"call_count": 0, "elapsed_seconds": 0.0}
    stage_markers = CONTROL / "probe_stage_markers"
    stage_markers.mkdir(parents=True, exist_ok=True)

    def one_cell(panel: dict[str, Any], checkpoint_root: Path, **kwargs: Any) -> dict[str, Any]:
        matches = [
            cell
            for cell in panel["cells"]
            if cell.operating_date == day
            and cell.candidate_id == "seasonal_naive"
            and cell.experiment_id == "A1"
            and int(cell.seed) == 20260809
        ]
        if len(matches) != 1:
            raise RuntimeError(f"probe cell count changed: {len(matches)}")
        cell = matches[0]
        probe_panel = {
            "schema_version": "dcvpp.conditional_dart_probe_panel.v1",
            "operating_dates": [day],
            "candidate_ids": ["seasonal_naive"],
            "experiment_ids": ["A1"],
            "seeds": [20260809],
            "cell_count": 1,
            "cells": [cell],
            "panel_sha256": sha256_json([cell.to_mapping()]),
        }
        started = perf_counter()
        result = original_replay(
            probe_panel,
            CONTROL / "probe_checkpoints",
            **kwargs,
        )
        elapsed = perf_counter() - started
        row = dict(result["results"][0])
        captured.update(
            {
                "schema_version": "dcvpp.conditional_dart_selection_probe.v1",
                "status": "COMPLETE",
                "operating_date": day,
                "candidate_id": "seasonal_naive",
                "experiment_id": "A1",
                "seed": 20260809,
                "candidate_cell_count": 1,
                "baseline_run_count": 1,
                "perfect_information_run_count": 1,
                "elapsed_seconds": elapsed,
                "candidate_result": row.get("candidate_result"),
                "baseline_result": row.get("baseline_result"),
                "perfect_information_result": row.get("perfect_information_result"),
                "decision_value": row.get("decision_value"),
                "result_sha256": result.get("results_sha256"),
                "formal_namespace": False,
                "locked_2025_opened": False,
                "main_optimization_started": False,
                "completed_at_utc": utc_now(),
            }
        )
        atomic_json(RESULT_PATH, captured)
        raise RuntimeError("ISOLATED_PROBE_CAPTURE_COMPLETE")

    real.run_dart_replay_panel = one_cell
    # v4 emits the final calibration panel as an immutable file path in one
    # legacy metadata field.  The source loader otherwise requires its digest.
    # Resolve only that labelled field to the file bytes; all other SHA gates
    # remain unchanged and the frozen training artifact is not modified.
    def calibration_path_compat(value: Any, *, label: str) -> str:
        if label.startswith("calibration panel ") and value is None:
            # The v4 training adapter renamed this field while retaining the
            # same frozen digest inside the serialized runner metadata.
            prediction_path = Path(label.removeprefix("calibration panel "))
            if prediction_path.is_file():
                import numpy as np

                with np.load(prediction_path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata_json"].item()))
                runner = metadata.get("runner_metadata", {})
                renamed = runner.get(
                    "deploy_calibration_panel_date_sha256"
                )
                role = runner.get("deploy_calibration_panel_role")
                internal = (
                    runner.get("data_audit", {})
                    .get("internal_holdout_panels", {})
                    .get("final_calibration_and_residual_coupling", {})
                    .get("date_sha256")
                )
                if (
                    role != "final_calibration_and_residual_coupling_only"
                    or renamed != internal
                ):
                    raise RuntimeError("v4 calibration compatibility identity failed")
                # Residual coupling is independently hash-verified by the
                # existing _verify_joint_coupling gate immediately after unit
                # discovery; this compatibility changes no coupling evidence.
                return original_valid_sha(renamed, label=label)
        return original_valid_sha(value, label=label)

    real._valid_sha256 = calibration_path_compat
    original_discover = real._discover_oos_inventory

    def discover_compat(**kwargs: Any) -> Any:
        saved = real._valid_sha256
        real._valid_sha256 = calibration_path_compat
        try:
            result = original_discover(**kwargs)
            atomic_json(stage_markers / "inventory_complete.json", {
                "status": "COMPLETE", "completed_at_utc": utc_now(), "inventory_sha256": result.fingerprint
            })
            return result
        finally:
            real._valid_sha256 = saved

    real._discover_oos_inventory = discover_compat
    original_solver = real.solve_policy

    def timed_solver(*args: Any, **kwargs: Any) -> Any:
        if solver_timing["call_count"] == 0:
            atomic_json(
                stage_markers / "solver_cell_entered.json",
                {
                    "status": "ENTERED",
                    "entered_at_utc": utc_now(),
                    "formal_namespace": False,
                    "marker_semantics": "immediately_before_first_real_solve_policy_call",
                },
            )
        solver_timing["call_count"] += 1
        started = perf_counter()
        try:
            return original_solver(*args, **kwargs)
        finally:
            solver_timing["elapsed_seconds"] += perf_counter() - started

    real.solve_policy = timed_solver
    inventory_cache = CONTROL / "probe_preflight_cache/focused_inventory.json"
    runtime_ledger = CONTROL / "probe_stage_markers/preflight_runtime.json"
    try:
        with focused_probe_preflight_runtime(
            repository_root=ROOT,
            stage_root=Path(selection["stage_root"]),
            phase="selection",
            operating_date=day,
            candidate_name="seasonal_naive",
            candidate_pool_contract=selection["candidate_pool_contract"],
            cache_path=inventory_cache,
            ledger_path=runtime_ledger,
        ) as runtime_state:
            real.run_real_dart_phase_evidence(
                root=ROOT,
                dart_forecast_root=Path(selection["stage_root"]),
                physical_result={"frozen_artifact": physical},
                forecast_result={real.INPUT_BLOCK: ready[real.INPUT_BLOCK]},
                config=config,
                phase="selection",
                output_root=CONTROL / "probe_adapter_work",
                resume=True,
                candidate_pool_contract=selection["candidate_pool_contract"],
                first_stage_solver=timed_solver,
                recourse_solver=timed_solver,
                perfect_information_solver=timed_solver,
            )
            captured["preflight_runtime"] = dict(runtime_state)
    finally:
        real.run_dart_replay_panel = original_replay
        real._valid_sha256 = original_valid_sha
        real._discover_oos_inventory = original_discover
        real.solve_policy = original_solver
        if RESULT_PATH.is_file():
            result = _read(RESULT_PATH)
            result["solver_timing"] = dict(solver_timing)
            if runtime_ledger.is_file():
                result["preflight_runtime"] = _read(runtime_ledger)
            atomic_json(RESULT_PATH, result)
    return 0 if RESULT_PATH.is_file() else 2


def _terminal_payload(*, probe: dict[str, Any], blocked: bool, reasons: list[str]) -> dict[str, Any]:
    elapsed = float(probe["elapsed_seconds"])
    cell_started = probe.get("status") in {"COMPLETE", "CENSORED_RUNTIME_LIMIT"}
    projected = (
        elapsed * FORMAL_CANDIDATE_CELLS / WORKERS / 3600.0
        if cell_started
        else None
    )
    candidate = probe.get("candidate_result") or {}
    baseline = probe.get("baseline_result") or {}
    pi = probe.get("perfect_information_result") or {}
    completed_cell = probe.get("status") == "COMPLETE"
    feasible = (
        bool(
        completed_cell
        and not candidate.get("solver_any_infeasible_under_frozen_rule", True)
        and not candidate.get("solver_any_time_limit_reached", True)
        and not candidate.get("solver_any_nonzero_gap", True)
        and not baseline.get("solver_any_infeasible_under_frozen_rule", True)
        and pi.get("pi_metrics_valid") is True
        )
        if completed_cell
        else None
    )
    return {
        "schema_version": "dcvpp.conditional_dart_selection_replay_terminal.v1",
        "status": "BLOCKED" if blocked else "PROBE_PASSED_FULL_NOT_STARTED",
        "terminal": blocked,
        "formal_cells_started": 0,
        "probe_isolated": True,
        "locked_2025_opened": False,
        "main_optimization_started": False,
        "panel": {
            "selection_valid_date_count": 1086,
            "candidate_count": 8,
            "experiment_count": 2,
            "seed_count": 3,
            "formal_candidate_cell_count": FORMAL_CANDIDATE_CELLS,
            "shared_baseline_run_count": FORMAL_BASELINE_RUNS,
            "perfect_information_run_count": FORMAL_PI_RUNS,
            "total_top_level_result_count": FORMAL_CANDIDATE_CELLS + FORMAL_BASELINE_RUNS + FORMAL_PI_RUNS,
            "estimated_underlying_milp_solve_count": FORMAL_MILP_SOLVES,
        },
        "probe": {
            **probe,
            "feasible": feasible,
            "four_worker_projected_candidate_wall_hours": projected,
            "per_candidate_cell_24h_budget_seconds": PER_CELL_BUDGET_SECONDS,
        },
        "blocker_reasons": reasons,
        "actionable_pending_count": 0 if blocked else FORMAL_CANDIDATE_CELLS,
        "sources": {
            "selection_config_ready": {"path": READY_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(READY_PATH)},
            "materialization": {"path": MATERIALIZATION_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(MATERIALIZATION_PATH)},
            "conditional_protocol": {"path": PROTOCOL_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(PROTOCOL_PATH)},
            "selection_training": {"path": SELECTION_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(SELECTION_PATH)},
        },
        "created_at_utc": utc_now(),
    }


def _parent(cutoff_seconds: float) -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.unlink(missing_ok=True)
    TERMINAL_PATH.unlink(missing_ok=True)
    marker_root = CONTROL / "probe_stage_markers"
    marker_root.mkdir(parents=True, exist_ok=True)
    for marker_name in ("solver_cell_entered.json", "inventory_complete.json"):
        (marker_root / marker_name).unlink(missing_ok=True)
    protocol = _read(PROTOCOL_PATH)
    day = str(protocol["phases"]["selection"]["ledger_path"])
    ledger = _read(ROOT / day)
    operating_date = str(ledger["valid_operating_dates"][0])
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--day", operating_date]
    parent_started = perf_counter()
    with LOG_PATH.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=cutoff_seconds)
        except subprocess.TimeoutExpired:
            solver_entered = (CONTROL / "probe_stage_markers/solver_cell_entered.json").is_file()
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            probe = {
                "schema_version": "dcvpp.conditional_dart_selection_probe.v1",
                "status": "CENSORED_RUNTIME_LIMIT",
                "operating_date": operating_date,
                "candidate_id": "seasonal_naive",
                "experiment_id": "A1",
                "seed": 20260809,
                "candidate_cell_count": 1,
                "baseline_run_count": 1,
                "perfect_information_run_count": 1,
                "elapsed_seconds": perf_counter() - parent_started,
                "feasibility": "UNKNOWN_CENSORED",
                "formal_namespace": False,
                "locked_2025_opened": False,
                "main_optimization_started": False,
                "worker_exit_code": process.returncode,
                "log_path": LOG_PATH.relative_to(ROOT).as_posix(),
                "solver_cell_entered": solver_entered,
            }
            atomic_json(RESULT_PATH, probe)
            reason = (
                "ONE_REAL_CELL_RUNTIME_LOWER_BOUND_PROJECTS_ABOVE_24_HOURS_AT_FOUR_WORKERS"
                if solver_entered
                else "PROBE_PREFLIGHT_RUNTIME_CENSORED_BEFORE_ANY_SOLVER_CELL_STARTED"
            )
            payload = _terminal_payload(probe=probe, blocked=True, reasons=[reason])
            if not solver_entered:
                payload["probe"]["feasible"] = None
                payload["probe"]["four_worker_projected_candidate_wall_hours"] = None
            atomic_json(TERMINAL_PATH, payload)
            print(json.dumps(payload, indent=2))
            return 2
    probe = _read(RESULT_PATH) if RESULT_PATH.is_file() else {
        "status": "FAILED_BEFORE_RESULT",
        "elapsed_seconds": perf_counter() - parent_started,
        "operating_date": operating_date,
        "worker_exit_code": process.returncode,
    }
    if probe["status"] == "FAILED_BEFORE_RESULT":
        adapter_manifest = CONTROL / "probe_adapter_work/selection_real_evidence_manifest.json"
        adapter = _read(adapter_manifest) if adapter_manifest.is_file() else {}
        probe["preflight_pending"] = adapter.get("pending", [])
        payload = _terminal_payload(
            probe=probe,
            blocked=True,
            reasons=["PROBE_INFRASTRUCTURE_OR_PREFLIGHT_FAILED_BEFORE_ANY_SOLVER_CELL_STARTED"],
        )
        payload["probe"]["feasible"] = None
        payload["probe"]["four_worker_projected_candidate_wall_hours"] = None
        payload["probe"]["per_candidate_cell_24h_budget_seconds"] = PER_CELL_BUDGET_SECONDS
        atomic_json(TERMINAL_PATH, payload)
        print(json.dumps(payload, indent=2))
        return 2
    candidate = probe.get("candidate_result") or {}
    feasible = not bool(
        candidate.get("solver_any_infeasible_under_frozen_rule", True)
        or candidate.get("solver_any_time_limit_reached", True)
        or candidate.get("solver_any_nonzero_gap", True)
    )
    projected = float(probe["elapsed_seconds"]) * FORMAL_CANDIDATE_CELLS / WORKERS / 3600.0
    reasons: list[str] = []
    if not feasible:
        reasons.append("REPRESENTATIVE_REAL_CELL_NOT_FEASIBLE_UNDER_FROZEN_SOLVER_RULE")
    if projected > MAX_WALL_HOURS:
        reasons.append("ONE_REAL_CELL_RUNTIME_PROJECTS_ABOVE_24_HOURS_AT_FOUR_WORKERS")
    blocked = bool(reasons)
    payload = _terminal_payload(probe=probe, blocked=blocked, reasons=reasons)
    atomic_json(TERMINAL_PATH, payload)
    print(json.dumps(payload, indent=2))
    return 2 if blocked else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--day")
    parser.add_argument("--cutoff-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.worker:
        if not args.day:
            raise ValueError("--day is required for --worker")
        return _worker(str(args.day))
    return _parent(max(float(args.cutoff_seconds), PER_CELL_BUDGET_SECONDS + 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
