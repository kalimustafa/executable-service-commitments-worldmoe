from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from dcvpp.core import atomic_json, utc_now


PAPER_TABLE_SCHEMAS: dict[str, list[str]] = {
    "nlr_twin_validation.csv": [
        "run_id",
        "status",
        "evidence_role",
        "source_relationship",
        "measurement_relationship",
        "blocker_code",
        "message",
    ],
    "nyiso_forecast_main.csv": [
        "run_id",
        "task",
        "status",
        "evidence_role",
        "source_relationship",
        "measurement_relationship",
        "blocker_code",
        "message",
    ],
    "workload_forecast_main.csv": [
        "run_id",
        "status",
        "evidence_role",
        "source_relationship",
        "measurement_relationship",
        "blocker_code",
        "message",
    ],
    "counterfactual_optimization_main.csv": [
        "run_id",
        "policy",
        "status",
        "evidence_role",
        "source_relationship",
        "measurement_relationship",
        "counterfactual",
        "blocker_code",
        "message",
    ],
}


def _atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """Write a small CSV atomically without creating a second audit artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_blocked_paper_contract(
    root: Path,
    *,
    run_id: str,
    blockers: list[dict[str, Any]],
) -> list[Path]:
    """Materialize truthful no-result schemas when an upstream evidence gate blocks paper work.

    These records are not experimental results.  They make the absence of paper
    estimates machine-readable and prevent downstream tooling from accidentally
    falling back to demo tables.
    """
    if not blockers:
        raise ValueError("blocked paper contract requires at least one blocker")
    blocker_code = str(blockers[0].get("code", "PAPER_BLOCKED"))
    message = str(blockers[0].get("message", "paper execution was blocked"))
    table_root = root / "artifacts/tables"
    outputs: list[Path] = []
    for name, columns in PAPER_TABLE_SCHEMAS.items():
        row = {
            "run_id": run_id,
            "task": "not_run",
            "policy": "not_run",
            "status": "blocked_before_training",
            "evidence_role": "observed" if name == "nlr_twin_validation.csv" else "counterfactual",
            "source_relationship": (
                "counterfactual_using_third_party_referenced_public_data"
                if name == "counterfactual_optimization_main.csv"
                else "third_party_referenced_public_data"
            ),
            "measurement_relationship": "not_measured_by_this_study",
            "counterfactual": True,
            "blocker_code": blocker_code,
            "message": message,
        }
        path = table_root / name
        _atomic_csv(path, columns, [row])
        outputs.append(path)

    failures = table_root / "failures.csv"
    _atomic_csv(
        failures,
        ["stage", "work_unit", "status", "reason"],
        [
            {
                "stage": "paper_evidence_gate",
                "work_unit": item.get("code", "PAPER_BLOCKED"),
                "status": "blocked",
                "reason": item.get("message", ""),
            }
            for item in blockers
        ],
    )
    outputs.append(failures)

    calibration = root / "artifacts/manifests/physical_calibration_manifest.json"
    atomic_json(
        calibration,
        {
            "run_id": run_id,
            "generated_at": utc_now(),
            "profile": "paper",
            "status": "not_fitted_upstream_gate_blocked",
            "frozen": False,
            "parameters": None,
            "locked_test_metrics": None,
            "blockers": blockers,
            "synthetic_fallback": False,
        },
    )
    outputs.append(calibration)

    transfer = root / "artifacts/manifests/digital_twin_transfer_manifest.json"
    atomic_json(
        transfer,
        {
            "run_id": run_id,
            "generated_at": utc_now(),
            "profile": "paper",
            "status": "not_instantiated_upstream_gate_blocked",
            "observed_facility": False,
            "counterfactual": True,
            "calibration_id": None,
            "twins": [],
            "blockers": blockers,
            "synthetic_fallback": False,
        },
    )
    outputs.append(transfer)
    return outputs


def write_ready_paper_contract(
    root: Path,
    *,
    run_id: str,
    timezone_gate: dict[str, Any],
) -> list[Path]:
    """Replace stale blocked outputs with explicit, non-result pending records."""
    if timezone_gate.get("status") != "passed":
        raise ValueError("ready paper contract requires a passed timezone gate")
    source_clock = str(
        timezone_gate.get("source_clock_assumption", timezone_gate.get("resolved_timezone"))
    )
    analysis_timezone = str(
        timezone_gate.get("analysis_timezone", timezone_gate.get("resolved_timezone"))
    )
    message = (
        f"preflight passed using assumed source clock {source_clock} and local analysis zone "
        f"{analysis_timezone}; "
        "real H0-H6 experiment has not run"
    )
    table_root = root / "artifacts/tables"
    outputs: list[Path] = []
    for name, columns in PAPER_TABLE_SCHEMAS.items():
        row = {
            "run_id": run_id,
            "task": "pending",
            "policy": "pending",
            "status": "pending_real_experiment",
            "evidence_role": "observed" if name == "nlr_twin_validation.csv" else "counterfactual",
            "source_relationship": (
                "counterfactual_using_third_party_referenced_public_data"
                if name == "counterfactual_optimization_main.csv"
                else "third_party_referenced_public_data"
            ),
            "measurement_relationship": "not_measured_by_this_study",
            "counterfactual": True,
            "blocker_code": "",
            "message": message,
        }
        path = table_root / name
        _atomic_csv(path, columns, [row])
        outputs.append(path)

    failures = table_root / "failures.csv"
    _atomic_csv(failures, ["stage", "work_unit", "status", "reason"], [])
    outputs.append(failures)

    calibration = root / "artifacts/manifests/physical_calibration_manifest.json"
    atomic_json(
        calibration,
        {
            "run_id": run_id,
            "generated_at": utc_now(),
            "profile": "paper",
            "status": "not_fitted_preflight_ready",
            "frozen": False,
            "parameters": None,
            "locked_test_metrics": None,
            "timezone_gate": timezone_gate,
            "paper_results_available": False,
            "synthetic_fallback": False,
        },
    )
    outputs.append(calibration)

    transfer = root / "artifacts/manifests/digital_twin_transfer_manifest.json"
    atomic_json(
        transfer,
        {
            "run_id": run_id,
            "generated_at": utc_now(),
            "profile": "paper",
            "status": "not_instantiated_calibration_pending",
            "observed_facility": False,
            "counterfactual": True,
            "calibration_id": None,
            "twins": [],
            "timezone_gate": timezone_gate,
            "paper_results_available": False,
            "synthetic_fallback": False,
        },
    )
    outputs.append(transfer)
    return outputs
