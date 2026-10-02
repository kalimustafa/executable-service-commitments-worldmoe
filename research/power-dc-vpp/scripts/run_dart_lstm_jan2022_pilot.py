"""Launch/resume the formal scoped January-2022 LSTM A1/A2 pilot."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcvpp.core import atomic_json, load_yaml, sha256_file, utc_now  # noqa: E402
from dcvpp.experiments import dart_real_evidence as real  # noqa: E402
from dcvpp.experiments.dart_conditional_settlement_protocol import (  # noqa: E402
    conditional_settlement_runtime,
)
from dcvpp.experiments.dart_formal_preflight_runtime import (  # noqa: E402
    formal_preflight_runtime,
    load_formal_preflight,
)
from dcvpp.experiments.dart_replay_candidate_shard_runtime import (  # noqa: E402
    candidate_shard_dart_replay_runtime,
)
from dcvpp.experiments.dart_replay_equivalent_reuse_runtime import (  # noqa: E402
    equivalent_baseline_reuse_runtime,
)
from dcvpp.experiments.dart_selection_only import (  # noqa: E402
    validate_bounded_gap_amendment,
)


SCOPE_ID = "selection_lstm_jan2022_v1"
CANDIDATES = ("lstm",)
DATES = tuple(
    (date(2022, 1, 1) + timedelta(days=index)).isoformat() for index in range(31)
)
SEEDS = (20260809, 20260819, 20260829)
EXPECTED_CELLS = 31 * 1 * 2 * 3
CONTROL = (
    ROOT
    / "artifacts/dart_value_select/run_control/conditional_selection_replay_v1"
    / "pilot_lstm_jan2022"
)
OUTPUT = (
    ROOT
    / "artifacts/dart_value_select/run_id=paper-dart-value-select-v4/replay"
    / "run_id=paper-dart-replay-v4/selection_pilots/lstm_jan2022_v1"
)
PREFLIGHT = (
    ROOT
    / "artifacts/dart_value_select/run_control/conditional_selection_replay_v1"
    / "formal_preflight/pilot_lstm_jan2022/selection_formal_preflight_manifest.json"
)
SELECTION = (
    ROOT
    / "artifacts/dart_value_select/run_id=paper-dart-forecast-v4"
    / "forecast_oos/selection_stage_manifest.json"
)
READY = (
    ROOT
    / "artifacts/dart_value_select/counterfactual_inputs/v1"
    / "selection_config_ready_input_block.json"
)
PHYSICAL = (
    ROOT
    / "artifacts/research/physical/run_id=paper-physical-v1"
    / "frozen_physical_artifact.json"
)
PROTOCOL = (
    ROOT
    / "artifacts/dart_value_select/protocol_amendments/conditional_settlement_v1"
    / "conditional_settlement_protocol_manifest.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not a mapping: {path}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    runtime = config.get("optimization_runtime", {})
    feasibility = config.get("feasibility", {})
    expected = {
        "time_limit_seconds": 600,
        "mip_relative_gap": 0.0,
        "solver_threads": 1,
    }
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"latest config requires optimization_runtime.{key}={value}")
    if (
        feasibility.get("maximum_solver_relative_mip_gap") != 0.05
        or feasibility.get("require_solver_incumbent") is not True
        or feasibility.get("require_finite_reported_solver_gap") is not True
        or feasibility.get("feasible_timeout_policy")
        != "accepted_only_with_incumbent_and_finite_relative_mip_gap_le_0_05"
    ):
        raise RuntimeError("latest bounded-gap admission contract changed")


def _resource_snapshot() -> dict[str, Any]:
    path = CONTROL / "device_monitor_latest.json"
    if not path.is_file():
        return {}
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "peak_system_ram_gb": value.get("peak_system_ram_gb"),
        "peak_process_tree_rss_gb": value.get("peak_process_tree_rss_gb"),
        "system_cpu_percent": value.get("system_cpu_percent"),
    }


def _dry_run(
    config: dict[str, Any],
    manifest: dict[str, Any],
    bounded_gap_pins: dict[str, Any],
) -> dict[str, Any]:
    contract = manifest["contract"]
    result = {
        "schema_version": "dart_lstm_jan2022_pilot_dry_run_v1",
        "status": "DRY_RUN_COMPLETE_NO_SOLVER",
        "phase": "selection",
        "scope_id": SCOPE_ID,
        "candidate_ids": list(CANDIDATES),
        "operating_dates": list(DATES),
        "operating_date_count": len(DATES),
        "experiment_ids": ["A1", "A2"],
        "seeds": list(SEEDS),
        "candidate_cell_count": EXPECTED_CELLS,
        "strict_common_baseline_count": len(DATES) * len(SEEDS),
        "perfect_information_count": len(DATES),
        "candidate_date_unit_count": len(DATES) * len(CANDIDATES),
        "requested_workers": 8,
        "initial_workers": 4,
        "solver_threads_per_process": 1,
        "time_limit_seconds_per_milp": config["optimization_runtime"]["time_limit_seconds"],
        "target_mip_relative_gap": config["optimization_runtime"]["mip_relative_gap"],
        "timeout_admission_max_gap": config["feasibility"]["maximum_solver_relative_mip_gap"],
        "timeout_requires_incumbent": True,
        "timeout_requires_finite_gap": True,
        "formal_preflight_manifest_sha256": sha256_file(PREFLIGHT),
        "formal_preflight_manifest_fingerprint": manifest["manifest_fingerprint"],
        "formal_preflight_contract_sha256": manifest["contract_sha256"],
        "bounded_gap_protocol_pins": bounded_gap_pins,
        "scope_roles": bounded_gap_pins["scope_roles"],
        "formal_preflight_scope_exact": bool(
            contract["scope_id"] == SCOPE_ID
            and contract["candidate_ids"] == list(CANDIDATES)
            and contract.get("shared_baseline_dependency", {}).get("candidate_id")
            == "seasonal_naive"
            and contract.get("shared_baseline_dependency", {}).get("role")
            == "shared_baseline_dependency_only"
            and bounded_gap_pins["scope_roles"]["candidate_ids"]
            == list(CANDIDATES)
            and bounded_gap_pins["scope_roles"]["dependency_candidate_ids"]
            == ["seasonal_naive"]
            and int(contract["operating_date_count"]) == len(DATES)
        ),
        "resume": True,
        "selection_only": True,
        "locked_2025_opened": False,
        "gpu_accelerates_highs_milp": False,
        "solver_invocation_count": 0,
        "created_at_utc": utc_now(),
    }
    if result["candidate_cell_count"] != 186 or not result["formal_preflight_scope_exact"]:
        raise RuntimeError("pilot dry-run panel identity changed")
    atomic_json(CONTROL / "dry_run.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8, choices=(4, 6, 8))
    parser.add_argument("--initial-workers", type=int, default=4, choices=(4, 6, 8))
    args = parser.parse_args()
    if args.initial_workers > args.max_workers:
        raise ValueError("--initial-workers cannot exceed --max-workers")
    CONTROL.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = _read(SELECTION)
    ready = _read(READY)
    physical = _read(PHYSICAL)
    config = load_yaml(ROOT / "configs/dart_value_select.yaml")
    _validate_config(config)
    bounded_gap_pins = validate_bounded_gap_amendment(
        repository_root=ROOT,
        candidate_ids=CANDIDATES,
        dependency_candidate_ids=("seasonal_naive",),
    )
    config[real.INPUT_BLOCK] = dict(ready[real.INPUT_BLOCK])
    manifest, _cache = load_formal_preflight(PREFLIGHT, repository_root=ROOT)
    dry = _dry_run(config, manifest, bounded_gap_pins)
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return 0

    launch = {
        "schema_version": "dart_lstm_jan2022_pilot_launch_v1",
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_at_utc": utc_now(),
        "scope_id": SCOPE_ID,
        "candidate_cell_count": EXPECTED_CELLS,
        "requested_workers": args.max_workers,
        "initial_workers": args.initial_workers,
        "solver_threads_per_process": 1,
        "selection_only": True,
        "locked_2025_opened": False,
        "resume": True,
        "progress_path": (CONTROL / "progress.json").as_posix(),
        "output_root": OUTPUT.as_posix(),
        "formal_preflight_manifest_sha256": sha256_file(PREFLIGHT),
        "formal_preflight_manifest_fingerprint": manifest["manifest_fingerprint"],
        "formal_preflight_contract_sha256": manifest["contract_sha256"],
        "bounded_gap_protocol_pins": bounded_gap_pins,
        "scope_roles": bounded_gap_pins["scope_roles"],
    }
    atomic_json(CONTROL / "launch.json", launch)
    try:
        with ExitStack() as stack:
            # The conditional overlay remains broader, while the hash-pinned
            # formal cache below narrows it to the exact 31-day pilot scope.
            stack.enter_context(
                conditional_settlement_runtime(
                    repository_root=ROOT,
                    protocol_manifest=PROTOCOL,
                    runtime_ledger_path=CONTROL / "conditional_settlement_runtime.json",
                )
            )
            stack.enter_context(
                formal_preflight_runtime(
                    repository_root=ROOT,
                    manifest_path=PREFLIGHT,
                    runtime_ledger_path=CONTROL / "formal_preflight_runtime.json",
                    expected_scope_id=SCOPE_ID,
                    expected_candidate_ids=CANDIDATES,
                    expected_operating_dates=DATES,
                )
            )
            stack.enter_context(
                candidate_shard_dart_replay_runtime(
                    max_candidate_workers=args.max_workers,
                    initial_candidate_workers=args.initial_workers,
                    solver_threads_per_process=1,
                    progress_path=CONTROL / "progress.json",
                    ledger_path=CONTROL / "candidate_shard_runtime.json",
                    resource_snapshot=_resource_snapshot,
                )
            )
            # Keeps the exact baseline/seasonal-A2 operational hash cache in
            # force for future full-pool scopes; it is a no-op for LSTM cells.
            stack.enter_context(
                equivalent_baseline_reuse_runtime(
                    ledger_path=CONTROL / "equivalent_baseline_runtime.json"
                )
            )
            result = real.run_real_dart_phase_evidence(
                root=ROOT,
                dart_forecast_root=Path(selection["stage_root"]),
                physical_result={"frozen_artifact": physical},
                forecast_result={real.INPUT_BLOCK: ready[real.INPUT_BLOCK]},
                config=config,
                phase="selection",
                output_root=OUTPUT,
                resume=True,
                candidate_pool_contract=selection["candidate_pool_contract"],
            )
        status = str(result.get("status", "UNKNOWN"))
        launch.update(
            {
                "status": status,
                "finished_at_utc": utc_now(),
                "result": result,
            }
        )
        atomic_json(CONTROL / "launch.json", launch)
        print(json.dumps(launch, indent=2, default=str))
        return 0 if status == "COMPLETE" else 2
    except BaseException as exc:
        launch.update(
            {
                "status": "FAILED",
                "finished_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        atomic_json(CONTROL / "launch.json", launch)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
