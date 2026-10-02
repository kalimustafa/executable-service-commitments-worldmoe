from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter
import traceback

from dcvpp.research_branch.contracts import FrozenPhysicalArtifact, ScenarioBundle
from dcvpp.research_branch.optimization import solve_policy


ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = (
    ROOT
    / "artifacts"
    / "optimization_branch"
    / "isolated-b6-optimal-probe-v1"
)
STATUS_PATH = PROBE_ROOT / "status.json"
RESULT_PATH = PROBE_ROOT / "result.json"
SCENARIO_PATH = (
    ROOT
    / "artifacts"
    / "optimization_branch"
    / "throughput-probe-normalized-reassignment-v1"
    / "scenarios"
    / "2025-01-01"
    / "seed=20260809__regime=independent__sensitivity=20__source=2bb4fca452ab7bee.npz"
)
PHYSICAL_PATH = (
    ROOT
    / "artifacts"
    / "research"
    / "physical"
    / "run_id=paper-physical-v1"
    / "frozen_physical_artifact.json"
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    started_at = timestamp()
    settings = {
        "solver": "appsi_highs",
        "time_limit_seconds": 315_360_000.0,
        "mip_relative_gap": 0.0,
        "solver_threads": 8,
        "interval_hours": 0.25,
        "cvar_alpha": 0.95,
        "risk_weight": 0.50,
        "flexible_share": 0.30,
        "migration_eligible": 1.0,
        "migration_bandwidth_base_work_units_per_interval": 1.0,
        "migration_bandwidth_multiplier": 1.0,
        "transfer_energy_mwh_per_work_unit": 0.00002,
        "checkpoint_overhead_fraction": 0.02,
        "migration_energy_price_usd_per_mwh": 100.0,
        "pue_slope_multiplier": 1.0,
        "rt_sell_back_haircut": 0.0,
        "checkpoint_granularity_work_units": 1.0,
        "tee": True,
    }
    write_json(
        STATUS_PATH,
        {
            "schema_version": "isolated_b6_optimal_probe_status_v1",
            "status": "RUNNING",
            "pid": os.getpid(),
            "started_at_utc": started_at,
            "operating_date": "2025-01-01",
            "policy_id": "B6",
            "scenario_count": 20,
            "scenario_path": str(SCENARIO_PATH),
            "physical_path": str(PHYSICAL_PATH),
            "settings": settings,
            "formal_evidence": False,
            "a1_a2_started": False,
        },
    )
    started = perf_counter()
    try:
        bundle = ScenarioBundle.load(SCENARIO_PATH)
        physical_payload = json.loads(PHYSICAL_PATH.read_text(encoding="utf-8"))
        physical = FrozenPhysicalArtifact.from_mapping(
            {"frozen_artifact": physical_payload}
        )
        result = solve_policy(bundle, physical, "B6", settings)
        wall_seconds = perf_counter() - started
        result["probe_wall_time_seconds"] = wall_seconds
        result["probe_formal_evidence"] = False
        result["probe_a1_a2_started"] = False
        write_json(RESULT_PATH, result)
        write_json(
            STATUS_PATH,
            {
                "schema_version": "isolated_b6_optimal_probe_status_v1",
                "status": "COMPLETED",
                "pid": os.getpid(),
                "started_at_utc": started_at,
                "finished_at_utc": timestamp(),
                "operating_date": "2025-01-01",
                "policy_id": "B6",
                "scenario_count": 20,
                "solver_outcome": result.get("solver_outcome"),
                "termination_condition": result.get("termination_condition"),
                "has_incumbent": result.get("has_incumbent"),
                "objective_value": result.get("objective_value"),
                "objective_lower_bound": result.get("objective_lower_bound"),
                "objective_upper_bound": result.get("objective_upper_bound"),
                "relative_mip_gap": result.get("relative_mip_gap"),
                "wall_time_seconds": wall_seconds,
                "result_path": str(RESULT_PATH),
                "formal_evidence": False,
                "a1_a2_started": False,
            },
        )
    except BaseException as exc:
        write_json(
            STATUS_PATH,
            {
                "schema_version": "isolated_b6_optimal_probe_status_v1",
                "status": "FAILED",
                "pid": os.getpid(),
                "started_at_utc": started_at,
                "finished_at_utc": timestamp(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "formal_evidence": False,
                "a1_a2_started": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
