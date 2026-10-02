from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repository_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dcvpp.core import atomic_json, load_yaml, sha256_file, utc_now  # noqa: E402
from dcvpp.research_branch.hypotheses import (  # noqa: E402
    derive_physical_hypothesis_metrics,
)
from dcvpp.research_branch.runner import run_optimization_branch  # noqa: E402
from dcvpp.research_branch.workload_unit_bridge import (  # noqa: E402
    CHECKPOINT_GRANULARITY_WORK_UNITS,
    prepare_normalized_optimization_forecasts,
)


PLANNED_COUNTS = {
    "primary": 4608,
    "ofat": 6144,
    "timezone": 576,
    "total_top_level": 11328,
}


def _physical_result(root: Path) -> dict[str, Any]:
    preflight_path = (
        root
        / "artifacts/real_forecasting_recovery/run_id=paper-forecast-v1-recovery-v1/preflight_manifest.json"
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    recovery = preflight["physical_recovery_contract"]
    if recovery.get("status") != "VERIFIED_COMPLETED":
        raise ValueError("physical recovery contract is not VERIFIED_COMPLETED")
    records = recovery["verified_output_hashes"]
    for record in records.values():
        path = root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"verified physical output mismatch: {record['path']}")
    frozen = json.loads((root / records["frozen_artifact"]["path"]).read_text(encoding="utf-8"))
    result = {
        "status": "completed",
        "frozen_artifact": frozen,
        "outputs": {name: record["path"] for name, record in records.items()},
        "output_hashes": {name: record["sha256"] for name, record in records.items()},
    }
    hypothesis_path = root / "configs/hypotheses.yaml"
    rules: dict[str, Any] = {}
    if hypothesis_path.is_file():
        rules = dict(
            load_yaml(hypothesis_path)
            .get("hypotheses", {})
            .get("H0", {})
            .get("decision_rules", {})
        )
    result["hypothesis_metrics"] = derive_physical_hypothesis_metrics(root, result, rules)
    return result


def _write_direct_manifest(
    root: Path,
    run_id: str,
    *,
    status: str,
    terminal: bool,
    reason: str | None,
    admission: dict[str, Any],
    optimization_result: dict[str, Any],
) -> Path:
    path = root / "artifacts/optimization_branch" / run_id / "direct_execution_manifest.json"
    payload = {
        "schema_version": "direct_main96_optimization_execution_v1",
        "run_id": run_id,
        "status": status,
        "terminal": terminal,
        "reason": reason,
        "forecast_extension_started_or_resumed": False,
        "dart_overlay_used": False,
        "full_year_extension": False,
        "planned_cell_counts": PLANNED_COUNTS,
        "resource_contract": {
            "process_count": 1,
            "solver": "HiGHS",
            "solver_threads_per_cell": 1,
            "parallel_date_sharding_enabled": False,
            "reason": "shared runner manifests are not concurrency-safe; no unsafe parallel claim",
        },
        "normalization_admission": admission,
        "optimization_result": optimization_result,
        "updated_at_utc": utc_now(),
    }
    atomic_json(path, payload)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="paper-optimization-v1")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    admission = prepare_normalized_optimization_forecasts(ROOT, args.run_id)
    if admission["status"] != "ADMITTED":
        manifest = _write_direct_manifest(
            ROOT,
            args.run_id,
            status="BLOCKED",
            terminal=True,
            reason=admission.get("reason"),
            admission=admission,
            optimization_result=admission["optimization_result"],
        )
        print(json.dumps({"status": "BLOCKED", "manifest": str(manifest)}))
        return 2
    if not args.execute:
        result = {
            "status": "NOT_STARTED_ADMITTED",
            "completed_count": 0,
            "failure_count": 0,
            "pending_count": PLANNED_COUNTS["total_top_level"],
            "actionable_pending_count": PLANNED_COUNTS["total_top_level"],
            "paths": {},
        }
        manifest = _write_direct_manifest(
            ROOT,
            args.run_id,
            status="READY_NOT_STARTED",
            terminal=False,
            reason=None,
            admission=admission,
            optimization_result=result,
        )
        print(json.dumps({"status": "READY_NOT_STARTED", "manifest": str(manifest)}))
        return 0

    config = load_yaml(ROOT / "configs/paper.yaml")
    config["seed_list"] = [20260809, 20260819, 20260829]
    optimization = dict(config.get("optimization", {}))
    optimization["full_year_extension"] = False
    config["optimization"] = optimization
    branch = dict(config.get("optimization_branch", {}))
    branch.update(
        {
            "full_year_extension": False,
            "run_full_year_extension": False,
            "run_ofat_sensitivities": True,
            "seed_list": [20260809, 20260819, 20260829],
            "solver_threads": 1,
            "checkpoint_granularity_work_units": CHECKPOINT_GRANULARITY_WORK_UNITS,
        }
    )
    config["optimization_branch"] = branch
    result = run_optimization_branch(
        ROOT, config, args.run_id, _physical_result(ROOT), admission["forecast_result"]
    )
    terminal = int(result.get("actionable_pending_count", 1)) == 0
    manifest = _write_direct_manifest(
        ROOT,
        args.run_id,
        status=str(result.get("status", "UNKNOWN")),
        terminal=terminal,
        reason=None,
        admission=admission,
        optimization_result=result,
    )
    print(json.dumps({"status": result.get("status"), "manifest": str(manifest)}))
    return 0 if terminal and result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

