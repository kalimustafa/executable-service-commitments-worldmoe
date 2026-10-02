"""Run only the formal full-date DART main/control forecast selection stage.

This launcher intentionally stops before the frozen-selection, locked-2025,
replay, optimization, ablation, and ORACLE stages. Re-running the same command
resumes the hash-gated readiness and forecast-selection checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dcvpp.core import load_yaml, sha256_file
from dcvpp.experiments import dart_checkpoint_training as frozen_checkpoint_training
from dcvpp.experiments.dart_checkpoint_replay_compat import (
    build_v3_anchor_snapshot,
)
from dcvpp.experiments.dart_pipeline import run_v1_dart_readiness_stage
from dcvpp.experiments.dart_training import run_dart_forecast_training_stage



def _configured_run_ids(root: Path) -> tuple[str, str]:
    paper = load_yaml(root / "configs/paper.yaml")
    execution = paper["paper_execution"]
    return (
        str(execution["dart_v1_readiness_run_id"]),
        str(execution["dart_forecast_run_id"]),
    )


def _ensure_v3_anchor_snapshot(root: Path, dart_config: Path) -> Path:
    """Build the immutable v3 positive-anchor inventory before v8 readiness."""

    policy = load_yaml(dart_config)
    amendment = policy["pre_dart_v1_readiness"][
        "anchor_only_checkpoint_amendment"
    ]
    snapshot = root / str(amendment["source_snapshot_manifest_path"])
    selection_manifest = root / str(
        amendment["source_selection_stage_manifest_path"]
    )
    build_v3_anchor_snapshot(
        frozen_checkpoint_training,
        repository_root=root,
        source_stage_root=selection_manifest.parent,
        destination_manifest_path=snapshot,
        source_readiness_manifest_path=root
        / str(amendment["prior_readiness_manifest_path"]),
        source_policy_freeze_path=root / str(amendment["prior_freeze_path"]),
        expected_source_hashes={
            "selection_stage_manifest": str(
                amendment["source_selection_stage_manifest_sha256"]
            ),
            "scheduler_state": str(amendment["source_scheduler_state_sha256"]),
            "source_training_module": str(
                amendment["source_training_module_sha256"]
            ),
            "source_readiness_manifest": str(
                amendment["prior_readiness_manifest_sha256"]
            ),
            "source_policy_freeze": str(amendment["prior_freeze_sha256"]),
            "source_config": str(amendment["source_config_sha256"]),
        },
        expected_counts={
            "track_count": 108,
            "record_count": int(amendment["source_anchor_record_count"]),
            "total_bytes": int(amendment["source_anchor_total_bytes"]),
        },
    )
    return snapshot


def main() -> int:
    root = REPOSITORY_ROOT
    v1_manifest = (
        root
        / "artifacts/real_forecasting_recovery"
        / "run_id=paper-forecast-v1-recovery-v1"
        / "run_manifest.json"
    )
    dart_config = root / "configs" / "dart_value_select.yaml"
    readiness_run_id, selection_run_id = _configured_run_ids(root)
    dart_policy = load_yaml(dart_config)
    anchor_amendment = dart_policy["pre_dart_v1_readiness"][
        "anchor_only_checkpoint_amendment"
    ]
    if selection_run_id != anchor_amendment["destination_run_id"]:
        raise RuntimeError(
            "paper.yaml DART forecast run id differs from the v12 overlay destination"
        )
    _ensure_v3_anchor_snapshot(root, dart_config)

    readiness = run_v1_dart_readiness_stage(
        root,
        v1_manifest_path=v1_manifest,
        dart_config_path=dart_config,
        run_id=readiness_run_id,
        resume=True,
    )
    audit = readiness.get("readiness_audit", {})
    if not (
        readiness.get("status") == "COMPLETE"
        and readiness.get("downstream_ready") is True
        and audit.get("passed") is True
    ):
        print(json.dumps({"stage": "readiness", "result": readiness}, indent=2))
        return 2

    selection = run_dart_forecast_training_stage(
        root,
        root / "artifacts",
        dart_config,
        selection_run_id,
        phase="selection",
        resume=True,
        input_fingerprints={
            "immutable_v1_manifest": sha256_file(v1_manifest),
            "v1_readiness_manifest": str(audit["manifest_file_sha256"]),
            "v1_testing_selection_policy": str(audit["frozen_policy_sha256"]),
        },
    )
    print(json.dumps({"stage": "selection", "result": selection}, indent=2))
    return 0 if selection.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
