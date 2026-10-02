from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    execution_path = root / "artifacts/manifests/paper_execution_manifest.json"
    protocol_path = root / "artifacts/manifests/three_seed_runtime_amendment.json"
    readiness_path = root / (
        "artifacts/dart_value_select/v1_readiness/"
        "run_id=paper-dart-v1-readiness-v5/v1_dart_readiness_manifest.json"
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if execution.get("branch_ids", {}).get("dart_v1_readiness") != (
        "paper-dart-v1-readiness-v1"
    ):
        raise RuntimeError("recorded readiness run-id is not the disclosed failed v1")
    if execution.get("branch_results", {}).get("dart_v1_readiness", {}).get(
        "status"
    ) != "failed":
        raise RuntimeError("recorded readiness v1 is not failed")
    for branch in ("forecast_extension", "dart"):
        status = execution.get("branch_results", {}).get(branch, {}).get("status")
        if status not in {None, "not_run_upstream_incomplete", "not_run"}:
            raise RuntimeError(f"downstream branch already started: {branch}/{status}")
    if (
        readiness.get("status") != "COMPLETE"
        or readiness.get("formal_dart_training_allowed") is not True
        or readiness.get("all_candidate_replay_seed_slots")
        != [20260809, 20260819, 20260829]
    ):
        raise RuntimeError("replacement readiness manifest is not COMPLETE three-seed evidence")
    if not protocol_path.is_file():
        raise RuntimeError("three-seed protocol amendment manifest is absent")

    payload = {
        "schema_version": "dcvpp.paper_execution_readiness_run_id_amendment/v1",
        "status": "COMPLETE",
        "old_run_id": "paper-dart-v1-readiness-v1",
        "old_status": "failed",
        "old_paper_execution_manifest_path": execution_path.relative_to(root).as_posix(),
        "old_paper_execution_manifest_sha256": sha256(execution_path),
        "new_run_id": "paper-dart-v1-readiness-v5",
        "new_readiness_manifest_path": readiness_path.relative_to(root).as_posix(),
        "new_readiness_manifest_sha256": sha256(readiness_path),
        "new_readiness_status": "COMPLETE",
        "new_formal_dart_training_allowed": True,
        "confirmatory_study_seeds": [20260809, 20260819, 20260829],
        "three_seed_protocol_amendment_path": protocol_path.relative_to(root).as_posix(),
        "three_seed_protocol_amendment_sha256": sha256(protocol_path),
        "downstream_training_started_before_amendment": False,
        "old_evidence_preserved": True,
        "reason": "failed_pre_amendment_readiness_run_replaced_by_hash_verified_three_seed_v9_readiness_before_any_downstream_training",
    }
    destination = root / "artifacts/manifests/paper_execution_readiness_run_id_amendment.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


if __name__ == "__main__":
    main()
