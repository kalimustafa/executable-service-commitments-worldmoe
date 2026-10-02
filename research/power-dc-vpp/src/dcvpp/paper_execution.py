from __future__ import annotations

import json
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from dcvpp.core import (
    PipelineState,
    atomic_json,
    atomic_text,
    load_yaml,
    sha256_file,
    sha256_json,
    utc_now,
)


PAPER_EXECUTION_RUNNING = "PAPER_EXECUTION_RUNNING"

_PHYSICAL_FINGERPRINT_CONFLICT = "RUN_ID_FINGERPRINT_CONFLICT"
_FORECAST_RECOVERY_SCHEMA = "paper_forecast_v1_recovery_manifest/v1"
_FORECAST_RECOVERY_KIND = "paper_forecast_v1_integrity_recovery"
_FORECAST_TASKS = {"nyiso_da", "nyiso_rt", "nlr_workload"}


def _dart_downstream_admissible(value: Mapping[str, Any]) -> bool:
    """Accept no generic PARTIAL state; validate the one frozen exception."""

    status = str(value.get("status", "")).strip().upper()
    if status == "COMPLETE":
        return (
            value.get("downstream_ready") is True
            and value.get("candidate_pool_contract") is None
            and value.get("terminal_status_cap") in (None, "")
        )
    if status != "PARTIAL_WITH_FAILED_CANDIDATE":
        return False
    try:
        from dcvpp.experiments.dart_partial_candidate import (
            PARTIAL_CANDIDATE_TERMINAL_CAP,
            validate_partial_candidate_pool_contract,
        )

        validate_partial_candidate_pool_contract(value.get("candidate_pool_contract"))
    except (ImportError, TypeError, ValueError):
        return False
    return (
        value.get("downstream_ready") is True
        and value.get("terminal_status_cap") == PARTIAL_CANDIDATE_TERMINAL_CAP
    )


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _resolved(root: Path, value: Any) -> Path:
    root = Path(root).resolve()
    path = Path(str(value))
    if value in (None, "") or ".." in path.parts:
        raise ValueError("evidence path is empty or contains traversal")
    if path.is_absolute():
        raise ValueError("evidence path must be repository-relative")
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence path resolves outside the repository") from exc
    return candidate


def _has_physical_fingerprint_conflict(manifest: Mapping[str, Any]) -> bool:
    if str(manifest.get("status", "")).lower() != "blocked":
        return False
    blockers = manifest.get("blockers", [])
    return any(
        isinstance(item, Mapping)
        and str(item.get("code", "")) == _PHYSICAL_FINGERPRINT_CONFLICT
        for item in blockers
    )


def _load_verified_physical_recovery_result(
    root: Path,
    run_id: str,
    *,
    conflict_manifest_path: Path,
    conflict_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover a completed physical result without mutating a later conflict.

    A code/config fingerprint conflict can overwrite the canonical branch
    manifest even though a targeted repair already froze a complete, verified
    output set.  The repair's pre-mutation snapshot retains the complete result
    and workload contract, while ``recovery.json`` records the post-repair
    hashes.  This loader composes those two immutable records and verifies every
    current output before returning it.  It never rewrites the canonical
    conflict manifest, so the failed resume attempt remains evidence.
    """

    root = Path(root).resolve()
    run_root = root / "artifacts" / "research" / "physical" / f"run_id={run_id}"
    repairs_root = run_root / "repairs"
    failures: list[str] = []
    candidates = sorted(repairs_root.glob("*/recovery.json"), reverse=True)
    for recovery_path in candidates:
        recovery = _load_json_mapping(recovery_path)
        if recovery is None:
            continue
        if str(recovery.get("status", "")).lower() != "completed":
            continue
        if str(recovery.get("run_id", "")) != run_id:
            continue

        targets = recovery.get("targets")
        target = (
            targets.get("branch_manifest")
            if isinstance(targets, Mapping)
            else None
        )
        if not isinstance(target, Mapping):
            failures.append(f"{recovery_path}: missing branch-manifest target")
            continue
        target_canonical_path = _resolved(root, target.get("canonical_path"))
        if target_canonical_path.resolve() != conflict_manifest_path.resolve():
            failures.append(f"{recovery_path}: canonical manifest target changed")
            continue
        snapshot_path = _resolved(root, target.get("backup_path"))
        expected_snapshot_sha = str(target.get("sha256", ""))
        before_hashes = recovery.get("before_hashes")
        if not snapshot_path.is_file():
            failures.append(f"{recovery_path}: completed snapshot is missing")
            continue
        actual_snapshot_sha = sha256_file(snapshot_path)
        if (
            len(expected_snapshot_sha) != 64
            or actual_snapshot_sha != expected_snapshot_sha
            or not isinstance(before_hashes, Mapping)
            or before_hashes.get("branch_manifest") != actual_snapshot_sha
        ):
            failures.append(f"{recovery_path}: completed snapshot hash mismatch")
            continue
        completed_sha = str(recovery.get("branch_manifest_sha256_after", ""))
        if len(completed_sha) != 64:
            failures.append(f"{recovery_path}: missing post-repair manifest hash")
            continue

        snapshot = _load_json_mapping(snapshot_path)
        if snapshot is None or str(snapshot.get("status", "")).lower() != "completed":
            failures.append(f"{recovery_path}: snapshot is not completed")
            continue
        result_value = snapshot.get("result")
        if not isinstance(result_value, Mapping):
            failures.append(f"{recovery_path}: snapshot has no completed result")
            continue
        result = dict(result_value)
        if (
            str(result.get("status", "")).lower() != "completed"
            or str(result.get("run_id", "")) != run_id
        ):
            failures.append(f"{recovery_path}: completed result identity mismatch")
            continue

        snapshot_contract = snapshot.get("workload_events_contract")
        result_contract = result.get("workload_events_contract")
        if (
            not isinstance(snapshot_contract, Mapping)
            or not isinstance(result_contract, Mapping)
            or dict(snapshot_contract) != dict(result_contract)
        ):
            failures.append(f"{recovery_path}: workload contract mismatch")
            continue
        workload_path = _resolved(root, snapshot_contract.get("relative_path"))
        workload_sha = str(snapshot_contract.get("sha256", ""))
        if (
            not workload_path.is_file()
            or len(workload_sha) != 64
            or sha256_file(workload_path) != workload_sha
            or recovery.get("expected_workload_sha256") != workload_sha
            or recovery.get("protected_workload_sha256_before") != workload_sha
            or recovery.get("protected_workload_sha256_after") != workload_sha
            or recovery.get("protected_workload_modified") is not False
        ):
            failures.append(f"{recovery_path}: protected workload does not verify")
            continue

        outputs = result.get("outputs")
        output_hashes = result.get("output_hashes")
        after_hashes = recovery.get("after_hashes")
        if (
            not isinstance(outputs, Mapping)
            or not outputs
            or not isinstance(output_hashes, Mapping)
            or not isinstance(after_hashes, Mapping)
        ):
            failures.append(f"{recovery_path}: output ledger is incomplete")
            continue
        composed_hashes = dict(output_hashes)
        for name, digest in after_hashes.items():
            if name == "branch_manifest":
                continue
            if name not in outputs or not isinstance(digest, str) or len(digest) != 64:
                failures.append(f"{recovery_path}: invalid repaired output {name}")
                break
            composed_hashes[str(name)] = digest
        else:
            output_failures: list[str] = []
            for name, value in outputs.items():
                output_path = _resolved(root, value)
                expected = composed_hashes.get(name)
                if (
                    not output_path.is_file()
                    or not isinstance(expected, str)
                    or len(expected) != 64
                    or sha256_file(output_path) != expected
                ):
                    output_failures.append(str(name))
            if output_failures:
                failures.append(
                    f"{recovery_path}: current output hash mismatch: "
                    + ", ".join(output_failures)
                )
                continue

            conflict_sha_before = sha256_file(conflict_manifest_path)
            conflict_evidence = {
                "path": str(conflict_manifest_path.relative_to(root)),
                "sha256": conflict_sha_before,
                "status": conflict_manifest.get("status"),
                "request_fingerprint": conflict_manifest.get("request_fingerprint"),
                "blockers": conflict_manifest.get("blockers", []),
                "preserved_unchanged": True,
            }
            # Recheck after all potentially slow output hashing.  A concurrent
            # writer invalidates the recovery rather than producing a mixed view.
            if sha256_file(conflict_manifest_path) != conflict_sha_before:
                raise RuntimeError(
                    "physical conflict manifest changed during recovery validation"
                )
            result.update(
                {
                    "status": "completed",
                    "resumed": True,
                    "resume_reason": "verified_frozen_targeted_repair_recovery",
                    "manifest_path": str(snapshot_path.relative_to(root)),
                    "output_hashes": composed_hashes,
                    "workload_events_contract": dict(snapshot_contract),
                    "recovery_record": dict(recovery),
                    "physical_recovery_manifest": str(
                        recovery_path.relative_to(root)
                    ),
                    "physical_recovery_manifest_sha256": sha256_file(recovery_path),
                    "completed_branch_manifest_sha256": completed_sha,
                    "completed_branch_manifest_hash_attestation": {
                        "sha256": completed_sha,
                        "recorded_by": str(recovery_path.relative_to(root)),
                        "completed_manifest_bytes_preserved": False,
                        "before_snapshot_sha256_verified": actual_snapshot_sha,
                        "post_repair_output_ledger_verified": True,
                    },
                    "canonical_manifest_is_failed_attempt": True,
                    "failed_resume_attempt_evidence": conflict_evidence,
                    "recovered_from_frozen_evidence": True,
                }
            )
            return result

    detail = "; ".join(failures[-3:]) if failures else "no completed repair found"
    raise RuntimeError(
        "canonical physical manifest records RUN_ID_FINGERPRINT_CONFLICT, but "
        f"no frozen completed recovery verified: {detail}"
    )


def _load_verified_completed_physical_result(
    root: Path,
    run_id: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a frozen completed physical result without invoking live code."""

    failures: list[str] = []
    result_value = manifest.get("result")
    if not isinstance(result_value, Mapping):
        failures.append("completed manifest has no result")
        result: dict[str, Any] = {}
    else:
        result = dict(result_value)
    if str(result.get("status", "")).lower() != "completed":
        failures.append("nested result is not completed")
    if str(result.get("run_id", "")) != run_id:
        failures.append("nested result run_id changed")
    outputs = manifest.get("outputs")
    hashes = manifest.get("output_hashes")
    if (
        not isinstance(outputs, Mapping)
        or not outputs
        or not isinstance(hashes, Mapping)
        or result.get("outputs") != outputs
        or result.get("output_hashes") != hashes
    ):
        failures.append("completed output ledger is inconsistent")
    else:
        for name, value in outputs.items():
            output_path = _resolved(root, value)
            expected = hashes.get(name)
            if (
                not output_path.is_file()
                or not isinstance(expected, str)
                or len(expected) != 64
                or sha256_file(output_path) != expected
            ):
                failures.append(f"completed output hash mismatch: {name}")
    contract = manifest.get("workload_events_contract")
    if (
        not isinstance(contract, Mapping)
        or result.get("workload_events_contract") != contract
    ):
        failures.append("completed workload contract is inconsistent")
    else:
        workload_path = _resolved(root, contract.get("relative_path"))
        workload_sha = contract.get("sha256")
        if (
            not workload_path.is_file()
            or not isinstance(workload_sha, str)
            or len(workload_sha) != 64
            or sha256_file(workload_path) != workload_sha
        ):
            failures.append("completed workload contract hash mismatch")
    if failures:
        raise RuntimeError(
            "existing completed physical evidence failed frozen validation: "
            + "; ".join(failures)
        )
    result.update(
        {
            "status": "completed",
            "resumed": True,
            "resume_reason": "verified_frozen_completed_manifest",
            "manifest_path": str(manifest_path.relative_to(root)),
            "physical_completed_manifest_sha256": sha256_file(manifest_path),
            "recovered_from_frozen_evidence": False,
        }
    )
    return result


def _run_or_resume_physical_branch(
    root: Path,
    config: dict[str, Any],
    run_id: str,
    runner: Callable[[Path, dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = (
        root
        / "artifacts"
        / "research"
        / "physical"
        / f"run_id={run_id}"
        / "physical_branch_manifest.json"
    )
    manifest = _load_json_mapping(manifest_path)
    if manifest is not None:
        manifest_status = str(manifest.get("status", "")).lower()
        if manifest_status == "completed":
            return _load_verified_completed_physical_result(
                root, run_id, manifest_path, manifest
            )
        if _has_physical_fingerprint_conflict(manifest):
            return _load_verified_physical_recovery_result(
                root,
                run_id,
                conflict_manifest_path=manifest_path,
                conflict_manifest=manifest,
            )
    return runner(root, config, run_id)


def _verify_recovery_file_record(
    root: Path,
    record: Any,
    label: str,
    failures: list[str],
    *,
    allowed_root: Path | None = None,
) -> Path | None:
    if not isinstance(record, Mapping):
        failures.append(f"{label} is not a path/hash record")
        return None
    raw_path = Path(str(record.get("path", "")))
    if raw_path.is_absolute():
        failures.append(f"{label} uses a forbidden absolute path")
        return None
    try:
        path = _resolved(root, raw_path)
    except ValueError as exc:
        failures.append(f"{label} is outside the repository: {exc}")
        return None
    if allowed_root is not None:
        try:
            path.relative_to(Path(allowed_root).resolve())
        except ValueError:
            failures.append(f"{label} is outside the recovery artifact root")
            return None
    expected = record.get("sha256")
    if (
        not path.is_file()
        or not isinstance(expected, str)
        or len(expected) != 64
        or sha256_file(path) != expected
    ):
        failures.append(f"{label} hash does not verify")
        return None
    byte_count = record.get("byte_count")
    if byte_count is not None and int(byte_count) != path.stat().st_size:
        failures.append(f"{label} byte count changed")
        return None
    return path


def _verify_forecast_recovery_manifest(
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    recovery_run_id: str,
    physical_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a COMPLETE mixed-lineage v1 forecast recovery manifest."""

    failures: list[str] = []
    if manifest.get("schema_version") != _FORECAST_RECOVERY_SCHEMA:
        failures.append("unsupported recovery schema")
    if manifest.get("run_kind") != _FORECAST_RECOVERY_KIND:
        failures.append("recovery run kind changed")
    if str(manifest.get("run_id", "")) != recovery_run_id:
        failures.append("recovery run_id changed")
    if str(manifest.get("status", "")).upper() != "COMPLETE":
        failures.append("recovery is not COMPLETE")
    if manifest.get("terminal") is not True:
        failures.append("recovery is not terminal")
    if manifest.get("formal_readiness_allowed") is not True:
        failures.append("recovery did not authorize formal readiness")
    if int(manifest.get("actionable_pending_count", -1) or 0) != 0:
        failures.append("recovery still has actionable work")
    if manifest.get("integrity_failures") not in ([], None):
        failures.append("recovery retains integrity failures")
    if manifest.get("task_failures") not in ([], None):
        failures.append("recovery retains task failures")
    if manifest.get("started_from_real_referenced_data") is not True:
        failures.append("recovery lacks real referenced-data provenance")
    if (
        manifest.get("synthetic_data_used") is not False
        or manifest.get("synthetic_fallback_used") is not False
    ):
        failures.append("recovery synthetic-data boundary changed")
    if manifest.get("measurement_relationship") != "not_measured_by_this_study":
        failures.append("recovery measurement boundary changed")
    if (
        manifest.get("source_relationship")
        != "third_party_referenced_public_data"
        or manifest.get("evidence_role")
        != "referenced_public_counterfactual_input"
    ):
        failures.append("recovery source/evidence relationship changed")
    if (
        manifest.get("locked_2025_metrics_consumed") is not False
        or manifest.get("locked_2025_metrics_materialized") is not False
        or manifest.get("locked_2025_policy_decisions_consumed") is not False
    ):
        failures.append("locked 2025 evidence entered recovery policy decisions")

    fingerprint = manifest.get("manifest_fingerprint")
    core = {
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    }
    if fingerprint != sha256_json(core):
        failures.append("recovery manifest fingerprint changed")

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        failures.append("recovery counts are absent")
        counts = {}
    if (
        int(counts.get("tasks_requested", -1)) != 3
        or int(counts.get("tasks_completed", -1)) != 3
        or int(counts.get("tasks_failed", -1)) != 0
        or int(counts.get("failed_seeds", -1)) != 0
    ):
        failures.append("recovery task/seed coverage is incomplete")

    source = manifest.get("source_original_v1")
    if not isinstance(source, Mapping):
        failures.append("original-v1 source lineage is absent")
        source = {}
    expected_source_root = (
        root / "artifacts" / "real_forecasting" / "paper-forecast-v1"
    ).resolve()
    source_root = _resolved(root, source.get("root_path", source.get("root")))
    if source_root != expected_source_root or not source_root.is_dir():
        failures.append("original-v1 source root is absent")
    original_manifest_path = _resolved(
        root, source.get("original_manifest_path", "missing-original-manifest")
    )
    if original_manifest_path != (expected_source_root / "run_manifest.json"):
        failures.append("original-v1 run-manifest path changed")
    original_manifest_sha = source.get("original_manifest_sha256_or_null")
    if original_manifest_path.is_file():
        if (
            not isinstance(original_manifest_sha, str)
            or len(original_manifest_sha) != 64
            or sha256_file(original_manifest_path) != original_manifest_sha
        ):
            failures.append("preserved original v1 run-manifest hash changed")
    elif original_manifest_sha is not None:
        failures.append("absent original v1 manifest has a non-null hash")
    leaderboards = source.get("leaderboards")
    if not isinstance(leaderboards, Mapping) or not {"nyiso_da", "nyiso_rt"}.issubset(
        leaderboards
    ):
        failures.append("immutable original leaderboard lineage is incomplete")
    else:
        for task in ("nyiso_da", "nyiso_rt"):
            record = leaderboards.get(task)
            _verify_recovery_file_record(
                root, record, f"original {task} leaderboard", failures
            )
            validation_sha = (
                record.get("validation_only_content_sha256")
                if isinstance(record, Mapping)
                else None
            )
            if not isinstance(validation_sha, str) or len(validation_sha) != 64:
                failures.append(f"original {task} validation-only hash is absent")
    frozen_sources = source.get("frozen_source_files")
    if not isinstance(frozen_sources, list) or not frozen_sources:
        failures.append("frozen v1 source-file ledger is absent")
    else:
        for index, record in enumerate(frozen_sources):
            path = _verify_recovery_file_record(
                root, record, f"frozen source file {index}", failures
            )
            if (
                path is None
                or record.get("verified") is not True
                or record.get("expected_sha256") != record.get("sha256")
            ):
                failures.append(f"frozen source file {index} is not hash-pinned")

    incident = manifest.get("incident")
    if not isinstance(incident, Mapping):
        failures.append("corruption incident lineage is absent")
        incident = {}
    ledger_path = _verify_recovery_file_record(
        root, incident.get("corruption_ledger"), "corruption ledger", failures
    )
    ledger_reference = incident.get("corruption_ledger")
    ledger_record_count = (
        int(ledger_reference.get("record_count", -1))
        if isinstance(ledger_reference, Mapping)
        else -1
    )
    if ledger_path is not None:
        ledger = _load_json_mapping(ledger_path)
        ledger_records = ledger.get("records") if isinstance(ledger, Mapping) else None
        if (
            ledger_record_count != 28
            or not isinstance(ledger, Mapping)
            or int(ledger.get("record_count", -1)) != 28
            or not isinstance(ledger_records, list)
            or len(ledger_records) != 28
        ):
            failures.append("corruption ledger is not the exact 28-file incident")
        else:
            for index, record in enumerate(ledger_records):
                if not isinstance(record, Mapping) or record.get(
                    "excluded_from_canonical_outputs"
                ) is not True:
                    failures.append(
                        f"corruption ledger record {index} is not excluded"
                    )
                    continue
                _verify_recovery_file_record(
                    root,
                    {
                        "path": record.get("preserved_snapshot_path"),
                        "sha256": record.get("preserved_snapshot_sha256"),
                    },
                    f"corruption ledger snapshot {index}",
                    failures,
                )
    if (
        not incident.get("incident_id")
        or int(incident.get("overwritten_file_count", 0) or 0) != 28
        or not incident.get("preservation_mode")
    ):
        failures.append("corruption incident is not fully disclosed")

    physical = manifest.get("physical_recovery_contract")
    if not isinstance(physical, Mapping):
        failures.append("physical recovery contract is absent")
        physical = {}
    if physical.get("status") != "VERIFIED_COMPLETED":
        failures.append("physical recovery contract is not complete")
    _verify_recovery_file_record(
        root,
        {
            "path": physical.get("path"),
            "sha256": physical.get("sha256"),
        },
        "physical recovery contract",
        failures,
    )
    expected_physical = dict(physical_result or {})
    expected_workload = expected_physical.get("workload_events_contract")
    if not isinstance(expected_workload, Mapping):
        failures.append("active physical result lacks a workload contract")
    elif (
        physical.get("workload_events_contract_sha256")
        != sha256_json(dict(expected_workload))
        or physical.get("workload_events_artifact_sha256")
        != expected_workload.get("sha256")
    ):
        failures.append("physical workload contract hashes changed")
    expected_repair_path = expected_physical.get("physical_recovery_manifest")
    expected_repair_sha = expected_physical.get("physical_recovery_manifest_sha256")
    if (
        expected_repair_path
        and (
            _resolved(root, expected_repair_path).resolve()
            != _resolved(root, physical.get("path")).resolve()
            or expected_repair_sha != physical.get("sha256")
        )
    ):
        failures.append("forecast recovery points to a different physical repair")

    lineage = manifest.get("recovery_lineage")
    if not isinstance(lineage, Mapping):
        failures.append("mixed recovery lineage is absent")
        lineage = {}
    if lineage.get("composition_mode") != (
        "hash_verified_mixed_original_and_recovered_evidence"
    ):
        failures.append("mixed-lineage composition mode changed")
    original_tasks = set(lineage.get("original_intact_tasks", []))
    recovered_tasks = set(lineage.get("recovered_tasks", []))
    if (
        original_tasks != {"nyiso_rt"}
        or recovered_tasks != {"nyiso_da", "nlr_workload"}
    ):
        failures.append("original/recovered task lineage is not an exact partition")
    if not isinstance(lineage.get("fingerprint_mismatches"), list) or not lineage.get(
        "fingerprint_mismatches"
    ):
        failures.append("known fingerprint mismatches are not disclosed")
    locked_policy = lineage.get("locked_2025_policy")
    if not isinstance(locked_policy, Mapping) or any(
        locked_policy.get(key) is not False
        for key in (
            "metrics_consumed",
            "metrics_materialized",
            "strategy_decisions_used",
        )
    ):
        failures.append("mixed-lineage locked-2025 policy boundary changed")

    ledger_by_name: dict[str, list[Mapping[str, Any]]] = {}
    canonical_recovery_root = (
        root
        / "artifacts"
        / "real_forecasting_recovery"
        / f"run_id={recovery_run_id}"
    )
    for ledger_name in (
        "recovered_outputs",
        "intact_outputs",
        "corrupt_outputs",
        "outputs",
    ):
        records = manifest.get(ledger_name)
        if not isinstance(records, list):
            failures.append(f"{ledger_name} ledger is absent")
            ledger_by_name[ledger_name] = []
            continue
        valid_records: list[Mapping[str, Any]] = []
        for index, record in enumerate(records):
            record_path = _verify_recovery_file_record(
                root,
                record,
                f"{ledger_name}[{index}]",
                failures,
                allowed_root=(
                    canonical_recovery_root
                    if ledger_name
                    in {"recovered_outputs", "intact_outputs", "outputs"}
                    else None
                ),
            )
            if not isinstance(record, Mapping):
                continue
            lineage_source = _verify_recovery_file_record(
                root,
                {
                    "path": record.get("lineage_source_path"),
                    "sha256": record.get("lineage_source_sha256"),
                },
                f"{ledger_name}[{index}] lineage source",
                failures,
            )
            if (
                record.get("task") not in _FORECAST_TASKS
                or not record.get("evidence_role")
            ):
                failures.append(f"{ledger_name}[{index}] provenance is incomplete")
            if record_path is not None and lineage_source is not None:
                valid_records.append(record)
        ledger_by_name[ledger_name] = valid_records
    output_identity = {
        (str(record.get("path")), str(record.get("sha256")))
        for record in ledger_by_name["outputs"]
    }
    composed_identity = {
        (str(record.get("path")), str(record.get("sha256")))
        for name in ("recovered_outputs", "intact_outputs")
        for record in ledger_by_name[name]
    }
    corrupt_paths = {
        str(record.get("path")) for record in ledger_by_name["corrupt_outputs"]
    }
    if not output_identity or output_identity != composed_identity:
        failures.append("composite output ledger is not exactly intact plus recovered")
    if len(output_identity) != len(ledger_by_name["outputs"]):
        failures.append("composite output ledger contains duplicate identities")
    if corrupt_paths & {path for path, _digest in output_identity}:
        failures.append("known corrupt outputs remain in the composite ledger")
    for name, count_key in (
        ("recovered_outputs", "recovered_output_count"),
        ("intact_outputs", "intact_output_count"),
        ("corrupt_outputs", "corrupt_output_count"),
    ):
        if int(counts.get(count_key, -1)) != len(ledger_by_name[name]):
            failures.append(f"{name} count differs from the frozen ledger")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != _FORECAST_TASKS:
        failures.append("recovery task manifest is not exact")
    else:
        task_output_identity: set[tuple[str, str]] = set()
        for task, task_record in tasks.items():
            if (
                not isinstance(task_record, Mapping)
                or str(task_record.get("status", "")).upper() != "COMPLETE"
                or not task_record.get("evidence_role")
                or not task_record.get("lineage")
            ):
                failures.append(f"recovery task is incomplete: {task}")
                continue
            task_outputs = task_record.get("outputs")
            if not isinstance(task_outputs, list) or not task_outputs:
                failures.append(f"recovery task output ledger is absent: {task}")
                continue
            for index, record in enumerate(task_outputs):
                if _verify_recovery_file_record(
                    root,
                    record,
                    f"tasks.{task}.outputs[{index}]",
                    failures,
                    allowed_root=canonical_recovery_root,
                ) is not None:
                    task_output_identity.add(
                        (str(record.get("path")), str(record.get("sha256")))
                    )
        if task_output_identity != output_identity:
            failures.append("task output ledgers differ from composite outputs")

    if failures:
        raise RuntimeError(
            "forecast recovery manifest failed closed validation: "
            + "; ".join(failures)
        )
    return {
        "status": "COMPLETE",
        "resumed": True,
        "run_id": recovery_run_id,
        "run_kind": _FORECAST_RECOVERY_KIND,
        "counts": dict(counts),
        "output_paths": [str(record["path"]) for record in manifest["outputs"]],
        "run_manifest": str(manifest_path.relative_to(root)),
        "run_manifest_sha256": sha256_file(manifest_path),
        "forecast_day_index": manifest.get("forecast_day_index"),
        "energy_hypothesis_metrics": manifest.get("energy_hypothesis_metrics", []),
        "workload_hypothesis_metrics": manifest.get(
            "workload_hypothesis_metrics", []
        ),
        "forecast_decision_records": manifest.get("forecast_decision_records", []),
        "source_relationship": "third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
        "formal_readiness_allowed": True,
        "forecast_recovery": True,
        "recovery_lineage": dict(lineage),
        "actionable_pending_count": 0,
    }


def _run_or_resume_forecasting_branch(
    root: Path,
    forecast_config: dict[str, Any],
    forecast_run_id: str,
    execution: Mapping[str, Any],
    runner: Callable[[Path, dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    """Use the isolated mixed-lineage recovery when the incident gate is on."""

    if execution.get("forecast_recovery_enabled") is not True:
        return runner(root, forecast_config, forecast_run_id)
    recovery_run_id = str(
        execution.get("forecast_recovery_run_id", "paper-forecast-v1-recovery-v1")
    )
    recovery_value = execution.get("forecast_recovery_manifest_path")
    recovery_path = (
        _resolved(root, recovery_value)
        if recovery_value not in (None, "")
        else root
        / "artifacts"
        / "real_forecasting_recovery"
        / f"run_id={recovery_run_id}"
        / "run_manifest.json"
    )
    canonical_recovery_path = (
        root
        / "artifacts"
        / "real_forecasting_recovery"
        / f"run_id={recovery_run_id}"
        / "run_manifest.json"
    ).resolve()
    if recovery_path.resolve() != canonical_recovery_path:
        raise ValueError(
            "forecast recovery manifest path differs from the canonical recovery run"
        )
    manifest = _load_json_mapping(recovery_path)
    if manifest is None:
        if execution.get("forecast_recovery_required") is True:
            return {
                "status": "PARTIAL",
                "run_id": recovery_run_id,
                "formal_readiness_allowed": False,
                "optimization_allowed": False,
                "forecast_recovery": True,
                "actionable_pending_count": 1,
                "reason": "versioned paper-forecast-v1 recovery manifest is absent",
                "expected_run_manifest": str(recovery_path.relative_to(root)),
            }
        return runner(root, forecast_config, forecast_run_id)
    if str(manifest.get("status", "")).upper() != "COMPLETE":
        pending = int(manifest.get("actionable_pending_count", 0) or 0)
        return {
            "status": "PARTIAL",
            "run_id": recovery_run_id,
            "formal_readiness_allowed": False,
            "optimization_allowed": False,
            "forecast_recovery": True,
            "actionable_pending_count": max(1, pending),
            "reason": "versioned paper-forecast-v1 recovery is not COMPLETE",
            "run_manifest": str(recovery_path.relative_to(root)),
            "run_manifest_sha256": sha256_file(recovery_path),
            "recovery_status": manifest.get("status"),
            "integrity_failures": manifest.get("integrity_failures", []),
            "task_failures": manifest.get("task_failures", []),
        }
    return _verify_forecast_recovery_manifest(
        root,
        recovery_path,
        manifest,
        recovery_run_id=recovery_run_id,
        physical_result=(
            forecast_config.get("physical_result")
            if isinstance(forecast_config.get("physical_result"), Mapping)
            else None
        ),
    )


def _status(value: dict[str, Any] | None) -> str:
    return str((value or {}).get("status", "missing")).lower()


def _safe_result(
    name: str,
    function: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = function()
        if not isinstance(result, dict):
            raise TypeError(f"{name} returned {type(result).__name__}, expected dict")
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "branch": name,
            "paper_results_available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_utc": utc_now(),
        }


def _forecast_config(
    config: dict[str, Any],
    physical_result: dict[str, Any],
) -> dict[str, Any]:
    forecast = dict(config.get("forecast", {}))
    branch = dict(config.get("forecast_branch", {}))
    return {
        "tasks": branch.get("tasks", ["nyiso_da", "nyiso_rt", "nlr_workload"]),
        "seeds": branch.get("seeds", config.get("seed_list", [config.get("seed", 20260809)])),
        "advanced_n_estimators": int(branch.get("advanced_n_estimators", 500)),
        "n_jobs": int(branch.get("n_jobs", min(8, 22))),
        "max_train_rows": branch.get("max_train_rows"),
        "keep_nlr_selected_jobs": bool(branch.get("keep_nlr_selected_jobs", False)),
        "nlr_batch_size": int(branch.get("nlr_batch_size", 200_000)),
        "torch_epochs": int(branch.get("torch_epochs", forecast.get("epochs", 100))),
        "torch_batch_size": int(
            branch.get("torch_batch_size", forecast.get("batch_size", 32))
        ),
        "torch_device": str(branch.get("torch_device", forecast.get("device", "auto"))),
        "forecast_draw_count": int(branch.get("forecast_draw_count", 100)),
        "resume": True,
        "physical_result": physical_result,
    }


def _write_running_state(
    root: Path,
    state: PipelineState,
    *,
    stage: str,
    branch_ids: dict[str, str],
    branch_results: dict[str, Any],
) -> None:
    now = utc_now()
    state.data["profile"] = "paper"
    state.data.pop("terminal_status", None)
    paper_state = dict(state.data.get("paper", {}))
    paper_state.update({
        "run_status": PAPER_EXECUTION_RUNNING,
        "terminal": False,
        "paper_training_started": True,
        "paper_results_available": any(
            _status(result) in {"complete", "completed", "partial"}
            for result in branch_results.values()
            if isinstance(result, dict)
        ),
        "synthetic_fallback": False,
        "branch_ids": branch_ids,
        "branch_statuses": {
            name: str(result.get("status", "missing"))
            for name, result in branch_results.items()
            if isinstance(result, dict)
        },
    })
    state.data["paper"] = paper_state
    state.data["updated_at"] = now
    atomic_json(state.path, state.data)
    atomic_text(
        state.status_path,
        "\n".join(
            [
                "# Current Stage Status",
                "",
                f"- Run ID: `{state.data['run_id']}`",
                f"- Stage: `{stage}`",
                f"- Run status: `{PAPER_EXECUTION_RUNNING}`",
                "- Terminal: `false`",
                "- Paper training started: `true`",
                "- Synthetic paper fallback: `false`",
                f"- Branch statuses: `{state.data['paper']['branch_statuses']}`",
                f"- Updated: `{now}`",
                "",
            ]
        ),
    )


def run_paper_execution(
    root: Path,
    config: dict[str, Any],
    state: PipelineState,
    preflight_result: dict[str, Any],
) -> dict[str, Any]:
    """Run the real paper branches and hand exhausted work to final reporting.

    The child run IDs are stable across top-level resumes. Branches own
    their fine-grained checkpoints; this function never replaces a failed real
    source with demo or synthetic data.
    """

    from dcvpp.experiments.real_forecasting import run_forecasting_branch
    from dcvpp.experiments.forecast_extension_bridge import (
        load_complete_forecast_evidence,
    )
    from dcvpp.experiments.paper_forecast_extensions import (
        run_referenced_paper_forecast_extensions,
    )
    from dcvpp.research import run_physical_branch
    from dcvpp.research_branch import run_optimization_branch
    from dcvpp.research_branch.hypotheses import derive_physical_hypothesis_metrics

    root = Path(root).resolve()
    execution = dict(config.get("paper_execution", {}))
    dart_required = bool(
        execution.get("dart_enabled") is True
        or execution.get("dart_config_path") not in (None, "")
    )
    parent_run_id = str(state.data["run_id"])
    branch_ids = {
        "physical": str(execution.get("physical_run_id", f"{parent_run_id}-physical-v1")),
        "forecast": str(execution.get("forecast_run_id", f"{parent_run_id}-forecast-v1")),
        "forecast_recovery": str(
            execution.get(
                "forecast_recovery_run_id", "paper-forecast-v1-recovery-v1"
            )
        ),
        "forecast_extension": str(
            execution.get(
                "forecast_extension_run_id", "paper-forecast-extensions-v1"
            )
        ),
        "dart_forecast": str(
            execution.get("dart_forecast_run_id", "paper-dart-forecast-v1")
        ),
        "dart_v1_readiness": str(
            execution.get(
                "dart_v1_readiness_run_id", "paper-dart-v1-readiness-v1"
            )
        ),
        "dart_replay": str(
            execution.get("dart_replay_run_id", "paper-dart-replay-v1")
        ),
        "dart_value_select": str(
            execution.get(
                "dart_value_select_run_id", "paper-dart-value-select-v1"
            )
        ),
        "optimization": str(
            execution.get("optimization_run_id", f"{parent_run_id}-optimization-v1")
        ),
    }
    branch_results: dict[str, Any] = {}
    execution_manifest = root / "artifacts/manifests/paper_execution_manifest.json"
    retry_rounds = max(1, int(execution.get("optimization_retry_rounds", 3)))
    optimization_config = dict(config)
    optimization_branch_config = dict(config.get("optimization_branch", {}))
    optimization_branch_config.setdefault("max_attempts_per_cell", retry_rounds)
    optimization_config["optimization_branch"] = optimization_branch_config

    _write_running_state(
        root,
        state,
        stage="paper_physical",
        branch_ids=branch_ids,
        branch_results=branch_results,
    )
    physical_result = _safe_result(
        "physical",
        lambda: _run_or_resume_physical_branch(
            root,
            config,
            branch_ids["physical"],
            run_physical_branch,
        ),
    )
    hypothesis_config_path = root / "configs" / "hypotheses.yaml"
    h0_rules: dict[str, Any] = {}
    if hypothesis_config_path.is_file():
        hypothesis_config = load_yaml(hypothesis_config_path)
        h0_config = hypothesis_config.get("hypotheses", {}).get("H0", {})
        if isinstance(h0_config, dict) and isinstance(
            h0_config.get("decision_rules"), dict
        ):
            h0_rules = dict(h0_config["decision_rules"])
    physical_result["hypothesis_metrics"] = derive_physical_hypothesis_metrics(
        root,
        physical_result,
        h0_rules,
    )
    physical_result["hypothesis_config_path"] = (
        str(hypothesis_config_path) if hypothesis_config_path.is_file() else None
    )
    branch_results["physical"] = physical_result
    atomic_json(
        execution_manifest,
        {
            "run_id": parent_run_id,
            "status": PAPER_EXECUTION_RUNNING,
            "terminal": False,
            "branch_ids": branch_ids,
            "branch_results": branch_results,
            "updated_at_utc": utc_now(),
        },
    )

    _write_running_state(
        root,
        state,
        stage="paper_forecasting",
        branch_ids=branch_ids,
        branch_results=branch_results,
    )
    forecast_v1_result = _safe_result(
        "forecast",
        lambda: _run_or_resume_forecasting_branch(
            root,
            _forecast_config(config, physical_result),
            branch_ids["forecast"],
            execution,
            run_forecasting_branch,
        ),
    )
    branch_results["forecast_v1"] = forecast_v1_result
    if forecast_v1_result.get("forecast_recovery") is True:
        branch_results["forecast_recovery"] = forecast_v1_result
    if (
        dart_required
        and _status(forecast_v1_result) == "complete"
        and forecast_v1_result.get("run_manifest")
    ):
        from dcvpp.experiments.dart_pipeline import run_v1_dart_readiness_stage

        _write_running_state(
            root,
            state,
            stage="paper_dart_v1_readiness",
            branch_ids=branch_ids,
            branch_results=branch_results,
        )
        immutable_v1_manifest = Path(str(forecast_v1_result["run_manifest"]))
        if not immutable_v1_manifest.is_absolute():
            immutable_v1_manifest = root / immutable_v1_manifest
        dart_config_value = execution.get(
            "dart_config_path", "configs/dart_value_select.yaml"
        )
        readiness_config_path = Path(str(dart_config_value))
        if not readiness_config_path.is_absolute():
            readiness_config_path = root / readiness_config_path
        v1_readiness_result = _safe_result(
            "dart_v1_readiness",
            lambda: run_v1_dart_readiness_stage(
                root,
                v1_manifest_path=immutable_v1_manifest,
                dart_config_path=readiness_config_path,
                run_id=branch_ids["dart_v1_readiness"],
                resume=True,
            ),
        )
    elif dart_required:
        v1_readiness_result = {
            "status": "not_run_upstream_incomplete",
            "downstream_ready": False,
            "forecast_v1_status": forecast_v1_result.get("status"),
        }
    else:
        v1_readiness_result = {
            "status": "not_required_legacy_compatibility",
            "downstream_ready": True,
        }
    if dart_required:
        branch_results["dart_v1_readiness"] = v1_readiness_result
    readiness_ready = bool(
        not dart_required
        or (
            _status(v1_readiness_result) == "complete"
            and v1_readiness_result.get("downstream_ready") is True
            and isinstance(v1_readiness_result.get("readiness_audit"), Mapping)
            and v1_readiness_result["readiness_audit"].get("passed") is True
        )
    )
    forecast_result: dict[str, Any]
    forecast_extension_required = bool(
        isinstance(config.get("optimization_branch", {}), dict)
        and config.get("optimization_branch", {}).get("run_ofat_sensitivities")
        is True
    )
    if (
        forecast_extension_required
        and _status(forecast_v1_result) == "complete"
        and forecast_v1_result.get("run_manifest")
        and readiness_ready
    ):
        _write_running_state(
            root,
            state,
            stage="paper_forecast_extensions",
            branch_ids=branch_ids,
            branch_results=branch_results,
        )
        v1_manifest_path = Path(str(forecast_v1_result["run_manifest"]))
        if not v1_manifest_path.is_absolute():
            v1_manifest_path = root / v1_manifest_path
        v1_manifest_path = v1_manifest_path.resolve()
        extension_root = (
            root
            / "artifacts"
            / "real_forecasting_extensions"
            / f"run_id={branch_ids['forecast_extension']}"
        )
        extension_manifest_path = extension_root / "extension_run_manifest.json"
        extension_config = config.get("forecast_extension", {})
        extension_config = (
            extension_config if isinstance(extension_config, dict) else {}
        )
        forecast_branch_config = config.get("forecast_branch", {})
        forecast_branch_config = (
            forecast_branch_config
            if isinstance(forecast_branch_config, dict)
            else {}
        )
        workload_panel_value = extension_config.get("workload_panel_path")
        workload_panel_path = (
            None
            if workload_panel_value in (None, "")
            else (
                Path(str(workload_panel_value))
                if Path(str(workload_panel_value)).is_absolute()
                else root / str(workload_panel_value)
            )
        )
        forecast_extension_result = _safe_result(
            "forecast_extension",
            lambda: run_referenced_paper_forecast_extensions(
                root,
                run_id=branch_ids["forecast_extension"],
                v1_manifest_path=v1_manifest_path,
                output_root=extension_root,
                workload_panel_path=workload_panel_path,
                device_name=str(
                    extension_config.get(
                        "device",
                        forecast_branch_config.get("torch_device", "auto"),
                    )
                ),
            ),
        )
        branch_results["forecast_extension"] = forecast_extension_result
        forecast_result = _safe_result(
            "forecast_extension_bridge",
            lambda: load_complete_forecast_evidence(
                root, v1_manifest_path, extension_manifest_path
            ),
        )
        extension_run_manifest = forecast_result.get(
            "run_manifest", str(extension_manifest_path)
        )
        forecast_result["extension_run_manifest"] = extension_run_manifest
        forecast_result["extension_manifest_sha256"] = forecast_result.get("sha256")
        # The v1 producer manifest remains the authority for training scope,
        # split chronology, and its immutable output ledger.  The bridge owns
        # the post-v1 full-year deployment index and confirmatory H1/H2 records.
        forecast_result["run_manifest"] = str(forecast_v1_result["run_manifest"])
        forecast_result["v1_forecast_result"] = forecast_v1_result
        forecast_result.setdefault(
            "source_relationship", "third_party_referenced_public_data"
        )
        forecast_result.setdefault(
            "measurement_relationship", "not_measured_by_this_study"
        )
    elif _status(forecast_v1_result) == "complete" and not readiness_ready:
        forecast_result = {
            **forecast_v1_result,
            "status": "PARTIAL",
            "optimization_allowed": False,
            "actionable_pending_count": max(
                1,
                int(v1_readiness_result.get("actionable_pending_count", 0) or 0),
            ),
            "reason": (
                "forecast recovery is complete, but the frozen pre-DART v1 "
                "readiness contract is not COMPLETE"
            ),
        }
    elif _status(forecast_v1_result) == "complete" and not forecast_extension_required:
        forecast_result = {
            **forecast_v1_result,
            "optimization_allowed": True,
            "actionable_pending_count": 0,
        }
    else:
        forecast_result = {
            **forecast_v1_result,
            "optimization_allowed": False,
            "actionable_pending_count": int(
                forecast_v1_result.get("actionable_pending_count", 0) or 0
            ),
            "reason": "post-v1 extension requires a COMPLETE paper-forecast-v1 run",
        }
    branch_results["forecast"] = forecast_result
    atomic_json(
        execution_manifest,
        {
            "run_id": parent_run_id,
            "status": PAPER_EXECUTION_RUNNING,
            "terminal": False,
            "branch_ids": branch_ids,
            "branch_results": branch_results,
            "updated_at_utc": utc_now(),
        },
    )

    dart_result: dict[str, Any]
    if (
        _status(physical_result) in {"complete", "completed"}
        and _status(forecast_result) == "complete"
        and dart_required
    ):
        from dcvpp.experiments.dart_pipeline import (
            attach_dart_overlay_to_forecast,
            run_dart_value_select_pipeline,
        )

        _write_running_state(
            root,
            state,
            stage="paper_dart_value_select",
            branch_ids=branch_ids,
            branch_results=branch_results,
        )
        dart_config_value = execution.get(
            "dart_config_path", "configs/dart_value_select.yaml"
        )
        dart_config_path = Path(str(dart_config_value))
        if not dart_config_path.is_absolute():
            dart_config_path = root / dart_config_path
        dart_result = _safe_result(
            "dart_value_select",
            lambda: run_dart_value_select_pipeline(
                root,
                dart_config_path=dart_config_path,
                forecast_run_id=branch_ids["dart_forecast"],
                replay_run_id=branch_ids["dart_replay"],
                value_select_run_id=branch_ids["dart_value_select"],
                physical_result=physical_result,
                forecast_result=forecast_result,
                v1_readiness_result=v1_readiness_result,
                resume=True,
            ),
        )
        branch_results["dart"] = dart_result
        if (
            _dart_downstream_admissible(dart_result)
            and isinstance(dart_result.get("run_ids"), Mapping)
            and dart_result["run_ids"].get("dart_forecast")
            == branch_ids["dart_forecast"]
            and dart_result["run_ids"].get("dart_replay")
            == branch_ids["dart_replay"]
            and dart_result["run_ids"].get("dart_value_select")
            == branch_ids["dart_value_select"]
        ):
            overlay_result = _safe_result(
                "dart_overlay",
                lambda: attach_dart_overlay_to_forecast(
                    root, forecast_result, dart_result
                ),
            )
            branch_results["dart_overlay"] = {
                "status": overlay_result.get("status"),
                "dart_overlay_manifest": overlay_result.get(
                    "dart_overlay_manifest"
                ),
                "dart_overlay_manifest_sha256": overlay_result.get(
                    "dart_overlay_manifest_sha256"
                ),
                "error_type": overlay_result.get("error_type"),
                "error": overlay_result.get("error"),
                "candidate_pool_contract": overlay_result.get(
                    "dart_value_select", {}
                ).get("candidate_pool_contract")
                if isinstance(overlay_result.get("dart_value_select"), Mapping)
                else None,
                "terminal_status_cap": overlay_result.get(
                    "dart_value_select", {}
                ).get("terminal_status_cap")
                if isinstance(overlay_result.get("dart_value_select"), Mapping)
                else None,
            }
            if _status(overlay_result) == "complete":
                forecast_result = overlay_result
                branch_results["forecast"] = forecast_result
            else:
                dart_result = {
                    **dart_result,
                    "status": "failed_overlay_bridge",
                    "downstream_ready": False,
                    "overlay_error": overlay_result.get("error"),
                }
                branch_results["dart"] = dart_result
    elif dart_required:
        dart_result = {
            "status": "not_run_upstream_incomplete",
            "completed": False,
            "downstream_ready": False,
            "forecast_status": forecast_result.get("status"),
            "actionable_pending_count": 0,
        }
        branch_results["dart"] = dart_result
    else:
        dart_result = {
            "status": "not_required_legacy_compatibility",
            "completed": False,
            "downstream_ready": True,
            "actionable_pending_count": 0,
        }

    atomic_json(
        execution_manifest,
        {
            "run_id": parent_run_id,
            "status": PAPER_EXECUTION_RUNNING,
            "terminal": False,
            "branch_ids": branch_ids,
            "branch_results": branch_results,
            "updated_at_utc": utc_now(),
        },
    )

    physical_ready = _status(physical_result) in {"complete", "completed"}
    forecast_ready = bool(
        _status(forecast_result) == "complete"
        and forecast_result.get("optimization_allowed") is True
    )
    dart_ready = bool(
        not dart_required
        or (
            _dart_downstream_admissible(dart_result)
            and isinstance(dart_result.get("run_ids"), Mapping)
            and dart_result["run_ids"].get("dart_forecast")
            == branch_ids["dart_forecast"]
            and dart_result["run_ids"].get("dart_replay")
            == branch_ids["dart_replay"]
            and dart_result["run_ids"].get("dart_value_select")
            == branch_ids["dart_value_select"]
        )
    )
    if physical_ready and forecast_ready and dart_ready:
        _write_running_state(
            root,
            state,
            stage="paper_scenarios_optimization",
            branch_ids=branch_ids,
            branch_results=branch_results,
        )
        optimization_result = _safe_result(
            "optimization",
            lambda: run_optimization_branch(
                root,
                optimization_config,
                branch_ids["optimization"],
                physical_result,
                forecast_result,
            ),
        )
        failure_retry_round = 1
        resume_invocations = 1
        max_resume_invocations = execution.get("max_optimization_invocations_per_command")
        while True:
            pending = int(
                optimization_result.get(
                    "actionable_pending_count",
                    optimization_result.get("pending_count", 0),
                )
                or 0
            )
            failures = int(optimization_result.get("failure_count", 0) or 0) + int(
                optimization_result.get("auxiliary_failure_count", 0) or 0
            )
            if _status(optimization_result) == "complete" or (pending == 0 and failures == 0):
                break
            if pending > 0:
                processed = int(optimization_result.get("processed_this_invocation", 0) or 0)
                if processed == 0:
                    break
                if (
                    max_resume_invocations is not None
                    and resume_invocations >= int(max_resume_invocations)
                ):
                    break
                resume_invocations += 1
            elif failures > 0:
                if failure_retry_round >= retry_rounds:
                    break
                failure_retry_round += 1
            optimization_result = _safe_result(
                "optimization",
                lambda: run_optimization_branch(
                    root,
                    optimization_config,
                    branch_ids["optimization"],
                    physical_result,
                    forecast_result,
                ),
            )
    else:
        optimization_result = {
            "status": "not_run_upstream_incomplete",
            "completed": False,
            "pending_count": 0,
            "failure_count": 0,
            "auxiliary_failure_count": 0,
            "auxiliary_pending_count": 0,
            "physical_status": physical_result.get("status"),
            "forecast_status": forecast_result.get("status"),
            "dart_status": dart_result.get("status"),
        }
    branch_results["optimization"] = optimization_result

    atomic_json(
        execution_manifest,
        {
            "run_id": parent_run_id,
            "status": PAPER_EXECUTION_RUNNING,
            "terminal": False,
            "synthetic_fallback": False,
            "preflight_status": preflight_result.get("status"),
            "branch_ids": branch_ids,
            "branch_results": branch_results,
            "fingerprint": sha256_json(
                {
                    "branch_ids": branch_ids,
                    "statuses": {
                        key: value.get("status") for key, value in branch_results.items()
                    },
                }
            ),
            "updated_at_utc": utc_now(),
        },
    )

    pending = int(
        optimization_result.get(
            "actionable_pending_count", optimization_result.get("pending_count", 0)
        )
        or 0
    )
    forecast_pending = int(forecast_result.get("actionable_pending_count", 0) or 0)
    dart_pending = int(dart_result.get("actionable_pending_count", 0) or 0)
    if forecast_pending > 0:
        _write_running_state(
            root,
            state,
            stage="paper_forecast_extension_resume_pending",
            branch_ids=branch_ids,
            branch_results=branch_results,
        )
        return {
            "status": PAPER_EXECUTION_RUNNING,
            "terminal": False,
            "paper_training_started": True,
            "paper_results_available": True,
            "branch_results": branch_results,
            "execution_manifest": execution_manifest,
        }
    if dart_pending > 0:
        _write_running_state(
            root,
            state,
            stage="paper_dart_resume_pending",
            branch_ids=branch_ids,
            branch_results=branch_results,
        )
        return {
            "status": PAPER_EXECUTION_RUNNING,
            "terminal": False,
            "paper_training_started": True,
            "paper_results_available": True,
            "branch_results": branch_results,
            "execution_manifest": execution_manifest,
        }
    if pending > 0:
        _write_running_state(
            root,
            state,
            stage="paper_optimization_resume_pending",
            branch_ids=branch_ids,
            branch_results=branch_results,
        )
        return {
            "status": PAPER_EXECUTION_RUNNING,
            "terminal": False,
            "paper_training_started": True,
            "paper_results_available": True,
            "branch_results": branch_results,
            "execution_manifest": execution_manifest,
        }

    # Imported only after every safe branch has exhausted its current queue.
    from dcvpp.paper_finalization import finalize_paper_run

    return finalize_paper_run(
        root,
        config,
        parent_run_id,
        preflight_result,
        physical_result,
        forecast_result,
        optimization_result,
        dart_result=dart_result,
        dart_v1_readiness_result=v1_readiness_result,
        state=state,
    )
