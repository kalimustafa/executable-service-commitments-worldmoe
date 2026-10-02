from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from dcvpp.core import (
    PipelineState,
    atomic_json,
    atomic_text,
    environment_snapshot,
    load_yaml,
    sha256_file,
    sha256_json,
    utc_now,
)
from dcvpp.data.nlr import (
    NLR_RESOURCES,
    evaluate_nlr300_timezone,
    profile_nlr295_archive,
    profile_nlr300_parquet,
)
from dcvpp.paper_outputs import write_blocked_paper_contract, write_ready_paper_contract
from dcvpp.paper_status import (
    DEMO_COMPLETE_PAPER_BLOCKED,
    PAPER_PREFLIGHT_READY,
    TERMINAL_STATUSES,
    assert_preflight_not_promoted_to_paper_result,
    classify_paper_preflight,
)
from dcvpp.reporting import build_blocked_report, build_run_manifest


TERMINAL_STATUS = DEMO_COMPLETE_PAPER_BLOCKED
NLR300_TIMEZONE_BLOCKER = "BLOCKED_TIMEZONE_NLR300"
SOURCE_RELATIONSHIP_POLICY = {
    "scope": "third_party_referenced_sources",
    "study_measurement_claimed": False,
    "nlr295": "publisher-reported monitored and TDP-estimated variables remain distinct",
    "nlr300": "publisher public facility telemetry",
    "nasa_power": "reconstructed meteorology",
    "nyiso": "official historical market and load records",
    "xcel": "official tariff and program-rule publications",
}


def _execution_branch_ids(config: dict[str, Any], run_id: str) -> dict[str, str]:
    execution = dict(config.get("paper_execution", {}))
    return {
        "physical": str(execution.get("physical_run_id", f"{run_id}-physical-v1")),
        "forecast": str(execution.get("forecast_run_id", f"{run_id}-forecast-v1")),
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
            execution.get("optimization_run_id", f"{run_id}-optimization-v1")
        ),
    }


def _existing_terminal_result(root: Path, state: PipelineState) -> dict[str, Any] | None:
    paper_state = state.data.get("paper", {})
    status = state.data.get("terminal_status") or paper_state.get("run_status")
    if status not in TERMINAL_STATUSES:
        return None
    manifest_path = root / "artifacts/manifests/run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("terminal paper state exists but its run manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_status = manifest.get("terminal_status", manifest.get("status"))
    if manifest_status != status or manifest.get("terminal") is not True:
        raise RuntimeError("terminal paper state and run manifest disagree; refusing to overwrite either")
    return {
        "status": status,
        "terminal": True,
        "resumed": True,
        "paper_results_available": bool(manifest.get("paper_results_available", False)),
        "paper_training_started": bool(manifest.get("paper_training_started", False)),
        "run_manifest": manifest_path,
        "branch_results": manifest.get("branch_results", {}),
    }


def _readiness_run_id_amendment_is_valid(
    root: Path,
    execution_path: Path,
    execution_manifest: dict[str, Any],
    changed: dict[str, dict[str, Any]],
) -> bool:
    if set(changed) != {"dart_v1_readiness"}:
        return False
    amendment_path = (
        root / "artifacts/manifests/paper_execution_readiness_run_id_amendment.json"
    )


def _anchor_lineage_run_id_amendment_is_valid(
    root: Path,
    config: dict[str, Any],
    execution_manifest: dict[str, Any],
    changed: dict[str, dict[str, Any]],
) -> bool:
    """Allow only the frozen v7/v3 -> v8/v4 pre-OOS lineage transition."""

    permitted = {
        "dart_v1_readiness",
        "dart_forecast",
        "dart_replay",
        "dart_value_select",
    }
    if not changed or not set(changed).issubset(permitted):
        return False
    execution = config.get("paper_execution", {})
    if not isinstance(execution, dict):
        return False
    config_value = execution.get("dart_config_path", "configs/dart_value_select.yaml")
    config_path = Path(str(config_value))
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        dart_config = load_yaml(config_path)
    except (OSError, ValueError, TypeError):
        return False
    readiness = dart_config.get("pre_dart_v1_readiness", {})
    amendment = (
        readiness.get("anchor_only_checkpoint_amendment")
        if isinstance(readiness, dict)
        else None
    )
    if not isinstance(amendment, dict) or amendment.get("enabled") is not True:
        return False
    expected_ids = _execution_branch_ids(config, str(execution_manifest.get("run_id", "")))
    expected_transition = {
        "dart_v1_readiness": (
            amendment.get("prior_run_id"),
            readiness.get("formal_run_id"),
        ),
        "dart_forecast": (
            amendment.get("source_run_id"),
            amendment.get("destination_run_id"),
        ),
        "dart_replay": (
            "paper-dart-replay-v3",
            expected_ids["dart_replay"],
        ),
        "dart_value_select": (
            "paper-dart-value-select-v3",
            expected_ids["dart_value_select"],
        ),
    }
    for key, value in changed.items():
        recorded, expected = expected_transition[key]
        # Older paper manifests may predate v3 run-ID persistence. They are
        # admissible only while no DART or optimization result exists.
        if value["expected"] != expected or value["recorded"] not in {
            recorded,
            "paper-dart-v1-readiness-v5" if key == "dart_v1_readiness" else None,
            "paper-dart-forecast-v1" if key == "dart_forecast" else None,
            "paper-dart-replay-v1" if key == "dart_replay" else None,
            "paper-dart-value-select-v1" if key == "dart_value_select" else None,
        }:
            return False
    branch_results = execution_manifest.get("branch_results", {})
    if not isinstance(branch_results, dict):
        return False
    for key in ("dart", "optimization"):
        result = branch_results.get(key, {})
        if isinstance(result, dict) and str(result.get("status", "")).lower() in {
            "complete",
            "completed",
        }:
            return False
    snapshot_value = amendment.get("source_snapshot_manifest_path")
    snapshot_path = Path(str(snapshot_value))
    if not snapshot_path.is_absolute():
        snapshot_path = root / snapshot_path
    if not snapshot_path.is_file():
        return False
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    core = {key: value for key, value in snapshot.items() if key != "manifest_sha256"}
    return bool(
        snapshot.get("status") == "COMPLETE_HASH_PINNED_READ_ONLY_SOURCE"
        and snapshot.get("source_run_id") == amendment.get("source_run_id")
        and snapshot.get("destination_run_id") == amendment.get("destination_run_id")
        and snapshot.get("records_sha256")
        == amendment.get("source_anchor_records_sha256")
        and int(snapshot.get("record_count", -1))
        == int(amendment.get("source_anchor_record_count", -2))
        and int(snapshot.get("total_bytes", -1))
        == int(amendment.get("source_anchor_total_bytes", -2))
        and snapshot.get("completed_oos_unit_count") == 0
        and snapshot.get("copy_hardlink_or_source_rewrite_performed") is False
        and snapshot.get("manifest_sha256") == sha256_json(core)
        and amendment.get("source_snapshot_manifest_file_sha256")
        in (None, sha256_file(snapshot_path))
        and amendment.get("source_snapshot_manifest_byte_count")
        in (None, snapshot_path.stat().st_size)
        and amendment.get("source_snapshot_manifest_sha256")
        in (None, snapshot.get("manifest_sha256"))
    )
    protocol_path = root / "artifacts/manifests/three_seed_runtime_amendment.json"
    if not amendment_path.is_file() or not protocol_path.is_file():
        return False
    try:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        readiness_path = root / str(amendment["new_readiness_manifest_path"])
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError, TypeError):
        return False
    change = changed["dart_v1_readiness"]
    branch_results = execution_manifest.get("branch_results", {})
    old_result = branch_results.get("dart_v1_readiness", {})
    downstream_statuses = {
        branch: branch_results.get(branch, {}).get("status")
        for branch in ("forecast_extension", "dart")
    }
    return bool(
        amendment.get("schema_version")
        == "dcvpp.paper_execution_readiness_run_id_amendment/v1"
        and amendment.get("status") == "COMPLETE"
        and amendment.get("old_run_id") == change["recorded"]
        and amendment.get("new_run_id") == change["expected"]
        and amendment.get("old_paper_execution_manifest_sha256")
        == sha256_file(execution_path)
        and amendment.get("new_readiness_manifest_sha256")
        == sha256_file(readiness_path)
        and amendment.get("three_seed_protocol_amendment_sha256")
        == sha256_file(protocol_path)
        and amendment.get("confirmatory_study_seeds")
        == [20260809, 20260819, 20260829]
        and amendment.get("downstream_training_started_before_amendment") is False
        and old_result.get("status") == "failed"
        and all(
            status in {None, "not_run", "not_run_upstream_incomplete"}
            for status in downstream_statuses.values()
        )
        and readiness.get("status") == "COMPLETE"
        and readiness.get("formal_dart_training_allowed") is True
        and readiness.get("all_candidate_replay_seed_slots")
        == [20260809, 20260819, 20260829]
    )


def _existing_execution_preflight(
    root: Path,
    config: dict[str, Any],
    state: PipelineState,
) -> dict[str, Any] | None:
    """Recover an in-progress execution without rewriting preflight outputs."""

    run_id = str(state.data["run_id"])
    execution_path = root / "artifacts/manifests/paper_execution_manifest.json"
    execution_manifest: dict[str, Any] = {}
    if execution_path.is_file():
        execution_manifest = json.loads(execution_path.read_text(encoding="utf-8"))
    paper_state = state.data.get("paper", {})
    running_in_state = paper_state.get("run_status") == "PAPER_EXECUTION_RUNNING"
    running_in_manifest = (
        execution_manifest.get("status") == "PAPER_EXECUTION_RUNNING"
        and execution_manifest.get("terminal") is False
    )
    if not (running_in_state or running_in_manifest):
        return None
    manifest_run_id = execution_manifest.get("run_id", run_id)
    if str(manifest_run_id) != run_id:
        raise RuntimeError("paper execution manifest belongs to a different top-level run")
    expected_ids = _execution_branch_ids(config, run_id)
    recorded_ids = execution_manifest.get("branch_ids") or paper_state.get("branch_ids")
    if recorded_ids:
        recorded = dict(recorded_ids)
        unknown = set(recorded).difference(expected_ids)
        changed = {
            key: {"recorded": value, "expected": expected_ids.get(key)}
            for key, value in recorded.items()
            if key in expected_ids and expected_ids[key] != value
        }
        required_legacy_base = {"physical", "forecast", "optimization"}
        allowed_readiness_amendment = (
            not unknown
            and bool(changed)
            and _readiness_run_id_amendment_is_valid(
                root, execution_path, execution_manifest, changed
            )
        )
        allowed_anchor_lineage_amendment = (
            not unknown
            and bool(changed)
            and _anchor_lineage_run_id_amendment_is_valid(
                root, config, execution_manifest, changed
            )
        )
        if (
            unknown
            or (
                changed
                and not (
                    allowed_readiness_amendment
                    or allowed_anchor_lineage_amendment
                )
            )
            or not required_legacy_base.issubset(recorded)
        ):
            raise RuntimeError(
                "paper execution branch IDs changed during resume; use new explicit "
                f"run IDs (unknown={sorted(unknown)}, changed={changed})"
            )
    source_manifest = root / "artifacts/manifests/source_manifest.json"
    timezone_manifest = root / "artifacts/manifests/timezone_manifest.json"
    missing = [path for path in (source_manifest, timezone_manifest) if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"paper execution is in progress but preflight evidence is missing: {missing}"
        )
    return {
        "status": execution_manifest.get("preflight_status", PAPER_PREFLIGHT_READY),
        "terminal": False,
        "resumed": True,
        "synthetic_fallback": False,
        "blockers": [],
        "source_manifest": source_manifest,
        "timezone_manifest": timezone_manifest,
    }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _retrieved_at_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _contract_records(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for role, raw_path in config.get("contracts", {}).items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"governing {role} contract is missing: {path}")
        contracts.append({"role": str(role), **_file_record(root.parent, path)})
    if not contracts:
        raise ValueError("paper configuration has no governing contracts")
    return contracts


def _demo_reuse_evidence(root: Path, demo: dict[str, Any]) -> dict[str, Any]:
    run_manifest_path = root / "artifacts/demo/manifests/run_manifest.json"
    gate_path = root / "artifacts/demo/manifests/demo_gate.json"
    if not run_manifest_path.is_file() or not gate_path.is_file():
        raise RuntimeError("a completed demo run is required before paper preflight")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if demo.get("status") != "DEMO_COMPLETE":
        raise RuntimeError(f"caller did not provide a completed demo: {demo.get('status')!r}")
    if run_manifest.get("status") != "DEMO_COMPLETE" or run_manifest.get("synthetic") is not True:
        raise RuntimeError("stored demo run manifest is not a completed synthetic smoke test")
    if gate.get("status") != "passed":
        raise RuntimeError("stored demo evidence gate did not pass")
    return {
        "status": "DEMO_COMPLETE",
        "role": "engineering_smoke_test_only",
        "reused_without_rerun": True,
        "metrics_reused_for_paper": False,
        "run_manifest": _file_record(root, run_manifest_path),
        "gate": _file_record(root, gate_path),
    }


def _nlr_source_records(
    root: Path,
    *,
    nlr295_profile: dict[str, Any],
    nlr300_power_profile: dict[str, Any],
    nlr300_weather_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_root = root / "data/raw"
    known_hashes = {
        "nlr295_energy_metrics_archive": nlr295_profile["sha256"],
        "nlr300_pue_parquet": nlr300_power_profile["sha256"],
        "nlr300_weather_parquet": nlr300_weather_profile["sha256"],
    }
    records: list[dict[str, Any]] = []
    for resource in NLR_RESOURCES:
        path = raw_root / resource.relative_path
        if not path.is_file():
            continue
        record: dict[str, Any] = {
            "source_id": resource.source_id,
            "dataset_id": resource.dataset_id,
            "stable_url": resource.stable_url,
            "filename": path.name,
            "local_path": _relative(root, path),
            "byte_count": path.stat().st_size,
            "sha256": known_hashes.get(resource.source_id) or sha256_file(path),
            "retrieved_at_utc": _retrieved_at_utc(path),
            "source_relationship": "third_party_referenced_public_data",
            "provenance_scope": (
                "NLR publisher-reported monitored GPU/node energy, declared TDP-estimated CPU "
                "energy, and missing values remain distinct; none are measurements made by this study"
                if resource.dataset_id == 295
                else "NLR publisher public facility telemetry; referenced data, not measurement by this study"
            ),
        }
        if resource.catalog_version is not None:
            record["catalog_version"] = resource.catalog_version
        if resource.source_id == "nlr295_energy_metrics_archive":
            record.update(
                {
                    "publisher_md5_expected": nlr295_profile["publisher_md5_expected"],
                    "publisher_md5_actual": nlr295_profile["publisher_md5_actual"],
                    "publisher_md5_verified": nlr295_profile["publisher_md5_verified"],
                }
            )
        records.append(record)
    return records


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"provenance record must be an object: {path}")
    return value


def _resolve_local_source(root: Path, payload: dict[str, Any]) -> Path:
    path = Path(str(payload["local_path"]))
    return path if path.is_absolute() else (root / path).resolve()


def _verify_persisted_source(root: Path, payload: dict[str, Any]) -> Path:
    path = _resolve_local_source(root, payload)
    if not path.is_file():
        raise FileNotFoundError(f"persisted provenance points to a missing file: {path}")
    if int(payload["byte_count"]) != path.stat().st_size:
        raise ValueError(f"persisted provenance byte count mismatch: {path}")
    if str(payload["sha256"]).lower() != sha256_file(path).lower():
        raise ValueError(f"persisted provenance SHA256 mismatch: {path}")
    return path


def _provenance_file(root: Path, path: Path) -> dict[str, Any]:
    return {**_file_record(root, path), "retrieved_at_utc": _retrieved_at_utc(path)}


def _ingest_xcel_registry(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry_path = root / "data/raw/xcel/source_registry.json"
    if not registry_path.is_file():
        return [], {"status": "not_acquired", "record_count": 0}
    registry = _load_json_object(registry_path)
    extraction_payload: dict[str, Any] | None = None
    extraction_path_value = registry.get("extraction_status_path")
    extraction_path: Path | None = None
    if extraction_path_value:
        extraction_path = (root / str(extraction_path_value)).resolve()
        if extraction_path.is_file():
            extraction_payload = _load_json_object(extraction_path)
            if extraction_payload.get("source_relationship") != "third_party_referenced_public_document":
                raise ValueError("Xcel extraction record has an invalid source relationship")
            if extraction_payload.get("measurement_relationship") != "not_measured_by_this_study":
                raise ValueError("Xcel extraction record must deny study measurement")
    payloads = list(registry.get("sources", []))
    current_tariff = registry.get("current_direct_pdf")
    if isinstance(current_tariff, dict):
        payloads.append(current_tariff)
    records: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            continue
        path = _verify_persisted_source(root, payload)
        role = str(payload.get("role") or ("current_entire_tariff_snapshot" if payload is current_tariff else "source"))
        records.append(
            {
                "source_id": f"xcel_{role}_{index + 1}",
                "source_role": "exogenous_local_tariff_and_demand_response_rule",
                "training_target": False,
                "stable_url": payload["stable_url"],
                "filename": str(payload.get("filename") or path.name),
                "local_path": _relative(root, path),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
                "retrieved_at_utc": str(payload.get("retrieved_at_utc") or _retrieved_at_utc(path)),
                "content_type": payload.get("content_type", "application/pdf"),
                "evidence_role": payload.get("evidence_role", "observed"),
                "source_relationship": "third_party_referenced_public_document",
                "measurement_relationship": payload.get(
                    "measurement_relationship", "not_measured_by_this_study"
                ),
                "record_type": "referenced_official_tariff_or_program_rule",
                "provenance_scope": (
                    "Xcel tariff and program-rule snapshot referenced by this study; not a study "
                    "measurement, training label, or realized demand-response observation"
                ),
                "claim_boundary": payload.get(
                    "claim_boundary",
                    "tariff-rule snapshot only; not a training label or realized response",
                ),
                "provenance_path": _relative(root, registry_path),
            }
        )
    return records, {
        "status": registry.get("status", "unknown"),
        "source_owner": registry.get("source_owner"),
        "role": registry.get("role"),
        "training_target": registry.get("training_target", False),
        "probed_at_utc": registry.get("probed_at_utc"),
        "landing_url": registry.get("landing_url"),
        "current_tariff_automated": bool(
            isinstance(current_tariff, dict) and current_tariff.get("automated")
        ),
        "record_count": len(records),
        "registry_file": _provenance_file(root, registry_path),
        "extraction_status_file": (
            _provenance_file(root, extraction_path) if extraction_path is not None else None
        ),
        "observed_event_call_count": (
            extraction_payload.get("observed_xcel_event_calls", {}).get("record_count")
            if extraction_payload is not None
            else None
        ),
        "realized_response_evidence": (
            extraction_payload.get("realized_response_evidence")
            if extraction_payload is not None
            else False
        ),
        "large_business_schedule_candidates": (
            extraction_payload.get("large_business_schedule_candidates", [])
            if extraction_payload is not None
            else []
        ),
        "cpp_program_rules": (
            extraction_payload.get("cpp_program_rules")
            if extraction_payload is not None
            else None
        ),
    }


def _ingest_srrl_provenance(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provenance_paths = sorted((root / "data/raw/srrl").glob("*.provenance.json"))
    records: list[dict[str, Any]] = []
    provenance_files: list[dict[str, Any]] = []
    for provenance_path in provenance_paths:
        payload = _load_json_object(provenance_path)
        path = _verify_persisted_source(root, payload)
        schema = payload.get("schema", {}) if isinstance(payload.get("schema"), dict) else {}
        records.append(
            {
                "source_id": f"srrl_{payload.get('site', 'BMS')}_{payload.get('query_day', path.stem)}",
                "stable_url": payload["stable_url"],
                "filename": str(payload.get("filename") or path.name),
                "local_path": _relative(root, path),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
                "retrieved_at_utc": str(payload.get("retrieved_at_utc") or _retrieved_at_utc(path)),
                "evidence_role": payload.get("evidence_role", "observed"),
                "site": payload.get("site"),
                "query_day": payload.get("query_day"),
                "timezone": payload.get("timezone"),
                "missing_sentinel": payload.get("missing_sentinel"),
                "source_relationship": "third_party_referenced_public_data",
                "provenance_scope": (
                    "NLR-published SRRL irradiance and meteorology referenced by this study; "
                    "not measured by this study and not measured PV output"
                ),
                "schema": {
                    "version": schema.get("version"),
                    "sha256": schema.get("sha256"),
                    "column_count": schema.get("column_count"),
                    "time_column": schema.get("time_column"),
                    "timezone": schema.get("timezone"),
                    "api_contract": schema.get("api_contract"),
                },
                "provenance_path": _relative(root, provenance_path),
            }
        )
        provenance_files.append(_provenance_file(root, provenance_path))
    return records, {
        "status": "persisted_provenance_ingested" if records else "not_acquired",
        "record_count": len(records),
        "provenance_files": provenance_files,
    }


_NASA_FILENAME = re.compile(r"_(\d{8})_(\d{8})\.json$")


def _is_annual_nasa_payload(payload: dict[str, Any]) -> bool:
    match = _NASA_FILENAME.search(str(payload.get("filename", "")))
    if match is None:
        return False
    start, end = match.groups()
    return start[:4] == end[:4] and start[4:] == "0101" and end[4:] == "1231"


def _ingest_nasa_provenance(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provenance_paths = sorted((root / "data/raw/nasa_power").glob("*.provenance.json"))
    records: list[dict[str, Any]] = []
    provenance_files: list[dict[str, Any]] = []
    for provenance_path in provenance_paths:
        payload = _load_json_object(provenance_path)
        if not _is_annual_nasa_payload(payload):
            continue
        path = _verify_persisted_source(root, payload)
        match = _NASA_FILENAME.search(str(payload["filename"]))
        assert match is not None
        start, end = match.groups()
        zone = str(payload.get("zone"))
        records.append(
            {
                "source_id": f"nasa_power_{zone}_{start[:4]}",
                "stable_url": payload["stable_url"],
                "filename": str(payload["filename"]),
                "local_path": _relative(root, path),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
                "retrieved_at_utc": str(payload.get("retrieved_at_utc") or _retrieved_at_utc(path)),
                "evidence_role": payload.get("evidence_role", "reconstructed"),
                "zone": zone,
                "period_start": start,
                "period_end": end,
                "row_count": payload.get("row_count"),
                "time_standard": payload.get("time_standard"),
                "parameters": payload.get("parameters"),
                "frozen_site": payload.get("frozen_site"),
                "fill_value": payload.get("fill_value"),
                "source_relationship": "third_party_referenced_public_data",
                "record_type": "referenced_gridded_reconstruction",
                "provenance_scope": (
                    "NASA POWER reconstructed meteorology referenced by this study; not a "
                    "measurement made by this study"
                ),
                "provenance_path": _relative(root, provenance_path),
            }
        )
        provenance_files.append(_provenance_file(root, provenance_path))
    return records, {
        "status": "annual_provenance_ingested" if records else "not_acquired",
        "annual_record_count": len(records),
        "zones": sorted({str(record["zone"]) for record in records}),
        "years": sorted({str(record["period_start"])[:4] for record in records}),
        "provenance_files": provenance_files,
    }


def _nyiso_checkpoint_path(root: Path) -> Path | None:
    candidates = [
        root / "data/raw/nyiso/acquisition_checkpoint.json",
        root / "data/raw/nyiso/acquisition_manifest.json",
        root / "data/raw/nyiso/download_manifest.json",
        root / "artifacts/manifests/nyiso_acquisition_checkpoint.json",
        root / "artifacts/state/nyiso_acquisition_checkpoint.json",
        root / "artifacts/state/nyiso_download_state.json",
    ]
    candidates.extend(sorted((root / "data/raw/nyiso").rglob("*checkpoint*.json")))
    candidates.extend(sorted((root / "data/raw/nyiso").rglob("*manifest*.json")))
    return next((path for path in candidates if path.is_file()), None)


def _ingest_referenced_data_progress(root: Path) -> dict[str, Any]:
    path = root / "artifacts/manifests/referenced_data_progress_manifest.json"
    if not path.is_file():
        return {"status": "not_started", "counts": {"completed": 0, "failed": 0, "pending": None}}
    payload = _load_json_object(path)
    if payload.get("study_measurement_claimed") is not False:
        raise ValueError("referenced-data progress manifest must deny study measurement")
    if payload.get("paper_experiment_claimed") is not False:
        raise ValueError("referenced-data preparation cannot claim a paper experiment")
    if payload.get("training_claimed") is not False:
        raise ValueError("referenced-data preparation cannot claim training")
    sidecar = path.with_suffix(".sha256")
    return {
        "status": payload.get("status", "in_progress"),
        "stage": payload.get("stage"),
        "updated_at_utc": payload.get("updated_at_utc"),
        "source_relationship": payload.get("source_relationship"),
        "measurement_relationship": payload.get("measurement_relationship"),
        "counts": payload.get("counts", {}),
        "sources": {
            key: {
                field: value.get(field)
                for field in ("status", "expected_count", "completed_count", "failed_count", "pending_count")
            }
            for key, value in payload.get("sources", {}).items()
            if isinstance(value, dict)
        },
        "manifest_file": _provenance_file(root, path),
        "sha256_sidecar": _provenance_file(root, sidecar) if sidecar.is_file() else None,
    }


def _ingest_nyiso_checkpoint(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = _nyiso_checkpoint_path(root)
    if checkpoint_path is None:
        return [], {"status": "checkpoint_not_present", "archive_count": 0}
    checkpoint = _load_json_object(checkpoint_path)
    records: list[dict[str, Any]] = []
    for item in checkpoint.get("archives", []):
        if not isinstance(item, dict) or not isinstance(item.get("archive"), dict):
            continue
        archive = item["archive"]
        payload = {
            "local_path": item["local_path"],
            "byte_count": archive["byte_count"],
            "sha256": archive["sha256"],
        }
        path = _verify_persisted_source(root, payload)
        records.append(
            {
                "source_id": f"nyiso_{item.get('product')}_{item.get('month')}",
                "stable_url": item["stable_url"],
                "filename": str(archive.get("filename") or path.name),
                "local_path": _relative(root, path),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
                "retrieved_at_utc": str(item.get("retrieved_at") or _retrieved_at_utc(path)),
                "product": item.get("product"),
                "month": item.get("month"),
                "timezone": checkpoint.get("timezone"),
                "source_relationship": "third_party_referenced_public_data",
                "record_type": "referenced_official_historical_market_or_load_record",
                "provenance_scope": (
                    "NYISO official historical market/load archive referenced by this study; "
                    "not a measurement or market action made by this study"
                ),
                "provenance_path": _relative(root, checkpoint_path),
            }
        )
    failures = checkpoint.get("failures", [])
    return records, {
        "status": "checkpoint_ingested",
        "source": checkpoint.get("source"),
        "generated_at": checkpoint.get("generated_at"),
        "timezone": checkpoint.get("timezone"),
        "archive_count": len(records),
        "failure_count": len(failures) if isinstance(failures, list) else None,
        "archive_revision_boundary": (
            "NYISO may correct historical ZIPs; retrieval time and hashes identify the acquired "
            "revision, but an original row-level publication vintage is not available"
        ),
        "checkpoint_file": _provenance_file(root, checkpoint_path),
    }


def _assert_manifest_has_stable_urls_only(value: Any) -> None:
    """Allow deterministic official queries while rejecting signed credentials."""
    if isinstance(value, dict):
        forbidden = {"retrieval_url", "final_url", "presigned_url", "signed_url"}.intersection(value)
        if forbidden:
            raise ValueError(f"transient URL fields cannot be persisted: {sorted(forbidden)}")
        for key, item in value.items():
            if key.endswith("_url") and item is not None:
                parsed = urlsplit(str(item))
                if parsed.scheme != "https" or parsed.fragment:
                    raise ValueError(f"source manifest URL is not stable: {item}")
                if parsed.netloc.lower() == "data.nlr.gov" and parsed.query:
                    raise ValueError(f"NLR catalog/file URLs must be query-free: {item}")
                deterministic_query_hosts = {
                    "midcdmz.nlr.gov",
                    "power.larc.nasa.gov",
                    "www.xcelenergy.com",
                }
                if parsed.query and parsed.netloc.lower() not in deterministic_query_hosts:
                    raise ValueError(f"query URL is not from an admitted official endpoint: {item}")
                forbidden_query_tokens = ("signature", "token", "expires", "credential")
                for query_key, _ in parse_qsl(parsed.query, keep_blank_values=True):
                    normalized = query_key.lower()
                    if normalized.startswith("x-amz-") or any(
                        token in normalized for token in forbidden_query_tokens
                    ):
                        raise ValueError(f"signed/authenticated URL cannot be persisted: {item}")
            _assert_manifest_has_stable_urls_only(item)
    elif isinstance(value, list):
        for item in value:
            _assert_manifest_has_stable_urls_only(item)


def _paper_output_paths(root: Path) -> list[Path]:
    return [
        root / "artifacts/manifests/source_manifest.json",
        root / "artifacts/manifests/physical_calibration_manifest.json",
        root / "artifacts/manifests/digital_twin_transfer_manifest.json",
        root / "artifacts/manifests/run_manifest.json",
        root / "artifacts/tables/nlr_twin_validation.csv",
        root / "artifacts/tables/nyiso_forecast_main.csv",
        root / "artifacts/tables/workload_forecast_main.csv",
        root / "artifacts/tables/counterfactual_optimization_main.csv",
        root / "artifacts/tables/failures.csv",
        root / "artifacts/report/blockers.json",
        root / "artifacts/report/run_summary.md",
        root / "artifacts/state/pipeline_state.json",
        root / "artifacts/status/CURRENT_STAGE_STATUS.md",
    ]


def _write_terminal_state(
    root: Path,
    state: PipelineState,
    *,
    fingerprint: str,
    blockers: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
) -> None:
    outputs = [_relative(root, path) for path in _paper_output_paths(root)]
    state.data.setdefault("stages", {})["paper_preflight"] = {
        "name": "paper_preflight",
        "status": "blocked",
        "terminal_status": TERMINAL_STATUS,
        "fingerprint": fingerprint,
        "started_at": started_at,
        "finished_at": finished_at,
        "outputs": outputs,
        "message": "mandatory NLR300 authoritative-timezone gate blocked paper training",
        "blocker_codes": [item["code"] for item in blockers],
    }
    state.data["profile"] = "paper"
    state.data["terminal_status"] = TERMINAL_STATUS
    state.data["paper"] = {
        "gate_status": "blocked",
        "paper_results_available": False,
        "paper_training_started": False,
        "synthetic_fallback": False,
        "demo_reused_without_rerun": True,
        "demo_metrics_reused": False,
        "blocker_codes": [item["code"] for item in blockers],
    }
    state.data["updated_at"] = finished_at
    atomic_json(state.path, state.data)
    atomic_text(
        state.status_path,
        "\n".join(
            [
                "# Current Stage Status",
                "",
                f"- Run ID: `{state.data['run_id']}`",
                "- Stage: `paper_preflight`",
                f"- Terminal status: `{TERMINAL_STATUS}`",
                "- Paper results available: `false`",
                "- Paper training started: `false`",
                "- Synthetic fallback: `false`",
                "- Demo reused without rerun: `true`",
                f"- Fatal blocker: `{NLR300_TIMEZONE_BLOCKER}`",
                f"- Updated: `{finished_at}`",
                "",
            ]
        ),
    )


def _nlr300_timezone_options(config: dict[str, Any]) -> dict[str, str]:
    nlr300 = config.get("sources", {}).get("nlr300", {})
    timezone_config = nlr300.get("timezone", {}) if isinstance(nlr300, dict) else {}
    if not isinstance(timezone_config, dict):
        timezone_config = {}
    return {
        "policy": str(timezone_config.get("policy", "strict_authoritative")),
        "assumed_timezone": str(
            timezone_config.get(
                "name",
                timezone_config.get("direct_local_sensitivity", {}).get(
                    "timezone", "America/Denver"
                ),
            )
        ),
        "source_clock_assumption": str(timezone_config.get("source_clock_assumption", "UTC")),
        "analysis_timezone": str(timezone_config.get("analysis_timezone", "America/Denver")),
        "ambiguous": str(
            timezone_config.get("direct_local_sensitivity", {}).get(
                "ambiguous", timezone_config.get("ambiguous", "standard_time")
            )
        ),
        "nonexistent": str(
            timezone_config.get("direct_local_sensitivity", {}).get(
                "nonexistent", timezone_config.get("nonexistent", "mark_invalid")
            )
        ),
    }


def _write_ready_state(
    root: Path,
    state: PipelineState,
    *,
    fingerprint: str,
    timezone_gate: dict[str, Any],
    referenced_data_progress: dict[str, Any],
    started_at: str,
    finished_at: str,
) -> None:
    source_clock = timezone_gate.get("source_clock_assumption", timezone_gate.get("resolved_timezone"))
    analysis_timezone = timezone_gate.get("analysis_timezone", timezone_gate.get("resolved_timezone"))
    outputs = [_relative(root, path) for path in _paper_output_paths(root)]
    outputs.append("artifacts/manifests/timezone_manifest.json")
    state.data.setdefault("stages", {})["paper_preflight"] = {
        "name": "paper_preflight",
        "status": "completed",
        "run_status": PAPER_PREFLIGHT_READY,
        "terminal": False,
        "fingerprint": fingerprint,
        "started_at": started_at,
        "finished_at": finished_at,
        "outputs": outputs,
        "message": "timezone gate passed; real H0-H6 paper execution remains pending",
        "blocker_codes": [],
    }
    state.data.pop("terminal_status", None)
    state.data["profile"] = "paper"
    state.data["paper"] = {
        "run_status": PAPER_PREFLIGHT_READY,
        "terminal": False,
        "gate_status": "passed",
        "paper_results_available": False,
        "paper_training_started": False,
        "synthetic_fallback": False,
        "demo_reused_without_rerun": True,
        "demo_metrics_reused": False,
        "blocker_codes": [],
        "nlr300_timezone": timezone_gate,
        "time_encoding_sensitivity_status": "required_before_paper_claims",
        "referenced_data_panels": referenced_data_progress,
    }
    state.data["updated_at"] = finished_at
    atomic_json(state.path, state.data)
    atomic_text(
        state.status_path,
        "\n".join(
            [
                "# Current Stage Status",
                "",
                f"- Run ID: `{state.data['run_id']}`",
                "- Stage: `paper_preflight`",
                f"- Run status: `{PAPER_PREFLIGHT_READY}`",
                "- Terminal: `false`",
                "- Terminal paper status: `none`",
                "- NLR300 timezone gate: `passed`",
                f"- NLR300 source-clock assumption: `{source_clock}`",
                f"- Local analysis timezone: `{analysis_timezone}`",
                f"- Timezone evidence role: `{timezone_gate.get('evidence_role')}`",
                f"- Referenced-data panels: `{referenced_data_progress.get('status')}`",
                f"- Referenced-data counts: `{referenced_data_progress.get('counts')}`",
                "- Paper results available: `false`",
                "- Paper training started: `false`",
                "- Required next work: real physical calibration and H0-H6 execution",
                "- Time-encoding sensitivity: `required_before_paper_claims`",
                f"- Updated: `{finished_at}`",
                "",
            ]
        ),
    )


def _write_ready_report(
    root: Path,
    timezone_gate: dict[str, Any],
    referenced_data_progress: dict[str, Any],
) -> Path:
    path = root / "artifacts/report/run_summary.md"
    source_clock = timezone_gate.get("source_clock_assumption", timezone_gate.get("resolved_timezone"))
    analysis_timezone = timezone_gate.get("analysis_timezone", timezone_gate.get("resolved_timezone"))
    atomic_text(
        path,
        "\n".join(
            [
                "# Research Run Summary",
                "",
                f"- Run status: `{PAPER_PREFLIGHT_READY}`",
                "- Terminal: `false`",
                "- Terminal paper status: `none`",
                "- Paper results available: `false`",
                "- Paper training started: `false`",
                "- Counterfactual boundary: NLR was not operated or bid in NYISO.",
                "",
                "## Timezone decision",
                "",
                f"- NLR300 naive timestamps use `{source_clock}` as the user-authorized source-clock assumption and are converted to `{analysis_timezone}` for local analysis.",
                "- This is an `assumed` evidence role, not publisher-confirmed timestamp encoding.",
                "- The raw `ts` column is preserved; UTC localization followed by timezone conversion does not fabricate or collapse DST instants.",
                "- A declared time-encoding sensitivity is required before any paper claim because raw clock patterns do not by themselves confirm the storage encoding.",
                "",
                "## Execution boundary",
                "",
                "NLR295, NASA POWER, and NYISO inputs are third-party referenced public data. "
                "`observed` denotes a source-level historical record and never a measurement or market action performed by this study.",
                "",
                "The old NLR300 timezone blocker is cleared. Physical calibration, digital-twin transfer, forecasting, scenarios, optimization, and H0-H6 have not yet run. Preflight readiness is not PARTIAL_PAPER or COMPLETE_PAPER.",
                "",
                "## Referenced-data preparation",
                "",
                f"- Status: `{referenced_data_progress.get('status')}`",
                f"- Counts: `{referenced_data_progress.get('counts')}`",
                "- This stage creates source-separated data panels and inventories; it is not model training or a paper experiment.",
                "",
                "## Referenced-source boundary",
                "",
                "NLR, NASA POWER, NYISO, SRRL, and Xcel are third-party referenced public data or official records, not measurements made by this study. NLR 295 publisher-reported monitored energy remains distinct from its declared TDP-estimated energy; NASA POWER remains reconstructed meteorology; and NYISO remains an official historical record.",
                "",
            ]
        ),
    )
    return path


def _run_ready_paper_preflight(
    root: Path,
    *,
    config_record: dict[str, Any],
    contracts: list[dict[str, Any]],
    state: PipelineState,
    demo_evidence: dict[str, Any],
    environment: dict[str, Any],
    nlr295_profile: dict[str, Any],
    nlr300_power_profile: dict[str, Any],
    nlr300_weather_profile: dict[str, Any],
    timezone_evidence: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    transition = classify_paper_preflight(timezone_evidence)
    assert_preflight_not_promoted_to_paper_result(transition)
    run_id = str(state.data["run_id"])
    source_semantics_path = root / "configs/source_semantics.yaml"
    if not source_semantics_path.is_file():
        raise FileNotFoundError(source_semantics_path)

    sources = _nlr_source_records(
        root,
        nlr295_profile=nlr295_profile,
        nlr300_power_profile=nlr300_power_profile,
        nlr300_weather_profile=nlr300_weather_profile,
    )
    xcel_records, xcel_provenance = _ingest_xcel_registry(root)
    srrl_records, srrl_provenance = _ingest_srrl_provenance(root)
    nasa_records, nasa_provenance = _ingest_nasa_provenance(root)
    nyiso_records, nyiso_provenance = _ingest_nyiso_checkpoint(root)
    referenced_data_progress = _ingest_referenced_data_progress(root)
    sources.extend(xcel_records)
    sources.extend(srrl_records)
    sources.extend(nasa_records)
    sources.extend(nyiso_records)

    assumption_record = next(
        (
            item
            for item in timezone_evidence.get("evidence", [])
            if isinstance(item, dict) and item.get("evidence_role") == "assumed"
        ),
        {},
    )
    sensitivity = {
        "status": "required_before_paper_claims",
        "primary_policy": timezone_evidence.get("policy"),
        "source_clock_assumption": timezone_evidence.get(
            "source_clock_assumption", timezone_evidence.get("resolved_timezone")
        ),
        "analysis_timezone": timezone_evidence.get(
            "analysis_timezone", timezone_evidence.get("resolved_timezone")
        ),
        "evidence_role": "assumed",
        "publisher_encoding_confirmed": False,
        "source_timestamp_preserved": True,
        "dst_semantics": timezone_evidence.get("dst_semantics"),
        "alternative_source_clock": "fixed_MST_UTC-07:00",
        "comparison_scope": [
            "source_overlap",
            "held_out_physical_twin_metrics",
            "primary_pooled_economic_endpoint",
        ],
        "direct_local_wall_clock_interpretation": "structurally_invalid_not_run",
        "reason": (
            "user authorization clears preflight but does not establish the publisher's storage "
            "encoding; paper claims require a declared alternative-encoding sensitivity"
        ),
    }
    timezone_manifest_path = root / "artifacts/manifests/timezone_manifest.json"
    timezone_manifest = {
        "profile": "paper",
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": "passed_with_user_authorized_assumption",
        "evidence_role": "assumed",
        "publisher_metadata_claimed": False,
        "publisher_timezone_evidence": timezone_evidence.get("publisher_evidence", []),
        "source_timestamp_column": "ts",
        "source_timestamp_preserved": True,
        "source_clock_assumption": timezone_evidence.get("source_clock_assumption"),
        "derived_local_timezone": timezone_evidence.get(
            "analysis_timezone", timezone_evidence.get("resolved_timezone")
        ),
        "timestamp_utc_rule": assumption_record.get("timestamp_utc_rule"),
        "timestamp_local_rule": assumption_record.get("timestamp_local_rule"),
        "diagnostic_caveat": (
            "The raw naive clock includes patterns incompatible with direct America/Denver wall-time "
            "localization and its diurnal timing is more consistent with a UTC source clock. This is "
            "diagnostic support for the declared assumption, not publisher confirmation."
        ),
        "paper_claim_requirement": "alternative source-clock sensitivity required",
        "alternative_source_clock": "fixed_MST_UTC-07:00",
        "direct_local_wall_clock_interpretation": "structurally_invalid_not_run",
        "timezone_gate": timezone_evidence,
    }
    atomic_json(timezone_manifest_path, timezone_manifest)
    source_manifest = {
        "profile": "paper",
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": PAPER_PREFLIGHT_READY,
        "terminal": False,
        "source_policy": "actual_local_files_with_official_stable_urls_only",
        "source_relationship_policy": SOURCE_RELATIONSHIP_POLICY,
        "synthetic_fallback": False,
        "paper_results_available": False,
        "paper_training_started": False,
        "environment": environment,
        "code_tree_sha256": environment["code_tree_hash"],
        "configuration": config_record,
        "source_semantics": _file_record(root, source_semantics_path),
        "governing_contracts": contracts,
        "demo_reuse": demo_evidence,
        "sources": sources,
        "source_status": {
            "nlr295": {"status": "present_profiled", "mandatory": True},
            "nlr300": {
                "status": "present_profiled_timezone_assumption_passed",
                "mandatory": True,
                "source_clock_assumption": timezone_evidence.get(
                    "source_clock_assumption", timezone_evidence.get("resolved_timezone")
                ),
                "analysis_timezone": timezone_evidence.get(
                    "analysis_timezone", timezone_evidence.get("resolved_timezone")
                ),
                "evidence_role": timezone_evidence.get("evidence_role"),
            },
            "xcel": {"status": xcel_provenance["status"], "source_count": len(xcel_records)},
            "srrl": {"status": srrl_provenance["status"], "source_count": len(srrl_records)},
            "nasa_power": {
                "status": nasa_provenance["status"],
                "source_count": len(nasa_records),
            },
            "nyiso": {"status": nyiso_provenance["status"], "source_count": len(nyiso_records)},
            "referenced_data_panels": referenced_data_progress,
        },
        "persisted_provenance": {
            "xcel": xcel_provenance,
            "srrl": srrl_provenance,
            "nasa_power": nasa_provenance,
            "nyiso": nyiso_provenance,
            "referenced_data_panels": referenced_data_progress,
        },
        "profiles": {
            "nlr295_archive": nlr295_profile,
            "nlr300_power": nlr300_power_profile,
            "nlr300_weather": nlr300_weather_profile,
        },
        "mandatory_gates": {"nlr300_timezone": timezone_evidence},
        "timezone_manifest": _file_record(root, timezone_manifest_path),
        "time_encoding_sensitivity": sensitivity,
        "blockers": [],
    }
    _assert_manifest_has_stable_urls_only(source_manifest)
    source_manifest_path = root / "artifacts/manifests/source_manifest.json"
    atomic_json(source_manifest_path, source_manifest)
    ready_outputs = write_ready_paper_contract(
        root,
        run_id=run_id,
        timezone_gate=timezone_evidence,
    )
    report_path = _write_ready_report(root, timezone_evidence, referenced_data_progress)
    blockers_path = root / "artifacts/report/blockers.json"
    atomic_json(
        blockers_path,
        {
            "run_id": run_id,
            "status": PAPER_PREFLIGHT_READY,
            "terminal": False,
            "paper_results_available": False,
            "blockers": [],
        },
    )
    fingerprint = sha256_json(
        {
            "stage": "paper_preflight",
            "code_tree_sha256": environment["code_tree_hash"],
            "configuration": config_record,
            "source_semantics": _file_record(root, source_semantics_path),
            "contracts": contracts,
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "demo_run_manifest_sha256": demo_evidence["run_manifest"]["sha256"],
            "timezone_gate": timezone_evidence,
        }
    )
    finished_at = utc_now()
    _write_ready_state(
        root,
        state,
        fingerprint=fingerprint,
        timezone_gate=timezone_evidence,
        referenced_data_progress=referenced_data_progress,
        started_at=started_at,
        finished_at=finished_at,
    )
    run_manifest_path = build_run_manifest(
        root,
        root / "artifacts",
        {
            "profile": "paper",
            "run_id": run_id,
            "status": PAPER_PREFLIGHT_READY,
            "terminal": False,
            "paper_results_available": False,
            "paper_training_started": False,
            "synthetic_fallback": False,
            "demo_reused_without_rerun": True,
            "demo_metrics_reused": False,
            "environment": environment,
            "code_tree_sha256": environment["code_tree_hash"],
            "configuration": config_record,
            "governing_contracts": contracts,
            "source_manifest": _file_record(root, source_manifest_path),
            "timezone_manifest": _file_record(root, timezone_manifest_path),
            "referenced_data_progress": referenced_data_progress,
            "source_relationship_policy": SOURCE_RELATIONSHIP_POLICY,
            "timezone_gate": timezone_evidence,
            "time_encoding_sensitivity": sensitivity,
            "blockers": [],
        },
    )
    expected = set(_paper_output_paths(root))
    expected.update(ready_outputs)
    expected.update((report_path, blockers_path, run_manifest_path, timezone_manifest_path))
    missing = [path.as_posix() for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"paper preflight-ready contract is incomplete: {missing}")
    return {
        **transition,
        "synthetic_fallback": False,
        "demo_reused_without_rerun": True,
        "demo_metrics_reused": False,
        "blockers": [],
        "timezone_gate": timezone_evidence,
        "time_encoding_sensitivity": sensitivity,
        "source_manifest": source_manifest_path,
        "run_manifest": run_manifest_path,
    }


def run_paper(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    state: PipelineState,
    demo: dict[str, Any],
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run real-source gates and, by default, continue through paper execution.

    ``preflight_only`` exists for the focused gate contract test.  The public
    pipeline never sets it and therefore cannot stop at readiness while safe
    paper work remains.
    """
    root = root.resolve()
    config_path = config_path.resolve()
    if not preflight_only:
        terminal_result = _existing_terminal_result(root, state)
        if terminal_result is not None:
            return terminal_result
        resumed_preflight = _existing_execution_preflight(root, config, state)
        if resumed_preflight is not None:
            from dcvpp.paper_execution import run_paper_execution

            return run_paper_execution(root, config, state, resumed_preflight)
    started_at = utc_now()
    run_id = str(state.data["run_id"])
    demo_evidence = _demo_reuse_evidence(root, demo)
    environment = environment_snapshot(root)
    config_record = _file_record(root, config_path)
    contracts = _contract_records(root, config)

    raw_root = root / "data/raw"
    nlr295_archive = raw_root / "nlr295/esif.hpc.eagle.job-anon-energy-metrics.zip"
    nlr300_power = raw_root / "nlr300/esif.influx.buildingData.PUE.combined.parquet"
    nlr300_weather = raw_root / "nlr300/esif.influx.buildingData.outside.combined.parquet"
    nlr300_readme = raw_root / "nlr300/README.md"

    nlr295_profile = profile_nlr295_archive(nlr295_archive, parquet_sample_count=3)
    nlr300_power_profile = profile_nlr300_parquet(nlr300_power)
    nlr300_weather_profile = profile_nlr300_parquet(nlr300_weather)
    timezone_evidence = evaluate_nlr300_timezone(
        [nlr300_power, nlr300_weather],
        [nlr300_readme],
        **_nlr300_timezone_options(config),
    )
    transition = classify_paper_preflight(timezone_evidence)
    if transition["status"] == PAPER_PREFLIGHT_READY:
        preflight_result = _run_ready_paper_preflight(
            root,
            config_record=config_record,
            contracts=contracts,
            state=state,
            demo_evidence=demo_evidence,
            environment=environment,
            nlr295_profile=nlr295_profile,
            nlr300_power_profile=nlr300_power_profile,
            nlr300_weather_profile=nlr300_weather_profile,
            timezone_evidence=timezone_evidence,
            started_at=started_at,
        )
        if preflight_only:
            return preflight_result
        from dcvpp.paper_execution import run_paper_execution

        return run_paper_execution(root, config, state, preflight_result)
    if timezone_evidence.get("status") != "blocked" or timezone_evidence.get("code") != NLR300_TIMEZONE_BLOCKER:
        raise RuntimeError("this terminal branch requires the observed NLR300 timezone blocker")

    blockers = [
        {
            "code": NLR300_TIMEZONE_BLOCKER,
            "severity": "fatal",
            "mandatory": True,
            "source": "NLR300",
            "stage": "paper_evidence_gate",
            "message": (
                "NLR300 ts fields are timezone-naive and neither the Parquet metadata nor the "
                "official README states an authoritative timezone; physical calibration and all "
                "downstream paper training are blocked before fitting."
            ),
            "downstream_not_run": [
                "physical_calibration",
                "digital_twin_transfer",
                "paper_forecasting",
                "scenario_generation",
                "counterfactual_optimization",
                "paper_metrics",
            ],
            "location_inference_used": False,
            "synthetic_fallback_used": False,
        }
    ]

    sources = _nlr_source_records(
        root,
        nlr295_profile=nlr295_profile,
        nlr300_power_profile=nlr300_power_profile,
        nlr300_weather_profile=nlr300_weather_profile,
    )
    xcel_records, xcel_provenance = _ingest_xcel_registry(root)
    srrl_records, srrl_provenance = _ingest_srrl_provenance(root)
    nasa_records, nasa_provenance = _ingest_nasa_provenance(root)
    nyiso_records, nyiso_provenance = _ingest_nyiso_checkpoint(root)
    sources.extend(xcel_records)
    sources.extend(srrl_records)
    sources.extend(nasa_records)
    sources.extend(nyiso_records)

    source_manifest = {
        "profile": "paper",
        "run_id": run_id,
        "generated_at": utc_now(),
        "terminal_status": TERMINAL_STATUS,
        "source_policy": "actual_local_files_with_official_stable_urls_only",
        "synthetic_fallback": False,
        "paper_training_started": False,
        "environment": environment,
        "code_tree_sha256": environment["code_tree_hash"],
        "configuration": config_record,
        "governing_contracts": contracts,
        "demo_reuse": demo_evidence,
        "sources": sources,
        "source_status": {
            "nlr295": {"status": "present_profiled", "mandatory": True},
            "nlr300": {
                "status": "present_profiled_timezone_blocked",
                "mandatory": True,
            },
            "xcel": {
                "status": xcel_provenance["status"],
                "source_count": len(xcel_records),
                "blocking_scope": "local_subcase_only",
            },
            "srrl": {
                "status": srrl_provenance["status"],
                "source_count": len(srrl_records),
                "downstream_use": "not_reached_due_nlr300_gate",
            },
            "nasa_power": {
                "status": nasa_provenance["status"],
                "source_count": len(nasa_records),
                "downstream_use": "not_reached_due_nlr300_gate",
            },
            "nyiso": {
                "status": nyiso_provenance["status"],
                "source_count": len(nyiso_records),
                "downstream_use": "not_reached_due_nlr300_gate",
            },
        },
        "persisted_provenance": {
            "xcel": xcel_provenance,
            "srrl": srrl_provenance,
            "nasa_power": nasa_provenance,
            "nyiso": nyiso_provenance,
        },
        "profiles": {
            "nlr295_archive": nlr295_profile,
            "nlr300_power": nlr300_power_profile,
            "nlr300_weather": nlr300_weather_profile,
        },
        "mandatory_gates": {"nlr300_authoritative_timezone": timezone_evidence},
        "blockers": blockers,
    }
    _assert_manifest_has_stable_urls_only(source_manifest)
    source_manifest_path = root / "artifacts/manifests/source_manifest.json"
    atomic_json(source_manifest_path, source_manifest)

    blocked_outputs = write_blocked_paper_contract(root, run_id=run_id, blockers=blockers)
    completed = [
        "Existing clean demo reused as an engineering smoke test; no demo stage or metric was rerun or promoted.",
        "Real local NLR295 archive checksum and sampled schema profile completed.",
        "Real local NLR300 schema, time-range, null, and timezone metadata profiles completed.",
    ]
    if nyiso_provenance.get("archive_count"):
        completed.append(
            f"Official NYISO archive acquisition checkpoint ingested: {nyiso_provenance['archive_count']} archives, "
            f"{nyiso_provenance.get('failure_count', 0)} recorded download failures."
        )
    if nasa_provenance.get("annual_record_count"):
        completed.append(
            f"NASA POWER annual provenance ingested for {nasa_provenance['annual_record_count']} zone-year resources."
        )
    if xcel_provenance.get("status") == "XCEL_PUBLIC_DATA_READY":
        completed.append("Current official Xcel tariff PDF and the dated CPP rule snapshot were acquired and hashed.")
    if srrl_provenance.get("record_count"):
        completed.append("The official SRRL fixed-MST daily adapter was exercised on a persisted real-data sample.")
    completed.append("Mandatory evidence gate evaluated before any paper model fitting or optimization.")
    report_path = build_blocked_report(
        root,
        TERMINAL_STATUS,
        blockers,
        completed,
    )
    blockers_path = root / "artifacts/report/blockers.json"
    atomic_json(
        blockers_path,
        {
            "run_id": run_id,
            "terminal_status": TERMINAL_STATUS,
            "paper_results_available": False,
            "paper_training_started": False,
            "synthetic_fallback": False,
            "blockers": blockers,
        },
    )

    fingerprint = sha256_json(
        {
            "stage": "paper_preflight",
            "code_tree_sha256": environment["code_tree_hash"],
            "configuration": config_record,
            "contracts": contracts,
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "demo_run_manifest_sha256": demo_evidence["run_manifest"]["sha256"],
            "blocker_codes": [item["code"] for item in blockers],
        }
    )
    finished_at = utc_now()
    _write_terminal_state(
        root,
        state,
        fingerprint=fingerprint,
        blockers=blockers,
        started_at=started_at,
        finished_at=finished_at,
    )

    run_manifest_path = build_run_manifest(
        root,
        root / "artifacts",
        {
            "profile": "paper",
            "run_id": run_id,
            "status": TERMINAL_STATUS,
            "terminal": True,
            "paper_results_available": False,
            "paper_training_started": False,
            "synthetic_fallback": False,
            "demo_reused_without_rerun": True,
            "demo_metrics_reused": False,
            "environment": environment,
            "code_tree_sha256": environment["code_tree_hash"],
            "configuration": config_record,
            "governing_contracts": contracts,
            "source_manifest": _file_record(root, source_manifest_path),
            "blockers": blockers,
        },
    )

    expected = set(_paper_output_paths(root))
    expected.update(blocked_outputs)
    expected.update((report_path, blockers_path, run_manifest_path))
    missing = [path.as_posix() for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"paper terminal contract is incomplete: {missing}")
    return {
        "status": TERMINAL_STATUS,
        "terminal": True,
        "paper_results_available": False,
        "paper_training_started": False,
        "synthetic_fallback": False,
        "demo_reused_without_rerun": True,
        "demo_metrics_reused": False,
        "blockers": blockers,
        "source_manifest": source_manifest_path,
        "run_manifest": run_manifest_path,
    }
