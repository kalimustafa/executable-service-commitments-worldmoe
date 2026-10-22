"""Fast, evidence-preserving finalization of the admitted partial DART OOS run.

The normal collector is intentionally exhaustive, but it opens the same 400 MB
DART archive once for each DA and RT unit.  This module is a post-execution
finalizer: it never trains, predicts, samples, or rewrites an existing model
artifact.  It validates every prediction and every *unique* DART bundle once,
in parallel, then emits the same strict partial-stage contract.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_partial_candidate import (
    ELIGIBLE_CANDIDATES,
    PARTIAL_CANDIDATE_STATUS,
    PARTIAL_CANDIDATE_TERMINAL_CAP,
    SELECTION_ELIGIBLE_UNIT_COUNT,
    SELECTION_EXECUTION_UNIT_COUNT,
    _event_date_contract,
    build_partial_candidate_pool_contract,
    eligible_units_for_contract,
    failed_units_from_scheduler_manifest,
    selection_execution_units_for_contract,
    validate_partial_candidate_pool_contract,
)


class FastPartialFinalizeError(RuntimeError):
    """Raised when a persisted selection artifact fails its frozen contract."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FastPartialFinalizeError(f"JSON root is not a mapping: {path}")
    return value


def _prediction_record(producer: Any, unit: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = producer._checkpoint_path(unit)
    checkpoint = _load_json(checkpoint_path)
    if checkpoint.get("status") != "COMPLETE":
        raise FastPartialFinalizeError(f"eligible unit is incomplete: {unit.unit_id}")
    expected_request = producer._request_fingerprint(unit)
    if checkpoint.get("request_fingerprint") != expected_request:
        raise FastPartialFinalizeError(f"unit request changed: {unit.unit_id}")
    artifact = checkpoint.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        raise FastPartialFinalizeError(f"prediction ledger is absent: {unit.unit_id}")
    path = Path(str(artifact.get("path", "")))
    expected_sha = str(artifact.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise FastPartialFinalizeError(f"prediction bytes changed: {unit.unit_id}")
    prediction = producer._load_prediction(path, expected_request=expected_request)
    coverage = _event_date_contract(prediction, unit.origin.forecast_year)
    return (
        {
            "unit_id": unit.unit_id,
            "path": path.as_posix(),
            "sha256": expected_sha,
            "forecast_year": unit.origin.forecast_year,
            "target": unit.target.name,
            "candidate": unit.candidate.name,
            "seed_slot": unit.seed_slot.name,
            "coverage_sha256": sha256_json(coverage),
        },
        coverage,
    )


def _dart_record(producer: Any, unit: Any) -> dict[str, str]:
    path = producer._dart_path(unit)
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise FastPartialFinalizeError(f"paired DART artifact is absent: {unit.unit_id}")
    manifest = _load_json(manifest_path)
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise FastPartialFinalizeError(f"DART contract is malformed: {path}")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("contract_fingerprint") != sha256_json(contract)
        or manifest.get("sign_gate_after_generation") != "PASSED"
        or manifest.get("sign_gate_after_serialization") != "PASSED"
        or manifest.get("ambiguous_rt_spread_present") is not False
        or manifest.get("joint_coupling_provenance_verified") is not True
        or contract.get("formula") != "draw-wise da_lbmp - rt_lbmp"
        or contract.get("inverse_formula") != "rt_lbmp - da_lbmp"
        or contract.get("locked_targets_used_for_coupling") is not False
        or contract.get("origin") != unit.origin.name
        or contract.get("candidate") != unit.candidate.name
        or contract.get("seed_slot") != unit.seed_slot.name
    ):
        raise FastPartialFinalizeError(f"DART sign/provenance gate changed: {path}")
    if sha256_file(path) != manifest.get("artifact_sha256"):
        raise FastPartialFinalizeError(f"DART bytes changed: {path}")
    # Inspect the central directory only.  The arrays were already round-trip
    # checked immediately after generation; re-inflating 20+ GB here is not a
    # stronger provenance check than the pinned byte hash.
    required = {
        "event_keys.npy",
        "da_lbmp_draws.npy",
        "rt_lbmp_draws.npy",
        "dart_da_minus_rt.npy",
        "rt_spread_rt_minus_da.npy",
        "common_base_random_numbers_sha256.npy",
        "joint_uniforms_sha256.npy",
        "coupling_manifest_sha256.npy",
    }
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if not required.issubset(names) or "rt_spread.npy" in names:
            raise FastPartialFinalizeError(f"DART ZIP schema is incomplete: {path}")
    return {"path": path.as_posix(), "sha256": str(manifest["artifact_sha256"])}


def _fast_archive_inventory(stage_root: Path, repository_root: Path) -> dict[str, Any]:
    manifests = sorted(
        (stage_root / "checkpoint_groups").rglob(
            "epoch_archive/epoch_archive_manifest.json"
        )
    )
    if len(manifests) != 108:
        raise FastPartialFinalizeError("v4 epoch archive does not contain 108 tracks")
    records: list[dict[str, Any]] = []
    for path in manifests:
        value = _load_json(path)
        core = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if (
            value.get("status") != "COMPLETE_THROUGH_CURRENT_EPOCH"
            or value.get("manifest_sha256") != sha256_json(core)
            or value.get("every_completed_epoch_full_state_persisted") is not True
            or value.get("formal_selection_candidates_divisible_by_five_only") is not True
        ):
            raise FastPartialFinalizeError(f"epoch archive contract changed: {path}")
        epochs = [int(item["epoch"]) for item in value.get("records", ())]
        predecessor = int(value.get("predecessor_imported_epoch", -1))
        if epochs and (
            epochs[0] != predecessor + 1
            or epochs != list(range(epochs[0], epochs[-1] + 1))
        ):
            raise FastPartialFinalizeError(f"epoch archive is not contiguous: {path}")
        for item in value.get("records", ()):
            checkpoint = Path(str(item.get("path", "")))
            if not checkpoint.is_absolute():
                checkpoint = repository_root / checkpoint
            checkpoint = checkpoint.resolve()
            if repository_root not in checkpoint.parents:
                raise FastPartialFinalizeError("epoch checkpoint escapes repository")
            if (
                not checkpoint.is_file()
                or checkpoint.stat().st_size != int(item.get("byte_count", -1))
                or bool(item.get("formal_selection_candidate"))
                != (int(item.get("epoch", -1)) > 0 and int(item["epoch"]) % 5 == 0)
            ):
                raise FastPartialFinalizeError(f"epoch checkpoint role/size changed: {checkpoint}")
        records.append(
            {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
                "track_identity": value["track_identity"],
                "first_epoch": value.get("first_epoch"),
                "latest_epoch": value.get("latest_epoch"),
                "record_count": value.get("record_count"),
                "predecessor_imported_epoch": predecessor,
            }
        )
    core = {
        "schema_version": "dcvpp.dart_v4_full_epoch_archive_inventory.v1",
        "status": "COMPLETE",
        "track_count": len(records),
        "track_manifests": records,
        "track_manifests_sha256": sha256_json(records),
        "every_completed_epoch_full_state_persisted": True,
        "formal_selection_candidates_divisible_by_five_only": True,
        "validation_profile": "manifest_hash_continuity_role_and_checkpoint_size",
    }
    inventory = {**core, "manifest_sha256": sha256_json(core)}
    path = stage_root / "recovery_overlay/v4_epoch_archive_inventory.json"
    atomic_json(path, inventory)
    return {**inventory, "path": path.as_posix(), "sha256": sha256_file(path)}


def fast_finalize_partial_selection(
    *, producer: Any, stage_manifest_path: Path, scheduler_manifest_path: Path,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Finalize the persisted 123+3 selection without executing a model."""

    raw = _load_json(Path(stage_manifest_path))
    if (
        raw.get("phase") != "selection"
        or raw.get("status") != "PARTIAL"
        or raw.get("downstream_ready") is not False
        or int(raw.get("expected_unit_count", -1)) != 126
        or int(raw.get("completed_unit_count", -1)) != 123
        or int(raw.get("failed_unit_count", -1)) != 3
    ):
        raise FastPartialFinalizeError("raw selection is not the saved 123+3 state")
    scheduler_path = Path(scheduler_manifest_path)
    scheduler = _load_json(scheduler_path)
    failed_units = failed_units_from_scheduler_manifest(scheduler)
    contract = build_partial_candidate_pool_contract()
    validate_partial_candidate_pool_contract(
        contract, failed_units=failed_units, require_failed_units=True
    )
    all_units = tuple(producer.plan.work_units(role="selection_oos"))
    executable = selection_execution_units_for_contract(
        all_units, contract, failed_units=failed_units
    )
    if set(map(str, raw.get("completed_units", ()))) != {
        item.unit_id for item in executable
    }:
        raise FastPartialFinalizeError("raw completed-unit panel changed")
    if {str(item.get("unit_id")) for item in raw.get("failed_units", ())} != {
        str(item["unit_id"]) for item in failed_units
    }:
        raise FastPartialFinalizeError("raw failed-unit panel changed")

    eligible = eligible_units_for_contract(all_units, contract)
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 16))) as pool:
        pairs = list(pool.map(lambda item: _prediction_record(producer, item), eligible))
    prediction_records = [item[0] for item in pairs]
    coverage_by_unit = {
        unit.unit_id: pair[1] for unit, pair in zip(eligible, pairs, strict=True)
    }
    references: dict[tuple[int, str], dict[str, Any]] = {}
    for unit, pair in zip(eligible, pairs, strict=True):
        key = (unit.origin.forecast_year, unit.target.name)
        if key in references and references[key] != pair[1]:
            raise FastPartialFinalizeError("eligible candidates use different date panels")
        references[key] = pair[1]
    expected_counts = Counter(
        {
            candidate: 3 * 2 * (3 if candidate not in {"seasonal_naive", "persistence", "lightgbm_quantile"} else 1)
            for candidate in ELIGIBLE_CANDIDATES
        }
    )
    if Counter(item["candidate"] for item in prediction_records) != expected_counts:
        raise FastPartialFinalizeError("eligible candidate coverage is incomplete")

    unique_units: dict[str, Any] = {}
    for unit in eligible:
        unique_units.setdefault(producer._dart_path(unit).as_posix(), unit)
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 8))) as pool:
        dart_records = list(pool.map(lambda item: _dart_record(producer, item), unique_units.values()))

    info: dict[str, str] = {}
    uniforms: dict[str, str] = {}
    coupling: dict[str, str] = {}
    for unit in eligible:
        info_path = producer._information_manifest_path(unit.origin)
        _, uniform_path = producer._uniform_paths(unit.origin)
        _, coupling_path = producer._joint_uniform_paths(unit)
        info[unit.origin.name] = sha256_file(info_path)
        uniforms[unit.origin.name] = sha256_file(uniform_path)
        coupling[f"{unit.origin.name}/{unit.candidate.name}/{unit.seed_slot.name}"] = sha256_file(coupling_path)
    evidence = {
        "candidate_pool_contract": contract,
        "role": "selection_oos",
        "eligible_unit_count": len(eligible),
        "prediction_artifacts": sorted(prediction_records, key=lambda item: item["unit_id"]),
        "dart_artifacts": sorted(dart_records, key=lambda item: item["path"]),
        "common_information_manifest_sha256": dict(sorted(info.items())),
        "common_uniform_manifest_sha256": dict(sorted(uniforms.items())),
        "joint_coupling_manifest_sha256": dict(sorted(coupling.items())),
        "coverage_contract": {
            "required_selection_years": [2022, 2023, 2024],
            "required_targets": ["da_lbmp", "rt_lbmp_at_da_gate"],
            "required_study_seeds": [20260809, 20260819, 20260829],
            "eligible_candidates": list(ELIGIBLE_CANDIDATES),
            "date_drop_count": 0,
            "common_date_panel_preserved": True,
            "unit_coverage_sha256": sha256_json(coverage_by_unit),
            "year_target_reference_sha256": sha256_json(
                {f"{year}/{target}": value for (year, target), value in sorted(references.items())}
            ),
        },
        "finalization_profile": "parallel_single_pass_manifest_and_byte_hash",
    }
    evidence["evidence_sha256"] = sha256_json(evidence)
    repository_root = Path(str(raw["repository_root"])).resolve()
    inventory = _fast_archive_inventory(producer.output_root, repository_root)
    finalized = {
        **raw,
        "status": PARTIAL_CANDIDATE_STATUS,
        "downstream_ready": False,
        "partial_candidate_downstream_ready": True,
        "terminal_status_cap": PARTIAL_CANDIDATE_TERMINAL_CAP,
        "completed_unit_count": SELECTION_EXECUTION_UNIT_COUNT,
        "failed_unit_count": 3,
        "failed_units": failed_units,
        "eligible_expected_unit_count": SELECTION_ELIGIBLE_UNIT_COUNT,
        "eligible_completed_unit_count": SELECTION_ELIGIBLE_UNIT_COUNT,
        "globally_excluded_completed_unit_count": 15,
        "candidate_pool_contract": contract,
        "date_drop_count": 0,
        "common_date_panel_preserved": True,
        "v4_full_epoch_archive_inventory": inventory,
        "scheduler_scientific_status": scheduler["scheduler_scientific_status"],
        "scheduler_manifest_path": scheduler_path.as_posix(),
        "scheduler_manifest_sha256": sha256_file(scheduler_path),
        "phase_evidence": evidence,
        "fast_finalize_code_sha256": sha256_file(Path(__file__)),
        "finished_at_utc": utc_now(),
    }
    finalized.pop("manifest_fingerprint", None)
    finalized["manifest_fingerprint"] = sha256_json(finalized)
    atomic_json(Path(stage_manifest_path), finalized)
    return finalized


__all__ = ["FastPartialFinalizeError", "fast_finalize_partial_selection"]
