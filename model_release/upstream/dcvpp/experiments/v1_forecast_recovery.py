from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dcvpp.core import atomic_json, atomic_text, sha256_file, sha256_json, utc_now


SCHEMA_VERSION = "paper_forecast_v1_recovery_manifest/v1"
LEDGER_SCHEMA_VERSION = "paper_forecast_v1_corruption_ledger/v1"
PREFLIGHT_SCHEMA_VERSION = "paper_forecast_v1_recovery_preflight/v1"
WORKLOAD_SEQUENCE_BUNDLE_SCHEMA_VERSION = (
    "paper_forecast_v1_workload_sequence_bundle/v1"
)
WORKLOAD_NATIVE_FAILURE_SCHEMA_VERSION = (
    "paper_forecast_v1_workload_native_failure/v1"
)
WORKLOAD_TRAINING_SMOKE_SCHEMA_VERSION = (
    "paper_forecast_v1_workload_training_isolation_smoke/v1"
)
RUN_ID = "paper-forecast-v1-recovery-v1"
DA_COMPONENT_RUN_ID = "paper-forecast-v1-da-recovery-v1"
WORKLOAD_COMPONENT_RUN_ID = "paper-forecast-v1-workload-recovery-v1"
INCIDENT_ID = "paper-forecast-v1-erroneous-restart-20260810"
INCIDENT_START_UTC = datetime(2026, 8, 10, 13, 55, tzinfo=timezone.utc)
INCIDENT_END_UTC = datetime(2026, 8, 10, 14, 18, tzinfo=timezone.utc)
STUDY_SEEDS = (20260809, 20260819, 20260829, 20260908)
LEGACY_EXCLUDED_SEED = 20260918
DA_NEURAL_MODELS = (
    "lstm",
    "dlinear",
    "patchtst",
    "itransformer",
    "static_graph_patch_transformer",
)
DA_ADVANCED_MODEL = "dgpt_e"
WORKLOAD_HISTORY = 672
WORKLOAD_HORIZON = 96
WORKLOAD_COHORTS = 4
WORKLOAD_AGGREGATE_FEATURES = 12
WORKLOAD_EVENT_FEATURES = 14
WORKLOAD_SEQUENCE_SPLIT_SIZES = {
    "train": 1222,
    "validation": 409,
    "locked_test": 409,
}
EXPECTED_CORRUPTION_COUNT = 28
EXPECTED_CHECKPOINT_CORRUPTION_COUNT = 26
ORIGINAL_CFC_CHECKPOINT_SHA256 = (
    "17775b0238ba4e8bc4f492d3b908c7b5e98388606ddd6646919a529d8290596b"
)
ORIGINAL_CFC_RESUME_FINGERPRINT = (
    "b78ac38cf7830255500595624de6b78a4900cc3ae1bc2651c81774c399062618"
)
KNOWN_INCOMPLETE_GRU_20260819_SHA256 = (
    "1f9d7913237f3d0269037ca09025359fb24b8e4566dc88a8970dfc78df627a19"
)
WORKLOAD_EVENTS_ARTIFACT_SHA256 = (
    "b1a90b0274e5bc2e2959a24ed0cb762ccffa1f347a9461e9ac9b133ab2923b1a"
)
WORKLOAD_EVENTS_CONTRACT_SHA256 = (
    "999d12d3223d968d96feca64fafe6f2e1f3fbeba6ffd6c096dcd54f89927e1cc"
)
PHYSICAL_RECOVERY_RELATIVE_PATH = (
    "artifacts/research/physical/run_id=paper-physical-v1/repairs/"
    "20260810T062256Z-timezone-climate-v1/recovery.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "artifacts/manifests/referenced_data_progress_manifest.json"
)

FROZEN_SOURCE_HASHES = {
    "src/dcvpp/experiments/real_forecasting.py": (
        "cd392312e6eaa1ddce5fd67a25d7ab6ff053c4aabbffb8fcb2bb3bc5dc3fa124"
    ),
    "src/dcvpp/experiments/forecast_common.py": (
        "c3defcd56c2a908686ac11cb115e3b81f452895e9879dd43852bbb6792ffef13"
    ),
    "src/dcvpp/experiments/forecast_calibration.py": (
        "ee37f0f59c9e7b6de6b814a2735f79afaba6544b036654d72232a75785431ac9"
    ),
    "src/dcvpp/experiments/paper_training.py": (
        "2da49a9dcab0961f18b29e4a79d7af558d0c6f3586866a25f80bc891a2512058"
    ),
    "src/dcvpp/experiments/nlr_workload.py": (
        "43910f8d023f82aac31de69538aab77f90dff4ae4d11442feae0cb8c3556f499"
    ),
    "src/dcvpp/experiments/torch_forecasts.py": (
        "ebfbb05014a246bed3d8e0d8cf822b5bd2adfd504c85e39b4e02294ad8557158"
    ),
    "src/dcvpp/experiments/forecast_day_factory.py": (
        "7281cd32da5b9fb5f4570216284e4577e70e0568a285c0060d6d65bdeacc6e98"
    ),
    "src/dcvpp/experiments/weather_pv.py": (
        "f245a4719278b8b805087f14ed8accc3f46df72b1479e13c851badc3cf8e2e32"
    ),
    "src/dcvpp/models/energy/dgpt_e.py": (
        "d248429f249df5ab715c7b026610eb06e611a95520757373c2ede172a8b7126e"
    ),
    "src/dcvpp/models/workload/cfc_w.py": (
        "4b3c2b2f28340d337bcb451fcb8e70d4a73639e479cd6224ce68003e4586ec91"
    ),
}


class V1ForecastRecoveryError(RuntimeError):
    pass


def recovery_root(repository_root: Path) -> Path:
    return (
        Path(repository_root).resolve()
        / "artifacts/real_forecasting_recovery"
        / f"run_id={RUN_ID}"
    )


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _canonical_output_path(repository_root: Path, raw_path: str) -> Path:
    root = Path(repository_root).resolve()
    relative = Path(raw_path)
    if relative.is_absolute():
        raise V1ForecastRecoveryError("canonical recovery output uses an absolute path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(recovery_root(root).resolve())
    except ValueError as exc:
        raise V1ForecastRecoveryError(
            f"canonical recovery output is outside the recovery run root: {raw_path}"
        ) from exc
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V1ForecastRecoveryError(f"JSON root is not an object: {path}")
    return value


def _utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _record_for_file(
    root: Path,
    path: Path,
    *,
    task: str,
    evidence_role: str,
    lineage_source_path: Path | None = None,
) -> dict[str, Any]:
    source = lineage_source_path or path
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "byte_count": int(path.stat().st_size),
        "task": task,
        "evidence_role": evidence_role,
        "lineage_source_path": _relative(root, source),
        "lineage_source_sha256": sha256_file(source),
    }


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _preserve_file(source: Path, destination: Path) -> str:
    """Create an immutable incident snapshot without changing the source."""

    expected = sha256_file(source)
    if destination.is_file():
        if sha256_file(destination) != expected:
            raise V1ForecastRecoveryError(
                f"existing preserved snapshot disagrees with source: {destination}"
            )
        return "existing_verified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        method = "copy"
    if sha256_file(destination) != expected:
        raise V1ForecastRecoveryError(
            f"preserved snapshot hash verification failed: {destination}"
        )
    return method


def _materialize_copy(source: Path, destination: Path) -> str:
    """Materialize a verified canonical recovery output, never a corrupt source."""

    expected = sha256_file(source)
    if destination.is_file():
        if sha256_file(destination) != expected:
            raise V1ForecastRecoveryError(
                f"canonical recovery output already exists with different bytes: {destination}"
            )
        return "existing_verified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        method = "copy"
    if sha256_file(destination) != expected:
        raise V1ForecastRecoveryError(
            f"canonical recovery copy hash verification failed: {destination}"
        )
    return method


def verify_frozen_sources(repository_root: Path) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    records: list[dict[str, Any]] = []
    for relative_path, expected in FROZEN_SOURCE_HASHES.items():
        path = root / relative_path
        if not path.is_file():
            raise V1ForecastRecoveryError(f"frozen v1 source is absent: {relative_path}")
        actual = sha256_file(path)
        verified = actual == expected
        records.append(
            {
                "path": relative_path,
                "sha256": actual,
                "expected_sha256": expected,
                "verified": verified,
            }
        )
        if not verified:
            raise V1ForecastRecoveryError(
                f"frozen v1 source hash changed: {relative_path}: {actual} != {expected}"
            )
    return records


def _validation_only_leaderboard(path: Path) -> tuple[dict[str, Any], str]:
    # This loader lexically redacts every locked_test_metrics value before
    # json.loads. Recovery inventory therefore cannot accidentally consume a
    # locked-2025 metric value.
    from dcvpp.experiments.v1_dart_readiness import (
        _read_validation_only_leaderboard,
    )

    return _read_validation_only_leaderboard(path)


def _leaderboard_checkpoint_contract(
    leaderboard: Mapping[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    baselines = leaderboard.get("deployable_baselines")
    if not isinstance(baselines, list):
        raise V1ForecastRecoveryError("DA leaderboard lacks deployable_baselines")
    for record in baselines:
        if isinstance(record, Mapping) and isinstance(record.get("model"), str):
            records[str(record["model"])] = record
    dgpt = leaderboard.get("dgpt_e")
    if not isinstance(dgpt, Mapping):
        raise V1ForecastRecoveryError("DA leaderboard lacks dgpt_e")
    records[DA_ADVANCED_MODEL] = dgpt
    expected_models = set(DA_NEURAL_MODELS) | {DA_ADVANCED_MODEL}
    if not expected_models.issubset(records):
        raise V1ForecastRecoveryError("DA leaderboard neural candidate set is incomplete")
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for model in (*DA_NEURAL_MODELS, DA_ADVANCED_MODEL):
        trials = records[model].get("seed_trials")
        if not isinstance(trials, list):
            raise V1ForecastRecoveryError(f"DA leaderboard lacks seed trials: {model}")
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise V1ForecastRecoveryError(f"invalid DA seed trial: {model}")
            seed = int(trial["seed"])
            expected_sha = str(trial.get("checkpoint_sha256", ""))
            expected_fingerprint = str(trial.get("resume_fingerprint", ""))
            if len(expected_sha) != 64 or len(expected_fingerprint) != 64:
                raise V1ForecastRecoveryError(
                    f"DA seed trial lacks checkpoint identity: {model}/{seed}"
                )
            output[(model, seed)] = {
                "expected_original_sha256": expected_sha,
                "expected_original_resume_fingerprint": expected_fingerprint,
                "status": str(trial.get("status", "")),
            }
    expected_keys = {
        (model, seed)
        for model in (*DA_NEURAL_MODELS, DA_ADVANCED_MODEL)
        for seed in STUDY_SEEDS
    }
    if set(output) != expected_keys:
        raise V1ForecastRecoveryError("DA leaderboard seed matrix is not exactly 6 x 5")
    return output


def build_corruption_ledger(
    repository_root: Path,
    *,
    preserve: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output_root = recovery_root(root)
    ledger_path = output_root / "incident/corruption_ledger.json"
    if ledger_path.is_file():
        existing = _load_json(ledger_path)
        validate_corruption_ledger(root, existing)
        return {
            **existing,
            "path": _relative(root, ledger_path),
            "sha256": sha256_file(ledger_path),
            "resumed": True,
        }

    v1_root = root / "artifacts/real_forecasting/paper-forecast-v1"
    leaderboard_path = v1_root / "metrics/nyiso_da_paper_energy_leaderboard.json"
    leaderboard, validation_content_sha = _validation_only_leaderboard(leaderboard_path)
    checkpoint_contract = _leaderboard_checkpoint_contract(leaderboard)
    records: list[dict[str, Any]] = []
    intact_dgpt: list[dict[str, Any]] = []

    for (model, seed), expected in checkpoint_contract.items():
        path = v1_root / "checkpoints/nyiso_da" / model / f"seed_{seed}_latest.pt"
        if not path.is_file():
            raise V1ForecastRecoveryError(f"expected DA checkpoint is absent: {path}")
        current_sha = sha256_file(path)
        mismatch = current_sha != expected["expected_original_sha256"]
        if model == DA_ADVANCED_MODEL and not mismatch:
            intact_dgpt.append(
                {
                    "path": _relative(root, path),
                    "sha256": current_sha,
                    "seed": seed,
                }
            )
            continue
        if not mismatch:
            raise V1ForecastRecoveryError(
                f"expected incident-overwritten checkpoint unexpectedly matches: {path}"
            )
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if not (INCIDENT_START_UTC <= mtime <= INCIDENT_END_UTC):
            raise V1ForecastRecoveryError(
                f"checkpoint mismatch is outside the incident window: {path}: {mtime}"
            )
        snapshot = (
            output_root
            / "incident/preserved_corrupt_files"
            / path.relative_to(v1_root)
        )
        method = _preserve_file(path, snapshot) if preserve else "not_preserved"
        preserved_sha = sha256_file(snapshot) if preserve else current_sha
        records.append(
            {
                "path": _relative(root, path),
                "artifact_kind": "torch_checkpoint",
                "task": "nyiso_da",
                "model": model,
                "seed": seed,
                "incident_mtime_utc": mtime.isoformat(),
                "current_sha256": current_sha,
                "expected_original_sha256": expected["expected_original_sha256"],
                "expected_original_resume_fingerprint": expected[
                    "expected_original_resume_fingerprint"
                ],
                "preserved_snapshot_path": _relative(root, snapshot),
                "preserved_snapshot_sha256": preserved_sha,
                "preservation_method": method,
                "corruption_evidence": "original_leaderboard_hash_mismatch_and_incident_window_mtime",
                "excluded_from_canonical_outputs": True,
            }
        )

    for relative_path, kind in (
        ("predictions/nyiso_da_predictions.parquet", "baseline_prediction_panel"),
        ("models/nyiso_da_models.joblib", "baseline_model_bundle"),
    ):
        path = v1_root / relative_path
        if not path.is_file():
            raise V1ForecastRecoveryError(f"incident DA output is absent: {path}")
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if not (INCIDENT_START_UTC <= mtime <= INCIDENT_END_UTC):
            raise V1ForecastRecoveryError(
                f"incident DA output mtime is outside the incident window: {path}: {mtime}"
            )
        current_sha = sha256_file(path)
        snapshot = output_root / "incident/preserved_corrupt_files" / path.relative_to(v1_root)
        method = _preserve_file(path, snapshot) if preserve else "not_preserved"
        records.append(
            {
                "path": _relative(root, path),
                "artifact_kind": kind,
                "task": "nyiso_da",
                "model": None,
                "seed": None,
                "incident_mtime_utc": mtime.isoformat(),
                "current_sha256": current_sha,
                "expected_original_sha256": None,
                "preserved_snapshot_path": _relative(root, snapshot),
                "preserved_snapshot_sha256": (
                    sha256_file(snapshot) if preserve else current_sha
                ),
                "preservation_method": method,
                "corruption_evidence": "incident_window_mtime_without_preincident_hash",
                "excluded_from_canonical_outputs": True,
            }
        )

    records.sort(key=lambda item: str(item["path"]))
    ledger = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "incident_id": INCIDENT_ID,
        "status": "COMPLETE",
        "incident_window_utc": {
            "start": INCIDENT_START_UTC.isoformat(),
            "end": INCIDENT_END_UTC.isoformat(),
        },
        "source_original_v1_root": _relative(root, v1_root),
        "source_original_da_leaderboard": {
            "path": _relative(root, leaderboard_path),
            "sha256": sha256_file(leaderboard_path),
            "validation_only_content_sha256": validation_content_sha,
            "locked_2025_metric_values_loaded": False,
        },
        "record_count": len(records),
        "records": records,
        "intact_da_dgpt_checkpoint_count": len(intact_dgpt),
        "intact_da_dgpt_checkpoints": sorted(
            intact_dgpt, key=lambda item: int(item["seed"])
        ),
        "preservation_mode": "copy_or_hardlink_without_source_mutation",
        "canonical_output_policy": "all_records_excluded_and_regenerated_in_isolated_recovery",
        "created_at_utc": utc_now(),
    }
    validate_corruption_ledger(root, ledger)
    atomic_json(ledger_path, ledger)
    return {
        **ledger,
        "path": _relative(root, ledger_path),
        "sha256": sha256_file(ledger_path),
        "resumed": False,
    }


def validate_corruption_ledger(
    repository_root: Path,
    ledger: Mapping[str, Any],
) -> None:
    root = Path(repository_root).resolve()
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise V1ForecastRecoveryError("corruption ledger schema is invalid")
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_CORRUPTION_COUNT:
        raise V1ForecastRecoveryError(
            f"corruption ledger must contain exactly {EXPECTED_CORRUPTION_COUNT} records"
        )
    if int(ledger.get("record_count", -1)) != EXPECTED_CORRUPTION_COUNT:
        raise V1ForecastRecoveryError("corruption ledger record_count is not 28")
    checkpoint_count = 0
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise V1ForecastRecoveryError("corruption ledger contains a non-object record")
        path_value = str(record.get("path", ""))
        if not path_value or path_value in seen:
            raise V1ForecastRecoveryError("corruption ledger paths are missing or duplicated")
        seen.add(path_value)
        source = root / path_value
        snapshot_value = str(record.get("preserved_snapshot_path", ""))
        snapshot = root / snapshot_value
        current_sha = str(record.get("current_sha256", ""))
        preserved_sha = str(record.get("preserved_snapshot_sha256", ""))
        if not source.is_file() or sha256_file(source) != current_sha:
            raise V1ForecastRecoveryError(f"corrupt source bytes changed: {path_value}")
        if not snapshot.is_file() or sha256_file(snapshot) != preserved_sha:
            raise V1ForecastRecoveryError(
                f"corrupt preserved snapshot is absent or changed: {snapshot_value}"
            )
        if current_sha != preserved_sha:
            raise V1ForecastRecoveryError(
                f"corrupt source and preserved snapshot hashes differ: {path_value}"
            )
        if record.get("excluded_from_canonical_outputs") is not True:
            raise V1ForecastRecoveryError(
                f"corrupt record is not excluded from canonical outputs: {path_value}"
            )
        if record.get("artifact_kind") == "torch_checkpoint":
            checkpoint_count += 1
            expected = record.get("expected_original_sha256")
            if not isinstance(expected, str) or len(expected) != 64 or expected == current_sha:
                raise V1ForecastRecoveryError(
                    f"checkpoint original hash evidence is invalid: {path_value}"
                )
    if checkpoint_count != EXPECTED_CHECKPOINT_CORRUPTION_COUNT:
        raise V1ForecastRecoveryError("corruption ledger must contain exactly 26 checkpoints")


def verify_physical_recovery_contract(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    recovery_path = root / PHYSICAL_RECOVERY_RELATIVE_PATH
    if not recovery_path.is_file():
        raise V1ForecastRecoveryError("physical repair recovery.json is absent")
    recovery = _load_json(recovery_path)
    if recovery.get("status") != "completed" or recovery.get("run_id") != "paper-physical-v1":
        raise V1ForecastRecoveryError("physical recovery is not a completed paper-physical-v1 repair")
    targets = recovery.get("targets")
    before_hashes = recovery.get("before_hashes")
    after_hashes = recovery.get("after_hashes")
    if not isinstance(targets, Mapping) or not isinstance(before_hashes, Mapping) or not isinstance(after_hashes, Mapping):
        raise V1ForecastRecoveryError("physical recovery target/hash ledgers are incomplete")
    branch_target = targets.get("branch_manifest")
    if not isinstance(branch_target, Mapping):
        raise V1ForecastRecoveryError("physical recovery lacks branch-manifest snapshot")
    snapshot_path = root / str(branch_target.get("backup_path", ""))
    snapshot_expected = str(branch_target.get("sha256", ""))
    if (
        not snapshot_path.is_file()
        or sha256_file(snapshot_path) != snapshot_expected
        or str(before_hashes.get("branch_manifest", "")) != snapshot_expected
    ):
        raise V1ForecastRecoveryError("completed physical branch snapshot does not hash-verify")
    snapshot = _load_json(snapshot_path)
    result = snapshot.get("result")
    if (
        snapshot.get("status") != "completed"
        or not isinstance(result, Mapping)
        or result.get("status") != "completed"
        or result.get("run_id") != "paper-physical-v1"
    ):
        raise V1ForecastRecoveryError("physical branch snapshot is not a completed result")
    workload_contract = snapshot.get("workload_events_contract")
    if not isinstance(workload_contract, Mapping):
        raise V1ForecastRecoveryError("physical snapshot lacks workload event contract")
    if sha256_json(dict(workload_contract)) != WORKLOAD_EVENTS_CONTRACT_SHA256:
        raise V1ForecastRecoveryError("physical workload contract digest changed")
    if workload_contract.get("sha256") != WORKLOAD_EVENTS_ARTIFACT_SHA256:
        raise V1ForecastRecoveryError("physical workload artifact SHA changed in the contract")
    workload_path = root / str(workload_contract.get("relative_path", ""))
    if not workload_path.is_file() or sha256_file(workload_path) != WORKLOAD_EVENTS_ARTIFACT_SHA256:
        raise V1ForecastRecoveryError("physical workload artifact does not hash-verify")
    if (
        recovery.get("expected_workload_sha256") != WORKLOAD_EVENTS_ARTIFACT_SHA256
        or recovery.get("protected_workload_sha256_before") != WORKLOAD_EVENTS_ARTIFACT_SHA256
        or recovery.get("protected_workload_sha256_after") != WORKLOAD_EVENTS_ARTIFACT_SHA256
        or recovery.get("protected_workload_modified") is not False
    ):
        raise V1ForecastRecoveryError("physical repair did not preserve the workload artifact")

    result_outputs = result.get("outputs")
    result_hashes = result.get("output_hashes")
    if not isinstance(result_outputs, Mapping) or not isinstance(result_hashes, Mapping):
        raise V1ForecastRecoveryError("completed physical result has no output ledger")
    effective_hashes = {str(key): str(value) for key, value in result_hashes.items()}
    effective_hashes.update({str(key): str(value) for key, value in after_hashes.items()})
    verified_outputs: dict[str, dict[str, Any]] = {}
    for name, relative_path in result_outputs.items():
        expected = effective_hashes.get(str(name))
        target = targets.get(str(name))
        if isinstance(target, Mapping) and target.get("canonical_path"):
            relative_path = target["canonical_path"]
        path = root / str(relative_path)
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise V1ForecastRecoveryError(
                f"current physical output does not match the completed recovery ledger: {name}"
            )
        verified_outputs[str(name)] = {
            "path": _relative(root, path),
            "sha256": expected,
            "byte_count": int(path.stat().st_size),
        }
    if "nlr295_workload_events" not in verified_outputs:
        raise V1ForecastRecoveryError("physical verified outputs omit nlr295_workload_events")
    current_manifest = root / str(branch_target.get("canonical_path", ""))
    return {
        "path": _relative(root, recovery_path),
        "sha256": sha256_file(recovery_path),
        "status": "VERIFIED_COMPLETED",
        "repair_id": recovery.get("repair_id"),
        "completed_snapshot_path": _relative(root, snapshot_path),
        "completed_snapshot_sha256": snapshot_expected,
        "post_repair_completed_manifest_sha256": recovery.get(
            "branch_manifest_sha256_after"
        ),
        "current_conflict_manifest_path": _relative(root, current_manifest),
        "current_conflict_manifest_sha256": (
            sha256_file(current_manifest) if current_manifest.is_file() else None
        ),
        "workload_events_contract": dict(workload_contract),
        "workload_events_contract_sha256": WORKLOAD_EVENTS_CONTRACT_SHA256,
        "workload_events_artifact_sha256": WORKLOAD_EVENTS_ARTIFACT_SHA256,
        "verified_output_count": len(verified_outputs),
        "verified_output_hashes": verified_outputs,
    }


def _low_memory_event_level_cfc_sequences(
    aggregate_sequences: Mapping[
        str,
        tuple[Any, Any, Any, Any, list[Any]],
    ],
    *,
    event_artifact_path: Path,
    event_artifact_sha256: str,
    history_events: int,
) -> tuple[dict[str, tuple[Any, Any, Any, Any, list[Any]]], dict[str, Any]]:
    """Equivalent event-history construction without DuckDB full materialization.

    The physical contract pins a Parquet file sorted by submit_time_utc and
    job_key. We verify the file hash and timestamp monotonicity, binary-search
    the last event strictly before each origin, and gather only the required
    origin x history rows. This avoids materializing a 13.8M-row window table.
    """

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from dcvpp.experiments.paper_training import COHORTS

    path = Path(event_artifact_path)
    if not path.is_file() or sha256_file(path) != event_artifact_sha256:
        raise V1ForecastRecoveryError("CfC-W physical workload event artifact hash mismatch")
    origins: list[pd.Timestamp] = []
    for split in ("train", "validation", "locked_test"):
        origins.extend(
            pd.Timestamp(day).tz_convert("UTC")
            for day in aggregate_sequences[split][4]
        )
    if not origins:
        raise V1ForecastRecoveryError("CfC-W recovery has no forecast origins")
    table = pq.read_table(
        path,
        columns=[
            "submit_time_utc",
            "wallclock_requested_hours",
            "gpus_requested",
            "partition",
            "qos",
        ],
        memory_map=True,
        pre_buffer=False,
    )
    submit_array = table["submit_time_utc"].combine_chunks()
    submit_ns = submit_array.cast(pa.int64()).to_numpy(zero_copy_only=False)
    if len(submit_ns) < history_events or bool((np.diff(submit_ns) < 0).any()):
        raise V1ForecastRecoveryError(
            "physical workload events are not monotone by submit_time_utc"
        )
    origin_ns = np.asarray([origin.value for origin in origins], dtype="int64")
    positions = np.searchsorted(submit_ns, origin_ns, side="left")
    if bool((positions < history_events).any()):
        raise V1ForecastRecoveryError("an origin lacks its required event history")
    offsets = np.arange(-history_events, 0, dtype="int64")
    history_indices = positions[:, None] + offsets[None, :]
    flat_indices = pa.array(history_indices.reshape(-1), type=pa.int64())

    selected_submit = submit_ns[history_indices]
    previous_indices = history_indices - 1
    previous_submit = submit_ns[previous_indices]
    elapsed = (selected_submit - previous_submit).astype("float64") / 3.6e12
    elapsed = np.clip(elapsed, 1.0 / 3600.0, 168.0).astype("float32")
    if not bool((selected_submit < origin_ns[:, None]).all()):
        raise V1ForecastRecoveryError(
            "CfC-W event encoder contains an event at or after forecast origin"
        )

    def selected_numeric(name: str) -> pd.Series:
        selected = pc.take(table[name].combine_chunks(), flat_indices)
        return pd.to_numeric(selected.to_pandas(), errors="coerce")

    def selected_text(name: str) -> pd.Series:
        selected = pc.take(table[name].combine_chunks(), flat_indices)
        return selected.to_pandas().fillna("").astype(str)

    wallclock = selected_numeric("wallclock_requested_hours")
    gpu = selected_numeric("gpus_requested")
    partition = selected_text("partition")
    qos = selected_text("qos")
    text = (partition + " " + qos).str.lower()
    cohort = np.select(
        [
            text.str.contains("interactive|debug|urgent|high", regex=True),
            gpu.fillna(0).gt(0),
            wallclock.le(2),
        ],
        [0, 2, 1],
        default=3,
    ).astype("int64")
    one_hot = np.eye(len(COHORTS), dtype="float32")[cohort]
    selected_submit_flat = pd.to_datetime(
        selected_submit.reshape(-1), utc=True, unit="ns", errors="raise"
    )
    local = selected_submit_flat.tz_convert("America/Denver")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    dow = local.dayofweek.to_numpy()
    partition_hash = (
        pd.util.hash_pandas_object(partition, index=False).to_numpy(dtype="uint64")
        % 1024
    ).astype("float32") / 1023.0
    qos_hash = (
        pd.util.hash_pandas_object(qos, index=False).to_numpy(dtype="uint64") % 256
    ).astype("float32") / 255.0
    features_flat = np.column_stack(
        [
            np.log1p(wallclock.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
            np.log1p(gpu.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
            partition_hash,
            qos_hash,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            wallclock.isna().to_numpy(dtype="float32"),
            gpu.isna().to_numpy(dtype="float32"),
            one_hot,
        ]
    ).astype("float32")
    feature_count = features_flat.shape[1]
    features = features_flat.reshape(len(origins), history_events, feature_count)
    elapsed_values = elapsed.reshape(len(origins), history_events, 1)
    train_origin_count = len(aggregate_sequences["train"][0])
    center = features[:train_origin_count].mean(axis=(0, 1))
    scale = features[:train_origin_count].std(axis=(0, 1))
    scale[scale == 0] = 1.0
    features = ((features - center) / scale).astype("float32")
    output: dict[str, tuple[Any, Any, Any, Any, list[Any]]] = {}
    cursor = 0
    for split in ("train", "validation", "locked_test"):
        count = len(aggregate_sequences[split][0])
        source = aggregate_sequences[split]
        output[split] = (
            features[cursor : cursor + count],
            elapsed_values[cursor : cursor + count],
            source[2],
            source[3],
            source[4],
        )
        cursor += count
    return output, {
        "representation": "event_level_origin_available_features",
        "history_events": int(history_events),
        "explicit_elapsed_time": True,
        "input_features": [
            "log1p_requested_wallclock_hours",
            "log1p_gpus_requested",
            "partition_hash",
            "qos_hash",
            "submission_hour_sin",
            "submission_hour_cos",
            "submission_dow_sin",
            "submission_dow_cos",
            "wallclock_missing",
            "gpu_missing",
            *[f"cohort_{name}" for name in COHORTS],
        ],
        "forbidden_realized_features": [
            "wallclock_used_hours",
            "start_time_utc",
            "end_time_utc",
            "state",
            "nodes_used",
            "processors_used",
            "node_energy",
            "gpu_energy",
        ],
        "event_artifact_path": path.as_posix(),
        "event_artifact_sha256": event_artifact_sha256,
        "origin_count": len(origins),
        "max_event_at_or_after_origin": 0,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
    }


def _verify_cfc_checkpoint_state(
    repository_root: Path,
) -> dict[str, Any]:
    import torch

    from dcvpp.models.workload.cfc_w import CfCW

    root = Path(repository_root).resolve()
    checkpoint_path = (
        root
        / "artifacts/real_forecasting/paper-forecast-v1/checkpoints/"
        "nlr_workload/cfc_w/seed_20260809_latest.pt"
    )
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != ORIGINAL_CFC_CHECKPOINT_SHA256:
        raise V1ForecastRecoveryError("epoch-59 CfC-W source checkpoint does not hash-verify")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_scalars = {
        "model_name": "cfc_w",
        "model_class": "dcvpp.models.workload.cfc_w.CfCW",
        "epoch": 59,
        "best_epoch": 58,
        "seed": 20260809,
        "resume_fingerprint": ORIGINAL_CFC_RESUME_FINGERPRINT,
        "completed": False,
    }
    for key, expected in expected_scalars.items():
        if checkpoint.get(key) != expected:
            raise V1ForecastRecoveryError(
                f"epoch-59 CfC-W checkpoint field changed: {key}"
            )
    if not math.isclose(
        float(checkpoint.get("best_validation_distribution_score", math.inf)),
        6.303293383121616,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise V1ForecastRecoveryError("epoch-59 CfC-W validation score changed")
    model = CfCW(input_size=14, hidden_size=64, horizon=96, cohorts=4)
    if dict(model.architecture_contract) != dict(checkpoint.get("architecture", {})):
        raise V1ForecastRecoveryError("current CfC-W architecture contract differs from checkpoint")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    best_model = CfCW(input_size=14, hidden_size=64, horizon=96, cohorts=4)
    best_model.load_state_dict(checkpoint["best_model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    architecture_contract = dict(model.architecture_contract)

    result: dict[str, Any] = {
        "source_checkpoint_path": _relative(root, checkpoint_path),
        "source_checkpoint_sha256": ORIGINAL_CFC_CHECKPOINT_SHA256,
        "source_epoch": 59,
        "source_best_epoch": 58,
        "source_best_validation_distribution_score": 6.303293383121616,
        "source_resume_fingerprint": ORIGINAL_CFC_RESUME_FINGERPRINT,
        "model_state_strict_load": True,
        "best_model_state_strict_load": True,
        "optimizer_state_load": True,
        "architecture_contract": architecture_contract,
        "deep_sequence_compatibility_checked": False,
    }
    return result


def _verify_cfc_sequence_compatibility(
    repository_root: Path,
    checkpoint_state: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    architecture_contract = dict(checkpoint_state["architecture_contract"])

    import pandas as pd

    from dcvpp.experiments.paper_training import _array_sha256, _workload_sequences

    panel_path = (
        root
        / "artifacts/real_forecasting/paper-forecast-v1/data/"
        "nlr295_15m_workload.parquet"
    )
    if not panel_path.is_file():
        raise V1ForecastRecoveryError("v1 workload panel is absent")
    panel = pd.read_parquet(panel_path)
    print("v1-recovery-preflight: workload panel loaded", flush=True)
    aggregate_sequences, sequence_manifest = _workload_sequences(
        panel, history=672, horizon=96
    )
    print("v1-recovery-preflight: aggregate sequences reconstructed", flush=True)
    event_path = (
        root
        / "data/processed/research/physical/run_id=paper-physical-v1/"
        "nlr295_workload_events.parquet"
    )
    cfc_sequences, event_manifest = _low_memory_event_level_cfc_sequences(
        aggregate_sequences,
        event_artifact_path=event_path,
        event_artifact_sha256=WORKLOAD_EVENTS_ARTIFACT_SHA256,
        history_events=672,
    )
    print("v1-recovery-preflight: event sequences reconstructed", flush=True)
    expected_shapes = {
        "aggregate_train": [1222, 672, 12],
        "aggregate_validation": [409, 672, 12],
        "aggregate_locked_test": [409, 672, 12],
        "event_train": [1222, 672, 14],
        "event_validation": [409, 672, 14],
        "event_locked_test": [409, 672, 14],
    }
    observed_shapes = {
        "aggregate_train": list(aggregate_sequences["train"][0].shape),
        "aggregate_validation": list(aggregate_sequences["validation"][0].shape),
        "aggregate_locked_test": list(aggregate_sequences["locked_test"][0].shape),
        "event_train": list(cfc_sequences["train"][0].shape),
        "event_validation": list(cfc_sequences["validation"][0].shape),
        "event_locked_test": list(cfc_sequences["locked_test"][0].shape),
    }
    if observed_shapes != expected_shapes:
        raise V1ForecastRecoveryError(
            f"reconstructed CfC-W sequence shapes changed: {observed_shapes}"
        )
    recovery_parent_fingerprint = sha256_json(
        {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "policy": "explicit_checkpoint_fingerprint_adapter_v1",
            "source_checkpoint_sha256": ORIGINAL_CFC_CHECKPOINT_SHA256,
            "workload_events_contract_sha256": WORKLOAD_EVENTS_CONTRACT_SHA256,
            "frozen_source_hashes": FROZEN_SOURCE_HASHES,
        }
    )
    print("v1-recovery-preflight: recovery parent fingerprint computed", flush=True)
    train = aggregate_sequences["train"]
    validation = aggregate_sequences["validation"]
    data_fingerprint = sha256_json(
        {
            "parent": recovery_parent_fingerprint,
            "history": 672,
            "horizon": 96,
            "train_x": _array_sha256(train[0]),
            "train_dt": _array_sha256(train[1]),
            "train_count": _array_sha256(train[2]),
            "train_volume": _array_sha256(train[3]),
            "validation_x": _array_sha256(validation[0]),
            "validation_count": _array_sha256(validation[2]),
            "sequence_manifest": sequence_manifest,
            "event_manifest": event_manifest,
        }
    )
    print("v1-recovery-preflight: recovery data fingerprint computed", flush=True)
    adapted_seed_fingerprint = sha256_json(
        {
            "parent": data_fingerprint,
            "model": "cfc_w",
            "seed": 20260809,
            "architecture": architecture_contract,
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 1e-3,
        }
    )
    if adapted_seed_fingerprint == ORIGINAL_CFC_RESUME_FINGERPRINT:
        raise V1ForecastRecoveryError("CfC-W mismatch adapter unexpectedly has no mismatch")
    return {
        "panel_path": _relative(root, panel_path),
        "panel_sha256": sha256_file(panel_path),
        "sequence_shapes": observed_shapes,
        "sequence_manifest_sha256": sha256_json(sequence_manifest),
        "event_sequence_manifest_sha256": sha256_json(event_manifest),
        "recovery_parent_fingerprint": recovery_parent_fingerprint,
        "recovery_data_fingerprint": data_fingerprint,
        "adapted_seed_fingerprint": adapted_seed_fingerprint,
        "fingerprint_mismatch_explicit": True,
        "exact_uninterrupted_trajectory_claimed": False,
        "dataloader_rng_exact_continuation_claimed": False,
        "deep_sequence_compatibility_checked": True,
    }


def _checkpoint_state_gate_path(repository_root: Path) -> Path:
    return recovery_root(repository_root) / "cfc_checkpoint_state_gate.json"


def write_cfc_checkpoint_state_gate(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _checkpoint_state_gate_path(root)
    state = _verify_cfc_checkpoint_state(root)
    payload = {
        "schema_version": "paper_forecast_v1_cfc_checkpoint_state_gate/v1",
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "terminal": True,
        "checkpoint_state": state,
        "completed_at_utc": utc_now(),
    }
    payload["gate_fingerprint"] = sha256_json(payload)
    atomic_json(path, payload)
    return {
        **payload,
        "path": _relative(root, path),
        "sha256": sha256_file(path),
    }


def _load_checkpoint_state_gate(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _checkpoint_state_gate_path(root)
    if not path.is_file():
        raise V1ForecastRecoveryError("CfC checkpoint state gate is absent")
    payload = _load_json(path)
    fingerprint = payload.get("gate_fingerprint")
    base = {key: item for key, item in payload.items() if key != "gate_fingerprint"}
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("terminal") is not True
        or not isinstance(fingerprint, str)
        or sha256_json(base) != fingerprint
    ):
        raise V1ForecastRecoveryError("CfC checkpoint state gate is invalid")
    state = payload.get("checkpoint_state")
    if not isinstance(state, Mapping):
        raise V1ForecastRecoveryError("CfC checkpoint state gate lacks state evidence")
    checkpoint_path = root / str(state.get("source_checkpoint_path", ""))
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != state.get("source_checkpoint_sha256")
    ):
        raise V1ForecastRecoveryError("CfC source checkpoint changed after state gate")
    return {
        **dict(state),
        "checkpoint_state_gate_path": _relative(root, path),
        "checkpoint_state_gate_sha256": sha256_file(path),
    }


def verify_cfc_checkpoint_compatibility(
    repository_root: Path,
    *,
    deep_sequences: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not deep_sequences:
        return _verify_cfc_checkpoint_state(root)
    state = _load_checkpoint_state_gate(root)
    sequences = _verify_cfc_sequence_compatibility(root, state)
    return {**state, **sequences}


def run_recovery_preflight(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output_root = recovery_root(root)
    preflight_path = output_root / "preflight_manifest.json"
    ledger = build_corruption_ledger(root, preserve=True)
    frozen_sources = verify_frozen_sources(root)
    physical = verify_physical_recovery_contract(root)
    if not _checkpoint_state_gate_path(root).is_file():
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "dcvpp.experiments.v1_forecast_recovery",
                "checkpoint",
                "--root",
                str(root),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise V1ForecastRecoveryError(
                "CfC checkpoint state subprocess failed: "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
    checkpoint = verify_cfc_checkpoint_compatibility(root, deep_sequences=True)
    payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "terminal": True,
        "formal_recovery_training_allowed": True,
        "corruption_ledger": {
            "path": ledger["path"],
            "sha256": ledger["sha256"],
            "record_count": ledger["record_count"],
        },
        "frozen_source_files": frozen_sources,
        "physical_recovery_contract": physical,
        "cfc_checkpoint_compatibility": checkpoint,
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "locked_2025_policy_decisions_consumed": False,
        "completed_at_utc": utc_now(),
    }
    payload["preflight_fingerprint"] = sha256_json(payload)
    atomic_json(preflight_path, payload)
    return {
        **payload,
        "path": _relative(root, preflight_path),
        "sha256": sha256_file(preflight_path),
    }


def _load_verified_preflight(repository_root: Path) -> tuple[Path, dict[str, Any]]:
    root = Path(repository_root).resolve()
    path = recovery_root(root) / "preflight_manifest.json"
    if not path.is_file():
        raise V1ForecastRecoveryError("recovery preflight has not been completed")
    value = _load_json(path)
    if (
        value.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or value.get("status") != "COMPLETE"
        or value.get("terminal") is not True
        or value.get("formal_recovery_training_allowed") is not True
    ):
        raise V1ForecastRecoveryError("recovery preflight is not COMPLETE")
    expected = value.get("preflight_fingerprint")
    payload = {key: item for key, item in value.items() if key != "preflight_fingerprint"}
    if not isinstance(expected, str) or sha256_json(payload) != expected:
        raise V1ForecastRecoveryError("recovery preflight fingerprint changed")
    ledger = value.get("corruption_ledger")
    if not isinstance(ledger, Mapping):
        raise V1ForecastRecoveryError("recovery preflight lacks the corruption ledger")
    ledger_path = root / str(ledger.get("path", ""))
    if (
        not ledger_path.is_file()
        or sha256_file(ledger_path) != ledger.get("sha256")
        or int(ledger.get("record_count", -1)) != EXPECTED_CORRUPTION_COUNT
    ):
        raise V1ForecastRecoveryError("preflight corruption ledger no longer verifies")
    validate_corruption_ledger(root, _load_json(ledger_path))
    return path, value


def _expected_workload_seed_fingerprint(
    *,
    data_fingerprint: str,
    model_name: str,
    seed: int,
    architecture: Mapping[str, Any],
) -> str:
    return sha256_json(
        {
            "parent": data_fingerprint,
            "model": model_name,
            "seed": int(seed),
            "architecture": dict(architecture),
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 1e-3,
        }
    )


def _workload_source_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    model_name: str,
    seed: int,
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate whether an original workload checkpoint can be adapted safely.

    An incomplete checkpoint is not relabelled as complete.  It may only be
    copied into the isolated recovery tree and continued from its recorded
    epoch when all state required by the frozen trainer is present.
    """

    if checkpoint.get("model_name") != model_name or int(
        checkpoint.get("seed", -1)
    ) != int(seed):
        raise V1ForecastRecoveryError(
            f"workload checkpoint identity mismatch: {model_name}/{seed}"
        )
    completed = checkpoint.get("completed")
    if not isinstance(completed, bool):
        raise V1ForecastRecoveryError(
            f"workload checkpoint completion flag is invalid: {model_name}/{seed}"
        )
    epoch = int(checkpoint.get("epoch", -1))
    best_epoch = int(checkpoint.get("best_epoch", -1))
    if epoch < 1 or epoch > 100 or best_epoch < 1 or best_epoch > epoch:
        raise V1ForecastRecoveryError(
            f"workload checkpoint epoch contract is invalid: {model_name}/{seed}"
        )
    if not completed and epoch >= 100:
        raise V1ForecastRecoveryError(
            f"incomplete workload checkpoint has no remaining epoch: {model_name}/{seed}"
        )
    if dict(checkpoint.get("architecture", {})) != dict(architecture):
        raise V1ForecastRecoveryError(
            f"workload checkpoint architecture changed: {model_name}/{seed}"
        )
    for key in ("model_state", "best_model_state", "optimizer_state"):
        if not isinstance(checkpoint.get(key), Mapping) or not checkpoint[key]:
            raise V1ForecastRecoveryError(
                f"workload checkpoint lacks resumable {key}: {model_name}/{seed}"
            )
    resume_fingerprint = str(checkpoint.get("resume_fingerprint", ""))
    if len(resume_fingerprint) != 64:
        raise V1ForecastRecoveryError(
            f"workload checkpoint resume fingerprint is invalid: {model_name}/{seed}"
        )
    return {
        "source_epoch": epoch,
        "source_best_epoch": best_epoch,
        "source_completed": completed,
        "continue_training": not completed,
        "source_resume_fingerprint": resume_fingerprint,
    }


def _preserve_known_workload_adapter_failure(
    repository_root: Path,
    output_root: Path,
    adapters: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Materialize the observed pre-patch adapter failure as immutable evidence."""

    root = Path(repository_root).resolve()
    matching = next(
        (
            dict(record)
            for record in adapters
            if record.get("model") == "gru"
            and int(record.get("seed", -1)) == 20260819
            and record.get("source_sha256")
            == KNOWN_INCOMPLETE_GRU_20260819_SHA256
            and record.get("source_completed") is False
            and int(record.get("source_epoch", -1)) == 83
        ),
        None,
    )
    if matching is None:
        return None
    path = output_root / "failures/pre_patch_adapter_failure_gru_20260819.json"
    if path.is_file():
        existing = _load_json(path)
        expected = existing.get("failure_fingerprint")
        payload = {
            key: value
            for key, value in existing.items()
            if key != "failure_fingerprint"
        }
        if not isinstance(expected, str) or sha256_json(payload) != expected:
            raise V1ForecastRecoveryError(
                "preserved workload adapter failure evidence changed"
            )
    else:
        payload = {
            "schema_version": "paper_forecast_v1_workload_recovery_failure/v1",
            "run_id": WORKLOAD_COMPONENT_RUN_ID,
            "status": "PRESERVED_FAILURE_EVIDENCE",
            "attempt_terminal": True,
            "overall_recovery_terminal": False,
            "failure_stage": "prepare_workload_checkpoint_adapters",
            "error_type": "V1ForecastRecoveryError",
            "error": "intact workload baseline is not complete: gru/20260819",
            "observation_source": "root_supervisor_exec_cell_183",
            "recorded_after_observed_failure": True,
            "source_checkpoint": {
                "path": matching["source_path"],
                "sha256": matching["source_sha256"],
                "model": matching["model"],
                "seed": matching["seed"],
                "epoch": matching["source_epoch"],
                "best_epoch": matching["source_best_epoch"],
                "completed": matching["source_completed"],
                "resume_fingerprint": matching["source_resume_fingerprint"],
            },
            "resolution": (
                "preserve the incomplete source unchanged, adapt a hash-traced "
                "copy in the recovery tree, and continue from epoch 83"
            ),
            "original_checkpoint_modified": False,
            "locked_2025_metrics_consumed": False,
            "preserved_at_utc": utc_now(),
        }
        payload["failure_fingerprint"] = sha256_json(payload)
        atomic_json(path, payload)
        existing = payload
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "status": existing["status"],
        "error_type": existing["error_type"],
        "error": existing["error"],
    }


def _workload_component_root(repository_root: Path) -> Path:
    return (
        Path(repository_root).resolve()
        / "artifacts/real_forecasting_recovery/components"
        / WORKLOAD_COMPONENT_RUN_ID
    )


def preserve_workload_native_failure_evidence(
    repository_root: Path,
) -> dict[str, Any]:
    """Preserve the externally observed native failures without reinterpretation."""

    root = Path(repository_root).resolve()
    path = (
        _workload_component_root(root)
        / "failures/native_appcrash_workload_cell_190.json"
    )
    if path.is_file():
        existing = _load_json(path)
        fingerprint = existing.get("failure_fingerprint")
        payload = {
            key: value
            for key, value in existing.items()
            if key != "failure_fingerprint"
        }
        if not isinstance(fingerprint, str) or sha256_json(payload) != fingerprint:
            raise V1ForecastRecoveryError(
                "preserved workload native-failure evidence changed"
            )
    else:
        payload = {
            "schema_version": WORKLOAD_NATIVE_FAILURE_SCHEMA_VERSION,
            "run_id": WORKLOAD_COMPONENT_RUN_ID,
            "status": "PRESERVED_NATIVE_FAILURE_EVIDENCE",
            "attempt_terminal": True,
            "overall_recovery_terminal": False,
            "failed_stage": "monolithic_workload_retry_after_checkpoint_adapters",
            "supervisor_cell": 190,
            "process_exit_observation": {
                "exit_code_reported_by_supervisor": 1,
                "elapsed_seconds": 49.6,
                "python_traceback_present": False,
                "stdout_stderr_observations": [
                    "paper_training Transformer warning",
                    "divide-by-zero log warning",
                ],
                "workload_run_manifest_present": False,
                "workload_epoch_progress_present": False,
            },
            "windows_event_log_evidence": [
                {
                    "event_time_local": "2026-08-11T00:22:23+08:00",
                    "provider": "Application Error",
                    "event_id": 1000,
                    "application": "python.exe",
                    "application_version": "3.11.15150.1013",
                    "application_path": (
                        "C:/Users/Kali/anaconda3/envs/riskmoe311/python.exe"
                    ),
                    "fault_module": "python311.dll",
                    "exception_code": "0xc0000005",
                    "fault_offset": "0x0000000000209f42",
                    "report_id": "40e8c67f-4fd1-41ac-878e-c7926fa6d61b",
                    "interpretation": "native_access_violation_not_python_exception",
                },
                {
                    "event_time_local": "2026-08-11T00:20:25+08:00",
                    "provider": "Application Error",
                    "event_id": 1000,
                    "application": "python.exe",
                    "fault_module": "libiomp5md.dll",
                    "fault_module_path": (
                        "C:/Users/Kali/Desktop/电力市场/power-dc-vpp/.venv/"
                        "Lib/site-packages/torch/lib/libiomp5md.dll"
                    ),
                    "exception_code": "0xc0000005",
                    "fault_offset": "0x00000000000e2689",
                    "report_id": "798bb6a9-34fb-4828-9dca-6f83187bd2a3",
                    "relationship": "related_read_only_diagnostic_crash",
                },
                {
                    "event_time_local": "2026-08-11T00:19:53+08:00",
                    "provider": "Application Error",
                    "event_id": 1000,
                    "application": "python.exe",
                    "fault_module": "libiomp5md.dll",
                    "exception_code": "0xc0000005",
                    "fault_offset": "0x00000000000e2689",
                    "report_id": "2de82545-1549-41d7-af8d-f81b45d8ada8",
                    "relationship": "related_read_only_diagnostic_crash",
                },
            ],
            "diagnosis": (
                "Windows native access violation after adapter completion; "
                "not Python SystemExit and not a caught Python exception"
            ),
            "recovery_policy": {
                "process_isolation": [
                    "checkpoint_adapter",
                    "arrow_sequence_materialization",
                    "torch_training_without_arrow",
                ],
                "native_training_retry_limit": 2,
                "kmp_duplicate_lib_ok": False,
                "original_checkpoint_modified": False,
            },
            "locked_2025_metrics_consumed": False,
            "recorded_at_utc": utc_now(),
        }
        payload["failure_fingerprint"] = sha256_json(payload)
        atomic_json(path, payload)
        existing = payload
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "status": existing["status"],
        "failure_fingerprint": existing["failure_fingerprint"],
    }


def _atomic_numpy(path: Path, value: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        if path.is_file():
            if sha256_file(path) != sha256_file(temporary):
                raise V1ForecastRecoveryError(
                    f"existing workload sequence array changed: {path}"
                )
            return
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_workload_sequence_bundle(
    repository_root: Path,
    *,
    aggregate_sequences: Mapping[str, Sequence[Any]],
    event_sequences: Mapping[str, Sequence[Any]],
    sequence_manifest: Mapping[str, Any],
    event_manifest: Mapping[str, Any],
    panel_path: Path,
    event_path: Path,
) -> dict[str, Any]:
    import numpy as np

    root = Path(repository_root).resolve()
    bundle_root = _workload_component_root(root) / "sequence_bundle"
    manifest_path = bundle_root / "manifest.json"
    records: dict[str, dict[str, Any]] = {}
    days: dict[str, dict[str, list[str]]] = {"aggregate": {}, "event": {}}
    for representation, collection in (
        ("aggregate", aggregate_sequences),
        ("event", event_sequences),
    ):
        if set(collection) != {"train", "validation", "locked_test"}:
            raise V1ForecastRecoveryError(
                f"workload sequence split contract changed: {representation}"
            )
        for split in ("train", "validation", "locked_test"):
            values = collection[split]
            if len(values) != 5:
                raise V1ForecastRecoveryError(
                    f"workload sequence tuple changed: {representation}/{split}"
                )
            for index, name in enumerate(("x", "dt", "count", "volume")):
                array = np.asarray(values[index])
                path = bundle_root / representation / split / f"{name}.npy"
                _atomic_numpy(path, array)
                key = f"{representation}.{split}.{name}"
                records[key] = {
                    "path": _relative(root, path),
                    "sha256": sha256_file(path),
                    "byte_count": int(path.stat().st_size),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            days[representation][split] = [
                item.isoformat() if hasattr(item, "isoformat") else str(item)
                for item in values[4]
            ]
            if len(days[representation][split]) != int(values[0].shape[0]):
                raise V1ForecastRecoveryError(
                    f"workload sequence day count changed: {representation}/{split}"
                )
    payload = {
        "schema_version": WORKLOAD_SEQUENCE_BUNDLE_SCHEMA_VERSION,
        "run_id": WORKLOAD_COMPONENT_RUN_ID,
        "status": "COMPLETE",
        "terminal": True,
        "source_panel": {
            "path": _relative(root, panel_path),
            "sha256": sha256_file(panel_path),
            "expected_sha256": sha256_file(panel_path),
        },
        "source_event_artifact": {
            "path": _relative(root, event_path),
            "sha256": sha256_file(event_path),
            "expected_sha256": WORKLOAD_EVENTS_ARTIFACT_SHA256,
        },
        "arrays": records,
        "days": days,
        "sequence_manifest": dict(sequence_manifest),
        "event_sequence_manifest": dict(event_manifest),
        "event_parquet_parsed_or_materialized_with_arrow_in_training_process": False,
        "event_source_hash_revalidated_by_raw_bytes": True,
        "event_sequence_construction_performed_in_training_process": False,
        "locked_2025_metrics_consumed": False,
        "locked_2025_policy_decisions_consumed": False,
        "created_at_utc": utc_now(),
    }
    payload["bundle_fingerprint"] = sha256_json(payload)
    atomic_json(manifest_path, payload)
    return {
        **payload,
        "path": _relative(root, manifest_path),
        "sha256": sha256_file(manifest_path),
        "resumed": False,
    }


def _amend_workload_sequence_bundle_wording(
    repository_root: Path,
) -> dict[str, Any] | None:
    root = Path(repository_root).resolve()
    path = _workload_component_root(root) / "sequence_bundle/manifest.json"
    if not path.is_file():
        return None
    value = _load_json(path)
    if "arrow_loaded_in_training_process" not in value:
        return value
    fingerprint = value.get("bundle_fingerprint")
    payload = {
        key: item for key, item in value.items() if key != "bundle_fingerprint"
    }
    if not isinstance(fingerprint, str) or sha256_json(payload) != fingerprint:
        raise V1ForecastRecoveryError(
            "pre-amendment workload sequence bundle fingerprint changed"
        )
    snapshot = (
        _workload_component_root(root)
        / "failures/sequence_bundle_manifest_before_truthful_wording.json"
    )
    _materialize_copy(path, snapshot)
    value.pop("bundle_fingerprint", None)
    value.pop("arrow_loaded_in_training_process", None)
    value[
        "event_parquet_parsed_or_materialized_with_arrow_in_training_process"
    ] = False
    value["event_source_hash_revalidated_by_raw_bytes"] = True
    value["event_sequence_construction_performed_in_training_process"] = False
    value["wording_amendment"] = {
        "reason": (
            "training imports libraries that may import pyarrow, but it does "
            "not load event Parquet or construct event sequences"
        ),
        "preserved_prior_manifest_path": _relative(root, snapshot),
        "preserved_prior_manifest_sha256": sha256_file(snapshot),
        "prior_bundle_fingerprint": fingerprint,
        "scientific_array_bytes_changed": False,
    }
    value["bundle_fingerprint"] = sha256_json(value)
    atomic_json(path, value)
    return value


def _load_verified_workload_sequence_bundle(
    repository_root: Path,
    *,
    load_arrays: bool,
) -> dict[str, Any]:
    import numpy as np

    root = Path(repository_root).resolve()
    path = _workload_component_root(root) / "sequence_bundle/manifest.json"
    if not path.is_file():
        raise V1ForecastRecoveryError("workload sequence bundle is absent")
    value = _load_json(path)
    fingerprint = value.get("bundle_fingerprint")
    payload = {
        key: item for key, item in value.items() if key != "bundle_fingerprint"
    }
    if (
        value.get("schema_version") != WORKLOAD_SEQUENCE_BUNDLE_SCHEMA_VERSION
        or value.get("status") != "COMPLETE"
        or value.get("terminal") is not True
        or not isinstance(fingerprint, str)
        or sha256_json(payload) != fingerprint
        or "arrow_loaded_in_training_process" in value
        or value.get(
            "event_parquet_parsed_or_materialized_with_arrow_in_training_process"
        )
        is not False
        or value.get("event_source_hash_revalidated_by_raw_bytes") is not True
        or value.get("event_sequence_construction_performed_in_training_process")
        is not False
    ):
        raise V1ForecastRecoveryError("workload sequence bundle contract changed")
    source_panel = value.get("source_panel", {})
    source_event = value.get("source_event_artifact", {})
    for source, expected in (
        (source_panel, source_panel.get("expected_sha256")),
        (source_event, WORKLOAD_EVENTS_ARTIFACT_SHA256),
    ):
        source_path = root / str(source.get("path", ""))
        if (
            source.get("sha256") != expected
            or not source_path.is_file()
            or sha256_file(source_path) != expected
        ):
            raise V1ForecastRecoveryError("workload sequence source hash changed")
    preflight_path = recovery_root(root) / "preflight_manifest.json"
    if preflight_path.is_file():
        preflight = _load_json(preflight_path)
        preflight_panel_sha = (
            preflight.get("cfc_checkpoint_compatibility", {}).get("panel_sha256")
        )
        if preflight_panel_sha != source_panel.get("expected_sha256"):
            raise V1ForecastRecoveryError(
                "workload sequence panel no longer matches the recovery preflight"
            )
    arrays = value.get("arrays")
    expected_array_keys = {
        f"{representation}.{split}.{name}"
        for representation in ("aggregate", "event")
        for split in ("train", "validation", "locked_test")
        for name in ("x", "dt", "count", "volume")
    }
    if not isinstance(arrays, Mapping) or set(arrays) != expected_array_keys:
        raise V1ForecastRecoveryError("workload sequence array ledger is incomplete")
    loaded: dict[str, Any] = {}
    validated: dict[str, Any] = {}
    bundle_root = path.parent.resolve()
    for key, record in arrays.items():
        if not isinstance(record, Mapping):
            raise V1ForecastRecoveryError("workload sequence array record is invalid")
        array_path = root / str(record.get("path", ""))
        try:
            array_path.resolve().relative_to(bundle_root)
        except ValueError as exc:
            raise V1ForecastRecoveryError(
                f"workload sequence array escapes its bundle: {key}"
            ) from exc
        if not array_path.is_file() or sha256_file(array_path) != record.get("sha256"):
            raise V1ForecastRecoveryError(
                f"workload sequence array hash changed: {key}"
            )
        # Keep the training view copy-on-write and memory mapped.  Materializing
        # all 24 arrays (~166 MB) before checkpoint state loading reproducibly
        # triggers a native Windows access violation in this environment.  Mode
        # ``c`` remains writable to PyTorch but cannot mutate the evidence files.
        array = np.load(
            array_path,
            mmap_mode="c" if load_arrays else "r",
            allow_pickle=False,
        )
        if list(array.shape) != record.get("shape") or str(array.dtype) != record.get("dtype"):
            raise V1ForecastRecoveryError(
                f"workload sequence array shape or dtype changed: {key}"
            )
        if str(array.dtype) != "float32" or not np.isfinite(array).all():
            raise V1ForecastRecoveryError(
                f"workload sequence array is not finite float32: {key}"
            )
        name = str(key).rsplit(".", 1)[-1]
        if name == "dt" and not np.all(array > 0):
            raise V1ForecastRecoveryError(
                f"workload elapsed-time array is not strictly positive: {key}"
            )
        if name in {"count", "volume"} and not np.all(array >= 0):
            raise V1ForecastRecoveryError(
                f"workload target array is negative: {key}"
            )
        validated[str(key)] = array
        if load_arrays:
            loaded[str(key)] = array
    days = value.get("days")
    if not isinstance(days, Mapping) or set(days) != {"aggregate", "event"}:
        raise V1ForecastRecoveryError("workload sequence day ledger is invalid")
    split_sizes = WORKLOAD_SEQUENCE_SPLIT_SIZES
    for representation, feature_count in (
        ("aggregate", WORKLOAD_AGGREGATE_FEATURES),
        ("event", WORKLOAD_EVENT_FEATURES),
    ):
        representation_days = days.get(representation)
        if not isinstance(representation_days, Mapping) or set(
            representation_days
        ) != set(split_sizes):
            raise V1ForecastRecoveryError("workload sequence day splits changed")
        for split, size in split_sizes.items():
            expected_shapes = {
                "x": [size, WORKLOAD_HISTORY, feature_count],
                "dt": [size, WORKLOAD_HISTORY, 1],
                "count": [size, WORKLOAD_COHORTS, WORKLOAD_HORIZON],
                "volume": [size, WORKLOAD_COHORTS, WORKLOAD_HORIZON],
            }
            for name, expected_shape in expected_shapes.items():
                if list(validated[f"{representation}.{split}.{name}"].shape) != expected_shape:
                    raise V1ForecastRecoveryError(
                        f"workload sequence shape contract changed: "
                        f"{representation}/{split}/{name}"
                    )
            day_values = representation_days[split]
            if not isinstance(day_values, list) or len(day_values) != size:
                raise V1ForecastRecoveryError(
                    f"workload sequence day count changed: {representation}/{split}"
                )
            for item in day_values:
                parsed = datetime.fromisoformat(str(item).replace("Z", "+00:00"))
                if (
                    parsed.tzinfo is None
                    or parsed.utcoffset() is None
                    or parsed.utcoffset().total_seconds() != 0
                ):
                    raise V1ForecastRecoveryError(
                        f"workload sequence day is not UTC: {representation}/{split}"
                    )
    for split in split_sizes:
        for name in ("count", "volume"):
            if not np.array_equal(
                validated[f"aggregate.{split}.{name}"],
                validated[f"event.{split}.{name}"],
            ):
                raise V1ForecastRecoveryError(
                    f"aggregate/event workload targets differ: {split}/{name}"
                )
        if days["aggregate"][split] != days["event"][split]:
            raise V1ForecastRecoveryError(
                f"aggregate/event workload days differ: {split}"
            )
    in_memory_aliases: dict[str, str] = {}
    if load_arrays:
        # The two representations have hash-verified identical targets.  Keep
        # one mapped target view per split rather than retaining six redundant
        # mappings.  Twenty-four simultaneously live mappings reproduce the
        # Windows/PyTorch load_state_dict access violation; eighteen do not.
        for split in split_sizes:
            for name in ("count", "volume"):
                event_key = f"event.{split}.{name}"
                aggregate_key = f"aggregate.{split}.{name}"
                event_array = loaded[event_key]
                event_array._mmap.close()
                loaded[event_key] = loaded[aggregate_key]
                in_memory_aliases[event_key] = aggregate_key
    result = {
        **value,
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "resumed": True,
    }
    if load_arrays:
        result["loaded_arrays"] = loaded
        result["array_loading_mode"] = "copy_on_write_memory_map"
        result["in_memory_aliases"] = in_memory_aliases
        result["active_memory_map_count"] = len(
            {id(array) for array in loaded.values()}
        )
    return result


def prepare_workload_sequence_bundle(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    manifest_path = _workload_component_root(root) / "sequence_bundle/manifest.json"
    if manifest_path.is_file():
        _amend_workload_sequence_bundle_wording(root)
        return _load_verified_workload_sequence_bundle(root, load_arrays=False)

    import pandas as pd

    from dcvpp.experiments.paper_training import _workload_sequences

    panel_path = (
        root
        / "artifacts/real_forecasting/paper-forecast-v1/data/"
        "nlr295_15m_workload.parquet"
    )
    event_path = (
        root
        / "data/processed/research/physical/run_id=paper-physical-v1/"
        "nlr295_workload_events.parquet"
    )
    panel = pd.read_parquet(panel_path)
    aggregate, sequence_manifest = _workload_sequences(
        panel, history=672, horizon=96
    )
    event, event_manifest = _low_memory_event_level_cfc_sequences(
        aggregate,
        event_artifact_path=event_path,
        event_artifact_sha256=WORKLOAD_EVENTS_ARTIFACT_SHA256,
        history_events=672,
    )
    return _write_workload_sequence_bundle(
        root,
        aggregate_sequences=aggregate,
        event_sequences=event,
        sequence_manifest=sequence_manifest,
        event_manifest=event_manifest,
        panel_path=panel_path,
        event_path=event_path,
    )


def _workload_stage_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_WAIT_POLICY": "PASSIVE",
            "KMP_BLOCKTIME": "0",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    return environment


def _native_windows_exit(returncode: int) -> bool:
    unsigned = int(returncode) & 0xFFFFFFFF
    return unsigned in {0xC0000005, 0xC0000409}


def _run_isolated_workload_stage(
    repository_root: Path,
    *,
    phase: str,
    maximum_attempts: int,
    completion_probe: Any,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    log_root = _workload_component_root(root) / "orchestration" / phase
    attempts: list[dict[str, Any]] = []
    existing_attempt_numbers = [
        int(path.stem.removeprefix("attempt_"))
        for path in log_root.glob("attempt_*.json")
        if path.stem.removeprefix("attempt_").isdigit()
    ]
    first_attempt_number = max(existing_attempt_numbers, default=0) + 1
    for invocation_attempt in range(1, maximum_attempts + 1):
        attempt = first_attempt_number + invocation_attempt - 1
        started_at = utc_now()
        if phase in {
            "adapters",
            "workload-smoke",
            "workload-cfc-train",
            "workload-train",
        }:
            # This interpreter has a reproducible native ordering constraint:
            # importing this recovery module before torch causes
            # load_state_dict to fault with 0xC0000005.  A tiny bootstrap makes
            # the safe order explicit without changing any frozen v1 source.
            command = [
                sys.executable,
                "-u",
                "-c",
                (
                    "import sys; import torch; "
                    "from dcvpp.experiments.v1_forecast_recovery import main; "
                    "raise SystemExit(main(sys.argv[1:]))"
                ),
                phase,
                "--root",
                str(root),
            ]
        else:
            command = [
                sys.executable,
                "-u",
                "-m",
                "dcvpp.experiments.v1_forecast_recovery",
                phase,
                "--root",
                str(root),
            ]
        completed = subprocess.run(
            command,
            cwd=root,
            env=_workload_stage_environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        stdout_path = log_root / f"attempt_{attempt:02d}.stdout.log"
        stderr_path = log_root / f"attempt_{attempt:02d}.stderr.log"
        atomic_text(stdout_path, completed.stdout)
        atomic_text(stderr_path, completed.stderr)
        artifact_verified = False
        probe_error: str | None = None
        try:
            artifact_verified = completion_probe() is not None
        except Exception as exc:
            probe_error = f"{type(exc).__name__}: {exc}"
        record = {
            "attempt": attempt,
            "invocation_attempt": invocation_attempt,
            "phase": phase,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "returncode": int(completed.returncode),
            "native_windows_exit": _native_windows_exit(completed.returncode),
            "artifact_verified": artifact_verified,
            "completion_probe_error": probe_error,
            "stdout": {
                "path": _relative(root, stdout_path),
                "sha256": sha256_file(stdout_path),
                "byte_count": int(stdout_path.stat().st_size),
            },
            "stderr": {
                "path": _relative(root, stderr_path),
                "sha256": sha256_file(stderr_path),
                "byte_count": int(stderr_path.stat().st_size),
            },
        }
        attempt_path = log_root / f"attempt_{attempt:02d}.json"
        record["attempt_fingerprint"] = sha256_json(record)
        atomic_json(attempt_path, record)
        attempts.append(
            {
                "path": _relative(root, attempt_path),
                "sha256": sha256_file(attempt_path),
                **record,
            }
        )
        if completed.returncode == 0 and artifact_verified:
            return {"status": "COMPLETE", "attempts": attempts}
        if artifact_verified:
            return {
                "status": "COMPLETE_WITH_PRESERVED_NATIVE_EXIT",
                "attempts": attempts,
            }
        if not _native_windows_exit(completed.returncode):
            break
    raise V1ForecastRecoveryError(
        f"isolated workload stage failed after {len(attempts)} attempt(s): "
        f"{phase}; attempts={attempts}"
    )


def prepare_workload_checkpoint_adapters(repository_root: Path) -> dict[str, Any]:
    """Prepare isolated fingerprint-adapted copies of valid checkpoints.

    The original v1 checkpoint bytes are never edited. Every adapted byte set
    has an explicit source hash and a separate immutable adapter snapshot.
    Completed checkpoints are reused; valid incomplete checkpoints retain that
    status and continue only inside the recovery tree.
    """

    import torch

    from dcvpp.experiments.paper_training import _workload_factory

    root = Path(repository_root).resolve()
    _, preflight = _load_verified_preflight(root)
    compatibility = preflight.get("cfc_checkpoint_compatibility")
    if not isinstance(compatibility, Mapping):
        raise V1ForecastRecoveryError("preflight lacks CfC compatibility evidence")
    data_fingerprint = str(compatibility.get("recovery_data_fingerprint", ""))
    if len(data_fingerprint) != 64:
        raise V1ForecastRecoveryError("preflight lacks recovery data fingerprint")
    output_root = (
        root
        / "artifacts/real_forecasting_recovery/components"
        / WORKLOAD_COMPONENT_RUN_ID
    )
    adapter_root = output_root / "adapter_snapshots"
    component_checkpoint_root = output_root / "checkpoints/nlr_workload"
    manifest_path = output_root / "checkpoint_adapter_manifest.json"
    if manifest_path.is_file():
        existing = _load_json(manifest_path)
        if existing.get("status") == "COMPLETE":
            for record in existing.get("adapters", []):
                snapshot = root / str(record["adapted_snapshot_path"])
                if not snapshot.is_file() or sha256_file(snapshot) != record["adapted_snapshot_sha256"]:
                    raise V1ForecastRecoveryError("workload adapter snapshot changed")
                destination = root / str(record["component_checkpoint_path"])
                if not destination.is_file():
                    _materialize_copy(snapshot, destination)
            for record in existing.get("prior_failure_evidence", []):
                evidence = root / str(record.get("path", ""))
                if (
                    not evidence.is_file()
                    or sha256_file(evidence) != record.get("sha256")
                ):
                    raise V1ForecastRecoveryError(
                        "preserved workload adapter failure evidence changed"
                    )
            return {
                **existing,
                "path": _relative(root, manifest_path),
                "sha256": sha256_file(manifest_path),
                "resumed": True,
            }

    original_root = root / "artifacts/real_forecasting/paper-forecast-v1"
    adapters: list[dict[str, Any]] = []
    for model_name in ("gru", "aggregate_patchtst", "cfc_w"):
        seeds = STUDY_SEEDS if model_name != "cfc_w" else (STUDY_SEEDS[0],)
        input_size = 14 if model_name == "cfc_w" else 12
        model = _workload_factory(
            model_name,
            input_size=input_size,
            hidden_size=64,
            horizon=96,
        )
        architecture = (
            dict(model.architecture_contract)
            if hasattr(model, "architecture_contract")
            else {
                "encoder": model_name,
                "hidden_size": 64,
                "horizon": 96,
                "cohorts": 4,
                "count_distribution": "negative_binomial",
                "volume_candidates": "lognormal_and_gamma",
            }
        )
        for seed in seeds:
            source = (
                original_root
                / "checkpoints/nlr_workload"
                / model_name
                / f"seed_{seed}_latest.pt"
            )
            if not source.is_file():
                raise V1ForecastRecoveryError(
                    f"intact workload source checkpoint is absent: {source}"
                )
            source_sha = sha256_file(source)
            if model_name == "cfc_w" and source_sha != ORIGINAL_CFC_CHECKPOINT_SHA256:
                raise V1ForecastRecoveryError("CfC source checkpoint hash changed before adaptation")
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            source_contract = _workload_source_checkpoint_contract(
                checkpoint,
                model_name=model_name,
                seed=seed,
                architecture=architecture,
            )
            if model_name == "cfc_w" and source_contract["source_completed"] is not False:
                raise V1ForecastRecoveryError("epoch-59 CfC source is unexpectedly complete")
            original_fingerprint = source_contract["source_resume_fingerprint"]
            adapted_fingerprint = _expected_workload_seed_fingerprint(
                data_fingerprint=data_fingerprint,
                model_name=model_name,
                seed=seed,
                architecture=architecture,
            )
            destination = component_checkpoint_root / model_name / f"seed_{seed}_latest.pt"
            snapshot = adapter_root / model_name / f"seed_{seed}_adapted_source.pt"
            if snapshot.is_file():
                adapted_payload = torch.load(snapshot, map_location="cpu", weights_only=False)
                if adapted_payload.get("resume_fingerprint") != adapted_fingerprint:
                    raise V1ForecastRecoveryError("existing adapter snapshot fingerprint changed")
            else:
                adapted_payload = dict(checkpoint)
                adapted_payload["resume_fingerprint"] = adapted_fingerprint
                adapted_payload["recovery_adapter"] = {
                    "schema_version": "paper_forecast_v1_checkpoint_adapter/v1",
                    "run_id": RUN_ID,
                    "source_path": _relative(root, source),
                    "source_sha256": source_sha,
                    "source_resume_fingerprint": original_fingerprint,
                    "adapted_resume_fingerprint": adapted_fingerprint,
                    "fingerprint_mismatch_explicit": (
                        original_fingerprint != adapted_fingerprint
                    ),
                    "source_epoch": int(checkpoint.get("epoch", 0)),
                    "exact_uninterrupted_trajectory_claimed": False,
                    "dataloader_rng_exact_continuation_claimed": False,
                    "created_at_utc": utc_now(),
                }
                _atomic_torch(snapshot, adapted_payload)
            snapshot_sha = sha256_file(snapshot)
            destination_state = "materialized_adapter"
            if destination.is_file():
                current = torch.load(destination, map_location="cpu", weights_only=False)
                if (
                    current.get("resume_fingerprint") != adapted_fingerprint
                    or int(current.get("epoch", -1)) < int(checkpoint.get("epoch", 0))
                ):
                    raise V1ForecastRecoveryError(
                        f"existing recovery checkpoint is incompatible: {destination}"
                    )
                destination_state = "existing_compatible_progress"
            else:
                _materialize_copy(snapshot, destination)
            adapters.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "source_path": _relative(root, source),
                    "source_sha256": source_sha,
                    "source_epoch": source_contract["source_epoch"],
                    "source_best_epoch": source_contract["source_best_epoch"],
                    "source_completed": source_contract["source_completed"],
                    "source_resume_fingerprint": original_fingerprint,
                    "adapted_resume_fingerprint": adapted_fingerprint,
                    "fingerprint_mismatch_explicit": (
                        original_fingerprint != adapted_fingerprint
                    ),
                    "adapted_snapshot_path": _relative(root, snapshot),
                    "adapted_snapshot_sha256": snapshot_sha,
                    "component_checkpoint_path": _relative(root, destination),
                    "component_checkpoint_state": destination_state,
                    "continue_training": source_contract["continue_training"],
                    "exact_uninterrupted_trajectory_claimed": False,
                }
            )
    prior_failure = _preserve_known_workload_adapter_failure(
        root, output_root, adapters
    )
    incomplete_sources = [
        {
            "model": record["model"],
            "seed": record["seed"],
            "source_path": record["source_path"],
            "source_sha256": record["source_sha256"],
            "source_epoch": record["source_epoch"],
            "source_best_epoch": record["source_best_epoch"],
            "source_completed": False,
            "continue_training": True,
        }
        for record in adapters
        if record["source_completed"] is False
    ]
    payload = {
        "schema_version": "paper_forecast_v1_checkpoint_adapter_manifest/v1",
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "terminal": True,
        "recovery_data_fingerprint": data_fingerprint,
        "adapter_count": len(adapters),
        "adapters": adapters,
        "completed_source_checkpoint_count": sum(
            record["source_completed"] is True for record in adapters
        ),
        "incomplete_source_checkpoint_count": len(incomplete_sources),
        "incomplete_source_checkpoints": incomplete_sources,
        "prior_failure_evidence": [prior_failure] if prior_failure else [],
        "original_v1_files_modified": False,
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "completed_at_utc": utc_now(),
    }
    payload["adapter_manifest_fingerprint"] = sha256_json(payload)
    atomic_json(manifest_path, payload)
    return {
        **payload,
        "path": _relative(root, manifest_path),
        "sha256": sha256_file(manifest_path),
        "resumed": False,
    }


def _verify_component_manifest(
    repository_root: Path,
    path: Path,
    *,
    expected_run_id: str,
    expected_tasks: int,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not path.is_file():
        raise V1ForecastRecoveryError(f"component manifest is absent: {path}")
    value = _load_json(path)
    counts = value.get("counts")
    if (
        value.get("status") != "COMPLETE"
        or value.get("run_id") != expected_run_id
        or not isinstance(counts, Mapping)
        or int(counts.get("tasks_requested", -1)) != expected_tasks
        or int(counts.get("tasks_completed", -1)) != expected_tasks
        or int(counts.get("tasks_failed", -1)) != 0
        or int(counts.get("failed_seeds", -1)) != 0
        or value.get("task_failures") not in (None, [], {})
    ):
        raise V1ForecastRecoveryError(f"component is not COMPLETE: {path}")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise V1ForecastRecoveryError(f"component has no output ledger: {path}")
    for record in outputs:
        if not isinstance(record, Mapping):
            raise V1ForecastRecoveryError("component output record is invalid")
        output = root / str(record.get("path", ""))
        if not output.is_file() or sha256_file(output) != record.get("sha256"):
            raise V1ForecastRecoveryError(f"component output hash mismatch: {output}")
    return value


def run_da_recovery(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    preflight_path, _ = _load_verified_preflight(root)
    component_base = root / "artifacts/real_forecasting_recovery/components"
    component_manifest = component_base / DA_COMPONENT_RUN_ID / "run_manifest.json"
    if component_manifest.is_file():
        value = _verify_component_manifest(
            root,
            component_manifest,
            expected_run_id=DA_COMPONENT_RUN_ID,
            expected_tasks=1,
        )
        return {
            "status": "COMPLETE",
            "resumed": True,
            "run_manifest": _relative(root, component_manifest),
            "run_manifest_sha256": sha256_file(component_manifest),
            "counts": value["counts"],
        }

    from dcvpp.experiments.real_forecasting import run_forecasting_branch

    config = {
        "tasks": ["nyiso_da"],
        "seeds": list(STUDY_SEEDS),
        "output_root": _relative(root, component_base),
        "referenced_manifest": SOURCE_MANIFEST_RELATIVE_PATH,
        "advanced_n_estimators": 500,
        "n_jobs": 8,
        "max_train_rows": None,
        "torch_epochs": 100,
        "torch_batch_size": 32,
        "torch_device": "auto",
        "history_steps": 672,
        "horizon_steps": 96,
        "dgpt_width": 128,
        "minimum_calibration_samples": 500,
        "forecast_draw_count": 100,
        "resume": True,
        "recovery_preflight_path": _relative(root, preflight_path),
        "recovery_preflight_sha256": sha256_file(preflight_path),
    }
    result = run_forecasting_branch(root, config, DA_COMPONENT_RUN_ID)
    value = _verify_component_manifest(
        root,
        component_manifest,
        expected_run_id=DA_COMPONENT_RUN_ID,
        expected_tasks=1,
    )
    return {
        "status": "COMPLETE",
        "resumed": bool(result.get("resumed")),
        "run_manifest": _relative(root, component_manifest),
        "run_manifest_sha256": sha256_file(component_manifest),
        "counts": value["counts"],
    }


def _load_completed_workload_component(
    repository_root: Path,
    manifest_path: Path,
) -> dict[str, Any] | None:
    root = Path(repository_root).resolve()
    if not manifest_path.is_file():
        return None
    value = _load_json(manifest_path)
    if value.get("status") != "COMPLETE":
        return None
    fingerprint = value.get("component_fingerprint")
    payload = {
        key: item for key, item in value.items() if key != "component_fingerprint"
    }
    counts = value.get("counts")
    if (
        value.get("terminal") is not True
        or value.get("run_id") != WORKLOAD_COMPONENT_RUN_ID
        or not isinstance(fingerprint, str)
        or sha256_json(payload) != fingerprint
        or not isinstance(counts, Mapping)
        or int(counts.get("tasks_requested", -1)) != 1
        or int(counts.get("tasks_completed", -1)) != 1
        or int(counts.get("tasks_failed", -1)) != 0
        or int(counts.get("failed_seeds", -1)) != 0
        or int(counts.get("cfc_seeds_requested", -1)) != len(STUDY_SEEDS)
        or int(counts.get("cfc_seeds_completed", -1)) != len(STUDY_SEEDS)
        or value.get("task_failures") not in ([], {})
    ):
        raise V1ForecastRecoveryError(
            "claimed COMPLETE workload component contract is invalid"
        )
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise V1ForecastRecoveryError("workload component output ledger is absent")
    component_root = _workload_component_root(root).resolve()
    seen: set[str] = set()
    for record in outputs:
        if not isinstance(record, Mapping):
            raise V1ForecastRecoveryError("workload component output record is invalid")
        path = root / str(record.get("path", ""))
        try:
            path.resolve().relative_to(component_root)
        except ValueError as exc:
            raise V1ForecastRecoveryError(
                "workload component output escapes its recovery root"
            ) from exc
        relative = _relative(root, path)
        if relative in seen:
            raise V1ForecastRecoveryError(
                "workload component output paths are duplicated"
            )
        seen.add(relative)
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or int(path.stat().st_size) != int(record.get("byte_count", -1))
        ):
            raise V1ForecastRecoveryError("completed workload component output changed")
    required = {
        _relative(root, component_root / "predictions/nlr_workload_cfc_w_predictions.parquet"),
        _relative(root, component_root / "metrics/nlr_workload_paper_leaderboard.json"),
        _relative(root, component_root / "checkpoint_adapter_manifest.json"),
        _relative(root, component_root / "sequence_bundle/manifest.json"),
        _relative(root, component_root / "training_isolation_smoke.json"),
    }
    if not required.issubset(seen):
        raise V1ForecastRecoveryError(
            "workload component required output evidence is incomplete"
        )
    return value


def _load_verified_workload_adapter_manifest(
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _workload_component_root(root) / "checkpoint_adapter_manifest.json"
    if not path.is_file():
        raise V1ForecastRecoveryError("workload checkpoint adapter manifest is absent")
    value = _load_json(path)
    fingerprint = value.get("adapter_manifest_fingerprint")
    payload = {
        key: item
        for key, item in value.items()
        if key != "adapter_manifest_fingerprint"
    }
    if (
        value.get("status") != "COMPLETE"
        or value.get("terminal") is not True
        or value.get("run_id") != RUN_ID
        or not isinstance(fingerprint, str)
        or sha256_json(payload) != fingerprint
    ):
        raise V1ForecastRecoveryError("workload checkpoint adapter contract changed")
    adapters = value.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != 11:
        raise V1ForecastRecoveryError("workload checkpoint adapter count changed")
    for record in adapters:
        source = root / str(record.get("source_path", ""))
        snapshot = root / str(record.get("adapted_snapshot_path", ""))
        destination = root / str(record.get("component_checkpoint_path", ""))
        if not source.is_file() or sha256_file(source) != record.get("source_sha256"):
            raise V1ForecastRecoveryError("workload adapter source hash changed")
        if (
            not snapshot.is_file()
            or sha256_file(snapshot) != record.get("adapted_snapshot_sha256")
        ):
            raise V1ForecastRecoveryError("workload adapter snapshot hash changed")
        if not destination.is_file():
            raise V1ForecastRecoveryError(
                "workload adapter component checkpoint is absent"
            )
    for record in value.get("prior_failure_evidence", []):
        evidence = root / str(record.get("path", ""))
        if not evidence.is_file() or sha256_file(evidence) != record.get("sha256"):
            raise V1ForecastRecoveryError(
                "workload adapter prior-failure evidence changed"
            )
    return {
        **value,
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "resumed": True,
    }


def _workload_sequences_from_bundle(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import pandas as pd

    arrays = bundle.get("loaded_arrays")
    days = bundle.get("days")
    if not isinstance(arrays, Mapping) or not isinstance(days, Mapping):
        raise V1ForecastRecoveryError("loaded workload sequence bundle is incomplete")
    collections: dict[str, dict[str, Any]] = {"aggregate": {}, "event": {}}
    for representation in ("aggregate", "event"):
        representation_days = days.get(representation)
        if not isinstance(representation_days, Mapping):
            raise V1ForecastRecoveryError("workload sequence days are invalid")
        for split in ("train", "validation", "locked_test"):
            collections[representation][split] = (
                arrays[f"{representation}.{split}.x"],
                arrays[f"{representation}.{split}.dt"],
                arrays[f"{representation}.{split}.count"],
                arrays[f"{representation}.{split}.volume"],
                [pd.Timestamp(item) for item in representation_days[split]],
            )
    return collections["aggregate"], collections["event"]


def _recovery_array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _verify_workload_bundle_preflight_lineage(
    bundle: Mapping[str, Any],
    preflight: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    compatibility = preflight.get("cfc_checkpoint_compatibility")
    arrays = bundle.get("loaded_arrays")
    if not isinstance(compatibility, Mapping) or not isinstance(arrays, Mapping):
        raise V1ForecastRecoveryError(
            "workload bundle/preflight lineage inputs are incomplete"
        )
    sequence_manifest = bundle.get("sequence_manifest")
    event_manifest = bundle.get("event_sequence_manifest")
    sequence_manifest_sha = sha256_json(sequence_manifest)
    event_manifest_sha = sha256_json(event_manifest)
    if (
        sequence_manifest_sha != compatibility.get("sequence_manifest_sha256")
        or event_manifest_sha
        != compatibility.get("event_sequence_manifest_sha256")
    ):
        raise V1ForecastRecoveryError(
            "workload sequence manifests no longer match the recovery preflight"
        )
    data_fingerprint = sha256_json(
        {
            "parent": compatibility.get("recovery_parent_fingerprint"),
            "history": 672,
            "horizon": 96,
            "train_x": _recovery_array_sha256(arrays["aggregate.train.x"]),
            "train_dt": _recovery_array_sha256(arrays["aggregate.train.dt"]),
            "train_count": _recovery_array_sha256(
                arrays["aggregate.train.count"]
            ),
            "train_volume": _recovery_array_sha256(
                arrays["aggregate.train.volume"]
            ),
            "validation_x": _recovery_array_sha256(
                arrays["aggregate.validation.x"]
            ),
            "validation_count": _recovery_array_sha256(
                arrays["aggregate.validation.count"]
            ),
            "sequence_manifest": sequence_manifest,
            "event_manifest": event_manifest,
        }
    )
    expected = compatibility.get("recovery_data_fingerprint")
    if (
        data_fingerprint != expected
        or adapter.get("recovery_data_fingerprint") != expected
    ):
        raise V1ForecastRecoveryError(
            "workload bundle fingerprint would bypass adapted checkpoints"
        )
    return {
        "sequence_manifest_sha256": sequence_manifest_sha,
        "event_sequence_manifest_sha256": event_manifest_sha,
        "recovery_data_fingerprint": data_fingerprint,
        "adapter_recovery_data_fingerprint": adapter[
            "recovery_data_fingerprint"
        ],
        "exact_frozen_fingerprint_recomputed": True,
    }


def _strict_verify_workload_recovery_checkpoints(
    repository_root: Path,
    *,
    adapter: Mapping[str, Any],
    recovery_data_fingerprint: str,
    torch_module: Any,
    workload_factory: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(repository_root).resolve()
    checks: list[dict[str, Any]] = []
    forward_models: dict[str, Any] = {}

    def verify_one(
        *,
        model_name: str,
        seed: int,
        path: Path,
        expected_fingerprint: str,
        minimum_epoch: int,
        lineage: str,
    ) -> None:
        progress_prefix = f"checkpoint_gate model={model_name} seed={seed}"
        print(f"{progress_prefix} phase=torch_load", flush=True)
        checkpoint = torch_module.load(
            path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("model_name") != model_name
            or int(checkpoint.get("seed", -1)) != seed
            or checkpoint.get("resume_fingerprint") != expected_fingerprint
            or int(checkpoint.get("epoch", -1)) < minimum_epoch
        ):
            raise V1ForecastRecoveryError(
                f"recovery checkpoint would bypass resume: {model_name}/{seed}"
            )
        input_size = 14 if model_name == "cfc_w" else 12
        model = workload_factory(
            model_name,
            input_size=input_size,
            hidden_size=64,
            horizon=96,
        )
        architecture = (
            dict(model.architecture_contract)
            if hasattr(model, "architecture_contract")
            else {
                "encoder": model_name,
                "hidden_size": 64,
                "horizon": 96,
                "cohorts": 4,
                "count_distribution": "negative_binomial",
                "volume_candidates": "lognormal_and_gamma",
            }
        )
        if dict(checkpoint.get("architecture", {})) != architecture:
            raise V1ForecastRecoveryError(
                f"recovery checkpoint architecture changed: {model_name}/{seed}"
            )
        print(f"{progress_prefix} phase=model_state", flush=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        best_model = workload_factory(
            model_name,
            input_size=input_size,
            hidden_size=64,
            horizon=96,
        )
        print(f"{progress_prefix} phase=best_model_state", flush=True)
        best_model.load_state_dict(checkpoint["best_model_state"], strict=True)
        optimizer = torch_module.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        print(f"{progress_prefix} phase=optimizer_state", flush=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        checks.append(
            {
                "model": model_name,
                "seed": seed,
                "checkpoint_path": _relative(root, path),
                "pre_training_checkpoint_sha256": sha256_file(path),
                "epoch": int(checkpoint["epoch"]),
                "completed": bool(checkpoint.get("completed")),
                "expected_to_change_on_continuation": not bool(
                    checkpoint.get("completed")
                ),
                "checked_at_utc": utc_now(),
                "resume_fingerprint": expected_fingerprint,
                "lineage": lineage,
                "model_state_strict_load": True,
                "best_model_state_strict_load": True,
                "optimizer_state_load": True,
            }
        )
        forward_models.setdefault(model_name, best_model)
        print(f"{progress_prefix} phase=complete", flush=True)

    adapted_keys: set[tuple[str, int]] = set()
    for record in adapter["adapters"]:
        model_name = str(record["model"])
        seed = int(record["seed"])
        adapted_keys.add((model_name, seed))
        verify_one(
            model_name=model_name,
            seed=seed,
            path=root / str(record["component_checkpoint_path"]),
            expected_fingerprint=str(record["adapted_resume_fingerprint"]),
            minimum_epoch=int(record["source_epoch"]),
            lineage="fingerprint_adapted_original_checkpoint",
        )
    cfc_probe = workload_factory(
        "cfc_w", input_size=14, hidden_size=64, horizon=96
    )
    cfc_architecture = dict(cfc_probe.architecture_contract)
    for seed in STUDY_SEEDS:
        if ("cfc_w", seed) in adapted_keys:
            continue
        path = (
            _workload_component_root(root)
            / "checkpoints/nlr_workload/cfc_w"
            / f"seed_{seed}_latest.pt"
        )
        if not path.is_file():
            continue
        expected = _expected_workload_seed_fingerprint(
            data_fingerprint=recovery_data_fingerprint,
            model_name="cfc_w",
            seed=seed,
            architecture=cfc_architecture,
        )
        verify_one(
            model_name="cfc_w",
            seed=seed,
            path=path,
            expected_fingerprint=expected,
            minimum_epoch=1,
            lineage="isolated_recovery_checkpoint_progress",
        )
    return checks, forward_models


def _load_verified_workload_training_smoke(
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _workload_component_root(root) / "training_isolation_smoke.json"
    if not path.is_file():
        raise V1ForecastRecoveryError("workload training isolation smoke is absent")
    value = _load_json(path)
    fingerprint = value.get("smoke_fingerprint")
    payload = {
        key: item for key, item in value.items() if key != "smoke_fingerprint"
    }
    if (
        value.get("schema_version") != WORKLOAD_TRAINING_SMOKE_SCHEMA_VERSION
        or value.get("status") != "COMPLETE"
        or value.get("terminal") is not True
        or value.get("formal_training_started") is not False
        or int(value.get("optimizer_steps", -1)) != 0
        or not isinstance(fingerprint, str)
        or sha256_json(payload) != fingerprint
    ):
        raise V1ForecastRecoveryError("workload training isolation smoke changed")
    adapter = _load_verified_workload_adapter_manifest(root)
    bundle = _load_verified_workload_sequence_bundle(root, load_arrays=False)
    if (
        value.get("checkpoint_adapter_sha256") != adapter["sha256"]
        or value.get("sequence_bundle_sha256") != bundle["sha256"]
        or len(value.get("checkpoint_state_checks", [])) < 11
        or set(value.get("forward_shape_checks", {}))
        != {"gru", "aggregate_patchtst", "cfc_w"}
    ):
        raise V1ForecastRecoveryError(
            "workload training isolation smoke lineage changed"
        )
    return {
        **value,
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "resumed": True,
    }


def run_workload_training_isolation_smoke(
    repository_root: Path,
) -> dict[str, Any]:
    """Strict-load and forward-shape gate; performs no optimizer step."""

    root = Path(repository_root).resolve()
    path = _workload_component_root(root) / "training_isolation_smoke.json"
    if path.is_file():
        return _load_verified_workload_training_smoke(root)

    # Initialize PyTorch's native runtime before NumPy traverses the large
    # sequence bundle.  On this Windows environment, doing this in the reverse
    # order reproducibly faults in load_state_dict (0xC0000005/libiomp5md).
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from dcvpp.experiments.paper_training import _workload_factory

    adapter = _load_verified_workload_adapter_manifest(root)
    _, preflight = _load_verified_preflight(root)
    expected_fingerprint = str(
        preflight["cfc_checkpoint_compatibility"]["recovery_data_fingerprint"]
    )
    state_checks, forward_models = _strict_verify_workload_recovery_checkpoints(
        root,
        adapter=adapter,
        recovery_data_fingerprint=expected_fingerprint,
        torch_module=torch,
        workload_factory=_workload_factory,
    )
    bundle = _load_verified_workload_sequence_bundle(root, load_arrays=True)
    lineage_gate = _verify_workload_bundle_preflight_lineage(
        bundle, preflight, adapter
    )
    if lineage_gate["recovery_data_fingerprint"] != expected_fingerprint:
        raise V1ForecastRecoveryError(
            "workload smoke checkpoint and sequence fingerprints diverged"
        )
    aggregate, event = _workload_sequences_from_bundle(bundle)

    forward_checks: dict[str, Any] = {}
    for model_name, model in forward_models.items():
        sequences = event if model_name == "cfc_w" else aggregate
        x = torch.from_numpy(sequences["train"][0][:1].copy())
        dt = torch.from_numpy(sequences["train"][1][:1].copy())
        model.eval()
        with torch.no_grad():
            output = model(x, dt)
        if not isinstance(output, Mapping) or not output:
            raise V1ForecastRecoveryError(
                f"workload smoke forward output changed: {model_name}"
            )
        forward_checks[model_name] = {
            key: list(value.shape) for key, value in sorted(output.items())
        }
    payload = {
        "schema_version": WORKLOAD_TRAINING_SMOKE_SCHEMA_VERSION,
        "run_id": WORKLOAD_COMPONENT_RUN_ID,
        "status": "COMPLETE",
        "terminal": True,
        "checkpoint_adapter_path": adapter["path"],
        "checkpoint_adapter_sha256": adapter["sha256"],
        "sequence_bundle_path": bundle["path"],
        "sequence_bundle_sha256": bundle["sha256"],
        "preflight_lineage_gate": lineage_gate,
        "checkpoint_state_checks": state_checks,
        "forward_shape_checks": forward_checks,
        "formal_training_started": False,
        "optimizer_steps": 0,
        "loss_values_computed": False,
        "locked_2025_metrics_consumed": False,
        "event_parquet_parsed_or_materialized_with_arrow_in_process": False,
        "event_source_hash_revalidated_by_raw_bytes": True,
        "event_sequence_construction_performed_in_process": False,
        "omp_num_threads": 1,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "kmp_duplicate_lib_ok": False,
        "native_runtime_initialization_order": (
            "torch_and_training_imports_before_numpy_bundle_validation"
        ),
        "sequence_array_loading_mode": "copy_on_write_memory_map",
        "sequence_active_memory_map_count": int(bundle["active_memory_map_count"]),
        "sequence_target_alias_count": len(bundle["in_memory_aliases"]),
        "completed_at_utc": utc_now(),
    }
    payload["smoke_fingerprint"] = sha256_json(payload)
    atomic_json(path, payload)
    return {
        **payload,
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "resumed": False,
    }


def _cfc_acceleration_stage_path(repository_root: Path) -> Path:
    return _workload_component_root(repository_root) / "cfc_cuda_graph_training_stage.json"


def _load_completed_cfc_acceleration_stage(
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _cfc_acceleration_stage_path(root)
    if not path.is_file():
        raise V1ForecastRecoveryError("CfC CUDA-graph training stage is absent")
    payload = _load_json(path)
    if payload.get("status") != "COMPLETE" or payload.get("terminal") is not True:
        raise V1ForecastRecoveryError("CfC CUDA-graph training stage is not COMPLETE")
    if payload.get("seeds") != list(STUDY_SEEDS):
        raise V1ForecastRecoveryError("CfC CUDA-graph stage seed contract changed")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(STUDY_SEEDS):
        raise V1ForecastRecoveryError("CfC CUDA-graph checkpoint ledger is incomplete")
    for record in checkpoints:
        checkpoint = root / str(record.get("path", ""))
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != record.get("sha256")
        ):
            raise V1ForecastRecoveryError("CfC CUDA-graph checkpoint hash changed")
    return {**payload, "path": _relative(root, path), "sha256": sha256_file(path)}


def run_cfc_accelerated_training_phase(repository_root: Path) -> dict[str, Any]:
    """Complete only unfinished CfC seeds in a clean CUDA-graph process."""

    root = Path(repository_root).resolve()
    try:
        completed = _load_completed_cfc_acceleration_stage(root)
        return {"status": "COMPLETE", "resumed": True, **completed}
    except V1ForecastRecoveryError:
        pass

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import dcvpp.experiments.paper_training as paper_training
    from dcvpp.experiments.cfc_cuda_graph_recovery import (
        build_cuda_graphed_cfc_w_recovery,
    )

    _, preflight = _load_verified_preflight(root)
    adapter = _load_verified_workload_adapter_manifest(root)
    compatibility = preflight["cfc_checkpoint_compatibility"]
    _strict_verify_workload_recovery_checkpoints(
        root,
        adapter=adapter,
        recovery_data_fingerprint=str(compatibility["recovery_data_fingerprint"]),
        torch_module=torch,
        workload_factory=paper_training._workload_factory,
    )
    sequence_bundle = _load_verified_workload_sequence_bundle(root, load_arrays=True)
    _, event_sequences = _workload_sequences_from_bundle(sequence_bundle)
    output_root = _workload_component_root(root)
    original_factory = paper_training._workload_factory

    def accelerated_factory(
        model_name: str,
        *,
        input_size: int,
        hidden_size: int,
        horizon: int,
    ):
        if model_name != "cfc_w":
            return original_factory(
                model_name,
                input_size=input_size,
                hidden_size=hidden_size,
                horizon=horizon,
            )
        return build_cuda_graphed_cfc_w_recovery(
            input_size=input_size,
            hidden_size=hidden_size,
            horizon=horizon,
            cohorts=4,
            capture_batch_size=32,
        )

    checkpoint_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    paper_training._workload_factory = accelerated_factory
    try:
        for seed in STUDY_SEEDS:
            try:
                fitted = paper_training._fit_workload_seed(
                    model_name="cfc_w",
                    seed=seed,
                    input_size=event_sequences["train"][0].shape[-1],
                    hidden_size=64,
                    horizon=96,
                    train=event_sequences["train"],
                    validation=event_sequences["validation"],
                    epochs=100,
                    batch_sizes=(32, 16, 8, 4),
                    learning_rate=1e-3,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    output_root=output_root,
                    resume_fingerprint=str(
                        compatibility["recovery_data_fingerprint"]
                    ),
                )
                checkpoint_path = Path(fitted["checkpoint_path"])
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if checkpoint.get("completed") is not True:
                    raise V1ForecastRecoveryError(
                        f"accelerated CfC seed {seed} did not complete"
                    )
                checkpoint_records.append(
                    {
                        "seed": seed,
                        "path": _relative(root, checkpoint_path),
                        "sha256": sha256_file(checkpoint_path),
                        "epoch": int(checkpoint["epoch"]),
                        "best_epoch": int(checkpoint["best_epoch"]),
                        "resumed_from_epoch": int(fitted["resumed_from_epoch"]),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                break
    finally:
        paper_training._workload_factory = original_factory

    acceleration_manifest = output_root / "cfc_cuda_graph_acceleration_manifest.json"
    payload = {
        "schema_version": "cfc_cuda_graph_training_stage/v1",
        "status": "COMPLETE" if not failures and len(checkpoint_records) == len(STUDY_SEEDS) else "FAILED",
        "terminal": True,
        "seeds": list(STUDY_SEEDS),
        "checkpoints": checkpoint_records,
        "failures": failures,
        "acceleration_manifest": {
            "path": _relative(root, acceleration_manifest),
            "sha256": sha256_file(acceleration_manifest),
        },
        "model_math_changed": False,
        "optimizer_changed": False,
        "batch_size_changed": False,
        "artificial_gpu_load_used": False,
        "completed_at_utc": utc_now(),
    }
    payload["stage_fingerprint"] = sha256_json(payload)
    stage_path = _cfc_acceleration_stage_path(root)
    atomic_json(stage_path, payload)
    if payload["status"] != "COMPLETE":
        raise V1ForecastRecoveryError(
            f"accelerated CfC stage failed: {failures}"
        )
    return {**payload, "path": _relative(root, stage_path), "sha256": sha256_file(stage_path)}


def run_workload_training_phase(repository_root: Path) -> dict[str, Any]:
    """Fit without event-Parquet parsing or event-sequence construction."""

    root = Path(repository_root).resolve()
    preflight_path, preflight = _load_verified_preflight(root)
    output_root = _workload_component_root(root)
    manifest_path = output_root / "run_manifest.json"
    completed = _load_completed_workload_component(root, manifest_path)
    if completed is not None:
        return {
            "status": "COMPLETE",
            "resumed": True,
            "run_manifest": _relative(root, manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "counts": completed["counts"],
        }
    _load_completed_cfc_acceleration_stage(root)

    # Keep the native runtime order identical to the successful isolation
    # smoke.  Importing/initializing torch after the large NumPy bundle scan is
    # a preserved, reproducible Windows access-violation path for this run.
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import pandas as pd
    import dcvpp.experiments.paper_training as paper_training
    from dcvpp.experiments.torch_forecasts import fit_cfc_w_five_seeds

    adapter = _load_verified_workload_adapter_manifest(root)
    expected_fingerprint = str(
        preflight["cfc_checkpoint_compatibility"]["recovery_data_fingerprint"]
    )
    resume_checkpoint_gate, _ = _strict_verify_workload_recovery_checkpoints(
        root,
        adapter=adapter,
        recovery_data_fingerprint=expected_fingerprint,
        torch_module=torch,
        workload_factory=paper_training._workload_factory,
    )
    sequence_bundle = _load_verified_workload_sequence_bundle(root, load_arrays=True)
    training_smoke = _load_verified_workload_training_smoke(root)
    lineage_gate = _verify_workload_bundle_preflight_lineage(
        sequence_bundle, preflight, adapter
    )
    if lineage_gate["recovery_data_fingerprint"] != expected_fingerprint:
        raise V1ForecastRecoveryError(
            "workload training checkpoint and sequence fingerprints diverged"
        )
    native_failure = preserve_workload_native_failure_evidence(root)

    panel_path = (
        root
        / "artifacts/real_forecasting/paper-forecast-v1/data/"
        "nlr295_15m_workload.parquet"
    )
    event_path = (
        root
        / "data/processed/research/physical/run_id=paper-physical-v1/"
        "nlr295_workload_events.parquet"
    )
    aggregate_sequences, event_sequences = _workload_sequences_from_bundle(
        sequence_bundle
    )
    panel = pd.DataFrame()
    failed_seeds: list[dict[str, Any]] = []
    compatibility = preflight["cfc_checkpoint_compatibility"]
    original_aggregate_builder = paper_training._workload_sequences
    original_event_builder = paper_training._event_level_cfc_sequences
    original_atomic_parquet = paper_training.atomic_parquet

    def isolated_aggregate_builder(*args: Any, **kwargs: Any):
        if (
            len(args) != 1
            or set(kwargs) != {"history", "horizon"}
            or int(kwargs["history"]) != WORKLOAD_HISTORY
            or int(kwargs["horizon"]) != WORKLOAD_HORIZON
        ):
            raise V1ForecastRecoveryError(
                "frozen aggregate workload sequence call contract changed"
            )
        return aggregate_sequences, sequence_bundle["sequence_manifest"]

    def isolated_event_builder(*args: Any, **kwargs: Any):
        if (
            len(args) != 1
            or set(kwargs)
            != {
                "event_artifact_path",
                "event_artifact_sha256",
                "history_events",
            }
            or Path(kwargs["event_artifact_path"]).resolve() != event_path.resolve()
            or kwargs["event_artifact_sha256"]
            != WORKLOAD_EVENTS_ARTIFACT_SHA256
            or int(kwargs["history_events"]) != WORKLOAD_HISTORY
        ):
            raise V1ForecastRecoveryError(
                "frozen event workload sequence call contract changed"
            )
        return event_sequences, sequence_bundle["event_sequence_manifest"]

    def isolated_atomic_parquet(
        frame: Any, path: Path, *, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Write Parquet in an Arrow-only child, outside the Torch process."""

        destination = Path(path).resolve()
        handoff_root = output_root / "arrow_handoff" / destination.stem
        handoff_root.mkdir(parents=True, exist_ok=True)
        pickle_path = handoff_root / "frame.pkl"
        temporary_pickle = handoff_root / "frame.tmp.pkl"
        frame.to_pickle(temporary_pickle)
        os.replace(temporary_pickle, pickle_path)
        request_path = handoff_root / "request.json"
        atomic_json(
            request_path,
            {
                "schema_version": "arrow_only_parquet_handoff/v1",
                "pickle_path": str(pickle_path),
                "pickle_sha256": sha256_file(pickle_path),
                "output_path": str(destination),
                "metadata": dict(metadata),
                "row_count": len(frame),
                "torch_process_writes_parquet": False,
            },
        )
        child_code = (
            "import json,sys; from pathlib import Path; import pandas as pd; "
            "from dcvpp.experiments.forecast_common import atomic_parquet; "
            "r=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); "
            "f=pd.read_pickle(r['pickle_path']); "
            "print(json.dumps(atomic_parquet(f,Path(r['output_path']),"
            "metadata=r['metadata']),sort_keys=True))"
        )
        completed = subprocess.run(
            [sys.executable, "-u", "-c", child_code, str(request_path)],
            cwd=root,
            env=_workload_stage_environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise V1ForecastRecoveryError(
                "Arrow-only Parquet child failed: " + completed.stderr
            )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        if (
            not destination.is_file()
            or result.get("sha256") != sha256_file(destination)
            or int(result.get("row_count", -1)) != len(frame)
        ):
            raise V1ForecastRecoveryError(
                "Arrow-only Parquet child output did not hash-verify"
            )
        atomic_json(
            handoff_root / "response.json",
            {
                "status": "COMPLETE",
                "request_path": _relative(root, request_path),
                "request_sha256": sha256_file(request_path),
                "result": result,
                "child_stderr": completed.stderr,
                "torch_imported_in_child": False,
            },
        )
        return result

    paper_training._workload_sequences = isolated_aggregate_builder
    paper_training._event_level_cfc_sequences = isolated_event_builder
    paper_training.atomic_parquet = isolated_atomic_parquet
    original_expected_paper_seeds = paper_training.EXPECTED_PAPER_SEEDS
    paper_training.EXPECTED_PAPER_SEEDS = STUDY_SEEDS
    try:
        try:
            result = fit_cfc_w_five_seeds(
                panel,
                seeds=list(STUDY_SEEDS),
                output_root=output_root,
                failed_seeds=failed_seeds,
                history=672,
                horizon=96,
                epochs=100,
                batch_size=32,
                hidden_size=64,
                learning_rate=1e-3,
                device_name="auto",
                resume_fingerprint=str(compatibility["recovery_parent_fingerprint"]),
                minimum_calibration_samples=500,
                event_artifact_path=event_path,
                event_artifact_sha256=WORKLOAD_EVENTS_ARTIFACT_SHA256,
                paper_evidence=True,
            )
        except Exception:
            atomic_json(
                output_root / "cfc_cuda_graph_failed_seed_diagnostics.json",
                {
                    "schema_version": "cfc_cuda_graph_failed_seed_diagnostics/v1",
                    "preserved": True,
                    "checkpoint_advanced": False,
                    "failed_seeds": failed_seeds,
                    "recorded_at_utc": utc_now(),
                },
            )
            raise
    finally:
        paper_training.EXPECTED_PAPER_SEEDS = original_expected_paper_seeds
        paper_training.atomic_parquet = original_atomic_parquet
        paper_training._workload_sequences = original_aggregate_builder
        paper_training._event_level_cfc_sequences = original_event_builder
    atomic_json(output_root / "failed_seeds.json", failed_seeds)
    cfc_trials = result.get("seed_trials")
    completed_cfc_seeds = (
        sum(
            1
            for trial in cfc_trials
            if isinstance(trial, Mapping) and trial.get("status") == "completed"
        )
        if isinstance(cfc_trials, list)
        else 0
    )
    status = (
        "COMPLETE"
        if result.get("full_preregistered_cfc_w_distribution_model_completed") is True
        and completed_cfc_seeds == len(STUDY_SEEDS)
        and not failed_seeds
        else "FAILED"
    )
    generated = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "orchestration" not in path.relative_to(output_root).parts
        and not any(
            part.startswith(".") or part.endswith(".tmp")
            for part in path.relative_to(output_root).parts
        )
        and not (path.suffix == ".pt" and path.name.startswith("tmp"))
    )
    outputs = [
        {
            "path": _relative(root, path),
            "sha256": sha256_file(path),
            "byte_count": int(path.stat().st_size),
        }
        for path in generated
    ]
    payload = {
        "schema_version": "paper_forecast_v1_workload_recovery_component/v1",
        "run_id": WORKLOAD_COMPONENT_RUN_ID,
        "status": status,
        "terminal": status in {"COMPLETE", "FAILED"},
        "source_panel": {
            "path": _relative(root, panel_path),
            "sha256": sha256_file(panel_path),
        },
        "physical_workload_contract_sha256": WORKLOAD_EVENTS_CONTRACT_SHA256,
        "physical_workload_artifact_sha256": WORKLOAD_EVENTS_ARTIFACT_SHA256,
        "preflight": {
            "path": _relative(root, preflight_path),
            "sha256": sha256_file(preflight_path),
        },
        "checkpoint_adapter": {
            "path": adapter["path"],
            "sha256": adapter["sha256"],
            "adapter_count": adapter["adapter_count"],
        },
        "sequence_bundle": {
            "path": sequence_bundle["path"],
            "sha256": sequence_bundle["sha256"],
            "array_count": len(sequence_bundle["arrays"]),
            "event_parquet_parsed_or_materialized_with_arrow_in_training_process": False,
            "event_source_hash_revalidated_by_raw_bytes": True,
            "event_sequence_construction_performed_in_training_process": False,
        },
        "preflight_lineage_gate": lineage_gate,
        "resume_checkpoint_gate": resume_checkpoint_gate,
        "training_isolation_smoke": {
            "path": training_smoke["path"],
            "sha256": training_smoke["sha256"],
            "formal_training_started": False,
            "optimizer_steps": 0,
        },
        "native_failure_evidence": native_failure,
        "process_isolation": {
            "checkpoint_adapter_process_separate": True,
            "arrow_sequence_process_separate": True,
            "event_parquet_arrow_parsing_and_sequence_construction_excluded_from_training_process": True,
            "omp_num_threads": 1,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "kmp_duplicate_lib_ok": False,
            "native_runtime_initialization_order": (
                "torch_and_training_imports_before_numpy_bundle_validation"
            ),
            "sequence_array_loading_mode": "copy_on_write_memory_map",
            "sequence_active_memory_map_count": int(
                sequence_bundle["active_memory_map_count"]
            ),
            "sequence_target_alias_count": len(
                sequence_bundle["in_memory_aliases"]
            ),
        },
        "execution_acceleration": {
            "backend": "torch.cuda.make_graphed_callables",
            "applies_to": "CfCW full training batches only",
            "capture_batch_size": 32,
            "partial_batch_and_evaluation_fallback": "same eager default CfC core",
            "model_math_changed": False,
            "loss_changed": False,
            "optimizer_changed": False,
            "seed_order_changed": False,
            "checkpoint_state_key_order_changed": False,
            "artificial_gpu_load_used": False,
        },
        "cfc_lineage": {
            "source_checkpoint_path": compatibility["source_checkpoint_path"],
            "source_checkpoint_sha256": compatibility["source_checkpoint_sha256"],
            "source_epoch": compatibility["source_epoch"],
            "source_resume_fingerprint": compatibility["source_resume_fingerprint"],
            "adapted_seed_fingerprint": compatibility["adapted_seed_fingerprint"],
            "fingerprint_mismatch_explicit": True,
            "exact_uninterrupted_trajectory_claimed": False,
            "dataloader_rng_exact_continuation_claimed": False,
        },
        "counts": {
            "tasks_requested": 1,
            "tasks_completed": int(status == "COMPLETE"),
            "tasks_failed": int(status != "COMPLETE"),
            "failed_seeds": len(failed_seeds),
            "cfc_seeds_requested": len(STUDY_SEEDS),
            "cfc_seeds_completed": completed_cfc_seeds,
        },
        "task_failures": failed_seeds,
        "outputs": outputs,
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "locked_2025_metric_artifacts_written_by_forecast_evaluator": True,
        "locked_2025_metrics_used_for_training_or_selection": False,
        "completed_at_utc": utc_now(),
    }
    payload["component_fingerprint"] = sha256_json(payload)
    atomic_json(manifest_path, payload)
    if status != "COMPLETE":
        raise V1ForecastRecoveryError("workload recovery completed with preserved failures")
    return {
        "status": "COMPLETE",
        "resumed": False,
        "run_manifest": _relative(root, manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "counts": payload["counts"],
    }


def run_workload_recovery(repository_root: Path) -> dict[str, Any]:
    """Coordinate isolated, bounded workload recovery subprocesses."""

    root = Path(repository_root).resolve()
    output_root = _workload_component_root(root)
    manifest_path = output_root / "run_manifest.json"
    completed = _load_completed_workload_component(root, manifest_path)
    if completed is not None:
        return {
            "status": "COMPLETE",
            "resumed": True,
            "run_manifest": _relative(root, manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "counts": completed["counts"],
        }
    native_failure = preserve_workload_native_failure_evidence(root)

    try:
        _load_verified_workload_adapter_manifest(root)
    except V1ForecastRecoveryError:
        _run_isolated_workload_stage(
            root,
            phase="adapters",
            maximum_attempts=1,
            completion_probe=lambda: _load_verified_workload_adapter_manifest(root),
        )
    try:
        _amend_workload_sequence_bundle_wording(root)
        _load_verified_workload_sequence_bundle(root, load_arrays=False)
    except V1ForecastRecoveryError:
        _run_isolated_workload_stage(
            root,
            phase="workload-sequences",
            maximum_attempts=1,
            completion_probe=lambda: _load_verified_workload_sequence_bundle(
                root, load_arrays=False
            ),
        )
    try:
        _load_verified_workload_training_smoke(root)
    except V1ForecastRecoveryError:
        _run_isolated_workload_stage(
            root,
            phase="workload-smoke",
            maximum_attempts=1,
            completion_probe=lambda: _load_verified_workload_training_smoke(root),
        )
    try:
        _load_completed_cfc_acceleration_stage(root)
    except V1ForecastRecoveryError:
        _run_isolated_workload_stage(
            root,
            phase="workload-cfc-train",
            maximum_attempts=1,
            completion_probe=lambda: _load_completed_cfc_acceleration_stage(root),
        )
    training_stage = _run_isolated_workload_stage(
        root,
        phase="workload-train",
        maximum_attempts=2,
        completion_probe=lambda: _load_completed_workload_component(
            root, manifest_path
        ),
    )
    completed = _load_completed_workload_component(root, manifest_path)
    if completed is None:
        raise V1ForecastRecoveryError(
            "isolated workload training exited without a COMPLETE component"
        )
    return {
        "status": "COMPLETE",
        "resumed": True,
        "run_manifest": _relative(root, manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "counts": completed["counts"],
        "native_failure_evidence": native_failure,
        "training_stage": training_stage,
    }


def _materialize_task_file(
    *,
    repository_root: Path,
    source: Path,
    destination: Path,
    task: str,
    evidence_role: str,
) -> dict[str, Any]:
    if not source.is_file():
        raise V1ForecastRecoveryError(f"recovery source artifact is absent: {source}")
    _materialize_copy(source, destination)
    return _record_for_file(
        repository_root,
        destination,
        task=task,
        evidence_role=evidence_role,
        lineage_source_path=source,
    )


def _materialize_task_tree(
    *,
    repository_root: Path,
    source_root: Path,
    destination_root: Path,
    task: str,
    evidence_role: str,
) -> list[dict[str, Any]]:
    if not source_root.is_dir():
        raise V1ForecastRecoveryError(f"recovery source directory is absent: {source_root}")
    records: list[dict[str, Any]] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        records.append(
            _materialize_task_file(
                repository_root=repository_root,
                source=source,
                destination=destination_root / source.relative_to(source_root),
                task=task,
                evidence_role=evidence_role,
            )
        )
    return records


def _safe_leaderboard_identity(
    repository_root: Path,
    path: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _, validation_sha = _validation_only_leaderboard(path)
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "validation_only_content_sha256": validation_sha,
        "locked_2025_metric_values_loaded": False,
    }


def _source_manifest_record(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = root / SOURCE_MANIFEST_RELATIVE_PATH
    if not path.is_file():
        raise V1ForecastRecoveryError("referenced-data source manifest is absent")
    return {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "byte_count": int(path.stat().st_size),
    }


def _canonical_task_outputs(
    repository_root: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    root = Path(repository_root).resolve()
    canonical = recovery_root(root)
    original = root / "artifacts/real_forecasting/paper-forecast-v1"
    components = root / "artifacts/real_forecasting_recovery/components"
    da_component = components / DA_COMPONENT_RUN_ID
    workload_component = components / WORKLOAD_COMPONENT_RUN_ID
    task_outputs: dict[str, list[dict[str, Any]]] = {
        "nyiso_da": [],
        "nyiso_rt": [],
        "nlr_workload": [],
    }
    recovered: list[dict[str, Any]] = []
    intact: list[dict[str, Any]] = []

    da_specs = (
        ("metrics/nyiso_da_paper_energy_leaderboard.json", "metrics/nyiso_da_paper_energy_leaderboard.json"),
        ("predictions/nyiso_da_predictions.parquet", "predictions/nyiso_da_predictions.parquet"),
        ("predictions/nyiso_da_dgpt_e_quantiles.parquet", "predictions/nyiso_da_dgpt_e_quantiles.parquet"),
        ("models/nyiso_da_models.joblib", "models/nyiso_da_models.joblib"),
    )
    for source_rel, destination_rel in da_specs:
        record = _materialize_task_file(
            repository_root=root,
            source=da_component / source_rel,
            destination=canonical / destination_rel,
            task="nyiso_da",
            evidence_role="isolated_recovery_retraining",
        )
        task_outputs["nyiso_da"].append(record)
        recovered.append(record)
    da_checkpoints = _materialize_task_tree(
        repository_root=root,
        source_root=da_component / "checkpoints/nyiso_da",
        destination_root=canonical / "checkpoints/nyiso_da",
        task="nyiso_da",
        evidence_role="isolated_recovery_retraining",
    )
    task_outputs["nyiso_da"].extend(da_checkpoints)
    recovered.extend(da_checkpoints)

    rt_specs = (
        "metrics/nyiso_rt_paper_energy_leaderboard.json",
        "predictions/nyiso_rt_predictions.parquet",
        "predictions/nyiso_rt_dgpt_e_quantiles.parquet",
        "models/nyiso_rt_models.joblib",
    )
    for relative_path in rt_specs:
        record = _materialize_task_file(
            repository_root=root,
            source=original / relative_path,
            destination=canonical / relative_path,
            task="nyiso_rt",
            evidence_role="hash_verified_intact_original",
        )
        task_outputs["nyiso_rt"].append(record)
        intact.append(record)
    rt_checkpoints = _materialize_task_tree(
        repository_root=root,
        source_root=original / "checkpoints/nyiso_rt",
        destination_root=canonical / "checkpoints/nyiso_rt",
        task="nyiso_rt",
        evidence_role="hash_verified_intact_original",
    )
    task_outputs["nyiso_rt"].extend(rt_checkpoints)
    intact.extend(rt_checkpoints)

    workload_intact_specs = (
        "data/nlr295_15m_workload.parquet",
        "predictions/nlr_workload_predictions.parquet",
        "models/nlr_workload_models.joblib",
    )
    for relative_path in workload_intact_specs:
        record = _materialize_task_file(
            repository_root=root,
            source=original / relative_path,
            destination=canonical / relative_path,
            task="nlr_workload",
            evidence_role="hash_verified_intact_original_baseline",
        )
        task_outputs["nlr_workload"].append(record)
        intact.append(record)
    workload_recovered_specs = (
        "metrics/nlr_workload_paper_leaderboard.json",
        "predictions/nlr_workload_cfc_w_predictions.parquet",
    )
    for relative_path in workload_recovered_specs:
        record = _materialize_task_file(
            repository_root=root,
            source=workload_component / relative_path,
            destination=canonical / relative_path,
            task="nlr_workload",
            evidence_role="epoch59_checkpoint_recovery_and_remaining_seed_training",
        )
        task_outputs["nlr_workload"].append(record)
        recovered.append(record)
    workload_checkpoints = _materialize_task_tree(
        repository_root=root,
        source_root=workload_component / "checkpoints/nlr_workload",
        destination_root=canonical / "checkpoints/nlr_workload",
        task="nlr_workload",
        evidence_role="explicit_fingerprint_adapter_and_recovery_training",
    )
    task_outputs["nlr_workload"].extend(workload_checkpoints)
    recovered.extend(workload_checkpoints)
    return task_outputs, recovered, intact


def validate_recovery_manifest(
    repository_root: Path,
    manifest: Mapping[str, Any] | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path: Path | None = None
    if isinstance(manifest, Path):
        path = manifest.resolve()
        value = _load_json(path)
    else:
        value = dict(manifest)
    required_exact = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "run_kind": "paper_forecast_v1_integrity_recovery",
        "status": "COMPLETE",
        "terminal": True,
        "formal_readiness_allowed": True,
        "actionable_pending_count": 0,
        "measurement_relationship": "not_measured_by_this_study",
        "source_relationship": "third_party_referenced_public_data",
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "locked_2025_policy_decisions_consumed": False,
        "synthetic_data_used": False,
        "synthetic_fallback_used": False,
    }
    for key, expected in required_exact.items():
        if value.get(key) != expected:
            raise V1ForecastRecoveryError(
                f"recovery manifest field {key!r} is not {expected!r}"
            )
    counts = value.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(key, -1)) != expected
        for key, expected in {
            "tasks_requested": 3,
            "tasks_completed": 3,
            "tasks_failed": 0,
            "failed_seeds": 0,
            "corrupt_output_count": EXPECTED_CORRUPTION_COUNT,
        }.items()
    ):
        raise V1ForecastRecoveryError("recovery manifest task/corruption counts are invalid")
    if value.get("integrity_failures") not in (None, [], {}) or value.get("task_failures") not in (None, [], {}):
        raise V1ForecastRecoveryError("recovery manifest contains failures")
    incident = value.get("incident")
    ledger_ref = incident.get("corruption_ledger") if isinstance(incident, Mapping) else None
    if not isinstance(ledger_ref, Mapping) or int(ledger_ref.get("record_count", -1)) != 28:
        raise V1ForecastRecoveryError("recovery manifest corruption-ledger reference is invalid")
    ledger_path = root / str(ledger_ref.get("path", ""))
    if not ledger_path.is_file() or sha256_file(ledger_path) != ledger_ref.get("sha256"):
        raise V1ForecastRecoveryError("recovery manifest corruption ledger changed")
    ledger = _load_json(ledger_path)
    validate_corruption_ledger(root, ledger)
    physical = value.get("physical_recovery_contract")
    if not isinstance(physical, Mapping) or physical.get("status") != "VERIFIED_COMPLETED":
        raise V1ForecastRecoveryError("recovery manifest physical contract is not verified")
    physical_path = root / str(physical.get("path", ""))
    if not physical_path.is_file() or sha256_file(physical_path) != physical.get("sha256"):
        raise V1ForecastRecoveryError("recovery manifest physical repair record changed")
    if (
        physical.get("workload_events_contract_sha256")
        != WORKLOAD_EVENTS_CONTRACT_SHA256
        or physical.get("workload_events_artifact_sha256")
        != WORKLOAD_EVENTS_ARTIFACT_SHA256
    ):
        raise V1ForecastRecoveryError("recovery manifest workload contract identity changed")
    source = value.get("source_original_v1")
    if not isinstance(source, Mapping):
        raise V1ForecastRecoveryError("recovery manifest lacks source_original_v1")
    if (
        source.get("root_path") != "artifacts/real_forecasting/paper-forecast-v1"
        or source.get("original_manifest_path")
        != "artifacts/real_forecasting/paper-forecast-v1/run_manifest.json"
        or source.get("original_manifest_sha256_or_null") is not None
    ):
        raise V1ForecastRecoveryError("original incomplete v1 manifest lineage is misstated")
    lineage = value.get("recovery_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("composition_mode") != "hash_verified_mixed_original_and_recovered_evidence":
        raise V1ForecastRecoveryError("recovery composition mode is invalid")
    if lineage.get("original_intact_tasks") != ["nyiso_rt"]:
        raise V1ForecastRecoveryError("recovery intact task partition is invalid")
    if lineage.get("recovered_tasks") != ["nyiso_da", "nlr_workload"]:
        raise V1ForecastRecoveryError("recovery recovered task partition is invalid")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise V1ForecastRecoveryError("recovery manifest has no canonical output ledger")
    output_paths: set[str] = set()
    for record in outputs:
        if not isinstance(record, Mapping):
            raise V1ForecastRecoveryError("recovery output record is invalid")
        raw_path = str(record.get("path", ""))
        output_path = _canonical_output_path(root, raw_path)
        if raw_path in output_paths:
            raise V1ForecastRecoveryError("recovery output ledger contains duplicates")
        output_paths.add(raw_path)
        if (
            not output_path.is_file()
            or sha256_file(output_path) != record.get("sha256")
            or int(output_path.stat().st_size) != int(record.get("byte_count", -1))
        ):
            raise V1ForecastRecoveryError(f"recovery output changed: {raw_path}")
    corrupt_paths = {str(record["path"]) for record in ledger["records"]}
    if output_paths & corrupt_paths:
        raise V1ForecastRecoveryError("corrupt v1 paths entered the canonical output ledger")
    recovered_declared = value.get("recovered_outputs")
    intact_declared = value.get("intact_outputs")
    if not isinstance(recovered_declared, list) or not isinstance(intact_declared, list):
        raise V1ForecastRecoveryError("recovery intact/recovered output ledgers are absent")
    partition_paths = {
        str(record.get("path", ""))
        for record in [*recovered_declared, *intact_declared]
        if isinstance(record, Mapping)
    }
    if partition_paths != output_paths:
        raise V1ForecastRecoveryError(
            "canonical outputs are not exactly the intact plus recovered partitions"
        )
    recovered_paths = {
        str(record.get("path", ""))
        for record in recovered_declared
        if isinstance(record, Mapping)
    }
    intact_paths = {
        str(record.get("path", ""))
        for record in intact_declared
        if isinstance(record, Mapping)
    }
    if recovered_paths & intact_paths:
        raise V1ForecastRecoveryError("intact and recovered output partitions overlap")
    declared_corrupt = value.get("corrupt_outputs")
    if not isinstance(declared_corrupt, list) or len(declared_corrupt) != 28:
        raise V1ForecastRecoveryError("recovery manifest corrupt_outputs is not exact")
    fingerprint = value.get("manifest_fingerprint")
    fingerprint_payload = {
        key: item for key, item in value.items() if key != "manifest_fingerprint"
    }
    if not isinstance(fingerprint, str) or sha256_json(fingerprint_payload) != fingerprint:
        raise V1ForecastRecoveryError("recovery manifest fingerprint changed")
    return {
        "status": "COMPLETE",
        "verified_output_count": len(outputs),
        "verified_output_bytes": sum(int(item["byte_count"]) for item in outputs),
        "manifest_path": _relative(root, path) if path is not None else None,
        "manifest_sha256": sha256_file(path) if path is not None else None,
    }


def compose_recovery_manifest(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    canonical = recovery_root(root)
    preflight_path, preflight = _load_verified_preflight(root)
    ledger_path = root / str(preflight["corruption_ledger"]["path"])
    ledger = _load_json(ledger_path)
    validate_corruption_ledger(root, ledger)
    physical = verify_physical_recovery_contract(root)

    da_component_path = (
        root
        / "artifacts/real_forecasting_recovery/components"
        / DA_COMPONENT_RUN_ID
        / "run_manifest.json"
    )
    da_component = _verify_component_manifest(
        root,
        da_component_path,
        expected_run_id=DA_COMPONENT_RUN_ID,
        expected_tasks=1,
    )
    workload_component_path = (
        root
        / "artifacts/real_forecasting_recovery/components"
        / WORKLOAD_COMPONENT_RUN_ID
        / "run_manifest.json"
    )
    workload_component = _load_completed_workload_component(root, workload_component_path)
    if workload_component is None:
        raise V1ForecastRecoveryError("workload recovery component is not COMPLETE")

    task_outputs, recovered_outputs, intact_outputs = _canonical_task_outputs(root)
    from dcvpp.experiments.forecast_day_factory import build_frozen_forecast_day_index

    day_contract = dict(physical["workload_events_contract"])
    day_contract["contract_sha256"] = WORKLOAD_EVENTS_CONTRACT_SHA256
    forecast_day_index = build_frozen_forecast_day_index(
        repository_root=root,
        output_root=canonical,
        da_prediction_path=canonical / "predictions/nyiso_da_dgpt_e_quantiles.parquet",
        rt_prediction_path=canonical / "predictions/nyiso_rt_dgpt_e_quantiles.parquet",
        workload_prediction_path=canonical / "predictions/nlr_workload_cfc_w_predictions.parquet",
        pv_panel_path=root / "data/processed/nyiso_zone_pv_15m.parquet",
        draw_count=100,
        physical_workload_contract=day_contract,
    )
    if forecast_day_index.get("status") != "COMPLETE":
        raise V1ForecastRecoveryError("recovery forecast-day index is not COMPLETE")
    day_records = [
        _record_for_file(
            root,
            path,
            task="nyiso_da",
            evidence_role="recovered_composite_counterfactual",
        )
        for path in sorted((canonical / "forecast_days").glob("*"))
        if path.is_file()
    ]
    recovered_outputs.extend(day_records)

    canonical_da_component_manifest = canonical / "lineage/da_component_run_manifest.json"
    canonical_workload_component_manifest = (
        canonical / "lineage/workload_component_run_manifest.json"
    )
    _materialize_copy(da_component_path, canonical_da_component_manifest)
    _materialize_copy(workload_component_path, canonical_workload_component_manifest)

    provenance_records = [
        _record_for_file(
            root,
            preflight_path,
            task="nyiso_da",
            evidence_role="recovery_gate",
        ),
        _record_for_file(
            root,
            ledger_path,
            task="nyiso_da",
            evidence_role="corruption_ledger",
        ),
        _record_for_file(
            root,
            canonical_da_component_manifest,
            task="nyiso_da",
            evidence_role="recovery_component_manifest",
            lineage_source_path=da_component_path,
        ),
        _record_for_file(
            root,
            canonical_workload_component_manifest,
            task="nlr_workload",
            evidence_role="recovery_component_manifest",
            lineage_source_path=workload_component_path,
        ),
    ]
    recovered_outputs.extend(provenance_records)
    task_outputs["nyiso_da"].extend(day_records)
    task_outputs["nyiso_da"].extend(provenance_records[:3])
    task_outputs["nlr_workload"].append(provenance_records[3])
    outputs_by_path = {
        str(record["path"]): record
        for record in (
            [item for values in task_outputs.values() for item in values]
            + day_records
            + provenance_records
        )
    }
    outputs = [outputs_by_path[key] for key in sorted(outputs_by_path)]

    original_root = root / "artifacts/real_forecasting/paper-forecast-v1"
    original_manifest = original_root / "run_manifest.json"
    if original_manifest.exists():
        raise V1ForecastRecoveryError(
            "an original v1 run manifest appeared after the incident; recovery lineage must be re-audited"
        )
    frozen_sources = verify_frozen_sources(root)
    code_hashes = {record["path"]: record["sha256"] for record in frozen_sources}
    source_manifest = _source_manifest_record(root)
    effective_config = {
        "recovery_schema_version": SCHEMA_VERSION,
        "da_component_run_id": DA_COMPONENT_RUN_ID,
        "workload_component_run_id": WORKLOAD_COMPONENT_RUN_ID,
        "seeds": list(STUDY_SEEDS),
        "epochs": 100,
        "batch_size": 32,
        "history_steps": 672,
        "horizon_steps": 96,
        "physical_workload_contract_sha256": WORKLOAD_EVENTS_CONTRACT_SHA256,
        "corruption_ledger_sha256": sha256_file(ledger_path),
    }
    fingerprint = sha256_json(
        {
            "run_id": RUN_ID,
            "effective_config": effective_config,
            "source_manifest_sha256": source_manifest["sha256"],
            "code_hashes": code_hashes,
        }
    )
    corrupt_outputs = []
    for record in ledger["records"]:
        snapshot_path = root / str(record["preserved_snapshot_path"])
        original_path = root / str(record["path"])
        corrupt_outputs.append(
            {
                "path": _relative(root, snapshot_path),
                "sha256": str(record["preserved_snapshot_sha256"]),
                "byte_count": int(snapshot_path.stat().st_size),
                "task": "nyiso_da",
                "evidence_role": "excluded_corrupt_incident_snapshot",
                "lineage_source_path": _relative(root, original_path),
                "lineage_source_sha256": str(record["current_sha256"]),
                "original_corrupt_path": str(record["path"]),
                "current_sha256": str(record["current_sha256"]),
                "preserved_snapshot_path": _relative(root, snapshot_path),
                "preserved_snapshot_sha256": str(
                    record["preserved_snapshot_sha256"]
                ),
                "expected_original_sha256": record.get(
                    "expected_original_sha256"
                ),
                "excluded_from_canonical_outputs": True,
            }
        )
    compatibility = preflight["cfc_checkpoint_compatibility"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "run_kind": "paper_forecast_v1_integrity_recovery",
        "status": "COMPLETE",
        "terminal": True,
        "formal_readiness_allowed": True,
        "actionable_pending_count": 0,
        "integrity_failures": [],
        "task_failures": [],
        "counts": {
            "tasks_requested": 3,
            "tasks_completed": 3,
            "tasks_failed": 0,
            "failed_seeds": 0,
            "recovered_output_count": len(recovered_outputs),
            "intact_output_count": len(intact_outputs),
            "corrupt_output_count": len(corrupt_outputs),
        },
        "four_seed_protocol_amendment": {
            "effective_date": "2026-08-11",
            "confirmatory_seeds": list(STUDY_SEEDS),
            "excluded_legacy_seed": LEGACY_EXCLUDED_SEED,
            "legacy_artifact_policy": "legacy_pre_amendment_excluded_from_confirmatory_four_seed_panel",
            "legacy_seed_outputs_deleted": False,
            "legacy_seed_values_used_in_amended_statistics": False,
        },
        "source_original_v1": {
            "root_path": _relative(root, original_root),
            "original_manifest_path": _relative(root, original_manifest),
            "original_manifest_sha256_or_null": None,
            "original_manifest_status": "NEVER_COMPLETED_OR_WRITTEN_BEFORE_INCIDENT",
            "leaderboards": {
                "nyiso_da": _safe_leaderboard_identity(
                    root,
                    original_root / "metrics/nyiso_da_paper_energy_leaderboard.json",
                ),
                "nyiso_rt": _safe_leaderboard_identity(
                    root,
                    original_root / "metrics/nyiso_rt_paper_energy_leaderboard.json",
                ),
            },
            "frozen_source_files": frozen_sources,
        },
        "incident": {
            "incident_id": INCIDENT_ID,
            "window_utc": ledger["incident_window_utc"],
            "corruption_ledger": {
                "path": _relative(root, ledger_path),
                "sha256": sha256_file(ledger_path),
                "record_count": int(ledger["record_count"]),
            },
            "overwritten_file_count": int(ledger["record_count"]),
            "preservation_mode": ledger["preservation_mode"],
        },
        "physical_recovery_contract": physical,
        "recovery_lineage": {
            "composition_mode": "hash_verified_mixed_original_and_recovered_evidence",
            "original_intact_tasks": ["nyiso_rt"],
            "recovered_tasks": ["nyiso_da", "nlr_workload"],
            "mixed_tasks": ["nlr_workload"],
            "fingerprint_mismatches": [
                {
                    "task": "nlr_workload",
                    "model": "cfc_w",
                    "seed": 20260809,
                    "source_checkpoint_sha256": compatibility[
                        "source_checkpoint_sha256"
                    ],
                    "source_resume_fingerprint": compatibility[
                        "source_resume_fingerprint"
                    ],
                    "adapted_resume_fingerprint": compatibility[
                        "adapted_seed_fingerprint"
                    ],
                    "explicitly_adapted": True,
                    "exact_uninterrupted_trajectory_claimed": False,
                }
            ],
            "locked_2025_policy": {
                "metrics_consumed": False,
                "metrics_materialized": False,
                "strategy_decisions_used": False,
                "forecast_evaluator_artifacts_may_contain_locked_metrics": True,
            },
        },
        "tasks": {
            "nyiso_da": {
                "status": "COMPLETE",
                "evidence_role": "isolated_recovery_retraining",
                "lineage": {
                    "source_mode": "isolated_recovery_component",
                    "source_original_corrupt_outputs_excluded": True,
                    "source_component_manifest_path": _relative(root, da_component_path),
                    "component_manifest_path": _relative(
                        root, canonical_da_component_manifest
                    ),
                    "component_manifest_sha256": sha256_file(
                        canonical_da_component_manifest
                    ),
                },
                "component_manifest_path": _relative(
                    root, canonical_da_component_manifest
                ),
                "component_manifest_sha256": sha256_file(
                    canonical_da_component_manifest
                ),
                "component_counts": da_component["counts"],
                "legacy_seed_20260918": "retained_in_source_component_but_excluded_from_confirmatory_four_seed_panel",
                "outputs": task_outputs["nyiso_da"],
            },
            "nyiso_rt": {
                "status": "COMPLETE",
                "evidence_role": "hash_verified_intact_original",
                "lineage": {
                    "source_mode": "intact_original_outside_incident_execution_scope",
                    "incident_retraining_executed": False,
                    "source_leaderboard_path": _relative(
                        root,
                        original_root / "metrics/nyiso_rt_paper_energy_leaderboard.json",
                    ),
                    "source_leaderboard_sha256": sha256_file(
                        original_root / "metrics/nyiso_rt_paper_energy_leaderboard.json"
                    ),
                },
                "incident_retraining_executed": False,
                "legacy_seed_20260918": "retained_in_source_component_but_excluded_from_confirmatory_four_seed_panel",
                "outputs": task_outputs["nyiso_rt"],
            },
            "nlr_workload": {
                "status": "COMPLETE",
                "evidence_role": "mixed_intact_baseline_and_explicit_checkpoint_recovery",
                "lineage": {
                    "source_mode": "intact_baseline_plus_isolated_checkpoint_recovery",
                    "source_component_manifest_path": _relative(
                        root, workload_component_path
                    ),
                    "component_manifest_path": _relative(
                        root, canonical_workload_component_manifest
                    ),
                    "component_manifest_sha256": sha256_file(
                        canonical_workload_component_manifest
                    ),
                    "epoch59_source_checkpoint_path": compatibility[
                        "source_checkpoint_path"
                    ],
                    "epoch59_source_checkpoint_sha256": compatibility[
                        "source_checkpoint_sha256"
                    ],
                    "fingerprint_adapter_used": True,
                    "exact_uninterrupted_trajectory_claimed": False,
                },
                "component_manifest_path": _relative(
                    root, canonical_workload_component_manifest
                ),
                "component_manifest_sha256": sha256_file(
                    canonical_workload_component_manifest
                ),
                "component_counts": workload_component["counts"],
                "epoch59_checkpoint_resumed": True,
                "exact_uninterrupted_trajectory_claimed": False,
                "outputs": task_outputs["nlr_workload"],
            },
        },
        "recovered_outputs": sorted(recovered_outputs, key=lambda item: item["path"]),
        "intact_outputs": sorted(intact_outputs, key=lambda item: item["path"]),
        "corrupt_outputs": corrupt_outputs,
        "outputs": outputs,
        "forecast_day_index": forecast_day_index,
        "forecast_decision_records": [
            {
                "forecast_day_index": forecast_day_index["index_path"],
                "day_count": forecast_day_index["day_count"],
                "draw_count": forecast_day_index["draw_count"],
                "horizon_intervals": 96,
                "status": forecast_day_index["status"],
                "source": "paper_forecast_v1_integrity_recovery",
            }
        ],
        "source_manifest": source_manifest,
        "code_hashes": code_hashes,
        "effective_config": effective_config,
        "fingerprint": fingerprint,
        "started_from_real_referenced_data": True,
        "synthetic_data_used": False,
        "synthetic_fallback_used": False,
        "paper_experiment_claimed": True,
        "paper_results_available": True,
        "measurement_relationship": "not_measured_by_this_study",
        "source_relationship": "third_party_referenced_public_data",
        "evidence_role": "referenced_public_counterfactual_input",
        "locked_2025_metrics_consumed": False,
        "locked_2025_metrics_materialized": False,
        "locked_2025_policy_decisions_consumed": False,
        "locked_2025_metric_artifacts_written_by_forecast_evaluator": True,
        "completed_at_utc": utc_now(),
    }
    payload["manifest_fingerprint"] = sha256_json(payload)
    manifest_path = canonical / "run_manifest.json"
    atomic_json(manifest_path, payload)
    validation = validate_recovery_manifest(root, manifest_path)
    return {
        **payload,
        "run_manifest": _relative(root, manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "validation": validation,
        "resumed": False,
    }


def run_full_recovery(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    run_recovery_preflight(root)
    run_da_recovery(root)
    run_workload_recovery(root)
    return compose_recovery_manifest(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated, evidence-gated paper-forecast-v1 recovery"
    )
    parser.add_argument(
        "phase",
        choices=(
            "ledger",
            "checkpoint",
            "preflight",
            "adapters",
            "workload-sequences",
            "workload-smoke",
            "workload-cfc-train",
            "workload-train",
            "native-evidence",
            "da",
            "workload",
            "compose",
            "all",
            "validate",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.phase == "ledger":
        result = build_corruption_ledger(root)
    elif args.phase == "checkpoint":
        result = write_cfc_checkpoint_state_gate(root)
    elif args.phase == "preflight":
        result = run_recovery_preflight(root)
    elif args.phase == "adapters":
        result = prepare_workload_checkpoint_adapters(root)
    elif args.phase == "workload-sequences":
        result = prepare_workload_sequence_bundle(root)
    elif args.phase == "workload-smoke":
        result = run_workload_training_isolation_smoke(root)
    elif args.phase == "workload-cfc-train":
        result = run_cfc_accelerated_training_phase(root)
    elif args.phase == "workload-train":
        result = run_workload_training_phase(root)
    elif args.phase == "native-evidence":
        result = preserve_workload_native_failure_evidence(root)
    elif args.phase == "da":
        result = run_da_recovery(root)
    elif args.phase == "workload":
        result = run_workload_recovery(root)
    elif args.phase == "compose":
        result = compose_recovery_manifest(root)
    elif args.phase == "validate":
        result = validate_recovery_manifest(root, recovery_root(root) / "run_manifest.json")
    else:
        result = run_full_recovery(root)
    summary = {
        key: result.get(key)
        for key in (
            "status",
            "terminal",
            "formal_readiness_allowed",
            "record_count",
            "run_manifest",
            "run_manifest_sha256",
            "path",
            "sha256",
            "resumed",
        )
        if key in result
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DA_COMPONENT_RUN_ID",
    "EXPECTED_CORRUPTION_COUNT",
    "RUN_ID",
    "SCHEMA_VERSION",
    "V1ForecastRecoveryError",
    "WORKLOAD_COMPONENT_RUN_ID",
    "build_corruption_ledger",
    "compose_recovery_manifest",
    "prepare_workload_checkpoint_adapters",
    "prepare_workload_sequence_bundle",
    "preserve_workload_native_failure_evidence",
    "recovery_root",
    "run_da_recovery",
    "run_full_recovery",
    "run_recovery_preflight",
    "run_workload_recovery",
    "run_workload_training_phase",
    "validate_corruption_ledger",
    "validate_recovery_manifest",
    "verify_cfc_checkpoint_compatibility",
    "verify_frozen_sources",
    "verify_physical_recovery_contract",
    "write_cfc_checkpoint_state_gate",
]
