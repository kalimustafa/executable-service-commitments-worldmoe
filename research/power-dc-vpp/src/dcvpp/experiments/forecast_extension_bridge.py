from __future__ import annotations

"""Read-only bridge from post-v1 forecast extensions to optimization.

The bridge never trains, repairs, or rewrites either forecasting run.  It
returns an optimization-facing result only after every extension evidence gate
has been verified from disk.  Missing evidence is reported as actionable
pending; contradictory hashes are reported as integrity failures.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from dcvpp.core import sha256_file, sha256_json
from dcvpp.experiments.paper_forecast_extensions import (
    ENERGY_VARIANTS,
    PREREGISTERED_SEEDS,
    WORKLOAD_VARIANTS,
)
from dcvpp.research_branch.contracts import FrozenForecastArtifact, NYISO_ZONES


ENERGY_TASKS = ("da_lbmp", "rt_lbmp")
TASK_TO_HYPOTHESIS_NAME = {"da_lbmp": "nyiso_da", "rt_lbmp": "nyiso_rt"}
PRIMARY_ENERGY_VARIANT = "dgpt_e_primary"
ORACLE_ENERGY_VARIANT = "ORACLE_future_realized_weather"
PRIMARY_WORKLOAD_VARIANT = "cfc_w_primary"
WORKLOAD_H2_INELIGIBLE = {"no_volume_distribution_head"}
EXPECTED_CELL_COUNT = len(ENERGY_TASKS) * len(ENERGY_VARIANTS) + len(WORKLOAD_VARIANTS)
EXPECTED_SEED_RUN_COUNT = EXPECTED_CELL_COUNT * len(PREREGISTERED_SEEDS)
EXPECTED_EXTENSION_RUN_ID = "paper-forecast-extensions-v1"


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _expected_2025_dates() -> tuple[str, ...]:
    cursor = date(2025, 1, 1)
    end = date(2026, 1, 1)
    values: list[str] = []
    while cursor < end:
        values.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(values)


def _root_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _manifest_path(root: Path, value: str | Path, *, name: str) -> Path:
    if isinstance(value, Mapping):
        raise TypeError(f"{name} must be a path so its read-only hash can be verified")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{name} must remain inside the repository root")
    return path


def _artifact_path(
    root: Path,
    manifest_path: Path,
    raw: str | Path,
) -> Path:
    path = Path(raw)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        repository_candidate = (root / path).resolve()
        sibling_candidate = (manifest_path.parent / path).resolve()
        resolved = (
            repository_candidate
            if repository_candidate.exists() or not sibling_candidate.exists()
            else sibling_candidate
        )
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact path escapes repository root: {raw}")
    return resolved


@dataclass
class _Audit:
    actionable_pending: list[dict[str, Any]] = field(default_factory=list)
    integrity_failures: list[dict[str, Any]] = field(default_factory=list)

    def pending(self, code: str, detail: str, **evidence: Any) -> None:
        self.actionable_pending.append(
            {"code": code, "detail": detail, **evidence}
        )

    def integrity(self, code: str, detail: str, **evidence: Any) -> None:
        self.integrity_failures.append(
            {"code": code, "detail": detail, **evidence}
        )

    def require(
        self,
        condition: bool,
        code: str,
        detail: str,
        **evidence: Any,
    ) -> bool:
        if not condition:
            self.pending(code, detail, **evidence)
            return False
        return True


def _read_json(path: Path, audit: _Audit, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        audit.pending(f"{label.upper()}_MANIFEST_MISSING", f"manifest is absent: {path}")
        return {}, ""
    digest = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit.integrity(
            f"{label.upper()}_MANIFEST_INVALID_JSON",
            f"{type(exc).__name__}: {exc}",
            path=str(path),
        )
        return {}, digest
    if not isinstance(payload, dict):
        audit.integrity(
            f"{label.upper()}_MANIFEST_NOT_MAPPING",
            "manifest JSON root must be a mapping",
            path=str(path),
        )
        return {}, digest
    return payload, digest


def _valid_sha(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text)


def _verify_artifact_record(
    record: Mapping[str, Any] | None,
    *,
    root: Path,
    manifest_path: Path,
    audit: _Audit,
    code_prefix: str,
    required_suffix: str | None = None,
) -> Path | None:
    if not isinstance(record, Mapping) or not record.get("path"):
        audit.pending(f"{code_prefix}_RECORD_MISSING", "artifact path/hash record is missing")
        return None
    raw_hash = record.get("sha256")
    if not _valid_sha(raw_hash):
        audit.pending(
            f"{code_prefix}_HASH_UNDECLARED",
            "artifact does not declare a valid SHA-256",
            path=str(record.get("path")),
        )
        return None
    try:
        path = _artifact_path(root, manifest_path, str(record["path"]))
    except Exception as exc:
        audit.integrity(
            f"{code_prefix}_PATH_INVALID",
            str(exc),
            path=str(record.get("path")),
        )
        return None
    if required_suffix is not None and path.suffix.lower() != required_suffix.lower():
        audit.integrity(
            f"{code_prefix}_TYPE_INVALID",
            f"artifact must be {required_suffix}",
            path=str(path),
        )
        return None
    if not path.is_file():
        audit.pending(f"{code_prefix}_FILE_MISSING", f"artifact is absent: {path}")
        return None
    actual = sha256_file(path)
    if actual.lower() != str(raw_hash).lower():
        audit.integrity(
            f"{code_prefix}_HASH_MISMATCH",
            "artifact bytes disagree with the declared SHA-256",
            path=str(path),
            expected_sha256=str(raw_hash),
            actual_sha256=actual,
        )
        return None
    return path


def _verify_v1(
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    audit: _Audit,
) -> dict[str, Any]:
    status_complete = str(manifest.get("status", "")).upper() == "COMPLETE"
    audit.require(
        status_complete,
        "V1_NOT_COMPLETE",
        "v1 status must be COMPLETE",
        observed=manifest.get("status"),
    )
    run_id = str(manifest.get("run_id", ""))
    is_integrity_recovery = (
        run_id == "paper-forecast-v1-recovery-v1"
        and manifest.get("run_kind") == "paper_forecast_v1_integrity_recovery"
    )
    source_original = manifest.get("source_original_v1", {})
    recovery_contract_valid = bool(
        is_integrity_recovery
        and manifest.get("terminal") is True
        and manifest.get("formal_readiness_allowed") is True
        and int(manifest.get("actionable_pending_count", -1)) == 0
        and not manifest.get("integrity_failures")
        and not manifest.get("task_failures")
        and manifest.get("locked_2025_metrics_consumed") is False
        and manifest.get("locked_2025_metrics_materialized") is False
        and manifest.get("locked_2025_policy_decisions_consumed") is False
        and isinstance(source_original, Mapping)
        and source_original.get("root_path")
        == "artifacts/real_forecasting/paper-forecast-v1"
        and source_original.get("original_manifest_path")
        == "artifacts/real_forecasting/paper-forecast-v1/run_manifest.json"
    )
    audit.require(
        run_id == "paper-forecast-v1" or recovery_contract_valid,
        "V1_RUN_ID_INVALID",
        "v1 must be the original run or its canonical hash-verified integrity recovery",
        observed=manifest.get("run_id"),
    )
    audit.require(
        manifest.get("synthetic_fallback_used") is False,
        "V1_SYNTHETIC_STATUS_UNSAFE",
        "v1 must explicitly declare synthetic_fallback_used=false",
        observed=manifest.get("synthetic_fallback_used"),
    )
    counts = manifest.get("counts", {})
    task_count_gate = False
    failed_seed_gate = False
    if isinstance(counts, Mapping):
        task_count_gate = bool(
            int(counts.get("tasks_requested", -1)) == 3
            and int(counts.get("tasks_completed", -1)) == 3
            and int(counts.get("tasks_failed", -1)) == 0
        )
        failed_seed_gate = int(counts.get("failed_seeds", -1)) == 0
        audit.require(
            task_count_gate,
            "V1_TASK_COUNTS_INCOMPLETE",
            "v1 requires exactly three requested/completed tasks and zero failed tasks",
            counts=dict(counts),
        )
        audit.require(
            failed_seed_gate,
            "V1_FAILED_SEEDS_PRESENT",
            "v1 contains failed seeds",
            counts=dict(counts),
        )
    task_failures = manifest.get("task_failures", [])
    task_failure_gate = task_failures in (None, [], {})
    audit.require(
        task_failure_gate,
        "V1_TASK_FAILURE_RECORDS_PRESENT",
        "v1 task_failures must be empty",
        observed=task_failures,
    )
    outputs = manifest.get("outputs", [])
    audit.require(
        isinstance(outputs, list) and bool(outputs),
        "V1_OUTPUT_HASH_LEDGER_MISSING",
        "v1 requires a nonempty output hash ledger",
    )
    verified = 0
    if isinstance(outputs, list):
        for index, record in enumerate(outputs):
            path = _verify_artifact_record(
                record if isinstance(record, Mapping) else None,
                root=root,
                manifest_path=manifest_path,
                audit=audit,
                code_prefix=f"V1_OUTPUT_{index}",
            )
            verified += int(path is not None)
    output_gate = bool(isinstance(outputs, list) and outputs and verified == len(outputs))

    source_record = manifest.get("source_manifest")
    source_path = _verify_artifact_record(
        source_record if isinstance(source_record, Mapping) else None,
        root=root,
        manifest_path=manifest_path,
        audit=audit,
        code_prefix="V1_SOURCE_MANIFEST",
    )
    source_gate = source_path is not None
    source_sha = (
        str(source_record.get("sha256"))
        if isinstance(source_record, Mapping) and source_record.get("sha256")
        else ""
    )

    code_hashes = manifest.get("code_hashes", {})
    code_verified = 0
    if not isinstance(code_hashes, Mapping) or not code_hashes:
        audit.pending(
            "V1_CODE_HASH_CONTRACT_MISSING",
            "v1 requires a nonempty code_hashes mapping",
        )
    else:
        for raw_path, declared_hash in code_hashes.items():
            try:
                code_path = _artifact_path(root, manifest_path, str(raw_path))
            except Exception as exc:
                audit.integrity(
                    "V1_CODE_PATH_INVALID",
                    str(exc),
                    path=str(raw_path),
                )
                continue
            if not _valid_sha(declared_hash):
                audit.pending(
                    "V1_CODE_HASH_UNDECLARED",
                    "v1 code hash is not a valid SHA-256",
                    path=str(raw_path),
                )
                continue
            if not code_path.is_file():
                audit.pending(
                    "V1_CODE_FILE_MISSING",
                    f"v1 frozen code file is absent: {code_path}",
                )
                continue
            actual = sha256_file(code_path)
            if actual.lower() != str(declared_hash).lower():
                audit.integrity(
                    "V1_CODE_HASH_MISMATCH",
                    "v1 frozen code bytes disagree with code_hashes",
                    path=str(code_path),
                    expected_sha256=str(declared_hash),
                    actual_sha256=actual,
                )
                continue
            code_verified += 1
    code_gate = bool(
        isinstance(code_hashes, Mapping)
        and code_hashes
        and code_verified == len(code_hashes)
    )

    fingerprint = str(manifest.get("fingerprint", ""))
    expected_fingerprint = sha256_json(
        {
            "run_id": manifest.get("run_id"),
            "effective_config": manifest.get("effective_config"),
            "source_manifest_sha256": source_sha,
            "code_hashes": dict(code_hashes) if isinstance(code_hashes, Mapping) else {},
        }
    )
    fingerprint_gate = bool(
        _valid_sha(fingerprint) and fingerprint.lower() == expected_fingerprint.lower()
    )
    audit.require(
        fingerprint_gate,
        "V1_FINGERPRINT_CONTRACT_MISMATCH",
        "v1 fingerprint must match run_id/config/source/code hashes",
        declared=fingerprint or None,
        recomputed=expected_fingerprint,
    )

    terminal_field_present = "terminal" in manifest
    explicit_terminal = manifest.get("terminal") is True
    inferred_terminal = bool(
        not terminal_field_present
        and status_complete
        and task_count_gate
        and failed_seed_gate
        and task_failure_gate
        and output_gate
        and source_gate
        and code_gate
        and fingerprint_gate
    )
    audit.require(
        explicit_terminal or inferred_terminal,
        "V1_FORECAST_BRANCH_NOT_TERMINAL",
        (
            "v1 must explicitly declare terminal=true or satisfy the producer-native "
            "forecast-branch COMPLETE task/seed/output/source/code/fingerprint gates"
        ),
        terminal_field_present=terminal_field_present,
        terminal_value=manifest.get("terminal"),
    )
    return {
        "run_id": manifest.get("run_id"),
        "manifest_path": _root_relative(manifest_path, root),
        "manifest_sha256": manifest_sha256,
        "verified_output_count": verified,
        "verified_code_hash_count": code_verified,
        "source_manifest_sha256": source_sha or None,
        "fingerprint": fingerprint or None,
        "terminal_field_present": terminal_field_present,
        "terminal_inferred_from_complete_hash_gates": inferred_terminal,
        "inference_scope": "forecast_branch_only",
        "read_only": True,
    }


def _trial_groups(extension: Mapping[str, Any]) -> tuple[Any, Any]:
    energy = extension.get("energy", {})
    workload = extension.get("workload", {})
    return (
        energy.get("tasks", {}) if isinstance(energy, Mapping) else {},
        workload.get("variants", {}) if isinstance(workload, Mapping) else {},
    )


def _prediction_arrays_valid(
    path: Path,
    *,
    task: str,
    locked_hash: str,
    audit: _Audit,
    code_prefix: str,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            if task in ENERGY_TASKS:
                required = {
                    "validation_prediction",
                    "locked_test_prediction",
                    "validation_target",
                    "locked_test_target",
                }
                missing = sorted(required - set(data.files))
                if missing:
                    raise ValueError(f"missing arrays: {missing}")
                target = np.asarray(data["locked_test_target"])
                if target.size == 0 or not np.isfinite(target).all():
                    raise ValueError("locked-test target is empty or non-finite")
                observed = _array_sha256(target)
                if observed != locked_hash:
                    raise ValueError(
                        f"locked target hash {observed} disagrees with trial {locked_hash}"
                    )
            else:
                required = {
                    "validation_count_prediction",
                    "locked_test_count_prediction",
                    "validation_count_target",
                    "locked_test_count_target",
                }
                missing = sorted(required - set(data.files))
                if missing:
                    raise ValueError(f"missing arrays: {missing}")
                target = np.asarray(data["locked_test_count_target"])
                if target.size == 0 or not np.isfinite(target).all():
                    raise ValueError("locked-test workload target is empty or non-finite")
    except Exception as exc:
        audit.integrity(
            f"{code_prefix}_NPZ_INVALID",
            f"{type(exc).__name__}: {exc}",
            path=str(path),
        )
        return False
    return True


def _verify_extension_trials(
    root: Path,
    manifest_path: Path,
    extension: Mapping[str, Any],
    audit: _Audit,
) -> dict[str, Any]:
    energy = extension.get("energy", {})
    workload = extension.get("workload", {})
    audit.require(
        isinstance(energy, Mapping) and energy.get("status") == "COMPLETE",
        "EXTENSION_ENERGY_NOT_COMPLETE",
        "extension energy summary must be COMPLETE",
    )
    audit.require(
        isinstance(workload, Mapping) and workload.get("status") == "COMPLETE",
        "EXTENSION_WORKLOAD_NOT_COMPLETE",
        "extension workload summary must be COMPLETE",
    )
    expected_seeds = tuple(PREREGISTERED_SEEDS)
    energy_tasks, workload_variants = _trial_groups(extension)
    audit.require(
        isinstance(energy_tasks, Mapping) and set(energy_tasks) == set(ENERGY_TASKS),
        "EXTENSION_ENERGY_TASK_SET_INCOMPLETE",
        "extension must contain separate DA and RT tasks",
        observed=sorted(energy_tasks) if isinstance(energy_tasks, Mapping) else None,
    )
    expected_energy_variants = set(ENERGY_VARIANTS)
    expected_workload_variants = set(WORKLOAD_VARIANTS)
    trial_count = 0
    prediction_paths: list[str] = []
    primary_prediction_hashes: dict[str, set[str]] = {
        "da_lbmp": set(),
        "rt_lbmp": set(),
        "nlr_workload": set(),
    }
    locked_hashes: dict[str, set[str]] = {
        "da_lbmp": set(),
        "rt_lbmp": set(),
        "nlr_workload": set(),
    }
    if isinstance(energy_tasks, Mapping):
        for task in ENERGY_TASKS:
            variants = energy_tasks.get(task, {})
            if not isinstance(variants, Mapping) or set(variants) != expected_energy_variants:
                audit.pending(
                    f"{task.upper()}_VARIANT_SET_INCOMPLETE",
                    "energy task does not contain all eight preregistered variants",
                    observed=sorted(variants) if isinstance(variants, Mapping) else None,
                )
                continue
            for variant_name in sorted(expected_energy_variants):
                group = variants[variant_name]
                prefix = f"{task.upper()}_{variant_name.upper()}"
                if not isinstance(group, Mapping) or group.get("status") != "COMPLETE":
                    audit.pending(f"{prefix}_NOT_COMPLETE", "variant group must be COMPLETE")
                    continue
                trials = group.get("seed_trials", [])
                observed_seeds = tuple(
                    sorted(int(trial.get("seed")) for trial in trials if isinstance(trial, Mapping))
                )
                audit.require(
                    observed_seeds == tuple(sorted(expected_seeds))
                    and len(trials) == len(expected_seeds),
                    f"{prefix}_SEEDS_INCOMPLETE",
                    "variant requires exactly the three amended seeds",
                    observed=list(observed_seeds),
                )
                for trial in trials if isinstance(trials, list) else []:
                    if not isinstance(trial, Mapping):
                        audit.integrity(f"{prefix}_TRIAL_INVALID", "seed trial is not a mapping")
                        continue
                    trial_count += 1
                    seed = int(trial.get("seed", -1))
                    trial_prefix = f"{prefix}_SEED_{seed}"
                    audit.require(
                        trial.get("status") == "COMPLETE",
                        f"{trial_prefix}_NOT_COMPLETE",
                        "seed trial must be COMPLETE",
                    )
                    audit.require(
                        trial.get("locked_test_used_for_selection_or_calibration") is False,
                        f"{trial_prefix}_LOCKED_SELECTION_UNSAFE",
                        "locked test must not be used for selection or calibration",
                    )
                    checkpoint = trial.get("checkpoint", {})
                    audit.require(
                        isinstance(checkpoint, Mapping)
                        and checkpoint.get("reload_prediction_bitwise_equal") is True,
                        f"{trial_prefix}_CHECKPOINT_RELOAD_UNVERIFIED",
                        "checkpoint reload equality must be true",
                    )
                    locked_hash = str(trial.get("locked_test_target_sha256", ""))
                    if _valid_sha(locked_hash):
                        locked_hashes[task].add(locked_hash)
                    else:
                        audit.pending(
                            f"{trial_prefix}_LOCKED_HASH_MISSING",
                            "seed trial lacks a locked-test target hash",
                        )
                    prediction_record = trial.get("prediction_artifact")
                    prediction_path = _verify_artifact_record(
                        prediction_record if isinstance(prediction_record, Mapping) else None,
                        root=root,
                        manifest_path=manifest_path,
                        audit=audit,
                        code_prefix=f"{trial_prefix}_PREDICTION",
                        required_suffix=".npz",
                    )
                    if prediction_path is not None:
                        prediction_paths.append(_root_relative(prediction_path, root))
                        _prediction_arrays_valid(
                            prediction_path,
                            task=task,
                            locked_hash=locked_hash,
                            audit=audit,
                            code_prefix=f"{trial_prefix}_PREDICTION",
                        )
                        if variant_name == PRIMARY_ENERGY_VARIANT:
                            primary_prediction_hashes[task].add(
                                str(prediction_record["sha256"]).lower()
                            )
                    calibration = trial.get("calibration", {})
                    if variant_name == "no_conformal_calibration":
                        audit.require(
                            isinstance(calibration, Mapping)
                            and calibration.get("method")
                            == "disabled_by_no_conformal_calibration_ablation"
                            and calibration.get("locked_test_used") is False,
                            f"{trial_prefix}_CALIBRATION_ABLATION_INVALID",
                            "no-conformal variant must explicitly disable calibration",
                        )
                    else:
                        audit.require(
                            isinstance(calibration, Mapping)
                            and calibration.get("fit_split") == "validation"
                            and calibration.get("locked_test_used") is False,
                            f"{trial_prefix}_CALIBRATION_INVALID",
                            "calibration must use validation only",
                        )
                    if variant_name == ORACLE_ENERGY_VARIANT:
                        audit.require(
                            trial.get("evidence_role") == "oracle"
                            and trial.get("h1_eligible") is False
                            and trial.get("deployable") is False
                            and trial.get("future_realized_weather_consumed") is True,
                            f"{trial_prefix}_ORACLE_NOT_ISOLATED",
                            "ORACLE must be nondeployable and excluded from H1",
                        )
                    elif variant_name == PRIMARY_ENERGY_VARIANT:
                        audit.require(
                            trial.get("h1_eligible") is True
                            and trial.get("future_realized_weather_consumed") is False,
                            f"{trial_prefix}_PRIMARY_ROLE_INVALID",
                            "primary must be H1-eligible and non-ORACLE",
                        )
    if not isinstance(workload_variants, Mapping) or set(workload_variants) != expected_workload_variants:
        audit.pending(
            "WORKLOAD_VARIANT_SET_INCOMPLETE",
            "workload task does not contain all three amended-seed variants",
            observed=(sorted(workload_variants) if isinstance(workload_variants, Mapping) else None),
        )
    elif isinstance(workload_variants, Mapping):
        for variant_name in sorted(expected_workload_variants):
            group = workload_variants[variant_name]
            prefix = f"WORKLOAD_{variant_name.upper()}"
            if not isinstance(group, Mapping) or group.get("status") != "COMPLETE":
                audit.pending(f"{prefix}_NOT_COMPLETE", "variant group must be COMPLETE")
                continue
            trials = group.get("seed_trials", [])
            observed_seeds = tuple(
                sorted(int(trial.get("seed")) for trial in trials if isinstance(trial, Mapping))
            )
            audit.require(
                observed_seeds == tuple(sorted(expected_seeds))
                and len(trials) == len(expected_seeds),
                f"{prefix}_SEEDS_INCOMPLETE",
                "variant requires exactly the three amended seeds",
                observed=list(observed_seeds),
            )
            for trial in trials if isinstance(trials, list) else []:
                if not isinstance(trial, Mapping):
                    audit.integrity(f"{prefix}_TRIAL_INVALID", "seed trial is not a mapping")
                    continue
                trial_count += 1
                seed = int(trial.get("seed", -1))
                trial_prefix = f"{prefix}_SEED_{seed}"
                audit.require(
                    trial.get("status") == "COMPLETE",
                    f"{trial_prefix}_NOT_COMPLETE",
                    "seed trial must be COMPLETE",
                )
                audit.require(
                    trial.get("locked_test_used_for_selection_or_calibration") is False,
                    f"{trial_prefix}_LOCKED_SELECTION_UNSAFE",
                    "locked test must not be used for selection or calibration",
                )
                checkpoint = trial.get("checkpoint", {})
                audit.require(
                    isinstance(checkpoint, Mapping)
                    and checkpoint.get("reload_prediction_bitwise_equal") is True,
                    f"{trial_prefix}_CHECKPOINT_RELOAD_UNVERIFIED",
                    "checkpoint reload equality must be true",
                )
                locked_hash = str(trial.get("locked_test_target_sha256", ""))
                if _valid_sha(locked_hash):
                    locked_hashes["nlr_workload"].add(locked_hash)
                else:
                    audit.pending(
                        f"{trial_prefix}_LOCKED_HASH_MISSING",
                        "seed trial lacks a locked-test target hash",
                    )
                prediction_record = trial.get("prediction_artifact")
                prediction_path = _verify_artifact_record(
                    prediction_record if isinstance(prediction_record, Mapping) else None,
                    root=root,
                    manifest_path=manifest_path,
                    audit=audit,
                    code_prefix=f"{trial_prefix}_PREDICTION",
                    required_suffix=".npz",
                )
                if prediction_path is not None:
                    prediction_paths.append(_root_relative(prediction_path, root))
                    _prediction_arrays_valid(
                        prediction_path,
                        task="nlr_workload",
                        locked_hash=locked_hash,
                        audit=audit,
                        code_prefix=f"{trial_prefix}_PREDICTION",
                    )
                    if variant_name == PRIMARY_WORKLOAD_VARIANT:
                        primary_prediction_hashes["nlr_workload"].add(
                            str(prediction_record["sha256"]).lower()
                        )
                calibration = trial.get("calibration", {})
                audit.require(
                    isinstance(calibration, Mapping)
                    and calibration.get("fit_split") == "validation"
                    and calibration.get("locked_test_used") is False,
                    f"{trial_prefix}_CALIBRATION_INVALID",
                    "workload calibration must use validation only",
                )
    audit.require(
        trial_count == EXPECTED_SEED_RUN_COUNT,
        "EXTENSION_SEED_RUN_COUNT_INCOMPLETE",
        "extension requires 21 cells x 3 seeds = 63 COMPLETE trials",
        observed=trial_count,
        expected=EXPECTED_SEED_RUN_COUNT,
    )
    for task, values in locked_hashes.items():
        audit.require(
            len(values) == 1,
            f"{task.upper()}_LOCKED_HASH_NOT_UNIQUE",
            "all variants and seeds for a task must use one locked-test hash",
            observed=sorted(values),
        )
    if isinstance(energy, Mapping):
        declared = energy.get("locked_test_target_sha256_by_task", {})
        for task in ENERGY_TASKS:
            if len(locked_hashes[task]) == 1:
                audit.require(
                    isinstance(declared, Mapping)
                    and str(declared.get(task)) == next(iter(locked_hashes[task])),
                    f"{task.upper()}_SUMMARY_LOCKED_HASH_MISMATCH",
                    "energy summary locked hash must match every seed trial",
                )
        audit.require(
            energy.get("oracle_isolated_from_H1") is True,
            "EXTENSION_ORACLE_SUMMARY_NOT_ISOLATED",
            "energy summary must declare ORACLE isolation",
        )
        audit.require(
            energy.get("da_rt_separate") is True,
            "EXTENSION_DA_RT_NOT_SEPARATE",
            "DA and RT must have separate tensor/checkpoint/calibration paths",
        )
    if isinstance(workload, Mapping) and len(locked_hashes["nlr_workload"]) == 1:
        audit.require(
            workload.get("same_locked_test_for_every_seed_and_ablation") is True
            and str(workload.get("locked_test_target_sha256"))
            == next(iter(locked_hashes["nlr_workload"])),
            "WORKLOAD_SUMMARY_LOCKED_HASH_MISMATCH",
            "workload summary locked hash must match every seed trial",
        )
    failed_seeds = extension.get("failed_seeds", [])
    audit.require(
        failed_seeds in (None, [], {}),
        "EXTENSION_FAILED_SEEDS_PRESENT",
        "extension contains failed seeds",
        observed=failed_seeds,
    )
    return {
        "cell_count": EXPECTED_CELL_COUNT,
        "seed_count": len(expected_seeds),
        "seed_run_count": trial_count,
        "prediction_paths": sorted(set(prediction_paths)),
        "primary_prediction_hashes": {
            task: sorted(values) for task, values in primary_prediction_hashes.items()
        },
        "locked_test_hashes": {
            task: next(iter(values)) if len(values) == 1 else None
            for task, values in locked_hashes.items()
        },
    }


def _extract_hash_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.lower()} if _valid_sha(value) else set()
    if isinstance(value, Mapping):
        return {
            str(item).lower()
            for item in value.values()
            if isinstance(item, str) and _valid_sha(item)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(item).lower() for item in value if _valid_sha(item)}
    return set()


def _source_hashes_link_primary(
    source_hashes: Mapping[str, Any],
    primary_hashes: Mapping[str, Sequence[str]],
) -> bool:
    aliases = {
        "da_lbmp": ("extension_da_predictions", "da_predictions", "da_lbmp_predictions"),
        "rt_lbmp": ("extension_rt_predictions", "rt_predictions", "rt_lbmp_predictions"),
        "nlr_workload": (
            "extension_workload_predictions",
            "workload_predictions",
            "nlr_workload_predictions",
        ),
    }
    for task, keys in aliases.items():
        values: set[str] = set()
        for key in keys:
            values.update(_extract_hash_values(source_hashes.get(key)))
        if not values or not values.issubset(set(primary_hashes.get(task, ()))):
            return False
    return True


def _declared_index_hash(
    extension: Mapping[str, Any],
    index_record: Mapping[str, Any] | None,
    index_path: Path,
    *,
    root: Path,
    manifest_path: Path,
) -> str | None:
    if isinstance(index_record, Mapping):
        for key in ("index_sha256", "sha256"):
            if _valid_sha(index_record.get(key)):
                return str(index_record[key])

    def visit(value: Any) -> str | None:
        if isinstance(value, Mapping):
            raw_path = value.get("path", value.get("index_path"))
            raw_hash = value.get("sha256", value.get("index_sha256"))
            if raw_path and _valid_sha(raw_hash):
                try:
                    candidate = _artifact_path(root, manifest_path, str(raw_path))
                except Exception:
                    candidate = None
                if candidate == index_path:
                    return str(raw_hash)
            for nested in value.values():
                result = visit(nested)
                if result:
                    return result
        elif isinstance(value, list):
            for nested in value:
                result = visit(nested)
                if result:
                    return result
        return None

    return visit(extension.get("output_hashes", {}))


def _validate_day_npz(
    path: Path,
    operating_date: str,
    primary_hashes: Mapping[str, Sequence[str]],
) -> FrozenForecastArtifact:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "issue_time_utc",
            "operating_date",
            "forecast_id",
            "timestamps_utc",
            "da_price",
            "rt_price_pool",
            "workload_pool",
            "pv_pool",
            "zones",
            "source_hashes_json",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"scenario NPZ misses arrays: {missing}")
        source_hashes = json.loads(str(data["source_hashes_json"].item()))
        if not isinstance(source_hashes, dict) or not _source_hashes_link_primary(
            source_hashes, primary_hashes
        ):
            raise ValueError("scenario source hashes do not link to extension primary predictions")
        payload = {
            "issue_time_utc": str(data["issue_time_utc"].item()),
            "operating_date": str(data["operating_date"].item()),
            "forecast_id": str(data["forecast_id"].item()),
            "timestamps_utc": data["timestamps_utc"],
            "da_price": data["da_price"],
            "rt_price_pool": data["rt_price_pool"],
            "workload_pool": data["workload_pool"],
            "pv_pool": data["pv_pool"],
            "zones": tuple(map(str, data["zones"].tolist())),
            "source_hashes": source_hashes,
            "locked_test": bool(data["locked_test"].item()) if "locked_test" in data.files else True,
        }
    artifact = FrozenForecastArtifact.from_mapping(payload)
    if artifact.operating_date != operating_date:
        raise ValueError("index date and scenario NPZ operating_date disagree")
    if len(artifact.timestamps_utc) != 96:
        raise ValueError("scenario horizon must contain 96 quarter-hour intervals")
    expected_timestamps = np.arange(
        np.datetime64(operating_date),
        np.datetime64(operating_date) + np.timedelta64(24, "h"),
        np.timedelta64(15, "m"),
    ).astype("datetime64[ns]")
    observed_timestamps = np.asarray(artifact.timestamps_utc).astype("datetime64[ns]")
    if not np.array_equal(observed_timestamps, expected_timestamps):
        raise ValueError("scenario timestamps must be the 96 UTC quarter-hours of its date")
    issue = np.datetime64(str(artifact.issue_time_utc).replace("+00:00", ""))
    if issue >= np.datetime64(operating_date):
        raise ValueError("scenario issue time must precede its operating date")
    return artifact


def _verify_forecast_day_index(
    root: Path,
    manifest_path: Path,
    extension: Mapping[str, Any],
    trial_audit: Mapping[str, Any],
    audit: _Audit,
) -> dict[str, Any]:
    raw = extension.get("frozen_day_index", extension.get("forecast_day_index"))
    index_record = raw if isinstance(raw, Mapping) else None
    raw_path = (
        index_record.get("index_path", index_record.get("path"))
        if isinstance(index_record, Mapping)
        else raw
    )
    if not raw_path:
        audit.pending(
            "EXTENSION_FORECAST_DAY_INDEX_MISSING",
            "extension must provide its 365-day frozen forecast index",
        )
        return {}
    try:
        index_path = _artifact_path(root, manifest_path, str(raw_path))
    except Exception as exc:
        audit.integrity("EXTENSION_FORECAST_DAY_INDEX_PATH_INVALID", str(exc))
        return {}
    if not index_path.is_file():
        audit.pending(
            "EXTENSION_FORECAST_DAY_INDEX_FILE_MISSING",
            f"forecast day index is absent: {index_path}",
        )
        return {}
    expected_hash = _declared_index_hash(
        extension,
        index_record,
        index_path,
        root=root,
        manifest_path=manifest_path,
    )
    if not _valid_sha(expected_hash):
        audit.pending(
            "EXTENSION_FORECAST_DAY_INDEX_HASH_UNDECLARED",
            "forecast day index needs a manifest-declared SHA-256",
            path=str(index_path),
        )
    actual_hash = sha256_file(index_path)
    if _valid_sha(expected_hash) and str(expected_hash).lower() != actual_hash.lower():
        audit.integrity(
            "EXTENSION_FORECAST_DAY_INDEX_HASH_MISMATCH",
            "forecast day index bytes disagree with its declared hash",
            expected_sha256=expected_hash,
            actual_sha256=actual_hash,
        )
        return {}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit.integrity(
            "EXTENSION_FORECAST_DAY_INDEX_INVALID_JSON",
            f"{type(exc).__name__}: {exc}",
        )
        return {}
    if not isinstance(index, Mapping):
        audit.integrity(
            "EXTENSION_FORECAST_DAY_INDEX_NOT_MAPPING",
            "forecast day index JSON root must be a mapping",
        )
        return {}
    audit.require(
        index.get("status") == "COMPLETE",
        "EXTENSION_FORECAST_DAY_INDEX_NOT_COMPLETE",
        "forecast day index must be COMPLETE",
    )
    audit.require(
        int(index.get("day_count", -1)) == 365
        and int(index.get("deployment_day_count", -1)) == 365
        and index.get("deployment_coverage_complete") is True,
        "EXTENSION_DEPLOYMENT_COVERAGE_INCOMPLETE",
        "forecast day index must cover all 365 deployment origins",
        day_count=index.get("day_count"),
        deployment_day_count=index.get("deployment_day_count"),
    )
    audit.require(
        index.get("deployment_predictions_use_realized_targets") is False
        and index.get("missing_realized_targets_imputed") is False,
        "EXTENSION_DEPLOYMENT_LEAKAGE_UNSAFE",
        "deployment scenarios must not consume or impute realized targets",
    )
    days = index.get("days", [])
    if not isinstance(days, list):
        audit.integrity(
            "EXTENSION_FORECAST_DAY_RECORDS_INVALID",
            "forecast day index days must be a list",
        )
        return {}
    by_date: dict[str, Mapping[str, Any]] = {}
    for record in days:
        if not isinstance(record, Mapping) or not record.get("operating_date"):
            audit.integrity(
                "EXTENSION_FORECAST_DAY_RECORD_INVALID",
                "every forecast day record needs an operating_date",
            )
            continue
        operating_date = str(record["operating_date"])
        if operating_date in by_date:
            audit.integrity(
                "EXTENSION_FORECAST_DAY_DUPLICATE",
                "forecast day index contains a duplicate date",
                operating_date=operating_date,
            )
        by_date[operating_date] = record
    expected_dates = set(_expected_2025_dates())
    audit.require(
        set(by_date) == expected_dates,
        "EXTENSION_FORECAST_DAY_DATE_SET_INCOMPLETE",
        "forecast day index must contain exactly every date in 2025",
        missing=sorted(expected_dates - set(by_date)),
        extra=sorted(set(by_date) - expected_dates),
    )
    verified = 0
    scenario_paths: list[str] = []
    for operating_date, record in sorted(by_date.items()):
        audit.require(
            record.get("realized_targets_used_to_generate_forecast") is False
            and record.get("missing_realized_targets_imputed") is False,
            f"EXTENSION_SCENARIO_{operating_date}_LEAKAGE_STATUS_UNSAFE",
            "each deployment day must explicitly exclude realized-target use and imputation",
        )
        path = _verify_artifact_record(
            record,
            root=root,
            manifest_path=index_path,
            audit=audit,
            code_prefix=f"EXTENSION_SCENARIO_{operating_date}",
            required_suffix=".npz",
        )
        if path is None:
            continue
        try:
            _validate_day_npz(
                path,
                operating_date,
                trial_audit["primary_prediction_hashes"],
            )
        except Exception as exc:
            audit.integrity(
                f"EXTENSION_SCENARIO_{operating_date}_INVALID",
                f"{type(exc).__name__}: {exc}",
                path=str(path),
            )
            continue
        verified += 1
        scenario_paths.append(_root_relative(path, root))
    audit.require(
        verified == 365,
        "EXTENSION_SCENARIO_COUNT_INCOMPLETE",
        "all 365 scenario NPZ files must be hash-verified and contract-valid",
        verified=verified,
    )
    return {
        "status": index.get("status"),
        "index_path": _root_relative(index_path, root),
        "index_sha256": actual_hash,
        "day_count": len(by_date),
        "deployment_day_count": index.get("deployment_day_count"),
        "deployment_coverage_complete": index.get("deployment_coverage_complete"),
        "draw_count": index.get("draw_count"),
        "forecast_role": index.get("forecast_role"),
        "scenario_npz_verified_count": verified,
        "scenario_paths": scenario_paths,
    }


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype="float64")
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("metric aggregation requires finite values")
    return float(array.mean())


def _energy_metric(group: Mapping[str, Any], split: str, *keys: str) -> float:
    values = []
    for trial in group["seed_trials"]:
        current: Any = trial["metrics"][split]
        for key in keys:
            current = current[key]
        values.append(float(current))
    return _mean(values)


def _workload_score(group: Mapping[str, Any], split: str) -> float:
    scores: list[float] = []
    for trial in group["seed_trials"]:
        metrics = trial["metrics"][split]
        count = metrics["count_distribution"]
        count_score = (
            count.get("negative_binomial_nll")
            if "negative_binomial_nll" in count
            else count.get("poisson_nll")
        )
        volume = metrics.get("volume_quantiles")
        if count_score is None or not isinstance(volume, Mapping):
            raise ValueError("H2 comparable score requires count NLL and volume quantile score")
        scores.append(float(count_score) + float(volume["mean_weighted_pinball"]))
    return _mean(scores)


def _workload_calibration(group: Mapping[str, Any], split: str) -> float:
    values: list[float] = []
    for trial in group["seed_trials"]:
        metrics = trial["metrics"][split]
        values.append(
            0.5
            * (
                float(metrics["count_quantiles"]["absolute_calibration_error"])
                + float(metrics["volume_quantiles"]["absolute_calibration_error"])
            )
        )
    return _mean(values)


def _build_hypothesis_records(
    extension: Mapping[str, Any],
    audit: _Audit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    energy_tasks, workload_variants = _trial_groups(extension)
    energy_records: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    try:
        for task in ENERGY_TASKS:
            groups = energy_tasks[task]
            primary = groups[PRIMARY_ENERGY_VARIANT]
            candidates = {
                name: group
                for name, group in groups.items()
                if name not in {PRIMARY_ENERGY_VARIANT, ORACLE_ENERGY_VARIANT}
            }
            validation_scores = {
                name: _energy_metric(group, "validation", "mean_weighted_pinball")
                for name, group in candidates.items()
            }
            baseline_name = min(validation_scores, key=validation_scores.get)
            baseline = candidates[baseline_name]
            advanced_value = _energy_metric(primary, "locked_test", "mean_weighted_pinball")
            baseline_value = _energy_metric(baseline, "locked_test", "mean_weighted_pinball")
            advanced_calibration = _energy_metric(
                primary, "locked_test", "absolute_calibration_error"
            )
            baseline_calibration = _energy_metric(
                baseline, "locked_test", "absolute_calibration_error"
            )
            advanced_tail = _energy_metric(primary, "locked_test", "tail", "f1")
            baseline_tail = _energy_metric(baseline, "locked_test", "tail", "f1")
            improvement = baseline_value - advanced_value
            calibration_harm = advanced_calibration - baseline_calibration
            energy_records.append(
                {
                    "hypothesis": "H1",
                    "task": TASK_TO_HYPOTHESIS_NAME[task],
                    "advanced_model": "DGPT-E extension primary",
                    "advanced_variant": PRIMARY_ENERGY_VARIANT,
                    "strongest_eligible_baseline": baseline_name,
                    "baseline_selected_on": "mean_validation_weighted_pinball_across_three_seeds",
                    "baseline_validation_value": validation_scores[baseline_name],
                    "comparison_metric": "locked_test_mean_weighted_pinball",
                    "advanced_value": advanced_value,
                    "baseline_value": baseline_value,
                    "improvement_absolute": improvement,
                    "absolute_calibration_error_change": calibration_harm,
                    "tail_f1_advanced": advanced_tail,
                    "tail_f1_baseline": baseline_tail,
                    "support": bool(
                        improvement > 0.0
                        and calibration_harm <= 0.02
                        and advanced_tail >= baseline_tail
                    ),
                    "evaluated": True,
                    "seed_aggregation": "mean_across_three_amended_seeds",
                    "locked_test_used_for_baseline_selection": False,
                    "oracle_excluded": True,
                    "evidence_source": "paper_forecast_extensions",
                }
            )
            selections[task] = {
                "primary": PRIMARY_ENERGY_VARIANT,
                "baseline": baseline_name,
                "validation_scores": validation_scores,
            }
    except Exception as exc:
        audit.pending(
            "EXTENSION_H1_METRICS_INCOMPLETE",
            f"{type(exc).__name__}: {exc}",
        )
        energy_records = []
    workload_records: list[dict[str, Any]] = []
    try:
        primary = workload_variants[PRIMARY_WORKLOAD_VARIANT]
        candidates = {
            name: group
            for name, group in workload_variants.items()
            if name != PRIMARY_WORKLOAD_VARIANT and name not in WORKLOAD_H2_INELIGIBLE
        }
        validation_scores = {
            name: _workload_score(group, "validation")
            for name, group in candidates.items()
        }
        baseline_name = min(validation_scores, key=validation_scores.get)
        baseline = candidates[baseline_name]
        advanced_value = _workload_score(primary, "locked_test")
        baseline_value = _workload_score(baseline, "locked_test")
        advanced_calibration = _workload_calibration(primary, "locked_test")
        baseline_calibration = _workload_calibration(baseline, "locked_test")
        improvement = baseline_value - advanced_value
        calibration_improvement = baseline_calibration - advanced_calibration
        workload_records.append(
            {
                "hypothesis": "H2",
                "task": "nlr_workload",
                "advanced_model": "CfC-W extension primary",
                "advanced_variant": PRIMARY_WORKLOAD_VARIANT,
                "strongest_eligible_baseline": baseline_name,
                "baseline_selected_on": "mean_validation_comparable_distribution_score_across_three_seeds",
                "baseline_validation_value": validation_scores[baseline_name],
                "comparison_metric": "locked_test_count_nll_plus_volume_weighted_pinball",
                "advanced_value": advanced_value,
                "baseline_value": baseline_value,
                "improvement_absolute": improvement,
                "interval_calibration_improvement": calibration_improvement,
                "support": bool(improvement > 0.0 and calibration_improvement >= 0.0),
                "evaluated": True,
                "seed_aggregation": "mean_across_three_amended_seeds",
                "locked_test_used_for_baseline_selection": False,
                "evidence_source": "paper_forecast_extensions",
            }
        )
        selections["nlr_workload"] = {
            "primary": PRIMARY_WORKLOAD_VARIANT,
            "baseline": baseline_name,
            "validation_scores": validation_scores,
            "ineligible_for_H2": sorted(WORKLOAD_H2_INELIGIBLE),
        }
    except Exception as exc:
        audit.pending(
            "EXTENSION_H2_METRICS_INCOMPLETE",
            f"{type(exc).__name__}: {exc}",
        )
        workload_records = []
    return energy_records, workload_records, selections


def _auxiliary_v1(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evidence_role": "auxiliary_ablation",
        "roles": ["price_only", "no_spike"],
        "excluded_from_H1_H2": True,
        "energy_records": list(manifest.get("energy_hypothesis_metrics", []) or []),
        "workload_records": list(manifest.get("workload_hypothesis_metrics", []) or []),
        "forecast_day_index": manifest.get("forecast_day_index"),
        "reason": (
            "v1 remains frozen lineage and an auxiliary price-only/no-spike ablation; "
            "confirmatory H1/H2 use extension primary and extension baselines only"
        ),
    }


def _pending_result(
    *,
    root: Path,
    extension_path: Path,
    extension_sha256: str,
    audit: _Audit,
    v1_lineage: Mapping[str, Any] | None = None,
    extension_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = "BLOCKED_INTEGRITY" if audit.integrity_failures else "ACTIONABLE_PENDING"
    return {
        "status": status,
        "terminal": False,
        "optimization_allowed": False,
        "actionable_pending": audit.actionable_pending,
        "actionable_pending_count": len(audit.actionable_pending),
        "integrity_failures": audit.integrity_failures,
        "integrity_failure_count": len(audit.integrity_failures),
        "processed_this_invocation": 0,
        "frozen_day_index": None,
        "forecast_day_index": None,
        "energy_hypothesis_metrics": [],
        "workload_hypothesis_metrics": [],
        "forecast_decision_records": [],
        "output_paths": [],
        "run_manifest": (
            _root_relative(extension_path, root)
            if extension_path.is_relative_to(root)
            else str(extension_path)
        ),
        "path": (
            _root_relative(extension_path, root)
            if extension_path.is_relative_to(root)
            else str(extension_path)
        ),
        "sha256": extension_sha256 or None,
        "v1_lineage": dict(v1_lineage or {}),
        "extension_lineage": dict(extension_lineage or {}),
    }


def load_complete_forecast_evidence(
    root: Path,
    v1_manifest: str | Path,
    extension_manifest: str | Path,
) -> dict[str, Any]:
    """Load verified extension evidence into the optimization forecast contract.

    The returned mapping can be passed directly to ``run_optimization_branch``
    only when ``optimization_allowed`` is true.  The function performs no
    filesystem writes.
    """

    repository_root = Path(root).resolve()
    audit = _Audit()
    try:
        v1_path = _manifest_path(repository_root, v1_manifest, name="v1_manifest")
        extension_path = _manifest_path(
            repository_root, extension_manifest, name="extension_manifest"
        )
    except Exception as exc:
        audit.integrity("FORECAST_MANIFEST_PATH_INVALID", str(exc))
        fallback = Path(extension_manifest) if not isinstance(extension_manifest, Mapping) else repository_root
        return _pending_result(
            root=repository_root,
            extension_path=fallback,
            extension_sha256="",
            audit=audit,
        )
    v1, v1_sha = _read_json(v1_path, audit, label="v1")
    extension, extension_sha = _read_json(extension_path, audit, label="extension")
    v1_lineage = _verify_v1(
        repository_root, v1_path, v1, v1_sha, audit
    ) if v1 else {}
    audit.require(
        str(extension.get("status", "")).upper() == "COMPLETE",
        "EXTENSION_NOT_COMPLETE",
        "extension manifest status must be COMPLETE",
        observed=extension.get("status"),
    )
    audit.require(
        extension.get("terminal") is True,
        "EXTENSION_NOT_TERMINAL",
        "extension manifest must explicitly declare terminal=true",
        observed=extension.get("terminal"),
    )
    audit.require(
        extension.get("v1_outputs_rewritten") is False,
        "EXTENSION_V1_REWRITE_STATUS_UNSAFE",
        "extension must explicitly declare v1_outputs_rewritten=false",
        observed=extension.get("v1_outputs_rewritten"),
    )
    audit.require(
        extension.get("run_id") == EXPECTED_EXTENSION_RUN_ID,
        "EXTENSION_RUN_ID_INVALID",
        f"extension run_id must be {EXPECTED_EXTENSION_RUN_ID}",
        observed=extension.get("run_id"),
    )
    completion_gates = extension.get("completion_gates", {})
    audit.require(
        isinstance(completion_gates, Mapping)
        and bool(completion_gates)
        and all(value is True for value in completion_gates.values()),
        "EXTENSION_COMPLETION_GATES_NOT_PASSED",
        "every declared extension completion gate must be true",
        observed=(dict(completion_gates) if isinstance(completion_gates, Mapping) else None),
    )
    baseline = extension.get("baseline_lineage", {})
    audit.require(
        isinstance(baseline, Mapping)
        and str(baseline.get("baseline_manifest_sha256", "")).lower() == v1_sha.lower(),
        "EXTENSION_V1_LINEAGE_HASH_MISMATCH",
        "extension baseline lineage must pin the current v1 manifest hash",
        current_v1_sha256=v1_sha,
        declared=(baseline.get("baseline_manifest_sha256") if isinstance(baseline, Mapping) else None),
    )
    audit.require(
        isinstance(baseline, Mapping)
        and baseline.get("baseline_run_id") == v1.get("run_id")
        and baseline.get("original_v1_run_id", "paper-forecast-v1")
        == "paper-forecast-v1"
        and baseline.get("baseline_read_only") is True
        and baseline.get("baseline_outputs_rewritten") is False
        and extension.get("baseline_manifest_unchanged") is True,
        "EXTENSION_V1_READ_ONLY_CONTRACT_MISSING",
        "extension must explicitly preserve the frozen paper-forecast-v1 lineage",
    )
    after_hash = extension.get("baseline_manifest_sha256_after")
    audit.require(
        str(after_hash or "").lower() == v1_sha.lower(),
        "EXTENSION_V1_FINAL_HASH_MISMATCH",
        "extension must prove the v1 manifest stayed unchanged through completion",
        current_v1_sha256=v1_sha,
        declared=after_hash,
    )
    trial_audit = _verify_extension_trials(
        repository_root, extension_path, extension, audit
    ) if extension else {
        "cell_count": 0,
        "seed_count": 0,
        "seed_run_count": 0,
        "prediction_paths": [],
        "primary_prediction_hashes": {},
        "locked_test_hashes": {},
    }
    day_index = _verify_forecast_day_index(
        repository_root,
        extension_path,
        extension,
        trial_audit,
        audit,
    ) if extension else {}
    energy_hypotheses, workload_hypotheses, baseline_selections = (
        _build_hypothesis_records(extension, audit)
        if extension
        else ([], [], {})
    )
    extension_lineage = {
        "manifest_path": _root_relative(extension_path, repository_root),
        "manifest_sha256": extension_sha,
        "namespace": extension.get("namespace", "paper_forecast_extensions"),
        "run_id": extension.get("run_id"),
        "cell_count": trial_audit["cell_count"],
        "seed_count": trial_audit["seed_count"],
        "seed_run_count": trial_audit["seed_run_count"],
        "preregistered_seeds": list(PREREGISTERED_SEEDS),
        "locked_test_hashes": trial_audit["locked_test_hashes"],
        "baseline_selections": baseline_selections,
        "oracle_isolated_from_H1": bool(
            isinstance(extension.get("energy"), Mapping)
            and extension["energy"].get("oracle_isolated_from_H1") is True
        ),
        "v1_manifest_sha256": v1_sha,
        "v1_read_only": True,
    }
    # Re-read both hashes after all artifact traversal. A concurrent mutation
    # invalidates the bridge even if every earlier check passed.
    if v1_path.is_file() and sha256_file(v1_path) != v1_sha:
        audit.integrity(
            "V1_MANIFEST_CHANGED_DURING_BRIDGE",
            "v1 manifest changed while bridge evidence was being loaded",
        )
    if extension_path.is_file() and sha256_file(extension_path) != extension_sha:
        audit.integrity(
            "EXTENSION_MANIFEST_CHANGED_DURING_BRIDGE",
            "extension manifest changed while bridge evidence was being loaded",
        )
    if audit.actionable_pending or audit.integrity_failures:
        return _pending_result(
            root=repository_root,
            extension_path=extension_path,
            extension_sha256=extension_sha,
            audit=audit,
            v1_lineage=v1_lineage,
            extension_lineage=extension_lineage,
        )
    frozen_day_index = day_index["index_path"]
    output_paths = sorted(
        set(trial_audit["prediction_paths"] + day_index["scenario_paths"])
    )
    return {
        "status": "COMPLETE",
        "terminal": True,
        "optimization_allowed": True,
        "actionable_pending": [],
        "actionable_pending_count": 0,
        "integrity_failures": [],
        "integrity_failure_count": 0,
        "processed_this_invocation": 0,
        "frozen_day_index": frozen_day_index,
        "forecast_day_index": {
            key: value for key, value in day_index.items() if key != "scenario_paths"
        },
        "energy_hypothesis_metrics": energy_hypotheses,
        "workload_hypothesis_metrics": workload_hypotheses,
        "forecast_decision_records": [
            {
                "forecast_day_index": frozen_day_index,
                "day_count": 365,
                "draw_count": day_index.get("draw_count"),
                "horizon_intervals": 96,
                "status": "COMPLETE",
                "source": "paper_forecast_extensions",
            }
        ],
        "auxiliary_v1_ablations": _auxiliary_v1(v1),
        "output_paths": output_paths,
        "run_manifest": _root_relative(extension_path, repository_root),
        "path": _root_relative(extension_path, repository_root),
        "sha256": extension_sha,
        "counts": {
            "task_variant_cells": EXPECTED_CELL_COUNT,
            "seeds": len(PREREGISTERED_SEEDS),
            "seed_runs": EXPECTED_SEED_RUN_COUNT,
            "prediction_npz_verified": len(trial_audit["prediction_paths"]),
            "scenario_npz_verified": day_index["scenario_npz_verified_count"],
            "deployment_days": day_index["day_count"],
            "failed_seeds": 0,
        },
        "v1_lineage": v1_lineage,
        "extension_lineage": extension_lineage,
    }


__all__ = ["load_complete_forecast_evidence"]
