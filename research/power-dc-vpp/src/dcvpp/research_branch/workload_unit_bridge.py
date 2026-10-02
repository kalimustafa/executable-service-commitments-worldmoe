from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now


RECOVERY_RUN_ID = "paper-forecast-v1-recovery-v1"
DEFAULT_OPTIMIZATION_RUN_ID = "paper-optimization-v1"
TRAIN_FRACTION = 0.60
LOWER_QUANTILE = 0.01
CAPACITY_QUANTILE = 0.99
CHECKPOINT_GRANULARITY_WORK_UNITS = 0.01


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def derive_workload_normalization_contract(
    panel: pd.DataFrame,
    physical_artifact: Mapping[str, Any],
    *,
    panel_sha256: str,
    physical_artifact_sha256: str,
) -> dict[str, Any]:
    """Derive the candidate-independent raw-node to physical-unit bridge.

    The chronological training prefix is the same 60% prefix used by the
    workload forecaster.  Its q01--q99 range is mapped to zero--aggregate
    frozen site capacity.  Values above q99 are deliberately not clipped so
    that tail overload remains visible to the SLA-shortfall variables.
    """

    required = {"timestamp_utc", "submitted_nodes_requested"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"workload panel is missing columns: {sorted(missing)}")
    values = panel["submitted_nodes_requested"].to_numpy(dtype=float)
    if values.size < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("workload panel values must be finite, nonnegative, and nonempty")
    train_rows = int(math.floor(values.size * TRAIN_FRACTION))
    if train_rows < 2:
        raise ValueError("workload training prefix has fewer than two rows")
    train = values[:train_rows]
    q01, q99 = np.quantile(train, [LOWER_QUANTILE, CAPACITY_QUANTILE])
    if not np.isfinite([q01, q99]).all() or q99 <= q01:
        raise ValueError("training-only q01/q99 workload scale is degenerate")
    capacities = np.asarray(physical_artifact.get("capacity_work_units"), dtype=float)
    if capacities.ndim != 1 or capacities.size != 4:
        raise ValueError("frozen physical artifact must contain four site capacities")
    if not np.isfinite(capacities).all() or (capacities <= 0).any():
        raise ValueError("frozen physical capacities must be finite and positive")
    aggregate_capacity = float(capacities.sum())
    scale = aggregate_capacity / float(q99 - q01)
    timestamps = pd.to_datetime(panel["timestamp_utc"], utc=True)
    return {
        "schema_version": "workload_unit_bridge_v1",
        "contract_name": "train_q01_q99_to_frozen_aggregate_capacity_v1",
        "source_quantity": "submitted_nodes_requested",
        "target_quantity": "aggregate_normalized_physical_work_units",
        "formula": "max(raw_nodes - train_q01_raw_nodes, 0) * scale_work_units_per_raw_node",
        "tail_policy": "not_clipped_above_train_q99",
        "candidate_dependency": "none",
        "locked_2025_outcomes_used": False,
        "training_fraction": TRAIN_FRACTION,
        "training_row_count": train_rows,
        "training_start_utc": timestamps.iloc[0].isoformat(),
        "training_end_utc_inclusive": timestamps.iloc[train_rows - 1].isoformat(),
        "first_nontraining_timestamp_utc": timestamps.iloc[train_rows].isoformat(),
        "train_q01_raw_nodes": float(q01),
        "train_q99_raw_nodes": float(q99),
        "aggregate_frozen_capacity_work_units": aggregate_capacity,
        "site_capacity_work_units": capacities.tolist(),
        "scale_work_units_per_raw_node": float(scale),
        "checkpoint_granularity_work_units": CHECKPOINT_GRANULARITY_WORK_UNITS,
        "panel_sha256": panel_sha256,
        "physical_artifact_sha256": physical_artifact_sha256,
        "evidence_role": "derived",
        "measurement_relationship": "not_measured_by_this_study",
    }


def apply_workload_normalization(
    workload_pool: np.ndarray, contract: Mapping[str, Any]
) -> np.ndarray:
    raw = np.asarray(workload_pool, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("forecast workload_pool must be finite, nonnegative, and two-dimensional")
    lower = float(contract["train_q01_raw_nodes"])
    scale = float(contract["scale_work_units_per_raw_node"])
    return np.maximum(raw - lower, 0.0) * scale


def _blocked(
    admission_path: Path,
    *,
    run_id: str,
    reason: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "workload_normalization_admission_v1",
        "run_id": run_id,
        "status": "BLOCKED",
        "admitted": False,
        "terminal": True,
        "reason": reason,
        "evidence": dict(evidence or {}),
        "optimization_result": {
            "status": "NOT_RUN_NORMALIZATION_BLOCKED",
            "completed_count": 0,
            "failure_count": 0,
            "pending_count": 0,
            "actionable_pending_count": 0,
            "paths": {},
        },
        "completed_at_utc": utc_now(),
    }
    atomic_json(admission_path, payload)
    return {
        "status": "BLOCKED",
        "admitted": False,
        "terminal": True,
        "reason": reason,
        "admission_manifest_path": str(admission_path),
        "admission_manifest_sha256": sha256_file(admission_path),
        "optimization_result": payload["optimization_result"],
    }


def prepare_normalized_optimization_forecasts(
    root: Path, run_id: str = DEFAULT_OPTIMIZATION_RUN_ID
) -> dict[str, Any]:
    """Create an immutable runtime wrapper around COMPLETE v1 forecast days.

    The source recovery days are never modified.  Every failure becomes a
    persisted BLOCKED admission record so terminal orchestration can close the
    run without mistaking it for actionable optimization work.
    """

    root = Path(root).resolve()
    branch_root = root / "artifacts" / "optimization_branch" / run_id
    admission_path = branch_root / "workload_normalization_admission.json"
    branch_root.mkdir(parents=True, exist_ok=True)
    try:
        recovery_root = (
            root / "artifacts" / "real_forecasting_recovery" / f"run_id={RECOVERY_RUN_ID}"
        )
        recovery_manifest_path = recovery_root / "run_manifest.json"
        preflight_path = recovery_root / "preflight_manifest.json"
        source_index_path = recovery_root / "forecast_days" / "index.json"
        for required in (recovery_manifest_path, preflight_path, source_index_path):
            if not required.is_file():
                raise FileNotFoundError(f"required frozen input is missing: {_relative(root, required)}")
        recovery = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
        if recovery.get("status") != "COMPLETE" or recovery.get("terminal") is not True:
            raise ValueError("v1 recovery manifest is not terminal COMPLETE")
        if source_index.get("status") != "COMPLETE" or int(source_index.get("day_count", 0)) != 365:
            raise ValueError("v1 frozen day index is not COMPLETE with 365 deployment days")
        source_days = list(source_index.get("days", []))
        if len(source_days) != 365 or len({str(day.get("operating_date")) for day in source_days}) != 365:
            raise ValueError("v1 frozen day index does not contain 365 unique days")

        physical_contract = preflight.get("physical_recovery_contract", {})
        if physical_contract.get("status") != "VERIFIED_COMPLETED":
            raise ValueError("frozen physical recovery contract is not VERIFIED_COMPLETED")
        outputs = physical_contract.get("verified_output_hashes", {})
        physical_record = outputs.get("frozen_artifact", {})
        physical_path = _resolve(root, physical_record.get("path", ""))
        expected_physical_sha = str(physical_record.get("sha256", ""))
        if not physical_path.is_file() or sha256_file(physical_path) != expected_physical_sha:
            raise ValueError("frozen physical artifact does not match verified recovery hash")
        physical_artifact = json.loads(physical_path.read_text(encoding="utf-8"))

        cfc = preflight.get("cfc_checkpoint_compatibility", {})
        panel_path = _resolve(root, cfc.get("panel_path", ""))
        expected_panel_sha = str(cfc.get("panel_sha256", ""))
        if not panel_path.is_file() or sha256_file(panel_path) != expected_panel_sha:
            raise ValueError("frozen workload training panel does not match recovery hash")
        panel = pd.read_parquet(
            panel_path, columns=["timestamp_utc", "submitted_nodes_requested"]
        )
        contract = derive_workload_normalization_contract(
            panel,
            physical_artifact,
            panel_sha256=expected_panel_sha,
            physical_artifact_sha256=expected_physical_sha,
        )
        contract["implementation_sha256"] = sha256_file(Path(__file__))
        contract_hash = sha256_json(contract)

        normalized_root = branch_root / "normalized_forecast_days"
        normalized_root.mkdir(parents=True, exist_ok=True)
        normalized_days: list[dict[str, Any]] = []
        raw_min = float("inf")
        raw_max = float("-inf")
        normalized_min = float("inf")
        normalized_max = float("-inf")
        for record in source_days:
            source_path = _resolve(root, str(record["path"]))
            expected_source_sha = str(record["sha256"])
            if not source_path.is_file() or sha256_file(source_path) != expected_source_sha:
                raise ValueError(
                    f"frozen forecast day hash mismatch: {record.get('operating_date')}"
                )
            with np.load(source_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            raw = np.asarray(arrays["workload_pool"], dtype=float)
            normalized = apply_workload_normalization(raw, contract)
            raw_min = min(raw_min, float(raw.min()))
            raw_max = max(raw_max, float(raw.max()))
            normalized_min = min(normalized_min, float(normalized.min()))
            normalized_max = max(normalized_max, float(normalized.max()))
            day = str(record["operating_date"])
            forecast_id = f"{str(record['forecast_id'])}__workunit_{contract_hash[:12]}"
            source_hashes = json.loads(str(np.asarray(arrays["source_hashes_json"]).item()))
            source_hashes.update(
                {
                    "source_frozen_forecast_day": expected_source_sha,
                    "workload_training_panel": expected_panel_sha,
                    "frozen_physical_artifact": expected_physical_sha,
                    "workload_unit_bridge_contract": contract_hash,
                }
            )
            arrays["workload_pool"] = normalized
            arrays["forecast_id"] = np.asarray(forecast_id)
            arrays["source_hashes_json"] = np.asarray(
                json.dumps(source_hashes, sort_keys=True, separators=(",", ":"))
            )
            destination = normalized_root / f"{day}.npz"
            with tempfile.NamedTemporaryFile(
                suffix=".npz", dir=normalized_root, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                np.savez_compressed(temporary_path, **arrays)
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            derived_record = dict(record)
            derived_record.update(
                {
                    "path": _relative(root, destination),
                    "sha256": sha256_file(destination),
                    "forecast_id": forecast_id,
                    "source_forecast_id": str(record["forecast_id"]),
                    "source_forecast_sha256": expected_source_sha,
                    "workload_unit_bridge_contract_sha256": contract_hash,
                }
            )
            normalized_days.append(derived_record)

        normalized_index = dict(source_index)
        normalized_index.update(
            {
                "schema_version": "normalized_optimization_forecast_day_index_v1",
                "status": "COMPLETE",
                "factory": "runtime_workload_unit_bridge_from_complete_v1_recovery",
                "source_index_path": _relative(root, source_index_path),
                "source_index_sha256": sha256_file(source_index_path),
                "workload_unit_bridge_contract": contract,
                "workload_unit_bridge_contract_sha256": contract_hash,
                "days": normalized_days,
            }
        )
        normalized_index_path = normalized_root / "index.json"
        atomic_json(normalized_index_path, normalized_index)
        index_sha = sha256_file(normalized_index_path)
        manifest = {
            "schema_version": "workload_normalization_admission_v1",
            "run_id": run_id,
            "status": "ADMITTED",
            "admitted": True,
            "terminal": True,
            "source_recovery_status": "COMPLETE",
            "source_recovery_extension_used": False,
            "dart_overlay_used": False,
            "contract": contract,
            "contract_sha256": contract_hash,
            "source_index_path": _relative(root, source_index_path),
            "source_index_sha256": sha256_file(source_index_path),
            "normalized_index_path": _relative(root, normalized_index_path),
            "normalized_index_sha256": index_sha,
            "day_count": len(normalized_days),
            "raw_workload_range": [raw_min, raw_max],
            "normalized_workload_range": [normalized_min, normalized_max],
            "optimization_checkpoint_granularity_work_units": CHECKPOINT_GRANULARITY_WORK_UNITS,
            "optimization_result": {
                "status": "NOT_STARTED_ADMITTED",
                "completed_count": 0,
                "failure_count": 0,
                "pending_count": 11328,
                "actionable_pending_count": 11328,
                "paths": {},
            },
            "completed_at_utc": utc_now(),
        }
        atomic_json(admission_path, manifest)
        forecast_result = dict(recovery)
        forecast_result.update(
            {
                "status": "COMPLETE",
                "forecast_day_index": _relative(root, normalized_index_path),
                "frozen_day_index": {
                    "index_path": _relative(root, normalized_index_path),
                    "sha256": index_sha,
                },
                "workload_unit_bridge_admission_path": _relative(root, admission_path),
            }
        )
        return {
            "status": "ADMITTED",
            "admitted": True,
            "terminal": True,
            "index_path": _relative(root, normalized_index_path),
            "index_sha256": index_sha,
            "normalization_manifest_path": _relative(root, admission_path),
            "normalization_manifest_sha256": sha256_file(admission_path),
            "contract_sha256": contract_hash,
            "checkpoint_granularity_work_units": CHECKPOINT_GRANULARITY_WORK_UNITS,
            "forecast_result": forecast_result,
            "optimization_result": manifest["optimization_result"],
        }
    except Exception as exc:
        return _blocked(
            admission_path,
            run_id=run_id,
            reason=f"{type(exc).__name__}: {exc}",
        )

