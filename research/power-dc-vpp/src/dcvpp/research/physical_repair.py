from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dcvpp.core import atomic_json, load_yaml, sha256_file, sha256_json, utc_now
from dcvpp.research.physical_ablations import (
    evaluate_physical_baselines_and_ablations,
)
from dcvpp.research.physical import (
    FIXED_MST,
    NYISO_ZONES,
    PhysicalBranchError,
    _branch_config,
    _atomic_parquet,
    _build_transfer_and_freeze,
    _calibrate_facility_panel,
    _completed_manifest_is_reusable,
    _configured_zone_weather_identity,
    _jsonable_config,
    _known_outputs,
    _load_json,
    _outputs_with_hashes,
    _paths,
    _physical_implementation_sha256,
    _prepare_overlaps,
    _relative,
    _required_source_identities,
    _same_source_content,
)


REPAIR_KIND = "targeted_physical_derivative_repair_v1"
NUMERIC_FROZEN_FIELDS = (
    "idle_mw",
    "work_mw_per_unit",
    "capacity_work_units",
    "grid_import_limit_mw",
    "battery_power_mw",
    "battery_energy_mwh",
    "physical_multiplier_pool",
)


def _atomic_restore(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _target_paths(paths: Any) -> dict[str, Path]:
    return {
        "overlap_primary": paths.overlap_primary,
        "overlap_fixed_mst": paths.overlap_fixed_mst,
        "overlap_manifest": paths.overlap_manifest,
        "facility_metrics": paths.facility_metrics,
        "pue_metrics": paths.pue_metrics,
        "surrogate": paths.surrogate,
        "physical_manifest": paths.physical_manifest,
        "transfer_manifest": paths.transfer_manifest,
        "zone_twins": paths.zone_twins,
        "transfer_table": paths.transfer_table,
        "frozen_artifact": paths.frozen_artifact,
        "physical_ablation_table": paths.physical_ablation_table,
        "physical_ablation_manifest": paths.physical_ablation_manifest,
        "branch_manifest": paths.manifest,
    }


def _new_repair_directory(paths: Any) -> tuple[str, Path]:
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-timezone-climate-v1"
    repairs = paths.artifacts / "repairs"
    candidate = repairs / stem
    suffix = 1
    while candidate.exists():
        candidate = repairs / f"{stem}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate.name, candidate


def _snapshot_targets(root: Path, paths: Any, repair_directory: Path) -> dict[str, Any]:
    before = repair_directory / "before"
    before.mkdir(parents=True)
    targets: dict[str, Any] = {}
    for name, source in _target_paths(paths).items():
        if not source.is_file():
            if name in {"physical_ablation_table", "physical_ablation_manifest"}:
                targets[name] = {
                    "canonical_path": _relative(root, source),
                    "backup_path": None,
                    "sha256": None,
                    "byte_count": 0,
                    "existed_before": False,
                }
                continue
            raise FileNotFoundError(f"required derivative repair target is missing: {source}")
        backup = before / f"{name}{source.suffix}"
        shutil.copy2(source, backup)
        source_hash = sha256_file(source)
        backup_hash = sha256_file(backup)
        if source_hash != backup_hash:
            raise RuntimeError(f"snapshot hash mismatch for {name}")
        targets[name] = {
            "canonical_path": _relative(root, source),
            "backup_path": _relative(root, backup),
            "sha256": source_hash,
            "byte_count": source.stat().st_size,
            "existed_before": True,
        }
    return targets


def _restore_targets(root: Path, targets: Mapping[str, Any]) -> None:
    for name, record in targets.items():
        destination = root / str(record["canonical_path"])
        if not bool(record.get("existed_before", True)):
            destination.unlink(missing_ok=True)
            continue
        source = root / str(record["backup_path"])
        expected = str(record["sha256"])
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"cannot restore {name}: backup is absent or corrupt")
        _atomic_restore(source, destination)
        if sha256_file(destination) != expected:
            raise RuntimeError(f"restored hash mismatch for {name}")


def _recover_interrupted_repairs(root: Path, paths: Any) -> list[str]:
    repairs_root = paths.artifacts / "repairs"
    restored: list[str] = []
    if not repairs_root.is_dir():
        return restored
    current_manifest = _load_json(paths.manifest, {})
    current_recovery = current_manifest.get("recovery_record", {})
    for record_path in sorted(repairs_root.glob("*/recovery.json")):
        record = _load_json(record_path, {})
        if record.get("status") != "in_progress":
            continue
        repair_id = str(record.get("repair_id", record_path.parent.name))
        if (
            isinstance(current_recovery, Mapping)
            and current_recovery.get("repair_id") == repair_id
            and current_recovery.get("status") == "completed"
        ):
            record["status"] = "completed_reconciled_from_branch_manifest"
            record["reconciled_at"] = utc_now()
            atomic_json(record_path, record)
            continue
        _restore_targets(root, record.get("targets", {}))
        record["status"] = "rolled_back_before_resume"
        record["rolled_back_at"] = utc_now()
        atomic_json(record_path, record)
        restored.append(repair_id)
        current_manifest = _load_json(paths.manifest, {})
        current_recovery = current_manifest.get("recovery_record", {})
    return restored


def _assert_overlap_content_invariant(old_path: Path, new_path: Path) -> None:
    # Read explicit files, not a Hive-style dataset: the formal artifact path
    # contains ``run_id=...`` and the table itself also carries a run_id field.
    old = pq.ParquetFile(old_path).read()
    new = pq.ParquetFile(new_path).read()
    if old.column_names != new.column_names:
        raise RuntimeError("overlap schema columns changed during timezone-only repair")
    retained = [name for name in old.column_names if name != "timestamp_local"]
    if not old.select(retained).equals(new.select(retained)):
        raise RuntimeError(
            "overlap values outside timestamp_local changed; numeric calibration cannot be reused"
        )


def _assert_overlap_clock(path: Path, expected_timezone: str) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["timestamp_utc", "timestamp_local"])
    utc = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    local = pd.to_datetime(frame["timestamp_local"], errors="raise")
    if expected_timezone == "America/Denver":
        expected = utc.dt.tz_convert("America/Denver")
        timezone_ok = str(local.dt.tz) == "America/Denver"
    elif expected_timezone == "UTC-07:00":
        expected = utc.dt.tz_convert(FIXED_MST)
        offsets = local.map(lambda value: value.utcoffset().total_seconds() / 3600.0)
        timezone_ok = bool(offsets.eq(-7.0).all())
    else:
        raise ValueError(expected_timezone)
    wall_clock_ok = bool(
        local.dt.tz_localize(None).equals(expected.dt.tz_localize(None))
    )
    instant_ok = bool(local.astype("int64").equals(utc.astype("int64")))
    if not (timezone_ok and wall_clock_ok and instant_ok):
        raise RuntimeError(f"invalid reconstructed {expected_timezone} overlap clock")
    return {
        "timezone": expected_timezone,
        "row_count": len(frame),
        "instant_matches_utc": instant_ok,
        "wall_clock_matches_tz_convert": wall_clock_ok,
        "minimum_utc": utc.min().isoformat(),
        "maximum_utc": utc.max().isoformat(),
    }


def _public_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not str(key).startswith("_")}


def _assert_calibrated_twin_invariant(
    previous: Mapping[str, Any],
    primary: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = {
        "primary": (
            previous.get("primary"),
            _public_calibration(primary),
        ),
        "fixed_mst_sensitivity": (
            previous.get("fixed_mst_sensitivity"),
            _public_calibration(fixed),
        ),
    }
    invariant_keys = (
        "clock_interpretation",
        "row_count",
        "source_day_count",
        "split_manifest",
        "facility_model",
        "pue_model",
        "weather_free_model",
        "engineering_baseline",
        "constant_train_pue",
        "pue_bounds",
        "uncertainty",
    )
    evidence: dict[str, Any] = {}
    for name, (old, new) in comparisons.items():
        old_core = {key: old.get(key) for key in invariant_keys}
        new_core = {key: new.get(key) for key in invariant_keys}
        old_hash = sha256_json(old_core)
        new_hash = sha256_json(new_core)
        if old_hash != new_hash:
            raise RuntimeError(
                f"{name} calibrated statistical twin changed during derivative repair"
            )
        old_nonsurrogate_metrics = [
            row for row in old.get("metrics", [])
            if row.get("model") != "frozen_piecewise_linear_surrogate"
        ]
        new_nonsurrogate_metrics = [
            row for row in new.get("metrics", [])
            if row.get("model") != "frozen_piecewise_linear_surrogate"
        ]
        if sha256_json(old_nonsurrogate_metrics) != sha256_json(new_nonsurrogate_metrics):
            raise RuntimeError(f"{name} non-surrogate held-out metrics changed")
        acceptance = new.get("surrogate_acceptance", {})
        if not acceptance.get("accepted_for_optimization"):
            raise RuntimeError(f"{name} repaired MILP surrogate failed acceptance")
        evidence[name] = {
            "calibrated_twin_sha256": new_hash,
            "old_surrogate_sha256": sha256_json(old.get("surrogate")),
            "new_surrogate_sha256": sha256_json(new.get("surrogate")),
            "surrogate_acceptance": acceptance,
        }
    return evidence


def _assert_frozen_numeric_invariant(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for field in NUMERIC_FROZEN_FIELDS:
        old = np.asarray(previous[field], dtype=float)
        new = np.asarray(current[field], dtype=float)
        if not np.array_equal(old, new):
            raise RuntimeError(f"frozen physical numeric field changed: {field}")
        hashes[field] = sha256_json(current[field])
    return hashes


def _current_source_identities(
    root: Path, paths: Any, cfg: Mapping[str, Any], previous: Mapping[str, Any]
) -> dict[str, Any]:
    current = _required_source_identities(root, paths)
    previous_core = {
        key: value
        for key, value in previous.get("source_identities", {}).items()
        if key in current
    }
    if previous_core and not _same_source_content(previous_core, current):
        raise RuntimeError("immutable NLR/SRRL source identities changed before repair")
    zone_weather = _configured_zone_weather_identity(root, cfg)
    if zone_weather is None:
        raise PhysicalBranchError(
            "NYISO_ZONE_WEATHER_FILE_MISSING",
            str(cfg.get("nyiso_zone_weather_path")),
        )
    current["nyiso_zone_weather"] = zone_weather
    return current


def _request_fingerprint(
    run_id: str, cfg: Mapping[str, Any], source_identities: Mapping[str, Any]
) -> str:
    return sha256_json(
        {
            "schema_version": 1,
            "run_id": run_id,
            "config": _jsonable_config(cfg),
            "source_identities": source_identities,
            "implementation_sha256": _physical_implementation_sha256(),
        }
    )


def _already_repaired(
    root: Path,
    paths: Any,
    manifest: Mapping[str, Any],
    fingerprint: str,
    workload_sha256: str,
) -> bool:
    recovery = manifest.get("recovery_record", {})
    if not (
        isinstance(recovery, Mapping)
        and recovery.get("kind") == REPAIR_KIND
        and recovery.get("status") == "completed"
        and recovery.get("protected_workload_sha256_after") == workload_sha256
    ):
        return False
    if not _completed_manifest_is_reusable(root, manifest, fingerprint):
        return False
    _assert_overlap_clock(paths.overlap_primary, "America/Denver")
    _assert_overlap_clock(paths.overlap_fixed_mst, "UTC-07:00")
    transfer = _load_json(paths.transfer_manifest, {})
    return transfer.get("climate_support_diagnostics", {}).get("status") == "computed"


def repair_completed_physical_derivatives(
    root: Path,
    config: Mapping[str, Any],
    run_id: str,
    *,
    expected_workload_sha256: str | None = None,
) -> dict[str, Any]:
    """Repair timezone/climate derived artifacts without touching workload events.

    The operation snapshots every target, records an in-progress recovery file,
    atomically rewrites only overlap/transfer/frozen/manifests, and rolls back on
    any content, calibration, frozen-array, or protected-input invariant failure.
    """

    repository_root = Path(root).resolve()
    cfg = _branch_config(config)
    cfg["_root"] = repository_root
    paths = _paths(repository_root, cfg, run_id)
    recovered_repairs = _recover_interrupted_repairs(repository_root, paths)
    previous = _load_json(paths.manifest, {})
    if previous.get("status") != "completed":
        raise RuntimeError("targeted repair requires a completed physical branch manifest")

    workload_path = paths.nlr295_workload_events
    workload_sha256 = sha256_file(workload_path)
    contracted_sha256 = str(
        previous.get("workload_events_contract", {}).get("sha256", "")
    )
    expected = expected_workload_sha256 or contracted_sha256
    if not expected or workload_sha256 != expected or workload_sha256 != contracted_sha256:
        raise RuntimeError("protected nlr295_workload_events.parquet hash mismatch")

    source_identities = _current_source_identities(
        repository_root, paths, cfg, previous
    )
    request_fingerprint = _request_fingerprint(
        paths.run_id, cfg, source_identities
    )
    if _already_repaired(
        repository_root,
        paths,
        previous,
        request_fingerprint,
        workload_sha256,
    ):
        result = dict(previous["result"])
        result["repair_resumed"] = True
        result["recovered_interrupted_repairs"] = recovered_repairs
        return result

    repair_id, repair_directory = _new_repair_directory(paths)
    targets = _snapshot_targets(repository_root, paths, repair_directory)
    recovery_path = repair_directory / "recovery.json"
    recovery: dict[str, Any] = {
        "schema_version": 1,
        "kind": REPAIR_KIND,
        "repair_id": repair_id,
        "status": "in_progress",
        "started_at": utc_now(),
        "run_id": paths.run_id,
        "targets": targets,
        "protected_workload_path": _relative(repository_root, workload_path),
        "protected_workload_sha256_before": workload_sha256,
        "expected_workload_sha256": expected,
        "zone_weather_identity": source_identities["nyiso_zone_weather"],
        "recovered_interrupted_repairs": recovered_repairs,
    }
    atomic_json(recovery_path, recovery)

    try:
        old_physical_manifest = _load_json(
            repository_root / targets["physical_manifest"]["backup_path"], {}
        )
        old_frozen = _load_json(
            repository_root / targets["frozen_artifact"]["backup_path"], {}
        )
        old_calibration_id = old_frozen.get("calibration_id")
        stages = dict(previous.get("stages", {}))
        overlap = _prepare_overlaps(
            repository_root,
            paths,
            stages["nlr295"],
            stages["nlr300"],
            stages["srrl_acquisition"],
            stages["srrl_panel"],
            cfg,
        )
        _assert_overlap_content_invariant(
            repository_root / targets["overlap_primary"]["backup_path"],
            paths.overlap_primary,
        )
        _assert_overlap_content_invariant(
            repository_root / targets["overlap_fixed_mst"]["backup_path"],
            paths.overlap_fixed_mst,
        )
        clock_checks = {
            "primary": _assert_overlap_clock(
                paths.overlap_primary, "America/Denver"
            ),
            "fixed_mst_sensitivity": _assert_overlap_clock(
                paths.overlap_fixed_mst, "UTC-07:00"
            ),
        }

        primary = _calibrate_facility_panel(
            paths.overlap_primary,
            interpretation="primary_raw_ts_assumed_utc",
            cfg=cfg,
        )
        fixed = _calibrate_facility_panel(
            paths.overlap_fixed_mst,
            interpretation="sensitivity_raw_ts_assumed_fixed_mst_utc_minus_07",
            cfg=cfg,
        )
        calibration_evidence = _assert_calibrated_twin_invariant(
            old_physical_manifest, primary, fixed
        )
        surrogate_changed = any(
            item["old_surrogate_sha256"] != item["new_surrogate_sha256"]
            for item in calibration_evidence.values()
        )
        facility_metrics = pd.DataFrame([*primary["metrics"], *fixed["metrics"]])
        _atomic_parquet(facility_metrics, paths.facility_metrics)
        pue_metrics = facility_metrics[
            [
                "clock_interpretation",
                "split",
                "model",
                "row_count",
                "pue_mae",
                "interval_coverage",
                "source_relationship",
                "measurement_relationship",
                "evidence_role",
            ]
        ].copy()
        _atomic_parquet(pue_metrics, paths.pue_metrics)
        atomic_json(paths.surrogate, primary["surrogate"])
        transfer = _build_transfer_and_freeze(
            repository_root,
            paths,
            source_identities,
            {"primary": primary, "fixed_mst": fixed},
            cfg,
        )
        frozen_numeric_hashes = _assert_frozen_numeric_invariant(
            old_frozen, transfer["frozen_artifact"]
        )
        climate_manifest = transfer["climate_support_diagnostics"]
        if climate_manifest.get("status") != "computed":
            raise RuntimeError(
                f"zone climate diagnostics incomplete: {climate_manifest}"
            )
        zone_twins = pd.read_parquet(paths.zone_twins)
        if set(zone_twins["zone"].astype(str)) != set(NYISO_ZONES):
            raise RuntimeError("repaired transfer table does not contain every NYISO zone")
        if zone_twins[
            ["climate_support_distance", "out_of_support_fraction"]
        ].isna().any().any():
            raise RuntimeError("repaired transfer table still has missing climate support")
        climate_values = {
            str(row.zone): {
                "climate_support_distance": float(row.climate_support_distance),
                "out_of_support_fraction": float(row.out_of_support_fraction),
                "zone_weather_row_count": int(row.zone_weather_row_count),
            }
            for row in zone_twins.itertuples(index=False)
        }
        physical_ablations = evaluate_physical_baselines_and_ablations(
            repository_root,
            paths,
            {"primary": primary, "fixed_mst": fixed},
            transfer,
            cfg,
        )

        prior_recovery = previous.get("recovery_record", {})
        reason_codes = [
            "DUCKDB_FETCHDF_SESSION_TIMEZONE_DISPLAY_REPLACED",
            "NYISO_ZONE_WEATHER_CONNECTED_TO_TRANSFER_DIAGNOSTICS",
            (
                "MILP_SURROGATE_ENDPOINT_CLIPPING_REPAIRED"
                if surrogate_changed
                else "MILP_SURROGATE_ENDPOINT_EXTRAPOLATION_VERIFIED"
            ),
        ]
        public_repair = {
            "kind": REPAIR_KIND,
            "repair_id": repair_id,
            "status": "completed",
            "completed_at": utc_now(),
            "reason_codes": reason_codes,
            "snapshot_manifest": _relative(repository_root, recovery_path),
            "repair_implementation_sha256": sha256_file(Path(__file__)),
            "protected_workload_sha256_before": workload_sha256,
            "protected_workload_sha256_after": sha256_file(workload_path),
            "protected_workload_modified": False,
            "forecast_code_or_artifact_modified": False,
            "overlap_nonlocal_columns_invariant": True,
            "calibration_input_columns": [
                "timestamp_utc",
                "it_power_kw",
                "facility_power_kw",
                "pue",
                "srrl_air_temp_c",
                "srrl_relative_humidity_pct",
                "srrl_ghi_wm2",
            ],
            "timestamp_local_used_by_numeric_model": False,
            "timestamp_local_historical_numeric_model_impact": "none",
            "historical_numeric_model_impact": (
                "calibrated statistical twin unchanged; prior clipped MILP surrogate "
                "was not acceptable for optimization and was replaced before downstream use"
            ),
            "numeric_calibration_reused": True,
            "calibrated_twin_invariant": True,
            "calibration_evidence": calibration_evidence,
            "milp_surrogate_repaired": surrogate_changed,
            "milp_surrogate_verified_accepted": True,
            "milp_surrogate_change": (
                "replace_np_interp_endpoint_clipping_with_linear_endpoint_extrapolation"
                if surrogate_changed
                else "linear_endpoint_extrapolation_already_present_and_reverified"
            ),
            "milp_surrogate_primary_acceptance": primary["surrogate_acceptance"],
            "milp_surrogate_fixed_mst_acceptance": fixed["surrogate_acceptance"],
            "frozen_numeric_fields_invariant": True,
            "frozen_numeric_hashes": frozen_numeric_hashes,
            "clock_checks": clock_checks,
            "zone_weather_identity": source_identities["nyiso_zone_weather"],
            "climate_support": climate_values,
            "physical_ablation_status": physical_ablations["status"],
            "physical_ablation_table": physical_ablations["output"],
            "physical_ablation_not_evaluable_variants": physical_ablations[
                "not_evaluable_variants"
            ],
            "old_calibration_id": old_calibration_id,
            "new_calibration_id": transfer["calibration_id"],
            "prior_completed_repair_id": (
                prior_recovery.get("repair_id")
                if isinstance(prior_recovery, Mapping)
                and prior_recovery.get("status") == "completed"
                else None
            ),
            "prior_repair_manifest": (
                prior_recovery.get("snapshot_manifest")
                if isinstance(prior_recovery, Mapping)
                and prior_recovery.get("status") == "completed"
                else None
            ),
            "before_hashes": {
                name: record["sha256"] for name, record in targets.items()
            },
        }
        if public_repair["protected_workload_sha256_after"] != workload_sha256:
            raise RuntimeError("protected workload changed during derivative repair")

        overlap_manifest = _load_json(paths.overlap_manifest, {})
        overlap_manifest["derivative_repair"] = public_repair
        atomic_json(paths.overlap_manifest, overlap_manifest)
        transfer_manifest = _load_json(paths.transfer_manifest, {})
        transfer_manifest["derivative_repair"] = public_repair
        atomic_json(paths.transfer_manifest, transfer_manifest)
        physical_manifest = _load_json(paths.physical_manifest, {})
        physical_manifest["primary"] = _public_calibration(primary)
        physical_manifest["fixed_mst_sensitivity"] = _public_calibration(fixed)
        physical_manifest["clock_sensitivity"] = old_physical_manifest.get(
            "clock_sensitivity"
        )
        physical_manifest["derivative_repair"] = public_repair
        atomic_json(paths.physical_manifest, physical_manifest)

        outputs = _known_outputs(repository_root, paths)
        output_hashes = _outputs_with_hashes(repository_root, outputs)
        after_hashes = {
            name: sha256_file(path) for name, path in _target_paths(paths).items()
            if name != "branch_manifest"
        }
        public_repair["after_hashes"] = after_hashes
        stages["overlap"] = overlap
        stages["overlap"]["derivative_repair"] = public_repair
        stages["calibration"] = {
            **dict(stages.get("calibration", {})),
            "numeric_model_reused_after_local_clock_repair": True,
            "timestamp_local_used_by_numeric_model": False,
            "calibrated_twin_invariant": True,
            "calibration_evidence": calibration_evidence,
            "milp_surrogate_repaired_and_accepted": True,
        }
        stages["transfer"] = {
            "calibration_id": transfer["calibration_id"],
            "transfer_manifest": _relative(repository_root, paths.transfer_manifest),
            "climate_support_diagnostics": climate_manifest,
            "derivative_repair": public_repair,
        }
        stages["physical_ablations"] = {
            "status": physical_ablations["status"],
            "manifest": _relative(
                repository_root, paths.physical_ablation_manifest
            ),
            "table": _relative(repository_root, paths.physical_ablation_table),
            "row_count": physical_ablations["row_count"],
            "not_evaluable_variants": physical_ablations[
                "not_evaluable_variants"
            ],
        }
        result = dict(previous["result"])
        result.update(
            {
                "status": "completed",
                "resumed": True,
                "resume_reason": "targeted_timezone_and_climate_derivative_repair",
                "request_fingerprint": request_fingerprint,
                "finished_at": utc_now(),
                "outputs": outputs,
                "output_hashes": output_hashes,
                "workload_events_contract": previous["workload_events_contract"],
                "frozen_artifact": transfer["frozen_artifact"],
                "recovery_record": public_repair,
                "blockers": [],
            }
        )
        manifest = {
            **dict(previous),
            "status": "completed",
            "request_fingerprint": request_fingerprint,
            "source_identities": source_identities,
            "config": _jsonable_config(cfg),
            "stages": stages,
            "outputs": outputs,
            "output_hashes": output_hashes,
            "workload_events_contract": previous["workload_events_contract"],
            "recovery_record": public_repair,
            "result": result,
        }
        atomic_json(paths.manifest, manifest)
        final_workload_sha256 = sha256_file(workload_path)
        if final_workload_sha256 != workload_sha256:
            raise RuntimeError("protected workload changed before repair commit")
        recovery.update(public_repair)
        recovery.update(
            {
                "status": "completed",
                "branch_manifest_sha256_after": sha256_file(paths.manifest),
                "protected_workload_sha256_after": final_workload_sha256,
            }
        )
        atomic_json(recovery_path, recovery)
        result["repair_manifest_path"] = _relative(repository_root, recovery_path)
        result["repair_manifest_sha256"] = sha256_file(recovery_path)
        result["repair_resumed"] = False
        return result
    except Exception as exc:
        _restore_targets(repository_root, targets)
        after_rollback = sha256_file(workload_path)
        recovery.update(
            {
                "status": "rolled_back_after_failure",
                "rolled_back_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "protected_workload_sha256_after": after_rollback,
                "protected_workload_modified": after_rollback != workload_sha256,
            }
        )
        atomic_json(recovery_path, recovery)
        if after_rollback != workload_sha256:
            raise RuntimeError(
                "protected workload hash changed despite derivative-only repair"
            ) from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely repair completed physical overlap/transfer derivatives."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--run-id", default="paper-physical-v1")
    parser.add_argument("--expected-workload-sha256")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = root / config_path
    result = repair_completed_physical_derivatives(
        root,
        load_yaml(config_path),
        arguments.run_id,
        expected_workload_sha256=arguments.expected_workload_sha256,
    )
    summary = {
        "status": result["status"],
        "run_id": result["run_id"],
        "request_fingerprint": result["request_fingerprint"],
        "repair_resumed": result.get("repair_resumed"),
        "repair_manifest_path": result.get("repair_manifest_path"),
        "repair_manifest_sha256": result.get("repair_manifest_sha256"),
        "recovery_record": result.get("recovery_record"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
