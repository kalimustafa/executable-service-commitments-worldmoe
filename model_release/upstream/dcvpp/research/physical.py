from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.data.srrl import fetch_and_persist_srrl_daily, parse_srrl_daily_csv


SOURCE_RELATIONSHIP = "third_party_referenced_public_data"
MEASUREMENT_RELATIONSHIP = "not_measured_by_this_study"
NYISO_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
FIXED_MST = timezone(timedelta(hours=-7))


class PhysicalBranchError(RuntimeError):
    """A truthful, machine-readable physical-branch gate failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class BranchPaths:
    run_id: str
    nlr295_archive: Path
    nlr300_pue: Path
    nlr300_weather: Path
    srrl_raw: Path
    processed: Path
    artifacts: Path
    nlr295_parts: Path
    nlr295_deduplicated: Path
    srrl_parts: Path
    nlr295_workload_events: Path
    nlr295_job_energy: Path
    nlr300_primary: Path
    nlr300_weather_15m: Path
    nlr300_fixed_mst: Path
    srrl_solar_15m: Path
    overlap_primary: Path
    overlap_fixed_mst: Path
    job_progress: Path
    nlr300_progress: Path
    srrl_download_progress: Path
    srrl_panel_progress: Path
    overlap_manifest: Path
    split_manifest: Path
    physical_manifest: Path
    transfer_manifest: Path
    job_metrics: Path
    facility_metrics: Path
    pue_metrics: Path
    uncertainty_metrics: Path
    energy_coverage: Path
    component_closure: Path
    physical_parameters: Path
    zone_twins: Path
    transfer_table: Path
    surrogate: Path
    frozen_artifact: Path
    physical_ablation_table: Path
    physical_ablation_manifest: Path
    manifest: Path


def _safe_run_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    if not safe:
        raise ValueError("run_id must contain at least one safe character")
    return safe[:120]


def _resolve(root: Path, value: Any, default: str) -> Path:
    candidate = Path(str(value)) if value not in (None, "") else Path(default)
    return candidate if candidate.is_absolute() else root / candidate


def _branch_config(config: Mapping[str, Any]) -> dict[str, Any]:
    nested = config.get("research_physical", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise TypeError("config['research_physical'] must be a mapping")
    cfg = dict(nested)
    # Keep branch-local access to declared transfer assumptions without
    # mutating or depending on a shared configuration module.
    for key in ("physical_twin", "digital_twins", "sites"):
        if key in config and key not in cfg:
            cfg[key] = config[key]
    cfg.setdefault("seed", int(config.get("seed", 20260809)))
    cfg.setdefault("interval_minutes", int(config.get("interval_minutes", 15)))
    cfg.setdefault("batch_size", 65_536)
    cfg.setdefault("job_model_epochs", 3)
    cfg.setdefault("job_min_train_rows", 1_000)
    cfg.setdefault("job_gpu_min_train_rows", 500)
    cfg.setdefault("facility_min_rows", 2_880)
    cfg.setdefault("facility_min_source_days", 30)
    cfg.setdefault("surrogate_knots", 6)
    cfg.setdefault("surrogate_acceptance_max_observed_rmse_ratio", 1.25)
    cfg.setdefault("surrogate_acceptance_max_calibrated_mae_fraction", 0.01)
    cfg.setdefault("surrogate_acceptance_min_calibrated_mae_kw", 5.0)
    cfg.setdefault("physical_multiplier_draws", 256)
    cfg.setdefault("download_srrl", True)
    cfg.setdefault("srrl_workers", 8)
    cfg.setdefault("srrl_retries", 3)
    cfg.setdefault("srrl_checkpoint_every", 10)
    cfg.setdefault("minimum_srrl_usable_day_fraction", 0.90)
    cfg.setdefault("resume", True)
    return cfg


def _configured_zone_weather_identity(
    root: Path, cfg: Mapping[str, Any]
) -> dict[str, Any] | None:
    configured = cfg.get("nyiso_zone_weather_path")
    if not configured:
        return None
    path = Path(str(configured))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return None
    return _file_identity(root, path)


def _physical_implementation_sha256() -> str:
    implementation_files = (
        Path(__file__),
        Path(__file__).with_name("physical_ablations.py"),
    )
    return sha256_json(
        {path.name: sha256_file(path) for path in implementation_files}
    )


def _paths(root: Path, cfg: Mapping[str, Any], run_id: str) -> BranchPaths:
    run = _safe_run_id(run_id)
    processed = _resolve(
        root,
        cfg.get("processed_root"),
        f"data/processed/research/physical/run_id={run}",
    )
    artifacts = _resolve(
        root,
        cfg.get("artifact_root"),
        f"artifacts/research/physical/run_id={run}",
    )
    return BranchPaths(
        run_id=run,
        nlr295_archive=_resolve(
            root,
            cfg.get("nlr295_archive"),
            "data/raw/nlr295/esif.hpc.eagle.job-anon-energy-metrics.zip",
        ),
        nlr300_pue=_resolve(
            root,
            cfg.get("nlr300_pue"),
            "data/raw/nlr300/esif.influx.buildingData.PUE.combined.parquet",
        ),
        nlr300_weather=_resolve(
            root,
            cfg.get("nlr300_weather"),
            "data/raw/nlr300/esif.influx.buildingData.outside.combined.parquet",
        ),
        srrl_raw=_resolve(root, cfg.get("srrl_raw"), "data/raw/srrl"),
        processed=processed,
        artifacts=artifacts,
        nlr295_parts=processed / "_streaming_nlr295_parts",
        nlr295_deduplicated=processed / "_nlr295_deduplicated.parquet",
        srrl_parts=processed / "_streaming_srrl_parts",
        nlr295_workload_events=processed / "nlr295_workload_events.parquet",
        nlr295_job_energy=processed / "nlr295_job_energy.parquet",
        nlr300_primary=processed / "nlr300_facility_15m.parquet",
        nlr300_weather_15m=processed / "nlr300_weather_15m.parquet",
        nlr300_fixed_mst=processed / "nlr300_facility_15m_fixed_mst.parquet",
        srrl_solar_15m=processed / "srrl_solar_15m.parquet",
        overlap_primary=processed / "verified_overlap_primary_15m.parquet",
        overlap_fixed_mst=processed / "verified_overlap_fixed_mst_15m.parquet",
        job_progress=artifacts / "nlr295_member_progress.json",
        nlr300_progress=artifacts / "nlr300_processing_progress.json",
        srrl_download_progress=artifacts / "srrl_download_progress.json",
        srrl_panel_progress=artifacts / "srrl_panel_progress.json",
        overlap_manifest=artifacts / "source_overlap_manifest.json",
        split_manifest=artifacts / "physical_split_manifest.json",
        physical_manifest=artifacts / "physical_calibration_manifest.json",
        transfer_manifest=artifacts / "digital_twin_transfer_manifest.json",
        job_metrics=artifacts / "nlr295_job_energy_metrics.parquet",
        facility_metrics=artifacts / "nlr_physical_validation_metrics.parquet",
        pue_metrics=artifacts / "pue_validation.parquet",
        uncertainty_metrics=artifacts / "uncertainty_calibration.parquet",
        energy_coverage=artifacts / "nlr295_energy_coverage.parquet",
        component_closure=artifacts / "component_closure.parquet",
        physical_parameters=processed / "physical_calibration_parameters.parquet",
        zone_twins=processed / "nyiso_zone_twins.parquet",
        transfer_table=artifacts / "digital_twin_transfer_parameters.parquet",
        surrogate=artifacts / "frozen_piecewise_surrogate.json",
        frozen_artifact=artifacts / "frozen_physical_artifact.json",
        physical_ablation_table=artifacts / "physical_transfer_ablation_results.parquet",
        physical_ablation_manifest=artifacts / "physical_transfer_ablation_manifest.json",
        manifest=artifacts / "physical_branch_manifest.json",
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _jsonable_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        if str(key).startswith("_") or callable(value):
            continue
        if isinstance(value, Path):
            result[str(key)] = value.as_posix()
        elif isinstance(value, Mapping):
            result[str(key)] = _jsonable_config(value)
        elif isinstance(value, (str, int, float, bool, list, tuple)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _recorded_output_hashes_match(records: Mapping[str, Any]) -> bool:
    if not records:
        return False
    for value in records.values():
        if not isinstance(value, Mapping):
            return False
        path = Path(str(value.get("path", "")))
        if not path.is_file() or value.get("sha256") != sha256_file(path):
            return False
    return True


def _nlr295_identity(root: Path, archive: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "local_path": _relative(root, archive),
        "byte_count": archive.stat().st_size,
        "mtime_ns": archive.stat().st_mtime_ns,
    }
    inventory_path = root / "artifacts/manifests/nlr295_inventory_manifest.json"
    inventory = _load_json(inventory_path, {})
    archived = inventory.get("archive", {}) if isinstance(inventory, Mapping) else {}
    if (
        isinstance(archived, Mapping)
        and archived.get("byte_count") == identity["byte_count"]
        and archived.get("sha256")
    ):
        identity.update(
            {
                "sha256": str(archived["sha256"]),
                "sha256_source": "verified_inventory_manifest",
                "publisher_md5": archived.get("publisher_md5_actual"),
                "publisher_md5_verified": bool(archived.get("publisher_md5_verified")),
                "inventory_manifest_sha256": sha256_file(inventory_path),
            }
        )
    else:
        identity.update({"sha256": sha256_file(archive), "sha256_source": "computed"})
    return identity


def _file_identity(root: Path, path: Path) -> dict[str, Any]:
    return {
        "local_path": _relative(root, path),
        "byte_count": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": sha256_file(path),
    }


def _required_source_identities(root: Path, paths: BranchPaths) -> dict[str, Any]:
    missing = [
        _relative(root, path)
        for path in (paths.nlr295_archive, paths.nlr300_pue, paths.nlr300_weather)
        if not path.is_file()
    ]
    if missing:
        raise PhysicalBranchError("MISSING_REQUIRED_PHYSICAL_SOURCE", ", ".join(missing))
    return {
        "nlr295": _nlr295_identity(root, paths.nlr295_archive),
        "nlr300_pue": _file_identity(root, paths.nlr300_pue),
        "nlr300_weather": _file_identity(root, paths.nlr300_weather),
    }


def _output_paths_exist(root: Path, manifest: Mapping[str, Any]) -> bool:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, Mapping):
        return False
    for value in outputs.values():
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            return False
    return True


NLR295_SOURCE_COLUMNS = (
    "id",
    "job_id",
    "array_pos",
    "name_hash",
    "user_hash",
    "submit_line_hash",
    "submit_time_tz",
    "start_time_tz",
    "end_time_tz",
    "partition",
    "qos",
    "state",
    "state_simple",
    "queue_wait",
    "nodes_req",
    "nodes_used",
    "processors_req",
    "processors_used",
    "memory_req",
    "wallclock_req",
    "wallclock_used",
    "gpus_requested",
    "node_energy_total_watt_hours",
    "gpu0_energy_total_watt_hours",
    "gpu1_energy_total_watt_hours",
    "cpu_energy_tdp_estimated_max_watt_hours",
    "cpu_energy_tdp_estimated_used_watt_hours",
)


NLR295_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("job_key", pa.string()),
        pa.field("job_family_key", pa.string()),
        pa.field("array_pos", pa.float64()),
        pa.field("submit_time_utc", pa.timestamp("us", tz="UTC")),
        pa.field("start_time_utc", pa.timestamp("us", tz="UTC")),
        pa.field("end_time_utc", pa.timestamp("us", tz="UTC")),
        pa.field("partition", pa.string()),
        pa.field("qos", pa.string()),
        pa.field("state", pa.string()),
        pa.field("state_simple", pa.string()),
        pa.field("queue_wait_hours", pa.float64()),
        pa.field("wallclock_requested_hours", pa.float64()),
        pa.field("wallclock_used_hours", pa.float64()),
        pa.field("memory_requested_raw", pa.string()),
        pa.field("duration_hours", pa.float64()),
        pa.field("nodes_used", pa.float64()),
        pa.field("processors_used", pa.float64()),
        pa.field("gpus_requested", pa.float64()),
        pa.field("node_hours", pa.float64()),
        pa.field("processor_hours", pa.float64()),
        pa.field("gpu_hours", pa.float64()),
        pa.field("resource_scale", pa.string()),
        pa.field("node_energy_monitoring_wh", pa.float64()),
        pa.field("gpu_energy_monitoring_wh", pa.float64()),
        pa.field("cpu_energy_tdp_estimated_max_wh", pa.float64()),
        pa.field("cpu_energy_tdp_estimated_used_wh", pa.float64()),
        pa.field("node_monitoring_available", pa.bool_()),
        pa.field("gpu_monitoring_available", pa.bool_()),
        pa.field("cpu_tdp_estimate_available", pa.bool_()),
        pa.field("energy_observed_node", pa.bool_()),
        pa.field("energy_observed_gpu", pa.bool_()),
        pa.field("energy_estimated_cpu", pa.bool_()),
        pa.field("source_member", pa.string()),
        pa.field("source_member_crc32", pa.string()),
        pa.field("raw_source_file", pa.string()),
        pa.field("raw_source_hash", pa.string()),
        pa.field("raw_source_hash_algorithm", pa.string()),
        pa.field("source_timezone", pa.string()),
        pa.field("source_relationship", pa.string()),
        pa.field("measurement_relationship", pa.string()),
        pa.field("evidence_role", pa.string()),
        pa.field("record_type", pa.string()),
    ]
)


def _safe_zip_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return not (
        normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or ".." in Path(normalized).parts
    )


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _duration_hours(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = frame[column]
    if pd.api.types.is_timedelta64_dtype(values.dtype):
        return values.dt.total_seconds().astype("float64") / 3600.0
    parsed = pd.to_timedelta(values, errors="coerce")
    return parsed.dt.total_seconds().astype("float64") / 3600.0


def _standardize_nlr295_batch(
    batch: pa.RecordBatch,
    member: str,
    crc32: str,
    member_sha256: str,
) -> pa.Table:
    frame = batch.to_pandas()
    for required in ("submit_time_tz", "start_time_tz", "end_time_tz"):
        if required not in frame:
            raise PhysicalBranchError(
                "NLR295_AUTHORITATIVE_TIME_COLUMN_MISSING",
                f"{member} lacks {required}",
            )
    submit = pd.to_datetime(frame["submit_time_tz"], utc=True, errors="coerce")
    start = pd.to_datetime(frame["start_time_tz"], utc=True, errors="coerce")
    end = pd.to_datetime(frame["end_time_tz"], utc=True, errors="coerce")
    duration = (end - start).dt.total_seconds() / 3600.0
    nodes = _number(frame, "nodes_used")
    nodes = nodes.where(nodes.notna(), _number(frame, "nodes_req"))
    processors = _number(frame, "processors_used")
    processors = processors.where(processors.notna(), _number(frame, "processors_req"))
    gpus = _number(frame, "gpus_requested").fillna(0.0)
    node_energy = _number(frame, "node_energy_total_watt_hours")
    gpu0 = _number(frame, "gpu0_energy_total_watt_hours")
    gpu1 = _number(frame, "gpu1_energy_total_watt_hours")
    gpu_energy = pd.concat([gpu0, gpu1], axis=1).sum(axis=1, min_count=1)
    job_id = (
        frame["id"].astype("string")
        if "id" in frame
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    if "job_id" in frame:
        job_id = job_id.fillna(frame["job_id"].astype("string"))
    array_pos = _number(frame, "array_pos")
    name_hash = (
        frame["name_hash"].astype("string")
        if "name_hash" in frame
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    submit_line = (
        frame["submit_line_hash"].astype("string")
        if "submit_line_hash" in frame
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    missing_job_id = job_id.isna()
    if missing_job_id.any():
        fingerprint_frame = pd.DataFrame(
            {
                "member": member,
                "submit": submit.astype("string"),
                "start": start.astype("string"),
                "end": end.astype("string"),
                "name": name_hash,
                "submit_line": submit_line,
                "array_pos": array_pos,
            }
        )
        fallback_hash = pd.util.hash_pandas_object(
            fingerprint_frame, index=True
        ).astype("uint64")
        job_id.loc[missing_job_id] = (
            "anonymous:" + fallback_hash.loc[missing_job_id].astype("string")
        )
    family = job_id.copy()
    has_array = array_pos.notna() & name_hash.notna()
    family.loc[has_array] = (
        "array:"
        + name_hash.loc[has_array].fillna("")
        + ":"
        + submit_line.loc[has_array].fillna("")
    )
    valid = submit.notna()
    resource_scale = pd.Series("unknown", index=frame.index, dtype="string")
    resource_scale.loc[nodes.notna() & nodes.le(2)] = "small_1_2_nodes"
    resource_scale.loc[nodes.gt(2) & nodes.le(16)] = "medium_3_16_nodes"
    resource_scale.loc[nodes.gt(16)] = "large_gt_16_nodes"
    cpu_max = _number(frame, "cpu_energy_tdp_estimated_max_watt_hours")
    cpu_used = _number(frame, "cpu_energy_tdp_estimated_used_watt_hours")
    node_available = node_energy.where(node_energy.ge(0)).notna()
    gpu_available = gpu_energy.where(gpu_energy.ge(0)).notna()
    cpu_available = cpu_max.notna() | cpu_used.notna()
    out = pd.DataFrame(
        {
            "job_key": job_id,
            "job_family_key": family,
            "array_pos": array_pos,
            "submit_time_utc": submit,
            "start_time_utc": start,
            "end_time_utc": end,
            "partition": frame.get("partition", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "qos": frame.get("qos", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "state": frame.get("state", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "state_simple": frame.get(
                "state_simple", pd.Series(pd.NA, index=frame.index)
            ).astype("string"),
            "queue_wait_hours": _duration_hours(frame, "queue_wait"),
            "wallclock_requested_hours": _duration_hours(frame, "wallclock_req"),
            "wallclock_used_hours": _duration_hours(frame, "wallclock_used"),
            "memory_requested_raw": frame.get(
                "memory_req", pd.Series(pd.NA, index=frame.index)
            ).astype("string"),
            "duration_hours": duration,
            "nodes_used": nodes,
            "processors_used": processors,
            "gpus_requested": gpus,
            "node_hours": nodes * duration,
            "processor_hours": processors * duration,
            "gpu_hours": gpus * duration,
            "resource_scale": resource_scale,
            "node_energy_monitoring_wh": node_energy.where(node_energy.ge(0)),
            "gpu_energy_monitoring_wh": gpu_energy.where(gpu_energy.ge(0)),
            "cpu_energy_tdp_estimated_max_wh": cpu_max,
            "cpu_energy_tdp_estimated_used_wh": cpu_used,
            "node_monitoring_available": node_available,
            "gpu_monitoring_available": gpu_available,
            "cpu_tdp_estimate_available": cpu_available,
            "energy_observed_node": node_available,
            "energy_observed_gpu": gpu_available,
            "energy_estimated_cpu": cpu_available,
            "source_member": member,
            "source_member_crc32": crc32,
            "raw_source_file": member,
            "raw_source_hash": member_sha256,
            "raw_source_hash_algorithm": "sha256",
            "source_timezone": "UTC_as_encoded_in_authoritative_tz_fields",
            "source_relationship": SOURCE_RELATIONSHIP,
            "measurement_relationship": MEASUREMENT_RELATIONSHIP,
            "evidence_role": "observed",
            "record_type": "referenced_publisher_job_energy_record",
        }
    ).loc[valid]
    out = out[list(field.name for field in NLR295_OUTPUT_SCHEMA)]
    return pa.Table.from_pandas(out, schema=NLR295_OUTPUT_SCHEMA, preserve_index=False)


def _nlr295_part_name(index: int, member: str) -> str:
    suffix = sha256_json(member)[:10]
    return f"part-{index:05d}-{suffix}.parquet"


def _process_nlr295_member(
    archive: ZipFile,
    member: str,
    destination: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    info = archive.getinfo(member)
    crc32 = f"{info.CRC:08x}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nlr295-member-", dir=destination.parent) as tmp_dir:
        extracted = Path(tmp_dir) / "member.parquet"
        with archive.open(info) as source, extracted.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        member_sha256 = sha256_file(extracted)
        parquet = pq.ParquetFile(extracted)
        missing = [
            column
            for column in ("submit_time_tz", "start_time_tz", "end_time_tz")
            if column not in parquet.schema_arrow.names
        ]
        if missing:
            raise PhysicalBranchError(
                "NLR295_AUTHORITATIVE_TIME_COLUMN_MISSING",
                f"{member}: {missing}",
            )
        selected = [column for column in NLR295_SOURCE_COLUMNS if column in parquet.schema_arrow.names]
        temporary = Path(tmp_dir) / "output.parquet"
        writer = pq.ParquetWriter(temporary, NLR295_OUTPUT_SCHEMA, compression="zstd")
        row_count = 0
        monitoring_node = 0
        monitoring_gpu = 0
        min_submit: pd.Timestamp | None = None
        min_start: pd.Timestamp | None = None
        max_end: pd.Timestamp | None = None
        try:
            for batch in parquet.iter_batches(columns=selected, batch_size=batch_size):
                table = _standardize_nlr295_batch(
                    batch, member, crc32, member_sha256
                )
                if not table.num_rows:
                    continue
                writer.write_table(table)
                row_count += table.num_rows
                monitoring_node += int(pc.sum(table["node_monitoring_available"]).as_py() or 0)
                monitoring_gpu += int(pc.sum(table["gpu_monitoring_available"]).as_py() or 0)
                submits = table["submit_time_utc"].to_pandas()
                starts = table["start_time_utc"].to_pandas()
                ends = table["end_time_utc"].to_pandas()
                current_submit = submits.min()
                current_min = starts.min()
                current_max = ends.max()
                if pd.notna(current_submit):
                    min_submit = (
                        current_submit if min_submit is None else min(min_submit, current_submit)
                    )
                if pd.notna(current_min):
                    min_start = current_min if min_start is None else min(min_start, current_min)
                if pd.notna(current_max):
                    max_end = current_max if max_end is None else max(max_end, current_max)
        finally:
            writer.close()
            parquet.close()
        os.replace(temporary, destination)
    return {
        "member": member,
        "member_crc32": crc32,
        "member_sha256": member_sha256,
        "member_compressed_bytes": info.compress_size,
        "member_uncompressed_bytes": info.file_size,
        "output": destination.name,
        "output_sha256": sha256_file(destination),
        "row_count": row_count,
        "node_monitoring_rows": monitoring_node,
        "gpu_monitoring_rows": monitoring_gpu,
        "min_submit_utc": min_submit.isoformat() if min_submit is not None else None,
        "min_start_utc": min_start.isoformat() if min_start is not None else None,
        "max_end_utc": max_end.isoformat() if max_end is not None else None,
    }


def _duckdb_glob(directory: Path) -> str:
    return (directory.resolve() / "*.parquet").as_posix().replace("'", "''")


def _duckdb_source(path: Path) -> str:
    if path.is_dir():
        return _duckdb_glob(path)
    return path.resolve().as_posix().replace("'", "''")


def _duckdb_copy(query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.parquet")
    temporary.unlink(missing_ok=True)
    escaped = temporary.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
    finally:
        con.close()
    os.replace(temporary, destination)


def _sql_timestamptz(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"TIMESTAMPTZ '{escaped}'"


def _nlr295_stats(source: Path) -> dict[str, Any]:
    glob = _duckdb_source(source)
    con = duckdb.connect()
    row = con.execute(
        f"""
        SELECT count(*) AS row_count,
               min(submit_time_utc) AS min_submit,
               max(submit_time_utc) AS max_submit,
               min(start_time_utc) AS min_start,
               max(end_time_utc) AS max_end,
               sum(node_monitoring_available) AS node_rows,
               sum(gpu_monitoring_available) AS gpu_rows,
               quantile_disc(submit_time_utc, 0.60) AS split_60,
               quantile_disc(submit_time_utc, 0.80) AS split_80
        FROM read_parquet('{glob}', union_by_name=true)
        """
    ).fetchone()
    if not row or not row[0]:
        raise PhysicalBranchError("NLR295_NO_VALID_JOBS", "no valid start/end job rows")
    split_60 = pd.Timestamp(row[7]).tz_convert("UTC")
    split_80 = pd.Timestamp(row[8]).tz_convert("UTC")
    counts = con.execute(
        f"""
        WITH b AS (
          SELECT *, min(submit_time_utc) OVER (PARTITION BY job_family_key) AS family_submit_time_utc
          FROM read_parquet('{glob}', union_by_name=true)
        )
        SELECT
          sum(family_submit_time_utc < ?) AS train_rows,
          sum(family_submit_time_utc >= ? AND family_submit_time_utc < ?) AS validation_rows,
          sum(family_submit_time_utc >= ?) AS locked_test_rows
        FROM b
        """,
        [split_60, split_60, split_80, split_80],
    ).fetchone()
    con.close()
    return {
        "row_count": int(row[0]),
        "min_submit_utc": pd.Timestamp(row[1]).tz_convert("UTC").isoformat(),
        "max_submit_utc": pd.Timestamp(row[2]).tz_convert("UTC").isoformat(),
        "min_start_utc": pd.Timestamp(row[3]).tz_convert("UTC").isoformat(),
        "max_end_utc": pd.Timestamp(row[4]).tz_convert("UTC").isoformat(),
        "node_monitoring_rows": int(row[5] or 0),
        "gpu_monitoring_rows": int(row[6] or 0),
        "split_boundaries": {
            "train_end_exclusive": split_60.isoformat(),
            "validation_end_exclusive": split_80.isoformat(),
        },
        "split_counts": {
            "train": int(counts[0] or 0),
            "validation": int(counts[1] or 0),
            "locked_test": int(counts[2] or 0),
        },
    }


def _prepare_nlr295(
    paths: BranchPaths,
    archive_identity: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    paths.nlr295_parts.mkdir(parents=True, exist_ok=True)
    previous = _load_json(paths.job_progress, {}) if bool(cfg["resume"]) else {}
    stable_archive_identity = {
        field: archive_identity.get(field)
        for field in ("local_path", "byte_count", "sha256")
    }
    fingerprint = sha256_json(stable_archive_identity)
    # Legacy progress fingerprints included provenance fields that are
    # enriched after inventory validation.  Accept the exact legacy computed
    # identity only when immutable path/size/SHA content is unchanged.
    legacy_computed_identity = {
        field: archive_identity.get(field)
        for field in ("local_path", "byte_count", "mtime_ns", "sha256")
    }
    legacy_computed_identity["sha256_source"] = "computed"
    compatible_fingerprints = {
        fingerprint,
        sha256_json(legacy_computed_identity),
    }
    if previous and previous.get("archive_fingerprint") not in (
        None,
        *compatible_fingerprints,
    ):
        raise PhysicalBranchError(
            "RUN_ID_FINGERPRINT_CONFLICT",
            "NLR295 source changed for an existing run_id; use a new run_id",
        )
    finalized = previous.get("finalized", {}) if isinstance(previous, Mapping) else {}
    if (
        bool(cfg["resume"])
        and isinstance(finalized, Mapping)
        and isinstance(finalized.get("stats"), Mapping)
        and _recorded_output_hashes_match(finalized.get("outputs", {}))
    ):
        return dict(finalized["stats"])
    completed: dict[str, Any] = dict(previous.get("completed", {}))
    failures: list[dict[str, str]] = []
    with ZipFile(paths.nlr295_archive) as archive:
        members = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".parquet")
        ]
        if not members:
            raise PhysicalBranchError("NLR295_ARCHIVE_EMPTY", "no Parquet members")
        unsafe = [member for member in members if not _safe_zip_member(member)]
        if unsafe:
            raise PhysicalBranchError("NLR295_UNSAFE_ARCHIVE_PATH", unsafe[0])
        for index, member in enumerate(members):
            output = paths.nlr295_parts / _nlr295_part_name(index, member)
            recorded = completed.get(member, {})
            if (
                bool(cfg["resume"])
                and output.is_file()
                and recorded.get("member_crc32") == f"{archive.getinfo(member).CRC:08x}"
                and recorded.get("output_sha256") == sha256_file(output)
            ):
                continue
            try:
                completed[member] = _process_nlr295_member(
                    archive,
                    member,
                    output,
                    batch_size=int(cfg["batch_size"]),
                )
            except Exception as exc:  # retained per-member and raised after checkpoint
                failures.append({"member": member, "error": f"{type(exc).__name__}: {exc}"})
                break
            atomic_json(
                paths.job_progress,
                {
                    "schema_version": 1,
                    "archive_fingerprint": fingerprint,
                    "updated_at": utc_now(),
                    "completed": completed,
                    "failures": failures,
                },
            )
    atomic_json(
        paths.job_progress,
        {
            "schema_version": 1,
            "archive_fingerprint": fingerprint,
            "updated_at": utc_now(),
            "completed": completed,
            "failures": failures,
        },
    )
    if failures:
        raise PhysicalBranchError("NLR295_MEMBER_PROCESSING_FAILED", failures[0]["error"])
    parts_glob = _duckdb_glob(paths.nlr295_parts)
    con = duckdb.connect()
    raw_valid_rows = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet('{parts_glob}', union_by_name=true)"
        ).fetchone()[0]
    )
    con.close()
    _duckdb_copy(
        f"""
        SELECT *
        FROM read_parquet('{parts_glob}', union_by_name=true)
        QUALIFY row_number() OVER (
          PARTITION BY job_key
          ORDER BY submit_time_utc, source_member, source_member_crc32
        ) = 1
        ORDER BY submit_time_utc, job_key
        """,
        paths.nlr295_deduplicated,
    )
    stats = _nlr295_stats(paths.nlr295_deduplicated)
    stats["pre_deduplication_valid_submit_rows"] = raw_valid_rows
    stats["duplicate_job_rows_removed"] = raw_valid_rows - stats["row_count"]
    glob = _duckdb_source(paths.nlr295_deduplicated)
    split_60 = _sql_timestamptz(stats["split_boundaries"]["train_end_exclusive"])
    split_80 = _sql_timestamptz(stats["split_boundaries"]["validation_end_exclusive"])
    split_expr = (
        f"CASE WHEN family_submit_time_utc < {split_60} THEN 'train' "
        f"WHEN family_submit_time_utc < {split_80} THEN 'validation' ELSE 'locked_test' END"
    )
    _duckdb_copy(
        f"""
        WITH b AS (
          SELECT *, min(submit_time_utc) OVER (PARTITION BY job_family_key) AS family_submit_time_utc
          FROM read_parquet('{glob}', union_by_name=true)
        )
        SELECT job_key, job_family_key, array_pos, submit_time_utc,
               start_time_utc, end_time_utc, partition, qos, state,
               state_simple, queue_wait_hours, wallclock_requested_hours,
               wallclock_used_hours, memory_requested_raw, duration_hours,
               nodes_used, processors_used, gpus_requested, node_hours,
               processor_hours, gpu_hours, resource_scale,
               source_member, source_member_crc32, raw_source_file,
               raw_source_hash, raw_source_hash_algorithm, source_timezone,
               {split_expr} AS split, source_relationship,
               measurement_relationship, evidence_role, record_type
        FROM b
        ORDER BY submit_time_utc, job_family_key, job_key
        """,
        paths.nlr295_workload_events,
    )
    _duckdb_copy(
        f"""
        WITH b AS (
          SELECT *, min(submit_time_utc) OVER (PARTITION BY job_family_key) AS family_submit_time_utc
          FROM read_parquet('{glob}', union_by_name=true)
        )
        SELECT * EXCLUDE (family_submit_time_utc), {split_expr} AS split
        FROM b
        WHERE node_monitoring_available OR gpu_monitoring_available
              OR cpu_energy_tdp_estimated_used_wh IS NOT NULL
              OR cpu_energy_tdp_estimated_max_wh IS NOT NULL
        ORDER BY submit_time_utc, job_family_key, job_key
        """,
        paths.nlr295_job_energy,
    )
    con = duckdb.connect()
    coverage = con.execute(
        f"""
        WITH b AS (
          SELECT *, min(submit_time_utc) OVER (PARTITION BY job_family_key) AS family_submit_time_utc
          FROM read_parquet('{glob}', union_by_name=true)
        )
        SELECT {split_expr} AS split, 'overall' AS dimension, 'all' AS value,
               count(*) AS job_rows,
               sum(node_monitoring_available) AS node_monitoring_rows,
               sum(gpu_monitoring_available) AS gpu_monitoring_rows,
               sum(cpu_energy_tdp_estimated_used_wh IS NOT NULL) AS cpu_tdp_estimated_rows,
               avg(node_monitoring_available::INTEGER) AS node_monitoring_fraction,
               avg(gpu_monitoring_available::INTEGER) AS gpu_monitoring_fraction
        FROM b GROUP BY 1
        UNION ALL
        SELECT {split_expr}, 'partition', coalesce(partition, '<missing>'),
               count(*), sum(node_monitoring_available), sum(gpu_monitoring_available),
               sum(cpu_energy_tdp_estimated_used_wh IS NOT NULL),
               avg(node_monitoring_available::INTEGER), avg(gpu_monitoring_available::INTEGER)
        FROM b GROUP BY 1, 3
        UNION ALL
        SELECT {split_expr}, 'qos', coalesce(qos, '<missing>'),
               count(*), sum(node_monitoring_available), sum(gpu_monitoring_available),
               sum(cpu_energy_tdp_estimated_used_wh IS NOT NULL),
               avg(node_monitoring_available::INTEGER), avg(gpu_monitoring_available::INTEGER)
        FROM b GROUP BY 1, 3
        UNION ALL
        SELECT {split_expr}, 'state_simple', coalesce(state_simple, '<missing>'),
               count(*), sum(node_monitoring_available), sum(gpu_monitoring_available),
               sum(cpu_energy_tdp_estimated_used_wh IS NOT NULL),
               avg(node_monitoring_available::INTEGER), avg(gpu_monitoring_available::INTEGER)
        FROM b GROUP BY 1, 3
        UNION ALL
        SELECT {split_expr}, 'resource_scale', coalesce(resource_scale, '<missing>'),
               count(*), sum(node_monitoring_available), sum(gpu_monitoring_available),
               sum(cpu_energy_tdp_estimated_used_wh IS NOT NULL),
               avg(node_monitoring_available::INTEGER), avg(gpu_monitoring_available::INTEGER)
        FROM b GROUP BY 1, 3
        UNION ALL
        SELECT {split_expr}, 'submit_month', strftime(submit_time_utc, '%Y-%m'),
               count(*), sum(node_monitoring_available), sum(gpu_monitoring_available),
               sum(cpu_energy_tdp_estimated_used_wh IS NOT NULL),
               avg(node_monitoring_available::INTEGER), avg(gpu_monitoring_available::INTEGER)
        FROM b GROUP BY 1, 3
        ORDER BY split, dimension, value
        """
    ).fetchdf()
    con.close()
    coverage["source_relationship"] = SOURCE_RELATIONSHIP
    coverage["measurement_relationship"] = MEASUREMENT_RELATIONSHIP
    _atomic_parquet(coverage, paths.energy_coverage)
    stats.update(
        {
            "parquet_member_count": len(completed),
            "dataset_path": paths.nlr295_deduplicated.as_posix(),
            "streaming_parts_path": paths.nlr295_parts.as_posix(),
            "workload_events_path": paths.nlr295_workload_events.as_posix(),
            "job_energy_path": paths.nlr295_job_energy.as_posix(),
            "energy_coverage_path": paths.energy_coverage.as_posix(),
            "streaming_member_checkpoint": paths.job_progress.as_posix(),
            "source_relationship": SOURCE_RELATIONSHIP,
            "measurement_relationship": MEASUREMENT_RELATIONSHIP,
            "variable_boundary": {
                "node_gpu_energy": "publisher_reported_monitoring_derived",
                "cpu_energy": "publisher_reported_tdp_estimated",
                "missing_monitoring": "null_not_zero",
            },
        }
    )
    finalized_outputs = {
        "deduplicated": {
            "path": paths.nlr295_deduplicated.as_posix(),
            "sha256": sha256_file(paths.nlr295_deduplicated),
        },
        "workload_events": {
            "path": paths.nlr295_workload_events.as_posix(),
            "sha256": sha256_file(paths.nlr295_workload_events),
        },
        "job_energy": {
            "path": paths.nlr295_job_energy.as_posix(),
            "sha256": sha256_file(paths.nlr295_job_energy),
        },
        "energy_coverage": {
            "path": paths.energy_coverage.as_posix(),
            "sha256": sha256_file(paths.energy_coverage),
        },
    }
    atomic_json(
        paths.job_progress,
        {
            "schema_version": 1,
            "archive_fingerprint": fingerprint,
            "updated_at": utc_now(),
            "completed": completed,
            "failures": [],
            "finalized": {"stats": stats, "outputs": finalized_outputs},
        },
    )
    return stats


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _choose_column(columns: set[str], candidates: Sequence[str], *, source: Path) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise PhysicalBranchError(
        "PHYSICAL_SOURCE_SCHEMA_MISSING",
        f"{source.name} lacks one of {list(candidates)}",
    )


def _prepare_nlr300(
    paths: BranchPaths,
    source_identities: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    interval = int(cfg["interval_minutes"])
    if interval <= 0 or 60 % interval:
        raise ValueError("interval_minutes must be a positive divisor of 60")
    pue_columns = _parquet_columns(paths.nlr300_pue)
    weather_columns = _parquet_columns(paths.nlr300_weather)
    required_pue = {
        "ts",
        "cooling_kw",
        "hvac_kw",
        "it_power_kw",
        "plug_and_light_kw",
        "pue",
        "pump_kw",
    }
    missing_pue = sorted(required_pue - pue_columns)
    if missing_pue:
        raise PhysicalBranchError("NLR300_PUE_SCHEMA_MISSING", str(missing_pue))
    weather_temp = _choose_column(
        weather_columns,
        ("outdoor_air_temp", "outside_air_temp"),
        source=paths.nlr300_weather,
    )
    weather_humidity = _choose_column(
        weather_columns,
        ("outdoor_air_humidity", "outside_air_humidity"),
        source=paths.nlr300_weather,
    )
    pue_path = paths.nlr300_pue.resolve().as_posix().replace("'", "''")
    weather_path = paths.nlr300_weather.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    query = f"""
        WITH p AS (
          SELECT time_bucket(INTERVAL '{interval} minutes', ts) AS source_ts,
                 avg(cooling_kw) AS cooling_kw,
                 avg(hvac_kw) AS hvac_kw,
                 avg(it_power_kw) AS it_power_kw,
                 avg(plug_and_light_kw) AS plug_and_light_kw,
                 avg(pue) AS pue,
                 avg(pump_kw) AS pump_kw,
                 count(*) AS pue_source_rows
          FROM read_parquet('{pue_path}')
          WHERE ts >= TIMESTAMP '2000-01-01'
          GROUP BY 1
        ), w AS (
          SELECT time_bucket(INTERVAL '{interval} minutes', ts) AS source_ts,
                 avg({weather_temp}) AS outdoor_air_temp_f,
                 avg({weather_humidity}) AS outdoor_air_humidity,
                 count(*) AS weather_source_rows
          FROM read_parquet('{weather_path}')
          WHERE ts >= TIMESTAMP '2000-01-01'
          GROUP BY 1
        )
        SELECT p.*, w.outdoor_air_temp_f, w.outdoor_air_humidity,
               w.weather_source_rows
        FROM p INNER JOIN w USING (source_ts)
        ORDER BY source_ts
    """
    frame = con.execute(query).fetchdf()
    con.close()
    if frame.empty:
        raise PhysicalBranchError("NLR300_NO_COMMON_POWER_WEATHER", "15-minute join is empty")
    frame["source_ts"] = pd.to_datetime(frame["source_ts"], errors="raise")
    frame["timestamp_utc"] = frame["source_ts"].dt.tz_localize("UTC")
    frame["timestamp_local"] = frame["timestamp_utc"].dt.tz_convert("America/Denver")
    frame["outdoor_air_temp_c"] = (frame["outdoor_air_temp_f"] - 32.0) * (5.0 / 9.0)
    frame["facility_power_kw"] = frame["pue"] * frame["it_power_kw"]
    component_columns = [
        "it_power_kw",
        "cooling_kw",
        "hvac_kw",
        "plug_and_light_kw",
        "pump_kw",
    ]
    frame["component_power_kw"] = frame[component_columns].sum(axis=1, min_count=len(component_columns))
    # The publisher README states that 2.67 kW of tower-filter-pump load is not
    # reflected in pump_kw. Keep the correction explicit rather than changing
    # the source field.
    frame["documented_unmetered_pump_kw"] = 2.67
    frame["component_power_with_documented_residual_kw"] = (
        frame["component_power_kw"] + frame["documented_unmetered_pump_kw"]
    )
    frame["component_closure_residual_kw"] = (
        frame["facility_power_kw"] - frame["component_power_with_documented_residual_kw"]
    )
    frame["source_clock_assumption"] = "UTC"
    frame["analysis_timezone"] = "America/Denver"
    frame["timezone_evidence_role"] = "assumed"
    frame["publisher_timezone_metadata"] = "absent"
    frame["source_relationship"] = SOURCE_RELATIONSHIP
    frame["measurement_relationship"] = MEASUREMENT_RELATIONSHIP
    frame["evidence_role"] = "derived"
    frame["record_type"] = "referenced_facility_record_with_derived_total"
    frame["synthetic"] = False
    primary_columns = [
        "source_ts",
        "timestamp_utc",
        "timestamp_local",
        "cooling_kw",
        "hvac_kw",
        "it_power_kw",
        "plug_and_light_kw",
        "pump_kw",
        "pue",
        "facility_power_kw",
        "component_power_kw",
        "documented_unmetered_pump_kw",
        "component_power_with_documented_residual_kw",
        "component_closure_residual_kw",
        "outdoor_air_temp_f",
        "outdoor_air_temp_c",
        "outdoor_air_humidity",
        "pue_source_rows",
        "weather_source_rows",
        "source_clock_assumption",
        "analysis_timezone",
        "timezone_evidence_role",
        "publisher_timezone_metadata",
        "source_relationship",
        "measurement_relationship",
        "evidence_role",
        "record_type",
        "synthetic",
    ]
    frame = frame[primary_columns]
    _atomic_parquet(frame, paths.nlr300_primary)
    _atomic_parquet(
        frame[
            [
                "source_ts",
                "timestamp_utc",
                "timestamp_local",
                "outdoor_air_temp_f",
                "outdoor_air_temp_c",
                "outdoor_air_humidity",
                "weather_source_rows",
                "source_clock_assumption",
                "analysis_timezone",
                "timezone_evidence_role",
                "publisher_timezone_metadata",
                "source_relationship",
                "measurement_relationship",
                "evidence_role",
                "synthetic",
            ]
        ],
        paths.nlr300_weather_15m,
    )
    fixed = frame.copy()
    fixed["timestamp_utc"] = fixed["source_ts"].dt.tz_localize(FIXED_MST).dt.tz_convert("UTC")
    fixed["timestamp_local"] = fixed["source_ts"].dt.tz_localize(FIXED_MST)
    fixed["source_clock_assumption"] = "fixed_MST_UTC_minus_07_sensitivity"
    fixed["analysis_timezone"] = "fixed_MST"
    fixed["timezone_evidence_role"] = "assumed_sensitivity"
    _atomic_parquet(fixed, paths.nlr300_fixed_mst)

    valid = frame[
        frame["facility_power_kw"].gt(0)
        & frame["it_power_kw"].gt(0)
        & frame["pue"].between(1.0, 3.0)
    ].copy()
    if valid.empty:
        raise PhysicalBranchError("NLR300_NO_VALID_FACILITY_ROWS", "PUE/IT filters removed all rows")
    closure = pd.DataFrame(
        [
            {
                "interpretation": "primary_assumed_utc",
                "row_count": len(valid),
                "closure_rows": int(valid["component_closure_residual_kw"].notna().sum()),
                "closure_mae_kw": float(valid["component_closure_residual_kw"].abs().mean()),
                "closure_bias_kw": float(valid["component_closure_residual_kw"].mean()),
                "closure_p95_abs_kw": float(
                    valid["component_closure_residual_kw"].abs().quantile(0.95)
                ),
                "documented_unmetered_pump_kw": 2.67,
                "source_relationship": SOURCE_RELATIONSHIP,
                "evidence_role": "derived",
            }
        ]
    )
    _atomic_parquet(closure, paths.component_closure)
    return {
        "row_count": len(frame),
        "valid_calibration_rows": len(valid),
        "min_utc": frame["timestamp_utc"].min().isoformat(),
        "max_utc": frame["timestamp_utc"].max().isoformat(),
        "primary_path": paths.nlr300_primary.as_posix(),
        "weather_path": paths.nlr300_weather_15m.as_posix(),
        "fixed_mst_sensitivity_path": paths.nlr300_fixed_mst.as_posix(),
        "component_closure_path": paths.component_closure.as_posix(),
        "source_hashes": {
            "nlr300_pue": source_identities["nlr300_pue"]["sha256"],
            "nlr300_weather": source_identities["nlr300_weather"]["sha256"],
        },
        "timezone_protocol": {
            "publisher_metadata": "absent",
            "primary_source_clock_assumption": "UTC",
            "canonical_join_clock": "UTC",
            "local_analysis_timezone": "America/Denver",
            "sensitivity": "fixed_MST_UTC_minus_07",
            "evidence_role": "assumed",
        },
    }


def _common_support_days(start_utc: str, end_utc: str) -> list[date]:
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    if end <= start:
        raise PhysicalBranchError("NO_THREE_SOURCE_TIME_SUPPORT", f"{start} to {end}")
    local_start = start.tz_convert(FIXED_MST).date()
    local_end = end.tz_convert(FIXED_MST).date()
    return [timestamp.date() for timestamp in pd.date_range(local_start, local_end, freq="D")]


def _srrl_paths(raw_root: Path, day: date) -> tuple[Path, Path]:
    stem = f"BMS_{day.strftime('%Y%m%d')}"
    return raw_root / f"{stem}.csv", raw_root / f"{stem}.provenance.json"


def _existing_srrl_day(raw_root: Path, day: date) -> dict[str, Any] | None:
    csv_path, provenance_path = _srrl_paths(raw_root, day)
    if not csv_path.is_file() or not provenance_path.is_file():
        return None
    provenance = _load_json(provenance_path, {})
    if not isinstance(provenance, Mapping) or provenance.get("query_day") != day.isoformat():
        return None
    if provenance.get("byte_count") not in (None, csv_path.stat().st_size):
        return None
    return {
        "day": day.isoformat(),
        "status": "existing",
        "local_path": csv_path.as_posix(),
        "byte_count": csv_path.stat().st_size,
        "sha256": provenance.get("sha256"),
        "schema": provenance.get("schema"),
    }


def _download_srrl_day(
    root: Path,
    day: date,
    *,
    retries: int,
    fetcher: Callable[..., Any],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _, provenance = fetcher(root, day)
            return {
                "day": day.isoformat(),
                "status": "downloaded",
                "attempts": attempt,
                "local_path": provenance["local_path"],
                "byte_count": provenance["byte_count"],
                "sha256": provenance["sha256"],
                "schema": provenance["schema"],
            }
        except Exception as exc:  # exact error is persisted after bounded retry
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    return {
        "day": day.isoformat(),
        "status": "failed",
        "attempts": retries,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def _acquire_srrl(
    root: Path,
    paths: BranchPaths,
    expected_days: Sequence[date],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    paths.srrl_raw.mkdir(parents=True, exist_ok=True)
    requested_days = list(expected_days)
    scope_truncated = False
    maximum = cfg.get("max_srrl_days")
    if maximum is not None and len(requested_days) > int(maximum):
        requested_days = requested_days[: int(maximum)]
        scope_truncated = True
    records: dict[str, Any] = {}
    missing: list[date] = []
    for day in requested_days:
        existing = _existing_srrl_day(paths.srrl_raw, day)
        if existing is None:
            missing.append(day)
        else:
            records[day.isoformat()] = existing
    if bool(cfg["download_srrl"]) and missing:
        custom_fetcher = cfg.get("_srrl_fetcher")
        if custom_fetcher is None:
            timeout_value = cfg.get("srrl_timeout", (20, 120))
            timeout = tuple(map(int, timeout_value))

            def fetcher(repository_root: Path, query_day: date) -> Any:
                return fetch_and_persist_srrl_daily(
                    repository_root,
                    query_day,
                    timeout=(timeout[0], timeout[1]),
                )

        else:
            fetcher = custom_fetcher
        workers = max(1, int(cfg["srrl_workers"]))
        checkpoint_every = max(1, int(cfg["srrl_checkpoint_every"]))
        completed_since_checkpoint = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _download_srrl_day,
                    root,
                    day,
                    retries=max(1, int(cfg["srrl_retries"])),
                    fetcher=fetcher,
                ): day
                for day in missing
            }
            for future in as_completed(futures):
                record = future.result()
                records[record["day"]] = record
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= checkpoint_every:
                    atomic_json(
                        paths.srrl_download_progress,
                        {
                            "schema_version": 1,
                            "expected_start": expected_days[0].isoformat(),
                            "expected_end": expected_days[-1].isoformat(),
                            "expected_day_count": len(expected_days),
                            "requested_day_count": len(requested_days),
                            "scope_truncated": scope_truncated,
                            "updated_at": utc_now(),
                            "days": records,
                        },
                    )
                    completed_since_checkpoint = 0
    elif missing:
        for day in missing:
            records[day.isoformat()] = {
                "day": day.isoformat(),
                "status": "missing_download_disabled",
                "error": "SRRL day is absent and download_srrl=false",
            }
    # Include expected days omitted only by an explicit bounded test/debug scope.
    if scope_truncated:
        for day in expected_days[len(requested_days) :]:
            records[day.isoformat()] = {
                "day": day.isoformat(),
                "status": "not_requested_scope_truncated",
                "error": "max_srrl_days truncated the governing common-support scope",
            }
    failures = [
        record
        for day, record in sorted(records.items())
        if record.get("status") not in {"existing", "downloaded"}
    ]
    successes = [
        record
        for day, record in sorted(records.items())
        if record.get("status") in {"existing", "downloaded"}
    ]
    result = {
        "schema_version": 1,
        "expected_start": expected_days[0].isoformat(),
        "expected_end": expected_days[-1].isoformat(),
        "expected_day_count": len(expected_days),
        "requested_day_count": len(requested_days),
        "available_day_count": len(successes),
        "failed_or_missing_day_count": len(failures),
        "scope_truncated": scope_truncated,
        "full_common_support_available": len(successes) == len(expected_days),
        "source_timezone": "fixed_MST_UTC_minus_07",
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "updated_at": utc_now(),
        "days": records,
        "failures": failures,
    }
    atomic_json(paths.srrl_download_progress, result)
    return result


SRRL_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ghi_wm2": (
        "Global CMP22 (vent/cor) [W/m^2]",
        "Global SR25 (vent/cor) [W/m^2]",
        "Global CMP22-1 (cor) [W/m^2]",
        "Global PSP (cor) [W/m^2]",
    ),
    "dni_wm2": (
        "Direct NIP [W/m^2]",
        "Direct sNIP [W/m^2]",
        "Direct CHP1-1 [W/m^2]",
        "Direct DR20 (cor) [W/m^2]",
    ),
    "dhi_wm2": (
        "Diffuse SR25 (vent/cor) [W/m^2]",
        "Diffuse CM22-1 (vent/cor) [W/m^2]",
        "Diffuse SB (cor) [W/m^2]",
        "Diffuse Research 1 [W/m^2]",
    ),
    "air_temp_c": (
        "Tower Dry Bulb Temp [deg C]",
        "Deck Dry Bulb Temp [deg C]",
        "SE Dry Bulb Temp [deg C]",
        "Data lab Dry Bulb Temp [deg C]",
    ),
    "relative_humidity_pct": (
        "Tower RH [%]",
        "Deck RH [%]",
        "SE RH [%]",
        "Data lab RH [%]",
    ),
}


def _resolve_srrl_columns(columns: Iterable[str]) -> dict[str, str]:
    available = set(columns)
    result: dict[str, str] = {}
    for canonical, candidates in SRRL_COLUMN_CANDIDATES.items():
        selected = next((candidate for candidate in candidates if candidate in available), None)
        if selected is None:
            raise PhysicalBranchError(
                "SRRL_REQUIRED_FIELD_MISSING",
                f"{canonical}; tried {list(candidates)}",
            )
        result[canonical] = selected
    return result


def _process_srrl_day(
    raw_path: Path,
    day: date,
    destination: Path,
    *,
    interval_minutes: int,
) -> dict[str, Any]:
    frame, schema = parse_srrl_daily_csv(raw_path.read_bytes(), expected_day=day)
    selected = _resolve_srrl_columns(frame.columns)
    values = pd.DataFrame({"timestamp_utc": frame["timestamp_utc"]})
    for canonical, source in selected.items():
        values[canonical] = pd.to_numeric(frame[source], errors="coerce")
    values = values.set_index("timestamp_utc").sort_index()
    frequency = f"{interval_minutes}min"
    panel = values.resample(frequency, origin="epoch", label="left", closed="left").mean()
    counts = values["ghi_wm2"].resample(
        frequency, origin="epoch", label="left", closed="left"
    ).count()
    panel["valid_ghi_source_rows"] = counts.astype("int32")
    panel["coverage_fraction"] = counts / float(interval_minutes)
    panel = panel.reset_index()
    panel["source_day"] = day.isoformat()
    panel["source_file"] = raw_path.name
    panel["source_schema_sha256"] = schema.sha256
    panel["source_relationship"] = SOURCE_RELATIONSHIP
    panel["measurement_relationship"] = MEASUREMENT_RELATIONSHIP
    panel["evidence_role"] = "observed"
    panel["record_type"] = "referenced_srrl_source_record"
    panel["synthetic"] = False
    _atomic_parquet(panel, destination)
    return {
        "day": day.isoformat(),
        "source_sha256": sha256_file(raw_path),
        "source_schema_sha256": schema.sha256,
        "field_mapping": selected,
        "output": destination.name,
        "output_sha256": sha256_file(destination),
        "row_count": len(panel),
        "min_utc": panel["timestamp_utc"].min().isoformat(),
        "max_utc": panel["timestamp_utc"].max().isoformat(),
    }


def _prepare_srrl_panel(
    paths: BranchPaths,
    acquisition: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    paths.srrl_parts.mkdir(parents=True, exist_ok=True)
    previous = _load_json(paths.srrl_panel_progress, {}) if bool(cfg["resume"]) else {}
    completed: dict[str, Any] = dict(previous.get("completed", {}))
    failures: list[dict[str, str]] = []
    for day_text, record in sorted(acquisition["days"].items()):
        if record.get("status") not in {"existing", "downloaded"}:
            continue
        day = date.fromisoformat(day_text)
        raw_path, _ = _srrl_paths(paths.srrl_raw, day)
        destination = paths.srrl_parts / f"part-{day.strftime('%Y%m%d')}.parquet"
        prior = completed.get(day_text, {})
        if (
            bool(cfg["resume"])
            and destination.is_file()
            and prior.get("source_sha256") == record.get("sha256")
            and prior.get("output_sha256") == sha256_file(destination)
        ):
            continue
        try:
            completed[day_text] = _process_srrl_day(
                raw_path,
                day,
                destination,
                interval_minutes=int(cfg["interval_minutes"]),
            )
        except Exception as exc:
            failures.append({"day": day_text, "error": f"{type(exc).__name__}: {exc}"})
        if len(completed) % max(1, int(cfg["srrl_checkpoint_every"])) == 0:
            atomic_json(
                paths.srrl_panel_progress,
                {
                    "schema_version": 1,
                    "updated_at": utc_now(),
                    "completed": completed,
                    "failures": failures,
                },
            )
    atomic_json(
        paths.srrl_panel_progress,
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "completed": completed,
            "failures": failures,
        },
    )
    if not completed:
        raise PhysicalBranchError("NO_SRRL_PANEL_ROWS", "no daily SRRL file could be normalized")
    glob = _duckdb_glob(paths.srrl_parts)
    _duckdb_copy(
        f"SELECT * FROM read_parquet('{glob}', union_by_name=true) ORDER BY timestamp_utc",
        paths.srrl_solar_15m,
    )
    con = duckdb.connect()
    stats = con.execute(
        f"""
        SELECT count(*) AS rows, min(timestamp_utc), max(timestamp_utc),
               count(DISTINCT source_day) AS source_days,
               avg(coverage_fraction) AS mean_coverage
        FROM read_parquet('{glob}', union_by_name=true)
        """
    ).fetchone()
    con.close()
    return {
        "row_count": int(stats[0]),
        "min_utc": pd.Timestamp(stats[1]).tz_convert("UTC").isoformat(),
        "max_utc": pd.Timestamp(stats[2]).tz_convert("UTC").isoformat(),
        "source_day_count": int(stats[3]),
        "mean_coverage_fraction": float(stats[4]),
        "failed_processing_days": failures,
        "dataset_path": paths.srrl_solar_15m.as_posix(),
        "daily_part_path": paths.srrl_parts.as_posix(),
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "timezone": "fixed_MST_UTC_minus_07",
    }


def _enrich_overlap_with_nlr295(
    frame: pd.DataFrame,
    nlr295_parts: Path,
    *,
    batch_size: int,
    interval_minutes: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp_utc", kind="stable").drop_duplicates(
        "timestamp_utc", keep="last"
    )
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    bucket_start = timestamps.astype("int64").to_numpy()
    bucket_end = bucket_start + int(pd.Timedelta(minutes=interval_minutes).value)
    count_diff = np.zeros(len(frame) + 1, dtype=np.float64)
    node_diff = np.zeros(len(frame) + 1, dtype=np.float64)
    monitoring_count_diff = np.zeros(len(frame) + 1, dtype=np.float64)
    power_diff = np.zeros(len(frame) + 1, dtype=np.float64)
    dataset = pads.dataset(nlr295_parts, format="parquet")
    columns = [
        "start_time_utc",
        "end_time_utc",
        "nodes_used",
        "duration_hours",
        "node_energy_monitoring_wh",
        "node_monitoring_available",
    ]
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    minimum = bucket_start[0]
    maximum = bucket_end[-1]
    overlapping_jobs = 0
    for batch in scanner.to_batches():
        jobs = batch.to_pandas()
        start = pd.to_datetime(jobs["start_time_utc"], utc=True, errors="coerce")
        end = pd.to_datetime(jobs["end_time_utc"], utc=True, errors="coerce")
        start_ns = start.astype("int64").to_numpy()
        end_ns = end.astype("int64").to_numpy()
        valid = start.notna().to_numpy() & end.notna().to_numpy()
        valid &= start_ns < maximum
        valid &= end_ns > minimum
        if not valid.any():
            continue
        start_ns = start_ns[valid]
        end_ns = end_ns[valid]
        first = np.searchsorted(bucket_end, start_ns, side="right")
        last = np.searchsorted(bucket_start, end_ns, side="left")
        valid_span = first < last
        if not valid_span.any():
            continue
        first = first[valid_span]
        last = last[valid_span]
        overlapping_jobs += len(first)
        nodes = pd.to_numeric(jobs.loc[valid, "nodes_used"], errors="coerce").to_numpy()[
            valid_span
        ]
        nodes = np.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)
        duration = pd.to_numeric(
            jobs.loc[valid, "duration_hours"], errors="coerce"
        ).to_numpy()[valid_span]
        energy = pd.to_numeric(
            jobs.loc[valid, "node_energy_monitoring_wh"], errors="coerce"
        ).to_numpy()[valid_span]
        monitoring = np.isfinite(energy) & np.isfinite(duration) & (duration > 0)
        average_power_kw = np.zeros_like(energy, dtype=float)
        average_power_kw[monitoring] = energy[monitoring] / duration[monitoring] / 1000.0
        np.add.at(count_diff, first, 1.0)
        np.add.at(count_diff, last, -1.0)
        np.add.at(node_diff, first, nodes)
        np.add.at(node_diff, last, -nodes)
        np.add.at(monitoring_count_diff, first, monitoring.astype(float))
        np.add.at(monitoring_count_diff, last, -monitoring.astype(float))
        np.add.at(power_diff, first, average_power_kw)
        np.add.at(power_diff, last, -average_power_kw)
    frame = frame.reset_index(drop=True)
    frame["nlr295_active_jobs"] = np.cumsum(count_diff[:-1]).astype("int64")
    frame["nlr295_active_nodes"] = np.cumsum(node_diff[:-1])
    frame["nlr295_active_monitoring_jobs"] = np.cumsum(monitoring_count_diff[:-1]).astype(
        "int64"
    )
    frame["nlr295_reconstructed_node_power_kw"] = np.cumsum(power_diff[:-1])
    frame["nlr295_job_overlap_found"] = frame["nlr295_active_jobs"].gt(0)
    frame["nlr295_workload_evidence_role"] = "reconstructed"
    frame["nlr295_workload_record_type"] = "uniform_job_energy_reconstruction"
    frame.attrs["overlapping_nlr295_job_rows"] = overlapping_jobs
    return frame


def _build_overlap_panel(
    facility_path: Path,
    srrl_path: Path,
    nlr295_parts: Path,
    destination: Path,
    *,
    cfg: Mapping[str, Any],
    interpretation: str,
    local_timezone: str | tzinfo,
) -> dict[str, Any]:
    facility = facility_path.resolve().as_posix().replace("'", "''")
    srrl = srrl_path.resolve().as_posix().replace("'", "''")
    minimum_coverage = float(cfg.get("srrl_min_interval_coverage", 0.80))
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT f.*,
               s.ghi_wm2 AS srrl_ghi_wm2,
               s.dni_wm2 AS srrl_dni_wm2,
               s.dhi_wm2 AS srrl_dhi_wm2,
               s.air_temp_c AS srrl_air_temp_c,
               s.relative_humidity_pct AS srrl_relative_humidity_pct,
               s.coverage_fraction AS srrl_coverage_fraction,
               s.source_day AS srrl_source_day,
               s.source_file AS srrl_source_file,
               s.source_schema_sha256 AS srrl_source_schema_sha256
        FROM read_parquet('{facility}') f
        INNER JOIN read_parquet('{srrl}') s USING (timestamp_utc)
        WHERE s.coverage_fraction >= {minimum_coverage}
        ORDER BY timestamp_utc
        """
    ).fetchdf()
    con.close()
    if frame.empty:
        raise PhysicalBranchError(
            "NO_VERIFIED_NLR300_SRRL_OVERLAP",
            f"{interpretation} produced no exact 15-minute UTC joins",
        )
    # DuckDB TIMESTAMPTZ values are rendered in the connection/session timezone
    # by fetchdf().  Never retain that display timezone as evidence.  UTC is the
    # join/model key and every local clock is deterministically derived from it.
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["timestamp_local"] = frame["timestamp_utc"].dt.tz_convert(local_timezone)
    frame = _enrich_overlap_with_nlr295(
        frame,
        nlr295_parts,
        batch_size=int(cfg["batch_size"]),
        interval_minutes=int(cfg["interval_minutes"]),
    )
    frame["clock_interpretation"] = interpretation
    frame["source_relationship"] = SOURCE_RELATIONSHIP
    frame["measurement_relationship"] = MEASUREMENT_RELATIONSHIP
    frame["synthetic"] = False
    _atomic_parquet(frame, destination)
    return {
        "path": destination.as_posix(),
        "sha256": sha256_file(destination),
        "row_count": len(frame),
        "min_utc": frame["timestamp_utc"].min().isoformat(),
        "max_utc": frame["timestamp_utc"].max().isoformat(),
        "source_day_count": int(frame["srrl_source_day"].nunique()),
        "intervals_with_active_nlr295_jobs": int(frame["nlr295_job_overlap_found"].sum()),
        "intervals_with_monitoring_energy_reconstruction": int(
            frame["nlr295_active_monitoring_jobs"].gt(0).sum()
        ),
        "overlapping_nlr295_job_rows": int(frame.attrs.get("overlapping_nlr295_job_rows", 0)),
        "clock_interpretation": interpretation,
        "timestamp_utc_role": "authoritative_join_and_model_key",
        "timestamp_local_derivation": "timestamp_utc_tz_convert",
        "timestamp_local_timezone": str(frame["timestamp_local"].dt.tz),
    }


def _prepare_overlaps(
    root: Path,
    paths: BranchPaths,
    nlr295: Mapping[str, Any],
    nlr300: Mapping[str, Any],
    srrl_acquisition: Mapping[str, Any],
    srrl_panel: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    primary = _build_overlap_panel(
        paths.nlr300_primary,
        paths.srrl_solar_15m,
        paths.nlr295_deduplicated,
        paths.overlap_primary,
        cfg=cfg,
        interpretation="raw_ts_assumed_UTC_canonical_UTC_Denver_local",
        local_timezone="America/Denver",
    )
    fixed = _build_overlap_panel(
        paths.nlr300_fixed_mst,
        paths.srrl_solar_15m,
        paths.nlr295_deduplicated,
        paths.overlap_fixed_mst,
        cfg=cfg,
        interpretation="raw_ts_assumed_fixed_MST_UTC_minus_07_sensitivity",
        local_timezone=FIXED_MST,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "synthetic": False,
        "source_coverage": {
            "nlr295": {
                "min_submit_utc": nlr295["min_submit_utc"],
                "max_submit_utc": nlr295["max_submit_utc"],
                "min_execution_start_utc": nlr295["min_start_utc"],
                "max_execution_end_utc": nlr295["max_end_utc"],
            },
            "nlr300": {"min_utc": nlr300["min_utc"], "max_utc": nlr300["max_utc"]},
            "srrl": {"min_utc": srrl_panel["min_utc"], "max_utc": srrl_panel["max_utc"]},
        },
        "governing_common_support_daily_scope": {
            "start": srrl_acquisition["expected_start"],
            "end": srrl_acquisition["expected_end"],
            "expected_days": srrl_acquisition["expected_day_count"],
            "available_days": srrl_acquisition["available_day_count"],
            "failed_or_missing_days": srrl_acquisition["failed_or_missing_day_count"],
            "full_support_available": srrl_acquisition["full_common_support_available"],
            "full_scope_requested": bool(
                not srrl_acquisition["scope_truncated"]
                and srrl_acquisition["requested_day_count"]
                == srrl_acquisition["expected_day_count"]
            ),
            "usable_day_fraction": srrl_panel["source_day_count"]
            / max(int(srrl_acquisition["expected_day_count"]), 1),
            "minimum_usable_day_fraction_gate": float(
                cfg["minimum_srrl_usable_day_fraction"]
            ),
            "gate_policy": (
                "request_every_day_in_common_support_then_require_minimum_usable_fraction_"
                "and_nonempty_train_validation_locked_test"
            ),
            "per_day_failures": srrl_acquisition["failures"],
        },
        "primary_assumed_utc": primary,
        "fixed_mst_sensitivity": fixed,
        "clock_sensitivity_overlap_row_delta": fixed["row_count"] - primary["row_count"],
        "clock_sensitivity_active_job_interval_delta": (
            fixed["intervals_with_active_nlr295_jobs"]
            - primary["intervals_with_active_nlr295_jobs"]
        ),
    }
    atomic_json(paths.overlap_manifest, manifest)
    manifest["manifest_path"] = _relative(root, paths.overlap_manifest)
    return manifest


JOB_FEATURES = (
    "duration_hours",
    "nodes_used",
    "processors_used",
    "gpus_requested",
    "node_hours",
    "processor_hours",
    "gpu_hours",
)


def _iter_job_model_batches(
    path: Path,
    *,
    target: str,
    split: str,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    parquet = pq.ParquetFile(path)
    columns = [*JOB_FEATURES, target, "split"]
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas()
        frame = frame.loc[frame["split"].eq(split)]
        if frame.empty:
            continue
        raw_features = frame[list(JOB_FEATURES)].apply(pd.to_numeric, errors="coerce")
        raw_target = pd.to_numeric(frame[target], errors="coerce")
        valid = raw_target.gt(0) & np.isfinite(raw_target)
        valid &= raw_features.notna().all(axis=1)
        valid &= raw_features.ge(0).all(axis=1)
        if not valid.any():
            continue
        raw = raw_features.loc[valid].to_numpy(dtype=np.float64)
        y = raw_target.loc[valid].to_numpy(dtype=np.float64)
        yield np.log1p(raw), np.log1p(y), raw


def _job_target_counts(path: Path, target: str) -> dict[str, int]:
    escaped = path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT split, count(*)
        FROM read_parquet('{escaped}')
        WHERE {target} IS NOT NULL AND {target} > 0
        GROUP BY split
        """
    ).fetchall()
    con.close()
    return {str(split): int(count) for split, count in rows}


def _job_baseline_rate(path: Path, target: str, exposure: str) -> float:
    escaped = path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    value = con.execute(
        f"""
        SELECT median({target} / nullif({exposure}, 0))
        FROM read_parquet('{escaped}')
        WHERE split='train' AND {target} > 0 AND {exposure} > 0
        """
    ).fetchone()[0]
    con.close()
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise PhysicalBranchError(
            "NLR295_INVALID_ENGINEERING_RATE",
            f"cannot estimate train-only {target}/{exposure}",
        )
    return float(value)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "mean_actual": float(np.mean(actual)),
        "mean_predicted": float(np.mean(predicted)),
    }


def _evaluate_job_model(
    path: Path,
    *,
    target: str,
    exposure_index: int,
    split: str,
    batch_size: int,
    baseline_rate: float,
    scaler: StandardScaler,
    model: SGDRegressor,
) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = {
        "train_specific_energy_baseline": {
            "n": 0.0,
            "abs": 0.0,
            "sq": 0.0,
            "bias": 0.0,
            "actual": 0.0,
            "pred": 0.0,
        },
        "calibrated_streaming_sgd": {
            "n": 0.0,
            "abs": 0.0,
            "sq": 0.0,
            "bias": 0.0,
            "actual": 0.0,
            "pred": 0.0,
        },
    }
    for features, target_log, raw in _iter_job_model_batches(
        path, target=target, split=split, batch_size=batch_size
    ):
        actual = np.expm1(target_log)
        predictions = {
            "train_specific_energy_baseline": baseline_rate * raw[:, exposure_index],
            "calibrated_streaming_sgd": np.maximum(
                0.0, np.expm1(model.predict(scaler.transform(features)))
            ),
        }
        for name, prediction in predictions.items():
            error = prediction - actual
            total = totals[name]
            total["n"] += len(actual)
            total["abs"] += float(np.abs(error).sum())
            total["sq"] += float(np.square(error).sum())
            total["bias"] += float(error.sum())
            total["actual"] += float(actual.sum())
            total["pred"] += float(prediction.sum())
    rows: list[dict[str, Any]] = []
    for model_name, total in totals.items():
        count = int(total["n"])
        if not count:
            continue
        rows.append(
            {
                "target": target,
                "split": split,
                "model": model_name,
                "row_count": count,
                "mae_wh": total["abs"] / count,
                "rmse_wh": math.sqrt(total["sq"] / count),
                "bias_wh": total["bias"] / count,
                "energy_bias_fraction": (
                    total["pred"] - total["actual"]
                )
                / max(total["actual"], 1e-12),
                "source_relationship": SOURCE_RELATIONSHIP,
                "measurement_relationship": MEASUREMENT_RELATIONSHIP,
                "evidence_role": "derived",
            }
        )
    return rows


def _fit_streaming_job_component(
    path: Path,
    *,
    target: str,
    exposure: str,
    minimum_train_rows: int,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    counts = _job_target_counts(path, target)
    if counts.get("train", 0) < minimum_train_rows:
        return {
            "target": target,
            "status": "not_fit_insufficient_monitoring_rows",
            "required_train_rows": minimum_train_rows,
            "split_counts": counts,
        }
    if counts.get("validation", 0) == 0 or counts.get("locked_test", 0) == 0:
        return {
            "target": target,
            "status": "not_fit_empty_validation_or_locked_test",
            "required_train_rows": minimum_train_rows,
            "split_counts": counts,
        }
    batch_size = int(cfg["batch_size"])
    scaler = StandardScaler()
    scaler_rows = 0
    for features, _, _ in _iter_job_model_batches(
        path, target=target, split="train", batch_size=batch_size
    ):
        scaler.partial_fit(features)
        scaler_rows += len(features)
    if scaler_rows < minimum_train_rows:
        raise PhysicalBranchError(
            "NLR295_JOB_FEATURE_ROWS_INSUFFICIENT",
            f"{target}: {scaler_rows} usable train rows",
        )
    alpha_values = [float(value) for value in cfg.get("job_model_alpha", [1e-5, 1e-4, 1e-3])]
    candidates: list[tuple[float, SGDRegressor, float]] = []
    for alpha in alpha_values:
        model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=alpha,
            max_iter=1,
            tol=None,
            shuffle=False,
            random_state=int(cfg["seed"]),
            learning_rate="invscaling",
            eta0=float(cfg.get("job_model_eta0", 0.01)),
            power_t=0.25,
        )
        fitted = False
        for _ in range(max(1, int(cfg["job_model_epochs"]))):
            for features, target_log, _ in _iter_job_model_batches(
                path, target=target, split="train", batch_size=batch_size
            ):
                model.partial_fit(scaler.transform(features), target_log)
                fitted = True
        if not fitted:
            raise PhysicalBranchError("NLR295_JOB_MODEL_NOT_FIT", target)
        squared_error = 0.0
        validation_rows = 0
        for features, target_log, _ in _iter_job_model_batches(
            path, target=target, split="validation", batch_size=batch_size
        ):
            prediction = np.maximum(0.0, np.expm1(model.predict(scaler.transform(features))))
            actual = np.expm1(target_log)
            squared_error += float(np.square(prediction - actual).sum())
            validation_rows += len(actual)
        candidates.append((math.sqrt(squared_error / validation_rows), model, alpha))
    _, selected_model, selected_alpha = min(candidates, key=lambda item: item[0])
    baseline_rate = _job_baseline_rate(path, target, exposure)
    exposure_index = JOB_FEATURES.index(exposure)
    metrics: list[dict[str, Any]] = []
    for split in ("train", "validation", "locked_test"):
        metrics.extend(
            _evaluate_job_model(
                path,
                target=target,
                exposure_index=exposure_index,
                split=split,
                batch_size=batch_size,
                baseline_rate=baseline_rate,
                scaler=scaler,
                model=selected_model,
            )
        )
    return {
        "target": target,
        "status": "fitted",
        "split_counts": counts,
        "selected_alpha_on_validation": selected_alpha,
        "baseline_exposure": exposure,
        "baseline_wh_per_exposure_hour": baseline_rate,
        "feature_names": list(JOB_FEATURES),
        "feature_transform": "log1p_then_train_only_standard_scaler",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": selected_model.coef_.tolist(),
        "intercept": selected_model.intercept_.tolist(),
        "target_transform": "log1p",
        "metrics": metrics,
    }


def _calibrate_job_energy(paths: BranchPaths, cfg: Mapping[str, Any]) -> dict[str, Any]:
    components = [
        _fit_streaming_job_component(
            paths.nlr295_job_energy,
            target="node_energy_monitoring_wh",
            exposure="node_hours",
            minimum_train_rows=int(cfg["job_min_train_rows"]),
            cfg=cfg,
        ),
        _fit_streaming_job_component(
            paths.nlr295_job_energy,
            target="gpu_energy_monitoring_wh",
            exposure="gpu_hours",
            minimum_train_rows=int(cfg["job_gpu_min_train_rows"]),
            cfg=cfg,
        ),
    ]
    node = components[0]
    if node["status"] != "fitted":
        raise PhysicalBranchError(
            "NLR295_NODE_ENERGY_CALIBRATION_INSUFFICIENT",
            json.dumps(node, sort_keys=True),
        )
    metric_rows = [row for component in components for row in component.get("metrics", [])]
    metrics = pd.DataFrame(metric_rows)
    _atomic_parquet(metrics, paths.job_metrics)
    return {
        "components": components,
        "metric_row_count": len(metrics),
        "metrics_path": paths.job_metrics.as_posix(),
        "monitoring_semantics": "publisher_reported_monitoring_derived_not_investigator_measured",
        "cpu_tdp_semantics": "publisher_reported_estimate_kept_separate_not_a_monitoring_target",
    }


FACILITY_FEATURES = (
    "it_power_kw",
    "hot_degree_c",
    "cold_degree_c",
    "relative_humidity_pct",
    "ghi_kwm2",
    "it_hot_interaction",
)


def _facility_design(frame: pd.DataFrame) -> np.ndarray:
    it_power = pd.to_numeric(frame["it_power_kw"], errors="coerce").to_numpy(dtype=float)
    temperature = pd.to_numeric(frame["srrl_air_temp_c"], errors="coerce").to_numpy(
        dtype=float
    )
    humidity = pd.to_numeric(
        frame["srrl_relative_humidity_pct"], errors="coerce"
    ).to_numpy(dtype=float)
    irradiance = pd.to_numeric(frame["srrl_ghi_wm2"], errors="coerce").to_numpy(
        dtype=float
    )
    hot = np.maximum(temperature - 18.0, 0.0)
    cold = np.maximum(10.0 - temperature, 0.0)
    return np.column_stack(
        [
            it_power,
            hot,
            cold,
            humidity,
            np.maximum(irradiance, 0.0) / 1000.0,
            it_power * hot / 1000.0,
        ]
    )


def _clean_facility_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    required = [
        "timestamp_utc",
        "it_power_kw",
        "facility_power_kw",
        "pue",
        "srrl_air_temp_c",
        "srrl_relative_humidity_pct",
        "srrl_ghi_wm2",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise PhysicalBranchError("PHYSICAL_OVERLAP_SCHEMA_MISSING", str(missing))
    numeric = [column for column in required if column != "timestamp_utc"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = frame["timestamp_utc"].notna()
    valid &= np.isfinite(frame[numeric]).all(axis=1)
    valid &= frame["it_power_kw"].gt(0)
    valid &= frame["facility_power_kw"].gt(0)
    valid &= frame["pue"].between(1.0, 3.0)
    frame = frame.loc[valid].sort_values("timestamp_utc", kind="stable")
    return frame.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _assign_physical_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        raise PhysicalBranchError("NO_VALID_PHYSICAL_OVERLAP_ROWS", "empty after quality filters")
    start = frame["timestamp_utc"].iloc[0]
    end = frame["timestamp_utc"].iloc[-1]
    span = end - start
    if span <= pd.Timedelta(0):
        raise PhysicalBranchError("PHYSICAL_SUPPORT_HAS_NO_DURATION", str(start))
    boundary_60 = start + 0.60 * span
    boundary_80 = start + 0.80 * span
    split = np.where(
        frame["timestamp_utc"].lt(boundary_60),
        "train",
        np.where(frame["timestamp_utc"].lt(boundary_80), "validation", "locked_test"),
    )
    frame = frame.copy()
    frame["split"] = split
    counts = frame["split"].value_counts().to_dict()
    if any(int(counts.get(name, 0)) == 0 for name in ("train", "validation", "locked_test")):
        raise PhysicalBranchError(
            "PHYSICAL_CHRONOLOGICAL_SPLIT_EMPTY",
            json.dumps(counts, sort_keys=True),
        )
    timestamp_hash = sha256_json(
        [timestamp.isoformat() for timestamp in frame["timestamp_utc"].tolist()]
    )
    return frame, {
        "method": "first_60_next_20_final_20_of_verified_time_support",
        "sort_key": "timestamp_utc",
        "support_start_utc": start.isoformat(),
        "support_end_utc": end.isoformat(),
        "train_end_exclusive": boundary_60.isoformat(),
        "validation_end_exclusive": boundary_80.isoformat(),
        "locked_test": True,
        "row_counts": {name: int(counts.get(name, 0)) for name in ("train", "validation", "locked_test")},
        "timestamp_sequence_sha256": timestamp_hash,
    }


def _fit_selected_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    alpha_values: Sequence[float],
) -> tuple[StandardScaler, Ridge, dict[str, Any]]:
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    candidates: list[dict[str, Any]] = []
    fitted: list[Ridge] = []
    for alpha in alpha_values:
        model = Ridge(alpha=float(alpha), fit_intercept=True).fit(train_scaled, train_y)
        prediction = model.predict(validation_scaled)
        score = _regression_metrics(validation_y, prediction)["rmse"]
        fitted.append(model)
        candidates.append({"alpha": float(alpha), "validation_rmse": score})
    selected_index = min(range(len(candidates)), key=lambda index: candidates[index]["validation_rmse"])
    return scaler, fitted[selected_index], {
        "selection_split": "validation",
        "selection_metric": "rmse",
        "candidates": candidates,
        "selected_alpha": candidates[selected_index]["alpha"],
    }


def _ridge_predict(
    scaler: StandardScaler,
    model: Ridge,
    design: np.ndarray,
) -> np.ndarray:
    return np.asarray(model.predict(scaler.transform(design)), dtype=float)


def _ridge_parameters(
    name: str,
    scaler: StandardScaler,
    model: Ridge,
    selection: Mapping[str, Any],
    *,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "clock_interpretation": interpretation,
        "model": name,
        "feature_names": list(FACILITY_FEATURES),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": float(model.intercept_),
        "selection": dict(selection),
        "fit_split": "train",
        "selection_split": "validation",
        "evaluation_split": "locked_test",
    }


def _distill_piecewise_surrogate(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    model: Ridge,
    *,
    knots: int,
    interpretation: str,
) -> dict[str, Any]:
    train = frame.loc[frame["split"].eq("train")]
    it_min = float(train["it_power_kw"].quantile(0.01))
    it_peak = float(train["it_power_kw"].quantile(0.99))
    if not math.isfinite(it_min) or not math.isfinite(it_peak) or it_peak <= it_min:
        raise PhysicalBranchError("PHYSICAL_IT_SUPPORT_DEGENERATE", f"{it_min}, {it_peak}")
    knot_count = max(2, int(knots))
    it_knots = np.linspace(it_min, it_peak, knot_count)
    reference = pd.DataFrame(
        {
            "it_power_kw": it_knots,
            "srrl_air_temp_c": float(train["srrl_air_temp_c"].median()),
            "srrl_relative_humidity_pct": float(
                train["srrl_relative_humidity_pct"].median()
            ),
            "srrl_ghi_wm2": float(train["srrl_ghi_wm2"].median()),
        }
    )
    facility_knots = _ridge_predict(scaler, model, _facility_design(reference))
    facility_knots = np.maximum(facility_knots, it_knots)
    facility_knots = np.maximum.accumulate(facility_knots)
    utilization = (it_knots - it_min) / (it_peak - it_min)
    slopes = np.diff(facility_knots) / np.diff(utilization)
    if (slopes < -1e-9).any() or not np.isfinite(slopes).all():
        raise PhysicalBranchError("NONMONOTONE_FROZEN_SURROGATE", interpretation)
    return {
        "schema_version": 1,
        "clock_interpretation": interpretation,
        "calibration_model": "facility_ridge_selected_on_validation",
        "distillation_split": "train",
        "selection_split": "validation",
        "x_axis": "normalized_it_power_between_train_q01_and_q99",
        "utilization_knots": utilization.tolist(),
        "it_power_knots_kw": it_knots.tolist(),
        "facility_power_knots_kw": facility_knots.tolist(),
        "segment_slopes_kw_per_utilization": slopes.tolist(),
        "domain_behavior": {
            "within_knots": "linear_interpolation",
            "below_first_knot": "linear_endpoint_extrapolation_using_first_segment",
            "above_last_knot": "linear_endpoint_extrapolation_using_last_segment",
            "facility_power_floor": "it_power_kw",
            "locked_test_used_to_fit_or_select": False,
        },
        "milp_representation": (
            "SOS2_piecewise_linear_segments_with_first_and_last_segment_rays"
        ),
        "idle_facility_power_kw": float(facility_knots[0]),
        "peak_facility_power_kw": float(facility_knots[-1]),
        "reference_weather": {
            "air_temp_c": float(reference["srrl_air_temp_c"].iloc[0]),
            "relative_humidity_pct": float(
                reference["srrl_relative_humidity_pct"].iloc[0]
            ),
            "ghi_wm2": float(reference["srrl_ghi_wm2"].iloc[0]),
        },
        "monotone": True,
        "frozen": True,
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "evidence_role": "derived",
        "synthetic": False,
    }


def _surrogate_predict(frame: pd.DataFrame, surrogate: Mapping[str, Any]) -> np.ndarray:
    values = frame["it_power_kw"].to_numpy(dtype=float)
    it_knots = np.asarray(surrogate["it_power_knots_kw"], dtype=float)
    facility_knots = np.asarray(surrogate["facility_power_knots_kw"], dtype=float)
    prediction = np.interp(values, it_knots, facility_knots)
    below = values < it_knots[0]
    above = values > it_knots[-1]
    first_slope = (facility_knots[1] - facility_knots[0]) / (
        it_knots[1] - it_knots[0]
    )
    last_slope = (facility_knots[-1] - facility_knots[-2]) / (
        it_knots[-1] - it_knots[-2]
    )
    prediction[below] = facility_knots[0] + first_slope * (
        values[below] - it_knots[0]
    )
    prediction[above] = facility_knots[-1] + last_slope * (
        values[above] - it_knots[-1]
    )
    return np.maximum(prediction, values)


def _surrogate_acceptance_report(
    metric_rows: Sequence[Mapping[str, Any]],
    split_diagnostics: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    max_rmse_ratio = float(
        cfg.get("surrogate_acceptance_max_observed_rmse_ratio", 1.25)
    )
    max_approximation_fraction = float(
        cfg.get("surrogate_acceptance_max_calibrated_mae_fraction", 0.01)
    )
    minimum_approximation_kw = float(
        cfg.get("surrogate_acceptance_min_calibrated_mae_kw", 5.0)
    )
    if max_rmse_ratio < 1.0 or not 0 < max_approximation_fraction < 1.0:
        raise ValueError("invalid MILP surrogate acceptance thresholds")
    by_key = {
        (str(row["split"]), str(row["model"])): row for row in metric_rows
    }
    gates: dict[str, Any] = {}
    for split in ("validation", "locked_test"):
        calibrated = by_key[(split, "calibrated_statistical_twin")]
        surrogate = by_key[(split, "frozen_piecewise_linear_surrogate")]
        calibrated_rmse = float(calibrated["facility_rmse_kw"])
        surrogate_rmse = float(surrogate["facility_rmse_kw"])
        rmse_ratio = surrogate_rmse / max(calibrated_rmse, 1e-9)
        calibrated_mean = float(
            split_diagnostics[split]["mean_calibrated_facility_kw"]
        )
        approximation_limit = max(
            minimum_approximation_kw,
            max_approximation_fraction * calibrated_mean,
        )
        approximation_mae = float(surrogate["surrogate_vs_calibrated_mae_kw"])
        observed_rmse_excess = surrogate_rmse - calibrated_rmse
        observed_gate_passed = bool(
            rmse_ratio <= max_rmse_ratio
            or observed_rmse_excess <= approximation_limit
        )
        passed = bool(
            observed_gate_passed
            and approximation_mae <= approximation_limit
        )
        gates[split] = {
            **dict(split_diagnostics[split]),
            "calibrated_observed_rmse_kw": calibrated_rmse,
            "surrogate_observed_rmse_kw": surrogate_rmse,
            "observed_rmse_ratio_to_calibrated": rmse_ratio,
            "maximum_observed_rmse_ratio": max_rmse_ratio,
            "observed_rmse_excess_kw": observed_rmse_excess,
            "maximum_observed_rmse_excess_kw": approximation_limit,
            "observed_error_gate": (
                "rmse_ratio_or_absolute_excess_within_calibrated_approximation_allowance"
            ),
            "observed_error_gate_passed": observed_gate_passed,
            "surrogate_vs_calibrated_mae_kw": approximation_mae,
            "maximum_surrogate_vs_calibrated_mae_kw": approximation_limit,
            "passed": passed,
        }
    return {
        "schema_version": 1,
        "status": "accepted" if all(item["passed"] for item in gates.values()) else "rejected",
        "accepted_for_optimization": all(item["passed"] for item in gates.values()),
        "selection_policy": "thresholds_fixed_in_config_and_checked_on_validation_before_locked_test",
        "locked_test_used_to_fit_or_select": False,
        "extrapolation_policy": "linear_endpoint_extrapolation",
        "thresholds": {
            "maximum_observed_rmse_ratio_to_calibrated": max_rmse_ratio,
            "maximum_calibrated_approximation_mae_fraction": max_approximation_fraction,
            "minimum_absolute_approximation_allowance_kw": minimum_approximation_kw,
        },
        "gates": gates,
    }


def _physical_metric_row(
    frame: pd.DataFrame,
    predicted_facility: np.ndarray,
    predicted_pue: np.ndarray,
    *,
    model_name: str,
    interpretation: str,
    interval_hours: float,
    calibrated_prediction: np.ndarray | None = None,
) -> dict[str, Any]:
    actual = frame["facility_power_kw"].to_numpy(dtype=float)
    actual_pue = frame["pue"].to_numpy(dtype=float)
    prediction = np.asarray(predicted_facility, dtype=float)
    pue_prediction = np.asarray(predicted_pue, dtype=float)
    error = prediction - actual
    dates = frame["timestamp_utc"].dt.floor("D")
    daily = pd.DataFrame(
        {"date": dates, "actual": actual * interval_hours, "prediction": prediction * interval_hours}
    ).groupby("date", sort=True).sum()
    daily_error = daily["prediction"] - daily["actual"]
    contiguous = (
        frame["timestamp_utc"].diff().iloc[1:]
        == pd.Timedelta(hours=interval_hours)
    ).to_numpy()
    actual_ramp = np.diff(actual)[contiguous]
    predicted_ramp = np.diff(prediction)[contiguous]
    row = {
        "clock_interpretation": interpretation,
        "split": str(frame["split"].iloc[0]),
        "model": model_name,
        "row_count": len(frame),
        "facility_mae_kw": float(np.mean(np.abs(error))),
        "facility_rmse_kw": float(np.sqrt(np.mean(np.square(error)))),
        "facility_bias_kw": float(np.mean(error)),
        "daily_energy_bias_kwh": float(daily_error.mean()),
        "daily_energy_mae_kwh": float(daily_error.abs().mean()),
        "peak_error_kw": float(prediction.max() - actual.max()),
        "ramp_mae_kw_per_interval": (
            float(np.mean(np.abs(predicted_ramp - actual_ramp))) if len(actual_ramp) else 0.0
        ),
        "peak_ramp_error_kw_per_interval": (
            float(np.max(np.abs(predicted_ramp)) - np.max(np.abs(actual_ramp)))
            if len(actual_ramp)
            else 0.0
        ),
        "pue_mae": float(np.mean(np.abs(pue_prediction - actual_pue))),
        "interval_coverage": 1.0,
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "evidence_role": "derived",
    }
    if calibrated_prediction is not None:
        row["surrogate_vs_calibrated_mae_kw"] = float(
            np.mean(np.abs(prediction - np.asarray(calibrated_prediction, dtype=float)))
        )
        row["mean_calibrated_facility_kw"] = float(
            np.mean(np.asarray(calibrated_prediction, dtype=float))
        )
    else:
        row["surrogate_vs_calibrated_mae_kw"] = np.nan
    return row


def _calibrate_facility_panel(
    path: Path,
    *,
    interpretation: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    frame = _clean_facility_panel(path)
    source_days = int(frame.get("srrl_source_day", pd.Series(dtype=str)).nunique())
    if len(frame) < int(cfg["facility_min_rows"]):
        raise PhysicalBranchError(
            "INSUFFICIENT_PHYSICAL_CALIBRATION_ROWS",
            f"{interpretation}: {len(frame)} < {cfg['facility_min_rows']}",
        )
    if source_days < int(cfg["facility_min_source_days"]):
        raise PhysicalBranchError(
            "INSUFFICIENT_PHYSICAL_CALIBRATION_DAYS",
            f"{interpretation}: {source_days} < {cfg['facility_min_source_days']}",
        )
    frame, split_manifest = _assign_physical_split(frame)
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    design_train = _facility_design(train)
    design_validation = _facility_design(validation)
    alphas = [float(value) for value in cfg.get("facility_ridge_alpha", [0.01, 0.1, 1.0, 10.0])]
    facility_scaler, facility_model, facility_selection = _fit_selected_ridge(
        design_train,
        train["facility_power_kw"].to_numpy(dtype=float),
        design_validation,
        validation["facility_power_kw"].to_numpy(dtype=float),
        alpha_values=alphas,
    )
    pue_scaler, pue_model, pue_selection = _fit_selected_ridge(
        design_train,
        train["pue"].to_numpy(dtype=float),
        design_validation,
        validation["pue"].to_numpy(dtype=float),
        alpha_values=alphas,
    )
    weather_free_scaler = StandardScaler().fit(design_train[:, :1])
    weather_free_model = Ridge(alpha=1.0).fit(
        weather_free_scaler.transform(design_train[:, :1]),
        train["facility_power_kw"].to_numpy(dtype=float),
    )
    pue_bounds = (
        max(1.0, float(train["pue"].quantile(0.005))),
        max(1.01, float(train["pue"].quantile(0.995))),
    )
    constant_pue = float(train["pue"].median())
    engineering = dict(cfg.get("engineering_pue", {}))
    engineering.setdefault("base", 1.15)
    engineering.setdefault("reference_temp_c", 20.0)
    engineering.setdefault("temp_slope_per_c", 0.01)
    engineering.setdefault("minimum", 1.0)
    engineering.setdefault("maximum", 1.60)
    surrogate = _distill_piecewise_surrogate(
        frame,
        facility_scaler,
        facility_model,
        knots=int(cfg["surrogate_knots"]),
        interpretation=interpretation,
    )
    metric_rows: list[dict[str, Any]] = []
    surrogate_split_diagnostics: dict[str, dict[str, Any]] = {}
    residual_multipliers: list[np.ndarray] = []
    calibration_residuals: list[np.ndarray] = []
    for split in ("train", "validation", "locked_test"):
        subset = frame.loc[frame["split"].eq(split)].copy()
        design = _facility_design(subset)
        it_power = subset["it_power_kw"].to_numpy(dtype=float)
        temperature = subset["srrl_air_temp_c"].to_numpy(dtype=float)
        engineering_pue = np.clip(
            float(engineering["base"])
            + float(engineering["temp_slope_per_c"])
            * np.maximum(temperature - float(engineering["reference_temp_c"]), 0.0),
            float(engineering["minimum"]),
            float(engineering["maximum"]),
        )
        calibrated_facility = np.maximum(
            it_power, _ridge_predict(facility_scaler, facility_model, design)
        )
        calibrated_pue = np.clip(
            _ridge_predict(pue_scaler, pue_model, design), pue_bounds[0], pue_bounds[1]
        )
        weather_free = np.maximum(
            it_power,
            weather_free_model.predict(weather_free_scaler.transform(design[:, :1])),
        )
        surrogate_prediction = _surrogate_predict(subset, surrogate)
        predictions = {
            "uncalibrated_engineering_curve": (engineering_pue * it_power, engineering_pue),
            "constant_train_pue": (
                constant_pue * it_power,
                np.full(len(subset), constant_pue),
            ),
            "weather_free_static_linear": (weather_free, weather_free / it_power),
            "calibrated_statistical_twin": (calibrated_facility, calibrated_pue),
            "frozen_piecewise_linear_surrogate": (
                surrogate_prediction,
                surrogate_prediction / it_power,
            ),
        }
        for model_name, (facility_prediction, pue_prediction) in predictions.items():
            metric_row = _physical_metric_row(
                subset,
                facility_prediction,
                pue_prediction,
                model_name=model_name,
                interpretation=interpretation,
                interval_hours=float(cfg["interval_minutes"]) / 60.0,
                calibrated_prediction=(
                    calibrated_facility
                    if model_name == "frozen_piecewise_linear_surrogate"
                    else None
                ),
            )
            if model_name == "frozen_piecewise_linear_surrogate":
                below = it_power < float(surrogate["it_power_knots_kw"][0])
                above = it_power > float(surrogate["it_power_knots_kw"][-1])
                metric_row["below_training_knot_fraction"] = float(np.mean(below))
                metric_row["above_training_knot_fraction"] = float(np.mean(above))
                metric_row["outside_training_knot_fraction"] = float(
                    np.mean(below | above)
                )
                metric_row["endpoint_extrapolation"] = "linear"
                surrogate_split_diagnostics[split] = {
                    "row_count": len(subset),
                    "it_power_minimum_kw": float(it_power.min()),
                    "it_power_maximum_kw": float(it_power.max()),
                    "below_training_knot_fraction": float(np.mean(below)),
                    "above_training_knot_fraction": float(np.mean(above)),
                    "outside_training_knot_fraction": float(np.mean(below | above)),
                    "mean_calibrated_facility_kw": float(calibrated_facility.mean()),
                }
            metric_rows.append(metric_row)
        actual = subset["facility_power_kw"].to_numpy(dtype=float)
        if split in {"train", "validation"}:
            residual_multipliers.append(
                np.clip(actual / np.maximum(calibrated_facility, 1e-9), 0.25, 4.0)
            )
            calibration_residuals.append(actual - calibrated_facility)
    residual_pool = np.concatenate(residual_multipliers)
    residual_kw = np.concatenate(calibration_residuals)
    q_low, q_high = np.quantile(residual_kw, [0.05, 0.95])
    test = frame.loc[frame["split"].eq("locked_test")]
    test_prediction = np.maximum(
        test["it_power_kw"].to_numpy(dtype=float),
        _ridge_predict(facility_scaler, facility_model, _facility_design(test)),
    )
    test_actual = test["facility_power_kw"].to_numpy(dtype=float)
    interval_coverage = float(
        np.mean((test_actual >= test_prediction + q_low) & (test_actual <= test_prediction + q_high))
    )
    surrogate_acceptance = _surrogate_acceptance_report(
        metric_rows, surrogate_split_diagnostics, cfg
    )
    surrogate["acceptance"] = surrogate_acceptance
    if not surrogate_acceptance["accepted_for_optimization"]:
        raise PhysicalBranchError(
            "MILP_SURROGATE_ACCEPTANCE_FAILED",
            json.dumps(surrogate_acceptance, sort_keys=True),
        )
    return {
        "clock_interpretation": interpretation,
        "row_count": len(frame),
        "source_day_count": source_days,
        "split_manifest": split_manifest,
        "facility_model": _ridge_parameters(
            "calibrated_facility_ridge",
            facility_scaler,
            facility_model,
            facility_selection,
            interpretation=interpretation,
        ),
        "pue_model": _ridge_parameters(
            "calibrated_pue_ridge",
            pue_scaler,
            pue_model,
            pue_selection,
            interpretation=interpretation,
        ),
        "weather_free_model": {
            "scaler_mean": weather_free_scaler.mean_.tolist(),
            "scaler_scale": weather_free_scaler.scale_.tolist(),
            "coefficient": weather_free_model.coef_.tolist(),
            "intercept": float(weather_free_model.intercept_),
        },
        "engineering_baseline": engineering,
        "constant_train_pue": constant_pue,
        "pue_bounds": list(pue_bounds),
        "surrogate": surrogate,
        "surrogate_acceptance": surrogate_acceptance,
        "metrics": metric_rows,
        "uncertainty": {
            "method": "train_validation_residual_empirical_interval",
            "lower_residual_kw_q05": float(q_low),
            "upper_residual_kw_q95": float(q_high),
            "locked_test_interval_coverage": interval_coverage,
            "nominal_coverage": 0.90,
            "residual_row_count": len(residual_kw),
        },
        "_residual_multiplier_pool": residual_pool,
    }


def _calibrate_physical(
    root: Path,
    paths: BranchPaths,
    nlr295: Mapping[str, Any],
    overlap: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    job = _calibrate_job_energy(paths, cfg)
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
    uncertainty = pd.DataFrame(
        [
            {"clock_interpretation": item["clock_interpretation"], **item["uncertainty"]}
            for item in (primary, fixed)
        ]
    )
    uncertainty["source_relationship"] = SOURCE_RELATIONSHIP
    uncertainty["measurement_relationship"] = MEASUREMENT_RELATIONSHIP
    uncertainty["evidence_role"] = "derived"
    _atomic_parquet(uncertainty, paths.uncertainty_metrics)
    parameter_rows: list[dict[str, Any]] = []
    for item in (primary, fixed):
        for model_key in ("facility_model", "pue_model"):
            model = item[model_key]
            parameter_rows.append(
                {
                    "clock_interpretation": item["clock_interpretation"],
                    "model": model["model"],
                    "feature_names_json": json.dumps(model["feature_names"]),
                    "scaler_mean_json": json.dumps(model["scaler_mean"]),
                    "scaler_scale_json": json.dumps(model["scaler_scale"]),
                    "coefficient_json": json.dumps(model["coefficient"]),
                    "intercept": model["intercept"],
                    "selected_alpha": model["selection"]["selected_alpha"],
                    "fit_split": "train",
                    "selection_split": "validation",
                    "evaluation_split": "locked_test",
                    "source_relationship": SOURCE_RELATIONSHIP,
                    "measurement_relationship": MEASUREMENT_RELATIONSHIP,
                    "evidence_role": "derived",
                    "synthetic": False,
                }
            )
    for component in job["components"]:
        if component["status"] != "fitted":
            continue
        parameter_rows.append(
            {
                "clock_interpretation": "nlr295_authoritative_submit_time_tz",
                "model": f"streaming_job_energy_{component['target']}",
                "feature_names_json": json.dumps(component["feature_names"]),
                "scaler_mean_json": json.dumps(component["scaler_mean"]),
                "scaler_scale_json": json.dumps(component["scaler_scale"]),
                "coefficient_json": json.dumps(component["coefficient"]),
                "intercept": float(component["intercept"][0]),
                "selected_alpha": component["selected_alpha_on_validation"],
                "fit_split": "train",
                "selection_split": "validation",
                "evaluation_split": "locked_test",
                "source_relationship": SOURCE_RELATIONSHIP,
                "measurement_relationship": MEASUREMENT_RELATIONSHIP,
                "evidence_role": "derived",
                "synthetic": False,
            }
        )
    parameters = pd.DataFrame(parameter_rows)
    _atomic_parquet(parameters, paths.physical_parameters)
    atomic_json(paths.surrogate, primary["surrogate"])
    split_manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "nlr295_job_workload": {
            "sort_key": "submit_time_utc_from_submit_time_tz",
            "execution_keys_retained": ["start_time_utc", "end_time_utc"],
            "family_grouping": "identifiable_array_and_job_families_are_not_split",
            "boundaries": nlr295["split_boundaries"],
            "row_counts": nlr295["split_counts"],
            "locked_test": True,
            "workload_events_sha256": sha256_file(paths.nlr295_workload_events),
            "job_energy_sha256": sha256_file(paths.nlr295_job_energy),
        },
        "nlr300_srrl_primary": primary["split_manifest"],
        "nlr300_srrl_fixed_mst_sensitivity": fixed["split_manifest"],
    }
    atomic_json(paths.split_manifest, split_manifest)
    public_primary = {key: value for key, value in primary.items() if not key.startswith("_")}
    public_fixed = {key: value for key, value in fixed.items() if not key.startswith("_")}
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "calibrated_pending_freeze",
        "fit_split": "train",
        "select_on": "validation",
        "evaluate_once_on": "locked_test",
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "synthetic": False,
        "job_energy": job,
        "primary": public_primary,
        "fixed_mst_sensitivity": public_fixed,
        "clock_sensitivity": {
            "primary_locked_test_rmse_kw": next(
                row["facility_rmse_kw"]
                for row in primary["metrics"]
                if row["split"] == "locked_test"
                and row["model"] == "calibrated_statistical_twin"
            ),
            "fixed_mst_locked_test_rmse_kw": next(
                row["facility_rmse_kw"]
                for row in fixed["metrics"]
                if row["split"] == "locked_test"
                and row["model"] == "calibrated_statistical_twin"
            ),
        },
        "outputs": {
            "job_metrics": _relative(root, paths.job_metrics),
            "facility_metrics": _relative(root, paths.facility_metrics),
            "pue_metrics": _relative(root, paths.pue_metrics),
            "uncertainty_metrics": _relative(root, paths.uncertainty_metrics),
            "physical_parameters": _relative(root, paths.physical_parameters),
            "surrogate": _relative(root, paths.surrogate),
            "split_manifest": _relative(root, paths.split_manifest),
        },
    }
    atomic_json(paths.physical_manifest, manifest)
    return {
        "job": job,
        "primary": primary,
        "fixed_mst": fixed,
        "split_manifest": split_manifest,
        "manifest": manifest,
    }


DEFAULT_ZONE_SITES = (
    ("DC1", "WEST", "Buffalo"),
    ("DC2", "CAPITL", "Albany"),
    ("DC3", "N.Y.C.", "New York City"),
    ("DC4", "LONGIL", "Long Island"),
)


def _declared_site_rows(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    supplied = cfg.get("sites", {})
    if isinstance(supplied, Mapping) and isinstance(supplied.get("sites"), Mapping):
        supplied = supplied["sites"]
    by_zone: dict[str, Mapping[str, Any]] = {}
    by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(supplied, Mapping):
        for key, value in supplied.items():
            if not isinstance(value, Mapping):
                continue
            by_id[str(key)] = value
            if value.get("zone"):
                by_zone[str(value["zone"])] = value
    rows: list[dict[str, Any]] = []
    for twin_name, zone, city in DEFAULT_ZONE_SITES:
        record = dict(by_zone.get(zone, by_id.get(twin_name, {})))
        rows.append(
            {
                "site_id": twin_name,
                "zone": zone,
                "city_label": str(record.get("city_label", city)),
                "capacity_scale": float(record.get("capacity_scale", 1.0)),
                "hardware_multiplier": float(record.get("hardware_multiplier", 1.0)),
                "battery_power_mw": record.get("battery_power_mw"),
                "battery_energy_mwh": record.get("battery_energy_mwh"),
                "grid_import_limit_mw": record.get("grid_import_limit_mw"),
            }
        )
    for row in rows:
        if row["capacity_scale"] <= 0 or row["hardware_multiplier"] <= 0:
            raise PhysicalBranchError(
                "INVALID_TWIN_TRANSFER_SCALE", f"{row['zone']}: {row}"
            )
    return rows


def _zone_climate_diagnostics(
    cfg: Mapping[str, Any],
    train_temperature: pd.Series,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    configured = cfg.get("nyiso_zone_weather_path")
    if not configured:
        return (
            {
                zone: {
                    "climate_support_status": "not_computed_zone_weather_not_supplied",
                    "climate_support_distance": None,
                    "out_of_support_fraction": None,
                }
                for zone in NYISO_ZONES
            },
            {
                "status": "not_computed_zone_weather_not_supplied",
                "blocker_code": "NYISO_ZONE_WEATHER_NOT_SUPPLIED_TO_PHYSICAL_BRANCH",
            },
        )
    path = Path(str(configured))
    if not path.is_absolute():
        path = Path(str(cfg.get("_root", "."))) / path
    if not path.is_file():
        return (
            {
                zone: {
                    "climate_support_status": "not_computed_zone_weather_missing",
                    "climate_support_distance": None,
                    "out_of_support_fraction": None,
                }
                for zone in NYISO_ZONES
            },
            {
                "status": "not_computed_zone_weather_missing",
                "blocker_code": "NYISO_ZONE_WEATHER_FILE_MISSING",
                "path": path.as_posix(),
            },
        )
    frame = pd.read_parquet(path)
    zone_column = next(
        (column for column in ("zone", "nyiso_zone", "location_domain") if column in frame),
        None,
    )
    temperature_column = next(
        (
            column
            for column in (
                "air_temp_c",
                "temperature_c",
                "temperature_2m_c",
                "T2M",
                "t2m_c",
                "temp_c",
            )
            if column in frame
        ),
        None,
    )
    if zone_column is None or temperature_column is None:
        raise PhysicalBranchError(
            "NYISO_ZONE_WEATHER_SCHEMA_MISSING",
            f"zone={zone_column}, temperature={temperature_column}",
        )
    local = pd.to_numeric(train_temperature, errors="coerce").dropna()
    if local.empty:
        raise PhysicalBranchError(
            "NLR_TRAIN_CLIMATE_EMPTY",
            "no finite SRRL training temperatures are available for transfer diagnostics",
        )
    local_min = float(local.min())
    local_max = float(local.max())
    local_mean = float(local.mean())
    local_std = max(float(local.std(ddof=0)), 1e-6)
    diagnostics: dict[str, dict[str, Any]] = {}
    for zone in NYISO_ZONES:
        values = pd.to_numeric(
            frame.loc[frame[zone_column].astype(str).eq(zone), temperature_column],
            errors="coerce",
        ).dropna()
        if values.empty:
            diagnostics[zone] = {
                "climate_support_status": "zone_rows_missing",
                "climate_support_distance": None,
                "out_of_support_fraction": None,
            }
            continue
        diagnostics[zone] = {
            "climate_support_status": "computed_from_referenced_zone_weather",
            "climate_support_distance": abs(float(values.mean()) - local_mean) / local_std,
            "out_of_support_fraction": float(
                ((values < local_min) | (values > local_max)).mean()
            ),
            "zone_weather_row_count": len(values),
            "zone_weather_path": path.as_posix(),
            "zone_weather_sha256": sha256_file(path),
        }
    status = (
        "computed"
        if all(item["climate_support_status"].startswith("computed") for item in diagnostics.values())
        else "partial_zone_coverage"
    )
    return diagnostics, {
        "status": status,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "zone_column": zone_column,
        "temperature_column": temperature_column,
        "local_training_temperature_c": {
            "minimum": local_min,
            "maximum": local_max,
            "mean": local_mean,
            "standard_deviation": local_std,
            "row_count": len(local),
        },
        "zone_diagnostics": diagnostics,
    }


def _build_transfer_and_freeze(
    root: Path,
    paths: BranchPaths,
    source_identities: Mapping[str, Any],
    calibration: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    surrogate = calibration["primary"]["surrogate"]
    source_hashes = {
        "nlr295_archive": str(source_identities["nlr295"]["sha256"]),
        "nlr300_pue": str(source_identities["nlr300_pue"]["sha256"]),
        "nlr300_weather": str(source_identities["nlr300_weather"]["sha256"]),
        "srrl_solar_15m": sha256_file(paths.srrl_solar_15m),
        "verified_overlap_primary": sha256_file(paths.overlap_primary),
        "physical_parameters": sha256_file(paths.physical_parameters),
        "split_manifest": sha256_file(paths.split_manifest),
        "piecewise_surrogate": sha256_file(paths.surrogate),
    }
    zone_weather_identity = _configured_zone_weather_identity(root, cfg)
    if zone_weather_identity is not None:
        source_hashes["nyiso_zone_weather"] = str(zone_weather_identity["sha256"])
    calibration_id = "nlr-esif-" + sha256_json(
        {
            "source_hashes": source_hashes,
            "primary_model": calibration["primary"]["facility_model"],
            "pue_model": calibration["primary"]["pue_model"],
            "surrogate": surrogate,
        }
    )[:20]
    sites = _declared_site_rows(cfg)
    primary_panel = pd.read_parquet(paths.overlap_primary, columns=["timestamp_utc", "srrl_air_temp_c"])
    primary_panel["timestamp_utc"] = pd.to_datetime(
        primary_panel["timestamp_utc"], utc=True, errors="coerce"
    )
    split_start = pd.Timestamp(
        calibration["primary"]["split_manifest"]["train_end_exclusive"]
    )
    train_temperature = primary_panel.loc[
        primary_panel["timestamp_utc"].lt(split_start), "srrl_air_temp_c"
    ]
    climate, climate_manifest = _zone_climate_diagnostics(cfg, train_temperature)
    idle_base_mw = float(surrogate["idle_facility_power_kw"]) / 1000.0
    peak_base_mw = float(surrogate["peak_facility_power_kw"]) / 1000.0
    dynamic_base_mw = max(peak_base_mw - idle_base_mw, 1e-6)
    rng = np.random.default_rng(int(cfg["seed"]))
    residual_pool = np.asarray(
        calibration["primary"]["_residual_multiplier_pool"], dtype=float
    )
    residual_pool = residual_pool[np.isfinite(residual_pool) & (residual_pool > 0)]
    if residual_pool.size == 0:
        raise PhysicalBranchError(
            "PHYSICAL_RESIDUAL_POOL_EMPTY", "no positive train/validation multipliers"
        )
    draws = max(100, int(cfg["physical_multiplier_draws"]))
    transfer_sigma = float(cfg.get("transfer_multiplier_log_sigma", 0.05))
    multiplier_pool = np.empty((draws, len(NYISO_ZONES)), dtype=float)
    for zone_index in range(len(NYISO_ZONES)):
        empirical = rng.choice(residual_pool, size=draws, replace=True)
        transfer_noise = rng.lognormal(mean=-(transfer_sigma**2) / 2.0, sigma=transfer_sigma, size=draws)
        multiplier_pool[:, zone_index] = np.clip(empirical * transfer_noise, 0.10, 10.0)
    idle_mw: list[float] = []
    work_mw_per_unit: list[float] = []
    capacity_work_units: list[float] = []
    grid_import_limit_mw: list[float] = []
    battery_power_mw: list[float] = []
    battery_energy_mwh: list[float] = []
    twin_ids: list[str] = []
    transfer_rows: list[dict[str, Any]] = []
    for zone_index, site in enumerate(sites):
        scale = float(site["capacity_scale"])
        hardware = float(site["hardware_multiplier"])
        idle = idle_base_mw * scale * hardware
        work = dynamic_base_mw * hardware
        capacity = scale
        peak = idle + work * capacity
        grid_limit = (
            float(site["grid_import_limit_mw"])
            if site["grid_import_limit_mw"] is not None
            else 1.15 * peak
        )
        battery_power = (
            float(site["battery_power_mw"])
            if site["battery_power_mw"] is not None
            else 0.25 * peak
        )
        battery_energy = (
            float(site["battery_energy_mwh"])
            if site["battery_energy_mwh"] is not None
            else 4.0 * battery_power
        )
        if min(idle, work, capacity, grid_limit, battery_power, battery_energy) < 0:
            raise PhysicalBranchError("NEGATIVE_TWIN_PARAMETER", site["zone"])
        twin_id = f"{calibration_id}-{site['site_id'].lower()}"
        twin_ids.append(twin_id)
        idle_mw.append(idle)
        work_mw_per_unit.append(work)
        capacity_work_units.append(capacity)
        grid_import_limit_mw.append(grid_limit)
        battery_power_mw.append(battery_power)
        battery_energy_mwh.append(battery_energy)
        zone_climate = climate[site["zone"]]
        transfer_rows.append(
            {
                **site,
                "calibration_id": calibration_id,
                "twin_id": twin_id,
                "twin_type": "nlr_calibrated_transfer",
                "transfer_regime": "frozen_transfer_ensemble",
                "idle_mw": idle,
                "work_mw_per_unit": work,
                "capacity_work_units": capacity,
                "peak_facility_mw": peak,
                "grid_import_limit_mw": grid_limit,
                "battery_power_mw": battery_power,
                "battery_energy_mwh": battery_energy,
                "physical_multiplier_p05": float(
                    np.quantile(multiplier_pool[:, zone_index], 0.05)
                ),
                "physical_multiplier_median": float(
                    np.quantile(multiplier_pool[:, zone_index], 0.50)
                ),
                "physical_multiplier_p95": float(
                    np.quantile(multiplier_pool[:, zone_index], 0.95)
                ),
                **zone_climate,
                "observed_facility": False,
                "counterfactual": True,
                "source_relationship": "counterfactual_parameter_transfer_from_third_party_referenced_public_data",
                "measurement_relationship": MEASUREMENT_RELATIONSHIP,
                "evidence_role": "counterfactual",
                "synthetic": False,
            }
        )
    twins = pd.DataFrame(transfer_rows)
    _atomic_parquet(twins, paths.zone_twins)
    _atomic_parquet(twins, paths.transfer_table)
    frozen_artifact = {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "twin_ids": twin_ids,
        "transfer_regime": "frozen_transfer_ensemble",
        "zones": list(NYISO_ZONES),
        "idle_mw": idle_mw,
        "work_mw_per_unit": work_mw_per_unit,
        "capacity_work_units": capacity_work_units,
        "grid_import_limit_mw": grid_import_limit_mw,
        "battery_power_mw": battery_power_mw,
        "battery_energy_mwh": battery_energy_mwh,
        "physical_multiplier_pool": multiplier_pool.tolist(),
        "source_hashes": source_hashes,
        "frozen": True,
        "frozen_at": utc_now(),
        "source_relationship": "counterfactual_parameter_transfer_from_third_party_referenced_public_data",
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "evidence_role": "counterfactual",
        "synthetic": False,
    }
    atomic_json(paths.frozen_artifact, frozen_artifact)
    transfer_manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "calibration_id": calibration_id,
        "zones": list(NYISO_ZONES),
        "twin_ids": twin_ids,
        "observed_nyiso_facilities": False,
        "counterfactual": True,
        "transfer_regime": "frozen_transfer_ensemble",
        "transfer_assumption_semantics": (
            "capacity_hardware_storage_and_grid_values_are_declared_counterfactual_assumptions"
        ),
        "climate_support_diagnostics": climate_manifest,
        "physical_multiplier_pool_shape": list(multiplier_pool.shape),
        "physical_multiplier_source": "train_validation_nlr_residual_bootstrap_plus_declared_transfer_uncertainty",
        "locked_test_excluded_from_uncertainty_fitting": True,
        "source_hashes": source_hashes,
        "outputs": {
            "zone_twins": _relative(root, paths.zone_twins),
            "transfer_table": _relative(root, paths.transfer_table),
            "frozen_artifact": _relative(root, paths.frozen_artifact),
        },
        "output_hashes": {
            "zone_twins": sha256_file(paths.zone_twins),
            "transfer_table": sha256_file(paths.transfer_table),
            "frozen_artifact": sha256_file(paths.frozen_artifact),
        },
    }
    atomic_json(paths.transfer_manifest, transfer_manifest)
    physical_manifest = _load_json(paths.physical_manifest, {})
    physical_manifest.update(
        {
            "status": "frozen",
            "frozen": True,
            "frozen_at": utc_now(),
            "calibration_id": calibration_id,
            "source_hashes": source_hashes,
            "parameter_sha256": sha256_file(paths.physical_parameters),
            "surrogate_sha256": sha256_file(paths.surrogate),
            "transfer_manifest_path": _relative(root, paths.transfer_manifest),
        }
    )
    atomic_json(paths.physical_manifest, physical_manifest)
    return {
        "calibration_id": calibration_id,
        "frozen_artifact": frozen_artifact,
        "transfer_manifest": transfer_manifest,
        "climate_support_diagnostics": climate_manifest,
    }


def _workload_events_contract(
    root: Path,
    paths: BranchPaths,
    source_identities: Mapping[str, Any],
) -> dict[str, Any]:
    schema = pq.ParquetFile(paths.nlr295_workload_events).schema_arrow
    schema_fields = [{"name": field.name, "type": str(field.type)} for field in schema]
    return {
        "schema_version": 1,
        "contract_name": "nlr295_workload_events_submit_chronology_v1",
        "relative_path": _relative(root, paths.nlr295_workload_events),
        "sha256": sha256_file(paths.nlr295_workload_events),
        "byte_count": paths.nlr295_workload_events.stat().st_size,
        "parquet_schema": schema_fields,
        "schema_sha256": sha256_json(schema_fields),
        "source_archive_sha256": str(source_identities["nlr295"]["sha256"]),
        "authoritative_sort_key": "submit_time_utc_from_submit_time_tz",
        "retained_execution_timestamps": ["start_time_utc", "end_time_utc"],
        "family_split_constraint": True,
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "synthetic": False,
    }


def _known_outputs(root: Path, paths: BranchPaths) -> dict[str, str]:
    names = (
        "nlr295_workload_events",
        "nlr295_job_energy",
        "nlr300_primary",
        "nlr300_weather_15m",
        "nlr300_fixed_mst",
        "srrl_solar_15m",
        "overlap_primary",
        "overlap_fixed_mst",
        "overlap_manifest",
        "split_manifest",
        "physical_manifest",
        "transfer_manifest",
        "job_metrics",
        "facility_metrics",
        "pue_metrics",
        "uncertainty_metrics",
        "energy_coverage",
        "component_closure",
        "physical_parameters",
        "zone_twins",
        "transfer_table",
        "surrogate",
        "frozen_artifact",
        "physical_ablation_table",
        "physical_ablation_manifest",
    )
    return {
        name: _relative(root, getattr(paths, name))
        for name in names
        if getattr(paths, name).is_file()
    }


def _outputs_with_hashes(root: Path, outputs: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in outputs.items():
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def _completed_manifest_is_reusable(
    root: Path,
    manifest: Mapping[str, Any],
    fingerprint: str,
) -> bool:
    if manifest.get("status") != "completed" or manifest.get("request_fingerprint") != fingerprint:
        return False
    outputs = manifest.get("outputs", {})
    hashes = manifest.get("output_hashes", {})
    if not isinstance(outputs, Mapping) or not isinstance(hashes, Mapping) or not outputs:
        return False
    for name, value in outputs.items():
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or hashes.get(name) != sha256_file(path):
            return False
    return True


def _same_source_content(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Compare immutable local source content, not enriched provenance fields.

    The first pass may only know a computed SHA-256.  After inventory
    validation, the same archive additionally carries publisher MD5 and
    inventory-manifest fields.  Those provenance enrichments must not turn a
    resumable code repair into an apparent source change.
    """

    if set(previous) != set(current):
        return False
    identity_fields = ("local_path", "byte_count", "sha256")
    for name in previous:
        left = previous.get(name)
        right = current.get(name)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if any(left.get(field) != right.get(field) for field in identity_fields):
            return False
    return True


def run_physical_branch(root: Path, config: dict, run_id: str) -> dict[str, Any]:
    """Run the resumable, source-backed NLR physical calibration branch.

    The function never substitutes synthetic data. Expected public-source gaps
    are returned as machine-readable blockers after all safe resumable work has
    been persisted.
    """

    repository_root = Path(root).resolve()
    cfg = _branch_config(config)
    cfg["_root"] = repository_root
    paths = _paths(repository_root, cfg, run_id)
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    resumed = paths.manifest.is_file()
    source_identities: dict[str, Any] = {}
    stages: dict[str, Any] = {}
    workload_contract: dict[str, Any] | None = None
    request_fingerprint: str | None = None
    recovery_record: dict[str, Any] | None = None
    try:
        source_identities = _required_source_identities(repository_root, paths)
        zone_weather_identity = _configured_zone_weather_identity(repository_root, cfg)
        if zone_weather_identity is not None:
            source_identities["nyiso_zone_weather"] = zone_weather_identity
        request_fingerprint = sha256_json(
            {
                "schema_version": 1,
                "run_id": paths.run_id,
                "config": _jsonable_config(cfg),
                "source_identities": source_identities,
                "implementation_sha256": _physical_implementation_sha256(),
            }
        )
        previous = _load_json(paths.manifest, {})
        if _completed_manifest_is_reusable(repository_root, previous, request_fingerprint):
            result = dict(previous["result"])
            result["resumed"] = True
            result["resume_reason"] = "matching_input_config_code_and_output_hashes"
            return result
        if previous and previous.get("request_fingerprint") not in (None, request_fingerprint):
            # A code-only repair may resume an internally failed run when the
            # source identities and effective configuration are byte-for-byte
            # unchanged.  Evidence-gate failures and changed inputs still
            # require a new run ID.  The exact failed manifest is archived by
            # the orchestrator before retry and referenced in the new record.
            same_sources = _same_source_content(
                previous.get("source_identities", {}), source_identities
            )
            same_config = previous.get("config") == _jsonable_config(cfg)
            blockers = previous.get("blockers", [])
            internal_failure = (
                previous.get("status") == "failed"
                and any(
                    isinstance(item, Mapping)
                    and item.get("code") == "PHYSICAL_BRANCH_INTERNAL_ERROR"
                    for item in blockers
                )
            )
            if not (same_sources and same_config and internal_failure):
                raise PhysicalBranchError(
                    "RUN_ID_FINGERPRINT_CONFLICT",
                    "inputs, config, or branch implementation changed; use a new run_id",
                )
            recovery_record = {
                "reason": "code_only_recovery_after_internal_failure",
                "previous_status": previous.get("status"),
                "previous_request_fingerprint": previous.get("request_fingerprint"),
                "previous_blockers": blockers,
                "failed_manifest_archive": _relative(
                    repository_root,
                    paths.artifacts / "physical_branch_manifest.failed-attempt-1.json",
                ),
            }
        nlr295 = _prepare_nlr295(paths, source_identities["nlr295"], cfg)
        stages["nlr295"] = nlr295
        workload_contract = _workload_events_contract(
            repository_root, paths, source_identities
        )
        nlr300 = _prepare_nlr300(paths, source_identities, cfg)
        stages["nlr300"] = nlr300
        support_start = max(
            pd.Timestamp(nlr295["min_start_utc"]), pd.Timestamp(nlr300["min_utc"])
        )
        support_end = min(
            pd.Timestamp(nlr295["max_end_utc"]), pd.Timestamp(nlr300["max_utc"])
        )
        expected_days = _common_support_days(support_start.isoformat(), support_end.isoformat())
        acquisition = _acquire_srrl(repository_root, paths, expected_days, cfg)
        stages["srrl_acquisition"] = acquisition
        srrl_panel = _prepare_srrl_panel(paths, acquisition, cfg)
        stages["srrl_panel"] = srrl_panel
        overlap = _prepare_overlaps(
            repository_root,
            paths,
            nlr295,
            nlr300,
            acquisition,
            srrl_panel,
            cfg,
        )
        stages["overlap"] = overlap
        requested_full_scope = bool(
            not acquisition["scope_truncated"]
            and acquisition["requested_day_count"] == acquisition["expected_day_count"]
        )
        usable_fraction = srrl_panel["source_day_count"] / max(
            int(acquisition["expected_day_count"]), 1
        )
        minimum_fraction = float(cfg["minimum_srrl_usable_day_fraction"])
        if not 0 < minimum_fraction <= 1:
            raise ValueError("minimum_srrl_usable_day_fraction must be in (0, 1]")
        if not requested_full_scope:
            raise PhysicalBranchError(
                "SRRL_FULL_COMMON_SUPPORT_NOT_REQUESTED",
                f"requested={acquisition['requested_day_count']}, expected={acquisition['expected_day_count']}",
            )
        if usable_fraction < minimum_fraction:
            raise PhysicalBranchError(
                "SRRL_USABLE_DAY_FRACTION_BELOW_GATE",
                f"usable_fraction={usable_fraction:.6f}, required={minimum_fraction:.6f}",
            )
        calibration = _calibrate_physical(
            repository_root, paths, nlr295, overlap, cfg
        )
        stages["calibration"] = {
            "physical_manifest": _relative(repository_root, paths.physical_manifest),
            "split_manifest": _relative(repository_root, paths.split_manifest),
            "primary_rows": calibration["primary"]["row_count"],
            "fixed_mst_rows": calibration["fixed_mst"]["row_count"],
        }
        transfer = _build_transfer_and_freeze(
            repository_root, paths, source_identities, calibration, cfg
        )
        stages["transfer"] = {
            "calibration_id": transfer["calibration_id"],
            "transfer_manifest": _relative(repository_root, paths.transfer_manifest),
            "climate_support_diagnostics": transfer["climate_support_diagnostics"],
        }
        from dcvpp.research.physical_ablations import (
            evaluate_physical_baselines_and_ablations,
        )

        physical_ablations = evaluate_physical_baselines_and_ablations(
            repository_root, paths, calibration, transfer, cfg
        )
        stages["physical_ablations"] = {
            "status": physical_ablations["status"],
            "manifest": _relative(repository_root, paths.physical_ablation_manifest),
            "table": _relative(repository_root, paths.physical_ablation_table),
            "row_count": physical_ablations["row_count"],
            "not_evaluable_variants": physical_ablations["not_evaluable_variants"],
        }
        outputs = _known_outputs(repository_root, paths)
        output_hashes = _outputs_with_hashes(repository_root, outputs)
        counts = {
            "nlr295_workload_rows": nlr295["row_count"],
            "nlr295_node_monitoring_rows": nlr295["node_monitoring_rows"],
            "nlr295_gpu_monitoring_rows": nlr295["gpu_monitoring_rows"],
            "nlr300_15m_rows": nlr300["row_count"],
            "srrl_expected_days": acquisition["expected_day_count"],
            "srrl_available_days": acquisition["available_day_count"],
            "srrl_failed_or_missing_days": acquisition["failed_or_missing_day_count"],
            "srrl_processed_days": srrl_panel["source_day_count"],
            "srrl_usable_day_fraction": usable_fraction,
            "primary_overlap_rows": overlap["primary_assumed_utc"]["row_count"],
            "fixed_mst_overlap_rows": overlap["fixed_mst_sensitivity"]["row_count"],
            "physical_multiplier_draws": len(
                transfer["frozen_artifact"]["physical_multiplier_pool"]
            ),
            "zone_twins": len(transfer["frozen_artifact"]["twin_ids"]),
            "physical_ablation_rows": physical_ablations["row_count"],
        }
        result = {
            "status": "completed",
            "run_id": paths.run_id,
            "resumed": resumed,
            "request_fingerprint": request_fingerprint,
            "started_at": started_at,
            "finished_at": utc_now(),
            "synthetic_fallback_used": False,
            "source_relationship": SOURCE_RELATIONSHIP,
            "measurement_relationship": MEASUREMENT_RELATIONSHIP,
            "counts": counts,
            "outputs": outputs,
            "output_hashes": output_hashes,
            "workload_events_contract": workload_contract,
            "frozen_artifact": transfer["frozen_artifact"],
            "blockers": [],
            "recovery_record": recovery_record,
            "manifest_path": _relative(repository_root, paths.manifest),
        }
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "request_fingerprint": request_fingerprint,
            "source_identities": source_identities,
            "config": _jsonable_config(cfg),
            "stages": stages,
            "outputs": outputs,
            "output_hashes": output_hashes,
            "workload_events_contract": workload_contract,
            "recovery_record": recovery_record,
            "result": result,
        }
        atomic_json(paths.manifest, manifest)
        return result
    except Exception as exc:
        if isinstance(exc, PhysicalBranchError):
            code = exc.code
            message = exc.message
            status = "blocked"
        else:
            code = "PHYSICAL_BRANCH_INTERNAL_ERROR"
            message = f"{type(exc).__name__}: {exc}"
            status = "failed"
        outputs = _known_outputs(repository_root, paths)
        output_hashes = _outputs_with_hashes(repository_root, outputs)
        blocker = {
            "code": code,
            "message": message,
            "exception_type": type(exc).__name__,
        }
        result = {
            "status": status,
            "run_id": paths.run_id,
            "resumed": resumed,
            "request_fingerprint": request_fingerprint,
            "started_at": started_at,
            "finished_at": utc_now(),
            "synthetic_fallback_used": False,
            "source_relationship": SOURCE_RELATIONSHIP,
            "measurement_relationship": MEASUREMENT_RELATIONSHIP,
            "counts": {
                "persisted_output_count": len(outputs),
                "completed_stage_count": len(stages),
            },
            "outputs": outputs,
            "output_hashes": output_hashes,
            "workload_events_contract": workload_contract,
            "frozen_artifact": None,
            "blockers": [blocker],
            "recovery_record": recovery_record,
            "manifest_path": _relative(repository_root, paths.manifest),
        }
        atomic_json(
            paths.manifest,
            {
                "schema_version": 1,
                "status": status,
                "request_fingerprint": request_fingerprint,
                "source_identities": source_identities,
                "config": _jsonable_config(cfg),
                "stages": stages,
                "outputs": outputs,
                "output_hashes": output_hashes,
                "workload_events_contract": workload_contract,
                "blockers": [blocker],
                "recovery_record": recovery_record,
                "result": result,
            },
        )
        return result
