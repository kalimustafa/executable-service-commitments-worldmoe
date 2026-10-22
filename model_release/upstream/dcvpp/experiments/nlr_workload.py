from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dcvpp.experiments.forecast_common import atomic_parquet


COHORTS = (
    "A_critical_interactive",
    "B_latency_sensitive",
    "C_preemptible_training",
    "D_elastic_batch",
)


SELECTED_JOB_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("submit_time_utc", pa.timestamp("ns", tz="UTC")),
        pa.field("nodes_req", pa.float64()),
        pa.field("processors_req", pa.float64()),
        pa.field("gpus_requested", pa.float64()),
        pa.field("source_member", pa.string()),
        pa.field("source_row_order", pa.int64()),
    ]
)


def _finalize_panel(
    aggregate: pd.DataFrame,
    *,
    output_root: Path,
    archive_sha256: str,
    volume_target_field: str = "nodes_req",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregate["timestamp_utc"] = pd.to_datetime(
        aggregate["timestamp_utc"], utc=True, errors="raise"
    )
    full_index = pd.date_range(
        aggregate["timestamp_utc"].min().floor("15min"),
        aggregate["timestamp_utc"].max().floor("15min"),
        freq="15min",
        tz="UTC",
    )
    aggregate = aggregate.set_index("timestamp_utc").reindex(full_index)
    aggregate.index.name = "timestamp_utc"
    count_columns = [
        "submitted_job_count",
        "jobs_missing_nodes",
        "jobs_missing_processors",
        "jobs_missing_gpus",
        *[f"cohort_{index}_count" for index in range(len(COHORTS))],
    ]
    value_columns = [
        "submitted_nodes_requested",
        "submitted_processors_requested",
        "submitted_gpus_requested",
        *[f"cohort_{index}_volume" for index in range(len(COHORTS))],
    ]
    for column in count_columns:
        if column not in aggregate:
            aggregate[column] = 0
    for column in value_columns:
        if column not in aggregate:
            aggregate[column] = 0.0
    aggregate[count_columns] = aggregate[count_columns].fillna(0).astype("int64")
    aggregate[value_columns] = aggregate[value_columns].fillna(0.0).astype("float64")
    if "mean_event_elapsed_hours" not in aggregate:
        aggregate["mean_event_elapsed_hours"] = np.nan
    event_present = aggregate["submitted_job_count"].gt(0).to_numpy()
    since = np.empty(len(aggregate), dtype="float64")
    running = 0.25
    for index, present in enumerate(event_present):
        since[index] = running
        running = 0.25 if present else min(running + 0.25, 168.0)
    aggregate["time_since_last_event_hours"] = since
    aggregate["mean_event_elapsed_hours"] = (
        pd.to_numeric(aggregate["mean_event_elapsed_hours"], errors="coerce")
        .where(event_present, aggregate["time_since_last_event_hours"])
        .fillna(aggregate["time_since_last_event_hours"])
        .clip(lower=1.0 / 3600.0, upper=168.0)
    )
    aggregate = aggregate.reset_index()
    aggregate["cohort_names"] = "|".join(COHORTS)
    aggregate["work_volume_target_field"] = volume_target_field
    aggregate["work_volume_origin_available"] = volume_target_field == "nodes_req"
    aggregate["evidence_role"] = "derived"
    aggregate["source_relationship"] = "third_party_referenced_public_data"
    aggregate["measurement_relationship"] = "not_measured_by_this_study"
    aggregate["source_record_type"] = "referenced_publisher_trace"
    panel_path = Path(output_root) / "data/nlr295_15m_workload.parquet"
    artifact = atomic_parquet(
        aggregate,
        panel_path,
        metadata={
            "source_archive_sha256": archive_sha256,
            "timestamp_source": "submit_time_tz",
            "deduplication_key": "id",
            "target": f"15min_sum_{volume_target_field}_grouped_by_submission_time",
            "work_volume_origin_available": volume_target_field == "nodes_req",
            "source_relationship": "third_party_referenced_public_data",
            "measurement_relationship": "not_measured_by_this_study",
        },
    )
    return aggregate, artifact


def _sidecar_sha256(path: Path) -> str | None:
    sha_sidecar = path.with_suffix(".sha256")
    if sha_sidecar.is_file():
        token = sha_sidecar.read_text(encoding="utf-8").split()
        return token[0] if token else None
    for suffix in (".manifest.json", ".provenance.json"):
        candidate = path.with_suffix(suffix)
        if not candidate.is_file():
            continue
        import json

        payload = json.loads(candidate.read_text(encoding="utf-8"))
        for value in (
            payload.get("sha256"),
            (payload.get("output") or {}).get("sha256"),
            (payload.get("artifact") or {}).get("sha256"),
        ):
            if value:
                return str(value)
    return None


def build_nlr_15m_from_processed_events(
    processed_events_path: Path,
    output_root: Path,
    *,
    archive_sha256: str,
    declared_sha256: str | None = None,
    duckdb_threads: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Reuse a hash-verified physical-branch event projection when available."""
    path = Path(processed_events_path)
    declared_hash = declared_sha256 or _sidecar_sha256(path)
    if declared_hash is None:
        raise ValueError("processed NLR295 event projection has no supported hash sidecar")
    from dcvpp.core import sha256_file

    actual_hash = sha256_file(path)
    if actual_hash != declared_hash:
        raise ValueError("processed NLR295 event projection hash mismatch")
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    timestamp_column = (
        "submit_time_utc"
        if "submit_time_utc" in schema_names
        else "submit_time_tz"
        if "submit_time_tz" in schema_names
        else None
    )
    node_column = "nodes_req" if "nodes_req" in schema_names else "nodes_used" if "nodes_used" in schema_names else None
    processor_column = (
        "processors_req"
        if "processors_req" in schema_names
        else "processors_used"
        if "processors_used" in schema_names
        else None
    )
    if timestamp_column is None or node_column is None or processor_column is None or "gpus_requested" not in schema_names:
        raise ValueError("processed NLR295 event projection has an incompatible schema")
    id_expression = (
        "cast(job_key AS VARCHAR)"
        if "job_key" in schema_names
        else "cast(id AS VARCHAR)"
        if "id" in schema_names
        else "cast(row_number() OVER () AS VARCHAR)"
    )
    wallclock_expression = (
        "cast(wallclock_requested_hours AS DOUBLE)"
        if "wallclock_requested_hours" in schema_names
        else "NULL::DOUBLE"
    )
    partition_expression = "coalesce(partition, '')" if "partition" in schema_names else "''"
    qos_expression = "coalesce(qos, '')" if "qos" in schema_names else "''"
    connection = duckdb.connect()
    try:
        connection.execute(f"PRAGMA threads={int(duckdb_threads)}")
        sql_path = path.as_posix().replace("'", "''")
        connection.execute(
            f"""
            CREATE TEMP VIEW events AS
            SELECT {id_expression} AS job_key,
                   {timestamp_column} AS submit_time_utc,
                   cast({node_column} AS DOUBLE) AS nodes_req,
                   cast({processor_column} AS DOUBLE) AS processors_req,
                   cast(gpus_requested AS DOUBLE) AS gpus_requested,
                   {wallclock_expression} AS wallclock_requested_hours,
                   {partition_expression} AS partition,
                   {qos_expression} AS qos
            FROM read_parquet('{sql_path}', hive_partitioning=false)
            WHERE {timestamp_column} IS NOT NULL
            """
        )
        raw_count, unique_count, null_nodes = connection.execute(
            "SELECT count(*), count(DISTINCT job_key), sum(nodes_req IS NULL)::BIGINT FROM events"
        ).fetchone()
        aggregate = connection.execute(
            """
            WITH ranked AS (
                SELECT *, row_number() OVER (PARTITION BY job_key ORDER BY submit_time_utc DESC) AS rank
                FROM events
            ), deduplicated AS (SELECT * EXCLUDE(rank) FROM ranked WHERE rank=1),
            marked AS (
                SELECT *,
                       CASE
                         WHEN regexp_matches(lower(partition || ' ' || qos), 'interactive|debug|urgent|high') THEN 0
                         WHEN coalesce(gpus_requested, 0) > 0 THEN 2
                         WHEN wallclock_requested_hours IS NOT NULL AND wallclock_requested_hours <= 2 THEN 1
                         ELSE 3
                       END AS cohort_index,
                       greatest(
                         epoch(submit_time_utc - lag(submit_time_utc) OVER (ORDER BY submit_time_utc, job_key)) / 3600.0,
                         1.0 / 3600.0
                       ) AS event_elapsed_hours
                FROM deduplicated
            )
            SELECT time_bucket(INTERVAL '15 minutes', submit_time_utc) AS timestamp_utc,
                   count(*)::BIGINT AS submitted_job_count,
                   sum(nodes_req)::DOUBLE AS submitted_nodes_requested,
                   sum(processors_req)::DOUBLE AS submitted_processors_requested,
                   sum(gpus_requested)::DOUBLE AS submitted_gpus_requested,
                   sum(nodes_req IS NULL)::BIGINT AS jobs_missing_nodes,
                   sum(processors_req IS NULL)::BIGINT AS jobs_missing_processors,
                   sum(gpus_requested IS NULL)::BIGINT AS jobs_missing_gpus,
                   avg(event_elapsed_hours)::DOUBLE AS mean_event_elapsed_hours,
                   sum(CASE WHEN cohort_index=0 THEN 1 ELSE 0 END)::BIGINT AS cohort_0_count,
                   sum(CASE WHEN cohort_index=1 THEN 1 ELSE 0 END)::BIGINT AS cohort_1_count,
                   sum(CASE WHEN cohort_index=2 THEN 1 ELSE 0 END)::BIGINT AS cohort_2_count,
                   sum(CASE WHEN cohort_index=3 THEN 1 ELSE 0 END)::BIGINT AS cohort_3_count,
                   sum(CASE WHEN cohort_index=0 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_0_volume,
                   sum(CASE WHEN cohort_index=1 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_1_volume,
                   sum(CASE WHEN cohort_index=2 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_2_volume,
                   sum(CASE WHEN cohort_index=3 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_3_volume
            FROM marked GROUP BY 1 ORDER BY 1
            """
        ).fetch_df()
    finally:
        connection.close()
    panel, artifact = _finalize_panel(
        aggregate,
        output_root=output_root,
        archive_sha256=archive_sha256,
        volume_target_field=node_column,
    )
    audit = {
        "input_mode": "reused_hash_verified_processed_event_projection",
        "processed_events_path": path.as_posix(),
        "processed_events_sha256": actual_hash,
        "archive_sha256": archive_sha256,
        "selected_rows_before_global_deduplication": int(raw_count),
        "unique_jobs_after_global_deduplication": int(unique_count),
        "duplicate_job_rows_removed": int(raw_count - unique_count),
        "null_nodes_rows": int(null_nodes or 0),
        "panel_15m_rows": len(panel),
        "coverage_min_utc": panel["timestamp_utc"].min().isoformat(),
        "coverage_max_utc": panel["timestamp_utc"].max().isoformat(),
        "timestamp_source": timestamp_column,
        "volume_target_column": node_column,
        "volume_target_is_submission_origin_available": node_column == "nodes_req",
        "cohorts": list(COHORTS),
        "cohort_mapping": (
            "A=interactive/debug/urgent/high partition-or-QoS; C=GPU request; "
            "B=requested wallclock <=2h; D=remaining jobs"
        ),
    }
    return panel, audit, artifact


def _selected_table(
    batch: pa.RecordBatch,
    *,
    member: str,
    row_offset: int,
) -> tuple[pa.Table, dict[str, int]]:
    frame = batch.to_pandas()
    timestamps = pd.to_datetime(frame["submit_time_tz"], utc=True, errors="coerce")
    nodes = pd.to_numeric(frame["nodes_req"], errors="coerce")
    processors = pd.to_numeric(frame["processors_req"], errors="coerce")
    gpus = pd.to_numeric(frame["gpus_requested"], errors="coerce")
    valid = timestamps.notna()
    negative_nodes = int(nodes.lt(0).sum())
    negative_processors = int(processors.lt(0).sum())
    negative_gpus = int(gpus.lt(0).sum())
    if negative_nodes or negative_processors or negative_gpus:
        raise ValueError(
            "NLR295 resource requests contain negative values: "
            f"nodes={negative_nodes}, processors={negative_processors}, gpus={negative_gpus}"
        )
    identifiers = frame["id"].astype("string")
    selected = pa.Table.from_arrays(
        [
            pa.array(identifiers.loc[valid], type=pa.string(), from_pandas=True),
            pa.array(timestamps.loc[valid], type=pa.timestamp("ns", tz="UTC"), from_pandas=True),
            pa.array(nodes.loc[valid], type=pa.float64(), from_pandas=True),
            pa.array(processors.loc[valid], type=pa.float64(), from_pandas=True),
            pa.array(gpus.loc[valid], type=pa.float64(), from_pandas=True),
            pa.array(np.repeat(member, int(valid.sum())), type=pa.string()),
            pa.array(
                np.arange(row_offset, row_offset + len(frame), dtype="int64")[valid.to_numpy()],
                type=pa.int64(),
            ),
        ],
        schema=SELECTED_JOB_SCHEMA,
    )
    return selected, {
        "raw_rows": len(frame),
        "valid_timestamp_rows": int(valid.sum()),
        "null_timestamp_rows": int((~valid).sum()),
        "null_nodes_rows": int(nodes.loc[valid].isna().sum()),
    }


def build_nlr_15m_workload_panel(
    archive_path: Path,
    output_root: Path,
    *,
    archive_sha256: str,
    batch_size: int = 200_000,
    duckdb_threads: int = 8,
    keep_selected_jobs: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Stream NLR295, globally deduplicate jobs, and aggregate 15-minute submissions."""
    archive_path = Path(archive_path)
    data_root = Path(output_root) / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    selected_path = data_root / "nlr295_selected_jobs.parquet"
    selected_partial = selected_path.with_suffix(".partial.parquet")
    selected_partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(selected_partial, SELECTED_JOB_SCHEMA, compression="zstd")
    member_records: list[dict[str, Any]] = []
    totals = {
        "raw_rows": 0,
        "valid_timestamp_rows": 0,
        "null_timestamp_rows": 0,
        "null_nodes_rows": 0,
    }
    try:
        with ZipFile(archive_path) as archive:
            members = sorted(
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".parquet")
            )
            if not members:
                raise ValueError("NLR295 archive contains no Parquet partitions")
            for member in members:
                member_counts = {key: 0 for key in totals}
                row_offset = 0
                with archive.open(member) as handle:
                    parquet_file = pq.ParquetFile(handle)
                    missing = sorted(
                        {
                            "id",
                            "submit_time_tz",
                            "nodes_req",
                            "processors_req",
                            "gpus_requested",
                        }
                        - set(parquet_file.schema_arrow.names)
                    )
                    if missing:
                        raise ValueError(f"NLR295 member {member} misses workload fields: {missing}")
                    for batch in parquet_file.iter_batches(
                        batch_size=batch_size,
                        columns=[
                            "id",
                            "submit_time_tz",
                            "nodes_req",
                            "processors_req",
                            "gpus_requested",
                        ],
                    ):
                        table, counts = _selected_table(
                            batch, member=member, row_offset=row_offset
                        )
                        writer.write_table(table)
                        row_offset += len(batch)
                        for key, value in counts.items():
                            member_counts[key] += value
                            totals[key] += value
                member_records.append({"member": member, **member_counts})
    except Exception:
        writer.close()
        selected_partial.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(selected_partial, selected_path)

    database_path = data_root / "nlr_workload_aggregate.duckdb"
    database_path.unlink(missing_ok=True)
    connection = duckdb.connect(str(database_path))
    selected_sql_path = selected_path.as_posix().replace("'", "''")
    try:
        connection.execute(f"PRAGMA threads={int(duckdb_threads)}")
        connection.execute(
            f"""
            CREATE TEMP VIEW selected_jobs AS
            SELECT *,
                   coalesce(id, source_member || ':' || cast(source_row_order AS VARCHAR)) AS job_key
            FROM read_parquet('{selected_sql_path}', hive_partitioning=false)
            """
        )
        raw_count, unique_count = connection.execute(
            "SELECT count(*), count(DISTINCT job_key) FROM selected_jobs"
        ).fetchone()
        aggregate = connection.execute(
            """
            WITH ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY job_key ORDER BY source_member DESC, source_row_order DESC
                ) AS occurrence_rank
                FROM selected_jobs
            ),
            deduplicated AS (
                SELECT * EXCLUDE (occurrence_rank) FROM ranked WHERE occurrence_rank = 1
            )
            SELECT
                time_bucket(INTERVAL '15 minutes', submit_time_utc) AS timestamp_utc,
                count(*)::BIGINT AS submitted_job_count,
                sum(nodes_req)::DOUBLE AS submitted_nodes_requested,
                sum(processors_req)::DOUBLE AS submitted_processors_requested,
                sum(gpus_requested)::DOUBLE AS submitted_gpus_requested,
                sum(CASE WHEN nodes_req IS NULL THEN 1 ELSE 0 END)::BIGINT AS jobs_missing_nodes,
                sum(CASE WHEN processors_req IS NULL THEN 1 ELSE 0 END)::BIGINT AS jobs_missing_processors,
                sum(CASE WHEN gpus_requested IS NULL THEN 1 ELSE 0 END)::BIGINT AS jobs_missing_gpus,
                avg(event_elapsed_hours)::DOUBLE AS mean_event_elapsed_hours,
                sum(CASE WHEN cohort_index=0 THEN 1 ELSE 0 END)::BIGINT AS cohort_0_count,
                sum(CASE WHEN cohort_index=1 THEN 1 ELSE 0 END)::BIGINT AS cohort_1_count,
                sum(CASE WHEN cohort_index=2 THEN 1 ELSE 0 END)::BIGINT AS cohort_2_count,
                sum(CASE WHEN cohort_index=3 THEN 1 ELSE 0 END)::BIGINT AS cohort_3_count,
                sum(CASE WHEN cohort_index=0 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_0_volume,
                sum(CASE WHEN cohort_index=1 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_1_volume,
                sum(CASE WHEN cohort_index=2 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_2_volume,
                sum(CASE WHEN cohort_index=3 THEN coalesce(nodes_req,0) ELSE 0 END)::DOUBLE AS cohort_3_volume
            FROM (
                SELECT *,
                       CASE
                         WHEN coalesce(gpus_requested,0) > 0 THEN 2
                         WHEN coalesce(nodes_req,0) <= 1 THEN 0
                         WHEN coalesce(nodes_req,0) <= 4 THEN 1
                         ELSE 3
                       END AS cohort_index,
                       greatest(
                         epoch(submit_time_utc - lag(submit_time_utc) OVER (ORDER BY submit_time_utc, job_key)) / 3600.0,
                         1.0 / 3600.0
                       ) AS event_elapsed_hours
                FROM deduplicated
            ) AS marked
            GROUP BY 1
            ORDER BY 1
            """
        ).fetch_df()
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    aggregate, panel_artifact = _finalize_panel(
        aggregate,
        output_root=output_root,
        archive_sha256=archive_sha256,
        volume_target_field="nodes_req",
    )
    if not keep_selected_jobs:
        selected_path.unlink(missing_ok=True)
    audit = {
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha256,
        "parquet_member_count": len(member_records),
        "members": member_records,
        **totals,
        "selected_rows_before_global_deduplication": int(raw_count),
        "unique_jobs_after_global_deduplication": int(unique_count),
        "duplicate_job_rows_removed": int(raw_count - unique_count),
        "panel_15m_rows": len(aggregate),
        "coverage_min_utc": aggregate["timestamp_utc"].min().isoformat(),
        "coverage_max_utc": aggregate["timestamp_utc"].max().isoformat(),
        "timestamp_source": "submit_time_tz",
        "deduplication_policy": (
            "unique publisher id; retain lexically latest source partition occurrence"
        ),
        "empty_interval_policy": "complete UTC 15-minute grid; zero denotes no submitted jobs",
        "missing_request_policy": "job remains in count; missing resource request contributes no sum",
        "selected_jobs_retained": keep_selected_jobs,
    }
    return aggregate, audit, panel_artifact
