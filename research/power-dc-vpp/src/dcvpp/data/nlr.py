from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.data.gates import GateFailure
from dcvpp.data.time_rules import (
    NLR295_REQUIRED_TZ_COLUMNS,
    NLR300_DEFAULT_AMBIGUOUS_POLICY,
    NLR300_DEFAULT_NONEXISTENT_POLICY,
    NLR300_LOCAL_TIMEZONE,
    NLR300_LOCAL_TIMEZONE_POLICY,
    NLR300_SOURCE_CLOCK_ASSUMPTION,
    NLR300_STRICT_TIMEZONE_POLICY,
    NLR300_UTC_SOURCE_LOCAL_ANALYSIS_POLICY,
    nlr300_timezone_policy_record,
    nlr300_utc_source_local_analysis_policy_record,
)


NLR295_LANDING_URL = "https://data.nlr.gov/submissions/295"
NLR300_LANDING_URL = "https://data.nlr.gov/submissions/300"
NLR295_DOI = "10.7799/3023273"
NLR300_DOI = "10.7799/3015212"
NLR295_CATALOG_VERSION = "3"
NLR295_ARCHIVE_MD5 = "cc60eac4d10b38a1bbfe3ef7dede5590"
NLR300_PLAUSIBLE_START = datetime(2000, 1, 1)


def _validate_stable_nlr_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "data.nlr.gov":
        raise ValueError(f"NLR source URL is not an official HTTPS URL: {url}")
    if parsed.query or parsed.fragment:
        raise ValueError("stable NLR source URLs cannot contain a query or fragment")


@dataclass(frozen=True)
class NLRResource:
    source_id: str
    dataset_id: int
    stable_url: str
    relative_path: str
    media_type: str
    required: bool = True
    catalog_version: str | None = None

    def __post_init__(self) -> None:
        _validate_stable_nlr_url(self.stable_url)


NLR_RESOURCES: tuple[NLRResource, ...] = (
    NLRResource(
        source_id="nlr295_datacard",
        dataset_id=295,
        catalog_version=NLR295_CATALOG_VERSION,
        stable_url="https://data.nlr.gov/system/files/295/1773549035-datacard_0.md",
        relative_path="nlr295/datacard.md",
        media_type="text/markdown",
    ),
    NLRResource(
        source_id="nlr295_energy_metrics_archive",
        dataset_id=295,
        catalog_version=NLR295_CATALOG_VERSION,
        stable_url=(
            "https://data.nlr.gov/system/files/295/"
            "1773547999-esif.hpc.eagle.job-anon-energy-metrics.zip"
        ),
        relative_path="nlr295/esif.hpc.eagle.job-anon-energy-metrics.zip",
        media_type="application/zip",
    ),
    NLRResource(
        source_id="nlr300_readme",
        dataset_id=300,
        stable_url="https://data.nlr.gov/system/files/300/1769711385-README.md",
        relative_path="nlr300/README.md",
        media_type="text/markdown",
    ),
    NLRResource(
        source_id="nlr300_pue_parquet",
        dataset_id=300,
        stable_url=(
            "https://data.nlr.gov/system/files/300/"
            "1757103411-esif.influx.buildingData.PUE.combined.parquet"
        ),
        relative_path="nlr300/esif.influx.buildingData.PUE.combined.parquet",
        media_type="application/vnd.apache.parquet",
    ),
    NLRResource(
        source_id="nlr300_weather_parquet",
        dataset_id=300,
        stable_url=(
            "https://data.nlr.gov/system/files/300/"
            "1757105566-esif.influx.buildingData.outside.combined_2.parquet"
        ),
        relative_path="nlr300/esif.influx.buildingData.outside.combined.parquet",
        media_type="application/vnd.apache.parquet",
    ),
)


def discover_nlr_resources(raw_root: Path) -> dict[str, Any]:
    """Discover configured NLR files without following transient download URLs."""
    resources: list[dict[str, Any]] = []
    for resource in NLR_RESOURCES:
        path = raw_root / resource.relative_path
        record: dict[str, Any] = {
            "source_id": resource.source_id,
            "dataset_id": resource.dataset_id,
            "stable_url": resource.stable_url,
            "relative_path": resource.relative_path,
            "media_type": resource.media_type,
            "required": resource.required,
            "present": path.is_file(),
            "source_relationship": "third_party_referenced_public_data",
            "measurement_relationship": "not_measured_by_this_study",
        }
        if resource.dataset_id == 295:
            record["publisher_variable_boundary"] = (
                "publisher-reported monitoring-derived node/GPU energy, "
                "publisher-declared TDP-estimated CPU energy, and nulls remain distinct"
            )
        if resource.catalog_version is not None:
            record["catalog_version"] = resource.catalog_version
        if path.is_file():
            record["byte_count"] = path.stat().st_size
        resources.append(record)
    return {
        "datasets": {
            "nlr295": {
                "dataset_id": 295,
                "landing_url": NLR295_LANDING_URL,
                "doi": NLR295_DOI,
                "catalog_version": NLR295_CATALOG_VERSION,
            },
            "nlr300": {
                "dataset_id": 300,
                "landing_url": NLR300_LANDING_URL,
                "doi": NLR300_DOI,
            },
        },
        "resources": resources,
        "all_required_present": all(item["present"] for item in resources if item["required"]),
    }


def nlr_source_manifest_records(raw_root: Path) -> list[dict[str, Any]]:
    """Return a strict persisted-provenance whitelist for locally present files.

    Download responses may contain presigned ``retrieval_url`` or ``final_url``
    values. This builder never accepts or serializes those fields; the official,
    query-free resource URL is selected from :data:`NLR_RESOURCES`.
    """
    records: list[dict[str, Any]] = []
    for resource in NLR_RESOURCES:
        path = raw_root / resource.relative_path
        if not path.is_file():
            continue
        record: dict[str, Any] = {
            "source_id": resource.source_id,
            "dataset_id": resource.dataset_id,
            "stable_url": resource.stable_url,
            "local_path": Path("data/raw") / resource.relative_path,
            "byte_count": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_relationship": "third_party_referenced_public_data",
            "measurement_relationship": "not_measured_by_this_study",
        }
        if resource.dataset_id == 295:
            record["publisher_variable_boundary"] = (
                "publisher-reported monitoring-derived node/GPU energy; "
                "TDP-estimated CPU energy and missing values are separate"
            )
        if resource.catalog_version is not None:
            record["catalog_version"] = resource.catalog_version
        record["local_path"] = record["local_path"].as_posix()
        records.append(record)
    return records


def write_nlr_source_manifest(raw_root: Path, destination: Path) -> dict[str, Any]:
    manifest = {
        "generated_at": utc_now(),
        "source_policy": "official_stable_urls_only",
        "source_relationship_policy": "third_party_referenced_public_data",
        "sources": nlr_source_manifest_records(raw_root),
    }
    atomic_json(destination, manifest)
    return manifest


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _duckdb_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _parquet_key_value_metadata(parquet_file: pq.ParquetFile) -> dict[str, str]:
    metadata = parquet_file.metadata.metadata or {}
    result: dict[str, str] = {}
    for key, value in metadata.items():
        decoded_key = key.decode("utf-8", "replace")
        if decoded_key == "ARROW:schema":
            continue
        result[decoded_key] = value.decode("utf-8", "replace")
    return result


def _timezone_candidates_from_parquet(path: Path) -> list[dict[str, str]]:
    parquet_file = pq.ParquetFile(path)
    candidates: list[dict[str, str]] = []
    ts_field = parquet_file.schema_arrow.field("ts")
    if pa.types.is_timestamp(ts_field.type) and ts_field.type.tz:
        candidates.append(
            {"source": f"{path.name}:arrow_timestamp_type", "timezone": ts_field.type.tz}
        )
    for key, value in _parquet_key_value_metadata(parquet_file).items():
        normalized = re.sub(r"[^a-z]", "", key.lower())
        if normalized in {"timezone", "tz"} and value.strip():
            candidates.append(
                {"source": f"{path.name}:parquet_metadata:{key}", "timezone": value.strip()}
            )
    return candidates


_STRUCTURED_TIMEZONE_LINE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:timezone|time[ _-]?zone|tz)\s*[:=]\s*`?([^`\r\n]+?)`?\s*$"
)


def _timezone_candidates_from_official_text(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        {"source": f"{path.name}:structured_timezone_line", "timezone": match.group(1).strip()}
        for match in _STRUCTURED_TIMEZONE_LINE.finditer(text)
    ]


def evaluate_nlr300_timezone(
    parquet_paths: Sequence[Path],
    official_metadata_paths: Sequence[Path] = (),
    *,
    policy: str = NLR300_STRICT_TIMEZONE_POLICY,
    assumed_timezone: str = NLR300_LOCAL_TIMEZONE,
    source_clock_assumption: str = NLR300_SOURCE_CLOCK_ASSUMPTION,
    analysis_timezone: str = NLR300_LOCAL_TIMEZONE,
    ambiguous: str = NLR300_DEFAULT_AMBIGUOUS_POLICY,
    nonexistent: str = NLR300_DEFAULT_NONEXISTENT_POLICY,
) -> dict[str, Any]:
    """Evaluate publisher evidence or an explicit user-authorized assumption.

    The default remains strict: only publisher/file metadata can pass. The
    local-time policy is opt-in and labels its result ``evidence_role=assumed``.
    """
    checked: list[dict[str, Any]] = []
    candidates: list[dict[str, str]] = []
    for path in parquet_paths:
        parquet_file = pq.ParquetFile(path)
        ts_field = parquet_file.schema_arrow.field("ts")
        item_candidates = _timezone_candidates_from_parquet(path)
        checked.append(
            {
                "path": path.as_posix(),
                "kind": "parquet",
                "timestamp_type": str(ts_field.type),
                "metadata_keys": sorted(_parquet_key_value_metadata(parquet_file)),
                "timezone_candidates": item_candidates,
            }
        )
        candidates.extend(item_candidates)
    for path in official_metadata_paths:
        item_candidates = _timezone_candidates_from_official_text(path)
        checked.append(
            {
                "path": path.as_posix(),
                "kind": "official_text",
                "timezone_candidates": item_candidates,
            }
        )
        candidates.extend(item_candidates)

    distinct = sorted({item["timezone"] for item in candidates})
    if policy == NLR300_UTC_SOURCE_LOCAL_ANALYSIS_POLICY:
        assumption = nlr300_utc_source_local_analysis_policy_record(
            source_clock_assumption=source_clock_assumption,
            analysis_timezone=analysis_timezone,
        )
        return {
            "status": "passed",
            "policy": NLR300_UTC_SOURCE_LOCAL_ANALYSIS_POLICY,
            "evidence_role": "assumed",
            "resolved_timezone": source_clock_assumption,
            "source_clock_assumption": source_clock_assumption,
            "analysis_timezone": analysis_timezone,
            "publisher_metadata_claimed": False,
            "authoritative_timezone": distinct[0] if len(distinct) == 1 else None,
            "publisher_evidence": candidates,
            "publisher_timezone_candidates": distinct,
            "evidence": [assumption],
            "checked": checked,
            "location_inference_used": False,
            "user_authorized_assumption": True,
            "preserve_source_timestamp": True,
            "dst_semantics": assumption["dst_semantics"],
        }
    if policy == NLR300_LOCAL_TIMEZONE_POLICY:
        assumption = nlr300_timezone_policy_record(
            timezone_name=assumed_timezone,
            ambiguous=ambiguous,
            nonexistent=nonexistent,
        )
        return {
            "status": "passed",
            "policy": NLR300_LOCAL_TIMEZONE_POLICY,
            "evidence_role": "assumed",
            "resolved_timezone": assumed_timezone,
            "authoritative_timezone": distinct[0] if len(distinct) == 1 else None,
            "publisher_evidence": candidates,
            "publisher_timezone_candidates": distinct,
            "evidence": [assumption],
            "checked": checked,
            "location_inference_used": True,
            "user_authorized_assumption": True,
            "dst_transition_policy": assumption["dst_transition_policy"],
        }
    if policy != NLR300_STRICT_TIMEZONE_POLICY:
        raise ValueError(f"unsupported NLR300 timezone policy: {policy}")
    if len(distinct) == 1:
        return {
            "status": "passed",
            "policy": NLR300_STRICT_TIMEZONE_POLICY,
            "evidence_role": "publisher_metadata",
            "resolved_timezone": distinct[0],
            "authoritative_timezone": distinct[0],
            "evidence": candidates,
            "checked": checked,
            "location_inference_used": False,
        }
    reason = "no explicit authoritative timezone metadata" if not distinct else "conflicting timezone metadata"
    return {
        "status": "blocked",
        "code": "BLOCKED_TIMEZONE_NLR300",
        "reason": reason,
        "policy": NLR300_STRICT_TIMEZONE_POLICY,
        "evidence_role": "publisher_metadata_required",
        "resolved_timezone": None,
        "authoritative_timezone": None,
        "evidence": candidates,
        "checked": checked,
        "location_inference_used": False,
    }


def require_nlr300_authoritative_timezone(
    parquet_paths: Sequence[Path], official_metadata_paths: Sequence[Path] = ()
) -> str:
    result = evaluate_nlr300_timezone(parquet_paths, official_metadata_paths)
    if result["status"] != "passed":
        raise GateFailure(f"BLOCKED_TIMEZONE_NLR300: {result['reason']}")
    return str(result["authoritative_timezone"])


def require_nlr300_timezone(
    parquet_paths: Sequence[Path],
    official_metadata_paths: Sequence[Path] = (),
    *,
    policy: str = NLR300_STRICT_TIMEZONE_POLICY,
    assumed_timezone: str = NLR300_LOCAL_TIMEZONE,
    source_clock_assumption: str = NLR300_SOURCE_CLOCK_ASSUMPTION,
    analysis_timezone: str = NLR300_LOCAL_TIMEZONE,
    ambiguous: str = NLR300_DEFAULT_AMBIGUOUS_POLICY,
    nonexistent: str = NLR300_DEFAULT_NONEXISTENT_POLICY,
) -> str:
    """Return the resolved timezone for either supported evidence policy."""
    result = evaluate_nlr300_timezone(
        parquet_paths,
        official_metadata_paths,
        policy=policy,
        assumed_timezone=assumed_timezone,
        source_clock_assumption=source_clock_assumption,
        analysis_timezone=analysis_timezone,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
    )
    if result["status"] != "passed":
        raise GateFailure(f"BLOCKED_TIMEZONE_NLR300: {result['reason']}")
    return str(result["resolved_timezone"])


def profile_nlr300_parquet(
    path: Path, *, plausible_start: datetime = NLR300_PLAUSIBLE_START
) -> dict[str, Any]:
    """Profile schema, timestamp coverage, nulls, and timestamp uniqueness."""
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    if "ts" not in schema.names:
        raise ValueError(f"NLR300 Parquet is missing ts: {path}")

    identifiers = [_quoted_identifier(name) for name in schema.names]
    aggregates = ["count(*) AS row_count"]
    aggregates.extend(
        f"count(*) - count({identifier}) AS {_quoted_identifier(name + '__null_count')}"
        for name, identifier in zip(schema.names, identifiers, strict=True)
    )
    threshold = plausible_start.isoformat(sep=" ")
    aggregates.extend(
        (
            "min(ts) AS raw_min",
            "max(ts) AS raw_max",
            f"min(ts) FILTER (WHERE ts >= TIMESTAMP '{threshold}') AS plausible_min",
            f"count(*) FILTER (WHERE ts < TIMESTAMP '{threshold}') AS implausible_early_count",
            "count(DISTINCT ts) AS distinct_timestamp_count",
        )
    )
    query = f"SELECT {', '.join(aggregates)} FROM read_parquet('{_duckdb_path(path)}')"
    connection = duckdb.connect(database=":memory:")
    try:
        row = connection.execute(query).fetchone()
        names = [item[0] for item in connection.description]
    finally:
        connection.close()
    values = dict(zip(names, row, strict=True))
    row_count = int(values["row_count"])
    nulls = {
        name: {
            "null_count": int(values[f"{name}__null_count"]),
            "null_rate": (float(values[f"{name}__null_count"]) / row_count if row_count else None),
        }
        for name in schema.names
    }
    ts_field = schema.field("ts")
    distinct_timestamp_count = int(values["distinct_timestamp_count"])
    return {
        "path": path.as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": row_count,
        "row_group_count": parquet_file.metadata.num_row_groups,
        "schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "parquet_metadata_keys": sorted(_parquet_key_value_metadata(parquet_file)),
        "timestamp": {
            "column": "ts",
            "arrow_type": str(ts_field.type),
            "arrow_timezone": ts_field.type.tz if pa.types.is_timestamp(ts_field.type) else None,
            "raw_min": _iso(values["raw_min"]),
            "raw_max": _iso(values["raw_max"]),
            "plausible_start_threshold": plausible_start.isoformat(),
            "plausible_min": _iso(values["plausible_min"]),
            "implausible_early_count": int(values["implausible_early_count"]),
            "distinct_count": distinct_timestamp_count,
            "duplicate_excess_count": row_count - distinct_timestamp_count,
        },
        "nulls": nulls,
    }


def _file_hashes(path: Path, algorithms: Iterable[str]) -> dict[str, str]:
    hashes = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            for digest in hashes.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in hashes.items()}


def _safe_zip_member(name: str) -> bool:
    member = PurePosixPath(name)
    return not member.is_absolute() and ".." not in member.parts and "\\" not in name


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


_PARTITION_RE = re.compile(r"/year=(\d{4})/month=(\d{1,2})/")


def _profile_zipped_parquet(zip_file: ZipFile, member_name: str) -> dict[str, Any]:
    with zip_file.open(member_name) as handle:
        parquet_file = pq.ParquetFile(handle)
        schema = parquet_file.schema_arrow
        timezone_types: dict[str, str | None] = {}
        for column in NLR295_REQUIRED_TZ_COLUMNS:
            if column not in schema.names:
                timezone_types[column] = None
                continue
            field = schema.field(column)
            timezone_types[column] = (
                field.type.tz if pa.types.is_timestamp(field.type) else None
            )
        fields = [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ]
        physical_columns: dict[str, list[Any]] = {}
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                physical_columns.setdefault(column.path_in_schema, []).append(column.statistics)

        nulls: dict[str, dict[str, int | float | None]] = {}
        for field in schema:
            if pa.types.is_null(field.type):
                null_count: int | None = parquet_file.metadata.num_rows
            else:
                statistics = physical_columns.get(field.name, [])
                null_count = (
                    sum(int(item.null_count or 0) for item in statistics)
                    if statistics and all(item is not None for item in statistics)
                    else None
                )
            nulls[field.name] = {
                "null_count": null_count,
                "null_rate": (
                    null_count / parquet_file.metadata.num_rows
                    if null_count is not None and parquet_file.metadata.num_rows
                    else None
                ),
            }

        time_range: dict[str, dict[str, Any]] = {}
        for column in NLR295_REQUIRED_TZ_COLUMNS:
            statistics = [
                item
                for item in physical_columns.get(column, [])
                if item is not None and item.has_min_max
            ]
            time_range[column] = {
                "min_utc": _iso(min(item.min for item in statistics)) if statistics else None,
                "max_utc": _iso(max(item.max for item in statistics)) if statistics else None,
                "null_count": nulls.get(column, {}).get("null_count"),
            }
        return {
            "member": member_name,
            "row_count": parquet_file.metadata.num_rows,
            "row_group_count": parquet_file.metadata.num_row_groups,
            "fields": fields,
            "field_name_fingerprint": sha256_json(schema.names),
            "schema_fingerprint": sha256_json(fields),
            "required_timezone_columns": timezone_types,
            "required_timezone_columns_present": all(
                column in schema.names for column in NLR295_REQUIRED_TZ_COLUMNS
            ),
            "required_timezone_columns_are_aware": all(timezone_types.values()),
            "time_range_from_footer_statistics": time_range,
            "nulls_from_footer_statistics": nulls,
        }


def profile_nlr295_archive(
    path: Path,
    *,
    publisher_md5: str = NLR295_ARCHIVE_MD5,
    parquet_sample_count: int = 3,
) -> dict[str, Any]:
    """Read the ZIP central directory and sample Parquet footers in place.

    The archive is not extracted and a full CRC scan is intentionally omitted.
    MD5 and SHA-256 are calculated together in one sequential pass.
    """
    hashes = _file_hashes(path, ("md5", "sha256"))
    try:
        with ZipFile(path) as zip_file:
            infos = zip_file.infolist()
            unsafe = [info.filename for info in infos if not _safe_zip_member(info.filename)]
            parquet_infos = [info for info in infos if info.filename.endswith(".parquet")]
            if not parquet_infos:
                raise GateFailure("NLR295 archive contains no Parquet members")
            sample_indices = _sample_indices(len(parquet_infos), parquet_sample_count)
            samples = [
                _profile_zipped_parquet(zip_file, parquet_infos[index].filename)
                for index in sample_indices
            ]
            partitions: list[tuple[int, int]] = []
            for info in parquet_infos:
                match = _PARTITION_RE.search(info.filename)
                if match:
                    partitions.append((int(match.group(1)), int(match.group(2))))
            compression_types = sorted({info.compress_type for info in infos})
            encrypted_member_count = sum(bool(info.flag_bits & 0x1) for info in infos)
            result = {
                "path": path.as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": hashes["sha256"],
                "publisher_md5_expected": publisher_md5.lower(),
                "publisher_md5_actual": hashes["md5"],
                "publisher_md5_verified": hashes["md5"].lower() == publisher_md5.lower(),
                "central_directory_readable": True,
                "member_count": len(infos),
                "parquet_member_count": len(parquet_infos),
                "directory_member_count": sum(info.is_dir() for info in infos),
                "total_member_compressed_bytes": sum(info.compress_size for info in infos),
                "total_member_uncompressed_bytes": sum(info.file_size for info in infos),
                "compression_types": compression_types,
                "encrypted_member_count": encrypted_member_count,
                "unsafe_member_names": unsafe,
                "partition_min": f"{min(partitions)[0]:04d}-{min(partitions)[1]:02d}" if partitions else None,
                "partition_max": f"{max(partitions)[0]:04d}-{max(partitions)[1]:02d}" if partitions else None,
                "partition_count": len(set(partitions)),
                "sampled_parquet_members": samples,
                "sample_profile_scope": "evenly_spaced_parquet_footer_sample",
                "sampled_row_count": sum(sample["row_count"] for sample in samples),
                "sampled_field_names_consistent": len(
                    {sample["field_name_fingerprint"] for sample in samples}
                )
                == 1,
                "sampled_schema_types_consistent": len(
                    {sample["schema_fingerprint"] for sample in samples}
                )
                == 1,
                "all_samples_have_required_timezone_columns": all(
                    sample["required_timezone_columns_present"] for sample in samples
                ),
                "all_samples_have_timezone_aware_columns": all(
                    sample["required_timezone_columns_are_aware"] for sample in samples
                ),
                "archive_extracted": False,
                "full_member_crc_scan_performed": False,
            }
    except BadZipFile as exc:
        raise GateFailure(f"NLR295 archive central directory is unreadable: {exc}") from exc
    if not result["publisher_md5_verified"]:
        raise GateFailure("NLR295 publisher MD5 mismatch")
    if result["unsafe_member_names"]:
        raise GateFailure("NLR295 archive contains unsafe member paths")
    return result
