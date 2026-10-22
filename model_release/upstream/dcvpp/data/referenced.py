from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dcvpp.core import atomic_json, atomic_text, sha256_file, utc_now
from dcvpp.data.nasa_power import (
    NASA_POWER_DEFAULT_PARAMETERS,
    NYISO_POWER_SITES,
    parse_nasa_power_hourly,
)
from dcvpp.data.nlr import NLR295_ARCHIVE_MD5, profile_nlr295_archive
from dcvpp.data.nyiso import (
    NYISO_ZONES,
    PRIMARY_PRODUCTS,
    PRODUCTS,
    build_pal_timezone_lookup,
    iter_months,
    parse_nyiso_archive,
    resolve_product,
)


SOURCE_RELATIONSHIP = "third_party_referenced_public_data"
MEASUREMENT_RELATIONSHIP = "not_measured_by_this_study"
NASA_RECORD_TYPE = "referenced_gridded_reconstruction"
NYISO_RECORD_TYPE = "referenced_official_historical_market_or_load_record"
NLR295_RECORD_TYPE = "referenced_publisher_trace"
PROGRESS_SCHEMA_VERSION = 1

_NASA_ANNUAL_RE = re.compile(
    r"^(?P<slug>[A-Z]+)_(?P<year>\d{4})0101_(?P=year)1231\.json$"
)
_NLR295_PARTITION_RE = re.compile(r"/year=(?P<year>\d{4})/month=(?P<month>\d{1,2})/")


class ReferencedDataError(ValueError):
    """Raised when a referenced-data partition violates its source contract."""


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReferencedDataError(f"expected a JSON object: {path}")
    return value


def _atomic_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    for key, value in metadata.items():
        schema_metadata[f"dcvpp.{key}".encode("utf-8")] = str(value).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".parquet", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="zstd", write_statistics=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "local_path": path.as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": len(frame),
        "schema": str(table.schema),
    }


def _semantic_columns(frame: pd.DataFrame, *, evidence_role: str, record_type: str) -> pd.DataFrame:
    result = frame.copy()
    result["evidence_role"] = evidence_role
    result["source_relationship"] = SOURCE_RELATIONSHIP
    result["measurement_relationship"] = MEASUREMENT_RELATIONSHIP
    result["record_type"] = record_type
    return result


def _timestamp_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    return {
        "min_utc": values.min().isoformat(),
        "max_utc": values.max().isoformat(),
        "distinct_timestamp_count": int(values.nunique()),
    }


def _record_is_resumable(
    record: Mapping[str, Any] | None,
    source_sha256: str,
    *,
    repository_root: Path,
) -> bool:
    if not record or record.get("status") != "completed":
        return False
    if record.get("source", {}).get("sha256") != source_sha256:
        return False
    output = record.get("output", {})
    path_value = output.get("local_path")
    if not path_value:
        return False
    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path(repository_root) / path
    return path.is_file() and output.get("sha256") == sha256_file(path)


def _nasa_zone_by_slug(slug: str) -> str:
    mapping = {
        zone.replace(".", "").replace(" ", "_"): zone for zone in NYISO_POWER_SITES
    }
    try:
        return mapping[slug]
    except KeyError as exc:
        raise ReferencedDataError(f"unknown NASA POWER site slug: {slug}") from exc


def discover_nasa_annual_files(
    raw_dir: Path,
    *,
    years: Sequence[int] = tuple(range(2021, 2026)),
    zones: Sequence[str] = tuple(NYISO_POWER_SITES),
) -> dict[str, Path]:
    expected = {f"{zone}:{year}" for zone in zones for year in years}
    discovered: dict[str, Path] = {}
    for path in sorted(Path(raw_dir).glob("*.json")):
        match = _NASA_ANNUAL_RE.fullmatch(path.name)
        if match is None:
            continue
        zone = _nasa_zone_by_slug(match.group("slug"))
        year = int(match.group("year"))
        record_id = f"{zone}:{year}"
        if record_id in expected:
            if record_id in discovered:
                raise ReferencedDataError(f"duplicate NASA POWER annual source: {record_id}")
            discovered[record_id] = path
    return discovered


def build_nasa_power_panels(
    repository_root: Path,
    *,
    years: Sequence[int] = tuple(range(2021, 2026)),
    zones: Sequence[str] = tuple(NYISO_POWER_SITES),
    previous_records: Mapping[str, Mapping[str, Any]] | None = None,
    resume: bool = True,
    checkpoint: Any | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """Build one canonical UTC Parquet partition per POWER site-year."""
    root = Path(repository_root)
    raw_dir = root / "data/raw/nasa_power"
    expected_ids = [f"{zone}:{year}" for zone in zones for year in years]
    discovered = discover_nasa_annual_files(raw_dir, years=years, zones=zones)
    records = {key: dict(value) for key, value in (previous_records or {}).items()}
    failures: dict[str, dict[str, str]] = {}

    for record_id in expected_ids:
        source_path = discovered.get(record_id)
        try:
            if source_path is None:
                raise ReferencedDataError(f"missing complete annual NASA POWER JSON: {record_id}")
            provenance_path = source_path.with_suffix(".provenance.json")
            if not provenance_path.is_file():
                raise ReferencedDataError(f"missing NASA POWER provenance sidecar: {provenance_path}")
            provenance = _json(provenance_path)
            source_sha256 = sha256_file(source_path)
            if provenance.get("sha256") != source_sha256:
                raise ReferencedDataError(f"NASA POWER source hash mismatch: {source_path}")
            if resume and _record_is_resumable(
                records.get(record_id), source_sha256, repository_root=root
            ):
                continue

            zone, year_text = record_id.rsplit(":", 1)
            year = int(year_text)
            selected_parameters = tuple(provenance.get("parameters") or NASA_POWER_DEFAULT_PARAMETERS)
            frame, parsed_metadata = parse_nasa_power_hourly(
                source_path.read_bytes(), zone=zone, parameters=selected_parameters
            )
            timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
            if not timestamps.dt.year.eq(year).all():
                raise ReferencedDataError(f"NASA POWER year leakage in {source_path.name}")
            if frame.duplicated(["timestamp_utc", "zone"]).any():
                raise ReferencedDataError(f"duplicate NASA POWER site-hour rows: {source_path.name}")
            frame = _semantic_columns(
                frame, evidence_role="reconstructed", record_type=NASA_RECORD_TYPE
            )
            frame["source_sha256"] = source_sha256
            frame["source_stable_url"] = provenance.get("stable_url")
            frame["source_local_path"] = _relative(source_path, root)
            output_path = (
                root
                / "data/processed/referenced/nasa_power"
                / f"zone={zone}"
                / f"year={year}"
                / "part-00000.parquet"
            )
            output = _atomic_parquet(
                frame,
                output_path,
                metadata={
                    "evidence_role": "reconstructed",
                    "source_relationship": SOURCE_RELATIONSHIP,
                    "measurement_relationship": MEASUREMENT_RELATIONSHIP,
                    "record_type": NASA_RECORD_TYPE,
                    "time_standard": "UTC",
                    "source_sha256": source_sha256,
                },
            )
            output["local_path"] = _relative(output_path, root)
            records[record_id] = {
                "id": record_id,
                "status": "completed",
                "zone": zone,
                "year": year,
                "evidence_role": parsed_metadata["evidence_role"],
                "source_relationship": SOURCE_RELATIONSHIP,
                "measurement_relationship": MEASUREMENT_RELATIONSHIP,
                "source": {
                    "local_path": _relative(source_path, root),
                    "stable_url": provenance.get("stable_url"),
                    "sha256": source_sha256,
                    "byte_count": source_path.stat().st_size,
                    "retrieved_at_utc": provenance.get("retrieved_at_utc"),
                },
                "output": output,
                "coverage": _timestamp_coverage(frame),
                "parameters": list(selected_parameters),
            }
            failures.pop(record_id, None)
        except Exception as exc:
            failures[record_id] = {"id": record_id, "error": str(exc)}
        if checkpoint is not None:
            checkpoint("nasa_power", expected_ids, records, failures)
    return expected_ids, records, failures


def _nyiso_archive_path(root: Path, product: str, month: pd.Timestamp) -> Path:
    spec = resolve_product(product)
    filename = f"{month.strftime('%Y%m01')}{spec.archive_stem}.zip"
    return root / "data/raw/nyiso" / spec.archive_directory / filename


def _nyiso_output_path(root: Path, product: str, month: pd.Timestamp) -> Path:
    return (
        root
        / "data/processed/referenced/nyiso"
        / f"product={product}"
        / f"year={month.year:04d}"
        / f"month={month.month:02d}"
        / "part-00000.parquet"
    )


def _decorate_nyiso(
    frame: pd.DataFrame,
    *,
    product: str,
    archive_sha256: str,
    source_path: Path,
    repository_root: Path,
) -> pd.DataFrame:
    result = _semantic_columns(frame, evidence_role="observed", record_type=NYISO_RECORD_TYPE)
    result["source_revision_sha256"] = archive_sha256
    result["source_archive_local_path"] = _relative(source_path, repository_root)
    value_column = PRODUCTS[product].value_column
    if product in {"da_lbmp", "rt_lbmp", "rt_lbmp_integrated"}:
        result["is_negative_price"] = result[value_column].lt(0)
    return result


def _nyiso_partition_record(
    root: Path,
    *,
    record_id: str,
    product: str,
    month: pd.Timestamp,
    source_path: Path,
    frame: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    archive_sha256 = str(provenance["sha256"])
    decorated = _decorate_nyiso(
        frame,
        product=product,
        archive_sha256=archive_sha256,
        source_path=source_path,
        repository_root=root,
    )
    output_path = _nyiso_output_path(root, product, month)
    output = _atomic_parquet(
        decorated,
        output_path,
        metadata={
            "evidence_role": "observed",
            "source_relationship": SOURCE_RELATIONSHIP,
            "measurement_relationship": MEASUREMENT_RELATIONSHIP,
            "record_type": NYISO_RECORD_TYPE,
            "market_timezone": "America/New_York",
            "source_revision_sha256": archive_sha256,
            "negative_prices_clipped": False,
        },
    )
    output["local_path"] = _relative(output_path, root)
    value_column = PRODUCTS[product].value_column
    negative_count = (
        int(decorated[value_column].lt(0).sum())
        if product in {"da_lbmp", "rt_lbmp", "rt_lbmp_integrated"}
        else None
    )
    return {
        "id": record_id,
        "status": "completed",
        "product": product,
        "year_month": month.strftime("%Y-%m"),
        "evidence_role": "observed",
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "source": {
            "local_path": _relative(source_path, root),
            "sha256": archive_sha256,
            "byte_count": int(provenance["byte_count"]),
            "member_hashes": [
                {"member": item["member"], "sha256": item["sha256"]}
                for item in provenance["members"]
            ],
        },
        "output": output,
        "coverage": _timestamp_coverage(decorated),
        "dst_fold_row_count": int(decorated["dst_fold"].eq(1).sum()),
        "negative_price_row_count": negative_count,
        "negative_prices_clipped": False,
    }


def build_nyiso_panels(
    repository_root: Path,
    *,
    start: str = "2021-01-01",
    end: str = "2025-12-31",
    products: Sequence[str] = PRIMARY_PRODUCTS,
    max_months: int | None = None,
    previous_records: Mapping[str, Mapping[str, Any]] | None = None,
    resume: bool = True,
    checkpoint: Any | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """Stream official NYISO archives one month and one product at a time."""
    root = Path(repository_root)
    canonical_products = tuple(resolve_product(product).key for product in products)
    months = iter_months(start, end)
    selected_months = months[:max_months] if max_months is not None else months
    expected_ids = [f"{month.strftime('%Y-%m')}:{product}" for month in months for product in canonical_products]
    records = {key: dict(value) for key, value in (previous_records or {}).items()}
    failures: dict[str, dict[str, str]] = {}

    for month in selected_months:
        pal_path = _nyiso_archive_path(root, "pal_actual_load", month)
        pal_record_id = f"{month.strftime('%Y-%m')}:pal_actual_load"
        pal_rows: pd.DataFrame | None = None
        lookup = None
        try:
            if not pal_path.is_file():
                raise ReferencedDataError(f"missing NYISO PAL archive: {pal_path}")
            pal_source_sha256 = sha256_file(pal_path)
            pal_previous = records.get(pal_record_id)
            if (
                "pal_actual_load" in canonical_products
                and resume
                and _record_is_resumable(
                    pal_previous, pal_source_sha256, repository_root=root
                )
            ):
                pal_output = root / str(pal_previous["output"]["local_path"])
                # Read the file directly instead of constructing a Hive dataset;
                # the canonical ``product`` column intentionally duplicates the
                # partition-directory key.
                pal_rows = pq.ParquetFile(pal_output).read(
                    columns=[
                        "timestamp_local",
                        "zone",
                        "dst_occurrence",
                        "timezone_abbreviation",
                    ]
                ).to_pandas()
            else:
                pal_rows, pal_provenance = parse_nyiso_archive(
                    pal_path, "pal_actual_load", zones=NYISO_ZONES
                )
                if "pal_actual_load" in canonical_products:
                    records[pal_record_id] = _nyiso_partition_record(
                        root,
                        record_id=pal_record_id,
                        product="pal_actual_load",
                        month=month,
                        source_path=pal_path,
                        frame=pal_rows,
                        provenance=pal_provenance,
                    )
                    failures.pop(pal_record_id, None)
                    if checkpoint is not None:
                        checkpoint("nyiso", expected_ids, records, failures)
            lookup = build_pal_timezone_lookup(pal_rows)
        except Exception as exc:
            if "pal_actual_load" in canonical_products:
                failures[pal_record_id] = {"id": pal_record_id, "error": str(exc)}
            for product in canonical_products:
                if product == "pal_actual_load":
                    continue
                record_id = f"{month.strftime('%Y-%m')}:{product}"
                failures[record_id] = {
                    "id": record_id,
                    "error": f"PAL timezone dependency failed: {exc}",
                }
            if checkpoint is not None:
                checkpoint("nyiso", expected_ids, records, failures)
            continue

        for product in canonical_products:
            if product == "pal_actual_load":
                continue
            record_id = f"{month.strftime('%Y-%m')}:{product}"
            source_path = _nyiso_archive_path(root, product, month)
            try:
                if not source_path.is_file():
                    raise ReferencedDataError(f"missing NYISO archive: {source_path}")
                source_sha256 = sha256_file(source_path)
                if resume and _record_is_resumable(
                    records.get(record_id), source_sha256, repository_root=root
                ):
                    continue
                rows, provenance = parse_nyiso_archive(
                    source_path,
                    product,
                    zones=NYISO_ZONES,
                    pal_timezone_lookup=lookup,
                )
                records[record_id] = _nyiso_partition_record(
                    root,
                    record_id=record_id,
                    product=product,
                    month=month,
                    source_path=source_path,
                    frame=rows,
                    provenance=provenance,
                )
                failures.pop(record_id, None)
            except Exception as exc:
                failures[record_id] = {"id": record_id, "error": str(exc)}
            if checkpoint is not None:
                checkpoint("nyiso", expected_ids, records, failures)
        del pal_rows, lookup
    return expected_ids, records, failures


def _safe_zip_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _month_sequence(start: tuple[int, int], end: tuple[int, int]) -> list[str]:
    values: list[str] = []
    cursor = pd.Timestamp(year=start[0], month=start[1], day=1)
    finish = pd.Timestamp(year=end[0], month=end[1], day=1)
    while cursor <= finish:
        values.append(cursor.strftime("%Y-%m"))
        cursor += pd.offsets.MonthBegin(1)
    return values


def build_nlr295_inventory(
    repository_root: Path,
    *,
    archive_path: Path | None = None,
    publisher_md5: str = NLR295_ARCHIVE_MD5,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Persist a safe central-directory inventory without extracting the 1.44 GB ZIP."""
    root = Path(repository_root)
    source = archive_path or root / "data/raw/nlr295/esif.hpc.eagle.job-anon-energy-metrics.zip"
    destination = destination or root / "artifacts/manifests/nlr295_inventory_manifest.json"
    profile = profile_nlr295_archive(source, publisher_md5=publisher_md5, parquet_sample_count=3)
    members: list[dict[str, Any]] = []
    partitions: list[str] = []
    try:
        with ZipFile(source) as archive:
            for info in archive.infolist():
                match = _NLR295_PARTITION_RE.search(info.filename)
                partition = None
                if match is not None:
                    partition = f"{int(match.group('year')):04d}-{int(match.group('month')):02d}"
                    if info.filename.lower().endswith(".parquet"):
                        partitions.append(partition)
                members.append(
                    {
                        "member": info.filename,
                        "safe_path": _safe_zip_member(info.filename),
                        "is_directory": info.is_dir(),
                        "is_parquet": info.filename.lower().endswith(".parquet"),
                        "partition": partition,
                        "compressed_byte_count": info.compress_size,
                        "uncompressed_byte_count": info.file_size,
                        "crc32_declared": f"{info.CRC:08x}",
                        "encrypted": bool(info.flag_bits & 0x1),
                    }
                )
    except BadZipFile as exc:
        raise ReferencedDataError(f"invalid NLR295 archive: {exc}") from exc

    unique_partitions = sorted(set(partitions))
    expected_partitions: list[str] = []
    if unique_partitions:
        first = tuple(map(int, unique_partitions[0].split("-")))
        last = tuple(map(int, unique_partitions[-1].split("-")))
        expected_partitions = _month_sequence(first, last)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "source": "NLR dataset 295 publisher archive",
        "source_relationship": SOURCE_RELATIONSHIP,
        "measurement_relationship": MEASUREMENT_RELATIONSHIP,
        "evidence_role": "observed",
        "record_type": NLR295_RECORD_TYPE,
        "study_measurement_claimed": False,
        "archive": {
            "local_path": _relative(source, root),
            "byte_count": profile["byte_count"],
            "sha256": profile["sha256"],
            "publisher_md5_expected": profile["publisher_md5_expected"],
            "publisher_md5_actual": profile["publisher_md5_actual"],
            "publisher_md5_verified": profile["publisher_md5_verified"],
            "central_directory_readable": profile["central_directory_readable"],
            "archive_extracted": False,
        },
        "inventory": {
            "scope": "complete_zip_central_directory",
            "member_count": len(members),
            "parquet_member_count": sum(item["is_parquet"] for item in members),
            "all_member_paths_safe": all(item["safe_path"] for item in members),
            "encrypted_member_count": sum(item["encrypted"] for item in members),
            "members": members,
        },
        "partition_coverage": {
            "partition_count": len(unique_partitions),
            "partition_min": unique_partitions[0] if unique_partitions else None,
            "partition_max": unique_partitions[-1] if unique_partitions else None,
            "partitions": unique_partitions,
            "missing_months_inside_range": sorted(set(expected_partitions) - set(unique_partitions)),
            "duplicate_partition_member_count": len(partitions) - len(unique_partitions),
        },
        "schema_readiness": {
            "scope": profile["sample_profile_scope"],
            "sampled_parquet_members": profile["sampled_parquet_members"],
            "sampled_field_names_consistent": profile["sampled_field_names_consistent"],
            "sampled_schema_types_consistent": profile["sampled_schema_types_consistent"],
            "full_partition_schema_scan_completed": False,
        },
        "checksum_readiness": {
            "archive_sha256_computed": True,
            "publisher_md5_verified": profile["publisher_md5_verified"],
            "central_directory_crc32_declared_for_all_members": all(
                bool(item["crc32_declared"]) for item in members
            ),
            "full_member_crc_scan_completed": False,
            "member_sha256_scan_completed": False,
        },
        "resumable_extraction_and_scan_plan": {
            "status": "ready_not_started",
            "checkpoint_unit": "one_safe_parquet_member",
            "memory_bound": "stream one ZIP member at a time; never load the full archive",
            "steps": [
                "reject unsafe or encrypted member before extraction",
                "stream member into a same-filesystem temporary staging file",
                "verify uncompressed byte count and declared CRC32 while streaming",
                "read the staged Parquet footer and validate required fields and timezone types",
                "compute member SHA-256 and atomically checkpoint the member record",
                "atomically promote the staged partition only after all member gates pass",
            ],
            "resume_rule": "skip only a member whose source archive SHA-256, declared CRC32, staged output SHA-256, schema fingerprint, and completed checkpoint all match",
            "failure_rule": "retain the source ZIP, remove only the failed temporary member, and continue from the next pending member on resume",
        },
    }
    if not manifest["inventory"]["all_member_paths_safe"]:
        raise ReferencedDataError("NLR295 archive contains unsafe member paths")
    atomic_json(destination, manifest)
    return manifest


def _source_summary(
    expected_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    failures: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = list(expected_ids)
    completed_ids = sorted(
        record_id
        for record_id in expected
        if records.get(record_id, {}).get("status") == "completed"
    )
    failed_ids = sorted(record_id for record_id in expected if record_id in failures)
    pending_ids = sorted(set(expected) - set(completed_ids) - set(failed_ids))
    if len(completed_ids) == len(expected) and not failed_ids:
        status = "COMPLETE"
    elif not completed_ids and len(failed_ids) == len(expected) and expected:
        status = "FAILED"
    else:
        status = "PARTIAL"
    return {
        "status": status,
        "expected_count": len(expected),
        "completed_count": len(completed_ids),
        "failed_count": len(failed_ids),
        "pending_count": len(pending_ids),
        "completed_ids": completed_ids,
        "failed_ids": failed_ids,
        "failed": [failures[key] for key in failed_ids],
        "pending_ids": pending_ids,
        "records": [records[key] for key in completed_ids],
    }


def build_referenced_data_stage(
    repository_root: Path,
    *,
    sources: Iterable[str] = ("nasa_power", "nyiso", "nlr295"),
    nyiso_start: str = "2021-01-01",
    nyiso_end: str = "2025-12-31",
    nyiso_max_months: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run and checkpoint the bounded third-party referenced-data panel stage."""
    root = Path(repository_root)
    progress_path = root / "artifacts/manifests/referenced_data_progress_manifest.json"
    progress_hash_path = progress_path.with_suffix(".sha256")
    selected = tuple(dict.fromkeys(str(source).strip().lower() for source in sources))
    unknown = sorted(set(selected) - {"nasa_power", "nyiso", "nlr295"})
    if unknown:
        raise ReferencedDataError(f"unknown referenced-data sources: {unknown}")
    previous = _json(progress_path) if resume and progress_path.is_file() else {}
    summaries: dict[str, Any] = dict(previous.get("sources") or {})
    internal: dict[str, tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, str]]]] = {}

    for source, summary in summaries.items():
        records = {str(item["id"]): item for item in summary.get("records", [])}
        failures = {str(item["id"]): item for item in summary.get("failed", [])}
        expected = list(summary.get("completed_ids", [])) + list(summary.get("failed_ids", [])) + list(
            summary.get("pending_ids", [])
        )
        internal[source] = (list(dict.fromkeys(expected)), records, failures)

    def write_checkpoint(
        source: str,
        expected_ids: Sequence[str],
        records: Mapping[str, Mapping[str, Any]],
        failures: Mapping[str, Mapping[str, Any]],
    ) -> None:
        internal[source] = (list(expected_ids), dict(records), dict(failures))
        summaries[source] = _source_summary(expected_ids, records, failures)
        completed = sum(item.get("completed_count", 0) for item in summaries.values())
        failed = sum(item.get("failed_count", 0) for item in summaries.values())
        pending = sum(item.get("pending_count", 0) for item in summaries.values())
        expected = completed + failed + pending
        if completed == expected and failed == 0:
            status = "COMPLETE"
        elif completed == 0 and failed == expected and expected:
            status = "FAILED"
        else:
            status = "PARTIAL"
        manifest = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "stage": "referenced_data_panels",
            "status": status,
            "updated_at_utc": utc_now(),
            "source_relationship": SOURCE_RELATIONSHIP,
            "measurement_relationship": MEASUREMENT_RELATIONSHIP,
            "study_measurement_claimed": False,
            "paper_experiment_claimed": False,
            "training_claimed": False,
            "execution_policy": "bounded_partition_streaming_with_atomic_checkpoint_resume",
            "integrity": {
                "algorithm": "sha256",
                "sidecar_path": _relative(progress_hash_path, root),
            },
            "selected_sources": list(selected),
            "counts": {"completed": completed, "failed": failed, "pending": pending},
            "sources": summaries,
        }
        atomic_json(progress_path, manifest)
        atomic_text(
            progress_hash_path,
            f"{sha256_file(progress_path)}  {progress_path.name}\n",
        )

    if "nasa_power" in selected:
        previous_records = internal.get("nasa_power", ([], {}, {}))[1]
        result = build_nasa_power_panels(
            root,
            previous_records=previous_records,
            resume=resume,
            checkpoint=write_checkpoint,
        )
        write_checkpoint("nasa_power", *result)

    if "nyiso" in selected:
        previous_records = internal.get("nyiso", ([], {}, {}))[1]
        result = build_nyiso_panels(
            root,
            start=nyiso_start,
            end=nyiso_end,
            max_months=nyiso_max_months,
            previous_records=previous_records,
            resume=resume,
            checkpoint=write_checkpoint,
        )
        write_checkpoint("nyiso", *result)

    if "nlr295" in selected:
        expected_ids = ["nlr295_archive_inventory"]
        records = dict(internal.get("nlr295", ([], {}, {}))[1])
        failures: dict[str, dict[str, str]] = {}
        record_id = expected_ids[0]
        inventory_path = root / "artifacts/manifests/nlr295_inventory_manifest.json"
        source_path = root / "data/raw/nlr295/esif.hpc.eagle.job-anon-energy-metrics.zip"
        previous_record = records.get(record_id)
        can_resume = (
            resume
            and previous_record is not None
            and inventory_path.is_file()
            and previous_record.get("source", {}).get("byte_count") == source_path.stat().st_size
            and previous_record.get("source", {}).get("mtime_ns") == source_path.stat().st_mtime_ns
            and previous_record.get("output", {}).get("sha256") == sha256_file(inventory_path)
        )
        if not can_resume:
            try:
                inventory = build_nlr295_inventory(root, destination=inventory_path)
                records[record_id] = {
                    "id": record_id,
                    "status": "completed",
                    "evidence_role": "observed",
                    "source_relationship": SOURCE_RELATIONSHIP,
                    "measurement_relationship": MEASUREMENT_RELATIONSHIP,
                    "source": {
                        "local_path": inventory["archive"]["local_path"],
                        "byte_count": source_path.stat().st_size,
                        "mtime_ns": source_path.stat().st_mtime_ns,
                        "sha256": inventory["archive"]["sha256"],
                    },
                    "output": {
                        "local_path": _relative(inventory_path, root),
                        "byte_count": inventory_path.stat().st_size,
                        "sha256": sha256_file(inventory_path),
                    },
                    "partition_count": inventory["partition_coverage"]["partition_count"],
                    "full_member_crc_scan_completed": False,
                    "archive_extracted": False,
                }
            except Exception as exc:
                failures[record_id] = {"id": record_id, "error": str(exc)}
        write_checkpoint("nlr295", expected_ids, records, failures)

    return _json(progress_path)
