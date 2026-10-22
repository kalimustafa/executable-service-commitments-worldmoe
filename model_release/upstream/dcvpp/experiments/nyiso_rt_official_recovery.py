from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.data.nyiso import NYISO_PUBLIC_ROOT, NYISO_ZONES, parse_nyiso_csv
from dcvpp.experiments.dart_real_input_preparation import (
    _settlement_coverage_from_frames,
    exact_phase_dates,
)
from dcvpp.experiments.dart_training import ReferencedNYISOPanelSource
from dcvpp.experiments.forecast_common import atomic_parquet


SCHEMA_VERSION = "dcvpp.nyiso_official_rt_recovery.v1"
DEFAULT_SLOT_TOLERANCE_SECONDS = 150


class OfficialRTRecoveryError(RuntimeError):
    """Raised when official recovery evidence is malformed or ambiguous."""


@dataclass(frozen=True)
class OfficialArchive:
    month: str
    url: str
    path: Path
    sha256: str
    byte_count: int
    live_download_verified: bool


def _archive_url(month: str) -> str:
    return f"{NYISO_PUBLIC_ROOT}/csv/realtime/{month}01realtime_zone_csv.zip"


def _download_official_archive(*, month: str, destination: Path) -> OfficialArchive:
    """Fetch the current official monthly archive and atomically retain it."""

    url = _archive_url(month)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".zip.part", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "dcvpp-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        with zipfile.ZipFile(temporary) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise OfficialRTRecoveryError(
                    f"official archive CRC failure for {month}: {bad_member}"
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return OfficialArchive(
        month=month,
        url=url,
        path=destination,
        sha256=sha256_file(destination),
        byte_count=destination.stat().st_size,
        live_download_verified=True,
    )


def _official_archive_with_fallback(
    *, month: str, destination: Path, original_path: Path
) -> tuple[OfficialArchive, str | None]:
    """Prefer a live official fetch but retain valid official cached bytes on failure."""

    try:
        return _download_official_archive(month=month, destination=destination), None
    except Exception as exc:
        if not original_path.is_file():
            raise
        return _cached_official_archive(month=month, path=original_path), (
            f"{type(exc).__name__}: {exc}"
        )


def _cached_official_archive(*, month: str, path: Path) -> OfficialArchive:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise OfficialRTRecoveryError(f"cached archive CRC failure: {bad_member}")
    return OfficialArchive(
        month=month,
        url=_archive_url(month),
        path=path,
        sha256=sha256_file(path),
        byte_count=path.stat().st_size,
        live_download_verified=False,
    )


def _member_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _daily_rows(
    *, archive: OfficialArchive, local_day: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    member = f"{local_day}realtime_zone.csv"
    with zipfile.ZipFile(archive.path) as handle:
        matches = [
            name
            for name in handle.namelist()
            if PurePosixPath(name).name.lower() == member.lower()
        ]
        if len(matches) != 1:
            raise OfficialRTRecoveryError(
                f"official archive {archive.path} contains {len(matches)} members for {member}"
            )
        member_name = matches[0]
        payload = handle.read(member_name)
    rows = parse_nyiso_csv(
        payload,
        "rt_lbmp",
        zones=NYISO_ZONES,
        source_name=member_name,
    )
    rows["source_member_sha256"] = _member_sha256(payload)
    rows["archive_sha256"] = archive.sha256
    return rows, {
        "member": member_name,
        "member_sha256": _member_sha256(payload),
        "archive_sha256": archive.sha256,
    }


def _complete_timestamp_candidates(
    rows: pd.DataFrame,
    *,
    expected_tick_utc: pd.Timestamp,
    tolerance_seconds: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return complete four-zone official records near one five-minute slot.

    A timestamp with conflicting duplicate zone rows is rejected. Identical
    duplicates are collapsed to the first official source row and counted.
    """

    expected = pd.Timestamp(expected_tick_utc)
    if expected.tzinfo is None:
        expected = expected.tz_localize("UTC")
    else:
        expected = expected.tz_convert("UTC")
    delta = (rows["timestamp_utc"] - expected).abs()
    window = rows.loc[delta.le(pd.Timedelta(seconds=tolerance_seconds))].copy()
    candidates: list[dict[str, Any]] = []
    identical_duplicates = 0
    for timestamp, group in window.groupby("timestamp_utc", sort=True):
        values: dict[str, dict[str, Any]] = {}
        valid = True
        for zone in NYISO_ZONES:
            zone_rows = group.loc[group["zone"].eq(zone)].sort_values(
                "source_row_order", kind="stable"
            )
            if zone_rows.empty or not np.isfinite(zone_rows["rt_lbmp"]).all():
                valid = False
                break
            distinct_values = zone_rows["rt_lbmp"].astype(float).unique()
            if len(distinct_values) != 1:
                valid = False
                break
            identical_duplicates += max(0, len(zone_rows) - 1)
            selected = zone_rows.iloc[0]
            values[zone] = {
                "rt_lbmp": float(selected["rt_lbmp"]),
                "source_row_order": int(selected["source_row_order"]),
            }
        if valid:
            candidates.append(
                {
                    "timestamp_utc": pd.Timestamp(timestamp),
                    "absolute_offset_seconds": float(
                        abs((pd.Timestamp(timestamp) - expected).total_seconds())
                    ),
                    "values": values,
                }
            )
    candidates.sort(
        key=lambda record: (
            record["absolute_offset_seconds"],
            record["timestamp_utc"],
        )
    )
    return candidates, identical_duplicates


def select_complete_official_block(
    rows: pd.DataFrame,
    *,
    block_start_utc: pd.Timestamp,
    tolerance_seconds: int = DEFAULT_SLOT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Select exactly three source-backed five-minute records for one block."""

    block = pd.Timestamp(block_start_utc)
    if block.tzinfo is None:
        block = block.tz_localize("UTC")
    else:
        block = block.tz_convert("UTC")
    rows = rows.copy()
    rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True, errors="raise")
    chosen: list[dict[str, Any]] = []
    duplicate_count = 0
    unresolved: list[dict[str, Any]] = []
    used_timestamps: set[pd.Timestamp] = set()
    for minute in (5, 10, 15):
        expected = block + pd.Timedelta(minutes=minute)
        candidates, duplicates = _complete_timestamp_candidates(
            rows,
            expected_tick_utc=expected,
            tolerance_seconds=tolerance_seconds,
        )
        duplicate_count += duplicates
        candidate = next(
            (
                record
                for record in candidates
                if record["timestamp_utc"] not in used_timestamps
            ),
            None,
        )
        if candidate is None:
            unresolved.append(
                {
                    "expected_tick_utc": expected.isoformat(),
                    "reason": "NO_COMPLETE_FOUR_ZONE_OFFICIAL_5M_RECORD",
                }
            )
            continue
        used_timestamps.add(candidate["timestamp_utc"])
        chosen.append(
            {
                "expected_tick_utc": expected.isoformat(),
                "source_timestamp_utc": candidate["timestamp_utc"].isoformat(),
                "absolute_offset_seconds": candidate["absolute_offset_seconds"],
                "values": candidate["values"],
            }
        )
    monotonic = len(chosen) == 3 and all(
        pd.Timestamp(left["source_timestamp_utc"])
        < pd.Timestamp(right["source_timestamp_utc"])
        for left, right in zip(chosen, chosen[1:])
    )
    recovered = len(chosen) == 3 and monotonic
    if len(chosen) == 3 and not monotonic:
        unresolved.append(
            {
                "expected_tick_utc": None,
                "reason": "SOURCE_TIMESTAMPS_NOT_STRICTLY_INCREASING",
            }
        )
    return {
        "block_start_utc": block.isoformat(),
        "recovered": recovered,
        "selection_rule": {
            "expected_interval_end_offsets_minutes": [5, 10, 15],
            "maximum_absolute_timestamp_offset_seconds": int(tolerance_seconds),
            "same_four_zones_required": list(NYISO_ZONES),
            "strictly_increasing_source_timestamps_required": True,
            "nearest_official_record_per_expected_tick": True,
            "interpolation_or_fill_used": False,
        },
        "selected_ticks": chosen if recovered else [],
        "unresolved_ticks": unresolved,
        "identical_source_duplicate_rows_collapsed": duplicate_count,
    }


def _missing_blocks(
    *, repository_root: Path, phase: str, maximum_year: int
) -> tuple[list[pd.Timestamp], Any]:
    bundle = ReferencedNYISOPanelSource(
        repository_root=repository_root,
        data_root=repository_root / "data/processed/referenced/nyiso",
    ).load_through_year(maximum_year)
    matrix = bundle.rt_15m.pivot(
        index="timestamp_utc", columns="zone", values="rt_lbmp"
    ).reindex(columns=NYISO_ZONES)
    missing: list[pd.Timestamp] = []
    for day in exact_phase_dates(phase):
        timestamps = pd.date_range(day, periods=96, freq="15min", tz="UTC")
        rows = matrix.reindex(timestamps)
        missing.extend(rows.index[rows.isna().any(axis=1)].tolist())
    return missing, bundle


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def recover_official_rt_settlement(
    *,
    repository_root: Path,
    destination: Path,
    refresh_official: bool = True,
    tolerance_seconds: int = DEFAULT_SLOT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Recover source-backed RT blocks and emit a complete audit ledger."""

    root = Path(repository_root).resolve()
    output = Path(destination).resolve()
    archive_root = output / "official_archives"
    output.mkdir(parents=True, exist_ok=True)

    phase_inputs: dict[str, dict[str, Any]] = {}
    required_local_months: set[str] = set()
    for phase, maximum_year in (("selection", 2024), ("locked_test", 2025)):
        missing, bundle = _missing_blocks(
            repository_root=root, phase=phase, maximum_year=maximum_year
        )
        phase_inputs[phase] = {
            "maximum_year": maximum_year,
            "missing": missing,
            "bundle": bundle,
        }
        for block in missing:
            for minute in (5, 10, 15):
                required_local_months.add(
                    (block + pd.Timedelta(minutes=minute))
                    .tz_convert("America/New_York")
                    .strftime("%Y%m")
                )

    archives: dict[str, OfficialArchive] = {}
    source_records: list[dict[str, Any]] = []
    for month in sorted(required_local_months):
        live_path = archive_root / f"{month}01realtime_zone_csv.zip"
        original_path = root / "data/raw/nyiso/realtime" / live_path.name
        original_sha = sha256_file(original_path) if original_path.is_file() else None
        retrieval_error = None
        if refresh_official:
            archive, retrieval_error = _official_archive_with_fallback(
                month=month,
                destination=live_path,
                original_path=original_path,
            )
        elif live_path.is_file():
            archive = _cached_official_archive(month=month, path=live_path)
        else:
            archive = _cached_official_archive(month=month, path=original_path)
        archives[month] = archive
        source_records.append(
            {
                "month": month,
                "official_url": archive.url,
                "local_path": _relative(archive.path, root),
                "sha256": archive.sha256,
                "byte_count": archive.byte_count,
                "live_download_verified": archive.live_download_verified,
                "live_retrieval_error": retrieval_error,
                "preexisting_local_archive_path": (
                    _relative(original_path, root) if original_path.is_file() else None
                ),
                "preexisting_local_archive_sha256": original_sha,
                "live_matches_preexisting_archive": (
                    archive.sha256 == original_sha if original_sha is not None else None
                ),
            }
        )

    day_cache: dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]] = {}

    def load_day(tick: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any], OfficialArchive]:
        local = tick.tz_convert("America/New_York")
        month = local.strftime("%Y%m")
        day = local.strftime("%Y%m%d")
        key = (month, day)
        if key not in day_cache:
            day_cache[key] = _daily_rows(archive=archives[month], local_day=day)
        rows, member = day_cache[key]
        return rows, member, archives[month]

    ledger: list[dict[str, Any]] = []
    recovered_rows: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    recovered_by_phase: dict[str, pd.DataFrame] = {}
    for phase, input_record in phase_inputs.items():
        missing = input_record["missing"]
        phase_recovered: list[dict[str, Any]] = []
        for block in missing:
            daily_frames: list[pd.DataFrame] = []
            day_evidence: dict[str, tuple[dict[str, Any], OfficialArchive]] = {}
            for minute in (5, 10, 15):
                tick = block + pd.Timedelta(minutes=minute)
                rows, member, archive = load_day(tick)
                daily_frames.append(rows)
                day_evidence[member["member"]] = (member, archive)
            raw_rows = pd.concat(daily_frames, ignore_index=True).drop_duplicates(
                ["timestamp_utc", "zone", "source_row_order", "source_name"],
                keep="first",
            )
            result = select_complete_official_block(
                raw_rows,
                block_start_utc=block,
                tolerance_seconds=tolerance_seconds,
            )
            record: dict[str, Any] = {
                "phase": phase,
                "operating_date": block.strftime("%Y-%m-%d"),
                **result,
                "status": (
                    "RECOVERED_OFFICIAL_3X5M"
                    if result["recovered"]
                    else "INVALID_MISSING_SETTLEMENT"
                ),
                "source_files": [
                    {
                        "official_url": archive.url,
                        "local_path": _relative(archive.path, root),
                        **member,
                    }
                    for member, archive in day_evidence.values()
                ],
            }
            if result["recovered"]:
                for tick in record["selected_ticks"]:
                    source_name = str(
                        raw_rows.loc[
                            raw_rows["timestamp_utc"].eq(
                                pd.Timestamp(tick["source_timestamp_utc"])
                            ),
                            "source_name",
                        ].iloc[0]
                    )
                    member, archive = day_evidence[source_name]
                    tick["official_url"] = archive.url
                    tick["local_archive_path"] = _relative(archive.path, root)
                    tick["archive_sha256"] = archive.sha256
                    tick["source_member"] = member["member"]
                    tick["source_member_sha256"] = member["member_sha256"]
                for zone in NYISO_ZONES:
                    values = [
                        float(tick["values"][zone]["rt_lbmp"])
                        for tick in record["selected_ticks"]
                    ]
                    value_record = {
                        "phase": phase,
                        "timestamp_utc": block,
                        "zone": zone,
                        "rt_lbmp": float(np.mean(values)),
                        "official_five_minute_values": values,
                        "source_timestamps_utc": [
                            tick["source_timestamp_utc"]
                            for tick in record["selected_ticks"]
                        ],
                        "source_archive_sha256": [
                            tick["archive_sha256"] for tick in record["selected_ticks"]
                        ],
                        "source_member_sha256": [
                            tick["source_member_sha256"]
                            for tick in record["selected_ticks"]
                        ],
                        "recovery_method": "ARITHMETIC_MEAN_OF_THREE_OFFICIAL_5M_VALUES",
                        "interpolation_or_fill_used": False,
                    }
                    value_record["recovered_value_sha256"] = sha256_json(
                        {
                            key: (
                                value.isoformat()
                                if isinstance(value, pd.Timestamp)
                                else value
                            )
                            for key, value in value_record.items()
                        }
                    )
                    phase_recovered.append(value_record)
                    recovered_rows.append(value_record)
            ledger.append(record)
        recovered_frame = pd.DataFrame(phase_recovered)
        recovered_by_phase[phase] = recovered_frame
        recovered_blocks = int(
            sum(1 for record in ledger if record["phase"] == phase and record["recovered"])
        )
        unresolved_records = [
            record
            for record in ledger
            if record["phase"] == phase and not record["recovered"]
        ]
        phase_summaries[phase] = {
            "original_missing_15m_block_count": len(missing),
            "recovered_15m_block_count": recovered_blocks,
            "unresolved_15m_block_count": len(unresolved_records),
            "unresolved_operating_date_count": len(
                {record["operating_date"] for record in unresolved_records}
            ),
            "recovered_zone_value_count": len(phase_recovered),
        }

    overlay_columns = [
        "phase",
        "timestamp_utc",
        "zone",
        "rt_lbmp",
        "official_five_minute_values",
        "source_timestamps_utc",
        "source_archive_sha256",
        "source_member_sha256",
        "recovery_method",
        "interpolation_or_fill_used",
        "recovered_value_sha256",
    ]
    overlay = pd.DataFrame(recovered_rows, columns=overlay_columns)
    overlay_path = output / "recovered_rt_15m_overlay.parquet"
    atomic_parquet(
        overlay,
        overlay_path,
        metadata={
            "schema_version": f"{SCHEMA_VERSION}.overlay",
            "source": "official_nyiso_realtime_zone_csv_archives",
            "interpolation_or_fill_used": False,
        },
    )

    unresolved_rows = [
        {
            "phase": record["phase"],
            "operating_date": record["operating_date"],
            "timestamp_utc": record["block_start_utc"],
            "status": record["status"],
            "unresolved_tick_count": len(record["unresolved_ticks"]),
            "unresolved_reasons": "|".join(
                item["reason"] for item in record["unresolved_ticks"]
            ),
        }
        for record in ledger
        if not record["recovered"]
    ]
    unresolved_path = output / "unresolved_rt_15m_intervals.csv"
    _atomic_csv(pd.DataFrame(unresolved_rows), unresolved_path)

    recovery_coverage: dict[str, dict[str, Any]] = {}
    replay_indices: dict[str, dict[str, Any]] = {}
    for phase, input_record in phase_inputs.items():
        bundle = input_record["bundle"]
        rt = bundle.rt_15m.copy().set_index(["timestamp_utc", "zone"])
        phase_overlay = recovered_by_phase[phase]
        if not phase_overlay.empty:
            for row in phase_overlay.itertuples(index=False):
                rt.loc[(pd.Timestamp(row.timestamp_utc), str(row.zone)), "rt_lbmp"] = float(
                    row.rt_lbmp
                )
        combined_rt = rt.reset_index()
        coverage = _settlement_coverage_from_frames(
            phase=phase,
            dates=exact_phase_dates(phase),
            da_15m=bundle.da_15m,
            rt_15m=combined_rt,
            source_sha256=sha256_json(
                {
                    "base_source_sha256": bundle.source_manifest["partitions_sha256"],
                    "official_overlay_sha256": sha256_file(overlay_path),
                    "phase": phase,
                }
            ),
        )
        coverage_path = output / f"{phase}_post_recovery_settlement_coverage.json"
        atomic_json(coverage_path, coverage)
        recovery_coverage[phase] = {
            "path": _relative(coverage_path, root),
            "sha256": sha256_file(coverage_path),
            "status": coverage["status"],
            "settlement_evaluable_date_count": coverage[
                "settlement_evaluable_date_count"
            ],
            "settlement_invalid_date_count": coverage[
                "settlement_invalid_date_count"
            ],
            "missing_rt_interval_count": coverage["missing_rt_interval_count"],
            "date_drop_count": coverage["date_drop_count"],
        }
        complete_dates = {
            record["operating_date"]
            for record in coverage["records"]
            if record["settlement_evaluable"] is True
        }
        replay_days: dict[str, dict[str, Any]] = {}
        da_matrix = bundle.da_15m.pivot(
            index="timestamp_utc", columns="zone", values="da_lbmp"
        ).reindex(columns=NYISO_ZONES)
        rt_matrix = combined_rt.pivot(
            index="timestamp_utc", columns="zone", values="rt_lbmp"
        ).reindex(columns=NYISO_ZONES)
        day_root = output / "nyiso_replay_by_date" / phase
        for day in sorted(complete_dates):
            timestamps = pd.date_range(day, periods=96, freq="15min", tz="UTC")
            da_values = da_matrix.reindex(timestamps).to_numpy(dtype=float).T
            rt_values = rt_matrix.reindex(timestamps).to_numpy(dtype=float).T
            if not np.isfinite(da_values).all() or not np.isfinite(rt_values).all():
                raise OfficialRTRecoveryError(
                    f"post-recovery complete day is non-finite: {phase}/{day}"
                )
            day_path = day_root / f"{day}.npz"
            source_hashes = {
                "referenced_nyiso_partitions": str(
                    bundle.source_manifest["partitions_sha256"]
                ),
                "official_rt_recovery_overlay": sha256_file(overlay_path),
            }
            _atomic_npz(
                day_path,
                timestamps_utc=np.asarray(
                    timestamps.to_numpy(dtype="datetime64[ns]")
                ),
                da_lbmp=da_values,
                rt_lbmp=rt_values,
                source_hashes_json=np.asarray(
                    json.dumps(source_hashes, sort_keys=True, separators=(",", ":"))
                ),
            )
            replay_days[day] = {
                "operating_date": day,
                "path": _relative(day_path, root),
                "sha256": sha256_file(day_path),
                "source_hashes": source_hashes,
            }
        replay_index = {
            "schema_version": "dcvpp.nyiso_replay_by_date.v1",
            "status": "COMPLETE",
            "phase": phase,
            "conditional_common_settlement_panel": True,
            "full_calendar_retained_in_coverage_ledger": True,
            "invalid_dates_excluded_from_replay_index_only": True,
            "day_count": len(replay_days),
            "days": replay_days,
        }
        replay_index["index_fingerprint"] = sha256_json(replay_index)
        replay_index_path = output / f"{phase}_nyiso_replay_by_date_index.json"
        atomic_json(replay_index_path, replay_index)
        replay_indices[phase] = {
            "path": _relative(replay_index_path, root),
            "sha256": sha256_file(replay_index_path),
            "day_count": len(replay_days),
        }

    coverage_index = {
        "schema_version": "dcvpp.nyiso_post_recovery_settlement_coverage_index.v1",
        "created_at": utc_now(),
        "status": (
            "COMPLETE"
            if all(
                value["status"] == "COMPLETE"
                for value in recovery_coverage.values()
            )
            else "PARTIAL_SOURCE_COVERAGE"
        ),
        "missing_settlement_imputed": False,
        "missing_dates_silently_dropped": False,
        "formal_full_panel_decision_selection_complete": all(
            value["status"] == "COMPLETE" for value in recovery_coverage.values()
        ),
        "conditional_common_panel_available": True,
        "phases": recovery_coverage,
    }
    coverage_index["manifest_fingerprint"] = sha256_json(coverage_index)
    coverage_index_path = output / "post_recovery_settlement_coverage_index.json"
    atomic_json(coverage_index_path, coverage_index)

    ledger_path = output / "official_rt_recovery_ledger.json"
    atomic_json(
        ledger_path,
        {
            "schema_version": f"{SCHEMA_VERSION}.ledger",
            "created_at": utc_now(),
            "records": ledger,
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": (
            "COMPLETE"
            if all(item["unresolved_15m_block_count"] == 0 for item in phase_summaries.values())
            else "PARTIAL_OFFICIAL_SOURCE_RECOVERY"
        ),
        "source_policy": {
            "official_nyiso_only": True,
            "official_archive_root": f"{NYISO_PUBLIC_ROOT}/csv/realtime/",
            "interpolation_used": False,
            "forward_fill_used": False,
            "forecast_price_used": False,
            "neighbor_zone_substitution_used": False,
            "unresolved_status": "INVALID_MISSING_SETTLEMENT",
        },
        "selection_rule": {
            "target_block_timezone": "UTC",
            "coverage_timestamp_semantics": "15-minute block start; identical to build_nyiso_price_15m",
            "source_timezone": "America/New_York",
            "dst_occurrences_preserved_by_utc_conversion": True,
            "expected_interval_end_offsets_minutes": [5, 10, 15],
            "maximum_absolute_timestamp_offset_seconds": int(tolerance_seconds),
            "same_four_zones_required": list(NYISO_ZONES),
            "three_strictly_increasing_official_records_required": True,
            "duplicate_policy": "identical duplicate values collapse; conflicting duplicate values reject the tick",
            "aggregation": "arithmetic mean of exactly three selected official 5-minute LBMP values",
        },
        "phase_summaries": phase_summaries,
        "post_recovery_coverage": recovery_coverage,
        "post_recovery_coverage_index": {
            "path": _relative(coverage_index_path, root),
            "sha256": sha256_file(coverage_index_path),
        },
        "nyiso_replay_by_date": replay_indices,
        "official_archives": source_records,
        "artifacts": {
            "overlay": {
                "path": _relative(overlay_path, root),
                "sha256": sha256_file(overlay_path),
                "row_count": len(overlay),
            },
            "ledger": {
                "path": _relative(ledger_path, root),
                "sha256": sha256_file(ledger_path),
                "block_count": len(ledger),
            },
            "unresolved": {
                "path": _relative(unresolved_path, root),
                "sha256": sha256_file(unresolved_path),
                "row_count": len(unresolved_rows),
            },
        },
        "formal_full_panel_a1_a2_mcs_lcb_winner_defined": all(
            item["unresolved_15m_block_count"] == 0
            for item in phase_summaries.values()
        ),
        "conditional_common_panel_allowed_only_after_protocol_amendment": True,
        "locked_2025_model_results_opened": False,
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    manifest_path = output / "official_rt_recovery_manifest.json"
    atomic_json(manifest_path, manifest)
    return {**manifest, "manifest_path": _relative(manifest_path, root)}
