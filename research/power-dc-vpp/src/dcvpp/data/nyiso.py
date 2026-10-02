from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TextIO

import pandas as pd

from dcvpp.core import atomic_json, sha256_file, utc_now
from dcvpp.data.download import resumable_download


NYISO_PUBLIC_ROOT = "https://mis.nyiso.com/public"
NYISO_TIMEZONE = "America/New_York"
NYISO_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
NYISO_LOAD_ZONES = (
    "WEST",
    "GENESE",
    "CENTRL",
    "NORTH",
    "MHK VL",
    "CAPITL",
    "HUD VL",
    "MILLWD",
    "DUNWOD",
    "N.Y.C.",
    "LONGIL",
)


class NyisoDataError(ValueError):
    """Raised when an official NYISO archive or row violates its data contract."""


@dataclass(frozen=True)
class NyisoProduct:
    key: str
    archive_directory: str
    archive_stem: str
    daily_stem: str
    index_page: str
    value_column: str

    @property
    def index_url(self) -> str:
        return f"{NYISO_PUBLIC_ROOT}/{self.index_page}"


PRODUCTS: dict[str, NyisoProduct] = {
    "da_lbmp": NyisoProduct(
        key="da_lbmp",
        archive_directory="damlbmp",
        archive_stem="damlbmp_zone_csv",
        daily_stem="damlbmp_zone",
        index_page="P-2Alist.htm",
        value_column="da_lbmp",
    ),
    "rt_lbmp": NyisoProduct(
        key="rt_lbmp",
        archive_directory="realtime",
        archive_stem="realtime_zone_csv",
        daily_stem="realtime_zone",
        index_page="P-24Alist.htm",
        value_column="rt_lbmp",
    ),
    "pal_actual_load": NyisoProduct(
        key="pal_actual_load",
        archive_directory="pal",
        archive_stem="pal_csv",
        daily_stem="pal",
        index_page="P-58Blist.htm",
        value_column="zone_load_mw",
    ),
    "iso_load_forecast": NyisoProduct(
        key="iso_load_forecast",
        archive_directory="isolf",
        archive_stem="isolf_csv",
        daily_stem="isolf",
        index_page="P-7list.htm",
        value_column="forecast_load_mw",
    ),
    # Optional compatibility products. They are deliberately excluded from
    # PRIMARY_PRODUCTS because the paper contract requires native 5-minute data.
    "rt_lbmp_integrated": NyisoProduct(
        key="rt_lbmp_integrated",
        archive_directory="rtlbmp",
        archive_stem="rtlbmp_zone_csv",
        daily_stem="rtlbmp_zone",
        index_page="P-4Alist.htm",
        value_column="rt_lbmp_integrated",
    ),
    "pal_actual_load_integrated": NyisoProduct(
        key="pal_actual_load_integrated",
        archive_directory="palIntegrated",
        archive_stem="palIntegrated_csv",
        daily_stem="palIntegrated",
        index_page="P-58Clist.htm",
        value_column="zone_load_mw_integrated",
    ),
}

PRIMARY_PRODUCTS = ("da_lbmp", "rt_lbmp", "pal_actual_load", "iso_load_forecast")

PRODUCT_ALIASES = {
    "da": "da_lbmp",
    "damlbmp": "da_lbmp",
    "dayahead": "da_lbmp",
    "dayaheadlbmp": "da_lbmp",
    "rt": "rt_lbmp",
    "realtime": "rt_lbmp",
    "realtimelbmp": "rt_lbmp",
    "rt5minute": "rt_lbmp",
    "rtlbmp": "rt_lbmp_integrated",
    "integratedrt": "rt_lbmp_integrated",
    "rtintegrated": "rt_lbmp_integrated",
    "pal": "pal_actual_load",
    "actual": "pal_actual_load",
    "actualload": "pal_actual_load",
    "palintegrated": "pal_actual_load_integrated",
    "integratedactual": "pal_actual_load_integrated",
    "forecast": "iso_load_forecast",
    "isolf": "iso_load_forecast",
    "isoloadforecast": "iso_load_forecast",
}

_ZONE_ALIASES = {
    "west": "WEST",
    "genese": "GENESE",
    "genesee": "GENESE",
    "centrl": "CENTRL",
    "central": "CENTRL",
    "north": "NORTH",
    "mhkvl": "MHK VL",
    "mohawkvalley": "MHK VL",
    "capitl": "CAPITL",
    "capital": "CAPITL",
    "hudvl": "HUD VL",
    "hudsonvalley": "HUD VL",
    "millwd": "MILLWD",
    "millwood": "MILLWD",
    "dunwod": "DUNWOD",
    "dunwoodie": "DUNWOD",
    "nyc": "N.Y.C.",
    "newyorkcity": "N.Y.C.",
    "longil": "LONGIL",
    "longisland": "LONGIL",
}

PalTimezoneLookup = Mapping[tuple[pd.Timestamp, str, int], str]
MetadataHook = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class NyisoArchiveRequest:
    product: str
    month: str
    archive_url: str
    index_url: str
    filename: str


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def resolve_product(product: str | NyisoProduct) -> NyisoProduct:
    if isinstance(product, NyisoProduct):
        return product
    raw = str(product).strip()
    if raw in PRODUCTS:
        return PRODUCTS[raw]
    key = PRODUCT_ALIASES.get(_token(raw))
    if key is None:
        raise NyisoDataError(f"unknown NYISO product: {product!r}")
    return PRODUCTS[key]


def _month_start(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.to_period("M").start_time


def iter_months(
    start: str | datetime | pd.Timestamp = "2021-01-01",
    end: str | datetime | pd.Timestamp = "2025-12-31",
) -> list[pd.Timestamp]:
    first = _month_start(start)
    last = _month_start(end)
    if first > last:
        raise NyisoDataError(f"NYISO month range is reversed: {first.date()} > {last.date()}")
    return list(pd.date_range(first, last, freq="MS"))


def monthly_archive_request(
    product: str | NyisoProduct,
    month: str | datetime | pd.Timestamp,
) -> NyisoArchiveRequest:
    spec = resolve_product(product)
    stamp = _month_start(month).strftime("%Y%m01")
    filename = f"{stamp}{spec.archive_stem}.zip"
    archive_url = f"{NYISO_PUBLIC_ROOT}/csv/{spec.archive_directory}/{filename}"
    return NyisoArchiveRequest(
        product=spec.key,
        month=_month_start(month).strftime("%Y-%m"),
        archive_url=archive_url,
        index_url=spec.index_url,
        filename=filename,
    )


def monthly_archive_requests(
    start: str | datetime | pd.Timestamp = "2021-01-01",
    end: str | datetime | pd.Timestamp = "2025-12-31",
    products: Iterable[str | NyisoProduct] = PRIMARY_PRODUCTS,
) -> list[NyisoArchiveRequest]:
    specs = [resolve_product(product) for product in products]
    return [monthly_archive_request(spec, month) for month in iter_months(start, end) for spec in specs]


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise NyisoDataError(f"unsafe ZIP member path: {name!r}")
    return normalized


def archive_provenance(path: Path) -> dict[str, Any]:
    """Hash an official ZIP and every file member; reading to EOF also verifies CRC."""
    path = Path(path)
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = _safe_member_name(info.filename)
                if name in seen:
                    raise NyisoDataError(f"duplicate ZIP member name: {name}")
                seen.add(name)
                digest = hashlib.sha256()
                byte_count = 0
                with archive.open(info) as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                        byte_count += len(chunk)
                if byte_count != info.file_size:
                    raise NyisoDataError(
                        f"ZIP member size mismatch for {name}: read={byte_count}, declared={info.file_size}"
                    )
                members.append(
                    {
                        "member": name,
                        "byte_count": byte_count,
                        "sha256": digest.hexdigest(),
                        "crc32": f"{info.CRC:08x}",
                    }
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise NyisoDataError(f"invalid NYISO ZIP {path}: {exc}") from exc
    if not members:
        raise NyisoDataError(f"NYISO ZIP has no file members: {path}")
    return {
        "filename": path.name,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": members,
    }


def download_monthly_archive(
    request: NyisoArchiveRequest,
    raw_root: Path,
    *,
    force: bool = False,
    metadata_hook: MetadataHook | None = None,
) -> dict[str, Any]:
    """Download one official archive, resuming a .part file through the shared downloader."""
    spec = resolve_product(request.product)
    destination = Path(raw_root) / spec.archive_directory / request.filename
    cache_hit = destination.exists() and not force
    if cache_hit:
        download_meta = {
            "stable_url": request.archive_url,
            "filename": destination.name,
            "byte_count": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "retrieved_at_epoch": destination.stat().st_mtime,
        }
    else:
        download_meta = resumable_download(
            request.archive_url,
            destination,
            stable_url=request.archive_url,
        )
    provenance = archive_provenance(destination)
    if provenance["sha256"] != download_meta["sha256"]:
        raise NyisoDataError(f"archive digest changed during verification: {destination}")
    record = {
        "product": spec.key,
        "month": request.month,
        "index_url": request.index_url,
        "stable_url": request.archive_url,
        "source_relationship": "third_party_referenced_public_data",
        "record_type": "referenced_official_historical_market_or_load_record",
        "measurement_relationship": "not_measured_by_this_study",
        "local_path": destination.as_posix(),
        "cache_hit": cache_hit,
        "retrieved_at": datetime.fromtimestamp(
            float(download_meta["retrieved_at_epoch"]), timezone.utc
        ).isoformat(),
        "archive": provenance,
    }
    if metadata_hook is not None:
        metadata_hook(record)
    return record


def download_archive_range(
    raw_root: Path,
    *,
    start: str | datetime | pd.Timestamp = "2021-01-01",
    end: str | datetime | pd.Timestamp = "2025-12-31",
    products: Iterable[str | NyisoProduct] = PRIMARY_PRODUCTS,
    manifest_path: Path | None = None,
    force: bool = False,
    continue_on_error: bool = False,
    metadata_hook: MetadataHook | None = None,
) -> dict[str, Any]:
    """Download an inclusive monthly range and checkpoint metadata after every archive."""
    records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    if manifest_path is not None and Path(manifest_path).exists():
        previous = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = {str(item["id"]): item for item in previous.get("archives", []) if "id" in item}

    def checkpoint() -> dict[str, Any]:
        manifest = {
            "source": "NYISO public MIS archives",
            "source_relationship": "third_party_referenced_public_data",
            "record_type": "referenced_official_historical_market_or_load_record",
            "measurement_relationship": "not_measured_by_this_study",
            "generated_at": utc_now(),
            "timezone": NYISO_TIMEZONE,
            "archives": [records[key] for key in sorted(records)],
            "failures": failures,
        }
        if manifest_path is not None:
            atomic_json(Path(manifest_path), manifest)
        return manifest

    for request in monthly_archive_requests(start, end, products):
        record_id = f"{request.month}:{request.product}"
        try:
            record = download_monthly_archive(
                request,
                raw_root,
                force=force,
                metadata_hook=metadata_hook,
            )
            records[record_id] = {"id": record_id, **record}
            checkpoint()
        except Exception as exc:
            failure = {"id": record_id, "stable_url": request.archive_url, "error": str(exc)}
            failures.append(failure)
            checkpoint()
            if not continue_on_error:
                raise
    return checkpoint()


def _read_csv(source: bytes | bytearray | BinaryIO | TextIO | str | Path) -> pd.DataFrame:
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return pd.read_csv(source, encoding="utf-8-sig")


def _find_column(frame: pd.DataFrame, *aliases: str, required: bool = True) -> str | None:
    columns = {_token(column): str(column) for column in frame.columns}
    for alias in aliases:
        if _token(alias) in columns:
            return columns[_token(alias)]
    if required:
        raise NyisoDataError(f"missing column {aliases}; available={list(frame.columns)}")
    return None


def _canonical_zone(value: Any) -> str | None:
    return _ZONE_ALIASES.get(_token(value))


def _parse_local_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values.astype(str).str.strip(), format="mixed", errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        raise NyisoDataError("NYISO public CSV timestamp must be a local naive wall-clock value")
    return parsed


def _ambiguous_mask(values: pd.Series) -> pd.Series:
    localized = pd.DatetimeIndex(values).tz_localize(
        NYISO_TIMEZONE,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    return pd.Series(localized.isna(), index=values.index)


def _localize_nyiso_rows(
    frame: pd.DataFrame,
    *,
    timezone_markers: pd.Series | None = None,
    pal_timezone_lookup: PalTimezoneLookup | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        raise NyisoDataError("NYISO parser produced no selected-zone rows")
    result["dst_occurrence"] = (
        result.groupby(["timestamp_local", "zone"], sort=False, dropna=False).cumcount().astype("int8")
    )
    ambiguous_mask = _ambiguous_mask(result["timestamp_local"])
    inferred_marker = pd.Series(index=result.index, dtype="object")

    if timezone_markers is not None:
        markers = timezone_markers.astype(str).str.strip().str.upper()
        bad = sorted(set(markers) - {"EST", "EDT"})
        if bad:
            raise NyisoDataError(f"unexpected PAL time-zone markers: {bad}")
        inferred_marker.loc[:] = markers

    for index, row in result.iterrows():
        key = (pd.Timestamp(row["timestamp_local"]), str(row["zone"]), int(row["dst_occurrence"]))
        pal_marker = pal_timezone_lookup.get(key) if pal_timezone_lookup is not None else None
        row_marker = inferred_marker.at[index]
        if pd.notna(row_marker) and pal_marker is not None and row_marker != pal_marker:
            raise NyisoDataError(f"PAL time-zone cross-check mismatch for {key}: {row_marker} != {pal_marker}")

    unresolved = ambiguous_mask & inferred_marker.isna()
    if unresolved.any():
        counts = result.loc[ambiguous_mask].groupby(["timestamp_local", "zone"], sort=False).size()
        complete_keys = set(counts[counts.eq(2)].index)
        for index, row in result.loc[unresolved].iterrows():
            group_key = (pd.Timestamp(row["timestamp_local"]), str(row["zone"]))
            lookup_key = (*group_key, int(row["dst_occurrence"]))
            if group_key in complete_keys:
                inferred_marker.at[index] = "EDT" if int(row["dst_occurrence"]) == 0 else "EST"
            elif pal_timezone_lookup is not None and pal_timezone_lookup.get(lookup_key) is not None:
                inferred_marker.at[index] = pal_timezone_lookup[lookup_key]
            else:
                raise NyisoDataError(
                    "ambiguous NYISO local time lacks an EST/EDT marker or both original-order folds: "
                    f"{group_key}"
                )

    ambiguous_flags = inferred_marker.eq("EDT").to_numpy()
    try:
        timestamp_local_tz = pd.DatetimeIndex(result["timestamp_local"]).tz_localize(
            NYISO_TIMEZONE,
            ambiguous=ambiguous_flags,
            nonexistent="raise",
        )
    except (ValueError, TypeError) as exc:
        raise NyisoDataError(f"cannot localize NYISO timestamp: {exc}") from exc
    actual_marker = pd.Series(timestamp_local_tz.strftime("%Z"), index=result.index)
    supplied = inferred_marker.notna()
    mismatch = supplied & actual_marker.ne(inferred_marker)
    if mismatch.any():
        sample = result.loc[mismatch, ["timestamp_local", "zone"]].head().to_dict("records")
        raise NyisoDataError(f"EST/EDT marker contradicts America/New_York rules: {sample}")

    if pal_timezone_lookup is not None:
        for index, row in result.iterrows():
            key = (pd.Timestamp(row["timestamp_local"]), str(row["zone"]), int(row["dst_occurrence"]))
            marker = pal_timezone_lookup.get(key)
            if marker is not None and actual_marker.at[index] != marker:
                raise NyisoDataError(
                    f"PAL time-zone cross-check mismatch for {key}: {actual_marker.at[index]} != {marker}"
                )

    result["timezone_abbreviation"] = actual_marker
    result["dst_fold"] = (ambiguous_mask & result["dst_occurrence"].eq(1)).astype("int8")
    result["timestamp_utc"] = timestamp_local_tz.tz_convert("UTC")
    return result


def _base_long_rows(raw: pd.DataFrame, zones: tuple[str, ...]) -> tuple[pd.DataFrame, str, str]:
    timestamp_column = _find_column(raw, "Time Stamp", "Timestamp")
    zone_column = _find_column(raw, "Name", "Zone")
    canonical = raw[zone_column].map(_canonical_zone)
    selected = raw.loc[canonical.isin(zones)].copy()
    selected["zone"] = canonical.loc[selected.index]
    selected["timestamp_local"] = _parse_local_time(selected[timestamp_column])
    selected["source_row_order"] = selected.index.astype("int64")
    return selected, timestamp_column, zone_column


def parse_nyiso_csv(
    source: bytes | bytearray | BinaryIO | TextIO | str | Path,
    product: str | NyisoProduct,
    *,
    zones: Iterable[str] = NYISO_ZONES,
    pal_timezone_lookup: PalTimezoneLookup | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    """Normalize one official NYISO daily CSV without clipping negative prices."""
    spec = resolve_product(product)
    target_zones = tuple(_canonical_zone(zone) or str(zone).strip().upper() for zone in zones)
    unknown = sorted(set(target_zones) - set(NYISO_ZONES))
    if unknown:
        raise NyisoDataError(f"unsupported target zones: {unknown}")
    raw = _read_csv(source)

    if spec.key == "iso_load_forecast":
        timestamp_column = _find_column(raw, "Time Stamp", "Timestamp")
        raw = raw.copy()
        raw["timestamp_local"] = _parse_local_time(raw[timestamp_column])
        raw["source_row_order"] = raw.index.astype("int64")
        zone_columns: dict[str, str] = {}
        for column in raw.columns:
            canonical = _canonical_zone(column)
            if canonical in target_zones:
                zone_columns[str(column)] = canonical
        missing = sorted(set(target_zones) - set(zone_columns.values()))
        if missing:
            raise NyisoDataError(f"ISO load forecast is missing target-zone columns: {missing}")
        selected = raw[["timestamp_local", "source_row_order", *zone_columns]].melt(
            id_vars=["timestamp_local", "source_row_order"],
            var_name="source_zone_column",
            value_name=spec.value_column,
        )
        selected["zone"] = selected["source_zone_column"].map(zone_columns)
        selected[spec.value_column] = pd.to_numeric(selected[spec.value_column], errors="raise")
        zone_rank = {zone: rank for rank, zone in enumerate(target_zones)}
        selected["_zone_rank"] = selected["zone"].map(zone_rank)
        selected = selected.sort_values(["source_row_order", "_zone_rank"], kind="stable").drop(
            columns=["_zone_rank"]
        )
        selected = _localize_nyiso_rows(selected, pal_timezone_lookup=pal_timezone_lookup)
        value_columns = [spec.value_column]
    else:
        if spec.key in {"da_lbmp", "rt_lbmp", "rt_lbmp_integrated"}:
            selected, _, _ = _base_long_rows(raw, target_zones)
            timezone_column = _find_column(selected, "Time Zone", "Timezone", required=False)
            lbmp_column = _find_column(selected, "LBMP ($/MWHr)", "LBMP")
            losses_column = _find_column(
                selected, "Marginal Cost Losses ($/MWHr)", "Marginal Cost Losses", required=False
            )
            congestion_column = _find_column(
                selected,
                "Marginal Cost Congestion ($/MWHr)",
                "Marginal Cost Congestion",
                required=False,
            )
            selected[spec.value_column] = pd.to_numeric(selected[lbmp_column], errors="raise")
            value_columns = [spec.value_column]
            if losses_column is not None:
                selected["marginal_loss"] = pd.to_numeric(selected[losses_column], errors="raise")
                value_columns.append("marginal_loss")
            if congestion_column is not None:
                selected["marginal_congestion"] = pd.to_numeric(
                    selected[congestion_column], errors="raise"
                )
                value_columns.append("marginal_congestion")
            markers = selected[timezone_column] if timezone_column is not None else None
            selected = _localize_nyiso_rows(
                selected,
                timezone_markers=markers,
                pal_timezone_lookup=pal_timezone_lookup,
            )
        else:
            selected, _, _ = _base_long_rows(raw, NYISO_LOAD_ZONES)
            timezone_column = _find_column(selected, "Time Zone", "Timezone", required=False)
            if timezone_column is None:
                raise NyisoDataError("PAL actual zonal load is missing its Time Zone column")
            load_column = _find_column(selected, "Integrated Load", "Load")
            selected[spec.value_column] = pd.to_numeric(selected[load_column], errors="raise")
            selected = _localize_nyiso_rows(
                selected,
                timezone_markers=selected[timezone_column],
                pal_timezone_lookup=pal_timezone_lookup,
            )
            system_value_column = (
                "system_load_mw"
                if spec.key == "pal_actual_load"
                else "system_load_mw_integrated"
            )
            system = (
                selected.groupby("timestamp_utc", sort=False, dropna=False)
                .agg(
                    **{
                        system_value_column: (spec.value_column, "sum"),
                        "system_zone_count": (spec.value_column, "count"),
                    }
                )
                .reset_index()
            )
            system["is_incomplete_system_load"] = system["system_zone_count"].ne(
                len(NYISO_LOAD_ZONES)
            )
            selected = selected.loc[selected["zone"].isin(target_zones)].merge(
                system, on="timestamp_utc", how="left", validate="many_to_one"
            )
            value_columns = [
                spec.value_column,
                system_value_column,
                "system_zone_count",
                "is_incomplete_system_load",
            ]

    selected.insert(0, "product", spec.key)
    selected["source_name"] = source_name
    columns = [
        "product",
        "timestamp_utc",
        "timestamp_local",
        "timezone_abbreviation",
        "dst_fold",
        "dst_occurrence",
        "zone",
        *value_columns,
        "source_row_order",
        "source_name",
    ]
    return selected[columns].reset_index(drop=True)


def aggregate_realtime_to_15m(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate native P-24A interval-ending 5-minute prices to complete 15-minute blocks.

    Blocks are formed in UTC so the fall-back hour remains chronological. A
    complete price is the arithmetic mean of exactly three valid 5-minute
    observations; incomplete groups are retained and explicitly flagged, with
    the paper-facing aggregate set to missing.
    """
    required = {"timestamp_utc", "zone", "rt_lbmp"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise NyisoDataError(f"native real-time rows are missing aggregation columns: {missing}")
    if rows.empty:
        raise NyisoDataError("cannot aggregate an empty native real-time table")
    source = rows.copy()
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
    sort_columns = ["timestamp_utc", "zone"]
    if "source_row_order" in source.columns:
        sort_columns.append("source_row_order")
    source = source.sort_values(sort_columns, kind="stable")
    source["is_nominal_5m"] = (
        source["timestamp_utc"].dt.minute.mod(5).eq(0)
        & source["timestamp_utc"].dt.second.eq(0)
        & source["timestamp_utc"].dt.microsecond.eq(0)
    )
    source["timestamp_15m_utc"] = source["timestamp_utc"].dt.ceil("15min")
    source["_is_negative"] = source["rt_lbmp"].lt(0)
    keys = ["timestamp_15m_utc", "zone"]
    nominal = source.loc[source["is_nominal_5m"]].copy()
    if nominal.empty:
        raise NyisoDataError("P-24A archive has no nominal five-minute settlement timestamps")
    grouped = nominal.groupby(keys, sort=True, dropna=False)
    aggregate_observed = grouped.agg(
        rt_lbmp=("rt_lbmp", "mean"),
        raw_interval_count=("rt_lbmp", "size"),
        valid_interval_count=("rt_lbmp", "count"),
        negative_interval_count=("_is_negative", "sum"),
    ).reset_index()
    if "source_name" in nominal.columns:
        aggregate_observed["source_name"] = grouped["source_name"].last().to_numpy()
    for value_column in ("marginal_loss", "marginal_congestion"):
        if value_column in nominal.columns:
            aggregate_observed[value_column] = grouped[value_column].mean().to_numpy()
    for provenance_column in ("source_member_sha256", "archive_sha256"):
        if provenance_column in nominal.columns:
            aggregate_observed[provenance_column] = grouped[provenance_column].last().to_numpy()

    source_counts = (
        source.groupby(keys, sort=True, dropna=False)
        .agg(
            raw_observation_count=("rt_lbmp", "size"),
            nominal_observation_count=("is_nominal_5m", "sum"),
        )
        .reset_index()
    )
    source_counts["off_cycle_observation_count"] = (
        source_counts["raw_observation_count"] - source_counts["nominal_observation_count"]
    )
    seen_zones = set(nominal["zone"].astype(str))
    zone_values = [zone for zone in NYISO_ZONES if zone in seen_zones]
    zone_values.extend(sorted(seen_zones - set(zone_values)))
    block_values = pd.date_range(
        nominal["timestamp_15m_utc"].min(),
        nominal["timestamp_15m_utc"].max(),
        freq="15min",
    )
    complete_grid = pd.MultiIndex.from_product(
        [block_values, zone_values], names=keys
    ).to_frame(index=False)
    aggregate = complete_grid.merge(aggregate_observed, on=keys, how="left", validate="one_to_one")
    aggregate = aggregate.merge(source_counts, on=keys, how="left", validate="one_to_one")
    count_columns = [
        "raw_interval_count",
        "valid_interval_count",
        "negative_interval_count",
        "raw_observation_count",
        "nominal_observation_count",
        "off_cycle_observation_count",
    ]
    aggregate[count_columns] = aggregate[count_columns].fillna(0).astype("int64")

    aggregate["is_incomplete_rt_15m"] = ~(
        aggregate["raw_interval_count"].eq(3) & aggregate["valid_interval_count"].eq(3)
    )
    incomplete = aggregate["is_incomplete_rt_15m"]
    aggregate.loc[incomplete, "rt_lbmp"] = float("nan")
    for value_column in ("marginal_loss", "marginal_congestion"):
        if value_column in aggregate.columns:
            aggregate.loc[incomplete, value_column] = float("nan")
    aggregate["timestamp_utc"] = aggregate.pop("timestamp_15m_utc")
    local_timestamps = pd.DatetimeIndex(aggregate["timestamp_utc"]).tz_convert(NYISO_TIMEZONE)
    aggregate["timestamp_local"] = local_timestamps.tz_localize(None)
    aggregate["timezone_abbreviation"] = local_timestamps.strftime("%Z")
    aggregate["dst_fold"] = [value.fold for value in local_timestamps.to_pydatetime()]
    aggregate["product"] = "rt_lbmp"
    aggregate["source_resolution_minutes"] = 5
    aggregate["target_resolution_minutes"] = 15
    aggregate["is_negative_rt"] = aggregate["rt_lbmp"].lt(0)
    ordered = [
        "product",
        "timestamp_utc",
        "timestamp_local",
        "timezone_abbreviation",
        "dst_fold",
        "zone",
        "rt_lbmp",
        "raw_interval_count",
        "valid_interval_count",
        "negative_interval_count",
        "raw_observation_count",
        "off_cycle_observation_count",
        "is_incomplete_rt_15m",
        "is_negative_rt",
        "source_resolution_minutes",
        "target_resolution_minutes",
    ]
    ordered.extend(column for column in aggregate.columns if column not in ordered)
    return aggregate[ordered].reset_index(drop=True)


def build_pal_timezone_lookup(pal_rows: pd.DataFrame) -> dict[tuple[pd.Timestamp, str, int], str]:
    required = ("timestamp_local", "zone", "dst_occurrence", "timezone_abbreviation")
    missing = sorted(set(required) - set(pal_rows.columns))
    if missing:
        raise NyisoDataError(f"PAL normalized rows are missing lookup columns: {missing}")
    lookup: dict[tuple[pd.Timestamp, str, int], str] = {}
    for row in pal_rows[list(required)].itertuples(index=False):
        values = dict(zip(required, row, strict=True))
        key = (
            pd.Timestamp(values["timestamp_local"]),
            str(values["zone"]),
            int(values["dst_occurrence"]),
        )
        marker = str(values["timezone_abbreviation"])
        if key in lookup and lookup[key] != marker:
            raise NyisoDataError(f"conflicting PAL time-zone markers for {key}")
        lookup[key] = marker
    return lookup


def parse_nyiso_archive(
    path: Path,
    product: str | NyisoProduct,
    *,
    zones: Iterable[str] = NYISO_ZONES,
    pal_timezone_lookup: PalTimezoneLookup | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = resolve_product(product)
    provenance = archive_provenance(Path(path))
    member_meta = {item["member"]: item for item in provenance["members"]}
    frames: list[pd.DataFrame] = []
    expected = re.compile(rf"^\d{{8}}{re.escape(spec.daily_stem)}\.csv$", re.IGNORECASE)
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            (_safe_member_name(info.filename) for info in archive.infolist() if not info.is_dir()),
            key=lambda value: PurePosixPath(value).name,
        )
        for member in members:
            basename = PurePosixPath(member).name
            if not expected.fullmatch(basename):
                continue
            payload = archive.read(member)
            frame = parse_nyiso_csv(
                payload,
                spec,
                zones=zones,
                pal_timezone_lookup=pal_timezone_lookup,
                source_name=member,
            )
            frame["source_member_sha256"] = member_meta[member]["sha256"]
            frames.append(frame)
    if not frames:
        raise NyisoDataError(f"no {spec.key} daily CSV members found in {path}")
    combined = pd.concat(frames, ignore_index=True)
    combined["archive_sha256"] = provenance["sha256"]
    return combined, provenance


def parse_monthly_archive_set(
    archive_paths: Mapping[str, Path],
    *,
    zones: Iterable[str] = NYISO_ZONES,
    aggregate_realtime: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Parse a same-month set, using PAL first as the authoritative EST/EDT cross-check."""
    canonical_paths = {resolve_product(key).key: Path(path) for key, path in archive_paths.items()}
    parsed: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict[str, Any]] = {}
    lookup: PalTimezoneLookup | None = None
    pal_product = next(
        (
            product
            for product in ("pal_actual_load", "pal_actual_load_integrated")
            if product in canonical_paths
        ),
        None,
    )
    if pal_product is not None:
        pal_rows, pal_provenance = parse_nyiso_archive(
            canonical_paths[pal_product], pal_product, zones=zones
        )
        parsed[pal_product] = pal_rows
        provenance[pal_product] = pal_provenance
        lookup = build_pal_timezone_lookup(pal_rows)
    for product, path in canonical_paths.items():
        if product == pal_product:
            continue
        rows, product_provenance = parse_nyiso_archive(
            path,
            product,
            zones=zones,
            pal_timezone_lookup=lookup,
        )
        if product == "rt_lbmp" and aggregate_realtime:
            rows = aggregate_realtime_to_15m(rows)
        parsed[product] = rows
        provenance[product] = product_provenance
    return parsed, provenance
