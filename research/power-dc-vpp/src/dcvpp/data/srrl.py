from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from dcvpp.core import atomic_json
from dcvpp.data.io_utils import atomic_bytes


SRRL_CATALOG_URL = "https://data.nlr.gov/submissions/7"
SRRL_SITE_HOME_URL = "https://midcdmz.nlr.gov/apps/sitehome.pl?site=BMS"
SRRL_API_DOCUMENTATION_URL = "https://midcdmz.nlr.gov/apps/data_api_doc.pl?BMS"
SRRL_FIELD_DICTIONARY_URL = "https://midcdmz.nlr.gov/apps/field_api.pl?BMS"
SRRL_RAW_API_URL = "https://midcdmz.nlr.gov/apps/data_api.pl"
SRRL_SITE_ID = "BMS"
SRRL_FIXED_MST = timezone(timedelta(hours=-7), name="MST")
SRRL_MISSING_SENTINEL = -7999.0


@dataclass(frozen=True)
class SrrlDailySchema:
    """Versioned description of one MIDC/BMS daily CSV schema."""

    version: str
    sha256: str
    columns: tuple[str, ...]
    column_count: int
    time_column: str
    timezone: str = "fixed MST (UTC-07:00)"
    missing_sentinel: float = SRRL_MISSING_SENTINEL
    api_contract: str = "midc-bms-raw-daily-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_day(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def build_srrl_daily_url(day: date | datetime | str, *, site: str = SRRL_SITE_ID) -> str:
    """Build the official MIDC raw-data URL for exactly one civil day."""
    requested_day = _coerce_day(day)
    query = urlencode(
        {
            "site": site,
            "begin": requested_day.strftime("%Y%m%d"),
            "end": requested_day.strftime("%Y%m%d"),
        }
    )
    return f"{SRRL_RAW_API_URL}?{query}"


def _find_column(columns: list[str], candidates: set[str]) -> str:
    for column in columns:
        normalized = " ".join(column.strip().lower().replace("_", " ").split())
        if normalized in candidates:
            return column
    raise ValueError(f"SRRL CSV is missing a required column; expected one of {sorted(candidates)}")


def _find_time_column(columns: list[str]) -> str:
    for column in columns:
        normalized = column.strip().upper()
        if normalized == "MST" or normalized.startswith("MST ") or normalized.endswith(" MST"):
            return column
    # The official MIDC raw contract identifies its fourth column as the local
    # time field. Keep that documented fallback while always interpreting BMS
    # timestamps using fixed UTC-07, never DST-aware America/Denver.
    if len(columns) >= 4:
        return columns[3]
    raise ValueError("SRRL CSV has no MIDC local-time column")


def _parse_hhmm(value: Any) -> tuple[int, int, int]:
    if pd.isna(value):
        raise ValueError("SRRL CSV contains a missing local time")
    raw = str(value).strip()
    if ":" in raw:
        hour_text, minute_text = raw.split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    else:
        # Pandas may represent an integer-like time as 5 or 5.0. MIDC means
        # HHMM, so 5 is 00:05 rather than 05:00.
        compact = str(int(float(raw))).zfill(4)
        hour, minute = int(compact[:-2]), int(compact[-2:])
    day_roll = 0
    if hour == 24 and minute == 0:
        hour, day_roll = 0, 1
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid SRRL MIDC HHMM value: {value!r}")
    return hour, minute, day_roll


def _schema_for(columns: list[str], time_column: str) -> SrrlDailySchema:
    serialized = json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    return SrrlDailySchema(
        version=f"midc-bms-{digest[:12]}",
        sha256=digest,
        columns=tuple(columns),
        column_count=len(columns),
        time_column=time_column,
    )


def parse_srrl_daily_csv(
    payload: str | bytes,
    *,
    expected_day: date | datetime | str | None = None,
) -> tuple[pd.DataFrame, SrrlDailySchema]:
    """Parse one official MIDC/BMS day and convert fixed MST to UTC.

    MIDC occasionally emits rows wider than the header around instrumentation
    changes. Mirroring the official ecosystem's robust reader, only the number
    of fields declared by the header is parsed. Each distinct ordered header is
    assigned a deterministic daily schema version.
    """
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload.lstrip("\ufeff")
    header = pd.read_csv(io.StringIO(text), nrows=0)
    declared_columns = [str(column).strip() for column in header.columns]
    if not declared_columns:
        raise ValueError("SRRL CSV has an empty header")

    frame = pd.read_csv(
        io.StringIO(text),
        usecols=range(len(declared_columns)),
        na_values=[SRRL_MISSING_SENTINEL, str(int(SRRL_MISSING_SENTINEL))],
        keep_default_na=True,
    )
    frame.columns = declared_columns
    if frame.empty:
        raise ValueError("SRRL CSV contains no observations")

    year_column = _find_column(declared_columns, {"year"})
    doy_column = _find_column(declared_columns, {"doy", "day of year", "julian day"})
    time_column = _find_time_column(declared_columns)

    years = pd.to_numeric(frame[year_column], errors="raise").astype(int)
    day_of_year = pd.to_numeric(frame[doy_column], errors="raise").astype(int)
    local_dates = pd.to_datetime(
        years.astype(str) + day_of_year.map(lambda value: f"{value:03d}"),
        format="%Y%j",
        errors="raise",
    )
    time_parts = frame[time_column].map(_parse_hhmm)
    local_naive = pd.DatetimeIndex(
        [
            local_date + pd.Timedelta(days=roll, hours=hour, minutes=minute)
            for local_date, (hour, minute, roll) in zip(local_dates, time_parts, strict=True)
        ]
    )
    timestamp_utc = local_naive.tz_localize(SRRL_FIXED_MST).tz_convert("UTC")

    requested_day = _coerce_day(expected_day) if expected_day is not None else None
    if requested_day is not None:
        observed_days = set(local_naive.date)
        # A terminal 24:00 record belongs to the requested file but rolls to the
        # following instant. All other rows must originate on the requested day.
        allowed = {requested_day, requested_day + timedelta(days=1)}
        if not observed_days.issubset(allowed) or requested_day not in observed_days:
            raise ValueError(
                f"SRRL response day mismatch: expected {requested_day.isoformat()}, "
                f"observed {sorted(day.isoformat() for day in observed_days)}"
            )

    # Cover object columns that may contain the sentinel as text after a schema
    # change; numeric columns were already handled through na_values.
    frame = frame.replace({str(int(SRRL_MISSING_SENTINEL)): np.nan, SRRL_MISSING_SENTINEL: np.nan})
    frame.insert(0, "timestamp_utc", timestamp_utc)
    frame.insert(1, "source_timezone", "MST_FIXED_UTC_MINUS_07")
    frame.insert(2, "source_site", SRRL_SITE_ID)

    schema = _schema_for(declared_columns, time_column)
    frame.attrs["schema"] = schema.to_dict()
    frame.attrs["evidence_role"] = "observed"
    return frame, schema


def fetch_srrl_daily(
    day: date | datetime | str,
    *,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (20, 120),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch and parse one SRRL/BMS day with response-level provenance."""
    requested_day = _coerce_day(day)
    url = build_srrl_daily_url(requested_day)
    client: Any = session or requests
    response = client.get(
        url,
        headers={"User-Agent": "power-dc-vpp/0.1 official-source-adapter"},
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.content
    frame, schema = parse_srrl_daily_csv(content, expected_day=requested_day)
    provenance = {
        "source": "NLR MIDC SRRL Baseline Measurement System",
        "catalog_url": SRRL_CATALOG_URL,
        "site_home_url": SRRL_SITE_HOME_URL,
        "api_documentation_url": SRRL_API_DOCUMENTATION_URL,
        "field_dictionary_url": SRRL_FIELD_DICTIONARY_URL,
        "stable_url": url,
        "query_day": requested_day.isoformat(),
        "site": SRRL_SITE_ID,
        "timezone": "fixed MST (UTC-07:00)",
        "missing_sentinel": SRRL_MISSING_SENTINEL,
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": schema.to_dict(),
        "evidence_role": "observed",
    }
    return frame, provenance


def fetch_and_persist_srrl_daily(
    repository_root: Path,
    day: date | datetime | str,
    *,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (20, 120),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one official BMS day and atomically preserve CSV plus provenance."""
    requested_day = _coerce_day(day)
    url = build_srrl_daily_url(requested_day)
    client: Any = session or requests
    response = client.get(
        url,
        headers={"User-Agent": "power-dc-vpp/0.1 official-source-adapter"},
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.content
    frame, schema = parse_srrl_daily_csv(content, expected_day=requested_day)

    stem = f"BMS_{requested_day.strftime('%Y%m%d')}"
    raw_path = repository_root / "data/raw/srrl" / f"{stem}.csv"
    provenance_path = repository_root / "data/raw/srrl" / f"{stem}.provenance.json"
    provenance = {
        "source": "NLR MIDC SRRL Baseline Measurement System",
        "catalog_url": SRRL_CATALOG_URL,
        "site_home_url": SRRL_SITE_HOME_URL,
        "api_documentation_url": SRRL_API_DOCUMENTATION_URL,
        "field_dictionary_url": SRRL_FIELD_DICTIONARY_URL,
        "stable_url": url,
        "query_day": requested_day.isoformat(),
        "site": SRRL_SITE_ID,
        "timezone": "fixed MST (UTC-07:00)",
        "missing_sentinel": SRRL_MISSING_SENTINEL,
        "filename": raw_path.name,
        "local_path": raw_path.relative_to(repository_root).as_posix(),
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": schema.to_dict(),
        "evidence_role": "observed",
    }
    atomic_bytes(raw_path, content)
    atomic_json(provenance_path, provenance)
    return frame, provenance
