from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from dcvpp.core import atomic_json
from dcvpp.data.io_utils import atomic_bytes


NASA_POWER_HOURLY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_HOURLY_DOCUMENTATION_URL = "https://power.larc.nasa.gov/docs/services/api/temporal/hourly/"
NASA_POWER_TIME_STANDARD = "UTC"
NASA_POWER_DEFAULT_PARAMETERS = (
    "T2M",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
)
NASA_POWER_DEFAULT_FILL_VALUE = -999.0


@dataclass(frozen=True)
class NasaPowerSite:
    zone: str
    city: str
    latitude: float
    longitude: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SITES = (
    NasaPowerSite("WEST", "Buffalo", 42.8864, -78.8784),
    NasaPowerSite("CAPITL", "Albany", 42.6526, -73.7562),
    NasaPowerSite("N.Y.C.", "New York City", 40.7128, -74.0060),
    NasaPowerSite("LONGIL", "Long Island", 40.7891, -73.1350),
)

# These coordinates are deliberately frozen experiment inputs, not centroids
# inferred afresh on each run.
NYISO_POWER_SITES: Mapping[str, NasaPowerSite] = MappingProxyType({site.zone: site for site in _SITES})


def _coerce_day(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def build_nasa_power_hourly_url(
    zone: str,
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    parameters: Sequence[str] = NASA_POWER_DEFAULT_PARAMETERS,
    community: str = "RE",
) -> str:
    """Build a deterministic NASA POWER hourly point request in explicit UTC."""
    if zone not in NYISO_POWER_SITES:
        raise KeyError(f"unknown frozen NYISO site: {zone}")
    start_day, end_day = _coerce_day(start), _coerce_day(end)
    if end_day < start_day:
        raise ValueError("NASA POWER end date precedes start date")
    unique_parameters = tuple(dict.fromkeys(str(parameter).strip().upper() for parameter in parameters))
    if not unique_parameters or len(unique_parameters) > 15:
        raise ValueError("NASA POWER hourly requests require 1 to 15 unique parameters")
    if any(not parameter for parameter in unique_parameters):
        raise ValueError("NASA POWER parameter names cannot be empty")

    site = NYISO_POWER_SITES[zone]
    query = urlencode(
        {
            "parameters": ",".join(unique_parameters),
            "community": community,
            "longitude": f"{site.longitude:.4f}",
            "latitude": f"{site.latitude:.4f}",
            "start": start_day.strftime("%Y%m%d"),
            "end": end_day.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": NASA_POWER_TIME_STANDARD,
        },
        safe=",",
    )
    return f"{NASA_POWER_HOURLY_POINT_URL}?{query}"


def _load_json(payload: Mapping[str, Any] | str | bytes) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("NASA POWER response must be a JSON object")
    return value


def parse_nasa_power_hourly(
    payload: Mapping[str, Any] | str | bytes,
    *,
    zone: str,
    parameters: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse NASA POWER point JSON without converting UTC to a market clock."""
    if zone not in NYISO_POWER_SITES:
        raise KeyError(f"unknown frozen NYISO site: {zone}")
    document = _load_json(payload)
    properties = document.get("properties")
    parameter_data = properties.get("parameter") if isinstance(properties, Mapping) else None
    if not isinstance(parameter_data, Mapping) or not parameter_data:
        raise ValueError("NASA POWER response has no properties.parameter data")

    selected = tuple(parameters) if parameters is not None else tuple(parameter_data.keys())
    missing_parameters = [parameter for parameter in selected if parameter not in parameter_data]
    if missing_parameters:
        raise ValueError(f"NASA POWER response is missing parameters: {missing_parameters}")

    timestamp_keys: set[str] = set()
    for parameter in selected:
        series = parameter_data[parameter]
        if not isinstance(series, Mapping):
            raise ValueError(f"NASA POWER parameter {parameter} is not a time-value mapping")
        timestamp_keys.update(str(key) for key in series)
    if not timestamp_keys:
        raise ValueError("NASA POWER response contains no hourly values")
    ordered_keys = sorted(timestamp_keys)
    timestamp_utc = pd.to_datetime(ordered_keys, format="%Y%m%d%H", errors="raise", utc=True)

    header = document.get("header") if isinstance(document.get("header"), Mapping) else {}
    fill_value = float(header.get("fill_value", NASA_POWER_DEFAULT_FILL_VALUE))
    site = NYISO_POWER_SITES[zone]
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamp_utc,
            "zone": zone,
            "latitude": site.latitude,
            "longitude": site.longitude,
        }
    )
    for parameter in selected:
        values = pd.to_numeric(
            pd.Series([parameter_data[parameter].get(key, np.nan) for key in ordered_keys]),
            errors="coerce",
        )
        frame[str(parameter)] = values.mask(np.isclose(values, fill_value), np.nan)

    response_coordinates: list[float] | None = None
    geometry = document.get("geometry")
    if isinstance(geometry, Mapping) and isinstance(geometry.get("coordinates"), list):
        response_coordinates = geometry["coordinates"]
    metadata = {
        "source": "NASA POWER hourly point API",
        "documentation_url": NASA_POWER_HOURLY_DOCUMENTATION_URL,
        "zone": zone,
        "frozen_site": site.to_dict(),
        "response_coordinates": response_coordinates,
        "time_standard": NASA_POWER_TIME_STANDARD,
        "parameters": list(selected),
        "fill_value": fill_value,
        "row_count": len(frame),
        # POWER is a gridded satellite/assimilation product, not an on-site
        # meter. Label it reconstructed to keep that scientific boundary clear.
        "evidence_role": "reconstructed",
        "source_relationship": "third_party_referenced_public_data",
        "record_type": "referenced_gridded_reconstruction",
        "measurement_relationship": "not_measured_by_this_study",
    }
    frame.attrs.update(metadata)
    return frame, metadata


def fetch_nasa_power_hourly(
    zone: str,
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    parameters: Sequence[str] = NASA_POWER_DEFAULT_PARAMETERS,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (20, 180),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one frozen NYISO representative point and return provenance."""
    url = build_nasa_power_hourly_url(zone, start, end, parameters=parameters)
    client: Any = session or requests
    response = client.get(
        url,
        headers={"User-Agent": "power-dc-vpp/0.1 official-source-adapter"},
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.content
    frame, metadata = parse_nasa_power_hourly(content, zone=zone, parameters=parameters)
    metadata = {
        **metadata,
        "stable_url": url,
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    frame.attrs.update(metadata)
    return frame, metadata


def fetch_and_persist_nasa_power_hourly(
    repository_root: Path,
    zone: str,
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    parameters: Sequence[str] = NASA_POWER_DEFAULT_PARAMETERS,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (20, 180),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire a deterministic zone-period JSON and atomically preserve it.

    Deterministic filenames make this suitable for sequential annual
    acquisition of each frozen zone over the 2021--2025 experiment window.
    """
    start_day, end_day = _coerce_day(start), _coerce_day(end)
    url = build_nasa_power_hourly_url(zone, start_day, end_day, parameters=parameters)
    client: Any = session or requests
    response = client.get(
        url,
        headers={"User-Agent": "power-dc-vpp/0.1 official-source-adapter"},
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.content
    frame, metadata = parse_nasa_power_hourly(content, zone=zone, parameters=parameters)

    zone_slug = zone.replace(".", "").replace(" ", "_")
    stem = f"{zone_slug}_{start_day.strftime('%Y%m%d')}_{end_day.strftime('%Y%m%d')}"
    raw_path = repository_root / "data/raw/nasa_power" / f"{stem}.json"
    provenance_path = repository_root / "data/raw/nasa_power" / f"{stem}.provenance.json"
    provenance = {
        **metadata,
        "stable_url": url,
        "filename": raw_path.name,
        "local_path": raw_path.relative_to(repository_root).as_posix(),
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_bytes(raw_path, content)
    atomic_json(provenance_path, provenance)
    frame.attrs.update(provenance)
    return frame, provenance


def acquire_nasa_power_annual_window(
    repository_root: Path,
    *,
    years: Sequence[int] = tuple(range(2021, 2026)),
    zones: Sequence[str] = tuple(NYISO_POWER_SITES),
    parameters: Sequence[str] = NASA_POWER_DEFAULT_PARAMETERS,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Sequentially persist annual POWER files for frozen zones and years."""
    unknown = [zone for zone in zones if zone not in NYISO_POWER_SITES]
    if unknown:
        raise KeyError(f"unknown frozen NYISO sites: {unknown}")
    invalid_years = [year for year in years if year < 2001]
    if invalid_years:
        raise ValueError(f"NASA POWER hourly data begin in 2001; invalid years: {invalid_years}")

    owned_session = session is None
    client = session or requests.Session()
    records: list[dict[str, Any]] = []
    try:
        for zone in zones:
            for year in years:
                _, provenance = fetch_and_persist_nasa_power_hourly(
                    repository_root,
                    zone,
                    date(year, 1, 1),
                    date(year, 12, 31),
                    parameters=parameters,
                    session=client,
                )
                records.append(provenance)
    finally:
        if owned_session:
            client.close()
    return records
