"""Causal NOAA forecast adapters for the VPP-WorldMoE weather tokens.

The module deliberately keeps weather acquisition independent from the main
WorldMoE corpus.  It downloads only selected GRIB messages by HTTP byte range,
materialises only the four frozen NYISO representative sites, and preserves the
operational publication time separately from the model initialisation time.

The 2025 final-test boundary is sealed by default.  Development acquisition and
queries are limited to 2022--2024 unless a caller explicitly constructs a store
with ``allow_sealed_2025=True`` during the one-shot final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time as wall_time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import requests


UTC = timezone.utc
WEATHER_MANIFEST_SCHEMA = "dcvpp.vpp_world_moe.weather-manifest.v1"
WEATHER_STORE_SCHEMA = "dcvpp.vpp_world_moe.weather-store.v1"
WEATHER_FRAME_SCHEMA = "dcvpp.vpp_world_moe.weather-frame.v1"
WEATHER_ARTIFACT_CONTRACT_VERSION = 2
DEVELOPMENT_YEARS = frozenset((2022, 2023, 2024))
SEALED_FINAL_TEST_YEAR = 2025
_WEATHER_METADATA_LOCK = threading.RLock()


class WeatherDataError(RuntimeError):
    """Base class for weather acquisition and lineage failures."""


class WeatherCausalityError(WeatherDataError):
    """Raised when a request crosses a temporal evidence boundary."""


class WeatherRemoteError(WeatherDataError):
    """Raised when an official remote object violates its expected contract."""


class WeatherDecodeError(WeatherDataError):
    """Raised when a selected GRIB message cannot be decoded."""


def _utc_datetime(value: datetime | pd.Timestamp | str) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    elif isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError(f"expected an aware datetime, received {type(value)!r}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise WeatherCausalityError("weather timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso_utc(value: datetime | pd.Timestamp | str) -> str:
    return _utc_datetime(value).isoformat().replace("+00:00", "Z")


def _parse_http_datetime(value: str | None) -> datetime:
    if not value:
        raise WeatherRemoteError("official object response has no Last-Modified header")
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed_years(values: Iterable[datetime]) -> set[int]:
    return {value.year for value in values if value.year >= SEALED_FINAL_TEST_YEAR}


def validate_development_times(*values: datetime) -> None:
    utc_values = tuple(_utc_datetime(value) for value in values)
    sealed = _sealed_years(utc_values)
    if sealed:
        raise WeatherCausalityError(
            f"weather development access cannot open sealed year(s): {sorted(sealed)}"
        )
    invalid = sorted({value.year for value in utc_values} - DEVELOPMENT_YEARS)
    if invalid:
        raise WeatherCausalityError(
            "weather development materialisation is restricted to 2022--2024; "
            f"received {invalid}"
        )


def _validate_development_forecast_times(
    run_init: datetime,
    valid_time: datetime,
) -> None:
    """Validate a forecast pair, including the late-2021 support cycle.

    The 18Z HRRR run on 2021-12-31 is a causal input for the first 2022
    operating day.  It is source support, not a 2021 training example.  No
    other pre-period or post-period valid time is admitted.
    """

    run = _utc_datetime(run_init)
    valid = _utc_datetime(valid_time)
    if (
        run.year == 2021
        and run >= datetime(2021, 12, 29, tzinfo=UTC)
        and valid.year == 2022
    ):
        return
    validate_development_times(run, valid)


@dataclass(frozen=True)
class WeatherSite:
    zone: str
    city: str
    latitude: float
    longitude: float


_SITES = (
    WeatherSite("WEST", "Buffalo", 42.8864, -78.8784),
    WeatherSite("CAPITL", "Albany", 42.6526, -73.7562),
    WeatherSite("N.Y.C.", "New York City", 40.7128, -74.0060),
    WeatherSite("LONGIL", "Long Island", 40.7891, -73.1350),
)

# Frozen experiment sites.  They are representative data-centre locations,
# not silently recomputed centroids of the NYISO load-zone polygons.
NYISO_WEATHER_SITES: Mapping[str, WeatherSite] = MappingProxyType(
    {site.zone: site for site in _SITES}
)


@dataclass(frozen=True)
class GribVariableSpec:
    feature: str
    parameter: str
    unit: str
    hrrr_level: str | None
    nbm_level: str | None

    def level_for(self, source: str) -> str | None:
        if source == "hrrr":
            return self.hrrr_level
        if source == "nbm":
            return self.nbm_level
        raise KeyError(f"unknown weather source: {source}")


WEATHER_VARIABLES: Mapping[str, GribVariableSpec] = MappingProxyType(
    {
        "air_temperature_2m_k": GribVariableSpec(
            "air_temperature_2m_k", "TMP", "K", "2 m above ground", "2 m above ground"
        ),
        "dew_point_2m_k": GribVariableSpec(
            "dew_point_2m_k", "DPT", "K", "2 m above ground", "2 m above ground"
        ),
        "relative_humidity_2m_pct": GribVariableSpec(
            "relative_humidity_2m_pct", "RH", "%", "2 m above ground", "2 m above ground"
        ),
        "wind_u_10m_mps": GribVariableSpec(
            "wind_u_10m_mps", "UGRD", "m/s", "10 m above ground", None
        ),
        "wind_v_10m_mps": GribVariableSpec(
            "wind_v_10m_mps", "VGRD", "m/s", "10 m above ground", None
        ),
        "wind_speed_10m_mps": GribVariableSpec(
            "wind_speed_10m_mps", "WIND", "m/s", None, "10 m above ground"
        ),
        "wind_direction_10m_deg": GribVariableSpec(
            "wind_direction_10m_deg", "WDIR", "degree", None, "10 m above ground"
        ),
        "cloud_cover_total_pct": GribVariableSpec(
            "cloud_cover_total_pct", "TCDC", "%", "entire atmosphere", "surface"
        ),
        "cloud_cover_low_pct": GribVariableSpec(
            "cloud_cover_low_pct", "LCDC", "%", "low cloud layer", None
        ),
        "cloud_cover_mid_pct": GribVariableSpec(
            "cloud_cover_mid_pct", "MCDC", "%", "middle cloud layer", None
        ),
        "cloud_cover_high_pct": GribVariableSpec(
            "cloud_cover_high_pct", "HCDC", "%", "high cloud layer", None
        ),
        "downward_shortwave_wm2": GribVariableSpec(
            "downward_shortwave_wm2", "DSWRF", "W/m^2", "surface", "surface"
        ),
        "visible_beam_downward_wm2": GribVariableSpec(
            "visible_beam_downward_wm2", "VBDSF", "W/m^2", "surface", None
        ),
        "visible_diffuse_downward_wm2": GribVariableSpec(
            "visible_diffuse_downward_wm2", "VDDSF", "W/m^2", "surface", None
        ),
    }
)

HRRR_DEFAULT_FEATURES = (
    "air_temperature_2m_k",
    "dew_point_2m_k",
    "relative_humidity_2m_pct",
    "wind_u_10m_mps",
    "wind_v_10m_mps",
    "cloud_cover_total_pct",
    "cloud_cover_low_pct",
    "cloud_cover_mid_pct",
    "cloud_cover_high_pct",
    "downward_shortwave_wm2",
    "visible_beam_downward_wm2",
    "visible_diffuse_downward_wm2",
)
NBM_DEFAULT_FEATURES = (
    "air_temperature_2m_k",
    "dew_point_2m_k",
    "relative_humidity_2m_pct",
    "wind_speed_10m_mps",
    "wind_direction_10m_deg",
    "cloud_cover_total_pct",
    "downward_shortwave_wm2",
)

HRRR_ZARR_DEFAULT_FEATURES = (
    "air_temperature_2m_k",
    "relative_humidity_2m_pct",
    "wind_u_10m_mps",
    "wind_v_10m_mps",
    "cloud_cover_total_pct",
    "downward_shortwave_wm2",
)


@dataclass(frozen=True)
class HrrrZarrVariable:
    feature: str
    level: str
    parameter: str
    unit: str
    temporal_interpolation: str

    @property
    def array_relative_path(self) -> str:
        return f"{self.level}/{self.parameter}/{self.level}/{self.parameter}"


HRRR_ZARR_VARIABLES: Mapping[str, HrrrZarrVariable] = MappingProxyType(
    {
        "air_temperature_2m_k": HrrrZarrVariable(
            "air_temperature_2m_k", "2m_above_ground", "TMP", "K", "linear"
        ),
        "relative_humidity_2m_pct": HrrrZarrVariable(
            "relative_humidity_2m_pct", "2m_above_ground", "RH", "%", "linear"
        ),
        "wind_u_10m_mps": HrrrZarrVariable(
            "wind_u_10m_mps", "10m_above_ground", "UGRD", "m/s", "linear"
        ),
        "wind_v_10m_mps": HrrrZarrVariable(
            "wind_v_10m_mps", "10m_above_ground", "VGRD", "m/s", "linear"
        ),
        "cloud_cover_total_pct": HrrrZarrVariable(
            "cloud_cover_total_pct", "entire_atmosphere", "TCDC", "%", "linear"
        ),
        # HRRR wrfsfc encodes DSWRF as an instantaneous valid-time field.
        "downward_shortwave_wm2": HrrrZarrVariable(
            "downward_shortwave_wm2", "surface", "DSWRF", "W/m^2", "linear"
        ),
    }
)


@dataclass(frozen=True)
class GribIndexEntry:
    record_number: int
    start_byte: int
    end_byte: int
    reference_token: str
    parameter: str
    level: str
    forecast_descriptor: str
    qualifiers: tuple[str, ...] = ()

    @property
    def byte_count(self) -> int:
        return self.end_byte - self.start_byte + 1

    @property
    def is_deterministic_primary(self) -> bool:
        text = " ".join(self.qualifiers).casefold()
        blocked = ("ens std dev", "prob", "% level", "percentile", "unknown derived")
        return not any(token in text for token in blocked)


_INDEX_REFERENCE_RE = re.compile(r"^d=(\d{10})$")


def parse_grib_index(text: str, *, object_size: int) -> tuple[GribIndexEntry, ...]:
    """Parse a NOAA ``.idx`` sidecar and close every message byte range."""

    if object_size <= 0:
        raise WeatherRemoteError("GRIB object size must be positive")
    partial: list[tuple[int, int, str, str, str, str, tuple[str, ...]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 6:
            raise WeatherRemoteError(f"malformed NOAA GRIB index line: {line!r}")
        try:
            number = int(parts[0])
            start = int(parts[1])
        except ValueError as exc:
            raise WeatherRemoteError(f"non-numeric NOAA GRIB index offset: {line!r}") from exc
        reference, parameter, level, descriptor = parts[2:6]
        qualifiers = tuple(part for part in parts[6:] if part)
        if not _INDEX_REFERENCE_RE.match(reference):
            raise WeatherRemoteError(f"unexpected NOAA GRIB reference token: {reference!r}")
        partial.append((number, start, reference, parameter, level, descriptor, qualifiers))

    if not partial:
        raise WeatherRemoteError("NOAA GRIB index is empty")
    record_numbers = [value[0] for value in partial]
    if record_numbers != list(range(1, len(record_numbers) + 1)):
        raise WeatherRemoteError(
            "NOAA GRIB record numbers must preserve their original sequential order"
        )
    offsets = [value[1] for value in partial]
    if offsets[0] != 0 or len(offsets) != len(set(offsets)):
        raise WeatherRemoteError("NOAA GRIB offsets must be unique and start at byte zero")
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        raise WeatherRemoteError("NOAA GRIB offsets are not strictly increasing")
    if offsets[-1] >= object_size:
        raise WeatherRemoteError("NOAA GRIB index extends beyond the remote object")

    entries: list[GribIndexEntry] = []
    for index, value in enumerate(partial):
        end = offsets[index + 1] - 1 if index + 1 < len(offsets) else object_size - 1
        entries.append(
            GribIndexEntry(
                record_number=value[0],
                start_byte=value[1],
                end_byte=end,
                reference_token=value[2],
                parameter=value[3],
                level=value[4],
                forecast_descriptor=value[5],
                qualifiers=value[6],
            )
        )
    return tuple(entries)


def select_grib_entries(
    entries: Sequence[GribIndexEntry],
    *,
    source: str,
    features: Sequence[str],
) -> dict[str, GribIndexEntry]:
    selected: dict[str, GribIndexEntry] = {}
    for feature in tuple(dict.fromkeys(features)):
        try:
            specification = WEATHER_VARIABLES[feature]
        except KeyError as exc:
            raise KeyError(f"unknown weather feature: {feature}") from exc
        level = specification.level_for(source)
        if level is None:
            raise WeatherDataError(f"{feature} is not available from {source}")
        matches = [
            entry
            for entry in entries
            if entry.parameter == specification.parameter
            and entry.level == level
            and entry.is_deterministic_primary
        ]
        if len(matches) != 1:
            details = [(entry.parameter, entry.level, entry.qualifiers) for entry in matches]
            raise WeatherRemoteError(
                f"expected exactly one {source}:{feature} GRIB message, found {details}"
            )
        selected[feature] = matches[0]
    return selected


@dataclass(frozen=True)
class RemoteObjectMetadata:
    url: str
    byte_count: int
    etag: str
    last_modified_utc: datetime


@dataclass(frozen=True)
class WeatherSourceAdapter:
    source: str
    base_url: str

    def validate_forecast(self, run_init: datetime, forecast_hour: int) -> None:
        run_init = _utc_datetime(run_init)
        _validate_development_forecast_times(
            run_init, run_init + timedelta(hours=forecast_hour)
        )
        if run_init.minute or run_init.second or run_init.microsecond:
            raise WeatherCausalityError("NOAA forecast cycles must start on an exact UTC hour")
        if forecast_hour < 0:
            raise ValueError("forecast_hour cannot be negative")
        if self.source == "hrrr":
            if forecast_hour > 48:
                raise ValueError("HRRR forecast_hour cannot exceed 48")
            if forecast_hour > 18 and run_init.hour not in (0, 6, 12, 18):
                raise WeatherCausalityError(
                    "HRRR horizons above 18h require a 00/06/12/18 UTC extended cycle"
                )
        elif self.source == "nbm":
            if forecast_hour < 1 or forecast_hour > 264:
                raise ValueError("NBM core forecast_hour must be in [1, 264]")
        else:
            raise KeyError(f"unknown weather source: {self.source}")

    def source_version(self, run_init: datetime) -> str:
        run_init = _utc_datetime(run_init)
        if self.source == "hrrr":
            return "HRRRv4"
        v41 = datetime(2023, 1, 17, 12, tzinfo=UTC)
        v42 = datetime(2024, 5, 15, 12, tzinfo=UTC)
        if run_init < v41:
            return "NBMv4.0"
        if run_init < v42:
            return "NBMv4.1"
        return "NBMv4.2"

    def object_key(self, run_init: datetime, forecast_hour: int) -> str:
        run_init = _utc_datetime(run_init)
        self.validate_forecast(run_init, forecast_hour)
        day = run_init.strftime("%Y%m%d")
        cycle = run_init.strftime("%H")
        if self.source == "hrrr":
            return f"hrrr.{day}/conus/hrrr.t{cycle}z.wrfsfcf{forecast_hour:02d}.grib2"
        return (
            f"blend.{day}/{cycle}/core/"
            f"blend.t{cycle}z.core.f{forecast_hour:03d}.co.grib2"
        )

    def object_url(self, run_init: datetime, forecast_hour: int) -> str:
        return f"{self.base_url.rstrip('/')}/{self.object_key(run_init, forecast_hour)}"

    def index_url(self, run_init: datetime, forecast_hour: int) -> str:
        return f"{self.object_url(run_init, forecast_hour)}.idx"


HRRR_ADAPTER = WeatherSourceAdapter(
    source="hrrr", base_url="https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
)
NBM_ADAPTER = WeatherSourceAdapter(
    source="nbm", base_url="https://noaa-nbm-grib2-pds.s3.amazonaws.com"
)
WEATHER_SOURCES: Mapping[str, WeatherSourceAdapter] = MappingProxyType(
    {"hrrr": HRRR_ADAPTER, "nbm": NBM_ADAPTER}
)


class WeatherHTTPClient:
    """Small HTTP client with strict range-response validation."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: tuple[int, int] = (20, 180),
        max_attempts: int = 4,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_attempts = int(max_attempts)
        self.backoff_seconds = float(backoff_seconds)
        if self.max_attempts < 1 or self.backoff_seconds < 0.0:
            raise ValueError("weather HTTP retry settings are invalid")
        self.headers = {"User-Agent": "power-dc-vpp/0.1 causal-noaa-weather-adapter"}

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        retryable_status = {408, 425, 429, 500, 502, 503, 504}
        for attempt in range(self.max_attempts):
            try:
                response = getattr(self.session, method)(url, **kwargs)
                if (
                    response.status_code in retryable_status
                    and attempt + 1 < self.max_attempts
                ):
                    wall_time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                return response
            except (requests.ConnectionError, requests.Timeout):
                if attempt + 1 >= self.max_attempts:
                    raise
                wall_time.sleep(self.backoff_seconds * (2**attempt))
        raise AssertionError("unreachable weather HTTP retry state")

    def metadata(self, url: str) -> RemoteObjectMetadata:
        response = self._request(
            "head",
            url,
            headers=self.headers,
            timeout=self.timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length is None:
            raise WeatherRemoteError(f"remote object has no Content-Length: {url}")
        etag = str(response.headers.get("ETag", "")).strip('"')
        if not etag:
            raise WeatherRemoteError(f"remote object has no ETag: {url}")
        return RemoteObjectMetadata(
            url=url,
            byte_count=int(length),
            etag=etag,
            last_modified_utc=_parse_http_datetime(response.headers.get("Last-Modified")),
        )

    def get_bytes(self, url: str) -> bytes:
        response = self._request(
            "get", url, headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()
        return bytes(response.content)

    def get_object(self, metadata: RemoteObjectMetadata) -> bytes:
        response = self._request(
            "get",
            metadata.url,
            headers={**self.headers, "If-Match": f'"{metadata.etag}"'},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = bytes(response.content)
        if response.status_code != 200 or len(payload) != metadata.byte_count:
            raise WeatherRemoteError(
                f"full-object response mismatch for {metadata.url}: "
                f"HTTP {response.status_code}, {len(payload)}/{metadata.byte_count} bytes"
            )
        return payload

    def get_range(
        self,
        metadata: RemoteObjectMetadata,
        *,
        start_byte: int,
        end_byte: int,
    ) -> bytes:
        if start_byte < 0 or end_byte < start_byte or end_byte >= metadata.byte_count:
            raise ValueError("invalid HTTP byte range")
        headers = {
            **self.headers,
            "Range": f"bytes={start_byte}-{end_byte}",
            "If-Match": f'"{metadata.etag}"',
        }
        response = self._request(
            "get", metadata.url, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        expected = end_byte - start_byte + 1
        payload = bytes(response.content)
        if response.status_code != 206:
            raise WeatherRemoteError(
                f"server ignored required byte range for {metadata.url}: HTTP {response.status_code}"
            )
        if len(payload) != expected:
            raise WeatherRemoteError(
                f"range length mismatch for {metadata.url}: expected {expected}, got {len(payload)}"
            )
        content_range = str(response.headers.get("Content-Range", ""))
        expected_range = f"bytes {start_byte}-{end_byte}/{metadata.byte_count}"
        if content_range != expected_range:
            raise WeatherRemoteError(
                f"unexpected Content-Range {content_range!r}; expected {expected_range!r}"
            )
        if not payload.startswith(b"GRIB") or not payload.endswith(b"7777"):
            raise WeatherRemoteError("selected NOAA byte range is not one complete GRIB message")
        return payload


@dataclass(frozen=True)
class WeatherFragmentRecord:
    record_id: str
    source: str
    source_version: str
    feature: str
    unit: str
    grib_parameter: str
    grib_level: str
    grib_forecast_descriptor: str
    run_init_utc: str
    forecast_hour: int
    valid_time_utc: str
    visible_at_utc: str
    remote_url: str
    index_url: str
    remote_etag: str
    remote_object_bytes: int
    byte_start: int
    byte_end: int
    byte_count: int
    fragment_sha256: str
    local_fragment: str
    index_sha256: str
    retrieved_at_utc: str
    sites: tuple[str, ...]
    index_etag: str = ""
    index_last_modified_utc: str = ""
    index_byte_count: int = 0
    visibility_rule: str = "max(data_last_modified,index_last_modified)"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sites"] = list(self.sites)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WeatherFragmentRecord":
        values = dict(payload)
        values["sites"] = tuple(values["sites"])
        return cls(**values)


class WeatherManifest:
    """Immutable raw-fragment records plus one atomically replaced index."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.root = self.cache_root / "manifest"
        self.records_root = self.root / "records"
        self.index_path = self.root / "index.json"

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "schema": WEATHER_MANIFEST_SCHEMA,
                "opened_final_test_2025": False,
                "record_count": 0,
            }
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("schema") != WEATHER_MANIFEST_SCHEMA:
            raise WeatherDataError("unexpected weather manifest schema")
        if payload.get("opened_final_test_2025") is not False:
            raise WeatherCausalityError("development weather manifest opened sealed 2025")
        return payload

    def record_path(self, record_id: str) -> Path:
        return self.records_root / f"{record_id}.json"

    def get(self, record_id: str) -> WeatherFragmentRecord | None:
        path = self.record_path(record_id)
        if not path.exists():
            return None
        return WeatherFragmentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def add(self, record: WeatherFragmentRecord) -> Path:
        with _WEATHER_METADATA_LOCK:
            return self._add_unlocked(record)

    def _add_unlocked(self, record: WeatherFragmentRecord) -> Path:
        run_init = _utc_datetime(record.run_init_utc)
        valid_time = _utc_datetime(record.valid_time_utc)
        _validate_development_forecast_times(run_init, valid_time)
        record_path = self.record_path(record.record_id)
        created = not record_path.exists()
        if record_path.exists():
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing != record.to_dict():
                raise WeatherDataError(f"weather record ID collision: {record.record_id}")
        else:
            _atomic_json(record_path, record.to_dict())
        index = self._load_index()
        if created or not self.index_path.exists():
            _atomic_json(
                self.index_path,
                {
                    "schema": WEATHER_MANIFEST_SCHEMA,
                    "opened_final_test_2025": False,
                    "record_count": int(index.get("record_count", 0)) + int(created),
                },
            )
        return record_path

    def records(self) -> tuple[WeatherFragmentRecord, ...]:
        records: list[WeatherFragmentRecord] = []
        for path in sorted(self.records_root.glob("*.json")):
            records.append(
                WeatherFragmentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            )
        return tuple(records)


def _record_identity(payload: Mapping[str, Any]) -> str:
    serialised = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialised).hexdigest()


class GribPointDecoder(Protocol):
    def decode(
        self,
        fragment_path: Path,
        *,
        sites: Sequence[WeatherSite],
    ) -> pd.DataFrame:
        """Return one row per site with value and selected grid metadata."""


class EccodesPointDecoder:
    """Decode a single GRIB message and retain only the four requested points."""

    def decode(
        self,
        fragment_path: Path,
        *,
        sites: Sequence[WeatherSite],
    ) -> pd.DataFrame:
        try:
            from eccodes import (  # type: ignore[import-not-found]
                codes_get,
                codes_get_array,
                codes_grib_new_from_file,
                codes_release,
            )
        except ImportError as exc:
            raise WeatherDecodeError(
                "materialisation requires the optional eccodes Python bindings; "
                "probe-only acquisition remains available without them"
            ) from exc

        with fragment_path.open("rb") as handle:
            message = codes_grib_new_from_file(handle)
            if message is None:
                raise WeatherDecodeError(f"no GRIB message in {fragment_path}")
            try:
                latitudes = np.asarray(codes_get_array(message, "latitudes"), dtype=np.float64)
                longitudes = np.asarray(codes_get_array(message, "longitudes"), dtype=np.float64)
                values = np.asarray(codes_get_array(message, "values"), dtype=np.float64)
                if not (latitudes.shape == longitudes.shape == values.shape):
                    raise WeatherDecodeError("GRIB latitude/longitude/value arrays disagree")
                try:
                    nx = int(codes_get(message, "Nx"))
                except Exception:
                    nx = 0
                try:
                    decoded_unit = str(codes_get(message, "units"))
                except Exception:
                    decoded_unit = ""
            finally:
                codes_release(message)

        rows: list[dict[str, Any]] = []
        for site in sites:
            longitude_delta = ((longitudes - site.longitude + 180.0) % 360.0) - 180.0
            score = np.square(latitudes - site.latitude) + np.square(
                longitude_delta * np.cos(np.deg2rad(site.latitude))
            )
            flat_index = int(np.nanargmin(score))
            rows.append(
                {
                    "zone": site.zone,
                    "city": site.city,
                    "site_latitude": site.latitude,
                    "site_longitude": site.longitude,
                    "grid_latitude": float(latitudes[flat_index]),
                    "grid_longitude": float(
                        ((longitudes[flat_index] + 180.0) % 360.0) - 180.0
                    ),
                    "grid_index_flat": flat_index,
                    "grid_x": flat_index % nx if nx > 0 else -1,
                    "grid_y": flat_index // nx if nx > 0 else -1,
                    "value": float(values[flat_index]),
                    "decoded_unit": decoded_unit,
                }
            )
        return pd.DataFrame.from_records(rows)


class WeatherRangeMaterializer:
    """Acquire selected NOAA messages and optionally decode four site points."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        client: WeatherHTTPClient | None = None,
        decoder: GribPointDecoder | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.client = client or WeatherHTTPClient()
        self.decoder = decoder
        self.manifest = WeatherManifest(self.cache_root)
        self._verified_cache: set[str] = set()

    def _fragment_path(
        self,
        source: str,
        run_init: datetime,
        forecast_hour: int,
        feature: str,
        entry: GribIndexEntry,
    ) -> Path:
        return (
            self.cache_root
            / "raw_ranges"
            / source
            / run_init.strftime("%Y%m%d")
            / run_init.strftime("%H")
            / f"f{forecast_hour:03d}"
            / f"{feature}.{entry.start_byte}-{entry.end_byte}.grib2"
        )

    def acquire(
        self,
        *,
        source: str,
        run_init: datetime,
        forecast_hour: int,
        features: Sequence[str],
        sites: Sequence[WeatherSite] = _SITES,
    ) -> tuple[WeatherFragmentRecord, ...]:
        run_init = _utc_datetime(run_init)
        adapter = WEATHER_SOURCES[source]
        adapter.validate_forecast(run_init, forecast_hour)
        if tuple(site.zone for site in sites) != tuple(NYISO_WEATHER_SITES):
            unknown = set(site.zone for site in sites) - set(NYISO_WEATHER_SITES)
            if unknown:
                raise KeyError(f"unknown weather sites: {sorted(unknown)}")

        object_url = adapter.object_url(run_init, forecast_hour)
        index_url = adapter.index_url(run_init, forecast_hour)
        metadata = self.client.metadata(object_url)
        index_metadata = self.client.metadata(index_url)
        index_payload = self.client.get_object(index_metadata)
        index_text = index_payload.decode("utf-8")
        entries = parse_grib_index(index_text, object_size=metadata.byte_count)
        reference = run_init.strftime("d=%Y%m%d%H")
        if any(entry.reference_token != reference for entry in entries):
            raise WeatherRemoteError(
                f"GRIB index reference time does not match requested cycle {reference}"
            )
        selected = select_grib_entries(entries, source=source, features=features)
        valid_time = run_init + timedelta(hours=forecast_hour)
        visible_at = max(
            metadata.last_modified_utc, index_metadata.last_modified_utc
        )
        retrieved_at = datetime.now(UTC)
        index_sha256 = _sha256_bytes(index_payload)
        index_cache = (
            self.cache_root
            / "indices"
            / source
            / run_init.strftime("%Y%m%d")
            / run_init.strftime("%H")
            / f"f{forecast_hour:03d}.idx"
        )
        _atomic_bytes(index_cache, index_payload)

        records: list[WeatherFragmentRecord] = []
        for feature, entry in selected.items():
            specification = WEATHER_VARIABLES[feature]
            fragment_path = self._fragment_path(
                source, run_init, forecast_hour, feature, entry
            )
            identity_payload = {
                "source": source,
                "source_version": adapter.source_version(run_init),
                "feature": feature,
                "run_init_utc": _iso_utc(run_init),
                "forecast_hour": forecast_hour,
                "valid_time_utc": _iso_utc(valid_time),
                "visible_at_utc": _iso_utc(visible_at),
                "remote_url": object_url,
                "remote_etag": metadata.etag,
                "index_etag": index_metadata.etag,
                "byte_start": entry.start_byte,
                "byte_end": entry.end_byte,
            }
            record_id = _record_identity(identity_payload)
            existing_record = self.manifest.get(record_id)
            # Reuse by expected size without repeatedly hashing a known cache.
            # The initial download hash remains pinned in the immutable manifest.
            if fragment_path.exists() and fragment_path.stat().st_size == entry.byte_count:
                if existing_record is not None:
                    if record_id not in self._verified_cache:
                        if _sha256_file(fragment_path) != existing_record.fragment_sha256:
                            raise WeatherDataError(
                                "cached GRIB range failed its pinned SHA-256"
                            )
                        self._verified_cache.add(record_id)
                    records.append(existing_record)
                    continue

            fragment = self.client.get_range(
                metadata,
                start_byte=entry.start_byte,
                end_byte=entry.end_byte,
            )
            _atomic_bytes(fragment_path, fragment)
            fragment_sha256 = _sha256_bytes(fragment)
            if existing_record is not None:
                if fragment_sha256 != existing_record.fragment_sha256:
                    raise WeatherRemoteError(
                        "re-downloaded GRIB range differs from its pinned manifest hash"
                    )
                records.append(existing_record)
                self._verified_cache.add(record_id)
                continue
            record = WeatherFragmentRecord(
                record_id=record_id,
                source=source,
                source_version=adapter.source_version(run_init),
                feature=feature,
                unit=specification.unit,
                grib_parameter=entry.parameter,
                grib_level=entry.level,
                grib_forecast_descriptor=entry.forecast_descriptor,
                run_init_utc=_iso_utc(run_init),
                forecast_hour=forecast_hour,
                valid_time_utc=_iso_utc(valid_time),
                visible_at_utc=_iso_utc(visible_at),
                remote_url=object_url,
                index_url=index_url,
                remote_etag=metadata.etag,
                remote_object_bytes=metadata.byte_count,
                byte_start=entry.start_byte,
                byte_end=entry.end_byte,
                byte_count=entry.byte_count,
                fragment_sha256=fragment_sha256,
                local_fragment=str(fragment_path),
                index_sha256=index_sha256,
                retrieved_at_utc=_iso_utc(retrieved_at),
                sites=tuple(site.zone for site in sites),
                index_etag=index_metadata.etag,
                index_last_modified_utc=_iso_utc(index_metadata.last_modified_utc),
                index_byte_count=index_metadata.byte_count,
            )
            self.manifest.add(record)
            self._verified_cache.add(record_id)
            records.append(record)
        return tuple(records)

    def materialize(
        self,
        *,
        store: "WorldWeatherStore",
        source: str,
        run_init: datetime,
        forecast_hour: int,
        features: Sequence[str],
        sites: Sequence[WeatherSite] = _SITES,
    ) -> Path:
        if self.decoder is None:
            raise WeatherDecodeError("a GRIB point decoder is required for materialisation")
        records = self.acquire(
            source=source,
            run_init=run_init,
            forecast_hour=forecast_hour,
            features=features,
            sites=sites,
        )
        return store.put(self.decode_records(records, sites=sites))

    def decode_records(
        self,
        records: Sequence[WeatherFragmentRecord],
        *,
        sites: Sequence[WeatherSite] = _SITES,
    ) -> pd.DataFrame:
        """Decode acquired messages without committing a store artifact."""

        if self.decoder is None:
            raise WeatherDecodeError("a GRIB point decoder is required for materialisation")
        frames: list[pd.DataFrame] = []
        for record in records:
            points = self.decoder.decode(Path(record.local_fragment), sites=sites).copy()
            points["schema"] = WEATHER_FRAME_SCHEMA
            points["source"] = record.source
            points["source_version"] = record.source_version
            points["feature"] = record.feature
            points["unit"] = record.unit
            points["grib_parameter"] = record.grib_parameter
            points["grib_level"] = record.grib_level
            points["grib_forecast_descriptor"] = record.grib_forecast_descriptor
            points["run_init_utc"] = pd.Timestamp(record.run_init_utc)
            points["forecast_hour"] = record.forecast_hour
            points["valid_time_utc"] = pd.Timestamp(record.valid_time_utc)
            points["visible_at_utc"] = pd.Timestamp(record.visible_at_utc)
            points["fragment_record_id"] = record.record_id
            index_object_id = _record_identity(
                {
                    "transport": "grib2_index",
                    "url": record.index_url,
                    "etag": record.index_etag,
                    "sha256": record.index_sha256,
                }
            )
            points["source_object_record_ids"] = json.dumps(
                sorted({record.record_id, index_object_id}),
                separators=(",", ":"),
            )
            points["transport"] = "grib2_idx_byte_range"
            points["native_resolution_minutes"] = 60
            points["temporal_interpolation"] = "causal_hold"
            frames.append(points)
        if not frames:
            raise WeatherDecodeError("no GRIB weather records were provided for decoding")
        return pd.concat(frames, ignore_index=True)


HRRR_ZARR_BASE_URL = "https://hrrrzarr.s3.amazonaws.com"
HRRR_GRID_X0_M = -2697520.1425219304
HRRR_GRID_Y0_M = -1587306.1525566636
HRRR_GRID_SPACING_M = 3000.0
HRRR_GRID_NX = 1799
HRRR_GRID_NY = 1059
HRRR_EARTH_RADIUS_M = 6371229.0
HRRR_PROJECTION_LAT_DEG = 38.5
HRRR_PROJECTION_LON_DEG = -97.5
HRRR_ZARR_CHUNK_XY = 150
HRRR_ZARR_MANIFEST_SCHEMA = "dcvpp.vpp_world_moe.hrrr-zarr-manifest.v1"


def _hrrr_forward_lambert(longitude: float, latitude: float) -> tuple[float, float]:
    radius = HRRR_EARTH_RADIUS_M
    phi0 = math.radians(HRRR_PROJECTION_LAT_DEG)
    phi1 = phi0
    lambda0 = math.radians(HRRR_PROJECTION_LON_DEG)
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    cone = math.sin(phi1)
    factor = (
        math.cos(phi1) * math.tan(math.pi / 4.0 + phi1 / 2.0) ** cone / cone
    )
    rho0 = radius * factor / math.tan(math.pi / 4.0 + phi0 / 2.0) ** cone
    rho = radius * factor / math.tan(math.pi / 4.0 + phi / 2.0) ** cone
    theta = cone * (lam - lambda0)
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


def _hrrr_inverse_lambert(x: float, y: float) -> tuple[float, float]:
    radius = HRRR_EARTH_RADIUS_M
    phi0 = math.radians(HRRR_PROJECTION_LAT_DEG)
    phi1 = phi0
    lambda0 = math.radians(HRRR_PROJECTION_LON_DEG)
    cone = math.sin(phi1)
    factor = (
        math.cos(phi1) * math.tan(math.pi / 4.0 + phi1 / 2.0) ** cone / cone
    )
    rho0 = radius * factor / math.tan(math.pi / 4.0 + phi0 / 2.0) ** cone
    rho = math.hypot(x, rho0 - y)
    theta = math.atan2(x, rho0 - y)
    phi = 2.0 * math.atan((radius * factor / rho) ** (1.0 / cone)) - math.pi / 2.0
    lam = lambda0 + theta / cone
    longitude = ((math.degrees(lam) + 180.0) % 360.0) - 180.0
    return math.degrees(phi), longitude


def _haversine_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    phi_a, phi_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_phi = phi_b - phi_a
    delta_lambda = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * HRRR_EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


@dataclass(frozen=True)
class HrrrGridPoint:
    zone: str
    city: str
    site_latitude: float
    site_longitude: float
    grid_latitude: float
    grid_longitude: float
    grid_x: int
    grid_y: int
    chunk_x: int
    chunk_y: int
    in_chunk_x: int
    in_chunk_y: int
    distance_m: float

    @property
    def forecast_chunk_id(self) -> str:
        return f"0.{self.chunk_y}.{self.chunk_x}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hrrr_grid_point(site: WeatherSite) -> HrrrGridPoint:
    projected_x, projected_y = _hrrr_forward_lambert(site.longitude, site.latitude)
    grid_x = int(round((projected_x - HRRR_GRID_X0_M) / HRRR_GRID_SPACING_M))
    grid_y = int(round((projected_y - HRRR_GRID_Y0_M) / HRRR_GRID_SPACING_M))
    if not (0 <= grid_x < HRRR_GRID_NX and 0 <= grid_y < HRRR_GRID_NY):
        raise WeatherDataError(f"site {site.zone} lies outside the HRRR CONUS grid")
    center_x = HRRR_GRID_X0_M + grid_x * HRRR_GRID_SPACING_M
    center_y = HRRR_GRID_Y0_M + grid_y * HRRR_GRID_SPACING_M
    grid_latitude, grid_longitude = _hrrr_inverse_lambert(center_x, center_y)
    return HrrrGridPoint(
        zone=site.zone,
        city=site.city,
        site_latitude=site.latitude,
        site_longitude=site.longitude,
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        grid_x=grid_x,
        grid_y=grid_y,
        chunk_x=grid_x // HRRR_ZARR_CHUNK_XY,
        chunk_y=grid_y // HRRR_ZARR_CHUNK_XY,
        in_chunk_x=grid_x % HRRR_ZARR_CHUNK_XY,
        in_chunk_y=grid_y % HRRR_ZARR_CHUNK_XY,
        distance_m=_haversine_m(
            site.latitude, site.longitude, grid_latitude, grid_longitude
        ),
    )


HRRR_GRID_POINTS: Mapping[str, HrrrGridPoint] = MappingProxyType(
    {zone: hrrr_grid_point(site) for zone, site in NYISO_WEATHER_SITES.items()}
)


class HrrrZarrManifest:
    def __init__(self, cache_root: str | Path) -> None:
        self.root = Path(cache_root).resolve() / "zarr_manifest"
        self.records_root = self.root / "records"
        self.index_path = self.root / "index.json"

    def record_path(self, record_id: str) -> Path:
        return self.records_root / f"{record_id}.json"

    def get(self, record_id: str) -> dict[str, Any] | None:
        path = self.record_path(record_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def add(self, record: Mapping[str, Any]) -> Path:
        with _WEATHER_METADATA_LOCK:
            return self._add_unlocked(record)

    def _add_unlocked(self, record: Mapping[str, Any]) -> Path:
        record = dict(record)
        if record.get("schema") != HRRR_ZARR_MANIFEST_SCHEMA:
            raise WeatherDataError("unexpected HRRR Zarr record schema")
        if int(record.get("operating_year", -1)) >= SEALED_FINAL_TEST_YEAR:
            raise WeatherCausalityError("HRRR Zarr manifest cannot open sealed 2025")
        record_id = str(record["record_id"])
        path = self.record_path(record_id)
        created = not path.exists()
        if created:
            _atomic_json(path, record)
        elif json.loads(path.read_text(encoding="utf-8")) != record:
            raise WeatherDataError(f"HRRR Zarr manifest collision: {record_id}")
        if self.index_path.exists():
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            index = {
                "schema": HRRR_ZARR_MANIFEST_SCHEMA,
                "opened_final_test_2025": False,
                "record_count": 0,
            }
        if created or not self.index_path.exists():
            _atomic_json(
                self.index_path,
                {
                    "schema": HRRR_ZARR_MANIFEST_SCHEMA,
                    "opened_final_test_2025": False,
                    "record_count": int(index.get("record_count", 0)) + int(created),
                },
            )
        return path


class HrrrZarrMaterializer:
    """Point materialiser using the Registry-listed HRRR Zarr archive.

    Each downloaded object is one 150x150 spatial chunk containing all 18 or
    48 lead hours.  Four sites map to only three distinct chunks.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        client: WeatherHTTPClient | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.client = client or WeatherHTTPClient()
        self.manifest = HrrrZarrManifest(self.cache_root)
        self._verified_cache: set[str] = set()

    @staticmethod
    def root_key(run_init: datetime) -> str:
        run_init = _utc_datetime(run_init)
        day = run_init.strftime("%Y%m%d")
        return f"sfc/{day}/{day}_{run_init:%H}z_fcst.zarr"

    @staticmethod
    def _validate_scope(run_init: datetime, operating_date: date) -> None:
        run_init = _utc_datetime(run_init)
        if operating_date.year not in DEVELOPMENT_YEARS:
            raise WeatherCausalityError(
                "HRRR Zarr development operating_date must be in 2022--2024"
            )
        if run_init >= datetime.combine(
            operating_date + timedelta(days=1), time.min, tzinfo=UTC
        ):
            raise WeatherCausalityError("HRRR run begins after its authorised operating day")
        if run_init < datetime.combine(
            operating_date - timedelta(days=3), time.min, tzinfo=UTC
        ):
            raise WeatherCausalityError("HRRR run is implausibly old for its operating day")
        if run_init.hour not in (0, 6, 12, 18):
            raise WeatherCausalityError(
                "formal day-ahead HRRR materialisation uses only 48h extended cycles"
            )
        if (run_init + timedelta(hours=48)).year >= SEALED_FINAL_TEST_YEAR:
            raise WeatherCausalityError(
                "HRRR Zarr run chunk contains sealed 2025 leads; use the mixed "
                "GRIB boundary path before remote I/O"
            )

    @staticmethod
    def _url(key: str) -> str:
        return f"{HRRR_ZARR_BASE_URL}/{key}"

    def _load_object(
        self,
        *,
        key: str,
        operating_date: date,
        record_context: Mapping[str, Any],
    ) -> tuple[bytes, RemoteObjectMetadata, dict[str, Any]]:
        metadata = self.client.metadata(self._url(key))
        identity = _record_identity(
            {"transport": "zarr", "key": key, "etag": metadata.etag}
        )
        existing = self.manifest.get(identity)
        if existing is not None:
            cache_path = Path(str(existing["local_cache_path"]))
            if cache_path.exists() and cache_path.stat().st_size == metadata.byte_count:
                if identity not in self._verified_cache:
                    if _sha256_file(cache_path) != existing["sha256"]:
                        raise WeatherDataError("cached HRRR Zarr object failed SHA-256")
                    self._verified_cache.add(identity)
                return cache_path.read_bytes(), metadata, existing

        payload = self.client.get_object(metadata)
        digest = _sha256_bytes(payload)
        cache_path = self.cache_root / "zarr_objects" / digest[:2] / f"{digest}.bin"
        if not cache_path.exists():
            _atomic_bytes(cache_path, payload)
        record = {
            "schema": HRRR_ZARR_MANIFEST_SCHEMA,
            "record_id": identity,
            "source": "hrrr",
            "source_version": "HRRRv4",
            "transport": "hrrrzarr_150x150_chunk",
            "operating_date": operating_date.isoformat(),
            "operating_year": operating_date.year,
            "s3_bucket": "hrrrzarr",
            "s3_key": key,
            "url": metadata.url,
            "etag": metadata.etag,
            "last_modified_utc": _iso_utc(metadata.last_modified_utc),
            "byte_count": metadata.byte_count,
            "sha256": digest,
            "local_cache_path": str(cache_path),
            "retrieved_at_utc": _iso_utc(datetime.now(UTC)),
            **dict(record_context),
        }
        self.manifest.add(record)
        self._verified_cache.add(identity)
        return payload, metadata, record

    @staticmethod
    def _decode_chunk(
        payload: bytes,
        *,
        zarray: Mapping[str, Any],
        chunk_y: int,
        chunk_x: int,
    ) -> np.ndarray:
        try:
            from numcodecs import get_codec
        except ImportError as exc:
            raise WeatherDecodeError(
                "HRRR Zarr decoding requires the numcodecs package"
            ) from exc
        shape = tuple(int(value) for value in zarray["shape"])
        chunks = tuple(int(value) for value in zarray["chunks"])
        if len(shape) != 3 or len(chunks) != 3 or chunks[1:] != (150, 150):
            raise WeatherDecodeError(
                f"unexpected HRRR forecast Zarr shape/chunks: {shape}/{chunks}"
            )
        decoded = get_codec(zarray["compressor"]).decode(payload)
        time_count = min(chunks[0], shape[0])
        y_count = min(chunks[1], shape[1] - chunk_y * chunks[1])
        x_count = min(chunks[2], shape[2] - chunk_x * chunks[2])
        expected = time_count * y_count * x_count
        values = np.frombuffer(decoded, dtype=np.dtype(str(zarray["dtype"])))
        if values.size != expected:
            raise WeatherDecodeError(
                f"HRRR Zarr chunk contains {values.size} values; expected {expected}"
            )
        result = values.reshape((time_count, y_count, x_count), order="C")
        fill_value = zarray.get("fill_value")
        if fill_value is not None:
            result = result.astype(np.float32, copy=True)
            result[result == float(fill_value)] = np.nan
        return result

    def materialize_run(
        self,
        *,
        store: "WorldWeatherStore",
        run_init: datetime,
        operating_date: date,
        features: Sequence[str] = HRRR_ZARR_DEFAULT_FEATURES,
        forecast_hours: Sequence[int] | None = None,
        sites: Sequence[WeatherSite] = _SITES,
    ) -> Path:
        run_init = _utc_datetime(run_init)
        self._validate_scope(run_init, operating_date)
        requested_features = tuple(dict.fromkeys(features))
        if not requested_features:
            raise ValueError("HRRR Zarr features cannot be empty")
        requested_hours = (
            None
            if forecast_hours is None
            else tuple(dict.fromkeys(int(value) for value in forecast_hours))
        )
        if requested_hours is not None:
            if not requested_hours or requested_hours != tuple(sorted(requested_hours)):
                raise ValueError("HRRR Zarr forecast_hours must be non-empty and increasing")
            for forecast_hour in requested_hours:
                HRRR_ADAPTER.validate_forecast(run_init, forecast_hour)
        unknown = set(requested_features) - set(HRRR_ZARR_VARIABLES)
        if unknown:
            raise KeyError(f"unsupported HRRR Zarr feature(s): {sorted(unknown)}")
        points = tuple(HRRR_GRID_POINTS[site.zone] for site in sites)
        root = self.root_key(run_init)
        rows: list[dict[str, Any]] = []
        visibility: list[datetime] = []
        feature_chunks: dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
        time_count: int | None = None

        for feature in requested_features:
            variable = HRRR_ZARR_VARIABLES[feature]
            array_root = f"{root}/{variable.array_relative_path}"
            metadata_key = f"{array_root}/.zarray"
            metadata_payload, metadata_remote, metadata_record = self._load_object(
                key=metadata_key,
                operating_date=operating_date,
                record_context={
                    "object_role": "zarr_array_metadata",
                    "run_init_utc": _iso_utc(run_init),
                    "feature": feature,
                    "level": variable.level,
                    "parameter": variable.parameter,
                },
            )
            visibility.append(metadata_remote.last_modified_utc)
            zarray = json.loads(metadata_payload.decode("utf-8"))
            feature_time_count = int(zarray["shape"][0])
            if time_count is None:
                time_count = feature_time_count
            elif time_count != feature_time_count:
                raise WeatherDecodeError("HRRR features expose different lead counts")
            chunks_for_feature: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
            for chunk_id in dict.fromkeys(point.forecast_chunk_id for point in points):
                _, chunk_y_text, chunk_x_text = chunk_id.split(".")
                chunk_y, chunk_x = int(chunk_y_text), int(chunk_x_text)
                key = f"{array_root}/{chunk_id}"
                payload, remote, record = self._load_object(
                    key=key,
                    operating_date=operating_date,
                    record_context={
                        "object_role": "zarr_forecast_chunk",
                        "run_init_utc": _iso_utc(run_init),
                        "feature": feature,
                        "level": variable.level,
                        "parameter": variable.parameter,
                        "chunk_id": chunk_id,
                        "grid_points": [
                            point.to_dict()
                            for point in points
                            if point.forecast_chunk_id == chunk_id
                        ],
                        "zarray_metadata_record_id": metadata_record["record_id"],
                    },
                )
                visibility.append(remote.last_modified_utc)
                chunks_for_feature[chunk_id] = (
                    self._decode_chunk(
                        payload, zarray=zarray, chunk_y=chunk_y, chunk_x=chunk_x
                    ),
                    record,
                )
            feature_chunks[feature] = chunks_for_feature

        if time_count is None:
            raise WeatherDecodeError("HRRR Zarr run produced no leads")
        visible_at = max(visibility)
        for forecast_index in range(time_count):
            forecast_hour = forecast_index + 1
            if requested_hours is not None and forecast_hour not in requested_hours:
                continue
            valid_time = run_init + timedelta(hours=forecast_hour)
            if valid_time.year not in DEVELOPMENT_YEARS:
                continue
            for point in points:
                for feature in requested_features:
                    variable = HRRR_ZARR_VARIABLES[feature]
                    chunk, record = feature_chunks[feature][point.forecast_chunk_id]
                    rows.append(
                        {
                            "schema": WEATHER_FRAME_SCHEMA,
                            "source": "hrrr",
                            "source_version": "HRRRv4",
                            "transport": "hrrrzarr_150x150_chunk",
                            "feature": feature,
                            "unit": variable.unit,
                            "grib_parameter": variable.parameter,
                            "grib_level": variable.level,
                            "grib_forecast_descriptor": f"{forecast_hour} hour fcst",
                            "zone": point.zone,
                            "city": point.city,
                            "site_latitude": point.site_latitude,
                            "site_longitude": point.site_longitude,
                            "grid_latitude": point.grid_latitude,
                            "grid_longitude": point.grid_longitude,
                            "grid_index_flat": point.grid_y * HRRR_GRID_NX + point.grid_x,
                            "grid_x": point.grid_x,
                            "grid_y": point.grid_y,
                            "grid_distance_m": point.distance_m,
                            "value": float(
                                chunk[
                                    forecast_index,
                                    point.in_chunk_y,
                                    point.in_chunk_x,
                                ]
                            ),
                            "decoded_unit": variable.unit,
                            "run_init_utc": pd.Timestamp(run_init),
                            "forecast_hour": forecast_hour,
                            "valid_time_utc": pd.Timestamp(valid_time),
                            "visible_at_utc": pd.Timestamp(visible_at),
                            "fragment_record_id": record["record_id"],
                            "source_object_record_ids": json.dumps(
                                sorted(
                                    {
                                        str(record["record_id"]),
                                        str(record["zarray_metadata_record_id"]),
                                    }
                                ),
                                separators=(",", ":"),
                            ),
                            "native_resolution_minutes": 60,
                            "temporal_interpolation": variable.temporal_interpolation,
                        }
                    )
        frame = pd.DataFrame.from_records(rows)
        if frame.empty:
            raise WeatherCausalityError("HRRR Zarr run has no unsealed development leads")
        return store.put(frame)


class HrrrMixedMaterializer:
    """Materialise the V1 weather pair from one coherent HRRR run.

    The historical HRRR forecast Zarr archive usually has compact TMP chunks,
    while DSWRF must come from GRIB2 because of the documented solar-field
    archive bug.  A small number of TMP Zarr runs are also absent.  Only an
    explicit HTTP 404 may route TMP to the matching same-run GRIB2 byte range;
    every other Zarr error fails closed.  Both features retain one HRRR run and
    the committed artifact receives one conservative visibility time: the
    maximum over every required object.
    """

    ZARR_FEATURES = ("air_temperature_2m_k",)
    GRIB_FEATURES = ("downward_shortwave_wm2",)

    @staticmethod
    def _requires_sealed_boundary_grib(run_init: datetime) -> bool:
        return (
            _utc_datetime(run_init) + timedelta(hours=48)
        ).year >= SEALED_FINAL_TEST_YEAR

    def __init__(
        self,
        cache_root: str | Path,
        *,
        client: WeatherHTTPClient | None = None,
        decoder: GribPointDecoder | None = None,
    ) -> None:
        shared_client = client or WeatherHTTPClient()
        self.zarr = HrrrZarrMaterializer(cache_root, client=shared_client)
        self.grib = WeatherRangeMaterializer(
            cache_root,
            client=shared_client,
            decoder=decoder or EccodesPointDecoder(),
        )

    def materialize_run(
        self,
        *,
        store: "WorldWeatherStore",
        run_init: datetime,
        operating_date: date,
        forecast_hours: Sequence[int] | None = None,
        sites: Sequence[WeatherSite] = _SITES,
    ) -> Path:
        run = _utc_datetime(run_init)
        # Validate the operating/run relationship before the first HEAD/GET.
        # A crossing Zarr cube is handled below with lead-specific GRIB ranges
        # so no sealed 2025 values are downloaded or decoded.
        if operating_date.year not in DEVELOPMENT_YEARS:
            raise WeatherCausalityError(
                "mixed HRRR development operating_date must be in 2022--2024"
            )
        if run.hour not in (0, 6, 12, 18):
            raise WeatherCausalityError(
                "formal day-ahead HRRR materialisation uses only 48h extended cycles"
            )
        if run >= datetime.combine(
            operating_date + timedelta(days=1), time.min, tzinfo=UTC
        ) or run < datetime.combine(
            operating_date - timedelta(days=3), time.min, tzinfo=UTC
        ):
            raise WeatherCausalityError(
                "mixed HRRR run lies outside its authorised operating-day window"
            )
        hours = (
            tuple(
                forecast_hour
                for forecast_hour in range(1, 49)
                if (run + timedelta(hours=forecast_hour)).date() == operating_date
            )
            if forecast_hours is None
            else tuple(dict.fromkeys(int(value) for value in forecast_hours))
        )
        if not hours:
            raise ValueError("mixed HRRR forecast_hours cannot be empty")
        if hours != tuple(sorted(hours)):
            raise ValueError("mixed HRRR forecast_hours must be increasing")
        for forecast_hour in hours:
            HRRR_ADAPTER.validate_forecast(run, forecast_hour)

        boundary_grib = self._requires_sealed_boundary_grib(run)
        zarr_not_found = False
        frames: list[pd.DataFrame] = []
        if not boundary_grib:
            try:
                zarr_path = self.zarr.materialize_run(
                    store=store,
                    run_init=run,
                    operating_date=operating_date,
                    features=self.ZARR_FEATURES,
                    forecast_hours=hours,
                    sites=sites,
                )
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status != 404:
                    raise
                zarr_not_found = True
            else:
                zarr_frame = pd.read_parquet(zarr_path)
                frames.append(
                    zarr_frame.loc[
                        zarr_frame["forecast_hour"].isin(hours)
                        & zarr_frame["feature"].isin(self.ZARR_FEATURES)
                    ].copy()
                )
        grib_features = (
            (*self.ZARR_FEATURES, *self.GRIB_FEATURES)
            if boundary_grib or zarr_not_found
            else self.GRIB_FEATURES
        )
        for forecast_hour in hours:
            records = self.grib.acquire(
                source="hrrr",
                run_init=run,
                forecast_hour=forecast_hour,
                features=grib_features,
                sites=sites,
            )
            frames.append(self.grib.decode_records(records, sites=sites))
        combined = pd.concat(frames, ignore_index=True)
        expected = len(hours) * len(sites) * (
            len(self.ZARR_FEATURES) + len(self.GRIB_FEATURES)
        )
        if len(combined) != expected:
            raise WeatherDataError(
                f"mixed HRRR run has {len(combined)}/{expected} required rows"
            )
        common_visibility = pd.to_datetime(
            combined["visible_at_utc"], utc=True, errors="raise"
        ).max()
        combined["visible_at_utc"] = common_visibility
        combined["transport_contract"] = (
            "same_hrrr_run_tmp_and_dswrf_grib_idx_sealed_boundary"
            if boundary_grib
            else (
                "same_hrrr_run_tmp_and_dswrf_grib_idx_zarr_http_404"
                if zarr_not_found
                else "same_hrrr_run_tmp_zarr_dswrf_grib_idx_byte_range"
            )
        )
        combined["transport_substitution_reason"] = (
            "sealed_2025_leads_share_the_zarr_chunk"
            if boundary_grib
            else (
                "required_hrrrzarr_tmp_object_http_404"
                if zarr_not_found
                else "none"
            )
        )
        return store.put(combined)


_REQUIRED_FRAME_COLUMNS = frozenset(
    {
        "schema",
        "source",
        "source_version",
        "feature",
        "unit",
        "zone",
        "city",
        "site_latitude",
        "site_longitude",
        "grid_latitude",
        "grid_longitude",
        "grid_index_flat",
        "grid_x",
        "grid_y",
        "value",
        "run_init_utc",
        "forecast_hour",
        "valid_time_utc",
        "visible_at_utc",
        "fragment_record_id",
    }
)


@dataclass(frozen=True)
class WeatherCube:
    sources: tuple[str, ...]
    valid_times: tuple[pd.Timestamp, ...]
    zones: tuple[str, ...]
    features: tuple[str, ...]
    values: np.ndarray
    mask: np.ndarray
    run_init_unix_s: np.ndarray
    visible_at_unix_s: np.ndarray
    forecast_hour: np.ndarray
    forecast_age_hours: np.ndarray


@dataclass(frozen=True)
class WeatherQueryResult:
    decision_origin_utc: pd.Timestamp
    frame: pd.DataFrame
    requested_sources: tuple[str, ...]
    requested_valid_times: tuple[pd.Timestamp, ...]
    requested_zones: tuple[str, ...]
    requested_features: tuple[str, ...]

    def to_cube(self) -> WeatherCube:
        sources = self.requested_sources
        times = self.requested_valid_times
        zones = self.requested_zones
        features = self.requested_features
        shape = (len(sources), len(times), len(zones), len(features))
        values = np.full(shape, np.nan, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        metadata_shape = shape[:3]
        run_init = np.full(metadata_shape, -1, dtype=np.int64)
        visible_at = np.full(metadata_shape, -1, dtype=np.int64)
        forecast_hour = np.full(metadata_shape, np.nan, dtype=np.float32)
        forecast_age = np.full(metadata_shape, np.nan, dtype=np.float32)
        source_index = {value: index for index, value in enumerate(sources)}
        time_index = {value: index for index, value in enumerate(times)}
        zone_index = {value: index for index, value in enumerate(zones)}
        feature_index = {value: index for index, value in enumerate(features)}
        for row in self.frame.itertuples(index=False):
            key = (
                source_index[row.source],
                time_index[pd.Timestamp(row.valid_time_utc)],
                zone_index[row.zone],
                feature_index[row.feature],
            )
            values[key] = np.float32(row.value)
            mask[key] = bool(np.isfinite(row.value))
            meta_key = key[:3]
            run_timestamp = pd.Timestamp(row.run_init_utc)
            visible_timestamp = pd.Timestamp(row.visible_at_utc)
            run_init[meta_key] = int(run_timestamp.timestamp())
            visible_at[meta_key] = int(visible_timestamp.timestamp())
            forecast_hour[meta_key] = np.float32(row.forecast_hour)
            forecast_age[meta_key] = np.float32(
                (self.decision_origin_utc - run_timestamp).total_seconds() / 3600.0
            )
        return WeatherCube(
            sources=sources,
            valid_times=times,
            zones=zones,
            features=features,
            values=values,
            mask=mask,
            run_init_unix_s=run_init,
            visible_at_unix_s=visible_at,
            forecast_hour=forecast_hour,
            forecast_age_hours=forecast_age,
        )


class WorldWeatherStore:
    """Materialised NOAA point forecasts with strict as-of selection.

    HRRR and NBM remain separate on the leading ``source`` dimension.  The
    store never averages, fills, or otherwise hard-fuses the two sources.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        allow_sealed_2025: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.materialized_root = self.root / "materialized"
        self.artifact_records_root = self.root / "materialized_manifest" / "records"
        self.index_path = self.root / "materialized_index.json"
        self.allow_sealed_2025 = bool(allow_sealed_2025)
        self._artifact_cache: tuple[dict[str, Any], ...] | None = None
        self._verified_artifacts: set[str] = set()

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "schema": WEATHER_STORE_SCHEMA,
                "opened_final_test_2025": False,
                "artifact_count": 0,
            }
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("schema") != WEATHER_STORE_SCHEMA:
            raise WeatherDataError("unexpected WorldWeatherStore schema")
        if payload.get("opened_final_test_2025") is not False and not self.allow_sealed_2025:
            raise WeatherCausalityError("WorldWeatherStore has opened sealed 2025 content")
        return payload

    def _artifact_records(self) -> tuple[dict[str, Any], ...]:
        if self._artifact_cache is None:
            self._artifact_cache = tuple(
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(self.artifact_records_root.glob("*.json"))
            )
        return self._artifact_cache

    def artifact_count(self) -> int:
        """Return the registered local artifact count without remote I/O."""

        return len(self._artifact_records())

    def _verified_artifact_path(self, artifact: Mapping[str, Any]) -> Path:
        artifact_id = str(artifact["artifact_id"])
        path = Path(str(artifact["path"]))
        if artifact_id not in self._verified_artifacts:
            if not path.exists() or _sha256_file(path) != str(artifact["sha256"]):
                raise WeatherDataError(
                    f"materialized weather artifact failed SHA-256: {artifact_id}"
                )
            self._verified_artifacts.add(artifact_id)
        return path

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(_REQUIRED_FRAME_COLUMNS - set(frame.columns))
        if missing:
            raise WeatherDataError(f"weather frame is missing columns: {missing}")
        result = frame.copy()
        if set(result["schema"].astype(str)) != {WEATHER_FRAME_SCHEMA}:
            raise WeatherDataError("weather frame schema marker is invalid")
        for column in ("run_init_utc", "valid_time_utc", "visible_at_utc"):
            result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
        if not self.allow_sealed_2025:
            run_years = set(result["run_init_utc"].dt.year)
            valid_years = set(result["valid_time_utc"].dt.year)
            if any(year >= SEALED_FINAL_TEST_YEAR for year in run_years | valid_years):
                raise WeatherCausalityError("cannot materialise sealed 2025 weather in development")
            if not valid_years <= DEVELOPMENT_YEARS:
                raise WeatherCausalityError(
                    "development weather valid times must be in 2022--2024, "
                    f"received {sorted(valid_years)}"
                )
            if not run_years <= DEVELOPMENT_YEARS | {2021}:
                raise WeatherCausalityError(
                    f"development weather source runs are out of scope: {sorted(run_years)}"
                )
        if set(result["source"].astype(str)) - set(WEATHER_SOURCES):
            raise WeatherDataError("weather frame contains an unsupported source")
        if set(result["zone"].astype(str)) != set(NYISO_WEATHER_SITES):
            raise WeatherDataError("weather frame must contain all four frozen NYISO sites")
        if (result["visible_at_utc"] < result["run_init_utc"]).any():
            raise WeatherDataError("visible_at precedes the model run initialisation")
        for (source, run_init, forecast_hour), group in result.groupby(
            ["source", "run_init_utc", "forecast_hour"], sort=False
        ):
            adapter = WEATHER_SOURCES[str(source)]
            run = pd.Timestamp(run_init).to_pydatetime()
            transports = set(group.get("transport", pd.Series(dtype=str)).astype(str))
            is_hrrr_zarr = transports == {"hrrrzarr_150x150_chunk"}
            if not self.allow_sealed_2025 and not is_hrrr_zarr:
                adapter.validate_forecast(run, int(forecast_hour))
            expected_valid = pd.Timestamp(run_init) + pd.Timedelta(
                hours=int(forecast_hour)
            )
            if set(group["valid_time_utc"]) != {expected_valid}:
                raise WeatherDataError("valid_time does not equal run_init + forecast_hour")
            if set(group["source_version"].astype(str)) != {
                adapter.source_version(run)
            }:
                raise WeatherDataError("weather source_version does not match its run date")
            for feature in set(group["feature"].astype(str)):
                if feature not in WEATHER_VARIABLES:
                    raise WeatherDataError(f"unknown materialized weather feature: {feature}")
        duplicated = result.duplicated(
            ["source", "run_init_utc", "forecast_hour", "zone", "feature"], keep=False
        )
        if duplicated.any():
            raise WeatherDataError("weather frame contains duplicate source/run/site/features")
        return result

    def put(self, frame: pd.DataFrame) -> Path:
        with _WEATHER_METADATA_LOCK:
            return self._put_unlocked(frame)

    def _put_unlocked(self, frame: pd.DataFrame) -> Path:
        frame = self._validate_frame(frame)
        identity_columns = ["source", "source_version", "run_init_utc"]
        identities = frame[identity_columns].drop_duplicates()
        if len(identities) != 1:
            raise WeatherDataError("one weather artifact must contain one source/run")
        identity = identities.iloc[0]
        run_init = pd.Timestamp(identity["run_init_utc"])
        source = str(identity["source"])
        forecast_hours = sorted(map(int, frame["forecast_hour"].unique()))
        features = sorted(frame["feature"].astype(str).unique())
        zones = sorted(frame["zone"].astype(str).unique())
        transports = sorted(
            frame.get("transport", pd.Series(dtype=str)).dropna().astype(str).unique()
        )
        transport_contracts = sorted(
            frame.get("transport_contract", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        )
        substitution_reasons = sorted(
            frame.get("transport_substitution_reason", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        )
        object_version_set = set(frame["fragment_record_id"].astype(str).unique())
        if "source_object_record_ids" in frame:
            for encoded in frame["source_object_record_ids"].dropna().astype(str):
                try:
                    values = json.loads(encoded)
                except json.JSONDecodeError as exc:
                    raise WeatherDataError(
                        "source_object_record_ids must be a JSON string array"
                    ) from exc
                if not isinstance(values, list) or not all(
                    isinstance(value, str) and value for value in values
                ):
                    raise WeatherDataError(
                        "source_object_record_ids must contain non-empty strings"
                    )
                object_version_set.update(values)
        object_versions = sorted(object_version_set)
        artifact_key = "|".join(
            (
                source,
                str(identity["source_version"]),
                run_init.isoformat(),
                ",".join(map(str, forecast_hours)),
                ",".join(features),
                ",".join(zones),
                ",".join(object_versions),
                str(WEATHER_ARTIFACT_CONTRACT_VERSION),
                ",".join(transports),
                ",".join(transport_contracts),
                ",".join(substitution_reasons),
            )
        )
        artifact_id = hashlib.sha256(artifact_key.encode()).hexdigest()
        target = (
            self.materialized_root
            / f"source={source}"
            / f"year={run_init.year:04d}"
            / f"month={run_init.month:02d}"
            / f"day={run_init.day:02d}"
            / f"run={run_init.hour:02d}"
            / f"forecast.{artifact_id[:12]}.parquet"
        )
        artifact_record_path = self.artifact_records_root / f"{artifact_id}.json"
        if target.exists() and artifact_record_path.exists():
            artifact = json.loads(artifact_record_path.read_text(encoding="utf-8"))
            return self._verified_artifact_path(artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        frame.sort_values(
            ["source", "run_init_utc", "forecast_hour", "zone", "feature"],
            kind="mergesort",
        ).to_parquet(temporary, index=False)
        os.replace(temporary, target)
        file_hash = _sha256_file(target)
        index = self._load_index()
        artifact = {
            "artifact_id": artifact_id,
            "artifact_contract_version": WEATHER_ARTIFACT_CONTRACT_VERSION,
            "source": source,
            "source_version": str(identity["source_version"]),
            "run_init_utc": _iso_utc(run_init),
            "forecast_hours": forecast_hours,
            "valid_times_utc": [
                _iso_utc(value)
                for value in sorted(frame["valid_time_utc"].drop_duplicates())
            ],
            "visible_at_utc": _iso_utc(frame["visible_at_utc"].max()),
            "features": features,
            "zones": zones,
            "source_object_record_ids": object_versions,
            "transports": transports,
            "transport_contracts": transport_contracts,
            "transport_substitution_reasons": substitution_reasons,
            "row_count": len(frame),
            "path": str(target),
            "sha256": file_hash,
        }
        artifact_record_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(artifact_record_path, artifact)
        created_count = int(index.get("artifact_count", 0)) + 1
        opened_2025 = bool(index.get("opened_final_test_2025", False)) or any(
            _utc_datetime(value).year >= SEALED_FINAL_TEST_YEAR
            for value in artifact["valid_times_utc"]
        )
        _atomic_json(
            self.index_path,
            {
                "schema": WEATHER_STORE_SCHEMA,
                "opened_final_test_2025": opened_2025,
                "artifact_count": created_count,
            },
        )
        self._artifact_cache = None
        self._verified_artifacts.add(artifact_id)
        return target

    def find_complete_run(
        self,
        *,
        source: str,
        run_init: datetime | pd.Timestamp | str,
        forecast_hours: Sequence[int],
        features: Sequence[str],
        zones: Sequence[str] = tuple(NYISO_WEATHER_SITES),
    ) -> Path | None:
        """Return an existing exact run artifact for resumable materialisation."""

        run = _utc_datetime(run_init)
        if run.year >= SEALED_FINAL_TEST_YEAR and not self.allow_sealed_2025:
            raise WeatherCausalityError("cannot inspect a sealed 2025 weather run")
        requested_hours = sorted(dict.fromkeys(int(value) for value in forecast_hours))
        requested_features = set(map(str, features))
        requested_zones = set(map(str, zones))
        candidates: list[dict[str, Any]] = []
        self._load_index()
        for artifact in self._artifact_records():
            if int(artifact.get("artifact_contract_version", 0)) != (
                WEATHER_ARTIFACT_CONTRACT_VERSION
            ):
                continue
            if artifact["source"] != source:
                continue
            if _utc_datetime(artifact["run_init_utc"]) != run:
                continue
            if sorted(map(int, artifact["forecast_hours"])) != requested_hours:
                continue
            if set(artifact["features"]) != requested_features:
                continue
            if set(artifact["zones"]) != requested_zones:
                continue
            path = Path(str(artifact["path"]))
            if path.exists():
                candidates.append(artifact)
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda item: (
                _utc_datetime(item["visible_at_utc"]),
                str(item["artifact_id"]),
            ),
        )
        return self._verified_artifact_path(selected)

    def query_as_of(
        self,
        decision_origin: datetime | pd.Timestamp | str,
        *,
        valid_times: Sequence[datetime | pd.Timestamp | str],
        sources: Sequence[str] = ("hrrr",),
        zones: Sequence[str] = tuple(NYISO_WEATHER_SITES),
        features: Sequence[str] = (
            "air_temperature_2m_k",
            "dew_point_2m_k",
            "relative_humidity_2m_pct",
            "cloud_cover_total_pct",
            "downward_shortwave_wm2",
        ),
        require_complete: bool = True,
    ) -> WeatherQueryResult:
        origin = pd.Timestamp(_utc_datetime(decision_origin))
        requested_times = tuple(pd.Timestamp(_utc_datetime(value)) for value in valid_times)
        requested_sources = tuple(dict.fromkeys(map(str, sources)))
        requested_zones = tuple(dict.fromkeys(map(str, zones)))
        requested_features = tuple(dict.fromkeys(map(str, features)))
        if not requested_times:
            raise ValueError("valid_times cannot be empty")
        if set(requested_sources) - set(WEATHER_SOURCES):
            raise KeyError("query contains an unsupported weather source")
        if set(requested_zones) - set(NYISO_WEATHER_SITES):
            raise KeyError("query contains an unsupported NYISO site")
        if set(requested_features) - set(WEATHER_VARIABLES):
            raise KeyError("query contains an unsupported weather feature")
        if not self.allow_sealed_2025:
            validate_development_times(origin.to_pydatetime())
            sealed = {value.year for value in requested_times if value.year >= 2025}
            if sealed:
                raise WeatherCausalityError("query cannot open sealed 2025 valid times")

        self._load_index()
        candidates = []
        requested_time_strings = {_iso_utc(value) for value in requested_times}
        for artifact in self._artifact_records():
            if artifact["source"] not in requested_sources:
                continue
            visible = pd.Timestamp(_utc_datetime(artifact["visible_at_utc"]))
            run_init = pd.Timestamp(_utc_datetime(artifact["run_init_utc"]))
            artifact_times = set(artifact["valid_times_utc"])
            if visible > origin or run_init > origin:
                continue
            if require_complete and not requested_time_strings <= artifact_times:
                continue
            if not require_complete and not requested_time_strings & artifact_times:
                continue
            if not set(requested_features) <= set(artifact["features"]):
                if require_complete:
                    continue
            if not set(requested_zones) <= set(artifact["zones"]):
                if require_complete:
                    continue
            candidates.append(artifact)

        if not candidates:
            empty = pd.DataFrame(columns=sorted(_REQUIRED_FRAME_COLUMNS))
            return WeatherQueryResult(
                decision_origin_utc=origin,
                frame=empty,
                requested_sources=requested_sources,
                requested_valid_times=requested_times,
                requested_zones=requested_zones,
                requested_features=requested_features,
            )

        candidate_frame = pd.DataFrame.from_records(candidates)
        if "artifact_contract_version" not in candidate_frame:
            candidate_frame["artifact_contract_version"] = 0
        else:
            candidate_frame["artifact_contract_version"] = candidate_frame[
                "artifact_contract_version"
            ].fillna(0).astype(int)
        candidate_frame["run_init_utc"] = pd.to_datetime(candidate_frame["run_init_utc"], utc=True)
        candidate_frame["visible_at_utc"] = pd.to_datetime(
            candidate_frame["visible_at_utc"], utc=True
        )
        # Select one coherent latest run per source for the entire requested
        # path; valid timestamps are never stitched across issue cycles.
        candidate_frame = candidate_frame.sort_values(
            [
                "run_init_utc",
                "visible_at_utc",
                "artifact_contract_version",
                "artifact_id",
            ]
        ).drop_duplicates(["source"], keep="last")
        selected_ids = set(candidate_frame["artifact_id"].astype(str))
        selected_records = {
            str(artifact["artifact_id"]): artifact
            for artifact in candidates
            if str(artifact["artifact_id"]) in selected_ids
        }
        frames = [
            pd.read_parquet(self._verified_artifact_path(artifact))
            for artifact in selected_records.values()
        ]
        result = pd.concat(frames, ignore_index=True)
        result["run_init_utc"] = pd.to_datetime(result["run_init_utc"], utc=True)
        result["valid_time_utc"] = pd.to_datetime(result["valid_time_utc"], utc=True)
        result["visible_at_utc"] = pd.to_datetime(result["visible_at_utc"], utc=True)
        result = result.loc[
            result["source"].isin(requested_sources)
            & result["valid_time_utc"].isin(requested_times)
            & result["zone"].isin(requested_zones)
            & result["feature"].isin(requested_features)
            & result["visible_at_utc"].le(origin)
            & result["run_init_utc"].le(origin)
        ].copy()
        result["forecast_age_hours"] = (
            origin - result["run_init_utc"]
        ).dt.total_seconds() / 3600.0
        result = result.sort_values(["source", "valid_time_utc", "zone", "feature"])
        return WeatherQueryResult(
            decision_origin_utc=origin,
            frame=result.reset_index(drop=True),
            requested_sources=requested_sources,
            requested_valid_times=requested_times,
            requested_zones=requested_zones,
            requested_features=requested_features,
        )

    def query_causal_hold_as_of(
        self,
        decision_origin: datetime | pd.Timestamp | str,
        *,
        valid_times: Sequence[datetime | pd.Timestamp | str],
        source: str = "hrrr",
        zones: Sequence[str] = tuple(NYISO_WEATHER_SITES),
        features: Sequence[str] = (
            "air_temperature_2m_k",
            "downward_shortwave_wm2",
        ),
    ) -> WeatherQueryResult:
        """Causally hold one coherent hourly run onto 15-minute tokens."""

        origin = pd.Timestamp(_utc_datetime(decision_origin))
        targets = tuple(pd.Timestamp(_utc_datetime(value)) for value in valid_times)
        if not targets:
            raise ValueError("valid_times cannot be empty")
        if source not in WEATHER_SOURCES:
            raise KeyError(f"unsupported weather source: {source}")
        requested_zones = tuple(dict.fromkeys(map(str, zones)))
        requested_features = tuple(dict.fromkeys(map(str, features)))
        if set(requested_zones) - set(NYISO_WEATHER_SITES):
            raise KeyError("query contains an unsupported NYISO site")
        if set(requested_features) - set(WEATHER_VARIABLES):
            raise KeyError("query contains an unsupported weather feature")
        if not self.allow_sealed_2025:
            validate_development_times(origin.to_pydatetime())
            if any(value.year >= SEALED_FINAL_TEST_YEAR for value in targets):
                raise WeatherCausalityError("query cannot open sealed 2025 valid times")

        anchors = tuple(value.floor("h") for value in targets)
        required_anchor_strings = {_iso_utc(value) for value in anchors}
        candidates: list[dict[str, Any]] = []
        self._load_index()
        for artifact in self._artifact_records():
            if artifact["source"] != source:
                continue
            run_init = pd.Timestamp(_utc_datetime(artifact["run_init_utc"]))
            visible_at = pd.Timestamp(_utc_datetime(artifact["visible_at_utc"]))
            if run_init > origin or visible_at > origin:
                continue
            if not required_anchor_strings <= set(artifact["valid_times_utc"]):
                continue
            if not set(requested_features) <= set(artifact["features"]):
                continue
            if not set(requested_zones) <= set(artifact["zones"]):
                continue
            candidates.append(artifact)

        empty = WeatherQueryResult(
            decision_origin_utc=origin,
            frame=pd.DataFrame(columns=sorted(_REQUIRED_FRAME_COLUMNS)),
            requested_sources=(source,),
            requested_valid_times=targets,
            requested_zones=requested_zones,
            requested_features=requested_features,
        )
        if not candidates:
            return empty

        selected = max(
            candidates,
            key=lambda item: (
                _utc_datetime(item["run_init_utc"]),
                _utc_datetime(item["visible_at_utc"]),
                int(item.get("artifact_contract_version", 0)),
                str(item["artifact_id"]),
            ),
        )
        native = pd.read_parquet(self._verified_artifact_path(selected))
        for column in ("run_init_utc", "valid_time_utc", "visible_at_utc"):
            native[column] = pd.to_datetime(native[column], utc=True, errors="raise")
        native = native.loc[
            native["valid_time_utc"].isin(anchors)
            & native["zone"].isin(requested_zones)
            & native["feature"].isin(requested_features)
        ].copy()
        expected_rows = len(set(anchors)) * len(requested_zones) * len(requested_features)
        if len(native) != expected_rows:
            return empty

        native_by_key = {
            (row.valid_time_utc, row.zone, row.feature): row
            for row in native.itertuples(index=False)
        }
        rows: list[dict[str, Any]] = []
        for target, anchor in zip(targets, anchors):
            for zone in requested_zones:
                for feature in requested_features:
                    row = native_by_key[(anchor, zone, feature)]
                    values = row._asdict()
                    values["native_valid_time_utc"] = anchor
                    values["valid_time_utc"] = target
                    values["causal_hold_minutes"] = int(
                        (target - anchor).total_seconds() // 60
                    )
                    values["forecast_age_hours"] = (
                        origin - pd.Timestamp(values["run_init_utc"])
                    ).total_seconds() / 3600.0
                    rows.append(values)
        frame = pd.DataFrame.from_records(rows).sort_values(
            ["source", "valid_time_utc", "zone", "feature"]
        )
        return WeatherQueryResult(
            decision_origin_utc=origin,
            frame=frame.reset_index(drop=True),
            requested_sources=(source,),
            requested_valid_times=targets,
            requested_zones=requested_zones,
            requested_features=requested_features,
        )


def iter_run_initialisations(
    start: date,
    end: date,
    *,
    cycles_utc: Sequence[int],
) -> tuple[datetime, ...]:
    if end < start:
        raise ValueError("weather end date precedes start date")
    if start.year not in DEVELOPMENT_YEARS or end.year not in DEVELOPMENT_YEARS:
        raise WeatherCausalityError("run iteration is restricted to 2022--2024")
    cycles = tuple(sorted(set(int(value) for value in cycles_utc)))
    if not cycles or any(value < 0 or value > 23 for value in cycles):
        raise ValueError("cycles_utc must contain hours in [0, 23]")
    values: list[datetime] = []
    current = start
    while current <= end:
        values.extend(datetime.combine(current, time(hour, tzinfo=UTC)) for hour in cycles)
        current += timedelta(days=1)
    return tuple(values)


__all__ = [
    "DEVELOPMENT_YEARS",
    "EccodesPointDecoder",
    "GribIndexEntry",
    "GribPointDecoder",
    "GribVariableSpec",
    "HRRR_ADAPTER",
    "HRRR_DEFAULT_FEATURES",
    "HRRR_GRID_POINTS",
    "HRRR_ZARR_DEFAULT_FEATURES",
    "HRRR_ZARR_VARIABLES",
    "HrrrGridPoint",
    "HrrrMixedMaterializer",
    "HrrrZarrMaterializer",
    "NBM_ADAPTER",
    "NBM_DEFAULT_FEATURES",
    "NYISO_WEATHER_SITES",
    "RemoteObjectMetadata",
    "SEALED_FINAL_TEST_YEAR",
    "WEATHER_FRAME_SCHEMA",
    "WEATHER_ARTIFACT_CONTRACT_VERSION",
    "WEATHER_MANIFEST_SCHEMA",
    "WEATHER_SOURCES",
    "WEATHER_STORE_SCHEMA",
    "WEATHER_VARIABLES",
    "WeatherCausalityError",
    "WeatherCube",
    "WeatherDataError",
    "WeatherDecodeError",
    "WeatherFragmentRecord",
    "WeatherHTTPClient",
    "WeatherManifest",
    "WeatherQueryResult",
    "WeatherRangeMaterializer",
    "WeatherRemoteError",
    "WeatherSite",
    "WeatherSourceAdapter",
    "WorldWeatherStore",
    "iter_run_initialisations",
    "parse_grib_index",
    "select_grib_entries",
    "validate_development_times",
]
