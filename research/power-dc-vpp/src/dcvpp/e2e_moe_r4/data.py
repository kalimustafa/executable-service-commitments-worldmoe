"""Causal, split-locked data contracts for solver-free E2E-MoE R4.

This module deliberately performs path/date gates *before* opening a payload.
The 2024 selection year is discoverable as metadata but sealed by default; the
2025 locked test is never openable through this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

INTERVALS = 96
SCENARIOS = 20
EMPIRICAL_SCENARIOS = 100
NYISO_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 6, 30)
VALIDATION_START = date(2023, 7, 1)
VALIDATION_END = date(2023, 12, 31)
SEALED_SELECTION_START = date(2024, 1, 1)
SEALED_SELECTION_END = date(2024, 12, 31)
LOCKED_TEST_YEAR = 2025

_DATE_TOKEN = re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)")
_YEAR_TOKEN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


class R4DataError(ValueError):
    """An R4 input violates its shape, provenance, or temporal contract."""


class LockedTestAccessError(R4DataError):
    """Raised before any attempted access to the 2025 locked-test payload."""


class SealedSelectionAccessError(R4DataError):
    """Raised before opening 2024 without an explicit selection authorization."""


def _date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise R4DataError(f"invalid operating date: {value!r}") from exc


def temporal_split(value: str | date | datetime) -> str:
    """Return the immutable R4 split label for an operating date."""

    day = _date(value)
    if TRAIN_START <= day <= TRAIN_END:
        return "train"
    if VALIDATION_START <= day <= VALIDATION_END:
        return "validation"
    if SEALED_SELECTION_START <= day <= SEALED_SELECTION_END:
        return "sealed_selection"
    if day.year == LOCKED_TEST_YEAR:
        return "locked_test"
    raise R4DataError(
        "R4 accepts only train=2022-01-01..2023-06-30, "
        "validation=2023-07-01..2023-12-31, sealed_selection=2024, "
        f"and locked_test={LOCKED_TEST_YEAR}; rejected {day.isoformat()}"
    )


def _path_dates(path: Path | str) -> tuple[date, ...]:
    text = str(Path(path))
    values: list[date] = []
    for year, month, day in _DATE_TOKEN.findall(text):
        try:
            values.append(date(int(year), int(month), int(day)))
        except ValueError as exc:
            raise R4DataError(f"invalid date token in path: {path}") from exc
    return tuple(values)


def _guard_path_before_open(
    path: Path | str,
    *,
    allow_sealed_selection: bool = False,
    locked_year: int = LOCKED_TEST_YEAR,
) -> Path:
    candidate = Path(path)
    tokens = _path_dates(candidate)
    years = {item.year for item in tokens}
    years.update(int(value) for value in _YEAR_TOKEN.findall(str(candidate)))
    if int(locked_year) in years:
        raise LockedTestAccessError(
            f"refusing to open locked {locked_year} path: {candidate}"
        )
    if 2024 in years and not allow_sealed_selection:
        raise SealedSelectionAccessError(
            f"2024 selection payload is sealed; refusing to open: {candidate}"
        )
    return candidate


@dataclass(frozen=True)
class DiscoveredDay:
    """Filename-derived metadata; constructing it never reads the payload."""

    path: Path
    operating_date: str
    split: str
    sealed: bool

    def __fspath__(self) -> str:
        return str(self.path)


def discover_eligible_days(
    root: Path | str,
    start: str | date,
    end: str | date,
    locked_year: int = LOCKED_TEST_YEAR,
    metadata_only: bool = True,
) -> list[DiscoveredDay] | list[Path]:
    """Discover dated bundles without opening them.

    Locked-year paths are excluded.  Sealed 2024 paths are returned only as
    metadata when ``metadata_only=True``; asking for openable paths omits them.
    Dates are derived from names, so this function cannot leak payload content.
    """

    lower, upper = _date(start), _date(end)
    if lower > upper:
        raise ValueError("start must not be after end")
    records: list[DiscoveredDay] = []
    for candidate in sorted(Path(root).rglob("*.npz")):
        tokens = _path_dates(candidate)
        if not tokens:
            continue
        operating = tokens[-1]
        if operating.year == int(locked_year) or not lower <= operating <= upper:
            continue
        try:
            split = temporal_split(operating)
        except R4DataError:
            continue
        sealed = split == "sealed_selection"
        if sealed and not metadata_only:
            continue
        records.append(
            DiscoveredDay(candidate.resolve(), operating.isoformat(), split, sealed)
        )
    if metadata_only:
        return records
    return [record.path for record in records]


def _finite(
    value: Any, *, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    if shape is not None and result.shape != shape:
        raise R4DataError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise R4DataError(f"{name} contains non-finite values")
    result.setflags(write=False)
    return result


def _utc(value: Any, *, name: str) -> datetime:
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise R4DataError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise R4DataError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class R4SelectionDay:
    """One complete 96-origin day with issue-time inputs and sealed outcomes."""

    operating_date: str
    timestamps_utc: np.ndarray
    issue_time_utc: str
    market: np.ndarray
    workload: np.ndarray
    physical_static: np.ndarray
    physical_dynamic: np.ndarray
    initial_soc_mwh: np.ndarray
    da_price: np.ndarray
    rt_price_pool: np.ndarray
    workload_pool: np.ndarray
    pv_pool: np.ndarray
    realized_da_price: np.ndarray
    realized_rt_price: np.ndarray
    realized_workload: np.ndarray
    realized_pv: np.ndarray
    physical_multiplier: np.ndarray
    feature_provenance: Mapping[str, Mapping[str, Any]]
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    history_dates: tuple[str, ...] = ()
    locked_test: bool = False
    realized_pue: np.ndarray | None = None
    physical_twin_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        split = temporal_split(self.operating_date)
        if split == "locked_test" or bool(self.locked_test):
            raise LockedTestAccessError("2025 locked-test day cannot be instantiated")
        expected = {
            "timestamps_utc": (INTERVALS,),
            "market": (INTERVALS, 16),
            "workload": (INTERVALS, 5),
            "physical_static": (4,),
            "physical_dynamic": (INTERVALS, 12),
            "initial_soc_mwh": (4,),
            "da_price": (4, INTERVALS),
            "rt_price_pool": (EMPIRICAL_SCENARIOS, 4, INTERVALS),
            "workload_pool": (EMPIRICAL_SCENARIOS, INTERVALS),
            "pv_pool": (EMPIRICAL_SCENARIOS, 4, INTERVALS),
            "realized_da_price": (4, INTERVALS),
            "realized_rt_price": (4, INTERVALS),
            "realized_workload": (INTERVALS,),
            "realized_pv": (4, INTERVALS),
            "physical_multiplier": (4,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if name == "timestamps_utc":
                array = np.asarray(value).copy()
                if array.shape != shape or len(np.unique(array)) != INTERVALS:
                    raise R4DataError("timestamps_utc must contain 96 unique origins")
                array.setflags(write=False)
            else:
                array = _finite(value, name=name, shape=shape)
            object.__setattr__(self, name, array)
        pue = (
            np.ones((INTERVALS, 4), dtype=float)
            if self.realized_pue is None
            else _finite(self.realized_pue, name="realized_pue", shape=(INTERVALS, 4))
        )
        object.__setattr__(self, "realized_pue", pue)
        object.__setattr__(
            self,
            "physical_twin_metadata",
            _freeze_mapping(self.physical_twin_metadata),
        )
        if (self.realized_workload < 0).any() or (self.realized_pv < 0).any():
            raise R4DataError("realized workload and PV must be nonnegative")
        if (self.physical_multiplier <= 0).any():
            raise R4DataError("physical multipliers must be strictly positive")
        issue = _utc(self.issue_time_utc, name="issue_time_utc")
        required = (
            "market",
            "workload",
            "physical_static",
            "physical_dynamic",
            "initial_soc",
        )
        missing = set(required).difference(self.feature_provenance)
        if missing:
            raise R4DataError(f"missing feature provenance: {sorted(missing)}")
        normalized_provenance: dict[str, Mapping[str, Any]] = {}
        for name in required:
            record = dict(self.feature_provenance[name])
            available = record.get("available_at_utc")
            source_hash = str(record.get("source_hash", ""))
            if not source_hash:
                raise R4DataError(f"feature provenance {name!r} requires source_hash")
            if isinstance(available, (list, tuple, np.ndarray)):
                values = tuple(_utc(item, name=f"{name}.available_at_utc") for item in available)
                if len(values) not in (1, INTERVALS):
                    raise R4DataError(
                        f"{name}.available_at_utc must contain 1 or 96 timestamps"
                    )
            else:
                values = (_utc(available, name=f"{name}.available_at_utc"),)
            if max(values) > issue:
                raise R4DataError(
                    f"feature {name!r} was not available at issue time"
                )
            record["available_at_utc"] = tuple(item.isoformat() for item in values)
            record["source_hash"] = source_hash
            normalized_provenance[name] = MappingProxyType(record)
        object.__setattr__(
            self, "feature_provenance", MappingProxyType(normalized_provenance)
        )
        object.__setattr__(self, "source_hashes", _freeze_mapping(self.source_hashes))
        operating = _date(self.operating_date)
        history = tuple(str(item)[:10] for item in self.history_dates)
        if any(_date(item) >= operating for item in history):
            raise R4DataError("history dates must strictly precede the operating day")
        object.__setattr__(self, "history_dates", history)

    @property
    def split(self) -> str:
        return temporal_split(self.operating_date)

    @property
    def sealed(self) -> bool:
        return self.split == "sealed_selection"

    @property
    def physical(self) -> np.ndarray:
        """Compatibility alias for the *static* frozen-twin input only."""

        return self.physical_static


def _json_field(bundle: Any, name: str, *, default: Any = None) -> Any:
    if name not in bundle.files:
        if default is not None:
            return default
        raise R4DataError(f"R4 bundle is missing required field {name!r}")
    try:
        return json.loads(str(bundle[name].item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise R4DataError(f"R4 bundle field {name!r} is not valid JSON") from exc


def load_r4_day(
    bundle_path: Path | str,
    settlement_path: Path | str | None = None,
    history_paths: Sequence[Path | str] = (),
    *,
    allow_sealed_selection: bool = False,
    capacity_work_units: Any | None = None,
    physical_twin_core: Any | None = None,
) -> R4SelectionDay:
    """Load an R4 day after path-level locked/sealed gates and provenance checks."""

    path = _guard_path_before_open(
        bundle_path, allow_sealed_selection=allow_sealed_selection
    )
    settlement = (
        None
        if settlement_path is None
        else _guard_path_before_open(
            settlement_path, allow_sealed_selection=allow_sealed_selection
        )
    )
    guarded_history = tuple(
        _guard_path_before_open(item, allow_sealed_selection=False)
        for item in history_paths
    )
    with np.load(path, allow_pickle=False) as bundle:
        operating_date = str(bundle["operating_date"].item())[:10]
        split = temporal_split(operating_date)
        if split == "locked_test":
            raise LockedTestAccessError("locked-test date found inside R4 bundle")
        if split == "sealed_selection" and not allow_sealed_selection:
            raise SealedSelectionAccessError("2024 payload is sealed")
        if bool(bundle["locked_test"].item()):
            raise LockedTestAccessError("locked_test=true found inside R4 bundle")
        issue_time = str(bundle["issue_time_utc"].item())
        source_hashes = _json_field(bundle, "source_hashes_json", default={})
        if "feature_provenance_json" in bundle.files:
            provenance = dict(_json_field(bundle, "feature_provenance_json"))
        else:
            # The verified selection bundles predate the R4 field name.  They
            # expose one immutable issue_time_utc and source hashes instead.
            # Derive the same fail-closed provenance contract without opening
            # another source or fabricating a later availability timestamp.
            source_hashes = dict(source_hashes)
            provenance = {
                "market": {
                    "available_at_utc": issue_time,
                    "source_hash": source_hashes.get(
                        "frozen_nyiso_twins",
                        source_hashes.get("materialization_contract", ""),
                    ),
                    "derived_role": "issue_time_price_pool_and_da_curve",
                },
                "workload": {
                    "available_at_utc": issue_time,
                    "source_hash": source_hashes.get(
                        "nlr295_workload",
                        source_hashes.get("workload_unit_bridge_contract", ""),
                    ),
                    "derived_role": "issue_time_workload_pool",
                },
                "physical": {
                    "available_at_utc": issue_time,
                    "source_hash": source_hashes.get(
                        "frozen_physical_artifact",
                        source_hashes.get("nasa_derived_pv", ""),
                    ),
                    "derived_role": "frozen_physical_and_issue_time_pv_pool",
                },
                "initial_soc": {
                    "available_at_utc": issue_time,
                    "source_hash": source_hashes.get(
                        "materialization_contract",
                        source_hashes.get("frozen_physical_artifact", ""),
                    ),
                    "derived_role": "issue_time_initial_soc",
                },
            }
        # Older issue-safe bundle manifests used one physical provenance key.
        # Splitting that record labels the two tensor roles without claiming a
        # new source or inventing a later availability timestamp.
        if "physical" in provenance:
            source = dict(provenance["physical"])
            provenance.setdefault(
                "physical_static",
                {**source, "derived_role": "frozen_multiplier_static"},
            )
            provenance.setdefault(
                "physical_dynamic",
                {
                    **source,
                    "derived_role": (
                        "issue_time_pv_distribution_and_capacity_constrained_"
                        "workload_utilization"
                    ),
                },
            )
        timestamps = np.asarray(bundle["timestamps_utc"]).copy()
        da = np.asarray(bundle["da_price"], dtype=float)
        rt_pool = np.asarray(bundle["rt_price_pool"], dtype=float)
        workload_pool = np.asarray(bundle["workload_pool"], dtype=float)
        pv_pool = np.asarray(bundle["pv_pool"], dtype=float)
        # Realized workload/PV belong to the causal selection bundle.  The
        # separate official settlement seam contains prices only.
        realized_workload = np.asarray(bundle["workload"], dtype=float)
        realized_pv = np.asarray(bundle["pv"], dtype=float)
        multiplier = np.asarray(bundle["physical_multiplier"], dtype=float)
        initial_soc = np.asarray(bundle["initial_soc_mwh_by_zone"], dtype=float)

    if settlement is None:
        raise R4DataError("R4 requires a separate official realized settlement path")
    with np.load(settlement, allow_pickle=False) as realized:
        if not np.array_equal(np.asarray(realized["timestamps_utc"]), timestamps):
            raise R4DataError("settlement timestamps do not match the R4 bundle")
        realized_da = np.asarray(realized["da_lbmp"], dtype=float)
        realized_rt = np.asarray(realized["rt_lbmp"], dtype=float)

    rt_mean = rt_pool.mean(axis=0).T
    rt_std = rt_pool.std(axis=0).T
    da_tz = da.T
    dart_tz = da_tz - rt_mean
    market = np.concatenate((da_tz, rt_mean, dart_tz, rt_std), axis=-1)
    workload_quantiles = np.quantile(workload_pool, (0.1, 0.5, 0.9), axis=0)
    workload = np.stack(
        (
            workload_pool.mean(axis=0),
            workload_pool.std(axis=0),
            workload_quantiles[0],
            workload_quantiles[1],
            workload_quantiles[2],
        ),
        axis=-1,
    )
    pv_mean = pv_pool.mean(axis=0).T
    pv_std = pv_pool.std(axis=0).T
    capacity = (
        np.ones(4, dtype=float)
        if capacity_work_units is None
        else np.asarray(capacity_work_units, dtype=float)
    )
    if capacity.shape != (4,) or not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise R4DataError("capacity_work_units must be four finite positive values")
    total_issue = np.minimum(np.maximum(workload[:, 0], 0.0), float(capacity.sum()))
    issue_work = total_issue[:, None] * capacity[None] / float(capacity.sum())
    issue_utilization = issue_work / capacity[None]
    physical_dynamic = np.concatenate((pv_mean, pv_std, issue_utilization), axis=-1)
    realized_pue = np.ones((INTERVALS, 4), dtype=float)
    twin_metadata: Mapping[str, Any] = {}
    if physical_twin_core is not None:
        realized_total = np.minimum(
            np.maximum(realized_workload, 0.0), float(capacity.sum())
        )
        realized_work = realized_total[:, None] * capacity[None] / float(capacity.sum())
        realized_utilization = realized_work / capacity[None]
        import torch

        with torch.inference_mode():
            twin_output = physical_twin_core(
                torch.as_tensor(multiplier[None], dtype=torch.float64),
                torch.as_tensor(realized_utilization[None], dtype=torch.float64),
            )
        realized_pue = np.asarray(twin_output["pue"][0].cpu(), dtype=float)
        twin_metadata = dict(getattr(physical_twin_core, "artifact_metadata", {}))
    history_dates: list[str] = []
    operating = _date(operating_date)
    for history in guarded_history:
        dates = _path_dates(history)
        if not dates or dates[-1] >= operating:
            raise R4DataError(
                f"history path must encode a date before {operating_date}: {history}"
            )
        history_dates.append(dates[-1].isoformat())
    return R4SelectionDay(
        operating_date=operating_date,
        timestamps_utc=timestamps,
        issue_time_utc=issue_time,
        market=market,
        workload=workload,
        physical_static=multiplier,
        physical_dynamic=physical_dynamic,
        initial_soc_mwh=initial_soc,
        da_price=da,
        rt_price_pool=rt_pool,
        workload_pool=workload_pool,
        pv_pool=pv_pool,
        realized_da_price=realized_da,
        realized_rt_price=realized_rt,
        realized_workload=realized_workload,
        realized_pv=realized_pv,
        physical_multiplier=multiplier,
        feature_provenance=provenance,
        source_hashes=source_hashes,
        history_dates=tuple(history_dates),
        locked_test=False,
        realized_pue=realized_pue,
        physical_twin_metadata=twin_metadata,
    )


__all__ = [
    "DiscoveredDay",
    "EMPIRICAL_SCENARIOS",
    "INTERVALS",
    "LOCKED_TEST_YEAR",
    "LockedTestAccessError",
    "NYISO_ZONES",
    "R4DataError",
    "R4SelectionDay",
    "SCENARIOS",
    "SEALED_SELECTION_END",
    "SEALED_SELECTION_START",
    "SealedSelectionAccessError",
    "TRAIN_END",
    "TRAIN_START",
    "VALIDATION_END",
    "VALIDATION_START",
    "discover_eligible_days",
    "load_r4_day",
    "temporal_split",
]
