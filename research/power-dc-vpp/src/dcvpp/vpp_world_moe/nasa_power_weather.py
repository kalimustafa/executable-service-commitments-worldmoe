"""Causal lag-persistence weather prior from the frozen NASA POWER table.

The registered NASA POWER product is a reconstructed historical weather
record, not an archived operational forecast.  This adapter therefore exposes
only a deterministic lag-persistence prior: at each rolling decision origin it
selects the latest *complete* historical source day whose every required value
was deployable by that origin, then maps that one day's time-of-day profile to
the operating day.  Realised values from the operating day are never inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .weather import WeatherCausalityError, WeatherQueryResult


UTC = timezone.utc
NASA_POWER_LAG_SOURCE = "nasa_power_lag_persistence"
NASA_POWER_SOURCE_VERSION = "NASA_POWER_hourly_reconstruction_lag_persistence_v1"
NASA_POWER_FEATURES = ("air_temperature_2m_k", "downward_shortwave_wm2")
NASA_POWER_SITE_ORDER = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
DEVELOPMENT_START = pd.Timestamp("2021-01-01T00:00:00Z")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2025-01-01T00:00:00Z")
SLOTS_PER_DAY = 96


@dataclass(frozen=True)
class LagPersistenceArray:
    """Dense result used by the hot corpus path without DataFrame overhead."""

    values: np.ndarray
    mask: np.ndarray
    source_day: date | None
    source_day_visible_at_utc: pd.Timestamp | None


class NasaPowerLagPersistenceStore:
    """Read-only, development-only NASA POWER lag-persistence store."""

    source = NASA_POWER_LAG_SOURCE

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"NASA POWER weather table is missing: {self.path}")

        columns = [
            "timestamp_utc",
            "source_hour_utc",
            "deployable_available_at_utc",
            "zone",
            "T2M",
            "ALLSKY_SFC_SW_DWN",
            "resampling_method",
            "evidence_role",
            "operational_weather_forecast_claimed",
        ]
        # Predicate pushdown is part of the sealed-test boundary: 2025 values
        # are not materialised into the development process even though the
        # registered physical file also contains the final-test year.
        frame = pd.read_parquet(
            self.path,
            columns=columns,
            filters=[
                ("timestamp_utc", ">=", DEVELOPMENT_START.to_pydatetime()),
                ("timestamp_utc", "<", DEVELOPMENT_END_EXCLUSIVE.to_pydatetime()),
            ],
        )
        for column in (
            "timestamp_utc",
            "source_hour_utc",
            "deployable_available_at_utc",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        if frame.empty or frame["timestamp_utc"].ge(DEVELOPMENT_END_EXCLUSIVE).any():
            raise WeatherCausalityError("NASA POWER development scan crossed into 2025")
        if frame["timestamp_utc"].lt(DEVELOPMENT_START).any():
            raise WeatherCausalityError("NASA POWER development scan opened pre-2021 data")
        if set(frame["zone"].astype(str)) != set(NASA_POWER_SITE_ORDER):
            raise ValueError("NASA POWER table does not contain the four frozen NYISO sites")
        if frame["operational_weather_forecast_claimed"].astype(bool).any():
            raise ValueError("NASA POWER table incorrectly claims operational forecasts")
        if set(frame["evidence_role"].astype(str)) != {"reconstructed"}:
            raise ValueError("NASA POWER evidence role must remain reconstructed")
        if set(frame["resampling_method"].astype(str)) != {
            "hourly_value_held_constant"
        }:
            raise ValueError("NASA POWER temporal reconstruction contract changed")
        if frame.duplicated(["timestamp_utc", "zone"]).any():
            raise ValueError("NASA POWER table has duplicate timestamp/zone rows")

        frame = frame.sort_values(["timestamp_utc", "zone"], kind="mergesort")
        frame["source_day"] = frame["timestamp_utc"].dt.date
        self._day_values: dict[date, np.ndarray] = {}
        self._day_visible_at: dict[date, pd.Timestamp] = {}
        for source_day, group in frame.groupby("source_day", sort=True):
            indexed = group.set_index(["timestamp_utc", "zone"])
            expected_times = pd.date_range(
                pd.Timestamp(datetime.combine(source_day, time.min, tzinfo=UTC)),
                periods=SLOTS_PER_DAY,
                freq="15min",
            )
            expected_index = pd.MultiIndex.from_product(
                [expected_times, NASA_POWER_SITE_ORDER],
                names=["timestamp_utc", "zone"],
            )
            if len(group) != SLOTS_PER_DAY * len(NASA_POWER_SITE_ORDER):
                continue
            ordered = indexed.reindex(expected_index)
            raw = ordered[["T2M", "ALLSKY_SFC_SW_DWN"]].to_numpy(dtype=np.float32)
            if not np.isfinite(raw).all():
                continue
            values = raw.reshape(SLOTS_PER_DAY, len(NASA_POWER_SITE_ORDER), 2)
            values[..., 0] += np.float32(273.15)
            self._day_values[source_day] = values
            self._day_visible_at[source_day] = pd.Timestamp(
                group["deployable_available_at_utc"].max()
            )
        self._complete_days = tuple(sorted(self._day_values))
        if not self._complete_days:
            raise ValueError("NASA POWER table contains no complete causal source day")

    def artifact_count(self) -> int:
        """The single registered local Parquet is the formal source artifact."""

        return 1

    def lineage(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "source": "NASA POWER referenced hourly reconstruction",
            "adapter": NASA_POWER_SOURCE_VERSION,
            "evidence_role": "reconstructed",
            "quality": "frozen_derived",
            "operational_weather_forecast_claimed": False,
            "loaded_window_utc": [
                DEVELOPMENT_START.isoformat(),
                DEVELOPMENT_END_EXCLUSIVE.isoformat(),
            ],
            "complete_source_days": len(self._complete_days),
            "first_source_day": self._complete_days[0].isoformat(),
            "last_source_day": self._complete_days[-1].isoformat(),
            "availability": "complete_source_day_max_deployable_available_at_lte_origin",
            "future_realized_values_allowed": False,
        }

    @staticmethod
    def _timestamps(
        decision_origin: datetime | pd.Timestamp | str,
        valid_times: Sequence[datetime | pd.Timestamp | str],
    ) -> tuple[pd.Timestamp, tuple[pd.Timestamp, ...], date]:
        origin = pd.Timestamp(decision_origin)
        if origin.tzinfo is None:
            raise WeatherCausalityError("weather decision origin must be timezone-aware")
        origin = origin.tz_convert("UTC")
        targets = tuple(pd.Timestamp(value) for value in valid_times)
        if not targets:
            raise ValueError("valid_times cannot be empty")
        if any(value.tzinfo is None for value in targets):
            raise WeatherCausalityError("weather valid times must be timezone-aware")
        targets = tuple(value.tz_convert("UTC") for value in targets)
        if origin.year >= 2025 or any(value.year >= 2025 for value in targets):
            raise WeatherCausalityError("NASA POWER development adapter cannot open 2025")
        operating_days = {value.date() for value in targets}
        if len(operating_days) != 1:
            raise ValueError("lag-persistence query must stay within one operating day")
        return origin, targets, next(iter(operating_days))

    def _latest_visible_day(self, operating_day: date, origin: pd.Timestamp) -> date | None:
        for source_day in reversed(self._complete_days):
            if source_day >= operating_day:
                continue
            if self._day_visible_at[source_day] <= origin:
                return source_day
        return None

    def query_causal_array_as_of(
        self,
        decision_origin: datetime | pd.Timestamp | str,
        *,
        valid_times: Sequence[datetime | pd.Timestamp | str],
        source: str = NASA_POWER_LAG_SOURCE,
        zones: Sequence[str] = NASA_POWER_SITE_ORDER,
        features: Sequence[str] = NASA_POWER_FEATURES,
    ) -> LagPersistenceArray:
        origin, targets, operating_day = self._timestamps(decision_origin, valid_times)
        if source != self.source:
            raise KeyError(f"unsupported NASA POWER weather source: {source}")
        if tuple(zones) != NASA_POWER_SITE_ORDER:
            raise KeyError("NASA POWER weather query must preserve frozen site order")
        if tuple(features) != NASA_POWER_FEATURES:
            raise KeyError("NASA POWER weather query must preserve frozen feature order")
        shape = (len(targets), len(NASA_POWER_SITE_ORDER), len(NASA_POWER_FEATURES))
        source_day = self._latest_visible_day(operating_day, origin)
        if source_day is None:
            return LagPersistenceArray(
                values=np.zeros(shape, dtype=np.float32),
                mask=np.zeros(shape, dtype=bool),
                source_day=None,
                source_day_visible_at_utc=None,
            )
        slots: list[int] = []
        for target in targets:
            if target.second or target.microsecond or target.minute % 15:
                raise ValueError("NASA POWER weather targets must use the 15-minute grid")
            slots.append(target.hour * 4 + target.minute // 15)
        values = self._day_values[source_day][np.asarray(slots)].copy()
        return LagPersistenceArray(
            values=values,
            mask=np.isfinite(values),
            source_day=source_day,
            source_day_visible_at_utc=self._day_visible_at[source_day],
        )

    def query_causal_hold_as_of(
        self,
        decision_origin: datetime | pd.Timestamp | str,
        *,
        valid_times: Sequence[datetime | pd.Timestamp | str],
        source: str = NASA_POWER_LAG_SOURCE,
        zones: Sequence[str] = NASA_POWER_SITE_ORDER,
        features: Sequence[str] = NASA_POWER_FEATURES,
    ) -> WeatherQueryResult:
        """Compatibility wrapper retaining explicit per-row causal lineage."""

        origin, targets, _ = self._timestamps(decision_origin, valid_times)
        dense = self.query_causal_array_as_of(
            origin,
            valid_times=targets,
            source=source,
            zones=zones,
            features=features,
        )
        rows: list[dict[str, object]] = []
        if dense.source_day is not None and dense.source_day_visible_at_utc is not None:
            source_start = pd.Timestamp(
                datetime.combine(dense.source_day, time.min, tzinfo=UTC)
            )
            for time_index, target in enumerate(targets):
                slot = target.hour * 4 + target.minute // 15
                native_time = source_start + pd.Timedelta(minutes=15 * slot)
                for zone_index, zone in enumerate(zones):
                    for feature_index, feature in enumerate(features):
                        rows.append(
                            {
                                "source": source,
                                "source_version": NASA_POWER_SOURCE_VERSION,
                                "valid_time_utc": target,
                                "native_valid_time_utc": native_time,
                                "zone": zone,
                                "feature": feature,
                                "value": float(
                                    dense.values[time_index, zone_index, feature_index]
                                ),
                                "run_init_utc": source_start,
                                "visible_at_utc": dense.source_day_visible_at_utc,
                                "forecast_hour": np.nan,
                                "evidence_role": "reconstructed",
                                "quality": "frozen_derived",
                                "operational_weather_forecast_claimed": False,
                            }
                        )
        return WeatherQueryResult(
            decision_origin_utc=origin,
            frame=pd.DataFrame.from_records(rows),
            requested_sources=(source,),
            requested_valid_times=targets,
            requested_zones=tuple(zones),
            requested_features=tuple(features),
        )


__all__ = [
    "DEVELOPMENT_END_EXCLUSIVE",
    "DEVELOPMENT_START",
    "LagPersistenceArray",
    "NASA_POWER_FEATURES",
    "NASA_POWER_LAG_SOURCE",
    "NASA_POWER_SITE_ORDER",
    "NASA_POWER_SOURCE_VERSION",
    "NasaPowerLagPersistenceStore",
]
