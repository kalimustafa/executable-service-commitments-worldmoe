from __future__ import annotations

"""Isolated post-v1 forecasting extensions required by the revised protocol.

This module deliberately does not participate in ``paper-forecast-v1``.  It
reads a terminal v1 manifest only as frozen lineage and writes to the separate
``artifacts/real_forecasting_extensions`` namespace.  The production models
are imported without modifying their source files.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.forecast_calibration import (
    DEFAULT_QUANTILES,
    apply_additive_conformal_quantiles,
    fit_additive_conformal_quantiles,
    negative_binomial_metrics,
    negative_binomial_quantiles,
    positive_volume_nll,
    positive_volume_quantiles,
    quantile_metrics,
    rearrange_quantiles,
)
from dcvpp.models.energy import DGPTE
from dcvpp.models.workload import (
    CfCW,
    gamma_volume_nll,
    lognormal_volume_nll,
    negative_binomial_nll,
)


PREREGISTERED_SEEDS = (20260809, 20260819, 20260829)
DEFAULT_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
DEFAULT_COHORTS = (
    "A_critical_interactive",
    "B_latency_sensitive",
    "C_preemptible_training",
    "D_elastic_batch",
)
QUANTILES = np.asarray(DEFAULT_QUANTILES, dtype="float32")
ENERGY_TASKS = ("da_lbmp", "rt_lbmp")
SPLITS = ("train", "validation", "locked_test")
TENSOR_CACHE_BUILDER_REVISION = "referenced_tensor_builder_v1_20260811"
ENERGY_TRAINER_CONTRACT_REVISION = "paper_energy_extension_trainer_v1"
WORKLOAD_TRAINER_CONTRACT_REVISION = "paper_workload_extension_trainer_v1"
LEGACY_SCHEDULER_ENERGY_CHECKPOINTS = {
    ("da_lbmp", "dgpt_e_primary", 20260809): {
        "resume_fingerprint": "ed8ed58c6c3817d5cb37947c8a135db1df7c6f778d42b8946070f383d9722dc9",
        "sha256": "b78eadb6722bcc1861959f240efaa2ad1df1b57358455f35da35f4e0574de5e2",
    },
    ("da_lbmp", "dgpt_e_primary", 20260819): {
        "resume_fingerprint": "6057e6c4eb77548d621bdf44339d999f6570d139147bf2ab3473bc79268f09c3",
        "sha256": "1b30c14a5a16203f7e00118042c8464fa8607fac2f847ba2d5733d88dc838c64",
    },
    ("da_lbmp", "dgpt_e_primary", 20260829): {
        "resume_fingerprint": "28f53258abf5fc382bab1b3bc8e6ac1af31fca2792eda6aec355f683b4d528ad",
        "sha256": "18b3432b909ca80f8c101625570fae33fbf8629e73292523d191f4412ad25d08",
    },
}

ENERGY_HISTORY_BASE_FEATURES = (
    "da_lbmp",
    "rt_lbmp",
    "rt_spread",
    "system_load",
    "zone_load",
    "load_forecast_error",
    "zone_weather",
    "zone_pv",
)
ENERGY_KNOWN_FUTURE_FEATURES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "is_dst",
    "is_issue_interval",
    "is_gate_interval",
    "hours_from_issue",
    "hours_from_gate",
    "solar_elevation_scaled",
    "solar_azimuth_sin",
    "solar_azimuth_cos",
)
ORACLE_FUTURE_FEATURES = (
    "ORACLE_future_realized_weather",
    "ORACLE_future_realized_weather_missing",
)

ZONE_COORDINATES = {
    "WEST": (42.8864, -78.8784),
    "CAPITL": (42.6526, -73.7562),
    "N.Y.C.": (40.7128, -74.0060),
    "LONGIL": (40.7891, -73.1350),
}


def _preprocess_thread_count() -> int:
    requested = int(os.environ.get("DCVPP_PREPROCESS_THREADS", "8"))
    return max(1, min(requested, os.cpu_count() or 1))


def _configure_duckdb_threads(connection: Any) -> None:
    connection.execute(f"SET threads TO {_preprocess_thread_count()}")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if not slug or slug in {".", ".."}:
        raise ValueError("run/model identifiers must contain a safe character")
    return slug


def _utc_series(
    values: pd.Series, *, name: str, allow_missing: bool = False
) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="raise")
    if result.isna().any() and not allow_missing:
        raise ValueError(f"{name} contains a missing timestamp")
    return result


def _utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    if pd.isna(result):
        raise ValueError(f"{name} is missing")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_gate(seeds: Sequence[int], *, paper_evidence: bool) -> tuple[int, ...]:
    values = tuple(int(value) for value in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("extension seeds must be nonempty and unique")
    if paper_evidence and values != PREREGISTERED_SEEDS:
        raise ValueError(
            f"paper forecast extensions require preregistered seeds {PREREGISTERED_SEEDS}"
        )
    return values


def _atomic_torch(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _atomic_pickle(payload: Any, path: Path) -> dict[str, Any]:
    """Persist a trusted, run-local tensor cache atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".pkl", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


@dataclass(frozen=True)
class EnergySplitTensors:
    history: np.ndarray
    future_known: np.ndarray
    future_oracle_weather: np.ndarray
    target_normalized: np.ndarray
    target_original: np.ndarray
    target_spike_mask: np.ndarray
    origins_utc: tuple[str, ...]
    target_timestamps_utc: tuple[tuple[str, ...], ...]
    issue_audit: tuple[dict[str, Any], ...]

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "history": _array_sha256(self.history),
                "future_known": _array_sha256(self.future_known),
                "future_oracle_weather": _array_sha256(self.future_oracle_weather),
                "target_normalized": _array_sha256(self.target_normalized),
                "target_original": _array_sha256(self.target_original),
                "target_spike_mask": _array_sha256(self.target_spike_mask),
                "origins_utc": self.origins_utc,
                "target_timestamps_utc": self.target_timestamps_utc,
                "issue_audit": self.issue_audit,
            }
        )


@dataclass(frozen=True)
class EnergyTaskTensors:
    task: str
    zones: tuple[str, ...]
    history_feature_names: tuple[str, ...]
    known_future_feature_names: tuple[str, ...]
    oracle_future_feature_names: tuple[str, ...]
    splits: Mapping[str, EnergySplitTensors]
    input_center: np.ndarray
    input_scale: np.ndarray
    target_center: np.ndarray
    target_scale: np.ndarray
    spike_thresholds: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "schema": "issue_safe_dgpte_tensors_v1",
                "task": self.task,
                "zones": self.zones,
                "history_features": self.history_feature_names,
                "known_future_features": self.known_future_feature_names,
                "oracle_future_features": self.oracle_future_feature_names,
                "splits": {key: self.splits[key].fingerprint for key in SPLITS},
                "input_center": _array_sha256(self.input_center),
                "input_scale": _array_sha256(self.input_scale),
                "target_center": _array_sha256(self.target_center),
                "target_scale": _array_sha256(self.target_scale),
                "spike_thresholds": {
                    key: _array_sha256(np.asarray(value))
                    for key, value in sorted(self.spike_thresholds.items())
                },
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class WorkloadSplitTensors:
    event_history: np.ndarray
    elapsed_hours: np.ndarray
    regular_grid_history: np.ndarray
    count_target: np.ndarray
    volume_target: np.ndarray
    origins_utc: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "event_history": _array_sha256(self.event_history),
                "elapsed_hours": _array_sha256(self.elapsed_hours),
                "regular_grid_history": _array_sha256(self.regular_grid_history),
                "count_target": _array_sha256(self.count_target),
                "volume_target": _array_sha256(self.volume_target),
                "origins_utc": self.origins_utc,
            }
        )


@dataclass(frozen=True)
class WorkloadExtensionTensors:
    event_feature_names: tuple[str, ...]
    regular_grid_feature_names: tuple[str, ...]
    cohorts: tuple[str, ...]
    splits: Mapping[str, WorkloadSplitTensors]
    metadata: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "schema": "cfc_w_ablation_tensors_v1",
                "event_feature_names": self.event_feature_names,
                "regular_grid_feature_names": self.regular_grid_feature_names,
                "cohorts": self.cohorts,
                "splits": {key: self.splits[key].fingerprint for key in SPLITS},
                "metadata": dict(self.metadata),
            }
        )


def _validate_panel(
    panel: pd.DataFrame,
    *,
    zones: Sequence[str],
    availability_columns: Mapping[str, str],
) -> pd.DataFrame:
    required = {"timestamp_utc", "zone", *ENERGY_HISTORY_BASE_FEATURES}
    required.update(availability_columns.values())
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"energy panel is missing issue-safety columns: {missing}")
    frame = panel.copy()
    frame["timestamp_utc"] = _utc_series(frame["timestamp_utc"], name="timestamp_utc")
    for column in set(availability_columns.values()):
        frame[column] = _utc_series(frame[column], name=column, allow_missing=True)
    frame["zone"] = frame["zone"].astype(str)
    unknown = sorted(set(frame["zone"]) - set(zones))
    if unknown:
        raise ValueError(f"energy panel contains unknown zones: {unknown}")
    if set(frame["zone"]) != set(zones):
        raise ValueError("energy panel must contain every configured zone")
    if frame.duplicated(["timestamp_utc", "zone"]).any():
        raise ValueError("energy panel contains duplicate timestamp-zone rows")
    for feature in ENERGY_HISTORY_BASE_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        availability_column = availability_columns[feature]
        if (frame[feature].notna() & frame[availability_column].isna()).any():
            raise ValueError(
                f"{feature} contains an observed value without an availability timestamp"
            )
    return frame.sort_values(["timestamp_utc", "zone"], kind="stable").reset_index(drop=True)


def _validate_origins(origins: pd.DataFrame, *, task: str) -> pd.DataFrame:
    required = {
        "task",
        "split",
        "issue_timestamp_utc",
        "gate_timestamp_utc",
        "target_start_utc",
    }
    missing = sorted(required - set(origins.columns))
    if missing:
        raise ValueError(f"energy origins are missing columns: {missing}")
    frame = origins.copy()
    if set(frame["task"].astype(str)) != {task}:
        raise ValueError("DA and RT origins must be passed in separate task tables")
    if set(frame["split"].astype(str)) != set(SPLITS):
        raise ValueError(f"energy origins require exactly the splits {SPLITS}")
    for column in ("issue_timestamp_utc", "gate_timestamp_utc", "target_start_utc"):
        frame[column] = _utc_series(frame[column], name=column)
    if "feature_cutoff_utc" in frame:
        frame["feature_cutoff_utc"] = _utc_series(
            frame["feature_cutoff_utc"], name="feature_cutoff_utc"
        )
    else:
        frame["feature_cutoff_utc"] = frame["issue_timestamp_utc"]
    if (frame["feature_cutoff_utc"] > frame["issue_timestamp_utc"]).any():
        raise ValueError("feature cutoff must not be later than the forecast issue")
    if (frame["issue_timestamp_utc"] >= frame["target_start_utc"]).any():
        raise ValueError("every forecast issue must precede its target horizon")
    if frame.duplicated(["task", "target_start_utc"]).any():
        raise ValueError("energy origins contain duplicate task/target horizons")
    return frame.sort_values("target_start_utc", kind="stable").reset_index(drop=True)


def _training_statistics(
    frame: pd.DataFrame,
    *,
    zones: tuple[str, ...],
    availability_columns: Mapping[str, str],
    train_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    center = np.zeros((len(zones), len(ENERGY_HISTORY_BASE_FEATURES)), dtype="float32")
    scale = np.ones_like(center)
    thresholds: dict[str, np.ndarray] = {
        task: np.zeros(len(zones), dtype="float32") for task in ENERGY_TASKS
    }
    for zone_index, zone in enumerate(zones):
        zone_rows = frame.loc[frame["zone"].eq(zone)]
        for feature_index, feature in enumerate(ENERGY_HISTORY_BASE_FEATURES):
            available = zone_rows[availability_columns[feature]].le(train_end)
            historical = zone_rows["timestamp_utc"].le(train_end)
            values = zone_rows.loc[available & historical, feature].dropna().to_numpy(dtype="float64")
            if not len(values):
                raise ValueError(f"no training-available values for {zone}/{feature}")
            center[zone_index, feature_index] = float(np.mean(values))
            deviation = float(np.std(values))
            scale[zone_index, feature_index] = deviation if deviation > 1e-8 else 1.0
        for product in ENERGY_TASKS:
            available = zone_rows[availability_columns[product]].le(train_end)
            historical = zone_rows["timestamp_utc"].le(train_end)
            values = zone_rows.loc[available & historical, product].dropna().to_numpy(dtype="float64")
            if not len(values):
                raise ValueError(f"no training values for {zone}/{product} spike threshold")
            thresholds[product][zone_index] = float(np.quantile(values, 0.99, method="linear"))
    target_center = center[:, ENERGY_HISTORY_BASE_FEATURES.index("da_lbmp")].copy()
    target_scale = scale[:, ENERGY_HISTORY_BASE_FEATURES.index("da_lbmp")].copy()
    return center, scale, thresholds, target_center, target_scale


def _solar_geometry(
    timestamps: pd.DatetimeIndex,
    *,
    zones: tuple[str, ...],
    coordinates: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    try:
        import pvlib
    except ImportError as exc:
        raise RuntimeError("deterministic known-future solar geometry requires pvlib") from exc
    result = np.zeros((len(timestamps), len(zones), 3), dtype="float32")
    for zone_index, zone in enumerate(zones):
        if zone not in coordinates:
            raise ValueError(f"zone coordinates are missing for {zone}")
        latitude, longitude = coordinates[zone]
        solar = pvlib.solarposition.get_solarposition(
            timestamps, latitude=float(latitude), longitude=float(longitude)
        )
        elevation = solar["apparent_elevation"].to_numpy(dtype="float64") / 90.0
        azimuth = np.deg2rad(solar["azimuth"].to_numpy(dtype="float64"))
        result[:, zone_index, 0] = elevation.astype("float32")
        result[:, zone_index, 1] = np.sin(azimuth).astype("float32")
        result[:, zone_index, 2] = np.cos(azimuth).astype("float32")
    if not np.isfinite(result).all():
        raise ValueError("known-future solar geometry contains non-finite values")
    return result


def _known_future(
    timestamps: pd.DatetimeIndex,
    *,
    issue: pd.Timestamp,
    gate: pd.Timestamp,
    zones: tuple[str, ...],
    coordinates: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    local = timestamps.tz_convert("America/New_York")
    hour = local.hour.to_numpy(dtype="float64") + local.minute.to_numpy(dtype="float64") / 60.0
    dow = local.dayofweek.to_numpy(dtype="float64")
    month = local.month.to_numpy(dtype="float64") - 1.0
    from pandas.tseries.holiday import USFederalHolidayCalendar

    holiday_dates = USFederalHolidayCalendar().holidays(
        start=local.min().tz_localize(None).normalize(),
        end=local.max().tz_localize(None).normalize(),
    )
    holiday = local.tz_localize(None).normalize().isin(holiday_dates).astype("float32")
    dst = np.asarray(
        [float(value.dst().total_seconds() != 0) for value in local.to_pydatetime()],
        dtype="float32",
    )
    issue_floor = issue.floor("15min")
    gate_floor = gate.floor("15min")
    common = np.column_stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * dow / 7.0),
            np.cos(2.0 * np.pi * dow / 7.0),
            np.sin(2.0 * np.pi * month / 12.0),
            np.cos(2.0 * np.pi * month / 12.0),
            holiday,
            dst,
            (timestamps == issue_floor).astype("float32"),
            (timestamps == gate_floor).astype("float32"),
            (timestamps - issue).total_seconds() / 3600.0,
            (timestamps - gate).total_seconds() / 3600.0,
        ]
    ).astype("float32")
    replicated = np.repeat(common[:, None, :], len(zones), axis=1)
    solar = _solar_geometry(timestamps, zones=zones, coordinates=coordinates)
    result = np.concatenate([replicated, solar], axis=-1).astype("float32")
    if result.shape[-1] != len(ENERGY_KNOWN_FUTURE_FEATURES):
        raise AssertionError("known-future feature contract drifted")
    return result


def build_issue_safe_energy_tensors(
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    task: str,
    train_end_utc: Any,
    history_steps: int = 672,
    horizon_steps: int = 96,
    interval_minutes: int = 15,
    zones: Sequence[str] = DEFAULT_ZONES,
    availability_columns: Mapping[str, str] | None = None,
    zone_coordinates: Mapping[str, tuple[float, float]] = ZONE_COORDINATES,
) -> EnergyTaskTensors:
    """Build separate, issue-time-safe DA or RT tensors.

    Every historical source requires an explicit publication/availability
    timestamp.  A value unavailable at an origin is treated as missing, its
    missingness bit is set, and it is filled only from an earlier value in the
    same history window or a training-only center.  Realized future weather is
    returned solely in a separately named ORACLE tensor.
    """

    if task not in ENERGY_TASKS:
        raise ValueError(f"task must be one of {ENERGY_TASKS}")
    if history_steps < 1 or horizon_steps < 1 or interval_minutes < 1:
        raise ValueError("history, horizon and interval must be positive")
    zone_values = tuple(str(zone) for zone in zones)
    if len(set(zone_values)) != len(zone_values):
        raise ValueError("zones must be unique")
    availability = dict(
        availability_columns
        or {feature: f"{feature}_available_at_utc" for feature in ENERGY_HISTORY_BASE_FEATURES}
    )
    if set(availability) != set(ENERGY_HISTORY_BASE_FEATURES):
        raise ValueError("availability mapping must cover every historical feature exactly")
    frame = _validate_panel(panel, zones=zone_values, availability_columns=availability)
    issue_frame = _validate_origins(origins, task=task)
    train_end = _utc_timestamp(train_end_utc, name="train_end_utc")
    center, scale, thresholds, _, _ = _training_statistics(
        frame,
        zones=zone_values,
        availability_columns=availability,
        train_end=train_end,
    )
    target_index = ENERGY_HISTORY_BASE_FEATURES.index(task)
    target_center = center[:, target_index].copy()
    target_scale = scale[:, target_index].copy()
    history_names = (
        *ENERGY_HISTORY_BASE_FEATURES,
        *(f"{feature}_missing" for feature in ENERGY_HISTORY_BASE_FEATURES),
        "da_price_spike_train_only",
        "rt_price_spike_train_only",
    )
    frequency = pd.Timedelta(minutes=interval_minutes)
    indexed = {
        zone: frame.loc[frame["zone"].eq(zone)].set_index("timestamp_utc").sort_index()
        for zone in zone_values
    }
    split_buckets: dict[str, dict[str, list[Any]]] = {
        split: {
            key: []
            for key in (
                "history",
                "future_known",
                "future_oracle",
                "target_normalized",
                "target_original",
                "target_spike",
                "origin",
                "target_timestamps",
                "audit",
            )
        }
        for split in SPLITS
    }
    for row in issue_frame.to_dict("records"):
        split = str(row["split"])
        issue = _utc_timestamp(row["issue_timestamp_utc"], name="issue_timestamp_utc")
        gate = _utc_timestamp(row["gate_timestamp_utc"], name="gate_timestamp_utc")
        cutoff = _utc_timestamp(row["feature_cutoff_utc"], name="feature_cutoff_utc")
        target_start = _utc_timestamp(row["target_start_utc"], name="target_start_utc")
        history_end = cutoff.floor(f"{interval_minutes}min")
        history_times = pd.date_range(
            history_end - frequency * (history_steps - 1),
            periods=history_steps,
            freq=frequency,
            tz="UTC",
        )
        target_times = pd.date_range(
            target_start, periods=horizon_steps, freq=frequency, tz="UTC"
        )
        history_values = np.empty(
            (history_steps, len(zone_values), len(ENERGY_HISTORY_BASE_FEATURES)),
            dtype="float32",
        )
        missing_values = np.empty_like(history_values)
        spike_values = np.empty((history_steps, len(zone_values), 2), dtype="float32")
        target_original = np.empty((len(zone_values), horizon_steps), dtype="float32")
        oracle = np.empty((horizon_steps, len(zone_values), 2), dtype="float32")
        max_consumed_availability: pd.Timestamp | None = None
        unavailable_value_count = 0
        for zone_index, zone in enumerate(zone_values):
            zone_rows = indexed[zone]
            history_rows = zone_rows.reindex(history_times)
            for feature_index, feature in enumerate(ENERGY_HISTORY_BASE_FEATURES):
                values = pd.to_numeric(history_rows[feature], errors="coerce")
                available_at = pd.to_datetime(
                    history_rows[availability[feature]], utc=True, errors="coerce"
                )
                unavailable = available_at.isna() | available_at.gt(cutoff)
                unavailable_value_count += int(unavailable.sum())
                consumed = available_at.loc[~unavailable]
                if len(consumed):
                    candidate = consumed.max()
                    max_consumed_availability = (
                        candidate
                        if max_consumed_availability is None
                        else max(max_consumed_availability, candidate)
                    )
                causal = values.mask(unavailable)
                missing = causal.isna().to_numpy(dtype="float32")
                filled = causal.ffill().fillna(float(center[zone_index, feature_index]))
                history_values[:, zone_index, feature_index] = (
                    (filled.to_numpy(dtype="float32") - center[zone_index, feature_index])
                    / scale[zone_index, feature_index]
                )
                missing_values[:, zone_index, feature_index] = missing
            for product_index, product in enumerate(ENERGY_TASKS):
                raw_index = ENERGY_HISTORY_BASE_FEATURES.index(product)
                raw = (
                    history_values[:, zone_index, raw_index] * scale[zone_index, raw_index]
                    + center[zone_index, raw_index]
                )
                spike_values[:, zone_index, product_index] = (
                    raw >= thresholds[product][zone_index]
                ).astype("float32")
            target_rows = zone_rows.reindex(target_times)
            targets = pd.to_numeric(target_rows[task], errors="coerce")
            if targets.isna().any():
                raise ValueError(
                    f"{task} target contains missing values for {zone}/{target_start.isoformat()}"
                )
            target_original[zone_index] = targets.to_numpy(dtype="float32")
            weather = pd.to_numeric(target_rows["zone_weather"], errors="coerce")
            weather_missing = weather.isna().to_numpy(dtype="float32")
            weather_index = ENERGY_HISTORY_BASE_FEATURES.index("zone_weather")
            weather_filled = weather.fillna(float(center[zone_index, weather_index]))
            oracle[:, zone_index, 0] = (
                weather_filled.to_numpy(dtype="float32") - center[zone_index, weather_index]
            ) / scale[zone_index, weather_index]
            oracle[:, zone_index, 1] = weather_missing
        if max_consumed_availability is not None and max_consumed_availability > cutoff:
            raise AssertionError("issue-time gate admitted a future-published historical value")
        history_tensor = np.concatenate(
            [history_values, missing_values, spike_values], axis=-1
        ).astype("float32")
        target_normalized = (
            (target_original - target_center[:, None]) / target_scale[:, None]
        ).astype("float32")
        target_spike = (
            target_original >= thresholds[task][:, None]
        ).astype("float32")
        future_known = _known_future(
            target_times,
            issue=issue,
            gate=gate,
            zones=zone_values,
            coordinates=zone_coordinates,
        )
        if not all(
            np.isfinite(value).all()
            for value in (history_tensor, target_original, target_normalized, future_known, oracle)
        ):
            raise ValueError("constructed DGPT-E tensors contain non-finite values")
        bucket = split_buckets[split]
        bucket["history"].append(history_tensor)
        bucket["future_known"].append(future_known)
        bucket["future_oracle"].append(oracle)
        bucket["target_normalized"].append(target_normalized)
        bucket["target_original"].append(target_original)
        bucket["target_spike"].append(target_spike)
        bucket["origin"].append(issue.isoformat())
        bucket["target_timestamps"].append(tuple(value.isoformat() for value in target_times))
        bucket["audit"].append(
            {
                "task": task,
                "issue_timestamp_utc": issue.isoformat(),
                "gate_timestamp_utc": gate.isoformat(),
                "feature_cutoff_utc": cutoff.isoformat(),
                "history_max_timestamp_utc": history_times.max().isoformat(),
                "target_start_utc": target_start.isoformat(),
                "max_consumed_feature_availability_utc": (
                    max_consumed_availability.isoformat()
                    if max_consumed_availability is not None
                    else None
                ),
                "unavailable_or_absent_history_value_count": unavailable_value_count,
                "future_realized_weather_location": "isolated_ORACLE_tensor_only",
                "causal_fill": "within_window_forward_fill_then_training_center",
            }
        )
    split_tensors: dict[str, EnergySplitTensors] = {}
    for split in SPLITS:
        bucket = split_buckets[split]
        if not bucket["history"]:
            raise ValueError(f"energy tensor split is empty: {split}")
        split_tensors[split] = EnergySplitTensors(
            history=np.stack(bucket["history"]).astype("float32"),
            future_known=np.stack(bucket["future_known"]).astype("float32"),
            future_oracle_weather=np.stack(bucket["future_oracle"]).astype("float32"),
            target_normalized=np.stack(bucket["target_normalized"]).astype("float32"),
            target_original=np.stack(bucket["target_original"]).astype("float32"),
            target_spike_mask=np.stack(bucket["target_spike"]).astype("float32"),
            origins_utc=tuple(bucket["origin"]),
            target_timestamps_utc=tuple(bucket["target_timestamps"]),
            issue_audit=tuple(bucket["audit"]),
        )
    locked = split_tensors["locked_test"]
    metadata = {
        "schema_version": 1,
        "issue_safety": "per_feature_publication_timestamp_le_feature_cutoff",
        "task_isolation": "separate_target_scaler_checkpoint_calibration_and_leaderboard",
        "training_end_inclusive_utc": train_end.isoformat(),
        "history_steps": int(history_steps),
        "horizon_steps": int(horizon_steps),
        "interval_minutes": int(interval_minutes),
        "spike_threshold_policy": "per_product_per_zone_training_q99_only",
        "historical_feature_semantics": {
            "da_lbmp": "issue_available_history_only",
            "rt_lbmp": "issue_available_history_only",
            "rt_spread": "issue_available_history_only",
            "system_load": "issue_available_history_only",
            "zone_load": "issue_available_history_only",
            "load_forecast_error": "lagged_and_issue_available_only",
            "zone_weather": "lagged_realized_and_issue_available_only",
            "zone_pv": "lagged_derived_and_issue_available_only",
            "missingness": "one_flag_per_raw_historical_feature",
            "price_spike": "training_thresholds_only",
        },
        "future_realized_weather_in_deployable_tensor": False,
        "oracle_evidence_role": "oracle",
        "locked_test_target_sha256": _array_sha256(locked.target_original),
        "locked_test_origin_sha256": sha256_json(locked.origins_utc),
    }
    return EnergyTaskTensors(
        task=task,
        zones=zone_values,
        history_feature_names=tuple(history_names),
        known_future_feature_names=ENERGY_KNOWN_FUTURE_FEATURES,
        oracle_future_feature_names=ORACLE_FUTURE_FEATURES,
        splits=split_tensors,
        input_center=center,
        input_scale=scale,
        target_center=target_center,
        target_scale=target_scale,
        spike_thresholds=thresholds,
        metadata=metadata,
    )


def make_workload_extension_tensors(
    splits: Mapping[str, WorkloadSplitTensors],
    *,
    event_feature_names: Sequence[str],
    regular_grid_feature_names: Sequence[str],
    cohorts: Sequence[str] = DEFAULT_COHORTS,
    metadata: Mapping[str, Any] | None = None,
) -> WorkloadExtensionTensors:
    if set(splits) != set(SPLITS):
        raise ValueError(f"workload tensors require exactly the splits {SPLITS}")
    feature_names = tuple(str(value) for value in event_feature_names)
    grid_names = tuple(str(value) for value in regular_grid_feature_names)
    cohort_names = tuple(str(value) for value in cohorts)
    if not feature_names or not grid_names or not cohort_names:
        raise ValueError("workload feature and cohort names must be nonempty")
    locked_target_hash: str | None = None
    for split in SPLITS:
        values = splits[split]
        sample_count = len(values.event_history)
        if values.event_history.ndim != 3 or values.event_history.shape[-1] != len(feature_names):
            raise ValueError(f"invalid event-history shape for {split}")
        if values.elapsed_hours.shape != (*values.event_history.shape[:2], 1):
            raise ValueError(f"invalid elapsed-time shape for {split}")
        if values.regular_grid_history.ndim != 3 or values.regular_grid_history.shape[-1] != len(grid_names):
            raise ValueError(f"invalid regular-grid shape for {split}")
        if len(values.regular_grid_history) != sample_count:
            raise ValueError(f"event and grid sample counts differ for {split}")
        if values.count_target.shape != values.volume_target.shape:
            raise ValueError(f"workload count and volume target shapes differ for {split}")
        if values.count_target.ndim != 3 or values.count_target.shape[1] != len(cohort_names):
            raise ValueError(f"invalid workload target shape for {split}")
        if len(values.count_target) != sample_count or len(values.origins_utc) != sample_count:
            raise ValueError(f"workload origin/sample counts differ for {split}")
        if (values.elapsed_hours <= 0).any() or not all(
            np.isfinite(array).all()
            for array in (
                values.event_history,
                values.elapsed_hours,
                values.regular_grid_history,
                values.count_target,
                values.volume_target,
            )
        ):
            raise ValueError(f"workload tensors contain invalid values for {split}")
        if (values.count_target < 0).any() or (values.volume_target < 0).any():
            raise ValueError("workload targets must be nonnegative")
        if split == "locked_test":
            locked_target_hash = sha256_json(
                {
                    "count": _array_sha256(values.count_target),
                    "volume": _array_sha256(values.volume_target),
                    "origins": values.origins_utc,
                }
            )
    required_mark = any(
        name.startswith(("mark_", "cohort_")) for name in feature_names
    )
    output_metadata = {
        "schema_version": 1,
        "locked_test_target_sha256": locked_target_hash,
        "event_marks_or_cohort_embedding_present": required_mark,
        "same_locked_test_for_all_ablations": True,
        "locked_test_used_for_selection_or_calibration": False,
        **dict(metadata or {}),
    }
    return WorkloadExtensionTensors(
        event_feature_names=feature_names,
        regular_grid_feature_names=grid_names,
        cohorts=cohort_names,
        splits=dict(splits),
        metadata=output_metadata,
    )


@dataclass(frozen=True)
class EnergyVariant:
    name: str
    graph_mode: str
    patch_length: int
    stride: int
    include_known_future: bool
    include_oracle_weather: bool
    spike_weighting: bool
    conformal_calibration: bool
    evidence_role: str
    h1_role: str


ENERGY_VARIANTS: Mapping[str, EnergyVariant] = {
    "dgpt_e_primary": EnergyVariant(
        "dgpt_e_primary", "dynamic", 16, 8, True, False, True, True,
        "derived", "primary_confirmatory",
    ),
    "no_graph": EnergyVariant(
        "no_graph", "none", 16, 8, True, False, True, True,
        "derived", "ablation",
    ),
    "static_graph_only": EnergyVariant(
        "static_graph_only", "static", 16, 8, True, False, True, True,
        "derived", "ablation",
    ),
    "no_patching": EnergyVariant(
        "no_patching", "dynamic", 1, 1, True, False, True, True,
        "derived", "ablation",
    ),
    "no_known_future": EnergyVariant(
        "no_known_future", "dynamic", 16, 8, False, False, True, True,
        "derived", "ablation",
    ),
    "no_spike_weighting": EnergyVariant(
        "no_spike_weighting", "dynamic", 16, 8, True, False, False, True,
        "derived", "ablation",
    ),
    "no_conformal_calibration": EnergyVariant(
        "no_conformal_calibration", "dynamic", 16, 8, True, False, True, False,
        "derived", "ablation",
    ),
    "ORACLE_future_realized_weather": EnergyVariant(
        "ORACLE_future_realized_weather", "dynamic", 16, 8, True, True, True, True,
        "oracle", "excluded_from_H1",
    ),
}


@dataclass(frozen=True)
class WorkloadVariant:
    name: str
    encoder: str
    remove_event_marks_and_cohort_embedding: bool
    count_distribution: str
    include_volume_head: bool
    role: str


WORKLOAD_VARIANTS: Mapping[str, WorkloadVariant] = {
    "cfc_w_primary": WorkloadVariant(
        "cfc_w_primary", "event_time_cfc", False, "negative_binomial", True,
        "primary_confirmatory",
    ),
    "regular_grid_lstm": WorkloadVariant(
        "regular_grid_lstm", "regular_grid_lstm", False, "negative_binomial", True,
        "ablation",
    ),
    "no_event_marks_or_cohort_embedding": WorkloadVariant(
        "no_event_marks_or_cohort_embedding", "event_time_cfc", True,
        "negative_binomial", True, "ablation",
    ),
    "poisson_count_head": WorkloadVariant(
        "poisson_count_head", "event_time_cfc", False, "poisson", True,
        "ablation",
    ),
    "no_volume_distribution_head": WorkloadVariant(
        "no_volume_distribution_head", "event_time_cfc", False,
        "negative_binomial", False, "ablation",
    ),
}


def weighted_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    spike_mask: torch.Tensor | None,
    spike_weight: float = 2.0,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    crossing_penalty: float = 0.1,
) -> torch.Tensor:
    if prediction.shape[:-1] != target.shape:
        raise ValueError("prediction and target shapes are incompatible for pinball loss")
    levels = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device)
    if prediction.shape[-1] != len(levels):
        raise ValueError("prediction quantile dimension is incorrect")
    if spike_weight < 1.0:
        raise ValueError("spike weight must be at least one")
    error = target.unsqueeze(-1) - prediction
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    if spike_mask is None:
        weights = torch.ones_like(target)
    else:
        if spike_mask.shape != target.shape:
            raise ValueError("spike mask and target shapes differ")
        weights = 1.0 + (float(spike_weight) - 1.0) * spike_mask.to(prediction)
    weighted = (pinball * weights.unsqueeze(-1)).sum() / (
        weights.sum() * prediction.shape[-1]
    ).clamp_min(1e-12)
    crossing = torch.relu(prediction[..., :-1] - prediction[..., 1:]).mean()
    return weighted + float(crossing_penalty) * crossing


def _energy_variant_inputs(
    split: EnergySplitTensors,
    variant: EnergyVariant,
) -> tuple[np.ndarray, np.ndarray]:
    if variant.include_known_future:
        future = split.future_known
    else:
        future = np.empty((*split.future_known.shape[:-1], 0), dtype="float32")
    if variant.include_oracle_weather:
        future = np.concatenate([future, split.future_oracle_weather], axis=-1)
    return split.history, future.astype("float32")


def build_energy_variant_model(
    tensors: EnergyTaskTensors,
    variant_name: str,
    *,
    geographic_adjacency: torch.Tensor | None = None,
) -> DGPTE:
    if variant_name not in ENERGY_VARIANTS:
        raise ValueError(f"unknown energy extension variant: {variant_name}")
    variant = ENERGY_VARIANTS[variant_name]
    training = tensors.splits["train"]
    history, future = _energy_variant_inputs(training, variant)
    return DGPTE(
        history=history.shape[1],
        horizon=training.target_normalized.shape[-1],
        sites=len(tensors.zones),
        input_features=history.shape[-1],
        future_features=future.shape[-1],
        quantiles=len(QUANTILES),
        patch_length=variant.patch_length,
        stride=variant.stride,
        graph_mode=variant.graph_mode,
        geographic_adjacency=geographic_adjacency,
    )


def _energy_forward(
    model: DGPTE,
    history: torch.Tensor,
    future: torch.Tensor,
) -> torch.Tensor:
    return model(history, future if model.future_features else None)


@torch.no_grad()
def _predict_energy(
    model: DGPTE,
    split: EnergySplitTensors,
    variant: EnergyVariant,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    history, future = _energy_variant_inputs(split, variant)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(history), torch.from_numpy(future)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    output: list[np.ndarray] = []
    for history_batch, future_batch in loader:
        prediction = _energy_forward(
            model,
            history_batch.to(device),
            future_batch.to(device),
        )
        output.append(prediction.float().cpu().numpy())
    return np.concatenate(output).astype("float32")


def _inverse_energy(tensors: EnergyTaskTensors, value: np.ndarray) -> np.ndarray:
    return (
        value * tensors.target_scale[None, :, None, None]
        + tensors.target_center[None, :, None, None]
    )


def _tail_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    ordered = rearrange_quantiles(prediction)
    upper = ordered[..., int(np.argmin(np.abs(QUANTILES - 0.95)))]
    threshold = np.asarray(thresholds, dtype="float64")[None, :, None]
    observed = truth >= threshold
    predicted = upper >= threshold
    tp = int(np.sum(observed & predicted))
    fp = int(np.sum(~observed & predicted))
    fn = int(np.sum(observed & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold_policy": "per_zone_training_q99_only",
        "thresholds_by_zone": thresholds.astype(float).tolist(),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
    }


def _fit_energy_seed(
    tensors: EnergyTaskTensors,
    variant: EnergyVariant,
    *,
    seed: int,
    output_root: Path,
    max_epochs: int,
    batch_sizes: Sequence[int],
    learning_rate: float,
    gradient_accumulation_steps: int,
    early_stopping_patience: int,
    spike_weight: float,
    device: torch.device,
    geographic_adjacency: torch.Tensor | None,
    paper_evidence: bool,
) -> dict[str, Any]:
    if max_epochs < 1 or gradient_accumulation_steps < 1:
        raise ValueError("training epochs and accumulation steps must be positive")
    train = tensors.splits["train"]
    validation = tensors.splits["validation"]
    last_oom: Exception | None = None
    for batch_size in tuple(dict.fromkeys(int(value) for value in batch_sizes if value > 0)):
        _seed_everything(seed)
        model = build_energy_variant_model(
            tensors, variant.name, geographic_adjacency=geographic_adjacency
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        architecture = {
            **model.architecture_contract,
            "input_features": model.input_features,
            "future_features": model.future_features,
            "known_future_consumed": variant.include_known_future,
            "oracle_weather_consumed": variant.include_oracle_weather,
            "temporal_patch_aggregation_consumed": variant.patch_length > 1,
            "no_patching_semantics": (
                "one_timestamp_per_token_no_temporal_patch_aggregation"
                if variant.patch_length == 1
                else None
            ),
            "intersite_graph_message_passing": variant.graph_mode != "none",
        }
        fingerprint = sha256_json(
            {
                "trainer_contract_revision": ENERGY_TRAINER_CONTRACT_REVISION,
                "tensor_fingerprint": tensors.fingerprint,
                "task": tensors.task,
                "variant": variant.__dict__,
                "seed": seed,
                "architecture": architecture,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "spike_weight": spike_weight if variant.spike_weighting else 1.0,
            }
        )
        checkpoint = (
            output_root
            / "checkpoints"
            / tensors.task
            / _safe_slug(variant.name)
            / f"seed_{seed}_latest.pt"
        )
        start_epoch = 1
        best_epoch = 0
        best_metric = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        resumed_from_epoch = 0
        checkpoint_migration: dict[str, Any] | None = None
        training_runtime_seconds = 0.0
        maximum_gradient_norm = 0.0
        nonfinite_gradient_steps = 0
        peak_cuda_memory_allocated_bytes = 0
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("resume_fingerprint") != fingerprint:
                legacy = LEGACY_SCHEDULER_ENERGY_CHECKPOINTS.get(
                    (tensors.task, variant.name, int(seed))
                )
                compatible = bool(
                    legacy
                    and saved.get("resume_fingerprint") == legacy["resume_fingerprint"]
                    and sha256_file(checkpoint) == legacy["sha256"]
                    and saved.get("task") == tensors.task
                    and saved.get("variant") == variant.name
                    and int(saved.get("seed", -1)) == int(seed)
                    and saved.get("architecture") == architecture
                    and int(saved.get("batch_size", -1)) == batch_size
                    and saved.get("paper_evidence") is paper_evidence
                    and saved.get("locked_test_used_for_selection_or_calibration") is False
                    and isinstance(saved.get("model_state"), Mapping)
                    and isinstance(saved.get("best_model_state"), Mapping)
                    and isinstance(saved.get("optimizer_state"), Mapping)
                )
                if not compatible:
                    raise ValueError(f"checkpoint fingerprint mismatch: {checkpoint}")
                checkpoint_migration = {
                    "reason": "seed_major_scheduler_only_protocol_amendment",
                    "legacy_resume_fingerprint": saved.get("resume_fingerprint"),
                    "legacy_checkpoint_sha256": legacy["sha256"],
                    "scientific_fields_reverified": True,
                    "locked_test_feedback_used": False,
                    "checkpoint_rewritten": not bool(saved.get("completed")),
                }
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            if saved.get("gradient_scaler_state"):
                scaler.load_state_dict(saved["gradient_scaler_state"])
            best_state = saved.get("best_model_state")
            best_epoch = int(saved.get("best_epoch", 0))
            best_metric = float(saved.get("best_validation_mean_weighted_pinball", math.inf))
            resumed_from_epoch = int(saved["epoch"])
            start_epoch = resumed_from_epoch + 1
            training_runtime_seconds = float(saved.get("training_runtime_seconds", 0.0))
            maximum_gradient_norm = float(saved.get("maximum_gradient_norm", 0.0))
            nonfinite_gradient_steps = int(saved.get("nonfinite_gradient_steps", 0))
            peak_cuda_memory_allocated_bytes = int(
                saved.get("peak_cuda_memory_allocated_bytes", 0)
            )
            if bool(saved.get("completed")):
                if best_state is None:
                    raise ValueError("completed energy extension checkpoint lacks best state")
                model.load_state_dict(best_state)
                return {
                    "model": model,
                    "checkpoint_path": checkpoint,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "epochs_written": resumed_from_epoch,
                    "resumed_from_epoch": resumed_from_epoch,
                    "batch_size": batch_size,
                    "resume_fingerprint": fingerprint,
                    "checkpoint_migration": checkpoint_migration,
                    "architecture": architecture,
                    "training_health": {
                        "runtime_seconds": training_runtime_seconds or None,
                        "maximum_gradient_norm_before_clipping": (
                            maximum_gradient_norm or None
                        ),
                        "nonfinite_gradient_steps": nonfinite_gradient_steps,
                        "peak_cuda_memory_allocated_bytes": (
                            peak_cuda_memory_allocated_bytes or None
                        ),
                    },
                }
        history, future = _energy_variant_inputs(train, variant)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(history),
                torch.from_numpy(future),
                torch.from_numpy(train.target_normalized),
                torch.from_numpy(train.target_spike_mask),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=device.type == "cuda",
        )
        no_improvement = max(0, resumed_from_epoch - best_epoch)
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                epoch_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for batch_index, (history_batch, future_batch, target_batch, spike_batch) in enumerate(
                    loader, start=1
                ):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        prediction = _energy_forward(
                            model,
                            history_batch.to(device, non_blocking=True),
                            future_batch.to(device, non_blocking=True),
                        )
                        loss = weighted_pinball_loss(
                            prediction,
                            target_batch.to(device, non_blocking=True),
                            spike_mask=(
                                spike_batch.to(device, non_blocking=True)
                                if variant.spike_weighting
                                else None
                            ),
                            spike_weight=spike_weight if variant.spike_weighting else 1.0,
                        ) / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    if (
                        batch_index % gradient_accumulation_steps == 0
                        or batch_index == len(loader)
                    ):
                        scaler.unscale_(optimizer)
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )
                        gradient_norm_value = float(gradient_norm.detach().cpu())
                        if not math.isfinite(gradient_norm_value):
                            nonfinite_gradient_steps += 1
                            raise RuntimeError("energy extension gradient norm is non-finite")
                        maximum_gradient_norm = max(
                            maximum_gradient_norm, gradient_norm_value
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                validation_prediction = _predict_energy(
                    model,
                    validation,
                    variant,
                    device=device,
                    batch_size=batch_size,
                )
                validation_metric = float(
                    quantile_metrics(
                        validation.target_normalized,
                        validation_prediction,
                        quantiles=QUANTILES,
                    )["mean_weighted_pinball"]
                )
                if not math.isfinite(validation_metric):
                    raise RuntimeError("energy extension validation metric is non-finite")
                training_runtime_seconds += time.perf_counter() - epoch_started
                if device.type == "cuda":
                    peak_cuda_memory_allocated_bytes = max(
                        peak_cuda_memory_allocated_bytes,
                        int(torch.cuda.max_memory_allocated(device)),
                    )
                if validation_metric < best_metric - 1e-8:
                    best_metric = validation_metric
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    no_improvement = 0
                else:
                    no_improvement += 1
                completed = epoch == max_epochs or no_improvement >= early_stopping_patience
                _atomic_torch(
                    {
                        "checkpoint_schema_version": 1,
                        "namespace": "paper_forecast_extensions",
                        "paper_evidence": bool(paper_evidence),
                        "task": tensors.task,
                        "variant": variant.name,
                        "model_class": "dcvpp.models.energy.dgpt_e.DGPTE",
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "gradient_scaler_state": scaler.state_dict(),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_validation_mean_weighted_pinball": best_metric,
                        "best_model_state": best_state,
                        "seed": seed,
                        "resume_fingerprint": fingerprint,
                        "architecture": architecture,
                        "batch_size": batch_size,
                        "completed": completed,
                        "training_runtime_seconds": training_runtime_seconds,
                        "maximum_gradient_norm": maximum_gradient_norm,
                        "nonfinite_gradient_steps": nonfinite_gradient_steps,
                        "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated_bytes,
                        "locked_test_used_for_selection_or_calibration": False,
                        "written_at_utc": utc_now(),
                    },
                    checkpoint,
                )
                if completed:
                    break
            if best_state is None:
                raise RuntimeError("energy extension produced no validation checkpoint")
            model.load_state_dict(best_state)
            latest = torch.load(checkpoint, map_location="cpu", weights_only=False)
            return {
                "model": model,
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "epochs_written": int(latest["epoch"]),
                "resumed_from_epoch": resumed_from_epoch,
                "batch_size": batch_size,
                "resume_fingerprint": fingerprint,
                "checkpoint_migration": checkpoint_migration,
                "architecture": architecture,
                "training_health": {
                    "runtime_seconds": training_runtime_seconds,
                    "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
                    "nonfinite_gradient_steps": nonfinite_gradient_steps,
                    "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated_bytes,
                },
            }
        except torch.OutOfMemoryError as exc:
            last_oom = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            checkpoint.unlink(missing_ok=True)
            continue
    raise RuntimeError(f"energy extension exhausted batch fallback: {last_oom}")


def train_energy_variant_seed(
    tensors: EnergyTaskTensors,
    variant_name: str,
    *,
    seed: int,
    output_root: Path,
    max_epochs: int = 100,
    batch_sizes: Sequence[int] = (32, 16, 8, 4, 2, 1),
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 2,
    early_stopping_patience: int = 10,
    spike_weight: float = 2.0,
    minimum_calibration_samples: int = 500,
    device_name: str = "auto",
    geographic_adjacency: torch.Tensor | None = None,
    paper_evidence: bool = True,
) -> dict[str, Any]:
    if variant_name not in ENERGY_VARIANTS:
        raise ValueError(f"unknown energy extension variant: {variant_name}")
    variant = ENERGY_VARIANTS[variant_name]
    if paper_evidence:
        train = tensors.splits["train"]
        if train.history.shape[1] != 672 or train.target_original.shape[-1] != 96:
            raise ValueError("paper DGPT-E extension fixes history=672 and horizon=96")
        if max_epochs != 100:
            raise ValueError("paper DGPT-E extension requires max_epochs=100")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    fitted = _fit_energy_seed(
        tensors,
        variant,
        seed=int(seed),
        output_root=Path(output_root),
        max_epochs=max_epochs,
        batch_sizes=batch_sizes,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        spike_weight=spike_weight,
        device=device,
        geographic_adjacency=geographic_adjacency,
        paper_evidence=paper_evidence,
    )
    validation = tensors.splits["validation"]
    locked = tensors.splits["locked_test"]
    validation_normalized = _predict_energy(
        fitted["model"], validation, variant, device=device, batch_size=fitted["batch_size"]
    )
    locked_normalized = _predict_energy(
        fitted["model"], locked, variant, device=device, batch_size=fitted["batch_size"]
    )
    validation_raw = _inverse_energy(tensors, validation_normalized)
    locked_raw = _inverse_energy(tensors, locked_normalized)
    if variant.conformal_calibration:
        calibrator = fit_additive_conformal_quantiles(
            validation.target_original,
            validation_raw,
            quantiles=QUANTILES,
            minimum_samples=minimum_calibration_samples,
        )
        validation_final = apply_additive_conformal_quantiles(validation_raw, calibrator)
        locked_final = apply_additive_conformal_quantiles(locked_raw, calibrator)
        corrections = np.asarray(calibrator["corrections"], dtype="float64")
        calibration_record = {
            key: value for key, value in calibrator.items() if key != "corrections"
        }
        calibration_record.update(
            {
                "corrections": corrections.tolist(),
                "corrections_sha256": _array_sha256(corrections),
                "fit_split": "validation",
                "locked_test_used": False,
            }
        )
    else:
        validation_final = rearrange_quantiles(validation_raw)
        locked_final = rearrange_quantiles(locked_raw)
        calibration_record = {
            "method": "disabled_by_no_conformal_calibration_ablation",
            "fit_split": None,
            "locked_test_used": False,
        }
    saved_checkpoint = torch.load(
        fitted["checkpoint_path"], map_location="cpu", weights_only=False
    )
    reloaded = build_energy_variant_model(
        tensors, variant.name, geographic_adjacency=geographic_adjacency
    ).to(device)
    reloaded.load_state_dict(saved_checkpoint["best_model_state"])
    reload_probe = _predict_energy(
        reloaded,
        EnergySplitTensors(
            history=validation.history[:1],
            future_known=validation.future_known[:1],
            future_oracle_weather=validation.future_oracle_weather[:1],
            target_normalized=validation.target_normalized[:1],
            target_original=validation.target_original[:1],
            target_spike_mask=validation.target_spike_mask[:1],
            origins_utc=validation.origins_utc[:1],
            target_timestamps_utc=validation.target_timestamps_utc[:1],
            issue_audit=validation.issue_audit[:1],
        ),
        variant,
        device=device,
        batch_size=1,
    )
    reload_reference = _predict_energy(
        fitted["model"],
        EnergySplitTensors(
            history=validation.history[:1],
            future_known=validation.future_known[:1],
            future_oracle_weather=validation.future_oracle_weather[:1],
            target_normalized=validation.target_normalized[:1],
            target_original=validation.target_original[:1],
            target_spike_mask=validation.target_spike_mask[:1],
            origins_utc=validation.origins_utc[:1],
            target_timestamps_utc=validation.target_timestamps_utc[:1],
            issue_audit=validation.issue_audit[:1],
        ),
        variant,
        device=device,
        batch_size=1,
    )
    reload_equal = bool(np.array_equal(reload_probe, reload_reference))
    if not reload_equal:
        raise RuntimeError("energy extension checkpoint reload changed inference")
    prediction_path = (
        Path(output_root)
        / "predictions"
        / tensors.task
        / _safe_slug(variant.name)
        / f"seed_{seed}.npz"
    )
    prediction_artifact = _atomic_npz(
        prediction_path,
        validation_prediction=np.asarray(validation_final, dtype="float32"),
        locked_test_prediction=np.asarray(locked_final, dtype="float32"),
        validation_target=validation.target_original,
        locked_test_target=locked.target_original,
    )
    metrics = {
        "validation": {
            **quantile_metrics(
                validation.target_original, validation_final, quantiles=QUANTILES
            ),
            "tail": _tail_metrics(
                validation.target_original,
                validation_final,
                np.asarray(tensors.spike_thresholds[tensors.task]),
            ),
        },
        "locked_test": {
            **quantile_metrics(locked.target_original, locked_final, quantiles=QUANTILES),
            "tail": _tail_metrics(
                locked.target_original,
                locked_final,
                np.asarray(tensors.spike_thresholds[tensors.task]),
            ),
        },
    }
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "namespace": "paper_forecast_extensions",
        "task": tensors.task,
        "variant": variant.name,
        "model_name": (
            "ORACLE_DGPT-E" if variant.include_oracle_weather else "DGPT-E"
        ),
        "model_class": "dcvpp.models.energy.dgpt_e.DGPTE",
        "seed": int(seed),
        "evidence_role": variant.evidence_role,
        "deployable": not variant.include_oracle_weather,
        "h1_role": variant.h1_role,
        "h1_eligible": variant.h1_role == "primary_confirmatory",
        "future_realized_weather_consumed": variant.include_oracle_weather,
        "known_future_consumed": variant.include_known_future,
        "spike_weighting": variant.spike_weighting,
        "spike_threshold_source": "training_only_per_product_per_zone_q99",
        "conformal_calibration": variant.conformal_calibration,
        "architecture": fitted["architecture"],
        "training_health": fitted["training_health"],
        "best_epoch_selected_on_validation": fitted["best_epoch"],
        "epochs_written": fitted["epochs_written"],
        "resumed_from_epoch": fitted["resumed_from_epoch"],
        "resume_fingerprint": fitted["resume_fingerprint"],
        "checkpoint_migration": fitted.get("checkpoint_migration"),
        "checkpoint": {
            "path": fitted["checkpoint_path"].as_posix(),
            "sha256": fitted["checkpoint_sha256"],
            "reload_prediction_bitwise_equal": reload_equal,
        },
        "prediction_artifact": prediction_artifact,
        "calibration": calibration_record,
        "metrics": metrics,
        "locked_test_target_sha256": tensors.metadata["locked_test_target_sha256"],
        "locked_test_origin_sha256": tensors.metadata["locked_test_origin_sha256"],
        "locked_test_used_for_selection_or_calibration": False,
    }
    metrics_path = (
        Path(output_root)
        / "metrics"
        / tensors.task
        / _safe_slug(variant.name)
        / f"seed_{seed}.json"
    )
    atomic_json(metrics_path, result)
    result["metrics_artifact"] = {
        "path": metrics_path.as_posix(),
        "sha256": sha256_file(metrics_path),
    }
    return result


def train_energy_extensions(
    tensors_by_task: Mapping[str, EnergyTaskTensors],
    *,
    output_root: Path,
    seeds: Sequence[int] = PREREGISTERED_SEEDS,
    variant_names: Sequence[str] = tuple(ENERGY_VARIANTS),
    paper_evidence: bool = True,
    **training_kwargs: Any,
) -> dict[str, Any]:
    seed_values = _seed_gate(seeds, paper_evidence=paper_evidence)
    if set(tensors_by_task) != set(ENERGY_TASKS):
        raise ValueError("energy extensions require separate DA and RT tensor bundles")
    unknown = sorted(set(variant_names) - set(ENERGY_VARIANTS))
    if unknown:
        raise ValueError(f"unknown energy extension variants: {unknown}")
    if paper_evidence and set(variant_names) != set(ENERGY_VARIANTS):
        raise ValueError("paper energy extensions require every preregistered ablation and ORACLE")
    results: dict[str, Any] = {}
    locked_hashes: dict[str, set[str]] = {}
    trials_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {
        (task, variant): [] for task in ENERGY_TASKS for variant in variant_names
    }
    for seed in seed_values:
        for task in ENERGY_TASKS:
            for variant_name in variant_names:
                trial = train_energy_variant_seed(
                    tensors_by_task[task],
                    variant_name,
                    seed=seed,
                    output_root=Path(output_root),
                    paper_evidence=paper_evidence,
                    **training_kwargs,
                )
                trials_by_cell[(task, variant_name)].append(trial)
    for task in ENERGY_TASKS:
        task_results: dict[str, Any] = {}
        for variant_name in variant_names:
            trials = trials_by_cell[(task, variant_name)]
            locked_hashes.setdefault(task, set()).update(
                str(trial["locked_test_target_sha256"]) for trial in trials
            )
            selected = min(
                trials,
                key=lambda trial: float(
                    trial["metrics"]["validation"]["mean_weighted_pinball"]
                ),
            )
            task_results[variant_name] = {
                "status": "COMPLETE",
                "seed_trials": trials,
                "selected_seed_by_validation_only": selected["seed"],
                "locked_test_used_for_selection": False,
            }
        results[task] = task_results
    if any(len(values) != 1 for values in locked_hashes.values()):
        raise RuntimeError("energy variants did not use the same locked test within task")
    for task in ENERGY_TASKS:
        oracle = results[task]["ORACLE_future_realized_weather"]
        if any(trial["h1_eligible"] for trial in oracle["seed_trials"]):
            raise RuntimeError("ORACLE weather variant leaked into H1 eligibility")
    return {
        "status": "COMPLETE",
        "tasks": results,
        "seeds": list(seed_values),
        "variants": list(variant_names),
        "da_rt_separate": True,
        "oracle_isolated_from_H1": True,
        "locked_test_target_sha256_by_task": {
            task: next(iter(values)) for task, values in locked_hashes.items()
        },
    }


class RegularGridWorkloadLSTM(nn.Module):
    """Regular-grid LSTM ablation with the CfC-W distribution contract."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        horizon: int,
        cohorts: int,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.cohorts = int(cohorts)
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        output_size = cohorts * horizon * 2
        self.count_head = nn.Linear(hidden_size, output_size)
        self.lognormal_head = nn.Linear(hidden_size, output_size)
        self.gamma_head = nn.Linear(hidden_size, output_size)

    @property
    def architecture_contract(self) -> dict[str, Any]:
        return {
            "encoder": "regular_grid_lstm",
            "layers": 2,
            "hidden_size": self.hidden_size,
            "horizon": self.horizon,
            "cohorts": self.cohorts,
            "event_elapsed_time_consumed": False,
        }

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> dict[str, torch.Tensor]:
        del dt
        sequence, _ = self.encoder(x)
        encoded = sequence[:, -1]
        shape = (len(x), self.cohorts, self.horizon, 2)
        count = self.count_head(encoded).reshape(shape)
        lognormal = self.lognormal_head(encoded).reshape(shape)
        gamma = self.gamma_head(encoded).reshape(shape)
        return {
            "count_mean": torch.nn.functional.softplus(count[..., 0]) + 1e-4,
            "count_dispersion": torch.nn.functional.softplus(count[..., 1]) + 1e-4,
            "volume_log_mu": lognormal[..., 0],
            "volume_log_sigma": torch.nn.functional.softplus(lognormal[..., 1]) + 1e-4,
            "volume_gamma_shape": torch.nn.functional.softplus(gamma[..., 0]) + 1e-4,
            "volume_gamma_rate": torch.nn.functional.softplus(gamma[..., 1]) + 1e-4,
        }


class CfCWHeadAblation(nn.Module):
    """Reuse the production CfC encoder with genuinely replaced/removed heads."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        horizon: int,
        cohorts: int,
        count_distribution: str,
        include_volume_head: bool,
    ) -> None:
        super().__init__()
        if count_distribution not in {"negative_binomial", "poisson"}:
            raise ValueError("CfC head ablation requires NB or Poisson count distribution")
        source = CfCW(
            input_size=input_size,
            hidden_size=hidden_size,
            horizon=horizon,
            cohorts=cohorts,
        )
        self.encoder = source.encoder
        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.cohorts = int(cohorts)
        self.count_distribution = count_distribution
        self.include_volume_head = bool(include_volume_head)
        if count_distribution == "poisson":
            self.poisson_count_head = nn.Linear(hidden_size, cohorts * horizon)
        else:
            self.count_head = source.count_head
        if include_volume_head:
            self.lognormal_head = source.lognormal_head
            self.gamma_head = source.gamma_head

    @property
    def architecture_contract(self) -> dict[str, Any]:
        return {
            "encoder": "ncps.torch.CfC_reused_from_CfCW",
            "hidden_size": self.hidden_size,
            "horizon": self.horizon,
            "cohorts": self.cohorts,
            "count_distribution": self.count_distribution,
            "count_dispersion_head_present": self.count_distribution == "negative_binomial",
            "volume_distribution_heads_present": self.include_volume_head,
        }

    def _encode_unchecked(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            (x.shape[0], self.encoder.state_size), dtype=x.dtype, device=x.device
        )
        encoded = hidden
        for step in range(x.shape[1]):
            encoded, hidden = self.encoder.rnn_cell.forward(x[:, step], hidden, dt[:, step])
        if self.encoder.fc is not None:
            encoded = self.encoder.fc(encoded)
        return encoded

    def _encode(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or dt.ndim != 3 or x.shape[:2] != dt.shape[:2]:
            raise ValueError("CfC head ablation expects aligned event and elapsed tensors")
        if dt.shape[-1] != 1 or (dt <= 0).any() or not torch.isfinite(dt).all():
            raise ValueError("CfC head ablation elapsed times must be finite and positive")
        return self._encode_unchecked(x, dt)

    def _cuda_graph_forward_unchecked(
        self, x: torch.Tensor, dt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        encoded = self._encode_unchecked(x, dt)
        output: dict[str, torch.Tensor]
        if self.count_distribution == "poisson":
            raw_mean = self.poisson_count_head(encoded).reshape(
                len(x), self.cohorts, self.horizon
            )
            output = {
                "count_mean": torch.nn.functional.softplus(raw_mean) + 1e-4,
            }
        else:
            count = self.count_head(encoded).reshape(
                len(x), self.cohorts, self.horizon, 2
            )
            output = {
                "count_mean": torch.nn.functional.softplus(count[..., 0]) + 1e-4,
                "count_dispersion": torch.nn.functional.softplus(count[..., 1]) + 1e-4,
            }
        if self.include_volume_head:
            shape = (len(x), self.cohorts, self.horizon, 2)
            lognormal = self.lognormal_head(encoded).reshape(shape)
            gamma = self.gamma_head(encoded).reshape(shape)
            output.update(
                {
                    "volume_log_mu": lognormal[..., 0],
                    "volume_log_sigma": torch.nn.functional.softplus(
                        lognormal[..., 1]
                    )
                    + 1e-4,
                    "volume_gamma_shape": torch.nn.functional.softplus(gamma[..., 0])
                    + 1e-4,
                    "volume_gamma_rate": torch.nn.functional.softplus(gamma[..., 1])
                    + 1e-4,
                }
            )
        return output

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._cuda_graph_forward_unchecked(x, dt)


def _workload_variant_inputs(
    tensors: WorkloadExtensionTensors,
    split: WorkloadSplitTensors,
    variant: WorkloadVariant,
) -> tuple[np.ndarray, np.ndarray]:
    if variant.encoder == "regular_grid_lstm":
        history = split.regular_grid_history.astype("float32", copy=False)
        elapsed = np.full((*history.shape[:2], 1), 0.25, dtype="float32")
        return history, elapsed
    history = split.event_history.astype("float32", copy=True)
    if variant.remove_event_marks_and_cohort_embedding:
        indexes = [
            index
            for index, name in enumerate(tensors.event_feature_names)
            if name.startswith(("mark_", "cohort_"))
        ]
        if not indexes:
            raise ValueError(
                "no-event-marks/cohort-embedding ablation requires declared mark_ or cohort_ features"
            )
        history[..., indexes] = 0.0
    return history, split.elapsed_hours.astype("float32", copy=False)


def build_workload_variant_model(
    tensors: WorkloadExtensionTensors,
    variant_name: str,
    *,
    hidden_size: int = 64,
) -> nn.Module:
    if variant_name not in WORKLOAD_VARIANTS:
        raise ValueError(f"unknown workload extension variant: {variant_name}")
    variant = WORKLOAD_VARIANTS[variant_name]
    training = tensors.splits["train"]
    history, _ = _workload_variant_inputs(tensors, training, variant)
    horizon = training.count_target.shape[-1]
    if variant.encoder == "regular_grid_lstm":
        model: nn.Module = RegularGridWorkloadLSTM(
            input_size=history.shape[-1],
            hidden_size=hidden_size,
            horizon=horizon,
            cohorts=len(tensors.cohorts),
        )
    elif variant.count_distribution == "poisson" or not variant.include_volume_head:
        model = CfCWHeadAblation(
            input_size=history.shape[-1],
            hidden_size=hidden_size,
            horizon=horizon,
            cohorts=len(tensors.cohorts),
            count_distribution=variant.count_distribution,
            include_volume_head=variant.include_volume_head,
        )
    else:
        model = CfCW(
            input_size=history.shape[-1],
            hidden_size=hidden_size,
            horizon=horizon,
            cohorts=len(tensors.cohorts),
        )
    return model


def _poisson_nll(target: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return -torch.distributions.Poisson(mean.clamp_min(1e-6)).log_prob(
        target.clamp_min(0)
    ).mean()


def _workload_loss_for_variant(
    output: Mapping[str, torch.Tensor],
    count_target: torch.Tensor,
    volume_target: torch.Tensor,
    variant: WorkloadVariant,
) -> torch.Tensor:
    if variant.count_distribution == "poisson":
        count_loss = _poisson_nll(count_target, output["count_mean"])
    else:
        count_loss = negative_binomial_nll(
            count_target, output["count_mean"], output["count_dispersion"]
        )
    if not variant.include_volume_head:
        return count_loss
    lognormal = lognormal_volume_nll(
        volume_target, output["volume_log_mu"], output["volume_log_sigma"]
    )
    gamma = gamma_volume_nll(
        volume_target, output["volume_gamma_shape"], output["volume_gamma_rate"]
    )
    return count_loss + 0.5 * (lognormal + gamma)


@torch.no_grad()
def _predict_workload(
    model: nn.Module,
    tensors: WorkloadExtensionTensors,
    split: WorkloadSplitTensors,
    variant: WorkloadVariant,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    history, elapsed = _workload_variant_inputs(tensors, split, variant)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(history), torch.from_numpy(elapsed)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    for history_batch, elapsed_batch in loader:
        output = model(history_batch.to(device), elapsed_batch.to(device))
        for key, value in output.items():
            collected.setdefault(key, []).append(value.float().cpu().numpy())
    return {
        key: np.concatenate(values).astype("float32")
        for key, values in collected.items()
    }


def _workload_distribution_score(
    count: np.ndarray,
    volume: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    variant: WorkloadVariant,
) -> tuple[float, str | None, dict[str, Any]]:
    if variant.count_distribution == "poisson":
        mean = np.asarray(prediction["count_mean"], dtype="float64").clip(1e-8)
        nll = -stats.poisson.logpmf(np.rint(count).astype("int64"), mean)
        count_record: dict[str, Any] = {
            "distribution": "poisson",
            "poisson_nll": float(np.mean(nll[np.isfinite(nll)])),
        }
        count_score = float(count_record["poisson_nll"])
    else:
        count_record = negative_binomial_metrics(
            count, prediction["count_mean"], prediction["count_dispersion"]
        )
        count_record["distribution"] = "negative_binomial"
        count_score = float(count_record["negative_binomial_nll"])
    if not variant.include_volume_head:
        return count_score, None, {"count": count_record, "volume": "disabled_by_ablation"}
    lognormal = positive_volume_nll(
        volume,
        {
            "log_mu": prediction["volume_log_mu"],
            "sigma": prediction["volume_log_sigma"],
        },
        distribution="lognormal",
    )
    gamma = positive_volume_nll(
        volume,
        {
            "shape": prediction["volume_gamma_shape"],
            "rate": prediction["volume_gamma_rate"],
        },
        distribution="gamma",
    )
    selected = "lognormal" if lognormal <= gamma else "gamma"
    return count_score + min(lognormal, gamma), selected, {
        "count": count_record,
        "volume_candidates": {
            "lognormal_positive_nll": float(lognormal),
            "gamma_positive_nll": float(gamma),
        },
    }


def _count_quantiles(
    prediction: Mapping[str, np.ndarray],
    variant: WorkloadVariant,
) -> np.ndarray:
    if variant.count_distribution == "poisson":
        return np.moveaxis(
            np.asarray(
                [
                    stats.poisson.ppf(float(level), prediction["count_mean"])
                    for level in QUANTILES
                ],
                dtype="float64",
            ),
            0,
            -1,
        )
    return negative_binomial_quantiles(
        prediction["count_mean"],
        prediction["count_dispersion"],
        quantiles=QUANTILES,
    )


def _volume_quantiles(
    prediction: Mapping[str, np.ndarray],
    variant: WorkloadVariant,
    *,
    distribution: str,
) -> np.ndarray:
    if variant.count_distribution == "negative_binomial":
        if distribution == "lognormal":
            parameters = {
                "log_mu": prediction["volume_log_mu"],
                "sigma": prediction["volume_log_sigma"],
            }
        else:
            parameters = {
                "shape": prediction["volume_gamma_shape"],
                "rate": prediction["volume_gamma_rate"],
            }
        return positive_volume_quantiles(
            prediction["count_mean"],
            prediction["count_dispersion"],
            parameters,
            distribution=distribution,
            quantiles=QUANTILES,
        )
    mean = np.asarray(prediction["count_mean"], dtype="float64").clip(1e-8)
    p_zero = np.exp(-mean)
    values: list[np.ndarray] = []
    for level in QUANTILES:
        conditional = np.clip(
            (float(level) - p_zero) / np.maximum(1.0 - p_zero, 1e-12), 0.0, 1.0
        )
        if distribution == "lognormal":
            value = stats.lognorm.ppf(
                conditional,
                s=np.asarray(prediction["volume_log_sigma"]).clip(1e-8),
                scale=np.exp(np.asarray(prediction["volume_log_mu"])),
            )
        else:
            value = stats.gamma.ppf(
                conditional,
                a=np.asarray(prediction["volume_gamma_shape"]).clip(1e-8),
                scale=1.0 / np.asarray(prediction["volume_gamma_rate"]).clip(1e-8),
            )
        values.append(np.nan_to_num(value, nan=0.0, posinf=1e12, neginf=0.0))
    return np.moveaxis(np.asarray(values), 0, -1)


def _fit_workload_seed(
    tensors: WorkloadExtensionTensors,
    variant: WorkloadVariant,
    *,
    seed: int,
    output_root: Path,
    hidden_size: int,
    max_epochs: int,
    batch_sizes: Sequence[int],
    learning_rate: float,
    early_stopping_patience: int,
    data_loader_workers: int,
    device: torch.device,
    paper_evidence: bool,
) -> dict[str, Any]:
    train = tensors.splits["train"]
    validation = tensors.splits["validation"]
    last_oom: Exception | None = None
    for batch_size in tuple(dict.fromkeys(int(value) for value in batch_sizes if value > 0)):
        _seed_everything(seed)
        model = build_workload_variant_model(
            tensors, variant.name, hidden_size=hidden_size
        ).to(device)
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        architecture = (
            model.architecture_contract
            if hasattr(model, "architecture_contract")
            else {
                "encoder": variant.encoder,
                "hidden_size": hidden_size,
                "horizon": train.count_target.shape[-1],
                "cohorts": len(tensors.cohorts),
            }
        )
        architecture = {
            **architecture,
            "count_distribution_consumed": variant.count_distribution,
            "volume_distribution_head_consumed": variant.include_volume_head,
            "event_marks_or_cohort_embedding_removed": (
                variant.remove_event_marks_and_cohort_embedding
            ),
        }
        fingerprint = sha256_json(
            {
                "trainer_contract_revision": WORKLOAD_TRAINER_CONTRACT_REVISION,
                "tensor_fingerprint": tensors.fingerprint,
                "variant": variant.__dict__,
                "seed": seed,
                "architecture": architecture,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )
        checkpoint = (
            output_root
            / "checkpoints"
            / "nlr_workload"
            / _safe_slug(variant.name)
            / f"seed_{seed}_latest.pt"
        )
        start_epoch = 1
        best_epoch = 0
        best_metric = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        resumed_from_epoch = 0
        training_runtime_seconds = 0.0
        maximum_gradient_norm = 0.0
        nonfinite_gradient_steps = 0
        peak_cuda_memory_allocated_bytes = 0
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("resume_fingerprint") != fingerprint:
                raise ValueError(f"checkpoint fingerprint mismatch: {checkpoint}")
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            if saved.get("gradient_scaler_state"):
                scaler.load_state_dict(saved["gradient_scaler_state"])
            best_state = saved.get("best_model_state")
            best_epoch = int(saved.get("best_epoch", 0))
            best_metric = float(saved.get("best_validation_distribution_score", math.inf))
            resumed_from_epoch = int(saved["epoch"])
            start_epoch = resumed_from_epoch + 1
            training_runtime_seconds = float(saved.get("training_runtime_seconds", 0.0))
            maximum_gradient_norm = float(saved.get("maximum_gradient_norm", 0.0))
            nonfinite_gradient_steps = int(saved.get("nonfinite_gradient_steps", 0))
            peak_cuda_memory_allocated_bytes = int(
                saved.get("peak_cuda_memory_allocated_bytes", 0)
            )
            if bool(saved.get("completed")):
                if best_state is None:
                    raise ValueError("completed workload extension checkpoint lacks best state")
                model.load_state_dict(best_state)
                return {
                    "model": model,
                    "checkpoint_path": checkpoint,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "epochs_written": resumed_from_epoch,
                    "resumed_from_epoch": resumed_from_epoch,
                    "batch_size": batch_size,
                    "resume_fingerprint": fingerprint,
                    "architecture": architecture,
                    "runtime_acceleration": {
                        "backend": "completed_checkpoint_eager_replay",
                        "data_loader_workers": int(data_loader_workers),
                    },
                    "training_health": {
                        "runtime_seconds": training_runtime_seconds or None,
                        "maximum_gradient_norm_before_clipping": (
                            maximum_gradient_norm or None
                        ),
                        "nonfinite_gradient_steps": nonfinite_gradient_steps,
                        "peak_cuda_memory_allocated_bytes": (
                            peak_cuda_memory_allocated_bytes or None
                        ),
                    },
                }
        runtime_acceleration: dict[str, Any] = {
            "backend": "eager",
            "data_loader_workers": int(data_loader_workers),
        }
        if device.type == "cuda" and variant.encoder != "regular_grid_lstm":
            from dcvpp.experiments.cfc_cuda_graph_recovery import (
                wrap_cuda_graphed_workload_model,
            )

            output_keys = ["count_mean"]
            if variant.count_distribution == "negative_binomial":
                output_keys.append("count_dispersion")
            if variant.include_volume_head:
                output_keys.extend(
                    [
                        "volume_log_mu",
                        "volume_log_sigma",
                        "volume_gamma_shape",
                        "volume_gamma_rate",
                    ]
                )
            model = wrap_cuda_graphed_workload_model(
                model,
                output_keys=output_keys,
                capture_batch_size=batch_size,
            )
            runtime_acceleration = {
                **dict(model.acceleration_contract),
                "data_loader_workers": int(data_loader_workers),
            }
        history, elapsed = _workload_variant_inputs(tensors, train, variant)
        worker_count = max(0, int(data_loader_workers))
        loader_kwargs: dict[str, Any] = {
            "num_workers": worker_count,
            "persistent_workers": worker_count > 0,
        }
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(history),
                torch.from_numpy(elapsed),
                torch.from_numpy(train.count_target),
                torch.from_numpy(train.volume_target),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=device.type == "cuda",
            **loader_kwargs,
        )
        no_improvement = max(0, resumed_from_epoch - best_epoch)
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                epoch_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                model.train()
                for history_batch, elapsed_batch, count_batch, volume_batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        output = model(
                            history_batch.to(device, non_blocking=True),
                            elapsed_batch.to(device, non_blocking=True),
                        )
                        loss = _workload_loss_for_variant(
                            output,
                            count_batch.to(device, non_blocking=True),
                            volume_batch.to(device, non_blocking=True),
                            variant,
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    gradient_norm_value = float(gradient_norm.detach().cpu())
                    if not math.isfinite(gradient_norm_value):
                        nonfinite_gradient_steps += 1
                        raise RuntimeError("workload extension gradient norm is non-finite")
                    maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm_value)
                    scaler.step(optimizer)
                    scaler.update()
                validation_prediction = _predict_workload(
                    model,
                    tensors,
                    validation,
                    variant,
                    device=device,
                    batch_size=batch_size,
                )
                validation_metric, _, _ = _workload_distribution_score(
                    validation.count_target,
                    validation.volume_target,
                    validation_prediction,
                    variant,
                )
                if not math.isfinite(validation_metric):
                    raise RuntimeError("workload extension validation metric is non-finite")
                training_runtime_seconds += time.perf_counter() - epoch_started
                if device.type == "cuda":
                    peak_cuda_memory_allocated_bytes = max(
                        peak_cuda_memory_allocated_bytes,
                        int(torch.cuda.max_memory_allocated(device)),
                    )
                if validation_metric < best_metric - 1e-8:
                    best_metric = validation_metric
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    no_improvement = 0
                else:
                    no_improvement += 1
                completed = epoch == max_epochs or no_improvement >= early_stopping_patience
                _atomic_torch(
                    {
                        "checkpoint_schema_version": 1,
                        "namespace": "paper_forecast_extensions",
                        "paper_evidence": bool(paper_evidence),
                        "task": "nlr_workload",
                        "variant": variant.name,
                        "model_class": f"{type(model).__module__}.{type(model).__name__}",
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "gradient_scaler_state": scaler.state_dict(),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_validation_distribution_score": best_metric,
                        "best_model_state": best_state,
                        "seed": seed,
                        "resume_fingerprint": fingerprint,
                        "architecture": architecture,
                        "batch_size": batch_size,
                        "completed": completed,
                        "training_runtime_seconds": training_runtime_seconds,
                        "maximum_gradient_norm": maximum_gradient_norm,
                        "nonfinite_gradient_steps": nonfinite_gradient_steps,
                        "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated_bytes,
                        "locked_test_used_for_selection_or_calibration": False,
                        "written_at_utc": utc_now(),
                    },
                    checkpoint,
                )
                if completed:
                    break
            if best_state is None:
                raise RuntimeError("workload extension produced no validation checkpoint")
            model.load_state_dict(best_state)
            latest = torch.load(checkpoint, map_location="cpu", weights_only=False)
            return {
                "model": model,
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "epochs_written": int(latest["epoch"]),
                "resumed_from_epoch": resumed_from_epoch,
                "batch_size": batch_size,
                "resume_fingerprint": fingerprint,
                "architecture": architecture,
                "runtime_acceleration": runtime_acceleration,
                "training_health": {
                    "runtime_seconds": training_runtime_seconds,
                    "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
                    "nonfinite_gradient_steps": nonfinite_gradient_steps,
                    "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated_bytes,
                },
            }
        except torch.OutOfMemoryError as exc:
            last_oom = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            checkpoint.unlink(missing_ok=True)
            continue
    raise RuntimeError(f"workload extension exhausted batch fallback: {last_oom}")


def _calibrate_quantiles(
    truth: np.ndarray,
    raw: np.ndarray,
    *,
    minimum_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    calibrator = fit_additive_conformal_quantiles(
        truth,
        raw,
        quantiles=QUANTILES,
        minimum_samples=minimum_samples,
    )
    corrections = np.asarray(calibrator["corrections"], dtype="float64")
    record = {key: value for key, value in calibrator.items() if key != "corrections"}
    record.update(
        {
            "corrections": corrections.tolist(),
            "corrections_sha256": _array_sha256(corrections),
            "fit_split": "validation",
            "locked_test_used": False,
        }
    )
    return apply_additive_conformal_quantiles(raw, calibrator), record


def train_workload_variant_seed(
    tensors: WorkloadExtensionTensors,
    variant_name: str,
    *,
    seed: int,
    output_root: Path,
    hidden_size: int = 64,
    max_epochs: int = 100,
    batch_sizes: Sequence[int] = (32, 16, 8, 4, 2, 1),
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 10,
    data_loader_workers: int = 2,
    minimum_calibration_samples: int = 500,
    device_name: str = "auto",
    paper_evidence: bool = True,
) -> dict[str, Any]:
    if variant_name not in WORKLOAD_VARIANTS:
        raise ValueError(f"unknown workload extension variant: {variant_name}")
    variant = WORKLOAD_VARIANTS[variant_name]
    if paper_evidence:
        if tensors.splits["train"].count_target.shape[-1] != 96:
            raise ValueError("paper CfC-W extension fixes horizon=96")
        if max_epochs != 100:
            raise ValueError("paper CfC-W extension requires max_epochs=100")
        if hidden_size not in {64, 96}:
            raise ValueError("paper CfC-W extension hidden size must be 64 or 96")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    fitted = _fit_workload_seed(
        tensors,
        variant,
        seed=int(seed),
        output_root=Path(output_root),
        hidden_size=hidden_size,
        max_epochs=max_epochs,
        batch_sizes=batch_sizes,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        data_loader_workers=data_loader_workers,
        device=device,
        paper_evidence=paper_evidence,
    )
    validation = tensors.splits["validation"]
    locked = tensors.splits["locked_test"]
    validation_prediction = _predict_workload(
        fitted["model"], tensors, validation, variant,
        device=device, batch_size=fitted["batch_size"],
    )
    locked_prediction = _predict_workload(
        fitted["model"], tensors, locked, variant,
        device=device, batch_size=fitted["batch_size"],
    )
    _, selected_distribution, validation_distribution = _workload_distribution_score(
        validation.count_target,
        validation.volume_target,
        validation_prediction,
        variant,
    )
    count_validation_raw = _count_quantiles(validation_prediction, variant)
    count_locked_raw = _count_quantiles(locked_prediction, variant)
    count_validation, count_calibration = _calibrate_quantiles(
        validation.count_target,
        count_validation_raw,
        minimum_samples=minimum_calibration_samples,
    )
    count_locked = apply_additive_conformal_quantiles(
        count_locked_raw,
        {
            "corrections": np.asarray(count_calibration["corrections"]),
        },
    )
    count_validation = np.maximum(count_validation, 0.0)
    count_locked = np.maximum(count_locked, 0.0)
    volume_validation: np.ndarray | None = None
    volume_locked: np.ndarray | None = None
    volume_calibration: dict[str, Any] | None = None
    if variant.include_volume_head:
        if selected_distribution not in {"lognormal", "gamma"}:
            raise AssertionError("validation did not select a volume distribution")
        volume_validation_raw = _volume_quantiles(
            validation_prediction, variant, distribution=selected_distribution
        )
        volume_locked_raw = _volume_quantiles(
            locked_prediction, variant, distribution=selected_distribution
        )
        volume_validation, volume_calibration = _calibrate_quantiles(
            validation.volume_target,
            volume_validation_raw,
            minimum_samples=minimum_calibration_samples,
        )
        volume_locked = apply_additive_conformal_quantiles(
            volume_locked_raw,
            {"corrections": np.asarray(volume_calibration["corrections"])},
        )
    saved = torch.load(fitted["checkpoint_path"], map_location="cpu", weights_only=False)
    reloaded = build_workload_variant_model(
        tensors, variant.name, hidden_size=hidden_size
    ).to(device)
    reloaded.load_state_dict(saved["best_model_state"])
    one = WorkloadSplitTensors(
        event_history=validation.event_history[:1],
        elapsed_hours=validation.elapsed_hours[:1],
        regular_grid_history=validation.regular_grid_history[:1],
        count_target=validation.count_target[:1],
        volume_target=validation.volume_target[:1],
        origins_utc=validation.origins_utc[:1],
    )
    reloaded_probe = _predict_workload(
        reloaded, tensors, one, variant, device=device, batch_size=1
    )
    reload_reference = _predict_workload(
        fitted["model"], tensors, one, variant, device=device, batch_size=1
    )
    reload_equal = all(
        np.array_equal(reloaded_probe[key], reload_reference[key])
        for key in reload_reference
    )
    if not reload_equal:
        raise RuntimeError("workload extension checkpoint reload changed inference")
    prediction_arrays: dict[str, np.ndarray] = {
        "validation_count_prediction": np.asarray(count_validation, dtype="float32"),
        "locked_test_count_prediction": np.asarray(count_locked, dtype="float32"),
        "validation_count_target": validation.count_target,
        "locked_test_count_target": locked.count_target,
    }
    if volume_validation is not None and volume_locked is not None:
        prediction_arrays.update(
            {
                "validation_volume_prediction": np.asarray(volume_validation, dtype="float32"),
                "locked_test_volume_prediction": np.asarray(volume_locked, dtype="float32"),
                "validation_volume_target": validation.volume_target,
                "locked_test_volume_target": locked.volume_target,
            }
        )
    prediction_path = (
        Path(output_root)
        / "predictions"
        / "nlr_workload"
        / _safe_slug(variant.name)
        / f"seed_{seed}.npz"
    )
    prediction_artifact = _atomic_npz(prediction_path, **prediction_arrays)
    validation_count_distribution = _workload_distribution_score(
        validation.count_target,
        validation.volume_target,
        validation_prediction,
        variant,
    )[2]["count"]
    locked_count_distribution = _workload_distribution_score(
        locked.count_target,
        locked.volume_target,
        locked_prediction,
        variant,
    )[2]["count"]
    metrics: dict[str, Any] = {
        "validation": {
            "count_distribution": validation_count_distribution,
            "count_quantiles": quantile_metrics(
                validation.count_target, count_validation, quantiles=QUANTILES
            ),
        },
        "locked_test": {
            "count_distribution": locked_count_distribution,
            "count_quantiles": quantile_metrics(
                locked.count_target, count_locked, quantiles=QUANTILES
            ),
        },
    }
    if volume_validation is not None and volume_locked is not None:
        metrics["validation"]["volume_quantiles"] = quantile_metrics(
            validation.volume_target, volume_validation, quantiles=QUANTILES
        )
        metrics["locked_test"]["volume_quantiles"] = quantile_metrics(
            locked.volume_target, volume_locked, quantiles=QUANTILES
        )
    else:
        metrics["validation"]["volume"] = "disabled_by_ablation"
        metrics["locked_test"]["volume"] = "disabled_by_ablation"
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "namespace": "paper_forecast_extensions",
        "task": "nlr_workload",
        "variant": variant.name,
        "seed": int(seed),
        "role": variant.role,
        "model_class": f"{type(fitted['model']).__module__}.{type(fitted['model']).__name__}",
        "architecture": fitted["architecture"],
        "runtime_acceleration": fitted["runtime_acceleration"],
        "training_health": fitted["training_health"],
        "count_distribution_consumed": variant.count_distribution,
        "volume_distribution_head_consumed": variant.include_volume_head,
        "volume_distribution_selected_on_validation": selected_distribution,
        "event_marks_or_cohort_embedding_removed": (
            variant.remove_event_marks_and_cohort_embedding
        ),
        "best_epoch_selected_on_validation": fitted["best_epoch"],
        "best_validation_distribution_score": fitted["best_metric"],
        "epochs_written": fitted["epochs_written"],
        "resumed_from_epoch": fitted["resumed_from_epoch"],
        "resume_fingerprint": fitted["resume_fingerprint"],
        "checkpoint": {
            "path": fitted["checkpoint_path"].as_posix(),
            "sha256": fitted["checkpoint_sha256"],
            "reload_prediction_bitwise_equal": reload_equal,
        },
        "prediction_artifact": prediction_artifact,
        "calibration": {
            "count": count_calibration,
            "volume": volume_calibration,
            "fit_split": "validation",
            "locked_test_used": False,
        },
        "validation_distribution_diagnostics": validation_distribution,
        "metrics": metrics,
        "locked_test_target_sha256": tensors.metadata["locked_test_target_sha256"],
        "locked_test_used_for_selection_or_calibration": False,
    }
    metrics_path = (
        Path(output_root)
        / "metrics"
        / "nlr_workload"
        / _safe_slug(variant.name)
        / f"seed_{seed}.json"
    )
    atomic_json(metrics_path, result)
    result["metrics_artifact"] = {
        "path": metrics_path.as_posix(),
        "sha256": sha256_file(metrics_path),
    }
    return result


def train_workload_extensions(
    tensors: WorkloadExtensionTensors,
    *,
    output_root: Path,
    seeds: Sequence[int] = PREREGISTERED_SEEDS,
    variant_names: Sequence[str] = tuple(WORKLOAD_VARIANTS),
    paper_evidence: bool = True,
    **training_kwargs: Any,
) -> dict[str, Any]:
    seed_values = _seed_gate(seeds, paper_evidence=paper_evidence)
    unknown = sorted(set(variant_names) - set(WORKLOAD_VARIANTS))
    if unknown:
        raise ValueError(f"unknown workload extension variants: {unknown}")
    if paper_evidence and set(variant_names) != set(WORKLOAD_VARIANTS):
        raise ValueError("paper workload extensions require every preregistered ablation")
    results: dict[str, Any] = {}
    locked_hashes: set[str] = set()
    trials_by_variant: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in variant_names
    }
    for seed in seed_values:
        for variant_name in variant_names:
            trial = train_workload_variant_seed(
                tensors,
                variant_name,
                seed=seed,
                output_root=Path(output_root),
                paper_evidence=paper_evidence,
                **training_kwargs,
            )
            trials_by_variant[variant_name].append(trial)
    for variant_name in variant_names:
        trials = trials_by_variant[variant_name]
        locked_hashes.update(str(trial["locked_test_target_sha256"]) for trial in trials)
        selected = min(
            trials,
            key=lambda trial: float(trial["best_validation_distribution_score"]),
        )
        results[variant_name] = {
            "status": "COMPLETE",
            "seed_trials": trials,
            "selected_seed_by_validation_only": selected["seed"],
            "locked_test_used_for_selection": False,
        }
    if len(locked_hashes) != 1:
        raise RuntimeError("workload variants did not use the same locked test")
    return {
        "status": "COMPLETE",
        "variants": results,
        "seeds": list(seed_values),
        "locked_test_target_sha256": next(iter(locked_hashes)),
        "same_locked_test_for_every_seed_and_ablation": True,
    }


def _run_extension_seed_rounds(
    energy_tensors_by_task: Mapping[str, EnergyTaskTensors],
    workload_tensors: WorkloadExtensionTensors,
    *,
    output_root: Path,
    seeds: Sequence[int],
    paper_evidence: bool,
    energy_training_kwargs: Mapping[str, Any],
    workload_training_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Run every model once per seed before advancing to the next seed.

    Round decisions consume training/validation health only. Locked-test metric
    values are never inspected by the diagnostic gate.
    """

    seed_values = _seed_gate(seeds, paper_evidence=paper_evidence)
    energy_trials: dict[tuple[str, str], list[dict[str, Any]]] = {
        (task, variant): []
        for task in ENERGY_TASKS
        for variant in tuple(ENERGY_VARIANTS)
    }
    workload_trials: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in tuple(WORKLOAD_VARIANTS)
    }
    round_records: list[dict[str, Any]] = []
    for round_index, seed in enumerate(seed_values, start=1):
        cells: list[dict[str, Any]] = []
        for task in ENERGY_TASKS:
            for variant in tuple(ENERGY_VARIANTS):
                trial = train_energy_variant_seed(
                    energy_tensors_by_task[task],
                    variant,
                    seed=seed,
                    output_root=output_root,
                    paper_evidence=paper_evidence,
                    **dict(energy_training_kwargs),
                )
                score = float(trial["metrics"]["validation"]["mean_weighted_pinball"])
                if not math.isfinite(score):
                    raise RuntimeError(
                        f"nonfinite validation score in seed round {seed}: {task}/{variant}"
                    )
                energy_trials[(task, variant)].append(trial)
                cells.append(
                    {
                        "family": "energy",
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "validation_score_name": "mean_weighted_pinball",
                        "validation_score": score,
                        "metrics_artifact": trial.get("metrics_artifact"),
                    }
                )
        for variant in tuple(WORKLOAD_VARIANTS):
            trial = train_workload_variant_seed(
                workload_tensors,
                variant,
                seed=seed,
                output_root=output_root,
                paper_evidence=paper_evidence,
                **dict(workload_training_kwargs),
            )
            score = float(trial["best_validation_distribution_score"])
            if not math.isfinite(score):
                raise RuntimeError(
                    f"nonfinite validation score in seed round {seed}: nlr_workload/{variant}"
                )
            workload_trials[variant].append(trial)
            cells.append(
                {
                    "family": "workload",
                    "task": "nlr_workload",
                    "variant": variant,
                    "seed": seed,
                    "validation_score_name": "best_validation_distribution_score",
                    "validation_score": score,
                    "metrics_artifact": trial.get("metrics_artifact"),
                }
            )
        expected_cells = len(ENERGY_TASKS) * len(ENERGY_VARIANTS) + len(WORKLOAD_VARIANTS)
        if len(cells) != expected_cells:
            raise RuntimeError("seed round did not complete every model exactly once")
        round_record = {
            "schema": "forecast_extension_seed_major_round_v1",
            "round_index": round_index,
            "seed": seed,
            "status": "COMPLETE",
            "cell_count": len(cells),
            "expected_cell_count": expected_cells,
            "cells": cells,
            "diagnostic_data_scope": "training_and_validation_only",
            "locked_test_metric_values_consumed_for_decision": False,
            "next_round_decision": (
                "CONTINUE_AUTOMATICALLY" if round_index < len(seed_values) else "ALL_ROUNDS_COMPLETE"
            ),
            "permitted_between_round_changes": [
                "verified_defect_fix",
                "runtime_efficiency",
                "checkpoint_and_resume_reliability",
            ],
            "forbidden_between_round_changes": [
                "locked_test_feedback",
                "metric_driven_model_or_hyperparameter_tuning",
                "candidate_or_data_split_change_without_protocol_amendment",
            ],
        }
        round_path = output_root / "manifests" / f"seed_round_{round_index}_{seed}.json"
        atomic_json(round_path, round_record)
        round_record["artifact"] = {
            "path": round_path.resolve().as_posix(),
            "sha256": sha256_file(round_path),
        }
        round_records.append(round_record)

    energy_tasks: dict[str, Any] = {}
    energy_locked: dict[str, set[str]] = {}
    for task in ENERGY_TASKS:
        task_results: dict[str, Any] = {}
        for variant in tuple(ENERGY_VARIANTS):
            trials = energy_trials[(task, variant)]
            energy_locked.setdefault(task, set()).update(
                str(trial["locked_test_target_sha256"]) for trial in trials
            )
            selected = min(
                trials,
                key=lambda trial: float(
                    trial["metrics"]["validation"]["mean_weighted_pinball"]
                ),
            )
            task_results[variant] = {
                "status": "COMPLETE",
                "seed_trials": trials,
                "selected_seed_by_validation_only": selected["seed"],
                "locked_test_used_for_selection": False,
            }
        energy_tasks[task] = task_results
    if any(len(values) != 1 for values in energy_locked.values()):
        raise RuntimeError("energy seed rounds did not use one locked target per task")
    for task in ENERGY_TASKS:
        if any(
            trial["h1_eligible"]
            for trial in energy_tasks[task]["ORACLE_future_realized_weather"]["seed_trials"]
        ):
            raise RuntimeError("ORACLE weather variant leaked into H1 eligibility")
    energy_result = {
        "status": "COMPLETE",
        "tasks": energy_tasks,
        "seeds": list(seed_values),
        "variants": list(ENERGY_VARIANTS),
        "da_rt_separate": True,
        "oracle_isolated_from_H1": True,
        "execution_order": "seed_major_all_models_once_per_round",
        "locked_test_target_sha256_by_task": {
            task: next(iter(values)) for task, values in energy_locked.items()
        },
    }
    workload_results: dict[str, Any] = {}
    workload_locked: set[str] = set()
    for variant in tuple(WORKLOAD_VARIANTS):
        trials = workload_trials[variant]
        workload_locked.update(str(trial["locked_test_target_sha256"]) for trial in trials)
        selected = min(
            trials, key=lambda trial: float(trial["best_validation_distribution_score"])
        )
        workload_results[variant] = {
            "status": "COMPLETE",
            "seed_trials": trials,
            "selected_seed_by_validation_only": selected["seed"],
            "locked_test_used_for_selection": False,
        }
    if len(workload_locked) != 1:
        raise RuntimeError("workload seed rounds did not use one locked target")
    workload_result = {
        "status": "COMPLETE",
        "variants": workload_results,
        "seeds": list(seed_values),
        "locked_test_target_sha256": next(iter(workload_locked)),
        "same_locked_test_for_every_seed_and_ablation": True,
        "execution_order": "seed_major_all_models_once_per_round",
    }
    return energy_result, workload_result, round_records


def _extract_baseline_run_id(manifest: Mapping[str, Any], path: Path) -> str | None:
    for key in ("run_id", "forecast_run_id", "real_forecasting_run_id"):
        value = manifest.get(key)
        if value:
            return str(value)
    for part in reversed(path.parts):
        if part == "paper-forecast-v1" or part.startswith("paper-forecast-v1"):
            return part
        if part.startswith("run_id="):
            return part.split("=", 1)[1]
    return None


def _baseline_is_terminal_complete(manifest: Mapping[str, Any]) -> bool:
    status = str(
        manifest.get("status", manifest.get("branch_status", ""))
    ).strip().lower()
    status_complete = status in {
        "complete",
        "completed",
        "complete_real_forecasts",
        "complete_forecasts",
    }
    terminal = manifest.get("terminal")
    if terminal is None:
        terminal = manifest.get("completed")
    return status_complete and (terminal is None or bool(terminal))


def validate_v1_baseline_lineage(
    v1_manifest_path: Path,
    *,
    expected_run_id: str = "paper-forecast-v1",
    require_terminal_complete: bool = True,
) -> dict[str, Any]:
    path = Path(v1_manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"v1 baseline manifest is absent: {path}")
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("v1 baseline manifest must be a JSON mapping")
    run_id = _extract_baseline_run_id(manifest, path)
    is_integrity_recovery = (
        run_id == "paper-forecast-v1-recovery-v1"
        and manifest.get("run_kind") == "paper_forecast_v1_integrity_recovery"
    )
    source_original = manifest.get("source_original_v1")
    if is_integrity_recovery:
        if not isinstance(source_original, Mapping):
            raise ValueError("v1 recovery manifest is missing source_original_v1")
        original_root = str(source_original.get("root_path", "")).replace("\\", "/").rstrip("/")
        original_manifest = str(source_original.get("original_manifest_path", "")).replace(
            "\\", "/"
        )
        if not original_root.endswith(f"/{expected_run_id}"):
            raise ValueError("v1 recovery does not identify the expected original run")
        if original_manifest != f"artifacts/real_forecasting/{expected_run_id}/run_manifest.json":
            raise ValueError("v1 recovery original manifest path is not canonical")
        if manifest.get("formal_readiness_allowed") is not True:
            raise ValueError("v1 recovery is not approved for formal readiness")
        if int(manifest.get("actionable_pending_count", -1)) != 0:
            raise ValueError("v1 recovery has actionable pending work")
        if any(manifest.get(key) for key in ("integrity_failures", "task_failures")):
            raise ValueError("v1 recovery contains integrity or task failures")
        for key in (
            "locked_2025_metrics_consumed",
            "locked_2025_metrics_materialized",
            "locked_2025_policy_decisions_consumed",
        ):
            if manifest.get(key) is not False:
                raise ValueError(f"v1 recovery violates the locked-test boundary: {key}")
    elif run_id != expected_run_id:
        raise ValueError(f"extension baseline must be {expected_run_id}, got {run_id!r}")
    if require_terminal_complete and not _baseline_is_terminal_complete(manifest):
        raise ValueError("v1 baseline manifest is not terminal COMPLETE")
    return {
        "baseline_namespace": (
            "artifacts/real_forecasting_recovery"
            if is_integrity_recovery
            else "artifacts/real_forecasting"
        ),
        "baseline_run_id": run_id,
        "baseline_evidence_kind": (
            "hash_verified_integrity_recovery" if is_integrity_recovery else "original_v1"
        ),
        "original_v1_run_id": expected_run_id,
        "original_v1_source": dict(source_original) if is_integrity_recovery else None,
        "baseline_manifest_path": path.as_posix(),
        "baseline_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "baseline_manifest_content_sha256": sha256_json(manifest),
        "baseline_status": manifest.get("status", manifest.get("branch_status")),
        "baseline_terminal_complete": _baseline_is_terminal_complete(manifest),
        "baseline_read_only": True,
        "baseline_outputs_rewritten": False,
    }


def plan_paper_forecast_extensions(
    repository_root: Path,
    *,
    run_id: str,
    v1_manifest_path: Path,
    energy_tensors_by_task: Mapping[str, EnergyTaskTensors],
    workload_tensors: WorkloadExtensionTensors,
    seeds: Sequence[int] = PREREGISTERED_SEEDS,
    output_root: Path | None = None,
    paper_evidence: bool = True,
    require_terminal_v1: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    safe_run_id = _safe_slug(run_id)
    if safe_run_id != run_id:
        raise ValueError("extension run_id must already be filesystem-safe")
    lineage = validate_v1_baseline_lineage(
        v1_manifest_path,
        require_terminal_complete=require_terminal_v1,
    )
    if run_id == lineage["baseline_run_id"]:
        raise ValueError("extension run_id must not reuse or masquerade as the v1 run_id")
    seed_values = _seed_gate(seeds, paper_evidence=paper_evidence)
    if set(energy_tensors_by_task) != set(ENERGY_TASKS):
        raise ValueError("extension plan requires independent DA and RT tensors")
    for task in ENERGY_TASKS:
        if energy_tensors_by_task[task].task != task:
            raise ValueError(f"energy tensor mapping is mislabeled for {task}")
    extension_root = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / "artifacts/real_forecasting_extensions" / f"run_id={run_id}").resolve()
    )
    baseline_path = Path(lineage["baseline_manifest_path"]).resolve()
    baseline_directory = baseline_path.parent
    if extension_root == baseline_directory or extension_root.is_relative_to(baseline_directory):
        raise ValueError("extension outputs must not be written inside the v1 baseline tree")
    v1_artifact_root = (root / "artifacts/real_forecasting").resolve()
    if extension_root == v1_artifact_root or extension_root.is_relative_to(v1_artifact_root):
        raise ValueError("extension outputs require the isolated real_forecasting_extensions namespace")
    request = {
        "schema": "paper_forecast_extension_plan_v1",
        "namespace": "paper_forecast_extensions",
        "run_id": run_id,
        "output_root": extension_root.as_posix(),
        "baseline_lineage": lineage,
        "seeds": list(seed_values),
        "energy_variants": list(ENERGY_VARIANTS),
        "workload_variants": list(WORKLOAD_VARIANTS),
        "energy_tensor_fingerprints": {
            task: energy_tensors_by_task[task].fingerprint for task in ENERGY_TASKS
        },
        "workload_tensor_fingerprint": workload_tensors.fingerprint,
        "locked_test_hashes": {
            "da_lbmp": energy_tensors_by_task["da_lbmp"].metadata[
                "locked_test_target_sha256"
            ],
            "rt_lbmp": energy_tensors_by_task["rt_lbmp"].metadata[
                "locked_test_target_sha256"
            ],
            "nlr_workload": workload_tensors.metadata["locked_test_target_sha256"],
        },
        "da_rt_separate": True,
        "oracle_model_name": "ORACLE_DGPT-E",
        "oracle_excluded_from_H1": True,
        "writes_or_updates_v1": False,
        "paper_evidence": bool(paper_evidence),
        "execution_order": "seed_major_all_models_once_per_round",
    }
    request["scientific_request_fingerprint"] = _extension_scientific_fingerprint(request)
    request["request_fingerprint"] = sha256_json(
        {
            **request,
            "extension_module_sha256": sha256_file(Path(__file__)),
        }
    )
    return request


def _extension_scientific_fingerprint(record: Mapping[str, Any]) -> str:
    baseline = record.get("baseline_lineage")
    baseline = baseline if isinstance(baseline, Mapping) else {}
    return sha256_json(
        {
            "schema": "paper_forecast_extension_scientific_contract_v1",
            "run_id": record.get("run_id"),
            "baseline_manifest_sha256": baseline.get("baseline_manifest_sha256"),
            "original_v1_run_id": baseline.get("original_v1_run_id", "paper-forecast-v1"),
            "seeds": record.get("seeds"),
            "energy_variants": record.get("energy_variants"),
            "workload_variants": record.get("workload_variants"),
            "energy_tensor_fingerprints": record.get("energy_tensor_fingerprints"),
            "workload_tensor_fingerprint": record.get("workload_tensor_fingerprint"),
            "locked_test_hashes": record.get("locked_test_hashes"),
            "da_rt_separate": record.get("da_rt_separate"),
            "oracle_excluded_from_H1": record.get("oracle_excluded_from_H1"),
            "paper_evidence": record.get("paper_evidence"),
        }
    )


def _verify_completed_extension_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "COMPLETE" or not bool(manifest.get("terminal")):
        raise ValueError("extension manifest is not terminal COMPLETE")
    for name, record in dict(manifest.get("output_hashes", {})).items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"completed extension output hash mismatch: {name}")


def run_paper_forecast_extensions(
    repository_root: Path,
    *,
    run_id: str,
    v1_manifest_path: Path,
    energy_tensors_by_task: Mapping[str, EnergyTaskTensors],
    workload_tensors: WorkloadExtensionTensors,
    seeds: Sequence[int] = PREREGISTERED_SEEDS,
    output_root: Path | None = None,
    paper_evidence: bool = True,
    require_terminal_v1: bool = True,
    energy_training_kwargs: Mapping[str, Any] | None = None,
    workload_training_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all post-v1 extensions without mutating the baseline namespace."""

    plan = plan_paper_forecast_extensions(
        repository_root,
        run_id=run_id,
        v1_manifest_path=v1_manifest_path,
        energy_tensors_by_task=energy_tensors_by_task,
        workload_tensors=workload_tensors,
        seeds=seeds,
        output_root=output_root,
        paper_evidence=paper_evidence,
        require_terminal_v1=require_terminal_v1,
    )
    run_root = Path(plan["output_root"])
    manifest_path = run_root / "extension_run_manifest.json"
    existing: dict[str, Any] | None = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("request_fingerprint") != plan["request_fingerprint"]:
            existing_scientific = existing.get("scientific_request_fingerprint")
            if existing_scientific is None:
                existing_scientific = _extension_scientific_fingerprint(existing)
            recoverable_scheduler_failure = bool(
                existing.get("status") == "FAILED"
                and existing.get("error_type") == "ValueError"
                and str(existing.get("error", "")).startswith(
                    "checkpoint fingerprint mismatch:"
                )
                and isinstance(existing.get("request_migration"), Mapping)
                and existing["request_migration"].get("scientific_contract_unchanged")
                is True
            )
            if not (
                existing_scientific == plan["scientific_request_fingerprint"]
                and (
                    existing.get("status") == "RUNNING"
                    or recoverable_scheduler_failure
                )
            ):
                raise ValueError("extension run directory belongs to a different request")
            plan["request_migration"] = {
                "reason": "seed_major_scheduler_only_protocol_amendment",
                "prior_request_fingerprint": existing.get("request_fingerprint"),
                "scientific_request_fingerprint": existing_scientific,
                "scientific_contract_unchanged": True,
                "existing_checkpoints_preserved": True,
                "locked_test_feedback_used": False,
            }
        if existing.get("status") == "COMPLETE" and bool(existing.get("terminal")):
            _verify_completed_extension_manifest(existing)
            return existing
    run_root.mkdir(parents=True, exist_ok=True)
    started = existing.get("started_at_utc", utc_now()) if existing else utc_now()
    atomic_json(
        manifest_path,
        {
            **plan,
            "status": "RUNNING",
            "terminal": False,
            "started_at_utc": started,
            "checkpoint_policy": "atomic_latest_after_every_completed_epoch",
            "resume_supported": True,
        },
    )
    baseline_path = Path(plan["baseline_lineage"]["baseline_manifest_path"])
    baseline_hash = plan["baseline_lineage"]["baseline_manifest_sha256"]
    try:
        energy_result, workload_result, seed_rounds = _run_extension_seed_rounds(
            energy_tensors_by_task,
            workload_tensors,
            output_root=run_root,
            seeds=seeds,
            paper_evidence=paper_evidence,
            energy_training_kwargs=dict(energy_training_kwargs or {}),
            workload_training_kwargs=dict(workload_training_kwargs or {}),
        )
        energy_summary_path = run_root / "manifests/energy_extensions.json"
        atomic_json(energy_summary_path, energy_result)
        workload_summary_path = run_root / "manifests/workload_extensions.json"
        atomic_json(workload_summary_path, workload_result)
        if sha256_file(baseline_path) != baseline_hash:
            raise RuntimeError("frozen v1 baseline manifest changed during extension execution")
        output_hashes = {
            "energy_extensions": {
                "path": energy_summary_path.resolve().as_posix(),
                "sha256": sha256_file(energy_summary_path),
            },
            "workload_extensions": {
                "path": workload_summary_path.resolve().as_posix(),
                "sha256": sha256_file(workload_summary_path),
            },
            **{
                f"seed_round_{record['round_index']}": record["artifact"]
                for record in seed_rounds
            },
        }
        final = {
            **plan,
            "status": "COMPLETE",
            "terminal": True,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "baseline_manifest_sha256_after": sha256_file(baseline_path),
            "baseline_manifest_unchanged": True,
            "v1_outputs_rewritten": False,
            "energy": energy_result,
            "workload": workload_result,
            "output_hashes": output_hashes,
            "seed_rounds": seed_rounds,
            "completion_gates": {
                "three_amended_seeds": tuple(seeds) == PREREGISTERED_SEEDS,
                "all_energy_variants": set(energy_result["variants"]) == set(ENERGY_VARIANTS),
                "all_workload_variants": set(workload_result["variants"]) == set(WORKLOAD_VARIANTS),
                "separate_da_rt": bool(energy_result["da_rt_separate"]),
                "oracle_isolated_from_H1": bool(energy_result["oracle_isolated_from_H1"]),
                "same_locked_tests": bool(
                    workload_result["same_locked_test_for_every_seed_and_ablation"]
                ),
                "baseline_lineage_unchanged": True,
            },
        }
        if paper_evidence and not all(final["completion_gates"].values()):
            raise RuntimeError("formal extension completion gates did not all pass")
        atomic_json(manifest_path, final)
        return final
    except Exception as exc:
        failed = {
            **plan,
            "status": "FAILED",
            "terminal": False,
            "started_at_utc": started,
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "baseline_manifest_unchanged": (
                baseline_path.is_file() and sha256_file(baseline_path) == baseline_hash
            ),
            "v1_outputs_rewritten": False,
            "resume_supported": True,
        }
        atomic_json(manifest_path, failed)
        raise


# ---------------------------------------------------------------------------
# Referenced-data adapter and no-training preflight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencedExtensionPaths:
    repository_root: Path
    nyiso_root: Path
    weather_15m: Path
    pv_15m: Path
    workload_15m: Path
    workload_events: Path
    physical_manifest: Path

    def source_files(self) -> tuple[Path, ...]:
        nyiso = tuple(
            sorted(
                path
                for product in (
                    "da_lbmp",
                    "rt_lbmp",
                    "pal_actual_load",
                    "iso_load_forecast",
                )
                for path in (self.nyiso_root / f"product={product}").rglob("*.parquet")
            )
        )
        return (
            *nyiso,
            self.weather_15m,
            self.pv_15m,
            self.workload_15m,
            self.workload_events,
            self.physical_manifest,
        )


def discover_referenced_extension_paths(
    repository_root: Path,
    *,
    workload_panel_path: Path | None = None,
) -> ReferencedExtensionPaths:
    root = Path(repository_root).resolve()
    paths = ReferencedExtensionPaths(
        repository_root=root,
        nyiso_root=root / "data/processed/referenced/nyiso",
        weather_15m=root / "data/processed/nyiso_zone_weather_15m.parquet",
        pv_15m=root / "data/processed/nyiso_zone_pv_15m.parquet",
        workload_15m=(
            Path(workload_panel_path).resolve()
            if workload_panel_path is not None
            else root
            / "artifacts/real_forecasting/paper-forecast-v1/data/nlr295_15m_workload.parquet"
        ),
        workload_events=root
        / "data/processed/research/physical/run_id=paper-physical-v1/nlr295_workload_events.parquet",
        physical_manifest=root
        / "artifacts/research/physical/run_id=paper-physical-v1/physical_branch_manifest.json",
    )
    missing: list[str] = []
    for product in ("da_lbmp", "rt_lbmp", "pal_actual_load", "iso_load_forecast"):
        product_root = paths.nyiso_root / f"product={product}"
        if not product_root.is_dir() or not any(product_root.rglob("*.parquet")):
            missing.append(product_root.as_posix())
    for path in (
        paths.weather_15m,
        paths.pv_15m,
        paths.workload_15m,
        paths.workload_events,
        paths.physical_manifest,
    ):
        if not path.is_file():
            missing.append(path.as_posix())
    if missing:
        raise FileNotFoundError(f"referenced extension inputs are missing: {missing}")
    return paths


def _source_identity(paths: ReferencedExtensionPaths) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths.source_files():
        resolved = path.resolve()
        records.append(
            {
                "path": resolved.relative_to(paths.repository_root).as_posix(),
                "byte_count": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "file_count": len(records),
        "files": records,
        "source_set_sha256": sha256_json(records),
    }


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _register_referenced_energy_views(connection: Any, paths: ReferencedExtensionPaths) -> None:
    zones = ",".join(f"'{zone.replace(chr(39), chr(39) * 2)}'" for zone in DEFAULT_ZONES)
    nyiso = _sql_path(paths.nyiso_root)
    weather = _sql_path(paths.weather_15m)
    pv = _sql_path(paths.pv_15m)
    connection.execute("SET TimeZone='UTC'")
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW extension_da_hourly AS
        SELECT timestamp_utc, zone, da_lbmp
        FROM read_parquet(
          '{nyiso}/product=da_lbmp/**/*.parquet',
          hive_partitioning=false,
          union_by_name=true
        )
        WHERE zone IN ({zones});

        CREATE OR REPLACE TEMP VIEW extension_rt_15m AS
        WITH nominal AS (
          SELECT timestamp_utc, zone, rt_lbmp
          FROM read_parquet(
            '{nyiso}/product=rt_lbmp/**/*.parquet',
            hive_partitioning=false,
            union_by_name=true
          )
          WHERE zone IN ({zones})
            AND minute(timestamp_utc) % 5 = 0
            AND second(timestamp_utc) = 0
        ), grouped AS (
          SELECT
            time_bucket(
              INTERVAL '15 minutes', timestamp_utc - INTERVAL '1 microsecond'
            ) AS timestamp_utc,
            zone,
            avg(rt_lbmp) AS mean_lbmp,
            count(*) AS valid_interval_count
          FROM nominal
          GROUP BY 1, 2
        )
        SELECT
          timestamp_utc,
          zone,
          CASE WHEN valid_interval_count = 3 THEN mean_lbmp END AS rt_lbmp,
          valid_interval_count
        FROM grouped;

        CREATE OR REPLACE TEMP VIEW extension_pal_15m AS
        WITH nominal AS (
          SELECT timestamp_utc, zone, zone_load_mw, system_load_mw
          FROM read_parquet(
            '{nyiso}/product=pal_actual_load/**/*.parquet',
            hive_partitioning=false,
            union_by_name=true
          )
          WHERE zone IN ({zones})
            AND minute(timestamp_utc) % 5 = 0
            AND second(timestamp_utc) = 0
        ), grouped AS (
          SELECT
            time_bucket(
              INTERVAL '15 minutes', timestamp_utc - INTERVAL '1 microsecond'
            ) AS timestamp_utc,
            zone,
            avg(zone_load_mw) AS zone_mean,
            avg(system_load_mw) AS system_mean,
            count(*) AS valid_interval_count
          FROM nominal
          GROUP BY 1, 2
        )
        SELECT
          timestamp_utc,
          zone,
          CASE WHEN valid_interval_count = 3 THEN zone_mean END AS zone_load,
          CASE WHEN valid_interval_count = 3 THEN system_mean END AS system_load,
          valid_interval_count
        FROM grouped;

        CREATE OR REPLACE TEMP VIEW extension_p7_vintage AS
        WITH source AS (
          SELECT
            timestamp_utc,
            zone,
            forecast_load_mw,
            source_name,
            source_member_sha256,
            strptime(substr(source_name, 1, 8), '%Y%m%d')::DATE AS source_date,
            (
              strptime(substr(source_name, 1, 8), '%Y%m%d')::DATE::TIMESTAMP
              AT TIME ZONE 'America/New_York'
            ) + INTERVAL '36 hours' AS conservative_available_at_utc
          FROM read_parquet(
            '{nyiso}/product=iso_load_forecast/**/*.parquet',
            hive_partitioning=false,
            union_by_name=true
          )
          WHERE zone IN ({zones})
            AND regexp_matches(source_name, '^[0-9]{{8}}isolf\\.csv$')
        ), eligible AS (
          SELECT
            *,
            row_number() OVER (
              PARTITION BY timestamp_utc, zone
              ORDER BY source_date DESC, source_member_sha256 DESC
            ) AS vintage_rank
          FROM source
          WHERE conservative_available_at_utc <= timestamp_utc
        )
        SELECT
          timestamp_utc,
          zone,
          forecast_load_mw,
          source_name,
          source_date,
          conservative_available_at_utc
        FROM eligible
        WHERE vintage_rank = 1;

        CREATE OR REPLACE TEMP VIEW extension_energy_panel AS
        WITH weather AS (
          SELECT
            timestamp_utc,
            zone,
            T2M,
            deployable_available_at_utc
          FROM read_parquet('{weather}', hive_partitioning=false)
          WHERE zone IN ({zones})
            AND timestamp_utc >= TIMESTAMPTZ '2021-01-01 00:00:00+00:00'
            AND timestamp_utc < TIMESTAMPTZ '2026-01-01 00:00:00+00:00'
        ), pv AS (
          SELECT
            timestamp_utc,
            zone,
            pv_dc_availability_mw_per_mwdc,
            deployable_available_at_utc
          FROM read_parquet('{pv}', hive_partitioning=false)
          WHERE zone IN ({zones})
            AND timestamp_utc >= TIMESTAMPTZ '2021-01-01 00:00:00+00:00'
            AND timestamp_utc < TIMESTAMPTZ '2026-01-01 00:00:00+00:00'
        )
        SELECT
          w.timestamp_utc,
          w.zone,
          d.da_lbmp,
          r.rt_lbmp,
          CASE
            WHEN r.rt_lbmp IS NOT NULL AND d.da_lbmp IS NOT NULL
            THEN r.rt_lbmp - d.da_lbmp
          END AS rt_spread,
          p.system_load,
          p.zone_load,
          CASE
            WHEN p.zone_load IS NOT NULL AND f.forecast_load_mw IS NOT NULL
            THEN p.zone_load - f.forecast_load_mw
          END AS load_forecast_error,
          w.T2M AS zone_weather,
          v.pv_dc_availability_mw_per_mwdc AS zone_pv,
          CASE WHEN d.da_lbmp IS NOT NULL
            THEN date_trunc('hour', w.timestamp_utc) + INTERVAL '36 hours'
          END AS da_lbmp_available_at_utc,
          CASE WHEN r.rt_lbmp IS NOT NULL
            THEN r.timestamp_utc + INTERVAL '24 hours 15 minutes'
          END AS rt_lbmp_available_at_utc,
          CASE WHEN r.rt_lbmp IS NOT NULL AND d.da_lbmp IS NOT NULL
            THEN greatest(
              r.timestamp_utc + INTERVAL '24 hours 15 minutes',
              date_trunc('hour', w.timestamp_utc) + INTERVAL '36 hours'
            )
          END AS rt_spread_available_at_utc,
          CASE WHEN p.system_load IS NOT NULL
            THEN p.timestamp_utc + INTERVAL '24 hours 15 minutes'
          END AS system_load_available_at_utc,
          CASE WHEN p.zone_load IS NOT NULL
            THEN p.timestamp_utc + INTERVAL '24 hours 15 minutes'
          END AS zone_load_available_at_utc,
          CASE WHEN p.zone_load IS NOT NULL AND f.forecast_load_mw IS NOT NULL
            THEN greatest(
              p.timestamp_utc + INTERVAL '24 hours 15 minutes',
              f.conservative_available_at_utc
            )
          END AS load_forecast_error_available_at_utc,
          CASE WHEN w.T2M IS NOT NULL THEN w.deployable_available_at_utc END
            AS zone_weather_available_at_utc,
          CASE WHEN v.pv_dc_availability_mw_per_mwdc IS NOT NULL
            THEN v.deployable_available_at_utc
          END AS zone_pv_available_at_utc,
          r.valid_interval_count AS rt_native_interval_count,
          p.valid_interval_count AS pal_native_interval_count,
          f.source_name AS p7_source_name,
          f.source_date AS p7_source_date,
          f.conservative_available_at_utc AS p7_conservative_available_at_utc,
          TRUE AS p7_direct_source_zone_field,
          FALSE AS p7_zone_forecast_disaggregated_or_fabricated
        FROM weather w
        LEFT JOIN extension_da_hourly d
          ON d.zone = w.zone
         AND d.timestamp_utc = date_trunc('hour', w.timestamp_utc)
        LEFT JOIN extension_rt_15m r
          ON r.zone = w.zone AND r.timestamp_utc = w.timestamp_utc
        LEFT JOIN extension_pal_15m p
          ON p.zone = w.zone AND p.timestamp_utc = w.timestamp_utc
        LEFT JOIN extension_p7_vintage f
          ON f.zone = w.zone
         AND f.timestamp_utc = date_trunc('hour', w.timestamp_utc)
        LEFT JOIN pv v
          ON v.zone = w.zone AND v.timestamp_utc = w.timestamp_utc;
        """
    )


def load_referenced_energy_panel(
    repository_root: Path,
    *,
    cache_path: Path | None = None,
    workload_panel_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble the complete issue-masked real-data panel.

    A cache, when requested by the later formal run, lives only in the
    independent forecast-extension namespace.  Preflight calls this adapter in
    streaming mode and does not create the cache.
    """

    import duckdb
    import pyarrow.parquet as pq

    paths = discover_referenced_extension_paths(
        repository_root, workload_panel_path=workload_panel_path
    )
    identity = _source_identity(paths)
    cache = Path(cache_path).resolve() if cache_path is not None else None
    cache_manifest = cache.with_suffix(".manifest.json") if cache is not None else None
    request = sha256_json(
        {
            "adapter_schema": "referenced_energy_issue_safe_panel_v1",
            "source_set_sha256": identity["source_set_sha256"],
            "module_sha256": sha256_file(Path(__file__)),
        }
    )
    if cache is not None and cache.is_file() and cache_manifest is not None and cache_manifest.is_file():
        saved = json.loads(cache_manifest.read_text(encoding="utf-8"))
        if (
            saved.get("request_fingerprint") == request
            and saved.get("output_sha256") == sha256_file(cache)
        ):
            frame = pq.ParquetFile(cache).read().to_pandas()
            return frame, {**saved, "resumed_from_cache": True}
    connection = duckdb.connect()
    _configure_duckdb_threads(connection)
    try:
        _register_referenced_energy_views(connection, paths)
        frame = connection.execute(
            "SELECT * FROM extension_energy_panel ORDER BY timestamp_utc, zone"
        ).fetch_df()
    finally:
        connection.close()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    for feature in ENERGY_HISTORY_BASE_FEATURES:
        availability = f"{feature}_available_at_utc"
        frame[availability] = pd.to_datetime(frame[availability], utc=True, errors="coerce")
    if cache is not None and cache_manifest is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb", suffix=".parquet", dir=cache.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "schema": "referenced_energy_issue_safe_panel_v1",
            "request_fingerprint": request,
            "source_set_sha256": identity["source_set_sha256"],
            "output_path": cache.as_posix(),
            "output_sha256": sha256_file(cache),
            "row_count": len(frame),
            "resumed_from_cache": False,
            "publication_masks": _referenced_publication_policy(),
        }
        atomic_json(cache_manifest, record)
    else:
        record = {
            "schema": "referenced_energy_issue_safe_panel_v1",
            "request_fingerprint": request,
            "source_set_sha256": identity["source_set_sha256"],
            "row_count": len(frame),
            "resumed_from_cache": False,
            "publication_masks": _referenced_publication_policy(),
        }
    return frame, record


def _referenced_publication_policy() -> dict[str, Any]:
    return {
        "da_lbmp": {
            "available_at": "delivery_hour_plus_36_hours",
            "reason": "row-level publication timestamp absent; conservative historical lag",
            "evidence_role": "assumed",
        },
        "rt_lbmp": {
            "available_at": "15min_block_end_plus_24_hours",
            "reason": "row-level publication timestamp absent; conservative historical lag",
            "evidence_role": "assumed",
        },
        "p58b_actual_load": {
            "available_at": "15min_block_end_plus_24_hours",
            "reason": "row-level publication timestamp absent; conservative historical lag",
            "evidence_role": "assumed",
        },
        "p7_load_forecast": {
            "vintage_key": "official_source_name_YYYYMMDDisolf.csv",
            "available_at": "source_date_local_midnight_plus_36_hours",
            "reason": (
                "source file proves a daily vintage but not an intraday publication time; "
                "the next-local-day-noon gate is deliberately conservative"
            ),
            "direct_source_zone_fields_only": True,
            "system_to_zone_disaggregation": False,
            "evidence_role": "assumed_availability_of_referenced_official_record",
        },
        "nasa_weather_and_derived_pv": {
            "available_at": "persisted_deployable_available_at_utc",
            "operational_forecast_claimed": False,
            "evidence_role": "reconstructed",
        },
    }


def _energy_issue_fields(task: str, target_start_utc: Any) -> dict[str, pd.Timestamp]:
    """Return the fixed, preregistered issue/gate contract for one UTC day."""

    target = _utc_timestamp(target_start_utc, name="target_start_utc")
    if task == "da_lbmp":
        issue = target - pd.Timedelta(hours=24)
        gate = target - pd.Timedelta(hours=12)
    elif task == "rt_lbmp":
        issue = target - pd.Timedelta(minutes=15)
        gate = issue
    else:
        raise ValueError(f"task must be one of {ENERGY_TASKS}")
    return {
        "issue_timestamp_utc": issue,
        "gate_timestamp_utc": gate,
        "feature_cutoff_utc": issue,
        "target_start_utc": target,
    }


def build_referenced_energy_origins(
    panel: pd.DataFrame,
    *,
    task: str,
    history_steps: int = 672,
    horizon_steps: int = 96,
) -> pd.DataFrame:
    """Select only complete real-label days and assign frozen chronological splits."""

    if task not in ENERGY_TASKS:
        raise ValueError(f"task must be one of {ENERGY_TASKS}")
    required = {"timestamp_utc", "zone", task}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"referenced energy panel is missing origin fields: {missing}")
    frame = panel.loc[:, list(required)].copy()
    frame["timestamp_utc"] = _utc_series(frame["timestamp_utc"], name="timestamp_utc")
    frame["zone"] = frame["zone"].astype(str)
    frame[task] = pd.to_numeric(frame[task], errors="coerce")
    frame["target_start_utc"] = frame["timestamp_utc"].dt.floor("D")
    daily = frame.groupby("target_start_utc", sort=True).agg(
        row_count=("zone", "size"),
        zone_count=("zone", "nunique"),
        timestamp_count=("timestamp_utc", "nunique"),
        observed_target_count=(task, "count"),
    )
    expected_rows = int(horizon_steps * len(DEFAULT_ZONES))
    complete = daily.loc[
        daily["row_count"].eq(expected_rows)
        & daily["zone_count"].eq(len(DEFAULT_ZONES))
        & daily["timestamp_count"].eq(horizon_steps)
        & daily["observed_target_count"].eq(expected_rows)
    ].reset_index()
    if complete.empty:
        raise ValueError(f"no complete real-label days are available for {task}")
    panel_min = frame["timestamp_utc"].min()
    rows: list[dict[str, Any]] = []
    for target in complete["target_start_utc"]:
        fields = _energy_issue_fields(task, target)
        history_start = fields["feature_cutoff_utc"] - pd.Timedelta(
            minutes=15 * (history_steps - 1)
        )
        if history_start < panel_min:
            continue
        year = int(fields["target_start_utc"].year)
        split = (
            "train"
            if year <= 2023
            else "validation"
            if year == 2024
            else "locked_test"
            if year == 2025
            else None
        )
        if split is None:
            continue
        rows.append({"task": task, "split": split, **fields})
    origins = pd.DataFrame(rows)
    if origins.empty or set(origins["split"]) != set(SPLITS):
        raise ValueError(f"referenced {task} origins do not cover train/validation/locked_test")
    return _validate_origins(origins, task=task)


def _deployment_origins_2025(task: str) -> pd.DataFrame:
    rows = []
    for target in pd.date_range(
        "2025-01-01 00:00:00+00:00",
        "2025-12-31 00:00:00+00:00",
        freq="D",
    ):
        rows.append({"task": task, "split": "deployment", **_energy_issue_fields(task, target)})
    return pd.DataFrame(rows)


def build_referenced_energy_tensors_from_sources(
    repository_root: Path,
    *,
    cache_path: Path | None = None,
    workload_panel_path: Path | None = None,
) -> tuple[dict[str, EnergyTaskTensors], dict[str, Any]]:
    """Stable real-source adapter used by the later formal extension run."""

    panel, adapter = load_referenced_energy_panel(
        repository_root,
        cache_path=cache_path,
        workload_panel_path=workload_panel_path,
    )
    def build_one(task: str) -> tuple[str, EnergyTaskTensors, dict[str, Any]]:
        origins = build_referenced_energy_origins(panel, task=task)
        built = build_issue_safe_energy_tensors(
            panel,
            origins,
            task=task,
            train_end_utc="2023-12-31 23:59:59+00:00",
        )
        record = {
            "split_counts": origins.groupby("split").size().astype(int).to_dict(),
            "origin_sha256": sha256_json(
                [value.isoformat() for value in origins["target_start_utc"]]
            ),
            "locked_test_target_sha256": built.metadata["locked_test_target_sha256"],
        }
        return task, built, record

    tensors: dict[str, EnergyTaskTensors] = {}
    origin_records: dict[str, Any] = {}
    worker_count = min(len(ENERGY_TASKS), _preprocess_thread_count())
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="energy-tensor") as pool:
        for task, built, record in pool.map(build_one, ENERGY_TASKS):
            tensors[task] = built
            origin_records[task] = record
    return tensors, {
        "schema": "referenced_energy_tensor_adapter_v1",
        "adapter": adapter,
        "origins": origin_records,
        "da_rt_separate": True,
        "p7_direct_source_zone_fields_only": True,
        "p7_zone_forecast_disaggregated_or_fabricated": False,
        "weather_scalar_channel": "NASA_POWER_T2M",
        "pv_channel": "NASA_POWER_derived_pv_dc_availability_mw_per_mwdc",
        "parallel_tensor_workers": worker_count,
        "duckdb_threads": _preprocess_thread_count(),
    }


def _load_referenced_workload_safe_panel(
    paths: ReferencedExtensionPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild request-time-safe cohort targets without altering either source.

    The legacy aggregate's cohort-volume columns are explicitly ignored because
    their Parquet metadata identifies ``nodes_used``.  The frozen physical event
    projection does contain request-time ``gpus_requested``; this adapter uses
    that narrower quantity and labels it accordingly.
    """

    import duckdb
    import pyarrow.parquet as pq

    aggregate = pq.ParquetFile(paths.workload_15m).read().to_pandas()
    aggregate["timestamp_utc"] = _utc_series(
        aggregate["timestamp_utc"], name="workload.timestamp_utc"
    )
    aggregate = aggregate.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    if aggregate["timestamp_utc"].duplicated().any():
        raise ValueError("workload aggregate contains duplicate 15-minute timestamps")
    if aggregate["timestamp_utc"].diff().dropna().ne(pd.Timedelta(minutes=15)).any():
        raise ValueError("workload aggregate is not a complete 15-minute grid")
    event_path = _sql_path(paths.workload_events)
    connection = duckdb.connect()
    _configure_duckdb_threads(connection)
    try:
        connection.execute("SET TimeZone='UTC'")
        safe = connection.execute(
            f"""
            WITH marked AS (
              SELECT
                time_bucket(INTERVAL '15 minutes', submit_time_utc) AS timestamp_utc,
                CASE
                  WHEN regexp_matches(
                    lower(coalesce(partition, '') || ' ' || coalesce(qos, '')),
                    'interactive|debug|urgent|high'
                  ) THEN 0
                  WHEN coalesce(gpus_requested, 0) > 0 THEN 2
                  WHEN wallclock_requested_hours IS NOT NULL
                    AND wallclock_requested_hours <= 2 THEN 1
                  ELSE 3
                END AS cohort_index,
                greatest(coalesce(gpus_requested, 0), 0) AS requested_gpu_units
              FROM read_parquet('{event_path}', hive_partitioning=false)
              WHERE submit_time_utc IS NOT NULL
            )
            SELECT
              timestamp_utc,
              count(*)::BIGINT AS safe_total_count,
              sum(CASE WHEN cohort_index=0 THEN 1 ELSE 0 END)::BIGINT
                AS safe_cohort_0_count,
              sum(CASE WHEN cohort_index=1 THEN 1 ELSE 0 END)::BIGINT
                AS safe_cohort_1_count,
              sum(CASE WHEN cohort_index=2 THEN 1 ELSE 0 END)::BIGINT
                AS safe_cohort_2_count,
              sum(CASE WHEN cohort_index=3 THEN 1 ELSE 0 END)::BIGINT
                AS safe_cohort_3_count,
              sum(CASE WHEN cohort_index=0 THEN requested_gpu_units ELSE 0 END)::DOUBLE
                AS safe_cohort_0_volume,
              sum(CASE WHEN cohort_index=1 THEN requested_gpu_units ELSE 0 END)::DOUBLE
                AS safe_cohort_1_volume,
              sum(CASE WHEN cohort_index=2 THEN requested_gpu_units ELSE 0 END)::DOUBLE
                AS safe_cohort_2_volume,
              sum(CASE WHEN cohort_index=3 THEN requested_gpu_units ELSE 0 END)::DOUBLE
                AS safe_cohort_3_volume
            FROM marked
            GROUP BY 1
            ORDER BY 1
            """
        ).fetch_df()
    finally:
        connection.close()
    safe["timestamp_utc"] = _utc_series(safe["timestamp_utc"], name="safe.timestamp_utc")
    merged = aggregate.merge(safe, on="timestamp_utc", how="left", validate="one_to_one")
    safe_columns = [
        "safe_total_count",
        *(f"safe_cohort_{index}_count" for index in range(len(DEFAULT_COHORTS))),
        *(f"safe_cohort_{index}_volume" for index in range(len(DEFAULT_COHORTS))),
    ]
    merged[safe_columns] = merged[safe_columns].fillna(0.0)
    count_mismatches: dict[str, int] = {}
    for index in range(len(DEFAULT_COHORTS)):
        old = pd.to_numeric(merged[f"cohort_{index}_count"], errors="raise").to_numpy()
        new = pd.to_numeric(merged[f"safe_cohort_{index}_count"], errors="raise").to_numpy()
        count_mismatches[f"cohort_{index}"] = int(np.count_nonzero(old != new))
    old_total = pd.to_numeric(merged["submitted_job_count"], errors="raise").to_numpy()
    safe_total = pd.to_numeric(merged["safe_total_count"], errors="raise").to_numpy()
    count_mismatches["total"] = int(np.count_nonzero(old_total != safe_total))
    if any(count_mismatches.values()):
        raise ValueError(
            "frozen event projection and workload aggregate cohort counts disagree: "
            f"{count_mismatches}"
        )
    return merged, {
        "schema": "referenced_workload_safe_panel_v1",
        "event_projection_sha256": sha256_file(paths.workload_events),
        "legacy_workload_panel_sha256": sha256_file(paths.workload_15m),
        "legacy_volume_target_field": str(
            merged["work_volume_target_field"].dropna().iloc[0]
        ),
        "legacy_volume_origin_available": bool(
            merged["work_volume_origin_available"].fillna(False).all()
        ),
        "legacy_cohort_volume_columns_consumed": False,
        "extension_volume_target": "sum_gpus_requested_by_submission_cohort_and_15min",
        "extension_volume_origin_available": True,
        "cohort_count_mismatches": count_mismatches,
        "retrospective_nodes_used_consumed": False,
        "retrospective_processors_used_consumed": False,
    }


def _workload_regular_sequences(
    panel: pd.DataFrame,
    *,
    history_steps: int,
    horizon_steps: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    count_columns = [f"safe_cohort_{index}_count" for index in range(len(DEFAULT_COHORTS))]
    volume_columns = [f"safe_cohort_{index}_volume" for index in range(len(DEFAULT_COHORTS))]
    required = {"timestamp_utc", *count_columns, *volume_columns}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"safe workload panel is missing fields: {missing}")
    frame = panel.sort_values("timestamp_utc", kind="stable").reset_index(drop=True).copy()
    timestamps = _utc_series(frame["timestamp_utc"], name="workload.timestamp_utc")
    if timestamps.diff().dropna().ne(pd.Timedelta(minutes=15)).any():
        raise ValueError("workload sequence adapter requires a complete 15-minute grid")
    counts = frame[count_columns].to_numpy(dtype="float32")
    volumes = frame[volume_columns].to_numpy(dtype="float32")
    if (counts < 0).any() or (volumes < 0).any():
        raise ValueError("workload request-time targets must be nonnegative")
    local = timestamps.dt.tz_convert("America/Denver")
    hour = local.dt.hour.to_numpy(dtype="float64") + local.dt.minute.to_numpy(
        dtype="float64"
    ) / 60.0
    dow = local.dt.dayofweek.to_numpy(dtype="float64")
    raw = np.concatenate(
        [
            np.log1p(counts),
            np.log1p(volumes),
            np.column_stack(
                [
                    np.sin(2 * np.pi * hour / 24.0),
                    np.cos(2 * np.pi * hour / 24.0),
                    np.sin(2 * np.pi * dow / 7.0),
                    np.cos(2 * np.pi * dow / 7.0),
                ]
            ).astype("float32"),
        ],
        axis=1,
    ).astype("float32")
    train_end = math.floor(len(frame) * 0.60)
    validation_end = math.floor(len(frame) * 0.80)
    center = raw[:train_end].mean(axis=0)
    scale = raw[:train_end].std(axis=0)
    scale[scale <= 1e-8] = 1.0
    features = ((raw - center) / scale).astype("float32")
    positions = np.flatnonzero(timestamps.dt.hour.eq(0) & timestamps.dt.minute.eq(0))
    buckets: dict[str, dict[str, list[Any]]] = {
        split: {key: [] for key in ("grid", "count", "volume", "origin")}
        for split in SPLITS
    }
    for position in positions:
        if position < history_steps or position + horizon_steps > len(frame):
            continue
        if position + horizon_steps <= train_end:
            split = "train"
        elif position >= train_end and position + horizon_steps <= validation_end:
            split = "validation"
        elif position >= validation_end:
            split = "locked_test"
        else:
            continue
        buckets[split]["grid"].append(features[position - history_steps : position])
        buckets[split]["count"].append(counts[position : position + horizon_steps].T)
        buckets[split]["volume"].append(volumes[position : position + horizon_steps].T)
        buckets[split]["origin"].append(timestamps.iloc[position])
    output: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        bucket = buckets[split]
        if not bucket["grid"]:
            raise ValueError(f"workload chronological split is empty: {split}")
        output[split] = {
            "regular_grid_history": np.stack(bucket["grid"]).astype("float32"),
            "count_target": np.stack(bucket["count"]).astype("float32"),
            "volume_target": np.stack(bucket["volume"]).astype("float32"),
            "origins": tuple(pd.Timestamp(value).isoformat() for value in bucket["origin"]),
        }
    return output, {
        "regular_grid_feature_names": [
            *(f"log1p_cohort_{index}_submission_count" for index in range(4)),
            *(f"log1p_cohort_{index}_gpus_requested" for index in range(4)),
            "local_hour_sin",
            "local_hour_cos",
            "local_dow_sin",
            "local_dow_cos",
        ],
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "split_counts": {split: len(output[split]["origins"]) for split in SPLITS},
        "split_policy": "chronological_60_20_20_complete_daily_horizons",
        "train_end_exclusive_utc": timestamps.iloc[train_end].isoformat(),
        "validation_end_exclusive_utc": timestamps.iloc[validation_end].isoformat(),
    }


def _workload_event_histories(
    paths: ReferencedExtensionPaths,
    sequences: Mapping[str, Mapping[str, Any]],
    *,
    history_events: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    import duckdb

    origins: list[dict[str, Any]] = []
    split_sizes: dict[str, int] = {}
    for split in SPLITS:
        values = tuple(sequences[split]["origins"])
        split_sizes[split] = len(values)
        for value in values:
            origins.append(
                {
                    "origin_id": len(origins),
                    "origin_utc": _utc_timestamp(value, name="workload_origin"),
                }
            )
    origin_frame = pd.DataFrame(origins)
    event_path = _sql_path(paths.workload_events)
    connection = duckdb.connect()
    _configure_duckdb_threads(connection)
    try:
        connection.execute("SET TimeZone='UTC'")
        connection.register("extension_workload_origins", origin_frame)
        rows = connection.execute(
            f"""
            WITH ordered AS MATERIALIZED (
              SELECT
                row_number() OVER (ORDER BY submit_time_utc, job_key) AS event_row,
                submit_time_utc,
                lag(submit_time_utc) OVER (ORDER BY submit_time_utc, job_key)
                  AS previous_submit_time_utc,
                wallclock_requested_hours,
                gpus_requested,
                partition,
                qos,
                memory_requested_raw
              FROM read_parquet('{event_path}', hive_partitioning=false)
              WHERE submit_time_utc IS NOT NULL
            ), origin_last AS (
              SELECT o.origin_id, o.origin_utc, e.event_row AS last_event_row
              FROM extension_workload_origins o
              ASOF LEFT JOIN ordered e ON o.origin_utc > e.submit_time_utc
            )
            SELECT
              o.origin_id,
              o.origin_utc,
              e.event_row,
              e.submit_time_utc,
              e.previous_submit_time_utc,
              e.wallclock_requested_hours,
              e.gpus_requested,
              e.partition,
              e.qos,
              e.memory_requested_raw
            FROM origin_last o
            JOIN ordered e
              ON e.event_row BETWEEN o.last_event_row - {int(history_events)} + 1
                                 AND o.last_event_row
            ORDER BY o.origin_id, e.event_row
            """
        ).fetch_df()
    finally:
        connection.close()
    counts = rows.groupby("origin_id", sort=True).size().reindex(
        range(len(origins)), fill_value=0
    )
    if not counts.eq(history_events).all():
        raise ValueError(
            "event histories lack the preregistered number of prior events: "
            f"{counts.loc[~counts.eq(history_events)].head().to_dict()}"
        )
    submit = _utc_series(rows["submit_time_utc"], name="event.submit_time_utc")
    origin = _utc_series(rows["origin_utc"], name="event.origin_utc")
    if not (submit < origin).all():
        raise ValueError("workload event history consumed an event at or after its origin")
    previous = pd.to_datetime(rows["previous_submit_time_utc"], utc=True, errors="coerce")
    elapsed = (
        (submit - previous)
        .dt.total_seconds()
        .div(3600.0)
        .fillna(1.0 / 3600.0)
        .clip(1.0 / 3600.0, 168.0)
        .to_numpy(dtype="float32")
    )
    wallclock = pd.to_numeric(rows["wallclock_requested_hours"], errors="coerce")
    gpu = pd.to_numeric(rows["gpus_requested"], errors="coerce")
    partition = rows["partition"].fillna("").astype(str)
    qos = rows["qos"].fillna("").astype(str)
    memory = rows["memory_requested_raw"].fillna("").astype(str)
    text_values = (partition + " " + qos).str.lower()
    cohort = np.select(
        [
            text_values.str.contains("interactive|debug|urgent|high", regex=True),
            gpu.fillna(0).gt(0),
            wallclock.le(2),
        ],
        [0, 2, 1],
        default=3,
    ).astype("int64")
    one_hot = np.eye(len(DEFAULT_COHORTS), dtype="float32")[cohort]
    local = submit.dt.tz_convert("America/Denver")
    hour = local.dt.hour.to_numpy(dtype="float64") + local.dt.minute.to_numpy(
        dtype="float64"
    ) / 60.0
    dow = local.dt.dayofweek.to_numpy(dtype="float64")

    def hashed(values: pd.Series, modulus: int) -> np.ndarray:
        return (
            pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64")
            % modulus
        ).astype("float32") / float(modulus - 1)

    features_flat = np.column_stack(
        [
            np.log1p(wallclock.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
            np.log1p(gpu.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
            hashed(partition, 1024),
            hashed(qos, 256),
            hashed(memory, 1024),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            wallclock.isna().to_numpy(dtype="float32"),
            gpu.isna().to_numpy(dtype="float32"),
            memory.eq("").to_numpy(dtype="float32"),
            one_hot,
        ]
    ).astype("float32")
    shaped = features_flat.reshape(len(origins), history_events, features_flat.shape[-1])
    elapsed_shaped = elapsed.reshape(len(origins), history_events, 1)
    train_count = split_sizes["train"]
    center = shaped[:train_count].mean(axis=(0, 1))
    scale = shaped[:train_count].std(axis=(0, 1))
    scale[scale <= 1e-8] = 1.0
    shaped = ((shaped - center) / scale).astype("float32")
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cursor = 0
    for split in SPLITS:
        count = split_sizes[split]
        output[split] = (
            shaped[cursor : cursor + count],
            elapsed_shaped[cursor : cursor + count],
        )
        cursor += count
    event_feature_names = [
        "mark_log1p_wallclock_requested_hours",
        "mark_log1p_gpus_requested",
        "mark_partition_hash",
        "mark_qos_hash",
        "mark_memory_requested_hash",
        "submission_local_hour_sin",
        "submission_local_hour_cos",
        "submission_local_dow_sin",
        "submission_local_dow_cos",
        "mark_wallclock_missing",
        "mark_gpu_missing",
        "mark_memory_requested_missing",
        *(f"cohort_{name}" for name in DEFAULT_COHORTS),
    ]
    return output, {
        "event_feature_names": event_feature_names,
        "event_feature_center": center.tolist(),
        "event_feature_scale": scale.tolist(),
        "history_events": int(history_events),
        "event_artifact_sha256": sha256_file(paths.workload_events),
        "strictly_pre_origin": True,
        "selected_origin_available_fields": [
            "submit_time_utc",
            "wallclock_requested_hours",
            "gpus_requested",
            "partition",
            "qos",
            "memory_requested_raw",
        ],
        "excluded_retrospective_fields": [
            "start_time_utc",
            "end_time_utc",
            "state",
            "queue_wait_hours",
            "wallclock_used_hours",
            "duration_hours",
            "nodes_used",
            "processors_used",
            "node_hours",
            "processor_hours",
            "gpu_hours",
        ],
        "requested_nodes_field_present": False,
        "requested_processors_field_present": False,
    }


def build_referenced_workload_tensors_from_sources(
    repository_root: Path,
    *,
    workload_panel_path: Path | None = None,
    history_steps: int = 672,
    horizon_steps: int = 96,
    history_events: int = 672,
) -> tuple[WorkloadExtensionTensors, dict[str, Any]]:
    paths = discover_referenced_extension_paths(
        repository_root, workload_panel_path=workload_panel_path
    )
    safe_panel, safe_audit = _load_referenced_workload_safe_panel(paths)
    sequences, regular_audit = _workload_regular_sequences(
        safe_panel,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    events, event_audit = _workload_event_histories(
        paths, sequences, history_events=history_events
    )
    split_tensors: dict[str, WorkloadSplitTensors] = {}
    for split in SPLITS:
        event_history, elapsed = events[split]
        split_tensors[split] = WorkloadSplitTensors(
            event_history=event_history,
            elapsed_hours=elapsed,
            regular_grid_history=sequences[split]["regular_grid_history"],
            count_target=sequences[split]["count_target"],
            volume_target=sequences[split]["volume_target"],
            origins_utc=tuple(sequences[split]["origins"]),
        )
    tensors = make_workload_extension_tensors(
        split_tensors,
        event_feature_names=event_audit["event_feature_names"],
        regular_grid_feature_names=regular_audit["regular_grid_feature_names"],
        metadata={
            "source_adapter": "referenced_workload_safe_panel_v1",
            "volume_target_semantics": safe_audit["extension_volume_target"],
            "volume_target_is_nodes_requested": False,
            "volume_target_is_gpus_requested": True,
            "legacy_nodes_used_volume_excluded": True,
            "physical_event_projection_sha256": event_audit["event_artifact_sha256"],
            "chronological_split_counts": regular_audit["split_counts"],
        },
    )
    return tensors, {
        "schema": "referenced_workload_tensor_adapter_v1",
        "safe_panel": safe_audit,
        "regular_grid": regular_audit,
        "event_history": event_audit,
        "locked_test_target_sha256": tensors.metadata["locked_test_target_sha256"],
    }


def build_referenced_extension_inputs(
    repository_root: Path,
    *,
    run_id: str,
    output_root: Path | None = None,
    workload_panel_path: Path | None = None,
) -> tuple[dict[str, EnergyTaskTensors], WorkloadExtensionTensors, dict[str, Any]]:
    """Build both extension tensor families from frozen referenced artifacts."""

    root = Path(repository_root).resolve()
    safe_run_id = _safe_slug(run_id)
    if safe_run_id != run_id:
        raise ValueError("extension run_id must already be filesystem-safe")
    run_root = (
        Path(output_root).resolve()
        if output_root is not None
        else root / "artifacts/real_forecasting_extensions" / f"run_id={run_id}"
    )
    forbidden = (root / "artifacts/real_forecasting").resolve()
    if run_root.resolve() == forbidden or run_root.resolve().is_relative_to(forbidden):
        raise ValueError("referenced extension inputs cannot be cached in the v1 namespace")
    input_cache = run_root / "input_cache"
    bundle_path = input_cache / "referenced_extension_tensors.pkl"
    bundle_manifest_path = input_cache / "referenced_extension_tensors.manifest.json"
    panel_manifest_path = input_cache / "referenced_energy_panel.manifest.json"
    source_paths = discover_referenced_extension_paths(
        root, workload_panel_path=workload_panel_path
    )
    source_identity = _source_identity(source_paths)
    cache_request = sha256_json(
        {
            "schema": "referenced_forecast_extension_tensor_cache_v1",
            "run_id": run_id,
            "source_set_sha256": source_identity["source_set_sha256"],
            "builder_contract_revision": TENSOR_CACHE_BUILDER_REVISION,
        }
    )
    if bundle_path.is_file() and bundle_manifest_path.is_file():
        saved = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        bundle_hash_matches = saved.get("bundle_sha256") == sha256_file(bundle_path)
        source_hash_matches = (
            saved.get("source_set_sha256") == source_identity["source_set_sha256"]
        )
        request_matches = saved.get("request_fingerprint") == cache_request
        legacy_module_hash_cache = (
            saved.get("schema") == "referenced_forecast_extension_tensor_cache_v1"
            and saved.get("builder_contract_revision") is None
            and source_hash_matches
        )
        if bundle_hash_matches and source_hash_matches and (
            request_matches or legacy_module_hash_cache
        ):
            with bundle_path.open("rb") as handle:
                cached = pickle.load(handle)
            energy = cached.get("energy") if isinstance(cached, Mapping) else None
            workload = cached.get("workload") if isinstance(cached, Mapping) else None
            energy_audit = cached.get("energy_audit") if isinstance(cached, Mapping) else None
            workload_audit = cached.get("workload_audit") if isinstance(cached, Mapping) else None
            fingerprints = {
                **{
                    task: energy[task].fingerprint
                    for task in ENERGY_TASKS
                    if isinstance(energy, Mapping) and task in energy
                },
                **(
                    {"nlr_workload": workload.fingerprint}
                    if isinstance(workload, WorkloadExtensionTensors)
                    else {}
                ),
            }
            if fingerprints != saved.get("tensor_fingerprints"):
                raise ValueError("referenced extension tensor cache fingerprint mismatch")
            if not isinstance(energy_audit, Mapping) or not isinstance(workload_audit, Mapping):
                raise ValueError("referenced extension tensor cache audit payload is invalid")
            cache_record = {
                **saved,
                "request_fingerprint": cache_request,
                "builder_contract_revision": TENSOR_CACHE_BUILDER_REVISION,
                "builder_module_sha256_at_validation": sha256_file(Path(__file__)),
                "resumed_from_cache": True,
            }
            if legacy_module_hash_cache:
                cache_record.update(
                    {
                        "legacy_request_fingerprint": saved.get("request_fingerprint"),
                        "cache_contract_migration": (
                            "scheduler_only_module_change_no_tensor_logic_change; "
                            "source_and_deserialized_tensor_fingerprints_reverified"
                        ),
                    }
                )
            atomic_json(bundle_manifest_path, cache_record)
        else:
            energy = workload = energy_audit = workload_audit = None
            cache_record = None
    else:
        energy = workload = energy_audit = workload_audit = None
        cache_record = None
    if cache_record is None:
        energy, energy_audit = build_referenced_energy_tensors_from_sources(
            root,
            cache_path=run_root / "input_cache/referenced_energy_panel.parquet",
            workload_panel_path=workload_panel_path,
        )
        workload, workload_audit = build_referenced_workload_tensors_from_sources(
            root,
            workload_panel_path=workload_panel_path,
        )
        tensor_fingerprints = {
            **{task: energy[task].fingerprint for task in ENERGY_TASKS},
            "nlr_workload": workload.fingerprint,
        }
        bundle = _atomic_pickle(
            {
                "energy": energy,
                "workload": workload,
                "energy_audit": energy_audit,
                "workload_audit": workload_audit,
            },
            bundle_path,
        )
        cache_record = {
            "schema": "referenced_forecast_extension_tensor_cache_v1",
            "request_fingerprint": cache_request,
            "source_set_sha256": source_identity["source_set_sha256"],
            "builder_contract_revision": TENSOR_CACHE_BUILDER_REVISION,
            "builder_module_sha256_at_creation": sha256_file(Path(__file__)),
            "bundle_path": bundle_path.as_posix(),
            "bundle_sha256": bundle["sha256"],
            "bundle_byte_count": bundle["byte_count"],
            "tensor_fingerprints": tensor_fingerprints,
            "panel_manifest_sha256": (
                sha256_file(panel_manifest_path) if panel_manifest_path.is_file() else None
            ),
            "resumed_from_cache": False,
        }
        atomic_json(bundle_manifest_path, cache_record)
    audit = {
        "schema": "referenced_forecast_extension_inputs_v1",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "energy": energy_audit,
        "workload": workload_audit,
        "tensor_fingerprints": {
            **{task: energy[task].fingerprint for task in ENERGY_TASKS},
            "nlr_workload": workload.fingerprint,
        },
        "locked_test_hashes": {
            **{
                task: energy[task].metadata["locked_test_target_sha256"]
                for task in ENERGY_TASKS
            },
            "nlr_workload": workload.metadata["locked_test_target_sha256"],
        },
        "same_locked_tensor_object_for_every_variant_and_seed": True,
        "tensor_cache": cache_record,
        "writes_or_updates_v1": False,
    }
    audit_path = run_root / "manifests/referenced_input_adapter.json"
    atomic_json(audit_path, audit)
    audit["artifact"] = {
        "path": audit_path.resolve().as_posix(),
        "sha256": sha256_file(audit_path),
    }
    return energy, workload, audit


def run_referenced_paper_forecast_extensions(
    repository_root: Path,
    *,
    run_id: str,
    v1_manifest_path: Path,
    output_root: Path | None = None,
    workload_panel_path: Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Stable one-call entry for the later formal, resumable training run."""

    root = Path(repository_root).resolve()
    run_root = (
        Path(output_root).resolve()
        if output_root is not None
        else root / "artifacts/real_forecasting_extensions" / f"run_id={run_id}"
    )
    energy, workload, _ = build_referenced_extension_inputs(
        root,
        run_id=run_id,
        output_root=run_root,
        workload_panel_path=workload_panel_path,
    )
    return run_paper_forecast_extensions(
        root,
        run_id=run_id,
        v1_manifest_path=Path(v1_manifest_path),
        energy_tensors_by_task=energy,
        workload_tensors=workload,
        output_root=run_root,
        seeds=PREREGISTERED_SEEDS,
        paper_evidence=True,
        require_terminal_v1=True,
        energy_training_kwargs={"device_name": device_name},
        workload_training_kwargs={"device_name": device_name},
    )


def _preflight_energy_sources(
    paths: ReferencedExtensionPaths,
) -> dict[str, Any]:
    import duckdb

    connection = duckdb.connect()
    _configure_duckdb_threads(connection)
    try:
        _register_referenced_energy_views(connection, paths)
        base = connection.execute(
            """
            SELECT
              count(*)::BIGINT AS row_count,
              count(DISTINCT timestamp_utc)::BIGINT AS timestamp_count,
              count(DISTINCT zone)::BIGINT AS zone_count,
              min(timestamp_utc) AS min_timestamp_utc,
              max(timestamp_utc) AS max_timestamp_utc,
              sum(p7_zone_forecast_disaggregated_or_fabricated)::BIGINT
                AS fabricated_p7_row_count,
              sum(p7_direct_source_zone_field)::BIGINT AS direct_p7_zone_row_count
            FROM extension_energy_panel
            """
        ).fetch_df().iloc[0].to_dict()
        panel_min = _utc_timestamp(base["min_timestamp_utc"], name="panel_min")
        feature_availability: dict[str, Any] = {}
        for feature in ENERGY_HISTORY_BASE_FEATURES:
            available = f"{feature}_available_at_utc"
            row = connection.execute(
                f"""
                SELECT
                  sum({feature} IS NOT NULL)::BIGINT AS observed_value_count,
                  sum({feature} IS NULL)::BIGINT AS missing_value_count,
                  sum({feature} IS NOT NULL AND {available} IS NULL)::BIGINT
                    AS observed_without_availability_count,
                  sum({feature} IS NULL AND {available} IS NOT NULL)::BIGINT
                    AS availability_without_observation_count,
                  min({available}) AS min_available_at_utc,
                  max({available}) AS max_available_at_utc
                FROM extension_energy_panel
                """
            ).fetch_df().iloc[0].to_dict()
            feature_availability[feature] = {
                key: (
                    pd.Timestamp(value).isoformat()
                    if isinstance(value, pd.Timestamp)
                    else int(value)
                    if isinstance(value, (np.integer, int))
                    else value
                )
                for key, value in row.items()
            }
        native = connection.execute(
            f"""
            SELECT
              (SELECT count(*) FROM extension_rt_15m)::BIGINT AS rt_group_count,
              (SELECT sum(valid_interval_count=3) FROM extension_rt_15m)::BIGINT
                AS rt_complete_group_count,
              (SELECT sum(valid_interval_count<3) FROM extension_rt_15m)::BIGINT
                AS rt_incomplete_group_count,
              (SELECT count(*) FROM extension_pal_15m)::BIGINT AS pal_group_count,
              (SELECT sum(valid_interval_count=3) FROM extension_pal_15m)::BIGINT
                AS pal_complete_group_count,
              (SELECT sum(valid_interval_count<3) FROM extension_pal_15m)::BIGINT
                AS pal_incomplete_group_count,
              (SELECT count(*) FROM read_parquet(
                '{_sql_path(paths.nyiso_root)}/product=rt_lbmp/**/*.parquet',
                hive_partitioning=false, union_by_name=true
               ) WHERE zone IN ({','.join(repr(zone) for zone in DEFAULT_ZONES)})
                 AND NOT (minute(timestamp_utc)%5=0 AND second(timestamp_utc)=0))::BIGINT
                AS rt_off_cycle_source_row_count,
              (SELECT count(*) FROM read_parquet(
                '{_sql_path(paths.nyiso_root)}/product=pal_actual_load/**/*.parquet',
                hive_partitioning=false, union_by_name=true
               ) WHERE zone IN ({','.join(repr(zone) for zone in DEFAULT_ZONES)})
                 AND NOT (minute(timestamp_utc)%5=0 AND second(timestamp_utc)=0))::BIGINT
                AS pal_off_cycle_source_row_count
            """
        ).fetch_df().iloc[0].to_dict()
        task_records: dict[str, Any] = {}
        for task in ENERGY_TASKS:
            complete_days = connection.execute(
                f"""
                SELECT date_trunc('day', timestamp_utc) AS target_start_utc
                FROM extension_energy_panel
                GROUP BY 1
                HAVING count(*) = {96 * len(DEFAULT_ZONES)}
                   AND count(DISTINCT timestamp_utc) = 96
                   AND count(DISTINCT zone) = {len(DEFAULT_ZONES)}
                   AND count({task}) = {96 * len(DEFAULT_ZONES)}
                ORDER BY 1
                """
            ).fetch_df()
            rows: list[dict[str, Any]] = []
            for target in pd.to_datetime(
                complete_days["target_start_utc"], utc=True, errors="raise"
            ):
                fields = _energy_issue_fields(task, target)
                if fields["feature_cutoff_utc"] - pd.Timedelta(minutes=15 * 671) < panel_min:
                    continue
                year = int(fields["target_start_utc"].year)
                split = (
                    "train"
                    if year <= 2023
                    else "validation"
                    if year == 2024
                    else "locked_test"
                    if year == 2025
                    else None
                )
                if split is not None:
                    rows.append({"task": task, "split": split, **fields})
            origins = pd.DataFrame(rows)
            if origins.empty or set(origins["split"]) != set(SPLITS):
                raise ValueError(f"preflight {task} origins miss a required split")
            locked = origins.loc[origins["split"].eq("locked_test")].copy()
            locked = locked.sort_values("target_start_utc", kind="stable").reset_index(drop=True)
            locked["origin_id"] = np.arange(len(locked), dtype="int64")
            connection.register(
                f"extension_{task}_locked_origins",
                locked[["origin_id", "target_start_utc"]],
            )
            zone_order = "CASE p.zone " + " ".join(
                f"WHEN '{zone}' THEN {index}" for index, zone in enumerate(DEFAULT_ZONES)
            ) + " END"
            target_rows = connection.execute(
                f"""
                SELECT o.origin_id, p.zone, p.timestamp_utc, p.{task} AS target
                FROM extension_{task}_locked_origins o
                JOIN extension_energy_panel p
                  ON p.timestamp_utc >= o.target_start_utc
                 AND p.timestamp_utc < o.target_start_utc + INTERVAL '24 hours'
                ORDER BY o.origin_id, {zone_order}, p.timestamp_utc
                """
            ).fetch_df()
            expected_locked_rows = len(locked) * len(DEFAULT_ZONES) * 96
            if len(target_rows) != expected_locked_rows or target_rows["target"].isna().any():
                raise ValueError(f"preflight {task} locked target tensor is incomplete")
            locked_array = target_rows["target"].to_numpy(dtype="float32").reshape(
                len(locked), len(DEFAULT_ZONES), 96
            )
            deployment = _deployment_origins_2025(task).copy()
            deployment["origin_id"] = np.arange(len(deployment), dtype="int64")
            connection.register(
                f"extension_{task}_deployment_origins",
                deployment[["origin_id", "feature_cutoff_utc", "target_start_utc"]],
            )
            counts_sql = ",\n".join(
                f"sum(p.{feature} IS NOT NULL AND "
                f"p.{feature}_available_at_utc <= o.feature_cutoff_utc)::BIGINT "
                f"AS {feature}_consumable_count"
                for feature in ENERGY_HISTORY_BASE_FEATURES
            )
            history_support = connection.execute(
                f"""
                SELECT
                  o.origin_id,
                  count(*)::BIGINT AS grid_row_count,
                  count(DISTINCT p.timestamp_utc)::BIGINT AS grid_timestamp_count,
                  count(DISTINCT p.zone)::BIGINT AS grid_zone_count,
                  {counts_sql}
                FROM extension_{task}_deployment_origins o
                JOIN extension_energy_panel p
                  ON p.timestamp_utc BETWEEN
                       o.feature_cutoff_utc - INTERVAL '15 minutes' * 671
                       AND o.feature_cutoff_utc
                GROUP BY o.origin_id
                ORDER BY o.origin_id
                """
            ).fetch_df()
            support_columns = [
                f"{feature}_consumable_count" for feature in ENERGY_HISTORY_BASE_FEATURES
            ]
            supported = (
                history_support["grid_row_count"].eq(672 * len(DEFAULT_ZONES))
                & history_support["grid_timestamp_count"].eq(672)
                & history_support["grid_zone_count"].eq(len(DEFAULT_ZONES))
                & history_support[support_columns].gt(0).all(axis=1)
            )
            target_year = complete_days.loc[
                pd.to_datetime(complete_days["target_start_utc"], utc=True).dt.year.eq(2025)
            ]
            task_records[task] = {
                "split_counts": origins.groupby("split").size().astype(int).to_dict(),
                "complete_real_label_day_count_by_year": {
                    str(year): int(count)
                    for year, count in pd.to_datetime(
                        complete_days["target_start_utc"], utc=True
                    ).dt.year.value_counts().sort_index().items()
                },
                "locked_test_complete_day_count": int(len(locked)),
                "locked_test_missing_label_day_count_in_2025": int(365 - len(target_year)),
                "locked_test_target_shape": list(locked_array.shape),
                "locked_test_target_sha256": _array_sha256(locked_array),
                "locked_test_origin_sha256": sha256_json(
                    tuple(value.isoformat() for value in locked["target_start_utc"])
                ),
                "deployment_origin_count": int(len(deployment)),
                "deployment_supported_origin_count": int(supported.sum()),
                "deployment_all_365_origins_supported": bool(
                    len(deployment) == 365 and supported.all()
                ),
                "minimum_consumable_history_values_by_feature": {
                    feature: int(
                        history_support[f"{feature}_consumable_count"].min()
                    )
                    for feature in ENERGY_HISTORY_BASE_FEATURES
                },
                "same_locked_test_for_all_variants_and_seeds": True,
                "issue_contract": (
                    "issue=target_start_minus_24h;gate=target_start_minus_12h"
                    if task == "da_lbmp"
                    else "issue=gate=target_start_minus_15min"
                ),
            }
        future = connection.execute(
            """
            SELECT
              sum(year(timestamp_utc)=2025)::BIGINT AS expected_2025_rows,
              sum(year(timestamp_utc)=2025 AND zone_weather IS NOT NULL)::BIGINT
                AS weather_2025_rows,
              sum(year(timestamp_utc)=2025 AND zone_pv IS NOT NULL)::BIGINT
                AS pv_2025_rows
            FROM extension_energy_panel
            """
        ).fetch_df().iloc[0].to_dict()
    finally:
        connection.close()
    base_record = {
        key: (
            pd.Timestamp(value).isoformat()
            if isinstance(value, pd.Timestamp)
            else int(value)
            if isinstance(value, (np.integer, int))
            else value
        )
        for key, value in base.items()
    }
    return {
        "schema": "referenced_energy_streaming_preflight_v1",
        "panel": base_record,
        "feature_availability": feature_availability,
        "native_interval_quality": {key: int(value) for key, value in native.items()},
        "tasks": task_records,
        "future_realized_weather_oracle_coverage": {
            key: int(value) for key, value in future.items()
        },
        "publication_policy": _referenced_publication_policy(),
        "p7_exact_intraday_publication_timestamp_present": False,
        "p7_vintage_proof": "official_source_filename_calendar_date",
        "p7_zone_values": "direct_fields_from_each_referenced_P7_record",
        "p7_system_to_zone_disaggregation_or_fabrication": False,
        "da_rt_separate": True,
    }


def _preflight_workload_sources(
    paths: ReferencedExtensionPaths,
) -> dict[str, Any]:
    import duckdb
    import pyarrow.parquet as pq

    safe_panel, safe_audit = _load_referenced_workload_safe_panel(paths)
    sequences, regular_audit = _workload_regular_sequences(
        safe_panel, history_steps=672, horizon_steps=96
    )
    event_schema = pq.ParquetFile(paths.workload_events).schema_arrow
    event_fields = set(event_schema.names)
    required_safe_fields = {
        "job_key",
        "submit_time_utc",
        "wallclock_requested_hours",
        "gpus_requested",
        "partition",
        "qos",
        "memory_requested_raw",
    }
    missing_safe_fields = sorted(required_safe_fields - event_fields)
    all_origins = [
        _utc_timestamp(value, name="workload_origin")
        for split in SPLITS
        for value in sequences[split]["origins"]
    ]
    earliest_origin = min(all_origins)
    connection = duckdb.connect()
    _configure_duckdb_threads(connection)
    try:
        connection.execute("SET TimeZone='UTC'")
        event_path = _sql_path(paths.workload_events)
        event_stats = connection.execute(
            f"""
            SELECT
              count(*)::BIGINT AS row_count,
              count(DISTINCT job_key)::BIGINT AS unique_job_count,
              min(submit_time_utc) AS min_submit_time_utc,
              max(submit_time_utc) AS max_submit_time_utc,
              sum(submit_time_utc < TIMESTAMPTZ '{earliest_origin.isoformat()}')::BIGINT
                AS events_before_earliest_origin,
              sum(gpus_requested IS NULL)::BIGINT AS missing_gpu_request_count,
              sum(coalesce(gpus_requested,0)>0)::BIGINT AS positive_gpu_request_count
            FROM read_parquet('{event_path}', hive_partitioning=false)
            WHERE submit_time_utc IS NOT NULL
            """
        ).fetch_df().iloc[0].to_dict()
    finally:
        connection.close()
    locked = sequences["locked_test"]
    locked_hash = sha256_json(
        {
            "count": _array_sha256(locked["count_target"]),
            "volume": _array_sha256(locked["volume_target"]),
            "origins": tuple(locked["origins"]),
        }
    )
    event_feature_count = 16
    return {
        "schema": "referenced_workload_streaming_preflight_v1",
        "safe_panel": safe_audit,
        "grid": {
            "row_count": int(len(safe_panel)),
            "min_timestamp_utc": safe_panel["timestamp_utc"].min().isoformat(),
            "max_timestamp_utc": safe_panel["timestamp_utc"].max().isoformat(),
            "interval_minutes": 15,
            "complete_grid": True,
        },
        "event_projection": {
            "schema_fields": list(event_schema.names),
            "required_origin_available_fields": sorted(required_safe_fields),
            "missing_required_origin_available_fields": missing_safe_fields,
            "requested_nodes_field_present": "nodes_req" in event_fields,
            "requested_processors_field_present": "processors_req" in event_fields,
            "retrospective_nodes_used_selected": False,
            "retrospective_processors_used_selected": False,
            **{
                key: (
                    pd.Timestamp(value).isoformat()
                    if isinstance(value, pd.Timestamp)
                    else int(value)
                    if isinstance(value, (np.integer, int))
                    else value
                )
                for key, value in event_stats.items()
            },
        },
        "tensor_contract": {
            "history_events": 672,
            "history_grid_steps": 672,
            "horizon_steps": 96,
            "event_feature_count": event_feature_count,
            "regular_grid_feature_count": len(regular_audit["regular_grid_feature_names"]),
            "cohort_count": len(DEFAULT_COHORTS),
            "split_counts": regular_audit["split_counts"],
            "expected_event_history_shapes": {
                split: [regular_audit["split_counts"][split], 672, event_feature_count]
                for split in SPLITS
            },
            "regular_grid_history_shapes": {
                split: list(sequences[split]["regular_grid_history"].shape)
                for split in SPLITS
            },
            "target_shapes": {
                split: list(sequences[split]["count_target"].shape)
                for split in SPLITS
            },
            "earliest_origin_utc": earliest_origin.isoformat(),
            "all_origins_have_at_least_672_prior_events": bool(
                int(event_stats["events_before_earliest_origin"]) >= 672
            ),
            "locked_test_target_sha256": locked_hash,
            "same_locked_test_for_all_variants_and_seeds": True,
        },
        "volume_target_semantics": "request_time_gpus_requested_not_nodes_used",
        "legacy_nodes_used_volume_rejected": True,
    }


def preflight_referenced_extension_data(
    repository_root: Path,
    *,
    run_id: str = "paper-forecast-extension-preflight-v1",
    output_path: Path | None = None,
    workload_panel_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the real-data E2E preflight without training or writing the v1 tree."""

    root = Path(repository_root).resolve()
    safe_run_id = _safe_slug(run_id)
    if safe_run_id != run_id:
        raise ValueError("preflight run_id must already be filesystem-safe")
    path = (
        Path(output_path).resolve()
        if output_path is not None
        else root
        / "artifacts/real_forecasting_extensions/preflight"
        / f"run_id={run_id}"
        / "preflight_manifest.json"
    )
    v1_root = (root / "artifacts/real_forecasting").resolve()
    if path == v1_root or path.is_relative_to(v1_root):
        raise ValueError("extension preflight manifest cannot be written in the v1 tree")
    paths = discover_referenced_extension_paths(
        root, workload_panel_path=workload_panel_path
    )
    source_identity = _source_identity(paths)
    request = {
        "schema": "paper_forecast_extension_real_data_preflight_v1",
        "run_id": run_id,
        "repository_root": root.as_posix(),
        "source_set_sha256": source_identity["source_set_sha256"],
        "extension_module_sha256": sha256_file(Path(__file__)),
        "training_requested": False,
        "writes_or_updates_v1": False,
    }
    request_fingerprint = sha256_json(request)
    if resume and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("request_fingerprint") == request_fingerprint
            and bool(existing.get("terminal"))
        ):
            return existing
    started = utc_now()
    atomic_json(
        path,
        {
            **request,
            "request_fingerprint": request_fingerprint,
            "status": "RUNNING",
            "terminal": False,
            "started_at_utc": started,
            "resume_supported": True,
        },
    )
    try:
        energy = _preflight_energy_sources(paths)
        workload = _preflight_workload_sources(paths)
        hard_blockers: list[str] = []
        for task in ENERGY_TASKS:
            record = energy["tasks"][task]
            if not record["deployment_all_365_origins_supported"]:
                hard_blockers.append(f"{task}:not_all_365_deployment_origins_supported")
            if any(int(record["split_counts"].get(split, 0)) == 0 for split in SPLITS):
                hard_blockers.append(f"{task}:empty_required_split")
        for feature, record in energy["feature_availability"].items():
            if int(record["observed_without_availability_count"]) != 0:
                hard_blockers.append(f"{feature}:observed_value_without_availability_time")
        if int(energy["panel"]["fabricated_p7_row_count"]) != 0:
            hard_blockers.append("p7_zone_forecast_was_fabricated")
        if workload["event_projection"]["missing_required_origin_available_fields"]:
            hard_blockers.append("workload_event_projection_missing_safe_features")
        if not workload["tensor_contract"]["all_origins_have_at_least_672_prior_events"]:
            hard_blockers.append("workload_event_history_too_short")
        if any(
            int(workload["tensor_contract"]["split_counts"].get(split, 0)) == 0
            for split in SPLITS
        ):
            hard_blockers.append("workload_empty_required_split")
        limitations = [
            {
                "code": "P7_INTRADAY_PUBLICATION_TIME_UNPROVEN",
                "handling": "official filename date plus conservative 36-hour lag",
                "blocks_formal_extension": False,
            },
            {
                "code": "NASA_POWER_IS_LAGGED_REANALYSIS_NOT_OPERATIONAL_WEATHER_FORECAST",
                "handling": "48-hour persisted availability mask; realized future only in ORACLE",
                "blocks_formal_extension": False,
            },
            {
                "code": "NLR_PHYSICAL_PROJECTION_LACKS_NODES_REQ_AND_PROCESSORS_REQ",
                "handling": (
                    "exclude retrospective nodes_used/processors_used; predict request-time "
                    "gpus_requested volume with explicit narrow semantics"
                ),
                "blocks_formal_extension": False,
            },
            {
                "code": "RT_2025_INCOMPLETE_REAL_LABEL_DAYS",
                "missing_day_count": int(
                    energy["tasks"]["rt_lbmp"][
                        "locked_test_missing_label_day_count_in_2025"
                    ]
                ),
                "handling": "retain missing native intervals; evaluate only complete real-label days",
                "blocks_365_deployment": False,
            },
        ]
        plan = {
            "energy_task_count": len(ENERGY_TASKS),
            "energy_variant_count_per_task": len(ENERGY_VARIANTS),
            "energy_task_variant_cells": len(ENERGY_TASKS) * len(ENERGY_VARIANTS),
            "workload_task_count": 1,
            "workload_variant_count": len(WORKLOAD_VARIANTS),
            "workload_task_variant_cells": len(WORKLOAD_VARIANTS),
            "total_task_variant_cells": (
                len(ENERGY_TASKS) * len(ENERGY_VARIANTS) + len(WORKLOAD_VARIANTS)
            ),
            "seed_count": len(PREREGISTERED_SEEDS),
            "seeds": list(PREREGISTERED_SEEDS),
            "total_seed_runs": (
                len(ENERGY_TASKS) * len(ENERGY_VARIANTS) + len(WORKLOAD_VARIANTS)
            )
            * len(PREREGISTERED_SEEDS),
            "epoch_budget_per_seed_run": 100,
            "formal_training_started_by_preflight": False,
        }
        final = {
            **request,
            "request_fingerprint": request_fingerprint,
            "status": (
                "BLOCKED" if hard_blockers else "READY_WITH_DISCLOSED_LIMITATIONS"
            ),
            "terminal": True,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "source_identity": source_identity,
            "energy": energy,
            "workload": workload,
            "formal_run_plan": plan,
            "hard_blockers": hard_blockers,
            "disclosed_limitations": limitations,
            "same_locked_hashes_contract": {
                "da_lbmp": energy["tasks"]["da_lbmp"]["locked_test_target_sha256"],
                "rt_lbmp": energy["tasks"]["rt_lbmp"]["locked_test_target_sha256"],
                "nlr_workload": workload["tensor_contract"]["locked_test_target_sha256"],
                "one_frozen_target_tensor_per_task_shared_by_all_variants_and_seeds": True,
            },
            "v1_baseline": {
                "manifest_path": (
                    root
                    / "artifacts/real_forecasting/paper-forecast-v1/run_manifest.json"
                ).as_posix(),
                "terminal_lineage_required_at_formal_start": True,
                "frozen_workload_panel_read_by_preflight": True,
                "mutated_by_preflight": False,
            },
            "writes_or_updates_v1": False,
            "training_started": False,
        }
        atomic_json(path, final)
        return final
    except Exception as exc:
        blocked = {
            **request,
            "request_fingerprint": request_fingerprint,
            "status": "BLOCKED",
            "terminal": True,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "source_identity": source_identity,
            "hard_blockers": [f"{type(exc).__name__}:{exc}"],
            "training_started": False,
            "writes_or_updates_v1": False,
            "resume_supported": True,
        }
        atomic_json(path, blocked)
        return blocked


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated post-v1 paper forecasting extensions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="validate real sources without training")
    preflight.add_argument("--root", type=Path, default=Path("."))
    preflight.add_argument(
        "--run-id", default="paper-forecast-extension-preflight-v1"
    )
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--no-resume", action="store_true")
    run = subparsers.add_parser(
        "run-referenced", help="build real tensors and execute the formal extensions"
    )
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--run-id", required=True)
    run.add_argument("--v1-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path)
    run.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.command == "preflight":
        result = preflight_referenced_extension_data(
            args.root,
            run_id=args.run_id,
            output_path=args.output,
            resume=not args.no_resume,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] != "BLOCKED" else 2
    if args.command == "run-referenced":
        result = run_referenced_paper_forecast_extensions(
            args.root,
            run_id=args.run_id,
            v1_manifest_path=args.v1_manifest,
            output_root=args.output_root,
            device_name=args.device,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
