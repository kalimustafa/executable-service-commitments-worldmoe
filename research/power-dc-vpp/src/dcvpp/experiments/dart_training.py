"""Concrete annual expanding-window training adapter for DART forecasts.

The adapter implements the :mod:`dcvpp.experiments.dart_forecast_oos` runner
boundary without changing the already frozen v1 forecasting implementation.
It deliberately keeps DA and RT as separate forecast heads while constructing
both from one source view and the same day-ahead information cutoff.

Paper mode has no synthetic or statistical fallback.  A missing dependency,
unsupported cell, insufficient causal history, or invalid residual-coupling
sample is written as a machine-readable unsupported record and raised.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy import stats

from dcvpp.core import atomic_json, load_yaml, sha256_file, sha256_json, utc_now
from dcvpp.experiments.dart_checkpoint_selection import (
    CheckpointSelectionConfig,
    fit_checkpoint_tail_thresholds,
)
from dcvpp.experiments.dart_forecast_oos import (
    DARTForecastOOSProducer,
    JointDrawRequest,
    JointDrawResult,
    OOSForecastRequest,
    OOSForecastResult,
    QUANTILES,
    STUDY_SEEDS,
)


SCHEMA_VERSION = "dart_training_adapter_v1"
ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
SUPPORTED_CANDIDATES = (
    "seasonal_naive",
    "persistence",
    "lightgbm_quantile",
    "lstm",
    "dlinear",
    "patchtst",
    "itransformer",
    "static_graph_patch_transformer",
    "dgpt_e",
)
NEURAL_CANDIDATES = SUPPORTED_CANDIDATES[3:]
DETERMINISTIC_CANDIDATES = SUPPORTED_CANDIDATES[:2]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DARTTrainingAdapterError(RuntimeError):
    """Base error for an executable training-adapter contract violation."""


class UnsupportedTrainingCellError(DARTTrainingAdapterError):
    """Raised after recording a cell that cannot be executed truthfully."""


class ResidualCouplingError(DARTTrainingAdapterError):
    """Raised when a causal validation-residual coupling cannot be fitted."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DARTTrainingAdapterError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _utc_text(value: Any) -> str:
    return _utc(value, name="timestamp").isoformat().replace("+00:00", "Z")


def _safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)


@dataclass(frozen=True)
class DARTTrainingConfig:
    """Frozen training controls for one adapter instance."""

    history_steps: int = 672
    horizon_steps: int = 96
    interval_minutes: int = 15
    issue_gap_steps: int = 96
    validation_fraction: float = 0.20
    minimum_validation_days: int = 112
    maximum_validation_days: int = 112
    minimum_core_training_days: int = 120
    minimum_coupling_residuals: int = 500
    maximum_feature_staleness_steps: int = 96
    max_epochs: int = 100
    batch_sizes: tuple[int, ...] = (32, 16, 8, 4, 2, 1)
    learning_rate: float = 1.0e-3
    gradient_accumulation_steps: int = 2
    minimum_calibration_samples: int = 500
    lightgbm_estimators: int = 500
    lightgbm_jobs: int = 8
    device_name: str = "auto"
    paper_evidence: bool = True
    feature_contract_id: str = "COMMON_DA_GATE_DART_V1"
    gate_rule: str = "target_start_utc_minus_24_hours"
    checkpoint_calibration_days: int = 56
    checkpoint_selection_days: int = 28
    final_calibration_days: int = 28
    checkpoint_anchor_interval_epochs: int = 5
    checkpoint_minimum_epochs: int = 40
    checkpoint_evaluation_interval_epochs: int = 5
    checkpoint_global_round_epochs: int = 10
    checkpoint_stability_evaluations: int = 3
    checkpoint_early_stopping_min_delta: float = 1.0e-8
    checkpoint_selection: CheckpointSelectionConfig = field(
        default_factory=CheckpointSelectionConfig
    )

    def __post_init__(self) -> None:
        positive = (
            self.history_steps,
            self.horizon_steps,
            self.interval_minutes,
            self.issue_gap_steps,
            self.minimum_validation_days,
            self.minimum_core_training_days,
            self.minimum_coupling_residuals,
            self.maximum_feature_staleness_steps,
            self.max_epochs,
            self.checkpoint_calibration_days,
            self.checkpoint_selection_days,
            self.final_calibration_days,
            self.checkpoint_anchor_interval_epochs,
            self.checkpoint_minimum_epochs,
            self.checkpoint_evaluation_interval_epochs,
            self.checkpoint_global_round_epochs,
            self.checkpoint_stability_evaluations,
        )
        if any(value < 1 for value in positive):
            raise ValueError("DART training dimensions and minima must be positive")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be inside (0, 0.5)")
        if self.maximum_validation_days < self.minimum_validation_days:
            raise ValueError("maximum_validation_days is below its minimum")
        if self.checkpoint_minimum_epochs > self.max_epochs:
            raise ValueError("checkpoint_minimum_epochs exceeds max_epochs")
        if (
            self.checkpoint_global_round_epochs
            < self.checkpoint_evaluation_interval_epochs
            or self.checkpoint_global_round_epochs
            % self.checkpoint_evaluation_interval_epochs
            != 0
        ):
            raise ValueError(
                "checkpoint global round must be a multiple of the assessment interval"
            )
        if (
            not math.isfinite(self.checkpoint_early_stopping_min_delta)
            or self.checkpoint_early_stopping_min_delta < 0.0
        ):
            raise ValueError(
                "checkpoint early-stopping min_delta must be finite and nonnegative"
            )
        minimum_internal_holdout = (
            self.checkpoint_calibration_days
            + self.checkpoint_selection_days
            + self.final_calibration_days
        )
        if (
            self.minimum_validation_days != minimum_internal_holdout
            or self.maximum_validation_days != minimum_internal_holdout
        ):
            raise ValueError(
                "DART internal holdout must equal the explicit checkpoint-calibration, "
                "checkpoint-selection, and final-calibration day counts"
            )
        if (
            self.checkpoint_calibration_days
            < self.checkpoint_selection.minimum_calibration_days
            or self.checkpoint_selection_days
            < self.checkpoint_selection.minimum_selection_days
        ):
            raise ValueError("explicit checkpoint panels are below selector minima")
        if self.paper_evidence:
            expected = (672, 96, 15, 96, 100)
            actual = (
                self.history_steps,
                self.horizon_steps,
                self.interval_minutes,
                self.issue_gap_steps,
                self.max_epochs,
            )
            if actual != expected:
                raise ValueError(
                    "paper DART training fixes history=672, horizon=96, "
                    "interval=15min, issue_gap=96 and max_epochs=100"
                )
            if (
                self.checkpoint_calibration_days,
                self.checkpoint_selection_days,
                self.final_calibration_days,
            ) != (56, 28, 28):
                raise ValueError(
                    "paper DART checkpoint selection fixes chronological 56/28/28-day panels"
                )
            if (
                self.checkpoint_minimum_epochs,
                self.checkpoint_evaluation_interval_epochs,
                self.checkpoint_global_round_epochs,
                self.checkpoint_stability_evaluations,
                self.checkpoint_early_stopping_min_delta,
            ) != (40, 5, 10, 3, 1.0e-8):
                raise ValueError(
                    "paper DART training fixes synchronized three-seed early stopping "
                    "at min=40, assessment=5, global_round=10, stability=3 "
                    "and min_delta=1e-8"
                )
            if (
                self.checkpoint_selection.bootstrap_resamples != 1000
                or self.checkpoint_selection.provisional_bootstrap_resamples != 200
            ):
                raise ValueError(
                    "paper DART checkpoint selection fixes 200 provisional and "
                    "1000 final bootstrap resamples"
                )

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class NYISOPanelBundle:
    """Separate referenced price panels plus source-level provenance."""

    da_15m: pd.DataFrame
    rt_15m: pd.DataFrame
    source_manifest: Mapping[str, Any]


class NYISOPanelSource(Protocol):
    def load(self) -> NYISOPanelBundle:
        """Load referenced DA and RT 15-minute panels without model fitting."""


@dataclass
class ReferencedNYISOPanelSource:
    """Production source backed only by referenced NYISO Parquet partitions."""

    repository_root: Path
    data_root: Path | None = None
    _cache_by_year: dict[int, NYISOPanelBundle] = field(
        default_factory=dict, init=False, repr=False
    )

    def load(self) -> NYISOPanelBundle:
        root = (
            Path(self.data_root)
            if self.data_root is not None
            else Path(self.repository_root) / "data/processed/referenced/nyiso"
        )
        years = [
            int(match.group(1))
            for path in root.rglob("*.parquet")
            if (match := re.search(r"(?:^|[\\/])year=(\d{4})(?:[\\/]|$)", str(path)))
        ]
        if not years:
            raise FileNotFoundError(f"no year-partitioned referenced NYISO data: {root}")
        return self.load_through_year(max(years))

    def load_through_year(self, maximum_year: int) -> NYISOPanelBundle:
        """Load no partition later than ``maximum_year``.

        This prevents the 2022--2024 selection phase from even opening a 2025
        Parquet file.  Within an OOS year, only pre-gate slices are later
        consumed by :func:`build_annual_training_data`.
        """

        maximum_year = int(maximum_year)
        if maximum_year in self._cache_by_year:
            return self._cache_by_year[maximum_year]
        from dcvpp.experiments.torch_forecasts import build_nyiso_price_15m
        import pyarrow.parquet as pq

        root = (
            Path(self.data_root)
            if self.data_root is not None
            else Path(self.repository_root) / "data/processed/referenced/nyiso"
        )
        product_paths: dict[str, list[Path]] = {}
        raw: dict[str, pd.DataFrame] = {}
        audits: dict[str, dict[str, Any]] = {}
        for product in ("da_lbmp", "rt_lbmp"):
            paths: list[Path] = []
            for path in sorted((root / f"product={product}").rglob("*.parquet")):
                match = re.search(r"(?:^|[\\/])year=(\d{4})(?:[\\/]|$)", str(path))
                if match and int(match.group(1)) <= maximum_year:
                    paths.append(path)
            if not paths:
                raise FileNotFoundError(
                    f"no {product} referenced partitions through {maximum_year}"
                )
            frames = [
                pq.ParquetFile(path)
                .read(columns=["timestamp_utc", "zone", product])
                .to_pandas()
                for path in paths
            ]
            frame = pd.concat(frames, ignore_index=True)
            frame["timestamp_utc"] = pd.to_datetime(
                frame["timestamp_utc"], utc=True, errors="raise"
            )
            frame["zone"] = frame["zone"].astype(str)
            frame[product] = pd.to_numeric(frame[product], errors="raise")
            off_cycle = 0
            if product == "rt_lbmp":
                nominal = (
                    frame["timestamp_utc"].dt.minute.mod(5).eq(0)
                    & frame["timestamp_utc"].dt.second.eq(0)
                    & frame["timestamp_utc"].dt.microsecond.eq(0)
                )
                off_cycle = int((~nominal).sum())
                frame = frame.loc[nominal].copy()
            if frame.duplicated(["zone", "timestamp_utc"]).any():
                raise DARTTrainingAdapterError(
                    f"{product} has duplicate zone-times through {maximum_year}"
                )
            if set(frame["zone"]) != set(ZONES):
                raise DARTTrainingAdapterError(
                    f"{product} does not contain the frozen four zones"
                )
            frame = frame.sort_values(["zone", "timestamp_utc"], kind="stable")
            raw[product] = frame.reset_index(drop=True)
            product_paths[product] = paths
            audits[product] = {
                "product": product,
                "maximum_partition_year_opened": maximum_year,
                "input_partition_count": len(paths),
                "input_rows": len(frame),
                "off_cycle_rows_excluded": off_cycle,
                "negative_price_rows": int(frame[product].lt(0).sum()),
                "coverage_min_utc": frame["timestamp_utc"].min().isoformat(),
                "coverage_max_utc": frame["timestamp_utc"].max().isoformat(),
            }
        paths = sorted(path for values in product_paths.values() for path in values)
        path_records = [
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
            }
            for path in paths
        ]
        bundle = NYISOPanelBundle(
            da_15m=build_nyiso_price_15m(raw["da_lbmp"], product="da_lbmp"),
            rt_15m=build_nyiso_price_15m(raw["rt_lbmp"], product="rt_lbmp"),
            source_manifest={
                "source_relationship": "third_party_referenced_public_data",
                "record_type": "referenced_official_historical_market_record",
                "measurement_relationship": "not_measured_by_this_study",
                "market_timezone": "America/New_York",
                "maximum_partition_year_opened": maximum_year,
                "later_partitions_opened": False,
                "da_audit": audits["da_lbmp"],
                "rt_audit": audits["rt_lbmp"],
                "partition_count": len(path_records),
                "partitions_sha256": sha256_json(path_records),
                "partitions": path_records,
            },
        )
        self._cache_by_year[maximum_year] = bundle
        return bundle


@dataclass(frozen=True)
class AnnualTrainingData:
    """One common-gate annual source view and its target-specific tensors."""

    x_core_by_target: Mapping[str, np.ndarray]
    y_core_by_target: Mapping[str, np.ndarray]
    x_checkpoint_calibration_by_target: Mapping[str, np.ndarray]
    y_checkpoint_calibration_by_target: Mapping[str, np.ndarray]
    x_checkpoint_selection_by_target: Mapping[str, np.ndarray]
    y_checkpoint_selection_by_target: Mapping[str, np.ndarray]
    x_validation_by_target: Mapping[str, np.ndarray]
    y_validation_by_target: Mapping[str, np.ndarray]
    x_forecast_by_target: Mapping[str, np.ndarray]
    center_by_target: Mapping[str, np.ndarray]
    scale_by_target: Mapping[str, np.ndarray]
    core_days: tuple[pd.Timestamp, ...]
    checkpoint_calibration_days: tuple[pd.Timestamp, ...]
    checkpoint_selection_days: tuple[pd.Timestamp, ...]
    validation_days: tuple[pd.Timestamp, ...]
    forecast_days: tuple[pd.Timestamp, ...]
    forecast_event_keys: tuple[str, ...]
    forecast_target_times_utc: tuple[str, ...]
    forecast_gate_times_utc: tuple[str, ...]
    forecast_max_feature_availability_utc: tuple[str, ...]
    validation_event_keys: tuple[str, ...]
    validation_target_times_utc: tuple[str, ...]
    tail_thresholds_by_target: Mapping[str, Mapping[str, Any]]
    tail_threshold_contract_sha256_by_target: Mapping[str, str]
    training_target_max_utc: str
    information_set_fingerprint: str
    training_data_fingerprint: str
    source_manifest: Mapping[str, Any]
    audit: Mapping[str, Any]


def _matrix(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    required = {"timestamp_utc", "zone", target}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DARTTrainingAdapterError(f"{target} panel lacks columns: {missing}")
    rows = frame[["timestamp_utc", "zone", target]].copy()
    rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True, errors="raise")
    rows["zone"] = rows["zone"].astype(str)
    rows[target] = pd.to_numeric(rows[target], errors="coerce")
    if rows.duplicated(["timestamp_utc", "zone"]).any():
        raise DARTTrainingAdapterError(f"{target} panel has duplicate zone-times")
    if set(rows["zone"]) != set(ZONES):
        raise DARTTrainingAdapterError(f"{target} panel does not contain the four frozen zones")
    return rows.pivot(index="timestamp_utc", columns="zone", values=target).reindex(columns=ZONES)


def _event_contract(
    days: Sequence[pd.Timestamp],
    *,
    horizon_steps: int,
    interval_minutes: int,
    issue_gap_steps: int,
    max_availability_by_day: Mapping[pd.Timestamp, pd.Timestamp],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    keys: list[str] = []
    targets: list[str] = []
    gates: list[str] = []
    availability: list[str] = []
    frequency = pd.Timedelta(minutes=interval_minutes)
    for day in days:
        target_times = pd.date_range(day, periods=horizon_steps, freq=frequency, tz="UTC")
        gate = day - frequency * issue_gap_steps
        max_available = max_availability_by_day[day]
        for zone in ZONES:
            for timestamp in target_times:
                target_text = _utc_text(timestamp)
                keys.append(f"{target_text}|{zone}")
                targets.append(target_text)
                gates.append(_utc_text(gate))
                availability.append(_utc_text(max_available))
    return tuple(keys), tuple(targets), tuple(gates), tuple(availability)


def build_annual_training_data(
    bundle: NYISOPanelBundle,
    request: OOSForecastRequest,
    config: DARTTrainingConfig,
) -> AnnualTrainingData:
    """Construct the common pre-gate view without consuming forecast targets."""

    if tuple(float(value) for value in request.quantiles) != QUANTILES:
        raise DARTTrainingAdapterError(f"quantiles must be exactly {QUANTILES}")
    if request.fit_policy != "expanding_annual_refit" or request.daily_retraining:
        raise DARTTrainingAdapterError("only annual expanding-window refits are supported")
    start = _utc(request.training_start_utc, name="training_start_utc").normalize()
    cutoff = _utc(
        request.training_cutoff_exclusive_utc,
        name="training_cutoff_exclusive_utc",
    )
    forecast_start = _utc(request.forecast_start_utc, name="forecast_start_utc").normalize()
    forecast_end = _utc(
        request.forecast_end_exclusive_utc,
        name="forecast_end_exclusive_utc",
    )
    if cutoff != forecast_start:
        raise DARTTrainingAdapterError("annual training cutoff must equal forecast-year start")
    if forecast_start >= forecast_end:
        raise DARTTrainingAdapterError("forecast interval is empty")
    if request.forecast_year != forecast_start.year or request.fit_through_year + 1 != request.forecast_year:
        raise DARTTrainingAdapterError("request does not describe a next-year annual origin")

    raw = {
        "da_lbmp": _matrix(bundle.da_15m, "da_lbmp"),
        "rt_lbmp_at_da_gate": _matrix(bundle.rt_15m, "rt_lbmp"),
    }
    frequency = pd.Timedelta(minutes=config.interval_minutes)
    minimum_source = min(frame.index.min() for frame in raw.values()).floor(
        f"{config.interval_minutes}min"
    )
    maximum_required = forecast_end - frequency * config.issue_gap_steps
    maximum_source = max(frame.index.max() for frame in raw.values()).floor(
        f"{config.interval_minutes}min"
    )
    grid_end = max(maximum_source, maximum_required)
    full_index = pd.date_range(
        minimum_source,
        grid_end,
        freq=frequency,
        tz="UTC",
    )
    raw = {name: frame.reindex(full_index) for name, frame in raw.items()}

    fit_days_all = tuple(
        pd.date_range(start, cutoff - pd.Timedelta(days=1), freq="D", tz="UTC")
    )
    eligible_fit_days: list[pd.Timestamp] = []
    positions = {timestamp: index for index, timestamp in enumerate(full_index)}
    for day in fit_days_all:
        gate = day - frequency * config.issue_gap_steps
        history_end = positions.get(gate)
        target_start = positions.get(day)
        if history_end is None or target_start is None or history_end < config.history_steps:
            continue
        target_slice = slice(target_start, target_start + config.horizon_steps)
        if target_slice.stop > len(full_index):
            continue
        if all(np.isfinite(frame.iloc[target_slice].to_numpy(dtype="float64")).all() for frame in raw.values()):
            eligible_fit_days.append(day)
    validation_count = (
        config.checkpoint_calibration_days
        + config.checkpoint_selection_days
        + config.final_calibration_days
    )
    if len(eligible_fit_days) < config.minimum_core_training_days + validation_count:
        raise UnsupportedTrainingCellError(
            "insufficient common complete pre-cutoff days for core training and validation"
        )
    internal_holdout_days = tuple(eligible_fit_days[-validation_count:])
    core_days = tuple(eligible_fit_days[:-validation_count])
    checkpoint_calibration_end = int(config.checkpoint_calibration_days)
    checkpoint_selection_end = (
        checkpoint_calibration_end + int(config.checkpoint_selection_days)
    )
    checkpoint_calibration_days = internal_holdout_days[:checkpoint_calibration_end]
    checkpoint_selection_days = internal_holdout_days[
        checkpoint_calibration_end:checkpoint_selection_end
    ]
    validation_days = internal_holdout_days[checkpoint_selection_end:]
    if not (
        len(checkpoint_calibration_days)
        == config.checkpoint_calibration_days
        and len(checkpoint_selection_days) == config.checkpoint_selection_days
        and len(validation_days) == config.final_calibration_days
    ):
        raise UnsupportedTrainingCellError(
            "a chronological checkpoint/final-calibration panel is below its minimum"
        )
    ordered_internal_days = (
        checkpoint_calibration_days + checkpoint_selection_days + validation_days
    )
    if ordered_internal_days != internal_holdout_days or not (
        core_days[-1] < checkpoint_calibration_days[0]
        < checkpoint_selection_days[0]
        < validation_days[0]
    ):
        raise DARTTrainingAdapterError(
            "core/checkpoint-calibration/checkpoint-selection/final-calibration "
            "chronology is invalid"
        )
    forecast_days = tuple(
        pd.date_range(
            forecast_start,
            forecast_end - pd.Timedelta(days=1),
            freq="D",
            tz="UTC",
        )
    )
    if not forecast_days:
        raise UnsupportedTrainingCellError("forecast-year day grid is empty")

    center: dict[str, np.ndarray] = {}
    scale: dict[str, np.ndarray] = {}
    for target, frame in raw.items():
        values: list[np.ndarray] = []
        for day in core_days:
            start_index = positions[day]
            values.append(
                frame.iloc[start_index : start_index + config.horizon_steps].to_numpy(
                    dtype="float64"
                )
            )
        stacked = np.concatenate(values, axis=0)
        center[target] = np.nanmean(stacked, axis=0).astype("float32")
        deviation = np.nanstd(stacked, axis=0)
        deviation[~np.isfinite(deviation) | (deviation <= 1.0e-8)] = 1.0
        scale[target] = deviation.astype("float32")
        if not np.isfinite(center[target]).all():
            raise UnsupportedTrainingCellError(f"{target} has no finite core-training center")

    filled = {
        target: frame.ffill().fillna(
            pd.Series(center[target], index=frame.columns)
        )
        for target, frame in raw.items()
    }

    def sample(
        days: Sequence[pd.Timestamp],
        *,
        require_targets: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[pd.Timestamp, pd.Timestamp]]:
        x_values: dict[str, list[np.ndarray]] = {target: [] for target in raw}
        y_values: dict[str, list[np.ndarray]] = {target: [] for target in raw}
        max_availability: dict[pd.Timestamp, pd.Timestamp] = {}
        for day in days:
            gate = day - frequency * config.issue_gap_steps
            history_end = positions.get(gate)
            target_start = positions.get(day)
            if history_end is None or history_end < config.history_steps:
                raise UnsupportedTrainingCellError(
                    f"causal history is absent at gate {gate.isoformat()}"
                )
            history_slice = slice(history_end - config.history_steps, history_end)
            lookback_start = max(0, history_end - config.maximum_feature_staleness_steps)
            recent_observed = [
                frame.iloc[lookback_start:history_end].notna().any(axis=0).all()
                for frame in raw.values()
            ]
            if not all(bool(value) for value in recent_observed):
                raise UnsupportedTrainingCellError(
                    f"feature staleness exceeds {config.maximum_feature_staleness_steps} "
                    f"steps at gate {gate.isoformat()}"
                )
            latest_candidates: list[pd.Timestamp] = []
            for frame in raw.values():
                availability_window = frame.iloc[lookback_start:history_end]
                observed = availability_window.notna().any(axis=1)
                if observed.any():
                    latest_candidates.append(
                        availability_window.index[observed.to_numpy()].max()
                    )
            if not latest_candidates:
                raise UnsupportedTrainingCellError(f"no causal feature is available at {gate}")
            max_availability[day] = max(latest_candidates)
            if max_availability[day] >= gate:
                raise DARTTrainingAdapterError("history admitted a value at or after the DA gate")
            for target in raw:
                history = filled[target].iloc[history_slice].to_numpy(dtype="float32")
                x_values[target].append(
                    ((history - center[target]) / scale[target]).astype("float32")
                )
                if require_targets:
                    if target_start is None:
                        raise UnsupportedTrainingCellError(f"target grid lacks {day}")
                    target_values = raw[target].iloc[
                        target_start : target_start + config.horizon_steps
                    ].to_numpy(dtype="float32").T
                    if not np.isfinite(target_values).all():
                        raise UnsupportedTrainingCellError(
                            f"pre-cutoff target is incomplete for {target}/{day.date()}"
                        )
                    y_values[target].append(target_values)
        x = {
            target: np.stack(values).astype("float32")
            for target, values in x_values.items()
        }
        y = {
            target: np.stack(values).astype("float32")
            for target, values in y_values.items()
        } if require_targets else {}
        return x, y, max_availability

    x_core, y_core, _ = sample(core_days, require_targets=True)
    x_checkpoint_calibration, y_checkpoint_calibration, _ = sample(
        checkpoint_calibration_days, require_targets=True
    )
    x_checkpoint_selection, y_checkpoint_selection, _ = sample(
        checkpoint_selection_days, require_targets=True
    )
    x_validation, y_validation, validation_availability = sample(
        validation_days, require_targets=True
    )
    x_forecast, _, forecast_availability = sample(
        forecast_days, require_targets=False
    )
    forecast_contract = _event_contract(
        forecast_days,
        horizon_steps=config.horizon_steps,
        interval_minutes=config.interval_minutes,
        issue_gap_steps=config.issue_gap_steps,
        max_availability_by_day=forecast_availability,
    )
    validation_contract = _event_contract(
        validation_days,
        horizon_steps=config.horizon_steps,
        interval_minutes=config.interval_minutes,
        issue_gap_steps=config.issue_gap_steps,
        max_availability_by_day=validation_availability,
    )
    training_target_max = validation_days[-1] + frequency * (config.horizon_steps - 1)
    if training_target_max >= cutoff:
        raise DARTTrainingAdapterError("training targets cross the annual cutoff")

    tail_thresholds: dict[str, dict[str, Any]] = {}
    tail_threshold_hashes: dict[str, str] = {}
    for target, target_values in y_core.items():
        contract = fit_checkpoint_tail_thresholds(
            np.asarray(target_values, dtype="float64"), zone_names=ZONES
        )
        tail_thresholds[target] = contract
        tail_threshold_hashes[target] = str(contract["threshold_contract_sha256"])

    feature_contract = {
        "profile_id": config.feature_contract_id,
        "gate_rule": config.gate_rule,
        "issue_gap_steps": config.issue_gap_steps,
        "history_steps": config.history_steps,
        "horizon_steps": config.horizon_steps,
        "interval_minutes": config.interval_minutes,
        "source_view": ["historical_da_lbmp", "historical_rt_lbmp"],
        "candidate_transform": "target_specific_history_from_common_source_view",
        "future_weather_consumed": False,
        "later_intraday_rt_updates_consumed": False,
        "realized_forecast_targets_consumed": False,
        "causal_missing_value_rule": "forward_fill_then_core_training_center",
    }
    information_contract = {
        "origin_name": request.origin_name,
        "forecast_year": request.forecast_year,
        "feature_contract": feature_contract,
        "forecast_event_keys_sha256": sha256_json(forecast_contract[0]),
        "forecast_target_times_sha256": sha256_json(forecast_contract[1]),
        "forecast_gate_times_sha256": sha256_json(forecast_contract[2]),
        "forecast_feature_availability_sha256": sha256_json(forecast_contract[3]),
        "forecast_da_history_sha256": _array_sha256(x_forecast["da_lbmp"]),
        "forecast_rt_history_sha256": _array_sha256(
            x_forecast["rt_lbmp_at_da_gate"]
        ),
        # Hash only values actually available in the common forecast views.
        # A repository-wide partition hash could disclose/hash future target
        # files even though their values are not consumed by inference.
        "source_relationship": "third_party_referenced_public_data",
    }
    information_fingerprint = sha256_json(information_contract)
    training_fingerprint = sha256_json(
        {
            "information_set_fingerprint": information_fingerprint,
            "config_fingerprint": config.fingerprint,
            "core_days": [_utc_text(value) for value in core_days],
            "checkpoint_calibration_days": [
                _utc_text(value) for value in checkpoint_calibration_days
            ],
            "checkpoint_selection_days": [
                _utc_text(value) for value in checkpoint_selection_days
            ],
            "final_calibration_days": [
                _utc_text(value) for value in validation_days
            ],
            "x_core": {key: _array_sha256(value) for key, value in x_core.items()},
            "y_core": {key: _array_sha256(value) for key, value in y_core.items()},
            "x_checkpoint_calibration": {
                key: _array_sha256(value)
                for key, value in x_checkpoint_calibration.items()
            },
            "y_checkpoint_calibration": {
                key: _array_sha256(value)
                for key, value in y_checkpoint_calibration.items()
            },
            "x_checkpoint_selection": {
                key: _array_sha256(value)
                for key, value in x_checkpoint_selection.items()
            },
            "y_checkpoint_selection": {
                key: _array_sha256(value)
                for key, value in y_checkpoint_selection.items()
            },
            "x_validation": {
                key: _array_sha256(value) for key, value in x_validation.items()
            },
            "y_validation": {
                key: _array_sha256(value) for key, value in y_validation.items()
            },
            "center": {key: value.tolist() for key, value in center.items()},
            "scale": {key: value.tolist() for key, value in scale.items()},
            "tail_threshold_contract_sha256_by_target": tail_threshold_hashes,
        }
    )
    audit = {
        "schema": SCHEMA_VERSION,
        "feature_contract": feature_contract,
        "common_pre_da_gate_information": True,
        "separate_da_rt_heads": True,
        "forecast_targets_read_for_inference": False,
        "future_information_used": False,
        "core_training_day_count": len(core_days),
        "checkpoint_calibration_day_count": len(checkpoint_calibration_days),
        "checkpoint_selection_day_count": len(checkpoint_selection_days),
        "final_calibration_day_count": len(validation_days),
        "validation_day_count": len(validation_days),
        "internal_holdout_chronology": [
            "checkpoint_calibration_fit",
            "checkpoint_selection",
            "final_calibration_and_residual_coupling",
        ],
        "internal_holdout_panels": {
            "checkpoint_calibration_fit": {
                "date_sha256": sha256_json(
                    [_utc_text(value) for value in checkpoint_calibration_days]
                ),
                "x_sha256_by_target": {
                    key: _array_sha256(value)
                    for key, value in x_checkpoint_calibration.items()
                },
                "y_sha256_by_target": {
                    key: _array_sha256(value)
                    for key, value in y_checkpoint_calibration.items()
                },
            },
            "checkpoint_selection": {
                "date_sha256": sha256_json(
                    [_utc_text(value) for value in checkpoint_selection_days]
                ),
                "x_sha256_by_target": {
                    key: _array_sha256(value)
                    for key, value in x_checkpoint_selection.items()
                },
                "y_sha256_by_target": {
                    key: _array_sha256(value)
                    for key, value in y_checkpoint_selection.items()
                },
            },
            "final_calibration_and_residual_coupling": {
                "date_sha256": sha256_json(
                    [_utc_text(value) for value in validation_days]
                ),
                "x_sha256_by_target": {
                    key: _array_sha256(value) for key, value in x_validation.items()
                },
                "y_sha256_by_target": {
                    key: _array_sha256(value) for key, value in y_validation.items()
                },
            },
        },
        "checkpoint_selection_config_fingerprint": (
            config.checkpoint_selection.fingerprint
        ),
        "tail_threshold_contract_sha256_by_target": tail_threshold_hashes,
        "forecast_day_count": len(forecast_days),
        "forecast_event_count": len(forecast_contract[0]),
        "training_target_max_utc": _utc_text(training_target_max),
        "information_contract": information_contract,
    }
    return AnnualTrainingData(
        x_core_by_target=x_core,
        y_core_by_target=y_core,
        x_checkpoint_calibration_by_target=x_checkpoint_calibration,
        y_checkpoint_calibration_by_target=y_checkpoint_calibration,
        x_checkpoint_selection_by_target=x_checkpoint_selection,
        y_checkpoint_selection_by_target=y_checkpoint_selection,
        x_validation_by_target=x_validation,
        y_validation_by_target=y_validation,
        x_forecast_by_target=x_forecast,
        center_by_target=center,
        scale_by_target=scale,
        core_days=core_days,
        checkpoint_calibration_days=checkpoint_calibration_days,
        checkpoint_selection_days=checkpoint_selection_days,
        validation_days=validation_days,
        forecast_days=forecast_days,
        forecast_event_keys=forecast_contract[0],
        forecast_target_times_utc=forecast_contract[1],
        forecast_gate_times_utc=forecast_contract[2],
        forecast_max_feature_availability_utc=forecast_contract[3],
        validation_event_keys=validation_contract[0],
        validation_target_times_utc=validation_contract[1],
        tail_thresholds_by_target=tail_thresholds,
        tail_threshold_contract_sha256_by_target=tail_threshold_hashes,
        training_target_max_utc=_utc_text(training_target_max),
        information_set_fingerprint=information_fingerprint,
        training_data_fingerprint=training_fingerprint,
        source_manifest=dict(bundle.source_manifest),
        audit=audit,
    )


def _point_baseline(
    x: np.ndarray,
    *,
    candidate: str,
    horizon: int,
    issue_gap: int,
) -> np.ndarray:
    if candidate == "persistence":
        return np.repeat(x[:, -1, :].T[:, :, None], horizon, axis=2).transpose(1, 0, 2)
    if candidate != "seasonal_naive":
        raise ValueError(candidate)
    lag_steps = 7 * horizon - issue_gap
    start = x.shape[1] - lag_steps
    stop = start + horizon
    if start < 0 or stop > x.shape[1]:
        raise UnsupportedTrainingCellError("history does not contain the seven-day seasonal slice")
    return x[:, start:stop, :].transpose(0, 2, 1)


def _design_lightgbm(x: np.ndarray, horizon: int) -> np.ndarray:
    samples, history, sites = x.shape
    rows = np.zeros((samples, sites, horizon, 11), dtype="float32")
    recent = x[:, -min(history, horizon) :, :]
    mean = recent.mean(axis=1)
    standard_deviation = recent.std(axis=1)
    for step in range(horizon):
        rows[:, :, step, 0] = x[:, -1, :]
        rows[:, :, step, 1] = mean
        rows[:, :, step, 2] = standard_deviation
        rows[:, :, step, 3] = math.sin(2.0 * math.pi * step / horizon)
        rows[:, :, step, 4] = math.cos(2.0 * math.pi * step / horizon)
        rows[:, :, step, 5] = step / max(horizon - 1, 1)
        for zone_index in range(sites):
            rows[:, zone_index, step, 6 + zone_index] = 1.0
        source_step = max(0, recent.shape[1] - horizon + step)
        source_step = min(source_step, recent.shape[1] - 1)
        rows[:, :, step, 10] = recent[:, source_step, :]
    return rows.reshape(-1, rows.shape[-1])


def _save_booster(booster: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        booster.save_model(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class ReferencedNYISODARTTrainingRunner:
    """Executable nine-family annual training implementation."""

    output_root: Path
    source: NYISOPanelSource
    config: DARTTrainingConfig = field(default_factory=DARTTrainingConfig)
    _data_cache: dict[str, AnnualTrainingData] = field(default_factory=dict, init=False, repr=False)

    def _unsupported(
        self, request: OOSForecastRequest, code: str, detail: str
    ) -> UnsupportedTrainingCellError:
        path = Path(request.checkpoint_directory) / "unsupported_cell.json"
        atomic_json(
            path,
            {
                "schema": SCHEMA_VERSION,
                "status": "UNSUPPORTED_CELL",
                "code": code,
                "detail": detail,
                "unit_id": request.unit_id,
                "request_fingerprint": request.request_fingerprint,
                "candidate": request.candidate_name,
                "target": request.target_name,
                "seed_slot": request.seed_slot_name,
                "synthetic_fallback_used": False,
                "recorded_at_utc": utc_now(),
            },
        )
        return UnsupportedTrainingCellError(f"{code}: {detail}")

    def _validate_request(self, request: OOSForecastRequest) -> None:
        if request.candidate_name not in SUPPORTED_CANDIDATES:
            raise self._unsupported(
                request, "UNSUPPORTED_CANDIDATE", request.candidate_name
            )
        if request.target_name not in {"da_lbmp", "rt_lbmp_at_da_gate"}:
            raise self._unsupported(request, "UNSUPPORTED_TARGET", request.target_name)
        expected_family = (
            "deterministic"
            if request.candidate_name in DETERMINISTIC_CANDIDATES
            else "statistical"
            if request.candidate_name == "lightgbm_quantile"
            else "neural"
        )
        if request.candidate_family != expected_family:
            raise self._unsupported(
                request,
                "CANDIDATE_FAMILY_MISMATCH",
                f"expected {expected_family}, received {request.candidate_family}",
            )
        if expected_family == "neural" and request.seed not in STUDY_SEEDS:
            raise self._unsupported(
                request,
                "NEURAL_SEED_NOT_PREREGISTERED",
                str(request.seed),
            )
        if request.issue_clock != "day_ahead_market_gate":
            raise self._unsupported(request, "ISSUE_CLOCK_MISMATCH", request.issue_clock)
        if request.information_policy != "common_pre_da_gate_only":
            raise self._unsupported(
                request, "INFORMATION_POLICY_MISMATCH", request.information_policy
            )

    def _data(self, request: OOSForecastRequest) -> AnnualTrainingData:
        cache_key = sha256_json(
            {
                "origin": request.origin_name,
                "training_start": request.training_start_utc,
                "cutoff": request.training_cutoff_exclusive_utc,
                "forecast_start": request.forecast_start_utc,
                "forecast_end": request.forecast_end_exclusive_utc,
                "config": self.config.fingerprint,
            }
        )
        if cache_key not in self._data_cache:
            try:
                load_through_year = getattr(self.source, "load_through_year", None)
                bundle = (
                    load_through_year(request.forecast_year)
                    if callable(load_through_year)
                    else self.source.load()
                )
                self._data_cache[cache_key] = build_annual_training_data(
                    bundle, request, self.config
                )
            except UnsupportedTrainingCellError as exc:
                raise self._unsupported(
                    request, "INSUFFICIENT_CAUSAL_DATA", str(exc)
                ) from exc
        data = self._data_cache[cache_key]
        if (
            request.expected_information_set_fingerprint is not None
            and request.expected_information_set_fingerprint
            != data.information_set_fingerprint
        ):
            raise DARTTrainingAdapterError(
                "common information-set fingerprint differs from the frozen origin"
            )
        return data

    def _cached_result(
        self, request: OOSForecastRequest, data: AnnualTrainingData
    ) -> OOSForecastResult | None:
        root = Path(request.checkpoint_directory)
        manifest_path = root / "runner_result.json"
        artifact_path = root / "runner_result.npz"
        if not manifest_path.is_file() and not artifact_path.is_file():
            return None
        if not manifest_path.is_file() or not artifact_path.is_file():
            raise DARTTrainingAdapterError("runner result cache is only partially present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "request_fingerprint": request.request_fingerprint,
            "training_data_fingerprint": data.training_data_fingerprint,
            "information_set_fingerprint": data.information_set_fingerprint,
            "artifact_sha256": sha256_file(artifact_path),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise DARTTrainingAdapterError(f"runner cache mismatch: {key}")
        with np.load(artifact_path, allow_pickle=False) as cached:
            quantiles = np.asarray(cached["quantile_values"], dtype="float64")
            event_keys = tuple(map(str, cached["event_keys"].tolist()))
        if event_keys != data.forecast_event_keys:
            raise DARTTrainingAdapterError("runner cache event keys changed")
        checkpoint = manifest.get("model_checkpoint", {})
        checkpoint_path = checkpoint.get("path")
        checkpoint_sha = checkpoint.get("sha256")
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.is_file() or sha256_file(path) != checkpoint_sha:
                raise DARTTrainingAdapterError("cached model checkpoint hash mismatch")
        return OOSForecastResult(
            event_keys=event_keys,
            quantile_values=quantiles,
            target_times_utc=data.forecast_target_times_utc,
            da_gate_times_utc=data.forecast_gate_times_utc,
            max_feature_availability_times_utc=(
                data.forecast_max_feature_availability_utc
            ),
            training_target_max_utc=data.training_target_max_utc,
            information_set_fingerprint=data.information_set_fingerprint,
            realized_targets_used_as_features=False,
            future_information_used=False,
            model_checkpoint_path=checkpoint_path,
            model_checkpoint_sha256=checkpoint_sha,
            metadata=dict(manifest["runner_metadata"]),
        )

    def _deterministic_raw(
        self,
        request: OOSForecastRequest,
        data: AnnualTrainingData,
    ) -> tuple[np.ndarray, np.ndarray, Path, dict[str, Any]]:
        target = request.target_name
        x_train = data.x_core_by_target[target]
        y_train = data.y_core_by_target[target]
        train_point = _point_baseline(
            x_train,
            candidate=request.candidate_name,
            horizon=self.config.horizon_steps,
            issue_gap=self.config.issue_gap_steps,
        )
        validation_point = _point_baseline(
            data.x_validation_by_target[target],
            candidate=request.candidate_name,
            horizon=self.config.horizon_steps,
            issue_gap=self.config.issue_gap_steps,
        )
        forecast_point = _point_baseline(
            data.x_forecast_by_target[target],
            candidate=request.candidate_name,
            horizon=self.config.horizon_steps,
            issue_gap=self.config.issue_gap_steps,
        )
        scale = data.scale_by_target[target][None, :, None]
        center = data.center_by_target[target][None, :, None]
        train_point_original = train_point * scale + center
        validation_point_original = validation_point * scale + center
        forecast_point_original = forecast_point * scale + center
        errors = (y_train - train_point_original).reshape(-1)
        offsets = np.quantile(errors, QUANTILES, method="linear")
        validation = validation_point_original[..., None] + offsets
        forecast = forecast_point_original[..., None] + offsets
        checkpoint = Path(request.checkpoint_directory) / "deterministic_fit.json"
        payload = {
            "schema": SCHEMA_VERSION,
            "status": "COMPLETE",
            "candidate": request.candidate_name,
            "target": target,
            "fit_randomness": "none",
            "actual_fit_count": 1,
            "quantile_offsets": offsets.tolist(),
            "training_data_fingerprint": data.training_data_fingerprint,
            "request_fingerprint": request.request_fingerprint,
            "study_seed_slots_reusing_this_fit": list(STUDY_SEEDS),
            "written_at_utc": utc_now(),
        }
        atomic_json(checkpoint, payload)
        return validation, forecast, checkpoint, payload

    def _lightgbm_raw(
        self,
        request: OOSForecastRequest,
        data: AnnualTrainingData,
    ) -> tuple[np.ndarray, np.ndarray, Path, dict[str, Any]]:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise self._unsupported(
                request,
                "LIGHTGBM_DEPENDENCY_MISSING",
                "install the paper optional dependency; no fallback is permitted",
            ) from exc
        target = request.target_name
        x_train = _design_lightgbm(
            data.x_core_by_target[target], self.config.horizon_steps
        )
        y_train = data.y_core_by_target[target].reshape(-1)
        x_validation = _design_lightgbm(
            data.x_validation_by_target[target], self.config.horizon_steps
        )
        x_forecast = _design_lightgbm(
            data.x_forecast_by_target[target], self.config.horizon_steps
        )
        root = Path(request.checkpoint_directory) / "lightgbm"
        manifest_path = root / "fit_manifest.json"
        fit_fingerprint = sha256_json(
            {
                "request": request.request_fingerprint,
                "training_data": data.training_data_fingerprint,
                "estimators": self.config.lightgbm_estimators,
                "seed": STUDY_SEEDS[0],
                "quantiles": list(QUANTILES),
            }
        )
        model_paths = [root / f"q{int(level * 100):02d}.txt" for level in QUANTILES]
        boosters: list[Any] = []
        resumed = False
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fit_fingerprint") != fit_fingerprint:
                raise DARTTrainingAdapterError("LightGBM fit fingerprint changed")
            for path, record in zip(model_paths, manifest["models"], strict=True):
                if not path.is_file() or sha256_file(path) != record["sha256"]:
                    raise DARTTrainingAdapterError("LightGBM checkpoint hash mismatch")
                boosters.append(lgb.Booster(model_file=str(path)))
            resumed = True
        else:
            root.mkdir(parents=True, exist_ok=True)
            for level, path in zip(QUANTILES, model_paths, strict=True):
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=float(level),
                    n_estimators=self.config.lightgbm_estimators,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=40,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    random_state=STUDY_SEEDS[0],
                    n_jobs=self.config.lightgbm_jobs,
                    deterministic=True,
                    force_col_wise=True,
                    verbosity=-1,
                )
                model.fit(x_train, y_train)
                _save_booster(model.booster_, path)
                boosters.append(model.booster_)
            manifest = {
                "schema": SCHEMA_VERSION,
                "status": "COMPLETE",
                "candidate": "lightgbm_quantile",
                "target": target,
                "fit_fingerprint": fit_fingerprint,
                "actual_fit_count": 1,
                "actual_fit_seed": STUDY_SEEDS[0],
                "study_seed_slots_reusing_this_fit": list(STUDY_SEEDS),
                "models": [
                    {
                        "quantile": level,
                        "path": path.as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for level, path in zip(QUANTILES, model_paths, strict=True)
                ],
                "written_at_utc": utc_now(),
            }
            atomic_json(manifest_path, manifest)
        validation = np.stack(
            [booster.predict(x_validation) for booster in boosters], axis=-1
        ).reshape(
            len(data.validation_days),
            len(ZONES),
            self.config.horizon_steps,
            len(QUANTILES),
        )
        forecast = np.stack(
            [booster.predict(x_forecast) for booster in boosters], axis=-1
        ).reshape(
            len(data.forecast_days),
            len(ZONES),
            self.config.horizon_steps,
            len(QUANTILES),
        )
        scale = data.scale_by_target[target][None, :, None, None]
        center = data.center_by_target[target][None, :, None, None]
        record = dict(manifest)
        record["resumed_from_checkpoint"] = resumed
        return validation * scale + center, forecast * scale + center, manifest_path, record

    def _neural_device(self) -> Any:
        import torch

        return torch.device(
            "cuda"
            if self.config.device_name == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.config.device_name == "auto"
            else self.config.device_name
        )

    def _neural_group_kwargs(
        self,
        request: OOSForecastRequest,
        data: AnnualTrainingData,
        *,
        device: Any,
    ) -> dict[str, Any]:
        target = request.target_name
        group_root = (
            Path(self.output_root)
            / "checkpoint_groups"
            / _safe_slug(request.origin_name)
            / _safe_slug(target)
            / _safe_slug(request.candidate_name)
        )
        group_resume_fingerprint = sha256_json(
            {
                "plan": request.plan_fingerprint,
                "origin": request.origin_name,
                "fit_through_year": request.fit_through_year,
                "forecast_year": request.forecast_year,
                "training_cutoff_exclusive_utc": (
                    request.training_cutoff_exclusive_utc
                ),
                "candidate": request.candidate_name,
                "target": target,
                "data": data.training_data_fingerprint,
                "config": self.config.fingerprint,
                "adapter_code": sha256_file(Path(__file__)),
            }
        )
        return {
            "task": target,
            "model_name": request.candidate_name,
            "requested_seed": request.seed,
            "study_seeds": STUDY_SEEDS,
            "x_train": data.x_core_by_target[target],
            "y_train": (
                (
                    data.y_core_by_target[target]
                    - data.center_by_target[target][None, :, None]
                )
                / data.scale_by_target[target][None, :, None]
            ).astype("float32"),
            "x_checkpoint_calibration": (
                data.x_checkpoint_calibration_by_target[target]
            ),
            "y_checkpoint_calibration": (
                data.y_checkpoint_calibration_by_target[target]
            ),
            "checkpoint_calibration_dates": tuple(
                _utc_text(day) for day in data.checkpoint_calibration_days
            ),
            "x_checkpoint_selection": (
                data.x_checkpoint_selection_by_target[target]
            ),
            "y_checkpoint_selection": (
                data.y_checkpoint_selection_by_target[target]
            ),
            "checkpoint_selection_dates": tuple(
                _utc_text(day) for day in data.checkpoint_selection_days
            ),
            "center": data.center_by_target[target],
            "scale": data.scale_by_target[target],
            "tail_threshold_contract": data.tail_thresholds_by_target[target],
            "history": self.config.history_steps,
            "horizon": self.config.horizon_steps,
            "max_epochs": self.config.max_epochs,
            "batch_sizes": self.config.batch_sizes,
            "gradient_accumulation_steps": (
                self.config.gradient_accumulation_steps
            ),
            "learning_rate": self.config.learning_rate,
            "device": device,
            "output_root": group_root,
            "resume_fingerprint": group_resume_fingerprint,
            "selection_config": self.config.checkpoint_selection,
            "anchor_interval_epochs": (
                self.config.checkpoint_anchor_interval_epochs
            ),
            "minimum_epochs": self.config.checkpoint_minimum_epochs,
            "evaluation_interval_epochs": (
                self.config.checkpoint_evaluation_interval_epochs
            ),
            "stop_commit_interval_epochs": (
                self.config.checkpoint_global_round_epochs
            ),
            "stability_evaluations": (
                self.config.checkpoint_stability_evaluations
            ),
            "early_stopping_min_delta": (
                self.config.checkpoint_early_stopping_min_delta
            ),
        }

    def _require_hybrid_pretraining(self, group_root: Path) -> None:
        if not self.config.paper_evidence:
            return
        path = (
            Path(self.output_root)
            / "checkpoint_scheduler"
            / "hybrid_scheduler_manifest.json"
        )
        if not path.is_file():
            raise DARTTrainingAdapterError(
                "formal neural forecast materialization requires the completed "
                "seed-major cross-model pretraining scheduler"
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("efficient_hybrid_group_scheduler") is not True
            or manifest.get("manifest_sha256") != sha256_json(core)
        ):
            raise DARTTrainingAdapterError(
                "hybrid neural pretraining manifest is incomplete or changed"
            )
        matching = [
            value
            for value in manifest.get("completed_groups", {}).values()
            if Path(str(value.get("output_root", ""))) == Path(group_root)
        ]
        if len(matching) != 1:
            raise DARTTrainingAdapterError(
                "neural group is absent from the hybrid pretraining manifest"
            )
        checkpoint_manifest = Path(
            str(matching[0]["checkpoint_group_manifest_path"])
        )
        if (
            not checkpoint_manifest.is_file()
            or sha256_file(checkpoint_manifest)
            != matching[0]["checkpoint_group_manifest_sha256"]
        ):
            raise DARTTrainingAdapterError(
                "hybrid pretraining group checkpoint hash changed"
            )

    def pretrain_neural_groups_hybrid(
        self,
        units: Sequence[Any],
        *,
        plan_fingerprint: str,
        quantiles: Sequence[float],
    ) -> dict[str, Any]:
        """Build every unique neural group, then schedule seed-major blocks."""

        from dcvpp.experiments.dart_checkpoint_training import (
            pretrain_dart_energy_groups_hybrid,
        )

        unique: list[Any] = []
        seen: set[tuple[str, str, str]] = set()
        for unit in units:
            if unit.candidate.family != "neural":
                continue
            identity = (
                unit.origin.name,
                unit.target.name,
                unit.candidate.name,
            )
            if identity not in seen:
                seen.add(identity)
                unique.append(unit)
        if not unique:
            raise DARTTrainingAdapterError(
                "formal selection plan has no neural checkpoint groups"
            )
        if self.config.paper_evidence:
            identities = {
                (unit.origin.name, unit.target.name, unit.candidate.name)
                for unit in unique
            }
            if len(identities) != 36 or {
                unit.candidate.name for unit in unique
            } != set(NEURAL_CANDIDATES):
                raise DARTTrainingAdapterError(
                    "paper hybrid scheduler requires all 36 origin/target/neural groups"
                )
        device = self._neural_device()
        jobs: list[dict[str, Any]] = []
        for unit in unique:
            request_fingerprint = sha256_json(
                {
                    "role": "efficient_hybrid_neural_pretraining",
                    "plan_fingerprint": plan_fingerprint,
                    "origin": unit.origin.name,
                    "target": unit.target.name,
                    "candidate": unit.candidate.name,
                }
            )
            request = OOSForecastRequest(
                unit_id=f"pretrain__{unit.unit_id}",
                request_fingerprint=request_fingerprint,
                plan_fingerprint=plan_fingerprint,
                origin_name=unit.origin.name,
                fit_through_year=unit.origin.fit_through_year,
                forecast_year=unit.origin.forecast_year,
                target_name=unit.target.name,
                target_field=unit.target.target_field,
                candidate_name=unit.candidate.name,
                candidate_family=unit.candidate.family,
                seed_slot_name=unit.seed_slot.name,
                seed=STUDY_SEEDS[0],
                quantiles=tuple(map(float, quantiles)),
                training_start_utc=unit.origin.training_start_utc,
                training_cutoff_exclusive_utc=(
                    unit.origin.training_cutoff_exclusive_utc
                ),
                forecast_start_utc=unit.origin.forecast_start_utc,
                forecast_end_exclusive_utc=unit.origin.forecast_end_exclusive_utc,
                checkpoint_directory=(
                    Path(self.output_root)
                    / "checkpoint_scheduler"
                    / "request_views"
                    / _safe_slug(unit.origin.name)
                    / _safe_slug(unit.target.name)
                    / _safe_slug(unit.candidate.name)
                ),
                resume_checkpoint=None,
                expected_information_set_fingerprint=None,
            )
            self._validate_request(request)
            data = self._data(request)
            jobs.append(self._neural_group_kwargs(request, data, device=device))
        manifest = pretrain_dart_energy_groups_hybrid(
            jobs,
            scheduler_root=Path(self.output_root) / "checkpoint_scheduler",
        )
        return {
            "path": (
                Path(self.output_root)
                / "checkpoint_scheduler"
                / "hybrid_scheduler_manifest.json"
            ).as_posix(),
            "sha256": sha256_file(
                Path(self.output_root)
                / "checkpoint_scheduler"
                / "hybrid_scheduler_manifest.json"
            ),
            "scheduler_fingerprint": manifest["scheduler_fingerprint"],
            "actual_training_order": manifest["actual_training_order"],
            "efficient_hybrid_group_scheduler": manifest[
                "efficient_hybrid_group_scheduler"
            ],
            "stop_committed_only_at_global_round_boundary": manifest[
                "stop_committed_only_at_global_round_boundary"
            ],
            "group_count": manifest["group_count"],
        }

    def _neural_raw(
        self,
        request: OOSForecastRequest,
        data: AnnualTrainingData,
    ) -> tuple[np.ndarray, np.ndarray, Path, dict[str, Any]]:
        try:
            import torch
            from dcvpp.experiments.dart_checkpoint_training import (
                fit_dart_energy_seed_group,
            )
            from dcvpp.experiments.paper_training import _energy_predict
        except ImportError as exc:
            raise self._unsupported(
                request,
                "NEURAL_DEPENDENCY_MISSING",
                "the frozen paper neural implementation is unavailable",
            ) from exc
        target = request.target_name
        device = self._neural_device()
        group_kwargs = self._neural_group_kwargs(request, data, device=device)
        self._require_hybrid_pretraining(Path(group_kwargs["output_root"]))
        fitted = fit_dart_energy_seed_group(**group_kwargs)
        validation_normalized = _energy_predict(
            fitted["model"],
            data.x_validation_by_target[target],
            device=device,
            batch_size=fitted["batch_size"],
        )
        forecast_normalized = _energy_predict(
            fitted["model"],
            data.x_forecast_by_target[target],
            device=device,
            batch_size=fitted["batch_size"],
        )
        scale = data.scale_by_target[target][None, :, None, None]
        center = data.center_by_target[target][None, :, None, None]
        record = {
            "candidate": request.candidate_name,
            "target": target,
            "seed": request.seed,
            "seed_slot": request.seed_slot_name,
            "best_epoch": fitted["best_epoch"],
            "selected_common_epoch": fitted["selected_epoch"],
            "completed_common_epoch": fitted["completed_common_epoch"],
            "best_validation_metric": fitted["best_metric"],
            "epochs_written": fitted["epochs_written"],
            "resumed_from_epoch": fitted["resumed_from_epoch"],
            "batch_size": fitted["batch_size"],
            "resume_fingerprint": fitted["resume_fingerprint"],
            "architecture": fitted["architecture"],
            "training_health": fitted["training_health"],
            "checkpoint_group_manifest": {
                "path": fitted["checkpoint_group_manifest_path"].as_posix(),
                "sha256": fitted["checkpoint_group_manifest_sha256"],
            },
            "checkpoint_selection": fitted["checkpoint_selection"],
            "seed_aggregation": fitted["seed_aggregation"],
            "selected_epoch_record": fitted["selected_epoch_record"],
            "synchronized_early_stopping_decision": (
                fitted["synchronized_early_stopping_decision"]
            ),
            "selection_input_role": "validation_only_no_action_or_market_replay",
            "exact_study_seeds": list(STUDY_SEEDS),
            "best_seed_selected": False,
            "individual_seed_early_stopping": False,
            "checkpoint_calibration_day_count": len(
                data.checkpoint_calibration_days
            ),
            "checkpoint_selection_day_count": len(data.checkpoint_selection_days),
            "final_deploy_calibration_day_count": len(data.validation_days),
            "checkpoint_calibration_dates_sha256": sha256_json(
                [_utc_text(day) for day in data.checkpoint_calibration_days]
            ),
            "checkpoint_selection_dates_sha256": sha256_json(
                [_utc_text(day) for day in data.checkpoint_selection_days]
            ),
            "final_deploy_calibration_dates_sha256": sha256_json(
                [_utc_text(day) for day in data.validation_days]
            ),
            "tail_threshold_contract_sha256": (
                data.tail_threshold_contract_sha256_by_target[target]
            ),
            "device": str(device),
        }
        return (
            validation_normalized * scale + center,
            forecast_normalized * scale + center,
            Path(fitted["checkpoint_path"]),
            record,
        )

    def _persist_residuals(
        self,
        request: OOSForecastRequest,
        data: AnnualTrainingData,
        validation_prediction: np.ndarray,
    ) -> tuple[Path, str]:
        target = request.target_name
        actual = data.y_validation_by_target[target]
        median = validation_prediction[..., QUANTILES.index(0.50)]
        residual = (actual - median).reshape(-1).astype("float64")
        path = (
            Path(self.output_root)
            / "validation_residuals"
            / request.origin_name
            / request.candidate_name
            / request.seed_slot_name
            / f"{target}.npz"
        )
        metadata = {
            "schema": SCHEMA_VERSION,
            "fit_role": "validation_residuals_only",
            "origin_name": request.origin_name,
            "candidate_name": request.candidate_name,
            "seed_slot_name": request.seed_slot_name,
            "target_name": target,
            "fit_cutoff_exclusive_utc": request.training_cutoff_exclusive_utc,
            "validation_residual_target_max_utc": data.training_target_max_utc,
            "event_keys_sha256": sha256_json(data.validation_event_keys),
            "residuals_sha256": _array_sha256(residual),
            "locked_targets_used": False,
            "locked_target_years_used": [],
            "created_at_utc": utc_now(),
        }
        _atomic_npz(
            path,
            event_keys=np.asarray(data.validation_event_keys),
            target_times_utc=np.asarray(data.validation_target_times_utc),
            median_residual=residual,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return path, sha256_file(path)

    def run(self, request: OOSForecastRequest) -> OOSForecastResult:
        self._validate_request(request)
        data = self._data(request)
        cached = self._cached_result(request, data)
        if cached is not None:
            return cached

        if request.candidate_name in DETERMINISTIC_CANDIDATES:
            raw_validation, raw_forecast, checkpoint, fit_record = self._deterministic_raw(
                request, data
            )
        elif request.candidate_name == "lightgbm_quantile":
            raw_validation, raw_forecast, checkpoint, fit_record = self._lightgbm_raw(
                request, data
            )
        else:
            raw_validation, raw_forecast, checkpoint, fit_record = self._neural_raw(
                request, data
            )
        from dcvpp.experiments.forecast_calibration import (
            apply_additive_conformal_quantiles,
            fit_additive_conformal_quantiles,
        )

        calibrator = fit_additive_conformal_quantiles(
            data.y_validation_by_target[request.target_name],
            raw_validation,
            quantiles=QUANTILES,
            minimum_samples=self.config.minimum_calibration_samples,
        )
        validation = apply_additive_conformal_quantiles(raw_validation, calibrator)
        forecast = apply_additive_conformal_quantiles(raw_forecast, calibrator)
        residual_path, residual_sha = self._persist_residuals(
            request, data, validation
        )
        quantile_values = np.asarray(forecast, dtype="float64").reshape(
            -1, len(QUANTILES)
        )
        if quantile_values.shape[0] != len(data.forecast_event_keys):
            raise DARTTrainingAdapterError("flattened forecast does not align to event keys")
        if not np.isfinite(quantile_values).all() or (
            np.diff(quantile_values, axis=1) < 0
        ).any():
            raise DARTTrainingAdapterError("calibrated forecast is non-finite or crossing")

        checkpoint_sha = sha256_file(checkpoint)
        calibration_record = {
            key: value for key, value in calibrator.items() if key != "corrections"
        }
        corrections = np.asarray(calibrator["corrections"], dtype="float64")
        calibration_record["corrections"] = corrections.tolist()
        calibration_record["corrections_sha256"] = _array_sha256(corrections)
        metadata = {
            "schema": SCHEMA_VERSION,
            "status": "COMPLETE",
            "paper_evidence": self.config.paper_evidence,
            "candidate_name": request.candidate_name,
            "candidate_family": request.candidate_family,
            "target_name": request.target_name,
            "separate_target_head": True,
            "common_pre_da_gate_information": True,
            "feature_contract_id": self.config.feature_contract_id,
            "target_specific_feature_subset": "own_price_history_only",
            "realized_targets_used_to_generate_forecast": False,
            "future_information_used": False,
            "daily_retraining": False,
            "fit_policy": "expanding_annual_refit",
            "fit_record": fit_record,
            "calibration": calibration_record,
            "deploy_calibration_panel_role": (
                "final_calibration_and_residual_coupling_only"
            ),
            "deploy_calibration_panel_date_sha256": sha256_json(
                [_utc_text(day) for day in data.validation_days]
            ),
            "validation_residual_artifact": {
                "path": residual_path.as_posix(),
                "sha256": residual_sha,
                "fit_role": "validation_residuals_only",
            },
            "information_set_fingerprint": data.information_set_fingerprint,
            "training_data_fingerprint": data.training_data_fingerprint,
            "source_manifest": data.source_manifest,
            "data_audit": data.audit,
            "non_neural_seed_replication": (
                {
                    "actual_fit_count": 1,
                    "replicated_study_seed_slots": list(STUDY_SEEDS),
                    "predictions_identical_across_replicated_slots": True,
                }
                if request.candidate_family != "neural"
                else None
            ),
        }
        artifact_path = Path(request.checkpoint_directory) / "runner_result.npz"
        _atomic_npz(
            artifact_path,
            event_keys=np.asarray(data.forecast_event_keys),
            quantile_values=quantile_values,
        )
        manifest = {
            "schema": SCHEMA_VERSION,
            "status": "COMPLETE",
            "request_fingerprint": request.request_fingerprint,
            "training_data_fingerprint": data.training_data_fingerprint,
            "information_set_fingerprint": data.information_set_fingerprint,
            "artifact_path": artifact_path.as_posix(),
            "artifact_sha256": sha256_file(artifact_path),
            "model_checkpoint": {
                "path": checkpoint.as_posix(),
                "sha256": checkpoint_sha,
            },
            "runner_metadata": metadata,
            "written_at_utc": utc_now(),
        }
        atomic_json(Path(request.checkpoint_directory) / "runner_result.json", manifest)
        return OOSForecastResult(
            event_keys=data.forecast_event_keys,
            quantile_values=quantile_values,
            target_times_utc=data.forecast_target_times_utc,
            da_gate_times_utc=data.forecast_gate_times_utc,
            max_feature_availability_times_utc=(
                data.forecast_max_feature_availability_utc
            ),
            training_target_max_utc=data.training_target_max_utc,
            information_set_fingerprint=data.information_set_fingerprint,
            realized_targets_used_as_features=False,
            future_information_used=False,
            model_checkpoint_path=checkpoint.as_posix(),
            model_checkpoint_sha256=checkpoint_sha,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ValidationResidualGaussianCopulaProvider:
    """Fit a nonsingular Gaussian copula from paired validation residuals."""

    residual_root: Path
    minimum_residuals: int = 500
    maximum_absolute_correlation: float = 0.995

    def _path(self, request: JointDrawRequest, target: str) -> Path:
        return (
            Path(self.residual_root)
            / "validation_residuals"
            / request.origin_name
            / request.candidate_name
            / request.seed_slot_name
            / f"{target}.npz"
        )

    @staticmethod
    def _load(path: Path) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
        if not path.is_file():
            raise ResidualCouplingError(f"validation residual artifact is absent: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"event_keys", "median_residual", "metadata_json"}
            if not required.issubset(data.files):
                raise ResidualCouplingError(f"residual artifact schema is incomplete: {path}")
            keys = tuple(map(str, data["event_keys"].tolist()))
            residual = np.asarray(data["median_residual"], dtype="float64")
            metadata = json.loads(str(data["metadata_json"].item()))
        if residual.shape != (len(keys),) or not np.isfinite(residual).all():
            raise ResidualCouplingError("validation residuals are unaligned or non-finite")
        if metadata.get("residuals_sha256") != _array_sha256(residual):
            raise ResidualCouplingError("validation residual content hash mismatch")
        return keys, residual, metadata

    def draw_joint_uniforms(self, request: JointDrawRequest) -> JointDrawResult:
        base = np.asarray(request.common_base_uniforms, dtype="float64")
        if base.ndim != 3:
            raise ResidualCouplingError(
                "common base uniforms must have shape (event, draw, 2)"
            )
        expected_shape = (len(request.event_keys), base.shape[1], 2)
        if base.shape != expected_shape:
            raise ResidualCouplingError(
                "common base uniforms must have shape (event, draw, 2)"
            )
        if not re.fullmatch(
            r"[0-9a-fA-F]{64}", request.common_base_random_numbers_sha256
        ):
            raise ResidualCouplingError(
                "common base-random-number artifact hash is invalid"
            )
        if not np.isfinite(base).all() or (base <= 0.0).any() or (base >= 1.0).any():
            raise ResidualCouplingError("common base uniforms must be finite inside (0,1)")
        da_path = self._path(request, "da_lbmp")
        rt_path = self._path(request, "rt_lbmp_at_da_gate")
        da_keys, da_residual, da_metadata = self._load(da_path)
        rt_keys, rt_residual, rt_metadata = self._load(rt_path)
        if da_keys != rt_keys:
            raise ResidualCouplingError("DA and RT validation residual event keys differ")
        if len(da_keys) < self.minimum_residuals:
            raise ResidualCouplingError(
                f"only {len(da_keys)} paired residuals; {self.minimum_residuals} required"
            )
        cutoff = _utc(request.fit_cutoff_exclusive_utc, name="fit_cutoff_exclusive_utc")
        residual_max_values = {
            _utc(
                metadata["validation_residual_target_max_utc"],
                name="validation_residual_target_max_utc",
            )
            for metadata in (da_metadata, rt_metadata)
        }
        if len(residual_max_values) != 1:
            raise ResidualCouplingError("DA and RT residual cutoffs differ")
        residual_max = residual_max_values.pop()
        if residual_max >= cutoff:
            raise ResidualCouplingError("coupling residuals cross the annual fit cutoff")
        for metadata in (da_metadata, rt_metadata):
            identity = (
                metadata.get("origin_name"),
                metadata.get("candidate_name"),
                metadata.get("seed_slot_name"),
                metadata.get("fit_cutoff_exclusive_utc"),
            )
            expected_identity = (
                request.origin_name,
                request.candidate_name,
                request.seed_slot_name,
                request.fit_cutoff_exclusive_utc,
            )
            if identity != expected_identity:
                raise ResidualCouplingError(
                    "validation residual identity differs from the joint-draw request"
                )
            if metadata.get("locked_targets_used") is not False:
                raise ResidualCouplingError("coupling artifact consumed locked targets")
            if metadata.get("locked_target_years_used") != []:
                raise ResidualCouplingError("coupling artifact lists locked target years")
            if metadata.get("fit_role") != "validation_residuals_only":
                raise ResidualCouplingError("coupling source is not validation-only")

        count = len(da_residual)
        da_rank = stats.rankdata(da_residual, method="average")
        rt_rank = stats.rankdata(rt_residual, method="average")
        da_normal = stats.norm.ppf((da_rank - 0.5) / count)
        rt_normal = stats.norm.ppf((rt_rank - 0.5) / count)
        raw_correlation = float(np.corrcoef(da_normal, rt_normal)[0, 1])
        if not math.isfinite(raw_correlation):
            raise ResidualCouplingError("validation residual copula correlation is non-finite")
        correlation = float(
            np.clip(
                raw_correlation,
                -self.maximum_absolute_correlation,
                self.maximum_absolute_correlation,
            )
        )
        epsilon = np.finfo("float64").eps
        first = stats.norm.ppf(np.clip(base[:, :, 0], epsilon, 1.0 - epsilon))
        independent = stats.norm.ppf(
            np.clip(base[:, :, 1], epsilon, 1.0 - epsilon)
        )
        second = correlation * first + math.sqrt(1.0 - correlation**2) * independent
        da_uniforms = stats.norm.cdf(first)
        rt_uniforms = stats.norm.cdf(second)
        if np.array_equal(da_uniforms, rt_uniforms):
            raise ResidualCouplingError(
                "joint provider produced identical uniforms without a verified perfect copula"
            )
        source_hashes = {
            "da_lbmp": sha256_file(da_path),
            "rt_lbmp_at_da_gate": sha256_file(rt_path),
        }
        manifest = {
            "provider_name": "validation_residual_gaussian_copula",
            "provider_version": 1,
            "method": "gaussian_copula_rank_transform",
            "fit_role": "validation_residuals_only",
            "fixture": False,
            "origin_name": request.origin_name,
            "candidate_name": request.candidate_name,
            "seed_slot_name": request.seed_slot_name,
            "fit_cutoff_exclusive_utc": request.fit_cutoff_exclusive_utc,
            "validation_residual_target_max_utc": _utc_text(residual_max),
            "paired_validation_residual_count": count,
            "raw_gaussian_rank_correlation": raw_correlation,
            "fitted_gaussian_rank_correlation": correlation,
            "maximum_absolute_correlation": self.maximum_absolute_correlation,
            "perfect_comonotonicity_fitted_and_verified": False,
            "source_residual_artifacts": source_hashes,
            "source_residuals_sha256": sha256_json(source_hashes),
            "common_base_random_numbers_sha256": (
                request.common_base_random_numbers_sha256
            ),
            "common_base_uniform_values_sha256": _array_sha256(base),
            "locked_targets_used": False,
            "locked_target_years_used": [],
        }
        manifest_sha = sha256_json(manifest)
        return JointDrawResult(
            da_uniforms=np.asarray(da_uniforms, dtype="float64"),
            rt_uniforms=np.asarray(rt_uniforms, dtype="float64"),
            origin_name=request.origin_name,
            candidate_name=request.candidate_name,
            seed_slot_name=request.seed_slot_name,
            fit_cutoff_exclusive_utc=request.fit_cutoff_exclusive_utc,
            validation_residual_target_max_utc=_utc_text(residual_max),
            common_base_random_numbers_sha256=(
                request.common_base_random_numbers_sha256
            ),
            coupling_manifest=manifest,
            coupling_manifest_sha256=manifest_sha,
            locked_targets_used=False,
        )


def build_referenced_dart_training_stack(
    repository_root: Path,
    output_root: Path,
    *,
    config: DARTTrainingConfig | None = None,
    data_root: Path | None = None,
) -> tuple[
    ReferencedNYISODARTTrainingRunner,
    ValidationResidualGaussianCopulaProvider,
]:
    """Construct the production runner and matching residual-coupling provider."""

    effective = config or DARTTrainingConfig()
    runner = ReferencedNYISODARTTrainingRunner(
        output_root=Path(output_root),
        source=ReferencedNYISOPanelSource(
            repository_root=Path(repository_root), data_root=data_root
        ),
        config=effective,
    )
    provider = ValidationResidualGaussianCopulaProvider(
        residual_root=Path(output_root),
        minimum_residuals=effective.minimum_coupling_residuals,
    )
    return runner, provider


def _validate_stage_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    candidates = tuple(
        config.get("candidate_contract", {}).get("frozen_deployable_pool", ())
    )
    seeds = tuple(config.get("randomization_contract", {}).get("study_seeds", ()))
    quantiles = tuple(
        float(value)
        for value in config.get("forecast_issue_contract", {}).get("quantiles", ())
    )
    chronology = config.get("chronology", {})
    errors: list[str] = []
    if candidates != SUPPORTED_CANDIDATES:
        errors.append("candidate pool differs from the fixed nine-family contract")
    if seeds != STUDY_SEEDS:
        errors.append("study seeds differ from the amended three-seed contract")
    if quantiles != QUANTILES:
        errors.append("quantiles differ from (0.05,0.10,0.50,0.90,0.95)")
    if chronology.get("selection_years") != [2022, 2023, 2024]:
        errors.append("selection years differ from 2022-2024")
    if chronology.get("locked_test_year") != 2025:
        errors.append("locked test year differs from 2025")
    if not bool(chronology.get("expanding_window")):
        errors.append("expanding-window fitting is not enabled")
    issue = config.get("forecast_issue_contract", {})
    if issue.get("profile_id") != "COMMON_DA_GATE_DART_V1":
        errors.append("common DA-gate profile is absent")
    if not bool(issue.get("both_targets_share_identical_information_cutoff")):
        errors.append("DA and RT do not share the identical information cutoff")
    if errors:
        raise DARTTrainingAdapterError("; ".join(errors))
    return config


def run_dart_forecast_training_stage(
    repository_root: Path,
    artifact_root: Path,
    config_path: Path,
    run_id: str,
    *,
    phase: str = "selection",
    resume: bool = True,
    input_fingerprints: Mapping[str, str] | None = None,
    dependency_fingerprints: Mapping[str, str] | None = None,
    data_root: Path | None = None,
    training_config: DARTTrainingConfig | None = None,
) -> dict[str, Any]:
    """Run one fail-closed annual DART forecast phase.

    This is the stable top-level callback for the paper orchestrator.  The
    accepted phases are ``selection`` (all 2022--2024 units), ``freeze``
    (freeze already completed selection evidence), and ``locked_test`` (the
    once-only 2025 refit/evaluation).  The callback never labels partial work
    complete: every exception is retained by unit and the returned stage
    manifest has ``status=PARTIAL`` and ``downstream_ready=false``.

    The runner callback consumed by :class:`DARTForecastOOSProducer` is exactly
    ``run(OOSForecastRequest) -> OOSForecastResult``.  Each result contains a
    unique event-key vector, an ``event x 5`` monotone quantile matrix, aligned
    target/gate/maximum-availability timestamps, the pre-cutoff training-target
    maximum, a common information-set SHA-256, and optional checkpoint
    path/hash.  The paired coupling callback is exactly
    ``draw_joint_uniforms(JointDrawRequest) -> JointDrawResult`` and returns
    ``event x draw`` DA/RT marginal uniforms fitted only from pre-cutoff paired
    validation residuals.
    """

    repository = Path(repository_root).resolve()
    artifacts = Path(artifact_root).resolve()
    config_file = Path(config_path).resolve()
    if not RUN_ID_PATTERN.fullmatch(str(run_id)):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    if phase not in {"selection", "freeze", "locked_test"}:
        raise ValueError("phase must be selection, freeze, or locked_test")
    stage_config = _validate_stage_config(config_file)
    anchor_amendment = (
        stage_config.get("pre_dart_v1_readiness", {}).get(
            "anchor_only_checkpoint_amendment", {}
        )
    )
    destination_run_id = str(anchor_amendment.get("destination_run_id", ""))
    v4_overlay_selection = phase == "selection" and run_id == destination_run_id
    if phase == "selection" and destination_run_id and not v4_overlay_selection:
        raise DARTTrainingAdapterError(
            "anchor-only v12 selection must use its configured destination run id"
        )
    stage_root = artifacts / "dart_value_select" / f"run_id={run_id}" / "forecast_oos"
    stage_root.mkdir(parents=True, exist_ok=True)
    effective_training = training_config or DARTTrainingConfig()
    source = ReferencedNYISOPanelSource(repository_root=repository, data_root=data_root)
    # The plan/freeze fingerprint is based only on data through 2024.  The
    # locked runner may later open 2025 sequential features under the once-only
    # ledger, but those bytes cannot alter the already frozen plan identity.
    source_manifest = source.load_through_year(2024).source_manifest
    default_fingerprints = {
        "dart_value_select_config": sha256_file(config_file),
        "referenced_nyiso_partitions": str(source_manifest["partitions_sha256"]),
        "dart_training_adapter_code": sha256_file(Path(__file__)),
        "frozen_paper_training_code": sha256_file(
            repository / "src/dcvpp/experiments/paper_training.py"
        ),
        "frozen_dgpt_e_model_code": sha256_file(
            repository / "src/dcvpp/models/energy/dgpt_e.py"
        ),
    }
    merged_fingerprints = {**default_fingerprints, **dict(input_fingerprints or {})}
    runner = ReferencedNYISODARTTrainingRunner(
        output_root=stage_root,
        source=source,
        config=effective_training,
    )
    provider = ValidationResidualGaussianCopulaProvider(
        residual_root=stage_root,
        minimum_residuals=effective_training.minimum_coupling_residuals,
    )
    producer = DARTForecastOOSProducer(
        stage_root,
        input_fingerprints=merged_fingerprints,
        joint_draw_provider=provider,
    )
    producer.initialize()
    started_at = utc_now()
    completed_units: list[str] = []
    failed_units: list[dict[str, Any]] = []
    phase_evidence: Mapping[str, Any] | None = None
    pretraining_scheduler_evidence: Mapping[str, Any] | None = None
    anchor_overlay_recovery: Mapping[str, Any] | None = None
    anchor_compatibility_evidence: Mapping[str, Any] | None = None
    full_epoch_archive_inventory: Mapping[str, Any] | None = None
    expected_unit_count = (
        len(producer.plan.work_units(role="selection_oos"))
        if phase == "selection"
        else len(producer.plan.work_units(role="locked_test"))
        if phase == "locked_test"
        else 1
    )
    if phase == "selection":
        selection_units = producer.plan.work_units(role="selection_oos")
        compatibility = nullcontext(None)
        if v4_overlay_selection:
            from dcvpp.experiments import dart_checkpoint_training as frozen
            from dcvpp.experiments.dart_checkpoint_replay_compat import (
                prepare_v4_overlay_recovery,
                selected_epoch_replay_compatibility,
            )

            snapshot_path = repository / str(
                anchor_amendment["source_snapshot_manifest_path"]
            )
            prepared = prepare_v4_overlay_recovery(
                frozen,
                runner=runner,
                units=selection_units,
                plan_fingerprint=producer.plan.fingerprint,
                quantiles=producer.plan.quantiles,
                snapshot_manifest_path=snapshot_path,
                v4_stage_root=stage_root,
                repository_root=repository,
                strict_paper_counts=True,
            )
            import_ledger = prepared["import_ledger"]
            scheduler_state_path = (
                stage_root / "checkpoint_scheduler/hybrid_scheduler_state.json"
            )
            anchor_overlay_recovery = {
                "schema_version": prepared["schema_version"],
                "status": prepared["status"],
                "snapshot_manifest_path": snapshot_path.relative_to(
                    repository
                ).as_posix(),
                "snapshot_manifest_sha256": import_ledger[
                    "v3_snapshot_manifest_sha256"
                ],
                "import_ledger_path": Path(
                    import_ledger["ledger_path"]
                ).relative_to(repository).as_posix(),
                "import_ledger_file_sha256": import_ledger["ledger_sha256"],
                "imported_highest_seed_state_count": import_ledger[
                    "imported_highest_seed_state_count"
                ],
                "scheduler_state_path": scheduler_state_path.relative_to(
                    repository
                ).as_posix(),
                "scheduler_state_file_sha256": sha256_file(scheduler_state_path),
                "training_launched_by_prepare": False,
                "v3_files_moved_modified_or_deleted": 0,
            }
            compatibility = selected_epoch_replay_compatibility(
                snapshot_manifest_path=snapshot_path,
                group_identity_by_root=prepared["group_identity_by_root"],
                ledger_path=stage_root
                / "recovery_overlay/anchor_only_compatibility_ledger.json",
                repository_root=repository,
                strict_paper_counts=True,
                v4_stage_root=stage_root,
                import_ledger=import_ledger,
            )
        with compatibility as compatibility_record:
            anchor_compatibility_evidence = compatibility_record
            try:
                pretraining_scheduler_evidence = (
                    runner.pretrain_neural_groups_hybrid(
                        selection_units,
                        plan_fingerprint=producer.plan.fingerprint,
                        quantiles=producer.plan.quantiles,
                    )
                )
            except Exception as exc:  # evidence ledger, not a silent fallback
                failed_units.append(
                    {
                        "unit_id": "neural_hybrid_pretraining_scheduler",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recorded_at_utc": utc_now(),
                    }
                )
            for unit in (() if failed_units else selection_units):
                try:
                    producer.execute_unit(unit, runner, resume=resume)
                    completed_units.append(unit.unit_id)
                except Exception as exc:  # evidence ledger, not a silent fallback
                    failed_units.append(
                        {
                            "unit_id": unit.unit_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "recorded_at_utc": utc_now(),
                        }
                    )
            if not failed_units:
                try:
                    phase_evidence = producer._collect_evidence(
                        role="selection_oos"
                    )
                except Exception as exc:
                    failed_units.append(
                        {
                            "unit_id": "selection_evidence_gate",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "recorded_at_utc": utc_now(),
                        }
                    )
        if not failed_units and v4_overlay_selection:
            from dcvpp.experiments.dart_checkpoint_replay_compat import (
                validate_v4_epoch_archives,
            )

            full_epoch_archive_inventory = validate_v4_epoch_archives(
                v4_stage_root=stage_root,
                expected_track_count=108,
                repository_root=repository,
            )
    elif phase == "freeze":
        if dependency_fingerprints is None:
            failed_units.append(
                {
                    "unit_id": "selection_freeze",
                    "error_type": "MissingDependencyFingerprints",
                    "error": "freeze requires all seven declared dependency fingerprints",
                    "recorded_at_utc": utc_now(),
                }
            )
        else:
            try:
                phase_evidence = producer.freeze_selection(dependency_fingerprints)
                completed_units.append("selection_freeze")
            except Exception as exc:
                failed_units.append(
                    {
                        "unit_id": "selection_freeze",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recorded_at_utc": utc_now(),
                    }
                )
    else:
        if dependency_fingerprints is None:
            failed_units.append(
                {
                    "unit_id": "locked_test_2025",
                    "error_type": "MissingDependencyFingerprints",
                    "error": "locked_test requires the exact pre-2025 frozen dependencies",
                    "recorded_at_utc": utc_now(),
                }
            )
        else:
            try:
                phase_evidence = producer.run_locked_test(
                    runner,
                    dependency_fingerprints=dependency_fingerprints,
                    resume=resume,
                )
                completed_units.extend(
                    unit.unit_id
                    for unit in producer.plan.work_units(role="locked_test")
                )
            except Exception as exc:
                failed_units.append(
                    {
                        "unit_id": "locked_test_2025",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recorded_at_utc": utc_now(),
                    }
                )
    status = (
        "COMPLETE"
        if not failed_units and len(completed_units) == expected_unit_count
        else "PARTIAL"
    )
    manifest = {
        "schema": SCHEMA_VERSION,
        "stage": "dart_annual_forecast_training",
        "phase": phase,
        "run_id": run_id,
        "status": status,
        "downstream_ready": status == "COMPLETE",
        "paper_mode_fail_closed": True,
        "synthetic_fallback_used": False,
        "repository_root": repository.as_posix(),
        "artifact_root": artifacts.as_posix(),
        "stage_root": stage_root.as_posix(),
        "config_path": config_file.as_posix(),
        "config_sha256": sha256_file(config_file),
        "training_config": asdict(effective_training),
        "training_config_sha256": effective_training.fingerprint,
        "input_fingerprints": merged_fingerprints,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "required_candidate_pool": list(SUPPORTED_CANDIDATES),
        "expected_unit_count": expected_unit_count,
        "completed_unit_count": len(completed_units),
        "completed_units": completed_units,
        "failed_unit_count": len(failed_units),
        "failed_units": failed_units,
        "neural_pretraining_scheduler": pretraining_scheduler_evidence,
        "anchor_overlay_recovery": anchor_overlay_recovery,
        "anchor_only_compatibility": anchor_compatibility_evidence,
        "v4_full_epoch_archive_inventory": full_epoch_archive_inventory,
        "phase_evidence": phase_evidence,
    }
    manifest["manifest_fingerprint"] = sha256_json(manifest)
    atomic_json(stage_root / f"{phase}_stage_manifest.json", manifest)
    return manifest


__all__ = [
    "AnnualTrainingData",
    "DARTTrainingAdapterError",
    "DARTTrainingConfig",
    "NYISOPanelBundle",
    "NYISOPanelSource",
    "ReferencedNYISODARTTrainingRunner",
    "ReferencedNYISOPanelSource",
    "ResidualCouplingError",
    "SUPPORTED_CANDIDATES",
    "UnsupportedTrainingCellError",
    "ValidationResidualGaussianCopulaProvider",
    "build_annual_training_data",
    "build_referenced_dart_training_stack",
    "run_dart_forecast_training_stage",
]
