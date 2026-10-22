"""Validation-only epoch selection for formal DART price models.

The outer DART-ValueSelect rule compares *model families* with market replay.
This module solves a different problem: selecting one checkpoint inside a
single neural fit without reading replay value or locked-test outcomes.

The policy is deliberately constraint-first and set-valued until the last
step.  It uses a chronologically earlier calibration-fit slice and a later
checkpoint-selection slice, then applies three gates:

1. finite/monotone/gradient-health checks;
2. simultaneous 80/90 percent coverage calibration checks;
3. a simultaneous global-CRPS confidence set followed by a simultaneous
   proper threshold-weighted-CRPS confidence set.

If several epochs remain statistically indistinguishable, the earliest epoch
is selected.  No arbitrary weighted score and no per-epoch market replay is
used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any

import numpy as np

from dcvpp.core import sha256_json
from dcvpp.evaluation.dart_metrics import (
    crps_from_quantiles,
    emphasized_tail_threshold_weighted_crps_from_quantiles,
)
from dcvpp.experiments.forecast_calibration import (
    apply_additive_conformal_quantiles,
    crossing_count,
    fit_additive_conformal_quantiles,
)
from dcvpp.statistics.block_bootstrap import moving_block_indices


SCHEMA_VERSION = "dcvpp.dart_checkpoint_selection.v2"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
TAIL_QUANTILES = (0.01, 0.05, 0.95, 0.99)


class DARTCheckpointSelectionError(RuntimeError):
    """Raised when an epoch panel cannot support truthful selection."""


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _seed(base: int, label: str) -> int:
    payload = f"DART_CHECKPOINT|{int(base)}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    """Frozen statistical and health controls for one neural fit."""

    confidence_level: float = 0.95
    block_length_days: int = 7
    bootstrap_resamples: int = 1000
    provisional_bootstrap_resamples: int = 200
    bootstrap_seed: int = 20260809
    minimum_calibration_days: int = 56
    minimum_selection_days: int = 28
    minimum_calibration_samples: int = 500
    maximum_absolute_coverage_error_80: float = 0.05
    maximum_absolute_coverage_error_90: float = 0.05
    maximum_raw_crossing_rate: float = 0.05
    require_zero_postcalibration_crossings: bool = True
    earlier_epoch_tie_rule: bool = True
    maximum_epochs: int = 100
    minimum_epochs: int = 40
    group_assessment_interval_epochs: int = 5
    required_stable_group_assessments: int = 3
    minimum_proper_score_improvement: float = 1.0e-8
    individual_seed_early_stopping_forbidden: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie inside (0, 1)")
        if self.block_length_days < 1:
            raise ValueError("block_length_days must be positive")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not 20 <= self.provisional_bootstrap_resamples <= self.bootstrap_resamples:
            raise ValueError(
                "provisional_bootstrap_resamples must lie in [20, bootstrap_resamples]"
            )
        if self.minimum_calibration_days < 1 or self.minimum_selection_days < 1:
            raise ValueError("checkpoint calibration/selection day minima must be positive")
        if self.minimum_calibration_samples < 1:
            raise ValueError("minimum_calibration_samples must be positive")
        if not 0.0 <= self.maximum_absolute_coverage_error_80 < 0.5:
            raise ValueError("80 percent coverage tolerance must lie in [0, 0.5)")
        if not 0.0 <= self.maximum_absolute_coverage_error_90 < 0.5:
            raise ValueError("90 percent coverage tolerance must lie in [0, 0.5)")
        if not 0.0 <= self.maximum_raw_crossing_rate <= 1.0:
            raise ValueError("maximum_raw_crossing_rate must lie in [0, 1]")
        if not 1 <= self.minimum_epochs <= self.maximum_epochs:
            raise ValueError("minimum_epochs must lie inside [1, maximum_epochs]")
        if self.group_assessment_interval_epochs < 1:
            raise ValueError("group assessment interval must be positive")
        if self.required_stable_group_assessments < 2:
            raise ValueError("at least two stable group assessments are required")
        if self.minimum_proper_score_improvement < 0.0:
            raise ValueError("minimum proper-score improvement must be nonnegative")
        if not self.individual_seed_early_stopping_forbidden:
            raise ValueError("formal DART forbids individual-seed early stopping")

    @property
    def fingerprint(self) -> str:
        return sha256_json({"schema_version": SCHEMA_VERSION, **asdict(self)})


def fit_checkpoint_tail_thresholds(
    core_training_actual: np.ndarray,
    *,
    zone_names: Sequence[str],
) -> dict[str, Any]:
    """Freeze candidate-independent q01/q05/q95/q99 thresholds from core train.

    The input must be ``[operating_date, zone, horizon]``.  No calibration,
    selection, OOS, replay, or locked-test outcome is admitted here.
    """

    actual = np.asarray(core_training_actual, dtype="float64")
    zones = tuple(map(str, zone_names))
    if actual.ndim != 3 or actual.shape[1] != len(zones) or not zones:
        raise DARTCheckpointSelectionError(
            "core-training thresholds require [day, zone, horizon] and exact zones"
        )
    if len(set(zones)) != len(zones) or not np.isfinite(actual).all():
        raise DARTCheckpointSelectionError(
            "core-training threshold panel has duplicate zones or non-finite values"
        )
    values = np.quantile(
        actual,
        np.asarray(TAIL_QUANTILES, dtype="float64"),
        axis=(0, 2),
        method="linear",
    )
    q01, q05, q95, q99 = values
    if np.any((q01 > q05) | (q05 >= q95) | (q95 > q99)):
        raise DARTCheckpointSelectionError(
            "training-core tail thresholds do not satisfy q01 <= q05 < q95 <= q99"
        )
    payload: dict[str, Any] = {
        "schema_version": "dcvpp.dart_checkpoint_tail_thresholds.v1",
        "fit_role": "candidate_independent_core_training_only",
        "locked_test_or_market_replay_consumed": False,
        "zone_names": list(zones),
        "tail_quantiles": list(TAIL_QUANTILES),
        "core_training_actual_sha256": _array_sha256(actual),
        "lower_extreme_q01_by_zone": q01.tolist(),
        "lower_tail_q05_by_zone": q05.tolist(),
        "upper_tail_q95_by_zone": q95.tolist(),
        "upper_extreme_q99_by_zone": q99.tolist(),
    }
    payload["threshold_contract_sha256"] = sha256_json(payload)
    return payload


def _validated_tail_thresholds(
    contract: Mapping[str, Any],
    *,
    target_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if len(target_shape) != 3:
        raise DARTCheckpointSelectionError(
            "checkpoint selection targets must be [day, zone, horizon]"
        )
    payload = dict(contract)
    claimed = str(payload.pop("threshold_contract_sha256", ""))
    if len(claimed) != 64 or claimed != sha256_json(payload):
        raise DARTCheckpointSelectionError("tail-threshold contract hash mismatch")
    if payload.get("fit_role") != "candidate_independent_core_training_only":
        raise DARTCheckpointSelectionError("tail thresholds are not core-training-only")
    if payload.get("locked_test_or_market_replay_consumed") is not False:
        raise DARTCheckpointSelectionError("tail thresholds consumed forbidden evidence")
    if tuple(map(float, payload.get("tail_quantiles", ()))) != TAIL_QUANTILES:
        raise DARTCheckpointSelectionError("tail-threshold quantiles changed")
    zones = tuple(map(str, payload.get("zone_names", ())))
    if len(zones) != target_shape[1] or len(set(zones)) != len(zones):
        raise DARTCheckpointSelectionError("tail-threshold zone contract changed")

    def vector(name: str) -> np.ndarray:
        value = np.asarray(payload.get(name), dtype="float64")
        if value.shape != (target_shape[1],) or not np.isfinite(value).all():
            raise DARTCheckpointSelectionError(
                f"tail-threshold field {name} is not one finite value per zone"
            )
        return value.reshape(1, target_shape[1], 1)

    q01 = vector("lower_extreme_q01_by_zone")
    q05 = vector("lower_tail_q05_by_zone")
    q95 = vector("upper_tail_q95_by_zone")
    q99 = vector("upper_extreme_q99_by_zone")
    if np.any((q01 > q05) | (q05 >= q95) | (q95 > q99)):
        raise DARTCheckpointSelectionError("tail-threshold ordering is invalid")
    return q01, q05, q95, q99, claimed


@dataclass(frozen=True)
class EpochMetricRecord:
    epoch: int
    training_loss: float
    maximum_gradient_norm: float
    nonfinite_gradient_steps: int
    gradient_clipped_steps: int
    raw_crossing_rate: float
    postcalibration_crossing_count: int
    finite_predictions: bool
    daily_global_crps: tuple[float, ...]
    daily_tail_proper_score: tuple[float, ...]
    daily_coverage_80: tuple[float, ...]
    daily_coverage_90: tuple[float, ...]
    daily_width_80: tuple[float, ...]
    daily_width_90: tuple[float, ...]
    calibration_fit_sha256: str
    calibration_panel_sha256: str
    calibration_date_sha256: str
    tail_threshold_contract_sha256: str
    selection_panel_sha256: str
    selection_date_sha256: str
    quantiles: tuple[float, ...] = QUANTILES

    @property
    def day_count(self) -> int:
        return len(self.daily_global_crps)

    @property
    def health_pass(self) -> bool:
        return bool(
            self.epoch >= 1
            and math.isfinite(self.training_loss)
            and math.isfinite(self.maximum_gradient_norm)
            and self.nonfinite_gradient_steps == 0
            and self.finite_predictions
            and 0.0 <= self.raw_crossing_rate <= 1.0
            and self.postcalibration_crossing_count >= 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "health_pass": self.health_pass,
            "day_count": self.day_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpochMetricRecord":
        fields = {
            "epoch": int(value["epoch"]),
            "training_loss": float(value["training_loss"]),
            "maximum_gradient_norm": float(value["maximum_gradient_norm"]),
            "nonfinite_gradient_steps": int(value["nonfinite_gradient_steps"]),
            "gradient_clipped_steps": int(value["gradient_clipped_steps"]),
            "raw_crossing_rate": float(value["raw_crossing_rate"]),
            "postcalibration_crossing_count": int(
                value["postcalibration_crossing_count"]
            ),
            "finite_predictions": bool(value["finite_predictions"]),
            "daily_global_crps": tuple(
                map(float, value["daily_global_crps"])
            ),
            "daily_tail_proper_score": tuple(
                map(
                    float,
                    value.get(
                        "daily_tail_proper_score", value.get("daily_tail_pinball", ())
                    ),
                )
            ),
            "daily_coverage_80": tuple(map(float, value["daily_coverage_80"])),
            "daily_coverage_90": tuple(map(float, value["daily_coverage_90"])),
            "daily_width_80": tuple(map(float, value["daily_width_80"])),
            "daily_width_90": tuple(map(float, value["daily_width_90"])),
            "calibration_fit_sha256": str(value["calibration_fit_sha256"]),
            "calibration_panel_sha256": str(value["calibration_panel_sha256"]),
            "calibration_date_sha256": str(value["calibration_date_sha256"]),
            "tail_threshold_contract_sha256": str(
                value["tail_threshold_contract_sha256"]
            ),
            "selection_panel_sha256": str(value["selection_panel_sha256"]),
            "selection_date_sha256": str(value["selection_date_sha256"]),
            "quantiles": tuple(map(float, value.get("quantiles", QUANTILES))),
        }
        return cls(**fields)


def build_epoch_metric_record(
    *,
    epoch: int,
    calibration_actual: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_dates: Sequence[str],
    selection_actual: np.ndarray,
    selection_prediction: np.ndarray,
    selection_dates: Sequence[str],
    tail_threshold_contract: Mapping[str, Any],
    training_loss: float,
    maximum_gradient_norm: float,
    nonfinite_gradient_steps: int,
    gradient_clipped_steps: int,
    minimum_calibration_days: int = 56,
    minimum_calibration_samples: int = 500,
) -> EpochMetricRecord:
    """Create one validation-only record from two chronological subpanels."""

    cal_y = np.asarray(calibration_actual, dtype="float64")
    cal_q = np.asarray(calibration_prediction, dtype="float64")
    select_y = np.asarray(selection_actual, dtype="float64")
    select_q_raw = np.asarray(selection_prediction, dtype="float64")
    if cal_q.shape[:-1] != cal_y.shape or select_q_raw.shape[:-1] != select_y.shape:
        raise DARTCheckpointSelectionError("actual/prediction epoch shapes differ")
    if cal_q.shape[-1] != len(QUANTILES) or select_q_raw.shape[-1] != len(QUANTILES):
        raise DARTCheckpointSelectionError("epoch forecasts require five fixed quantiles")
    if cal_y.shape[0] != len(calibration_dates):
        raise DARTCheckpointSelectionError("calibration date count differs from tensor days")
    if select_y.shape[0] != len(selection_dates):
        raise DARTCheckpointSelectionError("selection date count differs from tensor days")

    def ordered_utc(values: Sequence[str], *, label: str) -> tuple[str, ...]:
        texts = tuple(map(str, values))
        if len(set(texts)) != len(texts):
            raise DARTCheckpointSelectionError(f"{label} dates are not unique")
        parsed: list[datetime] = []
        for text in texts:
            try:
                value = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as error:
                raise DARTCheckpointSelectionError(
                    f"{label} dates are not ISO timestamps"
                ) from error
            if value.tzinfo is None or value.utcoffset() is None:
                raise DARTCheckpointSelectionError(f"{label} dates must be UTC-aware")
            parsed.append(value.astimezone(timezone.utc))
        if parsed != sorted(parsed):
            raise DARTCheckpointSelectionError(f"{label} dates are not increasing")
        return texts

    calibration_date_text = ordered_utc(calibration_dates, label="calibration")
    selection_date_text = ordered_utc(selection_dates, label="selection")
    if len(calibration_date_text) < int(minimum_calibration_days):
        raise DARTCheckpointSelectionError("checkpoint calibration panel is too short")
    if calibration_date_text and selection_date_text:
        calibration_end = datetime.fromisoformat(
            calibration_date_text[-1].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        selection_start = datetime.fromisoformat(
            selection_date_text[0].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        if calibration_end >= selection_start:
            raise DARTCheckpointSelectionError(
                "checkpoint calibration must precede checkpoint selection"
            )
    finite_predictions = bool(
        np.isfinite(cal_y).all()
        and np.isfinite(cal_q).all()
        and np.isfinite(select_y).all()
        and np.isfinite(select_q_raw).all()
    )
    if not finite_predictions:
        raise DARTCheckpointSelectionError("epoch metric panels contain non-finite values")
    raw_crossings = crossing_count(select_q_raw)
    raw_comparisons = int(np.prod(select_q_raw.shape[:-1])) * (len(QUANTILES) - 1)
    calibrator = fit_additive_conformal_quantiles(
        cal_y,
        cal_q,
        quantiles=QUANTILES,
        minimum_samples=int(minimum_calibration_samples),
    )
    calibrated = apply_additive_conformal_quantiles(select_q_raw, calibrator)
    post_crossings = crossing_count(calibrated)
    q01, q05, q95, q99, threshold_sha = _validated_tail_thresholds(
        tail_threshold_contract,
        target_shape=select_y.shape,
    )
    global_per_event = np.asarray(
        crps_from_quantiles(
            select_y,
            calibrated,
            QUANTILES,
            reduction="none",
        ),
        dtype="float64",
    )
    tail_per_event = np.asarray(
        emphasized_tail_threshold_weighted_crps_from_quantiles(
            select_y,
            calibrated,
            QUANTILES,
            lower_extreme_threshold=q01,
            lower_tail_threshold=q05,
            upper_tail_threshold=q95,
            upper_extreme_threshold=q99,
            reduction="none",
        ),
        dtype="float64",
    )
    # First axis is the chronological operating-date axis. Every remaining
    # zone/horizon element receives equal weight within the day.
    daily_global = np.mean(global_per_event, axis=tuple(range(1, select_y.ndim)))
    daily_tail = np.mean(tail_per_event, axis=tuple(range(1, select_y.ndim)))
    lower_80, upper_80 = calibrated[..., 1], calibrated[..., 3]
    lower_90, upper_90 = calibrated[..., 0], calibrated[..., 4]
    reduce_axes = tuple(range(1, select_y.ndim))
    daily_coverage_80 = np.mean(
        (select_y >= lower_80) & (select_y <= upper_80), axis=reduce_axes
    )
    daily_coverage_90 = np.mean(
        (select_y >= lower_90) & (select_y <= upper_90), axis=reduce_axes
    )
    daily_width_80 = np.mean(upper_80 - lower_80, axis=reduce_axes)
    daily_width_90 = np.mean(upper_90 - lower_90, axis=reduce_axes)
    calibration_manifest = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in calibrator.items()
    }
    return EpochMetricRecord(
        epoch=int(epoch),
        training_loss=float(training_loss),
        maximum_gradient_norm=float(maximum_gradient_norm),
        nonfinite_gradient_steps=int(nonfinite_gradient_steps),
        gradient_clipped_steps=int(gradient_clipped_steps),
        raw_crossing_rate=(float(raw_crossings / raw_comparisons) if raw_comparisons else 0.0),
        postcalibration_crossing_count=int(post_crossings),
        finite_predictions=finite_predictions,
        daily_global_crps=tuple(map(float, daily_global)),
        daily_tail_proper_score=tuple(map(float, daily_tail)),
        daily_coverage_80=tuple(map(float, daily_coverage_80)),
        daily_coverage_90=tuple(map(float, daily_coverage_90)),
        daily_width_80=tuple(map(float, daily_width_80)),
        daily_width_90=tuple(map(float, daily_width_90)),
        calibration_fit_sha256=sha256_json(calibration_manifest),
        calibration_panel_sha256=sha256_json(
            {
                "actual": _array_sha256(cal_y),
                "raw_prediction": _array_sha256(cal_q),
                "date_sha256": sha256_json(list(calibration_date_text)),
            }
        ),
        calibration_date_sha256=sha256_json(list(calibration_date_text)),
        tail_threshold_contract_sha256=threshold_sha,
        selection_panel_sha256=sha256_json(
            {
                "actual": _array_sha256(select_y),
                "raw_prediction": _array_sha256(select_q_raw),
                "calibrated_prediction": _array_sha256(calibrated),
            }
        ),
        selection_date_sha256=sha256_json(list(selection_date_text)),
    )


def aggregate_seed_epoch_records(
    records_by_seed: Mapping[int, Sequence[EpochMetricRecord | Mapping[str, Any]]],
    *,
    expected_seeds: Sequence[int] = (20260809, 20260819, 20260829),
) -> tuple[tuple[EpochMetricRecord, ...], dict[str, Any]]:
    """Average checkpoint evidence within date across the exact study seeds.

    Seed fits are repeated estimates of one model family, not independent
    operating dates.  All seeds must therefore expose the same epoch and date
    panels before any confidence set is built.
    """

    seeds = tuple(map(int, expected_seeds))
    if tuple(sorted(map(int, records_by_seed))) != tuple(sorted(seeds)):
        raise DARTCheckpointSelectionError("checkpoint evidence lacks the exact study seeds")
    converted: dict[int, tuple[EpochMetricRecord, ...]] = {
        seed: tuple(
            record
            if isinstance(record, EpochMetricRecord)
            else EpochMetricRecord.from_dict(record)
            for record in records_by_seed[seed]
        )
        for seed in seeds
    }
    epoch_sets = {tuple(record.epoch for record in rows) for rows in converted.values()}
    if len(epoch_sets) != 1 or not next(iter(epoch_sets), ()):
        raise DARTCheckpointSelectionError("seed checkpoint ledgers use different epochs")
    epochs = next(iter(epoch_sets))
    grouped: list[EpochMetricRecord] = []
    seed_panel_records: list[dict[str, Any]] = []
    for epoch_index, epoch in enumerate(epochs):
        rows = tuple(converted[seed][epoch_index] for seed in seeds)
        if {row.epoch for row in rows} != {epoch}:
            raise DARTCheckpointSelectionError("seed checkpoint epoch alignment changed")
        if len({row.selection_date_sha256 for row in rows}) != 1:
            raise DARTCheckpointSelectionError("seed checkpoint date panels differ")
        if len({row.calibration_date_sha256 for row in rows}) != 1:
            raise DARTCheckpointSelectionError("seed checkpoint calibration dates differ")
        if len({row.tail_threshold_contract_sha256 for row in rows}) != 1:
            raise DARTCheckpointSelectionError("seed checkpoint tail thresholds differ")
        if len({row.day_count for row in rows}) != 1:
            raise DARTCheckpointSelectionError("seed checkpoint day counts differ")

        def daily(field: str) -> tuple[float, ...]:
            matrix = np.asarray([getattr(row, field) for row in rows], dtype="float64")
            if not np.isfinite(matrix).all():
                raise DARTCheckpointSelectionError(
                    f"seed checkpoint field {field} contains non-finite values"
                )
            return tuple(map(float, np.mean(matrix, axis=0)))

        grouped.append(
            EpochMetricRecord(
                epoch=int(epoch),
                training_loss=float(np.mean([row.training_loss for row in rows])),
                maximum_gradient_norm=float(
                    np.max([row.maximum_gradient_norm for row in rows])
                ),
                nonfinite_gradient_steps=int(
                    sum(row.nonfinite_gradient_steps for row in rows)
                ),
                gradient_clipped_steps=int(
                    sum(row.gradient_clipped_steps for row in rows)
                ),
                raw_crossing_rate=float(np.max([row.raw_crossing_rate for row in rows])),
                postcalibration_crossing_count=int(
                    sum(row.postcalibration_crossing_count for row in rows)
                ),
                finite_predictions=all(row.finite_predictions for row in rows),
                daily_global_crps=daily("daily_global_crps"),
                daily_tail_proper_score=daily("daily_tail_proper_score"),
                daily_coverage_80=daily("daily_coverage_80"),
                daily_coverage_90=daily("daily_coverage_90"),
                daily_width_80=daily("daily_width_80"),
                daily_width_90=daily("daily_width_90"),
                calibration_fit_sha256=sha256_json(
                    {
                        str(seed): row.calibration_fit_sha256
                        for seed, row in zip(seeds, rows, strict=True)
                    }
                ),
                calibration_panel_sha256=sha256_json(
                    {
                        str(seed): row.calibration_panel_sha256
                        for seed, row in zip(seeds, rows, strict=True)
                    }
                ),
                calibration_date_sha256=rows[0].calibration_date_sha256,
                tail_threshold_contract_sha256=rows[0].tail_threshold_contract_sha256,
                selection_panel_sha256=sha256_json(
                    {
                        str(seed): row.selection_panel_sha256
                        for seed, row in zip(seeds, rows, strict=True)
                    }
                ),
                selection_date_sha256=rows[0].selection_date_sha256,
            )
        )
        seed_panel_records.append(
            {
                "epoch": int(epoch),
                "seed_record_sha256": {
                    str(seed): sha256_json(row.as_dict())
                    for seed, row in zip(seeds, rows, strict=True)
                },
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "aggregation": "within_operating_date_mean_across_exact_three_seeds",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "epoch_count": len(grouped),
        "selection_date_sha256": grouped[0].selection_date_sha256,
        "calibration_date_sha256": grouped[0].calibration_date_sha256,
        "tail_threshold_contract_sha256": grouped[0].tail_threshold_contract_sha256,
        "seed_panel_records_sha256": sha256_json(seed_panel_records),
        "seed_panel_records": seed_panel_records,
        "locked_test_or_market_replay_consumed": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return tuple(grouped), manifest


def _bootstrap_indices(
    day_count: int,
    config: CheckpointSelectionConfig,
    *,
    label: str,
    resamples: int,
) -> np.ndarray:
    return moving_block_indices(
        day_count,
        min(config.block_length_days, day_count),
        resamples=int(resamples),
        seed=_seed(config.bootstrap_seed, label),
        circular=True,
    )


def _standardized(observed: np.ndarray, centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    standard_error = np.std(centered, axis=0, ddof=1)
    observed_t = np.zeros_like(observed, dtype="float64")
    bootstrap_t = np.zeros_like(centered, dtype="float64")
    positive = standard_error > np.finfo(float).eps
    observed_t[positive] = observed[positive] / standard_error[positive]
    bootstrap_t[:, positive] = centered[:, positive] / standard_error[positive]
    deterministic = ~positive & (np.abs(observed) > np.finfo(float).eps)
    observed_t[deterministic] = np.sign(observed[deterministic]) * np.inf
    return observed_t, bootstrap_t


def _calibration_gate(
    records: Sequence[EpochMetricRecord],
    config: CheckpointSelectionConfig,
    *,
    resamples: int,
) -> dict[str, Any]:
    day_count = records[0].day_count
    columns: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    observed_deviation: list[float] = []
    observed_excess: list[float] = []
    for record in records:
        for nominal, values in (
            (80, record.daily_coverage_80),
            (90, record.daily_coverage_90),
        ):
            vector = np.asarray(values, dtype="float64")
            columns.append(vector)
            labels.append((record.epoch, nominal))
            deviation = float(np.mean(vector) - nominal / 100.0)
            tolerance = (
                config.maximum_absolute_coverage_error_80
                if nominal == 80
                else config.maximum_absolute_coverage_error_90
            )
            observed_deviation.append(deviation)
            observed_excess.append(
                math.copysign(max(abs(deviation) - tolerance, 0.0), deviation)
            )
    matrix = np.column_stack(columns)
    indices = _bootstrap_indices(
        day_count, config, label="calibration", resamples=resamples
    )
    bootstrap_means = np.mean(matrix[indices, :], axis=1)
    observed_means = np.mean(matrix, axis=0)
    centered = bootstrap_means - observed_means[None, :]
    observed_t, bootstrap_t = _standardized(
        np.asarray(observed_excess, dtype="float64"), centered
    )
    maximum = np.max(np.abs(bootstrap_t), axis=1)
    critical = float(np.quantile(maximum, config.confidence_level, method="higher"))
    by_epoch: dict[int, dict[str, Any]] = {}
    for index, (epoch, nominal) in enumerate(labels):
        row = by_epoch.setdefault(epoch, {"pass": True})
        passed = bool(abs(observed_t[index]) <= critical)
        row[f"coverage_{nominal}"] = float(observed_means[index])
        row[f"deviation_{nominal}"] = float(observed_deviation[index])
        row[f"allowed_absolute_deviation_{nominal}"] = float(
            config.maximum_absolute_coverage_error_80
            if nominal == 80
            else config.maximum_absolute_coverage_error_90
        )
        row[f"excess_deviation_{nominal}"] = float(observed_excess[index])
        row[f"studentized_deviation_{nominal}"] = float(observed_t[index])
        row[f"coverage_{nominal}_pass"] = passed
        row["pass"] = bool(row["pass"] and passed)
    return {
        "method": "simultaneous_two_interval_moving_block_bootstrap",
        "critical_value": critical,
        "confidence_level": config.confidence_level,
        "block_length_days": min(config.block_length_days, day_count),
        "resamples": int(resamples),
        "bootstrap_indices_sha256": _array_sha256(indices),
        "by_epoch": {str(key): value for key, value in sorted(by_epoch.items())},
    }


def _loss_confidence_set(
    records: Sequence[EpochMetricRecord],
    config: CheckpointSelectionConfig,
    *,
    field: str,
    label: str,
    resamples: int,
) -> dict[str, Any]:
    if not records:
        raise DARTCheckpointSelectionError(f"{label} confidence set has no candidates")
    epochs = tuple(record.epoch for record in records)
    matrix = np.column_stack(
        [np.asarray(getattr(record, field), dtype="float64") for record in records]
    )
    means = np.mean(matrix, axis=0)
    best_index = int(np.argmin(means))
    best_epoch = epochs[best_index]
    differences = matrix - matrix[:, [best_index]]
    observed = np.mean(differences, axis=0)
    indices = _bootstrap_indices(
        matrix.shape[0], config, label=label, resamples=resamples
    )
    bootstrap = np.mean(differences[indices, :], axis=1)
    centered = bootstrap - observed[None, :]
    observed_t, bootstrap_t = _standardized(observed, centered)
    # One-sided family-wise band: only a statistically positive loss
    # differential can exclude an epoch as worse than the empirical best.
    maximum = np.max(bootstrap_t, axis=1)
    critical = float(np.quantile(maximum, config.confidence_level, method="higher"))
    included = tuple(
        sorted(
            epoch
            for index, epoch in enumerate(epochs)
            if index == best_index or observed_t[index] <= critical
        )
    )
    return {
        "method": "one_sided_simultaneous_moving_block_confidence_set",
        "loss_field": field,
        "best_epoch": int(best_epoch),
        "included_epochs": list(included),
        "excluded_epochs": sorted(set(epochs) - set(included)),
        "mean_losses": {
            str(epoch): float(means[index]) for index, epoch in enumerate(epochs)
        },
        "mean_loss_differences_from_best": {
            str(epoch): float(observed[index]) for index, epoch in enumerate(epochs)
        },
        "studentized_differences": {
            str(epoch): float(observed_t[index]) for index, epoch in enumerate(epochs)
        },
        "critical_value": critical,
        "confidence_level": config.confidence_level,
        "block_length_days": min(config.block_length_days, matrix.shape[0]),
        "resamples": int(resamples),
        "bootstrap_indices_sha256": _array_sha256(indices),
    }


def select_checkpoint_epoch(
    records: Sequence[EpochMetricRecord | Mapping[str, Any]],
    *,
    config: CheckpointSelectionConfig | None = None,
    provisional: bool = False,
) -> dict[str, Any]:
    """Select the earliest epoch surviving health, calibration, and two ECS gates."""

    effective = config or CheckpointSelectionConfig()
    converted = tuple(
        record
        if isinstance(record, EpochMetricRecord)
        else EpochMetricRecord.from_dict(record)
        for record in records
    )
    if not converted:
        raise DARTCheckpointSelectionError("at least one epoch metric record is required")
    epochs = [record.epoch for record in converted]
    if epochs != sorted(set(epochs)):
        raise DARTCheckpointSelectionError("epoch records must be unique and increasing")
    day_counts = {record.day_count for record in converted}
    date_hashes = {record.selection_date_sha256 for record in converted}
    calibration_date_hashes = {
        record.calibration_date_sha256 for record in converted
    }
    threshold_hashes = {
        record.tail_threshold_contract_sha256 for record in converted
    }
    quantile_contracts = {record.quantiles for record in converted}
    if (
        len(day_counts) != 1
        or len(date_hashes) != 1
        or len(calibration_date_hashes) != 1
        or len(threshold_hashes) != 1
        or quantile_contracts != {QUANTILES}
    ):
        raise DARTCheckpointSelectionError(
            "all epochs must share one exact date/quantile checkpoint panel"
        )
    day_count = next(iter(day_counts))
    if day_count < effective.minimum_selection_days:
        raise DARTCheckpointSelectionError("checkpoint-selection panel is too short")
    for record in converted:
        numeric = (
            record.daily_global_crps,
            record.daily_tail_proper_score,
            record.daily_coverage_80,
            record.daily_coverage_90,
            record.daily_width_80,
            record.daily_width_90,
        )
        if any(len(values) != day_count for values in numeric):
            raise DARTCheckpointSelectionError("epoch daily metric lengths differ")
        if not all(np.isfinite(np.asarray(values, dtype=float)).all() for values in numeric):
            raise DARTCheckpointSelectionError("epoch daily metrics contain non-finite values")
        if not all(
            np.all((0.0 <= np.asarray(values)) & (np.asarray(values) <= 1.0))
            for values in (record.daily_coverage_80, record.daily_coverage_90)
        ):
            raise DARTCheckpointSelectionError("daily coverage values must lie in [0,1]")
    resamples = (
        effective.provisional_bootstrap_resamples
        if provisional
        else effective.bootstrap_resamples
    )
    health_pass = tuple(
        record
        for record in converted
        if record.health_pass
        and record.raw_crossing_rate <= effective.maximum_raw_crossing_rate
        and (
            not effective.require_zero_postcalibration_crossings
            or record.postcalibration_crossing_count == 0
        )
    )
    exclusions: dict[str, list[str]] = {}
    for record in converted:
        reasons: list[str] = []
        if not record.health_pass:
            reasons.append("HEALTH_GATE_FAILED")
        if record.raw_crossing_rate > effective.maximum_raw_crossing_rate:
            reasons.append("RAW_QUANTILE_CROSSING_RATE_EXCEEDED")
        if (
            effective.require_zero_postcalibration_crossings
            and record.postcalibration_crossing_count != 0
        ):
            reasons.append("POSTCALIBRATION_QUANTILE_CROSSING_PRESENT")
        if reasons:
            exclusions[str(record.epoch)] = reasons
    if not health_pass:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "NO_HEALTHY_EPOCH",
            "selected_epoch": None,
            "config": asdict(effective),
            "config_fingerprint": effective.fingerprint,
            "provisional": bool(provisional),
            "exclusions": exclusions,
        }
    calibration = _calibration_gate(
        health_pass, effective, resamples=resamples
    )
    calibration_pass = tuple(
        record
        for record in health_pass
        if calibration["by_epoch"][str(record.epoch)]["pass"]
    )
    for record in health_pass:
        if record not in calibration_pass:
            exclusions.setdefault(str(record.epoch), []).append(
                "SIMULTANEOUS_CALIBRATION_GATE_FAILED"
            )
    if not calibration_pass:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "NO_CALIBRATION_CREDIBLE_EPOCH",
            "selected_epoch": None,
            "config": asdict(effective),
            "config_fingerprint": effective.fingerprint,
            "provisional": bool(provisional),
            "health_pass_epochs": [record.epoch for record in health_pass],
            "calibration": calibration,
            "exclusions": exclusions,
        }
    global_set = _loss_confidence_set(
        calibration_pass,
        effective,
        field="daily_global_crps",
        label="global_crps",
        resamples=resamples,
    )
    global_epochs = set(map(int, global_set["included_epochs"]))
    global_pass = tuple(
        record for record in calibration_pass if record.epoch in global_epochs
    )
    tail_set = _loss_confidence_set(
        global_pass,
        effective,
        field="daily_tail_proper_score",
        label="proper_tail_twcrps",
        resamples=resamples,
    )
    tail_epochs = tuple(sorted(map(int, tail_set["included_epochs"])))
    if not tail_epochs:
        raise DARTCheckpointSelectionError("tail confidence set is unexpectedly empty")
    selected = min(tail_epochs) if effective.earlier_epoch_tie_rule else int(
        tail_set["best_epoch"]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "SELECTED",
        "selected_epoch": int(selected),
        "selection_rule": (
            "health_then_simultaneous_coverage_then_global_crps_confidence_set_"
            "then_proper_tail_twcrps_confidence_set_then_earliest_epoch"
        ),
        "no_arbitrary_weighted_composite": True,
        "market_replay_or_locked_test_consumed": False,
        "config": asdict(effective),
        "config_fingerprint": effective.fingerprint,
        "provisional": bool(provisional),
        "exact_selection_date_sha256": converted[0].selection_date_sha256,
        "tail_threshold_contract_sha256": converted[0].tail_threshold_contract_sha256,
        "selection_day_count": day_count,
        "health_pass_epochs": [record.epoch for record in health_pass],
        "calibration_pass_epochs": [record.epoch for record in calibration_pass],
        "global_crps_confidence_set": global_set,
        "proper_tail_twcrps_confidence_set": tail_set,
        "calibration": calibration,
        "exclusions": exclusions,
    }
    result["selection_manifest_sha256"] = sha256_json(result)
    return result


def synchronized_group_early_stopping_decision(
    audits: Sequence[Mapping[str, Any]],
    *,
    config: CheckpointSelectionConfig | None = None,
) -> dict[str, Any]:
    """Decide whether an exact three-seed group may stop before epoch 100.

    Each audit must contain ``completed_epoch`` and a provisional ``selection``
    result returned by :func:`select_checkpoint_epoch`.  This function never
    decides from an individual seed.  It requires a stable survivor-set
    signature and no material improvement in either proper-score family across
    the last preregistered assessments.
    """

    effective = config or CheckpointSelectionConfig()
    rows = tuple(audits)
    required = effective.required_stable_group_assessments
    base: dict[str, Any] = {
        "schema_version": "dcvpp.dart_synchronized_group_early_stopping.v1",
        "maximum_epochs": effective.maximum_epochs,
        "minimum_epochs": effective.minimum_epochs,
        "assessment_interval_epochs": effective.group_assessment_interval_epochs,
        "required_stable_assessments": required,
        "minimum_proper_score_improvement": (
            effective.minimum_proper_score_improvement
        ),
        "individual_seed_early_stopping_forbidden": True,
        "audit_count": len(rows),
    }
    if not rows:
        return {**base, "stop": False, "reason": "NO_GROUP_AUDIT"}
    completed_epochs = tuple(int(row.get("completed_epoch", -1)) for row in rows)
    if completed_epochs != tuple(sorted(set(completed_epochs))):
        raise DARTCheckpointSelectionError(
            "group early-stopping audits must be unique and increasing"
        )
    current_epoch = completed_epochs[-1]
    base["current_epoch"] = current_epoch
    if current_epoch >= effective.maximum_epochs:
        return {**base, "stop": True, "reason": "MAXIMUM_EPOCH_REACHED"}
    if current_epoch < effective.minimum_epochs or len(rows) < required:
        return {**base, "stop": False, "reason": "MINIMUM_GROUP_EVIDENCE_NOT_REACHED"}
    recent = rows[-required:]
    recent_epochs = tuple(int(row["completed_epoch"]) for row in recent)
    expected_epochs = tuple(
        range(
            recent_epochs[0],
            recent_epochs[0]
            + effective.group_assessment_interval_epochs * required,
            effective.group_assessment_interval_epochs,
        )
    )
    if recent_epochs != expected_epochs:
        raise DARTCheckpointSelectionError(
            "recent group audits do not follow the frozen epoch interval"
        )

    signatures: list[tuple[Any, ...]] = []
    best_global: list[float] = []
    best_tail: list[float] = []
    for row in recent:
        selection = row.get("selection")
        if not isinstance(selection, Mapping) or selection.get("status") != "SELECTED":
            return {**base, "stop": False, "reason": "GROUP_SELECTION_NOT_STABLE"}
        selected = int(selection["selected_epoch"])
        health = tuple(map(int, selection.get("health_pass_epochs", ())))
        calibration = tuple(
            map(int, selection.get("calibration_pass_epochs", ()))
        )
        global_set = selection.get("global_crps_confidence_set", {})
        tail_set = selection.get("proper_tail_twcrps_confidence_set", {})
        global_epochs = tuple(sorted(map(int, global_set.get("included_epochs", ()))))
        tail_epochs = tuple(sorted(map(int, tail_set.get("included_epochs", ()))))
        if (
            selected not in health
            or selected not in calibration
            or selected not in global_epochs
            or selected not in tail_epochs
        ):
            return {**base, "stop": False, "reason": "SELECTED_EPOCH_GATE_DRIFT"}
        signatures.append((selected, global_epochs, tail_epochs, True, True))
        global_means = tuple(map(float, global_set.get("mean_losses", {}).values()))
        tail_means = tuple(map(float, tail_set.get("mean_losses", {}).values()))
        if (
            not global_means
            or not tail_means
            or not np.isfinite(global_means).all()
            or not np.isfinite(tail_means).all()
        ):
            raise DARTCheckpointSelectionError(
                "group early-stopping audit lacks finite proper-score means"
            )
        best_global.append(min(global_means))
        best_tail.append(min(tail_means))
    if len(set(signatures)) != 1:
        return {**base, "stop": False, "reason": "SURVIVOR_SET_OR_EPOCH_CHANGED"}

    threshold = effective.minimum_proper_score_improvement
    global_improvements = tuple(
        max(best_global[index - 1] - best_global[index], 0.0)
        for index in range(1, len(best_global))
    )
    tail_improvements = tuple(
        max(best_tail[index - 1] - best_tail[index], 0.0)
        for index in range(1, len(best_tail))
    )
    if any(value > threshold for value in (*global_improvements, *tail_improvements)):
        return {
            **base,
            "stop": False,
            "reason": "PROPER_SCORE_STILL_IMPROVING",
            "global_improvements": list(global_improvements),
            "tail_improvements": list(tail_improvements),
        }
    return {
        **base,
        "stop": True,
        "reason": "THREE_SEED_GROUP_STABLE",
        "selected_epoch": signatures[0][0],
        "stable_audit_epochs": list(recent_epochs),
        "global_crps_survivor_epochs": list(signatures[0][1]),
        "proper_tail_twcrps_survivor_epochs": list(signatures[0][2]),
        "global_improvements": list(global_improvements),
        "tail_improvements": list(tail_improvements),
    }


__all__ = [
    "CheckpointSelectionConfig",
    "DARTCheckpointSelectionError",
    "EpochMetricRecord",
    "SCHEMA_VERSION",
    "aggregate_seed_epoch_records",
    "build_epoch_metric_record",
    "fit_checkpoint_tail_thresholds",
    "select_checkpoint_epoch",
    "synchronized_group_early_stopping_decision",
]
