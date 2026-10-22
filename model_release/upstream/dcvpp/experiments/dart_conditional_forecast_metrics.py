"""Forecast-only DART evidence on the common observable settlement panel.

This module deliberately stops before action construction, optimization, or
market replay.  It consumes only frozen 2022--2024 OOS DART distributions and
the candidate-independent post-recovery settlement index.  Dates are the
sampling unit: distinct forecast fits are averaged within each operating date
before the date panel is pooled.  Deterministic/statistical fits that occupy
three logical replay slots are evaluated once, so those slots are never
misrepresented as independent forecast fits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcvpp.core import atomic_json, sha256_file, sha256_json
from dcvpp.evaluation.dart_metrics import (
    emphasized_tail_threshold_weighted_crps_from_quantiles,
    pinball_loss,
)
from dcvpp.experiments.dart_training import ReferencedNYISOPanelSource


QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
REPLAY_SEEDS = (20260809, 20260819, 20260829)
EVENTS_PER_DAY = len(ZONES) * 96


class ConditionalForecastMetricError(RuntimeError):
    """Raised when frozen forecast evidence cannot be compared exactly."""


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _pr_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    score = np.asarray(probabilities, dtype=float).reshape(-1)
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        return None
    ordered = truth[np.argsort(-score, kind="stable")]
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float(np.sum(precision[ordered]) / positives)


def _classification(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    predicted = np.asarray(probabilities, dtype=float).reshape(-1) >= 0.50
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
    miss = fn / (tp + fn) if tp + fn else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miss_rate": miss,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_positive_count": tp + fp,
        "positive_count": tp + fn,
    }


def _pit_summary(values: np.ndarray) -> dict[str, Any]:
    pit = np.asarray(values, dtype=float).reshape(-1)
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return {"count": 0, "mean": None, "variance": None, "uniform_ks_distance": None}
    ordered = np.sort(pit)
    count = ordered.size
    upper = np.max(np.arange(1, count + 1) / count - ordered)
    lower = np.max(ordered - np.arange(count) / count)
    hist, edges = np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))
    return {
        "count": int(count),
        "mean": float(np.mean(pit)),
        "variance": float(np.var(pit)),
        "uniform_ks_distance": float(max(upper, lower)),
        "histogram_bin_edges": edges.tolist(),
        "histogram_counts": hist.astype(int).tolist(),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _training_thresholds(repository_root: Path, year: int) -> dict[str, Any]:
    """Reproduce the formal pre-origin per-zone DART threshold contract."""

    panel = ReferencedNYISOPanelSource(repository_root=repository_root).load_through_year(
        year - 1
    )
    matrices: dict[str, pd.DataFrame] = {}
    for name, frame in (("da_lbmp", panel.da_15m), ("rt_lbmp", panel.rt_15m)):
        rows = frame[["timestamp_utc", "zone", name]].copy()
        rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
        matrices[name] = rows.pivot(
            index="timestamp_utc", columns="zone", values=name
        ).reindex(columns=ZONES)
    cutoff = pd.Timestamp(f"{year}-01-01T00:00:00Z")
    da = matrices["da_lbmp"].loc[lambda value: value.index < cutoff]
    rt = matrices["rt_lbmp"].loc[lambda value: value.index < cutoff]
    common = da.index.intersection(rt.index)
    da_values = da.reindex(common).to_numpy(dtype=float)
    rt_values = rt.reindex(common).to_numpy(dtype=float)
    valid = np.isfinite(da_values).all(axis=1) & np.isfinite(rt_values).all(axis=1)
    if int(np.count_nonzero(valid)) < 500:
        raise ConditionalForecastMetricError(f"insufficient pre-{year} settlement history")
    da_values = da_values[valid]
    rt_values = rt_values[valid]
    dart = da_values - rt_values
    q01, q05, q95, q99 = np.quantile(dart, [0.01, 0.05, 0.95, 0.99], axis=0)
    core = {
        "schema_version": "dcvpp.per_zone_rolling_q01_q05_q95_q99_thresholds.v1",
        "forecast_year": year,
        "zone_order": list(ZONES),
        "dart_lower_q01_by_zone": q01.tolist(),
        "dart_lower_q05_by_zone": q05.tolist(),
        "dart_upper_q95_by_zone": q95.tolist(),
        "dart_upper_q99_by_zone": q99.tolist(),
        "fit_cutoff_exclusive_utc": cutoff.isoformat(),
        "training_row_count": int(dart.shape[0]),
        "source_sha256": sha256_json(
            {
                "forecast_year": year,
                "da_history": _array_sha256(da_values),
                "rt_history": _array_sha256(rt_values),
                "partitions_sha256": panel.source_manifest["partitions_sha256"],
            }
        ),
        "referenced_partitions_sha256": panel.source_manifest["partitions_sha256"],
        "future_targets_used": False,
        "locked_targets_used": False,
    }
    return {**core, "threshold_contract_sha256": sha256_json(core)}


def _settlement_year(
    repository_root: Path,
    records: Mapping[str, Any],
    dates: Sequence[str],
) -> np.ndarray:
    actual: list[np.ndarray] = []
    for day in dates:
        record = records[day]
        path = repository_root / str(record["path"])
        if sha256_file(path) != str(record["sha256"]):
            raise ConditionalForecastMetricError(f"settlement hash mismatch: {day}")
        with np.load(path, allow_pickle=False) as data:
            da = np.asarray(data["da_lbmp"], dtype=float)
            rt = np.asarray(data["rt_lbmp"], dtype=float)
        if da.shape != (4, 96) or rt.shape != da.shape or not np.isfinite(da - rt).all():
            raise ConditionalForecastMetricError(f"invalid settlement tensor: {day}")
        actual.append((da - rt).reshape(-1))
    return np.stack(actual)


def _artifact_stream(
    path: Path,
    repository_root: Path,
    valid_dates: Sequence[str],
    actual: np.ndarray,
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        event_keys = np.asarray(data["event_keys"]).astype(str)
        levels = np.asarray(data["quantile_levels"], dtype=float)
        quantile_values = np.asarray(data["dart_quantile_values"], dtype=float)
        draws = np.asarray(data["dart_da_minus_rt"], dtype=float)
    if not np.array_equal(levels, np.asarray(QUANTILES)):
        raise ConditionalForecastMetricError(f"quantile grid changed: {path}")
    if event_keys.size % EVENTS_PER_DAY or quantile_values.shape != (event_keys.size, 5):
        raise ConditionalForecastMetricError(f"invalid DART artifact geometry: {path}")
    if draws.shape != (event_keys.size, 100):
        raise ConditionalForecastMetricError(f"invalid draw geometry: {path}")
    all_days = event_keys[::EVENTS_PER_DAY]
    all_dates = np.asarray([value[:10] for value in all_days])
    requested = np.asarray(list(valid_dates))
    positions = np.searchsorted(all_dates, requested)
    if np.any(positions >= all_dates.size) or not np.array_equal(all_dates[positions], requested):
        raise ConditionalForecastMetricError(f"forecast artifact lacks common dates: {path}")
    quantile_values = quantile_values.reshape(-1, EVENTS_PER_DAY, 5)[positions]
    draws = draws.reshape(-1, EVENTS_PER_DAY, 100)[positions]
    days = len(valid_dates)
    q01 = np.tile(np.repeat(thresholds["dart_lower_q01_by_zone"], 96), (days, 1))
    q05 = np.tile(np.repeat(thresholds["dart_lower_q05_by_zone"], 96), (days, 1))
    q95 = np.tile(np.repeat(thresholds["dart_upper_q95_by_zone"], 96), (days, 1))
    q99 = np.tile(np.repeat(thresholds["dart_upper_q99_by_zone"], 96), (days, 1))
    median = quantile_values[..., 2]
    pinball = np.asarray(
        pinball_loss(actual, quantile_values, QUANTILES, reduction="none"), dtype=float
    )
    tail = np.asarray(
        emphasized_tail_threshold_weighted_crps_from_quantiles(
            actual,
            quantile_values,
            QUANTILES,
            lower_extreme_threshold=q01,
            lower_tail_threshold=q05,
            upper_tail_threshold=q95,
            upper_extreme_threshold=q99,
            reduction="none",
        ),
        dtype=float,
    )
    finite_draw_count = np.sum(np.isfinite(draws), axis=-1)
    if np.any(finite_draw_count != 100):
        raise ConditionalForecastMetricError(f"non-finite DART draws: {path}")
    upper_probability = np.mean(draws > q99[..., None], axis=-1)
    lower_probability = np.mean(draws < q01[..., None], axis=-1)
    less = np.sum(draws < actual[..., None], axis=-1)
    equal = np.sum(draws == actual[..., None], axis=-1)
    pit = (less + 0.5 * equal) / finite_draw_count
    values = {
        "absolute_error": np.abs(actual - median),
        "squared_error": (actual - median) ** 2,
        "pinball": np.mean(pinball, axis=-1),
        "crps": 2.0 * np.mean(pinball, axis=-1),
        "proper_tail_score": tail,
        "covered_80": ((actual >= quantile_values[..., 1]) & (actual <= quantile_values[..., 3])).astype(float),
        "width_80": quantile_values[..., 3] - quantile_values[..., 1],
        "covered_90": ((actual >= quantile_values[..., 0]) & (actual <= quantile_values[..., 4])).astype(float),
        "width_90": quantile_values[..., 4] - quantile_values[..., 0],
        "quantile_below": (actual[..., None] <= quantile_values).astype(float),
        "upper_probability": upper_probability,
        "combined_probability": np.clip(upper_probability + lower_probability, 0.0, 1.0),
        "pit": pit,
        "crossing_count": np.sum(np.diff(quantile_values, axis=-1) < 0.0, axis=(1, 2)).astype(float),
    }
    source = {
        "path": path.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(path),
        "event_keys_sha256": _array_sha256(event_keys),
        "valid_event_keys_sha256": _array_sha256(
            event_keys.reshape(-1, EVENTS_PER_DAY)[positions].reshape(-1)
        ),
    }
    return values, source


def _mean_streams(streams: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not streams:
        raise ConditionalForecastMetricError("no forecast stream is available")
    names = streams[0].keys()
    return {
        name: np.mean(np.stack([np.asarray(row[name]) for row in streams]), axis=0)
        for name in names
    }


def build_conditional_forecast_metrics(
    *,
    repository_root: Path,
    forecast_root: Path,
    settlement_index: Path,
    output_root: Path,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    forecast_root = Path(forecast_root).resolve()
    settlement_index = Path(settlement_index).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    index = json.loads(settlement_index.read_text(encoding="utf-8"))
    if index.get("phase") != "selection" or int(index.get("day_count", 0)) != 1086:
        raise ConditionalForecastMetricError("selection settlement index is not the 1086-day panel")
    selection_manifest_path = forecast_root / "selection_stage_manifest.json"
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    pool = selection_manifest["candidate_pool_contract"]
    candidates = tuple(map(str, pool["eligible_candidates"]))
    if len(candidates) != 8 or "itransformer" in candidates:
        raise ConditionalForecastMetricError("eligible candidate pool is not the frozen strict-eight set")

    dates = sorted(map(str, index["days"]))
    if dates[0] != "2022-01-01" or dates[-1] != "2024-12-31":
        raise ConditionalForecastMetricError("selection date range changed")
    dates_by_year = {
        year: [day for day in dates if day.startswith(f"{year}-")] for year in (2022, 2023, 2024)
    }
    thresholds = {year: _training_thresholds(repository_root, year) for year in dates_by_year}
    actual_by_year = {
        year: _settlement_year(repository_root, index["days"], year_dates)
        for year, year_dates in dates_by_year.items()
    }

    daily_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    source_artifacts: dict[str, list[dict[str, Any]]] = {}
    panel_arrays: dict[str, np.ndarray] = {}
    labels_upper: list[np.ndarray] = []
    labels_combined: list[np.ndarray] = []
    event_date_parts: list[np.ndarray] = []
    for year, year_dates in dates_by_year.items():
        count = len(year_dates)
        upper = np.tile(
            np.repeat(thresholds[year]["dart_upper_q99_by_zone"], 96), (count, 1)
        )
        lower = np.tile(
            np.repeat(thresholds[year]["dart_lower_q01_by_zone"], 96), (count, 1)
        )
        labels_upper.append(actual_by_year[year] > upper)
        labels_combined.append((actual_by_year[year] > upper) | (actual_by_year[year] < lower))
        event_date_parts.append(np.repeat(np.asarray(year_dates), EVENTS_PER_DAY))
    pooled_upper_labels = np.concatenate([value.reshape(-1) for value in labels_upper])
    pooled_combined_labels = np.concatenate([value.reshape(-1) for value in labels_combined])
    pooled_event_dates = np.concatenate(event_date_parts)

    for candidate in candidates:
        year_streams: list[dict[str, np.ndarray]] = []
        unique_fit_count = 0
        source_artifacts[candidate] = []
        for year, year_dates in dates_by_year.items():
            origin = f"through_{year - 1}_to_{year}_oos"
            paths = sorted((forecast_root / "dart" / origin / candidate).glob("*/dart_draws.npz"))
            expected = 3 if candidate not in {"seasonal_naive", "persistence", "lightgbm_quantile"} else 1
            if len(paths) != expected:
                raise ConditionalForecastMetricError(
                    f"{candidate}/{year} has {len(paths)} unique forecast fits, expected {expected}"
                )
            streams: list[dict[str, np.ndarray]] = []
            for path in paths:
                stream, source = _artifact_stream(
                    path,
                    repository_root,
                    year_dates,
                    actual_by_year[year],
                    thresholds[year],
                )
                streams.append(stream)
                source_artifacts[candidate].append({"forecast_year": year, **source})
            unique_fit_count += len(streams)
            averaged = _mean_streams(streams)
            year_streams.append(averaged)
            quantile_rates = np.mean(averaged["quantile_below"], axis=1)
            for day_index, day in enumerate(year_dates):
                daily_rows.append(
                    {
                        "candidate": candidate,
                        "operating_date": day,
                        "forecast_year": year,
                        "unique_forecast_fit_count": len(streams),
                        "logical_replay_seed_count": 3,
                        "mae": float(np.mean(averaged["absolute_error"][day_index])),
                        "mse": float(np.mean(averaged["squared_error"][day_index])),
                        "pinball_loss": float(np.mean(averaged["pinball"][day_index])),
                        "crps_from_quantiles": float(np.mean(averaged["crps"][day_index])),
                        "proper_tail_twcrps_from_quantiles": float(np.mean(averaged["proper_tail_score"][day_index])),
                        "coverage_80": float(np.mean(averaged["covered_80"][day_index])),
                        "width_80": float(np.mean(averaged["width_80"][day_index])),
                        "coverage_90": float(np.mean(averaged["covered_90"][day_index])),
                        "width_90": float(np.mean(averaged["width_90"][day_index])),
                        "quantile_calibration_mae": float(np.mean(np.abs(quantile_rates[day_index] - np.asarray(QUANTILES)))),
                        "final_quantile_crossing_rate": float(averaged["crossing_count"][day_index] / (EVENTS_PER_DAY * 4)),
                    }
                )
        combined = {
            name: np.concatenate([stream[name] for stream in year_streams], axis=0)
            for name in year_streams[0]
        }
        upper_probability = combined["upper_probability"].reshape(-1)
        combined_probability = combined["combined_probability"].reshape(-1)
        pit = combined["pit"].reshape(-1)
        upper_class = _classification(pooled_upper_labels, upper_probability)
        combined_class = _classification(pooled_combined_labels, combined_probability)
        q_rates = np.mean(combined["quantile_below"], axis=(0, 1))
        covered80 = float(np.mean(combined["covered_80"]))
        covered90 = float(np.mean(combined["covered_90"]))
        row = {
            "candidate": candidate,
            "operating_date_count": len(dates),
            "event_count": int(pooled_upper_labels.size),
            "logical_replay_seed_count": 3,
            "unique_forecast_fit_count_total_across_origins": unique_fit_count,
            "seed_aggregation": "mean_distinct_forecast_fits_within_date; deterministic_or_statistical_single_fit_reuse_not_replicated",
            "mae": float(np.mean(combined["absolute_error"])),
            "rmse": float(np.sqrt(np.mean(combined["squared_error"]))),
            "pinball_loss": float(np.mean(combined["pinball"])),
            "crps_from_quantiles": float(np.mean(combined["crps"])),
            "proper_tail_twcrps_from_quantiles": float(np.mean(combined["proper_tail_score"])),
            "coverage_80": covered80,
            "coverage_error_80": abs(covered80 - 0.80),
            "interval_width_80": float(np.mean(combined["width_80"])),
            "coverage_90": covered90,
            "coverage_error_90": abs(covered90 - 0.90),
            "interval_width_90": float(np.mean(combined["width_90"])),
            "quantile_calibration_mae": float(np.mean(np.abs(q_rates - np.asarray(QUANTILES)))),
            "final_quantile_crossing_rate": float(np.sum(combined["crossing_count"]) / (len(dates) * EVENTS_PER_DAY * 4)),
            "pit_mean": _pit_summary(pit)["mean"],
            "pit_variance": _pit_summary(pit)["variance"],
            "pit_uniform_ks_distance": _pit_summary(pit)["uniform_ks_distance"],
            "upper_q99_brier": float(np.mean((upper_probability - pooled_upper_labels.astype(float)) ** 2)),
            "upper_q99_pr_auc": _pr_auc(pooled_upper_labels, upper_probability),
            "upper_q99_precision": upper_class["precision"],
            "upper_q99_recall": upper_class["recall"],
            "upper_q99_f1": upper_class["f1"],
            "upper_q99_miss_rate": upper_class["miss_rate"],
            "combined_q01_q99_brier": float(np.mean((combined_probability - pooled_combined_labels.astype(float)) ** 2)),
            "combined_q01_q99_pr_auc": _pr_auc(pooled_combined_labels, combined_probability),
            "combined_q01_q99_precision": combined_class["precision"],
            "combined_q01_q99_recall": combined_class["recall"],
            "combined_q01_q99_f1": combined_class["f1"],
            "combined_q01_q99_miss_rate": combined_class["miss_rate"],
            "q99_probability_decision_threshold": 0.50,
            "raw_pre_rearrangement_crossing_rate": None,
            "raw_pre_rearrangement_crossing_rate_unsupported_reason": "frozen DART artifacts contain final empirical quantiles only",
        }
        candidate_rows.append(row)
        panel_arrays[f"{candidate}__upper_q99_probability"] = upper_probability.astype("float32")
        panel_arrays[f"{candidate}__combined_q01_q99_probability"] = combined_probability.astype("float32")
        panel_arrays[f"{candidate}__pit"] = pit.astype("float32")

    panel_arrays["upper_q99_labels"] = pooled_upper_labels.astype("uint8")
    panel_arrays["combined_q01_q99_labels"] = pooled_combined_labels.astype("uint8")
    panel_arrays["event_dates"] = pooled_event_dates.astype("U10")
    panel_path = output_root / "pooled_probability_panels.npz"
    np.savez_compressed(panel_path, **panel_arrays)
    daily_path = output_root / "daily_seed_averaged_metrics.csv"
    table_path = output_root / "candidate_forecast_metrics.csv"
    _write_csv(daily_path, daily_rows)
    _write_csv(table_path, candidate_rows)

    ranking_fields = (
        "mae",
        "crps_from_quantiles",
        "proper_tail_twcrps_from_quantiles",
        "upper_q99_f1",
    )
    rankings = {
        field: [
            row["candidate"]
            for row in sorted(
                candidate_rows,
                key=lambda value: (
                    -(float(value[field]) if value[field] is not None else -np.inf)
                    if field.endswith("_f1")
                    else (float(value[field]) if value[field] is not None else np.inf)
                ),
            )
        ]
        for field in ranking_fields
    }
    manifest_core = {
        "schema_version": "dcvpp.conditional_forecast_only_metrics.v1",
        "status": "COMPLETE_FORECAST_ONLY_CONDITIONAL_PANEL",
        "terminal_status_cap": "PARTIAL_PAPER",
        "claim_scope": "forecast_quality_on_common_fully_observed_2022_2024_settlement_days_only",
        "decision_value_winner_supported": False,
        "training_launched": False,
        "milp_solve_launched": False,
        "market_replay_launched": False,
        "locked_2025_opened": False,
        "selection_panel": {
            "full_calendar_day_count": 1096,
            "valid_day_count": len(dates),
            "invalid_missing_settlement_day_count": 10,
            "event_count": int(pooled_upper_labels.size),
            "date_sha256": sha256_json(dates),
            "settlement_index_path": settlement_index.relative_to(repository_root).as_posix(),
            "settlement_index_sha256": sha256_file(settlement_index),
        },
        "candidate_contract": {
            "candidate_count": len(candidates),
            "eligible_candidates": list(candidates),
            "globally_excluded_candidate": "itransformer",
            "candidate_pool_contract_sha256": pool["contract_sha256"],
            "replay_seeds": list(REPLAY_SEEDS),
            "date_is_sampling_unit": True,
            "deterministic_statistical_single_fit_pseudoreplication_forbidden": True,
        },
        "metric_contract": {
            "canonical_target": "DART=DA_LBMP-RT_LBMP",
            "quantile_grid": list(QUANTILES),
            "crps": "twice_mean_pinball_on_frozen_five_quantile_grid",
            "proper_tail_score": "proper_q05_q95_plus_q01_q99_threshold_weighted_crps_from_quantiles",
            "threshold_scope": "per_forecast_year_and_zone_pre_origin_history_only",
            "upper_q99_event": "actual_DART_above_training_zone_q99",
            "combined_extreme_event": "actual_DART_below_training_zone_q01_or_above_training_zone_q99",
            "event_probability_threshold": 0.50,
            "candidate_ranking_not_a_decision_value_selection": True,
        },
        "thresholds_by_year": thresholds,
        "candidate_metrics": candidate_rows,
        "diagnostic_rankings": rankings,
        "source_forecast_artifacts": source_artifacts,
        "outputs": {
            "candidate_table": {"path": table_path.relative_to(repository_root).as_posix(), "sha256": sha256_file(table_path)},
            "daily_table": {"path": daily_path.relative_to(repository_root).as_posix(), "sha256": sha256_file(daily_path)},
            "pooled_probability_panels": {"path": panel_path.relative_to(repository_root).as_posix(), "sha256": sha256_file(panel_path)},
        },
        "unsupported_metrics": {
            "raw_pre_rearrangement_crossing_rate": "frozen DART artifacts expose only final empirical quantiles",
            "action_metrics": "requires no-action bands and action construction; outside forecast-only scope",
            "decision_value_MCS_LCB": "requires successful common MILP market replay; not inferred from forecast metrics",
        },
        "selection_manifest": {
            "path": selection_manifest_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(selection_manifest_path),
        },
    }
    manifest = {
        **manifest_core,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_fingerprint": sha256_json(manifest_core),
    }
    manifest_path = output_root / "conditional_forecast_metrics_manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "evidence_fingerprint": manifest["evidence_fingerprint"],
        "candidate_rows": candidate_rows,
    }


__all__ = ["ConditionalForecastMetricError", "build_conditional_forecast_metrics"]
