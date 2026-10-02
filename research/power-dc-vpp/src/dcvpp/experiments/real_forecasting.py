from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dcvpp.core import atomic_json, sha256_file, sha256_json, utc_now
from dcvpp.experiments.forecast_common import fit_and_persist_task
from dcvpp.experiments.forecast_day_factory import build_frozen_forecast_day_index
from dcvpp.experiments.nlr_workload import (
    build_nlr_15m_from_processed_events,
    build_nlr_15m_workload_panel,
)
from dcvpp.experiments.torch_forecasts import (
    fit_cfc_w_five_seeds,
    fit_dgpt_e_five_seeds,
)
from dcvpp.experiments.weather_pv import build_weather_and_pv_15m_panels


NYISO_WEATHER_COLUMNS = (
    "T2M",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
)
NYISO_ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_parquet_files(paths: Sequence[Path], columns: Sequence[str]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("no referenced Parquet inputs matched")
    frames = [pq.ParquetFile(path).read(columns=list(columns)).to_pandas() for path in paths]
    return pd.concat(frames, ignore_index=True)


def load_referenced_nyiso(
    repository_root: Path,
    *,
    product: str,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if product not in {"da_lbmp", "rt_lbmp"}:
        raise ValueError(f"unsupported forecast product: {product}")
    root = data_root or repository_root / "data/processed/referenced/nyiso"
    paths = sorted((root / f"product={product}").rglob("*.parquet"))
    target = product
    frame = _read_parquet_files(
        paths,
        columns=[
            "timestamp_utc",
            "timestamp_local",
            "zone",
            target,
            "archive_sha256",
            "source_member_sha256",
        ],
    )
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"], errors="raise")
    frame[target] = pd.to_numeric(frame[target], errors="raise")
    raw_rows = len(frame)
    off_cycle_rows = 0
    if product == "rt_lbmp":
        nominal = (
            frame["timestamp_utc"].dt.minute.mod(5).eq(0)
            & frame["timestamp_utc"].dt.second.eq(0)
            & frame["timestamp_utc"].dt.microsecond.eq(0)
        )
        off_cycle_rows = int((~nominal).sum())
        frame = frame.loc[nominal].copy()
    if frame.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError(f"{product} has duplicate zone-timestamps after resolution gate")
    if set(frame["zone"].unique()) != set(NYISO_ZONES):
        raise ValueError(f"{product} does not contain the four frozen NYISO zones")
    frame = frame.sort_values(["zone", "timestamp_utc"], kind="stable").reset_index(drop=True)
    audit = {
        "product": product,
        "input_partition_count": len(paths),
        "input_rows": raw_rows,
        "nominal_rows": len(frame),
        "off_cycle_rows_excluded": off_cycle_rows,
        "negative_price_rows": int(frame[target].lt(0).sum()),
        "coverage_min_utc": frame["timestamp_utc"].min().isoformat(),
        "coverage_max_utc": frame["timestamp_utc"].max().isoformat(),
        "source_archive_sha256_count": int(frame["archive_sha256"].nunique()),
        "source_relationship": "third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
    }
    return frame, audit


def load_referenced_nasa(
    repository_root: Path,
    *,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = data_root or repository_root / "data/processed/referenced/nasa_power"
    paths = sorted(root.rglob("*.parquet"))
    frame = _read_parquet_files(
        paths,
        columns=[
            "timestamp_utc",
            "zone",
            "latitude",
            "longitude",
            *NYISO_WEATHER_COLUMNS,
            "source_sha256",
        ],
    )
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("NASA POWER referenced panel has duplicate site-hours")
    if set(frame["zone"].unique()) != set(NYISO_ZONES):
        raise ValueError("NASA POWER referenced panel does not contain the frozen four sites")
    audit = {
        "input_partition_count": len(paths),
        "input_rows": len(frame),
        "coverage_min_utc": frame["timestamp_utc"].min().isoformat(),
        "coverage_max_utc": frame["timestamp_utc"].max().isoformat(),
        "source_sha256_count": int(frame["source_sha256"].nunique()),
        "evidence_role": "reconstructed",
        "source_relationship": "third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
        "operational_weather_forecast_claimed": False,
    }
    return frame, audit


def _exact_group_lag(
    frame: pd.DataFrame,
    *,
    value_column: str,
    steps: int,
    interval: pd.Timedelta,
    output_column: str,
) -> tuple[pd.Series, pd.Series]:
    expected_delta = interval * steps
    lookup_index = pd.MultiIndex.from_frame(frame[["zone", "timestamp_utc"]])
    if lookup_index.has_duplicates:
        raise ValueError(f"{value_column} exact-lag lookup has duplicate zone timestamps")
    expected_timestamp = frame["timestamp_utc"] - expected_delta
    requested_index = pd.MultiIndex.from_arrays(
        [frame["zone"].to_numpy(), expected_timestamp.to_numpy()],
        names=["zone", "timestamp_utc"],
    )
    lookup = pd.Series(frame[value_column].to_numpy(), index=lookup_index)
    values = lookup.reindex(requested_index).to_numpy()
    exists = requested_index.isin(lookup_index)
    source_timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    source_timestamp.loc[exists] = expected_timestamp.loc[exists].to_numpy()
    # Timestamp-keyed lookup recovers immediately after an isolated source
    # gap; row-position shift would offset every subsequent lag until another
    # missing row happened to realign it.
    return pd.Series(values, index=frame.index), source_timestamp


def _calendar_features(frame: pd.DataFrame, local_column: str) -> None:
    local = pd.to_datetime(frame[local_column], errors="raise")
    hour = local.dt.hour + local.dt.minute / 60.0
    day_of_week = local.dt.dayofweek.astype("float64")
    day_of_year = local.dt.dayofyear.astype("float64")
    frame["calendar_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["calendar_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["calendar_dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    frame["calendar_dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    frame["calendar_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["calendar_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)


def build_nyiso_issue_safe_features(
    price_rows: pd.DataFrame,
    weather_rows: pd.DataFrame,
    *,
    product: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if product == "da_lbmp":
        interval = pd.Timedelta(hours=1)
        horizon = pd.Timedelta(hours=24)
        lag_steps = {"price_lag_24h": 24, "price_lag_48h": 48, "price_lag_168h": 168}
        persistence_column = "price_lag_24h"
        seasonal_column = "price_lag_168h"
    elif product == "rt_lbmp":
        interval = pd.Timedelta(minutes=5)
        horizon = pd.Timedelta(minutes=5)
        lag_steps = {
            "price_lag_5m": 1,
            "price_lag_15m": 3,
            "price_lag_1h": 12,
            "price_lag_24h": 288,
            "price_lag_168h": 2016,
        }
        persistence_column = "price_lag_5m"
        seasonal_column = "price_lag_168h"
    else:
        raise ValueError(f"unsupported product: {product}")
    frame = price_rows.copy().sort_values(["zone", "timestamp_utc"], kind="stable")
    frame["prediction_origin_utc"] = frame["timestamp_utc"] - horizon
    lag_source_columns: dict[str, str] = {}
    for output_column, steps in lag_steps.items():
        values, timestamps = _exact_group_lag(
            frame,
            value_column=product,
            steps=steps,
            interval=interval,
            output_column=output_column,
        )
        frame[output_column] = values
        timestamp_column = f"{output_column}_source_timestamp_utc"
        frame[timestamp_column] = timestamps
        lag_source_columns[output_column] = timestamp_column
    frame["baseline_persistence"] = frame[persistence_column]
    frame["baseline_seasonal"] = frame[seasonal_column]
    frame["latest_target_feature_timestamp_utc"] = frame[
        lag_source_columns[persistence_column]
    ]

    frame["weather_feature_timestamp_utc"] = (
        frame["prediction_origin_utc"].dt.floor("h") - pd.Timedelta(hours=24)
    )
    weather = weather_rows[["timestamp_utc", "zone", *NYISO_WEATHER_COLUMNS]].copy()
    weather = weather.rename(columns={"timestamp_utc": "weather_feature_timestamp_utc"})
    weather = weather.rename(
        columns={column: f"weather_{column}_lagged" for column in NYISO_WEATHER_COLUMNS}
    )
    frame = frame.merge(
        weather,
        on=["zone", "weather_feature_timestamp_utc"],
        how="left",
        validate="many_to_one",
    )
    _calendar_features(frame, "timestamp_local")
    for zone in NYISO_ZONES:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", zone).strip("_")
        frame[f"zone_{slug}"] = frame["zone"].eq(zone).astype("int8")

    market_year = frame["timestamp_local"].dt.year
    frame["split"] = np.select(
        [market_year.between(2021, 2023), market_year.eq(2024), market_year.eq(2025)],
        ["train", "validation", "locked_test"],
        default="outside_scope",
    )
    frame["feature_data_cutoff_utc"] = frame["prediction_origin_utc"]
    weather_features = [f"weather_{column}_lagged" for column in NYISO_WEATHER_COLUMNS]
    calendar_features = [
        "calendar_hour_sin",
        "calendar_hour_cos",
        "calendar_dow_sin",
        "calendar_dow_cos",
        "calendar_year_sin",
        "calendar_year_cos",
    ]
    zone_features = [
        f"zone_{re.sub(r'[^A-Za-z0-9]+', '_', zone).strip('_')}" for zone in NYISO_ZONES
    ]
    model_features = [*lag_steps, *weather_features, *calendar_features, *zone_features]
    required = [product, "baseline_persistence", "baseline_seasonal", *model_features]
    scoped = frame["split"].ne("outside_scope")
    eligible = scoped & frame[required].notna().all(axis=1)
    eligible &= np.isfinite(frame[required].to_numpy(dtype="float64")).all(axis=1)
    result = frame.loc[eligible].copy().reset_index(drop=True)

    if (result["latest_target_feature_timestamp_utc"] > result["prediction_origin_utc"]).any():
        raise ValueError(f"{product} target lag crosses the prediction origin")
    if (result["weather_feature_timestamp_utc"] > result["prediction_origin_utc"]).any():
        raise ValueError(f"{product} weather feature crosses the prediction origin")
    split_counts = result.groupby("split").size().to_dict()
    if set(split_counts) != {"train", "validation", "locked_test"}:
        raise ValueError(f"{product} issue-safe feature gate empties a required split")
    audit = {
        "product": product,
        "forecast_horizon_minutes": int(horizon.total_seconds() // 60),
        "issue_rule": "prediction_origin_utc = target timestamp minus forecast horizon",
        "weather_rule": "NASA POWER reconstruction timestamp is at least 24 hours before origin",
        "price_lag_rule": "exact UTC interval match; gaps are not row-shift imputed",
        "source_rows": len(frame),
        "eligible_rows": len(result),
        "ineligible_or_outside_scope_rows": int(len(frame) - len(result)),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "max_target_feature_lead_seconds": float(
            (result["latest_target_feature_timestamp_utc"] - result["prediction_origin_utc"])
            .dt.total_seconds()
            .max()
        ),
        "max_weather_feature_lead_seconds": float(
            (result["weather_feature_timestamp_utc"] - result["prediction_origin_utc"])
            .dt.total_seconds()
            .max()
        ),
    }
    return result, model_features, audit


def build_nlr_issue_safe_features(
    workload_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frame = workload_panel.copy().sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame["timestamp_utc"].diff().dropna().ne(pd.Timedelta(minutes=15)).any():
        raise ValueError("NLR workload panel is not a complete 15-minute UTC grid")
    target = "submitted_nodes_requested"
    frame["prediction_origin_utc"] = frame["timestamp_utc"] - pd.Timedelta(minutes=15)
    lag_steps = {"nodes_lag_15m": 1, "nodes_lag_24h": 96, "nodes_lag_168h": 672}
    for column, steps in lag_steps.items():
        frame[column] = frame[target].shift(steps)
    frame["baseline_persistence"] = frame["nodes_lag_15m"]
    frame["baseline_seasonal"] = frame["nodes_lag_168h"]
    frame["latest_target_feature_timestamp_utc"] = frame["timestamp_utc"].shift(1)
    frame["feature_data_cutoff_utc"] = frame["prediction_origin_utc"]
    frame["timestamp_local"] = frame["timestamp_utc"].dt.tz_convert("America/Denver")
    _calendar_features(frame, "timestamp_local")
    n_rows = len(frame)
    train_end = math.floor(n_rows * 0.60)
    validation_end = math.floor(n_rows * 0.80)
    positions = np.arange(n_rows)
    frame["split"] = np.select(
        [positions < train_end, positions < validation_end],
        ["train", "validation"],
        default="locked_test",
    )
    calendar_features = [
        "calendar_hour_sin",
        "calendar_hour_cos",
        "calendar_dow_sin",
        "calendar_dow_cos",
        "calendar_year_sin",
        "calendar_year_cos",
    ]
    model_features = [*lag_steps, *calendar_features]
    required = [target, "baseline_persistence", "baseline_seasonal", *model_features]
    eligible = frame[required].notna().all(axis=1)
    eligible &= np.isfinite(frame[required].to_numpy(dtype="float64")).all(axis=1)
    result = frame.loc[eligible].copy().reset_index(drop=True)
    if (result["latest_target_feature_timestamp_utc"] > result["prediction_origin_utc"]).any():
        raise ValueError("NLR workload lag crosses the deployable prediction origin")
    counts = result.groupby("split").size().to_dict()
    audit = {
        "target": target,
        "forecast_horizon_minutes": 15,
        "split_policy": "chronological 60/20/20 over the complete 15-minute target grid",
        "full_panel_row_count": n_rows,
        "train_end_exclusive_utc": frame.loc[train_end, "timestamp_utc"].isoformat(),
        "validation_end_exclusive_utc": frame.loc[validation_end, "timestamp_utc"].isoformat(),
        "eligible_split_counts": {str(key): int(value) for key, value in counts.items()},
        "calendar_timezone": "America/Denver",
        "feature_cutoff_rule": "all observed workload lags end at or before prediction_origin_utc",
    }
    return result, model_features, audit


def _split_record(frame: pd.DataFrame, *, target: str, policy: str) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in ("train", "validation", "locked_test"):
        selected = frame.loc[frame["split"].eq(split)]
        key_columns = [column for column in ("timestamp_utc", "zone") if column in selected]
        key_hashes = pd.util.hash_pandas_object(selected[key_columns], index=False).to_numpy()
        target_hashes = pd.util.hash_pandas_object(selected[[target]], index=False).to_numpy()
        splits[split] = {
            "row_count": len(selected),
            "target_min_timestamp_utc": selected["timestamp_utc"].min().isoformat(),
            "target_max_timestamp_utc": selected["timestamp_utc"].max().isoformat(),
            "prediction_origin_min_utc": selected["prediction_origin_utc"].min().isoformat(),
            "prediction_origin_max_utc": selected["prediction_origin_utc"].max().isoformat(),
            "key_sha256": hashlib.sha256(key_hashes.tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(target_hashes.tobytes()).hexdigest(),
        }
    return {
        "policy": policy,
        "overlap_allowed": False,
        "locked_test_used_for_model_or_seed_selection": False,
        "splits": splits,
    }


def _validate_referenced_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("referenced-data manifest is not COMPLETE")
    if manifest.get("measurement_relationship") != "not_measured_by_this_study":
        raise ValueError("referenced-data measurement boundary is missing")
    required = {"nasa_power", "nyiso", "nlr295"}
    if not required.issubset(manifest.get("sources", {})):
        raise ValueError("referenced-data manifest misses a forecasting source")
    return manifest


def _find_processed_event_record(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        path = value.get("path") or value.get("local_path")
        digest = value.get("sha256")
        if path and digest and str(path).replace("\\", "/").endswith(
            "/nlr295_workload_events.parquet"
        ):
            return {
                "path": str(path),
                "sha256": str(digest),
                "archive_sha256": value.get("archive_sha256"),
                "schema": value.get("schema"),
            }
        for nested in value.values():
            record = _find_processed_event_record(nested)
            if record is not None:
                return record
    elif isinstance(value, (list, tuple)):
        for nested in value:
            record = _find_processed_event_record(nested)
            if record is not None:
                return record
    return None


def _processed_event_record_from_config(
    repository_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    physical_result = config.get("physical_result")
    if isinstance(physical_result, Mapping):
        contract = physical_result.get("workload_events_contract")
        if isinstance(contract, Mapping):
            value = contract.get("relative_path") or contract.get("path")
            digest = contract.get("sha256")
            if value and digest:
                path = _path(repository_root, str(value))
                if not path.is_file():
                    raise FileNotFoundError(f"physical workload contract path is missing: {path}")
                actual = sha256_file(path)
                if actual != str(digest):
                    raise ValueError("physical workload contract hash mismatch")
                return {
                    "path": path.resolve().relative_to(repository_root).as_posix(),
                    "sha256": actual,
                    "archive_sha256": contract.get("source_archive_sha256"),
                    "schema": contract.get("parquet_schema"),
                    "schema_sha256": contract.get("schema_sha256"),
                    "contract_name": contract.get("contract_name"),
                    "contract_sha256": sha256_json(dict(contract)),
                }
    record = _find_processed_event_record(config.get("physical_result"))
    if record is not None:
        return record
    manifest_value = config.get("physical_run_manifest")
    if manifest_value:
        manifest_path = _path(repository_root, manifest_value)
        if manifest_path.is_file():
            return _find_processed_event_record(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
    return None


def _output_record(path: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _resume_result(
    run_manifest_path: Path,
    *,
    repository_root: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not run_manifest_path.is_file():
        return None
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("fingerprint") != fingerprint:
        return None
    for output in manifest.get("outputs", []):
        path = repository_root / output["path"]
        if not path.is_file() or sha256_file(path) != output.get("sha256"):
            return None
    return {
        "status": "COMPLETE",
        "resumed": True,
        "run_id": manifest["run_id"],
        "counts": manifest["counts"],
        "output_paths": [item["path"] for item in manifest["outputs"]],
        "run_manifest": run_manifest_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "forecast_day_index": manifest.get("forecast_day_index"),
        "energy_hypothesis_metrics": manifest.get("energy_hypothesis_metrics", []),
        "workload_hypothesis_metrics": manifest.get("workload_hypothesis_metrics", []),
        "forecast_decision_records": manifest.get("forecast_decision_records", []),
    }


def _hypothesis_records(
    task_results: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    energy: list[dict[str, Any]] = []
    for task in ("nyiso_da", "nyiso_rt"):
        result = task_results.get(task)
        if not result or "dgpt_e" not in result:
            continue
        baseline_rows = [
            row for row in result["dgpt_e"].get("baseline_results", [])
            if row.get("deployable") and row.get("locked_test_metrics")
        ]
        if not baseline_rows:
            continue
        strongest = min(
            baseline_rows,
            key=lambda row: float(row["locked_test_metrics"]["mean_weighted_pinball"]),
        )
        baseline_metrics = strongest["locked_test_metrics"]
        dgpt = result["dgpt_e"]["locked_test_metrics"]
        improvement = float(baseline_metrics["mean_weighted_pinball"]) - float(
            dgpt["mean_weighted_pinball"]
        )
        calibration_harm = float(dgpt["absolute_calibration_error"]) - float(
            baseline_metrics["absolute_calibration_error"]
        )
        tail_not_worse = float(dgpt.get("spike_f1", 0.0)) >= float(
            baseline_metrics.get("spike_f1", 0.0)
        )
        energy.append(
            {
                "hypothesis": "H1",
                "task": task,
                "advanced_model": result["dgpt_e"]["model_class"],
                "comparison_metric": "locked_test_mean_weighted_pinball",
                "advanced_value": float(dgpt["mean_weighted_pinball"]),
                "strongest_eligible_baseline": strongest["model"],
                "baseline_value": float(baseline_metrics["mean_weighted_pinball"]),
                "improvement_absolute": improvement,
                "improvement_fraction": (
                    improvement / float(baseline_metrics["mean_weighted_pinball"])
                    if float(baseline_metrics["mean_weighted_pinball"]) != 0
                    else None
                ),
                "proper_score_mean_pinball": float(dgpt["mean_weighted_pinball"]),
                "coverage_90": float(dgpt["coverage_90"]),
                "absolute_calibration_error_change": calibration_harm,
                "tail_f1_advanced": float(dgpt.get("spike_f1", 0.0)),
                "tail_f1_baseline": float(baseline_metrics.get("spike_f1", 0.0)),
                "support": bool(improvement > 0 and calibration_harm <= 0.02 and tail_not_worse),
                "evaluated": True,
                "decision_rule": (
                    "lower locked-test mean weighted pinball, absolute calibration error not worse "
                    "by more than 0.02, and spike F1 not lower than strongest deployable baseline"
                ),
            }
        )
    workload: list[dict[str, Any]] = []
    result = task_results.get("nlr_workload")
    if result and "cfc_w" in result:
        baseline_rows = [
            row for row in result["cfc_w"].get("baseline_results", [])
            if row.get("deployable") and row.get("locked_test_metrics")
        ]
        if not baseline_rows:
            return energy, workload
        strongest = min(
            baseline_rows,
            key=lambda row: float(row["locked_test_metrics"]["distribution_score"]),
        )
        baseline_metrics = strongest["locked_test_metrics"]
        cfc = result["cfc_w"]["locked_test_metrics"]
        improvement = float(baseline_metrics["distribution_score"]) - float(
            cfc["distribution_score"]
        )
        calibration_improvement = float(baseline_metrics["interval_calibration_error"]) - float(
            cfc["interval_calibration_error"]
        )
        workload.append(
            {
                "hypothesis": "H2",
                "task": "nlr_workload",
                "advanced_model": result["cfc_w"]["model_class"],
                "comparison_metric": "locked_test_count_plus_volume_distribution_score",
                "advanced_value": float(cfc["distribution_score"]),
                "strongest_eligible_baseline": strongest["model"],
                "baseline_value": float(baseline_metrics["distribution_score"]),
                "improvement_absolute": improvement,
                "improvement_fraction": (
                    improvement / float(baseline_metrics["distribution_score"])
                    if float(baseline_metrics["distribution_score"]) != 0
                    else None
                ),
                "interval_calibration_improvement": calibration_improvement,
                "support": bool(improvement > 0 and calibration_improvement > 0),
                "evaluated": True,
                "decision_rule": (
                    "lower locked-test count-plus-volume distribution score and improvement in at "
                    "least one preregistered interval-calibration aggregate"
                ),
            }
        )
    return energy, workload


def run_forecasting_branch(
    root: Path,
    config: dict,
    run_id: str,
) -> dict[str, Any]:
    """Run the real, issue-time-safe NYISO and NLR forecasting branch.

    This entrypoint never generates synthetic fallback data. A failed task is
    recorded as failed and does not inherit demo predictions or metrics.
    """
    repository_root = Path(root).resolve()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a safe 1-128 character filename token")
    selected_tasks = tuple(
        config.get("tasks", ("nyiso_da", "nyiso_rt", "nlr_workload"))
    )
    unknown = sorted(set(selected_tasks) - {"nyiso_da", "nyiso_rt", "nlr_workload"})
    if unknown:
        raise ValueError(f"unknown forecasting tasks: {unknown}")
    referenced_manifest_path = _path(
        repository_root,
        config.get(
            "referenced_manifest",
            "artifacts/manifests/referenced_data_progress_manifest.json",
        ),
    )
    referenced_manifest = _validate_referenced_manifest(referenced_manifest_path)
    source_manifest_sha256 = sha256_file(referenced_manifest_path)
    output_root = _path(
        repository_root,
        config.get("output_root", "artifacts/real_forecasting"),
    ) / run_id
    run_manifest_path = output_root / "run_manifest.json"
    code_hashes = {
        path.resolve().relative_to(repository_root).as_posix(): sha256_file(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("forecast_common.py"),
            Path(__file__).with_name("forecast_calibration.py"),
            Path(__file__).with_name("paper_training.py"),
            Path(__file__).with_name("nlr_workload.py"),
            Path(__file__).with_name("torch_forecasts.py"),
            Path(__file__).with_name("forecast_day_factory.py"),
            Path(__file__).with_name("weather_pv.py"),
            repository_root / "src/dcvpp/models/energy/dgpt_e.py",
            repository_root / "src/dcvpp/models/workload/cfc_w.py",
        )
    }
    configured_processed_event_record = _processed_event_record_from_config(
        repository_root, config
    )
    effective_config = {
        "tasks": list(selected_tasks),
        "seeds": [
            int(seed)
            for seed in config.get(
                "seeds", (20260809, 20260819, 20260829, 20260908, 20260918)
            )
        ],
        "advanced_n_estimators": int(config.get("advanced_n_estimators", 250)),
        "n_jobs": int(config.get("n_jobs", 8)),
        "max_train_rows": config.get("max_train_rows"),
        "keep_nlr_selected_jobs": bool(config.get("keep_nlr_selected_jobs", False)),
        "nlr_batch_size": int(config.get("nlr_batch_size", 200_000)),
        "torch_epochs": int(config.get("torch_epochs", 100)),
        "torch_batch_size": int(config.get("torch_batch_size", 32)),
        "torch_device": str(config.get("torch_device", "auto")),
        "history_steps": int(config.get("history_steps", 672)),
        "horizon_steps": int(config.get("horizon_steps", 96)),
        "dgpt_width": int(config.get("dgpt_width", 128)),
        "cfc_hidden_size": int(config.get("cfc_hidden_size", 64)),
        "minimum_calibration_samples": int(config.get("minimum_calibration_samples", 500)),
        "forecast_draw_count": int(config.get("forecast_draw_count", 100)),
        "processed_event_artifact": configured_processed_event_record,
        "physical_workload_contract": configured_processed_event_record,
    }
    fingerprint = sha256_json(
        {
            "run_id": run_id,
            "effective_config": effective_config,
            "source_manifest_sha256": source_manifest_sha256,
            "code_hashes": code_hashes,
        }
    )
    if bool(config.get("resume", True)):
        resumed = _resume_result(
            run_manifest_path,
            repository_root=repository_root,
            fingerprint=fingerprint,
        )
        if resumed is not None:
            return resumed

    output_root.mkdir(parents=True, exist_ok=True)
    failed_seeds: list[dict[str, Any]] = []
    task_failures: list[dict[str, Any]] = []
    task_results: dict[str, dict[str, Any]] = {}
    split_manifest: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "source_manifest_sha256": source_manifest_sha256,
        "tasks": {},
    }
    input_audits: dict[str, Any] = {}
    weather_rows: pd.DataFrame | None = None
    weather_audit: dict[str, Any] | None = None
    external_artifact_paths: list[Path] = []

    for task in selected_tasks:
        try:
            if task in {"nyiso_da", "nyiso_rt"}:
                if weather_rows is None:
                    weather_rows, weather_audit = load_referenced_nasa(
                        repository_root,
                        data_root=(
                            _path(repository_root, config["nasa_data_root"])
                            if "nasa_data_root" in config
                            else None
                        ),
                    )
                    input_audits["nasa_power"] = weather_audit
                    weather_pv = build_weather_and_pv_15m_panels(
                        weather_rows, repository_root
                    )
                    input_audits["weather_pv_15m"] = weather_pv
                    external_artifact_paths.extend(
                        [
                            Path(weather_pv["weather_artifact"]["path"]),
                            Path(weather_pv["pv_artifact"]["path"]),
                        ]
                    )
                product = "da_lbmp" if task == "nyiso_da" else "rt_lbmp"
                price_rows, price_audit = load_referenced_nyiso(
                    repository_root,
                    product=product,
                    data_root=(
                        _path(repository_root, config["nyiso_data_root"])
                        if "nyiso_data_root" in config
                        else None
                    ),
                )
                features, feature_columns, feature_audit = build_nyiso_issue_safe_features(
                    price_rows, weather_rows, product=product
                )
                input_audits[task] = {"price": price_audit, "features": feature_audit}
                split_manifest["tasks"][task] = _split_record(
                    features,
                    target=product,
                    policy="market-local years: 2021-2023 train, 2024 validation, 2025 locked test",
                )
                result = fit_and_persist_task(
                    features,
                    task=task,
                    target_column=product,
                    feature_columns=feature_columns,
                    seeds=effective_config["seeds"],
                    output_root=output_root,
                    forecast_horizon_minutes=1440 if task == "nyiso_da" else 5,
                    source_record_type="referenced_official_historical_market_record",
                    failed_seeds=failed_seeds,
                    max_train_rows=effective_config["max_train_rows"],
                    advanced_n_estimators=effective_config["advanced_n_estimators"],
                    n_jobs=effective_config["n_jobs"],
                )
                result["dgpt_e"] = fit_dgpt_e_five_seeds(
                    price_rows,
                    product=product,
                    task=task,
                    seeds=effective_config["seeds"],
                    output_root=output_root,
                    failed_seeds=failed_seeds,
                    epochs=effective_config["torch_epochs"],
                    batch_size=effective_config["torch_batch_size"],
                    width=effective_config["dgpt_width"],
                    device_name=effective_config["torch_device"],
                    history=effective_config["history_steps"],
                    horizon=effective_config["horizon_steps"],
                    resume_fingerprint=sha256_json(
                        {"forecast_branch": fingerprint, "task": task}
                    ),
                    minimum_calibration_samples=effective_config["minimum_calibration_samples"],
                    n_estimators=effective_config["advanced_n_estimators"],
                    n_jobs=effective_config["n_jobs"],
                )
                task_results[task] = result
                del price_rows, features
            else:
                nlr_record = referenced_manifest["sources"]["nlr295"]["records"][0]
                nlr_archive = (
                    _path(repository_root, config["nlr_archive"])
                    if "nlr_archive" in config
                    else repository_root / nlr_record["source"]["local_path"]
                )
                processed_record = configured_processed_event_record
                expected_archive_sha = str(nlr_record["source"]["sha256"])
                if (
                    processed_record is not None
                    and (
                        not processed_record.get("archive_sha256")
                        or str(processed_record["archive_sha256"]) == expected_archive_sha
                    )
                ):
                    try:
                        hourly_panel, panel_audit, panel_artifact = (
                            build_nlr_15m_from_processed_events(
                                _path(repository_root, processed_record["path"]),
                                output_root,
                                archive_sha256=expected_archive_sha,
                                declared_sha256=str(processed_record["sha256"]),
                                duckdb_threads=effective_config["n_jobs"],
                            )
                        )
                    except (FileNotFoundError, ValueError) as reuse_error:
                        hourly_panel, panel_audit, panel_artifact = build_nlr_15m_workload_panel(
                            nlr_archive,
                            output_root,
                            archive_sha256=expected_archive_sha,
                            batch_size=effective_config["nlr_batch_size"],
                            duckdb_threads=effective_config["n_jobs"],
                            keep_selected_jobs=effective_config["keep_nlr_selected_jobs"],
                        )
                        panel_audit["processed_event_reuse_fallback_reason"] = str(reuse_error)
                else:
                    hourly_panel, panel_audit, panel_artifact = build_nlr_15m_workload_panel(
                        nlr_archive,
                        output_root,
                        archive_sha256=expected_archive_sha,
                        batch_size=effective_config["nlr_batch_size"],
                        duckdb_threads=effective_config["n_jobs"],
                        keep_selected_jobs=effective_config["keep_nlr_selected_jobs"],
                    )
                features, feature_columns, feature_audit = build_nlr_issue_safe_features(
                    hourly_panel
                )
                input_audits[task] = {
                    "panel_build": panel_audit,
                    "features": feature_audit,
                    "panel_artifact": panel_artifact,
                }
                split_manifest["tasks"][task] = _split_record(
                    features,
                    target="submitted_nodes_requested",
                    policy="chronological 60/20/20 over the complete NLR295 15-minute grid",
                )
                result = fit_and_persist_task(
                    features,
                    task=task,
                    target_column="submitted_nodes_requested",
                    feature_columns=feature_columns,
                    seeds=effective_config["seeds"],
                    output_root=output_root,
                    forecast_horizon_minutes=15,
                    source_record_type="referenced_publisher_trace_replay",
                    failed_seeds=failed_seeds,
                    max_train_rows=effective_config["max_train_rows"],
                    advanced_n_estimators=effective_config["advanced_n_estimators"],
                    n_jobs=effective_config["n_jobs"],
                )
                result["cfc_w"] = fit_cfc_w_five_seeds(
                    hourly_panel,
                    seeds=effective_config["seeds"],
                    output_root=output_root,
                    failed_seeds=failed_seeds,
                    epochs=effective_config["torch_epochs"],
                    batch_size=effective_config["torch_batch_size"],
                    hidden_size=effective_config["cfc_hidden_size"],
                    device_name=effective_config["torch_device"],
                    history=effective_config["history_steps"],
                    horizon=effective_config["horizon_steps"],
                    resume_fingerprint=sha256_json(
                        {
                            "forecast_branch": fingerprint,
                            "task": task,
                            "physical_workload_contract": configured_processed_event_record,
                        }
                    ),
                    minimum_calibration_samples=effective_config["minimum_calibration_samples"],
                    event_artifact_path=(
                        _path(repository_root, configured_processed_event_record["path"])
                        if configured_processed_event_record is not None
                        else None
                    ),
                    event_artifact_sha256=(
                        configured_processed_event_record["sha256"]
                        if configured_processed_event_record is not None
                        else None
                    ),
                )
                result["hourly_panel_artifact"] = panel_artifact
                task_results[task] = result
                del hourly_panel, features
        except Exception as exc:
            task_failures.append(
                {
                    "task": task,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recorded_at_utc": utc_now(),
                }
            )
        atomic_json(output_root / "manifests/failed_seeds.json", failed_seeds)

    atomic_json(output_root / "manifests/split_manifest.json", split_manifest)
    model_manifest = {
        "generated_at_utc": utc_now(),
        "advanced_model_classes": [
            "dcvpp.models.energy.dgpt_e.DGPTE",
            "dcvpp.models.workload.cfc_w.CfCW",
        ],
        "energy_baseline_models": [
            "persistence", "seasonal_naive", "lightgbm_quantile", "lstm",
            "dlinear", "patchtst", "itransformer", "static_graph_patch_transformer",
        ],
        "workload_baseline_models": [
            "seasonal_poisson", "seasonal_negative_binomial",
            "negative_binomial_regression", "lightgbm_count_volume", "gru",
            "aggregate_patchtst",
        ],
        "selection_split": "validation",
        "locked_test_used_for_model_or_seed_selection": False,
        "tasks": {
            task: {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "validation_metrics",
                    "locked_test_metrics",
                    "prediction_artifact",
                    "model_artifact",
                    "hourly_panel_artifact",
                }
            }
            for task, result in task_results.items()
        },
        "failed_seeds": failed_seeds,
    }
    atomic_json(output_root / "manifests/model_manifest.json", model_manifest)
    metrics = {
        "generated_at_utc": utc_now(),
        "selection_split": "validation",
        "locked_test_used_for_selection": False,
        "tasks": {
            task: {
                "validation": result["validation_metrics"],
                "locked_test": result["locked_test_metrics"],
                **(
                    {
                        "dgpt_e": {
                            "validation": result["dgpt_e"]["validation_metrics"],
                            "locked_test": result["dgpt_e"]["locked_test_metrics"],
                            "h1_evaluable": bool(result["dgpt_e"].get("baseline_results")),
                        }
                    }
                    if "dgpt_e" in result
                    else {}
                ),
                **(
                    {
                        "cfc_w": {
                            "validation": result["cfc_w"]["validation_metrics"],
                            "locked_test": result["cfc_w"]["locked_test_metrics"],
                            "h2_evaluable": bool(result["cfc_w"].get("baseline_results")),
                        }
                    }
                    if "cfc_w" in result
                    else {}
                ),
            }
            for task, result in task_results.items()
        },
    }
    atomic_json(output_root / "metrics/metrics.json", metrics)
    atomic_json(output_root / "manifests/input_audits.json", input_audits)

    forecast_day_index: dict[str, Any] | None = None
    if {"nyiso_da", "nyiso_rt", "nlr_workload"}.issubset(task_results):
        pv_panel_path = repository_root / "data/processed/nyiso_zone_pv_15m.parquet"
        forecast_day_index = build_frozen_forecast_day_index(
            repository_root=repository_root,
            output_root=output_root,
            da_prediction_path=Path(
                task_results["nyiso_da"]["dgpt_e"]["prediction_artifact"]["path"]
            ),
            rt_prediction_path=Path(
                task_results["nyiso_rt"]["dgpt_e"]["prediction_artifact"]["path"]
            ),
            workload_prediction_path=Path(
                task_results["nlr_workload"]["cfc_w"]["prediction_artifact"]["path"]
            ),
            pv_panel_path=pv_panel_path,
            draw_count=effective_config["forecast_draw_count"],
            physical_workload_contract=configured_processed_event_record,
        )

    energy_hypothesis_metrics, workload_hypothesis_metrics = _hypothesis_records(
        task_results
    )
    forecast_decision_records = (
        [
            {
                "forecast_day_index": forecast_day_index["index_path"],
                "day_count": forecast_day_index["day_count"],
                "draw_count": forecast_day_index["draw_count"],
                "horizon_intervals": 96,
                "status": forecast_day_index["status"],
            }
        ]
        if forecast_day_index is not None
        else []
    )

    completed = len(task_results)
    failed = len(task_failures)
    status = "COMPLETE" if completed == len(selected_tasks) else ("PARTIAL" if completed else "FAILED")
    generated_paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path != run_manifest_path
    )
    generated_paths = sorted(set(generated_paths + external_artifact_paths))
    outputs = [_output_record(path, repository_root) for path in generated_paths]
    run_manifest = {
        "run_id": run_id,
        "status": status,
        "fingerprint": fingerprint,
        "started_from_real_referenced_data": True,
        "synthetic_fallback_used": False,
        "paper_experiment_claimed": bool(status in {"COMPLETE", "PARTIAL"} and completed > 0),
        "paper_results_available": bool(completed > 0),
        "source_relationship": "third_party_referenced_public_data",
        "measurement_relationship": "not_measured_by_this_study",
        "source_manifest": {
            "path": referenced_manifest_path.resolve().relative_to(repository_root).as_posix(),
            "sha256": source_manifest_sha256,
        },
        "code_hashes": code_hashes,
        "effective_config": effective_config,
        "counts": {
            "tasks_requested": len(selected_tasks),
            "tasks_completed": completed,
            "tasks_failed": failed,
            "failed_seeds": len(failed_seeds),
        },
        "task_failures": task_failures,
        "forecast_day_index": forecast_day_index,
        "energy_hypothesis_metrics": energy_hypothesis_metrics,
        "workload_hypothesis_metrics": workload_hypothesis_metrics,
        "forecast_decision_records": forecast_decision_records,
        "outputs": outputs,
        "completed_at_utc": utc_now(),
    }
    atomic_json(run_manifest_path, run_manifest)
    return {
        "status": status,
        "resumed": False,
        "run_id": run_id,
        "counts": run_manifest["counts"],
        "output_paths": [item["path"] for item in outputs],
        "run_manifest": run_manifest_path.resolve().relative_to(repository_root).as_posix(),
        "forecast_day_index": forecast_day_index,
        "energy_hypothesis_metrics": energy_hypothesis_metrics,
        "workload_hypothesis_metrics": workload_hypothesis_metrics,
        "forecast_decision_records": forecast_decision_records,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Import run_forecasting_branch(root, config, run_id); this module has no implicit default run."
    )
