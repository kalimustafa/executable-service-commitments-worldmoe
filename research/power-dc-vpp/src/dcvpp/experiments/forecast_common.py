from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dcvpp.core import sha256_file, utc_now


def atomic_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    for key, value in metadata.items():
        schema_metadata[f"dcvpp.{key}".encode()] = str(value).encode()
    table = table.replace_schema_metadata(schema_metadata)
    with tempfile.NamedTemporaryFile("wb", suffix=".parquet", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="zstd", write_statistics=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "row_count": len(frame),
    }


def atomic_joblib(value: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".joblib", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if not len(actual):
        raise ValueError("cannot score an empty finite prediction set")
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    total_variance = float(np.square(actual - actual.mean()).sum())
    residual_variance = float(np.square(error).sum())
    return {
        "n": int(len(actual)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(math.sqrt(np.square(error).mean())),
        "mean_error": float(error.mean()),
        "wape": float(np.abs(error).sum() / denominator) if denominator else float("nan"),
        "r2": float(1.0 - residual_variance / total_variance) if total_variance else float("nan"),
    }


def _bounded_rows(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame
    indices = np.linspace(0, len(frame) - 1, num=limit, dtype="int64")
    return frame.iloc[np.unique(indices)]


def _lightgbm_model(seed: int, *, n_estimators: int, n_jobs: int) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("real advanced forecasting requires installed lightgbm") from exc
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=int(seed),
        n_jobs=n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _metrics_rows(
    *,
    task: str,
    split: str,
    seed: int,
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, prediction in predictions.items():
        rows.append(
            {
                "task": task,
                "split": split,
                "model": model_name,
                "selected_advanced_seed": seed,
                **regression_metrics(actual, prediction),
            }
        )
    return rows


def fit_and_persist_task(
    frame: pd.DataFrame,
    *,
    task: str,
    target_column: str,
    feature_columns: Sequence[str],
    seeds: Sequence[int],
    output_root: Path,
    forecast_horizon_minutes: int,
    source_record_type: str,
    failed_seeds: list[dict[str, Any]],
    max_train_rows: int | None = None,
    advanced_n_estimators: int = 250,
    n_jobs: int = 8,
) -> dict[str, Any]:
    """Select on validation only, then refit once before locked-test scoring."""
    required_splits = {"train", "validation", "locked_test"}
    if set(frame["split"].unique()) != required_splits:
        raise ValueError(f"{task} requires exactly {sorted(required_splits)}")
    if not seeds:
        raise ValueError(f"{task} has no advanced-model seeds")
    finite = np.isfinite(frame[list(feature_columns)].to_numpy(dtype="float64")).all(axis=1)
    finite &= np.isfinite(frame[target_column].to_numpy(dtype="float64"))
    if not bool(finite.all()):
        raise ValueError(f"{task} contains non-finite model rows after feature eligibility gate")

    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("validation")].copy()
    locked_test = frame.loc[frame["split"].eq("locked_test")].copy()
    if min(len(train), len(validation), len(locked_test)) == 0:
        raise ValueError(f"{task} has an empty chronological split")
    fit_train = _bounded_rows(train, max_train_rows)
    x_train = fit_train[list(feature_columns)]
    y_train = fit_train[target_column].to_numpy(dtype="float64")
    x_validation = validation[list(feature_columns)]
    y_validation = validation[target_column].to_numpy(dtype="float64")

    ridge_validation_model = make_pipeline(
        StandardScaler(), Ridge(alpha=10.0, solver="lsqr")
    )
    ridge_validation_model.fit(x_train, y_train)
    ridge_validation = ridge_validation_model.predict(x_validation)

    seed_trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for seed in seeds:
        try:
            import lightgbm as lgb

            model = _lightgbm_model(
                int(seed), n_estimators=advanced_n_estimators, n_jobs=n_jobs
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_validation, y_validation)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            prediction = model.predict(x_validation)
            metrics = regression_metrics(y_validation, prediction)
            trial = {
                "seed": int(seed),
                "status": "completed",
                "best_iteration": int(model.best_iteration_ or advanced_n_estimators),
                "validation_metrics": metrics,
            }
            seed_trials.append(trial)
            if best is None or float(metrics["rmse"]) < float(best["metrics"]["rmse"]):
                best = {
                    "seed": int(seed),
                    "model": model,
                    "prediction": prediction,
                    "metrics": metrics,
                    "best_iteration": trial["best_iteration"],
                }
        except Exception as exc:
            failure = {
                "task": task,
                "seed": int(seed),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at_utc": utc_now(),
            }
            failed_seeds.append(failure)
            seed_trials.append({"seed": int(seed), "status": "failed", **failure})
    if best is None:
        raise RuntimeError(f"all advanced-model seeds failed for {task}")

    selected_seed = int(best["seed"])
    validation_predictions = {
        "baseline_persistence": validation["baseline_persistence"].to_numpy(dtype="float64"),
        "baseline_seasonal": validation["baseline_seasonal"].to_numpy(dtype="float64"),
        "ridge_autoregression": ridge_validation,
        "advanced_lightgbm": np.asarray(best["prediction"], dtype="float64"),
    }
    validation_metrics = _metrics_rows(
        task=task,
        split="validation",
        seed=selected_seed,
        actual=y_validation,
        predictions=validation_predictions,
    )

    # The selected seed and iteration count are frozen before any locked-test
    # target is passed to a scorer or model-selection calculation.
    selection_frozen_at = utc_now()
    refit_source = pd.concat([train, validation], ignore_index=True)
    refit_source = _bounded_rows(refit_source, max_train_rows)
    x_refit = refit_source[list(feature_columns)]
    y_refit = refit_source[target_column].to_numpy(dtype="float64")
    final_advanced = _lightgbm_model(
        selected_seed,
        n_estimators=max(1, int(best["best_iteration"])),
        n_jobs=n_jobs,
    )
    final_advanced.fit(x_refit, y_refit)
    final_ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr"))
    final_ridge.fit(x_refit, y_refit)

    x_test = locked_test[list(feature_columns)]
    y_test = locked_test[target_column].to_numpy(dtype="float64")
    test_predictions = {
        "baseline_persistence": locked_test["baseline_persistence"].to_numpy(dtype="float64"),
        "baseline_seasonal": locked_test["baseline_seasonal"].to_numpy(dtype="float64"),
        "ridge_autoregression": final_ridge.predict(x_test),
        "advanced_lightgbm": final_advanced.predict(x_test),
    }
    test_metrics = _metrics_rows(
        task=task,
        split="locked_test",
        seed=selected_seed,
        actual=y_test,
        predictions=test_predictions,
    )

    def prediction_frame(
        source: pd.DataFrame,
        predictions: Mapping[str, np.ndarray],
        *,
        training_scope: str,
    ) -> pd.DataFrame:
        identity = [
            column
            for column in (
                "timestamp_utc",
                "timestamp_local",
                "zone",
                "split",
                "prediction_origin_utc",
                "feature_data_cutoff_utc",
                "latest_target_feature_timestamp_utc",
                "weather_feature_timestamp_utc",
            )
            if column in source.columns
        ]
        result = source[identity].copy()
        result.insert(0, "task", task)
        result["forecast_horizon_minutes"] = forecast_horizon_minutes
        result["observed_target"] = source[target_column].to_numpy(dtype="float64")
        for name, values in predictions.items():
            result[name] = np.asarray(values, dtype="float64")
        result["selected_advanced_seed"] = selected_seed
        result["training_scope"] = training_scope
        result["target_evidence_role"] = "observed"
        result["prediction_evidence_role"] = "derived"
        result["source_relationship"] = "third_party_referenced_public_data"
        result["measurement_relationship"] = "not_measured_by_this_study"
        result["source_record_type"] = source_record_type
        return result

    predictions = pd.concat(
        [
            prediction_frame(
                validation, validation_predictions, training_scope="train_only_2021_2023_or_first_60pct"
            ),
            prediction_frame(
                locked_test, test_predictions, training_scope="train_plus_validation"
            ),
        ],
        ignore_index=True,
    )
    prediction_path = output_root / "predictions" / f"{task}_predictions.parquet"
    prediction_artifact = atomic_parquet(
        predictions,
        prediction_path,
        metadata={
            "task": task,
            "selection_split": "validation",
            "locked_test_used_for_selection": False,
            "selected_advanced_seed": selected_seed,
            "source_relationship": "third_party_referenced_public_data",
            "measurement_relationship": "not_measured_by_this_study",
        },
    )
    model_path = output_root / "models" / f"{task}_models.joblib"
    model_artifact = atomic_joblib(
        {
            "task": task,
            "feature_columns": list(feature_columns),
            "target_column": target_column,
            "selected_advanced_seed": selected_seed,
            "advanced_model": final_advanced,
            "ridge_model": final_ridge,
        },
        model_path,
    )
    return {
        "task": task,
        "status": "COMPLETE",
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "forecast_horizon_minutes": forecast_horizon_minutes,
        "row_counts": {
            "train_eligible": len(train),
            "train_fitted": len(fit_train),
            "validation": len(validation),
            "locked_test": len(locked_test),
            "refit_fitted": len(refit_source),
        },
        "seed_trials": seed_trials,
        "selected_advanced_seed": selected_seed,
        "selected_iteration_count": int(best["best_iteration"]),
        "selection_metric": "validation_rmse",
        "selection_frozen_at_utc": selection_frozen_at,
        "locked_test_used_for_selection": False,
        "validation_metrics": validation_metrics,
        "locked_test_metrics": test_metrics,
        "prediction_artifact": prediction_artifact,
        "model_artifact": model_artifact,
    }
