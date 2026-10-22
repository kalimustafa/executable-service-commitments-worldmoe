from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dcvpp.core import sha256_file, utc_now
from dcvpp.experiments.forecast_common import atomic_parquet, regression_metrics
from dcvpp.models.energy.dgpt_e import TinyDGPTE
from dcvpp.models.workload.cfc_w import TinyCfCW


QUANTILES = np.asarray([0.05, 0.10, 0.50, 0.90, 0.95], dtype="float32")
ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_torch(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def build_nyiso_price_15m(price_rows: pd.DataFrame, *, product: str) -> pd.DataFrame:
    source = price_rows[["timestamp_utc", "zone", product]].copy()
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
    if product == "da_lbmp":
        result = source.loc[source.index.repeat(4)].copy().reset_index(drop=True)
        result["timestamp_utc"] += pd.to_timedelta(
            np.tile([0, 15, 30, 45], len(source)), unit="m"
        )
    elif product == "rt_lbmp":
        source["block_start_utc"] = source["timestamp_utc"].dt.ceil("15min") - pd.Timedelta(
            minutes=15
        )
        result = (
            source.groupby(["block_start_utc", "zone"], sort=True)[product]
            .agg([("target", "mean"), ("valid_interval_count", "count")])
            .reset_index()
            .rename(columns={"block_start_utc": "timestamp_utc", "target": product})
        )
        result.loc[result["valid_interval_count"].ne(3), product] = np.nan
    else:
        raise ValueError(product)
    if result.duplicated(["timestamp_utc", "zone"]).any():
        raise ValueError(f"{product} 15-minute panel contains duplicates")
    return result.sort_values(["timestamp_utc", "zone"], kind="stable").reset_index(drop=True)


def _price_matrix(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    matrix = panel.pivot(index="timestamp_utc", columns="zone", values=target).reindex(columns=ZONES)
    full_index = pd.date_range(
        matrix.index.min().floor("15min"), matrix.index.max().floor("15min"), freq="15min", tz="UTC"
    )
    return matrix.reindex(full_index)


def _daily_price_sequences(
    matrix: pd.DataFrame,
    *,
    history: int,
    horizon: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]]:
    values = matrix.to_numpy(dtype="float32")
    index = matrix.index
    positions = {timestamp: position for position, timestamp in enumerate(index)}
    days = pd.date_range(index.min().ceil("D"), index.max().floor("D"), freq="D", tz="UTC")
    result: dict[str, list[Any]] = {key: [[], [], []] for key in ("train", "validation", "locked_test")}
    for day in days:
        position = positions.get(day)
        if position is None or position < history or position + horizon > len(values):
            continue
        x = values[position - history : position]
        y = values[position : position + horizon].T
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        split = "train" if day.year <= 2023 else "validation" if day.year == 2024 else "locked_test" if day.year == 2025 else None
        if split is None:
            continue
        result[split][0].append(x)
        result[split][1].append(y)
        result[split][2].append(day)
    output: dict[str, tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]] = {}
    for split, (xs, ys, split_days) in result.items():
        if not xs:
            raise ValueError(f"DGPT-E {split} sequence set is empty")
        output[split] = (np.stack(xs), np.stack(ys), split_days)
    return output


def _pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.as_tensor(QUANTILES, device=prediction.device, dtype=prediction.dtype)
    error = target.unsqueeze(-1) - prediction
    pinball = torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()
    crossing = torch.relu(prediction[..., :-1] - prediction[..., 1:]).mean()
    return pinball + 0.1 * crossing


@torch.no_grad()
def _dgpt_predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    values = []
    for (batch,) in loader:
        values.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(values)


def _quantile_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = actual[..., None] - predicted
    pinball = np.maximum(QUANTILES * error, (QUANTILES - 1.0) * error)
    median = predicted[..., 2]
    coverage_90 = np.mean((actual >= predicted[..., 0]) & (actual <= predicted[..., 4]))
    coverage_80 = np.mean((actual >= predicted[..., 1]) & (actual <= predicted[..., 3]))
    crossing = np.mean(np.any(np.diff(predicted, axis=-1) < 0, axis=-1))
    return {
        **regression_metrics(actual.ravel(), median.ravel()),
        "mean_pinball": float(pinball.mean()),
        "crps_quantile_approximation": float(2.0 * pinball.mean()),
        "coverage_90": float(coverage_90),
        "coverage_80": float(coverage_80),
        "mean_width_90": float(np.mean(predicted[..., 4] - predicted[..., 0])),
        "quantile_crossing_rate": float(crossing),
    }


def fit_tiny_dgpt_e_feasibility(
    price_rows: pd.DataFrame,
    *,
    product: str,
    task: str,
    seeds: list[int],
    output_root: Path,
    failed_seeds: list[dict[str, Any]],
    history: int = 672,
    horizon: int = 96,
    epochs: int = 10,
    batch_size: int = 32,
    width: int = 64,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
) -> dict[str, Any]:
    panel = build_nyiso_price_15m(price_rows, product=product)
    matrix = _price_matrix(panel, product)
    train_mask = matrix.index.year <= 2023
    center = matrix.loc[train_mask].mean().to_numpy(dtype="float32")
    scale = matrix.loc[train_mask].std().replace(0, 1.0).to_numpy(dtype="float32")
    normalized = (matrix - center) / scale
    sequences = _daily_price_sequences(normalized, history=history, horizon=horizon)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu"
    )
    x_train, y_train, _ = sequences["train"]
    x_validation, y_validation, validation_days = sequences["validation"]
    validation_actual_original = y_validation * scale[None, :, None] + center[None, :, None]

    def original_quantiles(values: np.ndarray) -> np.ndarray:
        return values * scale[None, :, None, None] + center[None, :, None, None]
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for seed in seeds:
        try:
            _seed_everything(seed)
            model = TinyDGPTE(history, horizon, len(ZONES), len(QUANTILES), width=width).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            best_metric = float("inf")
            best_state: dict[str, torch.Tensor] | None = None
            best_epoch = 0
            checkpoint_path = output_root / "checkpoints" / f"{task}_dgpt_e_seed_{seed}.pt"
            for epoch in range(1, epochs + 1):
                model.train()
                for x_batch, y_batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(x_batch.to(device))
                    loss = _pinball_loss(prediction, y_batch.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                validation_prediction = _dgpt_predict(model, x_validation, device, batch_size)
                validation_metric = _quantile_metrics(y_validation, validation_prediction)["mean_pinball"]
                if validation_metric < best_metric:
                    best_metric = float(validation_metric)
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                _atomic_torch(
                    {
                        "model_class": "dcvpp.models.energy.dgpt_e.TinyDGPTE",
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_validation_mean_pinball": best_metric,
                        "seed": seed,
                        "numpy_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "normalization_center": center,
                        "normalization_scale": scale,
                    },
                    checkpoint_path,
                )
            if best_state is None:
                raise RuntimeError("DGPT-E produced no finite validation checkpoint")
            model.load_state_dict(best_state)
            validation_prediction = _dgpt_predict(model, x_validation, device, batch_size)
            metrics = _quantile_metrics(
                validation_actual_original, original_quantiles(validation_prediction)
            )
            trial = {
                "seed": seed,
                "status": "completed",
                "best_epoch": best_epoch,
                "validation_metrics": metrics,
                "checkpoint_path": checkpoint_path.as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            trials.append(trial)
            if best is None or metrics["mean_pinball"] < best["metrics"]["mean_pinball"]:
                best = {
                    "seed": seed,
                    "model": model,
                    "metrics": metrics,
                    "validation_prediction": validation_prediction,
                    "best_epoch": best_epoch,
                    "checkpoint": trial,
                }
        except Exception as exc:
            failure = {
                "task": task,
                "model": "TinyDGPTE",
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at_utc": utc_now(),
            }
            failed_seeds.append(failure)
            trials.append({"seed": seed, "status": "failed", **failure})
    if best is None:
        raise RuntimeError(f"all TinyDGPTE seeds failed for {task}")
    x_test, y_test, test_days = sequences["locked_test"]
    test_prediction = _dgpt_predict(best["model"], x_test, device, batch_size)
    test_metrics = _quantile_metrics(
        y_test * scale[None, :, None] + center[None, :, None],
        original_quantiles(test_prediction),
    )

    def long_predictions(days: list[pd.Timestamp], actual: np.ndarray, predicted: np.ndarray, split: str) -> pd.DataFrame:
        records = []
        for day_index, day in enumerate(days):
            timestamps = pd.date_range(day, periods=horizon, freq="15min", tz="UTC")
            for zone_index, zone in enumerate(ZONES):
                for step, timestamp in enumerate(timestamps):
                    row = {
                        "task": task,
                        "split": split,
                        "operating_date": day.date().isoformat(),
                        "issue_time_utc": (day - pd.Timedelta(days=1)).isoformat(),
                        "timestamp_utc": timestamp,
                        "zone": zone,
                        "observed_target": float(actual[day_index, zone_index, step] * scale[zone_index] + center[zone_index]),
                        "selected_seed": int(best["seed"]),
                        "model_class": "dcvpp.models.energy.dgpt_e.TinyDGPTE",
                    }
                    for q_index, quantile in enumerate(QUANTILES):
                        row[f"q{int(round(float(quantile) * 100)):02d}"] = float(
                            predicted[day_index, zone_index, step, q_index] * scale[zone_index] + center[zone_index]
                        )
                    records.append(row)
        return pd.DataFrame(records)

    predictions = pd.concat(
        [
            long_predictions(validation_days, y_validation, best["validation_prediction"], "validation"),
            long_predictions(test_days, y_test, test_prediction, "locked_test"),
        ],
        ignore_index=True,
    )
    prediction_path = output_root / "predictions" / f"{task}_dgpt_e_quantiles.parquet"
    artifact = atomic_parquet(
        predictions,
        prediction_path,
        metadata={
            "model_class": "dcvpp.models.energy.dgpt_e.TinyDGPTE",
            "selected_seed": best["seed"],
            "locked_test_used_for_selection": False,
            "quantiles": QUANTILES.tolist(),
        },
    )
    return {
        "model_class": "dcvpp.models.energy.dgpt_e.TinyDGPTE",
        "architecture_scope": "compact actual PyTorch graph-patch Transformer implementation",
        "full_preregistered_dgpt_e_architecture_completed": False,
        "h1_support_claimed": False,
        "h1_nonclaim_reason": "full baseline suite, conformal calibration, and preregistered 3-block width-128 architecture are incomplete",
        "seed_trials": trials,
        "selected_seed": int(best["seed"]),
        "selected_epoch": int(best["best_epoch"]),
        "validation_metrics": best["metrics"],
        "locked_test_metrics": test_metrics,
        "prediction_artifact": artifact,
        "sequence_counts": {key: len(value[0]) for key, value in sequences.items()},
        "history_steps": history,
        "horizon_steps": horizon,
        "quantiles": QUANTILES.tolist(),
    }


def _workload_sequences(
    panel: pd.DataFrame,
    *,
    history: int,
    horizon: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]], np.ndarray, np.ndarray]:
    frame = panel.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    counts = frame["submitted_job_count"].to_numpy(dtype="float32")
    volume = frame["submitted_nodes_requested"].to_numpy(dtype="float32")
    input_values = np.column_stack([np.log1p(counts), np.log1p(volume)]).astype("float32")
    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)
    center = input_values[:train_end].mean(axis=0)
    scale = input_values[:train_end].std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (input_values - center) / scale
    nonzero = counts > 0
    elapsed = np.empty(len(frame), dtype="float32")
    since = 0.25
    for index, has_event in enumerate(nonzero):
        elapsed[index] = since
        since = 0.25 if has_event else min(since + 0.25, 168.0)
    timestamps = pd.DatetimeIndex(frame["timestamp_utc"])
    day_positions = np.flatnonzero(
        (timestamps.hour == 0) & (timestamps.minute == 0)
    )
    output: dict[str, list[Any]] = {key: [[], [], [], []] for key in ("train", "validation", "locked_test")}
    for position in day_positions:
        if position < history or position + horizon > len(frame):
            continue
        if position + horizon <= train_end:
            split = "train"
        elif position >= train_end and position + horizon <= validation_end:
            split = "validation"
        elif position >= validation_end:
            split = "locked_test"
        else:
            continue
        output[split][0].append(normalized[position - history : position])
        output[split][1].append(elapsed[position - history : position, None])
        output[split][2].append(input_values[position : position + horizon])
        output[split][3].append(timestamps[position])
    sequences = {
        split: (np.stack(values[0]), np.stack(values[1]), np.stack(values[2]), values[3])
        for split, values in output.items()
    }
    if any(not len(value[0]) for value in sequences.values()):
        raise ValueError("CfC-W chronological sequence split is empty")
    return sequences, center, scale


@torch.no_grad()
def _cfc_predict(model: nn.Module, x: np.ndarray, dt: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(dt)), batch_size=batch_size, shuffle=False
    )
    values = []
    for x_batch, dt_batch in loader:
        counts, volume = model(x_batch.to(device), dt_batch.to(device))
        values.append(torch.stack([counts, volume], dim=-1).cpu().numpy())
    return np.concatenate(values)


def fit_tiny_cfc_w_feasibility(
    workload_panel: pd.DataFrame,
    *,
    seeds: list[int],
    output_root: Path,
    failed_seeds: list[dict[str, Any]],
    history: int = 672,
    horizon: int = 96,
    epochs: int = 10,
    batch_size: int = 32,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
) -> dict[str, Any]:
    sequences, center, scale = _workload_sequences(workload_panel, history=history, horizon=horizon)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu"
    )
    x_train, dt_train, y_train, _ = sequences["train"]
    x_val, dt_val, y_val, val_days = sequences["validation"]
    trials = []
    best = None
    for seed in seeds:
        try:
            _seed_everything(seed)
            model = TinyCfCW(input_size=2, hidden_size=hidden_size, horizon=horizon).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(x_train), torch.from_numpy(dt_train), torch.from_numpy(y_train)),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            best_metric = float("inf")
            best_state = None
            best_epoch = 0
            checkpoint_path = output_root / "checkpoints" / f"nlr_workload_cfc_w_seed_{seed}.pt"
            for epoch in range(1, epochs + 1):
                model.train()
                for x_batch, dt_batch, y_batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    count_pred, volume_pred = model(x_batch.to(device), dt_batch.to(device))
                    target = y_batch.to(device)
                    loss = torch.mean((count_pred - target[..., 0]) ** 2) + torch.mean(
                        (volume_pred - target[..., 1]) ** 2
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                val_prediction = _cfc_predict(model, x_val, dt_val, device, batch_size)
                val_rmse = float(np.sqrt(np.mean((val_prediction - y_val) ** 2)))
                if val_rmse < best_metric:
                    best_metric = val_rmse
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                _atomic_torch(
                    {
                        "model_class": "dcvpp.models.workload.cfc_w.TinyCfCW",
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_validation_log_rmse": best_metric,
                        "seed": seed,
                        "numpy_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "normalization_center": center,
                        "normalization_scale": scale,
                    },
                    checkpoint_path,
                )
            if best_state is None:
                raise RuntimeError("CfC-W produced no validation checkpoint")
            model.load_state_dict(best_state)
            val_prediction = _cfc_predict(model, x_val, dt_val, device, batch_size)
            val_actual_raw = np.expm1(y_val)
            val_prediction_raw = np.maximum(np.expm1(val_prediction), 0.0)
            metrics = {
                "count": regression_metrics(val_actual_raw[..., 0].ravel(), val_prediction_raw[..., 0].ravel()),
                "volume": regression_metrics(val_actual_raw[..., 1].ravel(), val_prediction_raw[..., 1].ravel()),
                "joint_log_rmse": float(np.sqrt(np.mean((val_prediction - y_val) ** 2))),
            }
            trial = {
                "seed": seed,
                "status": "completed",
                "best_epoch": best_epoch,
                "validation_metrics": metrics,
                "checkpoint_path": checkpoint_path.as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            trials.append(trial)
            if best is None or metrics["joint_log_rmse"] < best["metrics"]["joint_log_rmse"]:
                best = {
                    "seed": seed,
                    "model": model,
                    "metrics": metrics,
                    "validation_prediction": val_prediction_raw,
                    "best_epoch": best_epoch,
                }
        except Exception as exc:
            failure = {
                "task": "nlr_workload",
                "model": "TinyCfCW",
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at_utc": utc_now(),
            }
            failed_seeds.append(failure)
            trials.append({"seed": seed, "status": "failed", **failure})
    if best is None:
        raise RuntimeError("all TinyCfCW seeds failed")
    x_test, dt_test, y_test, test_days = sequences["locked_test"]
    test_log_prediction = _cfc_predict(best["model"], x_test, dt_test, device, batch_size)
    test_prediction = np.maximum(np.expm1(test_log_prediction), 0.0)
    test_actual = np.expm1(y_test)
    test_metrics = {
        "count": regression_metrics(test_actual[..., 0].ravel(), test_prediction[..., 0].ravel()),
        "volume": regression_metrics(test_actual[..., 1].ravel(), test_prediction[..., 1].ravel()),
        "joint_log_rmse": float(np.sqrt(np.mean((test_log_prediction - y_test) ** 2))),
    }

    def long_predictions(days: list[pd.Timestamp], actual: np.ndarray, prediction: np.ndarray, split: str) -> pd.DataFrame:
        records = []
        for day_index, day in enumerate(days):
            for step, timestamp in enumerate(pd.date_range(day, periods=horizon, freq="15min", tz="UTC")):
                records.append(
                    {
                        "task": "nlr_workload",
                        "split": split,
                        "operating_date": day.date().isoformat(),
                        "prediction_origin_utc": day - pd.Timedelta(minutes=15),
                        "feature_data_cutoff_utc": day - pd.Timedelta(minutes=15),
                        "timestamp_utc": timestamp,
                        "observed_job_count": float(actual[day_index, step, 0]),
                        "observed_nodes_requested": float(actual[day_index, step, 1]),
                        "predicted_job_count": float(prediction[day_index, step, 0]),
                        "predicted_nodes_requested": float(prediction[day_index, step, 1]),
                        "selected_seed": int(best["seed"]),
                        "model_class": "dcvpp.models.workload.cfc_w.TinyCfCW",
                    }
                )
        return pd.DataFrame(records)

    val_actual = np.expm1(y_val)
    predictions = pd.concat(
        [
            long_predictions(val_days, val_actual, best["validation_prediction"], "validation"),
            long_predictions(test_days, test_actual, test_prediction, "locked_test"),
        ],
        ignore_index=True,
    )
    path = output_root / "predictions/nlr_workload_cfc_w_predictions.parquet"
    artifact = atomic_parquet(
        predictions,
        path,
        metadata={
            "model_class": "dcvpp.models.workload.cfc_w.TinyCfCW",
            "selected_seed": best["seed"],
            "locked_test_used_for_selection": False,
            "distribution_head": "deterministic log1p count and volume; not negative-binomial",
        },
    )
    return {
        "model_class": "dcvpp.models.workload.cfc_w.TinyCfCW",
        "architecture_scope": "compact actual PyTorch elapsed-time CfC implementation",
        "full_preregistered_cfc_w_distribution_model_completed": False,
        "h2_support_claimed": False,
        "h2_nonclaim_reason": "event-level cohort encoder, negative-binomial head, calibrated intervals, and full baseline suite are incomplete",
        "seed_trials": trials,
        "selected_seed": int(best["seed"]),
        "selected_epoch": int(best["best_epoch"]),
        "validation_metrics": best["metrics"],
        "locked_test_metrics": test_metrics,
        "prediction_artifact": artifact,
        "sequence_counts": {key: len(value[0]) for key, value in sequences.items()},
        "history_steps": history,
        "horizon_steps": horizon,
    }


def fit_dgpt_e_five_seeds(
    price_rows: pd.DataFrame,
    *,
    product: str,
    task: str,
    seeds: list[int],
    output_root: Path,
    failed_seeds: list[dict[str, Any]],
    history: int = 672,
    horizon: int = 96,
    epochs: int = 100,
    batch_size: int = 32,
    width: int = 128,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
    resume_fingerprint: str = "",
    minimum_calibration_samples: int = 500,
    n_estimators: int = 500,
    n_jobs: int = 8,
    paper_evidence: bool = True,
) -> dict[str, Any]:
    """Run the full paper DGPT-E branch; TinyDGPTE is demo-only."""
    from dcvpp.experiments.paper_training import fit_paper_dgpt_e_five_seeds

    return fit_paper_dgpt_e_five_seeds(
        price_rows,
        product=product,
        task=task,
        seeds=seeds,
        output_root=output_root,
        failed_seeds=failed_seeds,
        history=history,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        width=width,
        learning_rate=learning_rate,
        device_name=device_name,
        resume_fingerprint=resume_fingerprint,
        minimum_calibration_samples=minimum_calibration_samples,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        paper_evidence=paper_evidence,
    )


def fit_cfc_w_five_seeds(
    workload_panel: pd.DataFrame,
    *,
    seeds: list[int],
    output_root: Path,
    failed_seeds: list[dict[str, Any]],
    history: int = 672,
    horizon: int = 96,
    epochs: int = 100,
    batch_size: int = 32,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
    resume_fingerprint: str = "",
    minimum_calibration_samples: int = 500,
    event_artifact_path: Path | None = None,
    event_artifact_sha256: str | None = None,
    paper_evidence: bool = True,
) -> dict[str, Any]:
    """Run the full paper CfC-W branch; TinyCfCW is demo-only."""
    from dcvpp.experiments.paper_training import fit_paper_cfc_w_five_seeds

    return fit_paper_cfc_w_five_seeds(
        workload_panel,
        seeds=seeds,
        output_root=output_root,
        failed_seeds=failed_seeds,
        history=history,
        horizon=horizon,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        device_name=device_name,
        resume_fingerprint=resume_fingerprint,
        minimum_calibration_samples=minimum_calibration_samples,
        event_artifact_path=event_artifact_path,
        event_artifact_sha256=event_artifact_sha256,
        paper_evidence=paper_evidence,
    )
