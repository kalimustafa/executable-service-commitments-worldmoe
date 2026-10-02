from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import duckdb
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
)
from dcvpp.experiments.forecast_common import atomic_parquet, regression_metrics
from dcvpp.models.energy import DGPTE
from dcvpp.models.workload import (
    CfCW,
    gamma_volume_nll,
    lognormal_volume_nll,
    negative_binomial_nll,
)


QUANTILES = np.asarray(DEFAULT_QUANTILES, dtype="float32")
ZONES = ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
COHORTS = (
    "A_critical_interactive",
    "B_latency_sensitive",
    "C_preemptible_training",
    "D_elastic_batch",
)
EXPECTED_PAPER_SEEDS = (20260809, 20260819, 20260829, 20260908, 20260918)
PAPER_DGPT_CONTRACT = {
    "history": 672,
    "horizon": 96,
    "patch_length": 16,
    "stride": 8,
    "width": 128,
    "transformer_blocks": 3,
    "heads": 4,
    "feed_forward_width": 256,
    "dropout": 0.1,
    "graph_layers": 1,
    "dynamic_graph": True,
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


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


def _device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _paper_seed_gate(seeds: Sequence[int], *, paper_evidence: bool) -> list[int]:
    values = [int(value) for value in seeds]
    if paper_evidence and tuple(values) != EXPECTED_PAPER_SEEDS:
        raise ValueError(
            f"paper forecasting requires the preregistered five seeds {EXPECTED_PAPER_SEEDS}"
        )
    if not values:
        raise ValueError("forecast training requires at least one seed")
    return values


def _paper_shape_gate(history: int, horizon: int, *, paper_evidence: bool) -> None:
    if paper_evidence and (history, horizon) != (672, 96):
        raise ValueError("paper forecasting fixes history=672 and horizon=96")


class DLinearQuantile(nn.Module):
    def __init__(self, history: int, horizon: int, sites: int, quantiles: int) -> None:
        super().__init__()
        self.history, self.horizon, self.sites, self.quantiles = history, horizon, sites, quantiles
        self.linear = nn.Linear(history, horizon)
        self.quantile_offsets = nn.Parameter(torch.linspace(-0.5, 0.5, quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point = self.linear(x.transpose(1, 2))
        return point.unsqueeze(-1) + self.quantile_offsets


class LSTMQuantile(nn.Module):
    def __init__(self, history: int, horizon: int, sites: int, quantiles: int) -> None:
        super().__init__()
        self.horizon, self.sites, self.quantiles = horizon, sites, quantiles
        self.encoder = nn.LSTM(sites, 96, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Linear(96, sites * horizon * quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1]).reshape(
            len(x), self.sites, self.horizon, self.quantiles
        )


class PatchTSTQuantile(nn.Module):
    def __init__(self, history: int, horizon: int, sites: int, quantiles: int) -> None:
        super().__init__()
        self.history, self.horizon, self.sites, self.quantiles = history, horizon, sites, quantiles
        self.patch = nn.Conv1d(1, 128, kernel_size=16, stride=8)
        count = (history - 16) // 8 + 1
        self.position = nn.Parameter(torch.zeros(1, count, 128))
        layer = nn.TransformerEncoderLayer(
            128, nhead=4, dim_feedforward=256, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Linear(128, horizon * quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = len(x)
        tokens = self.patch(x.transpose(1, 2).reshape(batch * self.sites, 1, self.history))
        encoded = self.encoder(tokens.transpose(1, 2) + self.position).mean(dim=1)
        return self.head(encoded).reshape(batch, self.sites, self.horizon, self.quantiles)


class ITransformerQuantile(nn.Module):
    def __init__(self, history: int, horizon: int, sites: int, quantiles: int) -> None:
        super().__init__()
        self.horizon, self.sites, self.quantiles = horizon, sites, quantiles
        self.temporal_projection = nn.Linear(history, 128)
        self.site_embedding = nn.Parameter(torch.zeros(1, sites, 128))
        layer = nn.TransformerEncoderLayer(
            128, nhead=4, dim_feedforward=256, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Linear(128, horizon * quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.temporal_projection(x.transpose(1, 2)) + self.site_embedding
        encoded = self.encoder(tokens)
        return self.head(encoded).reshape(len(x), self.sites, self.horizon, self.quantiles)


def _energy_factory(
    name: str,
    *,
    history: int,
    horizon: int,
    sites: int,
    quantiles: int,
) -> nn.Module:
    if name == "dlinear":
        return DLinearQuantile(history, horizon, sites, quantiles)
    if name == "lstm":
        return LSTMQuantile(history, horizon, sites, quantiles)
    if name == "patchtst":
        return PatchTSTQuantile(history, horizon, sites, quantiles)
    if name == "itransformer":
        return ITransformerQuantile(history, horizon, sites, quantiles)
    if name == "static_graph_patch_transformer":
        return DGPTE(
            history=history, horizon=horizon, sites=sites, quantiles=quantiles,
            graph_mode="static",
        )
    if name == "dgpt_e":
        return DGPTE(
            history=history, horizon=horizon, sites=sites, quantiles=quantiles,
            graph_mode="dynamic",
        )
    raise ValueError(f"unknown paper energy model: {name}")


def _pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.as_tensor(QUANTILES, device=prediction.device, dtype=prediction.dtype)
    error = target.unsqueeze(-1) - prediction
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()
    crossing = torch.relu(prediction[..., :-1] - prediction[..., 1:]).mean()
    return loss + 0.1 * crossing


@torch.no_grad()
def _energy_predict(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    for (batch,) in loader:
        values.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(values)


def _checkpoint_payload(
    *,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    best_state: dict[str, torch.Tensor] | None,
    seed: int,
    fingerprint: str,
    architecture: Mapping[str, Any],
    batch_size: int,
    completed: bool,
    paper_evidence: bool,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 2,
        "paper_evidence": bool(paper_evidence),
        "model_name": model_name,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_validation_mean_weighted_pinball": float(best_metric),
        "best_model_state": best_state,
        "seed": int(seed),
        "resume_fingerprint": fingerprint,
        "architecture": dict(architecture),
        "batch_size": int(batch_size),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "completed": bool(completed),
        "written_at_utc": utc_now(),
    }


def _fit_energy_seed(
    *,
    task: str,
    model_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    history: int,
    horizon: int,
    epochs: int,
    batch_sizes: Sequence[int],
    gradient_accumulation_steps: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    output_root: Path,
    resume_fingerprint: str,
    paper_evidence: bool = True,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for batch_size in batch_sizes:
        model = _energy_factory(
            model_name, history=history, horizon=horizon, sites=len(ZONES), quantiles=len(QUANTILES)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        checkpoint_path = (
            output_root / "checkpoints" / task / model_name / f"seed_{seed}_latest.pt"
        )
        architecture = (
            model.architecture_contract
            if hasattr(model, "architecture_contract")
            else {
                "history": history,
                "horizon": horizon,
                "model": model_name,
            }
        )
        fingerprint = sha256_json(
            {
                "parent": resume_fingerprint,
                "task": task,
                "model": model_name,
                "seed": seed,
                "architecture": architecture,
                "epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
            }
        )
        start_epoch = 1
        best_epoch = 0
        best_metric = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        resumed_from_epoch = 0
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("resume_fingerprint") == fingerprint:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if checkpoint.get("gradient_scaler_state"):
                    scaler.load_state_dict(checkpoint["gradient_scaler_state"])
                start_epoch = int(checkpoint["epoch"]) + 1
                resumed_from_epoch = int(checkpoint["epoch"])
                best_epoch = int(checkpoint["best_epoch"])
                best_metric = float(checkpoint["best_validation_mean_weighted_pinball"])
                best_state = checkpoint.get("best_model_state")
                if bool(checkpoint.get("completed")):
                    if best_state is None:
                        raise ValueError("completed energy checkpoint lacks its best state")
                    model.load_state_dict(best_state)
                    return {
                        "model": model,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "epochs_written": int(checkpoint["epoch"]),
                        "resumed_from_epoch": resumed_from_epoch,
                        "batch_size": batch_size,
                        "resume_fingerprint": fingerprint,
                        "architecture": architecture,
                    }
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=device.type == "cuda",
        )
        no_improvement = max(0, resumed_from_epoch - best_epoch)
        try:
            for epoch in range(start_epoch, epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for index, (x_batch, y_batch) in enumerate(loader, start=1):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        prediction = model(x_batch.to(device, non_blocking=True))
                        loss = _pinball_loss(prediction, y_batch.to(device, non_blocking=True))
                        loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    if index % gradient_accumulation_steps == 0 or index == len(loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                validation_prediction = _energy_predict(
                    model, x_validation, device=device, batch_size=batch_size
                )
                validation_metric = quantile_metrics(
                    y_validation, validation_prediction, quantiles=QUANTILES
                )["mean_weighted_pinball"]
                if not math.isfinite(float(validation_metric)):
                    raise RuntimeError("energy validation metric is non-finite")
                if float(validation_metric) < best_metric - 1e-8:
                    best_metric = float(validation_metric)
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    no_improvement = 0
                else:
                    no_improvement += 1
                completed = epoch == epochs or no_improvement >= patience
                _atomic_torch(
                    _checkpoint_payload(
                        model_name=model_name,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_epoch=best_epoch,
                        best_metric=best_metric,
                        best_state=best_state,
                        seed=seed,
                        fingerprint=fingerprint,
                        architecture=architecture,
                        batch_size=batch_size,
                        completed=completed,
                        paper_evidence=paper_evidence,
                        scaler=scaler,
                    ),
                    checkpoint_path,
                )
                if completed:
                    break
            if best_state is None:
                raise RuntimeError(f"{model_name} produced no validation checkpoint")
            model.load_state_dict(best_state)
            latest = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            return {
                "model": model,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "epochs_written": int(latest["epoch"]),
                "resumed_from_epoch": resumed_from_epoch,
                "batch_size": batch_size,
                "resume_fingerprint": fingerprint,
                "architecture": architecture,
            }
        except torch.OutOfMemoryError as exc:
            last_error = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            checkpoint_path.unlink(missing_ok=True)
            continue
    raise RuntimeError(
        f"{model_name} exhausted batch fallback {list(batch_sizes)}: {last_error}"
    )


def _daily_sequences(
    history_matrix: pd.DataFrame,
    *,
    history: int,
    horizon: int,
    issue_gap_steps: int,
    target_matrix: pd.DataFrame | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]]:
    targets = history_matrix if target_matrix is None else target_matrix
    if not history_matrix.index.equals(targets.index) or list(history_matrix.columns) != list(targets.columns):
        raise ValueError("history and target matrices must share their exact index and columns")
    history_values_all = history_matrix.to_numpy(dtype="float32")
    target_values_all = targets.to_numpy(dtype="float32")
    positions = {timestamp: position for position, timestamp in enumerate(history_matrix.index)}
    days = pd.date_range(history_matrix.index.min().ceil("D"), history_matrix.index.max().floor("D"), freq="D", tz="UTC")
    buckets: dict[str, list[Any]] = {
        key: [[], [], []] for key in ("train", "validation", "locked_test")
    }
    for day in days:
        position = positions.get(day)
        history_end = None if position is None else position - issue_gap_steps
        if (
            position is None
            or history_end is None
            or history_end < history
            or position + horizon > len(history_values_all)
        ):
            continue
        history_values = history_values_all[history_end - history : history_end]
        target_values = target_values_all[position : position + horizon].T
        if not np.isfinite(history_values).all() or not np.isfinite(target_values).all():
            continue
        split = (
            "train" if day.year <= 2023 else
            "validation" if day.year == 2024 else
            "locked_test" if day.year == 2025 else None
        )
        if split is not None:
            buckets[split][0].append(history_values)
            buckets[split][1].append(target_values)
            buckets[split][2].append(day)
    result: dict[str, tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]] = {}
    for split, values_for_split in buckets.items():
        if not values_for_split[0]:
            raise ValueError(f"paper DGPT-E {split} sequence set is empty")
        result[split] = (
            np.stack(values_for_split[0]).astype("float32"),
            np.stack(values_for_split[1]).astype("float32"),
            list(values_for_split[2]),
        )
    return result


def _deployment_energy_sequences(
    history_matrix: pd.DataFrame,
    *,
    history: int,
    horizon: int,
    issue_gap_steps: int,
    target_matrix: pd.DataFrame,
    target_year: int = 2025,
    operating_dates: Sequence[pd.Timestamp | str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp], list[dict[str, Any]]]:
    """Build issue-time-safe deployment inputs without requiring realized targets.

    The returned target array intentionally retains NaNs.  It is carried only
    for a completeness audit and optional later evaluation; model inference
    receives ``history_inputs`` alone.  This is the deployment counterpart to
    :func:`_daily_sequences`, whose complete-target gate is appropriate for
    validation and locked-test scoring but not for counterfactual operation.
    """
    if not history_matrix.index.equals(target_matrix.index) or list(
        history_matrix.columns
    ) != list(target_matrix.columns):
        raise ValueError("deployment history and target matrices must share index and columns")
    if history <= 0 or horizon <= 0 or issue_gap_steps < 0:
        raise ValueError("deployment sequence dimensions must be positive")

    if operating_dates is None:
        requested_days = list(
            pd.date_range(
                f"{target_year:04d}-01-01",
                f"{target_year:04d}-12-31",
                freq="D",
                tz="UTC",
            )
        )
    else:
        requested_days = [pd.Timestamp(value) for value in operating_dates]
        requested_days = [
            value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
            for value in requested_days
        ]
    if len({value.normalize() for value in requested_days}) != len(requested_days):
        raise ValueError("deployment operating dates must be unique")
    requested_days = sorted(value.normalize() for value in requested_days)

    history_values_all = history_matrix.to_numpy(dtype="float32")
    target_values_all = target_matrix.to_numpy(dtype="float32")
    positions = {timestamp: position for position, timestamp in enumerate(history_matrix.index)}
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    days: list[pd.Timestamp] = []
    quality: list[dict[str, Any]] = []
    for day in requested_days:
        position = positions.get(day)
        history_end = None if position is None else position - issue_gap_steps
        if position is None:
            raise ValueError(f"deployment origin is absent from the 15-minute grid: {day}")
        if history_end is None or history_end < history:
            raise ValueError(f"deployment history is unavailable at origin: {day}")
        if position + horizon > len(history_values_all):
            raise ValueError(f"deployment horizon is outside the source grid: {day}")
        history_values = history_values_all[history_end - history : history_end]
        if not np.isfinite(history_values).all():
            raise ValueError(
                f"deployment history contains a non-finite value after causal filling: {day}"
            )
        target_values = target_values_all[position : position + horizon].T
        finite = np.isfinite(target_values)
        missing_by_site = {
            str(site): int((~finite[site_index]).sum())
            for site_index, site in enumerate(target_matrix.columns)
        }
        missing_intervals = int((~finite).any(axis=0).sum())
        missing_values = int((~finite).sum())
        inputs.append(history_values)
        targets.append(target_values)
        days.append(day)
        quality.append(
            {
                "operating_date": day.date().isoformat(),
                "realized_target_complete": missing_values == 0,
                "eligible_for_locked_test_scoring": missing_values == 0,
                "missing_realized_target_interval_count": missing_intervals,
                "missing_realized_target_value_count": missing_values,
                "missing_realized_target_intervals_by_site": missing_by_site,
                "missing_realized_targets_imputed": False,
                "deployment_history_causally_filled": True,
            }
        )
    if not inputs:
        raise ValueError("deployment sequence set is empty")
    return (
        np.stack(inputs).astype("float32"),
        np.stack(targets).astype("float32"),
        days,
        quality,
    )


def _deterministic_energy_quantiles(
    point: np.ndarray,
    train_error: np.ndarray,
) -> np.ndarray:
    offsets = np.quantile(train_error.reshape(-1), QUANTILES, method="linear")
    return point[..., None] + offsets


def _seasonal_and_persistence(
    x: np.ndarray,
    y: np.ndarray,
    *,
    history: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    persistence = np.repeat(x[:, -1, :].T[:, :, None], horizon, axis=2).transpose(1, 0, 2)
    if history < 7 * horizon:
        seasonal = np.repeat(x[:, -horizon:, :].transpose(0, 2, 1), 1, axis=0)
    else:
        seasonal = x[:, -7 * horizon : -6 * horizon, :].transpose(0, 2, 1)
    return {"persistence": persistence, "seasonal_naive": seasonal}


def _train_lgbm_quantile(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    *,
    n_estimators: int,
    n_jobs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("paper LightGBM quantile baseline requires lightgbm") from exc

    def design(x: np.ndarray) -> np.ndarray:
        sample_count = len(x)
        rows = np.zeros((sample_count, len(ZONES), 96, 11), dtype="float32")
        for horizon_index in range(96):
            rows[:, :, horizon_index, 0] = x[:, -1, :]
            rows[:, :, horizon_index, 1] = x[:, -96 + horizon_index, :]
            rows[:, :, horizon_index, 2] = x[:, -672 + horizon_index, :]
            rows[:, :, horizon_index, 3] = math.sin(2 * math.pi * horizon_index / 96)
            rows[:, :, horizon_index, 4] = math.cos(2 * math.pi * horizon_index / 96)
            rows[:, :, horizon_index, 5] = horizon_index / 95
            for zone_index in range(len(ZONES)):
                rows[:, zone_index, horizon_index, 6 + zone_index] = 1.0
            rows[:, :, horizon_index, 10] = x[:, -96:, :].std(axis=1)
        return rows.reshape(-1, rows.shape[-1])

    if train_y.shape[2] != 96:
        raise ValueError("paper LightGBM baseline requires horizon 96")
    x_fit = design(train_x)
    y_fit = train_y.reshape(-1)
    x_validation = design(validation_x)
    x_test = design(test_x)
    validation_quantiles = []
    test_quantiles = []
    model_hashes = []
    for level in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(level),
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(x_fit, y_fit)
        validation_quantiles.append(model.predict(x_validation))
        test_quantiles.append(model.predict(x_test))
        model_hashes.append(
            hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest()
        )
    validation = np.stack(validation_quantiles, axis=-1).reshape(
        len(validation_x), len(ZONES), 96, len(QUANTILES)
    )
    test = np.stack(test_quantiles, axis=-1).reshape(
        len(test_x), len(ZONES), 96, len(QUANTILES)
    )
    return validation, test, {
        "deterministic": True,
        "model_sha256_by_quantile": model_hashes,
        "n_estimators": n_estimators,
    }


def fit_paper_dgpt_e_five_seeds(
    price_rows: pd.DataFrame,
    *,
    product: str,
    task: str,
    seeds: Sequence[int],
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
    from dcvpp.experiments.torch_forecasts import build_nyiso_price_15m

    seed_values = _paper_seed_gate(seeds, paper_evidence=paper_evidence)
    _paper_shape_gate(history, horizon, paper_evidence=paper_evidence)
    if paper_evidence and width != 128:
        raise ValueError("paper DGPT-E fixes width=128")
    if epochs != 100 and paper_evidence:
        raise ValueError("paper DGPT-E requires max_epochs=100")
    panel = build_nyiso_price_15m(price_rows, product=product)
    matrix = panel.pivot(index="timestamp_utc", columns="zone", values=product).reindex(columns=ZONES)
    full_index = pd.date_range(matrix.index.min().floor("15min"), matrix.index.max().floor("15min"), freq="15min", tz="UTC")
    matrix = matrix.reindex(full_index)
    train_mask = matrix.index.year <= 2023
    center = matrix.loc[train_mask].mean().to_numpy(dtype="float32")
    scale = matrix.loc[train_mask].std().replace(0, 1.0).to_numpy(dtype="float32")
    normalized = (matrix - center) / scale
    history_missing = normalized.isna()
    normalized_history = normalized.ffill().fillna(0.0)
    issue_gap_steps = horizon if product == "da_lbmp" else 0
    sequences = _daily_sequences(
        normalized_history,
        history=history,
        horizon=horizon,
        issue_gap_steps=issue_gap_steps,
        target_matrix=normalized,
    )
    x_train, y_train, train_days = sequences["train"]
    x_validation, y_validation, validation_days = sequences["validation"]
    x_test, y_test, test_days = sequences["locked_test"]
    (
        x_deployment,
        y_deployment,
        deployment_days,
        deployment_target_quality,
    ) = _deployment_energy_sequences(
        normalized_history,
        history=history,
        horizon=horizon,
        issue_gap_steps=issue_gap_steps,
        target_matrix=normalized,
        target_year=2025,
    )
    data_fingerprint = sha256_json(
        {
            "parent": resume_fingerprint,
            "product": product,
            "history": history,
            "horizon": horizon,
            "train_x": _array_sha256(x_train),
            "train_y": _array_sha256(y_train),
            "validation_x": _array_sha256(x_validation),
            "validation_y": _array_sha256(y_validation),
            "deployment_x": _array_sha256(x_deployment),
            "deployment_target_audit": _array_sha256(y_deployment),
            "deployment_target_quality": deployment_target_quality,
            "normalization_center": center.tolist(),
            "normalization_scale": scale.tolist(),
            "causal_history_imputation": "per-zone forward fill then training-center zero fallback",
            "history_missing_interval_count": int(history_missing.any(axis=1).sum()),
        }
    )
    selected_device = _device(device_name)
    batch_fallback = tuple(value for value in (batch_size, 16, 8, 4) if value <= batch_size)
    batch_fallback = tuple(dict.fromkeys(batch_fallback))
    spike_threshold = float(np.quantile(y_train * scale[None, :, None] + center[None, :, None], 0.99))

    def original(values: np.ndarray) -> np.ndarray:
        return values * scale[None, :, None, None] + center[None, :, None, None]

    actual_validation = y_validation * scale[None, :, None] + center[None, :, None]
    actual_test = y_test * scale[None, :, None] + center[None, :, None]
    actual_deployment = y_deployment * scale[None, :, None] + center[None, :, None]
    actual_train = y_train * scale[None, :, None] + center[None, :, None]
    baseline_results: list[dict[str, Any]] = []
    baseline_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    deterministic_train = _seasonal_and_persistence(x_train, y_train, history=history, horizon=horizon)
    deterministic_validation = _seasonal_and_persistence(x_validation, y_validation, history=history, horizon=horizon)
    deterministic_test = _seasonal_and_persistence(x_test, y_test, history=history, horizon=horizon)
    for name in ("seasonal_naive", "persistence"):
        train_point = deterministic_train[name] * scale[None, :, None] + center[None, :, None]
        validation_point = deterministic_validation[name] * scale[None, :, None] + center[None, :, None]
        test_point = deterministic_test[name] * scale[None, :, None] + center[None, :, None]
        raw_validation = _deterministic_energy_quantiles(validation_point, actual_train - train_point)
        raw_test = _deterministic_energy_quantiles(test_point, actual_train - train_point)
        calibrator = fit_additive_conformal_quantiles(
            actual_validation, raw_validation, quantiles=QUANTILES,
            minimum_samples=minimum_calibration_samples,
        )
        calibrated_validation = apply_additive_conformal_quantiles(raw_validation, calibrator)
        calibrated_test = apply_additive_conformal_quantiles(raw_test, calibrator)
        baseline_predictions[name] = (calibrated_validation, calibrated_test)
        baseline_results.append(
            {
                "model": name,
                "deployable": True,
                "deterministic": True,
                "validation_metrics": quantile_metrics(actual_validation, calibrated_validation, quantiles=QUANTILES, spike_threshold=spike_threshold),
                "locked_test_metrics": quantile_metrics(actual_test, calibrated_test, quantiles=QUANTILES, spike_threshold=spike_threshold),
                "calibration": {key: value for key, value in calibrator.items() if key != "corrections"},
            }
        )

    lgb_validation, lgb_test, lgb_record = _train_lgbm_quantile(
        x_train, y_train, x_validation, y_validation, x_test,
        n_estimators=n_estimators, n_jobs=n_jobs, seed=seed_values[0],
    )
    lgb_validation = original(lgb_validation)
    lgb_test = original(lgb_test)
    lgb_calibrator = fit_additive_conformal_quantiles(
        actual_validation, lgb_validation, quantiles=QUANTILES,
        minimum_samples=minimum_calibration_samples,
    )
    lgb_validation = apply_additive_conformal_quantiles(lgb_validation, lgb_calibrator)
    lgb_test = apply_additive_conformal_quantiles(lgb_test, lgb_calibrator)
    baseline_predictions["lightgbm_quantile"] = (lgb_validation, lgb_test)
    baseline_results.append(
        {
            "model": "lightgbm_quantile",
            "deployable": True,
            **lgb_record,
            "validation_metrics": quantile_metrics(actual_validation, lgb_validation, quantiles=QUANTILES, spike_threshold=spike_threshold),
            "locked_test_metrics": quantile_metrics(actual_test, lgb_test, quantiles=QUANTILES, spike_threshold=spike_threshold),
            "calibration": {key: value for key, value in lgb_calibrator.items() if key != "corrections"},
        }
    )

    model_results: dict[str, Any] = {}
    for model_name in ("lstm", "dlinear", "patchtst", "itransformer", "static_graph_patch_transformer", "dgpt_e"):
        trials: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for seed in seed_values:
            try:
                _seed_everything(seed)
                fitted = _fit_energy_seed(
                    task=task,
                    model_name=model_name,
                    seed=seed,
                    x_train=x_train,
                    y_train=y_train,
                    x_validation=x_validation,
                    y_validation=y_validation,
                    history=history,
                    horizon=horizon,
                    epochs=epochs,
                    batch_sizes=batch_fallback,
                    gradient_accumulation_steps=2,
                    learning_rate=learning_rate,
                    patience=10,
                    device=selected_device,
                    output_root=output_root,
                    resume_fingerprint=data_fingerprint,
                )
                validation_raw_normalized = _energy_predict(
                    fitted["model"], x_validation, device=selected_device,
                    batch_size=fitted["batch_size"],
                )
                validation_raw = original(validation_raw_normalized)
                validation_metrics = quantile_metrics(
                    actual_validation, validation_raw, quantiles=QUANTILES,
                    spike_threshold=spike_threshold,
                )
                trial = {
                    "seed": seed,
                    "status": "completed",
                    "selected_epoch": fitted["best_epoch"],
                    "epochs_written": fitted["epochs_written"],
                    "resumed_from_epoch": fitted["resumed_from_epoch"],
                    "batch_size": fitted["batch_size"],
                    "validation_metrics_raw": validation_metrics,
                    "checkpoint_path": fitted["checkpoint_path"].as_posix(),
                    "checkpoint_sha256": fitted["checkpoint_sha256"],
                    "resume_fingerprint": fitted["resume_fingerprint"],
                }
                trials.append(trial)
                if best is None or validation_metrics["mean_weighted_pinball"] < best["validation_metrics"]["mean_weighted_pinball"]:
                    best = {
                        **fitted,
                        "seed": seed,
                        "validation_raw": validation_raw,
                        "validation_metrics": validation_metrics,
                    }
            except Exception as exc:
                failure = {
                    "task": task,
                    "model": model_name,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recorded_at_utc": utc_now(),
                }
                failed_seeds.append(failure)
                trials.append({"status": "failed", **failure})
        if best is None:
            raise RuntimeError(f"all five paper seeds failed for {task}/{model_name}")
        test_raw_normalized = _energy_predict(
            best["model"], x_test, device=selected_device, batch_size=best["batch_size"]
        )
        test_raw = original(test_raw_normalized)
        calibrator = fit_additive_conformal_quantiles(
            actual_validation, best["validation_raw"], quantiles=QUANTILES,
            minimum_samples=minimum_calibration_samples,
        )
        validation_calibrated = apply_additive_conformal_quantiles(best["validation_raw"], calibrator)
        test_calibrated = apply_additive_conformal_quantiles(test_raw, calibrator)
        record = {
            "model": model_name,
            "deployable": True,
            "deterministic": False,
            "seed_trials": trials,
            "selected_seed": int(best["seed"]),
            "selected_epoch": int(best["best_epoch"]),
            "batch_size": int(best["batch_size"]),
            "checkpoint_path": best["checkpoint_path"].as_posix(),
            "checkpoint_sha256": best["checkpoint_sha256"],
            "resume_fingerprint": best["resume_fingerprint"],
            "architecture": best["architecture"],
            "validation_metrics": quantile_metrics(actual_validation, validation_calibrated, quantiles=QUANTILES, spike_threshold=spike_threshold),
            "locked_test_metrics": quantile_metrics(actual_test, test_calibrated, quantiles=QUANTILES, spike_threshold=spike_threshold),
            "calibration": {
                **{key: value for key, value in calibrator.items() if key != "corrections"},
                "corrections": np.asarray(calibrator["corrections"]).tolist(),
            },
        }
        model_results[model_name] = record
        if model_name != "dgpt_e":
            baseline_results.append(record)
            baseline_predictions[model_name] = (validation_calibrated, test_calibrated)
        else:
            selected_validation = validation_calibrated
            selected_test = test_calibrated

    def long_predictions(
        days: list[pd.Timestamp], actual: np.ndarray, raw: np.ndarray,
        calibrated: np.ndarray, split: str,
        *,
        target_quality: Sequence[Mapping[str, Any]] | None = None,
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for day_index, day in enumerate(days):
            quality = (
                dict(target_quality[day_index])
                if target_quality is not None
                else {
                    "realized_target_complete": True,
                    "eligible_for_locked_test_scoring": split == "locked_test",
                    "missing_realized_target_interval_count": 0,
                    "missing_realized_target_value_count": 0,
                    "missing_realized_targets_imputed": False,
                    "deployment_history_causally_filled": False,
                }
            )
            timestamps = pd.date_range(day, periods=horizon, freq="15min", tz="UTC")
            for zone_index, zone in enumerate(ZONES):
                for step, timestamp in enumerate(timestamps):
                    observed_target = float(actual[day_index, zone_index, step])
                    row: dict[str, Any] = {
                        "task": task,
                        "split": split,
                        "operating_date": day.date().isoformat(),
                        "issue_time_utc": (
                            day - pd.Timedelta(days=1)
                            if product == "da_lbmp"
                            else day - pd.Timedelta(minutes=5)
                        ).isoformat(),
                        "feature_data_cutoff_utc": (
                            day - pd.Timedelta(days=1, minutes=15)
                            if product == "da_lbmp"
                            else day - pd.Timedelta(minutes=15)
                        ).isoformat(),
                        "timestamp_utc": timestamp,
                        "zone": zone,
                        "observed_target": observed_target,
                        "realized_target_available": bool(np.isfinite(observed_target)),
                        "day_realized_target_complete": bool(
                            quality["realized_target_complete"]
                        ),
                        "eligible_for_locked_test_scoring": bool(
                            quality["eligible_for_locked_test_scoring"]
                        ),
                        "missing_realized_target_interval_count": int(
                            quality["missing_realized_target_interval_count"]
                        ),
                        "missing_realized_target_value_count": int(
                            quality["missing_realized_target_value_count"]
                        ),
                        "missing_realized_targets_imputed": False,
                        "forecast_role": (
                            "deployment_counterfactual"
                            if split == "deployment"
                            else "historical_evaluation"
                        ),
                        "selected_seed": int(model_results["dgpt_e"]["selected_seed"]),
                        "model_class": "dcvpp.models.energy.dgpt_e.DGPTE",
                        "paper_evidence": True,
                    }
                    for index, level in enumerate(QUANTILES):
                        suffix = f"q{int(round(float(level) * 100)):02d}"
                        row[f"raw_{suffix}"] = float(raw[day_index, zone_index, step, index])
                        row[suffix] = float(calibrated[day_index, zone_index, step, index])
                    records.append(row)
        return pd.DataFrame(records)

    dgpt_best = model_results["dgpt_e"]
    selected_model = _energy_factory(
        "dgpt_e", history=history, horizon=horizon, sites=len(ZONES), quantiles=len(QUANTILES)
    ).to(selected_device)
    checkpoint = torch.load(dgpt_best["checkpoint_path"], map_location="cpu", weights_only=False)
    selected_model.load_state_dict(checkpoint["best_model_state"])
    validation_raw = original(
        _energy_predict(
            selected_model, x_validation, device=selected_device,
            batch_size=int(dgpt_best["batch_size"]),
        )
    )
    test_raw = original(
        _energy_predict(
            selected_model, x_test, device=selected_device,
            batch_size=int(dgpt_best["batch_size"]),
        )
    )
    deployment_raw = original(
        _energy_predict(
            selected_model,
            x_deployment,
            device=selected_device,
            batch_size=int(dgpt_best["batch_size"]),
        )
    )
    deployment_calibrated = apply_additive_conformal_quantiles(
        deployment_raw,
        dgpt_best["calibration"],
    )
    predictions = pd.concat(
        [
            long_predictions(validation_days, actual_validation, validation_raw, selected_validation, "validation"),
            long_predictions(test_days, actual_test, test_raw, selected_test, "locked_test"),
            long_predictions(
                deployment_days,
                actual_deployment,
                deployment_raw,
                deployment_calibrated,
                "deployment",
                target_quality=deployment_target_quality,
            ),
        ],
        ignore_index=True,
    )
    prediction_path = output_root / "predictions" / f"{task}_dgpt_e_quantiles.parquet"
    prediction_artifact = atomic_parquet(
        predictions,
        prediction_path,
        metadata={
            "model_class": "dcvpp.models.energy.dgpt_e.DGPTE",
            "architecture": PAPER_DGPT_CONTRACT,
            "selected_seed": dgpt_best["selected_seed"],
            "locked_test_used_for_selection": False,
            "quantiles": QUANTILES.tolist(),
            "calibration_split": "2024_validation_only",
            "deployment_target_year": 2025,
            "deployment_predictions_use_realized_targets": False,
            "missing_realized_targets_imputed": False,
            "paper_evidence": True,
        },
    )
    baseline_path = output_root / "metrics" / f"{task}_paper_energy_leaderboard.json"
    atomic_json(
        baseline_path,
        {
            "task": task,
            "selection_metric": "validation_mean_weighted_pinball",
            "locked_test_used_for_selection": False,
            "deployable_baselines": baseline_results,
            "oracle_future_realized_weather": {
                "status": "not_ranked",
                "deployable": False,
                "reason": "oracle future-realized-weather is intentionally excluded from H1 ranking",
            },
            "dgpt_e": dgpt_best,
        },
    )
    return {
        "model_class": "dcvpp.models.energy.dgpt_e.DGPTE",
        "architecture_scope": "full_preregistered_paper_architecture",
        "architecture": PAPER_DGPT_CONTRACT,
        "full_preregistered_dgpt_e_architecture_completed": True,
        "paper_evidence": True,
        "seed_trials": dgpt_best["seed_trials"],
        "selected_seed": int(dgpt_best["selected_seed"]),
        "selected_epoch": int(dgpt_best["selected_epoch"]),
        "validation_metrics": dgpt_best["validation_metrics"],
        "locked_test_metrics": dgpt_best["locked_test_metrics"],
        "baseline_results": baseline_results,
        "baseline_leaderboard_path": baseline_path.as_posix(),
        "prediction_artifact": prediction_artifact,
        "sequence_counts": {key: len(value[0]) for key, value in sequences.items()},
        "deployment_coverage": {
            "target_year": 2025,
            "requested_day_count": 365,
            "generated_day_count": len(deployment_days),
            "realized_target_complete_day_count": int(
                sum(bool(item["realized_target_complete"]) for item in deployment_target_quality)
            ),
            "missing_realized_target_day_count": int(
                sum(not bool(item["realized_target_complete"]) for item in deployment_target_quality)
            ),
            "missing_realized_target_dates": [
                item for item in deployment_target_quality
                if not bool(item["realized_target_complete"])
            ],
            "deployment_predictions_use_realized_targets": False,
            "missing_realized_targets_imputed": False,
            "history_policy": "issue-time-available per-zone forward fill then training-center fallback",
        },
        "history_steps": history,
        "horizon_steps": horizon,
        "quantiles": QUANTILES.tolist(),
        "resume_fingerprint": data_fingerprint,
        "checkpoint_policy": "atomic latest checkpoint written after every completed epoch",
        "max_epochs": epochs,
        "early_stopping_patience": 10,
        "locked_test_used_for_selection_or_calibration": False,
        "history_missingness": {
            "input_interval_count_with_any_missing_zone": int(history_missing.any(axis=1).sum()),
            "causal_imputation": "per-zone forward fill then training-center zero fallback",
            "target_missing_intervals_are_never_imputed": True,
        },
    }


class DiscreteWorkloadDistributionModel(nn.Module):
    """Aggregate recurrent/patch baseline with the same valid distribution heads."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        horizon: int,
        cohorts: int,
        encoder_type: str,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.cohorts = cohorts
        self.encoder_type = encoder_type
        if encoder_type == "gru":
            self.encoder: nn.Module = nn.GRU(
                input_size + 1, hidden_size, num_layers=2, dropout=0.1, batch_first=True
            )
            encoded_width = hidden_size
        elif encoder_type == "patchtst":
            self.patch = nn.Conv1d(input_size + 1, 96, kernel_size=16, stride=8)
            layer = nn.TransformerEncoderLayer(
                96, nhead=4, dim_feedforward=192, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            encoded_width = 96
        else:
            raise ValueError("workload baseline encoder must be gru or patchtst")
        output_size = cohorts * horizon * 2
        self.count_head = nn.Linear(encoded_width, output_size)
        self.lognormal_head = nn.Linear(encoded_width, output_size)
        self.gamma_head = nn.Linear(encoded_width, output_size)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.cat([x, torch.log1p(dt)], dim=-1)
        if self.encoder_type == "gru":
            sequence, _ = self.encoder(values)
            encoded = sequence[:, -1]
        else:
            tokens = self.patch(values.transpose(1, 2)).transpose(1, 2)
            encoded = self.encoder(tokens).mean(dim=1)
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


def _workload_sequences(
    panel: pd.DataFrame,
    *,
    history: int,
    horizon: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]],
    dict[str, Any],
]:
    frame = panel.sort_values("timestamp_utc", kind="stable").reset_index(drop=True).copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame["timestamp_utc"].diff().dropna().ne(pd.Timedelta(minutes=15)).any():
        raise ValueError("CfC-W requires a complete chronological 15-minute grid")
    count_columns = [f"cohort_{index}_count" for index in range(len(COHORTS))]
    volume_columns = [f"cohort_{index}_volume" for index in range(len(COHORTS))]
    if not set(count_columns + volume_columns).issubset(frame):
        for column in count_columns + volume_columns:
            frame[column] = 0.0
        frame[count_columns[-1]] = pd.to_numeric(frame["submitted_job_count"], errors="raise")
        frame[volume_columns[-1]] = pd.to_numeric(
            frame["submitted_nodes_requested"], errors="raise"
        )
        cohort_fallback = True
    else:
        cohort_fallback = False
    counts = frame[count_columns].to_numpy(dtype="float32")
    volumes = frame[volume_columns].to_numpy(dtype="float32")
    if (counts < 0).any() or (volumes < 0).any():
        raise ValueError("workload count and volume targets must be nonnegative")
    base_features = np.concatenate([np.log1p(counts), np.log1p(volumes)], axis=1)
    timestamp = frame["timestamp_utc"].dt.tz_convert("America/Denver")
    hour = timestamp.dt.hour.to_numpy() + timestamp.dt.minute.to_numpy() / 60.0
    dow = timestamp.dt.dayofweek.to_numpy()
    calendar = np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
        ]
    ).astype("float32")
    raw_features = np.concatenate([base_features, calendar], axis=1).astype("float32")
    train_end = math.floor(len(frame) * 0.60)
    validation_end = math.floor(len(frame) * 0.80)
    center = raw_features[:train_end].mean(axis=0)
    scale = raw_features[:train_end].std(axis=0)
    scale[scale == 0] = 1.0
    features = ((raw_features - center) / scale).astype("float32")
    if "mean_event_elapsed_hours" in frame:
        elapsed = pd.to_numeric(frame["mean_event_elapsed_hours"], errors="coerce").to_numpy()
    elif "time_since_last_event_hours" in frame:
        elapsed = pd.to_numeric(frame["time_since_last_event_hours"], errors="coerce").to_numpy()
    else:
        elapsed = np.empty(len(frame), dtype="float64")
        running = 0.25
        for index, event_count in enumerate(counts.sum(axis=1)):
            elapsed[index] = running
            running = 0.25 if event_count > 0 else min(running + 0.25, 168.0)
    elapsed = np.nan_to_num(elapsed, nan=0.25, posinf=168.0, neginf=0.25)
    elapsed = np.clip(elapsed, 1.0 / 3600.0, 168.0).astype("float32")[:, None]
    positions = np.flatnonzero(
        frame["timestamp_utc"].dt.hour.eq(0) & frame["timestamp_utc"].dt.minute.eq(0)
    )
    buckets: dict[str, list[Any]] = {
        key: [[], [], [], [], []] for key in ("train", "validation", "locked_test")
    }
    for position in positions:
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
        buckets[split][0].append(features[position - history : position])
        buckets[split][1].append(elapsed[position - history : position])
        buckets[split][2].append(counts[position : position + horizon].T)
        buckets[split][3].append(volumes[position : position + horizon].T)
        buckets[split][4].append(frame.loc[position, "timestamp_utc"])
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]] = {}
    for split, values in buckets.items():
        if not values[0]:
            raise ValueError(f"CfC-W {split} chronological split is empty")
        result[split] = (
            np.stack(values[0]).astype("float32"),
            np.stack(values[1]).astype("float32"),
            np.stack(values[2]).astype("float32"),
            np.stack(values[3]).astype("float32"),
            list(values[4]),
        )
    return result, {
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "feature_count": features.shape[1],
        "event_elapsed_time_explicit": True,
        "event_representation": "submission-event cohort aggregates with explicit elapsed hours",
        "cohort_names": list(COHORTS),
        "cohort_fallback_to_elastic_batch": cohort_fallback,
        "work_volume_target_fields": (
            sorted(frame["work_volume_target_field"].dropna().astype(str).unique().tolist())
            if "work_volume_target_field" in frame
            else ["nodes_req"]
        ),
        "work_volume_is_encoder_feature": False,
        "split_policy": "chronological 60/20/20 with complete forecast horizons",
        "train_end_exclusive_utc": frame.loc[train_end, "timestamp_utc"].isoformat(),
        "validation_end_exclusive_utc": frame.loc[validation_end, "timestamp_utc"].isoformat(),
    }


def _event_level_cfc_sequences(
    aggregate_sequences: Mapping[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]],
    ],
    *,
    event_artifact_path: Path,
    event_artifact_sha256: str,
    history_events: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]],
    dict[str, Any],
]:
    """Build exact pre-origin event histories without loading the full trace.

    DuckDB materializes only the ordered event projection and returns the last
    ``history_events`` rows for each daily origin.  No realized execution,
    completion, energy, or node-use field is selected as an encoder feature.
    """
    path = Path(event_artifact_path)
    if not path.is_file() or sha256_file(path) != event_artifact_sha256:
        raise ValueError("CfC-W physical workload event artifact hash mismatch")
    origins: list[dict[str, Any]] = []
    origin_lookup: list[tuple[str, int]] = []
    for split in ("train", "validation", "locked_test"):
        days = aggregate_sequences[split][4]
        for split_index, day in enumerate(days):
            origin_id = len(origins)
            origins.append(
                {
                    "origin_id": origin_id,
                    "origin_utc": pd.Timestamp(day).tz_convert("UTC"),
                }
            )
            origin_lookup.append((split, split_index))
    origin_frame = pd.DataFrame(origins)
    sql_path = path.as_posix().replace("'", "''")
    connection = duckdb.connect()
    try:
        connection.register("forecast_origins", origin_frame)
        history_rows = connection.execute(
            f"""
            WITH ordered AS MATERIALIZED (
              SELECT row_number() OVER (ORDER BY submit_time_utc, job_key) AS event_row,
                     submit_time_utc,
                     wallclock_requested_hours,
                     gpus_requested,
                     partition,
                     qos,
                     lag(submit_time_utc) OVER (
                       ORDER BY submit_time_utc, job_key
                     ) AS previous_submit_time_utc
              FROM read_parquet('{sql_path}', hive_partitioning=false)
              WHERE submit_time_utc IS NOT NULL
            ), origin_last AS (
              SELECT o.origin_id, o.origin_utc, e.event_row AS last_event_row
              FROM forecast_origins o
              ASOF LEFT JOIN ordered e
                ON o.origin_utc > e.submit_time_utc
            )
            SELECT o.origin_id, o.origin_utc, e.event_row, e.submit_time_utc,
                   e.previous_submit_time_utc, e.wallclock_requested_hours,
                   e.gpus_requested, e.partition, e.qos
            FROM origin_last o
            JOIN ordered e
              ON e.event_row BETWEEN o.last_event_row - {int(history_events)} + 1
                                 AND o.last_event_row
            ORDER BY o.origin_id, e.event_row
            """
        ).fetch_df()
    finally:
        connection.close()
    counts = history_rows.groupby("origin_id", sort=True).size().reindex(range(len(origins)), fill_value=0)
    if not counts.eq(history_events).all():
        missing = counts.loc[~counts.eq(history_events)].head().to_dict()
        raise ValueError(f"CfC-W event histories do not contain {history_events} events: {missing}")
    submit = pd.to_datetime(history_rows["submit_time_utc"], utc=True, errors="raise")
    origin = pd.to_datetime(history_rows["origin_utc"], utc=True, errors="raise")
    if not (submit < origin).all():
        raise ValueError("CfC-W event encoder contains an event at or after forecast origin")
    previous = pd.to_datetime(history_rows["previous_submit_time_utc"], utc=True, errors="coerce")
    elapsed = (submit - previous).dt.total_seconds().div(3600.0)
    elapsed = elapsed.fillna(1.0 / 3600.0).clip(1.0 / 3600.0, 168.0).to_numpy(dtype="float32")
    partition = history_rows["partition"].fillna("").astype(str)
    qos = history_rows["qos"].fillna("").astype(str)
    wallclock = pd.to_numeric(history_rows["wallclock_requested_hours"], errors="coerce")
    gpu = pd.to_numeric(history_rows["gpus_requested"], errors="coerce")
    text = (partition + " " + qos).str.lower()
    cohort = np.select(
        [
            text.str.contains("interactive|debug|urgent|high", regex=True),
            gpu.fillna(0).gt(0),
            wallclock.le(2),
        ],
        [0, 2, 1],
        default=3,
    ).astype("int64")
    one_hot = np.eye(len(COHORTS), dtype="float32")[cohort]
    local = submit.dt.tz_convert("America/Denver")
    hour = local.dt.hour.to_numpy() + local.dt.minute.to_numpy() / 60.0
    dow = local.dt.dayofweek.to_numpy()
    partition_hash = (
        pd.util.hash_pandas_object(partition, index=False).to_numpy(dtype="uint64") % 1024
    ).astype("float32") / 1023.0
    qos_hash = (
        pd.util.hash_pandas_object(qos, index=False).to_numpy(dtype="uint64") % 256
    ).astype("float32") / 255.0
    features_flat = np.column_stack(
        [
            np.log1p(wallclock.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
            np.log1p(gpu.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
            partition_hash,
            qos_hash,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            wallclock.isna().to_numpy(dtype="float32"),
            gpu.isna().to_numpy(dtype="float32"),
            one_hot,
        ]
    ).astype("float32")
    feature_count = features_flat.shape[1]
    features = features_flat.reshape(len(origins), history_events, feature_count)
    elapsed_values = elapsed.reshape(len(origins), history_events, 1)
    train_origin_count = len(aggregate_sequences["train"][0])
    center = features[:train_origin_count].mean(axis=(0, 1))
    scale = features[:train_origin_count].std(axis=(0, 1))
    scale[scale == 0] = 1.0
    features = ((features - center) / scale).astype("float32")
    output: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]],
    ] = {}
    cursor = 0
    for split in ("train", "validation", "locked_test"):
        count = len(aggregate_sequences[split][0])
        source = aggregate_sequences[split]
        output[split] = (
            features[cursor : cursor + count],
            elapsed_values[cursor : cursor + count],
            source[2],
            source[3],
            source[4],
        )
        cursor += count
    return output, {
        "representation": "event_level_origin_available_features",
        "history_events": int(history_events),
        "explicit_elapsed_time": True,
        "input_features": [
            "log1p_requested_wallclock_hours",
            "log1p_gpus_requested",
            "partition_hash",
            "qos_hash",
            "submission_hour_sin",
            "submission_hour_cos",
            "submission_dow_sin",
            "submission_dow_cos",
            "wallclock_missing",
            "gpu_missing",
            *[f"cohort_{name}" for name in COHORTS],
        ],
        "forbidden_realized_features": [
            "wallclock_used_hours", "start_time_utc", "end_time_utc", "state",
            "nodes_used", "processors_used", "node_energy", "gpu_energy",
        ],
        "event_artifact_path": path.as_posix(),
        "event_artifact_sha256": event_artifact_sha256,
        "origin_count": len(origins),
        "max_event_at_or_after_origin": 0,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
    }


def _workload_loss(
    output: Mapping[str, torch.Tensor],
    count_target: torch.Tensor,
    volume_target: torch.Tensor,
) -> torch.Tensor:
    count = negative_binomial_nll(
        count_target, output["count_mean"], output["count_dispersion"]
    )
    lognormal = lognormal_volume_nll(
        volume_target, output["volume_log_mu"], output["volume_log_sigma"]
    )
    gamma = gamma_volume_nll(
        volume_target, output["volume_gamma_shape"], output["volume_gamma_rate"]
    )
    return count + 0.5 * (lognormal + gamma)


@torch.no_grad()
def _workload_predict(
    model: nn.Module,
    x: np.ndarray,
    dt: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(dt)),
        batch_size=batch_size,
        shuffle=False,
    )
    for x_batch, dt_batch in loader:
        output = model(x_batch.to(device), dt_batch.to(device))
        for key, value in output.items():
            collected.setdefault(key, []).append(value.float().cpu().numpy())
    return {key: np.concatenate(values) for key, values in collected.items()}


def _select_volume_distribution(
    actual_volume: np.ndarray,
    prediction: Mapping[str, np.ndarray],
) -> tuple[str, dict[str, float]]:
    lognormal = positive_volume_nll(
        actual_volume,
        {"log_mu": prediction["volume_log_mu"], "sigma": prediction["volume_log_sigma"]},
        distribution="lognormal",
    )
    gamma = positive_volume_nll(
        actual_volume,
        {"shape": prediction["volume_gamma_shape"], "rate": prediction["volume_gamma_rate"]},
        distribution="gamma",
    )
    return ("lognormal" if lognormal <= gamma else "gamma"), {
        "lognormal_positive_nll": lognormal,
        "gamma_positive_nll": gamma,
    }


def _distribution_score(
    actual_count: np.ndarray,
    actual_volume: np.ndarray,
    prediction: Mapping[str, np.ndarray],
) -> tuple[float, str, dict[str, Any]]:
    count = negative_binomial_metrics(
        actual_count, prediction["count_mean"], prediction["count_dispersion"]
    )
    selected, candidate = _select_volume_distribution(actual_volume, prediction)
    score = float(count["negative_binomial_nll"] + min(candidate.values()))
    return score, selected, {"count": count, "volume_candidates": candidate}


def _workload_factory(
    model_name: str,
    *,
    input_size: int,
    hidden_size: int,
    horizon: int,
) -> nn.Module:
    if model_name == "cfc_w":
        return CfCW(
            input_size=input_size, hidden_size=hidden_size, horizon=horizon, cohorts=len(COHORTS)
        )
    if model_name == "gru":
        return DiscreteWorkloadDistributionModel(
            input_size=input_size, hidden_size=hidden_size, horizon=horizon,
            cohorts=len(COHORTS), encoder_type="gru",
        )
    if model_name == "aggregate_patchtst":
        return DiscreteWorkloadDistributionModel(
            input_size=input_size, hidden_size=hidden_size, horizon=horizon,
            cohorts=len(COHORTS), encoder_type="patchtst",
        )
    raise ValueError(f"unknown workload distribution model: {model_name}")


def _fit_workload_seed(
    *,
    model_name: str,
    seed: int,
    input_size: int,
    hidden_size: int,
    horizon: int,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]],
    epochs: int,
    batch_sizes: Sequence[int],
    learning_rate: float,
    device: torch.device,
    output_root: Path,
    resume_fingerprint: str,
    paper_evidence: bool = True,
) -> dict[str, Any]:
    x_train, dt_train, count_train, volume_train, _ = train
    x_validation, dt_validation, count_validation, volume_validation, _ = validation
    last_error: Exception | None = None
    for batch_size in batch_sizes:
        model = _workload_factory(
            model_name, input_size=input_size, hidden_size=hidden_size, horizon=horizon
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        architecture = (
            model.architecture_contract
            if hasattr(model, "architecture_contract")
            else {
                "encoder": model_name,
                "hidden_size": hidden_size,
                "horizon": horizon,
                "cohorts": len(COHORTS),
                "count_distribution": "negative_binomial",
                "volume_candidates": "lognormal_and_gamma",
            }
        )
        fingerprint = sha256_json(
            {
                "parent": resume_fingerprint,
                "model": model_name,
                "seed": seed,
                "architecture": architecture,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )
        checkpoint_path = (
            output_root / "checkpoints" / "nlr_workload" / model_name / f"seed_{seed}_latest.pt"
        )
        start_epoch = 1
        best_epoch = 0
        best_metric = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        resumed_from_epoch = 0
        selected_distribution = "lognormal"
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("resume_fingerprint") == fingerprint:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if checkpoint.get("gradient_scaler_state"):
                    scaler.load_state_dict(checkpoint["gradient_scaler_state"])
                start_epoch = int(checkpoint["epoch"]) + 1
                resumed_from_epoch = int(checkpoint["epoch"])
                best_epoch = int(checkpoint["best_epoch"])
                best_metric = float(checkpoint["best_validation_distribution_score"])
                best_state = checkpoint.get("best_model_state")
                selected_distribution = str(checkpoint.get("selected_volume_distribution", "lognormal"))
                if bool(checkpoint.get("completed")):
                    if best_state is None:
                        raise ValueError("completed workload checkpoint lacks its best state")
                    model.load_state_dict(best_state)
                    return {
                        "model": model,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "selected_volume_distribution": selected_distribution,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "epochs_written": int(checkpoint["epoch"]),
                        "resumed_from_epoch": resumed_from_epoch,
                        "batch_size": batch_size,
                        "resume_fingerprint": fingerprint,
                        "architecture": architecture,
                    }
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train), torch.from_numpy(dt_train),
                torch.from_numpy(count_train), torch.from_numpy(volume_train),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=device.type == "cuda",
        )
        no_improvement = max(0, resumed_from_epoch - best_epoch)
        try:
            for epoch in range(start_epoch, epochs + 1):
                model.train()
                for x_batch, dt_batch, count_batch, volume_batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        output = model(x_batch.to(device), dt_batch.to(device))
                        loss = _workload_loss(
                            output, count_batch.to(device), volume_batch.to(device)
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                prediction = _workload_predict(
                    model, x_validation, dt_validation, device=device, batch_size=batch_size
                )
                metric, candidate, _ = _distribution_score(
                    count_validation, volume_validation, prediction
                )
                if not math.isfinite(metric):
                    raise RuntimeError("workload validation distribution score is non-finite")
                if metric < best_metric - 1e-8:
                    best_metric = metric
                    best_epoch = epoch
                    selected_distribution = candidate
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    no_improvement = 0
                else:
                    no_improvement += 1
                completed = epoch == epochs or no_improvement >= 10
                payload = {
                    "checkpoint_schema_version": 2,
                    "paper_evidence": bool(paper_evidence),
                    "model_name": model_name,
                    "model_class": f"{type(model).__module__}.{type(model).__name__}",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "gradient_scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_distribution_score": best_metric,
                    "best_model_state": best_state,
                    "selected_volume_distribution": selected_distribution,
                    "seed": seed,
                    "resume_fingerprint": fingerprint,
                    "architecture": architecture,
                    "batch_size": batch_size,
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "completed": completed,
                    "written_at_utc": utc_now(),
                }
                _atomic_torch(payload, checkpoint_path)
                if completed:
                    break
            if best_state is None:
                raise RuntimeError(f"{model_name} produced no validation checkpoint")
            model.load_state_dict(best_state)
            latest = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            return {
                "model": model,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "selected_volume_distribution": selected_distribution,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "epochs_written": int(latest["epoch"]),
                "resumed_from_epoch": resumed_from_epoch,
                "batch_size": batch_size,
                "resume_fingerprint": fingerprint,
                "architecture": architecture,
            }
        except torch.OutOfMemoryError as exc:
            last_error = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            checkpoint_path.unlink(missing_ok=True)
            continue
    raise RuntimeError(
        f"{model_name} exhausted batch fallback {list(batch_sizes)}: {last_error}"
    )


def _volume_point(
    prediction: Mapping[str, np.ndarray],
    distribution: str,
) -> np.ndarray:
    size = np.asarray(prediction["count_dispersion"], dtype="float64").clip(1e-8)
    mean_count = np.asarray(prediction["count_mean"], dtype="float64").clip(1e-8)
    nonzero_probability = 1.0 - np.power(size / (size + mean_count), size)
    if distribution == "lognormal":
        conditional = np.exp(
            prediction["volume_log_mu"] + 0.5 * np.square(prediction["volume_log_sigma"])
        )
    else:
        conditional = prediction["volume_gamma_shape"] / np.maximum(
            prediction["volume_gamma_rate"], 1e-8
        )
    return nonzero_probability * conditional


def _cohort_calibration(
    actual: np.ndarray,
    raw: np.ndarray,
    *,
    minimum_samples: int,
) -> dict[str, Any]:
    corrections = []
    records = []
    for cohort_index, cohort in enumerate(COHORTS):
        calibrator = fit_additive_conformal_quantiles(
            actual[:, cohort_index, :].reshape(-1),
            raw[:, cohort_index, :, :].reshape(-1, len(QUANTILES)),
            quantiles=QUANTILES,
            minimum_samples=minimum_samples,
        )
        corrections.append(np.asarray(calibrator["corrections"]))
        records.append(
            {"cohort": cohort, **{key: value for key, value in calibrator.items() if key != "corrections"}}
        )
    return {"corrections": np.stack(corrections), "records": records}


def _apply_cohort_calibration(raw: np.ndarray, calibrator: Mapping[str, Any]) -> np.ndarray:
    correction = np.asarray(calibrator["corrections"], dtype="float64")
    return np.sort(np.asarray(raw, dtype="float64") + correction[None, :, None, :], axis=-1)


def _workload_metrics(
    actual_count: np.ndarray,
    actual_volume: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    *,
    volume_distribution: str,
    count_quantiles: np.ndarray,
    volume_quantiles: np.ndarray,
) -> dict[str, Any]:
    count = negative_binomial_metrics(
        actual_count, prediction["count_mean"], prediction["count_dispersion"]
    )
    volume_point = _volume_point(prediction, volume_distribution)
    volume = regression_metrics(actual_volume.ravel(), volume_point.ravel())
    count_interval = quantile_metrics(actual_count, count_quantiles, quantiles=QUANTILES)
    volume_interval = quantile_metrics(actual_volume, volume_quantiles, quantiles=QUANTILES)
    volume_nll = positive_volume_nll(
        actual_volume,
        (
            {"log_mu": prediction["volume_log_mu"], "sigma": prediction["volume_log_sigma"]}
            if volume_distribution == "lognormal"
            else {"shape": prediction["volume_gamma_shape"], "rate": prediction["volume_gamma_rate"]}
        ),
        distribution=volume_distribution,
    )
    return {
        "count": count,
        "volume": volume,
        "volume_positive_nll": volume_nll,
        "distribution_score": float(count["negative_binomial_nll"] + volume_nll),
        "count_interval": count_interval,
        "volume_interval": volume_interval,
        "interval_calibration_error": float(
            0.5 * (
                count_interval["absolute_calibration_error"]
                + volume_interval["absolute_calibration_error"]
            )
        ),
        "by_cohort": [
            {
                "cohort": cohort,
                "count_nll": negative_binomial_metrics(
                    actual_count[:, index, :],
                    prediction["count_mean"][:, index, :],
                    prediction["count_dispersion"][:, index, :],
                )["negative_binomial_nll"],
                "count_coverage_90": quantile_metrics(
                    actual_count[:, index, :], count_quantiles[:, index, :, :], quantiles=QUANTILES
                )["coverage_90"],
                "volume_coverage_90": quantile_metrics(
                    actual_volume[:, index, :], volume_quantiles[:, index, :, :], quantiles=QUANTILES
                )["coverage_90"],
            }
            for index, cohort in enumerate(COHORTS)
        ],
    }


def _distribution_quantiles(
    prediction: Mapping[str, np.ndarray],
    volume_distribution: str,
) -> tuple[np.ndarray, np.ndarray]:
    count = negative_binomial_quantiles(
        prediction["count_mean"], prediction["count_dispersion"], quantiles=QUANTILES
    )
    volume_parameters = (
        {"log_mu": prediction["volume_log_mu"], "sigma": prediction["volume_log_sigma"]}
        if volume_distribution == "lognormal"
        else {"shape": prediction["volume_gamma_shape"], "rate": prediction["volume_gamma_rate"]}
    )
    volume = positive_volume_quantiles(
        prediction["count_mean"], prediction["count_dispersion"], volume_parameters,
        distribution=volume_distribution, quantiles=QUANTILES,
    )
    return count, volume


def _empirical_baseline_prediction(
    count_mean: np.ndarray,
    volume_mean: np.ndarray,
    *,
    dispersion: np.ndarray,
    log_sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    mean = np.maximum(count_mean, 1e-4)
    shape = np.maximum(1.0 / np.square(log_sigma), 1e-4)
    rate = shape / np.maximum(volume_mean, 1e-4)
    return {
        "count_mean": mean,
        "count_dispersion": np.broadcast_to(dispersion, mean.shape).copy(),
        "volume_log_mu": np.log(np.maximum(volume_mean, 1e-4)) - 0.5 * np.square(log_sigma),
        "volume_log_sigma": np.broadcast_to(log_sigma, volume_mean.shape).copy(),
        "volume_gamma_shape": np.broadcast_to(shape, volume_mean.shape).copy(),
        "volume_gamma_rate": rate,
    }


def _tabular_workload_means(
    *,
    train_x: np.ndarray,
    train_count: np.ndarray,
    train_volume: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    feature_center: Sequence[float],
    feature_scale: Sequence[float],
    model_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cohort_count = len(COHORTS)
    center = np.asarray(feature_center, dtype="float64")
    scale = np.asarray(feature_scale, dtype="float64")

    def histories(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = np.expm1(
            x[..., :cohort_count] * scale[None, None, :cohort_count]
            + center[None, None, :cohort_count]
        )
        volumes = np.expm1(
            x[..., cohort_count : 2 * cohort_count]
            * scale[None, None, cohort_count : 2 * cohort_count]
            + center[None, None, cohort_count : 2 * cohort_count]
        )
        return np.maximum(counts, 0), np.maximum(volumes, 0)

    def design(x: np.ndarray) -> np.ndarray:
        counts, volumes = histories(x)
        rows = np.zeros((len(x), cohort_count, 96, 14), dtype="float32")
        recent_count = counts[:, -96:, :].mean(axis=1)
        recent_volume = volumes[:, -96:, :].mean(axis=1)
        for step in range(96):
            rows[:, :, step, 0] = counts[:, -1, :]
            rows[:, :, step, 1] = volumes[:, -1, :]
            rows[:, :, step, 2] = counts[:, -96 + step, :]
            rows[:, :, step, 3] = volumes[:, -96 + step, :]
            rows[:, :, step, 4] = counts[:, -672 + step, :]
            rows[:, :, step, 5] = volumes[:, -672 + step, :]
            rows[:, :, step, 6] = recent_count
            rows[:, :, step, 7] = recent_volume
            rows[:, :, step, 8] = math.sin(2 * math.pi * step / 96)
            rows[:, :, step, 9] = math.cos(2 * math.pi * step / 96)
            for cohort_index in range(cohort_count):
                rows[:, cohort_index, step, 10 + cohort_index] = 1.0
        return rows.reshape(-1, rows.shape[-1])

    validation_samples = len(validation_x)
    test_samples = len(test_x)
    x_train = design(train_x)
    y_count = train_count.reshape(-1)
    y_volume = train_volume.reshape(-1)
    x_validation = design(validation_x)
    x_test = design(test_x)
    if model_name == "negative_binomial_regression":
        from sklearn.linear_model import PoissonRegressor, TweedieRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        count_model = make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=300)
        )
        volume_model = make_pipeline(
            StandardScaler(),
            TweedieRegressor(power=1.5, alpha=1e-4, link="log", max_iter=300),
        )
    elif model_name == "lightgbm_count_volume":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("workload LightGBM baseline requires lightgbm") from exc
        common = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": seed,
            "n_jobs": 8,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        count_model = lgb.LGBMRegressor(objective="poisson", **common)
        volume_model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, **common)
    else:
        raise ValueError(f"unsupported tabular workload model: {model_name}")
    count_model.fit(x_train, y_count)
    volume_model.fit(x_train, y_volume)
    shape_validation = (validation_samples, cohort_count, 96)
    shape_test = (test_samples, cohort_count, 96)
    validation_count = np.maximum(count_model.predict(x_validation).reshape(shape_validation), 1e-4)
    validation_volume = np.maximum(volume_model.predict(x_validation).reshape(shape_validation), 1e-4)
    test_count = np.maximum(count_model.predict(x_test).reshape(shape_test), 1e-4)
    test_volume = np.maximum(volume_model.predict(x_test).reshape(shape_test), 1e-4)
    return validation_count, validation_volume, test_count, test_volume, {
        "fit_rows": int(len(x_train)),
        "deterministic": True,
        "implementation": (
            "sklearn Poisson mean plus validation-frozen NB dispersion and Tweedie volume"
            if model_name == "negative_binomial_regression"
            else "LightGBM Poisson count and Tweedie volume"
        ),
    }


def fit_paper_cfc_w_five_seeds(
    workload_panel: pd.DataFrame,
    *,
    seeds: Sequence[int],
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
    seed_values = _paper_seed_gate(seeds, paper_evidence=paper_evidence)
    _paper_shape_gate(history, horizon, paper_evidence=paper_evidence)
    if paper_evidence and hidden_size not in {64, 96}:
        raise ValueError("paper CfC-W hidden size must be 64 or 96")
    if paper_evidence and epochs != 100:
        raise ValueError("paper CfC-W requires max_epochs=100")
    aggregate_sequences, sequence_manifest = _workload_sequences(
        workload_panel, history=history, horizon=horizon
    )
    if event_artifact_path is not None and event_artifact_sha256 is not None:
        cfc_sequences, event_manifest = _event_level_cfc_sequences(
            aggregate_sequences,
            event_artifact_path=event_artifact_path,
            event_artifact_sha256=event_artifact_sha256,
            history_events=history,
        )
    elif paper_evidence:
        raise ValueError("paper CfC-W requires the hash-verified physical workload event contract")
    else:
        cfc_sequences = dict(aggregate_sequences)
        event_manifest = {
            "representation": "aggregate_feasibility_fallback",
            "explicit_elapsed_time": True,
            "paper_evidence": False,
        }
    x_train, dt_train, count_train, volume_train, _ = aggregate_sequences["train"]
    x_validation, dt_validation, count_validation, volume_validation, validation_days = aggregate_sequences["validation"]
    x_test, dt_test, count_test, volume_test, test_days = aggregate_sequences["locked_test"]
    data_fingerprint = sha256_json(
        {
            "parent": resume_fingerprint,
            "history": history,
            "horizon": horizon,
            "train_x": _array_sha256(x_train),
            "train_dt": _array_sha256(dt_train),
            "train_count": _array_sha256(count_train),
            "train_volume": _array_sha256(volume_train),
            "validation_x": _array_sha256(x_validation),
            "validation_count": _array_sha256(count_validation),
            "sequence_manifest": sequence_manifest,
            "event_manifest": event_manifest,
        }
    )
    selected_device = _device(device_name)
    batch_fallback = tuple(dict.fromkeys(value for value in (batch_size, 16, 8, 4) if value <= batch_size))
    baseline_results: list[dict[str, Any]] = []

    # Deterministic statistical baselines.  Dispersion and positive-volume
    # shape are estimated from training only; validation calibrates intervals.
    train_count_flat = count_train.reshape(-1, len(COHORTS), horizon)
    train_volume_flat = volume_train.reshape(-1, len(COHORTS), horizon)
    count_mean_train = np.maximum(train_count_flat.mean(axis=(0, 2), keepdims=True), 1e-4)
    count_variance_train = train_count_flat.var(axis=(0, 2), keepdims=True)
    nb_dispersion = np.where(
        count_variance_train > count_mean_train,
        np.square(count_mean_train) / np.maximum(count_variance_train - count_mean_train, 1e-6),
        1e6,
    )
    positive_volume = np.where(train_volume_flat > 0, np.log(train_volume_flat), np.nan)
    log_sigma = np.nanstd(positive_volume, axis=(0, 2), keepdims=True)
    log_sigma = np.nan_to_num(log_sigma, nan=1.0, posinf=1.0, neginf=1.0).clip(0.05, 5.0)

    def seasonal_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_counts = np.expm1(
            x[:, -7 * horizon : -6 * horizon, : len(COHORTS)]
            * np.asarray(sequence_manifest["feature_scale"][: len(COHORTS)])[None, None, :]
            + np.asarray(sequence_manifest["feature_center"][: len(COHORTS)])[None, None, :]
        ).transpose(0, 2, 1)
        offset = len(COHORTS)
        raw_volume = np.expm1(
            x[:, -7 * horizon : -6 * horizon, offset : 2 * offset]
            * np.asarray(sequence_manifest["feature_scale"][offset : 2 * offset])[None, None, :]
            + np.asarray(sequence_manifest["feature_center"][offset : 2 * offset])[None, None, :]
        ).transpose(0, 2, 1)
        return np.maximum(raw_counts, 0), np.maximum(raw_volume, 0)

    validation_count_point, validation_volume_point = seasonal_points(x_validation)
    test_count_point, test_volume_point = seasonal_points(x_test)
    tabular_predictions = {
        name: _tabular_workload_means(
            train_x=x_train,
            train_count=count_train,
            train_volume=volume_train,
            validation_x=x_validation,
            test_x=x_test,
            feature_center=sequence_manifest["feature_center"],
            feature_scale=sequence_manifest["feature_scale"],
            model_name=name,
            seed=seed_values[0],
        )
        for name in ("negative_binomial_regression", "lightgbm_count_volume")
    }
    for model_name, dispersion in (
        ("seasonal_poisson", np.full_like(nb_dispersion, 1e6)),
        ("seasonal_negative_binomial", nb_dispersion),
        ("negative_binomial_regression", nb_dispersion),
        ("lightgbm_count_volume", nb_dispersion),
    ):
        if model_name in tabular_predictions:
            validation_count_model, validation_volume_model, test_count_model, test_volume_model, fit_record = tabular_predictions[model_name]
        else:
            validation_count_model = validation_count_point
            validation_volume_model = validation_volume_point
            test_count_model = test_count_point
            test_volume_model = test_volume_point
            fit_record = {"deterministic": True, "implementation": model_name}
        validation_prediction = _empirical_baseline_prediction(
            validation_count_model,
            validation_volume_model,
            dispersion=dispersion,
            log_sigma=log_sigma,
        )
        test_prediction = _empirical_baseline_prediction(
            test_count_model,
            test_volume_model,
            dispersion=dispersion,
            log_sigma=log_sigma,
        )
        selected_distribution, _ = _select_volume_distribution(volume_validation, validation_prediction)
        val_count_q, val_volume_q = _distribution_quantiles(validation_prediction, selected_distribution)
        test_count_q, test_volume_q = _distribution_quantiles(test_prediction, selected_distribution)
        count_calibrator = _cohort_calibration(
            count_validation, val_count_q, minimum_samples=minimum_calibration_samples
        )
        volume_calibrator = _cohort_calibration(
            volume_validation, val_volume_q, minimum_samples=minimum_calibration_samples
        )
        val_count_q = _apply_cohort_calibration(val_count_q, count_calibrator)
        val_volume_q = _apply_cohort_calibration(val_volume_q, volume_calibrator)
        test_count_q = _apply_cohort_calibration(test_count_q, count_calibrator)
        test_volume_q = _apply_cohort_calibration(test_volume_q, volume_calibrator)
        baseline_results.append(
            {
                "model": model_name,
                "deployable": True,
                "deterministic": True,
                "fit_record": fit_record,
                "volume_distribution_selected_on_validation": selected_distribution,
                "validation_metrics": _workload_metrics(
                    count_validation, volume_validation, validation_prediction,
                    volume_distribution=selected_distribution,
                    count_quantiles=val_count_q, volume_quantiles=val_volume_q,
                ),
                "locked_test_metrics": _workload_metrics(
                    count_test, volume_test, test_prediction,
                    volume_distribution=selected_distribution,
                    count_quantiles=test_count_q, volume_quantiles=test_volume_q,
                ),
                "calibration": {
                    "count_by_cohort": count_calibrator["records"],
                    "volume_by_cohort": volume_calibrator["records"],
                },
            }
        )

    model_results: dict[str, dict[str, Any]] = {}
    selected_predictions: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_name in ("gru", "aggregate_patchtst", "cfc_w"):
        model_sequences = cfc_sequences if model_name == "cfc_w" else aggregate_sequences
        model_x_train = model_sequences["train"][0]
        model_x_validation = model_sequences["validation"][0]
        model_dt_validation = model_sequences["validation"][1]
        model_x_test = model_sequences["locked_test"][0]
        model_dt_test = model_sequences["locked_test"][1]
        trials: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for seed in seed_values:
            try:
                _seed_everything(seed)
                fitted = _fit_workload_seed(
                    model_name=model_name,
                    seed=seed,
                    input_size=model_x_train.shape[-1],
                    hidden_size=hidden_size,
                    horizon=horizon,
                    train=model_sequences["train"],
                    validation=model_sequences["validation"],
                    epochs=epochs,
                    batch_sizes=batch_fallback,
                    learning_rate=learning_rate,
                    device=selected_device,
                    output_root=output_root,
                    resume_fingerprint=data_fingerprint,
                )
                validation_prediction = _workload_predict(
                    fitted["model"], model_x_validation, model_dt_validation,
                    device=selected_device, batch_size=fitted["batch_size"],
                )
                score, volume_distribution, validation_distribution = _distribution_score(
                    count_validation, volume_validation, validation_prediction
                )
                trial = {
                    "seed": seed,
                    "status": "completed",
                    "selected_epoch": fitted["best_epoch"],
                    "epochs_written": fitted["epochs_written"],
                    "resumed_from_epoch": fitted["resumed_from_epoch"],
                    "batch_size": fitted["batch_size"],
                    "validation_distribution_score": score,
                    "volume_distribution": volume_distribution,
                    "validation_distribution": validation_distribution,
                    "checkpoint_path": fitted["checkpoint_path"].as_posix(),
                    "checkpoint_sha256": fitted["checkpoint_sha256"],
                    "resume_fingerprint": fitted["resume_fingerprint"],
                }
                trials.append(trial)
                if best is None or score < best["score"]:
                    best = {
                        **fitted,
                        "seed": seed,
                        "score": score,
                        "volume_distribution": volume_distribution,
                        "validation_prediction": validation_prediction,
                    }
            except Exception as exc:
                failure = {
                    "task": "nlr_workload",
                    "model": model_name,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recorded_at_utc": utc_now(),
                }
                failed_seeds.append(failure)
                trials.append({"status": "failed", **failure})
        if best is None:
            raise RuntimeError(f"all five paper seeds failed for nlr_workload/{model_name}")
        test_prediction = _workload_predict(
            best["model"], model_x_test, model_dt_test, device=selected_device,
            batch_size=best["batch_size"],
        )
        val_count_q, val_volume_q = _distribution_quantiles(
            best["validation_prediction"], best["volume_distribution"]
        )
        test_count_q, test_volume_q = _distribution_quantiles(
            test_prediction, best["volume_distribution"]
        )
        count_calibrator = _cohort_calibration(
            count_validation, val_count_q, minimum_samples=minimum_calibration_samples
        )
        volume_calibrator = _cohort_calibration(
            volume_validation, val_volume_q, minimum_samples=minimum_calibration_samples
        )
        val_count_calibrated = _apply_cohort_calibration(val_count_q, count_calibrator)
        val_volume_calibrated = _apply_cohort_calibration(val_volume_q, volume_calibrator)
        test_count_calibrated = _apply_cohort_calibration(test_count_q, count_calibrator)
        test_volume_calibrated = _apply_cohort_calibration(test_volume_q, volume_calibrator)
        record = {
            "model": model_name,
            "model_class": (
                "dcvpp.models.workload.cfc_w.CfCW"
                if model_name == "cfc_w"
                else f"dcvpp.experiments.paper_training.DiscreteWorkloadDistributionModel[{model_name}]"
            ),
            "deployable": True,
            "deterministic": False,
            "seed_trials": trials,
            "selected_seed": int(best["seed"]),
            "selected_epoch": int(best["best_epoch"]),
            "batch_size": int(best["batch_size"]),
            "checkpoint_path": best["checkpoint_path"].as_posix(),
            "checkpoint_sha256": best["checkpoint_sha256"],
            "resume_fingerprint": best["resume_fingerprint"],
            "architecture": best["architecture"],
            "volume_distribution_selected_on_validation": best["volume_distribution"],
            "validation_metrics": _workload_metrics(
                count_validation, volume_validation, best["validation_prediction"],
                volume_distribution=best["volume_distribution"],
                count_quantiles=val_count_calibrated,
                volume_quantiles=val_volume_calibrated,
            ),
            "locked_test_metrics": _workload_metrics(
                count_test, volume_test, test_prediction,
                volume_distribution=best["volume_distribution"],
                count_quantiles=test_count_calibrated,
                volume_quantiles=test_volume_calibrated,
            ),
            "calibration": {
                "count_by_cohort": count_calibrator["records"],
                "volume_by_cohort": volume_calibrator["records"],
            },
        }
        model_results[model_name] = record
        selected_predictions[model_name] = (
            best["validation_prediction"], test_prediction,
            val_count_calibrated, val_volume_calibrated,
            test_count_calibrated, test_volume_calibrated,
        )
        if model_name != "cfc_w":
            baseline_results.append(record)

    cfc_record = model_results["cfc_w"]
    validation_prediction, test_prediction, val_count_q, val_volume_q, test_count_q, test_volume_q = selected_predictions["cfc_w"]

    def long_predictions(
        days: list[pd.Timestamp], counts: np.ndarray, volumes: np.ndarray,
        prediction: Mapping[str, np.ndarray], count_q: np.ndarray, volume_q: np.ndarray,
        split: str,
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        volume_point = _volume_point(
            prediction, cfc_record["volume_distribution_selected_on_validation"]
        )
        for day_index, day in enumerate(days):
            for step, timestamp in enumerate(pd.date_range(day, periods=horizon, freq="15min", tz="UTC")):
                row: dict[str, Any] = {
                    "task": "nlr_workload",
                    "split": split,
                    "operating_date": day.date().isoformat(),
                    "prediction_origin_utc": (day - pd.Timedelta(minutes=15)).isoformat(),
                    "feature_data_cutoff_utc": (day - pd.Timedelta(minutes=15)).isoformat(),
                    "timestamp_utc": timestamp,
                    "observed_job_count": float(counts[day_index, :, step].sum()),
                    "observed_nodes_requested": float(volumes[day_index, :, step].sum()),
                    "observed_work_volume": float(volumes[day_index, :, step].sum()),
                    "predicted_job_count": float(prediction["count_mean"][day_index, :, step].sum()),
                    "predicted_nodes_requested": float(volume_point[day_index, :, step].sum()),
                    "predicted_work_volume": float(volume_point[day_index, :, step].sum()),
                    "selected_seed": cfc_record["selected_seed"],
                    "model_class": "dcvpp.models.workload.cfc_w.CfCW",
                    "paper_evidence": True,
                }
                for index, level in enumerate(QUANTILES):
                    suffix = f"q{int(round(float(level) * 100)):02d}"
                    row[f"count_{suffix}"] = float(count_q[day_index, :, step, index].sum())
                    row[f"volume_{suffix}"] = float(volume_q[day_index, :, step, index].sum())
                records.append(row)
        return pd.DataFrame(records)

    predictions = pd.concat(
        [
            long_predictions(
                validation_days, count_validation, volume_validation, validation_prediction,
                val_count_q, val_volume_q, "validation",
            ),
            long_predictions(
                test_days, count_test, volume_test, test_prediction,
                test_count_q, test_volume_q, "locked_test",
            ),
        ],
        ignore_index=True,
    )
    prediction_path = output_root / "predictions/nlr_workload_cfc_w_predictions.parquet"
    prediction_artifact = atomic_parquet(
        predictions,
        prediction_path,
        metadata={
            "model_class": "dcvpp.models.workload.cfc_w.CfCW",
            "hidden_size": hidden_size,
            "horizon": horizon,
            "cohorts": list(COHORTS),
            "count_distribution": "negative_binomial",
            "volume_distribution_selected_on_validation": cfc_record["volume_distribution_selected_on_validation"],
            "event_elapsed_time_explicit": True,
            "selected_seed": cfc_record["selected_seed"],
            "locked_test_used_for_selection": False,
            "paper_evidence": True,
        },
    )
    leaderboard_path = output_root / "metrics/nlr_workload_paper_leaderboard.json"
    atomic_json(
        leaderboard_path,
        {
            "selection_metric": "validation_distribution_score",
            "locked_test_used_for_selection": False,
            "cohorts": list(COHORTS),
            "baselines": baseline_results,
            "cfc_w": cfc_record,
        },
    )
    return {
        "model_class": "dcvpp.models.workload.cfc_w.CfCW",
        "architecture_scope": "full_preregistered_event_elapsed_distribution_model",
        "architecture": cfc_record["architecture"],
        "full_preregistered_cfc_w_distribution_model_completed": True,
        "paper_evidence": True,
        "seed_trials": cfc_record["seed_trials"],
        "selected_seed": cfc_record["selected_seed"],
        "selected_epoch": cfc_record["selected_epoch"],
        "validation_metrics": cfc_record["validation_metrics"],
        "locked_test_metrics": cfc_record["locked_test_metrics"],
        "baseline_results": baseline_results,
        "baseline_leaderboard_path": leaderboard_path.as_posix(),
        "prediction_artifact": prediction_artifact,
        "sequence_counts": {key: len(value[0]) for key, value in aggregate_sequences.items()},
        "sequence_manifest": sequence_manifest,
        "event_sequence_manifest": event_manifest,
        "history_steps": history,
        "horizon_steps": horizon,
        "resume_fingerprint": data_fingerprint,
        "checkpoint_policy": "atomic latest checkpoint written after every completed epoch",
        "max_epochs": epochs,
        "early_stopping_patience": 10,
        "locked_test_used_for_selection_or_calibration": False,
    }
