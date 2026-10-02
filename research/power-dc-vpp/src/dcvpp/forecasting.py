from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dcvpp.models.energy import TinyDGPTE
from dcvpp.models.workload import TinyCfCW


QUANTILES = np.array([0.05, 0.10, 0.50, 0.90, 0.95], dtype=np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pinball(y: np.ndarray, q_pred: np.ndarray) -> float:
    errors = y[..., None] - q_pred
    return float(np.mean(np.maximum(QUANTILES * errors, (QUANTILES - 1.0) * errors)))


def _window(values: np.ndarray, history: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    stride = max(1, horizon // 2)
    for end in range(history, len(values) - horizon + 1, stride):
        xs.append(values[end - history : end])
        ys.append(values[end : end + horizon])
    if len(xs) < 3:
        raise ValueError("not enough demo windows")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def run_demo_energy(market_path: Path, artifact_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 20260809))
    _seed_everything(seed)
    frame = pd.read_parquet(market_path)
    sites = sorted(frame["zone"].unique())
    history = int(config["forecast"].get("history_steps", 96))
    horizon = int(config["forecast"].get("horizon_steps", 96))
    epochs = int(config["forecast"].get("epochs", 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    checkpoint_dir = artifact_root / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for issue_type, column in (("DA", "da_lbmp"), ("RT", "rt_lbmp")):
        pivot = frame.pivot(index="timestamp_utc", columns="zone", values=column)[sites].sort_index()
        values = pivot.to_numpy(dtype=np.float32)
        mean, scale = values[: int(0.6 * len(values))].mean(axis=0), values[: int(0.6 * len(values))].std(axis=0) + 1e-6
        normalized = (values - mean) / scale
        x, y = _window(normalized, history, horizon)
        n_train = max(1, int(0.6 * len(x)))
        test_x, test_y = x[-1:], y[-1:]
        baseline_median = np.repeat(test_x[:, -1:, :], horizon, axis=1)
        residual_scale = np.std(y[:n_train] - np.repeat(x[:n_train, -1:, :], horizon, axis=1)) + 1e-4
        baseline_q = baseline_median[..., None] + residual_scale * np.array([-1.645, -1.282, 0, 1.282, 1.645], dtype=np.float32)
        rows.append(
            {
                "issue_type": issue_type,
                "model": "seasonal_naive",
                "seed": seed,
                "mae": float(np.mean(np.abs(test_y - baseline_median))),
                "mean_pinball": _pinball(test_y, baseline_q),
                "coverage_90": float(np.mean((test_y >= baseline_q[..., 0]) & (test_y <= baseline_q[..., -1]))),
                "profile": "demo",
                "synthetic": True,
                "evidence_role": "derived",
            }
        )
        model = TinyDGPTE(history, horizon, len(sites)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        q = torch.tensor(QUANTILES, device=device)
        tx = torch.tensor(x[:n_train], device=device)
        ty = torch.tensor(y[:n_train], device=device).transpose(1, 2)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred = torch.sort(model(tx), dim=-1).values
            error = ty[..., None] - pred
            loss = torch.maximum(q * error, (q - 1.0) * error).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = torch.sort(model(torch.tensor(test_x, device=device)), dim=-1).values.cpu().numpy()
        target = test_y.transpose(0, 2, 1)
        checkpoint = checkpoint_dir / f"dgpt_e_{issue_type.lower()}_latest.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "seed": seed,
                "issue_type": issue_type,
                "mean": mean,
                "scale": scale,
                "rng_state": torch.get_rng_state(),
            },
            checkpoint,
        )
        rows.append(
            {
                "issue_type": issue_type,
                "model": "DGPT-E-demo",
                "seed": seed,
                "mae": float(np.mean(np.abs(target - pred[..., 2]))),
                "mean_pinball": _pinball(target, pred),
                "coverage_90": float(np.mean((target >= pred[..., 0]) & (target <= pred[..., -1]))),
                "checkpoint": str(checkpoint),
                "profile": "demo",
                "synthetic": True,
                "evidence_role": "derived",
            }
        )
    table = artifact_root / "tables/nyiso_forecast_main.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table, index=False)
    return {"table": table, "rows": rows, "device": str(device)}


def run_demo_workload(workload_path: Path, artifact_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 20260809))
    _seed_everything(seed)
    frame = pd.read_parquet(workload_path).sort_values("timestamp_utc")
    history = int(config["forecast"].get("history_steps", 96))
    horizon = int(config["forecast"].get("horizon_steps", 96))
    epochs = int(config["forecast"].get("epochs", 2))
    count = frame["arrival_count"].to_numpy(np.float32)
    volume = frame["requested_work"].to_numpy(np.float32)
    count_scale, volume_scale = max(count[: int(0.6 * len(count))].mean(), 1.0), max(volume[: int(0.6 * len(volume))].mean(), 1.0)
    features = np.stack([count / count_scale, volume / volume_scale], axis=-1)
    x, y_count = _window(features, history, horizon)
    _, y_volume = _window(volume[:, None] / volume_scale, history, horizon)
    y_count = y_count[..., 0]
    y_volume = y_volume[..., 0]
    n_train = max(1, int(0.6 * len(x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCfCW(horizon=horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tx = torch.tensor(x[:n_train], device=device)
    dt = torch.full((len(tx), history, 1), 0.25, device=device)
    tc = torch.tensor(y_count[:n_train], device=device)
    tv = torch.tensor(y_volume[:n_train], device=device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred_c, pred_v = model(tx, dt)
        loss = torch.nn.functional.poisson_nll_loss(pred_c + 1e-4, tc, log_input=False) + torch.nn.functional.l1_loss(pred_v, tv)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_c, pred_v = model(torch.tensor(x[-1:], device=device), torch.full((1, history, 1), 0.25, device=device))
    pred_c_np, pred_v_np = pred_c.cpu().numpy()[0], pred_v.cpu().numpy()[0]
    target_c, target_v = y_count[-1], y_volume[-1]
    baseline_c = np.full_like(target_c, y_count[:n_train].mean())
    rows = [
        {
            "model": "seasonal_negative_binomial",
            "seed": seed,
            "count_mae": float(np.mean(np.abs(target_c - baseline_c))),
            "volume_mae": float(np.mean(np.abs(target_v - y_volume[:n_train].mean()))),
            "profile": "demo",
            "synthetic": True,
            "evidence_role": "derived",
        },
        {
            "model": "CfC-W-demo",
            "seed": seed,
            "count_mae": float(np.mean(np.abs(target_c - pred_c_np))),
            "volume_mae": float(np.mean(np.abs(target_v - pred_v_np))),
            "profile": "demo",
            "synthetic": True,
            "evidence_role": "derived",
        },
    ]
    checkpoint = artifact_root / "models/cfc_w_latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "rng_state": torch.get_rng_state()}, checkpoint)
    rows[-1]["checkpoint"] = str(checkpoint)
    table = artifact_root / "tables/workload_forecast_main.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table, index=False)
    return {"table": table, "rows": rows, "device": str(device)}

