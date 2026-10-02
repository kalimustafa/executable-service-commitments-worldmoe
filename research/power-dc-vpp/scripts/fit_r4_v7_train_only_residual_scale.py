from __future__ import annotations

"""Fit one global residual temperature on the trailing train-only panel."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcvpp.e2e_moe_r4.losses import (  # noqa: E402
    empirical_coverage,
    weighted_energy_score,
    weighted_interval_score,
    weighted_marginal_crps,
    weighted_quantile,
    weighted_variogram_score,
)
from dcvpp.e2e_moe_r4_v7.config import load_v7_config  # noqa: E402
from dcvpp.e2e_moe_r4_v7.data import load_frozen_residual_anchor  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-days", type=int, default=90)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--minimum", type=float, default=0.80)
    parser.add_argument("--maximum", type=float, default=1.05)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20240813)
    return parser.parse_args()


def _tail(samples: torch.Tensor, weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for quantile in (0.05, 0.95):
        forecast = weighted_quantile(samples, weights, quantile)
        error = target - forecast
        values.append(
            torch.maximum(quantile * error, (quantile - 1.0) * error)
            .flatten(1)
            .mean(1)
        )
    return torch.stack(values, dim=1).mean(1)


def _indices(
    day_indices: torch.Tensor,
    *,
    pool_size: int,
    sample_count: int,
    seed: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for day in day_indices.tolist():
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1_000_003 * int(day)
        )
        available = torch.cat((torch.arange(day), torch.arange(day + 1, pool_size)))
        order = torch.randperm(available.numel(), generator=generator)
        rows.append(available[order[:sample_count]])
    return torch.stack(rows)


def _score(
    residuals: torch.Tensor,
    channel_scale: torch.Tensor,
    day_indices: torch.Tensor,
    factors: torch.Tensor,
    *,
    sample_count: int,
    seed: int,
    device: torch.device,
    batch_size: int = 8,
) -> list[dict[str, float]]:
    totals: list[dict[str, float]] = [
        {
            "energy_score": 0.0,
            "marginal_crps": 0.0,
            "interval_score_90": 0.0,
            "interval_score_95": 0.0,
            "tail_score": 0.0,
            "dependence_score": 0.0,
            "coverage_90": 0.0,
            "coverage_95": 0.0,
        }
        for _ in factors
    ]
    scale = channel_scale.to(device=device, dtype=torch.float32)
    table = residuals.to(device=device, dtype=torch.float32) / scale
    count = int(day_indices.numel())
    for start in range(0, count, batch_size):
        selected = day_indices[start : start + batch_size]
        registered = _indices(
            selected,
            pool_size=residuals.shape[0],
            sample_count=sample_count,
            seed=seed,
        ).to(device)
        target = table[selected.to(device)]
        base = table[registered]
        weights = torch.full(
            base.shape[:2],
            1.0 / float(sample_count),
            device=device,
            dtype=torch.float32,
        )
        for factor_index, factor in enumerate(factors.tolist()):
            samples = base * float(factor)
            coverage90, _ = empirical_coverage(samples, weights, target, 0.90)
            coverage95, _ = empirical_coverage(samples, weights, target, 0.95)
            metrics = {
                "energy_score": weighted_energy_score(samples, weights, target),
                "marginal_crps": weighted_marginal_crps(samples, weights, target),
                "interval_score_90": weighted_interval_score(
                    samples, weights, target, 0.90
                ),
                "interval_score_95": weighted_interval_score(
                    samples, weights, target, 0.95
                ),
                "tail_score": _tail(samples, weights, target),
                "dependence_score": weighted_variogram_score(
                    samples, weights, target, lags=(1, 4, 16), power=0.5
                ),
                "coverage_90": coverage90,
                "coverage_95": coverage95,
            }
            weight = float(selected.numel()) / float(count)
            for name, value in metrics.items():
                totals[factor_index][name] += float(value.mean().cpu()) * weight
    return totals


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    arguments = _arguments()
    config_path = ROOT / "configs" / "e2e_moe_dart_vpp_R4_joint_world_policy_v7.yaml"
    _, _, resolved = load_v7_config(config_path)
    anchor = load_frozen_residual_anchor(resolved["residual_anchor"])
    if not 1 <= arguments.calibration_days < len(anchor.train_dates):
        raise ValueError("calibration-days must be smaller than the train panel")
    if not 2 <= arguments.sample_count < len(anchor.train_dates):
        raise ValueError("sample-count must permit leave-one-day-out sampling")
    steps = int(round((arguments.maximum - arguments.minimum) / arguments.step))
    factors = torch.linspace(arguments.minimum, arguments.maximum, steps + 1)
    start = len(anchor.train_dates) - arguments.calibration_days
    day_indices = torch.arange(start, len(anchor.train_dates))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = _score(
        anchor.residuals,
        anchor.channel_scale,
        day_indices,
        factors,
        sample_count=arguments.sample_count,
        seed=arguments.seed,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    for factor, values in zip(factors.tolist(), metrics, strict=True):
        rows.append(
            {
                "residual_scale": float(factor),
                **values,
                "absolute_coverage_error_90": abs(values["coverage_90"] - 0.90),
                "absolute_coverage_error_95": abs(values["coverage_95"] - 0.95),
            }
        )
    feasible = [
        row
        for row in rows
        if row["absolute_coverage_error_90"] <= 0.03
        and row["absolute_coverage_error_95"] <= 0.03
    ]
    if not feasible:
        raise RuntimeError("no train-only residual scale satisfies both coverage gates")
    selected = min(
        feasible,
        key=lambda row: (
            row["energy_score"],
            row["marginal_crps"],
            abs(row["residual_scale"] - 1.0),
        ),
    )
    output = (
        resolved["output_root"] / "data" / "train_only_global_residual_scale.json"
    )
    payload = {
        "schema": "dcvpp.r4.v7.train-only-global-residual-scale.v1",
        "fit_split": "trailing_train_only_leave_one_day_out",
        "fit_start": anchor.train_dates[start],
        "fit_end": anchor.train_dates[-1],
        "fit_days": arguments.calibration_days,
        "formal_validation_used": False,
        "test_2024_used": False,
        "locked_2025_opened": False,
        "mae_used": False,
        "sample_count": arguments.sample_count,
        "selection_rule": "coverage_feasible_then_lowest_joint_energy_score",
        "selected": selected,
        "grid": rows,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    _atomic_json(output, payload)
    print(json.dumps({"status": "PASSED", "output": str(output), **selected}, indent=2))


if __name__ == "__main__":
    main()
