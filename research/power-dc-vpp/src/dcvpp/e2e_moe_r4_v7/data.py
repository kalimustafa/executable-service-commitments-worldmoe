"""Read-only data adapters for the R4-V7 lineage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


HORIZON = 96
CHANNELS = 21
# Historical V7 default.  New split lineages pass their expected date window
# explicitly, so the residual pool is no longer required to contain 543 days.
TRAIN_DAYS = 543
RESIDUAL_BASELINE_SCHEMA = "dcvpp.r4.v6.variable-n-training-residual-baseline.v1"
DYNAMIC_RESIDUAL_BASELINE_SCHEMA = "dcvpp.r4.v7.development-residual-anchor.v1"


class V7DataError(ValueError):
    """Raised when a frozen predecessor artifact is unsuitable for V7."""


@dataclass(frozen=True)
class FrozenResidualAnchor:
    """The train-only whole-day residual distribution used as identity anchor."""

    residuals: Tensor
    channel_scale: Tensor
    pue_center: Tensor
    train_dates: tuple[str, ...]
    source_path: Path
    metadata: dict[str, Any]

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "FrozenResidualAnchor":
        return FrozenResidualAnchor(
            residuals=self.residuals.to(device=device, dtype=dtype),
            channel_scale=self.channel_scale.to(device=device, dtype=dtype),
            pue_center=self.pue_center.to(device=device, dtype=dtype),
            train_dates=self.train_dates,
            source_path=self.source_path,
            metadata=dict(self.metadata),
        )

    def deterministic_indices(self, sample_count: int, seed: int, *, device: torch.device | None = None) -> Tensor:
        count = int(sample_count)
        day_count = int(self.residuals.shape[0])
        if not 1 <= count <= day_count:
            raise V7DataError(f"sample_count must lie in [1,{day_count}]")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        indices = torch.randperm(day_count, generator=generator)[:count]
        return indices if device is None else indices.to(device)


def load_frozen_residual_anchor(
    path: str | Path,
    *,
    expected_start: str | date | None = None,
    expected_end: str | date | None = None,
    expected_days: int | None = None,
) -> FrozenResidualAnchor:
    """Load the existing V6 train-residual baseline without mutating it."""

    source = Path(path).resolve()
    if not source.is_file():
        raise V7DataError(f"frozen residual anchor is missing: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") not in {
        RESIDUAL_BASELINE_SCHEMA,
        DYNAMIC_RESIDUAL_BASELINE_SCHEMA,
    }:
        raise V7DataError("unexpected frozen residual anchor schema")
    residuals = torch.as_tensor(payload.get("residuals"), dtype=torch.float32)
    scale = torch.as_tensor(payload.get("scale"), dtype=torch.float32)
    pue_center = torch.as_tensor(payload.get("pue_center"), dtype=torch.float32)
    dates = tuple(str(item) for item in payload.get("train_dates", ()))
    if residuals.ndim != 3 or tuple(residuals.shape[1:]) != (HORIZON, CHANNELS):
        raise V7DataError("frozen residuals must be [days,96,21]")
    if scale.shape != (CHANNELS,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise V7DataError("frozen residual scale must contain 21 finite positive values")
    if pue_center.shape != (HORIZON, 4) or not bool(torch.isfinite(pue_center).all()):
        raise V7DataError("frozen PUE center must be [96,4]")
    day_count = int(residuals.shape[0])
    required_days = TRAIN_DAYS if expected_days is None else int(expected_days)
    if day_count != required_days or len(dates) != required_days or len(set(dates)) != required_days:
        raise V7DataError(
            f"frozen residual anchor must contain {required_days} unique train dates"
        )
    if dates != tuple(sorted(dates)):
        raise V7DataError("frozen residual anchor dates must be strictly ordered")
    start = "2022-01-01" if expected_start is None else str(expected_start)[:10]
    end = "2023-06-30" if expected_end is None else str(expected_end)[:10]
    try:
        start_day = date.fromisoformat(start)
        end_day = date.fromisoformat(end)
        parsed_dates = tuple(date.fromisoformat(value[:10]) for value in dates)
    except ValueError as exc:
        raise V7DataError("frozen residual anchor contains an invalid date") from exc
    if start_day > end_day or any(not start_day <= value <= end_day for value in parsed_dates):
        raise V7DataError("residual anchor contains dates outside the registered train split")
    if not bool(torch.isfinite(residuals).all()):
        raise V7DataError("frozen residual anchor contains non-finite values")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"residuals", "scale", "pue_center"}
    }
    return FrozenResidualAnchor(
        residuals=residuals,
        channel_scale=scale,
        pue_center=pue_center,
        train_dates=dates,
        source_path=source,
        metadata=metadata,
    )


def issue_center_from_batch(
    batch: dict[str, Any], anchor: FrozenResidualAnchor
) -> Tensor:
    """Reproduce the frozen baseline's causal issue-time center exactly."""

    try:
        market = torch.as_tensor(batch["market_raw"])
        workload = torch.as_tensor(batch["issue_workload_raw"])
        pv = torch.as_tensor(batch["issue_pv_raw"])
    except KeyError as exc:
        raise V7DataError(f"batch misses issue-time field {exc.args[0]!r}") from exc
    if market.ndim != 3 or market.shape[1:] != (HORIZON, 16):
        raise V7DataError("market_raw must be [B,96,16]")
    batch_size = market.shape[0]
    if workload.shape != (batch_size, HORIZON):
        raise V7DataError("issue_workload_raw must be [B,96]")
    if pv.shape != (batch_size, HORIZON, 4):
        raise V7DataError("issue_pv_raw must be [B,96,4]")
    pue = anchor.pue_center.to(device=market.device, dtype=market.dtype)
    pue = pue.unsqueeze(0).expand(batch_size, -1, -1)
    da = market[..., 0:4]
    rt = market[..., 4:8]
    return torch.cat(
        (da, rt, da - rt, workload[..., None].to(market), pv.to(market), pue),
        dim=-1,
    )


__all__ = [
    "CHANNELS",
    "DYNAMIC_RESIDUAL_BASELINE_SCHEMA",
    "FrozenResidualAnchor",
    "HORIZON",
    "TRAIN_DAYS",
    "V7DataError",
    "issue_center_from_batch",
    "load_frozen_residual_anchor",
]
