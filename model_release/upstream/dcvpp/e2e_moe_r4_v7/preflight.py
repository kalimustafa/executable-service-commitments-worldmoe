"""Minimal train-only preflight for the R4-V7 lineage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dcvpp.e2e_moe_r4.losses import distribution_proper_loss_terms

from .config import load_v7_config
from .data import FrozenResidualAnchor, load_frozen_residual_anchor


PROPER_SCALE_KEYS = (
    "center_scale",
    "energy_score",
    "marginal_crps",
    "interval",
    "tail",
    "dependence",
)


class V7PreflightError(RuntimeError):
    """Raised when V7 cannot start without changing its scientific contract."""


@dataclass(frozen=True)
class V7PreflightResult:
    config: dict[str, Any]
    repository_root: Path
    resolved_paths: dict[str, Path]
    residual_anchor: FrozenResidualAnchor
    proper_scales: dict[str, float]


def fit_train_only_proper_scales(
    anchor: FrozenResidualAnchor,
    *,
    sample_count: int = 64,
    batch_size: int = 4,
    seed: int = 20240813,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Fit immutable metric denominators using every registered train day.

    Every target day is scored against samples drawn without replacement from
    the other training days.  The calculation is translation invariant and therefore
    needs no validation-period center or outcome.
    """

    count = int(sample_count)
    if not 2 <= count < anchor.residuals.shape[0]:
        raise V7PreflightError(
            f"proper-scale sample_count must lie in [2,{anchor.residuals.shape[0] - 1}]"
        )
    compute_device = torch.device(device)
    residuals = anchor.residuals.to(compute_device, dtype=torch.float32)
    channel_scale = anchor.channel_scale.to(compute_device, dtype=torch.float32)
    normalized = residuals / channel_scale
    total_days = normalized.shape[0]
    index_rows: list[torch.Tensor] = []
    for target_index in range(total_days):
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1_000_003 * target_index
        )
        candidates = torch.cat(
            (torch.arange(target_index), torch.arange(target_index + 1, total_days))
        )
        selected = candidates[torch.randperm(total_days - 1, generator=generator)[:count]]
        index_rows.append(selected)
    indices = torch.stack(index_rows)
    sums = {key: 0.0 for key in PROPER_SCALE_KEYS}
    observed = 0
    with torch.inference_mode():
        for start in range(0, total_days, int(batch_size)):
            stop = min(start + int(batch_size), total_days)
            rows = torch.arange(start, stop)
            selected = indices[start:stop].to(compute_device)
            samples = normalized[selected]
            target = normalized[rows.to(compute_device)]
            terms = distribution_proper_loss_terms(
                samples,
                target,
                identity_samples=None,
            )
            size = stop - start
            for key in PROPER_SCALE_KEYS:
                sums[key] += float(terms[key].sum().cpu())
            observed += size
    scales = {key: max(value / observed, 1.0e-8) for key, value in sums.items()}
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in scales.values()):
        raise V7PreflightError("train-only proper metric scales are invalid")
    return scales


def run_v7_preflight(
    config_path: str | Path,
    *,
    scale_device: torch.device | str = "cpu",
) -> V7PreflightResult:
    config, root, resolved = load_v7_config(config_path)
    train = config["protocol"]["splits"]["train"]
    anchor = load_frozen_residual_anchor(
        resolved["residual_anchor"],
        expected_start=train["start"],
        expected_end=train["end"],
        expected_days=int(train["days"]),
    )
    proper_scales = fit_train_only_proper_scales(anchor, device=scale_device)
    return V7PreflightResult(config, root, resolved, anchor, proper_scales)


__all__ = [
    "PROPER_SCALE_KEYS",
    "V7PreflightError",
    "V7PreflightResult",
    "fit_train_only_proper_scales",
    "run_v7_preflight",
]
