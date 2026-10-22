"""Dimensionless local ramp and positive recovery-curvature penalties.

This research sensitivity term is a grid-impact surrogate, not a tariff or a
network-security guarantee. It deliberately leaves the original cost CVaR intact.
"""
from __future__ import annotations
import torch
from torch import Tensor


def grid_externality(grid: Tensor, previous: Tensor, previous_previous: Tensor,
                     grid_range: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-example total, local absolute ramp and aggregate curvature.

    grid: [B,H,Z], previous states/ranges: [B,Z]; signed import is positive.
    Each component is averaged over its time horizon, so horizon length alone
    does not change its coefficient relative to horizon-normalized economic cost.
    """
    if grid.ndim != 3 or previous.shape != grid.shape[::2]:
        raise ValueError('Expected grid[B,H,Z] and previous[B,Z]')
    if previous_previous.shape != previous.shape or grid_range.shape != previous.shape:
        raise ValueError('Grid history and physical range must have the same shape')
    g = grid.float()
    history = torch.cat((previous_previous[:, None].float(), previous[:, None].float(), g), dim=1)
    delta = history[:, 2:] - history[:, 1:-1]
    ramp = (delta.abs() / grid_range[:, None].float().clamp_min(1e-8)).mean((1, 2))
    total = history.sum(-1)
    curvature = (torch.relu(total[:, 2:] - 2 * total[:, 1:-1] + total[:, :-2])
                 / grid_range.sum(-1)[:, None].float().clamp_min(1e-8)).mean(1)
    return ramp + curvature, ramp, curvature
