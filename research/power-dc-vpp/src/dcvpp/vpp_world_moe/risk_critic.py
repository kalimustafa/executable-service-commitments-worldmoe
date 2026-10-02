"""Distribution-native daily cost critic for solver-free risk control."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


COST_QUANTILE_COUNT = 64
COST_QUANTILE_LEVELS = tuple(
    (index + 0.5) / COST_QUANTILE_COUNT for index in range(COST_QUANTILE_COUNT)
)


class MonotoneCostQuantileCritic(nn.Module):
    """Predict a monotone normalized daily-cost distribution from state/action.

    The action trajectory is encoded in parallel.  Positive increments make
    quantile crossing impossible, so expectation and CVaR can be integrated
    directly without sorting sampled scenario costs.
    """

    def __init__(
        self,
        state_width: int,
        *,
        action_width: int = 6,
        hidden_width: int = 128,
        horizon: int = 96,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.state_norm = nn.LayerNorm(state_width)
        self.state_projection = nn.Linear(state_width, hidden_width)
        self.action_projection = nn.Linear(action_width, hidden_width)
        self.position = nn.Parameter(torch.zeros(1, self.horizon, hidden_width))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_width,
            nhead=4,
            dim_feedforward=2 * hidden_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_width),
            nn.Linear(2 * hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, COST_QUANTILE_COUNT),
        )
        output = self.fusion[-1]
        if not isinstance(output, nn.Linear):
            raise RuntimeError("cost critic output must be linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        with torch.no_grad():
            output.bias[1:].fill_(-4.0)

    def forward(self, state: Tensor, action_summary: Tensor) -> dict[str, Tensor]:
        if state.ndim != 2:
            raise ValueError("cost critic state must be [B,D]")
        if action_summary.shape != (state.shape[0], self.horizon, 6):
            raise ValueError("cost critic action summary must be [B,96,6]")
        state_feature = self.state_projection(self.state_norm(state))
        action_feature = self.action_projection(action_summary.float())
        action_feature = self.action_encoder(action_feature + self.position).mean(dim=1)
        raw = self.fusion(torch.cat((state_feature, action_feature), dim=-1)).float()
        base = F.softplus(raw[:, :1])
        increments = F.softplus(raw[:, 1:])
        quantiles = torch.cat((base, base + torch.cumsum(increments, dim=-1)), dim=-1)
        levels = torch.tensor(
            COST_QUANTILE_LEVELS,
            device=quantiles.device,
            dtype=torch.float32,
        )
        tail = levels >= 0.95
        return {
            "quantiles": quantiles,
            "levels": levels,
            "expected_cost": quantiles.mean(dim=-1),
            "cvar95": quantiles[:, tail].mean(dim=-1),
        }


class TwinMonotoneCostQuantileCritic(nn.Module):
    """Two critics with a conservative upper-cost view for policy gradients."""

    def __init__(self, state_width: int, *, horizon: int = 96) -> None:
        super().__init__()
        self.first = MonotoneCostQuantileCritic(state_width, horizon=horizon)
        self.second = MonotoneCostQuantileCritic(state_width, horizon=horizon)

    def forward(self, state: Tensor, action_summary: Tensor) -> dict[str, Tensor]:
        first = self.first(state, action_summary)
        second = self.second(state, action_summary)
        conservative = torch.maximum(first["quantiles"], second["quantiles"])
        levels = first["levels"]
        tail = levels >= 0.95
        return {
            "first_quantiles": first["quantiles"],
            "second_quantiles": second["quantiles"],
            "quantiles": conservative,
            "levels": levels,
            "expected_cost": conservative.mean(dim=-1),
            "cvar95": conservative[:, tail].mean(dim=-1),
        }


__all__ = [
    "COST_QUANTILE_COUNT",
    "COST_QUANTILE_LEVELS",
    "MonotoneCostQuantileCritic",
    "TwinMonotoneCostQuantileCritic",
]
