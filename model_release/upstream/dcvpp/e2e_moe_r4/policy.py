"""Causal Liquid-state rolling policy for the solver-free R4 model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _origin_index(origin: int | Tensor, batch: int, reference: Tensor, horizon: int) -> Tensor:
    index = torch.as_tensor(origin, device=reference.device)
    if index.ndim == 0:
        index = index.expand(batch)
    if index.shape != (batch,):
        raise ValueError(f"origin must be scalar or [batch], got {tuple(index.shape)}")
    if index.dtype.is_floating_point and not torch.equal(index, index.round()):
        raise ValueError("origin must contain integer indices")
    index = index.long()
    if ((index < 0) | (index >= horizon)).any():
        raise ValueError(f"origin must lie in [0,{horizon - 1}]")
    return index


class LiquidRollingPolicy(nn.Module):
    """One-origin policy with future-plan attention and two neural refinements.

    Only the day-ahead schedule (at origin zero), signed battery power, compute
    allocation, and confidence are learned choices.  PV use and grid import are
    derived from the forecast energy identity.  Migration is structurally zero.
    The independent deterministic shield remains outside this module.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        horizon: int = 96,
        num_sites: int = 4,
        num_heads: int = 4,
        policy_hidden_dim: int = 256,
        refinement_steps: int = 2,
        max_grid_action: float = 1.0,
        max_battery_action: float = 1.0,
        max_compute_action: float = 1.0,
    ) -> None:
        super().__init__()
        if horizon != 96 or num_sites != 4:
            raise ValueError("formal R4 policy requires 96 origins and four sites")
        if refinement_steps != 2:
            raise ValueError("formal R4 policy requires exactly two refinement steps")
        self.hidden_dim = int(hidden_dim)
        self.horizon = 96
        self.num_sites = 4
        self.refinement_steps = 2
        self.max_grid_action = float(max_grid_action)
        self.max_battery_action = float(max_battery_action)
        self.max_compute_action = float(max_compute_action)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.plan_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.world_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.risk_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.rolling_projection = nn.Linear(hidden_dim, hidden_dim)
        self.origin_embedding = nn.Embedding(horizon, hidden_dim)
        self.revealed_projection = nn.Linear(hidden_dim, hidden_dim)
        self.policy_core = nn.Sequential(
            nn.Linear(hidden_dim * 2, policy_hidden_dim),
            nn.LayerNorm(policy_hidden_dim),
            nn.SiLU(),
        )
        # signed battery4, compute4, fallback/confidence2
        self.action_width = 10
        self.action_head = nn.Linear(policy_hidden_dim, self.action_width)
        self.refinement = nn.Sequential(
            nn.Linear(policy_hidden_dim + self.action_width, policy_hidden_dim),
            nn.LayerNorm(policy_hidden_dim),
            nn.SiLU(),
            nn.Linear(policy_hidden_dim, self.action_width),
        )
        self.day_ahead_head = nn.Sequential(
            nn.Linear(policy_hidden_dim, policy_hidden_dim),
            nn.SiLU(),
            nn.Linear(policy_hidden_dim, num_sites * horizon),
        )
        self.battery_plan_head = nn.Linear(hidden_dim, num_sites)
        self.compute_plan_head = nn.Linear(hidden_dim, num_sites)

    def _attend_future(
        self, plan_tokens: Tensor, state_hidden: Tensor, origin_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        periods = torch.arange(self.horizon, device=plan_tokens.device)[None]
        future_padding = periods < origin_index[:, None]
        query = self.query_norm(state_hidden).unsqueeze(1)
        values = self.plan_norm(plan_tokens)
        attended, attention = self.cross_attention(
            query,
            values,
            values,
            key_padding_mask=future_padding,
            need_weights=True,
            average_attn_weights=False,
        )
        return attended.squeeze(1), attention

    def forward(
        self,
        plan_tokens: Tensor,
        state_hidden: Tensor,
        forecast_zonal: Tensor,
        *,
        origin: int | Tensor,
        world_latent: Any,
        risk_features: Tensor | Mapping[str, Any],
        rolling_state_hidden: Tensor,
        revealed_context: Tensor | None = None,
        issued_da_schedule: Tensor | None = None,
        policy_memory: Tensor | None = None,
    ) -> dict[str, Any]:
        if plan_tokens.ndim != 3 or plan_tokens.shape[1:] != (
            self.horizon,
            self.hidden_dim,
        ):
            raise ValueError("plan_tokens must be [batch,96,128]")
        batch = plan_tokens.shape[0]
        if state_hidden.shape != (batch, self.hidden_dim):
            raise ValueError("state_hidden must be [batch,128]")
        if forecast_zonal.shape != (batch, self.horizon, 21):
            raise ValueError("forecast_zonal must be [batch,96,21]")
        index = _origin_index(origin, batch, plan_tokens, self.horizon)
        if isinstance(world_latent, Mapping):
            world_tokens = world_latent.get("token_features")
        else:
            world_tokens = getattr(world_latent, "token_features", None)
        if not isinstance(world_tokens, Tensor) or world_tokens.shape != plan_tokens.shape:
            raise ValueError(
                "world_latent must expose token_features with shape [batch,96,128]"
            )
        if isinstance(risk_features, Mapping):
            risk_tensor = risk_features.get("risk_features")
        else:
            risk_tensor = risk_features
        if not isinstance(risk_tensor, Tensor) or risk_tensor.shape != state_hidden.shape:
            raise ValueError("risk_features must resolve to [batch,128]")
        if rolling_state_hidden.shape != state_hidden.shape:
            raise ValueError("rolling_state_hidden must be [batch,128]")
        plan_tokens = plan_tokens + self.world_projection(world_tokens)
        state_hidden = (
            state_hidden
            + self.risk_projection(risk_tensor)
            + self.rolling_projection(rolling_state_hidden)
            + self.origin_embedding(index)
        )
        if revealed_context is not None:
            if revealed_context.shape != state_hidden.shape:
                raise ValueError("revealed_context must be [batch,128]")
            state_hidden = state_hidden + self.revealed_projection(revealed_context)
        if policy_memory is not None:
            if policy_memory.shape != state_hidden.shape:
                raise ValueError("policy_memory must be [batch,128]")
            state_hidden = state_hidden + policy_memory
        attended, attention = self._attend_future(plan_tokens, state_hidden, index)
        hidden = self.policy_core(torch.cat((state_hidden, attended), dim=-1))
        raw = self.action_head(hidden)
        for _ in range(self.refinement_steps):
            raw = raw + self.refinement(torch.cat((hidden, torch.tanh(raw)), dim=-1)) / 2.0
        battery = torch.tanh(raw[:, 0:4]) * self.max_battery_action
        compute = torch.sigmoid(raw[:, 4:8]) * self.max_compute_action
        fallback_confidence = torch.sigmoid(raw[:, 8])
        policy_confidence = torch.sigmoid(raw[:, 9])

        candidate_da = torch.sigmoid(self.day_ahead_head(hidden)).reshape(
            batch, self.num_sites, self.horizon
        ) * self.max_grid_action
        origin_zero = index.eq(0)
        if torch.any(~origin_zero):
            if issued_da_schedule is None:
                raise ValueError("issued_da_schedule is required after origin zero")
            issued = torch.as_tensor(
                issued_da_schedule, device=plan_tokens.device, dtype=plan_tokens.dtype
            )
            if issued.shape != candidate_da.shape:
                raise ValueError("issued_da_schedule must be [batch,4,96]")
            day_ahead = torch.where(origin_zero[:, None, None], candidate_da, issued)
        else:
            day_ahead = candidate_da

        current = forecast_zonal[torch.arange(batch, device=plan_tokens.device), index]
        workload = current[:, 12:13].expand(-1, 4)
        pv_available = current[:, 13:17].clamp_min(0.0)
        pue = current[:, 17:21].clamp_min(0.0)
        demand = compute * workload * pue
        # One signed-battery convention is used end to end: positive means
        # charging (additional demand) and negative means discharging.  This
        # is the same convention enforced by AnalyticConstraintShield.
        net_demand = demand + battery
        pv_used = torch.minimum(pv_available, net_demand.clamp_min(0.0))
        grid = (net_demand - pv_used).clamp_min(0.0)
        migration = plan_tokens.new_zeros((batch, 4, 4))
        battery_plan = torch.tanh(self.battery_plan_head(plan_tokens)).transpose(1, 2)
        battery_plan = battery_plan * self.max_battery_action
        compute_plan = torch.sigmoid(self.compute_plan_head(plan_tokens)).transpose(1, 2)
        compute_plan = compute_plan * self.max_compute_action
        # The executed current action is the two-step refined state-conditioned
        # choice; these suffix plans exist only for differentiable risk rollout.
        batch_rows = torch.arange(batch, device=plan_tokens.device)
        battery_plan = battery_plan.clone()
        compute_plan = compute_plan.clone()
        battery_plan[batch_rows, :, index] = battery
        compute_plan[batch_rows, :, index] = compute

        return {
            "day_ahead": day_ahead,
            "day_ahead_candidate": candidate_da,
            "day_ahead_issue_mask": origin_zero,
            "battery": battery,
            "compute": compute,
            "pv_used": pv_used,
            "grid": grid,
            "real_time": grid,
            "migration": migration,
            "battery_plan": battery_plan,
            "compute_plan": compute_plan,
            "fallback_confidence": fallback_confidence,
            "policy_confidence": policy_confidence,
            "policy_memory_next": state_hidden,
            "policy_attention": attention,
            "raw_neural_action": raw,
            "policy_input_contract": "world+risk+rolling+origin;no_particles",
            "free_actions": {
                "day_ahead": day_ahead,
                "battery": battery,
                "compute": compute,
                "fallback_confidence": fallback_confidence,
                "policy_confidence": policy_confidence,
            },
            "derived_diagnostics": {
                "pv_used": pv_used,
                "grid": grid,
                "real_time": grid,
                "migration_fixed_zero": migration,
            },
            "action_roles": {
                "free": (
                    "day_ahead_at_origin0",
                    "signed_battery",
                    "compute",
                    "fallback_confidence",
                    "policy_confidence",
                ),
                "derived_only": ("pv_used", "grid", "real_time"),
                "fixed_zero": ("migration",),
            },
        }


__all__ = ["LiquidRollingPolicy"]
