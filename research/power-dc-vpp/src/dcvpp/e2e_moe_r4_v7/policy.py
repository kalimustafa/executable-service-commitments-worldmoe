"""Origin-conditioned joint world-policy network for R4-V7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .distribution import StructuredDistributionMemory


HORIZON = 96
ZONES = 4


def _origin_index(origin: int | Tensor, batch: int, reference: Tensor) -> Tensor:
    value = torch.as_tensor(origin, device=reference.device)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,):
        raise ValueError("origin must be a scalar or [batch]")
    if value.is_floating_point() and not bool(torch.equal(value, value.round())):
        raise ValueError("origin must contain integers")
    value = value.long()
    if bool(((value < 0) | (value >= HORIZON)).any()):
        raise ValueError("origin must lie in [0,95]")
    return value


@dataclass(frozen=True)
class PhysicalActionLimits:
    """Batch-resolved action limits in deployment units."""

    grid_mw: Tensor
    battery_mw: Tensor
    compute_work: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "grid_mw": self.grid_mw,
            "battery_mw": self.battery_mw,
            "compute_work": self.compute_work,
        }


def _limit_tensor(
    raw: Any,
    *,
    batch: int,
    reference: Tensor,
    name: str,
    fallback: float,
) -> Tensor:
    value = fallback if raw is None else raw
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.reshape(1, 1).expand(batch, ZONES)
    elif result.shape == (ZONES,):
        result = result.unsqueeze(0).expand(batch, -1)
    elif result.shape == (batch, 1):
        result = result.expand(-1, ZONES)
    if result.shape != (batch, ZONES):
        raise ValueError(f"{name} must be scalar, [4], [batch,1], or [batch,4]")
    if not bool(torch.isfinite(result).all()) or bool((result <= 0.0).any()):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def resolve_action_limits(
    limits: Mapping[str, Any] | PhysicalActionLimits | None,
    reference: Tensor,
    *,
    grid_fallback: float,
    battery_fallback: float,
    compute_fallback: float,
) -> PhysicalActionLimits:
    if isinstance(limits, PhysicalActionLimits):
        raw: Mapping[str, Any] = limits.as_dict()
    elif limits is None:
        raw = {}
    elif isinstance(limits, Mapping):
        raw = limits
    else:
        raise ValueError("limits must be a mapping or PhysicalActionLimits")
    batch = reference.shape[0]

    def first(*names: str) -> Any:
        return next((raw[name] for name in names if raw.get(name) is not None), None)

    return PhysicalActionLimits(
        grid_mw=_limit_tensor(
            first("grid_mw", "grid_limit_mw", "max_grid_mw"),
            batch=batch,
            reference=reference,
            name="grid_mw",
            fallback=grid_fallback,
        ),
        battery_mw=_limit_tensor(
            first("battery_mw", "battery_power_mw", "max_battery_mw"),
            batch=batch,
            reference=reference,
            name="battery_mw",
            fallback=battery_fallback,
        ),
        compute_work=_limit_tensor(
            first("compute_work", "compute_capacity", "max_compute_work"),
            batch=batch,
            reference=reference,
            name="compute_work",
            fallback=compute_fallback,
        ),
    )


class OriginWorldReader(nn.Module):
    """Reads local, multi-scale, dependence and regime uncertainty at an origin."""

    def __init__(self, hidden_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("V7 OriginWorldReader fixes hidden_dim=128")
        self.hidden_dim = hidden_dim
        self.origin_embedding = nn.Embedding(HORIZON, hidden_dim)
        self.state_query = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        self.structure_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        # Current and future mean center/scale/tail are explicit policy inputs.
        self.stats_projection = nn.Sequential(
            nn.LayerNorm(2 * 3 * 21),
            nn.Linear(2 * 3 * 21, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.multiscale_projection = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(5 * hidden_dim),
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        memory: StructuredDistributionMemory,
        state_hidden: Tensor,
        *,
        origin: int | Tensor,
    ) -> dict[str, Tensor]:
        memory.validate()
        batch = memory.time_tokens.shape[0]
        if state_hidden.shape != (batch, self.hidden_dim):
            raise ValueError("state_hidden must be [batch,128]")
        index = _origin_index(origin, batch, state_hidden)
        row = torch.arange(batch, device=state_hidden.device)
        query = self.state_query(state_hidden) + self.origin_embedding(index)
        query_token = query[:, None]
        periods = torch.arange(HORIZON, device=state_hidden.device)[None]
        past_padding = periods < index[:, None]
        time_read = self.time_attention(
            query_token,
            memory.time_tokens,
            memory.time_tokens,
            key_padding_mask=past_padding,
            need_weights=False,
        )[0].squeeze(1)
        structure = torch.cat((memory.factor_tokens, memory.regime_tokens), dim=1)
        structure_read = self.structure_attention(
            query_token, structure, structure, need_weights=False
        )[0].squeeze(1)

        current_center = memory.center_zonal[row, index]
        current_scale = memory.scale_zonal[row, index]
        current_tail = memory.tail_zonal[row, index]
        future_mask = periods >= index[:, None]
        future_center = self._masked_mean(memory.center_zonal, future_mask)
        future_scale = self._masked_mean(memory.scale_zonal, future_mask)
        future_tail = self._masked_mean(memory.tail_zonal, future_mask)
        explicit_stats = torch.cat(
            (
                current_center,
                current_scale,
                current_tail,
                future_center,
                future_scale,
                future_tail,
            ),
            dim=-1,
        )
        stats_read = self.stats_projection(explicit_stats)

        def window(width: int | None) -> Tensor:
            if width is None:
                mask = periods >= index[:, None]
            else:
                mask = (periods >= index[:, None]) & (
                    periods < index[:, None] + width
                )
            return self._masked_mean(memory.time_tokens, mask)

        current_token = memory.time_tokens[row, index]
        multiscale = self.multiscale_projection(
            torch.cat((current_token, window(4), window(16), window(None)), dim=-1)
        )
        readout = self.output(
            torch.cat((query, time_read, structure_read, stats_read, multiscale), dim=-1)
        )
        return {
            "readout": readout,
            "time_read": time_read,
            "structure_read": structure_read,
            "stats_read": stats_read,
            "multiscale_read": multiscale,
            "explicit_center": torch.stack((current_center, future_center), dim=1),
            "explicit_scale": torch.stack((current_scale, future_scale), dim=1),
            "explicit_tail": torch.stack((current_tail, future_tail), dim=1),
            "origin_index": index,
        }


class DecisionExpert(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * expert_hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * expert_hidden_dim, hidden_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class DecisionTop2MoE(nn.Module):
    """Eight Top-2 experts specialize the shared future action plan."""

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        expert_hidden_dim: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if (num_experts, top_k) != (8, 2):
            raise ValueError("V7 Decision MoE requires eight Top-2 experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(2 * hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            DecisionExpert(hidden_dim, expert_hidden_dim)
            for _ in range(num_experts)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor, world_read: Tensor) -> dict[str, Tensor]:
        batch, horizon, hidden = tokens.shape
        context = world_read[:, None].expand(-1, horizon, -1)
        logits = self.router(torch.cat((tokens, context), dim=-1))
        probabilities = torch.softmax(logits, dim=-1)
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        expert_outputs = torch.stack(
            [expert(tokens) for expert in self.experts], dim=2
        )
        selected = expert_outputs.gather(
            2, top_indices[..., None].expand(-1, -1, -1, hidden)
        )
        delta = (selected * top_weights[..., None]).sum(dim=2)
        output = self.output_norm(tokens + delta)
        one_hot = F.one_hot(top_indices, self.num_experts).to(tokens.dtype)
        expert_load = one_hot.mean(dim=(0, 1, 2))
        importance = probabilities.mean(dim=(0, 1))
        return {
            "tokens": output,
            "router_logits": logits,
            "router_probabilities": probabilities,
            "topk_indices": top_indices,
            "topk_weights": top_weights,
            "router_balance_loss": self.num_experts
            * (expert_load * importance).sum(),
            "router_z_loss": torch.logsumexp(logits, dim=-1).square().mean(),
        }


class SharedPlanDecoder(nn.Module):
    """One decoder generates all three action families for all 96 intervals."""

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        heads: int = 4,
        expert_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.period_embedding = nn.Embedding(HORIZON, hidden_dim)
        self.context_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=3 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decision_moe = DecisionTop2MoE(
            hidden_dim, expert_hidden_dim=expert_hidden_dim
        )

    def forward(
        self, memory: StructuredDistributionMemory, world_read: Tensor, state: Tensor
    ) -> dict[str, Tensor]:
        periods = torch.arange(HORIZON, device=world_read.device)
        context = self.context_projection(torch.cat((world_read, state), dim=-1))
        tokens = (
            memory.time_tokens
            + self.period_embedding(periods)[None]
            + context[:, None]
        )
        tokens = self.sequence_layer(tokens)
        routed = self.decision_moe(tokens, world_read)
        return routed


class ActionConditionedRiskCritic(nn.Module):
    """Predicts decision risk from the world and the proposed physical action."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_projection = nn.Sequential(
            nn.Linear(3 * ZONES, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.LayerNorm(5 * hidden_dim),
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.expected_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)
        self.cvar_excess_head = nn.Linear(hidden_dim, 1)
        self.sla_head = nn.Linear(hidden_dim, 1)
        self.intervention_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        plan_tokens: Tensor,
        world_read: Tensor,
        state_hidden: Tensor,
        normalized_actions: Mapping[str, Tensor],
        origin_index: Tensor,
    ) -> dict[str, Tensor]:
        actions = torch.cat(
            (
                normalized_actions["day_ahead"],
                normalized_actions["battery"],
                normalized_actions["compute"],
            ),
            dim=-1,
        )
        action_tokens = self.action_projection(actions)
        batch = plan_tokens.shape[0]
        row = torch.arange(batch, device=plan_tokens.device)
        current_action = action_tokens[row, origin_index]
        future_mask = (
            torch.arange(HORIZON, device=plan_tokens.device)[None]
            >= origin_index[:, None]
        )
        weights = future_mask.to(plan_tokens.dtype)[..., None]
        future_plan = (plan_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        future_action = (action_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        hidden = self.net(
            torch.cat(
                (world_read, state_hidden, future_plan, current_action, future_action),
                dim=-1,
            )
        )
        expected = F.softplus(self.expected_head(hidden)).squeeze(-1)
        var = F.softplus(self.var_head(hidden)).squeeze(-1)
        cvar = var + F.softplus(self.cvar_excess_head(hidden)).squeeze(-1)
        return {
            "expected_cost": expected,
            "var": var,
            "cvar": cvar,
            "sla_risk": torch.sigmoid(self.sla_head(hidden)).squeeze(-1),
            "shield_intervention_risk": torch.sigmoid(
                self.intervention_head(hidden)
            ).squeeze(-1),
            "hidden": hidden,
        }


class RiskFeedbackRefiner(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.critic_projection = nn.Linear(hidden_dim, hidden_dim)
        self.delta = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self, plan_tokens: Tensor, world_read: Tensor, critic_hidden: Tensor
    ) -> Tensor:
        context = torch.cat(
            (
                plan_tokens,
                world_read[:, None].expand_as(plan_tokens),
                self.critic_projection(critic_hidden)[:, None].expand_as(plan_tokens),
            ),
            dim=-1,
        )
        return plan_tokens + 0.1 * torch.tanh(self.delta(context))


class JointWorldPolicy(nn.Module):
    """Shared-plan actor, action-conditioned critic and two risk refinements."""

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        heads: int = 4,
        expert_hidden_dim: int = 256,
        refinement_steps: int = 2,
        max_grid_mw: float = 1.0,
        max_battery_mw: float = 1.0,
        max_compute_work: float = 1.0,
    ) -> None:
        super().__init__()
        if refinement_steps != 2:
            raise ValueError("V7 requires exactly two risk-feedback refinements")
        self.reader = OriginWorldReader(hidden_dim, heads)
        self.plan_decoder = SharedPlanDecoder(
            hidden_dim, heads=heads, expert_hidden_dim=expert_hidden_dim
        )
        self.day_ahead_head = nn.Linear(hidden_dim, ZONES)
        self.battery_head = nn.Linear(hidden_dim, ZONES)
        self.compute_head = nn.Linear(hidden_dim, ZONES)
        for head in (self.day_ahead_head, self.battery_head, self.compute_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.critic = ActionConditionedRiskCritic(hidden_dim)
        self.refiners = nn.ModuleList(
            RiskFeedbackRefiner(hidden_dim) for _ in range(refinement_steps)
        )
        self.max_grid_mw = float(max_grid_mw)
        self.max_battery_mw = float(max_battery_mw)
        self.max_compute_work = float(max_compute_work)

    def _actions(
        self, plan_tokens: Tensor, limits: PhysicalActionLimits
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        normalized = {
            "day_ahead": torch.sigmoid(self.day_ahead_head(plan_tokens)),
            "battery": torch.tanh(self.battery_head(plan_tokens)),
            "compute": torch.sigmoid(self.compute_head(plan_tokens)),
        }
        physical = {
            "day_ahead": normalized["day_ahead"] * limits.grid_mw[:, None],
            "battery": normalized["battery"] * limits.battery_mw[:, None],
            "compute": normalized["compute"] * limits.compute_work[:, None],
        }
        return physical, normalized

    @staticmethod
    def _issued(value: Tensor, reference: Tensor) -> Tensor:
        issued = torch.as_tensor(
            value, device=reference.device, dtype=reference.dtype
        )
        if issued.shape == (reference.shape[0], ZONES, HORIZON):
            issued = issued.transpose(1, 2)
        if issued.shape != (reference.shape[0], HORIZON, ZONES):
            raise ValueError("issued_day_ahead must be [batch,96,4] or [batch,4,96]")
        if not bool(torch.isfinite(issued).all()):
            raise ValueError("issued_day_ahead contains non-finite values")
        return issued

    def forward(
        self,
        memory: StructuredDistributionMemory,
        state_hidden: Tensor,
        *,
        origin: int | Tensor,
        limits: Mapping[str, Any] | PhysicalActionLimits | None,
        issued_day_ahead: Tensor | None = None,
    ) -> dict[str, Any]:
        batch = state_hidden.shape[0]
        index = _origin_index(origin, batch, state_hidden)
        resolved_limits = resolve_action_limits(
            limits,
            state_hidden,
            grid_fallback=self.max_grid_mw,
            battery_fallback=self.max_battery_mw,
            compute_fallback=self.max_compute_work,
        )
        read = self.reader(memory, state_hidden, origin=index)
        decoded = self.plan_decoder(memory, read["readout"], state_hidden)
        plan_tokens = decoded["tokens"]
        trace: list[dict[str, Any]] = []
        for step, refiner in enumerate(self.refiners):
            physical, normalized = self._actions(plan_tokens, resolved_limits)
            critic = self.critic(
                plan_tokens, read["readout"], state_hidden, normalized, index
            )
            trace.append(
                {
                    "step": step,
                    "critic": critic,
                    "normalized_actions": normalized,
                }
            )
            plan_tokens = refiner(plan_tokens, read["readout"], critic["hidden"])

        physical_plan, normalized_plan = self._actions(plan_tokens, resolved_limits)
        critic = self.critic(
            plan_tokens, read["readout"], state_hidden, normalized_plan, index
        )
        origin_zero = index.eq(0)
        candidate_day_ahead = physical_plan["day_ahead"]
        if bool((~origin_zero).any()) and issued_day_ahead is None:
            raise ValueError("issued_day_ahead is required after origin zero")
        if issued_day_ahead is None:
            day_ahead = candidate_day_ahead
        else:
            frozen = self._issued(issued_day_ahead, candidate_day_ahead)
            day_ahead = torch.where(
                origin_zero[:, None, None], candidate_day_ahead, frozen
            )
        row = torch.arange(batch, device=state_hidden.device)
        current_battery = physical_plan["battery"][row, index]
        current_compute = physical_plan["compute"][row, index]
        normalized_current_battery = normalized_plan["battery"][row, index]
        normalized_current_compute = normalized_plan["compute"][row, index]
        free_actions = {
            "day_ahead": day_ahead,
            "battery": current_battery,
            "compute": current_compute,
        }
        normalized_actions = {
            "day_ahead": normalized_plan["day_ahead"],
            "battery": normalized_current_battery,
            "compute": normalized_current_compute,
        }
        return {
            "day_ahead": day_ahead,
            "day_ahead_candidate": candidate_day_ahead,
            "day_ahead_issue_mask": origin_zero,
            "battery": current_battery,
            "compute": current_compute,
            "battery_plan": physical_plan["battery"],
            "compute_plan": physical_plan["compute"],
            "free_actions": free_actions,
            "actions": free_actions,
            "normalized_actions": normalized_actions,
            "normalized_plans": normalized_plan,
            "critic": critic,
            "world_read": read,
            "plan_tokens": plan_tokens,
            "refinement_trace": tuple(trace),
            "refinement_steps": len(self.refiners),
            "decision_moe": {
                key: value for key, value in decoded.items() if key != "tokens"
            },
            "limits": resolved_limits.as_dict(),
            "origin_index": index,
            "action_units": {
                "day_ahead": "MW",
                "battery": "MW_signed",
                "compute": "work_units",
            },
            "policy_input_contract": (
                "origin+state+time_center_scale_tail+factor+regime;sample_free"
            ),
        }


__all__ = [
    "ActionConditionedRiskCritic",
    "DecisionTop2MoE",
    "JointWorldPolicy",
    "OriginWorldReader",
    "PhysicalActionLimits",
    "SharedPlanDecoder",
    "resolve_action_limits",
]
