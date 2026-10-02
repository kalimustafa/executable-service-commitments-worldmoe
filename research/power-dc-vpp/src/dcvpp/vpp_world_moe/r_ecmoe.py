"""Risk-conditioned expert-continuation MoE for direct VPP decisions.

The module deliberately bypasses the legacy candidate planner.  It executes the
released W0/W1 backbone once, continues only the active final-block experts,
assembles one structured action, and applies the lightweight W1 action response.
No external optimizer, candidate search, or repeated full-world rollout is used.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import VPPWorldMoE
from .packets import (
    NODES,
    WorldActionPacket,
    WorldConstraintPacket,
    WorldInputPacket,
    WorldObservationPacket,
    WorldStatePacket,
)
from .sparse_moe import TAIL_RISK_SLOT


ProfileName = Literal["fast", "balanced", "paper"]
StageName = Literal["ra", "rb", "rc"]
MechanismVariant = Literal[
    "full",
    "no_sparse_coordinator",
    "no_risk_condition",
    "no_decision_tail",
    "no_route_prior",
    "pooled_latent_instead_of_expert_states",
    "tail_always_on",
]
MECHANISM_VARIANTS = (
    "full",
    "no_sparse_coordinator",
    "no_risk_condition",
    "no_decision_tail",
    "no_route_prior",
    "pooled_latent_instead_of_expert_states",
    "tail_always_on",
)

COST_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class RECMoEConfig:
    """Small decision-continuation configuration independent of legacy gates."""

    profile: ProfileName = "fast"
    d_model: int = 256
    general_experts: int = 7
    active_top_k: int = 1
    adapter_bottleneck: int = 16
    temporal_rank: int = 8
    risk_hidden_dim: int = 64
    route_delta_cap: float = 2.0
    route_prior_beta: float = 1.0
    tail_decision_threshold: float = 0.05
    action_residual_fraction: float = 0.25
    deadline_intervals: int = 4
    dropout: float = 0.0
    tune_w1_action_response: bool = False
    mechanism_variant: MechanismVariant = "full"

    @classmethod
    def for_profile(cls, profile: ProfileName) -> "RECMoEConfig":
        if profile == "fast":
            return cls(
                profile="fast", active_top_k=1, adapter_bottleneck=16, temporal_rank=8
            )
        if profile == "balanced":
            return cls(
                profile="balanced",
                active_top_k=2,
                adapter_bottleneck=32,
                temporal_rank=16,
                tune_w1_action_response=False,
            )
        if profile == "paper":
            return cls(
                profile="paper",
                active_top_k=2,
                adapter_bottleneck=32,
                temporal_rank=16,
                tune_w1_action_response=False,
            )
        raise ValueError(f"unknown R-ECMoE profile: {profile!r}")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RECMoEConfig":
        profile = str(value.get("profile", "fast")).lower()
        base = cls.for_profile(profile)  # type: ignore[arg-type]
        fields = {
            name: value.get(name, getattr(base, name))
            for name in base.__dataclass_fields__
        }
        return cls(**fields)

    def __post_init__(self) -> None:
        if self.profile not in {"fast", "balanced", "paper"}:
            raise ValueError("R-ECMoE profile must be fast, balanced, or paper")
        if self.d_model < 8 or self.general_experts != 7:
            raise ValueError("R-ECMoE must continue the released seven-expert W1")
        if self.active_top_k not in {1, 2}:
            raise ValueError("R-ECMoE active_top_k must be one or two")
        if self.mechanism_variant not in MECHANISM_VARIANTS:
            raise ValueError("R-ECMoE mechanism_variant is not registered")
        if self.mechanism_variant != "full" and self.active_top_k != 2:
            raise ValueError("persistent R-ECMoE mechanism ablations require Top-2")
        if self.adapter_bottleneck < 4 or self.temporal_rank < 1:
            raise ValueError("R-ECMoE adapter and temporal ranks must be positive")
        if not 0.0 < self.tail_decision_threshold < 1.0:
            raise ValueError("tail decision threshold must lie in (0,1)")
        if not 0.0 < self.action_residual_fraction <= 1.0:
            raise ValueError("action residual fraction must lie in (0,1]")
        if self.deadline_intervals < 1 or self.deadline_intervals > 96:
            raise ValueError("deadline_intervals must lie in [1,96]")
        if self.tune_w1_action_response:
            raise ValueError(
                "shared W1 dynamics stay frozen; tune the R response adapter instead"
            )


@dataclass(frozen=True)
class RiskConditionPacket:
    """Deployable risk, service and stage preferences for one decision window."""

    risk_weight: Tensor
    tail_level: Tensor
    cost_budget: Tensor
    sla_budget: Tensor
    migration_budget: Tensor
    ramp_budget: Tensor
    battery_degradation_preference: Tensor
    action_range: Tensor
    stage: Tensor
    horizon: Tensor

    @classmethod
    def defaults(
        cls,
        batch: int,
        *,
        device: torch.device,
        horizon: int,
        stage: int = 1,
        action_range: float = 1.0,
    ) -> "RiskConditionPacket":
        one = torch.ones(batch, device=device, dtype=torch.float32)
        return cls(
            risk_weight=one.clone(),
            tail_level=one.new_full((batch,), 0.95),
            cost_budget=one.clone(),
            sla_budget=one.clone(),
            migration_budget=one.clone(),
            ramp_budget=one.clone(),
            battery_degradation_preference=one.clone(),
            action_range=one.new_full((batch,), float(action_range)),
            stage=torch.full((batch,), int(stage), device=device, dtype=torch.long),
            horizon=torch.full((batch,), int(horizon), device=device, dtype=torch.long),
        )

    def validate(self, batch: int, maximum_horizon: int) -> "RiskConditionPacket":
        scalar_names = (
            "risk_weight",
            "tail_level",
            "cost_budget",
            "sla_budget",
            "migration_budget",
            "ramp_budget",
            "battery_degradation_preference",
            "action_range",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if value.shape != (batch,):
                raise ValueError(f"risk condition {name} must be finite [B]")
            if value.device != self.risk_weight.device:
                raise ValueError("risk condition tensors must share one device")
            if value.device.type == "cpu" and not bool(torch.isfinite(value).all()):
                raise ValueError(f"risk condition {name} must be finite [B]")
        if self.tail_level.device.type == "cpu" and bool(
            ((self.tail_level <= 0.0) | (self.tail_level >= 1.0)).any()
        ):
            raise ValueError("tail_level must lie in (0,1)")
        for name in (
            "risk_weight",
            "cost_budget",
            "sla_budget",
            "migration_budget",
            "ramp_budget",
            "battery_degradation_preference",
        ):
            value = getattr(self, name)
            if value.device.type == "cpu" and bool((value < 0.0).any()):
                raise ValueError(f"risk condition {name} must be nonnegative")
        if self.action_range.device.type == "cpu" and bool(
            ((self.action_range < 0.0) | (self.action_range > 1.0)).any()
        ):
            raise ValueError("action_range must lie in [0,1]")
        if self.stage.shape != (batch,) or self.stage.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("risk condition stage must be integral [B]")
        if self.stage.device != self.risk_weight.device:
            raise ValueError("risk condition tensors must share one device")
        if self.stage.device.type == "cpu" and bool(
            ((self.stage < 0) | (self.stage > 1)).any()
        ):
            raise ValueError("stage uses 0=DA and 1=RT")
        if self.horizon.shape != (batch,) or self.horizon.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("risk condition horizon must be integral [B]")
        if self.horizon.device != self.risk_weight.device:
            raise ValueError("risk condition tensors must share one device")
        if self.horizon.device.type == "cpu" and bool(
            ((self.horizon < 1) | (self.horizon > maximum_horizon)).any()
        ):
            raise ValueError("risk horizon lies outside the active timeline")
        return self


@dataclass(frozen=True)
class W1ExpertPacket:
    """Compact final-block W1 expert context used by R-ECMoE."""

    state_latent: Tensor
    selected_expert_ids: Tensor
    selected_route_logits: Tensor
    selected_route_weights: Tensor
    selected_expert_states: Tensor
    pre_risk_state: Tensor
    tail_probability: Tensor
    tail_active: Tensor
    tail_state: Tensor
    action_response_basis: Tensor
    base_world_summary: Tensor
    uncertainty_summary: Tensor
    valid_action_masks: Tensor
    causal_context: Tensor
    origin_index: Tensor
    constraints: WorldConstraintPacket
    reference: WorldObservationPacket
    previous_actions: Tensor
    source_state: WorldStatePacket

    def validate(
        self, width: int, top_k: int, *, strict_values: bool = False
    ) -> "W1ExpertPacket":
        batch, time, actual_width = self.state_latent.shape
        if actual_width != width:
            raise ValueError("W1 expert packet width mismatch")
        if self.selected_expert_ids.shape != (batch, time, top_k):
            raise ValueError("W1 expert ids have an invalid shape")
        if self.selected_expert_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("W1 expert ids must be integral")
        if bool(
            ((self.selected_expert_ids < 0) | (self.selected_expert_ids >= 7)).any()
        ):
            raise ValueError("W1 expert ids lie outside the released seven experts")
        for name in ("selected_route_logits", "selected_route_weights"):
            if getattr(self, name).shape != (batch, time, top_k):
                raise ValueError(f"W1 expert packet {name} has an invalid shape")
        if self.selected_expert_states.shape != (batch, time, top_k, width):
            raise ValueError("W1 selected expert states have an invalid shape")
        if self.pre_risk_state.shape != (batch, time, width):
            raise ValueError("W1 pre-risk state must be [B,T,D]")
        if self.tail_probability.shape != (batch, time, 1):
            raise ValueError("W1 tail probability must be [B,T,1]")
        if bool(((self.tail_probability < 0.0) | (self.tail_probability > 1.0)).any()):
            raise ValueError("W1 tail probability must lie in [0,1]")
        if (
            self.tail_active.shape != (batch, time)
            or self.tail_active.dtype is not torch.bool
        ):
            raise ValueError("W1 tail active mask must be bool [B,T]")
        if self.tail_state.shape != (batch, time, width):
            raise ValueError("W1 tail state must be [B,T,D]")
        if self.action_response_basis.shape != (batch, time, width):
            raise ValueError("W1 action response basis must be [B,T,D]")
        if self.valid_action_masks.shape[:2] != (batch, time):
            raise ValueError("W1 action masks have an invalid timeline")
        if self.valid_action_masks.shape != (batch, time, NODES, 6):
            raise ValueError("W1 action masks must be [B,T,4,6]")
        if self.valid_action_masks.dtype is not torch.bool:
            raise ValueError("W1 action masks must be bool")
        if self.previous_actions.shape != (batch, time, NODES, 6):
            raise ValueError("W1 previous actions must be [B,T,4,6]")
        if self.base_world_summary.shape[:2] != (batch, time):
            raise ValueError("W1 base world summary has an invalid timeline")
        if self.uncertainty_summary.shape[:2] != (batch, time):
            raise ValueError("W1 uncertainty summary has an invalid timeline")
        if self.causal_context.shape != (batch, time, 2):
            raise ValueError("W1 causal context must be [B,T,2]")
        if self.origin_index.shape != (batch,):
            raise ValueError("W1 origin index must be [B]")
        if bool(((self.origin_index < 0) | (self.origin_index >= time)).any()):
            raise ValueError("W1 origin index lies outside the timeline")
        if strict_values:
            finite_values = {
                "state latent": self.state_latent,
                "route logits": self.selected_route_logits,
                "route weights": self.selected_route_weights,
                "selected expert states": self.selected_expert_states,
                "pre-risk state": self.pre_risk_state,
                "tail probability": self.tail_probability,
                "tail state": self.tail_state,
                "action response basis": self.action_response_basis,
                "base-world summary": self.base_world_summary,
                "uncertainty summary": self.uncertainty_summary,
                "causal context": self.causal_context,
                "previous actions": self.previous_actions,
            }
            for name, value in finite_values.items():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"W1 {name} contains non-finite values")
            if not torch.allclose(
                self.selected_route_weights.sum(dim=-1),
                torch.ones_like(self.selected_route_weights[..., 0]),
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                raise ValueError("W1 selected route weights must sum to one")
        self.constraints.validate(batch)
        self.reference.validate(batch=batch, strict_values=strict_values)
        return self


@dataclass(frozen=True)
class ConstraintAssemblyResult:
    actions: WorldActionPacket
    day_ahead_commitment: Tensor
    target_soc: Tensor
    raw_structured_actions: Tensor
    residuals: dict[str, Tensor]
    action_summary: Tensor
    decision_active_mask: Tensor
    deadline_expired_work: Tensor


@dataclass(frozen=True)
class RECMoEOutput:
    expert_packet: W1ExpertPacket
    actions: WorldActionPacket
    day_ahead_commitment: Tensor
    target_soc: Tensor
    raw_structured_actions: Tensor
    constraint_residuals: dict[str, Tensor]
    response_latent: Tensor
    response_log_scale: Tensor
    cost_quantiles: Tensor
    expected_cost: Tensor
    upper_tail_cost: Tensor
    sla_risk: Tensor
    constraint_risk: Tensor
    grid_risk: Tensor
    differentiable_operating_cost: Tensor
    route_weights: Tensor
    selected_expert_ids: Tensor
    tail_decision_active: Tensor
    decision_active_mask: Tensor
    deadline_expired_work: Tensor
    diagnostics: dict[str, Tensor | float | int | str]


class RiskStageTokenizer(nn.Module):
    def __init__(self, config: RECMoEConfig, *, maximum_horizon: int = 96) -> None:
        super().__init__()
        self.config = config
        self.maximum_horizon = int(maximum_horizon)
        self.stage_embedding = nn.Embedding(2, config.risk_hidden_dim)
        self.features = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, config.risk_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.risk_hidden_dim, config.d_model),
        )
        self.stage_projection = nn.Linear(config.risk_hidden_dim, config.d_model)
        self.time_embedding = nn.Parameter(
            torch.randn(maximum_horizon, config.d_model) / math.sqrt(config.d_model)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, risk: RiskConditionPacket, *, time: int) -> Tensor:
        batch = risk.risk_weight.shape[0]
        values = torch.stack(
            (
                risk.risk_weight,
                risk.tail_level,
                risk.cost_budget,
                risk.sla_budget,
                risk.migration_budget,
                risk.ramp_budget,
                risk.battery_degradation_preference,
                risk.horizon.float() / float(self.maximum_horizon),
            ),
            dim=-1,
        ).float()
        base = self.features(values)
        stage = self.stage_projection(self.stage_embedding(risk.stage))
        return self.output_norm(
            base[:, None] + stage[:, None] + self.time_embedding[:time][None]
        )


class ExpertContinuationRouter(nn.Module):
    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.delta = nn.Sequential(
            nn.LayerNorm(3 * config.d_model),
            nn.Linear(3 * config.d_model, config.risk_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.risk_hidden_dim, 2),
        )

    def forward(
        self,
        packet: W1ExpertPacket,
        risk_context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        top2_states = packet.selected_expert_states
        basis = packet.action_response_basis
        joined = torch.cat((packet.state_latent, basis, risk_context), dim=-1)
        delta = torch.tanh(self.delta(joined))[..., : top2_states.shape[2]]
        logits = (
            (0.0 if self.config.mechanism_variant == "no_route_prior"
             else self.config.route_prior_beta)
            * packet.selected_route_logits.detach().float()
            + self.config.route_delta_cap * delta.float()
        )
        weights = torch.softmax(logits, dim=-1).to(top2_states.dtype)
        # Top-1 is a deployment sparsity profile.  During training retain the
        # two already-computed W1 routes so the route-delta receives a smooth
        # decision and continuation gradient; no extra W1 expert is executed.
        keep = top2_states.shape[2] if self.training else self.config.active_top_k
        if keep == top2_states.shape[2]:
            return packet.selected_expert_ids, top2_states, weights, logits
        top_weight, route_index = torch.topk(weights.float(), keep, dim=-1)
        top_weight = top_weight / top_weight.sum(dim=-1, keepdim=True)
        ids = packet.selected_expert_ids.gather(-1, route_index)
        state_index = route_index[..., None].expand(-1, -1, -1, top2_states.shape[-1])
        states = top2_states.gather(2, state_index)
        selected_logits = logits.gather(-1, route_index)
        return ids, states, top_weight.to(states.dtype), selected_logits


class _DecisionExpertAdapter(nn.Module):
    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        width = config.d_model
        hidden = config.adapter_bottleneck
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, 2 * hidden)
        self.risk_film = nn.Linear(width, 2 * hidden)
        self.up = nn.Linear(hidden, width)
        self.output_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, value: Tensor, risk: Tensor) -> Tensor:
        gate, content = self.down(self.norm(value)).chunk(2, dim=-1)
        scale, shift = self.risk_film(risk).chunk(2, dim=-1)
        hidden = F.silu(gate + scale) * (content + shift)
        return value + torch.sigmoid(self.output_gate) * self.up(hidden)


class DecisionExpertAdapterBank(nn.Module):
    """Dispatch only the adapters whose W1 expert ids are active."""

    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleList(
            _DecisionExpertAdapter(config) for _ in range(config.general_experts)
        )
        self.last_executed_experts: tuple[int, ...] = ()

    def forward(self, states: Tensor, ids: Tensor, risk: Tensor) -> Tensor:
        batch, time, routes, width = states.shape
        flat_state = states.reshape(-1, width)
        flat_ids = ids.reshape(-1)
        flat_risk = risk[:, :, None].expand(-1, -1, routes, -1).reshape(-1, width)
        result = torch.zeros_like(flat_state)
        executed: list[int] = []
        for expert_id, adapter in enumerate(self.adapters):
            selected = torch.nonzero(flat_ids == expert_id, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            executed.append(expert_id)
            adapted = adapter(
                flat_state.index_select(0, selected),
                flat_risk.index_select(0, selected),
            )
            result = result.index_copy(0, selected, adapted)
        self.last_executed_experts = tuple(executed)
        return result.reshape(batch, time, routes, width)


class ConditionalTailDecisionAdapter(nn.Module):
    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter = _DecisionExpertAdapter(config)
        self.last_executed_tokens = 0

    def forward(
        self,
        packet: W1ExpertPacket,
        risk_context: Tensor,
        risk_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        threshold = self.config.tail_decision_threshold / (
            0.5 + risk_weight.float().clamp(0.0, 2.0)
        )
        active = packet.tail_active & (
            packet.tail_probability[..., 0].float() >= threshold[:, None]
        )
        flat_active = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
        output = torch.zeros_like(packet.tail_state).reshape(
            -1, packet.tail_state.shape[-1]
        )
        self.last_executed_tokens = int(flat_active.numel())
        if flat_active.numel() > 0:
            flat_state = packet.tail_state.reshape(-1, packet.tail_state.shape[-1])
            flat_risk = risk_context.reshape(-1, risk_context.shape[-1])
            adapted = self.adapter(
                flat_state.index_select(0, flat_active),
                flat_risk.index_select(0, flat_active),
            )
            output = output.index_copy(0, flat_active, adapted)
        return output.reshape_as(packet.tail_state), active


class SparseCrossExpertCoordinator(nn.Module):
    """One gated set-mixing block over active adapters, tail, risk and global."""

    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        width = config.d_model
        self.norm = nn.LayerNorm(width)
        self.score = nn.Linear(width, 1)
        self.update = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * config.risk_hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * config.risk_hidden_dim, width),
        )
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        adapter_states: Tensor,
        route_weights: Tensor,
        tail_state: Tensor,
        tail_active: Tensor,
        risk_context: Tensor,
        global_state: Tensor,
    ) -> Tensor:
        tokens = torch.cat(
            (
                adapter_states,
                tail_state[:, :, None],
                risk_context[:, :, None],
                global_state[:, :, None],
            ),
            dim=2,
        )
        extra = torch.ones(
            *route_weights.shape[:2],
            3,
            device=route_weights.device,
            dtype=route_weights.dtype,
        )
        prior = torch.cat((route_weights.float(), extra.float()), dim=-1)
        mask = torch.cat(
            (
                torch.ones_like(route_weights, dtype=torch.bool),
                tail_active[..., None],
                torch.ones_like(tail_active[..., None]).expand(-1, -1, 2),
            ),
            dim=-1,
        )
        score = self.score(torch.tanh(self.norm(tokens))).squeeze(-1).float()
        score = score + torch.log(prior.clamp_min(1.0e-8))
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1).to(tokens.dtype)
        mixed = (tokens * weight[..., None]).sum(dim=2)
        return global_state + torch.sigmoid(self.gate) * self.update(mixed)


class LowRankTemporalActionHead(nn.Module):
    """Decode all 96 steps with a learned low-rank basis and two einsums."""

    def __init__(self, config: RECMoEConfig, *, horizon: int, output_dim: int) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.rank = int(config.temporal_rank)
        self.output_dim = int(output_dim)
        self.basis = nn.Parameter(
            torch.randn(horizon, self.rank) / math.sqrt(self.rank)
        )
        self.coefficient = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.risk_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.risk_hidden_dim, output_dim),
        )
        self.local_residual = nn.Linear(config.d_model, output_dim)
        self.local_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, timeline: Tensor) -> Tensor:
        batch, time, width = timeline.shape
        if time > self.horizon:
            raise ValueError("R temporal input exceeds configured horizon")
        basis = torch.tanh(self.basis[:time])
        normalization = basis.square().sum(dim=0).clamp_min(1.0e-6).sqrt()
        normalized = basis / normalization[None]
        rank_state = torch.einsum("tr,btd->brd", normalized, timeline)
        coefficients = self.coefficient(rank_state)
        low_rank = torch.einsum("tr,bro->bto", normalized, coefficients)
        local = self.local_residual(timeline)
        return low_rank + torch.sigmoid(self.local_gate) * local


class StructuredActionHeads(nn.Module):
    """One head for DA, battery, compute, migration, PV and target SOC."""

    output_dim = 4 + 4 + 4 + 16 + 4 + 4

    def __init__(self, config: RECMoEConfig, *, horizon: int) -> None:
        super().__init__()
        self.temporal = LowRankTemporalActionHead(
            config, horizon=horizon, output_dim=self.output_dim
        )

    def forward(self, timeline: Tensor) -> dict[str, Tensor]:
        raw = self.temporal(timeline)
        return {
            "joined": raw,
            "day_ahead": raw[..., 0:4],
            "battery": raw[..., 4:8],
            "compute": raw[..., 8:12],
            "migration": raw[..., 12:28].reshape(*raw.shape[:2], NODES, NODES),
            "pv": raw[..., 28:32],
            "target_soc": raw[..., 32:36],
        }


def _prepend(value: Tensor, initial: Tensor) -> Tensor:
    return torch.cat((initial[:, None], value), dim=1)


def deadline_expired_work_parallel(
    arrivals: Tensor,
    service: Tensor,
    initial_queue: Tensor,
    active_mask: Tensor,
    *,
    deadline_intervals: int,
) -> Tensor:
    """Vectorized FIFO newly-expired work for one deterministic trajectory."""

    if arrivals.shape != service.shape or arrivals.ndim != 3:
        raise ValueError("deadline arrivals/service must be matching [B,T,4]")
    batch, time, nodes = arrivals.shape
    if nodes != NODES or initial_queue.shape != (batch, NODES):
        raise ValueError("deadline state must use four VPP nodes")
    if active_mask.shape != (batch, time) or active_mask.dtype is not torch.bool:
        raise ValueError("deadline active mask must be bool [B,T]")
    delay = int(deadline_intervals)
    if delay < 1:
        raise ValueError("deadline interval must be positive")
    if delay > time:
        return arrivals.new_zeros(batch, time, dtype=torch.float32)
    active = active_mask.float()
    arrival_total = arrivals.float().sum(dim=-1) * active
    service_total = service.float().sum(dim=-1) * active
    # Insert the issue-origin queue at the first active decision, not at t=0;
    # constraints.initial_queue already represents the state at that origin.
    first_active = active_mask & (torch.cumsum(active_mask.long(), dim=1) == 1)
    arrival_total = (
        arrival_total
        + first_active.float() * initial_queue.float().sum(dim=-1)[:, None]
    )
    cumulative_arrival = torch.cumsum(arrival_total, dim=1)
    cumulative_service = torch.cumsum(service_total, dim=1)
    due = torch.zeros_like(cumulative_arrival)
    due[:, delay - 1 :] = cumulative_arrival[:, : time - delay + 1]
    due_deficit = (due - cumulative_service).clamp_min(0.0)
    cumulative_expired = torch.cummax(due_deficit, dim=1).values
    expired = torch.diff(
        _prepend(cumulative_expired, torch.zeros_like(cumulative_expired[:, 0])),
        dim=1,
    ).clamp_min(0.0)
    # Work that would expire after the requested decision horizon belongs to a
    # later rolling window and must not enter this window's SLA loss or gate.
    return expired * active


class NeuralConstraintAssembler(nn.Module):
    """One vectorized, deterministic action mapping; not an iterative repair."""

    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _decision_masks(
        *,
        batch: int,
        time: int,
        origin_index: Tensor,
        stage: Tensor,
        device: torch.device,
        horizon: Tensor,
    ) -> tuple[Tensor, Tensor]:
        index = torch.arange(time, device=device)[None].expand(batch, -1)
        start = origin_index[:, None]
        stop = start + horizon[:, None]
        active = index.ge(start) & index.lt(stop)
        past = index.lt(start)
        return active, past

    def forward(
        self,
        raw: dict[str, Tensor],
        packet: W1ExpertPacket,
        risk: RiskConditionPacket,
    ) -> ConstraintAssemblyResult:
        reference = packet.reference
        constraints = packet.constraints
        batch, time, _ = raw["battery"].shape
        device = raw["battery"].device
        dtype = torch.float32
        active, past = self._decision_masks(
            batch=batch,
            time=time,
            origin_index=packet.origin_index,
            stage=risk.stage,
            device=device,
            horizon=risk.horizon,
        )
        previous = packet.previous_actions[:, :time].float()
        available_mask = packet.valid_action_masks[:, :time]
        action_range = risk.action_range.float()[:, None, None]

        dt = constraints.dt_hours.float()[:, None]
        power_limit = constraints.battery_power_limit.float()[:, None]
        energy_capacity = constraints.battery_energy_capacity.float()
        charge_efficiency = constraints.charge_efficiency.float()
        discharge_efficiency = constraints.discharge_efficiency.float()

        desired_battery = torch.tanh(
            raw["battery"].float() * action_range
        ) * power_limit
        historical_battery_available = available_mask[..., 1]
        historical_battery = torch.where(
            past[..., None] & historical_battery_available,
            previous[..., 1],
            torch.zeros_like(desired_battery),
        )
        active_battery_request = torch.where(active[..., None], desired_battery, 0.0)
        initial_soc = constraints.initial_soc.float().clamp_min(0.0)
        initial_soc = torch.minimum(initial_soc, energy_capacity)
        fixed_soc_lower = torch.zeros_like(initial_soc)
        fixed_soc_upper = torch.zeros_like(initial_soc)

        requested_discharge = active_battery_request.clamp_min(0.0)
        requested_charge = (-active_battery_request).clamp_min(0.0)
        discharge_energy = (requested_discharge / discharge_efficiency[:, None]) * dt[
            ..., None
        ]
        charge_energy = (requested_charge * charge_efficiency[:, None]) * dt[..., None]
        discharge_scale = torch.minimum(
            torch.ones_like(initial_soc),
            initial_soc / discharge_energy.sum(dim=1).clamp_min(1.0e-8),
        )
        charge_scale = torch.minimum(
            torch.ones_like(initial_soc),
            (energy_capacity - initial_soc).clamp_min(0.0)
            / charge_energy.sum(dim=1).clamp_min(1.0e-8),
        )
        planned_battery = (
            requested_discharge * discharge_scale[:, None]
            - requested_charge * charge_scale[:, None]
        )
        battery = historical_battery + planned_battery

        compute_capacity = constraints.compute_capacity.float()[:, None]
        desired_compute = torch.sigmoid(
            raw["compute"].float() * action_range
        ) * compute_capacity
        historical_compute = torch.where(
            past[..., None] & available_mask[..., 2],
            previous[..., 2].clamp_min(0.0),
            torch.zeros_like(desired_compute),
        )
        compute_request = torch.where(active[..., None], desired_compute, 0.0)

        migration_capacity = constraints.migration_capacity.float()[:, None]
        migration_allowed = constraints.migration_allowed[:, None]
        migration_request = torch.tanh(
            raw["migration"].float() * action_range[..., None]
        ).clamp_min(0.0)
        migration_request = migration_request * migration_capacity
        migration_request = (
            migration_request
            * risk.migration_budget.float().clamp(0.0, 1.0)[:, None, None, None]
        )
        migration_request = migration_request * migration_allowed.to(dtype)
        diagonal = torch.eye(NODES, device=device, dtype=torch.bool)[None, None]
        migration_request = migration_request.masked_fill(diagonal, 0.0)
        migration_request = torch.where(active[..., None, None], migration_request, 0.0)
        outgoing_request = migration_request.sum(dim=-1)
        service_request = compute_request + outgoing_request
        arrivals = (
            reference.arrivals[:, :time].float().clamp_min(0.0)
            * active[..., None].float()
        )
        cumulative_requested = torch.cumsum(service_request, dim=1)
        cumulative_available = constraints.initial_queue.float()[
            :, None
        ] + torch.cumsum(arrivals, dim=1)
        feasible_cumulative = torch.minimum(cumulative_requested, cumulative_available)
        feasible_increment = torch.diff(
            _prepend(feasible_cumulative, torch.zeros_like(feasible_cumulative[:, 0])),
            dim=1,
        ).clamp_min(0.0)
        service = torch.minimum(feasible_increment, service_request)
        service_scale = service / service_request.clamp_min(1.0e-8)
        active_compute = compute_request * service_scale
        migration = migration_request * service_scale[..., None]
        outgoing = migration.sum(dim=-1)
        incoming = migration.sum(dim=-2)
        queue_delta = arrivals + incoming - outgoing - active_compute
        queue = constraints.initial_queue.float()[:, None] + torch.cumsum(
            queue_delta, dim=1
        )
        queue_lower_residual = (-queue).clamp_min(0.0)
        queue = queue.clamp_min(0.0)
        queue_trace = _prepend(queue, constraints.initial_queue.float())
        compute = historical_compute + active_compute

        pue = reference.pue[:, :time].float()
        pv_available = reference.pv[:, :time].float().clamp_min(0.0)
        pv_requested = torch.sigmoid(
            raw["pv"].float() * action_range + 2.0
        ) * pv_available
        fixed_pv_curtail = torch.where(
            past[..., None] & available_mask[..., 5],
            previous[..., 5].clamp_min(0.0),
            torch.zeros_like(pv_requested),
        )
        fixed_pv_used = (pv_available - fixed_pv_curtail).clamp_min(0.0)
        pv_requested = torch.where(past[..., None], fixed_pv_used, pv_requested)
        facility = (
            constraints.idle_power.float()[:, None]
            + constraints.work_power_per_unit.float()[:, None] * compute
        ) * pue
        export_limit = constraints.grid_export_limit.float()[:, None]
        # A storage discharge cannot exceed contemporaneous facility demand
        # plus the export capability, even when PV is fully curtailed.  The
        # earlier SOC/power map is necessary but not sufficient for grid
        # closure.  Apply this single vectorized cross-family bound, then form
        # the SOC trace from the executable battery trajectory.
        maximum_export_safe_discharge = (facility + export_limit).clamp_min(0.0)
        planned_battery = torch.minimum(
            planned_battery.clamp_min(0.0), maximum_export_safe_discharge
        ) - (-planned_battery).clamp_min(0.0)
        battery = historical_battery + planned_battery
        battery_delta = (
            (-planned_battery).clamp_min(0.0) * charge_efficiency[:, None]
            - planned_battery.clamp_min(0.0) / discharge_efficiency[:, None]
        ) * dt[..., None]
        soc = initial_soc[:, None] + torch.cumsum(battery_delta, dim=1)
        soc_trace = _prepend(soc, initial_soc)
        maximum_pv_used = (facility - battery + export_limit).clamp_min(0.0)
        pv_used = torch.minimum(pv_requested, maximum_pv_used)
        pv_used = torch.minimum(pv_used, pv_available)
        pv_curtailed = pv_available - pv_used
        # Keep the grouping consistent with ``maximum_pv_used`` so the export
        # boundary does not lose another ulp by subtracting PV first.
        net = facility - battery - pv_used
        import_limit = constraints.grid_import_limit.float()[:, None]
        grid = torch.maximum(torch.minimum(net, import_limit), -export_limit)
        unserved = (net - grid).clamp_min(0.0)
        # Compute closure from the already formed net balance.  Reconstructing
        # ``facility - (grid + pv + battery + unserved)`` subtracts several
        # large physical-unit tensors in a different order and produced a
        # spurious ~2e-1 FP32 residual at RT origin 95 even though grid was
        # derived from this exact net.  The ordered form is algebraically
        # identical and is exactly zero for the by-construction branches.
        raw_power_balance_residual = net - grid - unserved
        closure_roundoff_scale = (
            facility.abs()
            + battery.abs()
            + pv_used.abs()
            + grid.abs()
            + unserved.abs()
        ).clamp_min(1.0)
        closure_roundoff_tolerance = (
            4.0 * torch.finfo(torch.float32).eps * closure_roundoff_scale
        )
        power_balance_residual = torch.where(
            raw_power_balance_residual.abs() <= closure_roundoff_tolerance,
            torch.zeros_like(raw_power_balance_residual),
            raw_power_balance_residual,
        )
        active_node = active[..., None].to(dtype)
        active_edge = active[..., None, None].to(dtype)
        state_active = torch.cat(
            (
                torch.zeros(batch, 1, device=device, dtype=torch.bool),
                active,
            ),
            dim=1,
        )[..., None].to(dtype)
        after_horizon = ~(active | past)
        # Past battery/compute/PV values remain available as causal logs, but
        # non-active electrical decisions and violations are not part of this
        # rolling decision window.
        pv_used = torch.where(after_horizon[..., None], 0.0, pv_used)
        pv_curtailed = torch.where(after_horizon[..., None], 0.0, pv_curtailed)
        grid = grid * active_node
        unserved = unserved * active_node
        power_balance_residual = power_balance_residual * active_node

        base_commitment = torch.where(
            available_mask[..., 0], previous[..., 0], grid.detach()
        )
        da_residual = (
            torch.tanh(raw["day_ahead"].float() * action_range)
            * self.config.action_residual_fraction
            * import_limit
        )
        generated_day_ahead = torch.maximum(
            torch.minimum(base_commitment + da_residual, import_limit),
            -export_limit,
        )
        # In RT the market commitment was already published and is frozen over
        # the complete day.  Only the DA stage may generate a new commitment.
        day_ahead = torch.where(
            risk.stage[:, None, None].eq(1) & available_mask[..., 0],
            previous[..., 0],
            generated_day_ahead,
        )
        target_soc = torch.sigmoid(
            raw["target_soc"].float() * action_range
        ) * energy_capacity[:, None]

        # Global migration conservation is structural: incoming and outgoing
        # are two views of the same directed flow tensor.  Re-summing them in
        # different FP32 orders can create a false residual at large scale, so
        # report the exact by-construction invariant here.  Per-edge capacity,
        # support, nonnegativity, and node queue effects remain checked above.
        migration_balance = torch.zeros_like(active, dtype=dtype)
        battery_power_excess = (battery.abs() - power_limit).clamp_min(
            0.0
        ) * active_node
        soc_lower = (-soc_trace).clamp_min(0.0) * state_active
        soc_upper = (soc_trace - energy_capacity[:, None]).clamp_min(0.0) * state_active
        compute_excess = (compute - compute_capacity).clamp_min(0.0) * active_node
        migration_excess = (migration - migration_capacity).clamp_min(0.0) * active_edge
        pv_excess = (pv_used - pv_available).clamp_min(0.0) * active_node
        grid_import_excess = (grid - import_limit).clamp_min(0.0) * active_node
        grid_export_excess = (-export_limit - grid).clamp_min(0.0) * active_node
        # Historical actions are retained only as logs.  They are not a
        # feasibility target and therefore contribute no residual.
        fixed_past_residual = torch.zeros_like(battery)
        critical_shortfall = queue_trace[:, -1]
        residuals = {
            "battery_power": battery_power_excess,
            "soc_lower": soc_lower,
            "soc_upper": soc_upper,
            "fixed_soc_lower": fixed_soc_lower,
            "fixed_soc_upper": fixed_soc_upper,
            "compute_capacity": compute_excess,
            "work_conservation": queue_lower_residual * active_node,
            "migration_capacity": migration_excess,
            "migration_conservation": migration_balance.abs(),
            "pv_limit": pv_excess,
            "grid_import": grid_import_excess,
            "grid_export": grid_export_excess,
            "power_balance": power_balance_residual.abs() * active_node,
            "fixed_past": fixed_past_residual,
            "critical_shortfall": critical_shortfall,
            "target_soc_tracking": (soc_trace[:, 1:] - target_soc).abs()
            * active[..., None].float(),
        }
        violation = (
            unserved
            + queue_lower_residual * active_node
            + battery_power_excess
            + compute_excess
            + grid_import_excess
            + grid_export_excess
        ) * active_node
        shield_intervention = (
            (desired_battery - battery).abs()
            + (desired_compute - compute).abs()
            + pv_curtailed
            + unserved
        ) * active_node
        actions = WorldActionPacket(
            battery=battery,
            compute=compute,
            migration=migration,
            grid=grid,
            pv_used=pv_used,
            pv_curtailed=pv_curtailed,
            unserved_power=unserved,
            soc=soc_trace,
            queue=queue_trace,
            power_balance_residual=power_balance_residual,
            migration_balance_residual=migration_balance,
            constraint_violation=violation,
            shield_intervention=shield_intervention,
        ).validate()
        migration_total = migration.sum(dim=(-1, -2)) / float(NODES)
        action_summary = torch.stack(
            (
                battery.mean(dim=-1),
                compute.mean(dim=-1),
                migration_total,
                grid.mean(dim=-1),
                pv_curtailed.mean(dim=-1),
                unserved.mean(dim=-1),
            ),
            dim=-1,
        )
        action_summary = action_summary * active[..., None].to(action_summary.dtype)
        deadline_expired = deadline_expired_work_parallel(
            reference.arrivals[:, :time],
            active_compute,
            constraints.initial_queue,
            active,
            deadline_intervals=self.config.deadline_intervals,
        )
        return ConstraintAssemblyResult(
            actions=actions,
            day_ahead_commitment=day_ahead,
            target_soc=target_soc,
            raw_structured_actions=raw["joined"],
            residuals=residuals,
            action_summary=action_summary,
            decision_active_mask=active,
            deadline_expired_work=deadline_expired,
        )


class DistributionalRiskValueHead(nn.Module):
    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        width = config.d_model
        hidden = 2 * config.risk_hidden_dim
        self.base_projection = nn.Linear(29, width)
        self.uncertainty_projection = nn.Linear(41, width)
        self.scale_projection = nn.Linear(1, width)
        self.attention = nn.Linear(width, 1)
        self.shared = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, hidden),
            nn.SiLU(),
        )
        self.quantile = nn.Linear(hidden, len(COST_QUANTILES))
        self.risks = nn.Linear(hidden, 3)
        self.register_buffer(
            "dense_probability_grid",
            torch.linspace(0.0, 1.0, 65, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _dense_quantile_grid(quantiles: Tensor, grid: Tensor) -> Tensor:
        levels = quantiles.new_tensor(COST_QUANTILES, dtype=torch.float32)
        upper = torch.searchsorted(levels, grid, right=False).clamp(
            0, len(COST_QUANTILES) - 1
        )
        lower = (upper - 1).clamp(0, len(COST_QUANTILES) - 1)
        lower_level = levels.index_select(0, lower)
        upper_level = levels.index_select(0, upper)
        denominator = (upper_level - lower_level).clamp_min(1.0e-8)
        fraction = ((grid - lower_level) / denominator).clamp(0.0, 1.0)
        lower_value = quantiles.index_select(-1, lower)
        upper_value = quantiles.index_select(-1, upper)
        dense = lower_value + fraction[None] * (upper_value - lower_value)
        dense = torch.where(grid[None] <= levels[0], quantiles[:, :1], dense)
        return torch.where(grid[None] >= levels[-1], quantiles[:, -1:], dense)

    def forward(
        self,
        decision_timeline: Tensor,
        response_timeline: Tensor,
        response_log_scale: Tensor,
        packet: W1ExpertPacket,
        risk_context: Tensor,
        active_mask: Tensor,
        tail_level: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        timeline = (
            decision_timeline
            + response_timeline
            + risk_context
            + self.base_projection(packet.base_world_summary.float()).to(
                decision_timeline.dtype
            )
            + self.uncertainty_projection(packet.uncertainty_summary.float()).to(
                decision_timeline.dtype
            )
            + self.scale_projection(response_log_scale.float()).to(
                decision_timeline.dtype
            )
        )
        score = self.attention(torch.tanh(timeline)).squeeze(-1).float()
        score = score.masked_fill(~active_mask, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1).to(timeline.dtype)
        pooled = (timeline * weight[..., None]).sum(dim=1)
        masked = timeline.float().masked_fill(~active_mask[..., None], -torch.inf)
        maximum = masked.amax(dim=1).to(timeline.dtype)
        terminal_index = (
            packet.origin_index + active_mask.long().sum(dim=1) - 1
        ).clamp(0, timeline.shape[1] - 1)
        row = torch.arange(timeline.shape[0], device=timeline.device)
        terminal = timeline[row, terminal_index]
        hidden = self.shared(torch.cat((pooled, maximum, terminal), dim=-1))
        raw = self.quantile(hidden).float()
        first = raw[:, :1]
        increments = F.softplus(raw[:, 1:])
        quantiles = torch.cat((first, first + torch.cumsum(increments, dim=-1)), dim=-1)
        grid = self.dense_probability_grid.to(device=quantiles.device)
        dense = self._dense_quantile_grid(quantiles, grid)
        expected = torch.trapz(dense, grid, dim=-1)
        tail_mask = grid[None] >= tail_level.float()[:, None]
        upper_tail = (dense * tail_mask.float()).sum(dim=-1) / tail_mask.float().sum(
            dim=-1
        ).clamp_min(1.0)
        risks = torch.sigmoid(self.risks(hidden).float())
        return quantiles, expected, upper_tail, risks[:, 0], risks[:, 1], risks[:, 2]


class RActionResponseAdapter(nn.Module):
    """R-owned residual over the frozen W1 parallel action response."""

    def __init__(self, config: RECMoEConfig) -> None:
        super().__init__()
        width = config.d_model
        self.norm = nn.LayerNorm(2 * width)
        self.down = nn.Linear(2 * width, config.adapter_bottleneck)
        self.up = nn.Linear(config.adapter_bottleneck, width)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, response: Tensor, basis: Tensor) -> Tensor:
        residual = self.up(
            F.silu(self.down(self.norm(torch.cat((response, basis), dim=-1))))
        )
        return response + torch.sigmoid(self.gate) * residual


def differentiable_operating_cost(
    *,
    reference: WorldObservationPacket,
    actions: WorldActionPacket,
    day_ahead_commitment: Tensor,
    constraints: WorldConstraintPacket,
    decision_active_mask: Tensor,
    deadline_expired_work: Tensor,
    battery_degradation_weight: float = 0.01,
    migration_weight: float = 0.02,
    sla_weight: float = 100.0,
) -> Tensor:
    """One-path two-settlement objective with true newly-expired work."""

    horizon = actions.horizon
    if decision_active_mask.shape != (actions.battery.shape[0], horizon):
        raise ValueError("decision_active_mask must be [B,T]")
    if deadline_expired_work.shape != decision_active_mask.shape:
        raise ValueError("deadline_expired_work must be [B,T]")
    active = decision_active_mask.float()
    dt = constraints.dt_hours.float()[:, None, None]
    da = reference.da[:, :horizon].float()
    rt = reference.rt[:, :horizon].float()
    commitment = day_ahead_commitment.float()
    grid = actions.grid.float()
    market = (da * commitment + rt * (grid - commitment)) * dt
    market = market * active[..., None]
    battery = battery_degradation_weight * actions.battery.float().abs() * dt
    battery = battery * active[..., None]
    migration = migration_weight * actions.migration.float().sum(dim=(-1, -2))
    migration = migration * constraints.dt_hours.float()[:, None] * active
    energy_service = (
        (
            actions.pv_curtailed.float().sum(dim=-1)
            + sla_weight * actions.unserved_power.float().sum(dim=-1)
        )
        * constraints.dt_hours.float()[:, None]
        * active
    )
    deadline_service = sla_weight * (deadline_expired_work.float() * active).sum(dim=-1)
    return (
        market.sum(dim=(-1, -2))
        + battery.sum(dim=(-1, -2))
        + migration.sum(dim=-1)
        + energy_service.sum(dim=-1)
        + deadline_service
    )


class RECMoE(nn.Module):
    """Direct W0 -> W1-MoE -> risk-conditioned sparse decision model."""

    architecture_version = "risk_conditioned_expert_continuation_moe_v1"
    external_solver_free = True
    candidate_search_free = True

    def __init__(self, world_model: VPPWorldMoE, config: RECMoEConfig) -> None:
        super().__init__()
        if world_model.config.d_model != config.d_model:
            raise ValueError("R-ECMoE width must match the released W1")
        if world_model.config.general_experts != config.general_experts:
            raise ValueError("R-ECMoE expert count must match the released W1")
        self.world_model = world_model
        self.config = config
        self.risk_tokenizer = RiskStageTokenizer(
            config, maximum_horizon=world_model.config.horizon
        )
        self.continuation_router = ExpertContinuationRouter(config)
        self.adapter_bank = DecisionExpertAdapterBank(config)
        self.tail_adapter = ConditionalTailDecisionAdapter(config)
        self.coordinator = SparseCrossExpertCoordinator(config)
        self.action_heads = StructuredActionHeads(
            config, horizon=world_model.config.horizon
        )
        self.assembler = NeuralConstraintAssembler(config)
        self.response_adapter = RActionResponseAdapter(config)
        self.risk_value_head = DistributionalRiskValueHead(config)
        self.causal_context_projection = nn.Linear(2, config.d_model)
        self.stage: StageName = "ra"
        self.w1_backbone_calls = 0
        # The released W1 emits one stable packet contract.  Full finite/value
        # validation synchronizes CUDA tensors with the host, so prove the
        # contract on the first packet only and keep steady-state forwards
        # entirely on device.
        self._packet_contract_verified = False
        self.configure_stage("ra")

    def train(self, mode: bool = True) -> "RECMoE":
        """Train R modules while keeping the frozen W0/W1 context deterministic."""

        super().train(mode)
        # The released W1 inherits nonzero dropout.  Its frozen packet must not
        # change stochastically between R updates or differ from deployment.
        self.world_model.eval()
        return self

    def configure_stage(self, stage: StageName) -> None:
        if stage not in {"ra", "rb", "rc"}:
            raise ValueError("R-ECMoE stage must be ra, rb, or rc")
        self.stage = stage
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)
        self.world_model.eval()
        self.apply_mechanism_trainability()

    @property
    def disabled_mechanism_modules(self) -> tuple[str, ...]:
        """Unused R modules, still initialized to preserve paired RNG streams."""

        return {
            "no_sparse_coordinator": ("coordinator",),
            "no_risk_condition": ("risk_tokenizer",),
            "no_decision_tail": ("tail_adapter",),
        }.get(self.config.mechanism_variant, ())

    def apply_mechanism_trainability(self) -> None:
        """Keep bypassed modules out of the optimizer after any stage change.

        Existing risk-value, cost, route and physical objectives remain valid
        for these decision-path interventions. In particular, the loss named
        ``tail_risk`` supervises the retained risk-value head, not the bypassed
        decision-tail adapter.
        """

        for name in self.disabled_mechanism_modules:
            module = getattr(self, name)
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def mechanism_definition(self) -> dict[str, Any]:
        return {
            "mechanism_variant": self.config.mechanism_variant,
            "active_top_k": self.config.active_top_k,
            "effective_route_prior_beta": (
                0.0 if self.config.mechanism_variant == "no_route_prior"
                else self.config.route_prior_beta
            ),
            "disabled_modules": list(self.disabled_mechanism_modules),
            "disabled_auxiliary_losses": [],
            "world_model_changed_by_variant": False,
        }

    @property
    def world_backbone_trainable(self) -> bool:
        return any(
            parameter.requires_grad
            for parameter in self.world_model.backbone.parameters()
        )

    def encode_expert_context(
        self,
        inputs: WorldInputPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> W1ExpertPacket:
        """Execute the released W0/W1 main experts exactly once."""

        self.w1_backbone_calls += 1
        context = (
            torch.no_grad() if not self.world_backbone_trainable else nullcontext()
        )
        with context:
            state = self.world_model.encode(
                inputs,
                sample_stochastic=False,
                phase="w1",
                capture_expert_context=True,
            )
            diagnostic = state.router_diagnostics[-1]
            selected_states = getattr(
                diagnostic, "captured_selected_expert_states", None
            )
            tail_state = getattr(diagnostic, "captured_tail_state", None)
            pre_risk_state = getattr(diagnostic, "captured_pre_state", None)
            capture_slot = getattr(diagnostic, "capture_slot", None)
            if (
                selected_states is None
                or tail_state is None
                or pre_risk_state is None
                or capture_slot != TAIL_RISK_SLOT
            ):
                raise RuntimeError(
                    "released W1 did not return the compact final risk-slot expert hook"
                )
            ids = diagnostic.topk_indices[:, :, TAIL_RISK_SLOT]
            weights = diagnostic.topk_weights[:, :, TAIL_RISK_SLOT]
            logits = diagnostic.logits[:, :, TAIL_RISK_SLOT].gather(-1, ids)
            reference = self.world_model.query_observations(
                state,
                inputs.constraints,
                da_anchor=da_anchor,
                da_residual_cap=da_residual_cap,
            )
            queue = reference.queue[:, 1:]
            base_summary = torch.cat(
                (reference.public_21, queue, reference.soc), dim=-1
            )
            uncertainty = torch.cat(
                (
                    state.regime_probabilities[:, :, TAIL_RISK_SLOT],
                    state.stochastic_log_scale[:, :, TAIL_RISK_SLOT],
                    diagnostic.tail_route_probabilities[..., None],
                ),
                dim=-1,
            )
            final_moe = self.world_model.backbone.blocks[
                max(self.world_model.config.moe_block_indices)
            ].moe
            if final_moe is None:
                raise RuntimeError("released W1 final MoE block is missing")
            # The hook captures raw expert deltas.  Continue a route-specific
            # effective state by combining the captured pre-MoE risk state and
            # the exact frozen W1 output gate.  The always-on shared contribution
            # remains available through the full post-backbone state_latent.
            selected_states = (
                pre_risk_state[:, :, None]
                + torch.sigmoid(final_moe.output_gate) * selected_states
            )
            tail_state = final_moe.tail_residual_scale * tail_state
            action_basis = (selected_states * weights[..., None]).sum(dim=2)
            origin_fraction = state.origin_index.float() / float(
                max(1, state.pooled_timeline.shape[1] - 1)
            )
            causal_context = torch.stack(
                (
                    origin_fraction[:, None].expand(-1, state.pooled_timeline.shape[1]),
                    inputs.constraints.dt_hours.float()[:, None].expand(
                        -1, state.pooled_timeline.shape[1]
                    ),
                ),
                dim=-1,
            )
        packet = W1ExpertPacket(
            state_latent=state.pooled_timeline,
            selected_expert_ids=ids,
            selected_route_logits=logits,
            selected_route_weights=weights,
            selected_expert_states=selected_states,
            pre_risk_state=pre_risk_state,
            tail_probability=diagnostic.tail_route_probabilities[..., None],
            tail_active=diagnostic.tail_active_mask,
            tail_state=tail_state,
            action_response_basis=action_basis,
            base_world_summary=base_summary,
            uncertainty_summary=uncertainty,
            valid_action_masks=inputs.action_mask,
            causal_context=causal_context,
            origin_index=state.origin_index,
            constraints=inputs.constraints,
            reference=reference,
            previous_actions=inputs.previous_actions,
            source_state=state,
        )
        if not self._packet_contract_verified:
            packet.validate(
                self.config.d_model,
                self.world_model.config.regular_top_k,
                strict_values=True,
            )
            self._packet_contract_verified = True
        return packet

    def decide(
        self,
        packet: W1ExpertPacket,
        risk: RiskConditionPacket,
    ) -> tuple[ConstraintAssemblyResult, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, time, _ = packet.state_latent.shape
        risk.validate(batch, time)
        variant = self.config.mechanism_variant
        if variant == "no_risk_condition":
            risk_context = self.causal_context_projection(packet.causal_context.float())
        else:
            risk_context = self.risk_tokenizer(risk, time=time)
            risk_context = risk_context + self.causal_context_projection(
                packet.causal_context.to(risk_context.dtype)
            )
        ids, selected_states, weights, _ = self.continuation_router(
            packet, risk_context
        )
        if variant == "pooled_latent_instead_of_expert_states":
            selected_states = packet.state_latent[:, :, None].expand(
                -1, -1, ids.shape[-1], -1
            )
        adapted = self.adapter_bank(selected_states, ids, risk_context)
        if variant == "no_decision_tail":
            tail_state = torch.zeros_like(packet.tail_state)
            tail_active = torch.zeros(
                batch, time, device=packet.state_latent.device, dtype=torch.bool
            )
            self.tail_adapter.last_executed_tokens = 0
        elif variant == "tail_always_on":
            tail_active = torch.ones(
                batch, time, device=packet.state_latent.device, dtype=torch.bool
            )
            # This changes only the R adapter gate; the frozen W1 packet and
            # its own conditional tail-expert computation stay unchanged.
            tail_state = self.tail_adapter.adapter(packet.tail_state, risk_context)
            self.tail_adapter.last_executed_tokens = batch * time
        else:
            risk_weight = (
                torch.ones_like(risk.risk_weight)
                if variant == "no_risk_condition" else risk.risk_weight
            )
            tail_state, tail_active = self.tail_adapter(packet, risk_context, risk_weight)
        if variant == "no_sparse_coordinator":
            coordinated = (adapted * weights.to(adapted.dtype)[..., None]).sum(dim=2)
            coordinated = coordinated + torch.where(
                tail_active[..., None], tail_state, torch.zeros_like(tail_state)
            )
        else:
            coordinated = self.coordinator(
                adapted,
                weights,
                tail_state,
                tail_active,
                risk_context,
                packet.state_latent,
            )
        raw = self.action_heads(coordinated)
        assembled = self.assembler(raw, packet, risk)
        return assembled, coordinated, weights, ids, tail_active, risk_context

    def apply_action_response(
        self,
        packet: W1ExpertPacket,
        action_summary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        method = getattr(self.world_model.dynamics, "parallel_action_response", None)
        if method is None:
            raise RuntimeError("W1 lightweight parallel action response is unavailable")
        response, scale = method(
            packet.state_latent.detach(), action_summary, packet.constraints
        )
        return (
            self.response_adapter(response, packet.action_response_basis.detach()),
            scale,
        )

    def forward(
        self,
        inputs: WorldInputPacket,
        risk: RiskConditionPacket | None = None,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> RECMoEOutput:
        calls_before = self.w1_backbone_calls
        packet = self.encode_expert_context(
            inputs,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )
        batch, time, _ = packet.state_latent.shape
        if risk is None:
            risk = RiskConditionPacket.defaults(
                batch,
                device=packet.state_latent.device,
                horizon=time,
                stage=1,
            )
        (
            assembled,
            decision_state,
            weights,
            ids,
            tail_active,
            risk_context,
        ) = self.decide(packet, risk)
        response, response_scale = self.apply_action_response(
            packet, assembled.action_summary
        )
        (
            cost_quantiles,
            expected_cost,
            upper_tail_cost,
            sla_risk,
            constraint_risk,
            grid_risk,
        ) = self.risk_value_head(
            decision_state,
            response,
            response_scale,
            packet,
            risk_context,
            assembled.decision_active_mask,
            risk.tail_level,
        )
        operating_cost = differentiable_operating_cost(
            reference=packet.reference,
            actions=assembled.actions,
            day_ahead_commitment=assembled.day_ahead_commitment,
            constraints=packet.constraints,
            decision_active_mask=assembled.decision_active_mask,
            deadline_expired_work=assembled.deadline_expired_work,
            battery_degradation_weight=self.world_model.config.battery_degradation_weight,
            migration_weight=self.world_model.config.migration_weight,
            sla_weight=self.world_model.config.unserved_power_weight,
        )
        diagnostics: dict[str, Tensor | float | int | str]
        if return_diagnostics:
            diagnostics = {
                "profile": self.config.profile,
                "mechanism_variant": self.config.mechanism_variant,
                "stage": self.stage,
                "w1_backbone_calls": self.w1_backbone_calls - calls_before,
                "active_top_k": weights.shape[-1],
                "executed_general_adapters": len(
                    self.adapter_bank.last_executed_experts
                ),
                "tail_executed_tokens": self.tail_adapter.last_executed_tokens,
                "tail_activation_rate": tail_active.float().mean(),
                "route_entropy": -(
                    weights.float() * torch.log(weights.float().clamp_min(1.0e-8))
                )
                .sum(dim=-1)
                .mean(),
            }
        else:
            diagnostics = {}
        return RECMoEOutput(
            expert_packet=packet,
            actions=assembled.actions,
            day_ahead_commitment=assembled.day_ahead_commitment,
            target_soc=assembled.target_soc,
            raw_structured_actions=assembled.raw_structured_actions,
            constraint_residuals=assembled.residuals,
            response_latent=response,
            response_log_scale=response_scale,
            cost_quantiles=cost_quantiles,
            expected_cost=expected_cost,
            upper_tail_cost=upper_tail_cost,
            sla_risk=sla_risk,
            constraint_risk=constraint_risk,
            grid_risk=grid_risk,
            differentiable_operating_cost=operating_cost,
            route_weights=weights,
            selected_expert_ids=ids,
            tail_decision_active=tail_active,
            decision_active_mask=assembled.decision_active_mask,
            deadline_expired_work=assembled.deadline_expired_work,
            diagnostics=diagnostics,
        )


__all__ = [
    "COST_QUANTILES",
    "MECHANISM_VARIANTS",
    "MechanismVariant",
    "ConstraintAssemblyResult",
    "DecisionExpertAdapterBank",
    "DistributionalRiskValueHead",
    "ExpertContinuationRouter",
    "NeuralConstraintAssembler",
    "RECMoE",
    "RECMoEConfig",
    "RECMoEOutput",
    "RiskConditionPacket",
    "RiskStageTokenizer",
    "SparseCrossExpertCoordinator",
    "StructuredActionHeads",
    "W1ExpertPacket",
    "differentiable_operating_cost",
]
