"""Matched-training decision baselines for the released W1 world model.

Both baselines execute the released W1 exactly once and retain its causal
pooled timeline.  Unlike :class:`~dcvpp.vpp_world_moe.r_ecmoe.RECMoE`, neither
baseline continues W1-selected experts, consumes the W1 route prior, or uses
the W1 tail decision state.  They share the primary action, constraint,
action-response, value-head, realized-replay, loss, and training contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import VPPWorldMoE
from .packets import WorldInputPacket
from .r_ecmoe import (
    ConstraintAssemblyResult,
    DecisionExpertAdapterBank,
    DistributionalRiskValueHead,
    NeuralConstraintAssembler,
    RActionResponseAdapter,
    RECMoE,
    RECMoEConfig,
    RECMoEOutput,
    RiskConditionPacket,
    RiskStageTokenizer,
    SparseCrossExpertCoordinator,
    StageName,
    StructuredActionHeads,
    W1ExpertPacket,
    differentiable_operating_cost,
)


BaselineType = Literal["dense_direct", "independent_r_moe"]


def _adapter_parameter_count(config: RECMoEConfig) -> int:
    width = int(config.d_model)
    hidden = int(config.adapter_bottleneck)
    return 5 * width * hidden + 4 * hidden + 3 * width + 1


def _continuation_router_parameter_count(config: RECMoEConfig) -> int:
    width = int(config.d_model)
    hidden = int(config.risk_hidden_dim)
    return 6 * width + 3 * width * hidden + 3 * hidden + 2


def _coordinator_parameter_count(config: RECMoEConfig) -> int:
    width = int(config.d_model)
    hidden = int(config.risk_hidden_dim)
    return 4 * width * hidden + 2 * hidden + 6 * width + 2


def reference_decision_parameter_budget(config: RECMoEConfig) -> int:
    """Trainable primary-R parameters replaced by either baseline block."""

    return (
        _continuation_router_parameter_count(config)
        + (int(config.general_experts) + 1) * _adapter_parameter_count(config)
        + _coordinator_parameter_count(config)
    )


@dataclass(frozen=True)
class BaselineParameterBudget:
    reference_decision_parameters: int
    baseline_decision_parameters: int

    @property
    def relative_error(self) -> float:
        return abs(
            self.baseline_decision_parameters - self.reference_decision_parameters
        ) / float(self.reference_decision_parameters)

    def validate(self, *, tolerance: float = 0.005) -> "BaselineParameterBudget":
        if self.reference_decision_parameters <= 0 or self.baseline_decision_parameters <= 0:
            raise ValueError("baseline parameter budgets must be positive")
        if self.relative_error > float(tolerance):
            raise ValueError(
                "baseline decision block is not parameter-budget matched: "
                f"relative error={self.relative_error:.6f}"
            )
        return self


class DenseResidualDecisionBlock(nn.Module):
    """One dense gated residual block matched to the sparse R decision budget."""

    def __init__(self, config: RECMoEConfig, *, target_parameters: int) -> None:
        super().__init__()
        width = int(config.d_model)
        denominator = 5 * width + 4
        hidden = max(4, round((int(target_parameters) - 3 * width - 1) / denominator))
        self.hidden_dim = int(hidden)
        self.norm = nn.LayerNorm(width)
        self.state_projection = nn.Linear(width, 2 * hidden)
        self.risk_projection = nn.Linear(width, 2 * hidden)
        self.output_projection = nn.Linear(hidden, width)
        self.output_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, state: Tensor, risk: Tensor) -> Tensor:
        gate, content = self.state_projection(self.norm(state)).chunk(2, dim=-1)
        scale, shift = self.risk_projection(risk).chunk(2, dim=-1)
        hidden = F.silu(gate + scale) * (content + shift)
        return state + torch.sigmoid(self.output_gate) * self.output_projection(hidden)


class IndependentSparseRouter(nn.Module):
    """Top-2 seven-expert router whose logits are independent of all W1 routes."""

    def __init__(self, config: RECMoEConfig, *, target_parameters: int) -> None:
        super().__init__()
        width = int(config.d_model)
        experts = int(config.general_experts)
        denominator = 2 * width + experts + 1
        hidden = max(
            4,
            round(
                (int(target_parameters) - 4 * width - experts) / denominator
            ),
        )
        self.hidden_dim = int(hidden)
        self.norm = nn.LayerNorm(2 * width)
        self.projection = nn.Sequential(
            nn.Linear(2 * width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, experts),
        )
        self.last_selected_ids: Tensor | None = None

    def forward(self, causal_state: Tensor, risk_context: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.projection(
            self.norm(torch.cat((causal_state, risk_context), dim=-1))
        ).float()
        selected_logits, selected_ids = torch.topk(logits, k=2, dim=-1)
        weights = torch.softmax(selected_logits, dim=-1).to(causal_state.dtype)
        self.last_selected_ids = selected_ids.detach()
        return selected_ids, weights


class _CausalPooledRBase(nn.Module):
    """Shared solver-free baseline path downstream of one released W1 encode."""

    baseline_type: BaselineType
    architecture_version = "matched_causal_pooled_r_baseline_v1"
    external_solver_free = True
    candidate_search_free = True

    def __init__(self, world_model: VPPWorldMoE, config: RECMoEConfig) -> None:
        super().__init__()
        if world_model.config.d_model != config.d_model:
            raise ValueError("baseline width must match the released W1")
        if world_model.config.general_experts != config.general_experts:
            raise ValueError("baseline expert count must match the released W1")
        if config.active_top_k != 2:
            raise ValueError("matched baselines use the balanced Top-2 profile")
        self.world_model = world_model
        self.config = config
        self.risk_tokenizer = RiskStageTokenizer(
            config, maximum_horizon=world_model.config.horizon
        )
        self.action_heads = StructuredActionHeads(
            config, horizon=world_model.config.horizon
        )
        self.assembler = NeuralConstraintAssembler(config)
        self.response_adapter = RActionResponseAdapter(config)
        self.risk_value_head = DistributionalRiskValueHead(config)
        self.causal_context_projection = nn.Linear(2, config.d_model)
        self.stage: StageName = "ra"
        self.w1_backbone_calls = 0
        self._packet_contract_verified = False
        self._realized_replay: Any | None = None

    def _finish_initialization(self) -> None:
        reference = reference_decision_parameter_budget(self.config)
        actual = sum(
            parameter.numel() for parameter in self._decision_modules().parameters()
        )
        self.parameter_budget = BaselineParameterBudget(reference, actual).validate()
        self.configure_stage("ra")

    def _decision_modules(self) -> nn.Module:
        raise NotImplementedError

    def train(self, mode: bool = True) -> "_CausalPooledRBase":
        super().train(mode)
        self.world_model.eval()
        return self

    def configure_stage(self, stage: StageName) -> None:
        if stage not in {"ra", "rb", "rc"}:
            raise ValueError("baseline stage must be ra, rb, or rc")
        self.stage = stage
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)
        self.world_model.eval()
        for name, parameter in self.named_parameters():
            if not name.startswith("world_model."):
                parameter.requires_grad_(True)

    @property
    def world_backbone_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.world_model.parameters())

    @staticmethod
    def _strip_w1_decision_signals(packet: W1ExpertPacket, top_k: int) -> W1ExpertPacket:
        """Replace W1 route/tail signals with fixed compatibility placeholders."""

        batch, timeline, width = packet.state_latent.shape
        ids = torch.arange(top_k, device=packet.state_latent.device, dtype=torch.long)
        ids = ids[None, None].expand(batch, timeline, -1)
        weights = packet.state_latent.new_full(
            (batch, timeline, top_k), 1.0 / float(top_k)
        )
        return replace(
            packet,
            selected_expert_ids=ids,
            selected_route_logits=torch.zeros_like(weights, dtype=torch.float32),
            selected_route_weights=weights,
            selected_expert_states=packet.state_latent.new_zeros(
                batch, timeline, top_k, width
            ),
            pre_risk_state=packet.state_latent,
            tail_probability=packet.state_latent.new_zeros(batch, timeline, 1),
            tail_active=torch.zeros(
                batch,
                timeline,
                device=packet.state_latent.device,
                dtype=torch.bool,
            ),
            tail_state=torch.zeros_like(packet.state_latent),
            action_response_basis=packet.state_latent,
            uncertainty_summary=torch.zeros_like(packet.uncertainty_summary),
        )

    def encode_expert_context(
        self,
        inputs: WorldInputPacket,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> W1ExpertPacket:
        # Reuse the released packet extractor without constructing any of the
        # primary R continuation modules.  The returned W1 decision signals are
        # immediately replaced before a baseline decision can consume them.
        packet = RECMoE.encode_expert_context(
            self,  # type: ignore[arg-type]
            inputs,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )
        return self._strip_w1_decision_signals(
            packet, int(self.world_model.config.regular_top_k)
        )

    def _risk_context(
        self, packet: W1ExpertPacket, risk: RiskConditionPacket
    ) -> Tensor:
        batch, timeline = packet.state_latent.shape[:2]
        risk.validate(batch, timeline)
        return self.risk_tokenizer(risk, time=timeline) + self.causal_context_projection(
            packet.causal_context.to(packet.state_latent.dtype)
        )

    def decide(
        self, packet: W1ExpertPacket, risk: RiskConditionPacket
    ) -> tuple[ConstraintAssemblyResult, Tensor, Tensor, Tensor, Tensor, Tensor]:
        risk_context = self._risk_context(packet, risk)
        decision_state, ids, weights = self._baseline_decision(packet, risk_context)
        assembled = self.assembler(self.action_heads(decision_state), packet, risk)
        tail_active = torch.zeros_like(assembled.decision_active_mask)
        return assembled, decision_state, weights, ids, tail_active, risk_context

    def _baseline_decision(
        self, packet: W1ExpertPacket, risk_context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def apply_action_response(
        self, packet: W1ExpertPacket, action_summary: Tensor
    ) -> tuple[Tensor, Tensor]:
        method = getattr(self.world_model.dynamics, "parallel_action_response", None)
        if method is None:
            raise RuntimeError("released W1 parallel action response is unavailable")
        response, scale = method(
            packet.state_latent.detach(), action_summary, packet.constraints
        )
        return self.response_adapter(response, packet.state_latent.detach()), scale

    def realized_historical_replay(
        self,
        output: RECMoEOutput,
        *,
        batch: dict[str, Any],
        risk: RiskConditionPacket,
        source_split: str,
    ) -> RECMoEOutput:
        if self._realized_replay is None:
            from .r_ecmoe_replay import RealizedHistoricalReplay

            self._realized_replay = RealizedHistoricalReplay.from_model(self)  # type: ignore[arg-type]
        return self._realized_replay(
            output,
            batch=batch,
            risk=risk,
            source_split=source_split,
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
        batch, timeline = packet.state_latent.shape[:2]
        if risk is None:
            risk = RiskConditionPacket.defaults(
                batch,
                device=packet.state_latent.device,
                horizon=timeline,
                stage=1,
            )
        assembled, state, weights, ids, tail_active, risk_context = self.decide(
            packet, risk
        )
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
            state,
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
        diagnostics: dict[str, Tensor | float | int | str] = {
            "baseline_type": self.baseline_type,
            "routing_source": self._routing_source,
            "w1_backbone_calls": self.w1_backbone_calls - calls_before,
            "active_top_k": int(weights.shape[-1]),
            "w1_selected_experts_consumed": 0,
            "w1_route_prior_consumed": 0,
            "w1_tail_state_consumed": 0,
            "parameter_budget_relative_error": self.parameter_budget.relative_error,
        }
        if not return_diagnostics:
            # Baseline lineage is retained even when verbose diagnostics are off.
            diagnostics = {
                "baseline_type": self.baseline_type,
                "routing_source": self._routing_source,
            }
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


class DenseDirectR(_CausalPooledRBase):
    """Parameter-matched dense residual policy over the causal W1 timeline."""

    baseline_type: BaselineType = "dense_direct"
    _routing_source = "none_dense_compatibility_route_only"

    def __init__(self, world_model: VPPWorldMoE, config: RECMoEConfig) -> None:
        super().__init__(world_model, config)
        self.dense_decision_block = DenseResidualDecisionBlock(
            config, target_parameters=reference_decision_parameter_budget(config)
        )
        self._finish_initialization()

    def _decision_modules(self) -> nn.Module:
        return self.dense_decision_block

    def _baseline_decision(
        self, packet: W1ExpertPacket, risk_context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = self.dense_decision_block(packet.state_latent, risk_context)
        batch, timeline = state.shape[:2]
        ids = torch.zeros(batch, timeline, 1, device=state.device, dtype=torch.long)
        weights = state.new_ones(batch, timeline, 1)
        return state, ids, weights


class IndependentRMoE(_CausalPooledRBase):
    """Parameter-matched independent seven-expert Top-2 decision MoE."""

    baseline_type: BaselineType = "independent_r_moe"
    _routing_source = "independent_r_router"

    def __init__(self, world_model: VPPWorldMoE, config: RECMoEConfig) -> None:
        super().__init__(world_model, config)
        route_and_tail_budget = (
            _continuation_router_parameter_count(config)
            + _adapter_parameter_count(config)
        )
        self.independent_router = IndependentSparseRouter(
            config, target_parameters=route_and_tail_budget
        )
        self.independent_adapter_bank = DecisionExpertAdapterBank(config)
        self.independent_coordinator = SparseCrossExpertCoordinator(config)
        self._finish_initialization()

    def _decision_modules(self) -> nn.Module:
        return nn.ModuleList(
            (
                self.independent_router,
                self.independent_adapter_bank,
                self.independent_coordinator,
            )
        )

    def _baseline_decision(
        self, packet: W1ExpertPacket, risk_context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        ids, weights = self.independent_router(packet.state_latent, risk_context)
        states = packet.state_latent[:, :, None].expand(-1, -1, 2, -1)
        adapted = self.independent_adapter_bank(states, ids, risk_context)
        inactive_tail = torch.zeros(
            packet.state_latent.shape[:2],
            device=packet.state_latent.device,
            dtype=torch.bool,
        )
        state = self.independent_coordinator(
            adapted,
            weights,
            torch.zeros_like(packet.state_latent),
            inactive_tail,
            risk_context,
            packet.state_latent,
        )
        return state, ids, weights


__all__ = [
    "BaselineParameterBudget",
    "BaselineType",
    "DenseDirectR",
    "DenseResidualDecisionBlock",
    "IndependentRMoE",
    "IndependentSparseRouter",
    "reference_decision_parameter_budget",
]
