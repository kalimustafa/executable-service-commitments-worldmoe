"""Always-on computation, general Top-2 routing, and a causal tail expert.

The registered eight-expert budget is split into seven interchangeable general
experts and one queue-tail expert.  The tail expert is deliberately *not* an
eighth competitor in the general router: ordinary tokens retain two general
routes, while causally high-risk time steps receive an additional bounded tail
residual.  This prevents common tokens from consuming the tail capacity and
keeps the general load-balancing objective scientifically separate.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import EXPERT_ASSIGNMENT_SCHEMA, WorldMoEConfig
from .packets import RouterDiagnostics


# Slot 37 is the registered global ``risk`` slot.  Once a time step is selected,
# the specialist is allowed to modify only workload/queue/constraint evidence;
# market, PV, battery, cooling and calendar slots cannot be polluted by it.
TAIL_RISK_SLOT = 37
TAIL_RELATED_SLOTS = (
    4,
    5,
    12,
    13,
    20,
    21,
    28,
    29,
    35,  # aggregate_workload
    36,  # system_constraint
    37,  # risk
)


class _SwiGLUExpert(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, value: Tensor) -> Tensor:
        gate, content = self.input(self.norm(value)).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)


class SharedSparseTop2MoE(nn.Module):
    """Run shared + general Top-2, then a capacity-isolated tail residual.

    Tail activation never accepts a realized event label.  It is a deterministic
    function of the causal global risk token, and the same rule is used in
    training and inference.  A small per-sample top-capacity floor prevents the
    rare expert from starving before its causal gate is calibrated; probabilities
    above the threshold may expand use only up to the independent tail capacity.
    """

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.general_expert_count = int(config.general_experts)
        self.general_top_k = int(config.regular_top_k)
        self.tail_exploration_fraction = float(config.tail_capacity_fraction)
        self.tail_capacity_max_fraction = float(config.tail_capacity_max_fraction)
        self.tail_min_tokens = int(config.tail_min_tokens)
        self.tail_route_threshold = float(config.tail_activation_probability_threshold)
        # The router is optimized with a frozen positive-class weight.  Its raw
        # logit is therefore a ranking score, not a calibrated event log-odds.
        # Correct the intercept before the probability is used by routing.
        self.tail_positive_weight = float(
            getattr(config, "tail_route_positive_weight", 49.08092411354107)
        )
        if (
            not math.isfinite(self.tail_positive_weight)
            or self.tail_positive_weight < 1.0
        ):
            raise ValueError("tail route positive weight must be finite and >= 1")
        self.tail_logit_correction = math.log(self.tail_positive_weight)
        self.tail_smooth_temperature = float(
            getattr(config, "tail_smooth_activation_temperature", 1.0)
        )
        if (
            not math.isfinite(self.tail_smooth_temperature)
            or self.tail_smooth_temperature <= 0.0
        ):
            raise ValueError("tail smooth activation temperature must be positive")
        self.tail_residual_min_scale = float(
            getattr(config, "tail_residual_min_scale", 0.08)
        )
        self.tail_residual_max_scale = float(
            getattr(config, "tail_residual_max_scale", 0.12)
        )
        if not (
            0.0 < self.tail_residual_min_scale < self.tail_residual_max_scale <= 1.0
        ):
            raise ValueError("tail residual scales must satisfy 0 < min < max <= 1")
        if self.general_expert_count < self.general_top_k:
            raise ValueError("general expert count must cover Top-k routing")
        self.router = nn.Linear(config.d_model, self.general_expert_count, bias=False)
        self.tail_router = nn.Linear(config.d_model, 1, bias=True)
        # Under weighted BCE, a prevalence-only optimum has raw logit zero:
        # logit(p) + log((1-p)/p) == 0.  The calibrated probability below then
        # recovers the frozen train prevalence instead of starting at 5%.
        nn.init.zeros_(self.tail_router.bias)
        self.shared_expert = _SwiGLUExpert(config.d_model, config.expert_hidden_dim)
        self.experts = nn.ModuleList(
            _SwiGLUExpert(config.d_model, config.expert_hidden_dim)
            for _ in range(self.general_expert_count)
        )
        self.tail_expert = _SwiGLUExpert(config.d_model, config.expert_hidden_dim)
        self.output_gate = nn.Parameter(torch.tensor(0.0))
        # This parameter maps to the bounded [0.08,0.12] specialist coefficient.
        # Early pollution is controlled by calibrated smooth routing (about 2%
        # at initialization), rather than by an effectively dead expert gate.
        self.tail_output_gate = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(config.dropout)

    @property
    def tail_residual_scale(self) -> Tensor:
        interval = self.tail_residual_max_scale - self.tail_residual_min_scale
        return self.tail_output_gate.sigmoid() * interval + self.tail_residual_min_scale

    def calibrated_tail_probability(self, weighted_logit: Tensor) -> Tensor:
        """Convert the weighted-BCE ranking logit to an event probability."""

        return torch.sigmoid(weighted_logit.float() - self.tail_logit_correction)

    def _tail_active_mask(self, probability: Tensor) -> tuple[Tensor, Tensor]:
        """Select causal high-risk time steps under an independent capacity.

        Selection is per sample, not across the batch, so changing batch peers
        cannot change a sample's route.  The small Top-5% floor is unioned with
        every threshold crossing.  The registered 100% ceiling is only a safety
        bound: it deliberately preserves long extreme episodes instead of
        truncating them at an arbitrary tail duration.
        """

        if probability.ndim != 2:
            raise ValueError("tail probability must have shape [batch,time]")
        time = probability.shape[1]
        minimum = max(
            1,
            math.ceil(time * self.tail_exploration_fraction),
            math.ceil(self.tail_min_tokens / len(TAIL_RELATED_SLOTS)),
        )
        minimum = min(time, minimum)
        capacity = max(
            minimum,
            math.ceil(time * self.tail_capacity_max_fraction),
        )
        capacity = min(time, capacity)
        order = torch.argsort(probability, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(time, device=probability.device)[None].expand_as(order),
        )
        within_capacity = rank < capacity
        active = (rank < minimum) | (
            probability.ge(self.tail_route_threshold) & within_capacity
        )
        capacities = torch.full(
            (probability.shape[0],),
            capacity,
            dtype=torch.long,
            device=probability.device,
        )
        return active, capacities

    def forward(
        self,
        tokens: Tensor,
        *,
        block_index: int,
        smooth_tail_activation: bool = False,
        capture_slot: int | None = None,
    ) -> tuple[Tensor, RouterDiagnostics]:
        batch, time, slots, width = tokens.shape
        if capture_slot is not None and (
            isinstance(capture_slot, bool) or not 0 <= int(capture_slot) < slots
        ):
            raise ValueError("capture_slot lies outside the token slot range")
        resolved_capture_slot = None if capture_slot is None else int(capture_slot)
        flat = tokens.reshape(-1, width)
        logits = self.router(flat.float()).to(flat.dtype)
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_weights, top_indices = torch.topk(probabilities, self.general_top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        checkpointing = self.config.gradient_checkpointing and torch.is_grad_enabled()
        shared = (
            checkpoint(
                self.shared_expert,
                flat,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if checkpointing
            else self.shared_expert(flat)
        )
        delta = shared.to(flat.dtype)
        sparse = torch.zeros_like(delta)
        captured_selected_flat = (
            None
            if resolved_capture_slot is None
            else flat.new_zeros(batch * time * self.general_top_k, width)
        )
        for expert_index, expert in enumerate(self.experts):
            assignment = torch.nonzero(top_indices == expert_index, as_tuple=False)
            if assignment.numel() == 0:
                continue
            token_index = assignment[:, 0]
            route_index = assignment[:, 1]
            selected = flat.index_select(0, token_index)
            weight = top_weights[token_index, route_index, None]
            expert_delta = (
                checkpoint(
                    expert,
                    selected,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if checkpointing
                else expert(selected)
            )
            if captured_selected_flat is not None:
                capture_assignment = token_index.remainder(slots).eq(
                    resolved_capture_slot
                )
                capture_index = (
                    token_index[capture_assignment].div(slots, rounding_mode="floor")
                    * self.general_top_k
                    + route_index[capture_assignment]
                )
                captured_selected_flat = captured_selected_flat.index_add(
                    0,
                    capture_index,
                    expert_delta[capture_assignment].to(captured_selected_flat.dtype),
                )
            contribution = (expert_delta * weight).to(sparse.dtype)
            sparse = sparse.index_add(0, token_index, contribution)

        # One aggregate causal score per time step.  Realized queue labels are
        # intentionally absent from this path; they may supervise the returned
        # logits in the loss but can never alter which latent receives a residual.
        tail_logits = self.tail_router(tokens[:, :, TAIL_RISK_SLOT].float()).squeeze(-1)
        tail_probabilities = self.calibrated_tail_probability(tail_logits)
        tail_active_mask, tail_capacity = self._tail_active_mask(tail_probabilities)
        related_slot_mask = torch.zeros(slots, dtype=torch.bool, device=tokens.device)
        related_slot_mask[list(TAIL_RELATED_SLOTS)] = True
        if smooth_tail_activation:
            # During W0, keep the tail path differentiable around the frozen
            # calibrated threshold.  Computing eleven related slots is modest,
            # removes hard 5%-84% route jumps, and leaves unrelated evidence
            # structurally untouched.
            selected_mask = related_slot_mask[None, None].expand(batch, time, -1)
            threshold = min(max(self.tail_route_threshold, 1.0e-6), 1.0 - 1.0e-6)
            threshold_logit = math.log(threshold / (1.0 - threshold))
            calibrated_logit = tail_logits.float() - self.tail_logit_correction
            tail_time_weight = torch.sigmoid(
                (calibrated_logit - threshold_logit) / self.tail_smooth_temperature
            )
        else:
            selected_mask = tail_active_mask[:, :, None] & related_slot_mask[None, None]
            tail_time_weight = tail_probabilities
        tail_token_index = torch.nonzero(
            selected_mask.reshape(-1), as_tuple=False
        ).flatten()
        tail_sparse = torch.zeros_like(delta)
        if tail_token_index.numel() > 0:
            tail_selected = flat.index_select(0, tail_token_index)
            tail_delta = (
                checkpoint(
                    self.tail_expert,
                    tail_selected,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if checkpointing
                else self.tail_expert(tail_selected)
            )
            time_probability = (
                tail_time_weight[:, :, None]
                .expand(-1, -1, slots)
                .reshape(-1)
                .index_select(0, tail_token_index)
            )
            tail_contribution = (
                tail_delta * time_probability[:, None].to(tail_delta.dtype)
            ).to(tail_sparse.dtype)
            tail_sparse = tail_sparse.index_add(0, tail_token_index, tail_contribution)

        general_delta = (delta + sparse).to(flat.dtype)
        gated = torch.sigmoid(self.output_gate) * general_delta
        tail_gated = self.tail_residual_scale * tail_sparse
        result = tokens + self.dropout(
            (gated + tail_gated).to(tokens.dtype)
        ).reshape_as(tokens)

        load = torch.zeros(
            self.general_expert_count,
            device=tokens.device,
            dtype=probabilities.dtype,
        )
        load.scatter_add_(
            0,
            top_indices.reshape(-1),
            top_weights.reshape(-1),
        )
        load = load / float(top_indices.shape[0])
        importance = probabilities.mean(dim=0)
        # Tail utilization is not part of general balancing: its capacity and
        # supervision are independent by construction.
        balance_loss = self.general_expert_count * torch.sum(importance * load)
        z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
        tail_selected_count = tail_active_mask.sum(dim=1)
        tail_load = tail_active_mask.float().mean()
        tail_utilization = (
            tail_selected_count.float() / tail_capacity.clamp_min(1).float()
        ).mean()
        diagnostics = RouterDiagnostics(
            block_index=int(block_index),
            logits=logits.reshape(batch, time, slots, -1),
            probabilities=probabilities.reshape(batch, time, slots, -1),
            topk_indices=top_indices.reshape(batch, time, slots, -1),
            topk_weights=top_weights.reshape(batch, time, slots, -1),
            expert_load=load,
            balance_loss=balance_loss,
            z_loss=z_loss,
            tail_route_logits=tail_logits,
            tail_route_probabilities=tail_probabilities,
            tail_active_mask=tail_active_mask,
            tail_selected_count=tail_selected_count,
            tail_capacity=tail_capacity,
            tail_load=tail_load,
            tail_utilization=tail_utilization,
            capture_slot=resolved_capture_slot,
            captured_pre_state=(
                None
                if resolved_capture_slot is None
                else tokens[:, :, resolved_capture_slot]
            ),
            captured_selected_expert_states=(
                None
                if captured_selected_flat is None
                else captured_selected_flat.reshape(
                    batch, time, self.general_top_k, width
                )
            ),
            captured_tail_state=(
                None
                if resolved_capture_slot is None
                else tail_sparse.reshape_as(tokens)[:, :, resolved_capture_slot]
            ),
        )
        return result, diagnostics


__all__ = [
    "EXPERT_ASSIGNMENT_SCHEMA",
    "SharedSparseTop2MoE",
    "TAIL_RELATED_SLOTS",
    "TAIL_RISK_SLOT",
]
