"""Multi-scale causal graph/LNN/SSM WorldMoE backbone."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import WorldMoEConfig
from .packets import (
    ENTITIES_PER_NODE,
    GLOBAL_SLOTS,
    NODES,
    RouterDiagnostics,
    TOTAL_SLOTS,
    WorldInputPacket,
    WorldStatePacket,
)
from .sparse_moe import TAIL_RISK_SLOT, SharedSparseTop2MoE


class LocalCrossModalAttention(nn.Module):
    """Attend within each site's eight entities plus shared global context."""

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(-1.0))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: Tensor) -> Tensor:
        batch, time, _, width = value.shape
        local = value[:, :, : NODES * ENTITIES_PER_NODE].reshape(
            batch, time, NODES, ENTITIES_PER_NODE, width
        )
        global_slots = value[:, :, -GLOBAL_SLOTS:]
        expanded_global = global_slots[:, :, None].expand(-1, -1, NODES, -1, -1)
        grouped = torch.cat((local, expanded_global), dim=3)
        flat = grouped.reshape(batch * time * NODES, 16, width)
        normalized = self.norm(flat)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        update = update.reshape(batch, time, NODES, 16, width)
        local_update = update[:, :, :, :ENTITIES_PER_NODE].reshape(
            batch, time, NODES * ENTITIES_PER_NODE, width
        )
        global_update = update[:, :, :, ENTITIES_PER_NODE:].mean(dim=2)
        joined = torch.cat((local_update, global_update), dim=2)
        return value + self.dropout(torch.sigmoid(self.gate) * joined)


class TemporalSelectiveStateSpace(nn.Module):
    """Input-selective causal recurrence with continuous-time decay."""

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        width = config.d_model
        self.norm = nn.LayerNorm(width)
        self.proposal = nn.Linear(width, width)
        self.rate = nn.Linear(width, width)
        self.mix = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: Tensor, dt_hours: Tensor) -> Tensor:
        normalized = self.norm(value)
        # Rate, proposal, and mixing gates depend only on the current input,
        # not on the recurrent state.  Evaluate each projection once over the
        # complete timeline so CUDA receives three useful GEMMs instead of
        # 3*T launch-bound micro-GEMMs.  Only the genuinely state-dependent
        # liquid recurrence remains sequential below.
        rate = F.softplus(self.rate(normalized)).clamp_max(20.0)
        dt = dt_hours[:, :, None, None].to(normalized.dtype)
        alpha = -torch.expm1(-(rate * dt).clamp_max(20.0))
        candidate = torch.tanh(self.proposal(normalized))
        mix = torch.sigmoid(self.mix(normalized))
        state = torch.zeros_like(normalized[:, 0])
        outputs: list[Tensor] = []
        for index in range(value.shape[1]):
            current = normalized[:, index]
            current_alpha = alpha[:, index]
            state = (1.0 - current_alpha) * state + current_alpha * candidate[:, index]
            current_mix = mix[:, index]
            outputs.append(current_mix * state + (1.0 - current_mix) * current)
        sequence = torch.stack(outputs, dim=1)
        return value + self.dropout(self.output(sequence))


class DirectedHeterogeneousGraphAttention(nn.Module):
    """Dense 40-slot implementation of directed typed graph message passing."""

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.edge_type_bias = nn.Embedding(32, config.num_heads)
        self.edge_feature_bias = nn.Linear(
            config.graph_edge_dim, config.num_heads, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def _bias_and_mask(
        self, packet: WorldInputPacket, *, batch: int, time: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        device = packet.graph_edge_index.device
        source, target = packet.graph_edge_index
        # Input convention is [source,target]; score rows are destinations.
        static_bias = self.edge_type_bias(packet.graph_edge_type.clamp(0, 31))
        static_bias = static_bias + self.edge_feature_bias(packet.graph_edge_features)
        bias = torch.zeros(
            batch,
            time,
            self.config.num_heads,
            TOTAL_SLOTS,
            TOTAL_SLOTS,
            device=device,
            dtype=dtype,
        )
        allowed = torch.eye(TOTAL_SLOTS, device=device, dtype=torch.bool)[None, None]
        allowed = allowed.expand(batch, time, -1, -1).clone()
        static_allowed = packet.graph_edge_mask
        if packet.dynamic_graph_edge_features is None:
            edge_bias = static_bias.to(dtype)[None, None].expand(batch, time, -1, -1)
            edge_allowed = static_allowed[None, None].expand(batch, time, -1)
        else:
            dynamic = self.edge_feature_bias(packet.dynamic_graph_edge_features)
            edge_bias = static_bias[None, None] + dynamic
            assert packet.dynamic_graph_edge_mask is not None
            edge_allowed = static_allowed[None, None] & packet.dynamic_graph_edge_mask
        # Forty slots keep this assignment far smaller than a general sparse
        # graph kernel while retaining directed, typed, time-varying edges.
        bias[:, :, :, target, source] = edge_bias.permute(0, 1, 3, 2).to(dtype)
        allowed[:, :, target, source] = edge_allowed
        return bias, allowed

    def forward(self, value: Tensor, packet: WorldInputPacket) -> Tensor:
        batch, time, slots, width = value.shape
        normalized = self.norm(value)
        head_width = width // self.config.num_heads

        def heads(layer: nn.Linear) -> Tensor:
            projected = layer(normalized).reshape(
                batch, time, slots, self.config.num_heads, head_width
            )
            return projected.permute(0, 1, 3, 2, 4)

        query, key, content = heads(self.query), heads(self.key), heads(self.value)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_width)
        bias, allowed = self._bias_and_mask(
            packet, batch=batch, time=time, dtype=score.dtype
        )
        score = (score + bias).masked_fill(
            ~allowed[:, :, None], torch.finfo(score.dtype).min
        )
        probability = torch.softmax(score.float(), dim=-1).to(score.dtype)
        update = torch.matmul(probability, content)
        update = update.permute(0, 1, 3, 2, 4).reshape(batch, time, slots, width)
        return value + self.dropout(torch.sigmoid(self.gate) * self.output(update))


class ActionConditionedLiquidTransition(nn.Module):
    """Continuous-time action-conditioned latent drift inside every block."""

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        width = config.d_model
        self.state_norm = nn.LayerNorm(width)
        self.action_norm = nn.LayerNorm(width)
        self.state_drive = nn.Linear(width, width)
        self.action_drive = nn.Linear(width, width, bias=False)
        self.inverse_tau = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self, value: Tensor, action_context: Tensor, dt_hours: Tensor
    ) -> Tensor:
        state = self.state_norm(value)
        action = self.action_norm(action_context)
        drive = torch.tanh(self.state_drive(state) + self.action_drive(action))
        inverse_tau = F.softplus(self.inverse_tau(state)).clamp_max(20.0)
        dt = dt_hours[:, :, None, None].to(value.dtype)
        alpha = -torch.expm1(-(inverse_tau * dt).clamp_max(20.0))
        liquid = state + alpha * (drive - state)
        return value + torch.sigmoid(self.gate) * self.output(liquid)


class StochasticRegimeResidualUpdate(nn.Module):
    """Couple a categorical operating regime with reparameterized innovation."""

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.norm = nn.LayerNorm(width)
        self.regime_head = nn.Linear(width, config.num_regimes)
        self.regime_embedding = nn.Parameter(
            torch.randn(config.num_regimes, width) / math.sqrt(width)
        )
        self.mean_head = nn.Linear(width, config.stochastic_dim)
        self.log_scale_head = nn.Linear(width, config.stochastic_dim)
        self.residual_projection = nn.Linear(config.stochastic_dim, width)
        self.regime_gate = nn.Parameter(torch.tensor(-2.0))
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, value: Tensor, *, sample: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        normalized = self.norm(value)
        logits = self.regime_head(normalized)
        probabilities = torch.softmax(logits.float(), dim=-1).to(value.dtype)
        if sample:
            uniform = torch.rand_like(probabilities.float()).clamp_(
                1.0e-6, 1.0 - 1.0e-6
            )
            gumbel = -torch.log(-torch.log(uniform))
            regime_weight = torch.softmax((logits.float() + gumbel) / 0.75, dim=-1)
            regime_weight = regime_weight.to(value.dtype)
        else:
            regime_weight = probabilities
        regime = regime_weight @ self.regime_embedding.to(value.dtype)
        mean = self.mean_head(normalized)
        log_scale = self.log_scale_head(normalized).clamp(-7.0, 2.0)
        innovation = mean
        if sample:
            innovation = mean + torch.exp(log_scale) * torch.randn_like(mean)
        update = torch.sigmoid(self.regime_gate) * regime + torch.sigmoid(
            self.residual_gate
        ) * self.residual_projection(innovation)
        return value + update, logits, probabilities, mean, log_scale


class PhysicsConstrainedResidual(nn.Module):
    """Inject a bounded residual projected onto node-wise energy balance."""

    _ENERGY_ENTITIES = (1, 2, 3, 6, 7)

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.d_model)
        self.flow_head = nn.Linear(config.d_model, 1)
        self.slack_gate = nn.Linear(config.d_model, 1)
        self.entity_basis = nn.Parameter(
            torch.randn(ENTITIES_PER_NODE, config.d_model) / math.sqrt(config.d_model)
        )
        self.global_projection = nn.Linear(config.d_model, config.d_model)

    def forward(self, value: Tensor, slack_context: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, _, width = value.shape
        with torch.autocast(device_type=value.device.type, enabled=False):
            local = (
                value[:, :, : NODES * ENTITIES_PER_NODE]
                .float()
                .reshape(batch, time, NODES, ENTITIES_PER_NODE, width)
            )
            slack = (
                slack_context[:, :, : NODES * ENTITIES_PER_NODE]
                .float()
                .reshape_as(local)
            )
            raw = torch.tanh(self.flow_head(self.norm(local))).squeeze(-1)
            cap = self.config.physics_residual_cap * torch.sigmoid(
                self.slack_gate(slack)
            ).squeeze(-1)
            flow = raw * cap
            selected = flow[..., list(self._ENERGY_ENTITIES)]
            first = selected[..., :4]
            subtotal = first.sum(dim=-1, keepdim=True)
            closed = torch.cat((first, -subtotal), dim=-1).to(flow.dtype)
            scalar = torch.zeros_like(flow)
            scalar[..., list(self._ENERGY_ENTITIES)] = closed
            local_update = scalar[..., None] * self.entity_basis.float()
            local_update = local_update.reshape(
                batch, time, NODES * ENTITIES_PER_NODE, width
            )
            global_update = self.global_projection(
                local_update.reshape(batch, time, NODES, ENTITIES_PER_NODE, width).mean(
                    dim=(2, 3)
                )
            )[:, :, None].expand(-1, -1, GLOBAL_SLOTS, -1)
            update = torch.cat((local_update, global_update), dim=2)
            # This expression reuses the exact subtotal used to construct the
            # closing component, avoiding an independent floating reduction.
            balance = subtotal.squeeze(-1) + closed[..., -1]
        return value + update.to(value.dtype), balance


class _FeedForward(nn.Module):
    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.input = nn.Linear(config.d_model, 2 * config.ffn_hidden_dim)
        self.output = nn.Linear(config.ffn_hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: Tensor) -> Tensor:
        gate, content = self.input(self.norm(value)).chunk(2, dim=-1)
        return value + self.dropout(self.output(F.silu(gate) * content))


class WorldInteractionBlock(nn.Module):
    """One complete causal world interaction layer in registered order."""

    def __init__(self, config: WorldMoEConfig, block_index: int) -> None:
        super().__init__()
        self.block_index = int(block_index)
        self.local_attention = LocalCrossModalAttention(config)
        self.temporal_ssm = TemporalSelectiveStateSpace(config)
        self.graph_attention = DirectedHeterogeneousGraphAttention(config)
        self.liquid_transition = ActionConditionedLiquidTransition(config)
        self.stochastic_update = StochasticRegimeResidualUpdate(config)
        self.moe = (
            SharedSparseTop2MoE(config)
            if self.block_index in config.moe_block_indices
            else None
        )
        self.physics_residual = PhysicsConstrainedResidual(config)
        self.feed_forward = _FeedForward(config)

    def forward_pre_moe(
        self,
        value: Tensor,
        action_context: Tensor,
        packet: WorldInputPacket,
        *,
        sample_stochastic: bool,
        apply_dynamic_residuals: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        value = self.local_attention(value)
        value = self.temporal_ssm(value, packet.dt_hours)
        value = self.graph_attention(value, packet)
        if apply_dynamic_residuals:
            value = self.liquid_transition(value, action_context, packet.dt_hours)
            value, logits, probabilities, mean, log_scale = self.stochastic_update(
                value, sample=sample_stochastic
            )
        else:
            # W0 learns the causal observation state before the action-conditioned
            # and stochastic transition modules receive their W1 objectives.  Do
            # not inject frozen random residuals into that state, and do not
            # create a train/eval mismatch by sampling them only in training.
            # Shape-correct neutral statistics preserve the WorldStatePacket API.
            shape = value.shape[:-1]
            logits = value.new_zeros(*shape, self.stochastic_update.config.num_regimes)
            probabilities = value.new_full(
                (*shape, self.stochastic_update.config.num_regimes),
                1.0 / float(self.stochastic_update.config.num_regimes),
            )
            mean = value.new_zeros(*shape, self.stochastic_update.config.stochastic_dim)
            log_scale = value.new_zeros(
                *shape, self.stochastic_update.config.stochastic_dim
            )
        return value, logits, probabilities, mean, log_scale

    def forward_post_moe(
        self, value: Tensor, slack_context: Tensor
    ) -> tuple[Tensor, Tensor]:
        value, balance = self.physics_residual(value, slack_context)
        return self.feed_forward(value), balance

    def forward(
        self,
        value: Tensor,
        action_context: Tensor,
        slack_context: Tensor,
        packet: WorldInputPacket,
        *,
        sample_stochastic: bool,
        apply_dynamic_residuals: bool = True,
        smooth_tail_activation: bool = False,
        capture_slot: int | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        RouterDiagnostics | None,
    ]:
        value, logits, probabilities, mean, log_scale = self.forward_pre_moe(
            value,
            action_context,
            packet,
            sample_stochastic=sample_stochastic,
            apply_dynamic_residuals=apply_dynamic_residuals,
        )
        diagnostics = None
        if self.moe is not None:
            value, diagnostics = self.moe(
                value,
                block_index=self.block_index,
                smooth_tail_activation=smooth_tail_activation,
                capture_slot=capture_slot,
            )
        value, balance = self.forward_post_moe(value, slack_context)
        return value, logits, probabilities, mean, log_scale, balance, diagnostics


class MultiScaleCausalWorldBackbone(nn.Module):
    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            WorldInteractionBlock(config, index) for index in range(config.num_blocks)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.pool_score = nn.Linear(config.d_model, 1)

    def forward(
        self,
        tokens: Tensor,
        action_context: Tensor,
        slack_context: Tensor,
        packet: WorldInputPacket,
        *,
        sample_stochastic: bool,
        apply_dynamic_residuals: bool = True,
        smooth_tail_activation: bool = False,
        capture_expert_context: bool = False,
    ) -> WorldStatePacket:
        value = tokens
        diagnostics: list[RouterDiagnostics] = []
        last: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        total_balance = tokens.new_zeros(tokens.shape[0], tokens.shape[1], NODES)
        final_moe_block = self.config.moe_block_indices[-1]
        for block in self.blocks:
            capture_slot = (
                TAIL_RISK_SLOT
                if capture_expert_context and block.block_index == final_moe_block
                else None
            )
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                router = None

                def pre_moe(
                    current: Tensor,
                    action: Tensor,
                    current_block: WorldInteractionBlock = block,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                    return current_block.forward_pre_moe(
                        current,
                        action,
                        packet,
                        sample_stochastic=sample_stochastic,
                        apply_dynamic_residuals=apply_dynamic_residuals,
                    )

                value, logits, probabilities, mean, log_scale = checkpoint(
                    pre_moe,
                    value,
                    action_context,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                # Routing itself executes once.  Recomputing Top-2 assignments
                # is not stable at BF16 ties; individual expert MLPs perform
                # their own fixed-shape activation checkpointing instead.
                if block.moe is not None:
                    value, router = block.moe(
                        value,
                        block_index=block.block_index,
                        smooth_tail_activation=smooth_tail_activation,
                        capture_slot=capture_slot,
                    )

                def post_moe(
                    current: Tensor,
                    slack: Tensor,
                    current_block: WorldInteractionBlock = block,
                ) -> tuple[Tensor, Tensor]:
                    return current_block.forward_post_moe(current, slack)

                value, balance = checkpoint(
                    post_moe,
                    value,
                    slack_context,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                (
                    value,
                    logits,
                    probabilities,
                    mean,
                    log_scale,
                    balance,
                    router,
                ) = block(
                    value,
                    action_context,
                    slack_context,
                    packet,
                    sample_stochastic=sample_stochastic,
                    apply_dynamic_residuals=apply_dynamic_residuals,
                    smooth_tail_activation=smooth_tail_activation,
                    capture_slot=capture_slot,
                )
            last = (logits, probabilities, mean, log_scale)
            total_balance = total_balance + balance
            if router is not None:
                diagnostics.append(router)
        if last is None:
            raise RuntimeError("WorldMoE backbone has no interaction blocks")
        value = self.final_norm(value)
        pool_weight = torch.softmax(self.pool_score(value).float(), dim=2).to(
            value.dtype
        )
        pooled_timeline = (pool_weight * value).sum(dim=2)
        row = torch.arange(value.shape[0], device=value.device)
        current_slots = value[row, packet.origin_index]
        pooled_state = pooled_timeline[row, packet.origin_index]
        logits, probabilities, mean, log_scale = last
        return WorldStatePacket(
            slots=value,
            current_slots=current_slots,
            pooled_timeline=pooled_timeline,
            pooled_state=pooled_state,
            regime_logits=logits,
            regime_probabilities=probabilities,
            stochastic_mean=mean,
            stochastic_log_scale=log_scale,
            physics_balance_residual=total_balance,
            router_diagnostics=tuple(diagnostics),
            origin_index=packet.origin_index,
        ).validate(self.config)


__all__ = [
    "ActionConditionedLiquidTransition",
    "DirectedHeterogeneousGraphAttention",
    "LocalCrossModalAttention",
    "MultiScaleCausalWorldBackbone",
    "PhysicsConstrainedResidual",
    "StochasticRegimeResidualUpdate",
    "TemporalSelectiveStateSpace",
    "WorldInteractionBlock",
]
