"""Parallel Representation, Distribution and Policy planes for R4-V8."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .packets import (
    CanonicalWorkspacePacket,
    DEPENDENCE_RANK,
    DecisionWorkspacePacket,
    DistributionSamplePacket,
    DistributionTruthPacket,
    HORIZON,
    INDEPENDENT_CHANNELS,
    PUBLIC_CHANNELS,
    PolicyPacket,
    REGIME_COUNT,
    RepresentationAnchorPacket,
    RepresentationPacket,
    WORKSPACE_DIM,
    ZONES,
    close_public_channels,
    independent_from_public,
)
from .sparse_moe import SparseTop2MoE


def _zero_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class TriFusionWorkspace(nn.Module):
    """Fuse four projected modality streams in the common 192-wide workspace."""

    def __init__(
        self, workspace_dim: int = WORKSPACE_DIM, *, heads: int = 6, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if workspace_dim != WORKSPACE_DIM:
            raise ValueError("V8 fixes workspace_dim=192")
        self.modality_norm = nn.LayerNorm(workspace_dim)
        self.modality_attention = nn.MultiheadAttention(
            workspace_dim, heads, dropout=dropout, batch_first=True
        )
        self.modality_gate = nn.Linear(5 * workspace_dim, 4)
        self.temporal = nn.TransformerEncoderLayer(
            d_model=workspace_dim,
            nhead=heads,
            dim_feedforward=3 * workspace_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(workspace_dim)

    def forward(self, modality_tokens: Tensor, state_tokens: Tensor) -> Tensor:
        if modality_tokens.ndim != 4 or modality_tokens.shape[1:] != (
            HORIZON,
            4,
            WORKSPACE_DIM,
        ):
            raise ValueError("modality_tokens must be [batch,96,4,192]")
        batch = modality_tokens.shape[0]
        if state_tokens.shape != (batch, HORIZON, WORKSPACE_DIM):
            raise ValueError("state_tokens must be [batch,96,192]")
        flat = self.modality_norm(modality_tokens).reshape(
            batch * HORIZON, 4, WORKSPACE_DIM
        )
        attended = self.modality_attention(flat, flat, flat, need_weights=False)[0]
        attended = attended.reshape(batch, HORIZON, 4, WORKSPACE_DIM)
        gates = torch.softmax(
            self.modality_gate(
                torch.cat((*modality_tokens.unbind(dim=2), state_tokens), dim=-1)
            ),
            dim=-1,
        )
        fused = (attended * gates[..., None]).sum(dim=2)
        return self.output_norm(fused + self.temporal(fused))


class RepresentationRepairHead(nn.Module):
    """Persistence-centred price residual and intermittent-workload hurdle.

    The input workspace is detached deliberately: a repair update can change
    only this module.  The inherited Representation MoE and the PV/PUE path
    therefore remain numerically fixed while the failed heads are repaired.
    """

    def __init__(
        self,
        *,
        level_cap: Tensor,
        spread_cap: Tensor,
        workload_cap: Tensor,
        zero_threshold: float,
        zero_prior: float,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        caps = {
            "level_cap": (level_cap, (ZONES,)),
            "spread_cap": (spread_cap, (ZONES,)),
            "workload_cap": (workload_cap, (1,)),
        }
        for name, (value, shape) in caps.items():
            tensor = torch.as_tensor(value, dtype=torch.float32).reshape(shape)
            if not bool(torch.isfinite(tensor).all()) or bool((tensor <= 0).any()):
                raise ValueError(f"Representation repair {name} must be positive")
            self.register_buffer(name, tensor, persistent=True)
        if not 0.5 <= float(zero_threshold) <= 0.99:
            raise ValueError("workload zero threshold must lie in [0.5,0.99]")
        if not 0.0 < float(zero_prior) < 0.5:
            raise ValueError("workload zero prior must lie in (0,0.5)")
        self.register_buffer(
            "workload_zero_threshold",
            torch.tensor(float(zero_threshold), dtype=torch.float32),
            persistent=True,
        )
        candidate_width = 2 * (ZONES + ZONES + 1)
        self.anchor_projection = nn.Sequential(
            nn.Linear(candidate_width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, WORKSPACE_DIM),
            nn.LayerNorm(WORKSPACE_DIM),
        )
        self.feature_norm = nn.LayerNorm(WORKSPACE_DIM)
        self.price_level_head = nn.Linear(WORKSPACE_DIM, ZONES)
        self.price_spread_head = nn.Linear(WORKSPACE_DIM, ZONES)
        self.workload_positive_head = nn.Linear(WORKSPACE_DIM, 1)
        self.workload_zero_head = nn.Linear(WORKSPACE_DIM, 1)
        for module in (
            self.price_level_head,
            self.price_spread_head,
            self.workload_positive_head,
        ):
            _zero_linear(module)
        nn.init.zeros_(self.workload_zero_head.weight)
        prior_logit = torch.logit(torch.tensor(float(zero_prior))).item()
        nn.init.constant_(self.workload_zero_head.bias, prior_logit)

    @staticmethod
    def _price_views(public: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        da = public[..., 0:4]
        rt = public[..., 4:8]
        return 0.5 * (da + rt), da - rt, public[..., 12:13]

    def forward(
        self,
        tokens: Tensor,
        canonical: CanonicalWorkspacePacket,
        anchors: RepresentationAnchorPacket,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        anchors.validate()
        canonical_issue = close_public_channels(canonical.anchor_independent)
        issue_public = anchors.issue_center_public.to(canonical_issue)
        if not torch.equal(issue_public, canonical_issue):
            raise ValueError(
                "Representation issue anchor must equal the frozen canonical center"
            )
        batch = tokens.shape[0]
        if anchors.persistence_public.shape[0] != batch:
            raise ValueError("Representation repair anchor batch mismatch")
        persistence_mask = anchors.availability[:, 0, None, None]
        seasonal_mask = anchors.availability[:, 1, None, None]
        persistence = torch.where(
            persistence_mask, anchors.persistence_public.to(issue_public), issue_public
        )
        seasonal = torch.where(
            seasonal_mask, anchors.seasonal_public.to(issue_public), persistence
        )
        issue_level, issue_spread, issue_workload = self._price_views(issue_public)
        base_level, base_spread, base_workload = self._price_views(persistence)
        seasonal_level, seasonal_spread, seasonal_workload = self._price_views(
            seasonal
        )
        level_cap = self.level_cap.to(tokens)
        spread_cap = self.spread_cap.to(tokens)
        workload_cap = self.workload_cap.to(tokens)
        candidate_features = torch.cat(
            (
                (issue_level - base_level) / level_cap,
                (issue_spread - base_spread) / spread_cap,
                (issue_workload - base_workload) / workload_cap,
                (seasonal_level - base_level) / level_cap,
                (seasonal_spread - base_spread) / spread_cap,
                (seasonal_workload - base_workload) / workload_cap,
            ),
            dim=-1,
        ).clamp(-12.0, 12.0)
        features = self.feature_norm(
            tokens.detach() + self.anchor_projection(candidate_features.to(tokens))
        )
        level_delta = level_cap * torch.tanh(self.price_level_head(features))
        spread_delta = spread_cap * torch.tanh(self.price_spread_head(features))
        base_da = persistence[..., 0:4].to(tokens)
        base_rt = persistence[..., 4:8].to(tokens)
        repaired_da = base_da + level_delta + 0.5 * spread_delta
        repaired_rt = base_rt + level_delta - 0.5 * spread_delta
        positive_workload = (
            base_workload.to(tokens)
            + workload_cap * torch.tanh(self.workload_positive_head(features))
        ).clamp_min(0.0)
        zero_logits = self.workload_zero_head(features.mean(dim=1)).squeeze(-1)
        zero_probabilities = torch.sigmoid(zero_logits)
        learned_hard = (
            zero_probabilities >= self.workload_zero_threshold.to(zero_probabilities)
        ).to(zero_probabilities)
        straight_through = (
            learned_hard + zero_probabilities - zero_probabilities.detach()
        )
        persistence_zero = (
            base_workload.detach().abs().amax(dim=(1, 2)) <= 1.0e-8
        ).to(straight_through)
        zero_gate = persistence_zero + (1.0 - persistence_zero) * straight_through
        workload = (1.0 - zero_gate[:, None, None]) * positive_workload
        repaired = torch.cat((repaired_da, repaired_rt, workload), dim=-1)
        return repaired, {
            "workload_zero_logits": zero_logits,
            "workload_zero_probabilities": zero_probabilities,
            "workload_positive_forecast": positive_workload,
        }


class RepresentationPlane(nn.Module):
    """Semantic plane with a point head that is invisible to Distribution."""

    def __init__(
        self,
        channel_scale_independent: Tensor,
        *,
        expert_hidden_dim: int = 384,
        num_experts: int = 8,
        repair_enabled: bool = False,
        repair_level_cap: Tensor | None = None,
        repair_spread_cap: Tensor | None = None,
        repair_workload_cap: Tensor | None = None,
        workload_zero_threshold: float = 0.9,
        workload_zero_prior: float = 11.0 / 720.0,
        repair_hidden_dim: int = WORKSPACE_DIM,
    ) -> None:
        super().__init__()
        scale = torch.as_tensor(channel_scale_independent, dtype=torch.float32)
        if scale.shape != (INDEPENDENT_CHANNELS,) or bool((scale <= 0).any()):
            raise ValueError("Representation requires 17 positive channel scales")
        self.register_buffer("point_cap", 0.1 * scale, persistent=True)
        self.moe = SparseTop2MoE(
            WORKSPACE_DIM,
            expert_hidden_dim,
            num_experts=num_experts,
            top_k=2,
            router_input_dim=2 * WORKSPACE_DIM,
        )
        self.point_head = nn.Linear(WORKSPACE_DIM, INDEPENDENT_CHANNELS)
        _zero_linear(self.point_head)
        self.repair = (
            RepresentationRepairHead(
                level_cap=(
                    scale[0:4]
                    if repair_level_cap is None
                    else repair_level_cap
                ),
                spread_cap=(
                    torch.sqrt(scale[0:4].square() + scale[4:8].square())
                    if repair_spread_cap is None
                    else repair_spread_cap
                ),
                workload_cap=(
                    scale[8:9]
                    if repair_workload_cap is None
                    else repair_workload_cap
                ),
                zero_threshold=workload_zero_threshold,
                zero_prior=workload_zero_prior,
                hidden_dim=repair_hidden_dim,
            )
            if repair_enabled
            else None
        )

    def forward(
        self,
        canonical: CanonicalWorkspacePacket,
        anchors: RepresentationAnchorPacket | None = None,
    ) -> RepresentationPacket:
        canonical.validate()
        routed = self.moe(
            canonical.time_tokens, router_context=canonical.state_tokens
        )
        tokens = routed["tokens"]
        cap = self.point_cap.to(device=tokens.device, dtype=tokens.dtype)
        legacy_point = canonical.anchor_independent + cap * torch.tanh(
            self.point_head(tokens)
        )
        workload_zero_logits: Tensor | None = None
        workload_zero_probabilities: Tensor | None = None
        workload_positive_forecast: Tensor | None = None
        if self.repair is None:
            point_independent = legacy_point
        else:
            if anchors is None:
                raise ValueError("R0 repair requires RepresentationAnchorPacket")
            repaired, diagnostics = self.repair(tokens, canonical, anchors)
            point_independent = torch.cat((repaired, legacy_point[..., 9:17]), dim=-1)
            workload_zero_logits = diagnostics["workload_zero_logits"]
            workload_zero_probabilities = diagnostics[
                "workload_zero_probabilities"
            ]
            workload_positive_forecast = diagnostics[
                "workload_positive_forecast"
            ]
        return RepresentationPacket(
            time_tokens=tokens,
            point_independent=point_independent,
            point_forecast=close_public_channels(point_independent),
            router_logits=routed["router_logits"],
            router_probabilities=routed["router_probabilities"],
            topk_indices=routed["topk_indices"],
            topk_weights=routed["topk_weights"],
            workload_zero_logits=workload_zero_logits,
            workload_zero_probabilities=workload_zero_probabilities,
            workload_positive_forecast=workload_positive_forecast,
        ).validate()


class DistributionPlane(nn.Module):
    """Anchor-conditioned probability truth independent of point forecasts."""

    def __init__(
        self,
        residuals_public: Tensor,
        channel_scale_public: Tensor,
        *,
        expert_hidden_dim: int = 384,
        num_experts: int = 8,
        transport_cap_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        residuals = independent_from_public(
            torch.as_tensor(residuals_public, dtype=torch.float32)
        )
        scale = independent_from_public(
            torch.as_tensor(channel_scale_public, dtype=torch.float32).reshape(1, 21)
        ).squeeze(0)
        if residuals.ndim != 3 or residuals.shape[1:] != (
            HORIZON,
            INDEPENDENT_CHANNELS,
        ):
            raise ValueError("residuals_public must be [days,96,21]")
        if scale.shape != (INDEPENDENT_CHANNELS,) or bool((scale <= 0).any()):
            raise ValueError("channel_scale_public must contain 21 positive values")
        if not 0.0 < float(transport_cap_fraction) <= 2.0:
            raise ValueError("transport_cap_fraction must lie in (0,2]")
        self.anchor_day_count = int(residuals.shape[0])
        self.transport_cap_fraction = float(transport_cap_fraction)
        self.register_buffer("baseline_residuals", residuals, persistent=False)
        self.register_buffer("channel_scale_independent", scale, persistent=True)
        self.anchor_projection = nn.Sequential(
            nn.LayerNorm(INDEPENDENT_CHANNELS),
            nn.Linear(INDEPENDENT_CHANNELS, WORKSPACE_DIM),
            nn.SiLU(),
        )
        self.conditioner = nn.TransformerEncoderLayer(
            d_model=WORKSPACE_DIM,
            nhead=6,
            dim_feedforward=3 * WORKSPACE_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.moe = SparseTop2MoE(
            WORKSPACE_DIM,
            expert_hidden_dim,
            num_experts=num_experts,
            top_k=2,
            router_input_dim=2 * WORKSPACE_DIM,
        )
        self.center_head = nn.Linear(WORKSPACE_DIM, INDEPENDENT_CHANNELS)
        self.location_head = nn.Linear(WORKSPACE_DIM, INDEPENDENT_CHANNELS)
        self.log_scale_head = nn.Linear(WORKSPACE_DIM, INDEPENDENT_CHANNELS)
        self.tail_head = nn.Linear(WORKSPACE_DIM, INDEPENDENT_CHANNELS)
        self.factor_head = nn.Linear(
            WORKSPACE_DIM, INDEPENDENT_CHANNELS * DEPENDENCE_RANK
        )
        self.regime_head = nn.Linear(WORKSPACE_DIM, REGIME_COUNT)
        for head in (
            self.center_head,
            self.location_head,
            self.log_scale_head,
            self.tail_head,
            self.factor_head,
            self.regime_head,
        ):
            _zero_linear(head)
        self.regime_shift_table = nn.Parameter(
            torch.zeros(REGIME_COUNT, INDEPENDENT_CHANNELS)
        )
        self.factor_basis = nn.Parameter(
            0.01 * torch.randn(DEPENDENCE_RANK, WORKSPACE_DIM)
        )
        self.regime_basis = nn.Parameter(
            0.01 * torch.randn(REGIME_COUNT, WORKSPACE_DIM)
        )
        self.stats_projection = nn.Linear(
            3 * INDEPENDENT_CHANNELS, WORKSPACE_DIM, bias=False
        )
        channel_axis = torch.arange(1, INDEPENDENT_CHANNELS + 1)[:, None]
        factor_axis = torch.arange(1, DEPENDENCE_RANK + 1)[None]
        projection = torch.sin(channel_axis * factor_axis).float()
        projection = projection / projection.square().sum(0, keepdim=True).sqrt()
        self.register_buffer("factor_projection", projection, persistent=True)

    def forward(self, canonical: CanonicalWorkspacePacket) -> DistributionTruthPacket:
        """Build truth from canonical inputs and anchors, never point forecasts."""

        canonical.validate()
        scale = self.channel_scale_independent.to(
            device=canonical.time_tokens.device, dtype=canonical.time_tokens.dtype
        ).reshape(1, 1, INDEPENDENT_CHANNELS)
        normalized_anchor = canonical.anchor_independent / scale
        conditioned = canonical.time_tokens + self.anchor_projection(normalized_anchor)
        conditioned = conditioned + self.conditioner(conditioned)
        routed = self.moe(conditioned, router_context=canonical.state_tokens)
        tokens = routed["tokens"]
        cap = scale.expand_as(canonical.anchor_independent) * self.transport_cap_fraction
        center = canonical.anchor_independent + cap * torch.tanh(
            self.center_head(tokens)
        )
        location = self.location_head(tokens)
        log_scale = 0.75 * torch.tanh(self.log_scale_head(tokens))
        truth_scale = scale.expand_as(center) * log_scale.exp()
        tail = 0.5 * torch.tanh(self.tail_head(tokens))
        factor_loadings = 0.5 * torch.tanh(
            self.factor_head(tokens).reshape(
                tokens.shape[0],
                HORIZON,
                INDEPENDENT_CHANNELS,
                DEPENDENCE_RANK,
            )
        )
        regime_logits = self.regime_head(tokens.mean(dim=1))
        regime_probabilities = torch.softmax(regime_logits, dim=-1)
        regime_shift = regime_probabilities @ self.regime_shift_table
        stats = torch.cat((center / scale, truth_scale / scale, tail), dim=-1)
        truth_time = F.layer_norm(
            tokens + 0.05 * self.stats_projection(stats), (WORKSPACE_DIM,)
        )
        loading_strength = factor_loadings.abs().mean(dim=2).transpose(1, 2)
        factor_weights = torch.softmax(loading_strength, dim=-1)
        factor_tokens = torch.einsum(
            "brt,btd->brd", factor_weights, truth_time
        ) + self.factor_basis[None]
        pooled = truth_time.mean(dim=1, keepdim=True)
        regime_tokens = self.regime_basis[None] + regime_probabilities[..., None] * pooled
        return DistributionTruthPacket(
            time_tokens=truth_time,
            factor_tokens=factor_tokens,
            regime_tokens=regime_tokens,
            anchor_independent=canonical.anchor_independent,
            center_independent=center,
            location_independent=location,
            scale_independent=truth_scale,
            tail_independent=tail,
            factor_loadings=factor_loadings,
            regime_probabilities=regime_probabilities,
            regime_shift=regime_shift,
            origin_index=canonical.origin_index,
            router_logits=routed["router_logits"],
            router_probabilities=routed["router_probabilities"],
            topk_indices=routed["topk_indices"],
            topk_weights=routed["topk_weights"],
        ).validate()

    def _indices(
        self,
        batch: int,
        count: int,
        reference: Tensor,
        *,
        base_indices: Tensor | None,
        seed: int | None,
    ) -> Tensor:
        if count < 1:
            raise ValueError("sample_count must be positive")
        if base_indices is not None:
            result = torch.as_tensor(base_indices, device=reference.device)
            if result.ndim == 1:
                result = result[None].expand(batch, -1)
            if result.shape != (batch, count):
                raise ValueError("base_indices must be [N] or [batch,N]")
            result = result.long()
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(torch.seed() if seed is None else int(seed))
            if count <= self.anchor_day_count:
                shared = torch.randperm(self.anchor_day_count, generator=generator)[:count]
            else:
                shared = torch.randint(
                    self.anchor_day_count, (count,), generator=generator
                )
            result = shared.to(reference.device)[None].expand(batch, -1)
        if bool(((result < 0) | (result >= self.anchor_day_count)).any()):
            raise ValueError("base_indices fall outside the frozen residual anchor")
        return result

    def sample(
        self,
        truth: DistributionTruthPacket,
        sample_count: int,
        *,
        base_indices: Tensor | None = None,
        seed: int | None = None,
    ) -> DistributionSamplePacket:
        truth.validate()
        batch = truth.time_tokens.shape[0]
        count = int(sample_count)
        indices = self._indices(
            batch,
            count,
            truth.time_tokens,
            base_indices=base_indices,
            seed=seed,
        )
        residuals = self.baseline_residuals.to(
            device=truth.time_tokens.device, dtype=truth.time_tokens.dtype
        )[indices]
        base_scale = self.channel_scale_independent.to(
            device=residuals.device, dtype=residuals.dtype
        ).reshape(1, 1, 1, INDEPENDENT_CHANNELS)
        standardized = residuals / base_scale
        factor_score = torch.einsum(
            "bntc,cr->bntr",
            standardized,
            self.factor_projection.to(residuals),
        )
        factor_delta = torch.einsum(
            "bntr,btcr->bntc", factor_score, truth.factor_loadings
        )
        relative_scale = truth.scale_independent[:, None] / base_scale
        raw_delta = (
            truth.location_independent[:, None]
            + (relative_scale - 1.0) * standardized
            + truth.tail_independent[:, None]
            * standardized
            * standardized.abs()
            + truth.regime_shift[:, None, None]
            + factor_delta
        )
        cap = self.transport_cap_fraction * torch.ones_like(raw_delta)
        transported = residuals + base_scale * cap * torch.tanh(raw_delta / cap)
        independent_samples = truth.center_independent[:, None] + transported
        public_samples = close_public_channels(independent_samples)
        weights = public_samples.new_full((batch, count), 1.0 / float(count))
        return DistributionSamplePacket(
            independent_samples=independent_samples,
            public_samples=public_samples,
            weights=weights,
            base_indices=indices,
        ).validate()


class DecisionBridge(nn.Module):
    """Create a decision-owned workspace from stop-gradient C/R/D snapshots."""

    def __init__(self) -> None:
        super().__init__()
        self.source_attention = nn.MultiheadAttention(
            WORKSPACE_DIM, 6, batch_first=True
        )
        self.structure_attention = nn.MultiheadAttention(
            WORKSPACE_DIM, 6, batch_first=True
        )
        self.stats_projection = nn.Sequential(
            nn.LayerNorm(3 * PUBLIC_CHANNELS),
            nn.Linear(3 * PUBLIC_CHANNELS, WORKSPACE_DIM),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(5 * WORKSPACE_DIM),
            nn.Linear(5 * WORKSPACE_DIM, 2 * WORKSPACE_DIM),
            nn.SiLU(),
            nn.Linear(2 * WORKSPACE_DIM, WORKSPACE_DIM),
            nn.LayerNorm(WORKSPACE_DIM),
        )

    def forward(
        self,
        canonical: CanonicalWorkspacePacket,
        representation: RepresentationPacket,
        truth: DistributionTruthPacket,
        *,
        detach_shared: bool = True,
    ) -> DecisionWorkspacePacket:
        # Representation and probabilistic truth are always protected.  The
        # canonical workspace may be opened only in registered joint phases so
        # Policy gradients can participate in conflict-projected shared updates.
        canonical = (
            canonical.detached().validate()
            if bool(detach_shared)
            else canonical.validate()
        )
        representation = representation.detached().validate()
        truth = truth.detached().validate()
        source = torch.cat((representation.time_tokens, truth.time_tokens), dim=1)
        source_read = self.source_attention(
            canonical.time_tokens, source, source, need_weights=False
        )[0]
        structure = torch.cat((truth.factor_tokens, truth.regime_tokens), dim=1)
        structure_read = self.structure_attention(
            canonical.time_tokens, structure, structure, need_weights=False
        )[0]
        stats = truth.stats_public
        stats_read = self.stats_projection(stats)
        state = canonical.state_hidden[:, None].expand(-1, HORIZON, -1)
        output = self.output(
            torch.cat(
                (
                    canonical.time_tokens,
                    source_read,
                    structure_read,
                    stats_read,
                    state,
                ),
                dim=-1,
            )
        )
        return DecisionWorkspacePacket(
            time_tokens=output,
            structure_tokens=structure,
            stats_public=stats,
            state_hidden=canonical.state_hidden,
            origin_index=canonical.origin_index,
        ).validate()


class DecisionRefinementRound(nn.Module):
    def __init__(self, expert_hidden_dim: int = 384) -> None:
        super().__init__()
        self.sequence = nn.TransformerEncoderLayer(
            d_model=WORKSPACE_DIM,
            nhead=6,
            dim_feedforward=3 * WORKSPACE_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.moe = SparseTop2MoE(
            WORKSPACE_DIM,
            expert_hidden_dim,
            num_experts=8,
            top_k=2,
            router_input_dim=2 * WORKSPACE_DIM,
        )
        self.feedback = nn.MultiheadAttention(WORKSPACE_DIM, 6, batch_first=True)
        self.feedback_delta = nn.Linear(WORKSPACE_DIM, WORKSPACE_DIM)
        _zero_linear(self.feedback_delta)

    def forward(
        self, workspace: DecisionWorkspacePacket
    ) -> tuple[DecisionWorkspacePacket, Tensor, dict[str, Tensor]]:
        workspace.validate()
        state = workspace.state_hidden[:, None].expand(-1, HORIZON, -1)
        plan = workspace.time_tokens + self.sequence(workspace.time_tokens)
        routed = self.moe(plan, router_context=state)
        plan = routed["tokens"]
        feedback = self.feedback(
            workspace.time_tokens, plan, plan, need_weights=False
        )[0]
        updated = replace(
            workspace,
            time_tokens=workspace.time_tokens
            + 0.1 * torch.tanh(self.feedback_delta(feedback)),
            refinement_round=workspace.refinement_round + 1,
        ).validate()
        diagnostics = {
            key: value
            for key, value in routed.items()
            if key != "tokens" and isinstance(value, Tensor)
        }
        return updated, plan, diagnostics


class PolicyPlane(nn.Module):
    """Two-round Policy/world refinement with action-conditioned risk output."""

    def __init__(self, *, expert_hidden_dim: int = 384) -> None:
        super().__init__()
        self.refinement_rounds = nn.ModuleList(
            DecisionRefinementRound(expert_hidden_dim) for _ in range(2)
        )
        self.day_ahead_head = nn.Linear(WORKSPACE_DIM, ZONES)
        self.battery_head = nn.Linear(WORKSPACE_DIM, ZONES)
        self.compute_head = nn.Linear(WORKSPACE_DIM, ZONES)
        for head in (self.day_ahead_head, self.battery_head, self.compute_head):
            _zero_linear(head)
        self.critic = nn.Sequential(
            nn.LayerNorm(3 * WORKSPACE_DIM),
            nn.Linear(3 * WORKSPACE_DIM, 2 * WORKSPACE_DIM),
            nn.SiLU(),
            nn.Linear(2 * WORKSPACE_DIM, 5),
        )

    @staticmethod
    def _limit(
        limits: Mapping[str, Any] | None,
        name: str,
        reference: Tensor,
    ) -> Tensor:
        raw = 1.0 if limits is None else limits.get(name, 1.0)
        value = torch.as_tensor(raw, device=reference.device, dtype=reference.dtype)
        batch = reference.shape[0]
        if value.ndim == 0:
            value = value.reshape(1, 1).expand(batch, ZONES)
        elif value.shape == (ZONES,):
            value = value[None].expand(batch, -1)
        if value.shape != (batch, ZONES) or bool((value <= 0).any()):
            raise ValueError(f"{name} limit must resolve to positive [batch,4]")
        return value[:, None]

    def forward(
        self,
        workspace: DecisionWorkspacePacket,
        *,
        limits: Mapping[str, Any] | None = None,
        issued_day_ahead: Tensor | None = None,
    ) -> PolicyPacket:
        workspace.validate()
        trace: list[Tensor] = []
        diagnostics: list[dict[str, Tensor]] = []
        current = workspace
        plan = workspace.time_tokens
        for refinement in self.refinement_rounds:
            current, plan, router = refinement(current)
            trace.append(plan)
            diagnostics.append(router)
        day_ahead = torch.sigmoid(self.day_ahead_head(plan)) * self._limit(
            limits, "grid_mw", plan
        )
        battery = torch.tanh(self.battery_head(plan)) * self._limit(
            limits, "battery_mw", plan
        )
        compute = torch.sigmoid(self.compute_head(plan)) * self._limit(
            limits, "compute_work", plan
        )
        batch = plan.shape[0]
        row = torch.arange(batch, device=plan.device)
        origin = workspace.origin_index
        if issued_day_ahead is not None:
            issued = torch.as_tensor(
                issued_day_ahead, device=plan.device, dtype=plan.dtype
            )
            if issued.shape != day_ahead.shape:
                raise ValueError("issued_day_ahead must be [batch,96,4]")
            day_ahead = torch.where(origin.eq(0)[:, None, None], day_ahead, issued)
        current_plan = plan[row, origin]
        future_mask = (
            torch.arange(HORIZON, device=plan.device)[None] >= origin[:, None]
        ).to(plan.dtype)[..., None]
        future_plan = (plan * future_mask).sum(1) / future_mask.sum(1).clamp_min(1.0)
        critic = self.critic(
            torch.cat((current_plan, future_plan, current.state_hidden), dim=-1)
        )
        return PolicyPacket(
            day_ahead_plan=day_ahead,
            battery_plan=battery,
            compute_plan=compute,
            current_battery=battery[row, origin],
            current_compute=compute[row, origin],
            critic=critic,
            plan_tokens=plan,
            origin_index=origin,
            refinement_trace=tuple(trace),
            router_diagnostics=tuple(diagnostics),
        ).validate()


__all__ = [
    "DecisionBridge",
    "DecisionRefinementRound",
    "DistributionPlane",
    "PolicyPlane",
    "RepresentationPlane",
    "TriFusionWorkspace",
]
