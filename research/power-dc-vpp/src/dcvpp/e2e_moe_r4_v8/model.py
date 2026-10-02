"""R4-V8 parallel three-plane causal world model.

V8 shares causal modality encoders and a 192-wide workspace, then forks into
parallel Representation and Distribution planes.  Policy reads typed,
stop-gradient snapshots through a decision-owned bridge.  No optimization
solver is imported or required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from dcvpp.e2e_moe_r4.liquid import ClosedFormLiquidSequence
from dcvpp.e2e_moe_r4.model import MarketTransformerEncoder, PhysicalStream
from dcvpp.e2e_moe_r4_v7.data import FrozenResidualAnchor, issue_center_from_batch
from dcvpp.e2e_moe_r4_v7.model import V7ModelConfig

from .packets import (
    CanonicalWorkspacePacket,
    DistributionSamplePacket,
    DistributionTruthPacket,
    HORIZON,
    MODALITY_DIM,
    PolicyPacket,
    RepresentationAnchorPacket,
    RepresentationPacket,
    WORKSPACE_DIM,
    close_public_channels,
    independent_from_public,
)
from .trifusion import (
    DecisionBridge,
    DistributionPlane,
    PolicyPlane,
    RepresentationPlane,
    TriFusionWorkspace,
)


ARCHITECTURE_VERSION = "causal_trifusion_joint_moe_r4_v8"
PARAMETER_GROUPS = (
    "representation",
    "distribution_truth",
    "decision",
    "shared",
    "router",
    "frozen",
)


@dataclass(frozen=True)
class V8ModelConfig:
    horizon: int = HORIZON
    modality_dim: int = MODALITY_DIM
    workspace_dim: int = WORKSPACE_DIM
    market_dim: int = 16
    workload_dim: int = 5
    physical_static_dim: int = 4
    physical_dynamic_dim: int = 12
    state_dim: int = 49
    history_days: int = 7
    market_layers: int = 2
    market_heads: int = 4
    workspace_heads: int = 6
    num_experts: int = 8
    top_k: int = 2
    expert_hidden_dim: int = 384
    transport_cap_fraction: float = 0.5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (self.horizon, self.modality_dim, self.workspace_dim) != (96, 128, 192):
            raise ValueError("V8 fixes horizon=96, modality_dim=128, workspace_dim=192")
        if (self.num_experts, self.top_k) != (8, 2):
            raise ValueError("V8 requires eight sparse Top-2 experts")
        if self.workspace_heads != 6:
            raise ValueError("V8 fixes six 32-wide workspace attention heads")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "V8ModelConfig":
        if raw is None:
            return cls()
        root: Mapping[str, Any] = raw.get("model", raw) if isinstance(raw, Mapping) else {}
        known = {field.name for field in fields(cls)}
        return cls(**{key: root[key] for key in known if key in root})

    def v7_representation_config(self) -> Any:
        return V7ModelConfig(
            horizon=self.horizon,
            history_days=self.history_days,
            market_dim=self.market_dim,
            workload_dim=self.workload_dim,
            physical_static_dim=self.physical_static_dim,
            physical_dynamic_dim=self.physical_dynamic_dim,
            state_dim=self.state_dim,
            hidden_dim=self.modality_dim,
            market_layers=self.market_layers,
            market_heads=self.market_heads,
            dropout=self.dropout,
        ).representation_config()


def _origin_index(
    origin: int | Tensor, batch: int, reference: Tensor
) -> Tensor:
    value = torch.as_tensor(origin, device=reference.device)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,):
        raise ValueError("origin must be scalar or [batch]")
    if value.is_floating_point() and not bool(torch.equal(value, value.round())):
        raise ValueError("origin must contain integers")
    value = value.long()
    if bool(((value < 0) | (value >= HORIZON)).any()):
        raise ValueError("origin must lie in [0,95]")
    return value


class TriPlaneCausalWorldModelV8(nn.Module):
    """Shared causal workspace with parallel semantic/truth/decision planes."""

    architecture_version = ARCHITECTURE_VERSION
    solver_free = True

    def __init__(
        self,
        residual_anchor: FrozenResidualAnchor,
        config: V8ModelConfig | Mapping[str, Any] | None = None,
        *,
        physical_twin_core: nn.Module | None = None,
        representation_repair_statistics: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, V8ModelConfig)
            else V8ModelConfig.from_mapping(config)
        )
        self.residual_anchor = residual_anchor
        rep = self.config.v7_representation_config()

        # Reused, proven 128-wide modality encoders.  The trainable V8 workspace
        # starts only at the explicit 128->192 projections below.
        self.market_encoder = MarketTransformerEncoder(rep)
        self.workload_encoder = ClosedFormLiquidSequence(
            self.config.workload_dim,
            self.config.modality_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
        )
        self.physical_stream = PhysicalStream(rep, physical_twin_core)
        self.state_encoder = ClosedFormLiquidSequence(
            self.config.state_dim,
            self.config.modality_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
        )
        self.modality_projections = nn.ModuleDict(
            {
                name: nn.Linear(self.config.modality_dim, self.config.workspace_dim)
                for name in ("market", "workload", "physical", "state")
            }
        )
        self.trifusion = TriFusionWorkspace(
            self.config.workspace_dim,
            heads=self.config.workspace_heads,
            dropout=self.config.dropout,
        )

        scale_independent = independent_from_public(
            torch.as_tensor(residual_anchor.channel_scale).reshape(1, 21)
        ).squeeze(0)
        raw_model = (
            config.get("model", config)
            if isinstance(config, Mapping)
            else {}
        )
        repair_config = raw_model.get("representation_repair", {})
        if not isinstance(repair_config, Mapping):
            raise ValueError("model.representation_repair must be a mapping")
        repair_enabled = bool(repair_config.get("enabled", False))
        repair_statistics = (
            representation_repair_statistics
            if representation_repair_statistics is not None
            else getattr(residual_anchor, "representation_repair_statistics", None)
        )
        persistence_statistics = getattr(repair_statistics, "persistence", None)
        level_cap = getattr(persistence_statistics, "price_level_cap", None)
        spread_cap = getattr(persistence_statistics, "price_spread_cap", None)
        workload_cap = getattr(persistence_statistics, "workload_cap", None)
        zero_days = int(getattr(repair_statistics, "workload_zero_day_count", 11))
        nonzero_days = int(
            getattr(repair_statistics, "workload_nonzero_day_count", 709)
        )
        zero_prior = float(zero_days) / float(max(1, zero_days + nonzero_days))
        self.representation = RepresentationPlane(
            scale_independent,
            expert_hidden_dim=self.config.expert_hidden_dim,
            num_experts=self.config.num_experts,
            repair_enabled=repair_enabled,
            repair_level_cap=level_cap,
            repair_spread_cap=spread_cap,
            repair_workload_cap=workload_cap,
            workload_zero_threshold=float(
                repair_config.get("workload_zero_threshold", 0.9)
            ),
            workload_zero_prior=zero_prior,
            repair_hidden_dim=int(
                repair_config.get("hidden_dim", self.config.workspace_dim)
            ),
        )
        self.distribution = DistributionPlane(
            residual_anchor.residuals,
            residual_anchor.channel_scale,
            expert_hidden_dim=self.config.expert_hidden_dim,
            num_experts=self.config.num_experts,
            transport_cap_fraction=self.config.transport_cap_fraction,
        )
        self.decision_bridge = DecisionBridge()
        self.policy = PolicyPlane(expert_hidden_dim=self.config.expert_hidden_dim)
        self.physical_stream.freeze_twin()

    def train(self, mode: bool = True) -> "TriPlaneCausalWorldModelV8":
        super().train(mode)
        self.physical_stream.freeze_twin()
        return self

    def _tensor(self, value: Any) -> Tensor:
        reference = next(self.parameters())
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _required(batch: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if batch.get(name) is not None:
                return batch[name]
        raise KeyError(f"missing required V8 input; expected one of {names}")

    def _workload(self, value: Any) -> Tensor:
        workload = self._tensor(value)
        if workload.ndim != 3 or workload.shape[1] != HORIZON:
            raise ValueError("workload must be [batch,96,features]")
        if workload.shape[-1] == self.config.workload_dim:
            return workload
        if workload.shape[-1] == 2 and self.config.workload_dim == 5:
            mean, std = workload.unbind(-1)
            return torch.stack(
                (mean, std, mean - 1.2815516 * std, mean, mean + 1.2815516 * std),
                dim=-1,
            )
        raise ValueError("workload feature width is incompatible with V8")

    def _state_sequence(
        self,
        raw: Any | None,
        *,
        batch: int,
        reference: Tensor,
        origin: Tensor,
    ) -> Tensor:
        if raw is None:
            sequence = reference.new_zeros((batch, HORIZON, self.config.state_dim))
        elif isinstance(raw, Mapping):
            parts: list[Tensor] = []
            widths = (
                ("current_soc", 4),
                ("backlog", 4),
                ("deadline_pressure", 4),
                ("previous_grid", 4),
                ("previous_battery", 4),
                ("previous_compute", 4),
                ("rt_innovation", 4),
                ("workload_innovation", 1),
                ("pv_innovation", 4),
            )
            for name, width in widths:
                value = raw.get(name)
                if value is None:
                    parts.append(reference.new_zeros((batch, width)))
                    continue
                tensor = self._tensor(value)
                if tensor.shape == (width,):
                    tensor = tensor[None].expand(batch, -1)
                if tensor.shape != (batch, width):
                    raise ValueError(f"rolling state {name} must be [batch,{width}]")
                parts.append(tensor)
            snapshot = torch.cat(parts, dim=-1)
            snapshot = torch.nn.functional.pad(
                snapshot, (0, self.config.state_dim - snapshot.shape[-1])
            )
            sequence = reference.new_zeros((batch, HORIZON, self.config.state_dim))
            sequence[torch.arange(batch, device=reference.device), origin] = snapshot
        else:
            tensor = self._tensor(raw)
            if tensor.ndim == 2:
                if tensor.shape[0] != batch or tensor.shape[-1] > self.config.state_dim:
                    raise ValueError("state snapshot must be [batch,features<=49]")
                tensor = torch.nn.functional.pad(
                    tensor, (0, self.config.state_dim - tensor.shape[-1])
                )
                sequence = reference.new_zeros((batch, HORIZON, self.config.state_dim))
                sequence[torch.arange(batch, device=reference.device), origin] = tensor
            elif tensor.ndim == 3:
                if tensor.shape[:2] != (batch, HORIZON) or tensor.shape[-1] > self.config.state_dim:
                    raise ValueError("state sequence must be [batch,96,features<=49]")
                sequence = torch.nn.functional.pad(
                    tensor, (0, self.config.state_dim - tensor.shape[-1])
                )
            else:
                raise ValueError("state must be mapping, [batch,F], or [batch,96,F]")
        periods = torch.arange(HORIZON, device=reference.device)[None]
        return sequence * (periods <= origin[:, None]).unsqueeze(-1).to(sequence.dtype)

    def _anchor_public(self, batch: Mapping[str, Any], reference: Tensor) -> Tensor:
        value = batch.get("anchor_center_zonal")
        if value is None:
            value = issue_center_from_batch(dict(batch), self.residual_anchor)
        anchor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if anchor.shape != (reference.shape[0], HORIZON, 21):
            raise ValueError("anchor_center_zonal must be [batch,96,21]")
        return anchor

    def _encode_exogenous(
        self, batch: Mapping[str, Any], market: Tensor
    ) -> dict[str, Tensor]:
        """Encode issue-time exogenous streams once for rolling Policy calls."""

        workload = self._workload(
            self._required(batch, "workload", "workload_features", "workload_inputs")
        )
        physical_static = self._tensor(
            self._required(batch, "physical_static", "physical", "physical_features")
        )
        physical_dynamic = batch.get("physical_dynamic")
        dynamic = None if physical_dynamic is None else self._tensor(physical_dynamic)
        history = batch.get("historical_market")
        history_tensor = None if history is None else self._tensor(history)
        market_tokens = self.market_encoder(
            market,
            historical_market=history_tensor,
            historical_market_available=batch.get("historical_market_available"),
            day_of_week=batch.get("day_of_week"),
            month=batch.get("month"),
            holiday=batch.get("holiday"),
            dst=batch.get("dst"),
        )["tokens"]
        workload_tokens, _ = self.workload_encoder(
            workload, dt=batch.get("workload_dt")
        )
        physical_tokens = self.physical_stream(physical_static, dynamic)["tokens"]
        return {
            "market": self.modality_projections["market"](market_tokens),
            "workload": self.modality_projections["workload"](workload_tokens),
            "physical": self.modality_projections["physical"](physical_tokens),
        }

    def _exogenous_tokens(
        self, batch: Mapping[str, Any], market: Tensor
    ) -> dict[str, Tensor]:
        cached = batch.get("_v8_exogenous_cache")
        if cached is None:
            return self._encode_exogenous(batch, market)
        if not isinstance(cached, Mapping):
            raise ValueError("_v8_exogenous_cache must be a mapping")
        expected = (market.shape[0], HORIZON, self.config.workspace_dim)
        result: dict[str, Tensor] = {}
        for name in ("market", "workload", "physical"):
            value = cached.get(name)
            if not isinstance(value, Tensor) or value.shape != expected:
                raise ValueError(f"cached {name} tokens must be {expected}")
            result[name] = value.to(device=market.device, dtype=market.dtype)
        return result

    def prepare_canonical(
        self, batch: Mapping[str, Any], origin: int | Tensor = 0
    ) -> CanonicalWorkspacePacket:
        market = self._tensor(
            self._required(batch, "market", "market_features", "market_inputs")
        )
        if market.ndim != 3 or market.shape[1:] != (HORIZON, self.config.market_dim):
            raise ValueError("market must be [batch,96,16]")
        batch_size = market.shape[0]
        origin_index = _origin_index(origin, batch_size, market)
        projected = self._exogenous_tokens(batch, market)
        raw_state = batch.get(
            "rolling_state",
            batch.get(
                "initial_state",
                batch.get(
                    "state_sequence", batch.get("state", batch.get("state_features"))
                ),
            ),
        )
        state_sequence = self._state_sequence(
            raw_state,
            batch=batch_size,
            reference=market,
            origin=origin_index,
        )
        state_tokens_128, _ = self.state_encoder(
            state_sequence, dt=batch.get("state_dt")
        )
        projected = {
            **projected,
            "state": self.modality_projections["state"](state_tokens_128),
        }
        modality_tokens = torch.stack(tuple(projected.values()), dim=2)
        state_tokens = projected["state"]
        time_tokens = self.trifusion(modality_tokens, state_tokens)
        row = torch.arange(batch_size, device=market.device)
        state_hidden = state_tokens[row, origin_index]
        anchor_independent = independent_from_public(
            self._anchor_public(batch, market)
        )
        return CanonicalWorkspacePacket(
            time_tokens=time_tokens,
            modality_tokens=modality_tokens,
            state_tokens=state_tokens,
            state_hidden=state_hidden,
            anchor_independent=anchor_independent,
            origin_index=origin_index,
        ).validate()

    def _representation_anchor_packet(
        self,
        batch: Mapping[str, Any],
        canonical: CanonicalWorkspacePacket,
    ) -> RepresentationAnchorPacket | None:
        if self.representation.repair is None:
            return None
        raw = batch.get("representation_anchors")
        if not isinstance(raw, Mapping):
            raise ValueError("R0 repair batch has no representation_anchors packet")
        reference = canonical.anchor_independent

        def tensor(name: str) -> Tensor:
            value = raw.get(name)
            if not isinstance(value, Tensor):
                raise ValueError(f"representation_anchors.{name} is required")
            return value.to(device=reference.device, dtype=reference.dtype)

        availability = raw.get("availability")
        if not isinstance(availability, Tensor):
            raise ValueError("representation_anchors.availability is required")
        packet = RepresentationAnchorPacket(
            issue_center_public=tensor("issue_center_public"),
            persistence_public=tensor("persistence_public"),
            seasonal_public=tensor("seasonal_public"),
            availability=availability.to(device=reference.device, dtype=torch.bool),
        ).validate()
        expected_issue = close_public_channels(canonical.anchor_independent)
        if not torch.equal(packet.issue_center_public, expected_issue):
            raise ValueError(
                "Representation issue anchor differs from Distribution issue center"
            )
        return packet

    def forward_planes(
        self, batch: Mapping[str, Any], origin: int | Tensor = 0
    ) -> dict[str, Any]:
        canonical = self.prepare_canonical(batch, origin)
        # Both planes consume the common packet directly.  Distribution never
        # receives RepresentationPacket or its point_forecast field.
        anchors = self._representation_anchor_packet(batch, canonical)
        representation = self.representation(canonical, anchors)
        distribution = self.distribution(canonical)
        return {
            "canonical": canonical,
            "representation": representation,
            "distribution": distribution,
        }

    def forward_representation(
        self,
        batch: Mapping[str, Any],
        *,
        origin: int | Tensor = 0,
        scientific_score_dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Trainer-facing semantic-plane interface."""

        del scientific_score_dtype
        canonical = self.prepare_canonical(batch, origin)
        anchors = self._representation_anchor_packet(batch, canonical)
        return {
            "canonical": canonical,
            "representation": self.representation(canonical, anchors),
        }

    def forward_distribution(
        self,
        batch: Mapping[str, Any],
        sample_count: int | None = None,
        *,
        origin: int | Tensor = 0,
        base_indices: Tensor | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        planes = self.forward_planes(batch, origin)
        truth: DistributionTruthPacket = planes["distribution"]
        samples: DistributionSamplePacket | None = None
        baseline_samples: Tensor | None = None
        if sample_count is not None:
            samples = self.distribution.sample(
                truth,
                int(sample_count),
                base_indices=base_indices,
                seed=seed,
            )
            residuals = self.distribution.baseline_residuals.to(
                device=truth.time_tokens.device,
                dtype=truth.time_tokens.dtype,
            )[samples.base_indices]
            baseline_samples = close_public_channels(
                truth.anchor_independent[:, None] + residuals
            )
        return {
            **planes,
            "samples": samples,
            "baseline_samples": baseline_samples,
        }

    def forward_decision(
        self,
        batch: Mapping[str, Any],
        *,
        origin: int | Tensor = 0,
        planes: Mapping[str, Any] | None = None,
        limits: Mapping[str, Any] | None = None,
        issued_day_ahead: Tensor | None = None,
        detach_shared: bool = True,
    ) -> dict[str, Any]:
        resolved = self.forward_planes(batch, origin) if planes is None else dict(planes)
        canonical = resolved.get("canonical")
        representation = resolved.get("representation")
        truth = resolved.get("distribution")
        if not isinstance(canonical, CanonicalWorkspacePacket):
            raise ValueError("planes.canonical must be CanonicalWorkspacePacket")
        if not isinstance(representation, RepresentationPacket):
            raise ValueError("planes.representation must be RepresentationPacket")
        if not isinstance(truth, DistributionTruthPacket):
            raise ValueError("planes.distribution must be DistributionTruthPacket")
        workspace = self.decision_bridge(
            canonical,
            representation,
            truth,
            detach_shared=detach_shared,
        )
        policy = self.policy(
            workspace, limits=limits, issued_day_ahead=issued_day_ahead
        )
        critic = {
            "expected": policy.critic[:, 0],
            "var": policy.critic[:, 1],
            "cvar": policy.critic[:, 2],
            "sla": policy.critic[:, 3],
            "intervention": policy.critic[:, 4],
        }
        actions = {
            "day_ahead": policy.day_ahead_plan,
            "battery": policy.battery_plan,
            "compute": policy.compute_plan,
        }
        return {
            **resolved,
            "decision_workspace": workspace,
            "policy": policy,
            "actions": actions,
            "critic": critic,
            "day_ahead": policy.day_ahead_plan,
            "battery_plan": policy.battery_plan,
            "compute_plan": policy.compute_plan,
            "current_battery": policy.current_battery,
            "current_compute": policy.current_compute,
        }

    def forward_policy(
        self,
        batch: Mapping[str, Any],
        *,
        origin: int | Tensor = 0,
        planes: Mapping[str, Any] | None = None,
        limits: Mapping[str, Any] | None = None,
        issued_day_ahead: Tensor | None = None,
        detach_distribution_truth: bool = True,
        detach_shared: bool = True,
        scientific_score_dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Trainer-facing Policy interface with a permanent truth firewall."""

        # Accepted for one common trainer call signature. Distribution truth
        # and Representation are always detached; joint stages may explicitly
        # expose the canonical shared packet for projected shared gradients.
        del detach_distribution_truth, scientific_score_dtype
        return self.forward_decision(
            batch,
            origin=origin,
            planes=planes,
            limits=limits,
            issued_day_ahead=issued_day_ahead,
            detach_shared=detach_shared,
        )

    def forward_joint(
        self,
        batch: Mapping[str, Any],
        *,
        origin: int | Tensor = 0,
        sample_count: int | None = None,
        base_indices: Tensor | None = None,
        seed: int | None = None,
        limits: Mapping[str, Any] | None = None,
        issued_day_ahead: Tensor | None = None,
        detach_distribution_truth: bool = True,
        detach_shared: bool = True,
        scientific_score_dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """One-call R/D/P interface used by the V8 joint trainer."""

        del detach_distribution_truth, scientific_score_dtype
        distribution = self.forward_distribution(
            batch,
            sample_count,
            origin=origin,
            base_indices=base_indices,
            seed=seed,
        )
        return self.forward_decision(
            batch,
            origin=origin,
            planes=distribution,
            limits=limits,
            issued_day_ahead=issued_day_ahead,
            detach_shared=detach_shared,
        )

    def forward_issue(
        self,
        batch: Mapping[str, Any],
        *,
        limits: Mapping[str, Any] | None = None,
        sample_count: int | None = None,
        base_indices: Tensor | None = None,
        seed: int | None = None,
        detach_shared: bool = True,
    ) -> dict[str, Any]:
        issue_batch = dict(batch)
        if detach_shared:
            market = self._tensor(
                self._required(batch, "market", "market_features", "market_inputs")
            )
            with torch.no_grad():
                issue_batch["_v8_exogenous_cache"] = self._encode_exogenous(
                    batch, market
                )
                distribution = self.forward_distribution(
                    issue_batch,
                    sample_count,
                    origin=0,
                    base_indices=base_indices,
                    seed=seed,
                )
        else:
            distribution = self.forward_distribution(
                issue_batch,
                sample_count,
                origin=0,
                base_indices=base_indices,
                seed=seed,
            )
        decision = self.forward_decision(
            issue_batch,
            origin=0,
            planes=distribution,
            limits=limits,
            detach_shared=detach_shared,
        )
        policy: PolicyPacket = decision["policy"]
        return {
            **decision,
            "issue_context": {
                "_v8_issue_context": True,
                "batch": issue_batch,
                "issued_day_ahead": policy.day_ahead_plan.detach(),
                "detach_shared": bool(detach_shared),
            },
            "issued_day_ahead": policy.day_ahead_plan,
        }

    def forward_origin(
        self,
        issue_context: Mapping[str, Any],
        rolling_state: Any,
        origin: int | Tensor,
        *,
        limits: Mapping[str, Any] | None = None,
        issued_day_ahead: Tensor | None = None,
        detach_shared: bool | None = None,
    ) -> dict[str, Any]:
        context = issue_context.get("issue_context", issue_context)
        if not isinstance(context, Mapping) or not context.get("_v8_issue_context"):
            raise ValueError("forward_origin requires a V8 issue_context")
        raw_batch = context.get("batch")
        if not isinstance(raw_batch, Mapping):
            raise ValueError("V8 issue_context omitted its causal batch")
        batch = {**raw_batch, "rolling_state": rolling_state}
        frozen_da = issued_day_ahead
        if frozen_da is None:
            frozen_da = context.get("issued_day_ahead")
        resolved_detach_shared = (
            bool(context.get("detach_shared", True))
            if detach_shared is None
            else bool(detach_shared)
        )
        planes: Mapping[str, Any] | None = None
        if resolved_detach_shared:
            with torch.no_grad():
                planes = self.forward_planes(batch, origin)
        return self.forward_decision(
            batch,
            origin=origin,
            planes=planes,
            limits=limits,
            issued_day_ahead=frozen_da,
            detach_shared=resolved_detach_shared,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        origin: int | Tensor = 0,
        sample_count: int | None = None,
        base_indices: Tensor | None = None,
        seed: int | None = None,
        limits: Mapping[str, Any] | None = None,
        issued_day_ahead: Tensor | None = None,
    ) -> dict[str, Any]:
        return self.forward_joint(
            batch,
            origin=origin,
            sample_count=sample_count,
            base_indices=base_indices,
            seed=seed,
            limits=limits,
            issued_day_ahead=issued_day_ahead,
        )

    def parameter_groups(self) -> dict[str, tuple[str, ...]]:
        """Return six disjoint, exhaustive registered parameter-name groups."""

        groups: dict[str, list[str]] = {name: [] for name in PARAMETER_GROUPS}
        for name, _ in self.named_parameters():
            if name.startswith("physical_stream.physical_twin_core."):
                group = "frozen"
            elif ".router." in name or name.startswith("router."):
                group = "router"
            elif name.startswith("representation."):
                group = "representation"
            elif name.startswith("distribution."):
                group = "distribution_truth"
            elif name.startswith("decision_bridge.") or name.startswith("policy."):
                group = "decision"
            elif name.startswith(
                (
                    "market_encoder.",
                    "workload_encoder.",
                    "physical_stream.",
                    "state_encoder.",
                    "modality_projections.",
                    "trifusion.",
                )
            ):
                group = "shared"
            else:
                raise RuntimeError(f"unregistered V8 parameter: {name}")
            groups[group].append(name)
        result = {key: tuple(value) for key, value in groups.items()}
        flattened = [name for values in result.values() for name in values]
        available = [name for name, _ in self.named_parameters()]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(available):
            raise RuntimeError("V8 parameter groups are not disjoint and exhaustive")
        if tuple(result) != PARAMETER_GROUPS:
            raise RuntimeError("V8 parameter group order changed")
        return result

    def configure_trainable_groups(
        self, active_groups: Sequence[str]
    ) -> tuple[str, ...]:
        active = {str(value) for value in active_groups}
        unknown = active - set(PARAMETER_GROUPS)
        if unknown:
            raise ValueError(f"unknown V8 parameter groups: {sorted(unknown)}")
        active.discard("frozen")
        groups = self.parameter_groups()
        enabled = {
            name for group in active for name in groups[group]
        }
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in enabled)
        self.physical_stream.freeze_twin()
        return tuple(sorted(enabled))

    def distribution_truth_parameter_names(self) -> tuple[str, ...]:
        return self.parameter_groups()["distribution_truth"]

    def representation_parameter_names(self) -> tuple[str, ...]:
        return self.parameter_groups()["representation"]

    def decision_parameter_names(self) -> tuple[str, ...]:
        return self.parameter_groups()["decision"]

    def assert_decision_gradient_boundary(self) -> None:
        groups = self.parameter_groups()
        protected = (
            groups["shared"]
            + groups["representation"]
            + groups["distribution_truth"]
            + groups["frozen"]
        )
        parameters = dict(self.named_parameters())
        leaked = [
            name
            for name in protected
            if parameters[name].grad is not None
            and bool((parameters[name].grad.detach() != 0).any())
        ]
        if leaked:
            raise RuntimeError(
                "Policy gradient crossed the V8 truth boundary: " + ", ".join(leaked)
            )


R4V8Model = TriPlaneCausalWorldModelV8


__all__ = [
    "ARCHITECTURE_VERSION",
    "PARAMETER_GROUPS",
    "R4V8Model",
    "TriPlaneCausalWorldModelV8",
    "V8ModelConfig",
]
