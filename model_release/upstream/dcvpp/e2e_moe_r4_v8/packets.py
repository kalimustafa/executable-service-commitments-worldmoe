"""Typed tensor contracts for the R4-V8 three-plane architecture.

The probability plane models only 17 independent random channels.  DART is a
public, deterministic view reconstructed from DA and RT and therefore can
never become a third stochastic degree of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor


HORIZON = 96
MODALITY_DIM = 128
WORKSPACE_DIM = 192
INDEPENDENT_CHANNELS = 17
PUBLIC_CHANNELS = 21
ZONES = 4
DEPENDENCE_RANK = 4
REGIME_COUNT = 8


def close_public_channels(independent: Tensor) -> Tensor:
    """Map 17 independent channels to the public 21-channel identity space."""

    if independent.ndim < 2 or independent.shape[-1] != INDEPENDENT_CHANNELS:
        raise ValueError("independent channels must end in 17")
    da = independent[..., 0:4]
    rt = independent[..., 4:8]
    workload = independent[..., 8:9]
    pv = independent[..., 9:13]
    pue = independent[..., 13:17]
    dart = da - rt
    return torch.cat((da, rt, dart, workload, pv, pue), dim=-1)


def independent_from_public(public: Tensor) -> Tensor:
    """Drop deterministic DART while preserving the public channel order."""

    if public.ndim < 2 or public.shape[-1] != PUBLIC_CHANNELS:
        raise ValueError("public channels must end in 21")
    return torch.cat(
        (
            public[..., 0:4],
            public[..., 4:8],
            public[..., 12:13],
            public[..., 13:17],
            public[..., 17:21],
        ),
        dim=-1,
    )


def public_scale_from_independent(scale: Tensor) -> Tensor:
    """Build a positive public marginal-scale view from 17 truth scales."""

    if scale.shape[-1] != INDEPENDENT_CHANNELS:
        raise ValueError("independent scale must end in 17")
    da = scale[..., 0:4]
    rt = scale[..., 4:8]
    dart = torch.sqrt(da.square() + rt.square()).clamp_min(1.0e-8)
    return torch.cat(
        (da, rt, dart, scale[..., 8:9], scale[..., 9:13], scale[..., 13:17]),
        dim=-1,
    )


def _finite(name: str, value: Tensor, shape: tuple[int | None, ...]) -> None:
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class RepresentationAnchorPacket:
    """Representation-only causal candidate anchors.

    This packet is deliberately separate from :class:`CanonicalWorkspacePacket`.
    In particular, neither candidate anchor is permitted to replace the frozen
    issue-time anchor consumed by the Distribution plane.
    """

    issue_center_public: Tensor
    persistence_public: Tensor
    seasonal_public: Tensor
    availability: Tensor

    def validate(self) -> "RepresentationAnchorPacket":
        batch = self.issue_center_public.shape[0]
        for name, value in (
            ("issue_center_public", self.issue_center_public),
            ("persistence_public", self.persistence_public),
            ("seasonal_public", self.seasonal_public),
        ):
            _finite(
                f"representation_anchor.{name}",
                value,
                (batch, HORIZON, PUBLIC_CHANNELS),
            )
        if (
            self.availability.shape != (batch, 2)
            or self.availability.dtype is not torch.bool
        ):
            raise ValueError(
                "representation_anchor.availability must be bool [batch,2] "
                "ordered as persistence,seasonal"
            )
        return self

    def detached(self) -> "RepresentationAnchorPacket":
        return replace(
            self,
            issue_center_public=self.issue_center_public.detach(),
            persistence_public=self.persistence_public.detach(),
            seasonal_public=self.seasonal_public.detach(),
            availability=self.availability.detach(),
        )


@dataclass(frozen=True)
class CanonicalWorkspacePacket:
    """Shared causal workspace produced before the three task planes diverge."""

    time_tokens: Tensor
    modality_tokens: Tensor
    state_tokens: Tensor
    state_hidden: Tensor
    anchor_independent: Tensor
    origin_index: Tensor

    def validate(self) -> "CanonicalWorkspacePacket":
        batch = self.time_tokens.shape[0]
        _finite("time_tokens", self.time_tokens, (batch, HORIZON, WORKSPACE_DIM))
        _finite(
            "modality_tokens",
            self.modality_tokens,
            (batch, HORIZON, 4, WORKSPACE_DIM),
        )
        _finite("state_tokens", self.state_tokens, (batch, HORIZON, WORKSPACE_DIM))
        _finite("state_hidden", self.state_hidden, (batch, WORKSPACE_DIM))
        _finite(
            "anchor_independent",
            self.anchor_independent,
            (batch, HORIZON, INDEPENDENT_CHANNELS),
        )
        if self.origin_index.shape != (batch,):
            raise ValueError("origin_index must be [batch]")
        if self.origin_index.is_floating_point():
            raise ValueError("origin_index must be integral")
        if bool(((self.origin_index < 0) | (self.origin_index >= HORIZON)).any()):
            raise ValueError("origin_index must lie in [0,95]")
        return self

    def detached(self) -> "CanonicalWorkspacePacket":
        return replace(
            self,
            time_tokens=self.time_tokens.detach(),
            modality_tokens=self.modality_tokens.detach(),
            state_tokens=self.state_tokens.detach(),
            state_hidden=self.state_hidden.detach(),
            anchor_independent=self.anchor_independent.detach(),
        )


@dataclass(frozen=True)
class RepresentationPacket:
    """Semantic plane output; point forecasts are never a Distribution input."""

    time_tokens: Tensor
    point_independent: Tensor
    point_forecast: Tensor
    router_logits: Tensor
    router_probabilities: Tensor
    topk_indices: Tensor
    topk_weights: Tensor
    workload_zero_logits: Tensor | None = None
    workload_zero_probabilities: Tensor | None = None
    workload_positive_forecast: Tensor | None = None

    def validate(self) -> "RepresentationPacket":
        batch = self.time_tokens.shape[0]
        _finite(
            "representation.time_tokens",
            self.time_tokens,
            (batch, HORIZON, WORKSPACE_DIM),
        )
        _finite(
            "representation.point_independent",
            self.point_independent,
            (batch, HORIZON, INDEPENDENT_CHANNELS),
        )
        _finite(
            "representation.point_forecast",
            self.point_forecast,
            (batch, HORIZON, PUBLIC_CHANNELS),
        )
        _finite("representation.router_logits", self.router_logits, (batch, HORIZON, 8))
        _finite(
            "representation.router_probabilities",
            self.router_probabilities,
            (batch, HORIZON, 8),
        )
        if self.topk_indices.shape != (batch, HORIZON, 2):
            raise ValueError("representation.topk_indices must be [batch,96,2]")
        _finite("representation.topk_weights", self.topk_weights, (batch, HORIZON, 2))
        workload_auxiliary = (
            self.workload_zero_logits,
            self.workload_zero_probabilities,
            self.workload_positive_forecast,
        )
        if any(value is not None for value in workload_auxiliary):
            if not all(isinstance(value, Tensor) for value in workload_auxiliary):
                raise ValueError(
                    "Representation workload hurdle diagnostics must be all present or all absent"
                )
            logits = self.workload_zero_logits
            probabilities = self.workload_zero_probabilities
            positive = self.workload_positive_forecast
            assert isinstance(logits, Tensor)
            assert isinstance(probabilities, Tensor)
            assert isinstance(positive, Tensor)
            _finite("representation.workload_zero_logits", logits, (batch,))
            _finite(
                "representation.workload_zero_probabilities",
                probabilities,
                (batch,),
            )
            _finite(
                "representation.workload_positive_forecast",
                positive,
                (batch, HORIZON, 1),
            )
            if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
                raise ValueError(
                    "representation.workload_zero_probabilities must lie in [0,1]"
                )
            if bool((positive < 0.0).any()):
                raise ValueError(
                    "representation.workload_positive_forecast must be nonnegative"
                )
        return self

    def detached(self) -> "RepresentationPacket":
        return replace(
            self,
            time_tokens=self.time_tokens.detach(),
            point_independent=self.point_independent.detach(),
            point_forecast=self.point_forecast.detach(),
            router_logits=self.router_logits.detach(),
            router_probabilities=self.router_probabilities.detach(),
            topk_indices=self.topk_indices.detach(),
            topk_weights=self.topk_weights.detach(),
            workload_zero_logits=(
                None
                if self.workload_zero_logits is None
                else self.workload_zero_logits.detach()
            ),
            workload_zero_probabilities=(
                None
                if self.workload_zero_probabilities is None
                else self.workload_zero_probabilities.detach()
            ),
            workload_positive_forecast=(
                None
                if self.workload_positive_forecast is None
                else self.workload_positive_forecast.detach()
            ),
        )


@dataclass(frozen=True)
class DistributionTruthPacket:
    """Calibrated truth memory with no materialized Monte-Carlo dimension."""

    time_tokens: Tensor
    factor_tokens: Tensor
    regime_tokens: Tensor
    anchor_independent: Tensor
    center_independent: Tensor
    location_independent: Tensor
    scale_independent: Tensor
    tail_independent: Tensor
    factor_loadings: Tensor
    regime_probabilities: Tensor
    regime_shift: Tensor
    origin_index: Tensor
    router_logits: Tensor
    router_probabilities: Tensor
    topk_indices: Tensor
    topk_weights: Tensor

    @property
    def center_public(self) -> Tensor:
        return close_public_channels(self.center_independent)

    @property
    def scale_public(self) -> Tensor:
        return public_scale_from_independent(self.scale_independent)

    @property
    def tail_public(self) -> Tensor:
        return close_public_channels(self.tail_independent)

    @property
    def stats_public(self) -> Tensor:
        return torch.cat(
            (self.center_public, self.scale_public, self.tail_public), dim=-1
        )

    def validate(self) -> "DistributionTruthPacket":
        batch = self.time_tokens.shape[0]
        _finite("truth.time_tokens", self.time_tokens, (batch, HORIZON, WORKSPACE_DIM))
        _finite(
            "truth.factor_tokens",
            self.factor_tokens,
            (batch, DEPENDENCE_RANK, WORKSPACE_DIM),
        )
        _finite(
            "truth.regime_tokens",
            self.regime_tokens,
            (batch, REGIME_COUNT, WORKSPACE_DIM),
        )
        for name, value in (
            ("anchor_independent", self.anchor_independent),
            ("center_independent", self.center_independent),
            ("location_independent", self.location_independent),
            ("scale_independent", self.scale_independent),
            ("tail_independent", self.tail_independent),
        ):
            _finite(f"truth.{name}", value, (batch, HORIZON, INDEPENDENT_CHANNELS))
        if bool((self.scale_independent <= 0).any()):
            raise ValueError("truth.scale_independent must be positive")
        _finite(
            "truth.factor_loadings",
            self.factor_loadings,
            (batch, HORIZON, INDEPENDENT_CHANNELS, DEPENDENCE_RANK),
        )
        _finite(
            "truth.regime_probabilities",
            self.regime_probabilities,
            (batch, REGIME_COUNT),
        )
        _finite("truth.regime_shift", self.regime_shift, (batch, INDEPENDENT_CHANNELS))
        if self.origin_index.shape != (batch,):
            raise ValueError("truth.origin_index must be [batch]")
        _finite("truth.router_logits", self.router_logits, (batch, HORIZON, 8))
        _finite(
            "truth.router_probabilities",
            self.router_probabilities,
            (batch, HORIZON, 8),
        )
        if self.topk_indices.shape != (batch, HORIZON, 2):
            raise ValueError("truth.topk_indices must be [batch,96,2]")
        _finite("truth.topk_weights", self.topk_weights, (batch, HORIZON, 2))
        if self.center_public.shape != (batch, HORIZON, PUBLIC_CHANNELS):
            raise AssertionError("truth public closure is malformed")
        return self

    def detached(self) -> "DistributionTruthPacket":
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = value.detach() if isinstance(value, Tensor) else value
        return replace(self, **values)


@dataclass(frozen=True)
class DecisionWorkspacePacket:
    """Decision-owned writable view constructed only from detached truth inputs."""

    time_tokens: Tensor
    structure_tokens: Tensor
    stats_public: Tensor
    state_hidden: Tensor
    origin_index: Tensor
    refinement_round: int = 0

    def validate(self) -> "DecisionWorkspacePacket":
        batch = self.time_tokens.shape[0]
        _finite(
            "decision.time_tokens", self.time_tokens, (batch, HORIZON, WORKSPACE_DIM)
        )
        _finite(
            "decision.structure_tokens",
            self.structure_tokens,
            (batch, None, WORKSPACE_DIM),
        )
        _finite(
            "decision.stats_public",
            self.stats_public,
            (batch, HORIZON, 3 * PUBLIC_CHANNELS),
        )
        _finite("decision.state_hidden", self.state_hidden, (batch, WORKSPACE_DIM))
        if self.origin_index.shape != (batch,):
            raise ValueError("decision.origin_index must be [batch]")
        if int(self.refinement_round) not in {0, 1, 2}:
            raise ValueError("decision.refinement_round must lie in [0,2]")
        return self


@dataclass(frozen=True)
class PolicyPacket:
    day_ahead_plan: Tensor
    battery_plan: Tensor
    compute_plan: Tensor
    current_battery: Tensor
    current_compute: Tensor
    critic: Tensor
    plan_tokens: Tensor
    origin_index: Tensor
    refinement_trace: tuple[Tensor, ...]
    router_diagnostics: tuple[dict[str, Tensor], ...]

    def validate(self) -> "PolicyPacket":
        batch = self.plan_tokens.shape[0]
        for name, value in (
            ("day_ahead_plan", self.day_ahead_plan),
            ("battery_plan", self.battery_plan),
            ("compute_plan", self.compute_plan),
        ):
            _finite(f"policy.{name}", value, (batch, HORIZON, ZONES))
        _finite("policy.current_battery", self.current_battery, (batch, ZONES))
        _finite("policy.current_compute", self.current_compute, (batch, ZONES))
        _finite("policy.critic", self.critic, (batch, 5))
        _finite("policy.plan_tokens", self.plan_tokens, (batch, HORIZON, WORKSPACE_DIM))
        if self.origin_index.shape != (batch,):
            raise ValueError("policy.origin_index must be [batch]")
        if len(self.refinement_trace) != 2 or len(self.router_diagnostics) != 2:
            raise ValueError("V8 Policy requires exactly two refinement rounds")
        return self


@dataclass(frozen=True)
class DistributionSamplePacket:
    independent_samples: Tensor
    public_samples: Tensor
    weights: Tensor
    base_indices: Tensor

    def validate(self) -> "DistributionSamplePacket":
        batch, count = self.weights.shape
        _finite(
            "samples.independent",
            self.independent_samples,
            (batch, count, HORIZON, INDEPENDENT_CHANNELS),
        )
        _finite(
            "samples.public",
            self.public_samples,
            (batch, count, HORIZON, PUBLIC_CHANNELS),
        )
        _finite("samples.weights", self.weights, (batch, count))
        if self.base_indices.shape != (batch, count):
            raise ValueError("samples.base_indices must be [batch,N]")
        identity = (
            self.public_samples[..., 0:4]
            - self.public_samples[..., 4:8]
            - self.public_samples[..., 8:12]
        )
        if not bool(torch.equal(identity, torch.zeros_like(identity))):
            raise ValueError("public samples violate exact DA-RT-DART closure")
        return self


__all__ = [
    "CanonicalWorkspacePacket",
    "DEPENDENCE_RANK",
    "DecisionWorkspacePacket",
    "DistributionSamplePacket",
    "DistributionTruthPacket",
    "HORIZON",
    "INDEPENDENT_CHANNELS",
    "MODALITY_DIM",
    "PUBLIC_CHANNELS",
    "PolicyPacket",
    "REGIME_COUNT",
    "RepresentationAnchorPacket",
    "RepresentationPacket",
    "WORKSPACE_DIM",
    "ZONES",
    "close_public_channels",
    "independent_from_public",
    "public_scale_from_independent",
]
