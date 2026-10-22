"""Baseline-anchored conditional distribution for the independent R4-V7 lineage.

The formal baseline is an empirical distribution of whole-day, 21-channel
residual paths.  V7 does not relearn that distribution from random weights.
Instead it starts as the exact empirical baseline and learns only bounded,
state-conditional transports.  The module has no fixed Monte-Carlo axis;
``sample`` materializes any requested number of paths from the frozen anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import FrozenResidualAnchor


HORIZON = 96
CHANNELS = 21
ZONES = 4


def _zero_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _origin_index(
    origin: int | Tensor, batch: int, reference: Tensor, horizon: int = HORIZON
) -> Tensor:
    value = torch.as_tensor(origin, device=reference.device)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,):
        raise ValueError("origin must be a scalar or [batch]")
    if value.is_floating_point() and not bool(torch.equal(value, value.round())):
        raise ValueError("origin must contain integer indices")
    value = value.long()
    if bool(((value < 0) | (value >= horizon)).any()):
        raise ValueError(f"origin must lie in [0,{horizon - 1}]")
    return value


@dataclass(frozen=True)
class StructuredDistributionMemory:
    """Time-preserving world memory read by the V7 decision model.

    ``center_zonal``, ``scale_zonal`` and ``tail_zonal`` remain explicit rather
    than being hidden in a single pooled latent.  This makes it impossible for
    the Policy interface to silently fall back to a day-level risk average.
    """

    time_tokens: Tensor
    factor_tokens: Tensor
    regime_tokens: Tensor
    center_zonal: Tensor
    scale_zonal: Tensor
    tail_zonal: Tensor
    factor_loadings: Tensor
    regime_probabilities: Tensor
    origin_index: Tensor
    decision_adapted: bool = False

    def validate(self) -> "StructuredDistributionMemory":
        batch = self.time_tokens.shape[0]
        if self.time_tokens.shape != (batch, HORIZON, 128):
            raise ValueError("time_tokens must be [batch,96,128]")
        for name, value in (
            ("center_zonal", self.center_zonal),
            ("scale_zonal", self.scale_zonal),
            ("tail_zonal", self.tail_zonal),
        ):
            if value.shape != (batch, HORIZON, CHANNELS):
                raise ValueError(f"{name} must be [batch,96,21]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if self.factor_tokens.ndim != 3 or self.factor_tokens.shape[0] != batch:
            raise ValueError("factor_tokens must be [batch,rank,128]")
        if self.regime_tokens.ndim != 3 or self.regime_tokens.shape[0] != batch:
            raise ValueError("regime_tokens must be [batch,regimes,128]")
        rank = self.factor_tokens.shape[1]
        if self.factor_loadings.shape != (batch, HORIZON, CHANNELS, rank):
            raise ValueError("factor_loadings must be [batch,96,21,rank]")
        regimes = self.regime_tokens.shape[1]
        if self.regime_probabilities.shape != (batch, regimes):
            raise ValueError("regime_probabilities must be [batch,regimes]")
        if self.origin_index.shape != (batch,):
            raise ValueError("origin_index must be [batch]")
        return self

    def detached(self) -> "StructuredDistributionMemory":
        return replace(
            self,
            time_tokens=self.time_tokens.detach(),
            factor_tokens=self.factor_tokens.detach(),
            regime_tokens=self.regime_tokens.detach(),
            center_zonal=self.center_zonal.detach(),
            scale_zonal=self.scale_zonal.detach(),
            tail_zonal=self.tail_zonal.detach(),
            factor_loadings=self.factor_loadings.detach(),
            regime_probabilities=self.regime_probabilities.detach(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_tokens": self.time_tokens,
            "factor_tokens": self.factor_tokens,
            "regime_tokens": self.regime_tokens,
            "center": self.center_zonal,
            "center_zonal": self.center_zonal,
            "scale": self.scale_zonal,
            "scale_zonal": self.scale_zonal,
            "tail": self.tail_zonal,
            "tail_zonal": self.tail_zonal,
            "factor_loadings": self.factor_loadings,
            "regime_probabilities": self.regime_probabilities,
            "origin_index": self.origin_index,
            "decision_adapted": self.decision_adapted,
            "sampling_axis_present": False,
        }


@dataclass(frozen=True)
class BaselineAnchoredWorld:
    """A conditional distribution with no materialized scenario dimension."""

    memory: StructuredDistributionMemory
    anchor_center_zonal: Tensor
    representation_center_zonal: Tensor
    center_zonal: Tensor
    transport_location: Tensor
    transport_log_scale: Tensor
    transport_tail: Tensor
    transport_cap_zonal: Tensor
    factor_loadings: Tensor
    regime_probabilities: Tensor
    regime_shift: Tensor
    source_path: str

    @property
    def token_features(self) -> Tensor:
        return self.memory.time_tokens

    @property
    def origin_index(self) -> Tensor:
        return self.memory.origin_index

    @property
    def has_sampling_axis(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "world": self,
            "world_latent": self,
            "memory": self.memory,
            "anchor_center": self.anchor_center_zonal,
            "anchor_center_zonal": self.anchor_center_zonal,
            "representation_center": self.representation_center_zonal,
            "representation_center_zonal": self.representation_center_zonal,
            "center": self.center_zonal,
            "center_zonal": self.center_zonal,
            "scale": self.memory.scale_zonal,
            "scale_zonal": self.memory.scale_zonal,
            "tail": self.memory.tail_zonal,
            "tail_zonal": self.memory.tail_zonal,
            "factor_tokens": self.memory.factor_tokens,
            "regime_tokens": self.memory.regime_tokens,
            "factor_loadings": self.factor_loadings,
            "regime_probabilities": self.regime_probabilities,
            "origin_index": self.origin_index,
            "sampling_axis_present": False,
            "fixed_particle_capacity": False,
            "baseline_anchor_source": self.source_path,
            "transport_contract": "frozen_empirical_anchor_plus_bounded_conditional_transport",
        }


class DistributionDecisionAdapter(nn.Module):
    """Decision-only adapter, deliberately separated from truth parameters."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.time_delta = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.factor_norm = nn.LayerNorm(hidden_dim)
        self.factor_delta = nn.Linear(hidden_dim, hidden_dim)
        self.regime_norm = nn.LayerNorm(hidden_dim)
        self.regime_delta = nn.Linear(hidden_dim, hidden_dim)
        # A1 has one narrow decision bridge.  It consumes only the four DART
        # center/scale/tail channels and cannot mutate the truth distribution.
        self.dart_bridge = nn.Sequential(
            nn.LayerNorm(3 * ZONES),
            nn.Linear(3 * ZONES, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        _zero_linear(self.time_delta[-1])
        _zero_linear(self.factor_delta)
        _zero_linear(self.regime_delta)
        _zero_linear(self.dart_bridge[-1])

    def forward(
        self, memory: StructuredDistributionMemory, *, detach_truth: bool = True
    ) -> StructuredDistributionMemory:
        source = memory.detached() if detach_truth else memory
        dart_features = torch.cat(
            (
                source.center_zonal[..., 8:12],
                source.scale_zonal[..., 8:12],
                source.tail_zonal[..., 8:12],
            ),
            dim=-1,
        )
        adapted = replace(
            source,
            time_tokens=source.time_tokens
            + 0.1 * torch.tanh(self.time_delta(self.time_norm(source.time_tokens)))
            + 0.1 * torch.tanh(self.dart_bridge(dart_features)),
            factor_tokens=source.factor_tokens
            + 0.1 * torch.tanh(self.factor_delta(self.factor_norm(source.factor_tokens))),
            regime_tokens=source.regime_tokens
            + 0.1 * torch.tanh(self.regime_delta(self.regime_norm(source.regime_tokens))),
            decision_adapted=True,
        )
        return adapted.validate()


class BaselineAnchoredConditionalTransport(nn.Module):
    """Structured empirical world model initialized as the frozen baseline.

    Parameters prefixed with ``truth_`` define the calibrated distribution.
    Policy loss must not update them.  ``decision_adapter`` is the only module
    in this class that belongs to the decision optimizer group.
    """

    def __init__(
        self,
        anchor: FrozenResidualAnchor,
        *,
        hidden_dim: int = 128,
        dependence_rank: int = 4,
        regime_count: int = 8,
        transport_cap_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("V7 StructuredDistributionMemory fixes hidden_dim=128")
        residuals = torch.as_tensor(anchor.residuals, dtype=torch.float32)
        scale = torch.as_tensor(anchor.channel_scale, dtype=torch.float32)
        if residuals.ndim != 3 or residuals.shape[1:] != (HORIZON, CHANNELS):
            raise ValueError("anchor residuals must be [days,96,21]")
        if scale.shape != (CHANNELS,) or bool((scale <= 0).any()):
            raise ValueError("anchor channel_scale must contain 21 positive values")
        if not bool(torch.isfinite(residuals).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("anchor residuals and scale must be finite")
        if dependence_rank < 1 or regime_count < 2:
            raise ValueError("V7 requires positive dependence rank and at least two regimes")
        if not 0.0 < float(transport_cap_fraction) <= 2.0:
            raise ValueError("transport_cap_fraction must lie in (0,2]")
        self.hidden_dim = int(hidden_dim)
        self.dependence_rank = int(dependence_rank)
        self.regime_count = int(regime_count)
        self.transport_cap_fraction = float(transport_cap_fraction)
        self.anchor_day_count = int(residuals.shape[0])
        self.anchor_source_path = str(Path(anchor.source_path))
        # The empirical baseline is immutable data, not a learned checkpoint
        # parameter.  It is reconstructed from the pinned artifact on resume.
        self.register_buffer("baseline_residuals", residuals, persistent=False)
        self.register_buffer("baseline_channel_scale", scale, persistent=False)
        channel_axis = torch.arange(1, CHANNELS + 1, dtype=torch.float32)[:, None]
        factor_axis = torch.arange(
            1, self.dependence_rank + 1, dtype=torch.float32
        )[None]
        projection = torch.sin(channel_axis * factor_axis) + torch.cos(
            channel_axis * (factor_axis + 0.5)
        )
        projection = projection / projection.square().sum(dim=0, keepdim=True).sqrt()
        self.register_buffer(
            "baseline_factor_projection", projection, persistent=True
        )
        stat_axis = torch.arange(1, 3 * CHANNELS + 1, dtype=torch.float32)[:, None]
        hidden_axis = torch.arange(1, hidden_dim + 1, dtype=torch.float32)[None]
        stat_projection = torch.sin(stat_axis * hidden_axis / hidden_dim)
        stat_projection = stat_projection / stat_projection.square().sum(
            dim=0, keepdim=True
        ).sqrt().clamp_min(1.0e-8)
        self.register_buffer(
            "structured_stat_projection", stat_projection, persistent=True
        )
        factor_basis = torch.sin(
            torch.arange(1, self.dependence_rank + 1, dtype=torch.float32)[:, None]
            * hidden_axis
            / hidden_dim
        )
        regime_basis = torch.cos(
            torch.arange(1, self.regime_count + 1, dtype=torch.float32)[:, None]
            * hidden_axis
            / hidden_dim
        )
        self.register_buffer("factor_token_basis", 0.01 * factor_basis, persistent=True)
        self.register_buffer("regime_token_basis", 0.01 * regime_basis, persistent=True)

        # Truth/calibration path.  Every head that can alter a sampled value is
        # zero initialized, making epoch zero the exact empirical baseline.
        self.truth_condition = nn.Sequential(
            nn.LayerNorm(hidden_dim + 2 * CHANNELS),
            nn.Linear(hidden_dim + 2 * CHANNELS, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.truth_center_head = nn.Linear(hidden_dim, CHANNELS)
        self.truth_location_head = nn.Linear(hidden_dim, CHANNELS)
        self.truth_dart_adapter = nn.Linear(hidden_dim, ZONES)
        self.truth_log_scale_head = nn.Linear(hidden_dim, CHANNELS)
        self.truth_tail_head = nn.Linear(hidden_dim, CHANNELS)
        self.truth_factor_head = nn.Linear(
            hidden_dim, CHANNELS * self.dependence_rank
        )
        self.truth_regime_head = nn.Linear(hidden_dim, self.regime_count)
        for head in (
            self.truth_center_head,
            self.truth_location_head,
            self.truth_dart_adapter,
            self.truth_log_scale_head,
            self.truth_tail_head,
            self.truth_factor_head,
            self.truth_regime_head,
        ):
            _zero_linear(head)
        self.truth_regime_shift = nn.Parameter(
            torch.zeros(self.regime_count, CHANNELS)
        )
        self.truth_regime_tail = nn.Parameter(
            torch.zeros(self.regime_count, CHANNELS)
        )
        self.decision_adapter = DistributionDecisionAdapter(hidden_dim)

    def _zonal(self, value: Tensor, name: str, reference: Tensor) -> Tensor:
        result = torch.as_tensor(
            value, device=reference.device, dtype=reference.dtype
        )
        expected = (reference.shape[0], HORIZON, CHANNELS)
        if result.shape != expected:
            raise ValueError(f"{name} must be {expected}")
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} contains non-finite values")
        return result

    def forward(
        self,
        token_features: Tensor,
        anchor_center_zonal: Tensor,
        *,
        representation_center_zonal: Tensor | None = None,
        origin: int | Tensor = 0,
    ) -> BaselineAnchoredWorld:
        if token_features.ndim != 3 or token_features.shape[1:] != (
            HORIZON,
            self.hidden_dim,
        ):
            raise ValueError("token_features must be [batch,96,128]")
        anchor_center = self._zonal(
            anchor_center_zonal, "anchor_center_zonal", token_features
        )
        proposal = (
            anchor_center
            if representation_center_zonal is None
            else self._zonal(
                representation_center_zonal,
                "representation_center_zonal",
                token_features,
            )
        )
        scale = self.baseline_channel_scale.to(
            device=token_features.device, dtype=token_features.dtype
        ).reshape(1, 1, CHANNELS)
        normalized_anchor = anchor_center / scale
        normalized_gap = (proposal - anchor_center) / scale
        condition = self.truth_condition(
            torch.cat((token_features, normalized_anchor, normalized_gap), dim=-1)
        )
        cap = scale.expand(anchor_center.shape) * self.transport_cap_fraction
        center_delta = cap * torch.tanh(self.truth_center_head(condition))
        center = anchor_center + center_delta
        generic_location = self.truth_location_head(condition)
        location = torch.cat(
            (
                generic_location[..., :8],
                generic_location[..., 8:12] + self.truth_dart_adapter(condition),
                generic_location[..., 12:],
            ),
            dim=-1,
        )
        log_scale = 0.75 * torch.tanh(self.truth_log_scale_head(condition))
        raw_tail = 0.5 * torch.tanh(self.truth_tail_head(condition))
        factor_loadings = 0.5 * torch.tanh(
            self.truth_factor_head(condition).reshape(
                token_features.shape[0],
                HORIZON,
                CHANNELS,
                self.dependence_rank,
            )
        )
        regime_logits = self.truth_regime_head(condition.mean(dim=1))
        regime_probabilities = torch.softmax(regime_logits, dim=-1)
        regime_shift = regime_probabilities @ self.truth_regime_shift
        regime_tail = regime_probabilities @ self.truth_regime_tail
        tail = raw_tail + 0.25 * torch.tanh(regime_tail[:, None])

        memory_stats = torch.cat(
            (
                center / scale,
                log_scale.exp().expand_as(center),
                tail,
            ),
            dim=-1,
        )
        stat_projection = self.structured_stat_projection.to(
            device=token_features.device, dtype=token_features.dtype
        )
        time_tokens = F.layer_norm(
            token_features + 0.05 * torch.matmul(memory_stats, stat_projection),
            (self.hidden_dim,),
        )
        loading_strength = factor_loadings.abs().mean(dim=2).transpose(1, 2)
        loading_weights = torch.softmax(loading_strength, dim=-1)
        factor_tokens = torch.einsum(
            "brt,btd->brd", loading_weights, time_tokens
        ) + self.factor_token_basis.to(
            device=time_tokens.device, dtype=time_tokens.dtype
        )[None]
        pooled = time_tokens.mean(dim=1, keepdim=True)
        regime_tokens = (
            self.regime_token_basis.to(
                device=time_tokens.device, dtype=time_tokens.dtype
            )[None]
            + regime_probabilities[..., None] * pooled
        )
        origin_index = _origin_index(
            origin, token_features.shape[0], token_features, HORIZON
        )
        memory = StructuredDistributionMemory(
            time_tokens=time_tokens,
            factor_tokens=factor_tokens,
            regime_tokens=regime_tokens,
            center_zonal=center,
            scale_zonal=scale.expand_as(center) * log_scale.exp(),
            tail_zonal=tail,
            factor_loadings=factor_loadings,
            regime_probabilities=regime_probabilities,
            origin_index=origin_index,
        ).validate()
        return BaselineAnchoredWorld(
            memory=memory,
            anchor_center_zonal=anchor_center,
            representation_center_zonal=proposal,
            center_zonal=center,
            transport_location=location,
            transport_log_scale=log_scale,
            transport_tail=tail,
            transport_cap_zonal=cap,
            factor_loadings=factor_loadings,
            regime_probabilities=regime_probabilities,
            regime_shift=regime_shift,
            source_path=self.anchor_source_path,
        )

    def decision_memory(
        self, world: BaselineAnchoredWorld, *, detach_truth: bool = True
    ) -> StructuredDistributionMemory:
        return self.decision_adapter(
            world.memory, detach_truth=bool(detach_truth)
        )

    def _indices(
        self,
        sample_count: int,
        batch: int,
        reference: Tensor,
        *,
        base_indices: Tensor | None,
        generator: torch.Generator | None,
        seed: int | None,
    ) -> Tensor:
        count = int(sample_count)
        if count < 1:
            raise ValueError("sample_count must be positive")
        if base_indices is not None:
            indices = torch.as_tensor(base_indices, device=reference.device)
            if indices.ndim == 1:
                indices = indices.unsqueeze(0).expand(batch, -1)
            if indices.shape != (batch, count):
                raise ValueError("base_indices must be [N] or [batch,N]")
            if indices.is_floating_point() and not bool(
                torch.equal(indices, indices.round())
            ):
                raise ValueError("base_indices must contain integers")
            indices = indices.long()
        else:
            if generator is not None and seed is not None:
                raise ValueError("provide generator or seed, not both")
            cpu_generator = generator
            if cpu_generator is None:
                cpu_generator = torch.Generator(device="cpu")
                if seed is not None:
                    cpu_generator.manual_seed(int(seed))
                else:
                    cpu_generator.manual_seed(torch.seed())
            if count <= self.anchor_day_count:
                shared = torch.randperm(
                    self.anchor_day_count, generator=cpu_generator, device="cpu"
                )[:count]
            else:
                shared = torch.randint(
                    self.anchor_day_count,
                    (count,),
                    generator=cpu_generator,
                    device="cpu",
                )
            indices = shared.to(reference.device)[None].expand(batch, -1)
        if bool(((indices < 0) | (indices >= self.anchor_day_count)).any()):
            raise ValueError("base_indices fall outside the frozen anchor")
        return indices

    def _factor_noise(
        self,
        noise: Tensor | None,
        world: BaselineAnchoredWorld,
        count: int,
        standardized_base: Tensor,
    ) -> Tensor:
        reference = world.center_zonal
        batch = reference.shape[0]
        if noise is None:
            # Whole-day anchor paths already contain empirical dependence.
            # Fixed factor scores extracted from those paths let conditional
            # loadings reshape dependence without injecting independent noise.
            projection = self.baseline_factor_projection.to(
                device=reference.device, dtype=reference.dtype
            )
            factor_scores = torch.einsum(
                "bntc,cr->bntr", standardized_base, projection
            )
            return torch.einsum(
                "bntr,btcr->bntc", factor_scores, world.factor_loadings
            )
        value = torch.as_tensor(noise, device=reference.device, dtype=reference.dtype)
        if value.ndim == 3:
            value = value.unsqueeze(0).expand(batch, -1, -1, -1)
        if value.shape == (batch, count, HORIZON, CHANNELS):
            return value
        if value.shape == (batch, count, HORIZON, self.dependence_rank):
            return torch.einsum(
                "bntr,btcr->bntc", value, world.factor_loadings
            )
        raise ValueError(
            "noise must be [N,96,rank/channels] or [batch,N,96,rank/channels]"
        )

    @staticmethod
    def _close_identity(
        samples: Tensor, *, dart_increment: Tensor | None = None
    ) -> Tensor:
        da = samples[..., 0:4]
        rt = samples[..., 4:8]
        if dart_increment is not None:
            # DART-specific learning must not be discarded by the closure.
            # Split its incremental change symmetrically across DA and RT;
            # epoch-zero increment is exactly zero, preserving the anchor.
            da = da + 0.5 * dart_increment
            rt = rt - 0.5 * dart_increment
        # DART is a deterministic identity, never a third independent public
        # degree of freedom.  Recomputing it last gives exact tensor equality.
        dart = da - rt
        return torch.cat((da, rt, dart, samples[..., 12:]), dim=-1)

    def sample(
        self,
        world: BaselineAnchoredWorld,
        sample_count: int,
        *,
        base_indices: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        revealed_zonal: Tensor | None = None,
        revealed_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        count = int(sample_count)
        batch = world.center_zonal.shape[0]
        indices = self._indices(
            count,
            batch,
            world.center_zonal,
            base_indices=base_indices,
            generator=generator,
            seed=seed,
        )
        residual_table = self.baseline_residuals.to(
            device=world.center_zonal.device, dtype=world.center_zonal.dtype
        )
        base = residual_table[indices]
        channel_scale = self.baseline_channel_scale.to(
            device=base.device, dtype=base.dtype
        ).reshape(1, 1, 1, CHANNELS)
        standardized = base / channel_scale
        relative_scale = world.transport_log_scale.exp()[:, None]
        regime_shift = world.regime_shift[:, None, None, :]
        raw_delta = (
            world.transport_location[:, None]
            + (relative_scale - 1.0) * standardized
            + world.transport_tail[:, None]
            * standardized
            * standardized.abs()
            + regime_shift
            + self._factor_noise(noise, world, count, standardized)
        )
        cap_standardized = (
            world.transport_cap_zonal[:, None] / channel_scale
        ).clamp_min(1.0e-6)
        bounded_delta = cap_standardized * torch.tanh(
            raw_delta / cap_standardized
        )
        transported_residual = base + channel_scale * bounded_delta
        samples = world.center_zonal[:, None] + transported_residual
        baseline_samples = world.anchor_center_zonal[:, None] + base
        dart_increment = (
            samples[..., 8:12] - baseline_samples[..., 8:12]
        )
        samples = self._close_identity(samples, dart_increment=dart_increment)

        if revealed_zonal is not None:
            revealed = self._zonal(
                revealed_zonal, "revealed_zonal", world.center_zonal
            )
            if revealed_mask is None:
                raise ValueError("revealed_mask is required with revealed_zonal")
            mask = torch.as_tensor(revealed_mask, device=samples.device).bool()
            if mask.shape != (batch, HORIZON):
                raise ValueError("revealed_mask must be [batch,96]")
            periods = torch.arange(HORIZON, device=samples.device)[None]
            causal_mask = mask & (periods <= world.origin_index[:, None])
            samples = torch.where(
                causal_mask[:, None, :, None], revealed[:, None], samples
            )
            samples = self._close_identity(samples)

        weights = samples.new_full((batch, count), 1.0 / float(count))
        return {
            "samples": samples,
            "samples_zonal": samples,
            "particles_zonal": samples,
            "weights": weights,
            "base_indices": indices,
            "regime_probabilities": world.regime_probabilities,
            "sample_count": count,
            "sampling_axis_present": True,
            "fixed_particle_capacity": False,
            "transport_is_bounded": True,
            "baseline_identity_at_zero_initialization": True,
        }


__all__ = [
    "BaselineAnchoredConditionalTransport",
    "BaselineAnchoredWorld",
    "DistributionDecisionAdapter",
    "StructuredDistributionMemory",
]
