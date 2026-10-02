"""Continuous conditional world distribution for the R4/V6 lineage.

The learned representation in this module has no particle axis.  Monte Carlo
samples are materialized only when :meth:`DistributionHead.sample` is called,
so changing the sampling budget cannot alter the checkpoint shape, the world
latent, or the deterministic policy graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DISTRIBUTIONAL_ARCHITECTURE_REVISION = "continuous_distributional_world_r4_v6"
HORIZON = 96
ZONAL_VARIABLE_COUNT = 21
FREE_VARIABLE_COUNT = 14

ZONAL_VARIABLE_SLICES = {
    "da": slice(0, 4),
    "rt": slice(4, 8),
    "dart": slice(8, 12),
    "workload": slice(12, 13),
    "pv": slice(13, 17),
    "pue": slice(17, 21),
}

# Compatibility-only raw-unit envelope.  A formal V6 run installs the frozen
# train-only support contract before Distribution training starts.
_FALLBACK_RAW_UNIT_CAPS_ZONAL = (
    *(10.0 for _ in range(4)),
    *(20.0 for _ in range(4)),
    *(10.0 for _ in range(4)),
    10.0,
    *(10.0 for _ in range(4)),
    *(0.1 for _ in range(4)),
)


def zonal_to_legacy(values: Tensor) -> Tensor:
    """Convert a final-width-21 tensor to the six-variable aggregate view."""

    if values.shape[-1] != ZONAL_VARIABLE_COUNT:
        raise ValueError("zonal tensor must end in 21 channels")
    return torch.cat(
        (
            values[..., 0:4].mean(dim=-1, keepdim=True),
            values[..., 4:8].mean(dim=-1, keepdim=True),
            values[..., 8:12].mean(dim=-1, keepdim=True),
            values[..., 12:13],
            values[..., 13:17].mean(dim=-1, keepdim=True),
            values[..., 17:21].mean(dim=-1, keepdim=True),
        ),
        dim=-1,
    )


def fp_safe_close_da_rt_dart(
    da: Tensor,
    dart_candidate: Tensor,
    *,
    rt_lower: Tensor,
    rt_upper: Tensor,
    dart_lower: Tensor,
    dart_upper: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return RT/DART with exact public closure and strict fp support.

    ``DART`` is always assigned from the exact audit expression ``DA - RT``.
    At an fp32 support face that subtraction can round one ULP outside the
    declared DART interval, so RT is moved a bounded number of representable
    steps toward the joint support interior.  This operation is shared by the
    learned sampler and the frozen formal baseline.
    """

    rt = torch.clamp(da - dart_candidate, min=rt_lower, max=rt_upper)
    negative_infinity = torch.full_like(rt, -torch.inf)
    positive_infinity = torch.full_like(rt, torch.inf)
    for _ in range(4):
        reclosed = da - rt
        rt = torch.where(
            reclosed < dart_lower,
            torch.nextafter(rt, negative_infinity),
            rt,
        )
        rt = torch.where(
            reclosed > dart_upper,
            torch.nextafter(rt, positive_infinity),
            rt,
        )
        rt = torch.clamp(rt, min=rt_lower, max=rt_upper)
    return rt, da - rt


@dataclass(frozen=True)
class DistributionalWorldLatent:
    """N-independent conditional representation of the complete 96-step world."""

    center_zonal: Tensor
    scale_zonal: Tensor
    residual_caps_zonal: Tensor
    support_lower_zonal: Tensor
    support_upper_zonal: Tensor
    scale_fraction_free: Tensor
    skew_free: Tensor
    tail_free: Tensor
    dependence_loadings_free: Tensor
    quantile_offsets_free: Tensor
    regime_logits: Tensor
    token_features: Tensor
    origin_index: Tensor
    architecture_revision: str = DISTRIBUTIONAL_ARCHITECTURE_REVISION

    @property
    def batch_size(self) -> int:
        return int(self.center_zonal.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.center_zonal.shape[1])

    @property
    def dependence_rank(self) -> int:
        return int(self.dependence_loadings_free.shape[-1])

    @property
    def has_sampling_axis(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_latent": self,
            "center_zonal": self.center_zonal,
            "scale_zonal": self.scale_zonal,
            "residual_caps_zonal": self.residual_caps_zonal,
            "support_lower_zonal": self.support_lower_zonal,
            "support_upper_zonal": self.support_upper_zonal,
            "scale_fraction_free": self.scale_fraction_free,
            "skew_free": self.skew_free,
            "tail_free": self.tail_free,
            "dependence_loadings_free": self.dependence_loadings_free,
            "quantile_offsets_free": self.quantile_offsets_free,
            "regime_logits": self.regime_logits,
            "token_features": self.token_features,
            "origin_index": self.origin_index,
            "architecture_revision": self.architecture_revision,
            "sampling_axis_present": False,
            "fixed_particle_capacity": False,
            "support_contract": (
                "train_frozen_absolute_bounds_intersected_with_centered_caps;"
                "RT_support_derived_from_DA_minus_DART"
            ),
        }


class DistributionHead(nn.Module):
    """Flow-compatible conditional residual head with arbitrary-N sampling.

    The first V6 correction cycle uses an analytic continuous residual
    transform (heteroscedastic scale, skew, tail and low-rank dependence).
    Its public latent and base-noise seam are deliberately compatible with a
    later conditional-flow or flow-matching transform without changing the
    policy interface.
    """

    def __init__(
        self,
        token_dim: int = 128,
        *,
        horizon: int = HORIZON,
        hidden_dim: int = 128,
        dependence_rank: int = 4,
        regime_count: int = 8,
        residual_scale: float = 1.0,
        debug_contract_checks: bool = False,
    ) -> None:
        super().__init__()
        if int(horizon) != HORIZON:
            raise ValueError("formal V6 Distribution requires a 96-interval horizon")
        if int(dependence_rank) < 1:
            raise ValueError("dependence_rank must be positive")
        if int(regime_count) < 2:
            raise ValueError("regime_count must be at least two")
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("residual_scale must lie in (0,1]")
        self.token_dim = int(token_dim)
        self.horizon = HORIZON
        self.hidden_dim = int(hidden_dim)
        self.dependence_rank = int(dependence_rank)
        self.regime_count = int(regime_count)
        self.residual_scale = float(residual_scale)
        self.debug_contract_checks = bool(debug_contract_checks)
        self.num_scenarios = None

        self.condition = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
        )
        self.scale_head = nn.Linear(hidden_dim, FREE_VARIABLE_COUNT)
        self.skew_head = nn.Linear(hidden_dim, FREE_VARIABLE_COUNT)
        self.tail_head = nn.Linear(hidden_dim, FREE_VARIABLE_COUNT)
        self.dependence_head = nn.Linear(
            hidden_dim, FREE_VARIABLE_COUNT * self.dependence_rank
        )
        self.regime_head = nn.Linear(hidden_dim, self.regime_count)
        self.dart_adapter = nn.Linear(token_dim, 4)
        self.dart_scale_adapter = nn.Linear(token_dim, 4)
        self.dart_tail_adapter = nn.Linear(token_dim, 4)
        nn.init.zeros_(self.dart_scale_adapter.weight)
        nn.init.zeros_(self.dart_scale_adapter.bias)
        nn.init.zeros_(self.dart_tail_adapter.weight)
        nn.init.zeros_(self.dart_tail_adapter.bias)
        latent_width = (
            FREE_VARIABLE_COUNT * 3
            + FREE_VARIABLE_COUNT * self.dependence_rank
            + self.regime_count
        )
        self.world_projection = nn.Sequential(
            nn.Linear(latent_width, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.policy_bridge = nn.Linear(token_dim, token_dim)
        nn.init.zeros_(self.policy_bridge.weight)
        nn.init.zeros_(self.policy_bridge.bias)
        self.world_norm = nn.LayerNorm(token_dim)
        self.register_buffer(
            "default_raw_unit_residual_caps_zonal",
            torch.tensor(_FALLBACK_RAW_UNIT_CAPS_ZONAL, dtype=torch.float32),
            persistent=False,
        )
        self._default_residual_caps_formally_set = False

    @property
    def residual_cap_contract(self) -> dict[str, object]:
        return {
            "schema": "r4_v6_continuous_distribution_support_v1",
            "formal_input": "scenario_residual_caps_zonal",
            "formal_input_shape": "[B,96,21]",
            "units": "raw_output_units",
            "fit_split": "train_only",
            "free_channels": "DA4,DART4,workload1,PV4,shared_PUE_latent",
            "derived_channel": "RT=DA-DART;DART=DA-RT_final",
            "sampling_capacity": "unbounded_runtime_budget",
            "compatibility_fallback_only": True,
        }

    def set_default_residual_caps(self, caps: Tensor) -> None:
        value = torch.as_tensor(
            caps,
            device=self.default_raw_unit_residual_caps_zonal.device,
            dtype=self.default_raw_unit_residual_caps_zonal.dtype,
        )
        if value.shape != (ZONAL_VARIABLE_COUNT,):
            raise ValueError("default distribution residual caps must be [21]")
        if not bool(torch.isfinite(value).all()) or not bool((value > 0.0).all()):
            raise ValueError("distribution residual caps must be finite and positive")
        value = value.clone()
        value[4:8] = value[0:4] + value[8:12]
        self.default_raw_unit_residual_caps_zonal.copy_(value)
        self._default_residual_caps_formally_set = True

    def initialize_scale_from_contract(
        self,
        score_scales: Tensor,
        caps: Tensor,
        *,
        minimum_fraction: float = 0.02,
        maximum_fraction: float = 0.50,
    ) -> Tensor:
        """Initialize conditional scale near the frozen train residual scale.

        The head starts with zero weights and a contract-derived bias.  It can
        subsequently learn heteroscedastic deviations, but it does not begin
        at the old random roughly-0.4-times-p99 envelope.
        """

        score = torch.as_tensor(score_scales, dtype=torch.float64).flatten()
        cap = torch.as_tensor(caps, dtype=torch.float64).flatten()
        if score.shape != (ZONAL_VARIABLE_COUNT,) or cap.shape != (
            ZONAL_VARIABLE_COUNT,
        ):
            raise ValueError("score_scales and caps must each contain 21 values")
        if not bool(torch.isfinite(score).all() and torch.isfinite(cap).all()):
            raise ValueError("score_scales and caps must be finite")
        if not bool((score > 0.0).all() and (cap > 0.0).all()):
            raise ValueError("score_scales and caps must be positive")
        lower = float(minimum_fraction)
        upper = float(maximum_fraction)
        if not 0.0 < lower <= upper < self.residual_scale:
            raise ValueError(
                "scale initialization fractions must satisfy 0 < min <= max < residual_scale"
            )
        ratio = (score / cap).clamp(lower, upper)
        free = torch.cat(
            (
                ratio[0:4],
                ratio[8:12],
                ratio[12:13],
                ratio[13:17],
                ratio[17:21].amax().reshape(1),
            )
        )
        # scale_fraction = softplus(b)/(1+softplus(b))*residual_scale
        positive = free / (self.residual_scale - free)
        bias = torch.log(torch.expm1(positive)).to(
            device=self.scale_head.bias.device,
            dtype=self.scale_head.bias.dtype,
        )
        with torch.no_grad():
            self.scale_head.weight.zero_()
            self.scale_head.bias.copy_(bias)
        return free.to(dtype=self.scale_head.bias.dtype)

    def _resolve_caps(self, value: Tensor | None, reference: Tensor) -> Tensor:
        batch = int(reference.shape[0])
        raw = (
            self.default_raw_unit_residual_caps_zonal.to(reference)
            if value is None
            else torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        )
        if raw.shape == (ZONAL_VARIABLE_COUNT,):
            caps = raw.reshape(1, 1, -1).expand(batch, HORIZON, -1)
        elif raw.shape == (HORIZON, ZONAL_VARIABLE_COUNT):
            caps = raw.unsqueeze(0).expand(batch, -1, -1)
        elif raw.shape == (batch, HORIZON, ZONAL_VARIABLE_COUNT):
            caps = raw
        else:
            raise ValueError(
                "scenario_residual_caps_zonal must be [21], [96,21], or "
                "[batch,96,21]"
            )
        caps = caps.clone()
        caps[..., 4:8] = caps[..., 0:4] + caps[..., 8:12]
        if self._default_residual_caps_formally_set and self.debug_contract_checks:
            expected = self.default_raw_unit_residual_caps_zonal.to(caps).reshape(
                1, 1, -1
            ).expand_as(caps)
            if not bool(torch.equal(caps, expected)):
                raise ValueError(
                    "scenario_residual_caps_zonal differs from the installed "
                    "frozen train-only contract"
                )
        return caps

    @staticmethod
    def _expand_support(
        value: Tensor | None,
        reference: Tensor,
        *,
        name: str,
    ) -> Tensor | None:
        if value is None:
            return None
        batch = int(reference.shape[0])
        raw = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if raw.shape == (ZONAL_VARIABLE_COUNT,):
            return raw.reshape(1, 1, -1).expand(batch, HORIZON, -1)
        if raw.shape == (HORIZON, ZONAL_VARIABLE_COUNT):
            return raw.unsqueeze(0).expand(batch, -1, -1)
        if raw.shape == (batch, HORIZON, ZONAL_VARIABLE_COUNT):
            return raw
        raise ValueError(f"{name} must be [21], [96,21], or [batch,96,21]")

    def _support_bounds(
        self,
        center: Tensor,
        caps: Tensor,
        lower_value: Tensor | None,
        upper_value: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        raw_lower = center - caps
        raw_upper = center + caps
        supplied_lower = self._expand_support(
            lower_value, center, name="distribution_support_lower_zonal"
        )
        supplied_upper = self._expand_support(
            upper_value, center, name="distribution_support_upper_zonal"
        )
        if supplied_lower is not None:
            raw_lower = torch.maximum(raw_lower, supplied_lower)
        if supplied_upper is not None:
            raw_upper = torch.minimum(raw_upper, supplied_upper)
        da_lower = raw_lower[..., 0:4]
        da_upper = raw_upper[..., 0:4]
        dart_lower = raw_lower[..., 8:12]
        dart_upper = raw_upper[..., 8:12]
        # RT is not an independently clipped variable.  Its canonical support
        # is exactly induced by the free DA and DART supports.
        rt_lower = da_lower - dart_upper
        rt_upper = da_upper - dart_lower
        if supplied_lower is not None:
            rt_lower = torch.maximum(rt_lower, supplied_lower[..., 4:8])
        if supplied_upper is not None:
            rt_upper = torch.minimum(rt_upper, supplied_upper[..., 4:8])
        workload_lower = raw_lower[..., 12:13].clamp_min(0.0)
        pv_lower = raw_lower[..., 13:17].clamp_min(0.0)
        pue_lower = raw_lower[..., 17:21].clamp_min(1.0e-4)
        lower = torch.cat(
            (
                da_lower,
                rt_lower,
                dart_lower,
                workload_lower,
                pv_lower,
                pue_lower,
            ),
            dim=-1,
        )
        upper = torch.cat(
            (
                da_upper,
                rt_upper,
                dart_upper,
                raw_upper[..., 12:13],
                raw_upper[..., 13:17],
                raw_upper[..., 17:21],
            ),
            dim=-1,
        )
        if self.debug_contract_checks and bool((lower > upper).any()):
            raise ValueError("distribution support has an empty interval")
        return lower, upper

    def _origin(
        self, origin: int | Tensor | None, reference: Tensor
    ) -> Tensor:
        batch = int(reference.shape[0])
        index = torch.as_tensor(
            0 if origin is None else origin, device=reference.device
        )
        if index.ndim == 0:
            index = index.expand(batch)
        if index.shape != (batch,):
            raise ValueError("origin must be scalar or [batch]")
        if (
            self.debug_contract_checks
            and index.dtype.is_floating_point
            and not torch.equal(index, index.round())
        ):
            raise ValueError("origin must contain integer indices")
        index = index.long()
        if self.debug_contract_checks and bool(
            ((index < 0) | (index >= HORIZON)).any()
        ):
            raise ValueError("origin must lie in [0,95]")
        return index

    @staticmethod
    def _canonical_center(center: Tensor) -> Tensor:
        da = center[..., 0:4]
        rt = center[..., 4:8]
        dart = da - rt
        workload = center[..., 12:13].clamp_min(0.0)
        pv = center[..., 13:17].clamp_min(0.0)
        pue = center[..., 17:21].clamp_min(1.0e-4)
        return torch.cat((da, rt, dart, workload, pv, pue), dim=-1)

    @staticmethod
    def _free_to_zonal_scale(scale_fraction: Tensor, caps: Tensor) -> Tensor:
        da = scale_fraction[..., 0:4] * caps[..., 0:4]
        dart = scale_fraction[..., 4:8] * caps[..., 8:12]
        rt = da + dart
        workload = scale_fraction[..., 8:9] * caps[..., 12:13]
        pv = scale_fraction[..., 9:13] * caps[..., 13:17]
        pue = scale_fraction[..., 13:14] * caps[..., 17:21]
        return torch.cat((da, rt, dart, workload, pv, pue), dim=-1)

    @staticmethod
    def _residual_transform(noise: Tensor, skew: Tensor, tail: Tensor) -> Tensor:
        # Smooth analytic correction-cycle-2 transform.  It starts from a
        # Gaussian base and exposes an explicit seam for a later learned flow.
        asymmetric = noise + 0.15 * skew * (noise.square() - 1.0)
        tail_excess = F.softplus(asymmetric.abs() - 1.0) - F.softplus(
            asymmetric.new_tensor(-1.0)
        )
        return asymmetric + 0.10 * tail * asymmetric.sign() * tail_excess.square()

    def forward(
        self,
        token_features: Tensor,
        center_zonal: Tensor,
        *,
        origin: int | Tensor | None = None,
        residual_caps_zonal: Tensor | None = None,
        support_lower_zonal: Tensor | None = None,
        support_upper_zonal: Tensor | None = None,
    ) -> DistributionalWorldLatent:
        if token_features.ndim != 3 or token_features.shape[1:] != (
            HORIZON,
            self.token_dim,
        ):
            raise ValueError(
                f"distribution token_features must be [batch,96,{self.token_dim}]"
            )
        if center_zonal.shape != (*token_features.shape[:2], ZONAL_VARIABLE_COUNT):
            raise ValueError("center_zonal must be [batch,96,21]")
        caps = self._resolve_caps(residual_caps_zonal, token_features)
        center = self._canonical_center(center_zonal)
        support_lower, support_upper = self._support_bounds(
            center,
            caps,
            support_lower_zonal,
            support_upper_zonal,
        )
        condition = self.condition(token_features)
        base_scale_logits = self.scale_head(condition)
        scale_logits = torch.cat(
            (
                base_scale_logits[..., 0:4],
                base_scale_logits[..., 4:8]
                + 0.1 * self.dart_scale_adapter(token_features),
                base_scale_logits[..., 8:14],
            ),
            dim=-1,
        )
        positive_scale = F.softplus(scale_logits)
        scale_fraction = (
            positive_scale / (1.0 + positive_scale)
        ) * self.residual_scale
        base_skew = torch.tanh(self.skew_head(condition))
        dart_skew = (
            base_skew[..., 4:8]
            + 0.05 * torch.tanh(self.dart_adapter(token_features))
        ).clamp(-1.0, 1.0)
        skew = torch.cat(
            (base_skew[..., 0:4], dart_skew, base_skew[..., 8:14]), dim=-1
        )
        base_tail_logits = self.tail_head(condition)
        tail_logits = torch.cat(
            (
                base_tail_logits[..., 0:4],
                base_tail_logits[..., 4:8]
                + 0.1 * self.dart_tail_adapter(token_features),
                base_tail_logits[..., 8:14],
            ),
            dim=-1,
        )
        tail = 1.5 * torch.sigmoid(tail_logits)
        dependence = torch.tanh(self.dependence_head(condition)).reshape(
            *condition.shape[:2], FREE_VARIABLE_COUNT, self.dependence_rank
        )
        regime_logits = self.regime_head(condition)
        latent_parameters = torch.cat(
            (
                scale_fraction,
                skew,
                tail,
                dependence.flatten(-2),
                regime_logits,
            ),
            dim=-1,
        )
        world_tokens = self.world_norm(
            token_features
            + self.world_projection(latent_parameters)
            + 0.1 * self.policy_bridge(token_features)
        )
        # Five declared marginal quantile offsets are diagnostic structure,
        # not a discrete support.  Sorting guarantees their public ordering.
        normal_knots = token_features.new_tensor(
            (-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536)
        )
        transformed_knots = self._residual_transform(
            normal_knots.reshape(1, 1, 1, -1),
            skew[..., None],
            tail[..., None],
        )
        quantile_offsets = (
            torch.tanh(scale_fraction[..., None] * transformed_knots)
            .sort(dim=-1)
            .values
        )
        return DistributionalWorldLatent(
            center_zonal=center,
            scale_zonal=self._free_to_zonal_scale(scale_fraction, caps),
            residual_caps_zonal=caps,
            support_lower_zonal=support_lower,
            support_upper_zonal=support_upper,
            scale_fraction_free=scale_fraction,
            skew_free=skew,
            tail_free=tail,
            dependence_loadings_free=dependence,
            quantile_offsets_free=quantile_offsets,
            regime_logits=regime_logits,
            token_features=world_tokens,
            origin_index=self._origin(origin, token_features),
        )

    @staticmethod
    def _generator(
        reference: Tensor,
        *,
        generator: torch.Generator | None,
        seed: int | None,
    ) -> torch.Generator | None:
        if generator is not None and seed is not None:
            raise ValueError("provide generator or seed, not both")
        if seed is None:
            return generator
        device = reference.device if reference.is_cuda else torch.device("cpu")
        created = torch.Generator(device=device)
        created.manual_seed(int(seed))
        return created

    def _validate_revealed(
        self,
        world: DistributionalWorldLatent,
        revealed_zonal: Tensor | None,
        revealed_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if revealed_zonal is None and revealed_mask is None:
            return None, None
        if revealed_zonal is None or revealed_mask is None:
            raise ValueError("revealed_zonal and revealed_mask must be provided together")
        values = torch.as_tensor(
            revealed_zonal,
            device=world.center_zonal.device,
            dtype=world.center_zonal.dtype,
        )
        mask = torch.as_tensor(revealed_mask, device=world.center_zonal.device).bool()
        if values.shape != world.center_zonal.shape:
            raise ValueError("revealed_zonal must be [batch,96,21]")
        if mask.shape != world.center_zonal.shape[:2]:
            raise ValueError("revealed_mask must be [batch,96]")
        periods = torch.arange(HORIZON, device=mask.device)[None]
        future_mask = mask & (periods > world.origin_index[:, None])
        if self.debug_contract_checks and bool(future_mask.any()):
            raise ValueError("revealed_mask exposes a future value")
        mask = mask & ~future_mask
        identity = values[..., 0:4] - values[..., 4:8] - values[..., 8:12]
        if self.debug_contract_checks and bool(
            (identity[mask].abs() > 1.0e-5).any()
        ):
            raise ValueError("revealed zonal values violate DA-RT-DART identity")
        return values, mask

    def sample(
        self,
        world: DistributionalWorldLatent,
        N: int,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        revealed_zonal: Tensor | None = None,
        revealed_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        """Draw ``N`` complete joint trajectories from one immutable world latent."""

        if isinstance(N, bool) or int(N) != N or int(N) < 1:
            raise ValueError("N must be a positive integer sampling budget")
        count = int(N)
        reference = world.center_zonal
        rng = self._generator(reference, generator=generator, seed=seed)
        independent = torch.randn(
            (
                world.batch_size,
                count,
                HORIZON,
                FREE_VARIABLE_COUNT,
            ),
            device=reference.device,
            dtype=reference.dtype,
            generator=rng,
        )
        factors = torch.randn(
            (world.batch_size, count, world.dependence_rank),
            device=reference.device,
            dtype=reference.dtype,
            generator=rng,
        )
        dependent = torch.einsum(
            "bnr,btcr->bntc", factors, world.dependence_loadings_free
        )
        variance = 1.0 + world.dependence_loadings_free.square().sum(-1)
        base = (independent + dependent) / variance[:, None].sqrt()
        transformed = self._residual_transform(
            base,
            world.skew_free[:, None],
            world.tail_free[:, None],
        )
        bounded_fraction = torch.tanh(
            world.scale_fraction_free[:, None] * transformed
        )
        caps = world.residual_caps_zonal[:, None]
        center = world.center_zonal[:, None]
        support_lower = world.support_lower_zonal[:, None]
        support_upper = world.support_upper_zonal[:, None]
        da = torch.clamp(
            center[..., 0:4] + bounded_fraction[..., 0:4] * caps[..., 0:4],
            min=support_lower[..., 0:4],
            max=support_upper[..., 0:4],
        )
        dart_candidate = (
            center[..., 8:12]
            + bounded_fraction[..., 4:8] * caps[..., 8:12]
        )
        dart_candidate = torch.clamp(
            dart_candidate,
            min=support_lower[..., 8:12],
            max=support_upper[..., 8:12],
        )
        # Project the two free variables onto any additionally declared RT
        # support without introducing RT as an independent stochastic output.
        difference = da - dart_candidate
        increase = (support_lower[..., 4:8] - difference).clamp_min(0.0)
        da_increase = torch.minimum(
            increase, (support_upper[..., 0:4] - da).clamp_min(0.0)
        )
        da = da + da_increase
        dart_candidate = dart_candidate - torch.minimum(
            (increase - da_increase).clamp_min(0.0),
            (dart_candidate - support_lower[..., 8:12]).clamp_min(0.0),
        )
        difference = da - dart_candidate
        decrease = (difference - support_upper[..., 4:8]).clamp_min(0.0)
        da_decrease = torch.minimum(
            decrease, (da - support_lower[..., 0:4]).clamp_min(0.0)
        )
        da = da - da_decrease
        dart_candidate = dart_candidate + torch.minimum(
            (decrease - da_decrease).clamp_min(0.0),
            (support_upper[..., 8:12] - dart_candidate).clamp_min(0.0),
        )
        rt, dart = fp_safe_close_da_rt_dart(
            da,
            dart_candidate,
            rt_lower=support_lower[..., 4:8],
            rt_upper=support_upper[..., 4:8],
            dart_lower=support_lower[..., 8:12],
            dart_upper=support_upper[..., 8:12],
        )
        workload = torch.clamp(
            center[..., 12:13]
            + bounded_fraction[..., 8:9] * caps[..., 12:13],
            min=support_lower[..., 12:13],
            max=support_upper[..., 12:13],
        )
        pv = torch.clamp(
            center[..., 13:17]
            + bounded_fraction[..., 9:13] * caps[..., 13:17],
            min=support_lower[..., 13:17],
            max=support_upper[..., 13:17],
        )
        pue = torch.clamp(
            center[..., 17:21]
            + bounded_fraction[..., 13:14] * caps[..., 17:21],
            min=support_lower[..., 17:21],
            max=support_upper[..., 17:21],
        )
        samples = torch.cat((da, rt, dart, workload, pv, pue), dim=-1)

        revealed_values, mask = self._validate_revealed(
            world, revealed_zonal, revealed_mask
        )
        if revealed_values is not None and mask is not None:
            samples = torch.where(
                mask[:, None, :, None], revealed_values[:, None], samples
            )
            # Re-close only the unrevealed suffix.  All 21 revealed channels
            # remain bitwise identical to the legally observed prefix.
            da = samples[..., 0:4]
            rt = samples[..., 4:8]
            closed_dart = da - rt
            dart = torch.where(
                mask[:, None, :, None],
                samples[..., 8:12],
                closed_dart,
            )
            samples = torch.cat(
                (
                    da,
                    rt,
                    dart,
                    samples[..., 12:13],
                    samples[..., 13:17],
                    samples[..., 17:21],
                ),
                dim=-1,
            )
        weights = samples.new_full(
            (world.batch_size, count), 1.0 / float(count)
        )
        return {
            "samples_zonal": samples,
            "particles_zonal": samples,
            "particles": samples,
            "samples_legacy": zonal_to_legacy(samples),
            "particles_legacy": zonal_to_legacy(samples),
            "weights": weights,
            "sampling_budget": count,
            "sampling_seed": None if seed is None else int(seed),
            "finite_continuous_joint": torch.isfinite(samples).all(),
            "effective_sample_size": weights.square().sum(-1).reciprocal(),
            "weight_entropy": -(weights * weights.log()).sum(-1),
            "distribution_architecture_revision": world.architecture_revision,
            "fixed_particle_capacity": False,
            "support_lower_zonal": world.support_lower_zonal,
            "support_upper_zonal": world.support_upper_zonal,
            "support_contract": (
                "train_frozen_absolute_bounds_intersected_with_centered_caps;"
                "revealed_prefix_is_observed_and_not_clipped"
            ),
        }


class RiskEncoder(nn.Module):
    """Neural risk features for Policy; never a formal empirical CVaR report."""

    def __init__(self, token_dim: int = 128, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        summary_width = token_dim + ZONAL_VARIABLE_COUNT + FREE_VARIABLE_COUNT * 2
        self.encoder = nn.Sequential(
            nn.Linear(summary_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.rolling_projection = nn.Linear(token_dim, hidden_dim)
        self.expected_head = nn.Linear(hidden_dim, 1)
        self.var_spread_head = nn.Linear(hidden_dim, 1)
        self.cvar_spread_head = nn.Linear(hidden_dim, 1)
        self.stress_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        world: DistributionalWorldLatent,
        rolling_state_hidden: Tensor | None = None,
    ) -> dict[str, Any]:
        pooled = world.token_features.mean(dim=1)
        scale_summary = world.scale_zonal.mean(dim=1)
        tail_summary = world.tail_free.mean(dim=1)
        dependence_summary = world.dependence_loadings_free.square().mean(
            dim=(1, 3)
        ).sqrt()
        latent = self.encoder(
            torch.cat(
                (pooled, scale_summary, tail_summary, dependence_summary), dim=-1
            )
        )
        if rolling_state_hidden is not None:
            if rolling_state_hidden.shape != (world.batch_size, self.token_dim):
                raise ValueError(
                    f"rolling_state_hidden must be [batch,{self.token_dim}]"
                )
            latent = latent + self.rolling_projection(rolling_state_hidden)
        expected_proxy = self.expected_head(latent).squeeze(-1)
        var_proxy = expected_proxy + F.softplus(
            self.var_spread_head(latent).squeeze(-1)
        )
        cvar_proxy = var_proxy + F.softplus(
            self.cvar_spread_head(latent).squeeze(-1)
        )
        uncertainty = torch.tanh(self.uncertainty_head(latent))
        return {
            "risk_features": latent,
            "tail_risk_latent": latent,
            "uncertainty_latent": uncertainty,
            "neural_expected_cost_proxy": expected_proxy,
            "neural_var95_proxy": var_proxy,
            "neural_cvar95_proxy": cvar_proxy,
            "stress_probability": torch.sigmoid(
                self.stress_head(latent).squeeze(-1)
            ),
            "formal_cvar": None,
            "formal_cvar_reported": False,
            "contract": (
                "neural_auxiliary_features_only; formal_VaR_CVaR_requires_"
                "independent_distribution_sampling_and_post_shield_replay"
            ),
        }


__all__ = [
    "DISTRIBUTIONAL_ARCHITECTURE_REVISION",
    "DistributionHead",
    "DistributionalWorldLatent",
    "FREE_VARIABLE_COUNT",
    "HORIZON",
    "RiskEncoder",
    "ZONAL_VARIABLE_COUNT",
    "ZONAL_VARIABLE_SLICES",
    "fp_safe_close_da_rt_dart",
    "zonal_to_legacy",
]
