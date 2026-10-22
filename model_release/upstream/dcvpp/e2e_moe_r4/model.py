"""Hybrid-CfC-Transformer-MoE E2E-DART-VPP R4 model core.

This package is an independent R4 lineage.  It imports no R3 model and no
mathematical solver.  The formal interfaces preserve zonal token sequences all
the way from issue-time encoding to forecasts, scenarios, risk, and the rolling
policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .liquid import ClosedFormLiquidSequence
from .policy import LiquidRollingPolicy
from .scenario_risk import (
    DISTRIBUTIONAL_ARCHITECTURE_REVISION,
    DistributionHead,
    DistributionalWorldLatent,
    EquationBasedRisk,
    RiskEncoder,
    SCENARIO_COUNT,
    ZONAL_VARIABLE_COUNT,
    zonal_to_legacy,
)


ARCHITECTURE_VERSION = "hybrid_cfc_transformer_moe_r4"
DISTRIBUTIONAL_JOINT_ARCHITECTURE_VERSION = (
    "continuous_distributional_joint_moe_r4_v6"
)


@dataclass(frozen=True)
class R4Config:
    architecture_version: str = ARCHITECTURE_VERSION
    representation_base_architecture: str = ARCHITECTURE_VERSION
    distributional_architecture_revision: str = DISTRIBUTIONAL_ARCHITECTURE_REVISION
    horizon: int = 96
    history_days: int = 7
    num_sites: int = 4
    market_dim: int = 16
    workload_dim: int = 5
    physical_static_dim: int = 4
    physical_dynamic_dim: int = 12
    state_dim: int = 49
    hidden_dim: int = 128
    market_layers: int = 2
    market_heads: int = 4
    market_ff_dim: int = 384
    fusion_heads: int = 4
    fusion_ff_dim: int = 256
    expert_hidden_dim: int = 256
    num_experts: int = 8
    top_k: int = 2
    num_scenarios: int = 20
    scenario_code_dim: int = 32
    scenario_hidden_dim: int = 64
    distribution_hidden_dim: int = 128
    distribution_dependence_rank: int = 4
    distribution_regime_count: int = 8
    distribution_debug_contract_checks: bool = False
    policy_hidden_dim: int = 256
    policy_refinement_steps: int = 2
    dropout: float = 0.1
    router_jitter: float = 0.05
    physical_residual_bound: float = 0.1
    forecast_da_residual_cap: float = 0.1
    forecast_pv_residual_cap: float = 0.01
    forecast_pue_residual_cap: float = 0.04
    cvar_alpha: float = 0.95
    max_grid_action: float = 1.0
    max_battery_action: float = 1.0
    max_compute_action: float = 1.0

    def __post_init__(self) -> None:
        if self.architecture_version not in {
            ARCHITECTURE_VERSION,
            DISTRIBUTIONAL_JOINT_ARCHITECTURE_VERSION,
        }:
            raise ValueError(
                "R4 architecture_version must be the archived R4/V5 value or "
                "the V6 continuous-distributional value"
            )
        if self.representation_base_architecture != ARCHITECTURE_VERSION:
            raise ValueError("V6 Representation base must remain hybrid R4")
        if self.distributional_architecture_revision != (
            DISTRIBUTIONAL_ARCHITECTURE_REVISION
        ):
            raise ValueError("unknown Distributional World architecture revision")
        if self.horizon != 96 or self.history_days != 7:
            raise ValueError("R4 fixes 96 current-day and 7 historical market tokens")
        if self.hidden_dim != 128:
            raise ValueError("R4 fixes d_model=128")
        if (self.market_layers, self.market_heads, self.market_ff_dim) != (2, 4, 384):
            raise ValueError("R4 market Transformer is fixed at 2L/4H/FF384")
        if (self.num_experts, self.top_k) != (8, 2):
            raise ValueError("R4 requires exactly eight Top-2 experts")
        if int(self.num_scenarios) < 1:
            raise ValueError("compatibility sampling budget must be positive")
        if int(self.distribution_dependence_rank) < 1:
            raise ValueError("distribution_dependence_rank must be positive")
        if int(self.distribution_regime_count) < 2:
            raise ValueError("distribution_regime_count must be at least two")
        if self.policy_refinement_steps != 2:
            raise ValueError("R4 requires two policy refinement steps")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie in (0,1)")
        for name in (
            "forecast_da_residual_cap",
            "forecast_pv_residual_cap",
            "forecast_pue_residual_cap",
        ):
            cap = float(getattr(self, name))
            if not isfinite(cap) or cap <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "R4Config":
        if value is None:
            return cls()
        root_mapping: Mapping[str, Any] = value
        project = root_mapping.get("project", {})
        if not isinstance(project, Mapping):
            project = {}
        project_architecture = project.get("architecture_version")
        mapping: Mapping[str, Any] = root_mapping
        nested_model = mapping.get("model")
        if isinstance(nested_model, Mapping):
            mapping = nested_model
        # Governing YAML uses a descriptive nested schema; translate it to the
        # compact constructor without weakening the dataclass invariants.
        if any(isinstance(item, Mapping) for item in mapping.values()):
            market = mapping.get("market", {})
            workload = mapping.get("workload", {})
            physical = mapping.get("physical", {})
            fusion = mapping.get("fusion", {})
            moe = mapping.get("moe", {})
            scenarios = mapping.get("scenarios", {})
            distribution = mapping.get("distribution", {})
            policy = mapping.get("policy", {})
            forecasts = mapping.get("forecasts", {})
            if not all(
                isinstance(item, Mapping)
                for item in (
                    market,
                    workload,
                    physical,
                    fusion,
                    moe,
                    scenarios,
                    distribution,
                    policy,
                    forecasts,
                )
            ):
                raise ValueError("nested R4 model configuration entries must be mappings")
            forecast_caps = forecasts.get(
                "bounded_neural_residual_raw_unit_fallback_caps",
                forecasts.get("bounded_neural_residuals", {}),
            )
            if not isinstance(forecast_caps, Mapping):
                raise ValueError(
                    "model.forecasts residual raw-unit fallback caps must be a mapping"
                )
            workload_features = workload.get("input_features", 5)
            translated = {
                "architecture_version": mapping.get(
                    "architecture_version",
                    project_architecture or ARCHITECTURE_VERSION,
                ),
                "representation_base_architecture": mapping.get(
                    "representation_base_architecture", ARCHITECTURE_VERSION
                ),
                "distributional_architecture_revision": mapping.get(
                    "distributional_architecture_revision",
                    DISTRIBUTIONAL_ARCHITECTURE_REVISION,
                ),
                "horizon": mapping.get("sequence_length", 96),
                "hidden_dim": mapping.get("latent_width", 128),
                "market_dim": market.get("input_features", 16),
                "history_days": market.get("history_tokens", 7),
                "market_layers": market.get("transformer_layers", 2),
                "market_heads": market.get("attention_heads", 4),
                "market_ff_dim": market.get("feedforward_width", 384),
                "dropout": market.get("dropout", 0.1),
                "workload_dim": len(workload_features)
                if isinstance(workload_features, (list, tuple))
                else int(workload_features),
                "physical_dynamic_dim": physical.get("input_features", 12),
                "fusion_heads": fusion.get("attention_heads", 4),
                "fusion_ff_dim": fusion.get("feedforward_width", 256),
                "num_experts": moe.get("experts", 8),
                "top_k": moe.get("top_k", 2),
                "expert_hidden_dim": moe.get("expert_hidden_width", 256),
                "router_jitter": moe.get("router_jitter_std_train", 0.05),
                "num_scenarios": scenarios.get("count", 20),
                "scenario_code_dim": scenarios.get("scenario_code_width", 32),
                "scenario_hidden_dim": scenarios.get("decoder_hidden_width", 64),
                "distribution_hidden_dim": distribution.get("hidden_width", 128),
                "distribution_dependence_rank": distribution.get(
                    "dependence_rank", 4
                ),
                "distribution_regime_count": distribution.get("regimes", 8),
                "distribution_debug_contract_checks": distribution.get(
                    "debug_contract_checks", False
                ),
                "policy_hidden_dim": policy.get("hidden_width", 256),
                "policy_refinement_steps": policy.get("residual_refinement_steps", 2),
                "physical_residual_bound": physical.get(
                    "residual_bound_fraction", 0.1
                ),
                "forecast_da_residual_cap": forecast_caps.get("da", 0.1),
                "forecast_pv_residual_cap": forecast_caps.get("pv", 0.01),
                "forecast_pue_residual_cap": forecast_caps.get("pue", 0.04),
            }
            return cls(**translated)
        for key in ("architecture", "r4_model"):
            nested = mapping.get(key)
            if isinstance(nested, Mapping):
                mapping = nested
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown R4 model configuration keys: {unknown}")
        return cls(**{key: mapping[key] for key in known if key in mapping})


class PreNormTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, relative_bias: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=relative_bias,
            need_weights=False,
        )[0]
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.ff(self.ff_norm(tokens))


class MarketTransformerEncoder(nn.Module):
    """Seven completed-day summary tokens plus 96 intraday tokens."""

    def __init__(self, config: R4Config) -> None:
        super().__init__()
        self.config = config
        self.context_length = config.history_days + config.horizon
        self.market_projection = nn.Linear(config.market_dim, config.hidden_dim)
        self.history_projection = nn.Linear(config.market_dim, config.hidden_dim)
        self.history_missing = nn.Parameter(torch.zeros(1, config.history_days, config.hidden_dim))
        self.position_embedding = nn.Embedding(self.context_length, config.hidden_dim)
        self.hour_projection = nn.Linear(6, config.hidden_dim, bias=False)
        self.day_embedding = nn.Embedding(7, config.hidden_dim)
        self.month_embedding = nn.Embedding(12, config.hidden_dim)
        self.holiday_embedding = nn.Embedding(2, config.hidden_dim)
        self.dst_embedding = nn.Embedding(2, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.relative_time_bias = nn.Parameter(torch.zeros(2 * self.context_length - 1))
        self.layers = nn.ModuleList(
            PreNormTransformerLayer(
                config.hidden_dim,
                config.market_heads,
                config.market_ff_dim,
                config.dropout,
            )
            for _ in range(config.market_layers)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def _calendar_index(
        self, value: Tensor | int | None, batch: int, reference: Tensor, maximum: int
    ) -> Tensor:
        result = torch.as_tensor(0 if value is None else value, device=reference.device)
        if result.ndim == 0:
            result = result.expand(batch)
        if result.shape != (batch,):
            raise ValueError("calendar indices must be scalar or [batch]")
        result = result.long()
        if ((result < 0) | (result >= maximum)).any():
            raise ValueError(f"calendar index must lie in [0,{maximum - 1}]")
        return result

    def _hour_harmonics(self, reference: Tensor) -> Tensor:
        step = torch.arange(self.config.horizon, device=reference.device, dtype=torch.float32)
        angle = 2.0 * torch.pi * step / float(self.config.horizon)
        features = []
        for harmonic in (1.0, 2.0, 4.0):
            features.extend((torch.sin(harmonic * angle), torch.cos(harmonic * angle)))
        return torch.stack(features, dim=-1).to(dtype=reference.dtype)

    def forward(
        self,
        market: Tensor,
        *,
        historical_market: Tensor | None = None,
        historical_market_available: Tensor | None = None,
        day_of_week: Tensor | int | None = None,
        month: Tensor | int | None = None,
        holiday: Tensor | int | None = None,
        dst: Tensor | int | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if market.ndim != 3 or market.shape[1:] != (cfg.horizon, cfg.market_dim):
            raise ValueError(f"market must be [batch,96,{cfg.market_dim}]")
        batch = market.shape[0]
        if historical_market is None:
            history = self.history_missing.expand(batch, -1, -1)
            history_available = market.new_zeros((batch, cfg.history_days), dtype=torch.bool)
        else:
            if historical_market.shape != (batch, cfg.history_days, cfg.market_dim):
                raise ValueError(f"historical_market must be [batch,7,{cfg.market_dim}]")
            if historical_market_available is None:
                history_available = market.new_ones(
                    (batch, cfg.history_days), dtype=torch.bool
                )
            else:
                history_available = torch.as_tensor(
                    historical_market_available, device=market.device, dtype=torch.bool
                )
                if history_available.shape != (batch, cfg.history_days):
                    raise ValueError(
                        "historical_market_available must be [batch,7]"
                    )
            projected_history = self.history_projection(historical_market)
            history = torch.where(
                history_available[..., None],
                projected_history,
                self.history_missing.expand(batch, -1, -1),
            )
        current = self.market_projection(market)
        current = current + self.hour_projection(self._hour_harmonics(market))[None]
        calendar = (
            self.day_embedding(self._calendar_index(day_of_week, batch, market, 7))
            + self.month_embedding(self._calendar_index(month, batch, market, 12))
            + self.holiday_embedding(self._calendar_index(holiday, batch, market, 2))
            + self.dst_embedding(self._calendar_index(dst, batch, market, 2))
        )
        current = current + calendar[:, None]
        tokens = torch.cat((history, current), dim=1)
        positions = torch.arange(self.context_length, device=market.device)
        tokens = tokens + self.position_embedding(positions)[None]
        tokens = self.input_dropout(self.input_norm(tokens))
        relative = positions[:, None] - positions[None, :] + self.context_length - 1
        relative_bias = self.relative_time_bias[relative].to(dtype=tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, relative_bias)
        tokens = self.output_norm(tokens)
        return {
            "context_tokens": tokens,
            "tokens": tokens[:, cfg.history_days :],
            "history_tokens": tokens[:, : cfg.history_days],
            "history_available": history_available,
        }


class FrozenPhysicalTwinCore(nn.Module):
    """Test-only compatibility core.

    Formal R4 execution must inject
    :class:`ArtifactBackedFrozenPhysicalTwinCore`.  Keeping this small core
    available makes the tensor-only unit tests self-contained, but a formal
    preflight rejects it rather than mistaking ``requires_grad=False`` for a
    calibrated physical lineage.
    """

    formal_artifact_backed = False

    def __init__(self, input_dim: int, hidden_dim: int, num_sites: int = 4) -> None:
        super().__init__()
        projection = torch.zeros(hidden_dim, input_dim)
        for row in range(hidden_dim):
            projection[row, row % input_dim] = 1.0 / max(1, hidden_dim // input_dim)
        self.register_buffer("calibrated_projection", projection)
        self.register_buffer("calibrated_bias", torch.zeros(hidden_dim))
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_sites = int(num_sites)

    def forward(self, physical_static: Tensor) -> tuple[Tensor, Tensor]:
        if physical_static.ndim != 2 or physical_static.shape[-1] != self.input_dim:
            raise ValueError(f"physical_static must be [batch,{self.input_dim}]")
        latent = F.linear(physical_static, self.calibrated_projection, self.calibrated_bias)
        if self.input_dim >= self.num_sites:
            driver = physical_static[:, : self.num_sites]
        else:
            driver = physical_static.mean(dim=-1, keepdim=True).expand(-1, self.num_sites)
        pue = 1.0 + 0.2 * torch.sigmoid(driver)
        return latent, pue


class PhysicalStream(nn.Module):
    def __init__(
        self, config: R4Config, physical_twin_core: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.physical_twin_core = physical_twin_core or FrozenPhysicalTwinCore(
            config.physical_static_dim, config.hidden_dim, config.num_sites
        )
        self.formal_artifact_backed = bool(
            getattr(self.physical_twin_core, "formal_artifact_backed", False)
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(config.physical_static_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.dynamic_liquid = ClosedFormLiquidSequence(
            config.physical_dynamic_dim,
            config.hidden_dim,
            tau_min_hours=0.25,
            tau_max_hours=48.0,
        )
        self.dynamic_gate = nn.Parameter(torch.tensor(0.0))
        self.freeze_twin()

    def freeze_twin(self) -> None:
        self.physical_twin_core.eval()
        for parameter in self.physical_twin_core.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "PhysicalStream":
        super().train(mode)
        self.freeze_twin()
        return self

    @staticmethod
    def _broadcast_zone_parameter(value: Tensor, reference: Tensor, name: str) -> Tensor:
        parameter = torch.as_tensor(
            value, device=reference.device, dtype=reference.dtype
        )
        if parameter.ndim == 1:
            parameter = parameter.unsqueeze(0)
        if parameter.ndim != 2 or parameter.shape != (reference.shape[0], 4):
            raise ValueError(f"{name} must resolve to [batch,4]")
        while parameter.ndim < reference.ndim:
            parameter = parameter.unsqueeze(1)
        return parameter

    def facility_load_from_compute(
        self,
        compute: Tensor,
        physical_multiplier: Tensor,
        capacity_work_units: Tensor,
        *,
        idle_mw: Tensor | None = None,
        work_mw_per_unit: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the same frozen physical law used by the formal twin.

        ``compute`` may be ``[B,4]`` or contain scenario/time dimensions
        between batch and zone.  The artifact-backed branch reproduces the
        calibrated piecewise-linear curve directly, while the small
        compatibility core retains the documented linear test seam.
        Gradients flow to ``compute`` but never to the frozen twin buffers.
        """

        if compute.ndim < 2 or compute.shape[-1] != self.config.num_sites:
            raise ValueError("compute must end in the four formal zones")
        if not bool(torch.isfinite(compute).all()):
            raise ValueError("compute contains non-finite values")
        multiplier = self._broadcast_zone_parameter(
            physical_multiplier, compute, "physical_multiplier"
        )
        capacity = self._broadcast_zone_parameter(
            capacity_work_units, compute, "capacity_work_units"
        )
        if not bool((capacity > 0.0).all()):
            raise ValueError("capacity_work_units must be strictly positive")

        core = self.physical_twin_core
        if self.formal_artifact_backed:
            required = (
                "it_power_knots_kw",
                "zone_scale",
                "capacity_work_units",
                "_piecewise_facility_kw",
            )
            if any(not hasattr(core, name) for name in required):
                raise RuntimeError(
                    "formal physical twin lacks the differentiable facility-load contract"
                )
            core_capacity = torch.as_tensor(
                core.capacity_work_units, device=compute.device, dtype=compute.dtype
            )
            supplied_capacity = capacity_work_units
            supplied_capacity = torch.as_tensor(
                supplied_capacity, device=compute.device, dtype=compute.dtype
            )
            if supplied_capacity.ndim == 1:
                supplied_capacity = supplied_capacity.unsqueeze(0)
            if not torch.allclose(
                supplied_capacity,
                core_capacity.unsqueeze(0).expand_as(supplied_capacity),
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError(
                    "rollout capacity differs from the frozen physical twin"
                )
            utilization = compute / capacity
            knots = core.it_power_knots_kw.to(
                device=compute.device, dtype=compute.dtype
            )
            it_reference_kw = knots[0] + utilization * (knots[-1] - knots[0])
            if not bool((it_reference_kw > 0.0).all()):
                raise ValueError("compute extrapolates to non-positive IT power")
            facility_reference_kw = core._piecewise_facility_kw(it_reference_kw)
            zone_scale = core.zone_scale.to(
                device=compute.device, dtype=compute.dtype
            )
            while zone_scale.ndim < compute.ndim:
                zone_scale = zone_scale.unsqueeze(0)
            return facility_reference_kw * zone_scale * multiplier / 1000.0

        if idle_mw is None or work_mw_per_unit is None:
            raise ValueError(
                "compatibility physical load requires idle_mw and work_mw_per_unit"
            )
        idle = self._broadcast_zone_parameter(idle_mw, compute, "idle_mw")
        coefficient = self._broadcast_zone_parameter(
            work_mw_per_unit, compute, "work_mw_per_unit"
        )
        return multiplier * (idle + coefficient * compute)

    def forward(
        self,
        physical_static: Tensor,
        physical_dynamic: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if physical_static.ndim == 3:
            if physical_static.shape[1:] != (cfg.horizon, cfg.physical_static_dim):
                raise ValueError("temporal physical_static must be [batch,96,static_dim]")
            first = physical_static[:, 0]
            if not torch.allclose(physical_static, first[:, None].expand_as(physical_static)):
                raise ValueError(
                    "physical_static varies through time; supply proven dynamic values "
                    "through physical_dynamic instead"
                )
            physical_static = first
        if physical_static.ndim != 2 or physical_static.shape[-1] != cfg.physical_static_dim:
            raise ValueError(f"physical_static must be [batch,{cfg.physical_static_dim}]")
        # The artifact-backed core consumes the final four dynamic channels as
        # the causal issue-utilization path.  The compatibility core remains a
        # static-only seam for tensor unit tests and is forbidden by formal
        # preflight/training.
        if self.formal_artifact_backed:
            if physical_dynamic is None:
                raise ValueError(
                    "artifact-backed physical twin requires issue utilization"
                )
            issue_utilization = physical_dynamic[..., 8:12]
            with torch.no_grad():
                core_output = self.physical_twin_core(
                    physical_static, issue_utilization
                )
        else:
            with torch.no_grad():
                core_output = self.physical_twin_core(physical_static)
        if isinstance(core_output, tuple) and len(core_output) == 2:
            core_latent, pue_core = core_output
        elif isinstance(core_output, Mapping):
            core_latent = core_output.get("latent", core_output.get("tokens"))
            pue_core = core_output.get("pue")
            if core_latent is None:
                raise ValueError("physical twin mapping must contain latent or tokens")
            if pue_core is None:
                pue_core = physical_static.new_ones((physical_static.shape[0], cfg.num_sites))
        else:
            core_latent = core_output
            pue_core = physical_static.new_ones((physical_static.shape[0], cfg.num_sites))
        if core_latent.ndim == 2:
            if core_latent.shape != (physical_static.shape[0], cfg.hidden_dim):
                raise ValueError("physical twin core must emit latent [batch,128]")
            core_tokens = core_latent[:, None].expand(-1, cfg.horizon, -1)
        elif core_latent.shape == (
            physical_static.shape[0], cfg.horizon, cfg.hidden_dim
        ):
            core_tokens = core_latent
        else:
            raise ValueError(
                "physical twin core must emit [batch,128] or [batch,96,128]"
            )
        if pue_core.ndim == 2:
            if pue_core.shape != (physical_static.shape[0], cfg.num_sites):
                raise ValueError("physical twin core must emit PUE [batch,4]")
            pue_tokens = pue_core[:, None].expand(-1, cfg.horizon, -1)
        elif pue_core.shape == (
            physical_static.shape[0], cfg.horizon, cfg.num_sites
        ):
            pue_tokens = pue_core
        else:
            raise ValueError(
                "physical twin core must emit PUE [batch,4] or [batch,96,4]"
            )
        bounded_residual = cfg.physical_residual_bound * torch.tanh(
            self.static_adapter(physical_static)
        )
        static_tokens = core_tokens + bounded_residual[:, None]
        if physical_dynamic is None:
            dynamic_tokens = torch.zeros_like(static_tokens)
            dynamic_state = torch.zeros_like(core_latent)
            dynamic_enabled = torch.tensor(False, device=physical_static.device)
        else:
            if physical_dynamic.shape != (
                physical_static.shape[0],
                cfg.horizon,
                cfg.physical_dynamic_dim,
            ):
                raise ValueError(
                    f"physical_dynamic must be [batch,96,{cfg.physical_dynamic_dim}]"
                )
            # The final four values are the exact utilization driver already
            # The final four channels are causal issue-time utilization.  They
            # drive both the immutable physical law and the trainable Liquid
            # residual stream: the former fixes the reference response while
            # the latter may model temporal deviations without ever replacing
            # or mutating the frozen twin.
            dynamic_tokens, dynamic_state = self.dynamic_liquid(
                physical_dynamic
            )
            dynamic_tokens = torch.sigmoid(self.dynamic_gate) * dynamic_tokens
            dynamic_enabled = torch.tensor(True, device=physical_static.device)
        return {
            "tokens": static_tokens + dynamic_tokens,
            "core_tokens": core_tokens,
            "bounded_residual": bounded_residual,
            "dynamic_tokens": dynamic_tokens,
            "dynamic_state": dynamic_state,
            "dynamic_enabled": dynamic_enabled,
            "pue_core": pue_tokens,
        }

    def pue_from_total_workload(
        self, physical_static: Tensor, total_workload: Tensor
    ) -> Tensor:
        """Map the learned total-workload forecast through the frozen twin.

        The formal data contain no independent observed PUE target.  PUE is a
        transferred frozen-twin output, so generating a second unconstrained
        prediction of it is both physically inconsistent and statistically
        noisy.  The total workload remains a learned causal forecast; its
        deterministic capacity-proportional allocation is passed through the
        exact artifact-backed piecewise plant.  Gradients may flow to the
        workload forecast, but the twin has no trainable parameters.
        """

        if not self.formal_artifact_backed:
            raise RuntimeError(
                "physics-linked PUE is available only with the formal artifact core"
            )
        if total_workload.ndim != 3 or total_workload.shape[-1] != 1:
            raise ValueError("total_workload must be [batch,96,1]")
        if physical_static.shape != (total_workload.shape[0], self.config.num_sites):
            raise ValueError("physical_static must be [batch,4]")
        core = self.physical_twin_core
        capacity = core.capacity_work_units.to(
            device=total_workload.device, dtype=total_workload.dtype
        )
        total_capacity = capacity.sum()
        served = total_workload.squeeze(-1).clamp_min(0.0)
        served = torch.minimum(served, total_capacity)
        # Proportional allocation divided by zonal capacity yields one common
        # utilization ratio while retaining the exact four-zone interface.
        utilization = (served / total_capacity)[..., None].expand(
            -1, -1, self.config.num_sites
        )
        return core(physical_static, utilization)["pue"].to(total_workload.dtype)


class CrossDomainFusion(nn.Module):
    """One state-FiLM conditioned four-stream attention block per horizon."""

    def __init__(self, config: R4Config) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.state_film = nn.Linear(dim, 2 * dim)
        self.stream_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, config.fusion_heads, dropout=config.dropout, batch_first=True
        )
        self.gate = nn.Linear(4 * dim + dim, 4)
        self.output_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, config.fusion_ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_ff_dim, dim),
        )

    def forward(
        self, market: Tensor, workload: Tensor, physical: Tensor, state: Tensor
    ) -> dict[str, Tensor]:
        if not (market.shape == workload.shape == physical.shape == state.shape):
            raise ValueError("all fusion streams must have the same [batch,96,128] shape")
        batch, horizon, dim = market.shape
        gamma, beta = self.state_film(state).chunk(2, dim=-1)
        streams = torch.stack(
            (market, workload, physical, state * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta),
            dim=2,
        )
        flat = self.stream_norm(streams).reshape(batch * horizon, 4, dim)
        attended = self.attention(flat, flat, flat, need_weights=False)[0].reshape(
            batch, horizon, 4, dim
        )
        gates = torch.softmax(
            self.gate(torch.cat((*streams.unbind(dim=2), state), dim=-1)), dim=-1
        )
        fused = (attended * gates[..., None]).sum(dim=2)
        fused = self.output_norm(fused + self.ff(self.output_norm(fused)))
        return {"tokens": fused, "stream_gates": gates, "attended_streams": attended}


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden_dim: int) -> None:
        super().__init__()
        # Two residual SwiGLU sub-blocks keep every expert at the registered
        # width 256 while giving the complete R4 model its intended capacity.
        self.norms = nn.ModuleList((nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)))
        self.inputs = nn.ModuleList(
            nn.Linear(hidden_dim, 2 * expert_hidden_dim) for _ in range(2)
        )
        self.outputs = nn.ModuleList(
            nn.Linear(expert_hidden_dim, hidden_dim) for _ in range(2)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for norm, input_projection, output_projection in zip(
            self.norms, self.inputs, self.outputs, strict=True
        ):
            gate, value = input_projection(norm(hidden)).chunk(2, dim=-1)
            hidden = hidden + output_projection(F.silu(gate) * value)
        return hidden - inputs


class TokenTop2MoE(nn.Module):
    """Dropless eight-expert Top-2 token router."""

    def __init__(self, config: R4Config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.router_jitter = float(config.router_jitter)
        self.router = nn.Linear(config.hidden_dim, config.num_experts)
        self.condition = nn.Linear(config.hidden_dim * 3 + 2, config.hidden_dim)
        self.experts = nn.ModuleList(
            SwiGLUExpert(config.hidden_dim, config.expert_hidden_dim)
            for _ in range(config.num_experts)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        tokens: Tensor,
        market_tokens: Tensor,
        state_tokens: Tensor,
        *,
        origin: int | Tensor = 0,
    ) -> dict[str, Tensor]:
        if not (tokens.shape == market_tokens.shape == state_tokens.shape):
            raise ValueError("MoE token/market/state inputs must share shape")
        batch, horizon, dim = tokens.shape
        index = _broadcast_origin(origin, batch, tokens, horizon)
        periods = torch.arange(horizon, device=tokens.device, dtype=tokens.dtype)[None]
        origin_fraction = index.to(tokens.dtype)[:, None] / float(horizon - 1)
        remaining = (periods - index[:, None].to(tokens.dtype)).clamp_min(0.0) / float(
            horizon - 1
        )
        conditioning = torch.cat(
            (
                tokens,
                market_tokens,
                state_tokens,
                origin_fraction[..., None].expand(-1, horizon, -1),
                remaining[..., None],
            ),
            dim=-1,
        )
        routed_input = tokens + self.condition(conditioning)
        clean_logits = self.router(routed_input)
        logits = clean_logits
        if self.training and self.router_jitter > 0.0:
            logits = logits + torch.randn_like(logits) * self.router_jitter
        probabilities = torch.softmax(logits, dim=-1)
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        flat = routed_input.reshape(batch * horizon, dim)
        flat_indices = top_indices.reshape(batch * horizon, self.top_k)
        flat_weights = top_weights.reshape(batch * horizon, self.top_k)
        output = torch.zeros_like(flat)
        selected_outputs: list[Tensor] = []
        for slot in range(self.top_k):
            slot_output = torch.zeros_like(flat)
            for expert_index, expert in enumerate(self.experts):
                rows = torch.nonzero(flat_indices[:, slot] == expert_index, as_tuple=False).flatten()
                if rows.numel() > 0:
                    values = expert(flat.index_select(0, rows))
                    slot_output = slot_output.index_copy(0, rows, values)
            output = output + slot_output * flat_weights[:, slot : slot + 1]
            selected_outputs.append(slot_output)
        output = self.output_norm(tokens + output.reshape(batch, horizon, dim))
        clean_probabilities = torch.softmax(clean_logits, dim=-1)
        expert_load = F.one_hot(top_indices, num_classes=self.num_experts).to(
            tokens.dtype
        ).mean(dim=(0, 1, 2))
        importance = clean_probabilities.mean(dim=(0, 1))
        balance_loss = self.num_experts * (expert_load * importance).sum()
        z_loss = torch.logsumexp(clean_logits, dim=-1).square().mean()
        disagreement = (
            selected_outputs[0] - selected_outputs[1]
        ).square().mean(dim=-1).sqrt().reshape(batch, horizon)
        return {
            "tokens": output,
            "latent": output,
            "router_logits": logits,
            "clean_router_logits": clean_logits,
            "router_probabilities": probabilities,
            "topk_indices": top_indices,
            "topk_weights": top_weights,
            "expert_load": expert_load,
            "router_balance_loss": balance_loss,
            "router_z_loss": z_loss,
            "expert_disagreement": disagreement,
        }


class ZonalForecastDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        da_residual_cap: float = 0.1,
        pv_residual_cap: float = 0.01,
        pue_residual_cap: float = 0.04,
    ) -> None:
        super().__init__()
        self.da_head = nn.Linear(hidden_dim, 4)
        self.dart_head = nn.Linear(hidden_dim, 4)
        self.dart_calibration_log_scale = nn.Parameter(torch.zeros(4))
        self.dart_calibration_bias = nn.Parameter(torch.zeros(4))
        self.workload_head = nn.Linear(hidden_dim, 1)
        self.pv_head = nn.Linear(hidden_dim, 4)
        self.pue_residual_head = nn.Linear(hidden_dim, 4)
        self._residual_cap_values = {
            "da": float(da_residual_cap),
            "pv": float(pv_residual_cap),
            "pue": float(pue_residual_cap),
        }
        self.register_buffer(
            "da_residual_cap", torch.tensor(float(da_residual_cap)), persistent=True
        )
        self.register_buffer(
            "pv_residual_cap", torch.tensor(float(pv_residual_cap)), persistent=True
        )
        self.register_buffer(
            "pue_residual_cap", torch.tensor(float(pue_residual_cap)), persistent=True
        )
        # A new lineage starts as the immutable causal anchors themselves.  A
        # random residual at epoch zero would discard their known skill before
        # the first gradient update and can immediately fail non-inferiority.
        for head in (self.da_head, self.pv_head, self.pue_residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def residual_contract(self) -> Mapping[str, Any]:
        return {
            "schema": "r4_causal_anchor_bounded_residual_v1",
            "formal_cap_source": "train_only_normalization.forecast_residual_caps",
            "formal_cap_statistic": "per_zone_absolute_residual_p95_raw_units",
            "compatibility_fallback_cap_source": (
                "model.forecasts.bounded_neural_residual_raw_unit_fallback_caps"
            ),
            "head_initialization": "zero_weight_zero_bias",
            "compatibility_fallback_caps_raw_units": dict(
                self._residual_cap_values
            ),
            "identity": "rt=da-dart_raw; dart=da-rt",
        }

    def forward(
        self,
        tokens: Tensor,
        pue_core: Tensor,
        *,
        da_anchor: Tensor | None = None,
        pv_anchor: Tensor | None = None,
        residual_caps: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        overrides = {} if residual_caps is None else residual_caps

        def cap(name: str, fallback: Tensor) -> Tensor:
            value = overrides.get(name, fallback)
            result = torch.as_tensor(
                value, device=tokens.device, dtype=tokens.dtype
            )
            if result.ndim == 0:
                return result
            if result.ndim == 1 and result.shape == (4,):
                return result.reshape(1, 1, 4)
            if result.ndim == 2 and result.shape == (tokens.shape[0], 4):
                return result[:, None, :]
            raise ValueError(
                f"forecast residual cap {name!r} must be scalar, [4], or [batch,4]"
            )

        da_cap = cap("da", self.da_residual_cap)
        pv_cap = cap("pv", self.pv_residual_cap)
        pue_cap = cap("pue", self.pue_residual_cap)
        raw_da = self.da_head(tokens)
        da = (
            raw_da
            if da_anchor is None
            else da_anchor + da_cap * torch.tanh(raw_da)
        )
        dart_raw = (
            self.dart_head(tokens) * self.dart_calibration_log_scale.exp()
            + self.dart_calibration_bias
        )
        rt = da - dart_raw
        # Recompute DART last so the public identity (DA - RT) - DART is
        # representable as exact zero in the forward dtype.
        dart = da - rt
        workload = F.softplus(self.workload_head(tokens))
        raw_pv = self.pv_head(tokens)
        pv = (
            F.softplus(raw_pv)
            if pv_anchor is None
            else (
                pv_anchor
                + pv_cap * torch.tanh(raw_pv)
            ).clamp_min(1.0e-8)
        )
        pue = (
            pue_core
            + pue_cap * torch.tanh(self.pue_residual_head(tokens))
        ).clamp_min(1e-4)
        zonal = torch.cat((da, rt, dart, workload, pv, pue), dim=-1)
        return {
            "da": da,
            "rt": rt,
            "dart": dart,
            "workload": workload,
            "pv": pv,
            "pue": pue,
            "zonal": zonal,
            "legacy": zonal_to_legacy(zonal),
            "residual_contract": self.residual_contract,
            "residual_caps": {
                "da": da_cap,
                "pv": pv_cap,
                "pue": pue_cap,
            },
        }


def _broadcast_origin(origin: int | Tensor, batch: int, reference: Tensor, horizon: int) -> Tensor:
    index = torch.as_tensor(origin, device=reference.device)
    if index.ndim == 0:
        index = index.expand(batch)
    if index.shape != (batch,):
        raise ValueError("origin must be scalar or [batch]")
    index = index.long()
    if ((index < 0) | (index >= horizon)).any():
        raise ValueError(f"origin must lie in [0,{horizon - 1}]")
    return index


class HybridCfCTransformerMoER4(nn.Module):
    """Unified R4 representation, uncertainty, risk, and rolling policy."""

    architecture_version = ARCHITECTURE_VERSION

    PHASE_PREFIXES = {
        "representation": (
            "market_encoder.",
            "workload_encoder.",
            "physical_stream.static_adapter.",
            "physical_stream.dynamic_liquid.",
            "physical_stream.dynamic_gate",
            "state_encoder.",
            "cross_fusion.",
            "moe.",
            "forecast_decoder.",
        ),
        "scenario": ("distribution_head.",),
        "distribution": ("distribution_head.",),
        "policy": (
            "state_encoder.",
            "moe.router.",
            "moe.condition.",
            "risk_encoder.",
            "policy.",
        ),
        "joint_policy": (
            "state_encoder.",
            "moe.router.",
            "moe.condition.",
            "distribution_head.condition.",
            "distribution_head.scale_head.",
            "distribution_head.skew_head.",
            "distribution_head.tail_head.",
            "distribution_head.dependence_head.",
            "distribution_head.regime_head.",
            "distribution_head.world_projection.",
            "distribution_head.world_norm.",
            "distribution_head.dart_adapter.",
            "distribution_head.dart_scale_adapter.",
            "distribution_head.dart_tail_adapter.",
            "distribution_head.policy_bridge.",
            "risk_encoder.",
            "policy.",
        ),
        "a1": (
            "forecast_decoder.dart_head.",
            "forecast_decoder.dart_calibration_",
            "distribution_head.dart_adapter.",
            "distribution_head.dart_scale_adapter.",
            "distribution_head.dart_tail_adapter.",
            "distribution_head.policy_bridge.",
            "moe.router.",
            "moe.condition.",
            "risk_encoder.",
        ),
        "a2": ("*",),
    }

    # The archived V5 capability contract is intentionally narrower than the
    # V6 Decision-Focused Distribution Adaptation scope.  Keeping the two
    # allowlists separate prevents merely importing the V6 implementation from
    # changing which tensors an old V5 checkpoint would update.
    V5_A1_PREFIXES = (
        "forecast_decoder.dart_head.",
        "forecast_decoder.dart_calibration_",
        "distribution_head.dart_adapter.",
    )

    def __init__(
        self,
        config: R4Config | Mapping[str, Any] | None = None,
        *,
        physical_twin_core: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, R4Config) else R4Config.from_mapping(config)
        cfg = self.config
        self.architecture_version = cfg.architecture_version
        self.distributional_architecture_revision = (
            cfg.distributional_architecture_revision
        )
        self.market_encoder = MarketTransformerEncoder(cfg)
        self.workload_encoder = ClosedFormLiquidSequence(
            cfg.workload_dim,
            cfg.hidden_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
        )
        self.physical_stream = PhysicalStream(cfg, physical_twin_core)
        self.state_encoder = ClosedFormLiquidSequence(
            cfg.state_dim,
            cfg.hidden_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
        )
        self.cross_fusion = CrossDomainFusion(cfg)
        self.moe = TokenTop2MoE(cfg)
        self.forecast_decoder = ZonalForecastDecoder(
            cfg.hidden_dim,
            da_residual_cap=cfg.forecast_da_residual_cap,
            pv_residual_cap=cfg.forecast_pv_residual_cap,
            pue_residual_cap=cfg.forecast_pue_residual_cap,
        )
        self.distribution_head = DistributionHead(
            cfg.hidden_dim,
            horizon=cfg.horizon,
            hidden_dim=cfg.distribution_hidden_dim,
            dependence_rank=cfg.distribution_dependence_rank,
            regime_count=cfg.distribution_regime_count,
            debug_contract_checks=cfg.distribution_debug_contract_checks,
        )
        self.risk_encoder = RiskEncoder(
            cfg.hidden_dim, hidden_dim=cfg.hidden_dim
        )
        # Parameter-free compatibility/evaluation helper.  It is never read by
        # the V6 Policy and cannot report post-shield formal CVaR by itself.
        self.empirical_risk_evaluator = EquationBasedRisk(
            cvar_alpha=cfg.cvar_alpha
        )
        self.policy = LiquidRollingPolicy(
            cfg.hidden_dim,
            horizon=cfg.horizon,
            num_sites=cfg.num_sites,
            num_heads=cfg.market_heads,
            policy_hidden_dim=cfg.policy_hidden_dim,
            refinement_steps=cfg.policy_refinement_steps,
            max_grid_action=cfg.max_grid_action,
            max_battery_action=cfg.max_battery_action,
            max_compute_action=cfg.max_compute_action,
        )
        self.physical_stream.freeze_twin()

    @property
    def scenario_generator(self) -> DistributionHead:
        """Read-only compatibility name; the module has no fixed scenario axis."""

        return self.distribution_head

    @property
    def risk(self) -> EquationBasedRisk:
        """Archived evaluator alias; V6 active Policy uses ``risk_encoder``."""

        return self.empirical_risk_evaluator

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> "HybridCfCTransformerMoER4":
        super().train(mode)
        self.physical_stream.freeze_twin()
        return self

    def _required(self, batch: Mapping[str, Any], *names: str) -> Tensor:
        for name in names:
            value = batch.get(name)
            if value is not None:
                return value
        raise KeyError(f"missing required model input; expected one of {names}")

    def _model_tensor(self, value: Any) -> Tensor:
        reference = next(self.parameters())
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    def _forecast_anchor_tensors(
        self, batch: Mapping[str, Any], reference: Tensor
    ) -> dict[str, Tensor]:
        """Resolve one validated anchor contract for training and deployment.

        Artifact-backed formal models fail closed.  The compatibility model
        retains legacy fallbacks solely for isolated tensor tests and ablation
        construction; it is rejected by formal preflight independently.
        """

        raw = batch.get("forecast_anchors")
        if raw is None:
            anchors: Mapping[str, Any] = {}
        elif not isinstance(raw, Mapping):
            raise ValueError("forecast_anchors must be a mapping")
        else:
            anchors = raw
        resolved: dict[str, Any] = {
            "da": anchors.get("da"),
            "pv": anchors.get("pv"),
        }
        if self.physical_stream.formal_artifact_backed:
            missing = [name for name, value in resolved.items() if value is None]
            if missing:
                raise KeyError(
                    "formal R4 requires explicit causal forecast_anchors for "
                    + ", ".join(missing)
                )
        else:
            if resolved["da"] is None and batch.get("market_raw") is not None:
                resolved["da"] = batch["market_raw"][..., :4]
            if resolved["pv"] is None and batch.get("issue_pv_raw") is not None:
                resolved["pv"] = batch["issue_pv_raw"]

        output: dict[str, Tensor] = {}
        expected = (reference.shape[0], self.config.horizon, self.config.num_sites)
        for name, value in resolved.items():
            if value is None:
                continue
            tensor = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype
            )
            if tensor.shape != expected:
                raise ValueError(
                    f"forecast_anchors.{name} must be {expected}, got {tuple(tensor.shape)}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"forecast_anchors.{name} contains non-finite values")
            output[name] = tensor
        return output

    def _forecast_residual_cap_tensors(
        self, batch: Mapping[str, Any], reference: Tensor
    ) -> dict[str, Tensor]:
        raw = batch.get("forecast_residual_caps")
        if raw is None:
            caps: Mapping[str, Any] = {}
        elif not isinstance(raw, Mapping):
            raise ValueError("forecast_residual_caps must be a mapping")
        else:
            caps = raw
        defaults = {
            "da": self.forecast_decoder.da_residual_cap,
            "pv": self.forecast_decoder.pv_residual_cap,
            "pue": self.forecast_decoder.pue_residual_cap,
        }
        if self.physical_stream.formal_artifact_backed:
            missing = [name for name in defaults if caps.get(name) is None]
            if missing:
                raise KeyError(
                    "formal R4 requires train-frozen forecast_residual_caps for "
                    + ", ".join(missing)
                )
        output: dict[str, Tensor] = {}
        for name, fallback in defaults.items():
            value = caps.get(name, fallback)
            tensor = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype
            )
            if tensor.ndim == 1 and tensor.shape == (self.config.num_sites,):
                pass
            elif tensor.ndim == 2 and tensor.shape == (
                reference.shape[0], self.config.num_sites
            ):
                pass
            elif tensor.ndim == 0 and not self.physical_stream.formal_artifact_backed:
                pass
            else:
                raise ValueError(
                    f"forecast_residual_caps.{name} must be [4] or [batch,4]"
                )
            if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0.0).all()):
                raise ValueError(
                    f"forecast_residual_caps.{name} must be finite and positive"
                )
            output[name] = tensor
        return output

    def _scenario_residual_cap_tensor(
        self, batch: Mapping[str, Any], reference: Tensor
    ) -> Tensor | None:
        """Validate the frozen raw-unit Scenario envelope, if supplied.

        Formal phase setup installs the same [21] contract directly on the
        generator and supplies the expanded batch key fail-closed.  ``None``
        remains available only for direct compatibility/unit-test models.
        """

        raw = batch.get("scenario_residual_caps_zonal")
        if raw is None:
            return None
        caps = torch.as_tensor(
            raw, device=reference.device, dtype=reference.dtype
        )
        accepted = {
            (21,),
            (self.config.horizon, 21),
            (reference.shape[0], self.config.horizon, 21),
        }
        if tuple(caps.shape) not in accepted:
            raise ValueError(
                "scenario_residual_caps_zonal must be [21], [96,21], or "
                "[batch,96,21]"
            )
        if not bool(torch.isfinite(caps).all()) or not bool((caps > 0.0).all()):
            raise ValueError(
                "scenario_residual_caps_zonal must be finite and strictly positive"
            )
        return caps

    @staticmethod
    def _distribution_support_tensors(
        batch: Mapping[str, Any],
    ) -> tuple[Any | None, Any | None]:
        raw = batch.get("distribution_support")
        support = raw if isinstance(raw, Mapping) else {}
        lower = support.get(
            "lower_zonal", support.get("lower", batch.get("distribution_support_lower_zonal"))
        )
        upper = support.get(
            "upper_zonal", support.get("upper", batch.get("distribution_support_upper_zonal"))
        )
        return lower, upper

    def _workload_features(self, value: Any) -> Tensor:
        workload = self._model_tensor(value)
        if workload.ndim != 3 or workload.shape[1] != self.config.horizon:
            raise ValueError("workload must be a [batch,96,features] tensor")
        if workload.shape[-1] == self.config.workload_dim:
            return workload
        if workload.shape[-1] == 2 and self.config.workload_dim == 5:
            mean, std = workload.unbind(dim=-1)
            # Explicit compatibility bridge for an older mean/std day object.
            # The formal data path should supply empirical p10/p50/p90 directly.
            return torch.stack(
                (mean, std, mean - 1.2815516 * std, mean, mean + 1.2815516 * std),
                dim=-1,
            )
        raise ValueError(
            f"workload must end in {self.config.workload_dim} formal features "
            "(or the registered two-feature compatibility input)"
        )

    def _state_tensor(
        self,
        value: Tensor | Mapping[str, Tensor],
        reference: Tensor,
        *,
        origin: int | Tensor = 0,
    ) -> Tensor:
        cfg = self.config
        if isinstance(value, Mapping):
            batch = reference.shape[0]
            index = _broadcast_origin(origin, batch, reference, cfg.horizon)

            def snapshot(name: str, width: int, *aliases: str) -> Tensor:
                item = next(
                    (value[key] for key in (name, *aliases) if value.get(key) is not None),
                    None,
                )
                if item is None:
                    return reference.new_zeros((batch, width))
                tensor = torch.as_tensor(
                    item, device=reference.device, dtype=reference.dtype
                )
                if tensor.ndim == 0:
                    tensor = tensor.reshape(1, 1).expand(batch, width)
                elif tensor.ndim == 1:
                    if tensor.numel() == width:
                        tensor = tensor.unsqueeze(0).expand(batch, -1)
                    elif tensor.numel() == batch and width == 1:
                        tensor = tensor.unsqueeze(-1)
                elif tensor.ndim == 2 and tensor.numel() == width and tensor.shape[0] != batch:
                    tensor = tensor.reshape(1, width).expand(batch, -1)
                if tensor.ndim == 3 and tensor.shape[1] == cfg.horizon:
                    tensor = tensor[torch.arange(batch, device=reference.device), index]
                if tensor.ndim > 2:
                    tensor = tensor.flatten(start_dim=1)
                if tensor.shape == (batch, 1) and width > 1:
                    tensor = tensor.expand(batch, width)
                if tensor.shape != (batch, width):
                    raise ValueError(
                        f"rolling state {name!r} must resolve to [batch,{width}], "
                        f"got {tuple(tensor.shape)}"
                    )
                return tensor

            result = torch.cat(
                (
                    snapshot("current_soc", 4, "soc"),
                    snapshot("backlog", 4),
                    snapshot("deadline_pressure", 4),
                    snapshot("previous_grid", 4),
                    snapshot("previous_battery", 4),
                    snapshot("previous_compute", 4),
                    snapshot("previous_migration", 16),
                    snapshot("rt_innovation", 4, "revealed_rt"),
                    snapshot("workload_innovation", 1, "revealed_workload"),
                    snapshot("pv_innovation", 4, "revealed_pv"),
                ),
                dim=-1,
            )
        else:
            result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if result.ndim != 2:
            raise ValueError("rolling state snapshot must be [batch,features]")
        if result.shape[-1] > cfg.state_dim:
            raise ValueError(f"rolling state width exceeds configured {cfg.state_dim}")
        if result.shape[-1] < cfg.state_dim:
            result = F.pad(result, (0, cfg.state_dim - result.shape[-1]))
        return result

    def prepare_issue_context(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Cache only origin-invariant, issue-time-safe stream encodings."""

        market = self._model_tensor(
            self._required(batch, "market", "market_features", "market_inputs")
        )
        workload = self._workload_features(
            self._required(batch, "workload", "workload_features", "workload_inputs")
        )
        physical_value = self._model_tensor(
            self._required(batch, "physical_static", "physical", "physical_features")
        )
        physical_dynamic_value = batch.get("physical_dynamic")
        if "physical_static" not in batch and physical_value.ndim == 3:
            if physical_value.shape[1:] != (
                self.config.horizon,
                self.config.physical_dynamic_dim,
            ):
                raise ValueError("physical compatibility input must be [batch,96,12]")
            physical_static = physical_value[:, 0, : self.config.physical_static_dim]
            repeated = physical_value[..., : self.config.physical_static_dim]
            if not torch.allclose(repeated, physical_static[:, None].expand_as(repeated)):
                raise ValueError("the first four physical compatibility channels must be static")
            physical_dynamic = physical_value
        else:
            physical_static = physical_value
            physical_dynamic = (
                None
                if physical_dynamic_value is None
                else self._model_tensor(physical_dynamic_value)
            )
        historical = batch.get("historical_market")
        historical = None if historical is None else self._model_tensor(historical)
        market_output = self.market_encoder(
            market,
            historical_market=historical,
            historical_market_available=batch.get("historical_market_available"),
            day_of_week=batch.get("day_of_week"),
            month=batch.get("month"),
            holiday=batch.get("holiday"),
            dst=batch.get("dst"),
        )
        workload_tokens, workload_state = self.workload_encoder(
            workload, dt=batch.get("workload_dt")
        )
        physical_output = self.physical_stream(
            physical_static, physical_dynamic
        )
        forecast_anchors = self._forecast_anchor_tensors(batch, market)
        forecast_residual_caps = self._forecast_residual_cap_tensors(batch, market)
        scenario_residual_caps_zonal = self._scenario_residual_cap_tensor(
            batch, market
        )
        support_lower_zonal, support_upper_zonal = (
            self._distribution_support_tensors(batch)
        )
        return {
            "architecture_version": self.architecture_version,
            "market_tokens": market_output["tokens"],
            "market_context_tokens": market_output["context_tokens"],
            "market_history_available": market_output["history_available"],
            "workload_tokens": workload_tokens,
            "workload_state": workload_state,
            "physical_tokens": physical_output["tokens"],
            "physical": physical_output,
            "physical_static": physical_static,
            "pue_core": physical_output["pue_core"],
            "forecast_anchors": forecast_anchors,
            "forecast_anchor_metadata": batch.get("forecast_anchor_metadata", {}),
            "forecast_residual_caps": forecast_residual_caps,
            "scenario_residual_caps_zonal": scenario_residual_caps_zonal,
            "distribution_support_lower_zonal": support_lower_zonal,
            "distribution_support_upper_zonal": support_upper_zonal,
            "forecast_residual_cap_metadata": batch.get(
                "forecast_residual_cap_metadata", {}
            ),
            "batch": batch,
        }

    def _decode_forecasts(
        self, context: Mapping[str, Any], tokens: Tensor
    ) -> dict[str, Any]:
        anchors = context["forecast_anchors"]
        forecasts = self.forecast_decoder(
            tokens,
            context["pue_core"],
            da_anchor=anchors.get("da"),
            pv_anchor=anchors.get("pv"),
            residual_caps=context["forecast_residual_caps"],
        )
        if self.physical_stream.formal_artifact_backed:
            # Formal PUE is not an independent free prediction.  It is the
            # frozen physical response to the model's causal workload forecast.
            pue = self.physical_stream.pue_from_total_workload(
                context["physical_static"], forecasts["workload"]
            )
            zonal = torch.cat(
                (
                    forecasts["da"],
                    forecasts["rt"],
                    forecasts["dart"],
                    forecasts["workload"],
                    forecasts["pv"],
                    pue,
                ),
                dim=-1,
            )
            forecasts = {
                **forecasts,
                "pue": pue,
                "zonal": zonal,
                "legacy": zonal_to_legacy(zonal),
                "pue_contract": (
                    "learned_total_workload_through_frozen_piecewise_twin"
                ),
            }
        return forecasts

    def _fuse_and_decode(
        self,
        context: Mapping[str, Any],
        state_tokens: Tensor,
        *,
        origin: int | Tensor,
        revealed: Mapping[str, Tensor] | Tensor | None = None,
        distribution_sample_count: int | None = None,
        distribution_generator: torch.Generator | None = None,
        distribution_seed: int | None = None,
    ) -> dict[str, Any]:
        fusion = self.cross_fusion(
            context["market_tokens"],
            context["workload_tokens"],
            context["physical_tokens"],
            state_tokens,
        )
        routing = self.moe(
            fusion["tokens"],
            context["market_tokens"],
            state_tokens,
            origin=origin,
        )
        forecasts = self._decode_forecasts(context, routing["tokens"])
        revealed_zonal = None
        revealed_mask = None
        if isinstance(revealed, Mapping):
            revealed_zonal = revealed.get("zonal")
            if revealed_zonal is None:
                revealed_zonal = revealed.get("revealed_zonal")
            revealed_mask = revealed.get("mask")
            if revealed_mask is None:
                revealed_mask = revealed.get("revealed_mask")
        elif revealed is not None:
            revealed_zonal = revealed
            revealed_mask = context["batch"].get("revealed_mask")
        if revealed_mask is not None:
            mask_tensor = torch.as_tensor(
                revealed_mask, device=routing["tokens"].device
            ).bool()
            origin_index = _broadcast_origin(
                origin,
                routing["tokens"].shape[0],
                routing["tokens"],
                self.config.horizon,
            )
            periods = torch.arange(
                self.config.horizon, device=mask_tensor.device
            )[None]
            if bool((mask_tensor & (periods > origin_index[:, None])).any()):
                raise ValueError("revealed_mask exposes a future value")
        world = self.distribution_head(
            routing["tokens"],
            forecasts["zonal"],
            origin=origin,
            residual_caps_zonal=context["scenario_residual_caps_zonal"],
            support_lower_zonal=context["distribution_support_lower_zonal"],
            support_upper_zonal=context["distribution_support_upper_zonal"],
        )
        distribution = world.as_dict()
        compatibility_scenarios: dict[str, Any] | None = None
        if distribution_sample_count is not None:
            sampled = self.distribution_head.sample(
                world,
                int(distribution_sample_count),
                generator=distribution_generator,
                seed=distribution_seed,
                revealed_zonal=revealed_zonal,
                revealed_mask=revealed_mask,
            )
            distribution.update(sampled)
            if int(distribution_sample_count) == SCENARIO_COUNT:
                compatibility_scenarios = {
                    **sampled,
                    "compatibility_view": "legacy_N20_only",
                }
        result = {
            "fusion": fusion,
            "moe": routing,
            "token_features": routing["tokens"],
            "forecasts": forecasts,
            "forecast_zonal": forecasts["zonal"],
            "forecast_legacy": forecasts["legacy"],
            "distribution": distribution,
        }
        if compatibility_scenarios is not None:
            result["scenarios"] = compatibility_scenarios
        return result

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        distribution_sample_count: int | None = None,
        distribution_generator: torch.Generator | None = None,
        distribution_seed: int | None = None,
    ) -> dict[str, Any]:
        context = self.prepare_issue_context(batch)
        state = self._required(batch, "state_sequence", "state", "state_features")
        state = self._model_tensor(state)
        if state.ndim == 2:
            state = state[:, None].expand(-1, self.config.horizon, -1)
        if state.ndim != 3 or state.shape[:2] != (
            context["market_tokens"].shape[0], self.config.horizon
        ):
            raise ValueError("state_sequence must be [batch,96,state_features]")
        if state.shape[-1] > self.config.state_dim:
            raise ValueError(f"state_sequence width exceeds configured {self.config.state_dim}")
        if state.shape[-1] < self.config.state_dim:
            state = F.pad(state, (0, self.config.state_dim - state.shape[-1]))
        origin = batch.get("origin", 0)
        index = _broadcast_origin(origin, state.shape[0], state, self.config.horizon)
        periods = torch.arange(self.config.horizon, device=state.device)[None]
        causal_state = state * (periods <= index[:, None]).unsqueeze(-1).to(state.dtype)
        state_tokens, state_memory = self.state_encoder(
            causal_state, dt=batch.get("state_dt")
        )
        revealed = batch.get("revealed")
        if revealed is None and batch.get("revealed_zonal") is not None:
            revealed = {
                "zonal": batch["revealed_zonal"],
                "mask": batch.get("revealed_mask"),
            }
        # Archived V5 callers retain an N=20 compatibility view.  Formal V6
        # inference has no implicit Monte Carlo work; sampling is opt-in.
        sample_count = distribution_sample_count
        if (
            sample_count is None
            and self.architecture_version == ARCHITECTURE_VERSION
        ):
            sample_count = SCENARIO_COUNT
        output = self._fuse_and_decode(
            context,
            state_tokens,
            origin=origin,
            revealed=revealed,
            distribution_sample_count=sample_count,
            distribution_generator=distribution_generator,
            distribution_seed=distribution_seed,
        )
        selected_state = state_tokens.gather(
            1, index[:, None, None].expand(-1, 1, self.config.hidden_dim)
        ).squeeze(1)
        world = output["distribution"]["world_latent"]
        risk = self.risk_encoder(world, selected_state)
        revealed_context = world.token_features.gather(
            1, index[:, None, None].expand(-1, 1, self.config.hidden_dim)
        ).squeeze(1)
        policy = self.policy(
            output["token_features"],
            selected_state,
            output["forecast_zonal"],
            origin=origin,
            world_latent=world,
            risk_features=risk,
            rolling_state_hidden=selected_state,
            revealed_context=revealed_context,
            issued_da_schedule=batch.get("issued_da_schedule", batch.get("frozen_day_ahead")),
            policy_memory=batch.get("policy_memory"),
        )
        reported_risk: Mapping[str, Any] = risk
        if self.architecture_version == ARCHITECTURE_VERSION:
            compatibility = output.get("scenarios")
            if isinstance(compatibility, Mapping):
                empirical = self.empirical_risk_evaluator(
                    compatibility["particles_zonal"],
                    compatibility["weights"],
                    day_ahead=policy["day_ahead"],
                    battery_plan=policy["battery_plan"],
                    compute_plan=policy["compute_plan"],
                )
                reported_risk = {
                    **risk,
                    **empirical,
                    "neural_risk": risk,
                    "formal_cvar_reported": False,
                }
        output.update(
            {
                "architecture_version": self.architecture_version,
                "distributional_architecture_revision": (
                    self.distributional_architecture_revision
                ),
                "state_sequence": state_tokens,
                "state_memory": state_memory,
                "policy": policy,
                "actions": policy,
                "risk": reported_risk,
                "parameter_count": self.parameter_count,
            }
        )
        return output

    def sample_distribution(
        self,
        batch_or_context: Mapping[str, Any] | DistributionalWorldLatent,
        N: int,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        revealed_zonal: Tensor | None = None,
        revealed_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        """Sample an arbitrary budget without changing the deterministic graph.

        ``batch_or_context`` may be a world latent, a model output, the public
        ``outputs['distribution']`` mapping, a prepared issue context, or a raw
        model batch.  Reusing an already encoded world is the preferred formal
        evaluation path because it makes N-invariance explicit.
        """

        world: DistributionalWorldLatent | None = None
        source: Mapping[str, Any] | None = None
        if isinstance(batch_or_context, DistributionalWorldLatent):
            world = batch_or_context
        elif isinstance(batch_or_context, Mapping):
            source = batch_or_context
            direct = source.get("world_latent")
            if isinstance(direct, DistributionalWorldLatent):
                world = direct
            distribution = source.get("distribution")
            if world is None and isinstance(distribution, Mapping):
                candidate = distribution.get("world_latent")
                if isinstance(candidate, DistributionalWorldLatent):
                    world = candidate
            if world is None:
                raw_batch = source.get("batch") if "market_tokens" in source else source
                if not isinstance(raw_batch, Mapping):
                    raise ValueError("prepared context does not retain its source batch")
                encoded = self.forward(raw_batch)
                world = encoded["distribution"]["world_latent"]
        else:
            raise TypeError("batch_or_context must be a mapping or world latent")

        if revealed_zonal is None and source is not None:
            revealed = source.get("revealed")
            if isinstance(revealed, Mapping):
                revealed_zonal = revealed.get("zonal", revealed.get("revealed_zonal"))
                revealed_mask = revealed.get("mask", revealed.get("revealed_mask"))
            elif source.get("revealed_zonal") is not None:
                revealed_zonal = source.get("revealed_zonal")
                revealed_mask = source.get("revealed_mask")
        return self.distribution_head.sample(
            world,
            N,
            generator=generator,
            seed=seed,
            revealed_zonal=revealed_zonal,
            revealed_mask=revealed_mask,
        )

    def forward_training_rollout(
        self, batch: Mapping[str, Any], horizon: int
    ) -> dict[str, Any]:
        """Differentiable K-origin analytic-projection rollout for training.

        Issue-time encodings and twenty joint scenarios are built once.  Each
        origin then queries the plan with the recurrent state produced from the
        *previous projected actions*.  This is the training surrogate for the
        deterministic hard shield; the formal phase-end gate still executes
        and independently validates the hard shield itself.
        """

        steps = int(horizon)
        if not 1 <= steps <= self.config.horizon:
            raise ValueError("training rollout horizon must lie in [1,96]")
        rollout = batch.get("rollout_targets")
        if not isinstance(rollout, Mapping):
            raise ValueError("forward_training_rollout requires rollout_targets")
        context = self.prepare_issue_context(batch)
        reference = context["market_tokens"]
        batch_size = reference.shape[0]

        def required(name: str) -> Tensor:
            value = rollout.get(name)
            if value is None:
                raise ValueError(f"rollout_targets missing {name!r}")
            return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

        da_price = required("da_price")
        rt_price = required("rt_price")
        workload = required("workload")
        pv_available = required("pv")
        multiplier = required("physical_multiplier")
        idle = required("idle_mw")
        work_coefficient = required("work_mw_per_unit")
        work_capacity = required("capacity_work_units")
        grid_limit = required("grid_limit_mw")
        battery_power = required("battery_power_mw")
        battery_energy = required("battery_energy_mwh")
        soc = required("initial_soc")
        expected_shapes = {
            "da_price": (batch_size, 4, 96),
            "rt_price": (batch_size, 4, 96),
            "workload": (batch_size, 96),
            "pv": (batch_size, 4, 96),
            "physical_multiplier": (batch_size, 4),
        }
        observed = {
            "da_price": tuple(da_price.shape),
            "rt_price": tuple(rt_price.shape),
            "workload": tuple(workload.shape),
            "pv": tuple(pv_available.shape),
            "physical_multiplier": tuple(multiplier.shape),
        }
        for name, shape in expected_shapes.items():
            if observed[name] != shape:
                raise ValueError(f"rollout target {name} must be {shape}")

        soc = torch.minimum(soc.clamp_min(0.0), battery_energy)
        queue_age = reference.new_zeros((batch_size, 5))
        previous_grid = reference.new_zeros((batch_size, 4))
        previous_battery = reference.new_zeros((batch_size, 4))
        previous_compute = reference.new_zeros((batch_size, 4))
        liquid_state: Tensor | None = None
        issued_da: Tensor | None = None
        scenario: Mapping[str, Any] | None = None
        first_world: DistributionalWorldLatent | None = None
        latest_neural_risk: Mapping[str, Any] | None = None
        first_output: dict[str, Any] | None = None
        battery_path: list[Tensor] = []
        compute_path: list[Tensor] = []
        grid_path: list[Tensor] = []
        raw_violation_path: list[Tensor] = []
        intervention_path: list[Tensor] = []
        sla_path: list[Tensor] = []
        soc_path: list[Tensor] = [soc]
        rolling_state_path: list[Tensor] = []

        eta_charge = 0.95
        eta_discharge = 0.95
        interval_hours = 0.25
        target = batch.get("targets", {})
        rt_target = torch.as_tensor(
            target["rt"], device=reference.device, dtype=reference.dtype
        )
        workload_target = torch.as_tensor(
            target["workload"], device=reference.device, dtype=reference.dtype
        )
        pv_target = torch.as_tensor(
            target["pv"], device=reference.device, dtype=reference.dtype
        )
        market_raw = torch.as_tensor(
            batch["market_raw"], device=reference.device, dtype=reference.dtype
        )
        issue_workload_raw = torch.as_tensor(
            batch["issue_workload_raw"], device=reference.device, dtype=reference.dtype
        )
        issue_pv_raw = torch.as_tensor(
            batch["issue_pv_raw"], device=reference.device, dtype=reference.dtype
        )
        for origin in range(steps):
            # Match the deployed runtime exactly: the current origin has been
            # revealed before the action is proposed, and its arrival is
            # already present in the EDF queue seen by the policy.
            rt_innovation = rt_target[:, origin] - market_raw[:, origin, 4:8]
            workload_innovation = (
                workload_target[:, origin] - issue_workload_raw[:, origin, None]
            )
            pv_innovation = pv_target[:, origin] - issue_pv_raw[:, origin]
            arrival = workload[:, origin]
            queue_available = queue_age.clone()
            queue_available[:, -1] = queue_available[:, -1] + arrival
            backlog_total = queue_available.sum(-1, keepdim=True)
            backlog = backlog_total.expand(-1, 4) / 4.0
            deadline_ratio = torch.where(
                backlog_total > 0.0,
                queue_available[:, 0:1] / backlog_total.clamp_min(1.0e-8),
                torch.zeros_like(backlog_total),
            )
            deadline_mass = deadline_ratio.expand(-1, 4)
            rolling_state = torch.cat(
                (
                    soc / battery_energy.clamp_min(1.0e-6),
                    backlog,
                    deadline_mass,
                    previous_grid,
                    previous_battery,
                    previous_compute,
                    reference.new_zeros((batch_size, 16)),
                    rt_innovation,
                    workload_innovation,
                    pv_innovation,
                ),
                dim=-1,
            )
            rolling_state_path.append(rolling_state)
            liquid_state = self.state_encoder.cell.step(
                self.state_encoder.input_norm(rolling_state), liquid_state, dt=0.25
            )
            state_token = self.state_encoder.output_norm(liquid_state)
            state_tokens = state_token[:, None].expand(-1, self.config.horizon, -1)
            fusion = self.cross_fusion(
                context["market_tokens"],
                context["workload_tokens"],
                context["physical_tokens"],
                state_tokens,
            )
            routing = self.moe(
                fusion["tokens"], context["market_tokens"], state_tokens, origin=origin
            )
            forecasts = self._decode_forecasts(context, routing["tokens"])
            world = self.distribution_head(
                routing["tokens"],
                forecasts["zonal"],
                origin=origin,
                residual_caps_zonal=context["scenario_residual_caps_zonal"],
                support_lower_zonal=context["distribution_support_lower_zonal"],
                support_upper_zonal=context["distribution_support_upper_zonal"],
            )
            latest_neural_risk = self.risk_encoder(world, state_token)
            if scenario is None:
                default_train_samples = (
                    SCENARIO_COUNT
                    if self.architecture_version == ARCHITECTURE_VERSION
                    else 32
                )
                train_samples = int(
                    batch.get("distribution_train_samples", default_train_samples)
                )
                scenario = self.distribution_head.sample(world, train_samples)
                first_world = world
                distribution = world.as_dict()
                distribution.update(scenario)
                first_output = {
                    "fusion": fusion,
                    "moe": routing,
                    "token_features": routing["tokens"],
                    "forecasts": forecasts,
                    "forecast_zonal": forecasts["zonal"],
                    "forecast_legacy": forecasts["legacy"],
                    "distribution": distribution,
                }
                if train_samples == SCENARIO_COUNT:
                    first_output["scenarios"] = {
                        **scenario,
                        "compatibility_view": "legacy_N20_only",
                    }
            assert scenario is not None
            revealed_context = world.token_features[:, origin]
            policy = self.policy(
                routing["tokens"],
                state_token,
                forecasts["zonal"],
                origin=origin,
                world_latent=world,
                risk_features=latest_neural_risk,
                rolling_state_hidden=state_token,
                revealed_context=revealed_context,
                issued_da_schedule=issued_da,
            )
            if issued_da is None:
                issued_da = policy["day_ahead"]

            raw_compute = policy["compute"]
            max_work_power = (
                (grid_limit + pv_available[:, :, origin] + battery_power)
                / multiplier.clamp_min(1.0e-6)
                - idle
            ) / work_coefficient.clamp_min(1.0e-6)
            feasible_work_cap = torch.minimum(work_capacity, max_work_power.clamp_min(0.0))
            requested_total = raw_compute.clamp_min(0.0).sum(-1)
            service_total = torch.minimum(
                requested_total,
                torch.minimum(feasible_work_cap.sum(-1), queue_available.sum(-1)),
            )
            site_weights = torch.softmax(raw_compute, dim=-1)
            projected_compute = torch.minimum(
                site_weights * service_total[:, None], feasible_work_cap
            )
            # Redistribute once after per-site caps so total service remains as
            # close as possible to the differentiable request.
            residual_service = (service_total - projected_compute.sum(-1)).clamp_min(0.0)
            spare = (feasible_work_cap - projected_compute).clamp_min(0.0)
            projected_compute = projected_compute + spare / spare.sum(-1, keepdim=True).clamp_min(1e-6) * residual_service[:, None]
            service_left = projected_compute.sum(-1)
            remaining_buckets: list[Tensor] = []
            for bucket in range(5):
                served_bucket = torch.minimum(service_left, queue_available[:, bucket])
                remaining_buckets.append(queue_available[:, bucket] - served_bucket)
                service_left = (service_left - served_bucket).clamp_min(0.0)
            remaining_queue = torch.stack(remaining_buckets, dim=-1)
            sla = remaining_queue[:, 0]
            queue_age_next = torch.cat(
                (remaining_queue[:, 1:], remaining_queue.new_zeros((batch_size, 1))),
                dim=-1,
            )

            load = self.physical_stream.facility_load_from_compute(
                projected_compute,
                multiplier,
                work_capacity,
                idle_mw=idle,
                work_mw_per_unit=work_coefficient,
            )
            raw_signed_battery = policy["battery"]
            discharge_cap = torch.minimum(
                battery_power,
                eta_discharge * soc / interval_hours,
            )
            charge_cap = torch.minimum(
                battery_power,
                (battery_energy - soc) / (interval_hours * eta_charge),
            )
            projected_battery = torch.maximum(
                torch.minimum(raw_signed_battery, charge_cap), -discharge_cap
            )
            charge = projected_battery.clamp_min(0.0)
            discharge = (-projected_battery).clamp_min(0.0)
            soc_next = (
                soc
                + interval_hours * (eta_charge * charge - discharge / eta_discharge)
            ).clamp_min(0.0)
            soc_next = torch.minimum(soc_next, battery_energy)
            net = load + projected_battery
            pv_used = torch.minimum(pv_available[:, :, origin].clamp_min(0.0), net.clamp_min(0.0))
            raw_grid = (net - pv_used).clamp_min(0.0)
            projected_grid = torch.minimum(raw_grid, grid_limit)
            # Any residual grid excess is recorded as a differentiable envelope
            # penalty.  The work cap above makes it zero for admissible inputs.
            violation = (
                (raw_compute - projected_compute).abs().sum(-1)
                + (raw_signed_battery - projected_battery).abs().sum(-1)
                + (raw_grid - projected_grid).abs().sum(-1)
            )
            battery_path.append(projected_battery)
            compute_path.append(projected_compute)
            grid_path.append(projected_grid)
            raw_violation_path.append(violation)
            intervention_path.append(violation)
            sla_path.append(sla.sum(-1))
            soc_path.append(soc_next)
            soc = soc_next
            queue_age = queue_age_next
            previous_grid = projected_grid
            previous_battery = projected_battery
            previous_compute = projected_compute

        assert (
            issued_da is not None
            and first_output is not None
            and scenario is not None
            and first_world is not None
            and latest_neural_risk is not None
        )
        battery_tensor = torch.stack(battery_path, dim=-1)
        compute_tensor = torch.stack(compute_path, dim=-1)
        grid_tensor = torch.stack(grid_path, dim=-1)
        da_used = issued_da[..., :steps]
        realized_cost = interval_hours * (
            da_used * da_price[..., :steps]
            + (grid_tensor - da_used) * rt_price[..., :steps]
        ).sum(dim=(-1, -2))
        total_arrival = workload[:, :steps].sum(-1).clamp_min(1.0e-6)
        normalized_sla = torch.stack(sla_path, -1).sum(-1) / total_arrival
        battery_degradation = interval_hours * battery_tensor.abs().sum(dim=(-1, -2))
        raw_violation = torch.stack(raw_violation_path, -1).mean(-1)
        intervention = torch.stack(intervention_path, -1).mean(-1)

        particles = scenario["particles_zonal"][:, :, :steps]
        particle_da_price = particles[..., 0:4].permute(0, 1, 3, 2)
        particle_rt_price = particles[..., 4:8].permute(0, 1, 3, 2)
        particle_workload = particles[..., 12:13].expand(-1, -1, -1, 4)
        particle_pv = particles[..., 13:17]
        particle_work = torch.minimum(
            compute_tensor.transpose(1, 2)[:, None],
            particle_workload,
        )
        particle_load = self.physical_stream.facility_load_from_compute(
            particle_work,
            multiplier,
            work_capacity,
            idle_mw=idle,
            work_mw_per_unit=work_coefficient,
        )
        particle_battery = battery_tensor.transpose(1, 2)[:, None]
        particle_net = particle_load + particle_battery
        particle_grid = (particle_net - particle_pv).clamp_min(0.0)
        particle_grid = torch.minimum(
            particle_grid, grid_limit[:, None, None]
        )
        particle_grid = particle_grid.permute(0, 1, 3, 2)
        particle_costs = interval_hours * (
            da_used[:, None] * particle_da_price
            + (particle_grid - da_used[:, None]) * particle_rt_price
        ).sum(dim=(-1, -2))
        particle_count = int(particles.shape[1])
        particle_queue = particles.new_zeros((batch_size, particle_count, 5))
        particle_sla_steps: list[Tensor] = []
        for origin in range(steps):
            particle_queue = particle_queue.clone()
            particle_queue[:, :, -1] = (
                particle_queue[:, :, -1] + particles[:, :, origin, 12]
            )
            service_left = particle_work[:, :, origin].sum(-1)
            remaining: list[Tensor] = []
            for bucket in range(5):
                served_bucket = torch.minimum(service_left, particle_queue[:, :, bucket])
                remaining.append(particle_queue[:, :, bucket] - served_bucket)
                service_left = (service_left - served_bucket).clamp_min(0.0)
            remaining_particle_queue = torch.stack(remaining, dim=-1)
            particle_sla_steps.append(remaining_particle_queue[:, :, 0])
            particle_queue = torch.cat(
                (
                    remaining_particle_queue[:, :, 1:],
                    remaining_particle_queue.new_zeros(
                        (batch_size, particle_count, 1)
                    ),
                ),
                dim=-1,
            )
        particle_sla = torch.stack(particle_sla_steps, dim=-1).sum(-1)
        particle_costs = particle_costs + 1000.0 * particle_sla
        weights = scenario["weights"]
        expected_cost = (weights * particle_costs).sum(-1)
        risk = {
            "scenario_costs": particle_costs,
            "expected_cost": expected_cost,
            "scenario_expected_cost": expected_cost,
            "neural_risk": latest_neural_risk,
            "formal_cvar_reported": False,
        }
        training_rollout = {
            "realized_settlement_cost": realized_cost,
            "normalized_sla": normalized_sla,
            "battery_degradation": battery_degradation,
            "raw_envelope_violation": raw_violation,
            "corrective_intervention": intervention,
            "value_consistency": (expected_cost - realized_cost).abs()
            / realized_cost.detach().abs().clamp_min(1.0),
            "day_ahead": issued_da,
            "battery": F.pad(battery_tensor, (0, 96 - steps)),
            "compute": F.pad(compute_tensor, (0, 96 - steps)),
            "grid": F.pad(grid_tensor, (0, 96 - steps)),
            "soc": torch.stack(soc_path, dim=-1),
            "rolling_state_before_action": torch.stack(
                rolling_state_path, dim=1
            ),
        }
        first_output.update(
            {
                "architecture_version": self.architecture_version,
                "distributional_architecture_revision": (
                    self.distributional_architecture_revision
                ),
                "risk": risk,
                "training_rollout": training_rollout,
                "actions": training_rollout,
                "policy": training_rollout,
                "parameter_count": self.parameter_count,
            }
        )
        return first_output

    def forward_origin(
        self,
        context: Mapping[str, Any],
        *,
        origin: int | Tensor,
        rolling_state: Tensor | Mapping[str, Tensor],
        liquid_state: Tensor | None = None,
        issued_da_schedule: Tensor | None = None,
        revealed: Mapping[str, Tensor] | Tensor | None = None,
        policy_memory: Tensor | None = None,
    ) -> dict[str, Any]:
        reference = context["market_tokens"]
        state_input = self._state_tensor(rolling_state, reference, origin=origin)
        state_memory = self.state_encoder.cell.step(
            self.state_encoder.input_norm(state_input), liquid_state, dt=0.25
        )
        state_token = self.state_encoder.output_norm(state_memory)
        state_tokens = state_token[:, None].expand(-1, self.config.horizon, -1)
        legacy_sample_count = (
            SCENARIO_COUNT
            if self.architecture_version == ARCHITECTURE_VERSION
            else None
        )
        output = self._fuse_and_decode(
            context,
            state_tokens,
            origin=origin,
            revealed=revealed,
            distribution_sample_count=legacy_sample_count,
            distribution_seed=0 if legacy_sample_count is not None else None,
        )
        world = output["distribution"]["world_latent"]
        risk = self.risk_encoder(world, state_token)
        origin_index = _broadcast_origin(
            origin, state_token.shape[0], state_token, self.config.horizon
        )
        revealed_context = world.token_features.gather(
            1,
            origin_index[:, None, None].expand(-1, 1, self.config.hidden_dim),
        ).squeeze(1)
        policy = self.policy(
            output["token_features"],
            state_token,
            output["forecast_zonal"],
            origin=origin,
            world_latent=world,
            risk_features=risk,
            rolling_state_hidden=state_token,
            revealed_context=revealed_context,
            issued_da_schedule=issued_da_schedule,
            policy_memory=policy_memory,
        )
        # Runtime memory is the raw recurrent state.  Reusing a normalized
        # policy token as recurrent memory would break prefix-replay equality.
        policy["policy_memory_next"] = state_memory
        reported_risk: Mapping[str, Any] = risk
        compatibility = output.get("scenarios")
        if (
            self.architecture_version == ARCHITECTURE_VERSION
            and isinstance(compatibility, Mapping)
        ):
            empirical = self.empirical_risk_evaluator(
                compatibility["particles_zonal"],
                compatibility["weights"],
                day_ahead=policy["day_ahead"],
                battery_plan=policy["battery_plan"],
                compute_plan=policy["compute_plan"],
            )
            reported_risk = {
                **risk,
                **empirical,
                "neural_risk": risk,
                "formal_cvar_reported": False,
            }
        output.update(
            {
                "architecture_version": self.architecture_version,
                "distributional_architecture_revision": (
                    self.distributional_architecture_revision
                ),
                "state_memory": state_memory,
                "liquid_state_next": state_memory,
                "policy": policy,
                "actions": policy,
                "risk": reported_risk,
                "parameter_count": self.parameter_count,
            }
        )
        return output

    def forward_session(self, session_input: Mapping[str, Any]) -> dict[str, Any]:
        """Runtime callback seam for a single rolling origin."""

        context = session_input.get("issue_context")
        if context is None:
            context = self.prepare_issue_context(session_input)
        rolling_state = session_input.get("rolling_state")
        if rolling_state is None:
            rolling_state = dict(
                session_input.get("state", {})
                if isinstance(session_input.get("state"), Mapping)
                else {}
            )
            rolling_state.update({
                key: session_input[key]
                for key in (
                    "current_soc",
                    "backlog",
                    "deadline_pressure",
                    "previous_grid",
                    "previous_battery",
                    "previous_compute",
                    "previous_migration",
                    "rt_innovation",
                    "workload_innovation",
                    "pv_innovation",
                )
                if key in session_input
            })
            previous = session_input.get("previous_action")
            if isinstance(previous, Mapping):
                for key, value in previous.items():
                    rolling_state[f"previous_{key}"] = value
        revealed = session_input.get("revealed")
        if revealed is None and session_input.get("revealed_zonal") is not None:
            revealed = {
                "zonal": session_input["revealed_zonal"],
                "mask": session_input.get("revealed_mask"),
            }
        if revealed is None and isinstance(session_input.get("state"), Mapping):
            state_mapping = session_input["state"]
            rt_value = state_mapping.get("revealed_rt")
            workload_value = state_mapping.get("revealed_workload")
            pv_value = state_mapping.get("revealed_pv")
            mask_value = session_input.get(
                "revealed_mask", state_mapping.get("revealed_mask")
            )
            if all(item is not None for item in (rt_value, workload_value, pv_value, mask_value)):
                market = self._model_tensor(session_input["market"])
                da = market[..., 0:4]
                rt = self._model_tensor(rt_value)
                workload = self._model_tensor(workload_value)
                pv = self._model_tensor(pv_value)
                pue = context["pue_core"]
                revealed = {
                    "zonal": torch.cat((da, rt, da - rt, workload, pv, pue), dim=-1),
                    "mask": self._model_tensor(mask_value).to(dtype=torch.bool),
                }
        return self.forward_origin(
            context,
            origin=session_input.get("origin", 0),
            rolling_state=rolling_state,
            liquid_state=session_input.get(
                "liquid_state", session_input.get("policy_memory")
            ),
            issued_da_schedule=session_input.get(
                "issued_da_schedule", session_input.get("frozen_day_ahead")
            ),
            revealed=revealed,
            policy_memory=session_input.get("policy_aux_memory"),
        )

    def phase_trainable_parameter_names(self, phase: str) -> tuple[str, ...]:
        phase = phase.lower()
        if phase not in self.PHASE_PREFIXES:
            raise ValueError(f"unknown R4 phase {phase!r}")
        prefixes = (
            self.V5_A1_PREFIXES
            if phase == "a1" and self.architecture_version == ARCHITECTURE_VERSION
            else self.PHASE_PREFIXES[phase]
        )
        return tuple(
            name
            for name, _ in self.named_parameters()
            if "physical_twin_core" not in name
            and not (
                self.physical_stream.formal_artifact_backed
                and name.startswith("forecast_decoder.pue_residual_head.")
            )
            and (prefixes == ("*",) or any(name.startswith(prefix) for prefix in prefixes))
        )

    def configure_trainable_phase(self, phase: str) -> tuple[str, ...]:
        selected = set(self.phase_trainable_parameter_names(phase))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in selected)
        self.physical_stream.freeze_twin()
        return tuple(sorted(selected))

    def load_representation_predecessor(
        self,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        expected_epoch: int = 28,
        require_capability_passed: bool = True,
    ) -> dict[str, Any]:
        """Allowlist-load the selected V5 Representation into a V6 model.

        Old fixed-particle Scenario, risk and Policy parameters are enumerated
        as rejected predecessor keys and are never passed to ``load_state_dict``.
        This preserves the accepted epoch-28 deterministic trunk without
        treating the failed V5 Scenario head as a V6 predecessor.
        """

        if isinstance(checkpoint, Mapping):
            payload = dict(checkpoint)
            source = "in_memory"
        else:
            path = Path(checkpoint).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(raw, Mapping):
                raise ValueError("Representation checkpoint payload is malformed")
            payload = dict(raw)
            source = str(path)
        phase = payload.get("phase")
        if phase is not None and str(phase) != "representation":
            raise ValueError("only a Representation checkpoint may seed V6")
        epoch_value = payload.get(
            "selected_candidate_epoch",
            payload.get(
                "selected_epoch",
                payload.get("epochs_completed", payload.get("epoch")),
            ),
        )
        if epoch_value is None or int(epoch_value) != int(expected_epoch):
            raise ValueError(
                f"V6 requires selected Representation epoch {int(expected_epoch)}"
            )
        if require_capability_passed:
            gate = str(payload.get("capability_gate_status", ""))
            role = str(payload.get("checkpoint_role", ""))
            if gate != "PASSED" or role != "capability_passed_selected_best":
                raise ValueError(
                    "Representation predecessor must be capability-passed selected best"
                )
        source_architecture = str(
            payload.get("architecture_version", ARCHITECTURE_VERSION)
        )
        if source_architecture != ARCHITECTURE_VERSION:
            raise ValueError("Representation predecessor must use the hybrid R4 base")
        raw_state = payload.get("model_state", payload.get("state_dict"))
        if not isinstance(raw_state, Mapping):
            raise ValueError("Representation checkpoint has no model_state mapping")
        current = self.state_dict()
        prefixes = self.PHASE_PREFIXES["representation"]
        allowed: dict[str, Tensor] = {}
        rejected: list[str] = []
        for raw_name, raw_value in raw_state.items():
            name = str(raw_name)
            if any(name.startswith(prefix) for prefix in prefixes):
                if name not in current:
                    raise ValueError(f"unknown Representation trunk key {name!r}")
                tensor = torch.as_tensor(raw_value)
                if tensor.shape != current[name].shape:
                    raise ValueError(
                        f"Representation key {name!r} has incompatible shape"
                    )
                allowed[name] = tensor
            else:
                rejected.append(name)
        required = {
            name
            for name, _ in self.named_parameters()
            if any(name.startswith(prefix) for prefix in prefixes)
        }
        missing = sorted(required - set(allowed))
        if missing:
            raise ValueError(
                "Representation predecessor is missing allowlisted trunk parameters: "
                + ", ".join(missing[:5])
            )
        self.load_state_dict(allowed, strict=False)
        self.physical_stream.freeze_twin()
        return {
            "source": source,
            "source_architecture": source_architecture,
            "target_architecture": self.architecture_version,
            "selected_representation_epoch": int(epoch_value),
            "loaded_key_count": len(allowed),
            "loaded_keys": tuple(sorted(allowed)),
            "rejected_predecessor_keys": tuple(sorted(rejected)),
            "fixed_particle_head_loaded": False,
            "old_risk_head_loaded": False,
            "old_policy_head_loaded": False,
        }

    # Compatibility alias for early design notes.
    set_trainable_phase = configure_trainable_phase


R4HybridModel = HybridCfCTransformerMoER4


__all__ = [
    "ARCHITECTURE_VERSION",
    "DISTRIBUTIONAL_ARCHITECTURE_REVISION",
    "DISTRIBUTIONAL_JOINT_ARCHITECTURE_VERSION",
    "CrossDomainFusion",
    "DistributionHead",
    "DistributionalWorldLatent",
    "FrozenPhysicalTwinCore",
    "HybridCfCTransformerMoER4",
    "MarketTransformerEncoder",
    "R4Config",
    "R4HybridModel",
    "TokenTop2MoE",
    "ZonalForecastDecoder",
]
