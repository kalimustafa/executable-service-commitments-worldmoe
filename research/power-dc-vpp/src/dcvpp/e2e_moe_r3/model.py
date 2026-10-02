"""Compact decision-native E2E-MoE-DART-VPP R3 model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .policy import FeasibilityFallbackHead, NeuralRollingPolicy
from .scenario_risk import (
    SCENARIO_COUNT,
    SCENARIO_VARIABLES,
    DecisionRiskHead,
    LearnedJointScenarioHead,
)


FORECAST_NAMES = ("da", "rt", "dart", "workload", "pv", "pue")


@dataclass(frozen=True)
class E2EMoER3Config:
    """Shape and capacity settings for the compact local R3 profile."""

    market_dim: int = 16
    workload_dim: int = 12
    physical_dim: int = 8
    state_dim: int = 12
    hidden_dim: int = 64
    expert_hidden_dim: int = 128
    horizon: int = 96
    action_dim: int = 1
    num_sites: int = 4
    num_experts: int = 8
    top_k: int = 2
    num_scenarios: int = SCENARIO_COUNT
    cvar_alpha: float = 0.95
    router_noise: bool = True
    router_noise_std: float = 0.1
    residual_bound: float = 0.1
    policy_origin_embedding_dim: int = 16
    policy_context_hidden_dim: int = 64
    policy_refinement_steps: int = 2

    @property
    def rolling_site_state_dim(self) -> int:
        """SOC, backlog, and deadline pressure features per controlled site."""

        return 3

    @property
    def rolling_previous_action_dim(self) -> int:
        """Flattened grid, battery, compute, and migration feedback width."""

        return 3 * self.num_sites + self.num_sites * self.num_sites

    @property
    def rolling_state_feature_dim(self) -> int:
        """Exact causal context width before the origin embedding."""

        return (
            self.rolling_site_state_dim * self.num_sites
            + self.rolling_previous_action_dim
            + 3
        )

    def __post_init__(self) -> None:
        dimensions = (
            self.market_dim,
            self.workload_dim,
            self.physical_dim,
            self.state_dim,
            self.hidden_dim,
            self.expert_hidden_dim,
            self.horizon,
            self.action_dim,
            self.num_sites,
            self.policy_origin_embedding_dim,
            self.policy_context_hidden_dim,
            self.policy_refinement_steps,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("all model dimensions must be positive")
        if self.num_experts != 8 or self.top_k != 2:
            raise ValueError("R3 fixes sparse routing at eight experts and Top-2")
        if self.num_scenarios != SCENARIO_COUNT:
            raise ValueError("R3 fixes formal learned scenarios at exactly 20")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie strictly between zero and one")
        if self.router_noise_std < 0.0 or self.residual_bound < 0.0:
            raise ValueError("noise and residual bounds must be non-negative")


class TemporalEncoder(nn.Module):
    """A small GRU encoder accepting either static or temporal feature input."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, time, {self.input_dim}] or "
                f"[batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        projected = self.input_projection(features)
        _, hidden = self.temporal(projected)
        return hidden[-1]


class FrozenPhysicalTwinCore(nn.Module):
    """A deterministic frozen projection standing in for calibrated twin state.

    Production callers can pass their calibrated torch module to
    :class:`E2EMoEDARTVPP`; this default remains frozen and contains no learned
    counterfactual degrees of freedom.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        weight = torch.zeros(output_dim, input_dim)
        for row in range(output_dim):
            weight[row, row % input_dim] = 1.0
        weight /= max(1.0, float(output_dim // input_dim))
        self.register_buffer("calibrated_projection", weight)
        self.register_buffer("calibrated_bias", torch.zeros(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, physical_features: Tensor) -> Tensor:
        if physical_features.ndim == 3:
            physical_features = physical_features.mean(dim=1)
        if physical_features.ndim != 2:
            raise ValueError("physical twin input must be [batch, features] or temporal")
        return F.linear(physical_features, self.calibrated_projection, self.calibrated_bias)


class BoundedResidualAdapter(nn.Module):
    """Bounded A2 adapter that cannot rewrite the frozen physical twin core."""

    def __init__(self, hidden_dim: int, bound: float = 0.1) -> None:
        super().__init__()
        self.bound = float(bound)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, physical_latent: Tensor) -> Tensor:
        return self.bound * torch.tanh(self.network(physical_latent))


class PhysicalUncertaintyAdapter(nn.Module):
    """Positive bounded uncertainty representation allowed to adapt in A2."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, physical_latent: Tensor) -> Tensor:
        return torch.sigmoid(self.network(physical_latent))


class Expert(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Linear(expert_hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, latent: Tensor) -> Tensor:
        return self.norm(latent + self.network(latent))


class SparseTop2MoE(nn.Module):
    """Eight-expert sparse router that executes only two experts per sample."""

    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        *,
        num_experts: int = 8,
        top_k: int = 2,
        noisy_router: bool = True,
        router_noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        if num_experts != 8 or top_k != 2:
            raise ValueError("R3 requires eight experts with Top-2 routing")
        self.num_experts = 8
        self.top_k = 2
        self.noisy_router = bool(noisy_router)
        self.router_noise_std = float(router_noise_std)
        self.router = nn.Linear(hidden_dim, self.num_experts)
        self.experts = nn.ModuleList(
            Expert(hidden_dim, expert_hidden_dim) for _ in range(self.num_experts)
        )

    def forward(self, latent: Tensor) -> dict[str, Tensor]:
        clean_logits = self.router(latent)
        clean_probabilities = torch.softmax(clean_logits, dim=-1)
        logits = clean_logits
        if self.training and self.noisy_router and self.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        probabilities = torch.softmax(logits, dim=-1)
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        selected_by_slot: list[Tensor] = []
        for slot in range(self.top_k):
            slot_output = torch.zeros_like(latent)
            for expert_index, expert in enumerate(self.experts):
                row_indices = torch.nonzero(
                    top_indices[:, slot] == expert_index, as_tuple=False
                ).flatten()
                if row_indices.numel():
                    expert_output = expert(latent.index_select(0, row_indices))
                    slot_output = slot_output.index_copy(0, row_indices, expert_output)
            selected_by_slot.append(slot_output)
        selected_outputs = torch.stack(selected_by_slot, dim=1)
        moe_latent = (selected_outputs * top_weights.unsqueeze(-1)).sum(dim=1)
        expert_load = F.one_hot(
            top_indices, num_classes=self.num_experts
        ).to(latent.dtype).mean(dim=(0, 1))
        disagreement = (selected_outputs[:, 0] - selected_outputs[:, 1]).pow(2).mean(dim=-1).sqrt()
        return {
            "latent": moe_latent,
            "router_logits": logits,
            "clean_router_logits": clean_logits,
            "router_probabilities": probabilities,
            "clean_router_probabilities": clean_probabilities,
            "topk_indices": top_indices,
            "topk_weights": top_weights,
            "expert_load": expert_load,
            "expert_disagreement": disagreement,
            "selected_expert_outputs": selected_outputs,
        }


class HorizonHead(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int, *, non_negative: bool = False) -> None:
        super().__init__()
        self.network = nn.Linear(hidden_dim, horizon)
        self.non_negative = bool(non_negative)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.network(latent)
        return F.softplus(output) if self.non_negative else output


class AffineDARTCalibration(nn.Module):
    """A1-only DART calibration initialized to the identity transform."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(horizon))
        self.bias = nn.Parameter(torch.zeros(horizon))

    def forward(self, dart: Tensor) -> Tensor:
        return dart * self.log_scale.exp() + self.bias


def _last_or_mean(value: Tensor, hidden_dim: int) -> Tensor:
    if value.ndim == 3:
        value = value.mean(dim=1)
    if value.ndim != 2 or value.shape[-1] != hidden_dim:
        raise ValueError(
            f"physical twin must emit [batch, {hidden_dim}] or temporal equivalent, "
            f"got {tuple(value.shape)}"
        )
    return value


class E2EMoEDARTVPP(nn.Module):
    """Unified forecasts -> scenarios -> risk -> decision policy model."""

    A1_PREFIXES = (
        "dart_head.",
        "dart_scenario_adapter.",
        "dart_calibration.",
    )
    A2_PREFIXES = (
        "market_encoder.",
        "workload_encoder.",
        "physical_residual_adapter.",
        "uncertainty_adapter.",
        "moe_fusion.",
        "moe.router.",
        "moe.experts.",
        "scenario_head.",
        "risk_head.",
        "policy.",
        "feasibility_head.",
        "value_head.",
    )

    def __init__(
        self,
        config: E2EMoER3Config | None = None,
        *,
        physical_twin_core: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or E2EMoER3Config()
        cfg = self.config
        self.market_encoder = TemporalEncoder(cfg.market_dim, cfg.hidden_dim)
        self.workload_encoder = TemporalEncoder(cfg.workload_dim, cfg.hidden_dim)
        self.physical_encoder = TemporalEncoder(cfg.physical_dim, cfg.hidden_dim)
        self.state_encoder = TemporalEncoder(cfg.state_dim, cfg.hidden_dim)

        self.physical_twin_core = physical_twin_core or FrozenPhysicalTwinCore(
            cfg.physical_dim, cfg.hidden_dim
        )
        self.freeze_physical_twin_core()
        self.physical_residual_adapter = BoundedResidualAdapter(
            cfg.hidden_dim, cfg.residual_bound
        )
        # Explicit alias in parameter-group reporting: this is the bounded
        # physical residual adapter named by the governing R3 specification.
        self.uncertainty_adapter = PhysicalUncertaintyAdapter(cfg.hidden_dim)

        self.moe_fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 5, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.moe = SparseTop2MoE(
            cfg.hidden_dim,
            cfg.expert_hidden_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            noisy_router=cfg.router_noise,
            router_noise_std=cfg.router_noise_std,
        )

        self.forecast_heads = nn.ModuleDict(
            {
                "da": HorizonHead(cfg.hidden_dim, cfg.horizon),
                "rt": HorizonHead(cfg.hidden_dim, cfg.horizon),
                "workload": HorizonHead(cfg.hidden_dim, cfg.horizon, non_negative=True),
                "pv": HorizonHead(cfg.hidden_dim, cfg.horizon, non_negative=True),
                "pue": HorizonHead(cfg.hidden_dim, cfg.horizon, non_negative=True),
            }
        )
        self.dart_head = HorizonHead(cfg.hidden_dim, cfg.horizon)
        self.dart_calibration = AffineDARTCalibration(cfg.horizon)
        self.dart_scenario_adapter = nn.Linear(
            cfg.hidden_dim, SCENARIO_COUNT * cfg.horizon
        )

        self.scenario_head = LearnedJointScenarioHead(
            cfg.hidden_dim,
            cfg.horizon,
            num_scenarios=cfg.num_scenarios,
        )
        self.risk_head = DecisionRiskHead(
            cfg.hidden_dim,
            num_scenarios=cfg.num_scenarios,
            cvar_alpha=cfg.cvar_alpha,
        )
        self.policy = NeuralRollingPolicy(
            cfg.hidden_dim,
            horizon=cfg.horizon,
            action_dim=cfg.action_dim,
            num_sites=cfg.num_sites,
            origin_embedding_dim=cfg.policy_origin_embedding_dim,
            context_hidden_dim=cfg.policy_context_hidden_dim,
            refinement_steps=cfg.policy_refinement_steps,
        )
        self.feasibility_head = FeasibilityFallbackHead(cfg.hidden_dim)
        self.value_head = nn.Linear(cfg.hidden_dim, 1)

    def freeze_physical_twin_core(self) -> None:
        self.physical_twin_core.eval()
        for parameter in self.physical_twin_core.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> E2EMoEDARTVPP:
        super().train(mode)
        # ``Module.train`` recurses into every child; immediately restore the
        # scientific invariant that the NLR-calibrated core stays frozen/eval.
        self.freeze_physical_twin_core()
        return self

    @staticmethod
    def _get_input(
        inputs: Mapping[str, Tensor], name: str, explicit: Tensor | None
    ) -> Tensor:
        if explicit is not None:
            return explicit
        for key in (name, f"{name}_features", f"{name}_inputs"):
            if key in inputs:
                return inputs[key]
        raise KeyError(f"missing required model input: {name}")

    def _causal_state_view(
        self,
        state: Tensor,
        revealed_mask: Tensor,
    ) -> Tensor:
        """Remove unrevealed temporal state before encoding it.

        Market, workload, and physical inputs are the issue-time forecast and
        frozen-twin views.  ``state`` is the only observation sequence accepted
        by this interface, and every unrevealed entry is structurally zeroed.
        A two-dimensional state tensor is already a current snapshot.
        """

        if state.ndim == 2:
            return state
        if state.ndim != 3 or state.shape[1] != self.config.horizon:
            raise ValueError(
                "rolling state must be [batch, state_features] or "
                f"[batch, {self.config.horizon}, state_features], got "
                f"{tuple(state.shape)}"
            )
        mask = torch.as_tensor(revealed_mask, device=state.device)
        if mask.shape != state.shape[:2]:
            raise ValueError(
                f"revealed_mask must match state prefix {tuple(state.shape[:2])}, "
                f"got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("revealed_mask must be boolean or binary")
            mask = mask.to(dtype=torch.bool)
        return state * mask.unsqueeze(-1).to(state.dtype)

    def prepare_origin_context(
        self,
        inputs: Mapping[str, Tensor] | None = None,
        *,
        market: Tensor | None = None,
        workload: Tensor | None = None,
        physical: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode origin-invariant issue-time inputs once per rolling rollout.

        The returned tensors retain their autograd graph, so reusing this
        context across all 96 origins is mathematically equivalent to invoking
        the three encoders independently at every origin.  State, queue,
        deadline, previous-action context, Top-2 routing, scenarios, risk, and
        actions are intentionally *not* cached because they remain
        origin-conditioned.
        """

        inputs = {} if inputs is None else inputs
        market = self._get_input(inputs, "market", market)
        workload = self._get_input(inputs, "workload", workload)
        physical = self._get_input(inputs, "physical", physical)
        market_latent = self.market_encoder(market)
        workload_latent = self.workload_encoder(workload)
        physical_encoded = self.physical_encoder(physical)
        with torch.no_grad():
            twin_latent = _last_or_mean(
                self.physical_twin_core(physical), self.config.hidden_dim
            )
        residual = self.physical_residual_adapter(physical_encoded)
        uncertainty = self.uncertainty_adapter(physical_encoded)
        return {
            "market": market_latent,
            "workload": workload_latent,
            "physical_encoded": physical_encoded,
            "physical_core": twin_latent,
            "physical_residual": residual,
            "physical_uncertainty": uncertainty,
            "physical": twin_latent + residual,
        }

    def forward(
        self,
        inputs: Mapping[str, Tensor] | None = None,
        *,
        market: Tensor | None = None,
        workload: Tensor | None = None,
        physical: Tensor | None = None,
        state: Tensor | None = None,
        scenario_noise: Tensor | None = None,
    ) -> dict[str, Any]:
        inputs = {} if inputs is None else inputs
        market = self._get_input(inputs, "market", market)
        workload = self._get_input(inputs, "workload", workload)
        physical = self._get_input(inputs, "physical", physical)
        state = self._get_input(inputs, "state", state)

        market_latent = self.market_encoder(market)
        workload_latent = self.workload_encoder(workload)
        physical_encoded = self.physical_encoder(physical)
        state_latent = self.state_encoder(state)
        with torch.no_grad():
            twin_latent = _last_or_mean(
                self.physical_twin_core(physical), self.config.hidden_dim
            )
        residual = self.physical_residual_adapter(physical_encoded)
        uncertainty = self.uncertainty_adapter(physical_encoded)
        physical_latent = twin_latent + residual

        fused = self.moe_fusion(
            torch.cat(
                (
                    market_latent,
                    workload_latent,
                    physical_latent,
                    state_latent,
                    uncertainty,
                ),
                dim=-1,
            )
        )
        routing = self.moe(fused)
        decision_latent = routing["latent"]

        forecasts = {
            name: head(decision_latent) for name, head in self.forecast_heads.items()
        }
        forecasts["dart"] = self.dart_calibration(self.dart_head(decision_latent))
        scenario = self.scenario_head(
            decision_latent, forecasts, noise=scenario_noise
        )
        dart_index = SCENARIO_VARIABLES.index("dart")
        dart_adjustment = self.dart_scenario_adapter(decision_latent).reshape(
            decision_latent.shape[0], SCENARIO_COUNT, self.config.horizon
        )
        particles = scenario["particles"].clone()
        particles[..., dart_index] = particles[..., dart_index] + dart_adjustment
        scenario = {**scenario, "particles": particles}

        risk = self.risk_head(
            decision_latent, scenario["particles"], scenario["weights"]
        )
        actions = self.policy(
            decision_latent,
            dart_forecast=forecasts["dart"],
            dart_scenarios=scenario["particles"][..., dart_index],
            scenario_weights=scenario["weights"],
        )
        confidence = self.feasibility_head(decision_latent)
        value = self.value_head(decision_latent).squeeze(-1)

        return {
            "latent": decision_latent,
            "encodings": {
                "market": market_latent,
                "workload": workload_latent,
                "physical": physical_latent,
                "state": state_latent,
                "physical_core": twin_latent,
                "physical_residual": residual,
                "physical_uncertainty": uncertainty,
            },
            "routing": routing,
            "forecasts": forecasts,
            "scenario": scenario,
            "scenarios": scenario["particles"],
            "scenario_weights": scenario["weights"],
            "risk": risk,
            "expected_cost": risk["expected_cost"],
            "tail_risk": risk["tail_risk"],
            "cvar": risk["cvar"],
            "actions": actions,
            "policy": actions,
            "confidence": confidence,
            "feasibility": confidence["feasibility"],
            "fallback_confidence": confidence["fallback_confidence"],
            "value": value,
        }

    def forward_origin(
        self,
        inputs: Mapping[str, Tensor] | None = None,
        *,
        origin: Tensor | int,
        current_soc: Tensor,
        backlog: Tensor,
        deadline_pressure: Tensor,
        previous_action: Mapping[str, Tensor] | Tensor,
        revealed_mask: Tensor,
        market: Tensor | None = None,
        workload: Tensor | None = None,
        physical: Tensor | None = None,
        state: Tensor | None = None,
        scenario_noise: Tensor | None = None,
        prepared_context: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        """Run the causal joint MoE policy at one rolling origin.

        All observed state is supplied explicitly.  ``revealed_mask`` must be
        ``[batch, horizon]`` and cannot expose a period after ``origin``.  The
        returned day-ahead schedule is ``[batch, sites, horizon]`` and is
        nonzero only at origin zero.  Current grid, signed battery, compute,
        and migration actions have shapes ``[batch, sites]`` and
        ``[batch, sites, sites]``.

        Temporal feature tensors are accepted exactly as in :meth:`forward`;
        callers must provide causal issue-time/current-origin views rather than
        arrays containing unrevealed future observations.
        """

        inputs = {} if inputs is None else inputs
        state = self._get_input(inputs, "state", state)
        if prepared_context is None:
            prepared_context = self.prepare_origin_context(
                inputs,
                market=market,
                workload=workload,
                physical=physical,
            )
        required_context = {
            "market",
            "workload",
            "physical_encoded",
            "physical_core",
            "physical_residual",
            "physical_uncertainty",
            "physical",
        }
        missing_context = required_context.difference(prepared_context)
        if missing_context:
            raise ValueError(
                "prepared_context is missing origin-invariant encodings: "
                f"{sorted(missing_context)}"
            )
        market_latent = prepared_context["market"]
        workload_latent = prepared_context["workload"]
        physical_encoded = prepared_context["physical_encoded"]
        state = self._causal_state_view(state, revealed_mask)
        state_latent = self.state_encoder(state)
        twin_latent = prepared_context["physical_core"]
        residual = prepared_context["physical_residual"]
        uncertainty = prepared_context["physical_uncertainty"]
        physical_latent = prepared_context["physical"]

        fused = self.moe_fusion(
            torch.cat(
                (
                    market_latent,
                    workload_latent,
                    physical_latent,
                    state_latent,
                    uncertainty,
                ),
                dim=-1,
            )
        )
        context, origin_index, causal_mask = self.policy.encode_origin_context(
            fused,
            origin=origin,
            current_soc=current_soc,
            backlog=backlog,
            deadline_pressure=deadline_pressure,
            previous_action=previous_action,
            revealed_mask=revealed_mask,
        )
        routing = self.moe(self.policy.condition_moe_input(fused, context))
        decision_latent = routing["latent"]

        forecasts = {
            name: head(decision_latent) for name, head in self.forecast_heads.items()
        }
        forecasts["dart"] = self.dart_calibration(self.dart_head(decision_latent))
        scenario = self.scenario_head(
            decision_latent, forecasts, noise=scenario_noise
        )
        dart_index = SCENARIO_VARIABLES.index("dart")
        dart_adjustment = self.dart_scenario_adapter(decision_latent).reshape(
            decision_latent.shape[0], SCENARIO_COUNT, self.config.horizon
        )
        particles = scenario["particles"].clone()
        particles[..., dart_index] = particles[..., dart_index] + dart_adjustment
        scenario = {**scenario, "particles": particles}
        risk = self.risk_head(
            decision_latent, scenario["particles"], scenario["weights"]
        )
        actions = self.policy.forward_origin(
            decision_latent,
            origin=origin_index,
            current_soc=current_soc,
            backlog=backlog,
            deadline_pressure=deadline_pressure,
            previous_action=previous_action,
            revealed_mask=causal_mask,
            context=context,
            dart_forecast=forecasts["dart"],
            dart_scenarios=scenario["particles"][..., dart_index],
            scenario_weights=scenario["weights"],
        )
        # Feasibility confidence is state-conditioned because context also
        # enters the Top-2 routing input and therefore decision_latent.
        confidence = self.feasibility_head(decision_latent)
        value = self.value_head(decision_latent).squeeze(-1)
        return {
            "origin": origin_index,
            "revealed_mask": causal_mask,
            "latent": decision_latent,
            "rolling_context": context,
            "encodings": {
                "market": market_latent,
                "workload": workload_latent,
                "physical": physical_latent,
                "state": state_latent,
                "physical_core": twin_latent,
                "physical_residual": residual,
                "physical_uncertainty": uncertainty,
            },
            "routing": routing,
            "forecasts": forecasts,
            "scenario": scenario,
            "scenarios": scenario["particles"],
            "scenario_weights": scenario["weights"],
            "risk": risk,
            "expected_cost": risk["expected_cost"],
            "tail_risk": risk["tail_risk"],
            "cvar": risk["cvar"],
            "actions": actions,
            "policy": actions,
            "confidence": confidence,
            "feasibility": confidence["feasibility"],
            "fallback_confidence": confidence["fallback_confidence"],
            "value": value,
        }

    def rollout_policy(
        self,
        inputs: Mapping[str, Tensor] | None = None,
        *,
        current_soc: Tensor,
        backlog: Tensor,
        deadline_pressure: Tensor,
        previous_action: Mapping[str, Tensor] | Tensor,
        revealed_mask: Tensor | None = None,
        market: Tensor | None = None,
        workload: Tensor | None = None,
        physical: Tensor | None = None,
        state: Tensor | None = None,
        scenario_noise: Tensor | None = None,
        state_transition: Any | None = None,
    ) -> dict[str, Any]:
        """Differentiably unroll the causal policy over all 96 origins.

        This is the training integration seam.  Returned shapes are
        ``day_ahead [B,4,96]``, ``grid/battery/compute [B,96,4]``, and
        ``migration [B,96,4,4]`` for the default contract.  The t0 day-ahead
        schedule is retained across the rollout.

        ``state_transition`` may be a differentiable callable receiving
        ``(origin, actions, state_dict)`` and returning updates for
        ``current_soc``, ``backlog``, ``deadline_pressure``, and optionally
        ``previous_action``.  Without it, exogenous state remains fixed while
        previous action feedback is updated causally.  Formal evaluation
        should use a physical transition plus the external constraint shield.
        """

        inputs = {} if inputs is None else inputs
        market = self._get_input(inputs, "market", market)
        workload = self._get_input(inputs, "workload", workload)
        physical = self._get_input(inputs, "physical", physical)
        state = self._get_input(inputs, "state", state)
        batch = market.shape[0]
        if revealed_mask is not None:
            requested_mask = torch.as_tensor(revealed_mask, device=market.device)
            if requested_mask.shape != (batch, self.config.horizon):
                raise ValueError(
                    "rollout revealed_mask must have shape "
                    f"{(batch, self.config.horizon)}, got "
                    f"{tuple(requested_mask.shape)}"
                )
            if requested_mask.dtype != torch.bool:
                if not torch.all((requested_mask == 0) | (requested_mask == 1)):
                    raise ValueError("rollout revealed_mask must be boolean or binary")
                requested_mask = requested_mask.to(dtype=torch.bool)
        else:
            requested_mask = torch.ones(
                (batch, self.config.horizon),
                dtype=torch.bool,
                device=market.device,
            )

        rolling_state: dict[str, Any] = {
            "current_soc": current_soc,
            "backlog": backlog,
            "deadline_pressure": deadline_pressure,
            "previous_action": previous_action,
        }
        prepared_context = self.prepare_origin_context(
            {"market": market, "workload": workload, "physical": physical}
        )
        grid_rows: list[Tensor] = []
        battery_rows: list[Tensor] = []
        compute_rows: list[Tensor] = []
        migration_rows: list[Tensor] = []
        confidence_rows: list[Tensor] = []
        fallback_rows: list[Tensor] = []
        origin_outputs: list[dict[str, Any]] = []
        day_ahead: Tensor | None = None
        periods = torch.arange(self.config.horizon, device=market.device).unsqueeze(0)
        for origin in range(self.config.horizon):
            # The caller may suppress known-unavailable historical entries,
            # while this prefix always structurally excludes future periods.
            causal_mask = requested_mask & periods.le(origin)
            causal_mask[:, origin] = True
            output = self.forward_origin(
                {
                    "market": market,
                    "workload": workload,
                    "physical": physical,
                    "state": state,
                },
                origin=origin,
                current_soc=rolling_state["current_soc"],
                backlog=rolling_state["backlog"],
                deadline_pressure=rolling_state["deadline_pressure"],
                previous_action=rolling_state["previous_action"],
                revealed_mask=causal_mask,
                scenario_noise=scenario_noise,
                prepared_context=prepared_context,
            )
            action = output["actions"]
            if day_ahead is None:
                day_ahead = action["day_ahead"]
            grid_rows.append(action["grid"])
            battery_rows.append(action["battery"])
            compute_rows.append(action["compute"])
            migration_rows.append(action["migration"])
            confidence_rows.append(output["feasibility"])
            fallback_rows.append(output["fallback_confidence"])
            origin_outputs.append(output)
            feedback: dict[str, Tensor] = {
                "grid": action["grid"],
                "battery": action["battery"],
                "compute": action["compute"],
                "migration": action["migration"],
            }
            rolling_state["previous_action"] = feedback
            if state_transition is not None:
                updates = state_transition(origin, action, dict(rolling_state))
                if not isinstance(updates, Mapping):
                    raise TypeError("state_transition must return a mapping")
                rolling_state.update(updates)

        assert day_ahead is not None
        actions = {
            "day_ahead": day_ahead,
            "grid": torch.stack(grid_rows, dim=1),
            "real_time": torch.stack(grid_rows, dim=1),
            "battery": torch.stack(battery_rows, dim=1),
            "compute": torch.stack(compute_rows, dim=1),
            "migration": torch.stack(migration_rows, dim=1),
        }
        return {
            "actions": actions,
            "policy": actions,
            "feasibility": torch.stack(confidence_rows, dim=1),
            "fallback_confidence": torch.stack(fallback_rows, dim=1),
            "origins": tuple(origin_outputs),
            "final_state": rolling_state,
        }

    def parameter_names_for_phase(self, phase: str) -> tuple[str, ...]:
        """Return the exact parameter allowlist for a resumable phase."""

        phase = phase.lower().strip()
        names = tuple(name for name, _ in self.named_parameters())
        if phase == "a1":
            prefixes = self.A1_PREFIXES
        elif phase == "a2":
            prefixes = self.A2_PREFIXES
        elif phase == "representation":
            prefixes = (
                "market_encoder.",
                "workload_encoder.",
                "physical_encoder.",
                "state_encoder.",
                "forecast_heads.",
                "dart_head.",
                "dart_calibration.",
            )
        elif phase == "scenario":
            prefixes = (
                "moe_fusion.",
                "moe.router.",
                "moe.experts.",
                "scenario_head.",
                "risk_head.",
            )
        elif phase in {"distillation", "policy"}:
            prefixes = (
                "policy.",
                "feasibility_head.",
                "value_head.",
            )
        elif phase in {"selection_evaluation", "evaluation", "eval"}:
            return ()
        else:
            raise ValueError(f"unknown R3 phase: {phase}")
        return tuple(name for name in names if name.startswith(prefixes))

    def configure_trainable_phase(self, phase: str) -> tuple[str, ...]:
        """Apply and return an exact phase-specific ``requires_grad`` allowlist."""

        allowed = set(self.parameter_names_for_phase(phase))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in allowed)
        self.freeze_physical_twin_core()
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def named_parameter_groups(self) -> dict[str, tuple[str, ...]]:
        """Expose auditable parameter names without duplicating parameters."""

        return {
            phase: self.parameter_names_for_phase(phase)
            for phase in (
                "representation",
                "scenario",
                "distillation",
                "policy",
                "a1",
                "a2",
                "selection_evaluation",
            )
        }


# Short aliases make the package convenient without weakening the descriptive
# paper-facing class name.
E2EMoER3 = E2EMoEDARTVPP
E2EMoEModel = E2EMoEDARTVPP


__all__ = [
    "FORECAST_NAMES",
    "AffineDARTCalibration",
    "BoundedResidualAdapter",
    "E2EMoEDARTVPP",
    "E2EMoEModel",
    "E2EMoER3",
    "E2EMoER3Config",
    "FrozenPhysicalTwinCore",
    "PhysicalUncertaintyAdapter",
    "SparseTop2MoE",
    "TemporalEncoder",
]
