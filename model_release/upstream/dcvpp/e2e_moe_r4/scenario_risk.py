"""Legacy scenario compatibility and arbitrary-N empirical risk for R4/V6."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .liquid import ClosedFormLiquidCell
from .distributional_world import (
    DISTRIBUTIONAL_ARCHITECTURE_REVISION,
    DistributionHead,
    DistributionalWorldLatent,
    RiskEncoder,
)


SCENARIO_COUNT = 20
ZONAL_VARIABLE_COUNT = 21
LEGACY_VARIABLE_COUNT = 6
ZONAL_VARIABLE_SLICES = {
    "da": slice(0, 4),
    "rt": slice(4, 8),
    "dart": slice(8, 12),
    "workload": slice(12, 13),
    "pv": slice(13, 17),
    "pue": slice(17, 21),
}
LEGACY_VARIABLE_NAMES = ("da", "rt", "dart", "workload", "pv", "pue")

# Compatibility-only raw-unit envelope.  Formal training supplies the frozen
# train-only p99 envelope through ``scenario_residual_caps_zonal``.  Keeping a
# non-persistent fallback lets architecture/unit tests construct the generator
# directly without silently changing the checkpoint state contract.
_FALLBACK_RAW_UNIT_CAPS_ZONAL = (
    *(10.0 for _ in range(4)),
    *(20.0 for _ in range(4)),
    *(10.0 for _ in range(4)),
    10.0,
    *(10.0 for _ in range(4)),
    *(0.1 for _ in range(4)),
)


def normalize_scenario_weights(weights: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize any positive Monte Carlo budget; N is not model capacity."""

    if weights.ndim != 2 or weights.shape[-1] < 1:
        raise ValueError("scenario weights must be [batch,N] with N >= 1")
    positive = weights.clamp_min(0.0)
    return positive / positive.sum(dim=-1, keepdim=True).clamp_min(eps)


def exact_weighted_cvar(costs: Tensor, weights: Tensor, alpha: float = 0.95) -> Tensor:
    """Exact forward weighted empirical upper-tail CVaR.

    Sorting selects the cost tail and partial boundary-particle mass is retained.
    The returned value is therefore computed from the final particles and is
    not an eta-head approximation.
    """

    if costs.ndim != 2 or costs.shape[-1] < 1:
        raise ValueError("costs must be [batch,N] with N >= 1")
    if costs.shape != weights.shape:
        raise ValueError("costs and weights must have identical shapes")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    weights = normalize_scenario_weights(weights)
    ordered_costs, order = torch.sort(costs, dim=-1, descending=True)
    ordered_weights = torch.gather(weights, -1, order)
    tail_mass = 1.0 - float(alpha)
    mass_before = ordered_weights.cumsum(dim=-1) - ordered_weights
    included_mass = torch.minimum(
        ordered_weights,
        (tail_mass - mass_before).clamp_min(0.0),
    )
    return (ordered_costs * included_mass).sum(dim=-1) / tail_mass


def zonal_to_legacy(values: Tensor) -> Tensor:
    """Convert final-width-21 tensors to the six-variable aggregate view."""

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


class LearnedJointScenarioGenerator(nn.Module):
    """Shared Liquid decoder for exactly twenty persistent joint trajectories."""

    def __init__(
        self,
        token_dim: int = 128,
        *,
        horizon: int = 96,
        scenario_code_dim: int = 32,
        hidden_dim: int = 64,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if horizon != 96:
            raise ValueError("formal R4 scenarios require a 96-interval horizon")
        self.horizon = 96
        self.num_scenarios = SCENARIO_COUNT
        self.scenario_code_dim = int(scenario_code_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("scenario residual_scale must lie in (0,1]")
        self.register_buffer(
            "default_raw_unit_residual_caps_zonal",
            torch.tensor(_FALLBACK_RAW_UNIT_CAPS_ZONAL, dtype=torch.float32),
            persistent=False,
        )
        self._default_residual_caps_formally_set = False
        self.scenario_codes = nn.Embedding(SCENARIO_COUNT, scenario_code_dim)
        self.decoder = ClosedFormLiquidCell(
            token_dim + scenario_code_dim,
            hidden_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
            default_dt_hours=0.25,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 14),  # DA4, DART4, workload1, PV4, PUE residual1
        )
        self.scale_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 14),
            nn.Softplus(),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, SCENARIO_COUNT),
        )

    @property
    def residual_cap_contract(self) -> dict[str, object]:
        return {
            "schema": "r4_scenario_raw_unit_residual_caps_v1",
            "formal_input": "scenario_residual_caps_zonal",
            "formal_input_shape": "[B,96,21]",
            "accepted_test_shapes": ("[21]", "[96,21]", "[B,96,21]"),
            "units": "raw_output_units",
            "fit_split": "train_only_p99",
            "scale_parameterization": "softplus_over_one_plus_softplus",
            "free_channels": "DA4,DART4,workload1,PV4,shared_PUE_latent_with_zone_caps4",
            "derived_channel": "RT=DA-DART_candidate;DART=DA-RT_final",
            "compatibility_fallback_only": True,
        }

    def set_default_residual_caps(self, caps: Tensor) -> None:
        """Install the frozen train-only [21] cap contract in-place.

        The buffer is intentionally non-persistent so a selected
        Representation checkpoint from before Scenario cap materialization can
        still load strictly.  Formal phase/final loaders bind the cap artifact
        in checkpoint metadata and call this method after model construction.
        """

        value = torch.as_tensor(
            caps,
            device=self.default_raw_unit_residual_caps_zonal.device,
            dtype=self.default_raw_unit_residual_caps_zonal.dtype,
        )
        if value.shape != (ZONAL_VARIABLE_COUNT,):
            raise ValueError("default scenario residual caps must be [21]")
        if not bool(torch.isfinite(value).all()) or not bool((value > 0.0).all()):
            raise ValueError(
                "default scenario residual caps must be finite and strictly positive"
            )
        value = value.clone()
        value[4:8] = value[0:4] + value[8:12]
        self.default_raw_unit_residual_caps_zonal.copy_(value)
        self._default_residual_caps_formally_set = True

    def _resolve_residual_caps_zonal(
        self, value: Tensor | None, token_features: Tensor
    ) -> Tensor:
        batch = token_features.shape[0]
        if value is None:
            caps = self.default_raw_unit_residual_caps_zonal.to(token_features)
        else:
            caps = torch.as_tensor(
                value, device=token_features.device, dtype=token_features.dtype
            )
        if caps.shape == (ZONAL_VARIABLE_COUNT,):
            caps = caps.reshape(1, 1, ZONAL_VARIABLE_COUNT).expand(
                batch, self.horizon, -1
            )
        elif caps.shape == (self.horizon, ZONAL_VARIABLE_COUNT):
            caps = caps.unsqueeze(0).expand(batch, -1, -1)
        elif caps.shape != (batch, self.horizon, ZONAL_VARIABLE_COUNT):
            raise ValueError(
                "scenario_residual_caps_zonal must be [21], [96,21], or "
                "[batch,96,21]"
            )
        if not bool(torch.isfinite(caps).all()) or not bool((caps > 0.0).all()):
            raise ValueError(
                "scenario_residual_caps_zonal must be finite and strictly positive"
            )
        caps = caps.clone()
        caps[..., 4:8] = caps[..., 0:4] + caps[..., 8:12]
        if self._default_residual_caps_formally_set:
            expected = self.default_raw_unit_residual_caps_zonal.to(caps).reshape(
                1, 1, ZONAL_VARIABLE_COUNT
            ).expand_as(caps)
            if not bool(torch.allclose(caps, expected, rtol=0.0, atol=0.0)):
                raise ValueError(
                    "scenario_residual_caps_zonal differs from the installed "
                    "frozen train-only contract"
                )
        return caps

    def _raw_unit_scale_zonal(
        self, scale_fraction: Tensor, caps_zonal: Tensor
    ) -> Tensor:
        """Map fourteen free amplitudes to a cap-safe 21-channel envelope.

        RT is not a free stochastic channel.  Its diagnostic amplitude is the
        triangle-inequality envelope of the independently calibrated DA and
        DART amplitudes; it must not shrink either free p99 support.
        """

        da_scale = scale_fraction[..., 0:4] * caps_zonal[..., 0:4]
        dart_scale = scale_fraction[..., 4:8] * caps_zonal[..., 8:12]
        rt_scale = da_scale + dart_scale
        workload_scale = scale_fraction[..., 8:9] * caps_zonal[..., 12:13]
        pv_scale = scale_fraction[..., 9:13] * caps_zonal[..., 13:17]
        # One liquid latent preserves the frozen-twin zonal linkage, while the
        # four raw-unit caps still calibrate each zone independently.
        pue_scale = scale_fraction[..., 13:14] * caps_zonal[..., 17:21]
        return torch.cat(
            (da_scale, rt_scale, dart_scale, workload_scale, pv_scale, pue_scale),
            dim=-1,
        )

    def _decode_output(
        self, raw: Tensor, baseline: Tensor, raw_unit_scale_zonal: Tensor
    ) -> Tensor:
        da_delta = (
            torch.tanh(raw[..., 0:4]) * raw_unit_scale_zonal[..., 0:4]
        )
        dart_delta = (
            torch.tanh(raw[..., 4:8]) * raw_unit_scale_zonal[..., 8:12]
        )
        da = baseline[..., 0:4] + da_delta
        # Build RT from bounded raw-unit innovations, then recompute DART last.
        # The public audit evaluates (DA - RT) - DART in this exact order, so
        # the final assignment makes its fp32 residual structurally zero.
        rt = baseline[..., 4:8] + da_delta - dart_delta
        dart = da - rt
        # Workload/PV caps are output-space raw-unit envelopes.  Applying them
        # inside inverse-softplus would change their units and could amplify a
        # nominal cap nonlinearly.  Add the bounded raw candidate first, then
        # project only the physically impossible negative tail to zero.
        workload = (
            baseline[..., 12:13]
            + torch.tanh(raw[..., 8:9]) * raw_unit_scale_zonal[..., 12:13]
        ).clamp_min(0.0)
        pv = (
            baseline[..., 13:17]
            + torch.tanh(raw[..., 9:13]) * raw_unit_scale_zonal[..., 13:17]
        ).clamp_min(0.0)
        # A common liquid disturbance preserves the zonal twin linkage, but
        # unlike the old fixed 0.1 path its four raw-unit amplitudes are now
        # learned through scale_head and bounded by the frozen zonal caps.
        pue = (
            baseline[..., 17:21]
            + torch.tanh(raw[..., 13:14]) * raw_unit_scale_zonal[..., 17:21]
        ).clamp_min(1e-4)
        return torch.cat((da, rt, dart, workload, pv, pue), dim=-1)

    def _default_decoder_step(self, inputs: Tensor, hidden: Tensor) -> Tensor:
        """Execute the decoder's validated default-time update without a sync.

        ``ClosedFormLiquidCell.step`` accepts arbitrary elapsed-time tensors and
        therefore validates them on every call.  This generator always uses the
        constructor-validated ``default_dt_hours`` value, so repeating CUDA
        ``isfinite``/``any`` checks at all 96 recurrent steps only introduces
        host-device synchronization.  The equations and fp32 exponential are
        intentionally identical to ``ClosedFormLiquidCell.step``.
        """

        joined = torch.cat((inputs, hidden), dim=-1)
        tau = self.decoder.tau_min_hours + (
            self.decoder.tau_max_hours - self.decoder.tau_min_hours
        ) * torch.sigmoid(self.decoder.tau_projection(joined))
        candidate = torch.tanh(self.decoder.candidate_projection(joined))
        decay = torch.exp(
            -self.decoder.default_dt_hours / tau.float()
        ).to(dtype=inputs.dtype)
        return decay * hidden + (1.0 - decay) * candidate

    def forward(
        self,
        token_features: Tensor,
        forecast_zonal: Tensor,
        *,
        origin: int | Tensor | None = None,
        revealed_zonal: Tensor | None = None,
        revealed_mask: Tensor | None = None,
        residual_caps_zonal: Tensor | None = None,
        dart_raw_adjustment: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if token_features.ndim != 3 or token_features.shape[1] != self.horizon:
            raise ValueError("scenario token features must be [batch,96,hidden]")
        if forecast_zonal.shape != (*token_features.shape[:2], ZONAL_VARIABLE_COUNT):
            raise ValueError("forecast_zonal must be [batch,96,21]")
        batch = token_features.shape[0]
        caps_zonal = self._resolve_residual_caps_zonal(
            residual_caps_zonal, token_features
        )
        code = self.scenario_codes.weight[None].expand(batch, -1, -1)
        hidden = token_features.new_zeros((batch, SCENARIO_COUNT, self.hidden_dim))
        positive_scale = self.scale_head(token_features)
        scale_fraction = (
            positive_scale / (1.0 + positive_scale)
        ) * self.residual_scale
        raw_unit_scale_zonal = self._raw_unit_scale_zonal(
            scale_fraction, caps_zonal
        )
        hidden_states: list[Tensor] = []
        for step in range(self.horizon):
            step_tokens = token_features[:, step, None].expand(-1, SCENARIO_COUNT, -1)
            decoder_input = torch.cat((step_tokens, code), dim=-1)
            hidden = self._default_decoder_step(
                decoder_input.reshape(batch * SCENARIO_COUNT, -1),
                hidden.reshape(batch * SCENARIO_COUNT, -1),
            ).reshape(batch, SCENARIO_COUNT, self.hidden_dim)
            hidden_states.append(hidden)

        # The decoded residual never feeds the recurrent state.  Applying the
        # heads once to the complete state sequence is therefore equivalent to
        # 96 small per-step calls while presenting much larger GEMMs to CUDA.
        hidden_sequence = torch.stack(hidden_states, dim=2)
        raw = self.residual_head(hidden_sequence)
        if dart_raw_adjustment is not None:
            adjustment = torch.as_tensor(
                dart_raw_adjustment, device=raw.device, dtype=raw.dtype
            )
            if adjustment.shape != (batch, self.horizon, 4):
                raise ValueError("dart_raw_adjustment must be [batch,96,4]")
            raw = raw.clone()
            raw[..., 4:8] = raw[..., 4:8] + adjustment[:, None]
        particles = self._decode_output(
            raw,
            forecast_zonal[:, None],
            raw_unit_scale_zonal[:, None],
        )

        if revealed_zonal is not None or revealed_mask is not None:
            if revealed_zonal is None or revealed_mask is None:
                raise ValueError("revealed_zonal and revealed_mask must be provided together")
            if revealed_zonal.shape != (batch, self.horizon, ZONAL_VARIABLE_COUNT):
                raise ValueError("revealed_zonal must be [batch,96,21]")
            mask = torch.as_tensor(revealed_mask, device=particles.device)
            if mask.shape != (batch, self.horizon):
                raise ValueError("revealed_mask must be [batch,96]")
            mask = mask.to(dtype=torch.bool)
            if origin is not None:
                index = torch.as_tensor(origin, device=particles.device)
                if index.ndim == 0:
                    index = index.expand(batch)
                periods = torch.arange(self.horizon, device=particles.device)[None]
                if torch.any(mask & (periods > index[:, None])):
                    raise ValueError("revealed_mask exposes a future value")
            identity_residual = (
                revealed_zonal[..., 0:4]
                - revealed_zonal[..., 4:8]
                - revealed_zonal[..., 8:12]
            )
            if torch.any(identity_residual[mask].abs() > 1e-5):
                raise ValueError("revealed zonal values violate DA-RT-DART identity")
            particles = torch.where(
                mask[:, None, :, None], revealed_zonal[:, None], particles
            )

        pooled = token_features.mean(dim=1)
        weight_logits = self.weight_head(pooled)
        weights = torch.softmax(weight_logits, dim=-1)
        return {
            "particles_zonal": particles,
            "particles": particles,
            "particles_legacy": zonal_to_legacy(particles),
            "weights": weights,
            "weight_logits": weight_logits,
            "scale": raw_unit_scale_zonal,
            "scale_fraction_free": scale_fraction,
            "raw_unit_scale_zonal": raw_unit_scale_zonal,
            "raw_unit_residual_caps_zonal": caps_zonal,
            "residual_cap_contract": self.residual_cap_contract,
            "effective_sample_size": weights.square().sum(dim=-1).reciprocal(),
            "weight_entropy": -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1),
        }


class EquationBasedRisk(nn.Module):
    """Differentiable settlement/physical rollout; no learned cost predictor."""

    def __init__(
        self,
        *,
        cvar_alpha: float = 0.95,
        interval_hours: float = 0.25,
        sla_penalty: float = 1000.0,
        physical_penalty: float = 1000.0,
    ) -> None:
        super().__init__()
        self.cvar_alpha = float(cvar_alpha)
        self.interval_hours = float(interval_hours)
        self.sla_penalty = float(sla_penalty)
        self.physical_penalty = float(physical_penalty)

    def forward(
        self,
        particles_zonal: Tensor,
        weights: Tensor,
        *,
        day_ahead: Tensor | None = None,
        battery_plan: Tensor | None = None,
        compute_plan: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if (
            particles_zonal.ndim != 4
            or particles_zonal.shape[1] < 1
            or particles_zonal.shape[2:] != (96, ZONAL_VARIABLE_COUNT)
        ):
            raise ValueError("particles_zonal must be [batch,N,96,21]")
        if weights.shape != particles_zonal.shape[:2]:
            raise ValueError("weights must match the [batch,N] sample axes")
        weights = normalize_scenario_weights(weights)
        batch = particles_zonal.shape[0]
        da_price = particles_zonal[..., 0:4]
        rt_price = particles_zonal[..., 4:8]
        workload = particles_zonal[..., 12:13].expand(-1, -1, -1, 4)
        pv = particles_zonal[..., 13:17]
        pue = particles_zonal[..., 17:21]
        load = workload * pue
        if day_ahead is None:
            da_schedule = torch.zeros_like(load)
        else:
            if day_ahead.shape != (batch, 4, 96):
                raise ValueError("day_ahead must be [batch,4,96]")
            da_schedule = day_ahead.transpose(1, 2)[:, None].expand_as(load)
        if battery_plan is None:
            battery = torch.zeros_like(load)
        else:
            if battery_plan.shape != (batch, 4, 96):
                raise ValueError("battery_plan must be [batch,4,96]")
            battery = battery_plan.transpose(1, 2)[:, None].expand_as(load)
        if compute_plan is None:
            served = workload
        else:
            if compute_plan.shape != (batch, 4, 96):
                raise ValueError("compute_plan must be [batch,4,96]")
            served = compute_plan.transpose(1, 2)[:, None].expand_as(load)

        # Positive signed battery is charge and therefore raises net demand;
        # negative signed battery is discharge.  Keep the differentiable risk
        # rollout identical to the formal analytic-shield convention.
        net_demand = load + battery - pv
        rt_settlement = net_demand - da_schedule
        market_cost = self.interval_hours * (
            da_price * da_schedule + rt_price * rt_settlement
        ).sum(dim=(-1, -2))
        sla_shortfall = F.relu(workload - served).sum(dim=(-1, -2))
        # Negative net demand is an export/curtailment request; the external
        # shield decides feasibility.  It remains an explicit tail diagnostic.
        physical_shortfall = F.relu(-net_demand).sum(dim=(-1, -2))
        costs = (
            market_cost
            + self.sla_penalty * sla_shortfall
            + self.physical_penalty * physical_shortfall
        )
        expected = (weights * costs).sum(dim=-1)
        cvar = exact_weighted_cvar(costs, weights, self.cvar_alpha)
        return {
            "scenario_costs": costs,
            "expected_cost": expected,
            "scenario_expected_cost": expected,
            "cvar": cvar,
            "cvar95": cvar,
            "particle_sla_shortfall": sla_shortfall,
            "particle_physical_shortfall": physical_shortfall,
        }


__all__ = [
    "DISTRIBUTIONAL_ARCHITECTURE_REVISION",
    "DistributionHead",
    "DistributionalWorldLatent",
    "EquationBasedRisk",
    "LEGACY_VARIABLE_COUNT",
    "LEGACY_VARIABLE_NAMES",
    "LearnedJointScenarioGenerator",
    "RiskEncoder",
    "SCENARIO_COUNT",
    "ZONAL_VARIABLE_COUNT",
    "ZONAL_VARIABLE_SLICES",
    "exact_weighted_cvar",
    "normalize_scenario_weights",
    "zonal_to_legacy",
]
