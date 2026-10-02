"""Learned joint scenarios and differentiable tail-risk summaries for R3.

The scenario axis is deliberately fixed at twenty.  Keeping that invariant in
the module (rather than only in a runner) prevents training and formal rolling
evaluation from silently using different uncertainty sets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCENARIO_COUNT = 20
SCENARIO_VARIABLES = ("da", "rt", "dart", "workload", "pv", "pue")


def _as_forecast_baseline(
    forecasts: Mapping[str, Tensor] | None,
    *,
    latent: Tensor,
    horizon: int,
    variables: Sequence[str],
) -> Tensor:
    """Return a ``[batch, horizon, variables]`` forecast baseline."""

    if forecasts is None:
        return latent.new_zeros((latent.shape[0], horizon, len(variables)))

    columns: list[Tensor] = []
    for name in variables:
        if name not in forecasts:
            raise KeyError(f"missing forecast head required by scenarios: {name}")
        value = forecasts[name]
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim != 2 or value.shape[0] != latent.shape[0]:
            raise ValueError(
                f"forecast {name!r} must have shape [batch, horizon], got "
                f"{tuple(value.shape)}"
            )
        if value.shape[1] != horizon:
            raise ValueError(
                f"forecast {name!r} has horizon {value.shape[1]}, expected {horizon}"
            )
        columns.append(value)
    return torch.stack(columns, dim=-1)


def normalize_scenario_weights(weights: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize non-negative weights while preserving gradients."""

    if weights.ndim < 1 or weights.shape[-1] != SCENARIO_COUNT:
        raise ValueError(
            f"scenario weights must end in exactly {SCENARIO_COUNT} entries, "
            f"got {tuple(weights.shape)}"
        )
    weights = weights.clamp_min(0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)


def differentiable_weighted_cvar(
    costs: Tensor,
    weights: Tensor,
    eta: Tensor,
    *,
    alpha: float = 0.95,
) -> Tensor:
    """Rockafellar-Uryasev upper-tail CVaR with a differentiable ``eta``.

    ``costs`` and ``weights`` are ``[batch, scenarios]``.  ``eta`` can be
    ``[batch]`` or ``[batch, 1]`` and is normally supplied by the risk head.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if costs.shape != weights.shape or costs.shape[-1] != SCENARIO_COUNT:
        raise ValueError(
            "costs and weights must have identical [batch, 20] shapes; "
            f"got {tuple(costs.shape)} and {tuple(weights.shape)}"
        )
    weights = normalize_scenario_weights(weights)
    eta = eta.reshape(costs.shape[:-1])
    if eta.shape != costs.shape[:-1]:
        raise ValueError(
            f"eta must have shape {tuple(costs.shape[:-1])}, got {tuple(eta.shape)}"
        )
    excess = F.relu(costs - eta.unsqueeze(-1))
    return eta + (weights * excess).sum(dim=-1) / (1.0 - alpha)


class LearnedJointScenarioHead(nn.Module):
    """Generate twenty weighted joint DA/RT/DART/workload/PV/PUE particles."""

    def __init__(
        self,
        hidden_dim: int,
        horizon: int = 96,
        *,
        num_scenarios: int = SCENARIO_COUNT,
        variables: Sequence[str] = SCENARIO_VARIABLES,
        scenario_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_scenarios != SCENARIO_COUNT:
            raise ValueError("R3 requires exactly 20 learned joint scenarios")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.num_scenarios = SCENARIO_COUNT
        self.variables = tuple(variables)
        embedding_dim = scenario_embedding_dim or max(8, hidden_dim // 4)

        self.scenario_embeddings = nn.Embedding(SCENARIO_COUNT, embedding_dim)
        particle_hidden = max(hidden_dim, 32)
        self.particle_net = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, particle_hidden),
            nn.GELU(),
            nn.Linear(particle_hidden, horizon * len(self.variables)),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * len(self.variables)),
        )
        self.weight_head = nn.Linear(hidden_dim, SCENARIO_COUNT)

    def forward(
        self,
        latent: Tensor,
        forecasts: Mapping[str, Tensor] | None = None,
        *,
        noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if latent.ndim != 2 or latent.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"latent must have shape [batch, {self.hidden_dim}], got "
                f"{tuple(latent.shape)}"
            )
        batch = latent.shape[0]
        scenario_ids = torch.arange(SCENARIO_COUNT, device=latent.device)
        embeddings = self.scenario_embeddings(scenario_ids)
        embeddings = embeddings.unsqueeze(0).expand(batch, -1, -1)
        expanded_latent = latent.unsqueeze(1).expand(-1, SCENARIO_COUNT, -1)
        raw_offsets = self.particle_net(
            torch.cat((expanded_latent, embeddings), dim=-1)
        ).reshape(batch, SCENARIO_COUNT, self.horizon, len(self.variables))
        scale = F.softplus(self.scale_head(latent)).reshape(
            batch, self.horizon, len(self.variables)
        )
        baseline = _as_forecast_baseline(
            forecasts,
            latent=latent,
            horizon=self.horizon,
            variables=self.variables,
        )
        particles = baseline.unsqueeze(1) + torch.tanh(raw_offsets) * scale.unsqueeze(1)
        if noise is not None:
            if noise.shape != particles.shape:
                raise ValueError(
                    f"scenario noise must have shape {tuple(particles.shape)}, got "
                    f"{tuple(noise.shape)}"
                )
            particles = particles + noise * scale.unsqueeze(1)

        weight_logits = self.weight_head(latent)
        weights = torch.softmax(weight_logits, dim=-1)
        return {
            "particles": particles,
            "weights": weights,
            "weight_logits": weight_logits,
            "scale": scale,
        }


class DecisionRiskHead(nn.Module):
    """Map learned scenarios to expected cost, tail probabilities and CVaR."""

    probability_names = (
        "tail_risk_probability",
        "price_spike_probability",
        "sla_risk_probability",
        "migration_congestion_risk",
        "energy_shortfall_risk",
    )

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_scenarios: int = SCENARIO_COUNT,
        num_variables: int = len(SCENARIO_VARIABLES),
        cvar_alpha: float = 0.95,
    ) -> None:
        super().__init__()
        if num_scenarios != SCENARIO_COUNT:
            raise ValueError("R3 risk evaluation requires exactly 20 scenarios")
        if not 0.0 < cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie strictly between zero and one")
        self.hidden_dim = int(hidden_dim)
        self.num_variables = int(num_variables)
        self.cvar_alpha = float(cvar_alpha)

        risk_hidden = max(hidden_dim // 2, 16)
        self.scenario_cost_head = nn.Sequential(
            nn.Linear(hidden_dim + num_variables, risk_hidden),
            nn.GELU(),
            nn.Linear(risk_hidden, 1),
        )
        self.expected_cost_residual = nn.Linear(hidden_dim, 1)
        self.eta_head = nn.Linear(hidden_dim, 1)
        self.probability_head = nn.Linear(hidden_dim, len(self.probability_names))

    def forward(
        self,
        latent: Tensor,
        particles: Tensor,
        weights: Tensor,
        *,
        alpha: float | None = None,
    ) -> dict[str, Tensor]:
        if particles.ndim != 4 or particles.shape[1] != SCENARIO_COUNT:
            raise ValueError(
                "particles must have shape [batch, 20, horizon, variables], got "
                f"{tuple(particles.shape)}"
            )
        if particles.shape[0] != latent.shape[0] or particles.shape[-1] != self.num_variables:
            raise ValueError("particle batch/variable dimensions do not match the risk head")
        weights = normalize_scenario_weights(weights)
        pooled_particles = particles.mean(dim=-2)
        expanded_latent = latent.unsqueeze(1).expand(-1, SCENARIO_COUNT, -1)
        scenario_costs = self.scenario_cost_head(
            torch.cat((expanded_latent, pooled_particles), dim=-1)
        ).squeeze(-1)
        # A deterministic latent-state cost applies to every particle.  Adding
        # it before both expectation and CVaR keeps the two risk measures on
        # exactly the same cost basis.
        scenario_costs = scenario_costs + self.expected_cost_residual(latent)
        expected_cost = (weights * scenario_costs).sum(dim=-1)
        eta = self.eta_head(latent).squeeze(-1)
        cvar = differentiable_weighted_cvar(
            scenario_costs,
            weights,
            eta,
            alpha=self.cvar_alpha if alpha is None else alpha,
        )
        # Upper-tail CVaR should not be below the corresponding expected cost.
        cvar = torch.maximum(cvar, expected_cost)
        probabilities = torch.sigmoid(self.probability_head(latent))
        output = {
            "scenario_costs": scenario_costs,
            "scenario_expected_cost": expected_cost,
            "expected_cost": expected_cost,
            "eta": eta,
            "cvar": cvar,
        }
        output.update(
            {name: probabilities[:, index] for index, name in enumerate(self.probability_names)}
        )
        output["tail_risk"] = output["tail_risk_probability"]
        return output


class ScenarioRiskHead(nn.Module):
    """Convenience composition used by small standalone callers."""

    def __init__(
        self,
        hidden_dim: int,
        horizon: int = 96,
        *,
        cvar_alpha: float = 0.95,
    ) -> None:
        super().__init__()
        self.scenario_head = LearnedJointScenarioHead(hidden_dim, horizon)
        self.risk_head = DecisionRiskHead(hidden_dim, cvar_alpha=cvar_alpha)

    def forward(
        self,
        latent: Tensor,
        forecasts: Mapping[str, Tensor] | None = None,
        *,
        noise: Tensor | None = None,
    ) -> dict[str, object]:
        scenario = self.scenario_head(latent, forecasts, noise=noise)
        risk = self.risk_head(latent, scenario["particles"], scenario["weights"])
        return {"scenario": scenario, "risk": risk}


__all__ = [
    "SCENARIO_COUNT",
    "SCENARIO_VARIABLES",
    "DecisionRiskHead",
    "LearnedJointScenarioHead",
    "ScenarioRiskHead",
    "differentiable_weighted_cvar",
    "normalize_scenario_weights",
]
