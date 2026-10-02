"""Differentiable decision, scenario, routing and constraint losses for R3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .scenario_risk import (
    SCENARIO_COUNT,
    SCENARIO_VARIABLES,
    differentiable_weighted_cvar,
    normalize_scenario_weights,
)


def _zero_like(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def routing_balance_loss(
    router_probabilities: Tensor,
    expert_load: Tensor | None = None,
) -> Tensor:
    """Switch-style expert utilization loss with zero at uniform utilization."""

    if router_probabilities.ndim != 2:
        raise ValueError("router probabilities must have shape [batch, experts]")
    probabilities = router_probabilities.clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    importance = probabilities.mean(dim=0)
    if expert_load is None:
        expert_load = F.one_hot(
            probabilities.argmax(dim=-1), num_classes=probabilities.shape[-1]
        ).to(probabilities.dtype).mean(dim=0)
    if expert_load.shape != importance.shape:
        raise ValueError("expert_load must have one entry per router expert")
    expert_load = expert_load / expert_load.sum().clamp_min(1e-8)
    raw = probabilities.shape[-1] * (importance * expert_load).sum() - 1.0
    return raw.clamp_min(0.0)


def routing_stability_loss(
    router_probabilities: Tensor,
    reference_probabilities: Tensor | None,
    *,
    detach_reference: bool = True,
) -> Tensor:
    """Penalize routing drift for paired/adjacent states."""

    if reference_probabilities is None:
        return _zero_like(router_probabilities)
    if router_probabilities.shape != reference_probabilities.shape:
        raise ValueError("current and reference routing tensors must match")
    reference = reference_probabilities.detach() if detach_reference else reference_probabilities
    current = router_probabilities.clamp_min(1e-8)
    reference = reference.clamp_min(1e-8)
    current = current / current.sum(dim=-1, keepdim=True)
    reference = reference / reference.sum(dim=-1, keepdim=True)
    forward_kl = (current * (current.log() - reference.log())).sum(dim=-1)
    reverse_kl = (reference * (reference.log() - current.log())).sum(dim=-1)
    return 0.5 * (forward_kl + reverse_kl).mean()


def _validate_scenarios(
    particles: Tensor, weights: Tensor, target: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if particles.ndim < 3 or particles.shape[1] != SCENARIO_COUNT:
        raise ValueError("particles must have shape [batch, 20, ...]")
    if target.shape != particles.shape[:1] + particles.shape[2:]:
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match particles "
            f"{tuple(particles.shape)}"
        )
    if weights.shape != particles.shape[:2]:
        raise ValueError("weights must have shape [batch, 20]")
    return particles, normalize_scenario_weights(weights), target


def marginal_scenario_loss(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
) -> Tensor:
    """Weighted empirical CRPS averaged over all marginal dimensions."""

    particles, weights, target = _validate_scenarios(particles, weights, target)
    broadcast_shape = (particles.shape[0], SCENARIO_COUNT) + (1,) * (particles.ndim - 2)
    expanded_weights = weights.reshape(broadcast_shape)
    first = (expanded_weights * (particles - target.unsqueeze(1)).abs()).sum(dim=1)
    pair_weights = weights.unsqueeze(2) * weights.unsqueeze(1)
    pair_shape = pair_weights.shape + (1,) * (particles.ndim - 2)
    pairwise = (particles.unsqueeze(2) - particles.unsqueeze(1)).abs()
    second = 0.5 * (pair_weights.reshape(pair_shape) * pairwise).sum(dim=(1, 2))
    return (first - second).mean()


weighted_marginal_crps = marginal_scenario_loss


def energy_score_loss(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    beta: float = 1.0,
) -> Tensor:
    """Weighted multivariate energy score for the complete joint trajectory."""

    if not 0.0 < beta <= 2.0:
        raise ValueError("energy-score beta must lie in (0, 2]")
    particles, weights, target = _validate_scenarios(particles, weights, target)
    flat_particles = particles.flatten(start_dim=2)
    flat_target = target.flatten(start_dim=1)
    observed_distance = torch.linalg.vector_norm(
        flat_particles - flat_target.unsqueeze(1), dim=-1
    ).pow(beta)
    first = (weights * observed_distance).sum(dim=-1)
    pairwise_distance = torch.cdist(flat_particles, flat_particles, p=2).pow(beta)
    second = 0.5 * (
        weights.unsqueeze(2) * weights.unsqueeze(1) * pairwise_distance
    ).sum(dim=(1, 2))
    return (first - second).mean()


def variogram_score_loss(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    power: float = 0.5,
    lags: tuple[int, ...] = (1, 4, 16),
) -> Tensor:
    """Memory-bounded weighted variogram score over deterministic feature lags."""

    if power <= 0.0:
        raise ValueError("variogram power must be positive")
    particles, weights, target = _validate_scenarios(particles, weights, target)
    flat_particles = particles.flatten(start_dim=2)
    flat_target = target.flatten(start_dim=1)
    components: list[Tensor] = []
    for lag in lags:
        if lag < 1 or lag >= flat_target.shape[-1]:
            continue
        target_variogram = (
            flat_target[:, lag:] - flat_target[:, :-lag]
        ).abs().pow(power)
        scenario_variogram = (
            flat_particles[:, :, lag:] - flat_particles[:, :, :-lag]
        ).abs().pow(power)
        predicted_variogram = (
            weights.unsqueeze(-1) * scenario_variogram
        ).sum(dim=1)
        components.append((target_variogram - predicted_variogram).pow(2).mean())
    if not components:
        return _zero_like(particles)
    return torch.stack(components).mean()


def weighted_quantile(samples: Tensor, weights: Tensor, quantile: float) -> Tensor:
    """Piecewise-differentiable empirical quantile over the scenario axis."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    if samples.ndim < 3 or samples.shape[1] != SCENARIO_COUNT:
        raise ValueError("samples must have shape [batch, 20, ...]")
    weights = normalize_scenario_weights(weights)
    flat = samples.flatten(start_dim=2)
    sorted_samples, indices = flat.sort(dim=1)
    expanded_weights = weights.unsqueeze(-1).expand_as(flat)
    sorted_weights = expanded_weights.gather(1, indices)
    cdf = sorted_weights.cumsum(dim=1)
    selected = (cdf < quantile).sum(dim=1).clamp_max(SCENARIO_COUNT - 1)
    result = sorted_samples.gather(1, selected.unsqueeze(1)).squeeze(1)
    return result.reshape(samples.shape[:1] + samples.shape[2:])


def tail_coverage_loss(
    particles: Tensor,
    weights: Tensor,
    target: Tensor,
    *,
    quantile: float = 0.95,
    temperature: float = 0.1,
) -> Tensor:
    """Upper-tail pinball loss plus a smooth batch coverage calibration term."""

    particles, weights, target = _validate_scenarios(particles, weights, target)
    predicted_quantile = weighted_quantile(particles, weights, quantile)
    error = target - predicted_quantile
    pinball = torch.maximum(quantile * error, (quantile - 1.0) * error).mean()
    scale = target.detach().std().clamp_min(temperature)
    soft_covered = torch.sigmoid((predicted_quantile - target) / scale)
    coverage = (soft_covered.mean() - quantile).pow(2)
    return pinball + coverage


def scenario_diversity_loss(
    particles: Tensor,
    *,
    minimum_distance: float = 0.1,
) -> Tensor:
    """Penalize collapsed particles without rewarding unbounded dispersion."""

    if particles.ndim < 3 or particles.shape[1] != SCENARIO_COUNT:
        raise ValueError("particles must have shape [batch, 20, ...]")
    flat = particles.flatten(start_dim=2)
    distances = torch.cdist(flat, flat, p=2) / math.sqrt(max(flat.shape[-1], 1))
    mask = ~torch.eye(SCENARIO_COUNT, dtype=torch.bool, device=particles.device)
    off_diagonal = distances[:, mask]
    return F.relu(minimum_distance - off_diagonal).pow(2).mean()


def scenario_weight_entropy_loss(weights: Tensor) -> Tensor:
    """Non-negative penalty whose minimum is uniform scenario utilization."""

    weights = normalize_scenario_weights(weights)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
    return (math.log(SCENARIO_COUNT) - entropy).mean().clamp_min(0.0)


def differentiable_realized_settlement_cost(
    day_ahead_schedule: Tensor,
    realized_grid_power: Tensor,
    day_ahead_price: Tensor,
    real_time_price: Tensor,
    *,
    interval_hours: float = 0.25,
    additional_cost: Tensor | None = None,
) -> Tensor:
    """Standard differentiable two-settlement cost for each batch member."""

    if interval_hours <= 0.0:
        raise ValueError("interval_hours must be positive")
    tensors = [day_ahead_schedule, realized_grid_power, day_ahead_price, real_time_price]
    tensors = [value.squeeze(-1) if value.ndim >= 3 and value.shape[-1] == 1 else value for value in tensors]
    try:
        day_ahead_schedule, realized_grid_power, day_ahead_price, real_time_price = torch.broadcast_tensors(*tensors)
    except RuntimeError as error:
        raise ValueError("settlement inputs are not broadcast-compatible") from error
    imbalance = realized_grid_power - day_ahead_schedule
    interval_cost = (
        day_ahead_price * day_ahead_schedule + real_time_price * imbalance
    ) * interval_hours
    cost = interval_cost.sum(dim=-1) if interval_cost.ndim > 1 else interval_cost
    if additional_cost is not None:
        cost = cost + additional_cost
    return cost


def realized_settlement_loss(*args: Any, **kwargs: Any) -> Tensor:
    return differentiable_realized_settlement_cost(*args, **kwargs).mean()


def cvar_loss(
    scenario_costs: Tensor,
    weights: Tensor,
    *,
    alpha: float = 0.95,
    eta: Tensor | None = None,
) -> Tensor:
    """Mean differentiable upper-tail CVaR across the batch."""

    if eta is None:
        eta = weighted_quantile(
            scenario_costs.unsqueeze(-1), weights, alpha
        ).squeeze(-1)
    return differentiable_weighted_cvar(
        scenario_costs, weights, eta, alpha=alpha
    ).mean()


def sla_violation_loss(
    violation: Tensor | None = None,
    *,
    demand: Tensor | None = None,
    served: Tensor | None = None,
) -> Tensor:
    """Positive unmet work/deadline violation loss."""

    if violation is None:
        if demand is None or served is None:
            raise ValueError("provide violation or both demand and served tensors")
        violation = demand - served
    return F.relu(violation).mean()


def migration_cost_loss(
    migration: Tensor,
    *,
    unit_cost: Tensor | float | None = None,
    capacity: Tensor | float | None = None,
) -> Tensor:
    """Migration operating cost plus any differentiable capacity excess."""

    if unit_cost is None:
        cost = migration.abs()
    else:
        cost = migration.abs() * torch.as_tensor(
            unit_cost, dtype=migration.dtype, device=migration.device
        )
    result = cost.mean()
    if capacity is not None:
        capacity_tensor = torch.as_tensor(
            capacity, dtype=migration.dtype, device=migration.device
        )
        result = result + F.relu(migration.abs() - capacity_tensor).pow(2).mean()
    return result


def feasibility_loss(
    feasibility_probability: Tensor,
    *,
    target: Tensor | None = None,
    constraint_violation: Tensor | None = None,
) -> Tensor:
    """Calibrate feasibility confidence and penalize physical violations."""

    probability = feasibility_probability.clamp(1e-6, 1.0 - 1e-6)
    if target is None:
        if constraint_violation is None:
            target = torch.ones_like(probability)
        else:
            reduced = constraint_violation
            while reduced.ndim > probability.ndim:
                reduced = reduced.amax(dim=-1)
            target = (reduced <= 1e-7).to(probability.dtype)
    target = target.to(dtype=probability.dtype, device=probability.device)
    result = F.binary_cross_entropy(probability, target)
    if constraint_violation is not None:
        result = result + F.relu(constraint_violation).pow(2).mean()
    return result


def teacher_imitation_loss(
    neural_actions: Mapping[str, Tensor],
    teacher_actions: Mapping[str, Tensor],
) -> Tensor:
    """Smooth-L1 action imitation over every shared policy component."""

    common = sorted(set(neural_actions).intersection(teacher_actions))
    common = [name for name in common if not name.endswith("logits")]
    if not common:
        reference = next(iter(neural_actions.values()), None)
        if reference is None:
            raise ValueError("neural_actions cannot be empty")
        return _zero_like(reference)
    components: list[Tensor] = []
    for name in common:
        prediction = neural_actions[name]
        target = teacher_actions[name].to(prediction)
        if prediction.shape != target.shape:
            raise ValueError(
                f"teacher action {name!r} shape {tuple(target.shape)} does not "
                f"match neural action {tuple(prediction.shape)}"
            )
        components.append(F.smooth_l1_loss(prediction, target))
    return torch.stack(components).mean()


imitation_loss = teacher_imitation_loss


@dataclass(frozen=True)
class LossWeights:
    """Governing default weights from the supplied R3 configuration."""

    router_balance: float = 0.02
    router_stability: float = 0.01
    marginal: float = 1.0
    energy_score: float = 0.5
    variogram: float = 0.25
    tail_coverage: float = 0.5
    diversity: float = 0.1
    weight_entropy: float = 0.02
    realized_cost: float = 1.0
    cvar: float = 0.25
    sla: float = 10.0
    migration: float = 1.0
    feasibility: float = 20.0
    teacher_imitation: float = 0.5


def _scenario_target(targets: Mapping[str, Any]) -> Tensor | None:
    direct = targets.get("scenario_target", targets.get("joint_target"))
    if direct is not None:
        return direct
    if all(name in targets for name in SCENARIO_VARIABLES):
        columns = []
        for name in SCENARIO_VARIABLES:
            value = targets[name]
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value.squeeze(-1)
            columns.append(value)
        return torch.stack(columns, dim=-1)
    return None


class E2ELoss(nn.Module):
    """Composable R3 objective; computes terms only when targets are supplied."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        cvar_alpha: float = 0.95,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.cvar_alpha = float(cvar_alpha)

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        *,
        previous_router_probabilities: Tensor | None = None,
        teacher_actions: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        reference: Tensor = outputs["latent"]
        terms: dict[str, Tensor] = {}
        routing = outputs["routing"]
        terms["routing_balance"] = routing_balance_loss(
            routing["router_probabilities"], routing.get("expert_load")
        )
        terms["routing_stability"] = routing_stability_loss(
            routing["router_probabilities"], previous_router_probabilities
        )

        particles = outputs["scenario"]["particles"]
        scenario_weights = outputs["scenario"]["weights"]
        scenario_target = _scenario_target(targets)
        if scenario_target is not None:
            scenario_target = scenario_target.to(particles)
            terms["marginal"] = marginal_scenario_loss(
                particles, scenario_weights, scenario_target
            )
            terms["energy_score"] = energy_score_loss(
                particles, scenario_weights, scenario_target
            )
            terms["variogram"] = variogram_score_loss(
                particles, scenario_weights, scenario_target
            )
            terms["tail_coverage"] = tail_coverage_loss(
                particles, scenario_weights, scenario_target
            )
        else:
            for name in ("marginal", "energy_score", "variogram", "tail_coverage"):
                terms[name] = _zero_like(reference)
        terms["diversity"] = scenario_diversity_loss(particles)
        terms["weight_entropy"] = scenario_weight_entropy_loss(scenario_weights)

        settlement = targets.get("settlement")
        if settlement is not None:
            realized_costs = differentiable_realized_settlement_cost(
                settlement["day_ahead_schedule"],
                settlement["realized_grid_power"],
                settlement["day_ahead_price"],
                settlement["real_time_price"],
                interval_hours=float(settlement.get("interval_hours", 0.25)),
                additional_cost=settlement.get("additional_cost"),
            )
            terms["realized_cost"] = realized_costs.mean()
        elif "realized_cost" in targets:
            terms["realized_cost"] = targets["realized_cost"].to(reference).mean()
        else:
            terms["realized_cost"] = outputs["risk"]["expected_cost"].mean()

        scenario_costs = targets.get(
            "scenario_costs", outputs["risk"]["scenario_costs"]
        ).to(reference)
        cost_weights = targets.get("scenario_cost_weights", scenario_weights).to(reference)
        terms["cvar"] = cvar_loss(
            scenario_costs,
            cost_weights,
            alpha=self.cvar_alpha,
            eta=outputs["risk"].get("eta"),
        )

        violation = targets.get("sla_violation", targets.get("sla_violations"))
        if violation is not None:
            terms["sla"] = sla_violation_loss(violation.to(reference))
        else:
            terms["sla"] = _zero_like(reference)
        terms["migration"] = migration_cost_loss(
            outputs["actions"]["migration"],
            unit_cost=targets.get("migration_unit_cost"),
            capacity=targets.get("migration_capacity"),
        )
        terms["feasibility"] = feasibility_loss(
            outputs["confidence"]["feasibility"],
            target=targets.get("feasibility_target"),
            constraint_violation=targets.get("constraint_violation"),
        )
        teacher_actions = teacher_actions or targets.get("teacher_actions")
        terms["teacher_imitation"] = (
            teacher_imitation_loss(outputs["actions"], teacher_actions)
            if teacher_actions is not None
            else _zero_like(reference)
        )

        weight_map = {
            "routing_balance": self.weights.router_balance,
            "routing_stability": self.weights.router_stability,
            "marginal": self.weights.marginal,
            "energy_score": self.weights.energy_score,
            "variogram": self.weights.variogram,
            "tail_coverage": self.weights.tail_coverage,
            "diversity": self.weights.diversity,
            "weight_entropy": self.weights.weight_entropy,
            "realized_cost": self.weights.realized_cost,
            "cvar": self.weights.cvar,
            "sla": self.weights.sla,
            "migration": self.weights.migration,
            "feasibility": self.weights.feasibility,
            "teacher_imitation": self.weights.teacher_imitation,
        }
        total = torch.stack(
            [terms[name] * weight for name, weight in weight_map.items()]
        ).sum()
        return {**terms, "total": total}


__all__ = [
    "E2ELoss",
    "LossWeights",
    "cvar_loss",
    "differentiable_realized_settlement_cost",
    "energy_score_loss",
    "feasibility_loss",
    "imitation_loss",
    "marginal_scenario_loss",
    "migration_cost_loss",
    "realized_settlement_loss",
    "routing_balance_loss",
    "routing_stability_loss",
    "scenario_diversity_loss",
    "scenario_weight_entropy_loss",
    "sla_violation_loss",
    "tail_coverage_loss",
    "teacher_imitation_loss",
    "variogram_score_loss",
    "weighted_marginal_crps",
    "weighted_quantile",
]
