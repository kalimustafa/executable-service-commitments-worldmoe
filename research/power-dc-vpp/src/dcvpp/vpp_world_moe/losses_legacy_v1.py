"""Frozen R5-V1 legacy phase losses for the unified VPP world model.

The functions in this module intentionally operate on packet-like objects
(dataclasses or mappings).  This keeps the scientific objectives independent
from the concrete network implementation while still enforcing one canonical
set of phase semantics.

This module is retained only for an explicit same-architecture loss ablation.
The formal R5-V2 path imports :mod:`dcvpp.vpp_world_moe.losses` instead.

All probabilistic, risk, and conservation reductions are performed in FP32.
Autocast may still be used for the neural forward pass by :mod:`trainer`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor


PhaseName = Literal["w0", "w1", "u0", "p1", "j1", "c1"]
TRAINING_PHASES: tuple[PhaseName, ...] = ("w0", "w1", "u0", "p1", "j1")
DYNAMICS_HORIZONS: tuple[int, ...] = (1, 4, 16, 96)


class WorldLossError(ValueError):
    """A phase packet cannot support its registered objective."""


@dataclass(frozen=True)
class WorldLossWeights:
    reconstruction: float = 1.0
    posterior_kl: float = 0.02
    dynamics: float = 1.0
    observation: float = 0.35
    nll: float = 0.25
    crps: float = 0.30
    energy: float = 0.20
    variogram: float = 0.10
    coverage: float = 0.05
    tail: float = 0.05
    diversity: float = 0.05
    expected_cost: float = 1.0
    cvar: float = 0.25
    feasibility: float = 10.0
    sla: float = 5.0
    terminal: float = 2.0
    smoothness: float = 0.01
    shield_correction: float = 0.05
    auxiliary: float = 0.01


@dataclass(frozen=True)
class PhaseLoss:
    """A differentiable scalar plus detached, audit-friendly components."""

    phase: PhaseName
    total: Tensor
    components: Mapping[str, Tensor]

    def detached(self) -> dict[str, float]:
        return {
            name: float(value.detach().to(dtype=torch.float32).cpu())
            for name, value in self.components.items()
        }


def packet_mapping(value: Any) -> Mapping[str, Any]:
    """Return a read-only mapping view for a dataclass or mapping packet."""

    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if hasattr(value, "__dict__"):
        return vars(value)
    raise WorldLossError(f"expected a packet-like value, got {type(value).__name__}")


def _lookup(value: Any, *names: str, required: bool = True) -> Any:
    current = packet_mapping(value)
    for name in names:
        if name in current:
            return current[name]
    if required:
        raise WorldLossError(f"packet is missing one of {list(names)}")
    return None


def _tensor(value: Any, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise WorldLossError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        value = value.float()
    return value


def _fp32(value: Tensor) -> Tensor:
    return value.to(dtype=torch.float32)


def _zero(reference: Tensor) -> Tensor:
    return reference.sum(dtype=torch.float32) * 0.0


def _finite(name: str, value: Tensor) -> Tensor:
    if value.ndim != 0:
        raise WorldLossError(f"{name} must reduce to one scalar")
    if not bool(torch.isfinite(value.detach())):
        raise WorldLossError(f"{name} is non-finite")
    return value


def _masked_huber(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    prediction32, target32 = _fp32(prediction), _fp32(target)
    if prediction32.shape != target32.shape:
        raise WorldLossError(
            f"reconstruction shape mismatch: {prediction32.shape} != {target32.shape}"
        )
    raw = F.smooth_l1_loss(prediction32, target32, reduction="none")
    if mask is None:
        return raw.mean()
    mask32 = _fp32(mask)
    try:
        mask32 = torch.broadcast_to(mask32, raw.shape)
    except RuntimeError as exc:
        raise WorldLossError("reconstruction mask is not broadcastable") from exc
    denominator = mask32.sum().clamp_min(1.0)
    return (raw * mask32).sum() / denominator


def _auxiliary_loss(packet: Any, reference: Tensor) -> Tensor:
    raw = _lookup(packet, "auxiliary_losses", "moe_auxiliary", required=False)
    if raw is None:
        return _zero(reference)
    if isinstance(raw, Tensor):
        return _fp32(raw).mean()
    if isinstance(raw, Mapping):
        values = [
            _fp32(_tensor(value, label=f"auxiliary_losses.{name}")).mean()
            for name, value in raw.items()
        ]
        return torch.stack(values).sum() if values else _zero(reference)
    raise WorldLossError("auxiliary losses must be a Tensor or mapping")


def _posterior_kl(packet: Any, reference: Tensor) -> Tensor:
    explicit = _lookup(packet, "posterior_kl", "latent_kl", required=False)
    if explicit is not None:
        return _fp32(_tensor(explicit, label="posterior_kl")).mean()
    mean = _lookup(packet, "posterior_mean", required=False)
    logvar = _lookup(packet, "posterior_logvar", required=False)
    prior_mean = _lookup(packet, "prior_mean", required=False)
    prior_logvar = _lookup(packet, "prior_logvar", required=False)
    if any(item is None for item in (mean, logvar, prior_mean, prior_logvar)):
        return _zero(reference)
    mean32 = _fp32(_tensor(mean, label="posterior_mean"))
    logvar32 = _fp32(_tensor(logvar, label="posterior_logvar"))
    prior_mean32 = _fp32(_tensor(prior_mean, label="prior_mean"))
    prior_logvar32 = _fp32(_tensor(prior_logvar, label="prior_logvar"))
    if not (
        mean32.shape == logvar32.shape == prior_mean32.shape == prior_logvar32.shape
    ):
        raise WorldLossError("posterior/prior Gaussian parameter shapes differ")
    variance_ratio = (logvar32 - prior_logvar32).exp()
    squared = (mean32 - prior_mean32).square() * (-prior_logvar32).exp()
    return 0.5 * (prior_logvar32 - logvar32 + variance_ratio + squared - 1.0).mean()


def w0_masked_world_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
) -> PhaseLoss:
    """Masked reconstruction and posterior regularization for W0."""

    predictions = packet_mapping(
        _lookup(output, "reconstructions", "reconstruction", "decoded")
    )
    targets = packet_mapping(
        _lookup(batch, "reconstruction_targets", "targets", "observations")
    )
    masks_raw = _lookup(batch, "reconstruction_masks", "masks", required=False)
    masks = packet_mapping(masks_raw) if masks_raw is not None else {}
    common = tuple(sorted(set(predictions).intersection(targets)))
    if not common:
        raise WorldLossError("W0 has no matching reconstruction targets")
    reconstruction_terms = []
    components: dict[str, Tensor] = {}
    reference: Tensor | None = None
    for name in common:
        prediction = _tensor(predictions[name], label=f"reconstructions.{name}")
        target = _tensor(targets[name], label=f"targets.{name}")
        mask_value = masks.get(name)
        mask = (
            None if mask_value is None else _tensor(mask_value, label=f"masks.{name}")
        )
        term = _masked_huber(prediction, target, mask)
        components[f"reconstruction/{name}"] = term
        reconstruction_terms.append(term)
        reference = prediction
    assert reference is not None
    reconstruction = torch.stack(reconstruction_terms).mean()
    posterior_kl = _posterior_kl(output, reference)
    auxiliary = _auxiliary_loss(output, reference)
    total = (
        weights.reconstruction * reconstruction
        + weights.posterior_kl * posterior_kl
        + weights.auxiliary * auxiliary
    )
    components.update(
        reconstruction=reconstruction,
        posterior_kl=posterior_kl,
        auxiliary=auxiliary,
        total=_finite("W0 total", total),
    )
    return PhaseLoss("w0", total, components)


def _horizon_mapping(value: Any, *, label: str) -> dict[int, Tensor]:
    raw = packet_mapping(value)
    result: dict[int, Tensor] = {}
    for key, item in raw.items():
        text = str(key).lower().removeprefix("h")
        try:
            horizon = int(text)
        except ValueError as exc:
            raise WorldLossError(f"{label} has invalid horizon {key!r}") from exc
        result[horizon] = _tensor(item, label=f"{label}.{key}")
    return result


def w1_action_dynamics_loss(
    output: Any,
    batch: Any,
    *,
    horizons: Sequence[int] = DYNAMICS_HORIZONS,
    weights: WorldLossWeights = WorldLossWeights(),
) -> PhaseLoss:
    """Action-conditioned latent overshooting at 1/4/16/96 steps."""

    raw_predictions = packet_mapping(
        _lookup(output, "latent_predictions", "dynamics_predictions")
    )
    embedded_predictions: dict[str, Any] = {}
    embedded_targets: dict[str, Any] = {}
    for key, value in raw_predictions.items():
        if isinstance(value, Mapping) or is_dataclass(value):
            item = packet_mapping(value)
            if "prediction" in item and "target" in item:
                embedded_predictions[str(key)] = item["prediction"]
                embedded_targets[str(key)] = item["target"]
                continue
        embedded_predictions[str(key)] = value
    predictions = _horizon_mapping(embedded_predictions, label="latent_predictions")
    raw_targets = _lookup(output, "latent_targets", "dynamics_targets", required=False)
    if raw_targets is None:
        raw_targets = _lookup(
            batch, "latent_targets", "dynamics_targets", required=False
        )
    if raw_targets is None:
        if not embedded_targets:
            raise WorldLossError("W1 requires true latent targets at every horizon")
        raw_targets = embedded_targets
    targets = _horizon_mapping(raw_targets, label="latent_targets")
    requested = tuple(int(value) for value in horizons)
    if requested != DYNAMICS_HORIZONS:
        raise WorldLossError(f"W1 horizons must be {list(DYNAMICS_HORIZONS)}")
    if any(value not in predictions or value not in targets for value in requested):
        raise WorldLossError("W1 requires latent predictions/targets at 1/4/16/96")
    components: dict[str, Tensor] = {}
    dynamics_terms = []
    horizon_weights = torch.tensor(
        [1.0, 1.0, 0.75, 0.5], device=predictions[1].device, dtype=torch.float32
    )
    for index, horizon in enumerate(requested):
        prediction, target = _fp32(predictions[horizon]), _fp32(targets[horizon])
        if prediction.shape != target.shape:
            raise WorldLossError(f"W1 h{horizon} prediction/target shape mismatch")
        term = F.smooth_l1_loss(prediction, target)
        components[f"latent_h{horizon}"] = term
        dynamics_terms.append(term * horizon_weights[index])
    dynamics = torch.stack(dynamics_terms).sum() / horizon_weights.sum()

    observation = _zero(predictions[1])
    observation_predictions = _lookup(output, "observation_predictions", required=False)
    observation_targets = _lookup(batch, "observation_targets", required=False)
    if observation_targets is not None:
        if observation_predictions is None:
            raise WorldLossError("W1 observation targets require rollout predictions")
        predicted = _horizon_mapping(
            observation_predictions, label="observation_predictions"
        )
        truth = _horizon_mapping(observation_targets, label="observation_targets")
        observation_terms = []
        for horizon in requested:
            if horizon not in predicted or horizon not in truth:
                raise WorldLossError(f"W1 observation rollout is missing h{horizon}")
            observation_terms.append(
                F.smooth_l1_loss(_fp32(predicted[horizon]), _fp32(truth[horizon]))
            )
        observation = torch.stack(observation_terms).mean()
    auxiliary = _auxiliary_loss(output, predictions[1])
    total = (
        weights.dynamics * dynamics
        + weights.observation * observation
        + weights.auxiliary * auxiliary
    )
    components.update(
        dynamics=dynamics,
        observation=observation,
        auxiliary=auxiliary,
        total=_finite("W1 total", total),
    )
    return PhaseLoss("w1", total, components)


def _scenario_and_target(output: Any, batch: Any) -> tuple[Tensor, Tensor]:
    raw_scenarios = _lookup(
        output, "scenarios", "stochastic_rollouts", "samples", required=False
    )
    if raw_scenarios is None:
        rollout = _lookup(output, "rollout", required=False)
        if rollout is not None:
            rollout_packet = packet_mapping(rollout)
            observations = rollout_packet.get("observations")
            if observations is not None:
                raw_scenarios = _lookup(observations, "public_21", required=False)
    if raw_scenarios is None:
        raise WorldLossError("U0 output has no stochastic paths")
    scenarios = _fp32(_tensor(raw_scenarios, label="scenarios"))
    target = _fp32(
        _tensor(
            _lookup(batch, "stochastic_target", "future_target", "scenario_target"),
            label="stochastic_target",
        )
    )
    if scenarios.ndim < 3:
        raise WorldLossError("scenarios must have shape [batch, samples, ...]")
    if scenarios.shape[0] != target.shape[0] or scenarios.shape[2:] != target.shape[1:]:
        raise WorldLossError("scenario and stochastic target shapes are incompatible")
    if scenarios.shape[1] < 2:
        raise WorldLossError("U0 requires at least two stochastic samples")
    return scenarios, target


def _ensemble_crps(scenarios: Tensor, target: Tensor) -> Tensor:
    accuracy = (scenarios - target.unsqueeze(1)).abs().mean()
    spread = (scenarios.unsqueeze(2) - scenarios.unsqueeze(1)).abs().mean()
    return accuracy - 0.5 * spread


def _energy_score(scenarios: Tensor, target: Tensor) -> Tensor:
    samples = scenarios.flatten(start_dim=2)
    truth = target.flatten(start_dim=1)
    accuracy = torch.linalg.vector_norm(samples - truth.unsqueeze(1), dim=-1).mean()
    spread = torch.linalg.vector_norm(
        samples.unsqueeze(2) - samples.unsqueeze(1), dim=-1
    ).mean()
    feature_count = max(int(samples.shape[-1]), 1)
    return (accuracy - 0.5 * spread) / feature_count**0.5


def _variogram_score(scenarios: Tensor, target: Tensor) -> Tensor:
    samples = scenarios.flatten(start_dim=2)
    truth = target.flatten(start_dim=1)
    feature_count = int(samples.shape[-1])
    if feature_count < 2:
        return _zero(scenarios)
    terms = []
    for lag in (1, 4, 16):
        if lag >= feature_count:
            continue
        epsilon = 1.0e-6
        observed = ((truth[:, lag:] - truth[:, :-lag]).abs() + epsilon).sqrt()
        generated = (
            ((samples[:, :, lag:] - samples[:, :, :-lag]).abs() + epsilon)
            .sqrt()
            .mean(dim=1)
        )
        terms.append((generated - observed).square().mean())
    return torch.stack(terms).mean() if terms else _zero(scenarios)


def _gaussian_nll(output: Any, scenarios: Tensor, target: Tensor) -> Tensor:
    mean_value = _lookup(output, "distribution_mean", "scenario_mean", required=False)
    log_scale_value = _lookup(
        output, "distribution_log_scale", "log_scale", required=False
    )
    mean = (
        scenarios.mean(dim=1)
        if mean_value is None
        else _fp32(_tensor(mean_value, label="distribution_mean"))
    )
    if mean.shape != target.shape:
        raise WorldLossError("distribution mean and target shapes differ")
    if log_scale_value is None:
        scale = scenarios.std(dim=1, unbiased=False).clamp_min(1.0e-4)
        log_scale = scale.log()
    else:
        log_scale = _fp32(_tensor(log_scale_value, label="distribution_log_scale"))
        if log_scale.shape != target.shape:
            raise WorldLossError("distribution log-scale and target shapes differ")
        log_scale = log_scale.clamp(-9.0, 8.0)
    standardized = (target - mean) * (-log_scale).exp()
    return (0.5 * standardized.square() + log_scale + 0.9189385332046727).mean()


def _coverage_penalty(scenarios: Tensor, target: Tensor) -> Tensor:
    penalties = []
    for nominal in (0.5, 0.8, 0.9):
        alpha = (1.0 - nominal) / 2.0
        lower = torch.quantile(scenarios, alpha, dim=1)
        upper = torch.quantile(scenarios, 1.0 - alpha, dim=1)
        empirical = ((target >= lower) & (target <= upper)).float().mean()
        penalties.append((empirical - nominal).square())
    return torch.stack(penalties).mean()


def _tail_score(scenarios: Tensor, target: Tensor) -> Tensor:
    losses = []
    for quantile in (0.05, 0.95):
        prediction = torch.quantile(scenarios, quantile, dim=1)
        error = target - prediction
        losses.append(torch.maximum(quantile * error, (quantile - 1.0) * error).mean())
    return torch.stack(losses).mean()


def _diversity_penalty(scenarios: Tensor) -> Tensor:
    flattened = scenarios.flatten(start_dim=2)
    pairwise = (
        torch.linalg.vector_norm(flattened[:, 1:] - flattened[:, :-1], dim=-1)
        / max(int(flattened.shape[-1]), 1) ** 0.5
    )
    scale = flattened.detach().std(dim=(1, 2), unbiased=False).mean().clamp_min(1.0e-4)
    minimum = 0.05 * scale
    collapse = F.relu(minimum - pairwise).mean()
    duplicate = (pairwise <= 1.0e-7).float().mean()
    return collapse + duplicate


def u0_stochastic_calibration_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
) -> PhaseLoss:
    """Proper scores, coverage, tails, and anti-collapse training for U0."""

    scenarios, target = _scenario_and_target(output, batch)
    nll = _gaussian_nll(output, scenarios, target)
    crps = _ensemble_crps(scenarios, target)
    energy = _energy_score(scenarios, target)
    variogram = _variogram_score(scenarios, target)
    coverage = _coverage_penalty(scenarios, target)
    tail = _tail_score(scenarios, target)
    diversity = _diversity_penalty(scenarios)
    auxiliary = _auxiliary_loss(output, scenarios)
    total = (
        weights.nll * nll
        + weights.crps * crps
        + weights.energy * energy
        + weights.variogram * variogram
        + weights.coverage * coverage
        + weights.tail * tail
        + weights.diversity * diversity
        + weights.auxiliary * auxiliary
    )
    components = {
        "nll": nll,
        "crps": crps,
        "energy": energy,
        "variogram": variogram,
        "coverage": coverage,
        "tail": tail,
        "diversity": diversity,
        "auxiliary": auxiliary,
        "total": _finite("U0 total", total),
    }
    return PhaseLoss("u0", total, components)


def _planner_field(output: Any, *names: str, required: bool = True) -> Any:
    planner = _lookup(output, "planner", "planning", "plan", required=False)
    if planner is not None:
        nested = packet_mapping(planner)
        for name in names:
            if name in nested:
                candidate = nested[name]
                if name not in {"actions", "shielded_actions"} or isinstance(
                    candidate, Tensor
                ):
                    return candidate
        rollout = nested.get("rollout")
        if rollout is not None:
            rollout_mapping = packet_mapping(rollout)
            if any(name in {"scenario_costs", "imagined_costs"} for name in names):
                if "path_cost" in rollout_mapping:
                    return rollout_mapping["path_cost"]
        actions = nested.get("actions", nested.get("shielded_actions"))
        if actions is not None:
            action_mapping = packet_mapping(actions)
            if any(name in {"shielded_actions", "actions"} for name in names):
                ordered = (
                    "battery",
                    "compute",
                    "grid",
                    "pv_used",
                    "pv_curtailed",
                    "unserved_power",
                )
                values = [
                    action_mapping[name] for name in ordered if name in action_mapping
                ]
                if values:
                    return torch.stack(values, dim=-1)
            if any(
                name in {"constraint_violations", "constraint_violation"}
                for name in names
            ):
                if "constraint_violation" in action_mapping:
                    return action_mapping["constraint_violation"]
            if any(name in {"sla_shortfall", "sla_violation"} for name in names):
                if "unserved_power" in action_mapping:
                    return action_mapping["unserved_power"]
            if any(
                name in {"shield_correction", "shield_intervention"} for name in names
            ):
                if "shield_intervention" in action_mapping:
                    return action_mapping["shield_intervention"]
    return _lookup(output, *names, required=required)


def _positive_mean(value: Any, *, label: str, reference: Tensor) -> Tensor:
    if value is None:
        return _zero(reference)
    return F.relu(_fp32(_tensor(value, label=label))).mean()


def p1_latent_planner_loss(
    output: Any,
    batch: Any,
    *,
    cvar_alpha: float = 0.9,
    weights: WorldLossWeights = WorldLossWeights(),
) -> PhaseLoss:
    """End-to-end risk and feasibility objective for the latent planner.

    No expert-action target is used.  The planner learns from differentiable
    imagined costs followed by deterministic physics and an analytic shield.
    """

    scenario_costs = _fp32(
        _tensor(
            _planner_field(output, "scenario_costs", "imagined_costs"),
            label="scenario_costs",
        )
    )
    if scenario_costs.ndim < 2:
        raise WorldLossError("planner scenario_costs must be [batch, scenarios, ...]")
    costs = (
        scenario_costs
        if scenario_costs.ndim == 2
        else scenario_costs.flatten(start_dim=2).sum(dim=-1)
    )
    expected_cost = costs.mean()
    alpha = float(cvar_alpha)
    if not 0.5 <= alpha < 1.0:
        raise WorldLossError("cvar_alpha must be in [0.5, 1)")
    count = max(1, int((1.0 - alpha) * costs.shape[1] + 0.999999))
    cvar = torch.topk(costs, k=count, dim=1).values.mean()
    actions = _fp32(
        _tensor(
            _planner_field(output, "shielded_actions", "actions"),
            label="shielded_actions",
        )
    )
    if actions.ndim < 3:
        raise WorldLossError("planner actions must contain a time axis")
    smoothness = (
        (actions[..., 1:, :] - actions[..., :-1, :]).square().mean()
        if actions.shape[-2] > 1
        else _zero(actions)
    )
    feasibility = _positive_mean(
        _planner_field(output, "constraint_violations", required=False),
        label="constraint_violations",
        reference=actions,
    )
    sla = _positive_mean(
        _planner_field(output, "sla_shortfall", "sla_violation", required=False),
        label="sla_shortfall",
        reference=actions,
    )
    terminal = _positive_mean(
        _planner_field(output, "terminal_violation", required=False),
        label="terminal_violation",
        reference=actions,
    )
    raw_actions = _planner_field(output, "raw_actions", required=False)
    shield_correction = _positive_mean(
        _planner_field(
            output, "shield_correction", "shield_intervention", required=False
        ),
        label="shield_intervention",
        reference=actions,
    )
    if raw_actions is not None:
        raw = _fp32(_tensor(raw_actions, label="raw_actions"))
        if raw.shape == actions.shape:
            shield_correction = shield_correction + (raw - actions).square().mean()
    auxiliary = _auxiliary_loss(output, actions)
    total = (
        weights.expected_cost * expected_cost
        + weights.cvar * cvar
        + weights.feasibility * feasibility
        + weights.sla * sla
        + weights.terminal * terminal
        + weights.smoothness * smoothness
        + weights.shield_correction * shield_correction
        + weights.auxiliary * auxiliary
    )
    components = {
        "expected_cost": expected_cost,
        "cvar": cvar,
        "feasibility": feasibility,
        "sla": sla,
        "terminal": terminal,
        "smoothness": smoothness,
        "shield_correction": shield_correction,
        "auxiliary": auxiliary,
        "total": _finite("P1 total", total),
    }
    return PhaseLoss("p1", total, components)


def j1_protected_joint_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    cvar_alpha: float = 0.9,
) -> PhaseLoss:
    """Protected joint loss over dynamics, uncertainty, and planning."""

    dynamics = w1_action_dynamics_loss(output, batch, weights=weights)
    stochastic = u0_stochastic_calibration_loss(output, batch, weights=weights)
    planner = p1_latent_planner_loss(
        output, batch, weights=weights, cvar_alpha=cvar_alpha
    )
    # The phase objectives are already internally weighted.  Equal outer
    # weights stop any one packet scale from silently replacing the others.
    total = (dynamics.total + stochastic.total + planner.total) / 3.0
    components: dict[str, Tensor] = {
        **{f"dynamics/{name}": value for name, value in dynamics.components.items()},
        **{
            f"stochastic/{name}": value for name, value in stochastic.components.items()
        },
        **{f"planner/{name}": value for name, value in planner.components.items()},
        "total": _finite("J1 total", total),
    }
    return PhaseLoss("j1", total, components)


def phase_loss(
    phase: str,
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    cvar_alpha: float = 0.9,
) -> PhaseLoss:
    """Dispatch one registered training phase objective."""

    normalized = str(phase).lower()
    if normalized == "w0":
        return w0_masked_world_loss(output, batch, weights=weights)
    if normalized == "w1":
        return w1_action_dynamics_loss(output, batch, weights=weights)
    if normalized == "u0":
        return u0_stochastic_calibration_loss(output, batch, weights=weights)
    if normalized == "p1":
        return p1_latent_planner_loss(
            output, batch, weights=weights, cvar_alpha=cvar_alpha
        )
    if normalized == "j1":
        return j1_protected_joint_loss(
            output, batch, weights=weights, cvar_alpha=cvar_alpha
        )
    if normalized == "c1":
        raise WorldLossError("C1 is evaluation-only and has no training loss")
    raise WorldLossError(f"unknown world-model phase: {phase}")
