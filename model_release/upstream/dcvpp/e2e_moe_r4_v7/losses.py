"""Loss contracts for the R4-V7 baseline-anchored joint world-policy model.

The scientific boundary in V7 is deliberate:

* distribution parameters are trained only by proper distribution scores;
* decision updates consume a detached distribution memory;
* policy risk uses the Rockafellar--Uryasev (RU) objective over executed,
  post-shield worlds rather than a free neural scalar; and
* every dimensional quantity is divided by one frozen train-only scale.

The functions accept mappings instead of model classes so the training
protocol remains independent of the concrete encoder/decoder implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from dcvpp.e2e_moe_r4 import losses as r4_losses


PROPER_SCORE_KEYS = (
    "center_scale",
    "energy_score",
    "marginal_crps",
    "interval",
    "tail",
    "dependence",
)

DEFAULT_DISTRIBUTION_WEIGHTS: dict[str, float] = {
    "center_scale": 0.10,
    "energy_score": 1.00,
    "marginal_crps": 1.00,
    "interval": 0.25,
    "tail": 0.75,
    "dependence": 0.50,
    "coverage90_hinge": 0.50,
    "coverage95_hinge": 0.75,
    "diversity": 0.02,
    # Identity is a structural assertion/gate, never a soft optimization term.
    "identity": 0.0,
}

DEFAULT_POLICY_WEIGHTS: dict[str, float] = {
    "relative_cost": 1.0,
    "ru_cvar": 0.25,
    "sla": 10.0,
    "terminal": 5.0,
    "shield_intervention": 1.0,
    "action_smoothness": 0.02,
    "critic": 0.25,
    "anchor_imitation": 0.0,
}


class V7LossError(ValueError):
    """Raised when loss inputs violate the registered V7 semantics."""


def _tensor(value: Any, *, like: Tensor) -> Tensor:
    return value if isinstance(value, Tensor) else torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _positive_scalar(value: Tensor | float, *, like: Tensor, label: str) -> Tensor:
    scale = _tensor(value, like=like).reshape(-1)
    if scale.numel() != 1:
        raise V7LossError(f"{label} must be one scalar")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise V7LossError(f"{label} must be finite and positive")
    return scale[0]


def _finite(value: Tensor, label: str) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise V7LossError(f"{label} contains non-finite values")
    return value


def coverage_hinge_loss(
    absolute_error: Tensor,
    *,
    tolerance: float = 0.03,
) -> Tensor:
    """Penalize calibration only beyond the preregistered error tolerance."""

    if not math.isfinite(float(tolerance)) or not 0.0 <= float(tolerance) < 1.0:
        raise V7LossError("coverage tolerance must lie in [0, 1)")
    return F.relu(absolute_error - float(tolerance)).square()


def assert_dart_identity(samples: Tensor, *, tolerance: float = 1.0e-6) -> None:
    """Fail closed when a model does not construct RT/DA/DART by identity."""

    if samples.ndim != 4 or samples.shape[-1] != 21:
        raise V7LossError("identity samples must have shape [day, world, 96, 21]")
    residual = samples[..., 0:4] - samples[..., 4:8] - samples[..., 8:12]
    if not bool(torch.isfinite(residual).all()):
        raise V7LossError("DA-RT-DART identity contains non-finite values")
    if float(residual.detach().abs().max().cpu()) > float(tolerance):
        raise V7LossError(
            f"DA-RT-DART identity exceeds structural tolerance {float(tolerance):.1e}"
        )


def normalize_distribution_channels(
    samples: Tensor,
    target: Tensor,
    channel_scale: Tensor,
    *,
    center: Tensor | None = None,
    scale: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Apply the frozen 21-channel residual scale before proper scoring.

    The scale is one train-only vector, never a batch statistic.  Identity is
    intentionally checked on the unscaled samples before calling this helper.
    """

    if samples.ndim != 4 or samples.shape[-1] != 21:
        raise V7LossError("distribution samples must be [day, world, 96, 21]")
    if target.shape != samples.shape[:1] + samples.shape[2:]:
        raise V7LossError("distribution target must be [day,96,21]")
    frozen = channel_scale.to(device=samples.device, dtype=samples.dtype)
    if frozen.shape != (21,):
        raise V7LossError("frozen channel_scale must have shape [21]")
    if not bool(torch.isfinite(frozen).all()) or bool((frozen <= 0).any()):
        raise V7LossError("frozen channel_scale must be finite and positive")
    normalized_samples = samples / frozen.view(1, 1, 1, 21)
    normalized_target = target.to(samples) / frozen.view(1, 1, 21)
    normalized_center = None
    if center is not None:
        if center.shape != target.shape:
            raise V7LossError("distribution center must match target")
        normalized_center = center.to(samples) / frozen.view(1, 1, 21)
    normalized_scale = None
    if scale is not None:
        if scale.shape != target.shape:
            raise V7LossError("distribution scale must match target")
        normalized_scale = scale.to(samples) / frozen.view(1, 1, 21)
    return (
        normalized_samples,
        normalized_target,
        normalized_center,
        normalized_scale,
    )


def baseline_normalized_distribution_terms(
    samples: Tensor,
    target: Tensor,
    *,
    baseline_scales: Mapping[str, Tensor | float],
    weights: Tensor | None = None,
    center: Tensor | None = None,
    scale: Tensor | None = None,
    identity_samples: Tensor | None = None,
    coverage_temperature: float = 0.05,
    coverage_tolerance: float = 0.03,
    diversity_floor: float = 0.05,
) -> dict[str, Tensor]:
    """Return V7 D0/calibration terms on frozen baseline metric scales.

    ``baseline_scales`` must come from the frozen residual baseline and may not
    be recomputed per batch.  Coverage is dimensionless and therefore uses a
    tolerance hinge rather than an arbitrary scale.  The raw metrics are
    returned with a ``raw_`` prefix for candidate selection and paper gates.
    """

    missing = set(PROPER_SCORE_KEYS) - set(baseline_scales)
    if missing:
        raise V7LossError(f"missing frozen baseline scales: {sorted(missing)}")
    raw = r4_losses.distribution_proper_loss_terms(
        samples,
        target,
        weights=weights,
        center=center,
        scale=scale,
        identity_samples=identity_samples,
        coverage_temperature=coverage_temperature,
        diversity_floor=diversity_floor,
    )
    terms: dict[str, Tensor] = {}
    for key in PROPER_SCORE_KEYS:
        denominator = _positive_scalar(
            baseline_scales[key], like=raw[key], label=f"baseline_scales.{key}"
        )
        terms[key] = raw[key] / denominator
        terms[f"raw_{key}"] = raw[key].detach()
    terms["coverage90_hinge"] = coverage_hinge_loss(
        raw["coverage90"], tolerance=coverage_tolerance
    )
    terms["coverage95_hinge"] = coverage_hinge_loss(
        raw["coverage95"], tolerance=coverage_tolerance
    )
    terms["raw_coverage90_error"] = raw["coverage90"].detach()
    terms["raw_coverage95_error"] = raw["coverage95"].detach()
    terms["diversity"] = raw["diversity"]
    terms["identity"] = raw["identity"]
    terms["raw_identity"] = raw["identity"].detach()
    return terms


def compose_distribution_loss(
    terms: Mapping[str, Tensor],
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Compose only differentiable registered terms, never ``raw_*`` metrics."""

    selected = DEFAULT_DISTRIBUTION_WEIGHTS if weights is None else dict(weights)
    if any(key.startswith("raw_") for key in selected):
        raise V7LossError("raw reporting metrics cannot enter the training loss")
    return r4_losses.compose_weighted_loss(terms, selected)


def rockafellar_uryasev_cvar(
    costs: Tensor,
    eta: Tensor,
    *,
    weights: Tensor | None = None,
    alpha: float = 0.95,
    smooth_temperature: float = 0.02,
) -> Tensor:
    """Differentiable RU-CVaR over post-shield world costs.

    ``costs`` is ``[day, world]``.  A small softplus continuation is used in
    training; exact empirical CVaR remains an evaluator-only metric at N=512.
    """

    if costs.ndim != 2 or costs.shape[1] < 2:
        raise V7LossError("costs must have shape [day, world] with world >= 2")
    _finite(costs, "costs")
    if not 0.0 < float(alpha) < 1.0:
        raise V7LossError("alpha must lie strictly between zero and one")
    if not math.isfinite(float(smooth_temperature)) or float(smooth_temperature) <= 0.0:
        raise V7LossError("smooth_temperature must be finite and positive")
    threshold = eta.reshape(-1).to(costs)
    if threshold.shape[0] != costs.shape[0]:
        raise V7LossError("eta must provide one threshold per day")
    if weights is None:
        normalized = costs.new_full(costs.shape, 1.0 / float(costs.shape[1]))
    else:
        if weights.shape != costs.shape:
            raise V7LossError("world weights must match costs")
        normalized = r4_losses.normalize_weights(weights).to(costs)
    temperature = float(smooth_temperature)
    excess = temperature * F.softplus((costs - threshold[:, None]) / temperature)
    return threshold + (normalized * excess).sum(1) / (1.0 - float(alpha))


def terminal_state_loss(
    terminal_soc_gap: Tensor,
    terminal_backlog: Tensor,
    *,
    soc_scale: Tensor | float = 1.0,
    backlog_scale: Tensor | float = 1.0,
) -> Tensor:
    """Per-day terminal feasibility penalty on frozen physical scales."""

    soc = terminal_soc_gap
    backlog = terminal_backlog.to(soc)
    if soc.shape[0] != backlog.shape[0]:
        raise V7LossError("terminal state tensors must share a day axis")
    soc_denominator = _positive_scalar(soc_scale, like=soc, label="soc_scale")
    backlog_denominator = _positive_scalar(
        backlog_scale, like=soc, label="backlog_scale"
    )
    soc_value = (soc / soc_denominator).square().flatten(1).mean(1) if soc.ndim > 1 else (soc / soc_denominator).square()
    backlog_value = (
        (backlog / backlog_denominator).square().flatten(1).mean(1)
        if backlog.ndim > 1
        else (backlog / backlog_denominator).square()
    )
    return soc_value + backlog_value


def action_smoothness_loss(actions: Mapping[str, Tensor], batch_size: int) -> Tensor:
    """Average squared temporal change for the three registered free actions."""

    values: list[Tensor] = []
    reference: Tensor | None = None
    for name in ("day_ahead", "battery", "compute"):
        action = actions.get(name)
        if not isinstance(action, Tensor):
            raise V7LossError(f"policy.actions.{name} is required")
        if action.shape[0] != batch_size:
            raise V7LossError(f"policy.actions.{name} has the wrong day axis")
        action = action.float()
        reference = action
        if action.ndim >= 2 and action.shape[1] > 1:
            difference = action[:, 1:] - action[:, :-1]
            values.append(difference.square().flatten(1).mean(1))
    if values:
        return torch.stack(values, 1).mean(1)
    assert reference is not None
    return reference.reshape(batch_size, -1).sum(1) * 0.0


def safe_anchor_imitation_loss(
    actions: Mapping[str, Tensor],
    anchors: Mapping[str, Tensor] | None,
    batch_size: int,
) -> Tensor:
    """Imitate the analytic EDF/physics anchor in normalized action space."""

    reference = actions.get("day_ahead")
    if not isinstance(reference, Tensor):
        raise V7LossError("policy.actions.day_ahead is required")
    if anchors is None:
        return reference.reshape(batch_size, -1).sum(1) * 0.0
    values: list[Tensor] = []
    for name in ("day_ahead", "battery", "compute"):
        action = actions.get(name)
        anchor = anchors.get(name)
        if not isinstance(action, Tensor) or not isinstance(anchor, Tensor):
            raise V7LossError(f"safe anchor must provide normalized {name} actions")
        if action.shape != anchor.shape or action.shape[0] != batch_size:
            raise V7LossError(f"safe anchor {name} shape must match policy action")
        values.append(
            (action.float() - anchor.to(action).float()).square().flatten(1).mean(1)
        )
    return torch.stack(values, dim=1).mean(1)


def _required(mapping: Mapping[str, Any], aliases: tuple[str, ...], label: str) -> Tensor:
    for alias in aliases:
        value = mapping.get(alias)
        if isinstance(value, Tensor):
            return value
    raise V7LossError(f"{label} is required (accepted keys: {aliases})")


def _per_day(value: Tensor, batch_size: int, label: str) -> Tensor:
    if value.shape[0] != batch_size:
        raise V7LossError(f"{label} has the wrong day axis")
    return value if value.ndim == 1 else value.flatten(1).mean(1)


def joint_policy_loss_terms(
    rollout: Mapping[str, Any],
    critic: Mapping[str, Any],
    actions: Mapping[str, Tensor],
    *,
    normalized_actions: Mapping[str, Tensor] | None = None,
    action_plans: Mapping[str, Tensor] | None = None,
    safe_anchor_actions: Mapping[str, Tensor] | None = None,
    cost_scale: Tensor | float,
    soc_scale: Tensor | float = 1.0,
    backlog_scale: Tensor | float = 1.0,
    alpha: float = 0.95,
    smooth_temperature: float = 0.02,
) -> dict[str, Tensor]:
    """Build the P0/J1--J3/A1--A2 decision objective.

    Rollout quantities must describe actions *after* the shared constraint
    shield.  Neural critic outputs are supervised auxiliaries; the actual
    policy risk term is always recomputed from the rollout world costs.
    """

    world_cost = _required(
        rollout, ("realized_cost", "world_cost", "post_shield_cost"), "rollout.realized_cost"
    ).float()
    if world_cost.ndim != 2 or world_cost.shape[1] < 2:
        raise V7LossError("rollout realized_cost must be [day, world>=2]")
    batch_size = world_cost.shape[0]
    denominator = _positive_scalar(cost_scale, like=world_cost, label="cost_scale")
    world_weights = rollout.get("weights", rollout.get("world_weights"))
    if world_weights is not None and not isinstance(world_weights, Tensor):
        world_weights = torch.as_tensor(world_weights, device=world_cost.device, dtype=world_cost.dtype)
    elif isinstance(world_weights, Tensor):
        world_weights = world_weights.float()

    baseline_cost = rollout.get("baseline_cost")
    expected_cost = (
        world_cost.mean(1)
        if world_weights is None
        else (
            r4_losses.normalize_weights(world_weights).to(world_cost) * world_cost
        ).sum(1)
    )
    if baseline_cost is None:
        relative_cost = expected_cost / denominator
    else:
        baseline = _tensor(baseline_cost, like=world_cost)
        baseline = _per_day(baseline, batch_size, "rollout.baseline_cost")
        relative_cost = (expected_cost - baseline) / denominator

    eta = _required(critic, ("var", "eta", "cvar"), "critic.var").float()
    eta = _per_day(eta, batch_size, "critic.var")
    raw_ru = rockafellar_uryasev_cvar(
        world_cost,
        eta,
        weights=world_weights,
        alpha=alpha,
        smooth_temperature=smooth_temperature,
    )
    ru = raw_ru
    baseline_cvar = rollout.get("baseline_cvar")
    if baseline_cvar is not None:
        ru = ru - _per_day(_tensor(baseline_cvar, like=world_cost), batch_size, "rollout.baseline_cvar")

    sla = _per_day(
        _required(rollout, ("sla_violation", "normalized_sla"), "rollout.sla_violation").float(),
        batch_size,
        "rollout.sla_violation",
    )
    intervention = _per_day(
        _required(
            rollout,
            ("shield_intervention", "intervention"),
            "rollout.shield_intervention",
        ),
        batch_size,
        "rollout.shield_intervention",
    )
    terminal = terminal_state_loss(
        _required(rollout, ("terminal_soc_gap",), "rollout.terminal_soc_gap").float(),
        _required(rollout, ("terminal_backlog",), "rollout.terminal_backlog").float(),
        soc_scale=soc_scale,
        backlog_scale=backlog_scale,
    )

    predicted_expected = _per_day(
        _required(critic, ("expected", "expected_cost"), "critic.expected").float(),
        batch_size,
        "critic.expected",
    )
    predicted_cvar = _per_day(
        _required(critic, ("cvar",), "critic.cvar").float(), batch_size, "critic.cvar"
    )
    predicted_sla = _per_day(
        _required(critic, ("sla", "sla_risk"), "critic.sla").float(),
        batch_size,
        "critic.sla",
    )
    predicted_intervention = _per_day(
        _required(
            critic,
            ("intervention", "shield_intervention_risk"),
            "critic.intervention",
        ),
        batch_size,
        "critic.intervention",
    )
    critic_loss = (
        ((predicted_expected - expected_cost.detach()) / denominator).square()
        + ((predicted_cvar - raw_ru.detach()) / denominator).square()
        + (predicted_sla - sla.detach()).square()
        + (predicted_intervention - intervention.detach()).square()
    )
    return {
        "relative_cost": relative_cost,
        "ru_cvar": ru / denominator,
        "sla": sla,
        "terminal": terminal,
        "shield_intervention": intervention,
        "action_smoothness": action_smoothness_loss(
            actions if action_plans is None else action_plans, batch_size
        ),
        "critic": critic_loss,
        "anchor_imitation": safe_anchor_imitation_loss(
            (
                action_plans
                if action_plans is not None
                else actions if normalized_actions is None else normalized_actions
            ),
            safe_anchor_actions,
            batch_size,
        ),
        "raw_expected_cost": expected_cost.detach(),
        "raw_ru_cvar": raw_ru.detach(),
    }


def compose_policy_loss(
    terms: Mapping[str, Tensor],
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    selected = DEFAULT_POLICY_WEIGHTS if weights is None else dict(weights)
    if any(key.startswith("raw_") for key in selected):
        raise V7LossError("raw reporting metrics cannot enter the training loss")
    return r4_losses.compose_weighted_loss(terms, selected)


__all__ = [
    "DEFAULT_DISTRIBUTION_WEIGHTS",
    "DEFAULT_POLICY_WEIGHTS",
    "PROPER_SCORE_KEYS",
    "V7LossError",
    "action_smoothness_loss",
    "assert_dart_identity",
    "baseline_normalized_distribution_terms",
    "compose_distribution_loss",
    "compose_policy_loss",
    "coverage_hinge_loss",
    "joint_policy_loss_terms",
    "normalize_distribution_channels",
    "rockafellar_uryasev_cvar",
    "safe_anchor_imitation_loss",
    "terminal_state_loss",
]
