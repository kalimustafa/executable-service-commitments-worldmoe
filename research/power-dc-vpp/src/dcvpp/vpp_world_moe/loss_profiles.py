"""Explicit R5-V2 training-loss profiles with one proper selection builder.

The matched Huber/mean profile consumes the native R5-V2 packet contract.  It
does not import the frozen R5-V1 loss module and therefore cannot inherit its
silent KL, missing rollout, flattened-variogram, or hard-coverage behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .losses import (
    DYNAMICS_HORIZONS,
    PUBLIC_GROUP_SLICES,
    QUANTILE_LEVELS,
    PhaseLoss,
    WorldLossWeights,
    _queue_tail_route_supervision_loss,
    p1_latent_planner_loss,
    phase_loss as proper_phase_loss,
)


PROPER_PROFILE = "proper_joint_risk_r5_v2"
MATCHED_HUBER_PROFILE = "matched_huber_mean_r5_v2"
MEDIAN_QUANTILE_INDEX = QUANTILE_LEVELS.index(0.50)


class LossProfileError(ValueError):
    """A requested loss profile or R5-V2 packet is incomplete."""


PhaseBuilder = Callable[..., PhaseLoss]


@dataclass(frozen=True)
class LossProfileBuilders:
    name: str
    training_builder: PhaseBuilder
    selection_builder: PhaseBuilder


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if hasattr(value, "__dict__"):
        return vars(value)
    raise LossProfileError(f"{label} must be packet-like")


def _lookup(value: Any, *names: str, label: str, required: bool = True) -> Any:
    packet = _mapping(value, label=label)
    for name in names:
        if name in packet and packet[name] is not None:
            return packet[name]
    if required:
        raise LossProfileError(f"{label} is missing one of {list(names)}")
    return None


def _tensor(value: Any, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise LossProfileError(f"{label} must be a Tensor")
    if not bool(torch.isfinite(value.detach().float()).all()):
        raise LossProfileError(f"{label} contains non-finite values")
    return value.float()


def _finite_scalar(value: Tensor, *, label: str) -> Tensor:
    if value.ndim != 0 or not bool(torch.isfinite(value.detach())):
        raise LossProfileError(f"{label} must be one finite scalar")
    return value


def _families(value: Any, *, label: str, quantile: bool = False) -> dict[str, Tensor]:
    if isinstance(value, Tensor):
        channel_axis = -2 if quantile else -1
        if value.shape[channel_axis] != 21:
            raise LossProfileError(f"{label} must contain 21 public channels")
        return {
            name: value[..., channel_slice, :] if quantile else value[..., channel_slice]
            for name, channel_slice in PUBLIC_GROUP_SLICES.items()
        }
    packet = _mapping(value, label=label)
    aliases = {"workload_volume": "workload", "arrival_volume": "workload"}
    result: dict[str, Tensor] = {}
    for raw_name, item in packet.items():
        name = aliases.get(str(raw_name).lower(), str(raw_name).lower())
        if name in PUBLIC_GROUP_SLICES:
            result[name] = _tensor(item, label=f"{label}.{raw_name}")
    missing = set(PUBLIC_GROUP_SLICES).difference(result)
    if missing:
        raise LossProfileError(f"{label} is missing families {sorted(missing)}")
    return result


def _horizons(value: Any, *, label: str) -> dict[int, Any]:
    packet = _mapping(value, label=label)
    result: dict[int, Any] = {}
    for key, item in packet.items():
        try:
            horizon = int(str(key).lower().removeprefix("h"))
        except ValueError as exc:
            raise LossProfileError(f"{label} has invalid horizon {key!r}") from exc
        result[horizon] = item
    missing = set(DYNAMICS_HORIZONS).difference(result)
    if missing:
        raise LossProfileError(
            f"{label} requires h1/h4/h16/h96; missing {sorted(missing)}"
        )
    return result


def _channel_weights(family: str, channels: int, reference: Tensor) -> Tensor:
    if family == "price":
        if channels != 12:
            raise LossProfileError("price family must be DA/RT/DART 4+4+4")
        return reference.new_tensor(
            [0.25 / 4.0] * 4 + [0.25 / 4.0] * 4 + [0.50 / 4.0] * 4
        )
    if channels <= 0:
        raise LossProfileError(f"{family} has no channels")
    return reference.new_full((channels,), 1.0 / channels)


def _broadcast(value: Tensor, shape: torch.Size, *, label: str) -> Tensor:
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as exc:
        raise LossProfileError(f"{label} is not broadcastable to {tuple(shape)}") from exc


def _transform(value: Tensor, family: str) -> Tensor:
    result = value.float()
    if family in {"workload", "queue", "count"}:
        if bool((result < 0).any()):
            raise LossProfileError(f"{family} must be nonnegative before log1p")
        return torch.log1p(result)
    return result


def _masked_group_huber(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    scale: Tensor,
    family: str,
    label: str,
) -> Tensor:
    prediction32 = _transform(_tensor(prediction, label=f"{label}.prediction"), family)
    target32 = _transform(_tensor(target, label=f"{label}.target"), family)
    if prediction32.shape != target32.shape:
        raise LossProfileError(f"{label} prediction/target shapes differ")
    mask32 = _broadcast(
        _tensor(mask, label=f"{label}.mask"), target32.shape, label=f"{label}.mask"
    )
    scale32 = _broadcast(
        _tensor(scale, label=f"{label}.scale"), target32.shape, label=f"{label}.scale"
    )
    if bool((scale32 <= 0).any()):
        raise LossProfileError(f"{label} scale must be positive")
    raw = F.smooth_l1_loss(
        prediction32 / scale32, target32 / scale32, reduction="none"
    )
    channel = _channel_weights(family, int(target32.shape[-1]), raw)
    channel = channel.reshape((1,) * (raw.ndim - 1) + (channel.numel(),))
    weighted_mask = mask32 * channel
    valid = weighted_mask.sum()
    if not bool(valid.detach() > 0):
        raise LossProfileError(f"{label} has no valid targets")
    return (raw * weighted_mask).sum() / valid


def _assert_price_identity(value: Tensor, *, label: str, channel_axis: int = -1) -> None:
    if value.shape[channel_axis] != 12:
        raise LossProfileError(f"{label} must contain 12 price channels")
    da, rt, dart = torch.split(value.float(), 4, dim=channel_axis)
    tolerance = 2.0e-5 + 2.0e-6 * value.detach().float().abs().amax()
    if bool(((da - rt - dart).abs().amax() > tolerance).detach()):
        raise LossProfileError(f"{label} violates DART=DA-RT")


def _assert_joint_paths(value: Any, *, label: str) -> None:
    packet = _mapping(value, label=label)
    da = _tensor(_lookup(packet, "da", label=label), label=f"{label}.da")
    rt = _tensor(_lookup(packet, "rt", label=label), label=f"{label}.rt")
    dart = _tensor(_lookup(packet, "dart", label=label), label=f"{label}.dart")
    if not (da.shape == rt.shape == dart.shape):
        raise LossProfileError(f"{label} path shapes differ")
    tolerance = 2.0e-5 + 2.0e-6 * da.detach().abs().amax()
    if bool(((da - rt - dart).abs().amax() > tolerance).detach()):
        raise LossProfileError(f"{label} violates DART=DA-RT")


def _auxiliary(output: Any, reference: Tensor) -> Tensor:
    raw = _lookup(
        output,
        "router_aux_loss",
        "auxiliary_losses",
        "moe_auxiliary",
        label="output",
        required=False,
    )
    if raw is None:
        return reference.sum() * 0.0
    if isinstance(raw, Tensor):
        return _tensor(raw, label="router_aux_loss").mean()
    packet = _mapping(raw, label="auxiliary_losses")
    values = [_tensor(item, label=f"auxiliary_losses.{name}").mean() for name, item in packet.items()]
    return torch.stack(values).sum() if values else reference.sum() * 0.0


def _assert_target_metadata(batch: Any) -> None:
    metadata = _mapping(
        _lookup(batch, "target_metadata", label="batch"), label="target_metadata"
    )
    if metadata.get("fit_split") != "train_2022_2023_only":
        raise LossProfileError("matched profile scales must be fitted on 2022-2023 train")
    if metadata.get("opened_final_test_2025") is not False:
        raise LossProfileError("matched profile must not open the sealed 2025 test set")


def _family_huber_terms(
    predictions: Any,
    targets: Any,
    masks: Any,
    scales: Any,
    *,
    label: str,
    predictions_are_quantiles: bool,
) -> dict[str, Tensor]:
    predicted = _families(
        predictions, label=f"{label}.predictions", quantile=predictions_are_quantiles
    )
    truth = _families(targets, label=f"{label}.targets")
    mask_packet = _families(masks, label=f"{label}.masks")
    scale_packet = _mapping(scales, label=f"{label}.scales")
    result: dict[str, Tensor] = {}
    for family in PUBLIC_GROUP_SLICES:
        prediction = predicted[family]
        if predictions_are_quantiles:
            if prediction.shape[-1] != len(QUANTILE_LEVELS):
                raise LossProfileError(f"{label}.{family} must expose nine quantiles")
            prediction = prediction[..., MEDIAN_QUANTILE_INDEX]
        if family == "price":
            _assert_price_identity(truth[family], label=f"{label}.price_target")
        if family not in mask_packet or family not in scale_packet:
            raise LossProfileError(f"{label} lacks {family} mask or scale")
        result[family] = _masked_group_huber(
            prediction,
            truth[family],
            mask=_tensor(mask_packet[family], label=f"{label}.masks.{family}"),
            scale=_tensor(scale_packet[family], label=f"{label}.scales.{family}"),
            family=family,
            label=f"{label}.{family}",
        )
    return result


def _state_huber(
    predictions: Any,
    targets: Any,
    masks: Any,
    scales: Any,
    *,
    label: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    predicted = _mapping(predictions, label=f"{label}.predictions")
    truth = _mapping(targets, label=f"{label}.targets")
    mask_packet = _mapping(masks, label=f"{label}.masks")
    scale_packet = _mapping(scales, label=f"{label}.scales")
    terms: dict[str, Tensor] = {}
    for state_name in ("soc", "queue"):
        if any(
            state_name not in packet
            for packet in (predicted, truth, mask_packet, scale_packet)
        ):
            raise LossProfileError(f"{label} lacks {state_name} prediction/target/mask/scale")
        prediction = _tensor(predicted[state_name], label=f"{label}.{state_name}.prediction")
        target = _tensor(truth[state_name], label=f"{label}.{state_name}.target")
        if prediction.shape != target.shape:
            raise LossProfileError(f"{label}.{state_name} shapes differ")
        family = "queue" if state_name == "queue" else "soc"
        terms[state_name] = _masked_group_huber(
            prediction,
            target,
            mask=_tensor(mask_packet[state_name], label=f"{label}.{state_name}.mask"),
            scale=_tensor(scale_packet[state_name], label=f"{label}.{state_name}.scale"),
            family=family,
            label=f"{label}.{state_name}",
        )
    return 0.5 * (terms["soc"] + terms["queue"]), terms


def matched_w0_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    **_: Any,
) -> PhaseLoss:
    _assert_target_metadata(batch)
    _assert_joint_paths(
        _lookup(output, "joint_price_paths", label="output"), label="joint_price_paths"
    )
    terms = _family_huber_terms(
        _lookup(output, "quantile_predictions", label="output"),
        _lookup(batch, "quantile_targets", label="batch"),
        _lookup(batch, "quantile_masks", label="batch"),
        _lookup(batch, "quantile_scales", "target_scales", label="batch"),
        label="w0",
        predictions_are_quantiles=True,
    )
    continuous = sum((0.25 * value for value in terms.values()), terms["price"].new_zeros(()))
    state, state_terms = _state_huber(
        _lookup(output, "state_predictions", "state_reconstructions", "reconstructions", label="output"),
        _lookup(batch, "state_targets", "state_reconstruction_targets", "reconstruction_targets", label="batch"),
        _lookup(batch, "state_masks", "state_reconstruction_masks", "reconstruction_masks", label="batch"),
        _lookup(batch, "state_scales", "state_target_scales", label="batch"),
        label="w0_state",
    )
    count = continuous.new_zeros(())
    count_active = False
    count_target = _lookup(
        batch, "workload_count_target", "arrival_count_target", label="batch", required=False
    )
    count_mask = _lookup(
        batch, "workload_count_mask", "arrival_count_mask", label="batch", required=False
    )
    if count_target is not None:
        if count_mask is None:
            raise LossProfileError("true count targets require an explicit valid mask")
        mask = _tensor(count_mask, label="workload_count_mask")
        if bool((mask > 0).any()):
            distribution = _mapping(
                _lookup(output, "count_distribution", label="output"),
                label="count_distribution",
            )
            mean = _tensor(
                _lookup(distribution, "mean", "mu", label="count_distribution"),
                label="count_distribution.mean",
            )
            target = _tensor(count_target, label="workload_count_target")
            selected = target[mask > 0]
            if bool((selected < 0).any()) or not bool(
                torch.isclose(selected, selected.round(), atol=1.0e-5, rtol=0.0).all()
            ):
                raise LossProfileError("count targets must be genuine nonnegative integers")
            count = _masked_group_huber(
                mean,
                target,
                mask=mask,
                scale=torch.ones((), device=target.device),
                family="count",
                label="w0.count",
            )
            count_active = True
    auxiliary = _auxiliary(output, continuous)
    route_supervision = _queue_tail_route_supervision_loss(
        output, batch, continuous
    )
    total = (
        weights.quantile * continuous
        + weights.state * state
        + weights.auxiliary * auxiliary
        + weights.route_supervision * route_supervision
    )
    if count_active:
        total = total + weights.count * count
    components = {
        **{f"continuous/{name}": value for name, value in terms.items()},
        **{f"state/{name}": value for name, value in state_terms.items()},
        "continuous": continuous,
        "state": state,
        "count": count,
        "count_active": continuous.new_tensor(float(count_active)),
        "auxiliary": auxiliary,
        "route_supervision": route_supervision,
        "total": _finite_scalar(total, label="matched W0 total"),
    }
    return PhaseLoss("w0", total, components)


def _gaussian_mean(value: Any, *, label: str) -> Tensor:
    packet = _mapping(value, label=label)
    return _tensor(_lookup(packet, "mean", "loc", label=label), label=f"{label}.mean")


def matched_w1_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    include_route_supervision: bool = True,
    **_: Any,
) -> PhaseLoss:
    _assert_target_metadata(batch)
    priors = _horizons(_lookup(output, "latent_prior", label="output"), label="latent_prior")
    posteriors = _horizons(
        _lookup(output, "latent_posterior", label="output"), label="latent_posterior"
    )
    observation_predictions = _horizons(
        _lookup(output, "observation_quantile_predictions", label="output"),
        label="observation_quantile_predictions",
    )
    observation_targets = _horizons(
        _lookup(batch, "observation_targets", label="batch"), label="observation_targets"
    )
    observation_masks = _horizons(
        _lookup(batch, "observation_masks", "observation_target_masks", label="batch"),
        label="observation_masks",
    )
    joint_paths = _horizons(
        _lookup(output, "observation_joint_price_paths", label="output"),
        label="observation_joint_price_paths",
    )
    state_predictions = _horizons(
        _lookup(output, "state_rollout_predictions", label="output"),
        label="state_rollout_predictions",
    )
    state_targets = _horizons(
        _lookup(batch, "state_rollout_targets", label="batch"), label="state_rollout_targets"
    )
    state_masks = _horizons(
        _lookup(batch, "state_rollout_masks", label="batch"), label="state_rollout_masks"
    )
    scales = _lookup(batch, "quantile_scales", "target_scales", label="batch")
    state_scales = _lookup(batch, "state_scales", "state_target_scales", label="batch")
    latent_terms: list[Tensor] = []
    rollout_terms: list[Tensor] = []
    state_terms: list[Tensor] = []
    components: dict[str, Tensor] = {}
    for horizon in DYNAMICS_HORIZONS:
        prior = _gaussian_mean(priors[horizon], label=f"latent_prior.h{horizon}")
        posterior = _gaussian_mean(
            posteriors[horizon], label=f"latent_posterior.h{horizon}"
        )
        if prior.shape != posterior.shape or prior.numel() == 0:
            raise LossProfileError(f"W1 h{horizon} prior/posterior positions are empty or differ")
        latent = F.smooth_l1_loss(prior, posterior.detach())
        latent_terms.append(latent)
        components[f"latent/h{horizon}"] = latent
        _assert_joint_paths(joint_paths[horizon], label=f"joint_paths.h{horizon}")
        family_terms = _family_huber_terms(
            observation_predictions[horizon],
            observation_targets[horizon],
            observation_masks[horizon],
            scales,
            label=f"w1.h{horizon}",
            predictions_are_quantiles=True,
        )
        rollout = sum(
            (0.25 * value for value in family_terms.values()),
            latent.new_zeros(()),
        )
        rollout_terms.append(rollout)
        components.update(
            {f"rollout/h{horizon}/{name}": value for name, value in family_terms.items()}
        )
        state, per_state = _state_huber(
            state_predictions[horizon],
            state_targets[horizon],
            state_masks[horizon],
            state_scales,
            label=f"w1_state.h{horizon}",
        )
        state_terms.append(state)
        components.update(
            {f"state/h{horizon}/{name}": value for name, value in per_state.items()}
        )
    latent = torch.stack(latent_terms).mean()
    rollout = torch.stack(rollout_terms).mean()
    state = torch.stack(state_terms).mean()
    auxiliary = _auxiliary(output, latent)
    route_supervision = (
        _queue_tail_route_supervision_loss(output, batch, latent)
        if include_route_supervision
        else latent.sum() * 0.0
    )
    total = (
        weights.latent * latent
        + weights.rollout * rollout
        + weights.rollout_state * state
        + weights.auxiliary * auxiliary
        + weights.route_supervision * route_supervision
    )
    components.update(
        latent=latent,
        rollout=rollout,
        state=state,
        auxiliary=auxiliary,
        route_supervision=route_supervision,
        total=_finite_scalar(total, label="matched W1 total"),
    )
    return PhaseLoss("w1", total, components)


def matched_u0_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    include_route_supervision: bool = True,
    **_: Any,
) -> PhaseLoss:
    _assert_target_metadata(batch)
    scenarios = _tensor(
        _lookup(output, "scenarios", "stochastic_rollouts", "samples", label="output"),
        label="scenarios",
    )
    target = _tensor(
        _lookup(batch, "stochastic_target", "scenario_target", label="batch"),
        label="stochastic_target",
    )
    if scenarios.ndim != target.ndim + 1 or scenarios.shape[1] != 20:
        raise LossProfileError("matched U0 requires exactly 20 paths [B,20,...,21]")
    if scenarios.shape[0] != target.shape[0] or scenarios.shape[2:] != target.shape[1:]:
        raise LossProfileError("U0 scenario/target shapes differ")
    _assert_price_identity(scenarios[..., :12], label="U0 scenarios")
    _assert_price_identity(target[..., :12], label="U0 target")
    mean = scenarios.mean(dim=1)
    masks = _lookup(batch, "quantile_masks", "stochastic_masks", label="batch")
    scales = _lookup(batch, "quantile_scales", "target_scales", label="batch")
    terms = _family_huber_terms(
        mean,
        target,
        masks,
        scales,
        label="u0",
        predictions_are_quantiles=False,
    )
    ensemble_mean = sum(
        (0.25 * value for value in terms.values()), terms["price"].new_zeros(())
    )
    auxiliary = _auxiliary(output, ensemble_mean)
    route_supervision = (
        _queue_tail_route_supervision_loss(output, batch, ensemble_mean)
        if include_route_supervision
        else ensemble_mean.sum() * 0.0
    )
    total = (
        ensemble_mean
        + weights.auxiliary * auxiliary
        + weights.route_supervision * route_supervision
    )
    components = {
        **{f"ensemble_mean/{name}": value for name, value in terms.items()},
        "ensemble_mean": ensemble_mean,
        "auxiliary": auxiliary,
        "route_supervision": route_supervision,
        "total": _finite_scalar(total, label="matched U0 total"),
    }
    return PhaseLoss("u0", total, components)


def matched_huber_mean_phase_loss(
    phase: str,
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    cvar_alpha: float = 0.95,
    include_route_supervision: bool = True,
) -> PhaseLoss:
    normalized = str(phase).lower()
    if normalized == "w0":
        return matched_w0_loss(output, batch, weights=weights)
    if normalized == "w1":
        return matched_w1_loss(
            output,
            batch,
            weights=weights,
            include_route_supervision=include_route_supervision,
        )
    if normalized == "u0":
        return matched_u0_loss(
            output,
            batch,
            weights=weights,
            include_route_supervision=include_route_supervision,
        )
    if normalized == "p1":
        return p1_latent_planner_loss(
            output,
            batch,
            weights=weights,
            cvar_alpha=cvar_alpha,
            include_route_supervision=include_route_supervision,
        )
    if normalized == "j1":
        world = matched_w1_loss(
            output, batch, weights=weights, include_route_supervision=False
        )
        uncertainty = matched_u0_loss(
            output, batch, weights=weights, include_route_supervision=False
        )
        policy = p1_latent_planner_loss(
            output,
            batch,
            weights=weights,
            cvar_alpha=cvar_alpha,
            include_route_supervision=False,
        )
        route_supervision = _queue_tail_route_supervision_loss(
            output, batch, world.total
        )
        world_objective = (
            world.total
            if _lookup(
                batch,
                "queue_risk_event_target",
                label="batch",
                required=False,
            )
            is None
            else world.total + weights.route_supervision * route_supervision
        )
        components = {
            **{f"world/{name}": value for name, value in world.components.items()},
            **{
                f"uncertainty/{name}": value
                for name, value in uncertainty.components.items()
            },
            **{f"policy/{name}": value for name, value in policy.components.items()},
            "world/route_supervision": route_supervision,
            "world/total": world_objective,
            "total": policy.total,
        }
        return PhaseLoss(
            "j1",
            policy.total,
            components,
            optimization_objectives={
                "world": world_objective,
                "uncertainty": uncertainty.total,
                "policy": policy.total,
            },
        )
    if normalized == "c1":
        raise LossProfileError("C1 is evaluation-only and has no training objective")
    raise LossProfileError(f"unknown WorldMoE phase: {phase}")


def resolve_loss_profile(name: str) -> LossProfileBuilders:
    normalized = str(name).strip().lower()
    if normalized == PROPER_PROFILE:
        return LossProfileBuilders(normalized, proper_phase_loss, proper_phase_loss)
    if normalized == MATCHED_HUBER_PROFILE:
        return LossProfileBuilders(
            normalized, matched_huber_mean_phase_loss, proper_phase_loss
        )
    raise LossProfileError(f"unknown R5-V2 loss profile: {name!r}")


__all__ = [
    "LossProfileBuilders",
    "LossProfileError",
    "MATCHED_HUBER_PROFILE",
    "PROPER_PROFILE",
    "matched_huber_mean_phase_loss",
    "matched_u0_loss",
    "matched_w0_loss",
    "matched_w1_loss",
    "resolve_loss_profile",
]
