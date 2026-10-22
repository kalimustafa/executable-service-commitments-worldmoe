"""Independent FP32 scientific objectives for V8 TriFusion training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from dcvpp.e2e_moe_r4 import losses as r4_losses
from dcvpp.e2e_moe_r4_v7 import losses as v7_losses


class V8LossError(ValueError):
    """A task packet cannot support the registered scientific objective."""


@dataclass(frozen=True)
class TaskLoss:
    total: Tensor
    terms: Mapping[str, Tensor]
    guard_violation: float


def packet_mapping(packet: Any, label: str = "packet") -> Mapping[str, Any]:
    if isinstance(packet, Mapping):
        return packet
    if is_dataclass(packet) and not isinstance(packet, type):
        return {field.name: getattr(packet, field.name) for field in fields(packet)}
    raise V8LossError(f"{label} must be a mapping or dataclass")


def _tensor(mapping: Mapping[str, Any], aliases: tuple[str, ...], label: str) -> Tensor:
    for alias in aliases:
        value = mapping.get(alias)
        if isinstance(value, Tensor):
            return value.float()
    raise V8LossError(f"{label} is required; accepted keys={aliases}")


def _weights(samples: Tensor, value: Any) -> Tensor:
    if value is None:
        return samples.new_full(samples.shape[:2], 1.0 / float(samples.shape[1]))
    weights = torch.as_tensor(value, device=samples.device, dtype=torch.float32)
    return r4_losses.normalize_weights(weights).to(samples)


PUBLIC_SLICES = {
    "da": slice(0, 4),
    "rt": slice(4, 8),
    "dart": slice(8, 12),
    "workload": slice(12, 13),
    "pv": slice(13, 17),
    "pue": slice(17, 21),
}

REPAIR_HEADS = ("da", "rt", "dart", "workload")
REPAIR_OBJECTIVE = "paired_daily_nmae_repair"


def _split_public(value: Tensor) -> dict[str, Tensor]:
    if value.ndim < 1 or value.shape[-1] != 21:
        raise V8LossError("public target/forecast tensor must end in 21 channels")
    return {name: value[..., channels] for name, channels in PUBLIC_SLICES.items()}


def _broadcast_scale(value: Any, reference: Tensor, label: str) -> Tensor:
    scale = torch.as_tensor(
        value, device=reference.device, dtype=torch.float32
    ).clamp_min(1.0e-6)
    if scale.numel() == 1:
        return scale
    if scale.ndim == 1 and scale.shape[0] == reference.shape[-1]:
        return scale.reshape((1,) * (reference.ndim - 1) + (scale.shape[0],))
    if (
        scale.ndim == 2
        and reference.ndim >= 2
        and scale.shape == (reference.shape[0], reference.shape[-1])
    ):
        return scale.reshape(
            (scale.shape[0],) + (1,) * (reference.ndim - 2) + (scale.shape[1],)
        )
    try:
        torch.broadcast_shapes(scale.shape, reference.shape)
    except RuntimeError as exc:
        raise V8LossError(f"{label} cannot broadcast to its supervised output") from exc
    return scale


def _router_health(packet: Mapping[str, Any], like: Tensor) -> Tensor:
    router = packet.get("router")
    if isinstance(router, Mapping):
        probabilities = router.get("probabilities")
        top_indices = router.get("top_indices", router.get("topk_indices"))
    else:
        probabilities = packet.get("router_probabilities")
        top_indices = packet.get("topk_indices")
    if not isinstance(probabilities, Tensor) or not isinstance(top_indices, Tensor):
        return like.sum() * 0.0
    return r4_losses.router_balance_loss(
        probabilities.float(), top_indices.long()
    )


def _daily_mae(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise V8LossError("daily MAE inputs must share a [batch,...] shape")
    return (prediction.float() - target.float()).abs().flatten(1).mean(1)


def _repair_persistence(
    batch: Mapping[str, Any], reference: Tensor
) -> tuple[dict[str, Tensor], Tensor]:
    anchors = batch.get("representation_anchors")
    if not isinstance(anchors, Mapping):
        raise V8LossError(
            "repair Representation loss requires batch.representation_anchors"
        )
    persistence = anchors.get("persistence_public")
    available = anchors.get("availability")
    if not isinstance(persistence, Tensor) or not isinstance(available, Tensor):
        raise V8LossError(
            "repair Representation loss requires persistence_public and "
            "availability"
        )
    persistence32 = persistence.to(device=reference.device, dtype=torch.float32)
    if persistence32.shape != reference.shape:
        raise V8LossError("persistence_public must match point_forecast [B,96,21]")
    availability = available.to(device=reference.device).bool()
    if availability.shape != (reference.shape[0], 2):
        raise V8LossError("Representation anchor availability must be [batch,2]")
    persistence_available = availability[:, 0]
    if not bool(persistence_available.any()):
        raise V8LossError("repair batch contains no available causal persistence day")
    return _split_public(persistence32), persistence_available


def _masked_mean(value: Tensor, mask: Tensor, *, like: Tensor) -> Tensor:
    selected = value[mask]
    return selected.mean() if selected.numel() else like.sum() * 0.0


def _upper_tail_mean(value: Tensor, fraction: float, *, like: Tensor) -> Tensor:
    if value.numel() == 0:
        return like.sum() * 0.0
    count = max(1, math.ceil(value.numel() * fraction))
    return value.topk(min(count, value.numel()), largest=True).values.mean()


def _repair_representation_loss(
    root: Mapping[str, Any],
    predictions: Mapping[str, Any],
    targets: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
    guard: float,
) -> TaskLoss:
    point = root.get("point_forecast")
    if not isinstance(point, Tensor):
        raise V8LossError("repair Representation packet requires point_forecast")
    point32 = point.float()
    baselines, persistence_available = _repair_persistence(batch, point32)

    statistics = batch.get("representation_repair_statistics", {})
    if not isinstance(statistics, Mapping):
        raise V8LossError("representation_repair_statistics must be a mapping")

    def setting(name: str, statistic_name: str, default: float) -> Tensor:
        return torch.as_tensor(
            statistics.get(statistic_name, selected.get(name, default)),
            device=point32.device,
            dtype=torch.float32,
        )

    zero_epsilon = setting(
        "workload_zero_epsilon", "zero_target_epsilon", 1.0e-6
    )
    near_threshold = setting(
        "workload_near_threshold", "near_zero_threshold", 0.0266533205
    )
    active_median = setting(
        "workload_active_median", "active_workload_median", 0.1877409667
    )
    positive_scale_floor = setting(
        "workload_positive_scale_floor", "positive_scale_floor", active_median
    )
    positive_weight = setting(
        "workload_zero_positive_weight", "workload_zero_pos_weight", 8.040
    )
    focal_gamma = float(selected.get("workload_zero_focal_gamma", 2.0))
    tail_fraction = float(selected.get("tail_fraction", 0.10))
    tail_weight = float(selected.get("tail_weight", 0.25))
    margin = float(selected.get("noninferiority_margin", 0.01))
    temperature = float(selected.get("smooth_temperature", 0.05))
    worst_temperature = float(selected.get("worst_head_temperature", 0.05))
    raw_scalar_settings = (
        zero_epsilon,
        near_threshold,
        active_median,
        positive_scale_floor,
        positive_weight,
    )
    if any(value.numel() != 1 for value in raw_scalar_settings):
        raise V8LossError("workload repair statistics must be scalar")
    scalar_settings = torch.stack(
        tuple(value.reshape(()) for value in raw_scalar_settings)
    )
    invalid_statistics = (
        (~torch.isfinite(scalar_settings)).any()
        | (scalar_settings <= 0.0).any()
        | (near_threshold <= zero_epsilon)
        | (active_median < near_threshold)
    )
    if (
        bool(invalid_statistics)
        or focal_gamma < 0.0
        or not 0.0 < tail_fraction <= 1.0
        or tail_weight < 0.0
        or temperature <= 0.0
        or worst_temperature <= 0.0
    ):
        raise V8LossError("invalid paired daily NMAE repair hyperparameters")

    logits = root.get("workload_zero_logits")
    positive = root.get("workload_positive_forecast")
    if not isinstance(logits, Tensor) or not isinstance(positive, Tensor):
        raise V8LossError(
            "repair Representation packet requires workload_zero_logits and "
            "workload_positive_forecast"
        )
    logits32 = logits.float()
    if logits32.shape != (point32.shape[0],):
        raise V8LossError("workload_zero_logits must be [batch]")
    positive32 = positive.float()
    if positive32.shape != (point32.shape[0], point32.shape[1], 1):
        raise V8LossError("workload_positive_forecast must be [batch,96,1]")

    terms: dict[str, Tensor] = {}
    head_objectives: list[Tensor] = []
    workload_target_scale: Tensor | None = None
    workload_prediction: Tensor | None = None
    workload_target: Tensor | None = None
    for name in REPAIR_HEADS:
        prediction = predictions.get(name)
        target_value = targets.get(name)
        baseline = baselines.get(name)
        if not all(isinstance(value, Tensor) for value in (prediction, target_value, baseline)):
            raise V8LossError(f"repair Representation output {name!r} is missing")
        prediction32 = prediction.float()  # type: ignore[union-attr]
        target32 = target_value.to(prediction32).float()  # type: ignore[union-attr]
        baseline32 = baseline.to(prediction32).float().detach()  # type: ignore[union-attr]
        if prediction32.shape != target32.shape or baseline32.shape != target32.shape:
            raise V8LossError(f"repair Representation output {name!r} shape mismatch")
        raw_scale = target32.abs().flatten(1).mean(1)
        daily_scale = raw_scale.clamp_min(zero_epsilon).detach()
        model_nmae = _daily_mae(prediction32, target32) / daily_scale
        baseline_nmae = _daily_mae(baseline32, target32) / daily_scale
        mask = persistence_available
        if name == "workload":
            mask = mask & raw_scale.gt(zero_epsilon)
            workload_target_scale = raw_scale
            workload_prediction = prediction32
            workload_target = target32
        normalizer = _masked_mean(baseline_nmae.detach(), mask, like=prediction32)
        normalizer = normalizer.detach().clamp_min(1.0e-3)
        delta = (model_nmae - baseline_nmae.detach()) / normalizer
        selected_delta = delta[mask]
        mean_delta = _masked_mean(delta, mask, like=prediction32)
        tail_delta = _upper_tail_mean(
            selected_delta, tail_fraction, like=prediction32
        )
        risk = mean_delta + tail_weight * tail_delta
        objective = temperature * F.softplus((risk + margin) / temperature)
        head_objectives.append(objective)
        terms[f"repair_model_nmae_{name}"] = model_nmae
        terms[f"repair_baseline_nmae_{name}"] = baseline_nmae.detach()
        terms[f"repair_delta_{name}"] = delta
        terms[f"repair_risk_{name}"] = risk.reshape(1)

    if workload_target_scale is None or workload_prediction is None or workload_target is None:
        raise V8LossError("repair Representation workload target is missing")
    zero_label = workload_target_scale.le(zero_epsilon).float()
    probabilities = torch.sigmoid(logits32)
    binary = F.binary_cross_entropy_with_logits(
        logits32, zero_label, reduction="none"
    )
    true_probability = torch.where(
        zero_label.bool(), probabilities, 1.0 - probabilities
    )
    class_weight = torch.where(
        zero_label.bool(),
        positive_weight.to(logits32).expand_as(logits32),
        logits32.new_ones(logits32.shape),
    )
    focal = (
        class_weight * (1.0 - true_probability).pow(focal_gamma) * binary
    ).mean()
    weighted_bce = (class_weight * binary).mean()
    zero_mask = zero_label.bool()
    zero_raw = _masked_mean(
        positive32.abs().flatten(1).mean(1)
        / torch.maximum(active_median, positive_scale_floor),
        zero_mask,
        like=positive32,
    )
    near_mask = workload_target_scale.gt(zero_epsilon) & workload_target_scale.le(
        near_threshold
    )
    near_raw = _masked_mean(
        _daily_mae(workload_prediction, workload_target) / near_threshold,
        near_mask,
        like=workload_prediction,
    )
    stacked = torch.stack(head_objectives)
    smooth_worst = worst_temperature * torch.logsumexp(
        stacked / worst_temperature, dim=0
    )
    total = (
        smooth_worst
        + float(selected.get("workload_zero_classifier", 1.0)) * focal
        + float(selected.get("workload_zero_raw", 1.0)) * zero_raw
        + float(selected.get("workload_near_raw", 0.25)) * near_raw
    )
    terms["repair_smooth_worst"] = smooth_worst.reshape(1)
    terms["repair_workload_zero_focal"] = focal.reshape(1)
    terms["repair_workload_zero_bce"] = weighted_bce.reshape(1)
    terms["repair_workload_zero_raw"] = zero_raw.reshape(1)
    terms["repair_workload_near_raw"] = near_raw.reshape(1)
    terms["repair_workload_zero_rate"] = zero_label.mean().reshape(1)
    terms["repair_workload_zero_probability_mean"] = probabilities.mean().reshape(1)
    return TaskLoss(
        total.float(),
        terms,
        max(0.0, float(total.detach().cpu()) - float(guard)),
    )


def representation_loss(
    packet: Any,
    batch: Mapping[str, Any],
    *,
    weights: Mapping[str, Any] | None = None,
    guard: float = 1.0,
) -> TaskLoss:
    """Legacy point loss or the explicitly selected train-only repair loss."""

    root = packet_mapping(packet, "Representation packet")
    nested = root.get("representation", root.get("representation_packet"))
    if nested is not None:
        root = packet_mapping(nested, "Representation packet")
    raw_predictions = root.get("predictions", root.get("forecasts"))
    if raw_predictions is None and isinstance(root.get("point_forecast"), Tensor):
        raw_predictions = _split_public(root["point_forecast"])
    predictions = packet_mapping(raw_predictions, "Representation predictions")
    targets = batch.get(
        "representation_targets", batch.get("forecast_targets", batch.get("targets"))
    )
    if targets is None:
        public_target = batch.get(
            "distribution_target", batch.get("scenario_target_zonal")
        )
        if isinstance(public_target, Tensor):
            targets = _split_public(public_target)
    targets = packet_mapping(targets, "batch Representation targets")
    scales = batch.get("representation_scales", {})
    if not isinstance(scales, Mapping):
        raise V8LossError("representation_scales must be a mapping")
    if not scales and isinstance(batch.get("channel_scale"), Tensor):
        scales = _split_public(batch["channel_scale"])
    selected = dict(weights or {})
    objective = str(selected.get("objective", "legacy_smooth_l1")).lower()
    if objective == REPAIR_OBJECTIVE:
        return _repair_representation_loss(
            root,
            predictions,
            targets,
            batch,
            selected=selected,
            guard=guard,
        )
    if objective not in {"legacy", "legacy_smooth_l1", "smooth_l1"}:
        raise V8LossError(f"unknown Representation objective: {objective}")
    terms: dict[str, Tensor] = {}
    totals: list[Tensor] = []
    for name, target_value in targets.items():
        prediction = predictions.get(name)
        if not isinstance(prediction, Tensor) or not isinstance(target_value, Tensor):
            raise V8LossError(f"Representation output {name!r} is missing")
        prediction32 = prediction.float()
        target32 = target_value.to(prediction32).float()
        if prediction32.shape != target32.shape:
            raise V8LossError(f"Representation output {name!r} shape mismatch")
        scale = _broadcast_scale(
            scales.get(name, 1.0), prediction32, f"Representation scale {name!r}"
        )
        forecast = F.smooth_l1_loss(
            prediction32 / scale, target32 / scale, reduction="none"
        ).flatten(1).mean(1)
        temporal = (
            F.smooth_l1_loss(
                (prediction32[:, 1:] - prediction32[:, :-1]) / scale,
                (target32[:, 1:] - target32[:, :-1]) / scale,
                reduction="none",
            ).flatten(1).mean(1)
            if prediction32.ndim >= 2 and prediction32.shape[1] > 1
            else forecast * 0.0
        )
        terms[f"forecast_{name}"] = forecast
        terms[f"temporal_{name}"] = temporal
        totals.append(
            float(selected.get("forecast", 1.0)) * forecast.mean()
            + float(selected.get("temporal", 0.25)) * temporal.mean()
        )
    if not totals:
        raise V8LossError("Representation packet has no supervised outputs")
    total = torch.stack(totals).mean()
    router = _router_health(root, total)
    terms["router_health"] = router.reshape(1)
    total = total + float(selected.get("router_health", 0.01)) * router
    return TaskLoss(
        total.float(),
        terms,
        max(0.0, float(total.detach().cpu()) - float(guard)),
    )


def distribution_loss(
    packet: Any,
    batch: Mapping[str, Any],
    *,
    secondary_guard: float = 0.005,
    constraint_weight: float = 1.0,
) -> TaskLoss:
    """Energy-first paired proper score with secondary non-inferiority guards."""

    root = packet_mapping(packet, "Distribution packet")
    nested = root.get("distribution_packet")
    if nested is None and "samples" not in root:
        nested = root.get("distribution")
    container = packet_mapping(nested, "Distribution packet") if nested is not None else root
    raw_sample_packet = root.get("samples", container.get("samples"))
    if isinstance(raw_sample_packet, Tensor):
        samples_raw = raw_sample_packet.float()
        sample_view: Mapping[str, Any] = container
    elif raw_sample_packet is not None:
        sample_view = packet_mapping(raw_sample_packet, "Distribution samples")
        samples_raw = _tensor(
            sample_view,
            ("public_samples", "samples", "samples_zonal"),
            "samples",
        )
    else:
        sample_view = container
        samples_raw = _tensor(container, ("samples_zonal",), "samples")
    baseline_value = root.get(
        "baseline_samples",
        container.get(
            "baseline_samples",
            container.get(
                "anchor_samples", batch.get("baseline_samples")
            ),
        ),
    )
    if not isinstance(baseline_value, Tensor):
        raise V8LossError("baseline samples are required")
    baseline_raw = baseline_value.float()
    target_raw = root.get("target")
    if not isinstance(target_raw, Tensor):
        target_raw = container.get("target")
    if not isinstance(target_raw, Tensor):
        target_raw = batch.get("distribution_target", batch.get("scenario_target_zonal"))
    if not isinstance(target_raw, Tensor):
        raise V8LossError("Distribution target is required")
    target_raw = target_raw.float().to(samples_raw)
    if samples_raw.shape != baseline_raw.shape:
        raise V8LossError("model and baseline samples must share common paths")
    if target_raw.shape != samples_raw.shape[:1] + samples_raw.shape[2:]:
        raise V8LossError("Distribution target shape is invalid")
    channel_scale = root.get(
        "channel_scale", container.get("channel_scale", batch.get("channel_scale"))
    )
    if not isinstance(channel_scale, Tensor):
        raise V8LossError("frozen channel_scale is required")
    scale = channel_scale.float().to(samples_raw)
    if bool((scale <= 0).any()):
        raise V8LossError("channel_scale must be positive")
    if scale.shape == (samples_raw.shape[-1],):
        sample_scale = scale.view(
            (1,) * (samples_raw.ndim - 1) + (scale.shape[-1],)
        )
        target_scale = scale.view(
            (1,) * (target_raw.ndim - 1) + (scale.shape[-1],)
        )
    elif scale.shape == (samples_raw.shape[0], samples_raw.shape[-1]):
        sample_scale = scale.view(
            scale.shape[0], *([1] * (samples_raw.ndim - 2)), scale.shape[-1]
        )
        target_scale = scale.view(
            scale.shape[0], *([1] * (target_raw.ndim - 2)), scale.shape[-1]
        )
    else:
        raise V8LossError("channel_scale must be [feature] or [day,feature]")
    samples = samples_raw.float() / sample_scale
    baseline = baseline_raw.float().to(samples) / sample_scale
    target = target_raw / target_scale
    weights = _weights(samples, sample_view.get("weights", root.get("weights")))
    baseline_weights = _weights(
        baseline, root.get("baseline_weights", container.get("baseline_weights"))
    )

    model_energy = r4_losses.weighted_energy_score(samples, weights, target)
    base_energy = r4_losses.weighted_energy_score(
        baseline, baseline_weights, target
    ).detach()
    energy_delta = model_energy - base_energy
    # Normalize per day so an 181-day logical update is invariant to its GPU
    # microbatch partition; a microbatch-local scalar denominator would change
    # the scientific objective when memory settings change.
    energy = energy_delta / base_energy.abs().clamp_min(1.0e-6)

    model_scores = {
        "marginal_crps": r4_losses.weighted_marginal_crps(samples, weights, target),
        "interval90": r4_losses.weighted_interval_score(samples, weights, target, 0.90),
        "interval95": r4_losses.weighted_interval_score(samples, weights, target, 0.95),
        "tail": r4_losses.weighted_tail_pinball_score(samples, weights, target),
        "dependence": r4_losses.weighted_variogram_score_stable(samples, weights, target),
    }
    baseline_scores = {
        "marginal_crps": r4_losses.weighted_marginal_crps(
            baseline, baseline_weights, target
        ).detach(),
        "interval90": r4_losses.weighted_interval_score(
            baseline, baseline_weights, target, 0.90
        ).detach(),
        "interval95": r4_losses.weighted_interval_score(
            baseline, baseline_weights, target, 0.95
        ).detach(),
        "tail": r4_losses.weighted_tail_pinball_score(
            baseline, baseline_weights, target
        ).detach(),
        "dependence": r4_losses.weighted_variogram_score_stable(
            baseline, baseline_weights, target
        ).detach(),
    }
    terms: dict[str, Tensor] = {
        "energy_delta": energy_delta,
        "normalized_energy_delta": energy,
    }
    penalty = energy.sum() * 0.0
    worst = 0.0
    for name, score in model_scores.items():
        ratio = score / baseline_scores[name].abs().clamp_min(1.0e-6)
        violation = F.relu(ratio - (1.0 + float(secondary_guard)))
        penalty = penalty + violation.square().mean()
        terms[f"{name}_ratio"] = ratio
        worst = max(worst, float(violation.detach().max().cpu()))
    coverage90, _ = r4_losses.empirical_coverage(samples, weights, target, 0.90)
    coverage95, _ = r4_losses.empirical_coverage(samples, weights, target, 0.95)
    terms["signed_coverage90_error"] = coverage90 - 0.90
    terms["signed_coverage95_error"] = coverage95 - 0.95
    total = energy.mean() + float(constraint_weight) * penalty
    terms["secondary_guard_penalty"] = penalty.reshape(1)
    return TaskLoss(total.float(), terms, worst)


def _policy_actions_from_typed(root: Mapping[str, Any]) -> Mapping[str, Tensor]:
    return {
        "day_ahead": _tensor(root, ("day_ahead_plan", "day_ahead"), "day-ahead plan"),
        "battery": _tensor(root, ("battery_plan", "battery"), "battery plan"),
        "compute": _tensor(root, ("compute_plan", "compute"), "compute plan"),
    }


def _normalized_actions(
    actions: Mapping[str, Tensor], batch: Mapping[str, Any]
) -> Mapping[str, Tensor]:
    targets = batch.get("rollout_targets")
    if not isinstance(targets, Mapping):
        return actions
    aliases = {
        "day_ahead": "grid_limit_mw",
        "battery": "battery_power_mw",
        "compute": "capacity_work_units",
    }
    if not all(isinstance(targets.get(name), Tensor) for name in aliases.values()):
        # Lightweight injected test adapters may already supply normalized
        # actions without the physical runtime contract.  Formal V8 batches
        # always carry all three frozen limits.
        return actions
    normalized: dict[str, Tensor] = {}
    for name, limit_name in aliases.items():
        action = actions.get(name)
        limit = targets.get(limit_name)
        if not isinstance(action, Tensor) or not isinstance(limit, Tensor):
            raise V8LossError(f"normalized Policy {name} requires {limit_name}")
        value = action.float()
        if value.ndim == 3 and value.shape[1] == 4 and value.shape[2] != 4:
            value = value.transpose(1, 2)
        bound = limit.float().to(value)
        if bound.ndim == 1:
            bound = bound.unsqueeze(0).expand(value.shape[0], -1)
        if value.ndim != 3 or bound.shape != (value.shape[0], value.shape[-1]):
            raise V8LossError(f"Policy {name} action/limit shapes are invalid")
        scaled = value / bound[:, None].clamp_min(1.0e-8)
        normalized[name] = (
            scaled.clamp(-1.0, 1.0)
            if name == "battery"
            else scaled.clamp(0.0, 1.0)
        )
    return normalized


def _normalized_safe_anchor_actions(
    batch: Mapping[str, Any],
) -> Mapping[str, Tensor] | None:
    anchors = batch.get("safe_anchor_actions")
    if not isinstance(anchors, Mapping):
        return None
    return _normalized_actions(packet_mapping(anchors, "safe anchor actions"), batch)


def policy_loss(
    packet: Any,
    batch: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> TaskLoss:
    """Post-shield Policy objective; all scientific operations are FP32."""

    root = packet_mapping(packet, "Policy packet")
    raw_policy = root.get("policy", root.get("policy_packet"))
    policy = packet_mapping(raw_policy, "Policy packet") if raw_policy is not None else root
    rollout = packet_mapping(
        root.get("rollout", policy.get("rollout")), "Policy rollout"
    )
    raw_critic = root.get("critic", policy.get("critic"))
    if isinstance(raw_critic, Tensor):
        if raw_critic.ndim != 2 or raw_critic.shape[1] != 5:
            raise V8LossError("typed Policy critic must be [day,5]")
        critic: Mapping[str, Tensor] = {
            "expected": raw_critic[:, 0],
            "var": raw_critic[:, 1],
            "cvar": raw_critic[:, 2],
            "sla": raw_critic[:, 3],
            "intervention": raw_critic[:, 4],
        }
    else:
        critic = packet_mapping(raw_critic, "Policy critic")
    raw_actions = root.get("actions", policy.get("actions"))
    actions = (
        packet_mapping(raw_actions, "Policy actions")
        if raw_actions is not None
        else _policy_actions_from_typed(policy)
    )
    raw_normalized = root.get(
        "normalized_actions", policy.get("normalized_actions")
    )
    # V8 actor heads emit physical plans.  Whenever frozen physical limits are
    # present, recompute the normalized view instead of trusting an adapter
    # alias that may simply point back to the physical tensors.
    target_limits = batch.get("rollout_targets")
    has_limits = isinstance(target_limits, Mapping) and all(
        isinstance(target_limits.get(name), Tensor)
        for name in ("grid_limit_mw", "battery_power_mw", "capacity_work_units")
    )
    normalized = (
        _normalized_actions(actions, batch)
        if has_limits
        else (
            packet_mapping(raw_normalized, "normalized actions")
            if raw_normalized is not None
            else actions
        )
    )
    raw_plans = root.get(
        "action_plans",
        root.get(
            "normalized_plans",
            policy.get("action_plans", policy.get("normalized_plans")),
        ),
    )
    plans = (
        normalized
        if has_limits
        else (
            packet_mapping(raw_plans, "action plans")
            if raw_plans is not None
            else normalized
        )
    )
    terms = v7_losses.joint_policy_loss_terms(
        rollout,
        critic,
        actions,
        normalized_actions=normalized,
        action_plans=plans,
        safe_anchor_actions=_normalized_safe_anchor_actions(batch),
        cost_scale=batch.get("cost_scale", 1.0),
        soc_scale=batch.get("soc_scale", 1.0),
        backlog_scale=batch.get("backlog_scale", 1.0),
    )
    selected = dict(weights or v7_losses.DEFAULT_POLICY_WEIGHTS)
    total = v7_losses.compose_policy_loss(terms, selected).float()
    guard = max(
        float(terms["sla"].detach().mean().cpu()),
        float(terms["terminal"].detach().mean().cpu()),
    )
    return TaskLoss(total, terms, guard)


__all__ = [
    "TaskLoss",
    "V8LossError",
    "distribution_loss",
    "packet_mapping",
    "policy_loss",
    "representation_loss",
]
