"""One-forward scientific objectives for risk-conditioned ECMoE.

The objective intentionally starts at :class:`RECMoEOutput`.  It does not
re-run or reproduce any W1 representation/dynamics loss.  Realized operating
cost drives the decision, while the distributional heads are trained with
proper pinball and Brier scores.  MAE and action-MSE are deliberately absent
from both the objective and the returned accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

import torch
from torch import Tensor

from .r_ecmoe import COST_QUANTILES, RECMoEOutput


GENERAL_EXPERTS = 7
_COMPONENT_NAMES = (
    "expected_operating_cost",
    "upper_tail_cost",
    "quantile_pinball",
    "sla_deadline",
    "constraint_residual",
    "route_continuation",
    "load_balance",
    "tail_risk",
    "action_smoothness",
    "safe_anchor",
)


class RECMoELossError(ValueError):
    """Raised when an R-ECMoE scientific objective cannot be trusted."""


def _cpu_value_checks(value: Tensor) -> bool:
    """Keep strict debug checks on CPU without synchronizing CUDA training.

    CUDA non-finite gradients/losses remain fail-closed in the training loop;
    output ranges are structurally parameterized by the model.  Shape, dtype,
    provenance and Python-metadata checks remain unconditional here.
    """

    return value.device.type == "cpu"


@dataclass(frozen=True)
class RECMoELossWeights:
    """Explicit weights for the independent R-ECMoE objective."""

    expected_operating_cost: float = 1.0
    upper_tail_cost: float = 0.25
    quantile_pinball: float = 0.25
    sla_deadline: float = 1.0
    constraint_residual: float = 1.0
    route_continuation: float = 0.05
    load_balance: float = 0.01
    tail_risk: float = 0.10
    action_smoothness: float = 0.02
    safe_anchor: float = 0.25

    def __post_init__(self) -> None:
        values = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RECMoELossError(f"loss weight {field.name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise RECMoELossError(
                    f"loss weight {field.name} must be finite and nonnegative"
                )
            values.append(numeric)
        if not any(value > 0.0 for value in values):
            raise RECMoELossError("at least one R-ECMoE loss weight must be positive")

    def as_mapping(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class RECMoELossResult:
    """Differentiable total plus auditable, FP32 atomic accounting."""

    total: Tensor
    components: Mapping[str, Tensor]
    weighted_components: Mapping[str, Tensor]
    effective_weights: Mapping[str, Tensor]
    safe_anchor_multiplier: Tensor
    route_load: Tensor
    next_rolling_route_load: Tensor

    def __post_init__(self) -> None:
        expected = set(_COMPONENT_NAMES)
        if set(self.components) != expected:
            raise RECMoELossError("R-ECMoE loss components are incomplete")
        if set(self.weighted_components) != expected:
            raise RECMoELossError("R-ECMoE weighted components are incomplete")
        if set(self.effective_weights) != expected:
            raise RECMoELossError("R-ECMoE effective weights are incomplete")
        tensors = (
            self.total,
            self.safe_anchor_multiplier,
            self.route_load,
            self.next_rolling_route_load,
            *self.components.values(),
            *self.weighted_components.values(),
            *self.effective_weights.values(),
        )
        if self.total.ndim != 0:
            raise RECMoELossError("R-ECMoE total loss must be scalar")
        if any(value.dtype != torch.float32 for value in tensors):
            raise RECMoELossError("all R-ECMoE scientific loss values must be FP32")
        if any(
            _cpu_value_checks(value) and not bool(torch.isfinite(value).all())
            for value in tensors
        ):
            raise RECMoELossError("R-ECMoE loss accounting contains non-finite values")


def _finite(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise RECMoELossError(f"{name} must be a Tensor")
    result = value.float()
    if _cpu_value_checks(result) and not bool(torch.isfinite(result).all()):
        raise RECMoELossError(f"{name} contains non-finite values")
    return result


def _scalar_schedule(
    name: str,
    value: float | Tensor,
    *,
    device: torch.device,
    lower: float,
    upper: float,
) -> Tensor:
    if isinstance(value, bool):
        raise RECMoELossError(f"{name} must be a scalar")
    if isinstance(value, Tensor) and value.requires_grad:
        raise RECMoELossError(f"{name} is schedule metadata and must not require grad")
    result = torch.as_tensor(value, device=device, dtype=torch.float32)
    if result.ndim != 0:
        raise RECMoELossError(f"{name} must be one finite scalar")
    if _cpu_value_checks(result):
        if not bool(torch.isfinite(result)):
            raise RECMoELossError(f"{name} must be one finite scalar")
        if not bool((result >= lower) & (result <= upper)):
            raise RECMoELossError(f"{name} must lie in [{lower},{upper}]")
    return result.detach()


def _train_cost_scale(
    value: float | Tensor,
    *,
    source: str,
    device: torch.device,
) -> Tensor:
    if source != "train":
        raise RECMoELossError("cost_scale_source must be exactly 'train'")
    if isinstance(value, bool):
        raise RECMoELossError("cost_scale must be one positive scalar")
    if isinstance(value, Tensor) and value.requires_grad:
        raise RECMoELossError("cost_scale must be frozen train-only metadata")
    result = torch.as_tensor(value, device=device, dtype=torch.float32)
    if result.ndim != 0:
        raise RECMoELossError("cost_scale must be one finite positive train-only scalar")
    if _cpu_value_checks(result):
        if not bool(torch.isfinite(result)) or not bool(result > 0.0):
            raise RECMoELossError(
                "cost_scale must be one finite positive train-only scalar"
            )
    return result.detach()


def _batch_mean_abs(value: Tensor) -> Tensor:
    value = _finite("physical loss value", value).abs()
    if value.ndim < 1:
        raise RECMoELossError("physical loss values must retain the batch axis")
    if value.ndim == 1:
        return value
    return value.flatten(start_dim=1).mean(dim=1)


def _batch_scale(value: Tensor) -> Tensor:
    value = _finite("physical scale", value).abs()
    if value.ndim < 1:
        raise RECMoELossError("physical scales must retain the batch axis")
    if value.ndim == 1:
        return value.clamp_min(1.0e-6)
    return value.flatten(start_dim=1).mean(dim=1).clamp_min(1.0e-6)


def _normalized_physical(value: Tensor, scale: Tensor) -> Tensor:
    numerator = _batch_mean_abs(value)
    denominator = _batch_scale(scale)
    if numerator.shape != denominator.shape:
        raise RECMoELossError("physical value and scale batch dimensions disagree")
    return numerator / denominator


def _constraint_severity(output: RECMoEOutput) -> tuple[Tensor, Tensor]:
    packet = output.expert_packet
    constraints = packet.constraints
    reference = packet.reference
    residuals = output.constraint_residuals
    if not residuals:
        raise RECMoELossError("constraint residual accounting cannot be empty")

    workload = constraints.initial_queue.float() + reference.arrivals.float().sum(dim=1)
    power = (
        constraints.grid_import_limit.float() + constraints.grid_export_limit.float()
    )
    fixed_scale = (
        constraints.battery_power_limit.float() + constraints.compute_capacity.float()
    )
    scale_by_name = {
        "battery_power": constraints.battery_power_limit,
        "soc_lower": constraints.battery_energy_capacity,
        "soc_upper": constraints.battery_energy_capacity,
        "fixed_soc_lower": constraints.battery_energy_capacity,
        "fixed_soc_upper": constraints.battery_energy_capacity,
        "compute_capacity": constraints.compute_capacity,
        "work_conservation": workload,
        "migration_capacity": constraints.migration_capacity,
        "migration_conservation": constraints.migration_capacity,
        "pv_limit": constraints.pv_capacity,
        "grid_import": constraints.grid_import_limit,
        "grid_export": constraints.grid_export_limit,
        "power_balance": power,
        "fixed_past": fixed_scale,
        "critical_shortfall": workload,
        "target_soc_tracking": constraints.battery_energy_capacity,
    }
    unknown = set(residuals).difference(scale_by_name)
    if unknown:
        raise RECMoELossError(
            "constraint residuals lack registered physical scales: "
            + ", ".join(sorted(unknown))
        )
    per_family = [
        _normalized_physical(value, scale_by_name[name])
        for name, value in residuals.items()
    ]
    per_batch = torch.stack(per_family, dim=-1).mean(dim=-1)
    return per_batch.mean(), per_batch


def _sla_deadline_severity(output: RECMoEOutput) -> tuple[Tensor, Tensor, Tensor]:
    actions = output.actions
    constraints = output.expert_packet.constraints
    reference = output.expert_packet.reference
    active = output.decision_active_mask.float()
    workload = constraints.initial_queue.float() + (
        reference.arrivals.float() * active[..., None]
    ).sum(dim=1)
    expired = output.deadline_expired_work.float() * active
    if expired.shape[0] != workload.shape[0] or expired.ndim < 2:
        raise RECMoELossError("deadline_expired_work must retain batch and time")
    deadline = expired.flatten(start_dim=1).sum(dim=-1) / workload.sum(
        dim=-1
    ).clamp_min(1.0e-6)
    power_scale = (
        constraints.grid_import_limit.float() + constraints.grid_export_limit.float()
    )
    unserved = _normalized_physical(
        actions.unserved_power * active[..., None], power_scale
    )
    per_batch = 0.5 * (deadline + unserved)
    return per_batch.mean(), per_batch, unserved


def _expert_distribution(ids: Tensor, weights: Tensor, *, label: str) -> Tensor:
    if ids.shape != weights.shape or ids.ndim != 3:
        raise RECMoELossError(f"{label} ids and weights must share [B,T,K]")
    if ids.dtype not in (torch.int32, torch.int64):
        raise RECMoELossError(f"{label} expert ids must be integral")
    values = _finite(f"{label} route weights", weights)
    if _cpu_value_checks(values):
        if (
            bool((values < 0.0).any())
            or bool((ids < 0).any())
            or bool((ids >= GENERAL_EXPERTS).any())
        ):
            raise RECMoELossError(
                f"{label} routes lie outside the seven-expert simplex"
            )
        if not torch.allclose(
            values.sum(dim=-1),
            torch.ones_like(values[..., 0]),
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise RECMoELossError(f"{label} route weights must sum to one")
    full = values.new_zeros((*values.shape[:2], GENERAL_EXPERTS))
    return full.scatter_add(-1, ids.long(), values)


def _route_objectives(
    output: RECMoEOutput,
    *,
    rolling_route_load: Tensor | None,
    rolling_decay: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    current = _expert_distribution(
        output.selected_expert_ids, output.route_weights, label="continued"
    )
    prior = _expert_distribution(
        output.expert_packet.selected_expert_ids,
        output.expert_packet.selected_route_weights,
        label="W1 prior",
    ).detach()
    if current.shape[:2] != prior.shape[:2]:
        raise RECMoELossError("continued and W1 route timelines disagree")
    active = output.decision_active_mask.float()
    if active.shape != current.shape[:2]:
        raise RECMoELossError("route objective requires a nonempty active horizon")
    if _cpu_value_checks(active) and not bool((active.sum(dim=1) > 0).all()):
        raise RECMoELossError("route objective requires a nonempty active horizon")
    epsilon = 1.0e-7
    current_safe = (current + epsilon) / (1.0 + GENERAL_EXPERTS * epsilon)
    prior_safe = (prior + epsilon) / (1.0 + GENERAL_EXPERTS * epsilon)
    continuation_by_time = (
        current_safe * (torch.log(current_safe) - torch.log(prior_safe))
    ).sum(dim=-1)
    continuation = (continuation_by_time * active).sum() / active.sum()

    current_load = (current * active[..., None]).sum(dim=(0, 1)) / active.sum()
    if rolling_route_load is None:
        balanced_load = current_load
        next_load = current_load.detach()
    else:
        rolling = _finite("rolling route load", rolling_route_load).detach()
        if rolling.shape != (GENERAL_EXPERTS,):
            raise RECMoELossError("rolling route load must be nonnegative [7]")
        total = rolling.sum()
        if _cpu_value_checks(rolling):
            if bool((rolling < 0.0).any()):
                raise RECMoELossError("rolling route load must be nonnegative [7]")
            if not bool(total > 0.0):
                raise RECMoELossError("rolling route load must have positive mass")
        rolling = rolling / total
        balanced_load = rolling_decay * rolling + (1.0 - rolling_decay) * current_load
        next_load = (
            rolling_decay * rolling + (1.0 - rolling_decay) * current_load.detach()
        )
    load_balance = (GENERAL_EXPERTS * balanced_load.square().sum() - 1.0).clamp_min(0.0)
    return continuation, load_balance, current_load, next_load


def _normalized_total_variation(
    value: Tensor, scale: Tensor, active_mask: Tensor
) -> Tensor:
    value = _finite("action timeline", value)
    if value.ndim < 2:
        raise RECMoELossError("action timelines must retain batch and time axes")
    if value.shape[1] < 2:
        return value.sum() * 0.0
    pair = active_mask[:, 1:] & active_mask[:, :-1]
    difference = (value[:, 1:] - value[:, :-1]) * pair.reshape(
        *pair.shape, *([1] * (value.ndim - 2))
    ).to(value.dtype)
    return _normalized_physical(difference, scale).mean()


def _action_smoothness(output: RECMoEOutput) -> Tensor:
    actions = output.actions
    constraints = output.expert_packet.constraints
    active = output.decision_active_mask
    power = (
        constraints.grid_import_limit.float() + constraints.grid_export_limit.float()
    )
    families = (
        _normalized_total_variation(
            actions.battery, constraints.battery_power_limit, active
        ),
        _normalized_total_variation(
            actions.compute, constraints.compute_capacity, active
        ),
        _normalized_total_variation(
            actions.migration, constraints.migration_capacity, active
        ),
        _normalized_total_variation(actions.grid, power, active),
        _normalized_total_variation(
            actions.pv_curtailed, constraints.pv_capacity, active
        ),
        _normalized_total_variation(output.day_ahead_commitment, power, active),
        _normalized_total_variation(
            output.target_soc, constraints.battery_energy_capacity, active
        ),
    )
    return torch.stack(families).mean()


def _safe_anchor(output: RECMoEOutput) -> Tensor:
    constraints = output.expert_packet.constraints
    anchor_scale = (
        constraints.battery_power_limit.float()
        + constraints.compute_capacity.float()
        + constraints.pv_capacity.float()
        + constraints.grid_import_limit.float()
        + constraints.grid_export_limit.float()
    )
    # Zero structured logits define the registered tensor-only safe baseline:
    # neutral battery/migration/DA residual, bounded mid compute/SOC and PV
    # self-use.  This is a decaying residual anchor, not an action-label MSE.
    raw = torch.tanh(_finite("raw structured action", output.raw_structured_actions))
    active = output.decision_active_mask.float()
    raw_residual = (raw.abs() * active[..., None]).sum() / (
        active.sum() * raw.shape[-1]
    ).clamp_min(1.0)
    assembler_intervention = _normalized_physical(
        output.actions.shield_intervention, anchor_scale
    ).mean()
    return 0.5 * raw_residual + 0.5 * assembler_intervention


def r_ecmoe_loss(
    output: RECMoEOutput,
    *,
    cost_scale: float | Tensor,
    cost_scale_source: str,
    progress: float | Tensor,
    weights: RECMoELossWeights = RECMoELossWeights(),
    rolling_route_load: Tensor | None = None,
    rolling_decay: float = 0.90,
    safe_anchor_decay_power: float = 1.0,
) -> RECMoELossResult:
    """Build all R-ECMoE objectives from exactly one model forward.

    ``cost_scale`` has no default and is accepted only when its provenance is
    explicitly declared as ``"train"``.  Validation/test-derived scales fail
    closed.  ``rolling_route_load`` is optional detached EMA state returned by
    the preceding call; the returned ``next_rolling_route_load`` can be carried
    across short rolling batches without backpropagating through history.
    """

    if not isinstance(output, RECMoEOutput):
        raise RECMoELossError("output must be one RECMoEOutput")
    operating_cost = _finite(
        "differentiable operating cost", output.differentiable_operating_cost
    )
    if operating_cost.ndim != 1:
        raise RECMoELossError("differentiable operating cost must be [B]")
    device = operating_cost.device
    scale = _train_cost_scale(cost_scale, source=cost_scale_source, device=device)
    training_progress = _scalar_schedule(
        "progress", progress, device=device, lower=0.0, upper=1.0
    )
    if (
        isinstance(rolling_decay, bool)
        or not math.isfinite(float(rolling_decay))
        or not 0.0 <= float(rolling_decay) < 1.0
    ):
        raise RECMoELossError("rolling_decay must lie in [0,1)")
    if (
        isinstance(safe_anchor_decay_power, bool)
        or not math.isfinite(float(safe_anchor_decay_power))
        or float(safe_anchor_decay_power) <= 0.0
    ):
        raise RECMoELossError("safe_anchor_decay_power must be finite and positive")

    normalized_cost = operating_cost / scale
    expected_operating_cost = normalized_cost.mean()

    quantiles = _finite("cost quantiles", output.cost_quantiles)
    expected_shape = (operating_cost.shape[0], len(COST_QUANTILES))
    if quantiles.shape != expected_shape:
        raise RECMoELossError(
            f"cost quantiles must be {expected_shape}, got {tuple(quantiles.shape)}"
        )
    if _cpu_value_checks(quantiles) and bool(
        (quantiles[:, 1:] < quantiles[:, :-1]).any()
    ):
        raise RECMoELossError("cost quantiles must be monotonically nondecreasing")
    normalized_quantiles = quantiles / scale
    levels = quantiles.new_tensor(COST_QUANTILES, dtype=torch.float32)
    error = normalized_cost.detach()[:, None] - normalized_quantiles
    quantile_pinball = torch.maximum(levels * error, (levels - 1.0) * error).mean()

    predicted_upper_tail = _finite("upper-tail cost", output.upper_tail_cost)
    if predicted_upper_tail.shape != operating_cost.shape:
        raise RECMoELossError("upper-tail cost must be [B]")
    # Use a Rockafellar-Uryasev replay objective for the policy gradient, with
    # eta taken from the *realized replay batch*.  The former learned-head eta
    # could be badly uncalibrated early in training and inflate this term by an
    # order of magnitude.  The proper pinball term remains solely responsible
    # for calibrating the distributional value head.
    eta = torch.quantile(normalized_cost.detach(), 0.95)
    upper_tail_cost = (
        eta + torch.relu(normalized_cost - eta) / (1.0 - 0.95)
    ).mean()

    sla_deadline, sla_batch, unserved_batch = _sla_deadline_severity(output)
    constraint_residual, constraint_batch = _constraint_severity(output)
    route_continuation, load_balance, route_load, next_load = _route_objectives(
        output,
        rolling_route_load=rolling_route_load,
        rolling_decay=float(rolling_decay),
    )

    risk_probabilities = torch.stack(
        (
            _finite("SLA risk", output.sla_risk),
            _finite("constraint risk", output.constraint_risk),
            _finite("grid risk", output.grid_risk),
        ),
        dim=-1,
    )
    if risk_probabilities.shape != (operating_cost.shape[0], 3):
        raise RECMoELossError("risk probabilities must each be [B]")
    if _cpu_value_checks(risk_probabilities) and bool(
        ((risk_probabilities < 0.0) | (risk_probabilities > 1.0)).any()
    ):
        raise RECMoELossError("risk probabilities must lie in [0,1]")
    risk_targets = torch.stack(
        (
            sla_batch.clamp(0.0, 1.0),
            constraint_batch.clamp(0.0, 1.0),
            unserved_batch.clamp(0.0, 1.0),
        ),
        dim=-1,
    ).detach()
    tail_risk = (risk_probabilities - risk_targets).square().mean()

    action_smoothness = _action_smoothness(output)
    safe_anchor = _safe_anchor(output)
    anchor_multiplier = (1.0 - training_progress).pow(float(safe_anchor_decay_power))
    components = {
        "expected_operating_cost": expected_operating_cost,
        "upper_tail_cost": upper_tail_cost,
        "quantile_pinball": quantile_pinball,
        "sla_deadline": sla_deadline,
        "constraint_residual": constraint_residual,
        "route_continuation": route_continuation,
        "load_balance": load_balance,
        "tail_risk": tail_risk,
        "action_smoothness": action_smoothness,
        "safe_anchor": safe_anchor,
    }
    weight_values = weights.as_mapping()
    effective_weights = {
        name: operating_cost.new_tensor(weight_values[name], dtype=torch.float32)
        * (anchor_multiplier if name == "safe_anchor" else 1.0)
        for name in _COMPONENT_NAMES
    }
    weighted = {
        name: components[name] * effective_weights[name] for name in _COMPONENT_NAMES
    }
    total = torch.stack(tuple(weighted.values())).sum()
    return RECMoELossResult(
        total=total,
        components=components,
        weighted_components=weighted,
        effective_weights=effective_weights,
        safe_anchor_multiplier=anchor_multiplier,
        route_load=route_load,
        next_rolling_route_load=next_load,
    )


__all__ = [
    "GENERAL_EXPERTS",
    "RECMoELossError",
    "RECMoELossResult",
    "RECMoELossWeights",
    "r_ecmoe_loss",
]
