"""Protected multi-phase training engine for :class:`VPPWorldMoE`."""

from __future__ import annotations

import inspect
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .coverage import (
    FORMAL_COVERAGE_PATHS,
    PUBLIC_FAMILY_SLICES,
    central_rank_interval_100,
    conditional_central_rank_interval_100,
    conditional_central_quantile_interval,
    public_family_channel_weights,
)
from .dynamics import deadline_expired_work, normalize_daily_path_cost
from .loss_profiles import PROPER_PROFILE
from .losses import (
    PhaseLoss,
    WorldLossWeights,
    p1_latent_planner_loss,
    phase_loss,
)


PHASE_ORDER: tuple[str, ...] = ("w0", "w1", "u0", "p1", "j1", "c1")
PARAMETER_GROUP_ORDER: tuple[str, ...] = (
    "encoder",
    "exogenous_dynamics",
    "action_dynamics",
    "stochastic",
    "planner",
    "shared",
    "router",
    "physics",
)
PHASE_ACTIVE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "w0": ("encoder", "shared", "router"),
    "w1": (
        "encoder",
        "exogenous_dynamics",
        "action_dynamics",
        "stochastic",
        "shared",
        "router",
    ),
    "u0": (
        "encoder",
        "exogenous_dynamics",
        "action_dynamics",
        "stochastic",
        "shared",
        "router",
    ),
    "p1": ("planner", "router"),
    "j1": (
        "encoder",
        "exogenous_dynamics",
        "action_dynamics",
        "stochastic",
        "planner",
        "shared",
        "router",
        "physics",
    ),
    "c1": (),
}
CHECKPOINT_SCHEMA = "dcvpp.vpp-world-moe.checkpoint.v1"
RISK_CHUNK_SEED_OFFSETS: tuple[int, ...] = (0, 104_729, 209_759, 314_159, 419_431)


class WorldTrainingError(RuntimeError):
    """Training would violate the world-model execution contract."""


def loss_weights_from_config(config: Mapping[str, Any]) -> WorldLossWeights:
    """Materialize versioned phase weights instead of using stale defaults.

    R5-V2.2 retains the historical 0.10 self-target mixture term for diagnostic
    replay, while R5-V2.3 sets it to zero. Reading the registered YAML is part
    of the scientific lineage rather than a convenience.
    """

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise WorldTrainingError("configuration.training must be a mapping")
    objectives = training.get("phase_objectives")
    if not isinstance(objectives, Mapping):
        raise WorldTrainingError("training.phase_objectives must be a mapping")

    def phase_weights(phase: str) -> Mapping[str, Any]:
        value = objectives.get(phase)
        if not isinstance(value, Mapping) or not isinstance(value.get("weights"), Mapping):
            raise WorldTrainingError(
                f"training.phase_objectives.{phase}.weights must be a mapping"
            )
        return value["weights"]

    w0 = phase_weights("w0")
    w1 = phase_weights("w1")
    u0 = phase_weights("u0")
    moe_values = tuple(float(item["moe"]) for item in (w0, w1, u0))
    if len(set(moe_values)) != 1:
        raise WorldTrainingError("shared MoE weight must agree across W0/W1/U0")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise WorldTrainingError("configuration.model must be a mapping")
    return WorldLossWeights(
        quantile=float(w0["quantile"]),
        count=float(w0["count"]),
        state=float(w0["state"]),
        latent=float(w1["latent"]),
        rollout=float(w1["rollout"]),
        rollout_state=float(w1["state"]),
        crps=float(u0["crps"]),
        energy=float(u0["energy"]),
        variogram=float(u0["variogram"]),
        tail=float(u0["tail_crps"]),
        mixture_nll=float(u0["mixture_nll"]),
        atom_brier=float(u0.get("atom_brier", 0.0)),
        cost_critic_distillation=float(
            model.get("cost_risk_critic_distillation_weight", 0.0)
        ),
        auxiliary=moe_values[0],
        route_supervision=float(model.get("tail_route_aux_weight", 0.05)),
    )


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    minimum_epochs: int
    maximum_epochs: int
    learning_rate: float
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    patience: int = 20
    batch_size: int = 32
    validation_batch_size: int | None = None
    sample_count: int = 20
    formal_sample_count: int | None = None
    horizon: int = 96
    cvar_alpha: float = 0.9
    short_epoch_factor: int = 1
    formal_validation_interval: int = 1
    sentinel_validation_batches: int = 12
    checkpoint_interval_updates: int = 4
    checkpoint_max_seconds: float = 180.0
    expected_updates_per_full_epoch: int = 0
    target_minimum_updates: int = 0
    target_maximum_updates: int = 0
    target_patience_updates: int = 0
    head_learning_rate: float | None = None
    router_learning_rate: float | None = None
    tail_learning_rate: float | None = None
    matrix_weight_decay: float | None = None
    warmup_updates: int = 0
    cosine_decay_updates: int = 0
    minimum_learning_rate_ratio: float = 1.0
    adamw_beta2: float = 0.999
    w0_pcgrad: bool = False

    def __post_init__(self) -> None:
        normalized = self.name.lower()
        if normalized not in PHASE_ORDER:
            raise WorldTrainingError(f"unknown phase {self.name!r}")
        if normalized == "c1":
            if self.minimum_epochs != 0 or self.maximum_epochs != 0:
                raise WorldTrainingError("C1 is evaluation-only")
        elif not 1 <= self.minimum_epochs <= self.maximum_epochs:
            raise WorldTrainingError("invalid epoch bounds")
        if self.learning_rate < 0.0 or self.weight_decay < 0.0:
            raise WorldTrainingError("optimizer hyperparameters must be nonnegative")
        optional_lrs = (
            self.head_learning_rate,
            self.router_learning_rate,
            self.tail_learning_rate,
        )
        if any(value is not None and value < 0.0 for value in optional_lrs):
            raise WorldTrainingError(
                "discriminative learning rates must be nonnegative"
            )
        if self.matrix_weight_decay is not None and self.matrix_weight_decay < 0.0:
            raise WorldTrainingError("matrix weight decay must be nonnegative")
        if self.warmup_updates < 0 or self.cosine_decay_updates < 0:
            raise WorldTrainingError(
                "learning-rate schedule updates must be nonnegative"
            )
        if (
            self.cosine_decay_updates
            and self.cosine_decay_updates <= self.warmup_updates
        ):
            raise WorldTrainingError("cosine decay must finish after warmup")
        if not 0.0 < self.minimum_learning_rate_ratio <= 1.0:
            raise WorldTrainingError("minimum learning-rate ratio must lie in (0,1]")
        if not 0.0 < self.adamw_beta2 < 1.0:
            raise WorldTrainingError("AdamW beta2 must lie in (0,1)")
        if self.gradient_clip_norm <= 0.0 or self.batch_size <= 0:
            raise WorldTrainingError("clip norm and batch size must be positive")
        if self.validation_batch_size is not None and self.validation_batch_size <= 0:
            raise WorldTrainingError("validation batch size must be positive")
        if self.sample_count < 2 or self.horizon != 96:
            raise WorldTrainingError(
                "training requires >=2 paths and a 96-step horizon"
            )
        if self.formal_sample_count is not None and self.formal_sample_count < 2:
            raise WorldTrainingError("formal sample count must be at least two")
        factor = int(self.short_epoch_factor)
        if factor < 1 or factor & (factor - 1):
            raise WorldTrainingError(
                "short_epoch_factor must be a positive power of two"
            )
        if (
            self.formal_validation_interval <= 0
            or self.formal_validation_interval % factor
        ):
            raise WorldTrainingError(
                "formal validation interval must be a positive multiple of "
                "short_epoch_factor"
            )
        if self.sentinel_validation_batches <= 0:
            raise WorldTrainingError("sentinel validation batch count must be positive")
        if self.checkpoint_interval_updates <= 0:
            raise WorldTrainingError("checkpoint update interval must be positive")
        if self.checkpoint_max_seconds <= 0.0:
            raise WorldTrainingError("checkpoint wall-clock interval must be positive")
        if normalized != "c1" and any(
            value % factor
            for value in (self.minimum_epochs, self.maximum_epochs, self.patience)
        ):
            raise WorldTrainingError(
                "short-epoch min/max/patience must be multiples of the factor"
            )
        update_budget = (
            self.expected_updates_per_full_epoch,
            self.target_minimum_updates,
            self.target_maximum_updates,
            self.target_patience_updates,
        )
        if any(value < 0 for value in update_budget):
            raise WorldTrainingError("target update budgets must be nonnegative")
        if any(update_budget) and (
            not all(update_budget)
            or self.target_minimum_updates > self.target_maximum_updates
        ):
            raise WorldTrainingError(
                "target update budgets must be complete and monotonically bounded"
            )

    @property
    def active_groups(self) -> tuple[str, ...]:
        return PHASE_ACTIVE_GROUPS[self.name.lower()]

    @property
    def effective_validation_batch_size(self) -> int:
        """Return the independent validation batch or the train-batch fallback."""

        return (
            self.batch_size
            if self.validation_batch_size is None
            else self.validation_batch_size
        )


@dataclass(frozen=True)
class TrainerStepResult:
    phase: str
    loss: float
    components: Mapping[str, float]
    gradient_norm: float
    updated_groups: tuple[str, ...]


@dataclass(frozen=True)
class ResumeState:
    phase: str
    epoch: int = 0
    batch_in_short_epoch: int = 0
    short_epoch_factor: int = 1
    update: int = 0
    global_update: int = 0
    best_validation: float = float("inf")
    best_sentinel_validation: float = float("inf")
    formal_stale_short_epochs: int = 0
    short_epoch_loss_sum: float = 0.0
    short_epoch_updates: int = 0
    last_full_validation_epoch: int = 0
    formal_evaluations: int = 0
    stable_formal_evaluations: int = 0
    failure_stable_formal_evaluations: int = 0
    failure_best_gate_gap: float = float("inf")
    failure_best_proper_score: float = float("inf")
    selected_record_json: str = ""
    selection_reference_json: str = ""

    def __post_init__(self) -> None:
        if str(self.phase).lower() not in PHASE_ORDER:
            raise WorldTrainingError(f"unknown resume phase {self.phase!r}")
        if any(
            value < 0
            for value in (
                self.epoch,
                self.batch_in_short_epoch,
                self.update,
                self.global_update,
                self.formal_stale_short_epochs,
                self.short_epoch_updates,
                self.last_full_validation_epoch,
                self.formal_evaluations,
                self.stable_formal_evaluations,
                self.failure_stable_formal_evaluations,
            )
        ):
            raise WorldTrainingError("resume counters must be nonnegative")
        if (
            math.isnan(float(self.failure_best_gate_gap))
            or math.isnan(float(self.failure_best_proper_score))
            or float(self.failure_best_gate_gap) < 0.0
            or float(self.failure_best_proper_score) < 0.0
        ):
            raise WorldTrainingError("resume failure-plateau scores are invalid")
        factor = int(self.short_epoch_factor)
        if factor < 1 or factor & (factor - 1):
            raise WorldTrainingError(
                "resume short_epoch_factor must be a positive power of two"
            )
        if self.last_full_validation_epoch > self.epoch:
            raise WorldTrainingError("resume full-validation epoch is inconsistent")
        if self.batch_in_short_epoch != self.short_epoch_updates:
            raise WorldTrainingError(
                "resume batch and partial-update counters must match"
            )


@dataclass(frozen=True)
class ResolvedParameterGroups:
    names: Mapping[str, tuple[str, ...]]
    parameters: Mapping[str, tuple[nn.Parameter, ...]]

    def active_parameters(self, groups: Sequence[str]) -> tuple[nn.Parameter, ...]:
        result: list[nn.Parameter] = []
        seen: set[int] = set()
        for group in groups:
            if group not in self.parameters:
                raise WorldTrainingError(f"model has no parameter group {group!r}")
            for parameter in self.parameters[group]:
                if id(parameter) not in seen:
                    result.append(parameter)
                    seen.add(id(parameter))
        return tuple(result)


def _parameter_group_declaration(model: nn.Module) -> Mapping[str, Any]:
    for method_name in ("world_parameter_groups", "parameter_groups"):
        method = getattr(model, method_name, None)
        if callable(method):
            result = method()
            if not isinstance(result, Mapping):
                raise WorldTrainingError(f"{method_name}() must return a mapping")
            return result
    raise WorldTrainingError(
        "VPPWorldMoE must expose world_parameter_groups() or parameter_groups()"
    )


def resolve_parameter_groups(model: nn.Module) -> ResolvedParameterGroups:
    """Resolve and validate the eight explicit world-model parameter groups."""

    declared = _parameter_group_declaration(model)
    if set(declared) != set(PARAMETER_GROUP_ORDER):
        missing = sorted(set(PARAMETER_GROUP_ORDER) - set(declared))
        extra = sorted(set(declared) - set(PARAMETER_GROUP_ORDER))
        raise WorldTrainingError(
            f"parameter groups mismatch; missing={missing}, extra={extra}"
        )
    named = dict(model.named_parameters())
    reverse = {id(parameter): name for name, parameter in named.items()}
    group_names: dict[str, tuple[str, ...]] = {}
    group_parameters: dict[str, tuple[nn.Parameter, ...]] = {}
    assigned: dict[int, str] = {}
    for group in PARAMETER_GROUP_ORDER:
        raw = declared[group]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise WorldTrainingError(f"parameter group {group!r} must be a sequence")
        names: list[str] = []
        parameters: list[nn.Parameter] = []
        for item in raw:
            if isinstance(item, str):
                if item not in named:
                    raise WorldTrainingError(f"unknown parameter name {item!r}")
                name, parameter = item, named[item]
            elif isinstance(item, nn.Parameter):
                if id(item) not in reverse:
                    raise WorldTrainingError(
                        "parameter group contains a foreign parameter"
                    )
                name, parameter = reverse[id(item)], item
            else:
                raise WorldTrainingError(
                    "parameter groups contain names or Parameters only"
                )
            previous = assigned.get(id(parameter))
            if previous is not None:
                raise WorldTrainingError(
                    f"parameter {name!r} appears in groups {previous!r} and {group!r}"
                )
            assigned[id(parameter)] = group
            names.append(name)
            parameters.append(parameter)
        group_names[group] = tuple(names)
        group_parameters[group] = tuple(parameters)
    missing_parameters = sorted(
        name for name, parameter in named.items() if id(parameter) not in assigned
    )
    if missing_parameters:
        raise WorldTrainingError(
            f"ungrouped model parameters are forbidden: {missing_parameters[:8]}"
        )
    return ResolvedParameterGroups(group_names, group_parameters)


class ParameterFirewall:
    """Apply phase trainability and prove protected groups receive no gradients."""

    def __init__(self, model: nn.Module, groups: ResolvedParameterGroups) -> None:
        self.model = model
        self.groups = groups
        self.phase: str | None = None

    def configure(self, phase: str) -> tuple[nn.Parameter, ...]:
        normalized = str(phase).lower()
        if normalized not in PHASE_ACTIVE_GROUPS:
            raise WorldTrainingError(f"unknown phase {phase!r}")
        active = PHASE_ACTIVE_GROUPS[normalized]
        configure = getattr(self.model, "configure_trainable_groups", None)
        if callable(configure):
            configure(active)
        active_ids = {
            id(parameter) for parameter in self.groups.active_parameters(active)
        }
        for group, parameters in self.groups.parameters.items():
            enabled = group in active
            for parameter in parameters:
                parameter.requires_grad_(enabled and id(parameter) in active_ids)
                parameter.grad = None
        self.phase = normalized
        return self.groups.active_parameters(active)

    def assert_after_backward(self) -> None:
        if self.phase is None:
            raise WorldTrainingError("parameter firewall was not configured")
        active = set(PHASE_ACTIVE_GROUPS[self.phase])
        protected = set(PARAMETER_GROUP_ORDER) - active
        offenders = [
            name
            for group in protected
            for name, parameter in zip(
                self.groups.names[group], self.groups.parameters[group], strict=True
            )
            if parameter.grad is not None
        ]
        if offenders:
            raise WorldTrainingError(
                f"gradient firewall violation in {self.phase}: {offenders[:8]}"
            )


def _call_supported(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(callback)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = (
        kwargs
        if accepts_kwargs
        else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
    )
    return callback(*args, **selected)


def _model_input(batch: Any) -> Any:
    if isinstance(batch, Mapping):
        for key in ("world_input", "input_packet", "packet", "model_input"):
            if key in batch:
                return batch[key]
    return batch


def _target_packet(batch: Any) -> Any | None:
    if isinstance(batch, Mapping):
        for key in ("target_packet", "next_day_packet", "future_input_packet"):
            if key in batch:
                return batch[key]
    return None


def _batch_value(batch: Any, *names: str) -> Any | None:
    if not isinstance(batch, Mapping):
        return None
    for name in names:
        if name in batch and batch[name] is not None:
            return batch[name]
    return None


def _required_batch_tensor(batch: Any, *names: str, label: str) -> torch.Tensor:
    value = _batch_value(batch, *names)
    if not isinstance(value, torch.Tensor):
        raise WorldTrainingError(f"{label} is required as a Tensor")
    return value


def _planner_mapping(output: Any) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise WorldTrainingError("policy output must be a mapping")
    planner = output.get("planner")
    if not isinstance(planner, Mapping):
        raise WorldTrainingError("P1/J1 output is missing the planner mapping")
    return dict(planner)


def _augment_normalized_policy_cost(
    output: Any,
    batch: Any,
    *,
    rollout: Any | None = None,
) -> dict[str, Any]:
    """Attach only train-frozen normalized two-settlement path costs."""

    if not isinstance(output, Mapping):
        raise WorldTrainingError("policy output must be a mapping")
    merged = dict(output)
    selected_rollout = output.get("rollout") if rollout is None else rollout
    costs = _member(selected_rollout, "path_cost")
    if not isinstance(costs, torch.Tensor) or costs.ndim != 2:
        raise WorldTrainingError("policy rollout must expose FP32 [B,20] path_cost")
    _require_fp32(costs, label="rollout.path_cost")
    baseline = _required_batch_tensor(
        batch,
        "train_frozen_baseline_daily_cost",
        label="train_frozen_baseline_daily_cost",
    ).to(device=costs.device, dtype=torch.float32)
    try:
        normalized = normalize_daily_path_cost(costs, baseline)
    except ValueError as exc:
        raise WorldTrainingError(str(exc)) from exc
    planner = _planner_mapping(output)
    planner["scenario_costs"] = costs
    planner["normalized_scenario_costs"] = normalized
    required = (
        "cvar_eta",
        "raw_constraint_violations",
        "constraint_multipliers",
        "constraint_penalties",
        "raw_actions",
        "shield_correction",
        "day_ahead_commitment",
    )
    missing = [name for name in required if planner.get(name) is None]
    if missing:
        raise WorldTrainingError(f"planner R5-V2 contract is missing {sorted(missing)}")
    merged["planner"] = planner
    merged["rollout"] = selected_rollout
    return merged


def _gradient_tuple(
    objective: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    if objective.ndim != 0 or not bool(torch.isfinite(objective.detach())):
        raise WorldTrainingError("PCGrad objective must be one finite scalar")
    return tuple(
        torch.autograd.grad(
            objective,
            tuple(parameters),
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def add_gradient_tuples(
    left: Sequence[torch.Tensor | None] | None,
    right: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    """Accumulate gradient chunks while preserving genuinely unused entries."""

    if left is None:
        return tuple(
            None if value is None else value.detach().clone() for value in right
        )
    if len(left) != len(right):
        raise WorldTrainingError("gradient tuples have different parameter axes")
    result: list[torch.Tensor | None] = []
    for first, second in zip(left, right, strict=True):
        if first is None:
            result.append(None if second is None else second.detach().clone())
        elif second is None:
            result.append(first)
        else:
            result.append(first + second.detach())
    return tuple(result)


def add_gradient_tuples_in_place(
    left: Sequence[torch.Tensor | None] | None,
    right: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    """Accumulate one risk chunk without reallocating earlier chunk gradients."""

    if left is None:
        return tuple(
            None if value is None else value.detach().clone() for value in right
        )
    if len(left) != len(right):
        raise WorldTrainingError("gradient tuples have different parameter axes")
    result = list(left)
    for index, (first, second) in enumerate(zip(result, right, strict=True)):
        if first is None:
            result[index] = None if second is None else second.detach().clone()
        elif second is not None:
            first.add_(second.detach())
    return tuple(result)


def pcgrad_project(
    task_gradients: Mapping[str, Sequence[torch.Tensor | None]],
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor | None, ...]:
    """Deterministic PCGrad projection and merge on one parameter axis."""

    names = tuple(task_gradients)
    allowed_orders = {
        ("world", "uncertainty", "policy"),
        ("general", "rare"),
    }
    if names not in allowed_orders:
        raise WorldTrainingError(
            "PCGrad requires ordered world/uncertainty/policy or general/rare gradients"
        )
    width = len(task_gradients[names[0]])
    if any(len(task_gradients[name]) != width for name in names):
        raise WorldTrainingError("PCGrad task gradients have different axes")
    originals = {
        name: tuple(
            None if value is None else value.detach().clone()
            for value in task_gradients[name]
        )
        for name in names
    }
    projected = {name: list(values) for name, values in originals.items()}
    for name in names:
        for other in names:
            if other == name:
                continue
            pairs = [
                (current, reference)
                for current, reference in zip(
                    projected[name], originals[other], strict=True
                )
                if current is not None and reference is not None
            ]
            if not pairs:
                continue
            dot = sum((current * reference).sum() for current, reference in pairs)
            norm = sum(reference.square().sum() for _, reference in pairs)
            if float(dot.detach().cpu()) < 0.0 and float(norm.detach().cpu()) > 0.0:
                coefficient = dot / norm.clamp_min(float(eps))
                projected[name] = [
                    (
                        current - coefficient * reference
                        if current is not None and reference is not None
                        else current
                    )
                    for current, reference in zip(
                        projected[name], originals[other], strict=True
                    )
                ]
    merged: list[torch.Tensor | None] = []
    for index in range(width):
        values = [
            projected[name][index]
            for name in names
            if projected[name][index] is not None
        ]
        merged.append(None if not values else torch.stack(values).mean(dim=0))
    return tuple(merged)


def pcgrad_project_flat(
    task_gradients: dict[str, tuple[torch.Tensor | None, ...]],
    *,
    eps: float = 1.0e-12,
    consume: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Equivalent PCGrad using flat participation groups and GPU-only scalars.

    Parameters are partitioned by the tasks that actually use them.  A
    task-exclusive partition bypasses projection and is merged directly.  For
    shared partitions, pairwise dot products span exactly the same parameter
    intersections as :func:`pcgrad_project`, but only a handful of flat CUDA
    kernels are launched and no scalar is copied to the CPU.  ``consume`` is
    used by J1 after the final task gradient is available so the original
    per-parameter tensors can be released before projected buffers are built.
    """

    names = tuple(task_gradients)
    if names != ("world", "uncertainty", "policy"):
        raise WorldTrainingError(
            "flat PCGrad requires ordered world/uncertainty/policy gradients"
        )
    width = len(task_gradients[names[0]])
    if any(len(task_gradients[name]) != width for name in names):
        raise WorldTrainingError("PCGrad task gradients have different axes")

    devices = {
        value.device
        for gradients in task_gradients.values()
        for value in gradients
        if value is not None
    }
    dtypes = {
        value.dtype
        for gradients in task_gradients.values()
        for value in gradients
        if value is not None
    }
    if len(devices) != 1 or len(dtypes) != 1:
        raise WorldTrainingError(
            "flat PCGrad requires one gradient device and one gradient dtype"
        )

    layouts: dict[int, list[tuple[int, torch.Size, int]]] = {}
    for index in range(width):
        values = tuple(task_gradients[name][index] for name in names)
        present = tuple(value for value in values if value is not None)
        if not present:
            continue
        shape = present[0].shape
        if any(value.shape != shape for value in present[1:]):
            raise WorldTrainingError("PCGrad task gradient shapes differ")
        mask = sum(
            1 << task_index
            for task_index, value in enumerate(values)
            if value is not None
        )
        layouts.setdefault(mask, []).append((index, shape, present[0].numel()))

    originals: dict[int, dict[str, torch.Tensor]] = {}
    for mask, layout in layouts.items():
        originals[mask] = {}
        for task_index, name in enumerate(names):
            if not mask & (1 << task_index):
                continue
            originals[mask][name] = torch.cat(
                [
                    task_gradients[name][index].detach().reshape(-1)
                    for index, _, _ in layout
                    if task_gradients[name][index] is not None
                ]
            )
    if consume:
        task_gradients.clear()

    projected: dict[int, dict[str, torch.Tensor]] = {}
    for mask, vectors in originals.items():
        projected[mask] = {
            name: value if mask.bit_count() == 1 else value.clone()
            for name, value in vectors.items()
        }

    for name_index, name in enumerate(names):
        for other_index, other in enumerate(names):
            if other == name:
                continue
            pair_masks = [
                mask
                for mask in layouts
                if mask & (1 << name_index) and mask & (1 << other_index)
            ]
            if not pair_masks:
                continue
            dot = torch.stack(
                [
                    torch.dot(projected[mask][name], originals[mask][other])
                    for mask in pair_masks
                ]
            ).sum()
            norm = torch.stack(
                [originals[mask][other].square().sum() for mask in pair_masks]
            ).sum()
            coefficient = torch.where(
                (dot < 0.0) & (norm > 0.0),
                dot / norm.clamp_min(float(eps)),
                torch.zeros_like(dot),
            )
            for mask in pair_masks:
                projected[mask][name].addcmul_(
                    originals[mask][other], coefficient, value=-1.0
                )

    merged_groups: dict[int, torch.Tensor] = {}
    for mask, vectors in projected.items():
        ordered = [vectors[name] for name in names if name in vectors]
        merged = ordered[0] if len(ordered) == 1 else ordered[0].clone()
        for value in ordered[1:]:
            merged.add_(value)
        if len(ordered) > 1:
            merged.div_(float(len(ordered)))
        merged_groups[mask] = merged

    merged: list[torch.Tensor | None] = [None] * width
    for mask, layout in layouts.items():
        flat = merged_groups[mask]
        offset = 0
        for index, shape, size in layout:
            merged[index] = flat.narrow(0, offset, size).view(shape)
            offset += size
        if offset != flat.numel():
            raise WorldTrainingError("flat PCGrad layout did not close")
    return tuple(merged)


def _merge_training_aliases(output: Any, batch: Any) -> Any:
    """Copy optional precomputed target aliases into a mutable output mapping."""

    if not isinstance(output, Mapping) or not isinstance(batch, Mapping):
        return output
    merged = dict(output)
    aliases = (
        "reconstructions",
        "latent_predictions",
        "observation_predictions",
        "scenarios",
        "distribution_mean",
        "distribution_log_scale",
        "scenario_costs",
        "shielded_actions",
        "raw_actions",
        "constraint_violations",
        "sla_shortfall",
        "terminal_violation",
        "normalized_scenario_costs",
        "cvar_eta",
        "raw_constraint_violations",
        "constraint_multipliers",
        "constraint_penalties",
        "shield_correction",
    )
    training_output = batch.get("training_output")
    if isinstance(training_output, Mapping):
        for name in aliases:
            if name not in merged and name in training_output:
                merged[name] = training_output[name]
    if "router_aux_loss" in merged and "auxiliary_losses" not in merged:
        merged["auxiliary_losses"] = merged["router_aux_loss"]
    return merged


def _member(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _require_fp32(value: Any, *, label: str) -> None:
    if isinstance(value, torch.Tensor) and value.dtype is not torch.float32:
        raise WorldTrainingError(f"{label} must cross the scientific boundary in FP32")


def assert_scientific_precision(output: Any) -> None:
    """Fail closed if physical or risk values escape in reduced precision."""

    if not isinstance(output, Mapping):
        return
    _require_fp32(output.get("physics_residual"), label="physics_residual")
    rollout = output.get("rollout")
    if rollout is not None:
        for name in ("path_cost", "expected_cost", "cvar_cost"):
            _require_fp32(_member(rollout, name), label=f"rollout.{name}")
    plan = output.get("plan")
    actions = None if plan is None else _member(plan, "actions")
    if actions is not None:
        for name in (
            "power_balance_residual",
            "migration_balance_residual",
            "constraint_violation",
            "shield_intervention",
        ):
            _require_fp32(_member(actions, name), label=f"plan.actions.{name}")


def _validation_support_bounds(
    packet: Any,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand train-frozen physical support bounds over the validation horizon."""

    constraints = _member(packet, "constraints")
    if constraints is None:
        raise WorldTrainingError("mixed-support validation requires constraints")

    def expand(name: str) -> torch.Tensor:
        value = _member(constraints, name)
        expected = (int(target.shape[0]), 4)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise WorldTrainingError(f"constraints.{name} must have shape {expected}")
        value32 = value.float()
        if not bool(torch.isfinite(value32).all()):
            raise WorldTrainingError(f"constraints.{name} is non-finite")
        return value32[:, None, :].expand(-1, int(target.shape[1]), -1)

    return expand("pv_capacity"), expand("pue_min"), expand("pue_max")


def _mixed_rank_diagnostic_fields(
    *,
    family: str,
    nominal: int,
    truth: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    finite_target: torch.Tensor,
    valid_mask: torch.Tensor,
    insufficient_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return exact additive coverage evidence and per-origin block statistics."""

    expected = truth.shape
    tensors = (lower, upper, finite_target, valid_mask, insufficient_mask)
    if any(value.shape != expected for value in tensors):
        raise WorldTrainingError(f"{family} conditional coverage shapes differ")
    if valid_mask.dtype is not torch.bool or insufficient_mask.dtype is not torch.bool:
        raise WorldTrainingError("coverage validity masks must be boolean")
    channel_weights = public_family_channel_weights(
        family,
        device=truth.device,
        dtype=torch.float32,
    )
    weights = channel_weights.view(1, 1, -1).expand_as(truth)
    valid_weights = valid_mask.float() * weights
    hit = ((truth >= lower) & (truth <= upper) & valid_mask).float()
    lower_miss = ((truth < lower) & valid_mask).float()
    upper_miss = ((truth > upper) & valid_mask).float()
    hit_sum = (hit * weights).sum()
    target_sum = (finite_target.float() * valid_weights).sum()
    valid_count = valid_weights.sum()
    lower_miss_sum = (lower_miss * weights).sum()
    upper_miss_sum = (upper_miss * weights).sum()
    insufficient_count = (insufficient_mask.float() * weights).sum()
    denominator = valid_count.clamp_min(1.0e-12)

    daily_hit = (hit * weights).sum(dim=(1, 2))
    daily_target = (finite_target.float() * valid_weights).sum(dim=(1, 2))
    daily_count = valid_weights.sum(dim=(1, 2))
    prefix = f"coverage_{family}_{nominal}"
    daily_prefix = f"_coverage_daily_{family}_{nominal}"
    return {
        prefix: float((hit_sum / denominator).detach().cpu()),
        f"coverage_target_{family}_{nominal}": float(
            (target_sum / denominator).detach().cpu()
        ),
        f"{prefix}_hit_sum": float(hit_sum.detach().cpu()),
        f"{prefix}_target_sum": float(target_sum.detach().cpu()),
        f"{prefix}_valid_count": float(valid_count.detach().cpu()),
        f"{prefix}_insufficient_count": float(insufficient_count.detach().cpu()),
        f"{prefix}_lower_miss_sum": float(lower_miss_sum.detach().cpu()),
        f"{prefix}_upper_miss_sum": float(upper_miss_sum.detach().cpu()),
        f"coverage_lower_miss_{family}_{nominal}": float(
            (lower_miss_sum / denominator).detach().cpu()
        ),
        f"coverage_upper_miss_{family}_{nominal}": float(
            (upper_miss_sum / denominator).detach().cpu()
        ),
        f"coverage_interior_valid_count_{family}_{nominal}": float(
            valid_count.detach().cpu()
        ),
        f"{daily_prefix}_hit_sum": tuple(
            float(value) for value in daily_hit.detach().cpu().tolist()
        ),
        f"{daily_prefix}_target_sum": tuple(
            float(value) for value in daily_target.detach().cpu().tolist()
        ),
        f"{daily_prefix}_valid_count": tuple(
            float(value) for value in daily_count.detach().cpu().tolist()
        ),
    }


def _atom_calibration_diagnostic_fields(
    output: Mapping[str, Any],
    truth: torch.Tensor,
    packet: Any,
) -> dict[str, Any]:
    """Score final unconditional boundary-event probabilities for validation."""

    analytic = output.get("analytic_distribution", output.get("mixed_distribution"))
    analytic_probabilities = (
        analytic.get("atom_probabilities")
        if isinstance(analytic, Mapping)
        else None
    )
    raw = (
        analytic_probabilities
        if isinstance(analytic_probabilities, Mapping)
        else output.get("support_probabilities")
    )
    if not isinstance(raw, Mapping):
        return {}
    paths = int(output["scenarios"].shape[1])
    batch, time = int(truth.shape[0]), int(truth.shape[1])
    pv_capacity, pue_min, pue_max = _validation_support_bounds(packet, truth)

    analytic_shapes = {
        "workload_zero": (batch, time, 1),
        "pv_zero": (batch, time, 4),
        "pv_capacity": (batch, time, 4),
        "pue_min": (batch, time, 4),
        "pue_max": (batch, time, 4),
    }
    expected_shapes = (
        analytic_shapes
        if isinstance(analytic_probabilities, Mapping)
        else {
            name: (shape[0], paths, *shape[1:])
            for name, shape in analytic_shapes.items()
        }
    )
    probability: dict[str, torch.Tensor] = {}
    for name, expected in expected_shapes.items():
        value = raw.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            return {"atom_support_fields_present": 0.0}
        value32 = value.float()
        if not bool(torch.isfinite(value32).all()) or bool(
            ((value32 < 0.0) | (value32 > 1.0)).any()
        ):
            return {"atom_support_fields_present": 0.0}
        probability[name] = (
            value32
            if isinstance(analytic_probabilities, Mapping)
            else value32.mean(dim=1)
        )

    workload_zero = probability["workload_zero"]
    pv_zero = probability["pv_zero"]
    pv_capacity_probability = probability["pv_capacity"]
    pue_min_probability = probability["pue_min"]
    pue_max_probability = probability["pue_max"]
    if bool((pv_zero + pv_capacity_probability > 1.0 + 2.0e-6).any()) or bool(
        (pue_min_probability + pue_max_probability > 1.0 + 2.0e-6).any()
    ):
        return {"atom_support_fields_present": 0.0}

    workload_truth = truth[..., 12:13]
    pv_truth = truth[..., 13:17]
    pue_truth = truth[..., 17:21]
    events = {
        "workload": {
            "zero": (workload_zero, workload_truth.eq(0.0)),
            "positive": (1.0 - workload_zero, workload_truth.ne(0.0)),
        },
        "pv": {
            "zero": (pv_zero, pv_truth.eq(0.0)),
            "capacity": (pv_capacity_probability, pv_truth.eq(pv_capacity)),
            "interior": (
                1.0 - pv_zero - pv_capacity_probability,
                ~(pv_truth.eq(0.0) | pv_truth.eq(pv_capacity)),
            ),
        },
        "pue": {
            "min": (pue_min_probability, pue_truth.eq(pue_min)),
            "max": (pue_max_probability, pue_truth.eq(pue_max)),
            "interior": (
                1.0 - pue_min_probability - pue_max_probability,
                ~(pue_truth.eq(pue_min) | pue_truth.eq(pue_max)),
            ),
        },
    }
    result: dict[str, Any] = {"atom_support_fields_present": 1.0}
    for family, family_events in events.items():
        for event, (forecast, observed_mask) in family_events.items():
            observed = observed_mask.float()
            forecast_sum_daily = forecast.sum(dim=(1, 2))
            observed_sum_daily = observed.sum(dim=(1, 2))
            count_daily = torch.full_like(
                forecast_sum_daily,
                float(forecast.shape[1] * forecast.shape[2]),
            )
            forecast_sum = forecast_sum_daily.sum()
            observed_sum = observed_sum_daily.sum()
            valid_count = count_daily.sum()
            prefix = f"atom_{family}_{event}"
            result.update(
                {
                    f"{prefix}_forecast_sum": float(forecast_sum.detach().cpu()),
                    f"{prefix}_observed_sum": float(observed_sum.detach().cpu()),
                    f"{prefix}_valid_count": float(valid_count.detach().cpu()),
                    f"{prefix}_forecast_probability": float(
                        (forecast_sum / valid_count).detach().cpu()
                    ),
                    f"{prefix}_observed_prevalence": float(
                        (observed_sum / valid_count).detach().cpu()
                    ),
                    f"{prefix}_reliability_error": float(
                        ((forecast_sum - observed_sum).abs() / valid_count)
                        .detach()
                        .cpu()
                    ),
                    f"_atom_daily_{family}_{event}_forecast_sum": tuple(
                        float(value)
                        for value in forecast_sum_daily.detach().cpu().tolist()
                    ),
                    f"_atom_daily_{family}_{event}_observed_sum": tuple(
                        float(value)
                        for value in observed_sum_daily.detach().cpu().tolist()
                    ),
                    f"_atom_daily_{family}_{event}_valid_count": tuple(
                        float(value) for value in count_daily.detach().cpu().tolist()
                    ),
                }
            )
    return result


def validation_diagnostics(output: Any, batch: Any) -> dict[str, Any]:
    """Non-differentiable validation gates; never included in training loss."""

    if not isinstance(output, Mapping):
        return {}
    diagnostics: dict[str, Any] = {}
    packet = _model_input(batch)
    origin_index = _member(packet, "origin_index")
    if isinstance(origin_index, torch.Tensor):
        diagnostics["_origin_count"] = float(origin_index.numel())
    price_paths = output.get("joint_price_paths")
    if isinstance(price_paths, Mapping):
        da = price_paths.get("da")
        rt = price_paths.get("rt")
        dart = price_paths.get("dart")
        if all(isinstance(value, torch.Tensor) for value in (da, rt, dart)):
            diagnostics["dart_identity_max_abs"] = float(
                (da.float() - rt.float() - dart.float()).abs().amax().detach().cpu()
            )
    latent_diagnostics = output.get("w1_latent_diagnostics")
    if isinstance(latent_diagnostics, Mapping):
        raw_kl = latent_diagnostics.get("raw_kl_mean")
        floor_fraction = latent_diagnostics.get("free_bit_floor_fraction")
        if isinstance(raw_kl, torch.Tensor) and isinstance(
            floor_fraction, torch.Tensor
        ):
            raw_value = float(raw_kl.detach().float().cpu())
            floor_value = float(floor_fraction.detach().float().cpu())
            diagnostics["w1_latent_raw_kl_mean"] = raw_value
            diagnostics["w1_latent_free_bit_floor_fraction"] = floor_value
            diagnostics["latent_kl_non_degenerate"] = float(
                np.isfinite(raw_value) and raw_value > 1.0e-6
            )
    queue_diagnostics = output.get("w1_queue_diagnostics")
    if isinstance(queue_diagnostics, Mapping):
        non_degenerate = queue_diagnostics.get("prediction_non_degenerate")
        if isinstance(non_degenerate, torch.Tensor):
            diagnostics["queue_rollout_non_degenerate"] = float(
                non_degenerate.detach().float().cpu()
            )
    scenarios = output.get("scenarios")
    target = _batch_value(batch, "stochastic_target", "scenario_target")
    if isinstance(scenarios, torch.Tensor) and isinstance(target, torch.Tensor):
        samples = scenarios.float()
        truth = target.float()
        if samples.ndim == truth.ndim + 1 and samples.shape[0] == truth.shape[0]:
            if int(samples.shape[1]) == FORMAL_COVERAGE_PATHS:
                support_probabilities = output.get("support_probabilities")
                analytic_distribution = output.get(
                    "analytic_distribution", output.get("mixed_distribution")
                )
                if isinstance(analytic_distribution, Mapping):
                    atom_fields = _atom_calibration_diagnostic_fields(
                        output, truth, packet
                    )
                    diagnostics.update(atom_fields)
                    raw_quantiles = analytic_distribution.get("quantiles")
                    if (
                        bool(atom_fields.get("atom_support_fields_present", 0.0))
                        and isinstance(raw_quantiles, Mapping)
                    ):
                        pv_capacity, pue_min, pue_max = _validation_support_bounds(
                            packet, truth
                        )
                        family_truth = {
                            family: truth[..., channel_slice]
                            for family, channel_slice in PUBLIC_FAMILY_SLICES.items()
                        }
                        truth_interior_masks = {
                            "price": torch.ones_like(
                                family_truth["price"], dtype=torch.bool
                            ),
                            "workload": family_truth["workload"].ne(0.0),
                            "pv": family_truth["pv"].ne(0.0)
                            & family_truth["pv"].ne(pv_capacity),
                            "pue": family_truth["pue"].ne(pue_min)
                            & family_truth["pue"].ne(pue_max),
                        }
                        for nominal in (50, 80, 90):
                            diagnostics[f"coverage_target_{nominal}"] = (
                                float(nominal) / 100.0
                            )
                            for family in ("price", "workload", "pv", "pue"):
                                family_quantiles = raw_quantiles.get(family)
                                if not isinstance(family_quantiles, torch.Tensor):
                                    continue
                                interval = conditional_central_quantile_interval(
                                    family_quantiles,
                                    truth_interior_masks[family],
                                    nominal,
                                )
                                diagnostics.update(
                                    _mixed_rank_diagnostic_fields(
                                        family=family,
                                        nominal=nominal,
                                        truth=family_truth[family],
                                        lower=interval.lower,
                                        upper=interval.upper,
                                        finite_target=(
                                            interval.finite_sample_target
                                        ),
                                        valid_mask=interval.valid_mask,
                                        insufficient_mask=torch.zeros_like(
                                            interval.valid_mask, dtype=torch.bool
                                        ),
                                    )
                                )
                        diagnostics["analytic_distribution_coverage"] = 1.0
                elif isinstance(support_probabilities, Mapping):
                    atom_fields = _atom_calibration_diagnostic_fields(
                        output, truth, packet
                    )
                    diagnostics.update(atom_fields)
                    if bool(atom_fields.get("atom_support_fields_present", 0.0)):
                        pv_capacity, pue_min, pue_max = _validation_support_bounds(
                            packet, truth
                        )
                        family_samples = {
                            family: samples[..., channel_slice]
                            for family, channel_slice in PUBLIC_FAMILY_SLICES.items()
                        }
                        family_truth = {
                            family: truth[..., channel_slice]
                            for family, channel_slice in PUBLIC_FAMILY_SLICES.items()
                        }
                        scenario_atom_masks = {
                            "workload": family_samples["workload"].eq(0.0),
                            "pv": family_samples["pv"].eq(0.0)
                            | family_samples["pv"].eq(pv_capacity.unsqueeze(1)),
                            "pue": family_samples["pue"].eq(pue_min.unsqueeze(1))
                            | family_samples["pue"].eq(pue_max.unsqueeze(1)),
                        }
                        truth_interior_masks = {
                            "workload": family_truth["workload"].ne(0.0),
                            "pv": family_truth["pv"].ne(0.0)
                            & family_truth["pv"].ne(pv_capacity),
                            "pue": family_truth["pue"].ne(pue_min)
                            & family_truth["pue"].ne(pue_max),
                        }
                        for nominal in (50, 80, 90):
                            price_low, price_high, price_spec = (
                                central_rank_interval_100(
                                    family_samples["price"], nominal, path_dim=1
                                )
                            )
                            price_truth = family_truth["price"]
                            diagnostics.update(
                                _mixed_rank_diagnostic_fields(
                                    family="price",
                                    nominal=nominal,
                                    truth=price_truth,
                                    lower=price_low,
                                    upper=price_high,
                                    finite_target=torch.full_like(
                                        price_truth,
                                        float(price_spec.finite_sample_target),
                                    ),
                                    valid_mask=torch.ones_like(
                                        price_truth, dtype=torch.bool
                                    ),
                                    insufficient_mask=torch.zeros_like(
                                        price_truth, dtype=torch.bool
                                    ),
                                )
                            )
                            # Preserve the historical scalar for price, the
                            # only all-continuous family in the mixed schema.
                            diagnostics[f"coverage_target_{nominal}"] = float(
                                price_spec.finite_sample_target
                            )
                            for family in ("workload", "pv", "pue"):
                                interval = conditional_central_rank_interval_100(
                                    family_samples[family],
                                    scenario_atom_masks[family],
                                    truth_interior_masks[family],
                                    nominal,
                                    path_dim=1,
                                )
                                diagnostics.update(
                                    _mixed_rank_diagnostic_fields(
                                        family=family,
                                        nominal=nominal,
                                        truth=family_truth[family],
                                        lower=interval.lower,
                                        upper=interval.upper,
                                        finite_target=interval.finite_sample_target,
                                        valid_mask=interval.valid_mask,
                                        insufficient_mask=(
                                            interval.insufficient_interior_mask
                                        ),
                                    )
                                )
                else:
                    # R5-V2.3 compatibility: all families were treated as
                    # continuous.  R5-V2.4 sets a nonzero atom loss weight, so
                    # this branch cannot release a new mixed-support run.
                    for nominal in (50, 80, 90):
                        low, high, rank_spec = central_rank_interval_100(
                            samples, nominal, path_dim=1
                        )
                        hits = ((truth >= low) & (truth <= high)).float()
                        diagnostics[f"coverage_target_{nominal}"] = float(
                            rank_spec.finite_sample_target
                        )
                        for family, channel_slice in PUBLIC_FAMILY_SLICES.items():
                            family_hits = hits[..., channel_slice]
                            channel_weights = public_family_channel_weights(
                                family,
                                device=family_hits.device,
                                dtype=family_hits.dtype,
                            )
                            daily_coverage = (family_hits * channel_weights).sum(
                                dim=-1
                            ).mean(dim=-1)
                            diagnostics[f"coverage_{family}_{nominal}"] = float(
                                daily_coverage.mean().detach().cpu()
                            )
                            diagnostics[
                                f"_coverage_daily_{family}_{nominal}"
                            ] = tuple(
                                float(value)
                                for value in daily_coverage.detach().cpu().tolist()
                            )
            else:
                # Twenty-path sentinel coverage is diagnostic only.  It keeps
                # the historical interpolated summaries for trend monitoring,
                # but it deliberately cannot satisfy the R5-V2.3 release gate.
                for nominal, lower, upper in (
                    (50, 0.25, 0.75),
                    (80, 0.10, 0.90),
                    (90, 0.05, 0.95),
                ):
                    low = torch.quantile(samples, lower, dim=1)
                    high = torch.quantile(samples, upper, dim=1)
                    diagnostics[f"coverage_{nominal}"] = float(
                        ((truth >= low) & (truth <= high)).float().mean().cpu()
                    )
            if samples.shape[1] > 1:
                adjacent = (
                    (samples[:, 1:] - samples[:, :-1])
                    .abs()
                    .amax(dim=tuple(range(2, samples.ndim)))
                )
                diagnostics["scenario_duplicate_rate"] = float(
                    (adjacent <= 1.0e-8).float().mean().cpu()
                )
                diagnostics["scenario_diversity"] = float(
                    (samples[:, 1:] - samples[:, :-1]).square().mean().sqrt().cpu()
                )
    plan = output.get("plan")
    actions = _member(plan, "actions")
    if actions is not None:
        violation = _member(actions, "constraint_violation")
        if isinstance(violation, torch.Tensor):
            diagnostics["constraint_violation_max_abs"] = float(
                violation.float().abs().amax().detach().cpu()
            )
        rollout = output.get("rollout")
        arrivals = _member(_member(rollout, "observations"), "arrivals")
        compute = _member(actions, "compute")
        constraints = _member(packet, "constraints")
        if (
            isinstance(arrivals, torch.Tensor)
            and isinstance(compute, torch.Tensor)
            and constraints is not None
        ):
            raw_deadline = _batch_value(batch, "deadline_intervals")
            deadline = (
                int(raw_deadline.reshape(-1)[0].item())
                if isinstance(raw_deadline, torch.Tensor)
                else 4
            )
            expired = deadline_expired_work(
                arrivals.float(),
                compute.float(),
                _member(constraints, "initial_queue"),
                deadline_intervals=deadline,
            )
            diagnostics["sla_expired_work_mean"] = float(expired.mean().detach().cpu())
            diagnostics["sla_expired_work_max"] = float(expired.amax().detach().cpu())
    return diagnostics


class WorldModelTrainer:
    """One-phase trainer with an explicit parameter and precision boundary."""

    def __init__(
        self,
        model: nn.Module,
        spec: PhaseSpec,
        *,
        device: torch.device | str,
        weights: WorldLossWeights = WorldLossWeights(),
        training_builder: Callable[..., PhaseLoss] = phase_loss,
        selection_builder: Callable[..., PhaseLoss] = phase_loss,
        training_loss_profile: str = PROPER_PROFILE,
        selection_profile: str = PROPER_PROFILE,
        # Backward-compatible test seam. Formal runners use the two explicit
        # builders above and never pass this alias.
        loss_builder: Callable[..., PhaseLoss] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.spec = spec
        self.device = torch.device(device)
        self.weights = weights
        if loss_builder is not None:
            training_builder = loss_builder
            selection_builder = loss_builder
        self.training_builder = training_builder
        self.selection_builder = selection_builder
        self.training_loss_profile = str(training_loss_profile)
        self.selection_profile = str(selection_profile)
        self.gradient_checkpointing_enabled = (
            self._configure_phase_gradient_checkpointing()
        )
        self.groups = resolve_parameter_groups(model)
        self.firewall = ParameterFirewall(model, self.groups)
        active = self.firewall.configure(spec.name)
        if spec.name == "c1":
            if optimizer is not None:
                raise WorldTrainingError("C1 cannot own an optimizer")
            self.optimizer = None
        else:
            if not active:
                raise WorldTrainingError(
                    f"phase {spec.name} has no trainable parameters"
                )
            self.optimizer = optimizer or self._build_adamw(active)
        self.use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    def _configure_phase_gradient_checkpointing(self) -> bool:
        """Apply the model's optional phase allow-list to the backbone.

        Older configurations only expose the global boolean and therefore keep
        their previous behavior.  R5-V2.1 can additionally register a phase
        tuple; changing phases then changes recomputation without changing the
        architecture or checkpoint parameter schema.
        """

        model_config = getattr(self.model, "config", None)
        if model_config is None:
            return False
        enabled = bool(getattr(model_config, "gradient_checkpointing", False))
        phase_allow_list = getattr(model_config, "gradient_checkpointing_phases", None)
        if phase_allow_list is not None:
            enabled = enabled and self.spec.name.lower() in {
                str(item).lower() for item in phase_allow_list
            }
        backbone = getattr(self.model, "backbone", None)
        backbone_config = getattr(backbone, "config", None)
        if backbone_config is not None and hasattr(
            backbone_config, "gradient_checkpointing"
        ):
            if not is_dataclass(backbone_config):
                raise WorldTrainingError(
                    "phase-specific gradient checkpointing requires a dataclass config"
                )
            backbone.config = replace(backbone_config, gradient_checkpointing=enabled)
        return enabled

    def _build_adamw(self, active: Sequence[nn.Parameter]) -> torch.optim.Optimizer:
        """Build resume-safe discriminative AdamW groups for R5-V2.1.

        Prediction queries receive the head rate, the causal Tail router and
        Tail residual receive their registered rates, and all remaining W0
        representation weights use the conservative backbone rate.  Only
        matrix weights decay; normalization, biases, intercepts, and gates do
        not.  Extra group metadata is serialized by ``optimizer.state_dict``
        so the analytic warmup/cosine schedule resumes without a second state
        file.
        """

        active_ids = {id(parameter) for parameter in active}
        declared_group = {
            name: group
            for group in PARAMETER_GROUP_ORDER
            for name in self.groups.names[group]
        }
        buckets: dict[tuple[float, float, str], list[nn.Parameter]] = {}
        for name, parameter in self.model.named_parameters():
            if id(parameter) not in active_ids:
                continue
            group = declared_group[name]
            learning_rate = float(self.spec.learning_rate)
            rate_role = "backbone"
            if self.spec.head_learning_rate is not None and (
                name.startswith("observation_queries.")
                or (
                    self.spec.name == "w1"
                    and group in {"exogenous_dynamics", "action_dynamics", "stochastic"}
                )
            ):
                learning_rate = float(self.spec.head_learning_rate)
                rate_role = (
                    "dynamics"
                    if self.spec.name == "w1"
                    and group in {"exogenous_dynamics", "action_dynamics", "stochastic"}
                    else "prediction_head"
                )
            if group == "router" and self.spec.router_learning_rate is not None:
                learning_rate = float(self.spec.router_learning_rate)
                rate_role = "router"
            if any(marker in name for marker in (".tail_expert.", ".tail_output_gate")):
                if self.spec.tail_learning_rate is not None:
                    learning_rate = float(self.spec.tail_learning_rate)
                    rate_role = "tail_expert"
            lowered = name.lower()
            no_decay = (
                parameter.ndim < 2
                or name.endswith(".bias")
                or "norm" in lowered
                or "gate" in lowered
                or "intercept" in lowered
            )
            matrix_decay = (
                self.spec.weight_decay
                if self.spec.matrix_weight_decay is None
                else self.spec.matrix_weight_decay
            )
            decay = 0.0 if no_decay else float(matrix_decay)
            buckets.setdefault((learning_rate, decay, rate_role), []).append(parameter)
        if not buckets:
            raise WorldTrainingError(f"phase {self.spec.name} has no optimizer buckets")
        parameter_groups = [
            {
                "params": parameters,
                "lr": learning_rate,
                "base_lr": learning_rate,
                "initial_lr": learning_rate,
                "weight_decay": decay,
                "rate_role": role,
            }
            for (learning_rate, decay, role), parameters in buckets.items()
        ]
        return torch.optim.AdamW(
            parameter_groups,
            betas=(0.9, float(self.spec.adamw_beta2)),
        )

    def _completed_optimizer_updates(self) -> int:
        if self.optimizer is None:
            return 0
        completed = 0
        for state in self.optimizer.state.values():
            raw = state.get("step", 0)
            value = (
                int(raw.detach().cpu().item())
                if isinstance(raw, torch.Tensor)
                else int(raw)
            )
            completed = max(completed, value)
        return completed

    def _apply_learning_rate_schedule(self) -> None:
        if self.optimizer is None:
            return
        next_update = self._completed_optimizer_updates() + 1
        warmup = int(self.spec.warmup_updates)
        finish = int(self.spec.cosine_decay_updates)
        floor = float(self.spec.minimum_learning_rate_ratio)
        if warmup > 0 and next_update <= warmup:
            factor = float(next_update) / float(warmup)
        elif finish > warmup and next_update < finish:
            progress = float(next_update - warmup) / float(finish - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = floor + (1.0 - floor) * cosine
        elif finish > 0:
            factor = floor
        else:
            factor = 1.0
        for group in self.optimizer.param_groups:
            base = float(group.get("base_lr", group.get("initial_lr", group["lr"])))
            group["base_lr"] = base
            group["lr"] = base * factor

    def _autocast_context(self) -> Any:
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward(
        self,
        batch: Any,
        *,
        seed: int | None = None,
        formal_seed_offsets: Sequence[int] | None = None,
    ) -> Any:
        plan = self.spec.name in {"p1", "j1", "c1"}
        offsets = (
            None
            if formal_seed_offsets is None
            else tuple(int(value) for value in formal_seed_offsets)
        )
        if offsets is not None and self.spec.name != "u0":
            raise WorldTrainingError(
                "formal seed offsets are only valid for U0 evaluation"
            )
        sample_count = None if self.spec.name in {"w0", "w1"} else self.spec.sample_count
        if offsets is not None:
            sample_count = (
                len(offsets) * int(self.spec.sample_count)
                if self.spec.formal_sample_count is None
                else int(self.spec.formal_sample_count)
            )
        with self._autocast_context():
            output = _call_supported(
                self.model,
                _model_input(batch),
                phase=self.spec.name,
                horizon=self.spec.horizon,
                sample_count=sample_count,
                seed=seed,
                path_seed_offsets=offsets,
                plan=plan,
                target_packet=_target_packet(batch),
                count_label_mask=_batch_value(
                    batch, "workload_count_mask", "arrival_count_mask"
                ),
                future_target_sequence=_batch_value(batch, "future_target_sequence"),
                future_state_sequence=_batch_value(batch, "future_state_sequence"),
                teacher_action_summary=_batch_value(batch, "teacher_action_summary"),
                teacher_action_mask=_batch_value(batch, "teacher_action_mask"),
                forecast_anchors=_batch_value(batch, "forecast_anchors"),
                forecast_residual_caps=_batch_value(batch, "forecast_residual_caps"),
                future_forecast_anchors=_batch_value(batch, "future_forecast_anchors"),
            )
        merged = _merge_training_aliases(output, batch)
        if self.spec.name in {"p1", "j1"}:
            merged = _augment_normalized_policy_cost(merged, batch)
        assert_scientific_precision(merged)
        return merged

    def _u0_formal_ensemble_output(
        self,
        batch: Any,
        *,
        base_seed: int,
        seed_offsets: Sequence[int],
    ) -> dict[str, Any]:
        """Generate one vectorized 100-path U0 ensemble before formal scoring.

        Training and sentinel monitoring retain the registered 20-path cost.
        A formal R5-V2.3 audit supplies five distinct seed offsets to one model
        call.  The dynamics layer preserves five independent 20-path random
        streams while vectorizing the shared 96-step rollout.  Scoring chunks
        separately would omit cross-chunk pair terms in fair CRPS and Energy
        Score, so that shortcut is deliberately rejected here.
        """

        offsets = tuple(int(value) for value in seed_offsets)
        if not offsets or len(offsets) != len(set(offsets)):
            raise WorldTrainingError(
                "formal U0 seed offsets must be nonempty and unique"
            )
        aggregate_paths = (
            len(offsets) * int(self.spec.sample_count)
            if self.spec.formal_sample_count is None
            else int(self.spec.formal_sample_count)
        )
        if aggregate_paths != 100:
            raise WorldTrainingError(
                "formal U0 validation must concatenate exactly 100 paths"
            )
        output = self.forward(
            batch,
            seed=int(base_seed),
            formal_seed_offsets=offsets,
        )
        if not isinstance(output, Mapping):
            raise WorldTrainingError("formal U0 model output must be a mapping")
        scenarios = output.get("scenarios")
        if not isinstance(scenarios, torch.Tensor):
            raise WorldTrainingError("formal U0 output omitted scenario paths")
        expected_shape = (
            int(scenarios.shape[0]),
            aggregate_paths,
            int(self.spec.horizon),
            21,
        )
        if tuple(scenarios.shape) != expected_shape:
            raise WorldTrainingError(
                "formal U0 rollout does not have the registered [B,100,96,21] shape"
            )
        scenarios = scenarios.float()
        if not bool(torch.isfinite(scenarios).all()):
            raise WorldTrainingError("formal U0 scenarios contain non-finite values")

        # Rebuild price paths from the same evaluated public tensor.  This keeps
        # the closure diagnostic on all 100 paths instead of accidentally
        # checking a separate quantile-head tensor from only one forward pass.
        da = scenarios[..., 0:4]
        rt = scenarios[..., 4:8]
        dart = scenarios[..., 8:12]
        identity_residual = (da - rt - dart).abs().amax()
        if float(identity_residual.detach().cpu()) > 1.0e-6:
            raise WorldTrainingError(
                "formal U0 paths violate deterministic DART = DA - RT"
            )

        combined = dict(output)
        combined["scenarios"] = scenarios
        combined["joint_price_paths"] = {"da": da, "rt": rt, "dart": dart}
        combined["distribution_mean"] = scenarios.mean(dim=1)
        minimum_scale = float(
            getattr(getattr(self.model, "config", None), "minimum_scale", 1.0e-4)
        )
        combined["distribution_log_scale"] = torch.log(
            scenarios.std(dim=1, unbiased=False).clamp_min(minimum_scale)
        )
        combined["distribution_scale_available"] = True
        combined["distribution_scale_source"] = "dynamics_residual_empirical"
        combined["formal_ensemble_paths"] = aggregate_paths
        combined["formal_ensemble_chunks"] = len(offsets)
        return combined

    def _risk_chunk_output(
        self,
        base_output: Mapping[str, Any],
        batch: Any,
        *,
        seed: int,
    ) -> dict[str, Any]:
        """Replay one 20-path risk chunk for the same state and chosen action."""

        plan = base_output.get("plan")
        state = base_output.get("state")
        actions = _member(plan, "actions")
        pooled_state = _member(state, "pooled_state")
        packet = _model_input(batch)
        constraints = _member(packet, "constraints")
        planner = _planner_mapping(base_output)
        commitment = planner.get("day_ahead_commitment")
        dynamics = getattr(self.model, "dynamics", None)
        if (
            actions is None
            or not isinstance(pooled_state, torch.Tensor)
            or constraints is None
            or not isinstance(commitment, torch.Tensor)
            or not callable(dynamics)
        ):
            raise WorldTrainingError(
                "five-chunk risk replay requires state, plan, DA commitment, and dynamics"
            )
        rollout = _call_supported(
            dynamics,
            pooled_state,
            actions,
            constraints,
            path_count=self.spec.sample_count,
            seed=int(seed),
            day_ahead_commitment=commitment,
            deadline_intervals=int(
                _batch_value(batch, "deadline_intervals").reshape(-1)[0].item()
                if isinstance(_batch_value(batch, "deadline_intervals"), torch.Tensor)
                else 4
            ),
        )
        result = _augment_normalized_policy_cost(base_output, batch, rollout=rollout)
        result["scenarios"] = _member(_member(rollout, "observations"), "public_21")
        assert_scientific_precision(result)
        return result

    def _policy_loss(
        self,
        output: Any,
        batch: Any,
        *,
        include_route_supervision: bool = True,
    ) -> PhaseLoss:
        return p1_latent_planner_loss(
            output,
            batch,
            weights=self.weights,
            cvar_alpha=self.spec.cvar_alpha,
            minimum_cvar_paths=self.spec.sample_count,
            include_route_supervision=include_route_supervision,
        )

    def _update_constraint_duals(self, output: Mapping[str, Any]) -> None:
        planner_output = _planner_mapping(output)
        violations = planner_output.get("raw_constraint_violations")
        planner_module = getattr(self.model, "planner", None)
        update = getattr(planner_module, "update_augmented_lagrangian", None)
        if not isinstance(violations, Mapping) or not callable(update):
            raise WorldTrainingError(
                "P1/J1 requires projected augmented-Lagrangian dual updates"
            )
        update(dict(violations))

    def _train_risk_step(self, batch: Any, *, seed: int | None) -> TrainerStepResult:
        """Train P1/J1 with 5x20 fixed-seed paths and only one chunk resident."""

        if self.optimizer is None:
            raise WorldTrainingError("risk phase has no optimizer")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        base_seed = 0 if seed is None else int(seed)
        base_output = self.forward(batch, seed=base_seed + RISK_CHUNK_SEED_OFFSETS[0])
        active = self.groups.active_parameters(self.spec.active_groups)
        task_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        components: dict[str, float] = {}

        if self.spec.name == "j1":
            # Resolve both non-policy tasks through the registered profile so
            # matched-Huber and proper training cannot silently diverge in this
            # risk-chunk fast path.  Route supervision belongs to the world
            # task once; uncertainty and the five policy chunks omit it.
            world = _call_supported(
                self.training_builder,
                "w1",
                base_output,
                batch,
                weights=self.weights,
                cvar_alpha=self.spec.cvar_alpha,
                include_route_supervision=True,
            )
            uncertainty = _call_supported(
                self.training_builder,
                "u0",
                base_output,
                batch,
                weights=self.weights,
                cvar_alpha=self.spec.cvar_alpha,
                include_route_supervision=False,
            )
            if not isinstance(world, PhaseLoss) or not isinstance(
                uncertainty, PhaseLoss
            ):
                raise WorldTrainingError(
                    "J1 world/uncertainty builder returned an invalid PhaseLoss"
                )
            task_gradients["world"] = _gradient_tuple(
                world.total, active, retain_graph=True
            )
            task_gradients["uncertainty"] = _gradient_tuple(
                uncertainty.total, active, retain_graph=True
            )
            components.update(
                {f"world/{key}": value for key, value in world.detached().items()}
            )
            components.update(
                {
                    f"uncertainty/{key}": value
                    for key, value in uncertainty.detached().items()
                }
            )

        policy_gradients: tuple[torch.Tensor | None, ...] | None = None
        policy_component_sums: dict[str, float] = {}
        policy_loss_sum = 0.0
        for index, offset in enumerate(RISK_CHUNK_SEED_OFFSETS):
            chunk_output = (
                base_output
                if index == 0
                else self._risk_chunk_output(
                    base_output,
                    batch,
                    seed=base_seed + int(offset),
                )
            )
            policy = self._policy_loss(
                chunk_output,
                batch,
                include_route_supervision=self.spec.name != "j1",
            )
            chunk_gradients = _gradient_tuple(
                policy.total / float(len(RISK_CHUNK_SEED_OFFSETS)),
                active,
                # Each risk chunk owns a separate rollout graph.  World and
                # uncertainty have already consumed the base graph before the
                # first policy call, so no policy graph is reused here.
                retain_graph=False,
            )
            policy_gradients = add_gradient_tuples_in_place(
                policy_gradients, chunk_gradients
            )
            del chunk_gradients
            detached = policy.detached()
            policy_loss_sum += float(policy.total.detach().float().cpu())
            for name, value in detached.items():
                policy_component_sums[name] = policy_component_sums.get(
                    name, 0.0
                ) + float(value)
            if index:
                del chunk_output
        if policy_gradients is None:
            raise WorldTrainingError("risk path chunking produced no policy gradients")

        if self.spec.name == "j1":
            task_gradients["policy"] = policy_gradients
            policy_gradients = None
            merged_gradients = pcgrad_project_flat(task_gradients, consume=True)
        else:
            merged_gradients = policy_gradients
        for parameter, gradient in zip(active, merged_gradients, strict=True):
            parameter.grad = None if gradient is None else gradient
        self.firewall.assert_after_backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            active, max_norm=self.spec.gradient_clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self._update_constraint_duals(base_output)
        self.firewall.assert_after_backward()

        divisor = float(len(RISK_CHUNK_SEED_OFFSETS))
        components.update(
            {
                f"policy/{name}" if self.spec.name == "j1" else name: value / divisor
                for name, value in policy_component_sums.items()
            }
        )
        if self.spec.name == "j1":
            components["pcgrad/task_count"] = 3.0
            components["pcgrad/path_chunks"] = divisor
        components["risk/aggregate_paths"] = float(
            len(RISK_CHUNK_SEED_OFFSETS) * self.spec.sample_count
        )
        return TrainerStepResult(
            phase=self.spec.name,
            loss=policy_loss_sum / divisor,
            components=components,
            gradient_norm=float(torch.as_tensor(gradient_norm).detach().float().cpu()),
            updated_groups=self.spec.active_groups,
        )

    def train_step(self, batch: Any, *, seed: int | None = None) -> TrainerStepResult:
        if self.optimizer is None or self.spec.name == "c1":
            raise WorldTrainingError("C1 has no train_step")
        if self.spec.name in {"p1", "j1"}:
            return self._train_risk_step(batch, seed=seed)
        self.model.train()
        self._apply_learning_rate_schedule()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.forward(batch, seed=seed)
        loss = _call_supported(
            self.training_builder,
            self.spec.name,
            output,
            batch,
            weights=self.weights,
            cvar_alpha=self.spec.cvar_alpha,
        )
        if not isinstance(loss, PhaseLoss):
            raise WorldTrainingError("loss builder returned an invalid PhaseLoss")
        active = self.groups.active_parameters(self.spec.active_groups)
        used_w0_pcgrad = False
        if self.spec.name == "w0" and self.spec.w0_pcgrad:
            general = loss.components.get("general_objective")
            rare = loss.components.get("rare_objective")
            if not isinstance(general, torch.Tensor) or not isinstance(
                rare, torch.Tensor
            ):
                raise WorldTrainingError(
                    "W0 PCGrad requires general_objective and rare_objective components"
                )
            gradients = {
                "general": _gradient_tuple(general, active, retain_graph=True),
                "rare": _gradient_tuple(rare, active, retain_graph=False),
            }
            merged = pcgrad_project(gradients)
            for parameter, gradient in zip(active, merged, strict=True):
                parameter.grad = gradient
            used_w0_pcgrad = True
        else:
            loss.total.backward()
        self.firewall.assert_after_backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            active, max_norm=self.spec.gradient_clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.firewall.assert_after_backward()
        logged_scalars = loss.detached(
            extra={
                "__trainer_loss": loss.total,
                "__trainer_gradient_norm": torch.as_tensor(gradient_norm),
            }
        )
        loss_value = logged_scalars.pop("__trainer_loss")
        gradient_norm_value = logged_scalars.pop("__trainer_gradient_norm")
        components = logged_scalars
        if used_w0_pcgrad:
            components["pcgrad/task_count"] = 2.0
        for group in self.optimizer.param_groups:
            role = str(group.get("rate_role", "uniform"))
            components[f"learning_rate/{role}"] = float(group["lr"])
        return TrainerStepResult(
            phase=self.spec.name,
            loss=loss_value,
            components=components,
            gradient_norm=gradient_norm_value,
            updated_groups=self.spec.active_groups,
        )

    @torch.no_grad()
    def evaluate_loss_and_diagnostics(
        self,
        batch: Any,
        *,
        seed: int | None = None,
        formal_seed_offsets: Sequence[int] | None = None,
    ) -> tuple[PhaseLoss, Mapping[str, float]]:
        if self.spec.name == "c1":
            raise WorldTrainingError(
                "C1 evaluation is performed by the closed-loop gate"
            )
        self.model.eval()
        base_seed = 0 if seed is None else int(seed)
        output = (
            self._u0_formal_ensemble_output(
                batch,
                base_seed=base_seed,
                seed_offsets=formal_seed_offsets,
            )
            if self.spec.name == "u0" and formal_seed_offsets is not None
            else self.forward(batch, seed=base_seed)
        )
        if self.spec.name in {"p1", "j1"}:
            normalized_chunks = [_planner_mapping(output)["normalized_scenario_costs"]]
            raw_chunks = [_planner_mapping(output)["scenario_costs"]]
            for offset in RISK_CHUNK_SEED_OFFSETS[1:]:
                chunk = self._risk_chunk_output(
                    output, batch, seed=base_seed + int(offset)
                )
                chunk_planner = _planner_mapping(chunk)
                normalized_chunks.append(chunk_planner["normalized_scenario_costs"])
                raw_chunks.append(chunk_planner["scenario_costs"])
            planner = _planner_mapping(output)
            planner["normalized_scenario_costs"] = torch.stack(normalized_chunks, dim=1)
            planner["scenario_costs"] = torch.stack(raw_chunks, dim=1)
            output = dict(output)
            output["planner"] = planner
        result = _call_supported(
            self.selection_builder,
            self.spec.name,
            output,
            batch,
            weights=self.weights,
            cvar_alpha=self.spec.cvar_alpha,
        )
        if not isinstance(result, PhaseLoss):
            raise WorldTrainingError("loss builder returned an invalid PhaseLoss")
        diagnostics = validation_diagnostics(output, batch)
        if self.spec.name == "u0" and formal_seed_offsets is not None:
            diagnostics = dict(diagnostics)
            diagnostics["formal_ensemble_paths"] = float(
                len(tuple(formal_seed_offsets)) * self.spec.sample_count
                if self.spec.formal_sample_count is None
                else self.spec.formal_sample_count
            )
            diagnostics["formal_ensemble_chunks"] = float(
                len(tuple(formal_seed_offsets))
            )
        return result, diagnostics

    @torch.no_grad()
    def evaluate_loss(self, batch: Any, *, seed: int | None = None) -> PhaseLoss:
        result, _ = self.evaluate_loss_and_diagnostics(batch, seed=seed)
        return result


def _rng_state(*, capture_cuda: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if capture_cuda:
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    def cpu_byte_tensor(value: Any, *, field: str) -> torch.Tensor:
        try:
            tensor = (
                value.detach()
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value)
            )
            return tensor.to(device="cpu", dtype=torch.uint8).contiguous()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise WorldTrainingError(
                f"checkpoint RNG field {field!r} cannot be restored as a CPU ByteTensor"
            ) from exc

    torch_cpu = cpu_byte_tensor(state["torch_cpu"], field="torch_cpu")
    raw_cuda = state.get("torch_cuda")
    if isinstance(raw_cuda, torch.Tensor):
        raw_cuda_states: tuple[Any, ...] = (raw_cuda,)
    elif raw_cuda is None:
        raw_cuda_states = ()
    else:
        try:
            raw_cuda_states = tuple(raw_cuda)
        except TypeError as exc:
            raise WorldTrainingError(
                "checkpoint RNG field 'torch_cuda' must be a tensor sequence"
            ) from exc
    torch_cuda = tuple(
        cpu_byte_tensor(value, field=f"torch_cuda[{index}]")
        for index, value in enumerate(raw_cuda_states)
    )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch_cpu)
    if torch_cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda)


class AtomicCheckpointStore:
    """Atomic, same-phase resume checkpoints with RNG continuity."""

    def __init__(self, root: str | Path, *, run_id: str) -> None:
        self.root = Path(root).resolve()
        self.run_id = str(run_id).strip()
        if not self.run_id:
            raise WorldTrainingError("run_id must be nonempty")

    def path(self, phase: str, role: str = "latest") -> Path:
        normalized = str(phase).lower()
        if normalized not in PHASE_ORDER:
            raise WorldTrainingError(f"unknown checkpoint phase {phase!r}")
        return self.root / f"phase={normalized}" / "checkpoints" / f"{role}.pt"

    def save(
        self,
        *,
        state: ResumeState,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        role: str = "latest",
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.path(state.phase, role)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "run_id": self.run_id,
            "phase": state.phase,
            "resume_state": asdict(state),
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "rng_state": _rng_state(
                capture_cuda=any(parameter.is_cuda for parameter in model.parameters())
            ),
            "locked_2025_opened": False,
            "extra": dict(extra or {}),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path

    def load(
        self,
        phase: str,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        role: str = "latest",
        map_location: str | torch.device = "cpu",
    ) -> ResumeState:
        path = self.path(phase, role)
        if not path.is_file():
            raise WorldTrainingError(f"checkpoint is missing: {path}")
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping):
            raise WorldTrainingError("checkpoint payload is malformed")
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise WorldTrainingError("checkpoint schema mismatch")
        if payload.get("run_id") != self.run_id:
            raise WorldTrainingError("checkpoint belongs to another run")
        if str(payload.get("phase", "")).lower() != str(phase).lower():
            raise WorldTrainingError("cross-phase resume is forbidden")
        if payload.get("locked_2025_opened") is not False:
            raise WorldTrainingError("checkpoint does not preserve the 2025 lock")
        model.load_state_dict(payload["model_state"])
        raw_optimizer = payload.get("optimizer_state")
        if optimizer is not None:
            if not isinstance(raw_optimizer, Mapping):
                raise WorldTrainingError("checkpoint has no optimizer state")
            optimizer.load_state_dict(raw_optimizer)
        raw_state = payload.get("resume_state")
        if not isinstance(raw_state, Mapping):
            raise WorldTrainingError("checkpoint has no resume state")
        rng = payload.get("rng_state")
        if not isinstance(rng, Mapping):
            raise WorldTrainingError("checkpoint has no RNG state")
        _restore_rng_state(rng)
        state_values = dict(raw_state)
        if (
            "stale_epochs" in state_values
            and "formal_stale_short_epochs" not in state_values
        ):
            state_values["formal_stale_short_epochs"] = state_values.pop("stale_epochs")
        return ResumeState(**state_values)
