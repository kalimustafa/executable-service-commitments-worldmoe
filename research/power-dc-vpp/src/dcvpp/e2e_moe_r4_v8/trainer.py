"""Executable training foundation for the V8 TriFusion network."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .config import PARAMETER_GROUPS, PHASE_ORDER
from .gradient_router import (
    ResolvedParameterGroups,
    assign_gradients,
    resolve_parameter_groups,
    route_shared_gradients,
)
from .losses import (
    TaskLoss,
    distribution_loss,
    packet_mapping,
    policy_loss,
    representation_loss,
)


TaskName = Literal["representation", "distribution", "policy"]
DEFAULT_TRAIN_DAYS = 722
DEFAULT_UPDATES = 4


class V8TrainingError(RuntimeError):
    """The unified trainer cannot preserve its task/gradient contract."""


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    minimum_cycles: int
    maximum_cycles: int
    primary_pattern: tuple[TaskName, ...]
    active_tasks: tuple[TaskName, ...]
    shared_updates: bool
    shared_primary: TaskName | None = None
    shared_constraints: tuple[TaskName, ...] = ()
    horizons: tuple[int, ...] = (96,)
    router_updates: bool = True
    trainable_subscope: tuple[str, ...] = ()

    def primary_at(self, optimizer_update_index: int) -> TaskName:
        index = int(optimizer_update_index)
        if index < 0:
            raise V8TrainingError("optimizer update index must be nonnegative")
        return self.primary_pattern[index % len(self.primary_pattern)]

    def horizon_at(self, optimizer_update_index: int) -> int:
        index = int(optimizer_update_index)
        if index < 0 or not self.horizons:
            raise V8TrainingError("invalid Policy horizon schedule")
        return int(self.horizons[index % len(self.horizons)])


@dataclass(frozen=True)
class V8TrainingContract:
    stages: tuple[StageSpec, ...]
    train_days: int = DEFAULT_TRAIN_DAYS
    updates_per_cycle: int = DEFAULT_UPDATES

    def stage(self, stage_id: str) -> StageSpec:
        normalized = str(stage_id).lower()
        for stage in self.stages:
            if stage.stage_id == normalized:
                return stage
        raise V8TrainingError(f"unknown V8 stage: {stage_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dcvpp.r4.v8-trifusion-training-contract.v1",
            "train_days": self.train_days,
            "updates_per_cycle": self.updates_per_cycle,
            "update_day_counts": [
                item.stop - item.start
                for item in coverage_slices(self.train_days, self.updates_per_cycle)
            ],
            "stages": [asdict(stage) for stage in self.stages],
            "policy_truth_gradient": "forbidden",
        }


def coverage_slices(
    day_count: int = DEFAULT_TRAIN_DAYS,
    updates: int = DEFAULT_UPDATES,
) -> tuple[slice, ...]:
    count, update_count = int(day_count), int(updates)
    if count <= 0 or update_count <= 0 or update_count > count:
        raise V8TrainingError("invalid full-coverage partition")
    quotient, remainder = divmod(count, update_count)
    start = 0
    output: list[slice] = []
    for index in range(update_count):
        size = quotient + (1 if index < remainder else 0)
        output.append(slice(start, start + size))
        start += size
    return tuple(output)


def _index_value(value: Any, indices: Tensor, day_count: int) -> Any:
    if isinstance(value, Tensor):
        return (
            value.index_select(0, indices.to(value.device))
            if value.ndim and value.shape[0] == day_count
            else value
        )
    if isinstance(value, Mapping):
        return {
            key: (
                item
                if key in {"channel_scale", "representation_scales", "base_indices"}
                and isinstance(item, Tensor)
                and item.ndim == 1
                else _index_value(item, indices, day_count)
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple) and len(value) == day_count:
        return tuple(value[int(index)] for index in indices.tolist())
    if isinstance(value, list) and len(value) == day_count:
        return [value[int(index)] for index in indices.tolist()]
    return value


def _batch_day_count(batch: Mapping[str, Any]) -> int:
    for key in (
        "distribution_target",
        "scenario_target_zonal",
        "operating_dates",
        "operating_date",
    ):
        value = batch.get(key)
        if isinstance(value, Tensor) and value.ndim:
            return int(value.shape[0])
        if isinstance(value, (tuple, list)):
            return len(value)
    targets = batch.get("representation_targets")
    if isinstance(targets, Mapping):
        for value in targets.values():
            if isinstance(value, Tensor) and value.ndim:
                return int(value.shape[0])
    raise V8TrainingError("cannot infer the complete batch day axis")


def coverage_batches(
    batch: Mapping[str, Any],
    *,
    cycle_index: int,
    expected_days: int | None = None,
    updates: int = DEFAULT_UPDATES,
    seed: int = 20240813,
    shuffle: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    days = _batch_day_count(batch)
    if expected_days is not None and days != int(expected_days):
        raise V8TrainingError(f"coverage expected {int(expected_days)} days, got {days}")
    cycle = int(cycle_index)
    if cycle < 0:
        raise V8TrainingError("cycle index must be nonnegative")
    order = torch.arange(days)
    if shuffle:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1_000_003 * cycle
        )
        order = order[torch.randperm(days, generator=generator)]
    return tuple(
        _index_value(batch, order[part], days)
        for part in coverage_slices(days, int(updates))
    )


def contract_from_config(config: Mapping[str, Any]) -> V8TrainingContract:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise V8TrainingError("training must be a mapping")
    cycle = training.get("coverage_cycle")
    if not isinstance(cycle, Mapping):
        raise V8TrainingError("training.coverage_cycle must be a mapping")
    days = int(cycle.get("days", -1))
    updates = int(cycle.get("optimizer_updates", -1))
    expected = [item.stop - item.start for item in coverage_slices(days, updates)]
    if list(cycle.get("update_day_counts", ())) != expected:
        raise V8TrainingError(f"update_day_counts must be {expected}")
    raw_phases = training.get("phases")
    if not isinstance(raw_phases, Mapping) or tuple(raw_phases) != PHASE_ORDER:
        raise V8TrainingError(f"training phases must be {list(PHASE_ORDER)}")
    stages: list[StageSpec] = []
    valid = {"representation", "distribution", "policy"}
    for stage_id in PHASE_ORDER:
        raw = raw_phases[stage_id]
        if not isinstance(raw, Mapping):
            raise V8TrainingError(f"phase {stage_id} must be a mapping")
        pattern = tuple(str(value) for value in raw.get("primary_pattern", ()))
        active = tuple(str(value) for value in raw.get("active_tasks", ()))
        if not pattern or not active or set(pattern) - valid or set(active) - valid:
            raise V8TrainingError(f"phase {stage_id} has invalid task declarations")
        if set(pattern) - set(active):
            raise V8TrainingError(f"phase {stage_id} primary task is inactive")
        shared_updates = bool(raw.get("shared_updates", False))
        shared_primary = raw.get("shared_primary")
        if shared_updates and shared_primary is None:
            shared_primary = pattern[0]
        constraints = tuple(str(value) for value in raw.get("shared_constraints", ()))
        if shared_primary is not None and str(shared_primary) not in active:
            raise V8TrainingError(f"phase {stage_id} shared primary is inactive")
        if set(constraints) - set(active) or str(shared_primary) in constraints:
            raise V8TrainingError(f"phase {stage_id} shared constraints are invalid")
        minimum = int(raw.get("min_cycles", -1))
        maximum = int(raw.get("max_cycles", -1))
        if not 1 <= minimum <= maximum:
            raise V8TrainingError(f"phase {stage_id} cycle bounds are invalid")
        horizons = tuple(int(value) for value in raw.get("horizons", (96,)))
        if not horizons or any(value not in {8, 24, 48, 96} for value in horizons):
            raise V8TrainingError(f"phase {stage_id} has invalid Policy horizons")
        router_updates = raw.get("router_updates", True)
        if not isinstance(router_updates, bool):
            raise V8TrainingError(f"phase {stage_id} router_updates must be boolean")
        raw_subscope = raw.get("trainable_subscope", ())
        if isinstance(raw_subscope, (str, bytes)) or not isinstance(
            raw_subscope, Sequence
        ):
            raise V8TrainingError(
                f"phase {stage_id} trainable_subscope must be a prefix sequence"
            )
        trainable_subscope = tuple(str(value).strip() for value in raw_subscope)
        if any(not value for value in trainable_subscope) or len(
            set(trainable_subscope)
        ) != len(trainable_subscope):
            raise V8TrainingError(
                f"phase {stage_id} trainable_subscope prefixes must be nonempty and unique"
            )
        stages.append(
            StageSpec(
                stage_id=stage_id,
                minimum_cycles=minimum,
                maximum_cycles=maximum,
                primary_pattern=pattern,  # type: ignore[arg-type]
                active_tasks=active,  # type: ignore[arg-type]
                shared_updates=shared_updates,
                shared_primary=(
                    None if shared_primary is None else str(shared_primary)
                ),  # type: ignore[arg-type]
                shared_constraints=constraints,  # type: ignore[arg-type]
                horizons=horizons,
                router_updates=router_updates,
                trainable_subscope=trainable_subscope,
            )
        )
    return V8TrainingContract(tuple(stages), days, updates)


def _call_supported(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(callback)
    accepts_kwargs = any(
        value.kind == inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    )
    selected = kwargs if accepts_kwargs else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return callback(*args, **selected)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _detach_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _detach_value(getattr(value, field.name))
                for field in fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(_detach_value(item) for item in value)
    if isinstance(value, list):
        return [_detach_value(item) for item in value]
    return value


def _decision_limits(batch: Mapping[str, Any]) -> Mapping[str, Tensor] | None:
    explicit = batch.get("limits")
    if isinstance(explicit, Mapping):
        return explicit  # type: ignore[return-value]
    targets = batch.get("rollout_targets")
    if not isinstance(targets, Mapping):
        return None
    aliases = {
        "grid_mw": "grid_limit_mw",
        "battery_mw": "battery_power_mw",
        "compute_work": "capacity_work_units",
    }
    limits = {output: targets.get(source) for output, source in aliases.items()}
    return (
        limits  # type: ignore[return-value]
        if all(isinstance(value, Tensor) for value in limits.values())
        else None
    )


def _add_gradients(
    accumulated: tuple[Tensor | None, ...] | None,
    incoming: tuple[Tensor | None, ...],
) -> tuple[Tensor | None, ...]:
    if accumulated is None:
        return tuple(
            None if value is None else value.detach().clone() for value in incoming
        )
    if len(accumulated) != len(incoming):
        raise V8TrainingError("microbatch gradient widths changed")
    output: list[Tensor | None] = []
    for current, value in zip(accumulated, incoming, strict=True):
        if current is None:
            output.append(None if value is None else value.detach().clone())
        elif value is None:
            output.append(current)
        else:
            output.append(current + value.detach().to(current))
    return tuple(output)


def _group_learning_rates(config: Mapping[str, Any]) -> Mapping[str, float]:
    training = config["training"]
    rates = training.get("learning_rates") if isinstance(training, Mapping) else None
    if not isinstance(rates, Mapping):
        raise V8TrainingError("training.learning_rates must be a mapping")
    return {name: float(rates[name]) for name in PARAMETER_GROUPS if name != "frozen"}


def build_optimizers(
    model: nn.Module,
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.optim.Optimizer], ResolvedParameterGroups]:
    groups = resolve_parameter_groups(model)
    rates = _group_learning_rates(config)
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    output = {
        name: torch.optim.AdamW(
            groups[name], lr=rates[name], weight_decay=weight_decay
        )
        for name in PARAMETER_GROUPS
        if name != "frozen"
    }
    return output, groups


@dataclass(frozen=True)
class UpdateResult:
    stage: str
    primary_task: str
    losses: Mapping[str, float]
    shared_conflicts: tuple[str, ...]
    router_updated: bool = True
    trainable_subscope: tuple[str, ...] = ()
    task_parameter_names: tuple[str, ...] = ()


class V8TriFusionTrainer:
    """Task-specific optimizers plus projected shared/router updates."""

    TASK_GROUP = {
        "representation": "representation",
        "distribution": "distribution_truth",
        "policy": "decision",
    }

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        *,
        optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
        groups: ResolvedParameterGroups | None = None,
        rollout_fn: Callable[..., Mapping[str, Any]] | None = None,
        origin_budget: int | None = 8,
        microbatch_days: int | Mapping[str, int] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.contract = contract_from_config(config)
        if optimizers is None or groups is None:
            built, resolved = build_optimizers(model, config)
            self.optimizers = built
            self.groups = resolved
        else:
            self.optimizers = dict(optimizers)
            self.groups = groups
        expected = set(PARAMETER_GROUPS) - {"frozen"}
        if set(self.optimizers) != expected:
            raise V8TrainingError(f"optimizers must cover {sorted(expected)}")
        training = config["training"]
        routing = training["gradient_router"]
        self.projection_passes = int(routing.get("passes", 2))
        self.projection_epsilon = float(routing.get("epsilon", 1.0e-12))
        self.firewall_tolerance = float(routing.get("policy_truth_tolerance", 0.0))
        self.gradient_clip = float(training.get("gradient_clip_norm", 1.0))
        self.use_bf16 = training["precision"].get("neural_forward") == "bf16"
        self.rollout_fn = rollout_fn
        self.origin_budget = None if origin_budget is None else int(origin_budget)
        if self.origin_budget is not None and self.origin_budget < 1:
            raise V8TrainingError("origin_budget must be positive or None")
        default_microbatches = {
            "representation": 181,
            "distribution": 4,
            "policy": 1,
            "joint": 1,
        }
        configured_microbatch = training.get(
            "microbatch_days", default_microbatches
        )
        selected_microbatch = (
            configured_microbatch if microbatch_days is None else microbatch_days
        )
        if isinstance(selected_microbatch, Mapping):
            if set(selected_microbatch) != set(default_microbatches):
                raise V8TrainingError(
                    "microbatch_days mapping must define representation/distribution/policy/joint"
                )
            self.microbatch_days = {
                name: int(selected_microbatch[name]) for name in default_microbatches
            }
        else:
            size = int(selected_microbatch)
            self.microbatch_days = {name: size for name in default_microbatches}
        if any(value < 1 for value in self.microbatch_days.values()):
            raise V8TrainingError("microbatch_days values must be positive")
        sampling = config.get("sampling", {})
        if not isinstance(sampling, Mapping):
            raise V8TrainingError("sampling must be a mapping")
        self.train_sample_count = int(sampling.get("train", 64))
        self.d0_warm_sample_count = int(
            sampling.get("d0_warmup", self.train_sample_count)
        )
        self.calibration_sample_count = int(
            sampling.get("calibration", self.train_sample_count)
        )
        if min(
            self.train_sample_count,
            self.d0_warm_sample_count,
            self.calibration_sample_count,
        ) < 2:
            raise V8TrainingError("training sample counts must be at least two")
        protocol = config.get("protocol", {})
        self.base_seed = int(
            protocol.get("active_seed", 20240813)
            if isinstance(protocol, Mapping)
            else 20240813
        )

    def _sample_count(self, stage: StageSpec, optimizer_update_index: int) -> int:
        if stage.stage_id == "c0":
            return self.calibration_sample_count
        warmup_updates = stage.minimum_cycles * self.contract.updates_per_cycle
        if stage.stage_id == "d0" and int(optimizer_update_index) < warmup_updates:
            return self.d0_warm_sample_count
        return self.train_sample_count

    def _task_subscope(
        self, stage: StageSpec, group_name: str
    ) -> tuple[tuple[nn.Parameter, ...], tuple[str, ...]]:
        parameters = self.groups[group_name]
        names = self.groups.names[group_name]
        if not stage.trainable_subscope:
            return parameters, names
        matched_by_prefix = {
            prefix: tuple(
                index for index, name in enumerate(names) if name.startswith(prefix)
            )
            for prefix in stage.trainable_subscope
        }
        missing = [
            prefix for prefix, indices in matched_by_prefix.items() if not indices
        ]
        if missing:
            raise V8TrainingError(
                f"phase {stage.stage_id} trainable_subscope matched no "
                f"{group_name} parameters for prefixes {missing}"
            )
        selected_indices = {
            index for indices in matched_by_prefix.values() for index in indices
        }
        if not selected_indices:
            raise V8TrainingError(
                f"phase {stage.stage_id} trainable_subscope selected no parameters"
            )
        ordered = tuple(sorted(selected_indices))
        return (
            tuple(parameters[index] for index in ordered),
            tuple(names[index] for index in ordered),
        )

    def _forward(
        self,
        batch: Mapping[str, Any],
        stage: StageSpec,
        *,
        optimizer_update_index: int,
        seed: int,
    ) -> Mapping[str, Any]:
        joint = len(stage.active_tasks) > 1
        callback = getattr(
            self.model,
            "forward_joint" if joint else f"forward_{stage.active_tasks[0]}",
            None,
        )
        if not callable(callback):
            raise V8TrainingError("model is missing a registered V8 forward method")
        device = next(self.model.parameters()).device
        enabled = self.use_bf16 and device.type == "cuda"
        sample_count = self._sample_count(stage, optimizer_update_index)
        horizon = stage.horizon_at(optimizer_update_index)
        limits = _decision_limits(batch)
        common = {
            "sample_count": sample_count,
            "base_indices": batch.get("base_indices"),
            "seed": int(seed),
            "origin": batch.get("origin", 0),
            "limits": limits,
            "issued_day_ahead": batch.get("issued_day_ahead"),
            "detach_distribution_truth": True,
            "detach_shared": not stage.shared_updates,
            "scientific_score_dtype": torch.float32,
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        ):
            if stage.active_tasks == ("policy",):
                distribution_callback = getattr(
                    self.model, "forward_distribution", None
                )
                if not callable(distribution_callback):
                    raise V8TrainingError(
                        "P0 requires forward_distribution for detached worlds"
                    )
                with torch.no_grad():
                    distribution_output = packet_mapping(
                        _call_supported(distribution_callback, batch, **common),
                        "detached P0 Distribution output",
                    )
                policy_output = packet_mapping(
                    _call_supported(
                        callback,
                        batch,
                        planes=distribution_output,
                        **common,
                    ),
                    "model Policy output",
                )
                output = {**distribution_output, **policy_output}
            else:
                output = packet_mapping(
                    _call_supported(callback, batch, **common),
                    "model forward output",
                )
                distribution_output = output
        if "policy" in stage.active_tasks and self.rollout_fn is not None:
            rollout = _call_supported(
                self.rollout_fn,
                self.model,
                batch,
                _detach_value(distribution_output),
                stage_id=stage.stage_id,
                horizon=horizon,
                origin_budget=self.origin_budget,
                sample_count=sample_count,
                seed=int(seed),
            )
            rollout_view = packet_mapping(rollout, "rollout_fn output")
            if "rollout" in rollout_view:
                rollout_view = packet_mapping(
                    rollout_view["rollout"], "rollout_fn output.rollout"
                )
            output = {**output, "rollout": rollout_view}
        return output

    @staticmethod
    def _task_packet(output: Mapping[str, Any], task: str) -> Any:
        # Loss adapters need sibling fields: Distribution samples/baseline and
        # Policy actions/critic/rollout live beside their typed plane packet.
        if task in {"representation", "distribution", "policy"}:
            return output
        return output

    def _losses(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        stage: StageSpec,
    ) -> dict[str, TaskLoss]:
        configured = self.config.get("losses", {})
        losses: dict[str, TaskLoss] = {}
        if "representation" in stage.active_tasks:
            losses["representation"] = representation_loss(
                self._task_packet(output, "representation"),
                batch,
                weights=configured.get("representation"),
            )
        if "distribution" in stage.active_tasks:
            guard = float(self.config["training"].get("secondary_noninferiority_guard", 0.005))
            losses["distribution"] = distribution_loss(
                self._task_packet(output, "distribution"),
                batch,
                secondary_guard=guard,
            )
        if "policy" in stage.active_tasks:
            policy_weights = dict(configured.get("policy", {}))
            if stage.stage_id == "p0":
                policy_weights.setdefault("anchor_imitation", 1.0)
            losses["policy"] = policy_loss(
                self._task_packet(output, "policy"),
                batch,
                weights=policy_weights,
            )
        return losses

    def _assert_policy_firewall(self, loss: Tensor) -> None:
        truth = self.groups["distribution_truth"]
        gradients = torch.autograd.grad(
            loss,
            truth,
            retain_graph=True,
            allow_unused=True,
        )
        leaked = [
            name
            for name, gradient in zip(
                self.groups.names["distribution_truth"], gradients, strict=True
            )
            if gradient is not None
            and bool((gradient.detach().abs() > self.firewall_tolerance).any())
        ]
        if leaked:
            raise V8TrainingError(
                "Policy gradient reached Distribution truth heads: " + ", ".join(leaked)
            )

    @staticmethod
    def _gradients(loss: Tensor, parameters: Sequence[nn.Parameter]) -> tuple[Tensor | None, ...]:
        return tuple(
            torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
        )

    def update(
        self,
        batch: Mapping[str, Any],
        *,
        stage_id: str,
        optimizer_update_index: int,
        seed: int | None = None,
    ) -> UpdateResult:
        stage = self.contract.stage(stage_id)
        primary_task = stage.primary_at(optimizer_update_index)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        task_group_name = self.TASK_GROUP[primary_task]
        task_parameters, task_parameter_names = self._task_subscope(
            stage, task_group_name
        )
        days = _batch_day_count(batch)
        microbatch_kind = (
            stage.active_tasks[0] if len(stage.active_tasks) == 1 else "joint"
        )
        microbatch_size = self.microbatch_days[microbatch_kind]
        resolved_seed = (
            self.base_seed + int(optimizer_update_index)
            if seed is None
            else int(seed)
        )
        task_gradients: tuple[Tensor | None, ...] | None = None
        router_gradients: tuple[Tensor | None, ...] | None = None
        routed_group_names = (
            tuple(
                name
                for name in ("shared", "router")
                if name != "router" or stage.router_updates
            )
            if stage.shared_updates
            else ()
        )
        shared_by_group: dict[
            str, dict[str, tuple[Tensor | None, ...] | None]
        ] = {
            group_name: {task: None for task in stage.active_tasks}
            for group_name in routed_group_names
        }
        aggregate_losses = {task: 0.0 for task in stage.active_tasks}
        aggregate_violations = {task: 0.0 for task in stage.active_tasks}
        for start in range(0, days, microbatch_size):
            stop = min(days, start + microbatch_size)
            indices = torch.arange(start, stop, dtype=torch.long)
            microbatch = _index_value(batch, indices, days)
            microbatch = _move_to_device(
                microbatch, next(self.model.parameters()).device
            )
            fraction = float(stop - start) / float(days)
            output = self._forward(
                microbatch,
                stage,
                optimizer_update_index=optimizer_update_index,
                seed=resolved_seed,
            )
            losses = self._losses(output, microbatch, stage)
            if "policy" in losses:
                self._assert_policy_firewall(losses["policy"].total)
            for task, task_loss in losses.items():
                aggregate_losses[task] += fraction * float(
                    task_loss.total.detach().float().cpu()
                )
                aggregate_violations[task] = max(
                    aggregate_violations[task], float(task_loss.guard_violation)
                )

            weighted_primary = losses[primary_task].total * fraction
            task_gradients = _add_gradients(
                task_gradients,
                self._gradients(weighted_primary, task_parameters),
            )
            if stage.shared_updates:
                for group_name in routed_group_names:
                    parameters = self.groups[group_name]
                    for task in stage.active_tasks:
                        incoming = self._gradients(
                            losses[task].total * fraction, parameters
                        )
                        shared_by_group[group_name][task] = _add_gradients(
                            shared_by_group[group_name][task], incoming
                        )
            elif stage.router_updates:
                router_gradients = _add_gradients(
                    router_gradients,
                    self._gradients(
                        weighted_primary, self.groups["router"]
                    ),
                )

        if task_gradients is None or not any(
            value is not None for value in task_gradients
        ):
            raise V8TrainingError(f"{primary_task} loss has no task-specific gradient")

        shared_routes: list[tuple[str, tuple[Tensor | None, ...]]] = []
        routed_conflicts: list[str] = []
        if stage.shared_updates:
            shared_primary = stage.shared_primary or primary_task
            for group_name in routed_group_names:
                gradients = shared_by_group[group_name]
                if any(value is None for value in gradients.values()):
                    raise V8TrainingError(
                        f"incomplete {group_name} microbatch gradient aggregation"
                    )
                route = route_shared_gradients(
                    gradients,  # type: ignore[arg-type]
                    primary=shared_primary,
                    constraints=stage.shared_constraints,
                    violations={
                        task: aggregate_violations[task]
                        for task in stage.shared_constraints
                    },
                    passes=self.projection_passes,
                    epsilon=self.projection_epsilon,
                )
                shared_routes.append((group_name, route.gradients))
                routed_conflicts.extend(route.projected_conflicts)
        elif stage.router_updates:
            if router_gradients is None or not any(
                value is not None for value in router_gradients
            ):
                raise V8TrainingError(
                    f"{primary_task} loss has no task Router gradient"
                )
            shared_routes.append(("router", router_gradients))

        assign_gradients(task_parameters, task_gradients)
        torch.nn.utils.clip_grad_norm_(task_parameters, self.gradient_clip)
        for group_name, gradients in shared_routes:
            assign_gradients(self.groups[group_name], gradients)
            torch.nn.utils.clip_grad_norm_(self.groups[group_name], self.gradient_clip)

        self.optimizers[task_group_name].step()
        for group_name, _ in shared_routes:
            self.optimizers[group_name].step()
        return UpdateResult(
            stage.stage_id,
            primary_task,
            aggregate_losses,
            tuple(dict.fromkeys(routed_conflicts)),
            router_updated=stage.router_updates,
            trainable_subscope=stage.trainable_subscope,
            task_parameter_names=task_parameter_names,
        )

    def run_coverage_cycle(
        self,
        batch: Mapping[str, Any],
        *,
        stage_id: str,
        cycle_index: int,
        seed: int = 20240813,
        shuffle: bool = True,
    ) -> tuple[UpdateResult, ...]:
        pieces = coverage_batches(
            batch,
            cycle_index=cycle_index,
            expected_days=self.contract.train_days,
            updates=self.contract.updates_per_cycle,
            seed=seed,
            shuffle=shuffle,
        )
        offset = int(cycle_index) * self.contract.updates_per_cycle
        return tuple(
            self.update(
                piece,
                stage_id=stage_id,
                optimizer_update_index=offset + index,
                seed=int(seed) + offset + index,
            )
            for index, piece in enumerate(pieces)
        )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    stage: str
    cycle: int
    feasibility: Mapping[str, bool]
    objectives: Mapping[str, float]
    checkpoint: str | None = None


class FeasibleCandidateRegistry:
    """Retain only same-checkpoint R/D/P-feasible candidates."""

    REQUIRED = ("representation", "distribution", "policy", "structural")

    def __init__(self, *, top_k: int = 5) -> None:
        if int(top_k) <= 0:
            raise V8TrainingError("candidate top_k must be positive")
        self.top_k = int(top_k)
        self._items: list[Candidate] = []

    def add(self, candidate: Candidate) -> bool:
        if set(candidate.feasibility) != set(self.REQUIRED):
            raise V8TrainingError("candidate feasibility schema is incomplete")
        if not all(candidate.feasibility.values()):
            return False
        if not candidate.objectives or not all(
            math.isfinite(float(value)) for value in candidate.objectives.values()
        ):
            raise V8TrainingError("candidate objectives must be finite")
        if any(item.candidate_id == candidate.candidate_id for item in self._items):
            raise V8TrainingError("duplicate candidate id")
        self._items.append(candidate)
        order = (
            "policy_cost_upper",
            "distribution_energy_upper",
            "representation_worst_upper",
            "cvar",
        )
        self._items.sort(
            key=lambda item: tuple(float(item.objectives.get(name, math.inf)) for name in order)
            + (int(item.cycle), item.candidate_id)
        )
        self._items = self._items[: self.top_k]
        return True

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(self._items)


class FeasibleEarlyStopping:
    """Patience counts only checks where all three scientific guards pass."""

    def __init__(self, patience: int = 8, min_delta: float = 1.0e-4) -> None:
        if int(patience) < 1 or float(min_delta) < 0.0:
            raise V8TrainingError("invalid early-stopping settings")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.stale = 0

    def update(self, metric: float, *, feasible: bool) -> bool:
        if not feasible:
            self.stale += 1
        elif float(metric) < self.best - self.min_delta:
            self.best = float(metric)
            self.stale = 0
        else:
            self.stale += 1
        return self.stale >= self.patience


__all__ = [
    "Candidate",
    "DEFAULT_TRAIN_DAYS",
    "FeasibleCandidateRegistry",
    "FeasibleEarlyStopping",
    "StageSpec",
    "UpdateResult",
    "V8TrainingContract",
    "V8TrainingError",
    "V8TriFusionTrainer",
    "build_optimizers",
    "contract_from_config",
    "coverage_batches",
    "coverage_slices",
]
