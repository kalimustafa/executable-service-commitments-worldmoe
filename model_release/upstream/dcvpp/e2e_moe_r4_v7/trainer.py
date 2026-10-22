"""R4-V7 staged trainer and scientific gate contracts.

This module intentionally does not inherit the V6 trainer.  V7 uses four
optimizer updates per configured full-day coverage cycle and alternates two *separate*
objectives: proper scoring updates for the conditional world model and
decision updates for the action-conditioned policy/critic.  The latter always
receives a recursively detached distribution mapping.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from . import losses as v7_losses


UpdateKind = Literal["distribution", "decision"]
TRAIN_DAYS = 543
UPDATES_PER_COVERAGE_CYCLE = 4
PHASE_ORDER = (
    "d0",
    "p0",
    "j1",
    "j2",
    "j3",
    "calibration_restore",
    "a1",
    "a2",
)


class V7TrainingError(RuntimeError):
    """Raised when an update would weaken the approved training contract."""


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    minimum_cycles: int
    maximum_cycles: int
    distribution_updates: int
    decision_updates: int
    sample_count: int = 64
    warmup_sample_count: int | None = None
    warmup_optimizer_updates: int = 0
    decision_horizons: tuple[int, ...] = (96,)
    early_stopping: bool = True

    @property
    def update_pattern(self) -> tuple[UpdateKind, ...]:
        pattern: tuple[UpdateKind, ...] = (
            ("distribution",) * self.distribution_updates
            + ("decision",) * self.decision_updates
        )
        if not pattern:
            raise V7TrainingError(f"phase {self.phase_id} has no optimizer updates")
        return pattern

    @property
    def maximum_optimizer_updates(self) -> int:
        return self.maximum_cycles * UPDATES_PER_COVERAGE_CYCLE

    def sample_count_at_update(self, optimizer_update_index: int) -> int:
        index = int(optimizer_update_index)
        if index < 0:
            raise V7TrainingError("optimizer update index must be nonnegative")
        if (
            self.warmup_sample_count is not None
            and index < self.warmup_optimizer_updates
        ):
            return int(self.warmup_sample_count)
        return int(self.sample_count)

    def decision_horizon_at_update(self, optimizer_update_index: int) -> int:
        index = int(optimizer_update_index)
        if index < 0 or not self.decision_horizons:
            raise V7TrainingError("invalid decision-horizon schedule")
        return int(self.decision_horizons[index % len(self.decision_horizons)])


DEFAULT_PHASE_SPECS: tuple[PhaseSpec, ...] = (
    # 10--30 full-data cycles = 40--120 optimizer updates.
    PhaseSpec("d0", 10, 100, 1, 0, 64, 32, 40),
    # Ten full-data cycles = the approved 40-update policy warm-up.
    PhaseSpec("p0", 10, 10, 0, 1, decision_horizons=(8, 24, 48, 96)),
    PhaseSpec("j1", 10, 60, 2, 1),
    PhaseSpec("j2", 10, 80, 1, 1),
    PhaseSpec("j3", 10, 60, 1, 2),
    # Five cycles = 20 final proper-score-only updates.
    PhaseSpec("calibration_restore", 5, 5, 1, 0, early_stopping=False),
    PhaseSpec("a1", 10, 30, 1, 1),
    PhaseSpec("a2", 10, 45, 1, 1),
)


@dataclass(frozen=True)
class V7TrainingContract:
    phases: tuple[PhaseSpec, ...]
    train_days: int = TRAIN_DAYS
    updates_per_coverage_cycle: int = UPDATES_PER_COVERAGE_CYCLE
    pareto_top_k: int = 5

    def phase(self, phase_id: str) -> PhaseSpec:
        normalized = str(phase_id).strip().lower()
        for phase in self.phases:
            if phase.phase_id == normalized:
                return phase
        raise V7TrainingError(f"unknown V7 phase: {phase_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dcvpp.r4.v7-joint-training-contract.v1",
            "phase_order": [phase.phase_id for phase in self.phases],
            "train_days": self.train_days,
            "updates_per_coverage_cycle": self.updates_per_coverage_cycle,
            "coverage_update_days": [
                part.stop - part.start for part in coverage_cycle_slices(self.train_days)
            ],
            "pareto_top_k": self.pareto_top_k,
            "phases": [asdict(phase) for phase in self.phases],
            "decision_gradient_boundary": "detached_distribution_truth",
        }


def coverage_cycle_slices(
    day_count: int = TRAIN_DAYS,
    updates: int = UPDATES_PER_COVERAGE_CYCLE,
) -> tuple[slice, ...]:
    """Partition every day exactly once into balanced optimizer updates."""

    count = int(day_count)
    update_count = int(updates)
    if count <= 0 or update_count <= 0 or update_count > count:
        raise V7TrainingError("invalid coverage-cycle partition")
    quotient, remainder = divmod(count, update_count)
    start = 0
    parts: list[slice] = []
    for index in range(update_count):
        size = quotient + (1 if index < remainder else 0)
        parts.append(slice(start, start + size))
        start += size
    if start != count:
        raise AssertionError("coverage-cycle partition lost training days")
    return tuple(parts)


def _index_day_value(value: Any, indices: Tensor, total_days: int) -> Any:
    if isinstance(value, Tensor):
        if value.ndim >= 1 and value.shape[0] == total_days:
            return value.index_select(0, indices.to(value.device))
        return value
    if isinstance(value, Mapping):
        return {
            key: _index_day_value(item, indices, total_days)
            for key, item in value.items()
        }
    if isinstance(value, tuple) and len(value) == total_days:
        return tuple(value[int(index)] for index in indices.tolist())
    if isinstance(value, list) and len(value) == total_days:
        return [value[int(index)] for index in indices.tolist()]
    return value


def coverage_cycle_batches(
    batch: Mapping[str, Any],
    *,
    coverage_cycle_index: int,
    seed: int = 20240813,
    shuffle: bool = True,
    expected_day_count: int | None = None,
    updates: int = UPDATES_PER_COVERAGE_CYCLE,
) -> tuple[Mapping[str, Any], ...]:
    """Materialize balanced batches with exact once-only inferred-day coverage.

    ``expected_day_count`` binds a caller to its frozen training contract.  It
    is optional so legacy callers that relied on the original 543-day default
    can continue to partition any explicitly supplied complete batch.
    """

    target = _required_tensor(
        batch,
        ("distribution_target", "distribution_target_zonal", "scenario_target_zonal", "target"),
        "coverage-cycle distribution target",
    )
    day_count = int(target.shape[0])
    if expected_day_count is not None and day_count != int(expected_day_count):
        raise V7TrainingError(
            f"coverage cycle expected {int(expected_day_count)} days, got {day_count}"
        )
    cycle = int(coverage_cycle_index)
    if cycle < 0:
        raise V7TrainingError("coverage cycle index must be nonnegative")
    indices = torch.arange(day_count)
    if shuffle:
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1_000_003 * cycle
        )
        indices = indices[torch.randperm(day_count, generator=generator)]
    return tuple(
        _index_day_value(batch, indices[part], day_count)
        for part in coverage_cycle_slices(day_count, int(updates))
    )


class JointUpdateScheduler:
    """Deterministic long-run D:P ratio scheduler across four-way cycles."""

    def __init__(self, contract: V7TrainingContract | None = None) -> None:
        self.contract = contract or V7TrainingContract(DEFAULT_PHASE_SPECS)

    def kind_at(self, phase_id: str, optimizer_update_index: int) -> UpdateKind:
        index = int(optimizer_update_index)
        if index < 0:
            raise V7TrainingError("optimizer update index must be nonnegative")
        pattern = self.contract.phase(phase_id).update_pattern
        return pattern[index % len(pattern)]

    def coverage_cycle(
        self, phase_id: str, coverage_cycle_index: int
    ) -> tuple[tuple[slice, UpdateKind], ...]:
        cycle = int(coverage_cycle_index)
        if cycle < 0:
            raise V7TrainingError("coverage cycle index must be nonnegative")
        offset = cycle * self.contract.updates_per_coverage_cycle
        return tuple(
            (part, self.kind_at(phase_id, offset + index))
            for index, part in enumerate(
                coverage_cycle_slices(
                    self.contract.train_days,
                    self.contract.updates_per_coverage_cycle,
                )
            )
        )


def training_contract_from_config(config: Mapping[str, Any]) -> V7TrainingContract:
    """Parse phase budgets while fail-closing on changed update ratios."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise V7TrainingError("training must be a mapping")
    cycle = training.get("coverage_cycle")
    if not isinstance(cycle, Mapping):
        raise V7TrainingError("training.coverage_cycle must be a mapping")
    train_days = int(cycle.get("days", -1))
    if train_days <= 0:
        raise V7TrainingError("V7 coverage cycle needs a positive training-day count")
    optimizer_updates = int(cycle.get("optimizer_updates", -1))
    if optimizer_updates != UPDATES_PER_COVERAGE_CYCLE:
        raise V7TrainingError("V7 coverage cycle must make four optimizer updates")
    expected_update_days = [
        part.stop - part.start
        for part in coverage_cycle_slices(train_days, optimizer_updates)
    ]
    declared_update_days = cycle.get("update_day_counts")
    if (
        not isinstance(declared_update_days, Sequence)
        or isinstance(declared_update_days, (str, bytes))
        or [int(value) for value in declared_update_days] != expected_update_days
    ):
        raise V7TrainingError(
            f"coverage update_day_counts must be {expected_update_days}"
        )
    protocol = config.get("protocol")
    if isinstance(protocol, Mapping):
        splits = protocol.get("splits")
        train_split = splits.get("train") if isinstance(splits, Mapping) else None
        if isinstance(train_split, Mapping) and int(train_split.get("days", -1)) != train_days:
            raise V7TrainingError(
                "coverage-cycle days must equal protocol.splits.train.days"
            )
    phases = training.get("phases")
    if not isinstance(phases, Mapping) or tuple(phases) != PHASE_ORDER:
        raise V7TrainingError(f"training phases must be ordered as {list(PHASE_ORDER)}")
    expected = {phase.phase_id: phase for phase in DEFAULT_PHASE_SPECS}
    parsed: list[PhaseSpec] = []
    for phase_id in PHASE_ORDER:
        raw = phases.get(phase_id)
        if not isinstance(raw, Mapping):
            raise V7TrainingError(f"training.phases.{phase_id} must be a mapping")
        reference = expected[phase_id]
        ratio = raw.get("update_ratio")
        if not isinstance(ratio, Mapping):
            raise V7TrainingError(f"{phase_id}.update_ratio must be a mapping")
        distribution_updates = int(ratio.get("distribution", -1))
        decision_updates = int(ratio.get("decision", -1))
        if (distribution_updates, decision_updates) != (
            reference.distribution_updates,
            reference.decision_updates,
        ):
            raise V7TrainingError(f"{phase_id} update ratio differs from the approved plan")
        minimum_cycles = int(raw.get("min_cycles", reference.minimum_cycles))
        maximum_cycles = int(raw.get("max_cycles", reference.maximum_cycles))
        if not 1 <= minimum_cycles <= maximum_cycles:
            raise V7TrainingError(f"{phase_id} has invalid cycle limits")
        if maximum_cycles != reference.maximum_cycles:
            raise V7TrainingError(f"{phase_id} maximum cycles differ from the approved plan")
        parsed.append(
            PhaseSpec(
                phase_id=phase_id,
                minimum_cycles=minimum_cycles,
                maximum_cycles=maximum_cycles,
                distribution_updates=distribution_updates,
                decision_updates=decision_updates,
                sample_count=int(raw.get("sample_count", 64)),
                warmup_sample_count=(
                    int(raw["sample_schedule"]["warmup_sample_count"])
                    if isinstance(raw.get("sample_schedule"), Mapping)
                    else None
                ),
                warmup_optimizer_updates=(
                    int(raw["sample_schedule"]["warmup_optimizer_updates"])
                    if isinstance(raw.get("sample_schedule"), Mapping)
                    else 0
                ),
                decision_horizons=tuple(
                    int(value)
                    for value in raw.get("required_horizons", (96,))
                ),
                early_stopping=bool(raw.get("early_stopping", reference.early_stopping)),
            )
        )
        if parsed[-1].sample_count != 64:
            raise V7TrainingError(f"{phase_id} main sample count must be 64")
        if (
            parsed[-1].warmup_sample_count != reference.warmup_sample_count
            or parsed[-1].warmup_optimizer_updates != reference.warmup_optimizer_updates
        ):
            raise V7TrainingError(f"{phase_id} sample schedule differs from the approved plan")
        if parsed[-1].decision_horizons != reference.decision_horizons:
            raise V7TrainingError(
                f"{phase_id} decision horizons differ from the approved plan"
            )
    candidate = training.get("candidate_selection", {})
    if not isinstance(candidate, Mapping) or int(candidate.get("top_k", -1)) != 5:
        raise V7TrainingError("candidate selection must retain the Pareto top five")
    return V7TrainingContract(
        tuple(parsed),
        train_days=train_days,
        updates_per_coverage_cycle=optimizer_updates,
        pareto_top_k=5,
    )


def detach_mapping(value: Any) -> Any:
    """Recursively detach tensors while preserving the model's public schema."""

    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: detach_mapping(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_mapping(item) for item in value)
    if isinstance(value, list):
        return [detach_mapping(item) for item in value]
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V7TrainingError(f"{label} must be a mapping")
    return value


def _decision_limits(batch: Mapping[str, Any]) -> Mapping[str, Tensor] | None:
    """Expose only static physical limits to the decision actor."""

    targets = batch.get("rollout_targets")
    if not isinstance(targets, Mapping):
        return None
    aliases = {
        "grid_mw": "grid_limit_mw",
        "battery_mw": "battery_power_mw",
        "compute_work": "capacity_work_units",
    }
    limits: dict[str, Tensor] = {}
    for output_name, input_name in aliases.items():
        value = targets.get(input_name)
        if not isinstance(value, Tensor):
            return None
        limits[output_name] = value
    return limits


def _normalized_safe_anchor_actions(
    batch: Mapping[str, Any],
) -> Mapping[str, Tensor] | None:
    """Convert the physical EDF baseline into Policy's normalized plan space."""

    anchors = batch.get("safe_anchor_actions")
    if not isinstance(anchors, Mapping):
        return None
    targets = batch.get("rollout_targets")
    if not isinstance(targets, Mapping):
        # Unit tests and external adapters may already provide normalized anchors.
        return _mapping(anchors, "batch.safe_anchor_actions")
    limits = {
        "day_ahead": targets.get("grid_limit_mw"),
        "battery": targets.get("battery_power_mw"),
        "compute": targets.get("capacity_work_units"),
    }
    normalized: dict[str, Tensor] = {}
    for name, bound in limits.items():
        value = anchors.get(name)
        if not isinstance(value, Tensor) or not isinstance(bound, Tensor):
            raise V7TrainingError(
                f"P0 safe anchor requires tensor {name} and its physical limit"
            )
        if value.ndim != 3:
            raise V7TrainingError(f"P0 safe anchor {name} must be a daily plan")
        if value.shape[1:] == (4, 96):
            value = value.transpose(1, 2)
        if value.shape[1:] != (96, 4):
            raise V7TrainingError(
                f"P0 safe anchor {name} must be [day,96,4] or [day,4,96]"
            )
        if bound.ndim == 1:
            bound = bound.unsqueeze(0).expand(value.shape[0], -1)
        if bound.shape != (value.shape[0], 4):
            raise V7TrainingError(f"P0 {name} physical limit must be [day,4]")
        scaled = value / bound.to(value)[:, None].clamp_min(1.0e-8)
        normalized[name] = (
            scaled.clamp(-1.0, 1.0)
            if name == "battery"
            else scaled.clamp(0.0, 1.0)
        )
    return normalized


def _distribution_view(output: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    distribution = output.get("distribution", output)
    distribution = _mapping(distribution, "distribution output")
    issue_context = output.get("issue_context")
    return distribution, issue_context


def _policy_view(
    output: Mapping[str, Any],
) -> tuple[
    Mapping[str, Tensor],
    Mapping[str, Tensor],
    Mapping[str, Tensor],
    Mapping[str, Any],
    Mapping[str, Any] | None,
]:
    policy = _mapping(output.get("policy"), "decision output.policy")
    actions = _mapping(policy.get("actions", output.get("actions")), "policy.actions")
    normalized_actions = _mapping(
        policy.get("normalized_actions", actions), "policy.normalized_actions"
    )
    action_plans = _mapping(
        policy.get("normalized_plans", normalized_actions), "policy.normalized_plans"
    )
    critic = _mapping(output.get("critic", policy.get("critic")), "decision output.critic")
    rollout = output.get("rollout")
    return (
        actions,
        normalized_actions,
        action_plans,
        critic,
        None if rollout is None else _mapping(rollout, "decision output.rollout"),
    )


def _call_supported(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(callback)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        selected = kwargs
    else:
        selected = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callback(*args, **selected)


def _named_parameter_dict(model: nn.Module) -> dict[str, nn.Parameter]:
    return dict(model.named_parameters())


@dataclass(frozen=True)
class ParameterPartition:
    distribution_truth: tuple[str, ...]
    decision: tuple[str, ...]

    def validate(self, model: nn.Module) -> None:
        available = _named_parameter_dict(model)
        unknown = (set(self.distribution_truth) | set(self.decision)) - set(available)
        if unknown:
            raise V7TrainingError(f"model parameter API returned unknown names: {sorted(unknown)}")
        overlap = set(self.distribution_truth) & set(self.decision)
        if overlap:
            raise V7TrainingError(f"distribution and decision parameter groups overlap: {sorted(overlap)}")
        if not self.distribution_truth or not self.decision:
            raise V7TrainingError("both distribution and decision parameter groups are required")


def parameter_partition(model: nn.Module) -> ParameterPartition:
    """Read the explicit parameter boundary promised by the V7 architecture."""

    distribution_api = getattr(model, "distribution_truth_parameter_names", None)
    decision_api = getattr(model, "decision_parameter_names", None)
    if not callable(distribution_api) or not callable(decision_api):
        raise V7TrainingError(
            "V7 model must expose distribution_truth_parameter_names() and decision_parameter_names()"
        )
    partition = ParameterPartition(
        tuple(str(name) for name in distribution_api()),
        tuple(str(name) for name in decision_api()),
    )
    partition.validate(model)
    return partition


def build_v7_optimizers(
    model: nn.Module,
    *,
    distribution_learning_rate: float,
    decision_learning_rate: float,
    weight_decay: float = 0.01,
) -> tuple[dict[UpdateKind, torch.optim.Optimizer], ParameterPartition]:
    """Create disjoint AdamW optimizers and freeze all out-of-contract params."""

    partition = parameter_partition(model)
    parameters = _named_parameter_dict(model)
    allowed = set(partition.distribution_truth) | set(partition.decision)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in allowed)
    optimizers: dict[UpdateKind, torch.optim.Optimizer] = {
        "distribution": torch.optim.AdamW(
            [parameters[name] for name in partition.distribution_truth],
            lr=float(distribution_learning_rate),
            weight_decay=float(weight_decay),
        ),
        "decision": torch.optim.AdamW(
            [parameters[name] for name in partition.decision],
            lr=float(decision_learning_rate),
            weight_decay=float(weight_decay),
        ),
    }
    return optimizers, partition


def assert_distribution_truth_has_no_gradient(
    model: nn.Module,
    partition: ParameterPartition,
) -> None:
    parameters = _named_parameter_dict(model)
    leaked = [
        name
        for name in partition.distribution_truth
        if parameters[name].grad is not None
        and bool((parameters[name].grad.detach().abs() > 0).any())
    ]
    if leaked:
        raise V7TrainingError(
            "decision objective leaked into distribution truth parameters: "
            + ", ".join(leaked)
        )


def _required_tensor(mapping: Mapping[str, Any], aliases: Sequence[str], label: str) -> Tensor:
    for key in aliases:
        value = mapping.get(key)
        if isinstance(value, Tensor):
            return value
    raise V7TrainingError(f"{label} is required; accepted keys={list(aliases)}")


def _clear_disallowed_gradients(
    model: nn.Module,
    allowed_names: Iterable[str],
) -> None:
    allowed = set(allowed_names)
    for name, parameter in model.named_parameters():
        if name not in allowed:
            parameter.grad = None


@dataclass(frozen=True)
class UpdateResult:
    phase: str
    update_kind: UpdateKind
    loss: float
    metrics: dict[str, float]


class V7JointTrainer:
    """One-update engine; the phase runner owns epochs, checkpoints and I/O."""

    def __init__(
        self,
        model: nn.Module,
        optimizers: Mapping[UpdateKind, torch.optim.Optimizer],
        *,
        baseline_proper_scales: Mapping[str, Tensor | float],
        cost_scale: Tensor | float,
        channel_scale: Tensor | None = None,
        partition: ParameterPartition | None = None,
        rollout_fn: Callable[..., Mapping[str, Any]] | None = None,
        contract: V7TrainingContract | None = None,
        distribution_loss_weights: Mapping[str, float] | None = None,
        policy_loss_weights: Mapping[str, Mapping[str, float]] | None = None,
        a1_name_contains: Sequence[str] = ("dart", "risk_bridge", "decision_router"),
        gradient_clip_norm: float = 1.0,
        cvar_alpha: float = 0.95,
        cvar_temperature: float = 0.02,
        decision_microbatch_days: int | None = None,
        distribution_microbatch_days: int | None = None,
        decision_autocast_dtype: torch.dtype | None = None,
        path_seed: int = 20240813,
    ) -> None:
        self.model = model
        self.optimizers = dict(optimizers)
        if set(self.optimizers) != {"distribution", "decision"}:
            raise V7TrainingError("V7 requires distribution and decision optimizers")
        self.partition = partition or parameter_partition(model)
        self.partition.validate(model)
        self.baseline_proper_scales = dict(baseline_proper_scales)
        self.cost_scale = cost_scale
        self.channel_scale = channel_scale
        self.rollout_fn = rollout_fn
        self.contract = contract or V7TrainingContract(DEFAULT_PHASE_SPECS)
        self.scheduler = JointUpdateScheduler(self.contract)
        self.distribution_loss_weights = (
            None if distribution_loss_weights is None else dict(distribution_loss_weights)
        )
        self.policy_loss_weights = {
            str(key): dict(value) for key, value in (policy_loss_weights or {}).items()
        }
        self.a1_name_contains = tuple(str(item) for item in a1_name_contains)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_temperature = float(cvar_temperature)
        if decision_microbatch_days is not None and int(decision_microbatch_days) <= 0:
            raise V7TrainingError("decision_microbatch_days must be positive")
        self.decision_microbatch_days = (
            None
            if decision_microbatch_days is None
            else int(decision_microbatch_days)
        )
        if (
            distribution_microbatch_days is not None
            and int(distribution_microbatch_days) <= 0
        ):
            raise V7TrainingError("distribution_microbatch_days must be positive")
        self.distribution_microbatch_days = (
            None
            if distribution_microbatch_days is None
            else int(distribution_microbatch_days)
        )
        if decision_autocast_dtype not in {None, torch.bfloat16, torch.float16}:
            raise V7TrainingError("decision autocast must be bf16, fp16, or disabled")
        self.decision_autocast_dtype = decision_autocast_dtype
        self.path_seed = int(path_seed)

    def _active_names(self, phase: str, update_kind: UpdateKind) -> tuple[str, ...]:
        names = (
            self.partition.distribution_truth
            if update_kind == "distribution"
            else self.partition.decision
        )
        if phase != "a1":
            return names
        explicit = getattr(self.model, "a1_parameter_names", None)
        if callable(explicit):
            allowed = set(str(name) for name in explicit())
            selected = tuple(name for name in names if name in allowed)
        else:
            selected = tuple(
                name for name in names if any(token in name for token in self.a1_name_contains)
            )
        if not selected:
            raise V7TrainingError("A1 has no parameter matching its DART-only allowlist")
        return selected

    def _forward_distribution(
        self,
        batch: Mapping[str, Any],
        sample_count: int,
        *,
        seed: int | None = None,
    ) -> tuple[Mapping[str, Any], Any]:
        callback = getattr(self.model, "forward_distribution", None)
        if not callable(callback):
            raise V7TrainingError("V7 model must expose forward_distribution")
        output = _call_supported(
            callback, batch, sample_count=sample_count, seed=seed
        )
        return _distribution_view(_mapping(output, "forward_distribution output"))

    def _forward_decision(
        self,
        batch: Mapping[str, Any],
        distribution: Mapping[str, Any],
        issue_context: Any,
        decision_horizon: int,
    ) -> Mapping[str, Any]:
        callback = getattr(self.model, "forward_decision", None)
        if not callable(callback):
            raise V7TrainingError("V7 model must expose forward_decision")
        primary = batch if issue_context is None else detach_mapping(issue_context)
        output = _call_supported(
            callback,
            primary,
            distribution=detach_mapping(distribution),
            detach_distribution_core=True,
            decision_horizon=int(decision_horizon),
            limits=_decision_limits(batch),
        )
        return _mapping(output, "forward_decision output")

    def update(
        self,
        batch: Mapping[str, Any],
        *,
        phase: str,
        optimizer_update_index: int,
    ) -> UpdateResult:
        phase_id = str(phase).lower()
        spec = self.contract.phase(phase_id)
        kind = self.scheduler.kind_at(phase_id, optimizer_update_index)
        sample_count = spec.sample_count_at_update(optimizer_update_index)
        decision_horizon = spec.decision_horizon_at_update(optimizer_update_index)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        target = _required_tensor(
            batch,
            (
                "distribution_target",
                "distribution_target_zonal",
                "scenario_target_zonal",
                "target",
            ),
            "optimizer-update distribution target",
        )
        day_count = int(target.shape[0])
        configured_microbatch = (
            self.distribution_microbatch_days
            if kind == "distribution"
            else self.decision_microbatch_days
        )
        microbatch_days = (
            day_count
            if configured_microbatch is None
            else min(day_count, configured_microbatch)
        )
        weighted_loss = 0.0
        weighted_metrics: dict[str, float] = {}
        for start in range(0, day_count, microbatch_days):
            stop = min(start + microbatch_days, day_count)
            indices = torch.arange(start, stop)
            piece = _index_day_value(batch, indices, day_count)
            use_autocast = (
                kind == "decision"
                and self.decision_autocast_dtype is not None
                and target.device.type == "cuda"
            )
            sample_seed = self.path_seed + 1_000_003 * int(optimizer_update_index)
            with torch.autocast(
                device_type=target.device.type,
                dtype=self.decision_autocast_dtype or torch.bfloat16,
                enabled=use_autocast,
            ):
                if kind == "distribution":
                    loss, terms = self._distribution_update(
                        piece, sample_count, seed=sample_seed
                    )
                else:
                    loss, terms = self._decision_update(
                        piece,
                        phase_id,
                        sample_count,
                        decision_horizon,
                        seed=sample_seed,
                    )
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise V7TrainingError("training objective must be one finite scalar")
            fraction = float(stop - start) / float(day_count)
            (loss * fraction).backward()
            weighted_loss += float(loss.detach().cpu()) * fraction
            for key, value in terms.items():
                if isinstance(value, Tensor) and key.startswith("raw_"):
                    weighted_metrics[key] = weighted_metrics.get(key, 0.0) + (
                        float(value.detach().mean().cpu()) * fraction
                    )
        if kind == "decision":
            assert_distribution_truth_has_no_gradient(self.model, self.partition)
        active_names = self._active_names(phase_id, kind)
        _clear_disallowed_gradients(self.model, active_names)
        parameters = _named_parameter_dict(self.model)
        torch.nn.utils.clip_grad_norm_(
            [parameters[name] for name in active_names], self.gradient_clip_norm
        )
        self.optimizers[kind].step()
        return UpdateResult(phase_id, kind, weighted_loss, weighted_metrics)

    def run_coverage_cycle(
        self,
        batch: Mapping[str, Any],
        *,
        phase: str,
        coverage_cycle_index: int,
        seed: int = 20240813,
        shuffle: bool = True,
    ) -> tuple[UpdateResult, ...]:
        """Execute one exact configured coverage cycle over every day once."""

        cycle = int(coverage_cycle_index)
        pieces = coverage_cycle_batches(
            batch,
            coverage_cycle_index=cycle,
            seed=seed,
            shuffle=shuffle,
            expected_day_count=self.contract.train_days,
            updates=self.contract.updates_per_coverage_cycle,
        )
        offset = cycle * self.contract.updates_per_coverage_cycle
        return tuple(
            self.update(
                piece,
                phase=phase,
                optimizer_update_index=offset + local_index,
            )
            for local_index, piece in enumerate(pieces)
        )

    def _distribution_update(
        self,
        batch: Mapping[str, Any],
        sample_count: int,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        distribution, _ = self._forward_distribution(
            batch, sample_count, seed=seed
        )
        samples = _required_tensor(
            distribution, ("samples", "samples_zonal"), "distribution.samples"
        )
        target = _required_tensor(
            batch,
            ("distribution_target", "distribution_target_zonal", "scenario_target_zonal", "target"),
            "distribution target",
        ).to(device=samples.device, dtype=torch.float32)
        weights = distribution.get("weights")
        if weights is not None and not isinstance(weights, Tensor):
            raise V7TrainingError("distribution.weights must be a tensor")
        center = distribution.get("center")
        scale = distribution.get("scale")
        identity_samples = distribution.get("identity_samples", samples)
        if not isinstance(identity_samples, Tensor):
            raise V7TrainingError("distribution.identity_samples must be a tensor")
        identity_samples = identity_samples.float()
        v7_losses.assert_dart_identity(identity_samples)
        channel_scale = distribution.get(
            "channel_scale", batch.get("channel_scale", self.channel_scale)
        )
        if not isinstance(channel_scale, Tensor):
            raise V7TrainingError(
                "frozen channel_scale[21] is required for proper-score training"
            )
        (
            normalized_samples,
            normalized_target,
            normalized_center,
            normalized_scale,
        ) = v7_losses.normalize_distribution_channels(
            samples.float(),
            target,
            channel_scale.float(),
            center=center.float() if isinstance(center, Tensor) else None,
            scale=scale.float() if isinstance(scale, Tensor) else None,
        )
        terms = v7_losses.baseline_normalized_distribution_terms(
            normalized_samples,
            normalized_target,
            baseline_scales=self.baseline_proper_scales,
            weights=weights.float() if isinstance(weights, Tensor) else None,
            center=normalized_center,
            scale=normalized_scale,
            identity_samples=identity_samples,
        )
        return v7_losses.compose_distribution_loss(
            terms, self.distribution_loss_weights
        ), terms

    def _decision_update(
        self,
        batch: Mapping[str, Any],
        phase: str,
        sample_count: int,
        decision_horizon: int,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # no_grad plus recursive detach is intentional redundancy: either one
        # alone prevents a weak model implementation from leaking decision loss
        # into truth parameters, while the explicit assertion verifies it.
        with torch.no_grad():
            distribution, issue_context = self._forward_distribution(
                batch, sample_count, seed=seed
            )
        decision = self._forward_decision(
            batch, distribution, issue_context, decision_horizon
        )
        (
            actions,
            normalized_actions,
            action_plans,
            critic,
            embedded_rollout,
        ) = _policy_view(decision)
        if self.rollout_fn is None:
            if embedded_rollout is None:
                raise V7TrainingError("decision update requires a shared-shield rollout")
            rollout = embedded_rollout
        else:
            rollout = _mapping(
                _call_supported(
                    self.rollout_fn,
                    actions=actions,
                    critic=critic,
                    batch=batch,
                    distribution=detach_mapping(distribution),
                    decision_horizon=int(decision_horizon),
                ),
                "rollout_fn output",
            )
        terms = v7_losses.joint_policy_loss_terms(
            rollout,
            critic,
            actions,
            normalized_actions=normalized_actions,
            action_plans=action_plans,
            safe_anchor_actions=(
                _normalized_safe_anchor_actions(batch)
                if phase == "p0"
                else None
            ),
            cost_scale=self.cost_scale,
            soc_scale=batch.get("soc_scale", 1.0),
            backlog_scale=batch.get("backlog_scale", 1.0),
            alpha=self.cvar_alpha,
            smooth_temperature=self.cvar_temperature,
        )
        weights = self.policy_loss_weights.get(phase)
        return v7_losses.compose_policy_loss(terms, weights), terms


@dataclass(frozen=True)
class DistributionGateMetrics:
    energy_score_delta_upper: float
    marginal_crps_ratio: float
    coverage90_error: float
    coverage95_error: float
    interval_score_ratio: float
    tail_score_ratio: float
    dependence_score_ratio: float
    identity_residual: float


@dataclass(frozen=True)
class PolicyGateMetrics:
    daily_cost_delta_upper: float
    normalized_sla_increase: float
    feasibility_fraction: float
    non_exogenous_rejected_origins: int


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    failed: tuple[str, ...]


def evaluate_distribution_gate(metrics: DistributionGateMetrics) -> GateDecision:
    """Apply the unchanged paper gate; no MAE substitution is permitted."""

    failed: list[str] = []
    if not metrics.energy_score_delta_upper < 0.0:
        failed.append("energy_score_paired_upper_not_below_zero")
    for name in (
        "marginal_crps_ratio",
        "interval_score_ratio",
        "tail_score_ratio",
        "dependence_score_ratio",
    ):
        if getattr(metrics, name) > 1.01:
            failed.append(f"{name}_exceeds_1pct_noninferiority")
    if abs(metrics.coverage90_error) > 0.03:
        failed.append("coverage90_error_exceeds_0.03")
    if abs(metrics.coverage95_error) > 0.03:
        failed.append("coverage95_error_exceeds_0.03")
    if abs(metrics.identity_residual) > 1.0e-6:
        failed.append("da_rt_dart_identity_exceeds_1e-6")
    return GateDecision(not failed, tuple(failed))


def evaluate_policy_gate(metrics: PolicyGateMetrics) -> GateDecision:
    failed: list[str] = []
    if metrics.daily_cost_delta_upper > 0.0:
        failed.append("daily_cost_paired_upper_above_zero")
    if metrics.normalized_sla_increase > 0.001:
        failed.append("normalized_sla_increase_above_0.001")
    if metrics.feasibility_fraction < 1.0:
        failed.append("feasibility_fraction_below_one")
    if metrics.non_exogenous_rejected_origins > 0:
        failed.append("non_exogenous_origin_rejected")
    return GateDecision(not failed, tuple(failed))


@dataclass(frozen=True)
class ParetoCandidate:
    candidate_id: str
    phase: str
    optimizer_update: int
    objectives: Mapping[str, float]
    checkpoint_path: str | None = None


class ParetoFrontier:
    """Small in-memory top-five candidate set; heavy audits run only later."""

    def __init__(self, objective_names: Sequence[str], top_k: int = 5) -> None:
        if not objective_names or int(top_k) <= 0:
            raise V7TrainingError("Pareto frontier needs objectives and positive top_k")
        self.objective_names = tuple(str(name) for name in objective_names)
        self.top_k = int(top_k)
        self._candidates: list[ParetoCandidate] = []

    def _values(self, candidate: ParetoCandidate) -> tuple[float, ...]:
        missing = set(self.objective_names) - set(candidate.objectives)
        if missing:
            raise V7TrainingError(f"candidate lacks objectives: {sorted(missing)}")
        values = tuple(float(candidate.objectives[name]) for name in self.objective_names)
        if not all(math.isfinite(value) for value in values):
            raise V7TrainingError("candidate objectives must be finite")
        return values

    def _dominates(self, left: ParetoCandidate, right: ParetoCandidate) -> bool:
        lhs, rhs = self._values(left), self._values(right)
        return all(a <= b for a, b in zip(lhs, rhs, strict=True)) and any(
            a < b for a, b in zip(lhs, rhs, strict=True)
        )

    def add(self, candidate: ParetoCandidate) -> tuple[ParetoCandidate, ...]:
        self._values(candidate)
        if any(existing.candidate_id == candidate.candidate_id for existing in self._candidates):
            raise V7TrainingError(f"duplicate candidate id: {candidate.candidate_id}")
        if any(self._dominates(existing, candidate) for existing in self._candidates):
            return self.candidates
        self._candidates = [
            existing for existing in self._candidates if not self._dominates(candidate, existing)
        ]
        self._candidates.append(candidate)
        # Deterministic bounded proxy ranking.  This is only candidate triage;
        # the frozen 179-day paired gates decide scientific selection.
        self._candidates.sort(
            key=lambda item: (sum(self._values(item)), self._values(item), item.candidate_id)
        )
        self._candidates = self._candidates[: self.top_k]
        return self.candidates

    @property
    def candidates(self) -> tuple[ParetoCandidate, ...]:
        return tuple(self._candidates)


__all__ = [
    "DEFAULT_PHASE_SPECS",
    "DistributionGateMetrics",
    "GateDecision",
    "JointUpdateScheduler",
    "PHASE_ORDER",
    "ParameterPartition",
    "ParetoCandidate",
    "ParetoFrontier",
    "PhaseSpec",
    "PolicyGateMetrics",
    "TRAIN_DAYS",
    "UPDATES_PER_COVERAGE_CYCLE",
    "UpdateResult",
    "V7JointTrainer",
    "V7TrainingContract",
    "V7TrainingError",
    "assert_distribution_truth_has_no_gradient",
    "build_v7_optimizers",
    "coverage_cycle_slices",
    "coverage_cycle_batches",
    "detach_mapping",
    "evaluate_distribution_gate",
    "evaluate_policy_gate",
    "parameter_partition",
    "training_contract_from_config",
]
