"""R4 phase training and capability-gate contracts.

This module deliberately contains no R3 imports and no dataset discovery.  A
caller supplies the model, iterables, cheap validator, and phase-end evaluator.
That boundary keeps the formal split sealed while still making the training
state machine executable and testable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
import inspect
import math
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import losses as r4_losses
from .artifact_io import (
    atomic_json,
    atomic_torch_save,
    read_json,
    sha256_file,
    write_state as write_artifact_state,
)
from .config import TRAINING_SEMANTICS_REVISION


R4_PROTOCOL_VERSION = "r4-hybrid-solver-free-formal-v1"
DISTRIBUTIONAL_R4_PROTOCOL_VERSION = (
    "r4-v6-continuous-distributional-joint-formal-v1"
)
R4_SEEDS = (20240813, 20240829, 20240917)
PHASE_ORDER = ("representation", "scenario", "policy", "a1", "a2")
DISTRIBUTIONAL_PHASE_ORDER = (
    "representation",
    "distribution",
    "joint_policy",
    "a1",
    "a2",
)
PHYSICAL_BATCH_FALLBACK = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
EFFECTIVE_BATCH_SIZE = 512
VALIDATION_START = date(2023, 7, 1)
VALIDATION_END = date(2023, 12, 31)
TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 6, 30)
CAPABILITY_GATE_DAYS = 179
FORMAL_TRAINING_DAYS = 543
HARD_GATE_VALIDATION_ORIGINS = CAPABILITY_GATE_DAYS * 96


def _artifact_protocol_version(payload: Mapping[str, Any]) -> str:
    architecture = str(payload.get("architecture_version", ""))
    phase = str(payload.get("phase", "")).lower()
    if (
        architecture == "continuous_distributional_joint_moe_r4_v6"
        or payload.get("distribution_training_contract_sha256") is not None
        or phase in {"distribution", "joint_policy"}
    ):
        return DISTRIBUTIONAL_R4_PROTOCOL_VERSION
    return R4_PROTOCOL_VERSION


def _checkpoint_value_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for checkpoint state, including extra_state."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_checkpoint_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_checkpoint_value_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


class R4ProtocolError(RuntimeError):
    """A fail-closed R4 training or protocol contract violation."""


@dataclass(frozen=True)
class CurriculumStage:
    horizon: int
    epochs: int
    patience: int | None = None


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    epochs: int
    patience: int
    minimum_epochs: int = 0
    curriculum: tuple[CurriculumStage, ...] = ()
    capability_days: int | None = None
    hard_gate_horizon: int | None = None
    min_delta: float = 1.0e-5


@dataclass(frozen=True)
class DistributionCurriculumStage:
    """One preregistered V6 Distribution optimization emphasis."""

    stage_id: str
    name: str
    minimum_epochs: int
    maximum_epochs: int
    patience: int
    min_delta: float
    loss_weights: Mapping[str, float]


@dataclass(frozen=True)
class DistributionCurriculumContract:
    """Frozen training/candidate contract independent of Monte Carlo N."""

    stages: tuple[DistributionCurriculumStage, ...]
    train_samples: int
    cheap_validation_samples: int
    physical_microbatch_days: int
    candidate_selection: Mapping[str, Any]

    @property
    def maximum_epochs(self) -> int:
        return sum(stage.maximum_epochs for stage in self.stages)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema": "dcvpp.r4.v6-distribution-training-contract.v1",
            "phase": "distribution",
            "train_samples": self.train_samples,
            "cheap_validation_samples": self.cheap_validation_samples,
            "physical_microbatch_days": self.physical_microbatch_days,
            "scientific_optimizer_effective_days": FORMAL_TRAINING_DAYS,
            "stages": [asdict(stage) for stage in self.stages],
            "candidate_selection": dict(self.candidate_selection),
            "fixed_particle_head_predecessor_allowed": False,
            "formal_gate_role": "level2_external_frozen_179_day_callback",
            "selected_checkpoint_rule": "level3_pass_only",
        }


_DISTRIBUTION_STAGE_IDS = ("S1", "S2", "S3", "S4", "S5")
_DISTRIBUTION_STAGE_NAMES = (
    "center_scale",
    "coverage",
    "tail",
    "dependence",
    "joint_proper_finetune",
)
_DISTRIBUTION_LOSS_KEYS = frozenset(
    {
        "center_scale",
        "energy_score",
        "marginal_crps",
        "coverage90",
        "coverage95",
        "interval",
        "tail",
        "dependence",
        "diversity",
        "identity",
    }
)


def distribution_curriculum_contract(
    config: Mapping[str, Any],
) -> DistributionCurriculumContract:
    """Parse and fail-close the V6 five-stage preregistration."""

    training = config.get("training")
    phases = training.get("phases") if isinstance(training, Mapping) else None
    entry = phases.get("distribution") if isinstance(phases, Mapping) else None
    if not isinstance(entry, Mapping):
        raise R4ProtocolError("training.phases.distribution must be a mapping")
    raw_stages = entry.get("curriculum")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        raise R4ProtocolError("Distribution curriculum must be a five-stage sequence")
    if len(raw_stages) != len(_DISTRIBUTION_STAGE_IDS):
        raise R4ProtocolError("Distribution curriculum must contain exactly S1--S5")
    stages: list[DistributionCurriculumStage] = []
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise R4ProtocolError("each Distribution curriculum stage must be a mapping")
        stage_id = str(raw.get("id", ""))
        name = str(raw.get("name", ""))
        if stage_id != _DISTRIBUTION_STAGE_IDS[index] or name != _DISTRIBUTION_STAGE_NAMES[index]:
            raise R4ProtocolError(
                "Distribution curriculum order/names must be "
                "S1 center_scale -> S2 coverage -> S3 tail -> "
                "S4 dependence -> S5 joint_proper_finetune"
            )
        minimum = int(raw.get("min_epochs", 0))
        maximum = int(raw.get("max_epochs", 0))
        patience = int(raw.get("patience", 0))
        min_delta = float(raw.get("min_delta", float("nan")))
        if not (1 <= minimum <= maximum and 1 <= patience <= maximum):
            raise R4ProtocolError(
                f"Distribution {stage_id} requires 1 <= min/patience <= max"
            )
        if not math.isfinite(min_delta) or min_delta < 0.0:
            raise R4ProtocolError(
                f"Distribution {stage_id} min_delta must be finite and nonnegative"
            )
        raw_weights = raw.get("loss_weights")
        if not isinstance(raw_weights, Mapping):
            raise R4ProtocolError(
                f"Distribution {stage_id} loss_weights must be a mapping"
            )
        if set(raw_weights) != _DISTRIBUTION_LOSS_KEYS:
            missing = sorted(_DISTRIBUTION_LOSS_KEYS - set(raw_weights))
            extra = sorted(set(raw_weights) - _DISTRIBUTION_LOSS_KEYS)
            raise R4ProtocolError(
                f"Distribution {stage_id} loss keys changed; missing={missing}, extra={extra}"
            )
        weights = {str(key): float(value) for key, value in raw_weights.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise R4ProtocolError(
                f"Distribution {stage_id} loss weights must be finite/nonnegative"
            )
        if not any(value > 0.0 for value in weights.values()):
            raise R4ProtocolError(f"Distribution {stage_id} has no active loss")
        stages.append(
            DistributionCurriculumStage(
                stage_id,
                name,
                minimum,
                maximum,
                patience,
                min_delta,
                weights,
            )
        )
    maximum_epochs = sum(stage.maximum_epochs for stage in stages)
    if int(entry.get("max_epochs", -1)) != maximum_epochs:
        raise R4ProtocolError(
            "Distribution max_epochs must equal the sum of S1--S5 max_epochs"
        )
    train_samples = int(entry.get("train_samples", 0))
    cheap_samples = int(entry.get("cheap_validation_samples", 0))
    physical = int(entry.get("physical_microbatch_days", 0))
    if (train_samples, cheap_samples, physical) != (32, 64, 32):
        raise R4ProtocolError(
            "V6 Distribution freezes train N=32, cheap N=64, physical microbatch=32"
        )
    selection = entry.get("candidate_selection")
    if not isinstance(selection, Mapping):
        raise R4ProtocolError("Distribution candidate_selection must be a mapping")
    required_selection = {
        "mode": "top_k_pareto_scheduled",
        "level2": "full_frozen_179_day_distribution_gate",
        "level3": "pass_only",
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise R4ProtocolError(
                f"Distribution candidate_selection.{key} must be {expected!r}"
            )
    if int(selection.get("top_k", 0)) < 1:
        raise R4ProtocolError("Distribution candidate top_k must be positive")
    if int(selection.get("scheduled_every_epochs", 0)) < 1:
        raise R4ProtocolError(
            "Distribution scheduled candidate interval must be positive"
        )
    if int(selection.get("max_candidates", 0)) < 5:
        raise R4ProtocolError("Distribution Level1 must retain at least five candidates")
    expected_pareto = (
        "energy_score",
        "marginal_crps",
        "coverage90_error",
        "coverage95_error",
        "tail",
        "dependence",
    )
    if tuple(selection.get("pareto_metrics", ())) != expected_pareto:
        raise R4ProtocolError("Distribution Pareto metric order changed")
    return DistributionCurriculumContract(
        tuple(stages), train_samples, cheap_samples, physical, dict(selection)
    )


R4_PHASE_SPECS: dict[str, PhaseSpec] = {
    "representation": PhaseSpec(
        "representation", 800, 30, minimum_epochs=100,
        capability_days=CAPABILITY_GATE_DAYS
    ),
    "scenario": PhaseSpec(
        "scenario", 200, 15, minimum_epochs=50,
        capability_days=CAPABILITY_GATE_DAYS
    ),
    "distribution": PhaseSpec(
        "distribution", 200, 12, minimum_epochs=20,
        capability_days=CAPABILITY_GATE_DAYS
    ),
    "policy": PhaseSpec(
        "policy",
        320,
        20,
        curriculum=(
            CurriculumStage(8, 40),
            CurriculumStage(24, 60),
            CurriculumStage(48, 80),
            CurriculumStage(96, 140, patience=20),
        ),
        hard_gate_horizon=96,
    ),
    "joint_policy": PhaseSpec(
        "joint_policy",
        320,
        20,
        curriculum=(
            CurriculumStage(8, 40),
            CurriculumStage(24, 60),
            CurriculumStage(48, 80),
            CurriculumStage(96, 140, patience=20),
        ),
        hard_gate_horizon=96,
    ),
    "a1": PhaseSpec(
        "a1",
        120,
        15,
        curriculum=(
            CurriculumStage(8, 15),
            CurriculumStage(24, 25),
            CurriculumStage(48, 30),
            CurriculumStage(96, 50, patience=15),
        ),
        hard_gate_horizon=96,
    ),
    "a2": PhaseSpec(
        "a2",
        180,
        20,
        curriculum=(
            CurriculumStage(8, 20),
            CurriculumStage(24, 30),
            CurriculumStage(48, 40),
            CurriculumStage(96, 90, patience=20),
        ),
        hard_gate_horizon=96,
    ),
}


def phase_spec(
    name: str, config: Mapping[str, Any] | None = None
) -> PhaseSpec:
    normalized = str(name).lower()
    if normalized == "distribution" and config is not None:
        contract = distribution_curriculum_contract(config)
        final = contract.stages[-1]
        return PhaseSpec(
            "distribution",
            contract.maximum_epochs,
            final.patience,
            minimum_epochs=sum(stage.minimum_epochs for stage in contract.stages),
            capability_days=CAPABILITY_GATE_DAYS,
            min_delta=final.min_delta,
        )
    try:
        return R4_PHASE_SPECS[normalized]
    except KeyError as exc:
        raise R4ProtocolError(f"unknown R4 phase: {name!r}") from exc


def accumulation_for_physical_batch(physical_batch_size: int) -> int:
    """Return exact accumulation needed for an unbiased effective batch of 512."""

    size = int(physical_batch_size)
    if size not in PHYSICAL_BATCH_FALLBACK:
        raise R4ProtocolError(
            f"physical batch must be in {PHYSICAL_BATCH_FALLBACK}, got {size}"
        )
    if EFFECTIVE_BATCH_SIZE % size:
        raise R4ProtocolError("physical batch must exactly divide effective batch 512")
    return EFFECTIVE_BATCH_SIZE // size


def unbiased_tail_normalized_mean(
    values: Tensor,
    tail_indicator: Tensor,
    *,
    tail_probability: float,
    inclusion_probability: Tensor | float | None = None,
) -> Tensor:
    """Horvitz-Thompson-style tail mean with a fixed population denominator.

    Dividing by the number of tail samples seen in a mini-batch is biased and
    becomes unstable when a batch contains no tail observations.  This
    estimator divides by the registered tail probability instead.  Optional
    inclusion probabilities correct active/stratified sampling as well.
    """

    if values.shape != tail_indicator.shape:
        raise R4ProtocolError("values and tail_indicator must have identical shapes")
    probability = float(tail_probability)
    if not 0.0 < probability <= 1.0:
        raise R4ProtocolError("tail_probability must lie in (0, 1]")
    indicator = tail_indicator.to(device=values.device, dtype=values.dtype)
    if not torch.all((indicator == 0) | (indicator == 1)):
        raise R4ProtocolError("tail_indicator must be binary")
    contribution = values * indicator
    if inclusion_probability is not None:
        inclusion = torch.as_tensor(
            inclusion_probability, device=values.device, dtype=values.dtype
        )
        if torch.any(inclusion <= 0) or torch.any(inclusion > 1):
            raise R4ProtocolError("inclusion_probability must lie in (0, 1]")
        contribution = contribution / inclusion
    return contribution.mean() / probability


def exact_weighted_cvar(costs: Tensor, weights: Tensor, *, alpha: float = 0.95) -> Tensor:
    """Exact discrete upper-tail CVaR, including the fractional VaR atom.

    The selected tail is normalized by exactly ``1-alpha`` rather than by the
    observed count or the uncorrected mass above a quantile.
    """

    if costs.ndim != 2 or weights.shape != costs.shape:
        raise R4ProtocolError("costs and weights must share [batch, particles] shape")
    if not 0.0 < alpha < 1.0:
        raise R4ProtocolError("alpha must lie in (0, 1)")
    if torch.any(weights < 0):
        raise R4ProtocolError("scenario weights must be nonnegative")
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    order = torch.argsort(costs, dim=-1, descending=True)
    sorted_costs = torch.gather(costs, -1, order)
    sorted_weights = torch.gather(weights, -1, order)
    tail_mass = 1.0 - float(alpha)
    cumulative_before = torch.cumsum(sorted_weights, dim=-1) - sorted_weights
    admitted = (tail_mass - cumulative_before).clamp(min=0.0)
    admitted = torch.minimum(admitted, sorted_weights)
    return (sorted_costs * admitted).sum(dim=-1) / tail_mass


@dataclass(frozen=True)
class CheapValidationPolicy:
    """Small deterministic validation used only for optimization feedback."""

    max_days: int = 16
    every_epochs: int = 1
    selection_metric: str = "loss"
    may_open_formal_selection: bool = False
    may_satisfy_capability_gate: bool = False


@dataclass(frozen=True)
class CapabilityGateResult:
    phase: str
    status: str
    evaluated_days: int
    horizon: int | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


@dataclass(frozen=True)
class EpochCoverage:
    epoch: int
    horizon: int
    physical_days: int
    effective_days: int
    optimizer_steps: int
    full_effective_batches: int
    partial_tail_days: int
    unique_operating_dates: int
    configured_effective_batch_days: int = EFFECTIVE_BATCH_SIZE
    optimizer_effective_batch_days: int = FORMAL_TRAINING_DAYS


def _parse_validation_dates(values: Sequence[str]) -> tuple[str, ...]:
    parsed: list[date] = []
    for value in values:
        try:
            item = date.fromisoformat(str(value)[:10])
        except ValueError as exc:
            raise R4ProtocolError(f"invalid capability-gate date: {value!r}") from exc
        if not VALIDATION_START <= item <= VALIDATION_END:
            raise R4ProtocolError(
                f"capability gate may use only 2023-H2 validation dates, got {item}"
            )
        parsed.append(item)
    if len(set(parsed)) != len(parsed):
        raise R4ProtocolError("capability-gate dates must be unique")
    return tuple(item.isoformat() for item in parsed)


class PhaseEndEvaluator(Protocol):
    def __call__(self, *, phase: str, horizon: int | None) -> Mapping[str, Any]: ...


def _batch_day_contract(batch: Any, *, physical_batch_size: int) -> tuple[int, tuple[str, ...]]:
    if not isinstance(batch, Mapping):
        raise R4ProtocolError(
            "formal R4 batches must be mappings with num_days and operating_dates"
        )
    if "num_days" not in batch or "operating_dates" not in batch:
        raise R4ProtocolError(
            "formal R4 batch is missing explicit num_days/operating_dates"
        )
    raw_days = batch["num_days"]
    if isinstance(raw_days, Tensor):
        if raw_days.numel() != 1:
            raise R4ProtocolError("batch num_days must be one scalar")
        raw_days = raw_days.item()
    if isinstance(raw_days, float) and not raw_days.is_integer():
        raise R4ProtocolError("batch num_days must be an integer")
    days = int(raw_days)
    if days < 1 or days > int(physical_batch_size):
        raise R4ProtocolError(
            f"batch num_days must lie in [1,{physical_batch_size}], got {days}"
        )
    raw_dates = batch["operating_dates"]
    if isinstance(raw_dates, (str, bytes)) or not isinstance(raw_dates, Sequence):
        raise R4ProtocolError("operating_dates must be a sequence")
    if len(raw_dates) != days:
        raise R4ProtocolError(
            f"num_days={days} but operating_dates has {len(raw_dates)} entries"
        )
    normalized: list[str] = []
    for value in raw_dates:
        try:
            parsed = date.fromisoformat(str(value)[:10])
        except ValueError as exc:
            raise R4ProtocolError(f"invalid training operating date: {value!r}") from exc
        if not TRAIN_START <= parsed <= TRAIN_END:
            raise R4ProtocolError(
                f"training batch contains date outside frozen train split: {parsed}"
            )
        normalized.append(parsed.isoformat())
    if len(set(normalized)) != len(normalized):
        raise R4ProtocolError("operating_dates are duplicated within one batch")
    return days, tuple(normalized)


def _require_tensor(mapping: Mapping[str, Any], names: Sequence[str], label: str) -> Tensor:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, Tensor):
            return value
    raise R4ProtocolError(f"missing required R4 {label}: expected one of {tuple(names)}")


def _as_loss_weights(config: Mapping[str, Any], phase: str) -> dict[str, float]:
    training = config.get("training", {})
    phases = training.get("phases", {}) if isinstance(training, Mapping) else {}
    entry = phases.get(phase, {}) if isinstance(phases, Mapping) else {}
    weights = entry.get("loss_weights", {}) if isinstance(entry, Mapping) else {}
    if not isinstance(weights, Mapping):
        raise R4ProtocolError(f"training.phases.{phase}.loss_weights must be a mapping")
    return {str(key): float(value) for key, value in weights.items()}


class PhaseLossComputer:
    """Strict adapter from R4 model outputs/batches to :mod:`e2e_moe_r4.losses`.

    Decision phases require costs produced by an actual differentiable
    rollout.  V5 retains its archived N=20 contract; V6 consumes an arbitrary
    N>=2 sample budget from the continuous Distribution head while Policy
    itself remains independent of that Monte Carlo axis.
    """

    def __init__(
        self,
        phase: str,
        config: Mapping[str, Any],
        *,
        fixed_training_cost_scale: float | Tensor | None = None,
        representation_scales: Mapping[str, float] | None = None,
        scenario_feature_scales: Sequence[float] | Tensor | None = None,
    ) -> None:
        self.phase = phase_spec(phase, config).name
        self.config = config
        self.is_distributional_architecture = (
            str(config.get("project", {}).get("architecture_version", ""))
            == "continuous_distributional_joint_moe_r4_v6"
        )
        self.distribution_contract: DistributionCurriculumContract | None = None
        self.distribution_stage_index: int | None = None
        self.distribution_sample_count: int | None = None
        if self.phase == "distribution" or (
            self.is_distributional_architecture
            and self.phase in {"joint_policy", "a1", "a2"}
        ):
            self.distribution_contract = distribution_curriculum_contract(config)
            if self.phase == "distribution":
                self.distribution_stage_index = 0
                self.distribution_sample_count = self.distribution_contract.train_samples
                self.weights = dict(self.distribution_contract.stages[0].loss_weights)
            else:
                self.weights = _as_loss_weights(config, self.phase)
        else:
            self.weights = _as_loss_weights(config, self.phase)
        self.fixed_training_cost_scale = fixed_training_cost_scale
        self.representation_scales = {
            str(name): float(value)
            for name, value in (representation_scales or {}).items()
        }
        for name, value in self.representation_scales.items():
            if not math.isfinite(value) or value <= 0.0:
                raise R4ProtocolError(
                    f"representation scale for {name!r} must be finite and positive"
                )
        if scenario_feature_scales is None:
            self.scenario_feature_scales: Tensor | None = None
        else:
            scales = torch.as_tensor(scenario_feature_scales, dtype=torch.float32).flatten()
            if scales.numel() != 21 or not bool(torch.isfinite(scales).all()) or bool(
                torch.any(scales <= 0)
            ):
                raise R4ProtocolError(
                    "scenario feature scales must be the 21 finite positive "
                    "train-residual scales frozen by the formal baseline artifact"
                )
            self.scenario_feature_scales = scales.detach().cpu()
        if (
            self.is_distributional_architecture
            and self.phase == "distribution"
            and self.scenario_feature_scales is None
        ):
            raise R4ProtocolError(
                f"{self.phase} training requires the 21 frozen residual score scales "
                "from its train-only contract"
            )
        if (
            self.is_distributional_architecture
            and self.phase in {"joint_policy", "a1", "a2"}
            and "distribution" in self.weights
            and self.scenario_feature_scales is None
        ):
            raise R4ProtocolError(
                "V6 decision phases preserving proper Distribution losses require "
                "the frozen 21-channel residual score scales"
            )
        if self.phase in {"policy", "joint_policy", "a1", "a2"} and fixed_training_cost_scale is None:
            raise R4ProtocolError(
                "decision phases require a frozen train-only cost scale"
            )

    def set_distribution_stage(self, stage_index: int) -> None:
        """Switch only preregistered scalar weights; formulas remain fixed."""

        if self.phase != "distribution" or self.distribution_contract is None:
            raise R4ProtocolError("only the Distribution loss has S1--S5 stages")
        index = int(stage_index)
        if not 0 <= index < len(self.distribution_contract.stages):
            raise R4ProtocolError("Distribution stage index is out of range")
        self.distribution_stage_index = index
        self.weights = dict(self.distribution_contract.stages[index].loss_weights)

    def set_distribution_sample_count(self, sample_count: int) -> None:
        if self.phase != "distribution" or self.distribution_contract is None:
            raise R4ProtocolError("only the Distribution loss has a sampling budget")
        count = int(sample_count)
        allowed = {
            self.distribution_contract.train_samples,
            self.distribution_contract.cheap_validation_samples,
        }
        if count not in allowed:
            raise R4ProtocolError(
                f"Distribution training/cheap sample count must be one of {sorted(allowed)}"
            )
        self.distribution_sample_count = count

    def __call__(
        self, model: nn.Module, batch: Any, horizon: int
    ) -> Mapping[str, Tensor]:
        if not isinstance(batch, Mapping):
            raise R4ProtocolError("formal R4 phase loss requires a mapping batch")
        if self.phase in {"policy", "joint_policy", "a1", "a2"}:
            rollout_forward = getattr(model, "forward_training_rollout", None)
            if rollout_forward is None:
                # Direct loss-adapter unit callers may supply already-realized
                # rollout targets.  Formal ``R4PhaseTrainer.run`` below still
                # requires the model's differentiable rollout entrypoint.
                outputs = model(batch)
            else:
                outputs = rollout_forward(batch, horizon)
        elif self.phase == "distribution":
            assert self.distribution_contract is not None
            assert self.distribution_sample_count is not None
            signature = inspect.signature(model.forward)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if "distribution_sample_count" in signature.parameters or accepts_kwargs:
                outputs = model(
                    batch,
                    distribution_sample_count=self.distribution_sample_count,
                )
            else:
                outputs = model(batch)
        else:
            outputs = model(batch)
        if not isinstance(outputs, Mapping):
            raise R4ProtocolError("R4 model output must be a mapping")
        if self.phase == "representation":
            return self._representation(outputs, batch)
        if self.phase == "distribution":
            return self._distribution(model, outputs, batch)
        if self.phase == "scenario":
            return self._scenario(outputs, batch)
        return self._decision(outputs, batch)

    def _distribution(
        self,
        model: nn.Module,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> Mapping[str, Tensor]:
        """V6 continuous Distribution loss at the preregistered train N=32."""

        assert self.distribution_contract is not None
        distribution = outputs.get("distribution")
        if not isinstance(distribution, Mapping):
            raise R4ProtocolError("Distribution phase requires outputs['distribution']")
        samples = distribution.get("samples_zonal", distribution.get("samples"))
        if not isinstance(samples, Tensor):
            sampler = getattr(model, "sample_distribution", None)
            if not callable(sampler):
                raise R4ProtocolError(
                    "Distribution output has no samples and model has no sample_distribution API"
                )
            samples = sampler(
                outputs,
                int(self.distribution_sample_count or self.distribution_contract.train_samples),
            )
            if isinstance(samples, Mapping):
                distribution = {**dict(distribution), **dict(samples)}
                samples = distribution.get("samples_zonal", distribution.get("samples"))
        if not isinstance(samples, Tensor):
            raise R4ProtocolError("Distribution sampler did not return a sample tensor")
        expected_n = int(
            self.distribution_sample_count or self.distribution_contract.train_samples
        )
        if samples.ndim != 4 or samples.shape[1:] != (expected_n, 96, 21):
            raise R4ProtocolError(
                f"Distribution training samples must be [batch,{expected_n},96,21]"
            )
        target = _require_tensor(
            batch, ("distribution_target_zonal", "scenario_target_zonal", "scenario_target"),
            "Distribution target",
        )
        if target.shape != (samples.shape[0], 96, 21):
            raise R4ProtocolError("Distribution target must be [batch,96,21]")
        raw_weights = distribution.get("weights")
        uniform_weights = r4_losses.uniform_sample_weights(samples)
        if raw_weights is None:
            weights = uniform_weights
        elif isinstance(raw_weights, Tensor) and raw_weights.shape == samples.shape[:2]:
            # V6 sampling is i.i.d.; model-reported weights are compatibility
            # metadata, never learned importance weights.  The formal
            # preflight/Level2 gate verifies their exact 1/N value once.  The
            # epoch hot path uses the structural uniform tensor directly.
            weights = uniform_weights
        else:
            raise R4ProtocolError("Distribution weights must be absent or [batch,N]")
        assert self.scenario_feature_scales is not None
        feature_scales = self.scenario_feature_scales.to(
            device=samples.device, dtype=samples.dtype
        )
        score_samples = samples / feature_scales
        score_target = target.to(samples) / feature_scales
        center = distribution.get("center_zonal")
        scale = distribution.get("scale_zonal")
        score_center: Tensor | None = None
        score_scale: Tensor | None = None
        if center is not None:
            if not isinstance(center, Tensor) or center.shape != target.shape:
                raise R4ProtocolError("Distribution center_zonal must be [batch,96,21]")
            score_center = center.to(samples) / feature_scales
        if scale is not None:
            if (
                not isinstance(scale, Tensor)
                or scale.shape != target.shape
            ):
                raise R4ProtocolError(
                    "Distribution scale_zonal must be [batch,96,21]"
                )
            score_scale = scale.to(samples) / feature_scales
        terms = r4_losses.distribution_proper_loss_terms(
            score_samples,
            score_target,
            weights=weights,
            center=score_center,
            scale=score_scale,
            identity_samples=samples,
            validate_inputs=False,
        )
        configured = {name: terms[name] for name in self.weights}
        with torch.no_grad():
            identity_max = (
                samples[..., 0:4] - samples[..., 4:8] - samples[..., 8:12]
            ).abs().flatten(1).amax(1)
        stage = self.distribution_contract.stages[
            int(self.distribution_stage_index or 0)
        ]
        return {
            "total": r4_losses.compose_weighted_loss(configured, self.weights),
            **terms,
            "coverage90_error": terms["coverage90"].detach(),
            "coverage95_error": terms["coverage95"].detach(),
            "identity_max_abs": identity_max,
            "sample_count": samples.new_full((samples.shape[0],), float(expected_n)),
            "stage_index": samples.new_full(
                (samples.shape[0],), float(self.distribution_stage_index or 0)
            ),
        }

    def _representation(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any]
    ) -> Mapping[str, Tensor]:
        forecasts = outputs.get("forecasts", outputs.get("forecast_legacy"))
        targets = batch.get("targets", batch.get("forecast_targets"))
        if not isinstance(forecasts, Mapping) or not isinstance(targets, Mapping):
            raise R4ProtocolError("representation requires named forecasts and targets")
        terms: dict[str, Tensor] = {}
        for name in self.weights:
            if name not in forecasts or name not in targets:
                raise R4ProtocolError(f"representation target missing for {name}")
            metrics = r4_losses.forecast_raw_metrics(forecasts[name], targets[name])
            scale = float(self.representation_scales.get(name, 1.0))
            # Capability is expressed as a head-wise normalized error.  Use
            # one immutable train-only scale per head here so price magnitudes
            # cannot suppress the PV/PUE gradients in the shared trunk.  Raw
            # MAE remains available as a diagnostic but never enters ``total``.
            terms[name] = metrics["mae"] / scale
            terms[f"{name}_raw_mae"] = metrics["mae"]
        return {"total": r4_losses.compose_weighted_loss(terms, self.weights), **terms}

    def _scenario(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any]
    ) -> Mapping[str, Tensor]:
        scenario = outputs.get("scenarios")
        if not isinstance(scenario, Mapping):
            raise R4ProtocolError("scenario phase requires model scenarios")
        particles = _require_tensor(
            scenario, ("particles_zonal", "particles", "particles_legacy"), "particles"
        )
        weights = _require_tensor(scenario, ("weights",), "scenario weights")
        target = _require_tensor(
            batch, ("scenario_target_zonal", "scenario_target"), "scenario target"
        )
        score_particles = particles
        score_target = target
        if self.scenario_feature_scales is not None:
            if particles.shape[-1] != 21 or target.shape[-1] != 21:
                raise R4ProtocolError(
                    "scenario feature scaling requires the registered 21-variable output"
                )
            scales = self.scenario_feature_scales.to(
                device=particles.device, dtype=particles.dtype
            )
            score_particles = particles / scales
            score_target = target / scales.to(target)
        terms: dict[str, Tensor] = {
            "energy_score": r4_losses.weighted_energy_score(
                score_particles, weights, score_target
            ),
            "marginal_crps": r4_losses.weighted_marginal_crps(
                score_particles, weights, score_target
            ),
            "variogram": r4_losses.weighted_variogram_score_stable(
                score_particles, weights, score_target
            ),
        }
        terms["coverage"] = self._scenario_coverage_surrogate(
            score_particles, weights, score_target
        )
        terms["tail"] = self._scenario_tail_score(
            score_particles, weights, score_target
        )
        diversity = r4_losses.mean_pairwise_particle_distance(score_particles)
        terms["diversity"] = 1.0 / diversity.clamp_min(1e-6)
        normalized = r4_losses.normalize_weights(weights)
        terms["weight_entropy"] = (
            normalized * normalized.clamp_min(1e-8).log()
        ).sum(-1).add(math.log(20.0)).abs()
        # Scenario training precedes the canonical policy rollout.  Driving an
        # untrained/frozen policy's expected cost toward zero changes the
        # probabilistic target and is not a valid decision-consistency loss.
        # Keep the configured term explicitly inert until a checkpoint-bound
        # canonical rollout target exists.
        terms["decision_consistency"] = particles.flatten(2).sum((1, 2)) * 0.0
        configured = {name: terms[name] for name in self.weights if name in terms}
        missing = set(self.weights) - set(configured)
        if missing:
            raise R4ProtocolError(f"scenario loss terms unavailable: {sorted(missing)}")
        return {"total": r4_losses.compose_weighted_loss(configured, self.weights), **configured}

    @staticmethod
    def _scenario_tail_score(
        particles: Tensor, weights: Tensor, target: Tensor
    ) -> Tensor:
        """Proper lower/upper quantile score, distinct from coverage error."""

        scores: list[Tensor] = []
        for quantile in (0.05, 0.95):
            forecast = r4_losses.weighted_quantile(particles, weights, quantile)
            error = target - forecast
            pinball = torch.maximum(
                float(quantile) * error,
                (float(quantile) - 1.0) * error,
            )
            scores.append(pinball.flatten(1).mean(1))
        return torch.stack(scores, dim=-1).mean(-1)

    @staticmethod
    def _scenario_coverage_surrogate(
        particles: Tensor, weights: Tensor, target: Tensor
    ) -> Tensor:
        """Differentiable proxy aligned with the exact 90/95% gate.

        A hard comparison such as ``target >= lower`` has no autograd path and
        cannot calibrate a scenario generator.  For each registered level we
        therefore combine (i) the absolute error of a logistic soft-coverage
        estimate and (ii) a scaled proper interval score.  Multiplying the
        interval score by ``alpha / 2`` keeps the two levels in comparable
        units while retaining the proper score's width and miss penalties.
        Exact Boolean coverage remains confined to the formal evaluator.
        """

        losses: list[Tensor] = []
        for level in (0.90, 0.95):
            smooth_coverage, _ = r4_losses.smooth_empirical_coverage(
                particles,
                weights,
                target,
                level,
                temperature=0.05,
            )
            interval = r4_losses.weighted_interval_score(
                particles, weights, target, level
            )
            alpha = 1.0 - float(level)
            losses.append(
                (smooth_coverage - float(level)).abs()
                + (alpha / 2.0) * interval
            )
        return torch.stack(losses, dim=-1).mean(-1)

    def _decision(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any]
    ) -> Mapping[str, Tensor]:
        if self.is_distributional_architecture:
            return self._distributional_decision(outputs, batch)
        rollout_inputs = batch.get("rollout_targets")
        rollout = outputs.get("training_rollout")
        if not isinstance(rollout_inputs, Mapping):
            raise R4ProtocolError(
                "policy/A1/A2 require actual rollout targets and costs; proxies are forbidden"
            )
        if not isinstance(rollout, Mapping):
            # Direct loss-adapter tests may supply a precomputed mapping.  The
            # formal trainer rejects models without forward_training_rollout,
            # so this compatibility seam cannot enter a formal phase run.
            rollout = rollout_inputs
        risk = outputs.get("risk")
        scenario = outputs.get("scenarios")
        if not isinstance(risk, Mapping) or not isinstance(scenario, Mapping):
            raise R4ProtocolError(
                "decision phases require equation-based model risk and scenarios"
            )
        particle_costs = _require_tensor(
            risk, ("scenario_costs", "particle_costs"), "20-particle rollout costs"
        )
        particle_weights = _require_tensor(
            scenario, ("weights", "scenario_weights"), "20-particle rollout weights"
        )
        if particle_costs.ndim != 2 or particle_costs.shape[-1] != 20:
            raise R4ProtocolError("actual rollout particle_costs must be [batch,20]")
        realized_cost = _require_tensor(
            rollout, ("realized_settlement_cost", "realized_cost"), "realized rollout cost"
        )
        expected = (r4_losses.normalize_weights(particle_weights).to(particle_costs) * particle_costs).sum(-1)
        cvar = r4_losses.weighted_cvar(particle_costs, particle_weights, alpha=0.95)
        scale = self.fixed_training_cost_scale
        assert scale is not None
        terms: dict[str, Tensor] = {
            "realized_settlement": realized_cost,
            "expected_particle_cost": expected,
            "particle_cvar": cvar,
        }
        if self.phase == "a1":
            forecasts = outputs.get("forecasts", {})
            targets = batch.get("targets", {})
            if not isinstance(forecasts, Mapping) or not isinstance(targets, Mapping):
                raise R4ProtocolError("A1 requires forecast targets for DART supervision")
            if "dart_supervised" in self.weights:
                if not isinstance(forecasts.get("dart"), Tensor) or not isinstance(targets.get("dart"), Tensor):
                    raise R4ProtocolError("A1 requires a DART forecast and DART target")
                terms["dart_supervised"] = r4_losses.forecast_raw_metrics(
                    forecasts["dart"], targets["dart"]
                )["mae"] / float(self.representation_scales.get("dart", 1.0))
            if "baseline_action_consistency" in self.weights:
                actions = outputs.get("actions", outputs.get("policy", {}))
                baseline = rollout_inputs.get("baseline_actions")
                if not isinstance(actions, Mapping) or not isinstance(baseline, Mapping):
                    raise R4ProtocolError("A1 requires frozen baseline_actions")
                distances: list[Tensor] = []
                for name in ("day_ahead", "battery", "compute"):
                    if isinstance(actions.get(name), Tensor) and isinstance(baseline.get(name), Tensor):
                        distances.append((actions[name] - baseline[name]).abs().flatten(1).mean(1))
                if not distances:
                    raise R4ProtocolError("A1 baseline_actions have no comparable tensors")
                terms["baseline_action_consistency"] = torch.stack(distances, -1).mean(-1)
            if "cost_noninferiority" in self.weights:
                baseline_cost = _require_tensor(
                    rollout_inputs, ("baseline_realized_cost",), "A1 baseline realized cost"
                )
                terms["cost_noninferiority"] = F.relu(realized_cost - baseline_cost)

        if self.phase == "a2":
            if "representation" in self.weights:
                rep_weights = _as_loss_weights(self.config, "representation")
                forecasts = outputs.get("forecasts", {})
                targets = batch.get("targets", {})
                if not isinstance(forecasts, Mapping) or not isinstance(targets, Mapping):
                    raise R4ProtocolError("A2 representation consistency requires targets")
                rep_terms: dict[str, Tensor] = {}
                for name in rep_weights:
                    if not isinstance(forecasts.get(name), Tensor) or not isinstance(targets.get(name), Tensor):
                        raise R4ProtocolError(f"A2 representation target missing for {name}")
                    raw_mae = r4_losses.forecast_raw_metrics(
                        forecasts[name], targets[name]
                    )["mae"]
                    rep_terms[name] = raw_mae / float(
                        self.representation_scales.get(name, 1.0)
                    )
                terms["representation"] = r4_losses.compose_weighted_loss(
                    rep_terms, rep_weights
                ).reshape(1)
            if "scenario" in self.weights:
                scenario_weights_config = _as_loss_weights(self.config, "scenario")
                scenario = outputs.get("scenarios", {})
                if not isinstance(scenario, Mapping):
                    raise R4ProtocolError("A2 scenario consistency requires particles")
                particles = _require_tensor(
                    scenario, ("particles_zonal", "particles", "particles_legacy"), "A2 particles"
                )
                scenario_weights = _require_tensor(scenario, ("weights",), "A2 scenario weights")
                target = _require_tensor(
                    batch, ("scenario_target_zonal", "scenario_target"), "A2 scenario target"
                )
                score_particles = particles
                score_target = target
                if self.scenario_feature_scales is not None:
                    if particles.shape[-1] != 21 or target.shape[-1] != 21:
                        raise R4ProtocolError(
                            "A2 scenario scaling requires the registered 21-variable output"
                        )
                    scales = self.scenario_feature_scales.to(
                        device=particles.device, dtype=particles.dtype
                    )
                    score_particles = particles / scales
                    score_target = target / scales.to(target)
                scenario_terms: dict[str, Tensor] = {
                    "energy_score": r4_losses.weighted_energy_score(
                        score_particles, scenario_weights, score_target
                    ),
                    "marginal_crps": r4_losses.weighted_marginal_crps(
                        score_particles, scenario_weights, score_target
                    ),
                    "variogram": r4_losses.weighted_variogram_score_stable(
                        score_particles, scenario_weights, score_target
                    ),
                }
                scenario_terms["coverage"] = self._scenario_coverage_surrogate(
                    score_particles, scenario_weights, score_target
                )
                scenario_terms["tail"] = self._scenario_tail_score(
                    score_particles, scenario_weights, score_target
                )
                scenario_terms["diversity"] = 1.0 / r4_losses.mean_pairwise_particle_distance(
                    score_particles
                ).clamp_min(1e-6)
                normalized_weights = r4_losses.normalize_weights(scenario_weights)
                scenario_terms["weight_entropy"] = (
                    normalized_weights * normalized_weights.clamp_min(1e-8).log()
                ).sum(-1).add(math.log(20.0)).abs()
                # No canonical checkpoint-bound counterfactual target exists
                # during A2 gradient epochs, so this term remains fail-safe.
                scenario_terms["decision_consistency"] = (
                    particles.flatten(2).sum((1, 2)) * 0.0
                )
                terms["scenario"] = r4_losses.compose_weighted_loss(
                    {name: scenario_terms[name] for name in scenario_weights_config},
                    scenario_weights_config,
                ).reshape(1)
        optional = {
            "normalized_sla": ("normalized_sla", "sla_shortfall"),
            "battery_degradation": ("battery_degradation",),
            "raw_envelope_violation": ("raw_envelope_violation",),
            "corrective_intervention": ("corrective_intervention", "intervention"),
            "value_consistency": ("value_consistency",),
        }
        for name, aliases in optional.items():
            if name in self.weights:
                terms[name] = _require_tensor(rollout, aliases, name)
        moe = outputs.get("moe", outputs.get("moe_aux", {}))
        if "router_balance" in self.weights:
            if not isinstance(moe, Mapping):
                raise R4ProtocolError("router_balance requires R4 MoE outputs")
            terms["router_balance"] = r4_losses.router_balance_loss(
                _require_tensor(moe, ("router_probabilities",), "router probabilities"),
                _require_tensor(moe, ("topk_indices",), "Top-2 indices"),
            )
        if "router_stability" in self.weights:
            if not isinstance(moe, Mapping):
                raise R4ProtocolError("router_stability requires R4 MoE outputs")
            current = _require_tensor(moe, ("router_probabilities",), "router probabilities")
            previous = rollout.get("previous_router_probabilities")
            terms["router_stability"] = r4_losses.router_stability_loss(current, previous)
        configured: dict[str, Tensor] = {}
        for name in self.weights:
            if name not in terms:
                raise R4ProtocolError(f"decision rollout term missing: {name}")
            if name in {"realized_settlement", "expected_particle_cost", "particle_cvar"}:
                configured[name] = r4_losses.fixed_scale_cost_loss(
                    terms[name], scale
                ).reshape(1)
            else:
                configured[name] = terms[name]
        total = r4_losses.compose_weighted_loss(configured, self.weights)
        return {"total": total, **configured}

    def _distributional_decision(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any]
    ) -> Mapping[str, Tensor]:
        """V6 joint loss with arbitrary-N simulation and preserved proper scores."""

        rollout_inputs = batch.get("rollout_targets")
        rollout = outputs.get("training_rollout")
        distribution = outputs.get("distribution")
        risk = outputs.get("risk")
        if not isinstance(rollout_inputs, Mapping) or not isinstance(rollout, Mapping):
            raise R4ProtocolError("V6 decision phases require differentiable rollout targets")
        if not isinstance(distribution, Mapping) or not isinstance(risk, Mapping):
            raise R4ProtocolError(
                "V6 decision phases require a continuous Distribution sample and risk output"
            )
        samples = _require_tensor(
            distribution,
            ("samples_zonal", "particles_zonal", "samples", "particles"),
            "V6 Distribution samples",
        )
        weights = _require_tensor(
            distribution, ("weights", "sample_weights"), "V6 sample weights"
        )
        particle_costs = _require_tensor(
            risk,
            ("scenario_costs", "particle_costs"),
            "V6 sampled rollout costs",
        )
        if samples.ndim != 4 or samples.shape[1] < 2 or samples.shape[2:] != (96, 21):
            raise R4ProtocolError(
                "V6 Distribution samples must be [batch,N,96,21] with N>=2"
            )
        if weights.shape != samples.shape[:2] or particle_costs.shape != weights.shape:
            raise R4ProtocolError(
                "V6 sampled costs and weights must both match [batch,N]"
            )
        normalized_weights = r4_losses.normalize_weights(weights).to(particle_costs)
        expected_cost = (normalized_weights * particle_costs).sum(-1)
        cvar = r4_losses.weighted_cvar(
            particle_costs, normalized_weights, alpha=0.95
        )
        realized_cost = _require_tensor(
            rollout,
            ("realized_settlement_cost", "realized_cost"),
            "V6 realized rollout cost",
        )
        scale = self.fixed_training_cost_scale
        assert scale is not None

        terms: dict[str, Tensor] = {
            "realized_cost": r4_losses.fixed_scale_cost_loss(
                realized_cost, scale
            ),
            "sla": _require_tensor(
                rollout, ("normalized_sla", "sla_shortfall"), "normalized SLA"
            ),
            "shield_intervention": _require_tensor(
                rollout,
                ("corrective_intervention", "raw_envelope_violation", "intervention"),
                "shield intervention",
            ),
        }

        neural_risk = risk.get("neural_risk")
        neural_alignment = particle_costs.sum() * 0.0
        if isinstance(neural_risk, Mapping):
            neural_cvar = neural_risk.get("neural_cvar95_proxy")
            neural_expected = neural_risk.get("neural_expected_cost_proxy")
            alignments: list[Tensor] = []
            scale_tensor = torch.as_tensor(
                scale, device=particle_costs.device, dtype=particle_costs.dtype
            )
            if isinstance(neural_cvar, Tensor) and neural_cvar.shape == cvar.shape:
                alignments.append(
                    F.smooth_l1_loss(
                        neural_cvar / scale_tensor,
                        cvar.detach() / scale_tensor,
                    )
                )
            if (
                isinstance(neural_expected, Tensor)
                and neural_expected.shape == expected_cost.shape
            ):
                alignments.append(
                    F.smooth_l1_loss(
                        neural_expected / scale_tensor,
                        expected_cost.detach() / scale_tensor,
                    )
                )
            if alignments:
                neural_alignment = torch.stack(alignments).mean()
        terms["risk"] = (
            r4_losses.fixed_scale_cost_loss(cvar, scale) + neural_alignment
        )

        if "distribution" in self.weights:
            target = _require_tensor(
                batch,
                ("distribution_target_zonal", "scenario_target_zonal", "scenario_target"),
                "V6 Distribution target",
            )
            if target.shape != samples.shape[:1] + samples.shape[2:]:
                raise R4ProtocolError("V6 Distribution target must be [batch,96,21]")
            assert self.scenario_feature_scales is not None
            feature_scales = self.scenario_feature_scales.to(
                device=samples.device, dtype=samples.dtype
            )
            center = distribution.get("center_zonal")
            distribution_scale = distribution.get("scale_zonal")
            score_center = (
                None
                if center is None
                else torch.as_tensor(center).to(samples) / feature_scales
            )
            score_scale = (
                None
                if distribution_scale is None
                else torch.as_tensor(distribution_scale).to(samples) / feature_scales
            )
            proper = r4_losses.distribution_proper_loss_terms(
                samples / feature_scales,
                target.to(samples) / feature_scales,
                weights=normalized_weights.to(samples),
                center=score_center,
                scale=score_scale,
                identity_samples=samples,
                validate_inputs=False,
            )
            if self.distribution_contract is None:
                raise R4ProtocolError("V6 decision loss lacks its Distribution curriculum")
            proper_weights = self.distribution_contract.stages[-1].loss_weights
            terms["distribution"] = r4_losses.compose_weighted_loss(
                proper, proper_weights
            )

        moe = outputs.get("moe", outputs.get("moe_aux"))
        if "router_balance" in self.weights:
            if not isinstance(moe, Mapping):
                raise R4ProtocolError("V6 router_balance requires MoE outputs")
            terms["router_balance"] = r4_losses.router_balance_loss(
                _require_tensor(
                    moe, ("router_probabilities",), "router probabilities"
                ),
                _require_tensor(moe, ("topk_indices",), "Top-2 indices"),
            )
        if "noninferiority" in self.weights:
            baseline_cost = _require_tensor(
                rollout_inputs,
                ("baseline_realized_cost",),
                "A1 baseline realized cost",
            )
            terms["noninferiority"] = r4_losses.fixed_scale_cost_loss(
                F.relu(realized_cost - baseline_cost), scale
            )

        missing = sorted(set(self.weights) - set(terms))
        if missing:
            raise R4ProtocolError(
                f"V6 decision rollout terms unavailable: {missing}"
            )
        configured = {name: terms[name] for name in self.weights}
        total = r4_losses.compose_weighted_loss(configured, self.weights)
        return {
            "total": total,
            **configured,
            "expected_sampled_cost": expected_cost.mean(),
            "sampled_cvar95": cvar.mean(),
            "sample_count": samples.new_tensor(float(samples.shape[1])),
            "neural_risk_alignment": neural_alignment,
        }


def build_phase_loss(
    phase: str,
    config: Mapping[str, Any],
    *,
    fixed_training_cost_scale: float | Tensor | None = None,
    representation_scales: Mapping[str, float] | None = None,
    scenario_feature_scales: Sequence[float] | Tensor | None = None,
) -> PhaseLossComputer:
    return PhaseLossComputer(
        phase,
        config,
        fixed_training_cost_scale=fixed_training_cost_scale,
        representation_scales=representation_scales,
        scenario_feature_scales=scenario_feature_scales,
    )


def run_capability_gate(
    phase: str,
    evaluator: PhaseEndEvaluator | None,
    *,
    checkpoint: Path | None = None,
    epoch: int | None = None,
    model: nn.Module | None = None,
    checkpoint_sha256: str | None = None,
) -> CapabilityGateResult:
    """Run a phase-end gate against the explicitly selected checkpoint.

    ``checkpoint``/``epoch``/``model`` are optional for backwards-compatible
    standalone validation.  The formal trainer always supplies all three, so
    an evaluator measures the reloaded immutable cheap-validation best rather
    than whichever weights happen to remain in memory after the last epoch.
    """

    spec = phase_spec(phase)
    required_days = spec.capability_days
    if evaluator is None:
        return CapabilityGateResult(
            phase=spec.name,
            status="NOT_READY",
            evaluated_days=0,
            horizon=96 if required_days is not None else spec.hard_gate_horizon,
            reasons=("phase-end capability evaluator is not connected",),
        )
    evaluation_horizon = (
        spec.hard_gate_horizon
        if spec.hard_gate_horizon is not None
        else (96 if required_days is not None else None)
    )
    if checkpoint is not None:
        if epoch is None or model is None:
            raise R4ProtocolError(
                "checkpoint capability evaluation requires epoch and model"
            )
        raw = dict(
            _invoke_candidate_evaluator(
                evaluator,
                phase=spec.name,
                checkpoint=checkpoint,
                epoch=epoch,
                model=model,
                horizon=evaluation_horizon,
                checkpoint_sha256=checkpoint_sha256,
            )
        )
    else:
        raw = dict(evaluator(phase=spec.name, horizon=evaluation_horizon))
    dates = _parse_validation_dates(tuple(raw.get("dates", ())))
    metrics_raw = raw.get("metrics", {})
    if not isinstance(metrics_raw, Mapping):
        raise R4ProtocolError("gate metrics must be a mapping")
    metrics = {str(key): float(value) for key, value in metrics_raw.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        return CapabilityGateResult(
            spec.name,
            "FAIL",
            len(dates),
            evaluation_horizon,
            metrics,
            ("non-finite gate metric",),
            dates,
        )

    reasons: list[str] = []
    if required_days is not None and len(dates) != required_days:
        reasons.append(
            f"requires exactly {required_days} validation days; received {len(dates)}"
        )
    if spec.hard_gate_horizon is not None:
        observed_horizon = raw.get("horizon")
        if int(observed_horizon or -1) != spec.hard_gate_horizon:
            reasons.append(
                f"hard gate requires K={spec.hard_gate_horizon}; got {observed_horizon}"
            )
        if not bool(raw.get("hard_shield", False)):
            reasons.append("hard-shield evaluation was not used")
    evaluator_passed = raw.get("passed")
    if evaluator_passed is not True:
        evaluator_reasons = raw.get("reasons", ())
        if isinstance(evaluator_reasons, (list, tuple)):
            reasons.extend(
                str(reason) for reason in evaluator_reasons if str(reason).strip()
            )
        if not evaluator_reasons:
            reasons.append("evaluator did not report an explicit pass")
    return CapabilityGateResult(
        phase=spec.name,
        status="FAIL" if reasons else "PASS",
        evaluated_days=len(dates),
        horizon=evaluation_horizon,
        metrics=metrics,
        reasons=tuple(reasons),
        dates=dates,
    )


def _invoke_candidate_evaluator(
    evaluator: PhaseEndEvaluator,
    *,
    phase: str,
    checkpoint: Path,
    epoch: int,
    model: nn.Module,
    horizon: int | None = 96,
    checkpoint_sha256: str | None = None,
) -> Mapping[str, Any]:
    available = {
        "phase": phase,
        "horizon": horizon,
        "checkpoint": str(checkpoint),
        "checkpoint_path": str(checkpoint),
        "epoch": int(epoch),
        "model": model,
        "checkpoint_sha256": checkpoint_sha256,
    }
    try:
        signature = inspect.signature(evaluator)
    except (TypeError, ValueError):
        return evaluator(**available)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = (
        available
        if accepts_kwargs
        else {name: value for name, value in available.items() if name in signature.parameters}
    )
    return evaluator(**kwargs)


def _finite_metric(raw: Mapping[str, Any], metrics: Mapping[str, Any], *names: str) -> float:
    value: Any = None
    for name in names:
        if name in metrics:
            value = metrics[name]
            break
        if name in raw:
            value = raw[name]
            break
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise R4ProtocolError(f"candidate metric {names[0]} must be scalar")
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise R4ProtocolError(f"candidate gate omitted finite metric {names[0]}") from exc
    if not math.isfinite(result):
        raise R4ProtocolError(f"candidate metric {names[0]} is non-finite")
    return result


def _boolean_field(
    raw: Mapping[str, Any], metrics: Mapping[str, Any], *names: str
) -> bool:
    for name in names:
        if name in raw:
            return bool(raw[name])
        if name in metrics:
            return bool(metrics[name])
    return False


def validate_decision_candidate_gate(
    phase: str,
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one full 179-day/17184-origin K96 hard-shield result."""

    raw = dict(raw_result)
    metrics_value = raw.get("metrics", {})
    if not isinstance(metrics_value, Mapping):
        raise R4ProtocolError("candidate gate metrics must be a mapping")
    metrics = dict(metrics_value)
    dates = _parse_validation_dates(tuple(raw.get("dates", ())))
    reasons: list[str] = []
    if len(dates) != CAPABILITY_GATE_DAYS:
        reasons.append(
            f"requires exactly {CAPABILITY_GATE_DAYS} validation days; got {len(dates)}"
        )
    origins = int(
        _finite_metric(raw, metrics, "validation_origins", "completed_origins")
    )
    if origins != HARD_GATE_VALIDATION_ORIGINS:
        reasons.append(
            f"requires {HARD_GATE_VALIDATION_ORIGINS} validation origins; got {origins}"
        )
    if int(raw.get("horizon", -1)) != 96:
        reasons.append("candidate hard gate requires K=96")
    if not _boolean_field(raw, metrics, "hard_shield", "hard_shield_used"):
        reasons.append("candidate was not evaluated through the hard shield")
    if not _boolean_field(
        raw,
        metrics,
        "all_actions_independently_validated",
        "independent_audit_passed",
    ):
        reasons.append("independent action/residual audit did not pass")
    if not _boolean_field(raw, metrics, "finite", "all_metrics_finite"):
        reasons.append("candidate did not attest finite metrics")
    feasibility = _finite_metric(
        raw, metrics, "feasibility_fraction", "post_shield_feasibility_fraction"
    )
    if feasibility != 1.0:
        reasons.append(f"post-shield feasibility must equal 1.0; got {feasibility}")
    rejected = int(
        _finite_metric(
            raw,
            metrics,
            "non_exogenous_rejected_origins",
            "rejected_origins_non_exogenous",
        )
    )
    if rejected != 0:
        reasons.append(f"non-exogenous rejected origins must equal zero; got {rejected}")
    sla = _finite_metric(
        raw, metrics, "normalized_sla_increase", "normalized_sla_shortfall_increase"
    )
    if sla > 0.001:
        reasons.append(f"normalized SLA increase exceeds 0.001: {sla}")
    cost_ci = _finite_metric(
        raw,
        metrics,
        "paired_daily_cost_upper_ci",
        "paired_daily_cost_difference_upper_ci",
    )
    if cost_ci > 0.0:
        reasons.append(f"paired daily cost upper CI must be <= 0: {cost_ci}")
    if not _boolean_field(raw, metrics, "day_ahead_frozen_after_origin_zero"):
        reasons.append("day-ahead schedule was not frozen after origin zero")
    if not _boolean_field(raw, metrics, "finite_and_resumable", "resumable"):
        reasons.append("candidate/checkpoint was not finite and resumable")

    if phase == "a1":
        required = (
            "representation_dart_gate_retained",
            "allowed_gradients_finite_nonzero",
            "frozen_parameters_unchanged",
            "policy_cost_and_sla_noninferiority",
        )
        reasons.extend(
            f"A1 requirement did not pass: {name}"
            for name in required
            if not _boolean_field(raw, metrics, name)
        )
    if phase == "a2" and not _boolean_field(
        raw, metrics, "inherit_full_policy_gate_without_relaxation"
    ):
        reasons.append("A2 did not inherit the full policy gate without relaxation")

    operating_cost = _finite_metric(
        raw, metrics, "operating_cost", "mean_operating_cost", "realized_operating_cost"
    )
    cvar = _finite_metric(raw, metrics, "cvar", "cvar95", "particle_cvar")
    correction = _finite_metric(
        raw,
        metrics,
        "corrective_magnitude",
        "mean_corrective_magnitude",
        "constraint_intervention_magnitude",
    )
    if raw.get("passed") is not True:
        reasons.append("candidate evaluator did not explicitly report passed=true")
    return {
        "status": "FAIL" if reasons else "PASS",
        "phase": phase,
        "horizon": 96,
        "evaluated_days": len(dates),
        "validation_origins": origins,
        "dates": list(dates),
        "metrics": {
            "operating_cost": operating_cost,
            "cvar": cvar,
            "corrective_magnitude": correction,
            "feasibility_fraction": feasibility,
            "non_exogenous_rejected_origins": rejected,
            "normalized_sla_increase": sla,
            "paired_daily_cost_upper_ci": cost_ci,
        },
        "reasons": reasons,
    }


def _checkpoint_scenario_contract(
    payload: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Validate and normalize an optional scenario-training contract binding."""

    raw_path = payload.get("scenario_training_contract_path")
    raw_sha256 = payload.get("scenario_training_contract_sha256")
    if raw_path is None and raw_sha256 is None:
        return None, None
    if raw_path is None or raw_sha256 is None:
        raise R4ProtocolError(
            "scenario-training contract path and SHA-256 must be supplied together"
        )
    digest = str(raw_sha256).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise R4ProtocolError(
            "scenario_training_contract_sha256 must be one lowercase SHA-256"
        )
    path = Path(str(raw_path)).resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise R4ProtocolError(
            "scenario-training contract artifact is missing or hash-mismatched"
        )
    return str(path), digest


def _checkpoint_distribution_contract(
    payload: Mapping[str, Any], *, verify_content: bool = False
) -> tuple[str | None, str | None]:
    """Normalize the V6 contract with one optional preflight hash read.

    Epoch checkpoints carry the already-frozen digest and never rehash the
    manifest.  ``verify_content=True`` is used exactly once when the trainer is
    created, before the first optimizer update.
    """

    raw_path = payload.get("distribution_training_contract_path")
    raw_sha256 = payload.get("distribution_training_contract_sha256")
    if raw_path is None and raw_sha256 is None:
        return None, None
    if raw_path is None or raw_sha256 is None:
        raise R4ProtocolError(
            "distribution-training contract path and SHA-256 must be supplied together"
        )
    digest = str(raw_sha256).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise R4ProtocolError(
            "distribution_training_contract_sha256 must be one lowercase SHA-256"
        )
    path = Path(str(raw_path)).resolve()
    if not path.is_file():
        raise R4ProtocolError("distribution-training contract artifact is missing")
    if verify_content and sha256_file(path) != digest:
        raise R4ProtocolError(
            "distribution-training contract artifact is hash-mismatched"
        )
    return str(path), digest


class AtomicRunStore:
    """Atomic, lineage-checked JSON state and Torch checkpoints."""

    def __init__(
        self,
        root: str | Path,
        *,
        lineage_id: str,
        config_sha256: str,
        config_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.lineage_id = str(lineage_id)
        self.config_sha256 = str(config_sha256)
        self.config_path = None if config_path is None else str(Path(config_path).resolve())

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def write_state(self, state: Mapping[str, Any]) -> Path:
        incoming = dict(state)
        internal_status = str(incoming.get("status", "PARTIAL"))
        phase = str(incoming.get("phase", "preflight"))
        contract_path, contract_sha256 = _checkpoint_scenario_contract(incoming)
        distribution_path, distribution_sha256 = _checkpoint_distribution_contract(
            incoming
        )
        if (
            phase in {"scenario", "policy"}
            and internal_status in {"TRAINING", "COMPLETE", "FAILED"}
            and contract_path is None
        ):
            raise R4ProtocolError(
                f"terminal/training {phase} state requires a frozen "
                "scenario-training contract"
            )
        if (
            phase in {"distribution", "joint_policy"}
            and internal_status in {"TRAINING", "COMPLETE", "FAILED"}
            and distribution_path is None
        ):
            raise R4ProtocolError(
                f"terminal/training {phase} state requires a frozen "
                "distribution-training contract"
            )
        if (
            phase in {"a1", "a2"}
            and internal_status in {"TRAINING", "COMPLETE", "FAILED"}
            and contract_path is None
            and distribution_path is None
        ):
            raise R4ProtocolError(
                f"terminal/training {phase} state requires its frozen uncertainty contract"
            )
        # Normalize the path before writing so state, latest, provisional and
        # selected artifacts carry exactly one byte-for-byte binding.
        incoming["scenario_training_contract_path"] = contract_path
        incoming["scenario_training_contract_sha256"] = contract_sha256
        incoming["distribution_training_contract_path"] = distribution_path
        incoming["distribution_training_contract_sha256"] = distribution_sha256
        gate = incoming.get("capability_gate")
        if isinstance(gate, Mapping):
            gate_status = str(gate.get("status", "NOT_RUN"))
        else:
            gate_status = str(incoming.get("capability_gate_status", "NOT_RUN"))
        normalized_gate = {
            "PASS": "PASSED", "PASSED": "PASSED", "FAIL": "FAILED",
            "FAILED": "FAILED", "NOT_READY": "NOT_RUN",
        }.get(gate_status, gate_status)
        if internal_status == "COMPLETE" and normalized_gate == "NOT_RUN":
            # COMPLETE enters this method only after run_phase has verified an
            # explicit gate PASS, or after seed1 has completed all five such phases.
            normalized_gate = "PASSED"
        artifact_status = (
            "COMPLETE" if internal_status == "COMPLETE"
            else internal_status if internal_status in {"BLOCKED", "FAILED"}
            else "PARTIAL"
        )
        seed = int(incoming.get("seed", R4_SEEDS[0]))
        config_literal = repr(self.config_path or "<R4_CONFIG_PATH>")
        resume_command = incoming.get(
            "resume_command",
            "python -c \"from dcvpp.e2e_moe_r4.formal_protocol import run_phase; "
            f"run_phase({config_literal}, {phase!r}, {seed}, resume=True)\"",
        )
        payload = {
            **incoming,
            "protocol_version": _artifact_protocol_version(incoming),
            "lineage_id": self.lineage_id,
            "config_sha256": self.config_sha256,
            "experiment_id": str(
                incoming.get("experiment_id", "e2e_moe_hybrid_r4_formal_v1")
            ),
            "architecture_version": str(
                incoming.get("architecture_version", "hybrid_cfc_transformer_moe_r4")
            ),
            "training_semantics_revision": str(
                incoming.get("training_semantics_revision", TRAINING_SEMANTICS_REVISION)
            ),
            "method": str(
                incoming.get("method", "hybrid_ltc_transformer_top2moe_r4")
            ),
            "seed": seed,
            "phase": phase,
            "split": str(incoming.get("split", "train_validation_only")),
            "predecessor_checkpoint": incoming.get("predecessor_checkpoint"),
            "capability_gate_status": normalized_gate,
            "internal_status": internal_status,
            "status": artifact_status,
            "resume_command": str(resume_command),
        }
        write_artifact_state(self.state_path, payload)
        return self.state_path

    def read_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        value = read_json(self.state_path)
        self._check_lineage(value)
        contract_path, contract_sha256 = _checkpoint_scenario_contract(value)
        distribution_path, distribution_sha256 = _checkpoint_distribution_contract(
            value, verify_content=False
        )
        phase = str(value.get("phase", "preflight"))
        internal_status = str(value.get("internal_status", value.get("status", "PARTIAL")))
        if (
            phase in {"scenario", "policy"}
            and internal_status in {"TRAINING", "COMPLETE", "FAILED"}
            and contract_path is None
        ):
            raise R4ProtocolError(
                f"terminal/training {phase} state lacks its frozen "
                "scenario-training contract"
            )
        if (
            phase in {"distribution", "joint_policy"}
            and internal_status in {"TRAINING", "COMPLETE", "FAILED"}
            and distribution_path is None
        ):
            raise R4ProtocolError(
                f"terminal/training {phase} state lacks its frozen "
                "distribution-training contract"
            )
        if (
            phase in {"a1", "a2"}
            and internal_status in {"TRAINING", "COMPLETE", "FAILED"}
            and contract_path is None
            and distribution_path is None
        ):
            raise R4ProtocolError(
                f"terminal/training {phase} state lacks its uncertainty contract"
            )
        return value

    def checkpoint_path(self, phase: str, seed: int) -> Path:
        return self.root / "checkpoints" / f"{phase}_seed{int(seed)}_latest.pt"

    def candidate_root(self, phase: str, seed: int) -> Path:
        return self.root / "candidates" / str(phase) / f"seed={int(seed)}"

    def candidate_path(self, phase: str, seed: int, epoch: int) -> Path:
        return self.candidate_root(phase, seed) / f"k96_epoch{int(epoch):04d}.pt"

    def candidate_registry_path(self, phase: str, seed: int) -> Path:
        return self.candidate_root(phase, seed) / "candidate_registry.json"

    def provisional_root(self, phase: str, seed: int) -> Path:
        return self.root / "provisional_best" / str(phase) / f"seed={int(seed)}"

    def provisional_path(self, phase: str, seed: int, epoch: int) -> Path:
        return self.provisional_root(phase, seed) / f"epoch{int(epoch):04d}.pt"

    def provisional_registry_path(self, phase: str, seed: int) -> Path:
        return self.provisional_root(phase, seed) / "provisional_registry.json"

    def selected_best_path(self, phase: str, seed: int) -> Path:
        return self.root / "checkpoints" / f"{phase}_seed{int(seed)}_selected_best.pt"

    def selected_best_manifest_path(self, phase: str, seed: int) -> Path:
        return self.root / "checkpoints" / f"{phase}_seed{int(seed)}_selected_best.sha256.json"

    @staticmethod
    def epochs_completed(payload: Mapping[str, Any]) -> int:
        """Read the unambiguous one-based completed-epoch count.

        New R4 checkpoints carry both fields.  The fallback is intentionally
        limited to old latest checkpoints, whose historical ``epoch`` was a
        zero-based loop index.  Immutable candidates written by this revision
        always carry ``epochs_completed`` and never need a heuristic.
        """

        if "epochs_completed" in payload:
            completed = int(payload["epochs_completed"])
        else:
            completed = int(payload.get("epoch", -1)) + 1
        if completed < 0:
            raise R4ProtocolError("checkpoint has an invalid completed-epoch count")
        return completed

    def save_checkpoint(
        self,
        phase: str,
        seed: int,
        payload: Mapping[str, Any],
    ) -> Path:
        path = self.checkpoint_path(phase, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        wrapped = {
            **dict(payload),
            "phase": str(phase),
            "seed": int(seed),
            "protocol_version": _artifact_protocol_version(
                {**dict(payload), "phase": str(phase)}
            ),
            "lineage_id": self.lineage_id,
            "config_sha256": self.config_sha256,
        }
        contract_path, contract_sha256 = _checkpoint_scenario_contract(wrapped)
        wrapped["scenario_training_contract_path"] = contract_path
        wrapped["scenario_training_contract_sha256"] = contract_sha256
        distribution_path, distribution_sha256 = _checkpoint_distribution_contract(
            wrapped
        )
        wrapped["distribution_training_contract_path"] = distribution_path
        wrapped["distribution_training_contract_sha256"] = distribution_sha256
        atomic_torch_save(path, wrapped)
        return path

    def read_candidate_registry(self, phase: str, seed: int) -> dict[str, Any]:
        path = self.candidate_registry_path(phase, seed)
        if not path.exists():
            return {
                "schema": "dcvpp.r4.k96-candidate-registry.v1",
                "protocol_version": _artifact_protocol_version({"phase": phase}),
                "lineage_id": self.lineage_id,
                "config_sha256": self.config_sha256,
                "phase": str(phase),
                "seed": int(seed),
                "frozen": False,
                "selected_best": None,
                "candidates": [],
            }
        registry = read_json(path)
        self._check_lineage(registry)
        if registry.get("phase") != phase or int(registry.get("seed", -1)) != int(seed):
            raise R4ProtocolError("candidate registry phase/seed mismatch")
        candidates = registry.get("candidates")
        if not isinstance(candidates, list):
            raise R4ProtocolError("candidate registry candidates must be a list")
        return registry

    def read_provisional_registry(self, phase: str, seed: int) -> dict[str, Any]:
        if phase not in {"representation", "scenario", "distribution"}:
            raise R4ProtocolError(
                "only representation/scenario/distribution have provisional candidates"
            )
        path = self.provisional_registry_path(phase, seed)
        if not path.exists():
            return {
                "schema": "dcvpp.r4.provisional-best-registry.v1",
                "protocol_version": _artifact_protocol_version({"phase": phase}),
                "lineage_id": self.lineage_id,
                "config_sha256": self.config_sha256,
                "phase": str(phase),
                "seed": int(seed),
                "best": None,
                "candidates": [],
            }
        registry = read_json(path)
        self._check_lineage(registry)
        if registry.get("phase") != phase or int(registry.get("seed", -1)) != int(seed):
            raise R4ProtocolError("provisional registry phase/seed mismatch")
        if not isinstance(registry.get("candidates"), list):
            raise R4ProtocolError("provisional registry candidates must be a list")
        return registry

    def write_provisional_registry(
        self, phase: str, seed: int, registry: Mapping[str, Any]
    ) -> Path:
        path = self.provisional_registry_path(phase, seed)
        payload = {
            **dict(registry),
            "schema": "dcvpp.r4.provisional-best-registry.v1",
            "protocol_version": _artifact_protocol_version(
                {**dict(registry), "phase": phase}
            ),
            "lineage_id": self.lineage_id,
            "config_sha256": self.config_sha256,
            "phase": str(phase),
            "seed": int(seed),
        }
        atomic_json(path, payload)
        return path

    def save_provisional_best(
        self,
        phase: str,
        seed: int,
        epoch: int,
        payload: Mapping[str, Any],
        *,
        cheap_validation: float,
        cheap_metrics: Mapping[str, float] | None = None,
        curriculum_stage: str | None = None,
        capture_reason: str = "cheap_improvement",
    ) -> Path:
        """Atomically register one immutable Level-1 candidate checkpoint."""

        registry = self.read_provisional_registry(phase, seed)
        candidates = list(registry["candidates"])
        existing = next(
            (item for item in candidates if int(item.get("epoch", -1)) == int(epoch)),
            None,
        )
        path = self.provisional_path(phase, seed, epoch)
        if existing is not None:
            if not path.is_file():
                raise R4ProtocolError("immutable provisional best is missing or hash-mismatched")
            stat = path.stat()
            if phase == "distribution":
                if (
                    int(existing.get("checkpoint_size_bytes", -1)) != stat.st_size
                    or int(existing.get("checkpoint_mtime_ns", -1)) != stat.st_mtime_ns
                ):
                    raise R4ProtocolError(
                        "immutable Distribution candidate changed after its one-time hash"
                    )
            elif sha256_file(path) != existing.get("checkpoint_sha256"):
                raise R4ProtocolError(
                    "immutable provisional best is missing or hash-mismatched"
                )
            return path
        if path.exists():
            raise R4ProtocolError(f"refusing to overwrite unregistered provisional best: {path}")
        completed = int(epoch)
        wrapped = {
            **dict(payload),
            "phase": phase,
            "seed": int(seed),
            "epoch": completed - 1,
            "epoch_index": completed - 1,
            "epochs_completed": completed,
            "horizon": 96,
            "protocol_version": _artifact_protocol_version(
                {**dict(payload), "phase": phase}
            ),
            "lineage_id": self.lineage_id,
            "config_sha256": self.config_sha256,
        }
        contract_path, contract_sha256 = _checkpoint_scenario_contract(wrapped)
        wrapped["scenario_training_contract_path"] = contract_path
        wrapped["scenario_training_contract_sha256"] = contract_sha256
        distribution_path, distribution_sha256 = _checkpoint_distribution_contract(
            wrapped
        )
        wrapped["distribution_training_contract_path"] = distribution_path
        wrapped["distribution_training_contract_sha256"] = distribution_sha256
        atomic_torch_save(path, wrapped)
        digest = sha256_file(path)
        stat = path.stat()
        item = {
            "epoch": completed,
            "checkpoint": str(path),
            "checkpoint_sha256": digest,
            "checkpoint_size_bytes": int(stat.st_size),
            "checkpoint_mtime_ns": int(stat.st_mtime_ns),
            "cheap_validation": float(cheap_validation),
            "scenario_training_contract_path": contract_path,
            "scenario_training_contract_sha256": contract_sha256,
            "distribution_training_contract_path": distribution_path,
            "distribution_training_contract_sha256": distribution_sha256,
            "cheap_metrics": {
                str(key): float(value) for key, value in (cheap_metrics or {}).items()
            },
            "curriculum_stage": curriculum_stage,
            "capture_reason": str(capture_reason),
            "level1_selected": phase != "distribution",
            "gate_status": "PENDING",
            "gate": None,
        }
        candidates.append(item)
        registry["candidates"] = sorted(candidates, key=lambda value: int(value["epoch"]))
        current = registry.get("best")
        if current is None or (float(item["cheap_validation"]), completed) < (
            float(current["cheap_validation"]), int(current["epoch"])
        ):
            registry["best"] = item
        self.write_provisional_registry(phase, seed, registry)
        return path

    def write_candidate_registry(
        self, phase: str, seed: int, registry: Mapping[str, Any]
    ) -> Path:
        path = self.candidate_registry_path(phase, seed)
        payload = {
            **dict(registry),
            "schema": "dcvpp.r4.k96-candidate-registry.v1",
            "protocol_version": _artifact_protocol_version(
                {**dict(registry), "phase": phase}
            ),
            "lineage_id": self.lineage_id,
            "config_sha256": self.config_sha256,
            "phase": str(phase),
            "seed": int(seed),
        }
        atomic_json(path, payload)
        return path

    def save_k96_candidate(
        self,
        phase: str,
        seed: int,
        epoch: int,
        payload: Mapping[str, Any],
        *,
        cheap_validation: float,
    ) -> Path:
        if phase not in {"policy", "joint_policy", "a1", "a2"}:
            raise R4ProtocolError("only decision phases have K96 candidates")
        registry = self.read_candidate_registry(phase, seed)
        candidates = list(registry["candidates"])
        existing = next(
            (item for item in candidates if int(item.get("epoch", -1)) == int(epoch)),
            None,
        )
        path = self.candidate_path(phase, seed, epoch)
        if existing is not None:
            if not path.is_file() or sha256_file(path) != existing.get("checkpoint_sha256"):
                raise R4ProtocolError("immutable K96 candidate is missing or hash-mismatched")
            return path
        if bool(registry.get("frozen", False)):
            raise R4ProtocolError("cannot add a K96 candidate after registry freeze")
        if path.exists():
            raise R4ProtocolError(f"refusing to overwrite unregistered candidate: {path}")
        wrapped = {
            **dict(payload),
            "phase": phase,
            "seed": int(seed),
            "epoch": int(epoch) - 1,
            "epoch_index": int(epoch) - 1,
            "epochs_completed": int(epoch),
            "horizon": 96,
            "protocol_version": _artifact_protocol_version(
                {**dict(payload), "phase": phase}
            ),
            "lineage_id": self.lineage_id,
            "config_sha256": self.config_sha256,
        }
        contract_path, contract_sha256 = _checkpoint_scenario_contract(wrapped)
        wrapped["scenario_training_contract_path"] = contract_path
        wrapped["scenario_training_contract_sha256"] = contract_sha256
        distribution_path, distribution_sha256 = _checkpoint_distribution_contract(
            wrapped
        )
        wrapped["distribution_training_contract_path"] = distribution_path
        wrapped["distribution_training_contract_sha256"] = distribution_sha256
        atomic_torch_save(path, wrapped)
        stat = path.stat()
        candidates.append(
            {
                "epoch": int(epoch),
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "cheap_validation": float(cheap_validation),
                "scenario_training_contract_path": contract_path,
                "scenario_training_contract_sha256": contract_sha256,
                "distribution_training_contract_path": distribution_path,
                "distribution_training_contract_sha256": distribution_sha256,
                "checkpoint_size_bytes": int(stat.st_size),
                "checkpoint_mtime_ns": int(stat.st_mtime_ns),
                "gate_status": "PENDING",
                "gate": None,
            }
        )
        registry["candidates"] = sorted(candidates, key=lambda item: int(item["epoch"]))
        self.write_candidate_registry(phase, seed, registry)
        return path

    def freeze_candidate_registry(self, phase: str, seed: int) -> dict[str, Any]:
        registry = self.read_candidate_registry(phase, seed)
        if not registry["candidates"]:
            raise R4ProtocolError(f"{phase} produced no K96 candidates")
        registry["frozen"] = True
        self.write_candidate_registry(phase, seed, registry)
        return registry

    def save_selected_best(
        self,
        phase: str,
        seed: int,
        candidate: Mapping[str, Any],
        *,
        use_registered_candidate_digest: bool = False,
    ) -> Path:
        source = Path(str(candidate["checkpoint"])).resolve()
        if not source.is_file():
            raise R4ProtocolError("selected candidate checkpoint is missing")
        if (
            not use_registered_candidate_digest
            and sha256_file(source) != candidate.get("checkpoint_sha256")
        ):
            raise R4ProtocolError("selected candidate checkpoint hash mismatch")
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise R4ProtocolError("selected candidate checkpoint must be a mapping")
        contract_path, contract_sha256 = _checkpoint_scenario_contract(payload)
        distribution_path, distribution_sha256 = _checkpoint_distribution_contract(
            payload, verify_content=False
        )
        if (
            candidate.get("scenario_training_contract_path") != contract_path
            or candidate.get("scenario_training_contract_sha256") != contract_sha256
        ):
            raise R4ProtocolError(
                "selected candidate scenario-training contract binding mismatch"
            )
        if (
            candidate.get("distribution_training_contract_path")
            != distribution_path
            or candidate.get("distribution_training_contract_sha256")
            != distribution_sha256
        ):
            raise R4ProtocolError(
                "selected candidate Distribution contract binding mismatch"
            )
        destination = self.selected_best_path(phase, seed)
        selected = {
            **dict(payload),
            "selected_from": str(source),
            "selected_from_sha256": candidate["checkpoint_sha256"],
            "selection_gate": candidate["gate"],
            "selected_epoch": int(candidate["epoch"]),
            "checkpoint_role": "capability_passed_selected_best",
            "capability_gate_status": "PASSED",
        }
        atomic_torch_save(destination, selected)
        selected_sha256 = sha256_file(destination)
        atomic_json(
            self.selected_best_manifest_path(phase, seed),
            {
                "schema": "dcvpp.r4.selected-best-binding.v1",
                "protocol_version": _artifact_protocol_version(selected),
                "lineage_id": self.lineage_id,
                "config_sha256": self.config_sha256,
                "phase": str(phase),
                "seed": int(seed),
                "selected_best": str(destination.resolve()),
                "selected_best_sha256": selected_sha256,
                "selected_from": str(source),
                "selected_from_sha256": candidate["checkpoint_sha256"],
                "scenario_training_contract_path": contract_path,
                "scenario_training_contract_sha256": contract_sha256,
                "distribution_training_contract_path": distribution_path,
                "distribution_training_contract_sha256": distribution_sha256,
            },
        )
        return destination

    def load_selected_checkpoint(
        self, phase: str, seed: int, *, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        """Load only a capability-passed predecessor; latest/provisional are invalid."""

        path = self.selected_best_path(phase, seed)
        if not path.is_file():
            raise R4ProtocolError(
                f"phase {phase} has no capability-passed selected checkpoint: {path}"
            )
        try:
            value = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            value = torch.load(path, map_location=map_location)
        if not isinstance(value, Mapping):
            raise R4ProtocolError("selected checkpoint payload must be a mapping")
        result = dict(value)
        binding_path = self.selected_best_manifest_path(phase, seed)
        if not binding_path.is_file():
            raise R4ProtocolError("selected checkpoint has no immutable SHA-256 binding")
        binding = read_json(binding_path)
        self._check_lineage(binding)
        if (
            binding.get("schema") != "dcvpp.r4.selected-best-binding.v1"
            or binding.get("phase") != phase
            or int(binding.get("seed", -1)) != int(seed)
            or Path(str(binding.get("selected_best", ""))).resolve() != path.resolve()
            or sha256_file(path) != binding.get("selected_best_sha256")
        ):
            raise R4ProtocolError("selected checkpoint content binding is invalid")
        self._check_lineage(result)
        if result.get("phase") != phase or int(result.get("seed", -1)) != int(seed):
            raise R4ProtocolError("selected checkpoint phase/seed mismatch")
        if result.get("checkpoint_role") != "capability_passed_selected_best":
            raise R4ProtocolError("predecessor is not a selected-best checkpoint")
        if result.get("capability_gate_status") != "PASSED":
            raise R4ProtocolError("predecessor capability gate did not pass")
        contract_path, contract_sha256 = _checkpoint_scenario_contract(result)
        if (
            binding.get("scenario_training_contract_path") != contract_path
            or binding.get("scenario_training_contract_sha256") != contract_sha256
        ):
            raise R4ProtocolError(
                "selected checkpoint scenario-training contract binding is invalid"
            )
        gate = result.get("selection_gate")
        if not isinstance(gate, Mapping) or gate.get("status") not in {"PASS", "PASSED"}:
            raise R4ProtocolError("selected checkpoint lacks a terminal PASS gate")
        source = Path(str(result.get("selected_from", ""))).resolve()
        expected_sha = result.get("selected_from_sha256")
        if (
            not source.is_file()
            or sha256_file(source) != expected_sha
            or binding.get("selected_from_sha256") != expected_sha
            or Path(str(binding.get("selected_from", ""))).resolve() != source
        ):
            raise R4ProtocolError("selected checkpoint source is missing or hash-mismatched")
        try:
            source_value = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            source_value = torch.load(source, map_location="cpu")
        if (
            not isinstance(source_value, Mapping)
            or not _checkpoint_value_equal(
                result.get("model_state"), source_value.get("model_state")
            )
        ):
            raise R4ProtocolError(
                "selected checkpoint model_state differs from its capability-passed source"
            )
        return result

    def load_checkpoint(
        self, phase: str, seed: int, *, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any] | None:
        path = self.checkpoint_path(phase, seed)
        if not path.exists():
            return None
        try:
            value = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:  # older supported Torch
            value = torch.load(path, map_location=map_location)
        if not isinstance(value, Mapping):
            raise R4ProtocolError("checkpoint payload must be a mapping")
        result = dict(value)
        self._check_lineage(result)
        if result.get("phase") != phase or int(result.get("seed", -1)) != int(seed):
            raise R4ProtocolError("checkpoint phase/seed does not match resume request")
        _checkpoint_scenario_contract(result)
        return result

    def _check_lineage(self, value: Mapping[str, Any]) -> None:
        protocol = value.get("protocol_version")
        if (
            protocol not in {R4_PROTOCOL_VERSION, DISTRIBUTIONAL_R4_PROTOCOL_VERSION}
            or value.get("lineage_id") != self.lineage_id
            or value.get("config_sha256") != self.config_sha256
        ):
            raise R4ProtocolError(
                "refusing cross-lineage resume: checkpoint/state is not this R4 protocol"
            )


LossFunction = Callable[[nn.Module, Any, int], Tensor | Mapping[str, Tensor]]
CheapValidator = Callable[..., float | Mapping[str, float]]


def _invoke_cheap_validator(
    validator: CheapValidator,
    model: nn.Module,
    phase: str,
    horizon: int,
    *,
    distribution_stage_index: int | None,
) -> float | Mapping[str, float]:
    signature = inspect.signature(validator)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "distribution_stage_index" in signature.parameters or accepts_kwargs:
        return validator(
            model,
            phase,
            horizon,
            distribution_stage_index=distribution_stage_index,
        )
    return validator(model, phase, horizon)


def _normalize_cheap_validation(
    raw: float | Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    if isinstance(raw, Mapping):
        metrics: dict[str, float] = {}
        for name, value in raw.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise R4ProtocolError(
                        f"cheap validation metric {name!r} must be scalar"
                    )
                scalar = float(value.detach().cpu())
            else:
                scalar = float(value)
            if not math.isfinite(scalar):
                raise R4ProtocolError(
                    f"cheap validation metric {name!r} is non-finite"
                )
            metrics[str(name)] = scalar
        if "total" not in metrics:
            raise R4ProtocolError("cheap validation mapping must contain total")
        return metrics["total"], metrics
    score = float(raw)
    if not math.isfinite(score):
        raise R4ProtocolError("cheap validation returned a non-finite score")
    return score, {"total": score}


@dataclass(frozen=True)
class PhaseRunResult:
    phase: str
    seed: int
    status: str
    epochs_completed: int
    best_cheap_validation: float | None
    capability_gate: CapabilityGateResult
    checkpoint: str | None
    trainable_parameters: tuple[str, ...]
    reason: str | None = None
    epoch_coverage: tuple[EpochCoverage, ...] = ()
    candidate_registry: str | None = None
    selected_candidate_epoch: int | None = None


def _total_loss(value: Tensor | Mapping[str, Tensor]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value.get("total"), Tensor):
        return value["total"]
    tensors = tuple(item for item in value.values() if isinstance(item, Tensor))
    if not tensors:
        raise R4ProtocolError("loss function returned no tensor losses")
    return torch.stack(tuple(item.reshape(()) for item in tensors)).sum()


class R4PhaseTrainer:
    """Minimal phase trainer with exact accumulation and fail-closed gates."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        store: AtomicRunStore,
        cheap_validator: CheapValidator,
        phase_end_evaluator: PhaseEndEvaluator | None,
        physical_batch_size: int = 512,
        cheap_validation_policy: CheapValidationPolicy | None = None,
        expected_training_dates: Sequence[str] | None = None,
        predecessor_checkpoint: str | Path | None = None,
        predecessor_sha256: str | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        gradient_clip_norm: float | None = None,
        normalization_sha256: str | None = None,
        scenario_training_contract_path: str | Path | None = None,
        scenario_training_contract_sha256: str | None = None,
        distribution_training_contract_path: str | Path | None = None,
        distribution_training_contract_sha256: str | None = None,
        distribution_contract_preverified: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.store = store
        self.cheap_validator = cheap_validator
        self.phase_end_evaluator = phase_end_evaluator
        self.physical_batch_size = int(physical_batch_size)
        self.accumulation_steps = accumulation_for_physical_batch(physical_batch_size)
        self.cheap_policy = cheap_validation_policy or CheapValidationPolicy()
        self.expected_training_dates = self._normalize_expected_training_dates(
            expected_training_dates
        )
        self.predecessor_checkpoint = (
            None if predecessor_checkpoint is None
            else str(Path(predecessor_checkpoint).resolve())
        )
        if self.predecessor_checkpoint is None:
            if predecessor_sha256 is not None:
                raise R4ProtocolError(
                    "predecessor_sha256 cannot be supplied without a checkpoint"
                )
            self.predecessor_sha256 = None
        elif predecessor_sha256 is None:
            self.predecessor_sha256 = sha256_file(self.predecessor_checkpoint)
        else:
            digest = str(predecessor_sha256).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise R4ProtocolError("predecessor_sha256 must be one lowercase SHA-256")
            # The V6 phase runner hashes the inherited epoch-28 checkpoint once
            # while producing its immutable import manifest.  Reuse that
            # binding here instead of re-reading a multi-gigabyte checkpoint.
            self.predecessor_sha256 = digest
        if gradient_clip_norm is not None and (
            not math.isfinite(float(gradient_clip_norm))
            or float(gradient_clip_norm) <= 0.0
        ):
            raise R4ProtocolError("gradient_clip_norm must be finite and positive")
        self.gradient_clip_norm = (
            None if gradient_clip_norm is None else float(gradient_clip_norm)
        )
        if normalization_sha256 is not None:
            digest = str(normalization_sha256).lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise R4ProtocolError("normalization_sha256 must be one lowercase SHA-256")
            self.normalization_sha256: str | None = digest
        else:
            self.normalization_sha256 = None
        contract_path, contract_sha256 = _checkpoint_scenario_contract(
            {
                "scenario_training_contract_path": scenario_training_contract_path,
                "scenario_training_contract_sha256": scenario_training_contract_sha256,
            }
        )
        self.scenario_training_contract_path = contract_path
        self.scenario_training_contract_sha256 = contract_sha256
        distribution_path, distribution_sha256 = _checkpoint_distribution_contract(
            {
                "distribution_training_contract_path": (
                    distribution_training_contract_path
                ),
                "distribution_training_contract_sha256": (
                    distribution_training_contract_sha256
                ),
            },
            verify_content=not bool(distribution_contract_preverified),
        )
        self.distribution_training_contract_path = distribution_path
        self.distribution_training_contract_sha256 = distribution_sha256

    def _scenario_contract_payload(self) -> dict[str, str | None]:
        return {
            "scenario_training_contract_path": self.scenario_training_contract_path,
            "scenario_training_contract_sha256": self.scenario_training_contract_sha256,
        }

    def _distribution_contract_payload(self) -> dict[str, str | None]:
        return {
            "distribution_training_contract_path": (
                self.distribution_training_contract_path
            ),
            "distribution_training_contract_sha256": (
                self.distribution_training_contract_sha256
            ),
        }

    def _validate_scenario_contract_for_phase(
        self,
        phase: str,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        """Fail closed on absent, changed, or cross-contract phase state."""

        if phase in {"scenario", "policy"} and self.scenario_training_contract_path is None:
            raise R4ProtocolError(
                f"phase {phase} requires a frozen scenario-training contract"
            )
        current_path, current_sha256 = _checkpoint_scenario_contract(
            self._scenario_contract_payload()
        )
        if (
            current_path != self.scenario_training_contract_path
            or current_sha256 != self.scenario_training_contract_sha256
        ):
            raise R4ProtocolError("scenario-training contract changed after trainer setup")
        if checkpoint is not None:
            checkpoint_path, checkpoint_sha256 = _checkpoint_scenario_contract(checkpoint)
            if (
                checkpoint_path != self.scenario_training_contract_path
                or checkpoint_sha256 != self.scenario_training_contract_sha256
            ):
                raise R4ProtocolError(
                    "resume checkpoint scenario-training contract differs from this run"
                )

    def _validate_distribution_contract_for_phase(
        self,
        phase: str,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            phase in {"distribution", "joint_policy"}
            and self.distribution_training_contract_path is None
        ):
            raise R4ProtocolError(
                f"phase {phase} requires a frozen distribution-training contract"
            )
        if phase in {"a1", "a2"} and (
            self.distribution_training_contract_path is None
            and self.scenario_training_contract_path is None
        ):
            raise R4ProtocolError(
                f"phase {phase} requires its frozen uncertainty-training contract"
            )
        current_path, current_sha = _checkpoint_distribution_contract(
            self._distribution_contract_payload(), verify_content=False
        )
        if (
            current_path != self.distribution_training_contract_path
            or current_sha != self.distribution_training_contract_sha256
        ):
            raise R4ProtocolError(
                "distribution-training contract changed after trainer setup"
            )
        if checkpoint is not None:
            checkpoint_path, checkpoint_sha = _checkpoint_distribution_contract(
                checkpoint, verify_content=False
            )
            if (
                checkpoint_path != self.distribution_training_contract_path
                or checkpoint_sha != self.distribution_training_contract_sha256
            ):
                raise R4ProtocolError(
                    "resume checkpoint Distribution contract differs from this run"
                )

    @staticmethod
    def _normalize_expected_training_dates(
        values: Sequence[str] | None,
    ) -> frozenset[str] | None:
        if values is None:
            return None
        normalized: list[str] = []
        for value in values:
            try:
                parsed = date.fromisoformat(str(value)[:10])
            except ValueError as exc:
                raise R4ProtocolError(
                    f"invalid expected training date: {value!r}"
                ) from exc
            if not TRAIN_START <= parsed <= TRAIN_END:
                raise R4ProtocolError(
                    f"expected training date is outside train split: {parsed}"
                )
            normalized.append(parsed.isoformat())
        result = frozenset(normalized)
        if len(values) != FORMAL_TRAINING_DAYS or len(result) != FORMAL_TRAINING_DAYS:
            raise R4ProtocolError(
                f"expected_training_dates must contain exactly {FORMAL_TRAINING_DAYS} "
                "unique dates"
            )
        return result

    def _set_phase(self, phase: str) -> tuple[str, ...]:
        setter = getattr(self.model, "set_trainable_phase", None)
        if setter is None:
            setter = getattr(self.model, "configure_trainable_phase", None)
        if setter is None:
            raise R4ProtocolError(
                "R4 model must expose set_trainable_phase or configure_trainable_phase"
            )
        reported = setter(phase)
        if reported is None:
            reported = tuple(
                name for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            )
        names = tuple(str(name) for name in reported)
        if not names:
            raise R4ProtocolError(f"phase {phase} has no trainable parameters")
        return names

    def run(
        self,
        phase: str,
        seed: int,
        batches: Iterable[Any] | Callable[[int, int], Iterable[Any]],
        loss_function: LossFunction,
        *,
        resume: bool = True,
        max_epochs_for_test: int | None = None,
    ) -> PhaseRunResult:
        if not isinstance(loss_function, PhaseLossComputer):
            raise R4ProtocolError(
                "formal R4 training requires build_phase_loss/PhaseLossComputer"
            )
        # Compatibility subclasses used by the training-engine contract tests
        # may deliberately bypass ``PhaseLossComputer.__init__``.  Only V6
        # Distribution requires the governing config to derive its five-stage
        # schedule; all other phase specs are static.
        loss_config = getattr(loss_function, "config", None)
        spec = phase_spec(phase, loss_config)
        is_distributional_run = bool(
            isinstance(loss_config, Mapping)
            and str(loss_config.get("project", {}).get("architecture_version", ""))
            == "continuous_distributional_joint_moe_r4_v6"
        )
        reported_effective_days = (
            FORMAL_TRAINING_DAYS if is_distributional_run else EFFECTIVE_BATCH_SIZE
        )
        scientific_accumulation_steps = math.ceil(
            FORMAL_TRAINING_DAYS / self.physical_batch_size
        )
        project = loss_config.get("project", {}) if isinstance(loss_config, Mapping) else {}
        run_identity = {
            "experiment_id": str(
                project.get("experiment_id", "e2e_moe_hybrid_r4_formal_v1")
            ),
            "architecture_version": str(
                project.get("architecture_version", "hybrid_cfc_transformer_moe_r4")
            ),
            "training_semantics_revision": str(
                project.get("training_semantics_revision", TRAINING_SEMANTICS_REVISION)
            ),
            "method": (
                "continuous_distributional_world_joint_moe_r4_v6"
                if is_distributional_run
                else "hybrid_ltc_transformer_top2moe_r4"
            ),
        }
        if loss_function.phase != spec.name:
            raise R4ProtocolError(
                f"loss phase {loss_function.phase} does not match run phase {spec.name}"
            )
        if spec.name in {"policy", "joint_policy", "a1", "a2"} and not callable(
            getattr(self.model, "forward_training_rollout", None)
        ):
            raise R4ProtocolError(
                "formal decision training requires model.forward_training_rollout"
            )
        if int(seed) not in R4_SEEDS:
            raise R4ProtocolError(f"unregistered R4 seed: {seed}")
        self._validate_scenario_contract_for_phase(spec.name)
        self._validate_distribution_contract_for_phase(spec.name)
        torch.manual_seed(int(seed))
        trainable = self._set_phase(spec.name)
        checkpoint = self.store.load_checkpoint(spec.name, seed) if resume else None
        checkpoint_completed = (
            0 if checkpoint is None else self.store.epochs_completed(checkpoint)
        )
        if resume and spec.name in {"policy", "joint_policy", "a1", "a2"}:
            registry_path = self.store.candidate_registry_path(spec.name, seed)
            if registry_path.exists():
                registry = self.store.read_candidate_registry(spec.name, seed)
                if registry["candidates"]:
                    newest = max(registry["candidates"], key=lambda item: int(item["epoch"]))
                    candidate_completed = int(newest["epoch"])
                    if candidate_completed > checkpoint_completed:
                        candidate_path = Path(str(newest["checkpoint"])).resolve()
                        if (
                            not candidate_path.is_file()
                            or sha256_file(candidate_path) != newest.get("checkpoint_sha256")
                        ):
                            raise R4ProtocolError(
                                "newest resume candidate is missing or hash-mismatched"
                            )
                        try:
                            candidate_payload = torch.load(
                                candidate_path, map_location="cpu", weights_only=False
                            )
                        except TypeError:
                            candidate_payload = torch.load(candidate_path, map_location="cpu")
                        if not isinstance(candidate_payload, Mapping):
                            raise R4ProtocolError("resume candidate must be a mapping")
                        self.store._check_lineage(candidate_payload)
                        checkpoint = dict(candidate_payload)
                        if "epochs_completed" in checkpoint:
                            payload_completed = self.store.epochs_completed(checkpoint)
                        elif (
                            "epoch_index" not in checkpoint
                            and int(checkpoint.get("epoch", -1)) == candidate_completed
                        ):
                            # Pre-hardening R4 candidates stored the one-based
                            # completed count in ``epoch``.  The immutable
                            # registry independently records the same value,
                            # so this exact equality is a safe one-time legacy
                            # interpretation rather than an epoch heuristic.
                            payload_completed = candidate_completed
                        else:
                            raise R4ProtocolError(
                                "legacy candidate has ambiguous epoch semantics"
                            )
                        if payload_completed != candidate_completed:
                            raise R4ProtocolError(
                                "candidate registry/checkpoint completed-epoch mismatch"
                            )
                        checkpoint_completed = payload_completed
        start_epoch = 0
        best = math.inf
        stale = 0
        distribution_contract = getattr(loss_function, "distribution_contract", None)
        distribution_stage_index = 0
        distribution_stage_epoch = 0
        if checkpoint is not None:
            self._validate_scenario_contract_for_phase(spec.name, checkpoint)
            self._validate_distribution_contract_for_phase(spec.name, checkpoint)
            checkpoint_normalization = checkpoint.get("normalization_sha256")
            if self.normalization_sha256 is not None and (
                checkpoint_normalization != self.normalization_sha256
            ):
                raise R4ProtocolError(
                    "resume checkpoint normalization hash differs from the frozen "
                    "train-only artifact"
                )
            checkpoint_dates = checkpoint.get("expected_training_dates")
            if checkpoint_dates is not None:
                restored_dates = self._normalize_expected_training_dates(
                    tuple(str(value) for value in checkpoint_dates)
                )
                if (
                    self.expected_training_dates is not None
                    and restored_dates != self.expected_training_dates
                ):
                    raise R4ProtocolError(
                        "resume checkpoint training-date set differs from configured set"
                    )
                self.expected_training_dates = restored_dates
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler_state = checkpoint.get("scheduler_state")
            if self.scheduler is None:
                if scheduler_state is not None:
                    raise R4ProtocolError(
                        "resume checkpoint has scheduler state but no scheduler is configured"
                    )
            else:
                if not isinstance(scheduler_state, Mapping):
                    raise R4ProtocolError(
                        "configured scheduler requires exact checkpoint scheduler state"
                    )
                expected_scheduler = (
                    f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
                )
                if checkpoint.get("scheduler_type") != expected_scheduler:
                    raise R4ProtocolError("resume checkpoint scheduler type mismatch")
                self.scheduler.load_state_dict(dict(scheduler_state))
            start_epoch = checkpoint_completed
            best = float(checkpoint.get("best_cheap_validation", math.inf))
            stale = int(checkpoint.get("stale_epochs", 0))
            if spec.name == "distribution":
                distribution_stage_index = int(
                    checkpoint.get("distribution_stage_index", 0)
                )
                distribution_stage_epoch = int(
                    checkpoint.get("distribution_stage_epoch", 0)
                )
                if (
                    distribution_contract is None
                    or not 0 <= distribution_stage_index < len(distribution_contract.stages)
                    or distribution_stage_epoch < 0
                    or distribution_stage_epoch
                    > distribution_contract.stages[
                        distribution_stage_index
                    ].maximum_epochs
                    or (
                        distribution_stage_epoch
                        == distribution_contract.stages[
                            distribution_stage_index
                        ].maximum_epochs
                        and not bool(checkpoint.get("distribution_training_complete", False))
                    )
                ):
                    raise R4ProtocolError(
                        "resume checkpoint has an invalid Distribution curriculum cursor"
                    )
            rng = checkpoint.get("torch_rng_state")
            if isinstance(rng, Tensor):
                torch.set_rng_state(rng.cpu())
            cuda_rng = checkpoint.get("cuda_rng_state_all")
            if torch.cuda.is_available() and isinstance(cuda_rng, Sequence) and cuda_rng:
                torch.cuda.set_rng_state_all(list(cuda_rng))

        epochs = spec.epochs
        if max_epochs_for_test is not None:
            epochs = min(epochs, int(max_epochs_for_test))
        completed = start_epoch
        latest_path: Path | None = (
            self.store.checkpoint_path(spec.name, seed) if checkpoint is not None else None
        )
        coverage_records: list[EpochCoverage] = []
        training_frozen = False
        if resume and spec.name == "representation" and checkpoint is not None:
            # A representation run may have finished gradient training but
            # failed because an older selector gated only the differentiable
            # cheap-best checkpoint.  Immutable provisional checkpoints still
            # contain the full pre-gate candidate set, so resume must re-run
            # only the scientific capability selection rather than silently
            # training additional epochs under an already-finished schedule.
            phase_state = self.store.read_state() or {}
            state_status = str(
                phase_state.get("internal_status", phase_state.get("status", ""))
            ).upper()
            registry_path = self.store.provisional_registry_path(spec.name, seed)
            if state_status == "FAILED" and registry_path.is_file():
                registry = self.store.read_provisional_registry(spec.name, seed)
                training_frozen = bool(registry.get("candidates"))
        if resume and spec.name == "distribution" and checkpoint is not None:
            training_frozen = bool(
                checkpoint.get("distribution_training_complete", False)
            )
        if spec.name in {"policy", "joint_policy", "a1", "a2"}:
            registry_path = self.store.candidate_registry_path(spec.name, seed)
            if registry_path.exists():
                training_frozen = bool(
                    self.store.read_candidate_registry(spec.name, seed).get("frozen", False)
                )
        training_epochs = start_epoch if training_frozen else epochs
        for epoch in range(start_epoch, training_epochs):
            executed_distribution_stage: DistributionCurriculumStage | None = None
            executed_distribution_stage_index: int | None = None
            if spec.name == "distribution":
                assert distribution_contract is not None
                executed_distribution_stage_index = distribution_stage_index
                executed_distribution_stage = distribution_contract.stages[
                    distribution_stage_index
                ]
                loss_function.set_distribution_stage(distribution_stage_index)
            if spec.curriculum and epoch in self._stage_start_epochs(spec):
                # Metrics at K=8/24/48/96 have different scales and must not
                # share early-stopping state.  In particular, stale epochs
                # accumulated in the fixed curriculum cannot stop K=96 early.
                best, stale = math.inf, 0
            horizon = self._horizon_at_epoch(spec, epoch)
            epoch_batches = batches(epoch, horizon) if callable(batches) else batches
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            physical_days = 0
            optimizer_steps = 0
            full_effective_batches = 0
            partial_tail_days = 0
            seen_dates: set[str] = set()
            epoch_loss_day_sum = 0.0
            gradient_nonzero_names: set[str] = set()

            def audit_gradients() -> None:
                observed = 0
                for name, parameter in self.model.named_parameters():
                    if not parameter.requires_grad or parameter.grad is None:
                        continue
                    observed += 1
                    gradient = parameter.grad
                    if not bool(torch.isfinite(gradient).all()):
                        raise R4ProtocolError(
                            f"non-finite gradient for {name} in {spec.name} epoch {epoch}"
                        )
                    if bool(torch.count_nonzero(gradient).item()):
                        gradient_nonzero_names.add(name)
                if observed == 0:
                    raise R4ProtocolError(
                        f"no trainable gradient was produced in {spec.name} epoch {epoch}"
                    )
            for batch_index, batch in enumerate(epoch_batches):
                batch_days, operating_dates = _batch_day_contract(
                    batch, physical_batch_size=self.physical_batch_size
                )
                duplicates = seen_dates.intersection(operating_dates)
                if duplicates:
                    raise R4ProtocolError(
                        "training operating dates repeated within epoch: "
                        f"{sorted(duplicates)[:3]}"
                    )
                if physical_days + batch_days > FORMAL_TRAINING_DAYS:
                    raise R4ProtocolError(
                        "a physical batch crosses the frozen 543-day coverage boundary"
                    )
                seen_dates.update(operating_dates)
                physical_days += batch_days
                loss = _total_loss(loss_function(self.model, batch, horizon))
                if not torch.isfinite(loss):
                    raise R4ProtocolError(
                        f"non-finite training loss in {spec.name} epoch {epoch}"
                    )
                epoch_loss_day_sum += float(loss.detach().cpu()) * float(batch_days)
                # One optimizer update represents the exact mean gradient of
                # all 543 unique train days.  In particular, the final 31-day
                # physical tail is never passed to AdamW as a second almost-
                # full adaptive update.
                (loss * (float(batch_days) / FORMAL_TRAINING_DAYS)).backward()
            if physical_days != FORMAL_TRAINING_DAYS:
                self.optimizer.zero_grad(set_to_none=True)
                raise R4ProtocolError(
                    f"epoch {epoch} must cover exactly {FORMAL_TRAINING_DAYS} "
                    f"training days, got {physical_days}"
                )
            if len(seen_dates) != FORMAL_TRAINING_DAYS:
                self.optimizer.zero_grad(set_to_none=True)
                raise R4ProtocolError(
                    f"epoch {epoch} must contain {FORMAL_TRAINING_DAYS} unique "
                    f"operating dates, got {len(seen_dates)}"
                )
            observed_date_set = frozenset(seen_dates)
            if self.expected_training_dates is None:
                # Freeze the first complete coverage cycle.  Formal callers
                # should pass the split-manifest dates explicitly; inference
                # here still prevents epoch-to-epoch sample substitution.
                self.expected_training_dates = observed_date_set
            elif observed_date_set != self.expected_training_dates:
                missing = sorted(self.expected_training_dates - observed_date_set)
                unexpected = sorted(observed_date_set - self.expected_training_dates)
                self.optimizer.zero_grad(set_to_none=True)
                raise R4ProtocolError(
                    "epoch operating-date set differs from the frozen 543-day set; "
                    f"missing={missing[:3]}, unexpected={unexpected[:3]}"
                )
            audit_gradients()
            gradient_norm_before_clip: float | None = None
            if self.gradient_clip_norm is not None:
                parameters = [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    self.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                gradient_norm_before_clip = float(norm.detach().cpu())
            learning_rates_before_step = [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            learning_rates_after_step = [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_steps = 1
            full_effective_batches = 1

            if spec.name == "a1":
                missing_gradients = sorted(set(trainable) - gradient_nonzero_names)
                if missing_gradients:
                    raise R4ProtocolError(
                        "A1 allowed parameters lack finite nonzero gradients: "
                        f"{missing_gradients}"
                    )

            coverage = EpochCoverage(
                epoch=epoch + 1,
                horizon=horizon,
                physical_days=physical_days,
                effective_days=physical_days,
                optimizer_steps=optimizer_steps,
                full_effective_batches=full_effective_batches,
                partial_tail_days=partial_tail_days,
                unique_operating_dates=len(seen_dates),
                configured_effective_batch_days=reported_effective_days,
                optimizer_effective_batch_days=FORMAL_TRAINING_DAYS,
            )
            coverage_records.append(coverage)

            cheap_improved = False
            cheap_metrics: dict[str, float] = {}
            score = best
            if (epoch + 1) % self.cheap_policy.every_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    raw_cheap = _invoke_cheap_validator(
                        self.cheap_validator,
                        self.model,
                        spec.name,
                        horizon,
                        distribution_stage_index=executed_distribution_stage_index,
                    )
                score, cheap_metrics = _normalize_cheap_validation(raw_cheap)
                epoch_min_delta = (
                    executed_distribution_stage.min_delta
                    if executed_distribution_stage is not None
                    else spec.min_delta
                )
                if score < best - epoch_min_delta:
                    best, stale = score, 0
                    cheap_improved = True
                else:
                    stale += 1
            completed = epoch + 1

            distribution_stage_completed = False
            distribution_training_complete = False
            captured_stage_id: str | None = None
            if executed_distribution_stage is not None:
                captured_stage_id = executed_distribution_stage.stage_id
                distribution_stage_epoch += 1
                stage_early_stop = (
                    distribution_stage_epoch
                    >= executed_distribution_stage.minimum_epochs
                    and stale >= executed_distribution_stage.patience
                )
                stage_max_stop = (
                    distribution_stage_epoch
                    >= executed_distribution_stage.maximum_epochs
                )
                distribution_stage_completed = stage_early_stop or stage_max_stop
                if distribution_stage_completed:
                    if distribution_stage_index + 1 < len(distribution_contract.stages):
                        distribution_stage_index += 1
                        distribution_stage_epoch = 0
                        best, stale = math.inf, 0
                    else:
                        distribution_training_complete = True
            mean_training_loss = epoch_loss_day_sum / float(physical_days)
            gradient_evidence = {
                "required_parameter_count": len(trainable),
                "finite_nonzero_parameter_count": len(gradient_nonzero_names),
                "finite_nonzero_parameter_names": sorted(gradient_nonzero_names),
                "all_allowed_gradients_finite_nonzero": (
                    spec.name != "a1" or set(trainable) == gradient_nonzero_names
                ),
                "gradient_clip_norm": self.gradient_clip_norm,
                "gradient_norm_before_clip": gradient_norm_before_clip,
            }
            scheduler_type = (
                None
                if self.scheduler is None
                else f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
            )
            scheduler_state = (
                None if self.scheduler is None else self.scheduler.state_dict()
            )
            checkpoint_payload = {
                    "epoch": epoch,
                    "epoch_index": epoch,
                    "epochs_completed": completed,
                    "horizon": horizon,
                    "best_cheap_validation": best,
                    "stale_epochs": stale,
                    "cheap_validation_metrics": cheap_metrics,
                    "mean_training_loss": mean_training_loss,
                    "gradient_evidence": gradient_evidence,
                    "epoch_coverage": asdict(coverage),
                    "expected_training_dates": sorted(self.expected_training_dates),
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_type": scheduler_type,
                    "scheduler_state": scheduler_state,
                    "learning_rates_before_step": learning_rates_before_step,
                    "learning_rates_after_step": learning_rates_after_step,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
                    ),
                    **run_identity,
                    "predecessor_checkpoint": self.predecessor_checkpoint,
                    "predecessor_sha256": self.predecessor_sha256,
                    **self._scenario_contract_payload(),
                    **self._distribution_contract_payload(),
                    "distribution_curriculum_stage_executed": captured_stage_id,
                    "distribution_curriculum_stage_index_executed": (
                        executed_distribution_stage_index
                    ),
                    "distribution_stage_index": distribution_stage_index,
                    "distribution_stage_epoch": distribution_stage_epoch,
                    "distribution_stage_completed": distribution_stage_completed,
                    "distribution_training_complete": distribution_training_complete,
                    "physical_microbatch_size": self.physical_batch_size,
                    "configured_effective_batch_size": reported_effective_days,
                    "scientific_effective_batch_days": FORMAL_TRAINING_DAYS,
                    "scientific_accumulation_steps": scientific_accumulation_steps,
                    "legacy_configured_effective_batch_size": (
                        EFFECTIVE_BATCH_SIZE if is_distributional_run else None
                    ),
                    "optimizer_effective_batch_days": FORMAL_TRAINING_DAYS,
                    "normalization_sha256": self.normalization_sha256,
                }
            if spec.name in {"representation", "scenario"} and cheap_improved:
                self.store.save_provisional_best(
                    spec.name,
                    seed,
                    completed,
                    checkpoint_payload,
                    cheap_validation=best,
                )
            if spec.name == "distribution":
                assert distribution_contract is not None
                selection = distribution_contract.candidate_selection
                scheduled = completed % int(selection["scheduled_every_epochs"]) == 0
                if cheap_improved or scheduled or distribution_stage_completed:
                    reason_parts: list[str] = []
                    if cheap_improved:
                        reason_parts.append("cheap_improvement")
                    if distribution_stage_completed:
                        reason_parts.append("stage_end")
                    if scheduled:
                        reason_parts.append("scheduled")
                    reason = "_and_".join(reason_parts)
                    self.store.save_provisional_best(
                        spec.name,
                        seed,
                        completed,
                        checkpoint_payload,
                        cheap_validation=score,
                        cheap_metrics=cheap_metrics,
                        curriculum_stage=captured_stage_id,
                        capture_reason=reason,
                    )
            if spec.name in {"policy", "joint_policy", "a1", "a2"} and horizon == 96:
                self.store.save_k96_candidate(
                    spec.name,
                    seed,
                    epoch + 1,
                    checkpoint_payload,
                    cheap_validation=best,
                )
            # Advance the resumable latest pointer only after an immutable
            # K96 candidate has been durably registered.
            latest_path = self.store.save_checkpoint(
                spec.name,
                seed,
                checkpoint_payload,
            )
            self.store.write_state(
                {
                    "status": "TRAINING",
                    "phase": spec.name,
                    "seed": int(seed),
                    "epochs_completed": completed,
                    "phase_epochs": spec.epochs,
                    "horizon": horizon,
                    "checkpoint": str(latest_path),
                    "physical_days": physical_days,
                    "effective_days": physical_days,
                    "partial_tail_days": partial_tail_days,
                    "mean_training_loss": mean_training_loss,
                    "cheap_validation_loss": score,
                    "cheap_validation_metrics": cheap_metrics,
                    "gradient_evidence": gradient_evidence,
                    "scheduler_type": scheduler_type,
                    "learning_rates_before_step": learning_rates_before_step,
                    "learning_rates_after_step": learning_rates_after_step,
                    **run_identity,
                    "predecessor_checkpoint": self.predecessor_checkpoint,
                    "predecessor_sha256": self.predecessor_sha256,
                    **self._scenario_contract_payload(),
                    **self._distribution_contract_payload(),
                    "distribution_curriculum_stage_executed": captured_stage_id,
                    "distribution_curriculum_stage_index_executed": (
                        executed_distribution_stage_index
                    ),
                    "distribution_stage_index": distribution_stage_index,
                    "distribution_stage_epoch": distribution_stage_epoch,
                    "distribution_stage_completed": distribution_stage_completed,
                    "distribution_training_complete": distribution_training_complete,
                    "physical_microbatch_size": self.physical_batch_size,
                    "configured_effective_batch_size": reported_effective_days,
                    "scientific_effective_batch_days": FORMAL_TRAINING_DAYS,
                    "scientific_accumulation_steps": scientific_accumulation_steps,
                    "legacy_configured_effective_batch_size": (
                        EFFECTIVE_BATCH_SIZE if is_distributional_run else None
                    ),
                    "optimizer_effective_batch_days": FORMAL_TRAINING_DAYS,
                    "normalization_sha256": self.normalization_sha256,
                }
            )
            if spec.name == "distribution":
                if distribution_training_complete:
                    break
            else:
                current_patience = self._patience_at_epoch(spec, epoch)
                if completed >= spec.minimum_epochs and stale >= current_patience:
                    break

        candidate_registry_path: str | None = None
        selected_epoch: int | None = None
        if spec.name in {"policy", "joint_policy", "a1", "a2"}:
            gate, selected_path, selected_epoch = self._gate_and_select_k96_candidates(
                spec.name, int(seed)
            )
            if selected_path is not None:
                latest_path = selected_path
            candidate_registry_path = str(
                self.store.candidate_registry_path(spec.name, seed)
            )
            status = "COMPLETE" if gate.passed else "FAILED"
        elif spec.name == "distribution":
            assert distribution_contract is not None
            gate, selected_path, selected_epoch = (
                self._gate_and_select_distribution_candidates(
                    int(seed), distribution_contract
                )
            )
            if selected_path is not None:
                latest_path = selected_path
            candidate_registry_path = str(
                self.store.provisional_registry_path(spec.name, seed)
            )
            status = (
                "COMPLETE" if gate.passed
                else "FAILED" if gate.status == "FAIL"
                else gate.status
            )
        else:
            gate, selected_path, selected_epoch = self._gate_and_select_provisional_best(
                spec.name, int(seed)
            )
            if selected_path is not None:
                latest_path = selected_path
            candidate_registry_path = str(
                self.store.provisional_registry_path(spec.name, seed)
            )
            status = (
                "COMPLETE" if gate.passed
                else "FAILED" if gate.status == "FAIL"
                else gate.status
            )
        self.store.write_state(
            {
                "status": status,
                "phase": spec.name,
                "seed": int(seed),
                "epochs_completed": completed,
                "phase_epochs": spec.epochs,
                "checkpoint": None if latest_path is None else str(latest_path),
                "capability_gate": asdict(gate),
                "candidate_registry": candidate_registry_path,
                "selected_candidate_epoch": selected_epoch,
                **run_identity,
                "predecessor_checkpoint": self.predecessor_checkpoint,
                "predecessor_sha256": self.predecessor_sha256,
                **self._scenario_contract_payload(),
                **self._distribution_contract_payload(),
                "distribution_stage_index": distribution_stage_index,
                "distribution_stage_epoch": distribution_stage_epoch,
                "distribution_training_complete": (
                    spec.name != "distribution" or training_frozen
                    or bool(locals().get("distribution_training_complete", False))
                ),
                "physical_microbatch_size": self.physical_batch_size,
                "configured_effective_batch_size": reported_effective_days,
                "scientific_effective_batch_days": FORMAL_TRAINING_DAYS,
                "scientific_accumulation_steps": scientific_accumulation_steps,
                "legacy_configured_effective_batch_size": (
                    EFFECTIVE_BATCH_SIZE if is_distributional_run else None
                ),
                "optimizer_effective_batch_days": FORMAL_TRAINING_DAYS,
                "normalization_sha256": self.normalization_sha256,
            }
        )
        return PhaseRunResult(
            phase=spec.name,
            seed=int(seed),
            status=status,
            epochs_completed=completed,
            best_cheap_validation=None if math.isinf(best) else best,
            capability_gate=gate,
            checkpoint=None if latest_path is None else str(latest_path),
            trainable_parameters=trainable,
            reason=None if gate.passed else "; ".join(gate.reasons),
            epoch_coverage=tuple(coverage_records),
            candidate_registry=candidate_registry_path,
            selected_candidate_epoch=selected_epoch,
        )

    def _gate_and_select_provisional_best(
        self, phase: str, seed: int
    ) -> tuple[CapabilityGateResult, Path | None, int | None]:
        """Reload and fully evaluate the immutable cheap-validation best."""

        if phase == "representation":
            return self._gate_and_select_representation_candidates(seed)

        registry = self.store.read_provisional_registry(phase, seed)
        candidate = registry.get("best")
        if not isinstance(candidate, Mapping):
            return (
                CapabilityGateResult(
                    phase,
                    "NOT_READY",
                    0,
                    96,
                    reasons=("no immutable cheap-validation best checkpoint exists",),
                ),
                None,
                None,
            )
        path = Path(str(candidate["checkpoint"])).resolve()
        if not path.is_file() or sha256_file(path) != candidate.get("checkpoint_sha256"):
            raise R4ProtocolError("provisional best checkpoint is missing or hash-mismatched")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise R4ProtocolError("provisional best checkpoint must be a mapping")
        self.store._check_lineage(payload)
        epoch = int(candidate["epoch"])
        if self.store.epochs_completed(payload) != epoch:
            raise R4ProtocolError("provisional registry/checkpoint epoch mismatch")
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        gate = run_capability_gate(
            phase,
            self.phase_end_evaluator,
            checkpoint=path,
            epoch=epoch,
            model=self.model,
        )
        if not gate.passed:
            return gate, None, None
        selected_candidate = {
            **dict(candidate),
            "gate": asdict(gate),
        }
        selected_path = self.store.save_selected_best(
            phase, seed, selected_candidate
        )
        registry["selected_best"] = {
            "epoch": epoch,
            "checkpoint": str(selected_path),
            "source_checkpoint": str(path),
            "source_sha256": candidate["checkpoint_sha256"],
            "capability_gate_status": "PASSED",
        }
        self.store.write_provisional_registry(phase, seed, registry)
        return gate, selected_path, epoch

    def _gate_and_select_distribution_candidates(
        self,
        seed: int,
        contract: DistributionCurriculumContract,
    ) -> tuple[CapabilityGateResult, Path | None, int | None]:
        """Execute V6 Level1 -> Level2 -> Level3 without cheap-best leakage."""

        phase = "distribution"
        registry = self.store.read_provisional_registry(phase, seed)
        candidates = list(registry.get("candidates", ()))
        if not candidates:
            return (
                CapabilityGateResult(
                    phase,
                    "NOT_READY",
                    0,
                    96,
                    reasons=("Distribution produced no immutable Level1 candidates",),
                ),
                None,
                None,
            )
        selection = contract.candidate_selection
        pareto_names = tuple(str(value) for value in selection["pareto_metrics"])

        def cheap_vector(candidate: Mapping[str, Any]) -> tuple[float, ...]:
            metrics = candidate.get("cheap_metrics")
            if not isinstance(metrics, Mapping):
                raise R4ProtocolError(
                    "Distribution candidate lacks its cheap metric vector"
                )
            vector: list[float] = []
            for name in pareto_names:
                if name not in metrics:
                    raise R4ProtocolError(
                        f"Distribution candidate cheap metrics omit {name!r}"
                    )
                value = float(metrics[name])
                if not math.isfinite(value):
                    raise R4ProtocolError(
                        f"Distribution candidate metric {name!r} is non-finite"
                    )
                vector.append(value)
            return tuple(vector)

        vectors = [cheap_vector(candidate) for candidate in candidates]
        pareto_indices: list[int] = []
        for index, vector in enumerate(vectors):
            dominated = False
            for other_index, other in enumerate(vectors):
                if index == other_index:
                    continue
                if all(left <= right for left, right in zip(other, vector, strict=True)) and any(
                    left < right for left, right in zip(other, vector, strict=True)
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(index)

        top_k = int(selection["top_k"])
        top_indices = sorted(
            range(len(candidates)),
            key=lambda index: (
                float(candidates[index]["cheap_validation"]),
                int(candidates[index]["epoch"]),
            ),
        )[:top_k]
        stage_end_indices = [
            index
            for index, candidate in enumerate(candidates)
            if "stage_end" in str(candidate.get("capture_reason", ""))
        ]
        scheduled_indices = sorted(
            range(len(candidates)), key=lambda index: int(candidates[index]["epoch"])
        )
        maximum = int(selection["max_candidates"])
        selected_indices: list[int] = []
        reasons_by_index: dict[int, list[str]] = {}

        def include(indices: Sequence[int], reason: str) -> None:
            for index in indices:
                reasons_by_index.setdefault(index, []).append(reason)
                if index not in selected_indices and len(selected_indices) < maximum:
                    selected_indices.append(index)

        include(stage_end_indices, "stage_end")
        include(top_indices, "top_k")
        include(
            sorted(
                pareto_indices,
                key=lambda index: (
                    float(candidates[index]["cheap_validation"]),
                    int(candidates[index]["epoch"]),
                ),
            ),
            "pareto",
        )
        include(scheduled_indices, "scheduled_fill")
        selected_set = set(selected_indices)
        for index, candidate in enumerate(candidates):
            candidate["level1_selected"] = index in selected_set
            candidate["level1_reasons"] = reasons_by_index.get(index, [])
            if index not in selected_set:
                candidate["gate_status"] = "SKIPPED_LEVEL1"
                candidate["gate"] = None
        registry["candidates"] = candidates
        registry["frozen"] = True
        registry["level1"] = {
            "mode": selection["mode"],
            "top_k": top_k,
            "pareto_metrics": list(pareto_names),
            "max_candidates": maximum,
            "selected_epochs": [int(candidates[index]["epoch"]) for index in selected_indices],
            "full_gate_not_run_during_training": True,
        }
        self.store.write_provisional_registry(phase, seed, registry)
        if self.phase_end_evaluator is None:
            return (
                CapabilityGateResult(
                    phase,
                    "NOT_READY",
                    0,
                    96,
                    reasons=("Distribution Level2 formal evaluator is not connected",),
                ),
                None,
                None,
            )

        for index in selected_indices:
            registry = self.store.read_provisional_registry(phase, seed)
            candidates = list(registry["candidates"])
            candidate = candidates[index]
            status = str(candidate.get("gate_status", "PENDING"))
            if status in {"PASS", "FAIL"}:
                if not isinstance(candidate.get("gate"), Mapping):
                    raise R4ProtocolError(
                        "terminal Distribution candidate lacks Level2 evidence"
                    )
                continue
            if status != "PENDING":
                raise R4ProtocolError(
                    f"unknown Distribution candidate gate status: {status}"
                )
            path = Path(str(candidate["checkpoint"])).resolve()
            if not path.is_file():
                raise R4ProtocolError("Distribution candidate checkpoint is missing")
            stat = path.stat()
            if (
                int(candidate.get("checkpoint_size_bytes", -1)) != stat.st_size
                or int(candidate.get("checkpoint_mtime_ns", -1)) != stat.st_mtime_ns
            ):
                raise R4ProtocolError(
                    "Distribution candidate changed after its one-time write hash"
                )
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, Mapping):
                raise R4ProtocolError("Distribution candidate must be a mapping")
            self.store._check_lineage(payload)
            if self.store.epochs_completed(payload) != int(candidate["epoch"]):
                raise R4ProtocolError(
                    "Distribution candidate registry/checkpoint epoch mismatch"
                )
            self._validate_distribution_contract_for_phase(phase, payload)
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.model.eval()
            gate = run_capability_gate(
                phase,
                self.phase_end_evaluator,
                checkpoint=path,
                epoch=int(candidate["epoch"]),
                model=self.model,
                checkpoint_sha256=str(candidate["checkpoint_sha256"]),
            )
            candidate["gate_status"] = gate.status
            candidate["gate"] = asdict(gate)
            candidates[index] = candidate
            registry["candidates"] = candidates
            self.store.write_provisional_registry(phase, seed, registry)

        registry = self.store.read_provisional_registry(phase, seed)
        candidates = list(registry["candidates"])
        selected_candidates = [
            candidate for candidate in candidates if candidate.get("level1_selected")
        ]
        pending = [
            candidate
            for candidate in selected_candidates
            if candidate.get("gate_status") == "PENDING"
        ]
        if pending:
            raise R4ProtocolError(
                "cannot perform Distribution Level3 selection with pending Level2 gates"
            )
        passing = [
            candidate
            for candidate in selected_candidates
            if candidate.get("gate_status") == "PASS"
        ]
        if not passing:
            reasons = tuple(
                f"epoch {candidate['epoch']}: "
                + "; ".join((candidate.get("gate") or {}).get("reasons", ("failed",)))
                for candidate in selected_candidates
            )
            return (
                CapabilityGateResult(phase, "FAIL", 0, 96, reasons=reasons),
                None,
                None,
            )

        metric_aliases: dict[str, tuple[str, ...]] = {
            "energy_score_delta_one_sided_upper": (
                "energy_score_delta_one_sided_upper",
                "energy_score_delta_upper",
            ),
            "marginal_crps": ("model_marginal_crps", "marginal_crps"),
            "coverage95_error": ("absolute_coverage_error_95", "coverage95_error"),
            "tail_score_noninferiority_one_sided_upper": (
                "tail_score_noninferiority_one_sided_upper",
                "tail_noninferiority_one_sided_upper",
            ),
        }

        def formal_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
            gate = candidate.get("gate")
            metrics = gate.get("metrics") if isinstance(gate, Mapping) else None
            if not isinstance(metrics, Mapping):
                raise R4ProtocolError("passing Distribution candidate lacks metrics")
            rank: list[float] = []
            for requested in selection.get("pass_rank", ()):
                name = str(requested)
                if name == "epoch":
                    rank.append(float(candidate["epoch"]))
                    continue
                aliases = metric_aliases.get(name, (name,))
                found = next((metrics[key] for key in aliases if key in metrics), None)
                if found is None or not math.isfinite(float(found)):
                    raise R4ProtocolError(
                        f"passing Distribution gate omits rank metric {name!r}"
                    )
                rank.append(float(found))
            return tuple(rank)

        selected = min(passing, key=formal_rank)
        selected_path = self.store.save_selected_best(
            phase,
            seed,
            selected,
            use_registered_candidate_digest=True,
        )
        registry["selected_best"] = {
            "epoch": int(selected["epoch"]),
            "checkpoint": str(selected_path),
            "source_checkpoint": selected["checkpoint"],
            "source_sha256": selected["checkpoint_sha256"],
            "capability_gate_status": "PASSED",
            "selection_level": 3,
            "selection_order": list(selection.get("pass_rank", ())),
        }
        self.store.write_provisional_registry(phase, seed, registry)
        selected_gate = selected["gate"]
        self.model.load_state_dict(
            torch.load(selected_path, map_location="cpu", weights_only=False)[
                "model_state"
            ],
            strict=True,
        )
        return (
            CapabilityGateResult(
                phase=phase,
                status="PASS",
                evaluated_days=int(selected_gate["evaluated_days"]),
                horizon=selected_gate.get("horizon"),
                metrics={
                    str(key): float(value)
                    for key, value in selected_gate["metrics"].items()
                },
                reasons=tuple(selected_gate.get("reasons", ())),
                dates=tuple(selected_gate.get("dates", ())),
            ),
            selected_path,
            int(selected["epoch"]),
        )

    def _gate_and_select_representation_candidates(
        self, seed: int
    ) -> tuple[CapabilityGateResult, Path | None, int | None]:
        """Gate every immutable improvement and select by the paper primary.

        Aggregate differentiable validation loss is not the paper selection
        metric and can improve while a per-head paired confidence bound
        deteriorates.  Every checkpoint that was frozen *before* the gate is
        therefore evaluated once; only six-head PASS candidates compete, with
        DART NMAE as the registered primary ranking metric.
        """

        phase = "representation"
        registry = self.store.read_provisional_registry(phase, seed)
        candidates = list(registry.get("candidates", ()))
        if not candidates:
            return (
                CapabilityGateResult(
                    phase,
                    "NOT_READY",
                    0,
                    96,
                    reasons=("no immutable representation candidates exist",),
                ),
                None,
                None,
            )
        for index, candidate in enumerate(candidates):
            prior_status = str(candidate.get("gate_status", "PENDING"))
            if prior_status in {"PASS", "FAIL"}:
                path = Path(str(candidate["checkpoint"])).resolve()
                if (
                    not path.is_file()
                    or sha256_file(path) != candidate.get("checkpoint_sha256")
                    or not isinstance(candidate.get("gate"), Mapping)
                ):
                    raise R4ProtocolError(
                        "terminal representation candidate evidence is invalid"
                    )
                continue
            if prior_status != "PENDING":
                raise R4ProtocolError(
                    f"unknown representation candidate gate status: {prior_status}"
                )
            path = Path(str(candidate["checkpoint"])).resolve()
            if not path.is_file() or sha256_file(path) != candidate.get(
                "checkpoint_sha256"
            ):
                raise R4ProtocolError(
                    "representation candidate is missing or hash-mismatched"
                )
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, Mapping):
                raise R4ProtocolError("representation candidate must be a mapping")
            self.store._check_lineage(payload)
            epoch = int(candidate["epoch"])
            if self.store.epochs_completed(payload) != epoch:
                raise R4ProtocolError(
                    "representation candidate registry/checkpoint epoch mismatch"
                )
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.model.eval()
            gate = run_capability_gate(
                phase,
                self.phase_end_evaluator,
                checkpoint=path,
                epoch=epoch,
                model=self.model,
            )
            candidate = {
                **dict(candidate),
                "gate_status": gate.status,
                "gate": asdict(gate),
            }
            candidates[index] = candidate
            registry["candidates"] = candidates
            self.store.write_provisional_registry(phase, seed, registry)

        registry = self.store.read_provisional_registry(phase, seed)
        candidates = list(registry["candidates"])
        passing = [item for item in candidates if item.get("gate_status") == "PASS"]
        if not passing:
            reasons = tuple(
                f"epoch {item['epoch']}: "
                + "; ".join((item.get("gate") or {}).get("reasons", ("failed",)))
                for item in candidates
            )
            return CapabilityGateResult(phase, "FAIL", 0, 96, reasons=reasons), None, None

        def rank(item: Mapping[str, Any]) -> tuple[float, float, float, int]:
            metrics = item["gate"]["metrics"]
            return (
                float(metrics["model_dart_nmae"]),
                float(metrics["dart_nmae_delta_one_sided_upper"]),
                float(item["cheap_validation"]),
                int(item["epoch"]),
            )

        selected = min(passing, key=rank)
        selected_path = self.store.save_selected_best(phase, seed, selected)
        registry["selected_best"] = {
            "epoch": int(selected["epoch"]),
            "checkpoint": str(selected_path),
            "source_checkpoint": selected["checkpoint"],
            "source_sha256": selected["checkpoint_sha256"],
            "capability_gate_status": "PASSED",
            "selection_order": [
                "lowest_model_dart_nmae",
                "lower_dart_paired_upper",
                "lower_cheap_validation",
                "lower_epoch",
            ],
        }
        self.store.write_provisional_registry(phase, seed, registry)
        selected_gate = selected["gate"]
        self.model.load_state_dict(
            torch.load(selected_path, map_location="cpu", weights_only=False)[
                "model_state"
            ],
            strict=True,
        )
        return (
            CapabilityGateResult(
                phase=phase,
                status="PASS",
                evaluated_days=int(selected_gate["evaluated_days"]),
                horizon=selected_gate.get("horizon"),
                metrics={
                    str(key): float(value)
                    for key, value in selected_gate["metrics"].items()
                },
                reasons=tuple(selected_gate.get("reasons", ())),
                dates=tuple(selected_gate.get("dates", ())),
            ),
            selected_path,
            int(selected["epoch"]),
        )

    def _gate_and_select_k96_candidates(
        self, phase: str, seed: int
    ) -> tuple[CapabilityGateResult, Path | None, int | None]:
        registry = self.store.freeze_candidate_registry(phase, seed)
        if self.phase_end_evaluator is None:
            return (
                CapabilityGateResult(
                    phase,
                    "NOT_READY",
                    0,
                    96,
                    reasons=("K96 hard-gate evaluator is not connected",),
                ),
                None,
                None,
            )
        candidates = list(registry["candidates"])
        for index, candidate in enumerate(candidates):
            status = str(candidate.get("gate_status", "PENDING"))
            path = Path(str(candidate["checkpoint"])).resolve()
            is_v6_candidate = candidate.get(
                "distribution_training_contract_sha256"
            ) is not None

            def candidate_binding_valid() -> bool:
                if not path.is_file():
                    return False
                if is_v6_candidate:
                    stat = path.stat()
                    return (
                        int(candidate.get("checkpoint_size_bytes", -1))
                        == stat.st_size
                        and int(candidate.get("checkpoint_mtime_ns", -1))
                        == stat.st_mtime_ns
                    )
                return sha256_file(path) == candidate.get("checkpoint_sha256")

            if status in {"PASS", "FAIL"}:
                # Resume skips only a hash-verified immutable candidate with a
                # persisted terminal gate result.
                if not candidate_binding_valid():
                    raise R4ProtocolError("terminal candidate checkpoint hash mismatch")
                continue
            if status != "PENDING":
                raise R4ProtocolError(f"unknown candidate gate status: {status}")
            if not candidate_binding_valid():
                raise R4ProtocolError("pending candidate checkpoint hash mismatch")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, Mapping):
                raise R4ProtocolError("K96 candidate checkpoint must be a mapping")
            self.store._check_lineage(payload)
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
            try:
                raw = _invoke_candidate_evaluator(
                    self.phase_end_evaluator,
                    phase=phase,
                    checkpoint=path,
                    epoch=int(candidate["epoch"]),
                    model=self.model,
                    checkpoint_sha256=(
                        str(candidate["checkpoint_sha256"])
                        if is_v6_candidate
                        else None
                    ),
                )
            except Exception:
                # External interruption remains PENDING so resume can continue
                # from this exact candidate without falsely terminalizing it.
                raise
            try:
                gate = validate_decision_candidate_gate(phase, raw)
            except R4ProtocolError as exc:
                gate = {
                    "status": "FAIL",
                    "phase": phase,
                    "horizon": 96,
                    "evaluated_days": 0,
                    "validation_origins": 0,
                    "dates": [],
                    "metrics": {},
                    "reasons": [str(exc)],
                }
            candidate["gate_status"] = gate["status"]
            candidate["gate"] = gate
            candidates[index] = candidate
            registry["candidates"] = candidates
            self.store.write_candidate_registry(phase, seed, registry)

        registry = self.store.read_candidate_registry(phase, seed)
        candidates = list(registry["candidates"])
        pending = [item for item in candidates if item.get("gate_status") == "PENDING"]
        if pending:
            raise R4ProtocolError("cannot select best while any K96 candidate gate is pending")
        passing = [item for item in candidates if item.get("gate_status") == "PASS"]
        if not passing:
            reasons = tuple(
                f"epoch {item['epoch']}: "
                + "; ".join((item.get("gate") or {}).get("reasons", ("failed",)))
                for item in candidates
            )
            return CapabilityGateResult(phase, "FAIL", 0, 96, reasons=reasons), None, None

        def rank(item: Mapping[str, Any]) -> tuple[float, float, float, int]:
            metrics = item["gate"]["metrics"]
            return (
                float(metrics["operating_cost"]),
                float(metrics["cvar"]),
                float(metrics["corrective_magnitude"]),
                int(item["epoch"]),
            )

        selected = min(passing, key=rank)
        selected_path = self.store.save_selected_best(
            phase,
            seed,
            selected,
            use_registered_candidate_digest=(
                selected.get("distribution_training_contract_sha256") is not None
            ),
        )
        registry["selected_best"] = {
            "epoch": int(selected["epoch"]),
            "checkpoint": str(selected_path),
            "source_checkpoint": selected["checkpoint"],
            "source_sha256": selected["checkpoint_sha256"],
            "selection_order": [
                "lowest_operating_cost",
                "lower_cvar",
                "lower_corrective_magnitude",
                "lower_epoch",
            ],
        }
        self.store.write_candidate_registry(phase, seed, registry)
        self.model.load_state_dict(
            torch.load(selected_path, map_location="cpu", weights_only=False)["model_state"]
        )
        selected_gate = selected["gate"]
        return (
            CapabilityGateResult(
                phase=phase,
                status="PASS",
                evaluated_days=int(selected_gate["evaluated_days"]),
                horizon=96,
                metrics={
                    key: float(value)
                    for key, value in selected_gate["metrics"].items()
                },
                dates=tuple(selected_gate["dates"]),
            ),
            selected_path,
            int(selected["epoch"]),
        )

    @staticmethod
    def _horizon_at_epoch(spec: PhaseSpec, epoch: int) -> int:
        if not spec.curriculum:
            return 96
        cursor = 0
        for stage in spec.curriculum:
            cursor += stage.epochs
            if epoch < cursor:
                return stage.horizon
        return spec.curriculum[-1].horizon

    @staticmethod
    def _patience_at_epoch(spec: PhaseSpec, epoch: int) -> int:
        if not spec.curriculum:
            return spec.patience
        cursor = 0
        for stage in spec.curriculum:
            cursor += stage.epochs
            if epoch < cursor:
                # K=8/24/48 are fixed-length curriculum stages.  Early
                # stopping is enabled only for the final K=96 stage (p=20).
                return spec.epochs + 1 if stage.patience is None else stage.patience
        return spec.patience

    @staticmethod
    def _stage_start_epochs(spec: PhaseSpec) -> tuple[int, ...]:
        starts: list[int] = []
        cursor = 0
        for stage in spec.curriculum:
            starts.append(cursor)
            cursor += stage.epochs
        return tuple(starts)


__all__ = [
    "AtomicRunStore",
    "CAPABILITY_GATE_DAYS",
    "CapabilityGateResult",
    "CheapValidationPolicy",
    "CurriculumStage",
    "DISTRIBUTIONAL_PHASE_ORDER",
    "DISTRIBUTIONAL_R4_PROTOCOL_VERSION",
    "DistributionCurriculumContract",
    "DistributionCurriculumStage",
    "EFFECTIVE_BATCH_SIZE",
    "EpochCoverage",
    "FORMAL_TRAINING_DAYS",
    "HARD_GATE_VALIDATION_ORIGINS",
    "PHASE_ORDER",
    "PHYSICAL_BATCH_FALLBACK",
    "PhaseRunResult",
    "PhaseLossComputer",
    "PhaseSpec",
    "R4PhaseTrainer",
    "R4ProtocolError",
    "R4_PROTOCOL_VERSION",
    "R4_PHASE_SPECS",
    "R4_SEEDS",
    "accumulation_for_physical_batch",
    "build_phase_loss",
    "distribution_curriculum_contract",
    "exact_weighted_cvar",
    "phase_spec",
    "run_capability_gate",
    "unbiased_tail_normalized_mean",
    "validate_decision_candidate_gate",
]
