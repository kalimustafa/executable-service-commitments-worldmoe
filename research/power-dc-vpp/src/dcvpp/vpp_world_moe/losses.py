"""R5-V2 proper-scoring objectives for the unified VPP world model.

This module is the formal R5-V2 training contract.  It deliberately fails
closed when a phase does not expose the probabilistic objects needed by its
registered objective.  The frozen R5-V1 ablation lives in
``losses_legacy_v1.py`` and is never selected implicitly.

Neural forward passes may run under BF16 autocast.  Every score, likelihood,
risk functional, and constraint reduction below is explicitly promoted to
FP32.  Atomic score terms retain a numerator and effective valid count so an
epoch can be aggregated independently of its batch partition.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .coverage import public_family_channel_weights
from .data import QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS


PhaseName = Literal["w0", "w1", "u0", "p1", "j1", "c1"]
TRAINING_PHASES: tuple[PhaseName, ...] = ("w0", "w1", "u0", "p1", "j1")
DYNAMICS_HORIZONS: tuple[int, ...] = (1, 4, 16, 96)
QUANTILE_LEVELS: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)
PUBLIC_GROUP_SLICES: Mapping[str, slice] = {
    "price": slice(0, 12),
    "workload": slice(12, 13),
    "pv": slice(13, 17),
    "pue": slice(17, 21),
}


class WorldLossError(ValueError):
    """A phase packet cannot support its registered R5-V2 objective."""


@dataclass(frozen=True)
class ScoreTerm:
    """An exactly aggregable score numerator and effective valid count."""

    numerator: Tensor
    valid_count: Tensor
    # Value checks are deliberately deferred to PhaseLoss.  This lets one
    # phase-wide device-to-host health snapshot replace several synchronizing
    # scalar reads for every atomic term.  Direct constructors historically
    # allowed a zero-count/zero-numerator term, hence the permissive default;
    # _score_term records its stricter allow_empty contract explicitly.
    allow_empty: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        if self.numerator.ndim != 0 or self.valid_count.ndim != 0:
            raise WorldLossError("ScoreTerm numerator and valid_count must be scalars")
        if self.numerator.dtype != torch.float32:
            raise WorldLossError("ScoreTerm numerator must be FP32")
        if self.valid_count.dtype != torch.float32:
            raise WorldLossError("ScoreTerm valid_count must be FP32")
        if not isinstance(self.allow_empty, bool):
            raise WorldLossError("ScoreTerm allow_empty must be boolean")
        if not isinstance(self.label, str):
            raise WorldLossError("ScoreTerm label must be a string")

    @property
    def scalar(self) -> Tensor:
        # Conditional scores such as positive-only queue severity can have no
        # observations in an individual sparse batch.  Keep the named term so
        # global numerator/count aggregation remains batch-partition invariant.
        return torch.where(
            self.valid_count > 0,
            self.numerator / self.valid_count.clamp_min(1.0),
            torch.zeros_like(self.numerator),
        )

    def detached(self) -> dict[str, float]:
        packed = torch.stack(
            (self.numerator.detach(), self.valid_count.detach(), self.scalar.detach())
        ).to(device="cpu", dtype=torch.float32)
        numerator, valid_count, scalar = packed.tolist()
        return {
            "numerator": float(numerator),
            "valid_count": float(valid_count),
            "scalar": float(scalar),
        }


def merge_score_terms(terms: Iterable[ScoreTerm]) -> ScoreTerm:
    """Merge batch terms without ever averaging their batch means."""

    materialized = tuple(terms)
    if not materialized:
        raise WorldLossError("cannot merge an empty ScoreTerm sequence")
    numerator = torch.stack([term.numerator for term in materialized]).sum()
    valid_count = torch.stack([term.valid_count for term in materialized]).sum()
    return ScoreTerm(numerator.float(), valid_count.float())


def aggregate_score_terms(
    batches: Iterable[Mapping[str, ScoreTerm]],
) -> dict[str, ScoreTerm]:
    """Aggregate named atomic scores across arbitrarily sized batches."""

    grouped: dict[str, list[ScoreTerm]] = {}
    for batch in batches:
        for name, term in batch.items():
            grouped.setdefault(name, []).append(term)
    if not grouped:
        raise WorldLossError("no score terms were supplied for aggregation")
    return {name: merge_score_terms(values) for name, values in grouped.items()}


@dataclass(frozen=True)
class WorldLossWeights:
    """Frozen R5-V2 phase weights from the experimental protocol."""

    # W0
    quantile: float = 1.0
    count: float = 0.25
    state: float = 0.25
    # W1
    latent: float = 0.5
    rollout: float = 0.5
    rollout_state: float = 0.1
    # U0
    crps: float = 0.30
    energy: float = 0.30
    variogram: float = 0.15
    tail: float = 0.15
    mixture_nll: float = 0.10
    atom_brier: float = 0.0
    # P1
    expected_cost: float = 1.0
    cvar: float = 0.25
    augmented_lagrangian: float = 1.0
    ramp: float = 0.02
    shield_correction: float = 0.05
    cost_critic_distillation: float = 0.0
    # Shared
    auxiliary: float = 0.01
    # Training-only teacher supervision for the causal tail router.  This is
    # deliberately separate from every proper score and checkpoint selector.
    route_supervision: float = 0.05


@dataclass(frozen=True)
class PhaseLoss:
    """Differentiable objective plus exact atomic score accounting."""

    phase: PhaseName
    total: Tensor
    components: Mapping[str, Tensor]
    score_terms: Mapping[str, ScoreTerm] = field(default_factory=dict)
    optimization_objectives: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.total, Tensor) or self.total.ndim != 0:
            raise WorldLossError("PhaseLoss total must be one scalar Tensor")
        device = self.total.device
        checks: list[tuple[str, str, Tensor]] = []

        def scalar(value: Any, *, label: str) -> Tensor:
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise WorldLossError(f"{label} must reduce to one scalar Tensor")
            if value.device != device:
                raise WorldLossError(f"{label} must be on the PhaseLoss device")
            return value

        def check(kind: str, label: str, predicate: Tensor) -> None:
            candidate = scalar(predicate, label=f"{label} health predicate")
            checks.append((kind, label, candidate.detach().bool()))

        check("total_finite", f"{self.phase.upper()} total", torch.isfinite(self.total))
        for name, value in self.components.items():
            component = scalar(value, label=f"PhaseLoss component {name}")
            check(
                "component_finite",
                f"PhaseLoss component {name}",
                torch.isfinite(component),
            )
        for name, objective in self.optimization_objectives.items():
            value = scalar(objective, label=f"PhaseLoss objective {name}")
            check(
                "objective_finite",
                f"PhaseLoss objective {name}",
                torch.isfinite(value),
            )
        for name, term in self.score_terms.items():
            if not isinstance(term, ScoreTerm):
                raise WorldLossError(f"PhaseLoss score term {name} is not a ScoreTerm")
            numerator = scalar(
                term.numerator, label=f"ScoreTerm {name} numerator"
            )
            valid_count = scalar(
                term.valid_count, label=f"ScoreTerm {name} valid_count"
            )
            term_label = term.label or name
            check(
                "score_numerator_finite",
                term_label,
                torch.isfinite(numerator),
            )
            check(
                "score_count_finite",
                term_label,
                torch.isfinite(valid_count),
            )
            check(
                "score_count_nonnegative",
                term_label,
                valid_count >= 0.0,
            )
            check(
                "score_empty_zero",
                term_label,
                (valid_count != 0.0) | (numerator == 0.0),
            )
            if not term.allow_empty:
                check(
                    "score_nonempty",
                    term_label,
                    valid_count != 0.0,
                )

        if self.phase == "w0":
            required = {"general_objective", "rare_objective"}
            present = required.intersection(self.components)
            # R5-V2.1 proper W0 exposes this split.  Legacy fixed-loss ablations
            # deliberately retain their historical scalar PhaseLoss contract.
            if present and present != required:
                raise WorldLossError(
                    "W0 must expose general_objective and rare_objective for PCGrad"
                )
            if present:
                general = scalar(
                    self.components["general_objective"],
                    label="W0 general objective",
                )
                rare = scalar(
                    self.components["rare_objective"], label="W0 rare objective"
                )
                check(
                    "w0_objective_sum",
                    "W0 general/rare objectives",
                    torch.isclose(
                        self.total.detach().float(),
                        (general + rare).detach().float(),
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    ),
                )
        if self.phase == "j1":
            expected = {"world", "uncertainty", "policy"}
            if set(self.optimization_objectives) != expected:
                raise WorldLossError(
                    "J1 must expose exactly world/uncertainty/policy optimization "
                    "objectives for PCGrad"
                )
            if self.total is not self.optimization_objectives["policy"]:
                raise WorldLossError(
                    "J1 total is the policy logging scalar; scalar objective "
                    "aggregation is prohibited"
                )

        # One packed health snapshot is the only device-to-host synchronization
        # performed by PhaseLoss construction, irrespective of term count.
        if checks:
            passed = torch.stack([item[2] for item in checks]).to(
                device="cpu", dtype=torch.bool
            )
            for (kind, label, _), ok in zip(checks, passed.tolist(), strict=True):
                if ok:
                    continue
                if kind in {
                    "total_finite",
                    "component_finite",
                    "objective_finite",
                }:
                    raise WorldLossError(f"{label} is non-finite")
                if kind == "score_numerator_finite":
                    raise WorldLossError(f"{label} numerator is non-finite")
                if kind == "score_count_finite":
                    raise WorldLossError(f"{label} valid_count is non-finite")
                if kind == "score_count_nonnegative":
                    raise WorldLossError(f"{label} valid_count cannot be negative")
                if kind == "score_empty_zero":
                    raise WorldLossError(
                        f"empty {label} ScoreTerm must have a zero numerator"
                    )
                if kind == "score_nonempty":
                    raise WorldLossError(f"{label} has no valid observations")
                if kind == "w0_objective_sum":
                    raise WorldLossError(
                        "W0 general/rare objectives must sum exactly to total"
                    )
                raise WorldLossError(f"unknown PhaseLoss health failure: {kind}")

    def detached(
        self, *, extra: Mapping[str, Tensor] | None = None
    ) -> dict[str, float]:
        values: dict[str, Tensor] = dict(self.components)
        values.update(
            {
                f"objective/{name}": value
                for name, value in self.optimization_objectives.items()
            }
        )
        for name, term in self.score_terms.items():
            values[f"{name}/numerator"] = term.numerator
            values[f"{name}/valid_count"] = term.valid_count
        for name, value in (extra or {}).items():
            if name in values:
                raise WorldLossError(f"detached scalar key is duplicated: {name}")
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise WorldLossError(f"detached extra scalar {name} is invalid")
            if value.device != self.total.device:
                raise WorldLossError(
                    f"detached extra scalar {name} must be on the PhaseLoss device"
                )
            values[name] = value
        if not values:
            return {}
        names = tuple(values)
        packed = torch.stack(
            tuple(values[name].detach().to(dtype=torch.float32) for name in names)
        ).to(device="cpu", dtype=torch.float32)
        return {
            name: float(value)
            for name, value in zip(names, packed.tolist(), strict=True)
        }


def packet_mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping view for a mapping, dataclass, or packet object."""

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
        if name in current and current[name] is not None:
            return current[name]
    if required:
        raise WorldLossError(f"packet is missing one of {list(names)}")
    return None


def _tensor(value: Any, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise WorldLossError(f"{label} must be a Tensor")
    return value.float()


def _fp32(value: Tensor) -> Tensor:
    return value.to(dtype=torch.float32)


def _zero(reference: Tensor) -> Tensor:
    return reference.float().sum() * 0.0


def _finite(name: str, value: Tensor) -> Tensor:
    if value.ndim != 0:
        raise WorldLossError(f"{name} must reduce to one scalar")
    # PhaseLoss performs the single packed finite-value synchronization.
    return value.float()


def _broadcast(value: Tensor, shape: torch.Size, *, label: str) -> Tensor:
    candidate = value.float()
    while candidate.ndim < len(shape):
        candidate = candidate.unsqueeze(0)
    try:
        return torch.broadcast_to(candidate, shape)
    except RuntimeError as exc:
        raise WorldLossError(f"{label} is not broadcastable to {tuple(shape)}") from exc


def _mask(value: Any | None, shape: torch.Size, *, label: str) -> Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.float32)
    tensor = _tensor(value, label=label)
    expanded = _broadcast(tensor, shape, label=label)
    if not bool(torch.isfinite(expanded).all()):
        raise WorldLossError(f"{label} contains non-finite values")
    return (expanded > 0).float()


def _score_term(
    values: Tensor,
    *,
    mask: Tensor | None = None,
    element_weights: Tensor | None = None,
    label: str,
    allow_empty: bool = False,
) -> ScoreTerm:
    values32 = values.float()
    if mask is None:
        effective = torch.ones_like(values32)
    else:
        effective = _broadcast(mask.float(), values32.shape, label=f"{label} mask")
    if element_weights is not None:
        effective = effective * _broadcast(
            element_weights.float(), values32.shape, label=f"{label} weights"
        )
    valid = effective > 0
    numerator = (
        torch.where(valid, values32, torch.zeros_like(values32)) * effective
    ).sum()
    valid_count = effective.sum()
    return ScoreTerm(
        numerator.float(),
        valid_count.float(),
        allow_empty=allow_empty,
        label=label,
    )


def _register_term(
    name: str,
    term: ScoreTerm,
    *,
    components: dict[str, Tensor],
    score_terms: dict[str, ScoreTerm],
) -> None:
    score_terms[name] = term
    components[name] = term.scalar


def _weighted_scalar(
    terms: Mapping[str, ScoreTerm], weights: Mapping[str, float], *, label: str
) -> Tensor:
    missing = set(weights).difference(terms)
    if missing:
        raise WorldLossError(f"{label} is missing score terms {sorted(missing)}")
    total_weight = sum(float(value) for value in weights.values())
    if total_weight <= 0.0:
        raise WorldLossError(f"{label} weights must have positive sum")
    return (
        sum(terms[name].scalar * float(weight) for name, weight in weights.items())
        / total_weight
    )


def _auxiliary_loss(packet: Any, reference: Tensor) -> Tensor:
    raw = _lookup(
        packet,
        "router_aux_loss",
        "auxiliary_losses",
        "moe_auxiliary",
        required=False,
    )
    if raw is None:
        return _zero(reference)
    if isinstance(raw, Tensor):
        return raw.float().mean()
    if isinstance(raw, Mapping):
        values = [
            _tensor(value, label=f"auxiliary_losses.{name}").mean()
            for name, value in raw.items()
        ]
        return torch.stack(values).sum() if values else _zero(reference)
    raise WorldLossError("auxiliary losses must be a Tensor or mapping")


def _queue_tail_route_supervision_loss(
    output: Any,
    batch: Any,
    reference: Tensor,
) -> Tensor:
    """Train the causal tail gate from an aggregate teacher event.

    The target is deliberately read here, after the model forward has already
    completed.  It is not a WorldInputPacket field and therefore cannot control
    expert activation or leak into inference.  A train-frozen prevalence
    supplies the positive weight, so the rare-event gradient does not depend on
    the accidental composition of an individual minibatch.
    """

    target_value = _lookup(batch, "queue_risk_event_target", required=False)
    if target_value is None:
        return _zero(reference)
    metadata = packet_mapping(_lookup(batch, "target_metadata"))
    threshold = float(
        metadata.get("queue_occurrence_threshold_work_units", float("nan"))
    )
    if not math.isclose(
        threshold,
        QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise WorldLossError(
            "queue tail-route supervision requires the registered 1e-4 threshold"
        )
    if metadata.get("aggregate_queue_event_rule") != "any_node_q_gt_threshold":
        raise WorldLossError("queue tail-route supervision has an unknown aggregation")

    target = _tensor(target_value, label="queue_risk_event_target")
    if target.shape[-1:] != (1,):
        raise WorldLossError("queue tail-route target must have [B,T,1] shape")
    if not bool(torch.isfinite(target).all()):
        raise WorldLossError("queue tail-route supervision contains non-finite values")
    if bool(((target != 0.0) & (target != 1.0)).any()):
        raise WorldLossError("queue tail-route teacher target must be binary")

    mask_value = _lookup(batch, "queue_risk_event_mask", required=False)
    valid_mask = (
        torch.ones_like(target, dtype=torch.bool)
        if mask_value is None
        else _broadcast(
            _tensor(mask_value, label="queue_risk_event_mask"),
            target.shape,
            label="queue_risk_event_mask",
        )
        > 0
    )
    prevalence = _tensor(
        _lookup(batch, "queue_risk_event_train_prevalence"),
        label="queue_risk_event_train_prevalence",
    )
    if prevalence.numel() != 1 or not bool(torch.isfinite(prevalence).all()):
        raise WorldLossError("train queue-event prevalence must be one finite scalar")
    prevalence = prevalence.reshape(()).clamp(1.0e-4, 1.0 - 1.0e-4)
    positive_weight = ((1.0 - prevalence) / prevalence).clamp(1.0, 100.0)
    configured_weight = _lookup(
        output, "queue_tail_route_positive_weight", required=False
    )
    if configured_weight is not None:
        configured = float(configured_weight)
        expected = float(positive_weight.detach().cpu())
        if not math.isclose(configured, expected, rel_tol=5.0e-3, abs_tol=5.0e-3):
            raise WorldLossError(
                "tail router calibration weight differs from its frozen weighted-BCE "
                "supervision weight"
            )
    diagnostics = _lookup(output, "moe_tail_diagnostics")
    if isinstance(diagnostics, (str, bytes)) or not isinstance(diagnostics, Sequence):
        raise WorldLossError("moe_tail_diagnostics must be a five-block sequence")
    if len(diagnostics) != 5:
        raise WorldLossError("queue tail supervision requires all five MoE gates")
    per_block: list[Tensor] = []
    for block_index, diagnostic in enumerate(diagnostics):
        logits = _tensor(
            _lookup(diagnostic, "tail_route_logits"),
            label=f"moe_tail_diagnostics[{block_index}].tail_route_logits",
        )
        if logits.shape != target.shape:
            raise WorldLossError(
                "each tail-route gate must expose logits with [B,T,1] target shape"
            )
        if not bool(torch.isfinite(logits).all()):
            raise WorldLossError("tail-route gate contains non-finite logits")
        raw = F.binary_cross_entropy_with_logits(
            logits.float(),
            target.float(),
            pos_weight=positive_weight,
            reduction="none",
        )
        per_block.append(
            _score_term(
                raw,
                mask=valid_mask,
                label=f"queue_tail_route_bce_block_{block_index}",
            ).scalar
        )
    return torch.stack(per_block).mean()


def _assert_frozen_target_metadata(batch: Any) -> None:
    metadata = packet_mapping(_lookup(batch, "target_metadata"))
    if metadata.get("fit_split") != "train_2022_2023_only":
        raise WorldLossError(
            "proper-score metadata must be fitted on 2022-2023 train only"
        )
    if bool(metadata.get("opened_final_test_2025", True)):
        raise WorldLossError(
            "proper-score metadata must not open the sealed 2025 test set"
        )


def _horizon_mapping(value: Any, *, label: str) -> dict[int, Any]:
    raw = packet_mapping(value)
    result: dict[int, Any] = {}
    for key, item in raw.items():
        text = str(key).lower().removeprefix("h")
        try:
            horizon = int(text)
        except ValueError as exc:
            raise WorldLossError(f"{label} has invalid horizon {key!r}") from exc
        result[horizon] = item
    return result


def _require_horizons(value: Any, *, label: str) -> dict[int, Any]:
    result = _horizon_mapping(value, label=label)
    missing = set(DYNAMICS_HORIZONS).difference(result)
    if missing:
        raise WorldLossError(
            f"{label} requires explicit h1/h4/h16/h96; missing {sorted(missing)}"
        )
    return result


def _group_mapping(value: Any, *, label: str) -> dict[str, Tensor]:
    raw = packet_mapping(value)
    if "public_21" in raw:
        public = _tensor(raw["public_21"], label=f"{label}.public_21")
        return _split_public(public, label=label)
    aliases = {
        "workload_volume": "workload",
        "arrival_volume": "workload",
    }
    result: dict[str, Tensor] = {}
    for raw_name, item in raw.items():
        name = aliases.get(str(raw_name).lower(), str(raw_name).lower())
        if name in PUBLIC_GROUP_SLICES:
            result[name] = _tensor(item, label=f"{label}.{raw_name}")
    missing = set(PUBLIC_GROUP_SLICES).difference(result)
    if missing:
        raise WorldLossError(f"{label} is missing target families {sorted(missing)}")
    return result


def _split_public(value: Tensor, *, label: str) -> dict[str, Tensor]:
    if value.shape[-1] != 21:
        raise WorldLossError(f"{label} public tensor must have 21 channels")
    return {
        name: value[..., channel_slice]
        for name, channel_slice in PUBLIC_GROUP_SLICES.items()
    }


def _family_transform(value: Tensor, family: str) -> Tensor:
    result = value.float()
    if family == "workload":
        if bool((result < 0).any()):
            raise WorldLossError("workload volume must be nonnegative before log1p")
        result = torch.log1p(result)
    return result


def _family_scale(scales: Mapping[str, Any], family: str, shape: torch.Size) -> Tensor:
    if family not in scales:
        raise WorldLossError(f"quantile_scales is missing {family}")
    scale = _tensor(scales[family], label=f"quantile_scales.{family}")
    expanded = _broadcast(scale, shape, label=f"quantile_scales.{family}")
    if not bool(torch.isfinite(expanded).all()) or bool((expanded <= 0).any()):
        raise WorldLossError(f"quantile_scales.{family} must be finite and positive")
    return expanded


def _channel_weights(
    family: str, channel_count: int, *, device: torch.device
) -> Tensor:
    channel_slice = PUBLIC_GROUP_SLICES.get(family)
    if channel_slice is None:
        raise WorldLossError(f"unknown public target family: {family!r}")
    expected = int(channel_slice.stop - channel_slice.start)
    if channel_count != expected:
        raise WorldLossError(
            f"{family} family requires {expected} channels, got {channel_count}"
        )
    return public_family_channel_weights(family, device=device)


def _assert_quantile_levels(output: Any) -> None:
    supplied = _lookup(output, "quantile_levels", required=False)
    if supplied is None:
        return
    levels = _tensor(supplied, label="quantile_levels").flatten()
    expected = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=levels.device)
    if levels.shape != expected.shape or not bool(
        torch.allclose(levels, expected, atol=1e-7, rtol=0.0)
    ):
        raise WorldLossError(f"quantile_levels must equal {list(QUANTILE_LEVELS)}")


def _assert_monotone_quantiles(prediction: Tensor, *, label: str) -> None:
    if prediction.shape[-1] != len(QUANTILE_LEVELS):
        raise WorldLossError(f"{label} must end in {len(QUANTILE_LEVELS)} quantiles")
    if bool((prediction[..., 1:] < prediction[..., :-1] - 1.0e-6).any()):
        raise WorldLossError(
            f"{label} has crossing quantiles; use monotone parameterization"
        )


def _quantile_integration_weights(*, device: torch.device) -> Tensor:
    """Voronoi quadrature weights on the fixed non-uniform quantile grid."""

    levels = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)
    boundaries = torch.cat(
        (
            torch.zeros(1, dtype=torch.float32, device=device),
            0.5 * (levels[:-1] + levels[1:]),
            torch.ones(1, dtype=torch.float32, device=device),
        )
    )
    return boundaries[1:] - boundaries[:-1]


def _assert_price_identity(
    value: Tensor, *, label: str, quantile_axis: bool = False
) -> None:
    channel_axis = -2 if quantile_axis else -1
    if value.shape[channel_axis] != 12:
        raise WorldLossError(f"{label} must contain 12 price channels")
    da, rt, dart = torch.split(value.float(), 4, dim=channel_axis)
    residual = da - rt - dart
    tolerance = 2.0e-5 + 2.0e-6 * value.float().abs().amax().detach()
    if float(residual.abs().amax().detach().cpu()) > float(tolerance.cpu()):
        raise WorldLossError(f"{label} violates DART = DA - RT")


def _assert_joint_price_paths(value: Any, *, label: str) -> None:
    """Check closure on coupled paths, not on independently sorted quantiles."""

    packet = packet_mapping(value)
    da = _tensor(_lookup(packet, "da"), label=f"{label}.da")
    rt = _tensor(_lookup(packet, "rt"), label=f"{label}.rt")
    dart = _tensor(_lookup(packet, "dart"), label=f"{label}.dart")
    if not (da.shape == rt.shape == dart.shape):
        raise WorldLossError(f"{label} DA/RT/DART path shapes differ")
    residual = da - rt - dart
    tolerance = (
        2.0e-5
        + 2.0e-6
        * torch.stack((da.abs().amax(), rt.abs().amax(), dart.abs().amax()))
        .amax()
        .detach()
    )
    if float(residual.abs().amax().detach().cpu()) > float(tolerance.cpu()):
        raise WorldLossError(f"{label} violates pathwise DART = DA - RT")


def quantile_pinball_term(
    prediction: Tensor,
    target: Tensor,
    *,
    scale: Tensor,
    mask: Tensor | None,
    family: str,
    label: str,
    allow_empty: bool = False,
    quantile_weights: Tensor | None = None,
) -> ScoreTerm:
    """Group/channel-balanced pinball score for the fixed R5-V2 quantiles."""

    prediction32, target32 = prediction.float(), target.float()
    _assert_monotone_quantiles(prediction32, label=label)
    if prediction32.shape[:-1] != target32.shape:
        raise WorldLossError(f"{label} prediction/target shape mismatch")
    if target32.ndim < 1:
        raise WorldLossError(f"{label} target must contain a channel axis")
    transformed_prediction = _family_transform(prediction32, family)
    transformed_target = _family_transform(target32, family)
    scale32 = _broadcast(scale.float(), target32.shape, label=f"{label} scale")
    if bool((scale32 <= 0).any()) or not bool(torch.isfinite(scale32).all()):
        raise WorldLossError(f"{label} scale must be finite and positive")
    error = (
        transformed_target.unsqueeze(-1) - transformed_prediction
    ) / scale32.unsqueeze(-1)
    quantiles = torch.tensor(
        QUANTILE_LEVELS, dtype=torch.float32, device=prediction32.device
    )
    raw = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    valid_mask = _mask(mask, target32.shape, label=f"{label} mask").unsqueeze(-1)
    channel = _channel_weights(
        family, int(target32.shape[-1]), device=prediction32.device
    )
    if quantile_weights is None:
        level_weights = torch.full_like(quantiles, 1.0 / len(QUANTILE_LEVELS))
    else:
        level_weights = quantile_weights.to(
            device=prediction32.device, dtype=torch.float32
        ).flatten()
        if level_weights.shape != quantiles.shape or bool(
            (~torch.isfinite(level_weights)).any()
        ):
            raise WorldLossError(f"{label} quantile weights are invalid")
        if bool((level_weights <= 0.0).any()):
            raise WorldLossError(f"{label} quantile weights must be positive")
        level_weights = level_weights / level_weights.sum()
    weights = channel.reshape((1,) * (raw.ndim - 2) + (channel.numel(), 1))
    weights = weights * level_weights
    return _score_term(
        raw,
        mask=valid_mask,
        element_weights=weights,
        label=label,
        allow_empty=allow_empty,
    )


def _quantile_family_terms(
    predictions: Any,
    targets: Any,
    masks: Any | None,
    scales: Any,
    *,
    prefix: str,
) -> dict[str, ScoreTerm]:
    predicted = _group_mapping(predictions, label=f"{prefix}.predictions")
    truth = _group_mapping(targets, label=f"{prefix}.targets")
    mask_mapping = {} if masks is None else packet_mapping(masks)
    scale_mapping = packet_mapping(scales)
    terms: dict[str, ScoreTerm] = {}
    for family in PUBLIC_GROUP_SLICES:
        prediction = predicted[family]
        target = truth[family]
        if family == "price":
            _assert_price_identity(target, label=f"{prefix}.price_target")
        scale = _family_scale(scale_mapping, family, target.shape)
        mask_value = mask_mapping.get(family)
        terms[family] = quantile_pinball_term(
            prediction,
            target,
            scale=scale,
            mask=None
            if mask_value is None
            else _tensor(mask_value, label=f"{prefix}.masks.{family}"),
            family=family,
            label=f"{prefix}/{family}",
        )
    return terms


def negative_binomial_nll_term(
    mean: Tensor,
    dispersion: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
) -> ScoreTerm:
    """NB2 likelihood for genuine nonnegative integer arrival counts."""

    mean32, dispersion32, target32 = mean.float(), dispersion.float(), target.float()
    if not (mean32.shape == dispersion32.shape == target32.shape):
        raise WorldLossError("count mean/dispersion/target shapes differ")
    valid_mask = _mask(mask, target32.shape, label="workload_count_mask")
    if not bool((valid_mask > 0).any()):
        raise WorldLossError("workload_count_mask has no true count labels")
    selected = valid_mask > 0
    if bool((target32[selected] < 0).any()) or not bool(
        torch.allclose(
            target32[selected], target32[selected].round(), atol=1e-5, rtol=0.0
        )
    ):
        raise WorldLossError("workload count targets must be true nonnegative integers")
    if bool((mean32[selected] <= 0).any()) or bool((dispersion32[selected] <= 0).any()):
        raise WorldLossError("negative-binomial mean and dispersion must be positive")
    mean32 = mean32.clamp_min(1.0e-8)
    dispersion32 = dispersion32.clamp_min(1.0e-8)
    log_total = torch.log(mean32 + dispersion32)
    nll = (
        torch.lgamma(dispersion32)
        + torch.lgamma(target32 + 1.0)
        - torch.lgamma(target32 + dispersion32)
        + dispersion32 * (log_total - torch.log(dispersion32))
        + target32 * (log_total - torch.log(mean32))
    )
    return _score_term(nll, mask=valid_mask, label="count_nll")


def _state_terms(
    predictions: Any,
    targets: Any,
    masks: Any | None,
    scales: Any,
    *,
    prefix: str,
) -> dict[str, ScoreTerm]:
    predicted = packet_mapping(predictions)
    truth = packet_mapping(targets)
    mask_mapping = {} if masks is None else packet_mapping(masks)
    scale_mapping = packet_mapping(scales)
    result: dict[str, ScoreTerm] = {}
    for state_name in ("soc", "queue"):
        if state_name not in predicted or state_name not in truth:
            raise WorldLossError(f"{prefix} requires both SOC and queue targets")
        prediction = _tensor(
            predicted[state_name], label=f"{prefix}.predictions.{state_name}"
        )
        target = _tensor(truth[state_name], label=f"{prefix}.targets.{state_name}")
        if prediction.shape != target.shape:
            raise WorldLossError(
                f"{prefix} {state_name} prediction/target shape mismatch"
            )
        if state_name == "soc":
            if bool(((target < 0) | (target > 1)).any()):
                raise WorldLossError("SOC targets must be normalized to [0,1]")
        else:
            if bool((prediction < 0).any()) or bool((target < 0).any()):
                raise WorldLossError("queue values must be nonnegative before log1p")
            prediction, target = torch.log1p(prediction), torch.log1p(target)
        if state_name not in scale_mapping:
            raise WorldLossError(f"{prefix} state_scales is missing {state_name}")
        scale = _broadcast(
            _tensor(scale_mapping[state_name], label=f"{prefix}.scales.{state_name}"),
            target.shape,
            label=f"{prefix}.scales.{state_name}",
        )
        if bool((scale <= 0).any()) or not bool(torch.isfinite(scale).all()):
            raise WorldLossError(
                f"{prefix} {state_name} scale must be finite and positive"
            )
        raw = F.smooth_l1_loss(
            prediction.float() / scale,
            target.float() / scale,
            reduction="none",
        )
        mask_value = mask_mapping.get(state_name)
        result[state_name] = _score_term(
            raw,
            mask=None
            if mask_value is None
            else _tensor(mask_value, label=f"{prefix}.masks.{state_name}"),
            label=f"{prefix}/{state_name}",
        )
    return result


def _w0_queue_terms(
    predictions: Any,
    targets: Any,
    masks: Any | None,
    scales: Any,
    *,
    distribution: Any | None = None,
    occurrence_target: Any | None = None,
    occurrence_mask: Any | None = None,
    severity_target: Any | None = None,
    severity_mask: Any | None = None,
) -> dict[str, ScoreTerm]:
    """Score sparse queues as occurrence plus positive conditional severity.

    A queue of zero is a structural operating regime, not a tiny severity.  The
    Bernoulli occurrence score therefore uses every valid target, while the
    log-severity score uses only genuine positive queue observations.  An
    explicit neural distribution head is preferred; the legacy continuous
    queue reconstruction remains a differentiable compatibility fallback.
    """

    predicted = packet_mapping(predictions)
    truth = packet_mapping(targets)
    mask_mapping = {} if masks is None else packet_mapping(masks)
    scale_mapping = packet_mapping(scales)
    if "queue" not in predicted or "queue" not in truth:
        raise WorldLossError("W0 queue scoring requires queue predictions and targets")
    fallback_mean = _tensor(predicted["queue"], label="w0_queue.predictions.queue")
    target = _tensor(
        truth["queue"] if severity_target is None else severity_target,
        label="w0_queue.targets.queue",
    )
    if fallback_mean.shape != target.shape:
        raise WorldLossError("W0 queue prediction/target shape mismatch")
    if bool((fallback_mean < 0).any()) or bool((target < 0).any()):
        raise WorldLossError("W0 queue values must be nonnegative")

    raw_mask = mask_mapping.get("queue") if occurrence_mask is None else occurrence_mask
    valid_mask = (
        torch.ones_like(target, dtype=torch.bool)
        if raw_mask is None
        else _broadcast(
            _tensor(raw_mask, label="w0_queue.occurrence_mask"),
            target.shape,
            label="w0_queue.occurrence_mask",
        )
        > 0
    )
    derived_occurrence = target > QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS
    if occurrence_target is None:
        occurrence_truth = derived_occurrence.float()
    else:
        occurrence_truth = _broadcast(
            _tensor(occurrence_target, label="w0_queue.occurrence_target"),
            target.shape,
            label="w0_queue.occurrence_target",
        )
        if bool(((occurrence_truth > 0) != derived_occurrence)[valid_mask].any()):
            raise WorldLossError(
                "queue occurrence target must equal the registered q>1e-4 mask"
            )
        occurrence_truth = (occurrence_truth > 0).float()

    scale_key = "queue_severity" if "queue_severity" in scale_mapping else "queue"
    if scale_key not in scale_mapping:
        raise WorldLossError("W0 state_scales is missing queue severity")
    scale = _broadcast(
        _tensor(scale_mapping[scale_key], label="w0_queue.severity_scale"),
        target.shape,
        label="w0_queue.severity_scale",
    )
    if bool((scale < 1.0e-3).any()) or not bool(torch.isfinite(scale).all()):
        raise WorldLossError(
            "W0 queue severity scale must be finite and nondegenerate (>=1e-3)"
        )

    distribution_mapping = {} if distribution is None else packet_mapping(distribution)
    if distribution_mapping:
        logits = _tensor(
            _lookup(distribution_mapping, "occurrence_logits", "logits"),
            label="queue_distribution.occurrence_logits",
        )
        positive_severity = _tensor(
            _lookup(
                distribution_mapping,
                "positive_severity",
                "severity",
                "conditional_severity",
            ),
            label="queue_distribution.positive_severity",
        )
        if logits.shape != target.shape or positive_severity.shape != target.shape:
            raise WorldLossError("queue distribution tensors must match queue targets")
        if not bool(torch.isfinite(logits).all()) or not bool(
            torch.isfinite(positive_severity).all()
        ):
            raise WorldLossError("queue distribution contains non-finite values")
        if bool((positive_severity < 0).any()):
            raise WorldLossError("positive queue severity must be nonnegative")
        occurrence_probability = torch.sigmoid(logits.float())
        severity_prediction = positive_severity.float()
    else:
        severity_prediction = fallback_mean.float()
        normalized_positive = torch.log1p(severity_prediction) / scale.float()
        occurrence_probability = -torch.expm1(-normalized_positive)

    occurrence_values = (
        occurrence_probability.float() - occurrence_truth.float()
    ).square()
    occurrence_term = _score_term(
        occurrence_values,
        mask=valid_mask,
        label="w0_queue_occurrence_brier",
    )
    occurrence_zero_baseline = _score_term(
        occurrence_truth.float().square(),
        mask=valid_mask,
        label="w0_queue_occurrence_zero_baseline",
    )

    raw_severity_mask = severity_mask
    if raw_severity_mask is None:
        positive_mask = valid_mask & derived_occurrence
    else:
        positive_mask = (
            _broadcast(
                _tensor(raw_severity_mask, label="w0_queue.severity_mask"),
                target.shape,
                label="w0_queue.severity_mask",
            )
            > 0
        )
        if bool((positive_mask & ~(valid_mask & derived_occurrence)).any()):
            raise WorldLossError(
                "queue severity mask may select positive valid targets only"
            )
    severity_values = F.smooth_l1_loss(
        torch.log1p(severity_prediction) / scale.float(),
        torch.log1p(target.float()) / scale.float(),
        reduction="none",
    )
    severity_term = _score_term(
        severity_values,
        mask=positive_mask,
        label="w0_queue_positive_severity",
        allow_empty=True,
    )
    severity_zero_baseline = _score_term(
        F.smooth_l1_loss(
            torch.zeros_like(target.float()),
            torch.log1p(target.float()) / scale.float(),
            reduction="none",
        ),
        mask=positive_mask,
        label="w0_queue_severity_zero_baseline",
        allow_empty=True,
    )
    # Compatibility only: older selectors expect state/queue.  Training and new
    # gates consume the two atomic terms above, whose counts remain independent.
    compatibility_term = ScoreTerm(
        (occurrence_term.numerator + severity_term.numerator).float(),
        (occurrence_term.valid_count + severity_term.valid_count).float(),
    )
    return {
        "queue_occurrence": occurrence_term,
        "queue_severity": severity_term,
        "queue_occurrence_zero_baseline": occurrence_zero_baseline,
        "queue_severity_zero_baseline": severity_zero_baseline,
        "queue": compatibility_term,
    }


def _w0_soc_persistence_baseline(
    batch: Any,
    state_targets: Any,
    state_masks: Any | None,
    state_scales: Any,
) -> ScoreTerm:
    """Causal issue-time SOC persistence baseline for the W0 capability gate."""

    targets = packet_mapping(state_targets)
    masks = {} if state_masks is None else packet_mapping(state_masks)
    scales = packet_mapping(state_scales)
    target = _tensor(targets["soc"], label="w0_soc.targets.soc")
    if "soc" not in scales:
        raise WorldLossError("W0 state_scales is missing soc")

    initial: Any | None = None
    world_input = _lookup(
        batch, "world_input", "input_packet", "packet", required=False
    )
    if world_input is not None:
        constraints = _lookup(world_input, "constraints", required=False)
        if constraints is not None:
            initial = _lookup(constraints, "initial_soc", required=False)
    if initial is None:
        centers = _lookup(batch, "state_centers", required=False)
        if centers is not None:
            initial = packet_mapping(centers).get("soc")
    if initial is None:
        # Compatibility for small synthetic tests.  Formal batches always carry
        # the issue-time constraint packet above.
        initial_tensor = target.new_zeros((target.shape[0], target.shape[-1]))
    else:
        initial_tensor = _tensor(initial, label="w0_soc.initial_soc")
    while initial_tensor.ndim < target.ndim:
        initial_tensor = initial_tensor.unsqueeze(-2)
    baseline = _broadcast(
        initial_tensor, target.shape, label="w0_soc.persistence_baseline"
    )
    scale = _broadcast(
        _tensor(scales["soc"], label="w0_soc.scales.soc"),
        target.shape,
        label="w0_soc.scales.soc",
    )
    values = F.smooth_l1_loss(
        baseline.float() / scale,
        target.float() / scale,
        reduction="none",
    )
    mask = masks.get("soc")
    return _score_term(
        values,
        mask=None if mask is None else _tensor(mask, label="w0_soc.masks.soc"),
        label="w0_soc_persistence_baseline",
    )


def _w0_training_baseline_scalar(
    batch: Any,
    name: str,
    *,
    fallback: ScoreTerm,
    reference: Tensor,
) -> Tensor:
    """Read a train-only frozen proper-score denominator.

    Formal R5-V2.1 batches carry one scalar per state family, computed once over
    all 2022-2023 training observations.  The detached atomic fallback preserves
    compatibility with small legacy unit packets; it must not be used by the
    formal corpus because minibatch-dependent normalization would reweight the
    scientific objective.
    """

    raw = _lookup(
        batch,
        "state_training_baselines",
        "w0_state_training_baselines",
        required=False,
    )
    aliases = {
        "soc": ("soc", "soc_persistence", "soc_persistence_baseline"),
        "queue_occurrence": (
            "queue_occurrence",
            "queue_occurrence_zero",
            "queue_occurrence_zero_baseline",
        ),
        "queue_severity": (
            "queue_severity",
            "queue_severity_zero",
            "queue_severity_zero_baseline",
        ),
    }
    value: Any | None = None
    if raw is not None:
        mapping = packet_mapping(raw)
        for key in aliases[name]:
            if key in mapping and mapping[key] is not None:
                value = mapping[key]
                break
        if value is None:
            raise WorldLossError(f"state_training_baselines is missing frozen {name}")
    if value is None:
        scalar = fallback.scalar.detach().to(device=reference.device).float()
    elif isinstance(value, Tensor):
        scalar = value.detach().to(device=reference.device, dtype=torch.float32)
    else:
        scalar = torch.as_tensor(value, device=reference.device, dtype=torch.float32)
    if scalar.numel() != 1:
        raise WorldLossError(f"frozen {name} baseline must be one scalar")
    scalar = scalar.reshape(())
    if name == "soc" and float(scalar.detach().cpu()) <= 1.0e-5:
        # The registered train trace has a persistence-perfect SOC teacher, so
        # its baseline is mathematically zero.  A score ratio is undefined in
        # that degenerate case; retain the original normalized Smooth-L1 scale
        # instead of amplifying it by the serialization safety floor.
        return scalar.new_tensor(1.0)
    if not bool(torch.isfinite(scalar)) or float(scalar.cpu()) <= 0.0:
        # Synthetic all-zero batches have a zero empirical occurrence baseline.
        # A small compatibility floor keeps their gradient finite.  Formal
        # train-only baselines are required to be strictly positive above.
        if raw is not None:
            raise WorldLossError(
                f"frozen {name} baseline must be finite and strictly positive"
            )
        scalar = scalar.new_tensor(1.0e-4)
    return scalar


def w0_masked_world_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
) -> PhaseLoss:
    """W0 group-balanced quantile, genuine-count, and state objective."""

    _assert_frozen_target_metadata(batch)
    _assert_quantile_levels(output)
    predictions = _lookup(output, "quantile_predictions", "quantiles")
    _assert_joint_price_paths(
        _lookup(output, "joint_price_paths"), label="w0_joint_price_paths"
    )
    targets = _lookup(batch, "quantile_targets")
    masks = _lookup(batch, "quantile_masks", required=False)
    scales = _lookup(batch, "quantile_scales", "target_scales")
    quantile_terms = _quantile_family_terms(
        predictions, targets, masks, scales, prefix="w0_quantile"
    )
    family_weights = {name: 0.25 for name in PUBLIC_GROUP_SLICES}
    quantile = _weighted_scalar(quantile_terms, family_weights, label="W0 quantile")

    state_predictions = _lookup(
        output, "state_predictions", "state_reconstructions", "reconstructions"
    )
    state_targets = _lookup(
        batch, "state_targets", "state_reconstruction_targets", "reconstruction_targets"
    )
    state_masks = _lookup(
        batch,
        "state_masks",
        "state_reconstruction_masks",
        "reconstruction_masks",
        required=False,
    )
    state_scales = _lookup(batch, "state_scales", "state_target_scales")
    legacy_state_terms = _state_terms(
        state_predictions,
        state_targets,
        state_masks,
        state_scales,
        prefix="w0_state",
    )
    queue_terms = _w0_queue_terms(
        state_predictions,
        state_targets,
        state_masks,
        state_scales,
        distribution=_lookup(output, "queue_distribution", required=False),
        occurrence_target=_lookup(batch, "queue_occurrence_target", required=False),
        occurrence_mask=_lookup(batch, "queue_occurrence_mask", required=False),
        severity_target=_lookup(batch, "queue_severity_target", required=False),
        severity_mask=_lookup(batch, "queue_severity_mask", required=False),
    )
    soc_persistence_baseline = _w0_soc_persistence_baseline(
        batch, state_targets, state_masks, state_scales
    )
    soc_baseline = _w0_training_baseline_scalar(
        batch,
        "soc",
        fallback=soc_persistence_baseline,
        reference=legacy_state_terms["soc"].numerator,
    )
    occurrence_baseline = _w0_training_baseline_scalar(
        batch,
        "queue_occurrence",
        fallback=queue_terms["queue_occurrence_zero_baseline"],
        reference=queue_terms["queue_occurrence"].numerator,
    )
    severity_baseline = _w0_training_baseline_scalar(
        batch,
        "queue_severity",
        fallback=queue_terms["queue_severity_zero_baseline"],
        reference=queue_terms["queue_severity"].numerator,
    )
    normalized_soc = legacy_state_terms["soc"].scalar / soc_baseline
    normalized_occurrence = queue_terms["queue_occurrence"].scalar / occurrence_baseline
    normalized_severity = queue_terms["queue_severity"].scalar / severity_baseline
    # Fixed thirds keep the training objective invariant to whether a sparse
    # minibatch happens to contain a positive queue severity target.
    state = (normalized_soc + normalized_occurrence + normalized_severity) / 3.0
    state_terms = {
        "soc": legacy_state_terms["soc"],
        "soc_persistence_baseline": soc_persistence_baseline,
        **queue_terms,
    }

    count: Tensor | None = None
    count_term: ScoreTerm | None = None
    count_target = _lookup(
        batch, "workload_count_target", "arrival_count_target", required=False
    )
    count_mask = _lookup(
        batch, "workload_count_mask", "arrival_count_mask", required=False
    )
    if count_target is not None:
        if count_mask is None:
            raise WorldLossError(
                "true workload count targets require an explicit count mask"
            )
        mask_tensor = _tensor(count_mask, label="workload_count_mask")
        if bool((mask_tensor > 0).any()):
            distribution = packet_mapping(
                _lookup(output, "count_distribution", "workload_count_distribution")
            )
            mean = _lookup(distribution, "mean", "mu")
            dispersion = _lookup(distribution, "dispersion", "total_count", "r")
            count_term = negative_binomial_nll_term(
                _tensor(mean, label="count_distribution.mean"),
                _tensor(dispersion, label="count_distribution.dispersion"),
                _tensor(count_target, label="workload_count_target"),
                mask=mask_tensor,
            )
            count = count_term.scalar
    auxiliary = _auxiliary_loss(output, next(iter(quantile_terms.values())).numerator)
    route_supervision = _queue_tail_route_supervision_loss(
        output,
        batch,
        next(iter(quantile_terms.values())).numerator,
    )
    general_objective = (
        weights.quantile * quantile
        + weights.state * normalized_soc / 3.0
        + weights.auxiliary * auxiliary
    )
    if count is not None:
        general_objective = general_objective + weights.count * count
    rare_objective = (
        weights.state * (normalized_occurrence + normalized_severity) / 3.0
        + weights.route_supervision * route_supervision
    )
    total = general_objective + rare_objective

    components: dict[str, Tensor] = {}
    score_terms: dict[str, ScoreTerm] = {}
    for name, term in quantile_terms.items():
        _register_term(
            f"quantile/{name}", term, components=components, score_terms=score_terms
        )
    for name, term in state_terms.items():
        _register_term(
            f"state/{name}", term, components=components, score_terms=score_terms
        )
    if count_term is not None:
        _register_term(
            "count/nll", count_term, components=components, score_terms=score_terms
        )
    components.update(
        quantile=quantile,
        state=state,
        normalized_soc=normalized_soc,
        normalized_queue_occurrence=normalized_occurrence,
        normalized_queue_severity=normalized_severity,
        general_objective=general_objective,
        rare_objective=rare_objective,
        auxiliary=auxiliary,
        route_supervision=route_supervision,
        total=_finite("W0 total", total),
    )
    # W0 intentionally has no KL term.  Prior/posterior learning starts in W1.
    return PhaseLoss("w0", total, components, score_terms)


def _gaussian_parameters(value: Any, *, label: str) -> tuple[Tensor, Tensor]:
    packet = packet_mapping(value)
    mean = _tensor(_lookup(packet, "mean", "loc"), label=f"{label}.mean")
    logvar_value = _lookup(packet, "logvar", "log_variance", required=False)
    if logvar_value is None:
        log_scale = _tensor(_lookup(packet, "log_scale"), label=f"{label}.log_scale")
        logvar = 2.0 * log_scale
    else:
        logvar = _tensor(logvar_value, label=f"{label}.logvar")
    if mean.shape != logvar.shape:
        raise WorldLossError(f"{label} mean and log-variance shapes differ")
    return mean.float(), logvar.float().clamp(-30.0, 20.0)


def _diagonal_gaussian_kl(
    q_mean: Tensor, q_logvar: Tensor, p_mean: Tensor, p_logvar: Tensor
) -> Tensor:
    if not (q_mean.shape == q_logvar.shape == p_mean.shape == p_logvar.shape):
        raise WorldLossError("prior/posterior Gaussian parameter shapes differ")
    return 0.5 * (
        p_logvar
        - q_logvar
        + (q_logvar - p_logvar).exp()
        + (q_mean - p_mean).square() * (-p_logvar).exp()
        - 1.0
    )


def balanced_kl_term(
    prior: Any,
    posterior: Any,
    *,
    free_nats: float = 1.0,
    dynamics_weight: float = 0.8,
) -> ScoreTerm:
    """Dreamer-style balanced KL with stop-gradient and one-nat free bits."""

    floored, _, _ = balanced_kl_diagnostic_terms(
        prior,
        posterior,
        free_nats=free_nats,
        dynamics_weight=dynamics_weight,
    )
    return floored


def balanced_kl_diagnostic_terms(
    prior: Any,
    posterior: Any,
    *,
    free_nats: float = 1.0,
    dynamics_weight: float = 0.8,
) -> tuple[ScoreTerm, ScoreTerm, ScoreTerm]:
    """Return floored KL, raw KL and the free-bit floor fraction.

    The raw term is retained as a small optimization path in W1.  Consequently
    a batch whose KL lies entirely below one nat no longer turns the complete
    stochastic prior/posterior objective into a gradient-free constant.  The
    third exact term is the fraction of latent positions at the floor and is a
    validation diagnostic, not an optimization target.
    """

    if not 0.0 <= dynamics_weight <= 1.0:
        raise WorldLossError("balanced KL weight must lie in [0,1]")
    if float(free_nats) < 0.0:
        raise WorldLossError("free_nats cannot be negative")
    q_mean, q_logvar = _gaussian_parameters(posterior, label="posterior")
    p_mean, p_logvar = _gaussian_parameters(prior, label="prior")
    # Sum only the latent feature axis.  Each time/horizon state receives its
    # own one-nat free bit and contributes one valid item, so h1 is not scaled
    # up merely because it contains more target positions than h96.
    dynamics_kl = _diagonal_gaussian_kl(
        q_mean.detach(), q_logvar.detach(), p_mean, p_logvar
    ).sum(dim=-1)
    representation_kl = _diagonal_gaussian_kl(
        q_mean, q_logvar, p_mean.detach(), p_logvar.detach()
    ).sum(dim=-1)
    balanced = (
        dynamics_weight * dynamics_kl + (1.0 - dynamics_weight) * representation_kl
    )
    floored = balanced.clamp_min(float(free_nats))
    floor_indicator = (balanced.detach() <= float(free_nats) + 1.0e-6).float()
    return (
        _score_term(floored, label="balanced_kl"),
        _score_term(balanced, label="raw_balanced_kl"),
        _score_term(floor_indicator, label="kl_free_bit_floor_fraction"),
    )


def _public_target_groups(value: Any, *, label: str) -> dict[str, Tensor]:
    if isinstance(value, Tensor):
        return _split_public(value.float(), label=label)
    return _group_mapping(value, label=label)


def _rollout_quantile_terms(
    predictions: Any,
    targets: Any,
    masks: Any | None,
    scales: Any,
    *,
    horizon: int,
) -> dict[str, ScoreTerm]:
    if isinstance(predictions, Tensor):
        if predictions.shape[-2] != 21:
            raise WorldLossError(f"W1 h{horizon} quantile tensor must be [...,21,Q]")
        prediction_groups = {
            name: predictions[..., channel_slice, :]
            for name, channel_slice in PUBLIC_GROUP_SLICES.items()
        }
    else:
        prediction_groups = _group_mapping(
            predictions, label=f"w1_h{horizon}.predictions"
        )
    target_groups = _public_target_groups(targets, label=f"w1_h{horizon}.targets")
    if masks is None:
        mask_groups: Mapping[str, Any] = {}
    elif isinstance(masks, Tensor):
        mask_groups = _split_public(masks.float(), label=f"w1_h{horizon}.masks")
    else:
        mask_groups = packet_mapping(masks)
    scale_mapping = packet_mapping(scales)
    terms: dict[str, ScoreTerm] = {}
    for family in PUBLIC_GROUP_SLICES:
        prediction = prediction_groups[family]
        target = target_groups[family]
        if family == "price":
            _assert_price_identity(target, label=f"w1_h{horizon}.price_target")
        terms[family] = quantile_pinball_term(
            prediction,
            target,
            scale=_family_scale(scale_mapping, family, target.shape),
            mask=None
            if family not in mask_groups
            else _tensor(mask_groups[family], label=f"w1_h{horizon}.masks.{family}"),
            family=family,
            label=f"rollout_h{horizon}/{family}",
        )
    return terms


def w1_action_dynamics_loss(
    output: Any,
    batch: Any,
    *,
    horizons: Sequence[int] = DYNAMICS_HORIZONS,
    weights: WorldLossWeights = WorldLossWeights(),
    include_route_supervision: bool = True,
) -> PhaseLoss:
    """W1 balanced latent KL plus proper 1/4/16/96 observation rollout."""

    _assert_frozen_target_metadata(batch)
    requested = tuple(int(value) for value in horizons)
    if requested != DYNAMICS_HORIZONS:
        raise WorldLossError(f"W1 horizons must be {list(DYNAMICS_HORIZONS)}")
    priors = _require_horizons(
        _lookup(output, "latent_prior", "latent_priors"), label="latent_prior"
    )
    posteriors = _require_horizons(
        _lookup(output, "latent_posterior", "latent_posteriors"),
        label="latent_posterior",
    )
    observation_predictions = _require_horizons(
        _lookup(output, "observation_quantile_predictions", "rollout_quantiles"),
        label="observation_quantile_predictions",
    )
    observation_targets = _require_horizons(
        _lookup(batch, "observation_targets", "future_target_sequence"),
        label="observation_targets",
    )
    joint_price_paths = _require_horizons(
        _lookup(output, "observation_joint_price_paths"),
        label="observation_joint_price_paths",
    )
    for horizon in requested:
        _assert_joint_price_paths(
            joint_price_paths[horizon],
            label=f"observation_joint_price_paths.h{horizon}",
        )
    observation_masks = _lookup(batch, "observation_masks", required=False)
    observation_mask_horizons = (
        {}
        if observation_masks is None
        else _require_horizons(observation_masks, label="observation_masks")
    )
    scales = _lookup(batch, "quantile_scales", "target_scales")
    latent_targets_value = _lookup(output, "latent_targets", required=False)
    latent_target_horizons = (
        {}
        if latent_targets_value is None
        else _require_horizons(latent_targets_value, label="latent_targets")
    )

    components: dict[str, Tensor] = {}
    score_terms: dict[str, ScoreTerm] = {}
    latent_scalars: list[Tensor] = []
    rollout_scalars: list[Tensor] = []
    for horizon in requested:
        latent_term, raw_latent_term, floor_fraction_term = (
            balanced_kl_diagnostic_terms(priors[horizon], posteriors[horizon])
        )
        _register_term(
            f"latent/h{horizon}",
            latent_term,
            components=components,
            score_terms=score_terms,
        )
        _register_term(
            f"latent/raw_h{horizon}",
            raw_latent_term,
            components=components,
            score_terms=score_terms,
        )
        _register_term(
            f"latent/free_bit_floor_fraction_h{horizon}",
            floor_fraction_term,
            components=components,
            score_terms=score_terms,
        )
        anchor_scalar = latent_term.scalar.new_zeros(())
        if horizon in latent_target_horizons:
            posterior_mean, _ = _gaussian_parameters(
                posteriors[horizon], label=f"posterior.h{horizon}"
            )
            target_latent = _tensor(
                latent_target_horizons[horizon], label=f"latent_targets.h{horizon}"
            ).detach()
            if posterior_mean.shape != target_latent.shape:
                raise WorldLossError(
                    f"W1 h{horizon} posterior anchor shape differs from target latent"
                )
            anchor_term = _score_term(
                F.smooth_l1_loss(
                    posterior_mean.float(), target_latent.float(), reduction="none"
                ).mean(dim=-1),
                label=f"posterior_anchor_h{horizon}",
            )
            _register_term(
                f"latent/posterior_anchor_h{horizon}",
                anchor_term,
                components=components,
                score_terms=score_terms,
            )
            anchor_scalar = anchor_term.scalar
        # The proper free-bit term remains the primary latent objective.  A
        # small raw-KL path prevents a full-floor batch from having zero
        # stochastic gradient; the W0-target anchor keeps q meaningful.
        latent_scalars.append(
            latent_term.scalar + 0.05 * raw_latent_term.scalar + 0.10 * anchor_scalar
        )
        family_terms = _rollout_quantile_terms(
            observation_predictions[horizon],
            observation_targets[horizon],
            observation_mask_horizons.get(horizon),
            scales,
            horizon=horizon,
        )
        for family, term in family_terms.items():
            _register_term(
                f"rollout/h{horizon}/{family}",
                term,
                components=components,
                score_terms=score_terms,
            )
        rollout_scalars.append(
            _weighted_scalar(
                family_terms,
                {name: 0.25 for name in PUBLIC_GROUP_SLICES},
                label=f"W1 rollout h{horizon}",
            )
        )
    latent = torch.stack(latent_scalars).mean()
    rollout = torch.stack(rollout_scalars).mean()

    state_predictions = _lookup(output, "state_rollout_predictions")
    state_targets = _lookup(batch, "state_rollout_targets", "future_state_targets")
    state_masks = _lookup(
        batch, "state_rollout_masks", "future_state_masks", required=False
    )
    state_scales = _lookup(batch, "state_scales", "state_target_scales")
    predicted_state_horizons = _require_horizons(
        state_predictions, label="state_rollout_predictions"
    )
    target_state_horizons = _require_horizons(
        state_targets, label="state_rollout_targets"
    )
    state_mask_horizons = (
        {}
        if state_masks is None
        else _require_horizons(state_masks, label="state_rollout_masks")
    )
    state_scalars: list[Tensor] = []
    for horizon in requested:
        horizon_terms = _state_terms(
            predicted_state_horizons[horizon],
            target_state_horizons[horizon],
            state_mask_horizons.get(horizon),
            state_scales,
            prefix=f"w1_state_h{horizon}",
        )
        _register_term(
            f"state/h{horizon}/soc",
            horizon_terms["soc"],
            components=components,
            score_terms=score_terms,
        )
        predicted_state = packet_mapping(predicted_state_horizons[horizon])
        queue_terms = _w0_queue_terms(
            predicted_state,
            target_state_horizons[horizon],
            state_mask_horizons.get(horizon),
            state_scales,
            distribution=predicted_state.get("queue_distribution"),
        )
        for name in ("queue_occurrence", "queue_severity", "queue"):
            _register_term(
                f"state/h{horizon}/{name}",
                queue_terms[name],
                components=components,
                score_terms=score_terms,
            )
        queue_scalar = 0.5 * (
            queue_terms["queue_occurrence"].scalar
            + queue_terms["queue_severity"].scalar
        )
        state_scalars.append(0.5 * horizon_terms["soc"].scalar + 0.5 * queue_scalar)
    state = torch.stack(state_scalars).mean()
    auxiliary = _auxiliary_loss(output, latent)
    route_supervision = (
        _queue_tail_route_supervision_loss(output, batch, latent)
        if include_route_supervision
        else _zero(latent)
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
        total=_finite("W1 total", total),
    )
    return PhaseLoss("w1", total, components, score_terms)


def _scenario_and_target(output: Any, batch: Any) -> tuple[Tensor, Tensor]:
    scenarios = _tensor(
        _lookup(output, "scenarios", "stochastic_rollouts", "samples"),
        label="scenarios",
    )
    target = _tensor(
        _lookup(batch, "stochastic_target", "future_target", "scenario_target"),
        label="stochastic_target",
    )
    if scenarios.ndim != 4 or target.ndim != 3:
        raise WorldLossError("U0 requires scenarios [B,S,T,21] and target [B,T,21]")
    if (
        scenarios.shape[0] != target.shape[0]
        or scenarios.shape[2:] != target.shape[1:]
    ):
        raise WorldLossError("scenario and stochastic target shapes are incompatible")
    if scenarios.shape[1] < 2 or scenarios.shape[-1] != 21:
        raise WorldLossError("U0 requires at least two 21-channel stochastic paths")
    _assert_price_identity(scenarios[..., :12], label="scenario prices")
    _assert_price_identity(target[..., :12], label="stochastic target prices")
    return scenarios.float(), target.float()


def _fair_ensemble_crps_values(scenarios: Tensor, target: Tensor) -> Tensor:
    """Return per-element fair CRPS without changing its finite-M definition.

    Gradient-enabled training retains the original pairwise absolute-distance
    expression.  Besides being efficient for the registered M=20 ensemble, it
    preserves PyTorch's zero subgradient for exact ties.  Formal M=100 scoring
    runs under ``no_grad`` and uses the algebraically identical sorted-sample
    identity, avoiding a quadratic validation tensor without introducing a
    Gram-matrix approximation.
    """

    samples = scenarios.float()
    truth = target.float()
    sample_count = int(samples.shape[1])
    accuracy = (samples - truth.unsqueeze(1)).abs().mean(dim=1)
    if sample_count == 100 and not torch.is_grad_enabled():
        ordered = samples.sort(dim=1).values
        rank_weight = (
            2.0
            * torch.arange(
                sample_count,
                device=ordered.device,
                dtype=torch.float32,
            )
            - float(sample_count)
            + 1.0
        )
        rank_shape = (1, sample_count) + (1,) * (samples.ndim - 2)
        half_spread = (
            ordered * rank_weight.reshape(rank_shape)
        ).sum(dim=1) / float(sample_count * (sample_count - 1))
        return accuracy - half_spread

    spread = (
        (samples.unsqueeze(2) - samples.unsqueeze(1))
        .abs()
        .mean(dim=(1, 2))
        * (float(sample_count) / float(sample_count - 1))
    )
    return accuracy - 0.5 * spread


def ensemble_crps_term(
    scenarios: Tensor,
    target: Tensor,
    *,
    label: str,
    element_weights: Tensor | None = None,
) -> ScoreTerm:
    """Marginal CRPS with the fair finite-ensemble estimator.

    The stochastic paths are Monte Carlo draws from a predictive population,
    rather than a fixed deterministic ensemble.  The diagonal pairs are
    therefore excluded from the spread expectation.  Including their exact
    zeros would multiply the population spread by ``(M - 1) / M`` and create a
    systematic contraction incentive when ``M`` is finite.
    """

    if scenarios.ndim != target.ndim + 1 or scenarios.ndim < 2:
        raise WorldLossError(f"{label} scenarios must add exactly one path dimension")
    if (
        scenarios.shape[0] != target.shape[0]
        or scenarios.shape[2:] != target.shape[1:]
    ):
        raise WorldLossError(f"{label} scenario/target shapes differ")
    sample_count = int(scenarios.shape[1])
    if sample_count < 2:
        raise WorldLossError(f"{label} requires at least two stochastic paths")
    return _score_term(
        _fair_ensemble_crps_values(scenarios, target),
        element_weights=element_weights,
        label=label,
    )


def energy_score_term(scenarios: Tensor, target: Tensor) -> ScoreTerm:
    """Joint Energy Score with a fair off-diagonal spread estimator."""

    if scenarios.ndim != target.ndim + 1 or scenarios.ndim < 2:
        raise WorldLossError("energy scenarios must add exactly one path dimension")
    if (
        scenarios.shape[0] != target.shape[0]
        or scenarios.shape[2:] != target.shape[1:]
    ):
        raise WorldLossError("energy scenario/target shapes differ")
    sample_count = int(scenarios.shape[1])
    if sample_count < 2:
        raise WorldLossError("energy requires at least two stochastic paths")
    samples = scenarios.float().flatten(start_dim=2)
    truth = target.float().flatten(start_dim=1)
    accuracy = torch.linalg.vector_norm(samples - truth.unsqueeze(1), dim=-1).mean(
        dim=1
    )
    if sample_count == 100 and not torch.is_grad_enabled():
        # The direct cdist kernel evaluates the same Euclidean distances while
        # avoiding [B,100,100,D] materialization.  Explicitly forbid its faster
        # Gram/matmul mode: squared-norm cancellation would alter the registered
        # scientific score for nearby paths.
        pair_distance = torch.cdist(
            samples,
            samples,
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        spread = pair_distance.mean(dim=(1, 2)) * (
            float(sample_count) / float(sample_count - 1)
        )
    else:
        # M=20 training intentionally retains the original expression and its
        # gradient.  At this small ensemble size direct cdist is slower.
        spread = torch.linalg.vector_norm(
            samples.unsqueeze(2) - samples.unsqueeze(1), dim=-1
        ).mean(dim=(1, 2)) * (float(sample_count) / float(sample_count - 1))
    score = (accuracy - 0.5 * spread) / math.sqrt(max(int(samples.shape[-1]), 1))
    return _score_term(score, label="energy")


def _pair_variogram(
    scenarios_a: Tensor,
    scenarios_b: Tensor,
    target_a: Tensor,
    target_b: Tensor,
) -> Tensor:
    # The common epsilon cancels when forecast and observation differences
    # match, while avoiding the singular derivative of sqrt at exactly zero.
    epsilon = 1.0e-6
    observed = ((target_a - target_b).abs() + epsilon).sqrt()
    generated = ((scenarios_a - scenarios_b).abs() + epsilon).sqrt().mean(dim=1)
    return (generated - observed).square()


def variogram_score_terms(scenarios: Tensor, target: Tensor) -> dict[str, ScoreTerm]:
    """Variogram components on semantic axes, never on flattened channels."""

    if scenarios.ndim != 4 or target.ndim != 3 or scenarios.shape[-1] != 21:
        raise WorldLossError("variogram expects [B,S,T,21] and [B,T,21]")
    terms: dict[str, ScoreTerm] = {}
    samples = scenarios.float()
    truth = target.float()
    groups_s = _split_public(samples, label="variogram scenarios")
    groups_y = _split_public(truth, label="variogram target")

    time_values: list[Tensor] = []
    for lag in (1, 4, 16):
        if scenarios.shape[2] <= lag:
            raise WorldLossError(f"variogram time axis is too short for lag {lag}")
        # One 21-channel path calculation replaces four independent family
        # kernels.  Family means remain separate below, preserving equal family
        # rather than channel-count weighting.
        value = _pair_variogram(
            samples[:, :, lag:, :],
            samples[:, :, :-lag, :],
            truth[:, lag:, :],
            truth[:, :-lag, :],
        )
        family_values = [
            value[..., channel_slice].flatten(start_dim=1).mean(dim=1)
            for channel_slice in PUBLIC_GROUP_SLICES.values()
        ]
        time_values.append(torch.stack(family_values).mean(dim=0))
    terms["time"] = _score_term(
        torch.stack(time_values).mean(dim=0), label="variogram/time"
    )

    # DA, RT, DART, PV and PUE each contribute all six unordered four-node
    # pairs.  Gather the registered 30 pairs into one path calculation.
    cross_node_pairs = tuple(
        (offset + left, offset + right)
        for offset in (0, 4, 8, 13, 17)
        for left in range(4)
        for right in range(left + 1, 4)
    )
    cross_node_left = torch.tensor(
        tuple(left for left, _ in cross_node_pairs),
        device=samples.device,
        dtype=torch.long,
    )
    cross_node_right = torch.tensor(
        tuple(right for _, right in cross_node_pairs),
        device=samples.device,
        dtype=torch.long,
    )
    cross_node_value = _pair_variogram(
        samples.index_select(-1, cross_node_left),
        samples.index_select(-1, cross_node_right),
        truth.index_select(-1, cross_node_left),
        truth.index_select(-1, cross_node_right),
    )
    terms["cross_node"] = _score_term(
        cross_node_value.flatten(start_dim=1).mean(dim=1),
        label="variogram/cross_node",
    )

    scenario_variables = torch.stack(
        (
            groups_s["price"][..., 0:4].mean(dim=-1),
            groups_s["price"][..., 4:8].mean(dim=-1),
            groups_s["price"][..., 8:12].mean(dim=-1),
            groups_s["workload"].squeeze(-1),
            groups_s["pv"].mean(dim=-1),
            groups_s["pue"].mean(dim=-1),
        ),
        dim=-1,
    )
    target_variables = torch.stack(
        (
            groups_y["price"][..., 0:4].mean(dim=-1),
            groups_y["price"][..., 4:8].mean(dim=-1),
            groups_y["price"][..., 8:12].mean(dim=-1),
            groups_y["workload"].squeeze(-1),
            groups_y["pv"].mean(dim=-1),
            groups_y["pue"].mean(dim=-1),
        ),
        dim=-1,
    )
    cross_variable_pairs = tuple(
        (left, right) for left in range(6) for right in range(left + 1, 6)
    )
    cross_variable_left = torch.tensor(
        tuple(left for left, _ in cross_variable_pairs),
        device=samples.device,
        dtype=torch.long,
    )
    cross_variable_right = torch.tensor(
        tuple(right for _, right in cross_variable_pairs),
        device=samples.device,
        dtype=torch.long,
    )
    cross_variable_value = _pair_variogram(
        scenario_variables.index_select(-1, cross_variable_left),
        scenario_variables.index_select(-1, cross_variable_right),
        target_variables.index_select(-1, cross_variable_left),
        target_variables.index_select(-1, cross_variable_right),
    )
    terms["cross_variable"] = _score_term(
        cross_variable_value.flatten(start_dim=1).mean(dim=1),
        label="variogram/cross_variable",
    )
    return terms


def _threshold_pair(
    value: Any, *, family: str, shape: torch.Size
) -> tuple[Tensor, Tensor]:
    if isinstance(value, Mapping) or is_dataclass(value):
        packet = packet_mapping(value)
        lower = _tensor(
            _lookup(packet, "lower", "low"), label=f"tail_thresholds.{family}.lower"
        )
        upper = _tensor(
            _lookup(packet, "upper", "high"), label=f"tail_thresholds.{family}.upper"
        )
    else:
        tensor = _tensor(value, label=f"tail_thresholds.{family}")
        if tensor.shape[-1] != 2:
            raise WorldLossError(f"tail_thresholds.{family} must end in [lower,upper]")
        lower, upper = tensor[..., 0], tensor[..., 1]
    lower = _broadcast(lower, shape, label=f"tail_thresholds.{family}.lower")
    upper = _broadcast(upper, shape, label=f"tail_thresholds.{family}.upper")
    if bool((lower >= upper).any()):
        raise WorldLossError(f"tail_thresholds.{family} lower must be below upper")
    return lower, upper


def threshold_weighted_crps_term(
    scenarios: Tensor,
    target: Tensor,
    *,
    lower: Tensor,
    upper: Tensor,
    label: str,
    element_weights: Tensor | None = None,
) -> ScoreTerm:
    """Proper two-tail CRPS using fixed threshold chaining functions."""

    lower32 = _broadcast(lower.float(), target.shape, label=f"{label} lower")
    upper32 = _broadcast(upper.float(), target.shape, label=f"{label} upper")
    lower_paths, upper_paths = lower32.unsqueeze(1), upper32.unsqueeze(1)
    lower_term = ensemble_crps_term(
        F.relu(lower_paths - scenarios),
        F.relu(lower32 - target),
        label=f"{label}/lower",
        element_weights=element_weights,
    )
    upper_term = ensemble_crps_term(
        F.relu(scenarios - upper_paths),
        F.relu(target - upper32),
        label=f"{label}/upper",
        element_weights=element_weights,
    )
    return ScoreTerm(
        0.5 * (lower_term.numerator + upper_term.numerator),
        0.5 * (lower_term.valid_count + upper_term.valid_count),
    )


def latent_mixture_nll_term(mixture: Any, target: Tensor) -> ScoreTerm:
    packet = packet_mapping(mixture)
    logits = _tensor(
        _lookup(packet, "logits", "log_weights"), label="latent_mixture.logits"
    )
    means = _tensor(_lookup(packet, "means", "loc"), label="latent_mixture.means")
    log_scales = _tensor(
        _lookup(packet, "log_scales", "log_scale"), label="latent_mixture.log_scales"
    ).clamp(-20.0, 10.0)
    target32 = target.float()
    if means.shape != log_scales.shape or means.ndim < 3:
        raise WorldLossError("latent mixture means/log_scales must be [...,K,D]")
    if target32.shape != means.shape[:-2] + (means.shape[-1],):
        raise WorldLossError("latent mixture target must be [...,D]")
    if logits.shape != means.shape[:-1]:
        raise WorldLossError("latent mixture logits must be [...,K]")
    standardized = (target32.unsqueeze(-2) - means) * torch.exp(-log_scales)
    component_log_prob = (
        -0.5 * standardized.square() - log_scales - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)
    log_prob = torch.logsumexp(
        F.log_softmax(logits, dim=-1) + component_log_prob, dim=-1
    )
    return _score_term(-log_prob, label="mixture_nll")


def _normalized_public_groups(
    scenarios: Tensor, target: Tensor, scales: Mapping[str, Any]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    scenario_groups = _split_public(scenarios, label="U0 scenarios")
    target_groups = _split_public(target, label="U0 target")
    normalized_scenarios: dict[str, Tensor] = {}
    normalized_targets: dict[str, Tensor] = {}
    for family in PUBLIC_GROUP_SLICES:
        scenario_value = _family_transform(scenario_groups[family], family)
        target_value = _family_transform(target_groups[family], family)
        scale = _family_scale(scales, family, target_value.shape)
        normalized_scenarios[family] = scenario_value / scale.unsqueeze(1)
        normalized_targets[family] = target_value / scale
    return normalized_scenarios, normalized_targets


def _join_public(groups: Mapping[str, Tensor]) -> Tensor:
    return torch.cat(tuple(groups[name] for name in PUBLIC_GROUP_SLICES), dim=-1)


def _joint_family_balancing(value: Tensor) -> Tensor:
    """Give each public target family one quarter of joint Euclidean mass."""

    if value.shape[-1] != 21:
        raise WorldLossError("joint family balancing expects 21 public channels")
    channel_mass = torch.cat(
        (
            0.25 * _channel_weights("price", 12, device=value.device),
            0.25 * _channel_weights("workload", 1, device=value.device),
            0.25 * _channel_weights("pv", 4, device=value.device),
            0.25 * _channel_weights("pue", 4, device=value.device),
        )
    )
    return value * channel_mass.sqrt()


def _atom_support_bounds(
    batch: Any,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    world_input = _lookup(
        batch, "world_input", "input_packet", "packet", "model_input"
    )
    constraints = _lookup(world_input, "constraints")

    def bound(name: str) -> Tensor:
        value = _tensor(_lookup(constraints, name), label=f"constraints.{name}")
        if value.shape != (target.shape[0], 4):
            raise WorldLossError(f"constraints.{name} must be [B,4]")
        return value.float()[:, None].expand(-1, target.shape[1], -1)

    return bound("pv_capacity"), bound("pue_min"), bound("pue_max")


def _support_probability(
    probabilities: Mapping[str, Any],
    name: str,
    *,
    target_shape: torch.Size,
) -> Tensor:
    value = _tensor(_lookup(probabilities, name), label=f"support_probabilities.{name}")
    value32 = value.float()
    if value32.shape == target_shape:
        result = value32
    elif (
        value32.ndim == len(target_shape) + 1
        and value32.shape[0] == target_shape[0]
        and value32.shape[2:] == target_shape[1:]
    ):
        # Legacy R5-V2.4 exposes one conditional probability per stochastic
        # latent path.  V2.5 exposes the path-free analytic probability.  Both
        # are accepted, but the latter never acquires Monte Carlo calibration
        # noise from an empirical path mean.
        result = value32.mean(dim=1)
    else:
        raise WorldLossError(
            f"support_probabilities.{name} must have analytic shape "
            f"{tuple(target_shape)} or add one path dimension"
        )
    if not bool(torch.isfinite(result).all()) or bool(
        ((result < 0.0) | (result > 1.0)).any()
    ):
        raise WorldLossError(f"support_probabilities.{name} must lie in [0,1]")
    return result


def _analytic_distribution(output: Any) -> Mapping[str, Any] | None:
    raw = _lookup(
        output,
        "analytic_distribution",
        "mixed_distribution",
        "distribution_parameters",
        required=False,
    )
    return None if raw is None else packet_mapping(raw)


def mixed_support_quantile_score_terms(
    output: Any,
    batch: Any,
    target: Tensor,
) -> dict[str, ScoreTerm]:
    """Composite proper scores for atoms and conditional interiors.

    A mixed distribution is identified by categorical boundary mass and a
    conditional continuous distribution.  The categorical part is scored by
    :func:`atom_brier_score_terms`; this function scores price unconditionally
    and workload/PV/PUE only when the realized value lies in their continuous
    interior.  The sum is a proper composite score without requiring a rare
    component to appear in a finite scenario ensemble.
    """

    distribution = _analytic_distribution(output)
    if distribution is None:
        raise WorldLossError("analytic mixed distribution is missing")
    _assert_quantile_levels(distribution)
    quantiles_raw = _lookup(distribution, "quantiles", "conditional_quantiles")
    quantiles = packet_mapping(quantiles_raw)
    missing = set(PUBLIC_GROUP_SLICES).difference(quantiles)
    if missing:
        raise WorldLossError(
            f"analytic distribution quantiles are missing {sorted(missing)}"
        )
    joint_price_paths = _lookup(
        distribution, "joint_price_paths", required=False
    )
    if joint_price_paths is not None:
        _assert_joint_price_paths(
            joint_price_paths, label="analytic_distribution.joint_price_paths"
        )

    groups = _split_public(target.float(), label="analytic distribution target")
    _assert_price_identity(groups["price"], label="analytic distribution target price")
    pv_capacity, pue_min, pue_max = _atom_support_bounds(batch, target)
    masks = {
        "price": torch.ones_like(groups["price"], dtype=torch.bool),
        "workload": groups["workload"].gt(0.0),
        "pv": ~(groups["pv"].eq(0.0) | groups["pv"].eq(pv_capacity)),
        "pue": ~(groups["pue"].eq(pue_min) | groups["pue"].eq(pue_max)),
    }
    scales = packet_mapping(_lookup(batch, "quantile_scales", "target_scales"))
    terms: dict[str, ScoreTerm] = {}
    for family in PUBLIC_GROUP_SLICES:
        prediction = _tensor(
            quantiles[family], label=f"analytic_distribution.quantiles.{family}"
        )
        truth = groups[family]
        terms[family] = quantile_pinball_term(
            prediction,
            truth,
            scale=_family_scale(scales, family, truth.shape),
            mask=masks[family],
            family=family,
            label=f"analytic_quantile/{family}",
            allow_empty=family != "price",
            quantile_weights=_quantile_integration_weights(
                device=prediction.device
            ),
        )
    return terms


def _multiclass_brier_term(
    forecast: Tensor,
    truth: Tensor,
    *,
    label: str,
) -> ScoreTerm:
    if forecast.shape != truth.shape or forecast.ndim < 2:
        raise WorldLossError(f"{label} forecast/truth class tensors differ")
    if not bool(torch.isfinite(forecast).all()) or not bool(
        torch.isfinite(truth).all()
    ):
        raise WorldLossError(f"{label} contains non-finite values")
    if bool((forecast < 0.0).any()) or bool((forecast > 1.0).any()):
        raise WorldLossError(f"{label} forecast probabilities must lie in [0,1]")
    if not torch.allclose(
        forecast.sum(dim=-1),
        torch.ones_like(forecast[..., 0]),
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise WorldLossError(f"{label} forecast classes must sum to one")
    if not torch.equal(truth.sum(dim=-1), torch.ones_like(truth[..., 0])):
        raise WorldLossError(f"{label} truth classes must be one-hot")
    # Averaging the class axis makes the binary hurdle reduce to the familiar
    # (p-y)^2 and prevents three-class families receiving extra weight merely
    # because their support has two boundary atoms.
    return _score_term((forecast - truth).square().mean(dim=-1), label=label)


def atom_brier_score_terms(
    output: Any,
    batch: Any,
    target: Tensor,
) -> dict[str, ScoreTerm]:
    """Proper occurrence scores for structural support atoms.

    The model supplies one atom probability per generated path.  Their path
    mean is the predictive categorical probability compared with the realized
    boundary label.  Continuous public scores remain unconditional elsewhere.
    """

    distribution = _analytic_distribution(output)
    raw_probabilities = (
        None
        if distribution is None
        else _lookup(
            distribution,
            "atom_probabilities",
            "support_probabilities",
            required=False,
        )
    )
    if raw_probabilities is None:
        raw_probabilities = _lookup(output, "support_probabilities")
    probabilities = packet_mapping(raw_probabilities)
    batch_size, time, _ = target.shape
    workload = target[..., 12:13].float()
    pv = target[..., 13:17].float()
    pue = target[..., 17:21].float()
    pv_capacity, pue_min, pue_max = _atom_support_bounds(batch, target)

    workload_zero = _support_probability(
        probabilities,
        "workload_zero",
        target_shape=torch.Size((batch_size, time, 1)),
    )
    workload_forecast = torch.stack(
        (workload_zero, 1.0 - workload_zero), dim=-1
    )
    workload_label = workload.eq(0.0).float()
    workload_truth = torch.stack((workload_label, 1.0 - workload_label), dim=-1)

    pv_zero = _support_probability(
        probabilities,
        "pv_zero",
        target_shape=torch.Size((batch_size, time, 4)),
    )
    pv_capacity_probability = _support_probability(
        probabilities,
        "pv_capacity",
        target_shape=torch.Size((batch_size, time, 4)),
    )
    pv_interior = 1.0 - pv_zero - pv_capacity_probability
    pv_forecast = torch.stack(
        (pv_zero, pv_capacity_probability, pv_interior), dim=-1
    )
    pv_zero_label = pv.eq(0.0)
    pv_capacity_label = pv.eq(pv_capacity)
    if bool((pv_zero_label & pv_capacity_label).any()):
        raise WorldLossError("PV zero and capacity labels overlap")
    pv_truth = torch.stack(
        (
            pv_zero_label.float(),
            pv_capacity_label.float(),
            (~(pv_zero_label | pv_capacity_label)).float(),
        ),
        dim=-1,
    )

    pue_min_probability = _support_probability(
        probabilities,
        "pue_min",
        target_shape=torch.Size((batch_size, time, 4)),
    )
    pue_max_probability = _support_probability(
        probabilities,
        "pue_max",
        target_shape=torch.Size((batch_size, time, 4)),
    )
    pue_interior = 1.0 - pue_min_probability - pue_max_probability
    pue_forecast = torch.stack(
        (pue_min_probability, pue_max_probability, pue_interior), dim=-1
    )
    pue_min_label = pue.eq(pue_min)
    pue_max_label = pue.eq(pue_max)
    if bool((pue_min_label & pue_max_label).any()):
        raise WorldLossError("PUE min and max labels overlap")
    pue_truth = torch.stack(
        (
            pue_min_label.float(),
            pue_max_label.float(),
            (~(pue_min_label | pue_max_label)).float(),
        ),
        dim=-1,
    )
    return {
        "workload": _multiclass_brier_term(
            workload_forecast, workload_truth, label="atom/workload"
        ),
        "pv": _multiclass_brier_term(pv_forecast, pv_truth, label="atom/pv"),
        "pue": _multiclass_brier_term(pue_forecast, pue_truth, label="atom/pue"),
    }


def u0_stochastic_calibration_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    include_route_supervision: bool = True,
) -> PhaseLoss:
    """U0 marginal and joint proper scores with fixed-tail weighting."""

    _assert_frozen_target_metadata(batch)
    scenarios, target = _scenario_and_target(output, batch)
    scale_mapping = packet_mapping(_lookup(batch, "quantile_scales", "target_scales"))
    scenario_groups, target_groups = _normalized_public_groups(
        scenarios, target, scale_mapping
    )
    analytic_distribution = _analytic_distribution(output)
    components: dict[str, Tensor] = {}
    score_terms: dict[str, ScoreTerm] = {}

    # Build normal, lower-tail and upper-tail transformed paths together.  One
    # [B,M,M,T,63] training reduction replaces twelve family/side-specific
    # pairwise graphs while retaining exactly the same per-family ScoreTerms.
    thresholds = packet_mapping(_lookup(batch, "tail_thresholds"))
    raw_target_groups = _split_public(target, label="tail target")
    lower_groups: dict[str, Tensor] = {}
    upper_groups: dict[str, Tensor] = {}
    for family in PUBLIC_GROUP_SLICES:
        if family not in thresholds:
            raise WorldLossError(f"tail_thresholds is missing {family}")
        raw_target = raw_target_groups[family]
        lower, upper = _threshold_pair(
            thresholds[family], family=family, shape=raw_target.shape
        )
        tail_scale = _family_scale(scale_mapping, family, raw_target.shape)
        lower_groups[family] = _family_transform(lower, family) / tail_scale
        upper_groups[family] = _family_transform(upper, family) / tail_scale

    joined_scenarios = _join_public(scenario_groups)
    joined_target = _join_public(target_groups)
    joined_lower = _join_public(lower_groups)
    joined_upper = _join_public(upper_groups)
    lower_scenarios = F.relu(joined_lower.unsqueeze(1) - joined_scenarios)
    lower_target = F.relu(joined_lower - joined_target)
    upper_scenarios = F.relu(joined_scenarios - joined_upper.unsqueeze(1))
    upper_target = F.relu(joined_target - joined_upper)
    fused_crps_values = _fair_ensemble_crps_values(
        torch.cat((joined_scenarios, lower_scenarios, upper_scenarios), dim=-1),
        torch.cat((joined_target, lower_target, upper_target), dim=-1),
    )

    scenario_crps_terms: dict[str, ScoreTerm] = {}
    tail_terms: dict[str, ScoreTerm] = {}
    for family, channel_slice in PUBLIC_GROUP_SLICES.items():
        channel_weights = _channel_weights(
            family, int(target_groups[family].shape[-1]), device=scenarios.device
        )
        scenario_crps_terms[family] = _score_term(
            fused_crps_values[..., channel_slice],
            label=f"scenario_crps/{family}",
            element_weights=channel_weights,
        )
        lower_slice = slice(channel_slice.start + 21, channel_slice.stop + 21)
        upper_slice = slice(channel_slice.start + 42, channel_slice.stop + 42)
        lower_term = _score_term(
            fused_crps_values[..., lower_slice],
            label=f"tail/{family}/lower",
            element_weights=channel_weights,
        )
        upper_term = _score_term(
            fused_crps_values[..., upper_slice],
            label=f"tail/{family}/upper",
            element_weights=channel_weights,
        )
        tail_terms[family] = ScoreTerm(
            0.5 * (lower_term.numerator + upper_term.numerator),
            0.5 * (lower_term.valid_count + upper_term.valid_count),
        )
        scenario_key = (
            f"scenario_crps/{family}"
            if analytic_distribution is not None
            else f"crps/{family}"
        )
        _register_term(
            scenario_key,
            scenario_crps_terms[family],
            components=components,
            score_terms=score_terms,
        )
    scenario_crps = _weighted_scalar(
        scenario_crps_terms,
        {name: 0.25 for name in PUBLIC_GROUP_SLICES},
        label="U0 scenario CRPS",
    )
    if analytic_distribution is None:
        crps = scenario_crps
    else:
        analytic_terms = mixed_support_quantile_score_terms(output, batch, target)
        for family, term in analytic_terms.items():
            _register_term(
                f"crps/{family}",
                term,
                components=components,
                score_terms=score_terms,
            )
        # CRPS equals twice the integral of quantile loss.  Nine fixed levels
        # provide the registered numerical quadrature; every constituent
        # pinball score remains proper.
        crps = 2.0 * _weighted_scalar(
            analytic_terms,
            {name: 0.25 for name in PUBLIC_GROUP_SLICES},
            label="U0 analytic quantile score",
        )
        components["scenario_crps"] = scenario_crps

    normalized_scenarios = _joint_family_balancing(_join_public(scenario_groups))
    normalized_target = _joint_family_balancing(_join_public(target_groups))
    energy_term = energy_score_term(normalized_scenarios, normalized_target)
    _register_term(
        "energy", energy_term, components=components, score_terms=score_terms
    )

    variogram_terms = variogram_score_terms(normalized_scenarios, normalized_target)
    for name, term in variogram_terms.items():
        _register_term(
            f"variogram/{name}", term, components=components, score_terms=score_terms
        )
    variogram = _weighted_scalar(
        variogram_terms,
        {"time": 1.0 / 3.0, "cross_node": 1.0 / 3.0, "cross_variable": 1.0 / 3.0},
        label="U0 variogram",
    )

    for family in PUBLIC_GROUP_SLICES:
        _register_term(
            f"tail/{family}",
            tail_terms[family],
            components=components,
            score_terms=score_terms,
        )
    tail = _weighted_scalar(
        tail_terms, {name: 0.25 for name in PUBLIC_GROUP_SLICES}, label="U0 tail"
    )

    raw_support_probabilities = (
        None
        if analytic_distribution is None
        else _lookup(
            analytic_distribution,
            "atom_probabilities",
            "support_probabilities",
            required=False,
        )
    )
    if raw_support_probabilities is None:
        raw_support_probabilities = _lookup(
            output, "support_probabilities", required=False
        )
    support_probability_mapping = (
        {}
        if raw_support_probabilities is None
        else packet_mapping(raw_support_probabilities)
    )
    atom_terms: dict[str, ScoreTerm] = {}
    atom_brier: Tensor | None = None
    if support_probability_mapping:
        atom_terms = atom_brier_score_terms(output, batch, target)
        for family, term in atom_terms.items():
            _register_term(
                f"atom/{family}",
                term,
                components=components,
                score_terms=score_terms,
            )
        atom_brier = _weighted_scalar(
            atom_terms,
            {name: 1.0 / 3.0 for name in atom_terms},
            label="U0 atom Brier",
        )
    elif float(weights.atom_brier) != 0.0:
        raise WorldLossError(
            "a nonzero atom Brier weight requires support_probabilities"
        )

    mixture = _lookup(output, "latent_mixture", "mixture_distribution")
    mixture_target_value = _lookup(
        output,
        "latent_mixture_target",
        "posterior_latent_target",
        required=False,
    )
    if mixture_target_value is None:
        mixture_target_value = _lookup(batch, "latent_mixture_target", "latent_target")
    mixture_target = _tensor(
        mixture_target_value, label="latent_mixture_target"
    ).detach()
    mixture_weight = float(weights.mixture_nll)
    if mixture_weight == 0.0:
        # R5-V2.3 keeps this legacy quantity for paper diagnostics only.  Do
        # not retain its large latent-density graph when its registered
        # objective weight is exactly zero.
        with torch.no_grad():
            mixture_term = latent_mixture_nll_term(mixture, mixture_target)
    else:
        mixture_term = latent_mixture_nll_term(mixture, mixture_target)
    _register_term(
        "mixture_nll", mixture_term, components=components, score_terms=score_terms
    )
    auxiliary = _auxiliary_loss(output, scenarios)
    route_supervision = (
        _queue_tail_route_supervision_loss(output, batch, scenarios)
        if include_route_supervision
        else _zero(scenarios)
    )
    total = (
        weights.crps * crps
        + weights.energy * energy_term.scalar
        + weights.variogram * variogram
        + weights.tail * tail
        + weights.auxiliary * auxiliary
        + weights.route_supervision * route_supervision
    )
    if atom_brier is not None:
        total = total + float(weights.atom_brier) * atom_brier
    if mixture_weight != 0.0:
        total = total + mixture_weight * mixture_term.scalar
    components.update(
        crps=crps,
        energy=energy_term.scalar,
        variogram=variogram,
        tail=tail,
        mixture_nll=mixture_term.scalar,
        auxiliary=auxiliary,
        route_supervision=route_supervision,
        total=_finite("U0 total", total),
    )
    if atom_brier is not None:
        components["atom_brier"] = atom_brier
    # Coverage and duplicate rate are validation gates, never gradients.
    return PhaseLoss("u0", total, components, score_terms)


def rockafellar_uryasev_cvar(
    costs: Tensor, eta: Tensor, *, alpha: float = 0.95
) -> ScoreTerm:
    """Differentiable Rockafellar-Uryasev CVaR objective."""

    costs32, eta32 = costs.float(), eta.float()
    if costs32.ndim != 2:
        raise WorldLossError("CVaR costs must be [batch,paths]")
    if not 0.5 <= float(alpha) < 1.0:
        raise WorldLossError("cvar_alpha must be in [0.5,1)")
    if eta32.ndim == 0:
        eta32 = eta32.expand(costs32.shape[0])
    elif eta32.shape == (costs32.shape[0], 1):
        eta32 = eta32.squeeze(-1)
    if eta32.shape != (costs32.shape[0],):
        raise WorldLossError("CVaR eta must be scalar or [batch]")
    per_origin = eta32 + F.relu(costs32 - eta32.unsqueeze(1)).mean(dim=1) / (
        1.0 - float(alpha)
    )
    return _score_term(per_origin, label="cvar")


def _planner_mapping(output: Any) -> Mapping[str, Any]:
    planner = _lookup(output, "planner", "planning", "plan", required=False)
    return packet_mapping(output if planner is None else planner)


def _planner_lookup(output: Any, *names: str, required: bool = True) -> Any:
    planner = _planner_mapping(output)
    for name in names:
        if name in planner and planner[name] is not None:
            return planner[name]
    if required:
        raise WorldLossError(f"planner output is missing one of {list(names)}")
    return None


def augmented_lagrangian_term(
    violations: Any,
    multipliers: Any,
    penalties: Any,
) -> ScoreTerm:
    """Inequality augmented Lagrangian evaluated on raw (pre-shield) actions."""

    violation_mapping = (
        packet_mapping(violations)
        if not isinstance(violations, Tensor)
        else {"all": violations}
    )
    multiplier_mapping = (
        packet_mapping(multipliers)
        if not isinstance(multipliers, Tensor)
        else {"all": multipliers}
    )
    penalty_mapping = (
        packet_mapping(penalties)
        if not isinstance(penalties, Tensor)
        else {"all": penalties}
    )
    if set(violation_mapping) != set(multiplier_mapping) or set(
        violation_mapping
    ) != set(penalty_mapping):
        raise WorldLossError(
            "raw violations, multipliers, and penalties must have matching keys"
        )
    raw_values: list[Tensor] = []
    for name in sorted(violation_mapping):
        violation = _tensor(
            violation_mapping[name], label=f"raw_constraint_violations.{name}"
        )
        multiplier = _broadcast(
            _tensor(multiplier_mapping[name], label=f"constraint_multipliers.{name}"),
            violation.shape,
            label=f"constraint_multipliers.{name}",
        )
        penalty = _broadcast(
            _tensor(penalty_mapping[name], label=f"constraint_penalties.{name}"),
            violation.shape,
            label=f"constraint_penalties.{name}",
        )
        if bool((multiplier < 0).any()) or bool((penalty <= 0).any()):
            raise WorldLossError(
                "augmented-Lagrangian multipliers must be nonnegative and penalties positive"
            )
        positive = F.relu(violation)
        raw_values.append(multiplier * positive + 0.5 * penalty * positive.square())
    return _score_term(
        torch.cat([value.reshape(-1) for value in raw_values]),
        label="augmented_lagrangian",
    )


def p1_latent_planner_loss(
    output: Any,
    batch: Any,
    *,
    cvar_alpha: float = 0.95,
    minimum_cvar_paths: int = 100,
    weights: WorldLossWeights = WorldLossWeights(),
    include_route_supervision: bool = True,
) -> PhaseLoss:
    """P1 normalized two-settlement cost, RU-CVaR, and raw constraints."""

    raw_costs = _tensor(
        _planner_lookup(output, "normalized_scenario_costs"),
        label="normalized_scenario_costs",
    )
    if raw_costs.ndim < 2:
        raise WorldLossError(
            "normalized scenario costs must include batch and path axes"
        )
    costs = raw_costs.flatten(start_dim=1)
    if costs.shape[1] < int(minimum_cvar_paths):
        raise WorldLossError(
            f"CVaR95 requires at least {minimum_cvar_paths} aggregated paths; got {costs.shape[1]}"
        )
    expected_term = _score_term(costs.mean(dim=1), label="expected_cost")
    eta = _tensor(_planner_lookup(output, "cvar_eta", "eta"), label="cvar_eta")
    cvar_term = rockafellar_uryasev_cvar(costs, eta, alpha=cvar_alpha)
    policy_expected_term = expected_term
    policy_cvar_term = cvar_term
    critic_distillation_term: ScoreTerm | None = None
    raw_cost_distribution = _planner_lookup(
        output, "cost_distribution", required=False
    )
    if raw_cost_distribution is not None:
        distribution = packet_mapping(raw_cost_distribution)
        levels = _tensor(
            _lookup(distribution, "levels"), label="cost_distribution.levels"
        ).float()
        first = _tensor(
            _lookup(distribution, "first_quantiles"),
            label="cost_distribution.first_quantiles",
        ).float()
        second = _tensor(
            _lookup(distribution, "second_quantiles"),
            label="cost_distribution.second_quantiles",
        ).float()
        conservative = _tensor(
            _lookup(distribution, "quantiles"),
            label="cost_distribution.quantiles",
        ).float()
        if (
            levels.ndim != 1
            or first.shape != second.shape
            or first.shape != conservative.shape
            or first.shape != (costs.shape[0], levels.numel())
            or bool((levels[1:] <= levels[:-1]).any())
            or bool((first[:, 1:] < first[:, :-1]).any())
            or bool((second[:, 1:] < second[:, :-1]).any())
        ):
            raise WorldLossError("cost critic quantile contract is invalid")

        def critic_pinball(prediction: Tensor) -> Tensor:
            error = costs.detach().unsqueeze(1) - prediction.unsqueeze(-1)
            tau = levels.view(1, -1, 1)
            return torch.maximum(tau * error, (tau - 1.0) * error)

        critic_distillation_term = _score_term(
            0.5 * (critic_pinball(first) + critic_pinball(second)),
            label="cost_critic_distillation",
        )
        policy_expected_term = _score_term(
            conservative.mean(dim=-1), label="critic_expected_cost"
        )
        tail_mask = levels >= float(cvar_alpha)
        if not bool(tail_mask.any()):
            raise WorldLossError("cost critic grid has no CVaR tail quantiles")
        policy_cvar_term = _score_term(
            conservative[:, tail_mask].mean(dim=-1), label="critic_cvar"
        )
    elif float(weights.cost_critic_distillation) != 0.0:
        raise WorldLossError(
            "nonzero cost critic weight requires a distributional cost critic"
        )
    raw_violations = _planner_lookup(output, "raw_constraint_violations")
    multipliers = _planner_lookup(
        output, "constraint_multipliers", "lagrange_multipliers"
    )
    penalties = _planner_lookup(output, "constraint_penalties", "penalty_coefficients")
    constraint_term = augmented_lagrangian_term(raw_violations, multipliers, penalties)

    raw_actions = _tensor(_planner_lookup(output, "raw_actions"), label="raw_actions")
    if raw_actions.ndim < 3:
        raise WorldLossError("raw_actions must contain batch/time/action axes")
    temporal_ramp = (
        (raw_actions[..., 1:, :] - raw_actions[..., :-1, :]).square()
        if raw_actions.shape[-2] > 1
        else raw_actions[..., :0, :].square()
    )
    previous_ramp = _planner_lookup(
        output, "previous_ramp_action", "ramp_initial_action", required=False
    )
    previous_mask = _planner_lookup(output, "previous_ramp_mask", required=False)
    if previous_ramp is None and previous_mask is None:
        # Compatibility for synthetic/legacy test packets. Formal P1 always
        # supplies the last causal action and its evidence mask.
        ramp_raw = temporal_ramp
        ramp_mask = None
    elif previous_ramp is None or previous_mask is None:
        raise WorldLossError("previous ramp action and mask must be paired")
    else:
        previous = _tensor(previous_ramp, label="previous_ramp_action")
        mask = _tensor(previous_mask, label="previous_ramp_mask")
        expected_previous = raw_actions.shape[:-2] + (raw_actions.shape[-1],)
        if previous.shape != expected_previous or mask.shape != expected_previous:
            raise WorldLossError(
                "previous ramp action/mask must match batch and action dimensions"
            )
        first = (raw_actions[..., 0, :] - previous).square().unsqueeze(-2)
        ramp_raw = torch.cat((first, temporal_ramp), dim=-2)
        ramp_mask = torch.cat(
            (
                (mask > 0).unsqueeze(-2),
                torch.ones_like(temporal_ramp, dtype=torch.bool),
            ),
            dim=-2,
        )
    ramp_term = _score_term(ramp_raw, mask=ramp_mask, label="ramp")
    shield = _tensor(
        _planner_lookup(output, "shield_correction", "shield_intervention"),
        label="shield_correction",
    ).abs()
    shield_term = _score_term(shield, label="shield_correction")
    auxiliary = _auxiliary_loss(output, costs)
    route_supervision = (
        _queue_tail_route_supervision_loss(output, batch, costs)
        if include_route_supervision
        else _zero(costs)
    )
    total = (
        weights.expected_cost * policy_expected_term.scalar
        + weights.cvar * policy_cvar_term.scalar
        + weights.augmented_lagrangian * constraint_term.scalar
        + weights.ramp * ramp_term.scalar
        + weights.shield_correction * shield_term.scalar
        + weights.auxiliary * auxiliary
        + weights.route_supervision * route_supervision
    )
    if critic_distillation_term is not None:
        total = total + (
            float(weights.cost_critic_distillation)
            * critic_distillation_term.scalar
        )
    components: dict[str, Tensor] = {}
    score_terms: dict[str, ScoreTerm] = {}
    for name, term in (
        ("expected_cost", expected_term),
        ("cvar", cvar_term),
        ("augmented_lagrangian", constraint_term),
        ("ramp", ramp_term),
        ("shield_correction", shield_term),
    ):
        _register_term(name, term, components=components, score_terms=score_terms)
    if critic_distillation_term is not None:
        _register_term(
            "critic/expected_cost",
            policy_expected_term,
            components=components,
            score_terms=score_terms,
        )
        _register_term(
            "critic/cvar",
            policy_cvar_term,
            components=components,
            score_terms=score_terms,
        )
        _register_term(
            "critic/distillation",
            critic_distillation_term,
            components=components,
            score_terms=score_terms,
        )
    components.update(
        auxiliary=auxiliary,
        route_supervision=route_supervision,
        total=_finite("P1 total", total),
    )
    return PhaseLoss("p1", total, components, score_terms)


def j1_pcgrad_joint_loss(
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    cvar_alpha: float = 0.95,
) -> PhaseLoss:
    """Expose W1, U0, and P1 as three independent J1 PCGrad objectives.

    ``total`` intentionally aliases the policy objective for scalar logging and
    checkpoint telemetry only.  It is not a joint optimization reduction; the
    trainer must consume ``optimization_objectives`` and apply PCGrad.
    """

    dynamics = w1_action_dynamics_loss(
        output, batch, weights=weights, include_route_supervision=False
    )
    stochastic = u0_stochastic_calibration_loss(
        output, batch, weights=weights, include_route_supervision=False
    )
    planner = p1_latent_planner_loss(
        output,
        batch,
        weights=weights,
        cvar_alpha=cvar_alpha,
        include_route_supervision=False,
    )
    route_supervision = _queue_tail_route_supervision_loss(
        output, batch, dynamics.total
    )
    world_objective = (
        dynamics.total
        if _lookup(batch, "queue_risk_event_target", required=False) is None
        else dynamics.total + weights.route_supervision * route_supervision
    )
    total = planner.total
    components: dict[str, Tensor] = {
        **{f"world/{name}": value for name, value in dynamics.components.items()},
        **{
            f"uncertainty/{name}": value
            for name, value in stochastic.components.items()
        },
        **{f"policy/{name}": value for name, value in planner.components.items()},
        "world/route_supervision": route_supervision,
        "world/total": world_objective,
        "total": _finite("J1 total", total),
    }
    score_terms = {
        **{f"world/{name}": term for name, term in dynamics.score_terms.items()},
        **{
            f"uncertainty/{name}": term for name, term in stochastic.score_terms.items()
        },
        **{f"policy/{name}": term for name, term in planner.score_terms.items()},
    }
    return PhaseLoss(
        "j1",
        total,
        components,
        score_terms,
        optimization_objectives={
            "world": world_objective,
            "uncertainty": stochastic.total,
            "policy": planner.total,
        },
    )


def phase_loss(
    phase: str,
    output: Any,
    batch: Any,
    *,
    weights: WorldLossWeights = WorldLossWeights(),
    cvar_alpha: float = 0.95,
    include_route_supervision: bool = True,
) -> PhaseLoss:
    """Dispatch the formal R5-V2 objective; no legacy fallback is allowed."""

    normalized = str(phase).lower()
    if normalized == "w0":
        return w0_masked_world_loss(output, batch, weights=weights)
    if normalized == "w1":
        return w1_action_dynamics_loss(
            output,
            batch,
            weights=weights,
            include_route_supervision=include_route_supervision,
        )
    if normalized == "u0":
        return u0_stochastic_calibration_loss(
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
        return j1_pcgrad_joint_loss(
            output, batch, weights=weights, cvar_alpha=cvar_alpha
        )
    if normalized == "c1":
        raise WorldLossError("C1 is evaluation-only and has no training loss")
    raise WorldLossError(f"unknown world-model phase: {phase}")


__all__ = [
    "DYNAMICS_HORIZONS",
    "PUBLIC_GROUP_SLICES",
    "QUANTILE_LEVELS",
    "PhaseLoss",
    "ScoreTerm",
    "TRAINING_PHASES",
    "WorldLossError",
    "WorldLossWeights",
    "aggregate_score_terms",
    "atom_brier_score_terms",
    "augmented_lagrangian_term",
    "balanced_kl_diagnostic_terms",
    "balanced_kl_term",
    "energy_score_term",
    "ensemble_crps_term",
    "j1_pcgrad_joint_loss",
    "latent_mixture_nll_term",
    "merge_score_terms",
    "mixed_support_quantile_score_terms",
    "negative_binomial_nll_term",
    "p1_latent_planner_loss",
    "phase_loss",
    "quantile_pinball_term",
    "rockafellar_uryasev_cvar",
    "threshold_weighted_crps_term",
    "u0_stochastic_calibration_loss",
    "variogram_score_terms",
    "w0_masked_world_loss",
    "w1_action_dynamics_loss",
]
