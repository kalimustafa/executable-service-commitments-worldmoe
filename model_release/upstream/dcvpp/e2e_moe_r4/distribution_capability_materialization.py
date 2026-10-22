"""Formal V6 capability materialization for the continuous Distribution head.

This module is intentionally separate from
``scenario_capability_materialization``.  The latter is immutable V5 history
for the failed fixed-20 Scenario lineage; this evaluator accepts only a V6
``distribution`` checkpoint and draws an arbitrary, preflight-frozen number of
samples through ``model.sample_distribution``.

The Level-2 candidate gate is probability-only.  It uses N=256 on all 179
2023-H2 validation dates.  The separately frozen N=512 empirical-risk budget is
reported but is not executed here because Joint Policy does not yet exist at
the Distribution stage.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date
import inspect
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .distributional_world import fp_safe_close_da_rt_dart

from .artifact_io import atomic_json, atomic_torch_save, read_json
from .capability_evaluation import (
    CapabilityEvidenceError,
    EXPECTED_VALIDATION_DAYS,
    evaluate_distribution,
)
from .hashing import sha256_file, sha256_json
from .scenario_training_contract import (
    ScenarioTrainingContract,
)
from .training_data import PreloadedR4Corpus, move_batch


SCHEMA = "dcvpp.r4.v6.distribution-capability-materialization.v1"
BASELINE_SCHEMA = "dcvpp.r4.v6.variable-n-training-residual-baseline.v1"
BASELINE_NAME = "frozen_training_residual_sampling_variable_n"
SAMPLING_CONTRACT_SCHEMA = "dcvpp.r4.v6.formal-sampling-contract.v1"
FORMAL_PROBABILITY_SAMPLES = 256
FORMAL_EMPIRICAL_RISK_SAMPLES = 512
HORIZON = 96
CHANNELS = 21
_FORBIDDEN_SOLVER_MODULES = ("pyomo", "gurobipy")
SUPPORT_CONTRACT_SCHEMA = "r4_v6_distribution_support.v1"


@dataclass(frozen=True)
class DistributionSupportContract:
    """Validated V6 support artifact and its immutable file binding.

    The V5 Scenario contract remains a read-only source lineage.  All active
    V6 scoring scales, residual caps, and absolute physical/statistical bounds
    are read from this new-root artifact.
    """

    path: Path
    artifact_sha256: str
    payload: Mapping[str, Any]

    @property
    def score_scales(self) -> Tensor:
        value = self.payload.get(
            "score_scales", self.payload.get("scenario_residual_score_scales")
        )
        return torch.as_tensor(value, dtype=torch.float32)

    @property
    def raw_p99_caps(self) -> Tensor:
        zonal = self.payload.get("residual_caps_zonal")
        if isinstance(zonal, list) and zonal:
            value = zonal[0]
        else:
            value = self.payload.get(
                "residual_caps",
                self.payload.get("scenario_residual_raw_p99_caps"),
            )
        return torch.as_tensor(value, dtype=torch.float32)

    @property
    def lower_zonal(self) -> Tensor:
        return torch.as_tensor(self.payload["lower_zonal"], dtype=torch.float32)

    @property
    def upper_zonal(self) -> Tensor:
        return torch.as_tensor(self.payload["upper_zonal"], dtype=torch.float32)

    @property
    def source_residual_contract_sha256(self) -> str:
        return str(
            self.payload.get(
                "source_residual_score_contract_sha256",
                self.payload.get("residual_score_contract_sha256", ""),
            )
        ).lower()


def _assert_solver_free_active_imports() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in _FORBIDDEN_SOLVER_MODULES)
    )
    if loaded:
        raise CapabilityEvidenceError(
            "V6 active Distribution path loaded an external mathematical solver: "
            + ", ".join(loaded[:8])
        )


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def load_distribution_support_contract(
    value: str | Path | Mapping[str, Any] | DistributionSupportContract,
    *,
    expected_source_residual_sha256: str | None = None,
) -> DistributionSupportContract:
    """Load the new-root V6 support contract without mutating V5 history."""

    if isinstance(value, DistributionSupportContract):
        contract = value
    else:
        if isinstance(value, Mapping):
            raw_path = value.get("path", value.get("artifact_path"))
            if raw_path is None:
                raise CapabilityEvidenceError(
                    "Distribution support mapping requires its immutable artifact path"
                )
            path = Path(str(raw_path)).resolve()
            payload_value = value.get("payload", value)
            if not isinstance(payload_value, Mapping):
                raise CapabilityEvidenceError("Distribution support payload is malformed")
            payload = dict(payload_value)
        else:
            path = Path(value).resolve()
            if not path.is_file():
                raise CapabilityEvidenceError(
                    f"V6 Distribution support artifact is missing: {path}"
                )
            payload = read_json(path)
        if not path.is_file():
            raise CapabilityEvidenceError(
                f"V6 Distribution support artifact is missing: {path}"
            )
        contract = DistributionSupportContract(path, sha256_file(path), payload)

    payload = contract.payload
    if payload.get("schema") != SUPPORT_CONTRACT_SCHEMA:
        raise CapabilityEvidenceError("wrong V6 Distribution support schema")
    if (
        payload.get("fit_split") != "train"
        or int(payload.get("fit_days", -1)) != 543
        or payload.get("fit_date_start") != "2022-01-01"
        or payload.get("fit_date_end") != "2023-06-30"
        or payload.get("future_years_opened") is not False
    ):
        raise CapabilityEvidenceError(
            "V6 Distribution support must be frozen on the 543 training days only"
        )
    source_sha = contract.source_residual_contract_sha256
    if not _is_sha256(source_sha):
        raise CapabilityEvidenceError(
            "V6 Distribution support lacks its read-only residual-score source hash"
        )
    if (
        expected_source_residual_sha256 is not None
        and source_sha != str(expected_source_residual_sha256).lower()
    ):
        raise CapabilityEvidenceError(
            "V6 Distribution support is bound to a different residual-score source"
        )
    scale = contract.score_scales
    caps = contract.raw_p99_caps
    lower = contract.lower_zonal
    upper = contract.upper_zonal
    if (
        scale.shape != (CHANNELS,)
        or caps.shape != (CHANNELS,)
        or lower.shape != (HORIZON, CHANNELS)
        or upper.shape != (HORIZON, CHANNELS)
        or not bool(torch.isfinite(scale).all())
        or not bool(torch.isfinite(caps).all())
        or not bool(torch.isfinite(lower).all())
        or not bool(torch.isfinite(upper).all())
        or bool((scale <= 0).any())
        or bool((caps <= 0).any())
        or bool((lower >= upper).any())
    ):
        raise CapabilityEvidenceError(
            "V6 Distribution support arrays have invalid shapes or values"
        )
    if not bool(
        torch.allclose(caps[4:8], caps[0:4] + caps[8:12], rtol=0.0, atol=1.0e-6)
    ):
        raise CapabilityEvidenceError("V6 RT residual caps are not induced by DA and DART")
    if not bool(
        torch.allclose(
            lower[:, 4:8], lower[:, 0:4] - upper[:, 8:12], rtol=0.0, atol=1.0e-5
        )
        and torch.allclose(
            upper[:, 4:8], upper[:, 0:4] - lower[:, 8:12], rtol=0.0, atol=1.0e-5
        )
    ):
        raise CapabilityEvidenceError("V6 RT absolute support is not induced by DA-DART")
    return contract


def attach_distribution_support_contract(
    batch: Mapping[str, Any],
    contract: DistributionSupportContract,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Attach the active V6 support values and their immutable binding."""

    result = dict(batch)
    scales = contract.score_scales.to(device=device)
    caps = contract.raw_p99_caps.to(device=device)
    lower = contract.lower_zonal.to(device=device)
    upper = contract.upper_zonal.to(device=device)
    result["scenario_residual_score_scales"] = scales
    result["scenario_residual_caps_zonal"] = caps[None].expand(HORIZON, CHANNELS)
    result["distribution_support"] = {
        "schema": SUPPORT_CONTRACT_SCHEMA,
        "lower_zonal": lower,
        "upper_zonal": upper,
        "contract_sha256": contract.artifact_sha256,
    }
    result["distribution_support_contract_sha256"] = contract.artifact_sha256
    return result


def _operating_date(record: Mapping[str, Any]) -> str:
    value = record.get("operating_date")
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError as exc:
        raise CapabilityEvidenceError(f"invalid corpus operating date {value!r}") from exc
    return parsed.isoformat()


def _validate_split_dates(
    train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train_dates = tuple(_operating_date(row) for row in train)
    validation_dates = tuple(_operating_date(row) for row in validation)
    if len(train_dates) != 543 or len(set(train_dates)) != 543:
        raise CapabilityEvidenceError(
            "Distribution baseline requires exactly 543 unique training dates"
        )
    if (
        len(validation_dates) != EXPECTED_VALIDATION_DAYS
        or len(set(validation_dates)) != EXPECTED_VALIDATION_DAYS
    ):
        raise CapabilityEvidenceError(
            "Distribution gate requires exactly 179 unique validation dates"
        )
    if train_dates != tuple(sorted(train_dates)) or validation_dates != tuple(
        sorted(validation_dates)
    ):
        raise CapabilityEvidenceError("formal corpus dates must be strictly ordered")
    if not all("2022-01-01" <= item <= "2023-06-30" for item in train_dates):
        raise CapabilityEvidenceError(
            "Distribution baseline attempted to use a non-training date"
        )
    if not all("2023-07-01" <= item <= "2023-12-31" for item in validation_dates):
        raise CapabilityEvidenceError(
            "Distribution gate attempted to use a date outside 2023-H2"
        )
    return train_dates, validation_dates


def _target(record: Mapping[str, Any]) -> Tensor:
    value = torch.as_tensor(record["scenario_target_zonal"], dtype=torch.float32).cpu()
    if value.shape != (HORIZON, CHANNELS) or not bool(torch.isfinite(value).all()):
        raise CapabilityEvidenceError(
            "Distribution target must be one finite [96,21] trajectory"
        )
    identity = value[:, 0:4] - value[:, 4:8] - value[:, 8:12]
    if float(identity.abs().max()) > 1.0e-5:
        raise CapabilityEvidenceError("Distribution target violates DA-RT-DART identity")
    return value


def _issue_center(record: Mapping[str, Any], pue_center: Tensor) -> Tensor:
    market = torch.as_tensor(record["market_raw"], dtype=torch.float32).cpu()
    workload = torch.as_tensor(record["issue_workload_raw"], dtype=torch.float32).cpu()
    pv = torch.as_tensor(record["issue_pv_raw"], dtype=torch.float32).cpu()
    if market.shape != (HORIZON, 16):
        raise CapabilityEvidenceError("raw issue-time market input must be [96,16]")
    if workload.shape != (HORIZON,) or pv.shape != (HORIZON, 4):
        raise CapabilityEvidenceError("raw issue-time workload/PV input has an invalid shape")
    da = market[:, 0:4]
    rt = market[:, 4:8]
    return torch.cat((da, rt, da - rt, workload[:, None], pv, pue_center), dim=-1)


def _corpus_signature(
    records: Sequence[Mapping[str, Any]], train_dates: Sequence[str]
) -> str:
    # Hash only already-registered source digests.  Large day payloads are not
    # repeatedly re-read or re-hashed by every candidate gate.
    sources: list[dict[str, Any]] = []
    for day, record in zip(train_dates, records, strict=True):
        provenance = record.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        sources.append(
            {
                "date": day,
                "bundle_sha256": provenance.get("bundle_sha256"),
                "settlement_sha256": provenance.get("settlement_sha256"),
                "physical_sha256": provenance.get("physical_sha256"),
            }
        )
    return sha256_json({"schema": BASELINE_SCHEMA, "training_sources": sources})


def _collate(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Tensor):
        return torch.stack([torch.as_tensor(value) for value in values])
    if isinstance(first, Mapping):
        return {key: _collate([value[key] for value in values]) for key in first}
    return list(values)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        first, second = left[name], right[name]
        if isinstance(first, Tensor) and isinstance(second, Tensor):
            if (
                first.shape != second.shape
                or first.dtype != second.dtype
                or not torch.equal(first.detach().cpu(), second.detach().cpu())
            ):
                return False
        elif first != second:
            return False
    return True


def _load_checkpoint_binding(
    checkpoint: Path,
    model: nn.Module,
    epoch: int | None,
    supplied_sha256: str | None,
) -> tuple[str, Mapping[str, Any]]:
    if not checkpoint.is_file():
        raise CapabilityEvidenceError(
            f"Distribution checkpoint does not exist: {checkpoint}"
        )
    if supplied_sha256 is None:
        checkpoint_sha = sha256_file(checkpoint)
    else:
        checkpoint_sha = str(supplied_sha256).lower()
        if not _is_sha256(checkpoint_sha):
            raise CapabilityEvidenceError("candidate checkpoint SHA-256 is malformed")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise CapabilityEvidenceError("Distribution checkpoint is missing model_state")
    if str(payload.get("phase", "")).lower() != "distribution":
        raise CapabilityEvidenceError(
            "V6 capability evaluator accepts only phase=distribution; "
            "fixed-20 Scenario checkpoints are history-only"
        )
    completed = int(payload.get("epochs_completed", int(payload.get("epoch", -1)) + 1))
    if epoch is not None and completed != int(epoch):
        raise CapabilityEvidenceError(
            f"Distribution evaluator epoch {epoch} does not match checkpoint epoch {completed}"
        )
    if not _states_equal(payload["model_state"], model.state_dict()):
        raise CapabilityEvidenceError(
            "live Distribution model is not the supplied immutable checkpoint"
        )
    return checkpoint_sha, payload


def _make_generator(reference: Tensor, seed: int) -> torch.Generator:
    device = reference.device if reference.is_cuda else torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _call_with_supported_keywords(
    target: Any, positional: tuple[Any, ...], keywords: Mapping[str, Any]
) -> Any:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(*positional, **dict(keywords))
    accepts_kwargs = any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    selected = (
        dict(keywords)
        if accepts_kwargs
        else {name: value for name, value in keywords.items() if name in signature.parameters}
    )
    return target(*positional, **selected)


def _revealed_from_batch(batch: Mapping[str, Any]) -> tuple[Any | None, Any | None]:
    revealed = batch.get("revealed")
    if isinstance(revealed, Mapping):
        return (
            revealed.get("zonal", revealed.get("revealed_zonal")),
            revealed.get("mask", revealed.get("revealed_mask")),
        )
    return batch.get("revealed_zonal"), batch.get("revealed_mask")


def sample_distribution(
    model: nn.Module,
    batch_or_context: Mapping[str, Any],
    sample_count: int,
    *,
    generator: torch.Generator,
) -> Mapping[str, Any]:
    """Invoke the formal arbitrary-N API, with one narrow V6 compatibility seam.

    The fixed-20 ``outputs['scenarios']`` view is deliberately never accepted.
    """

    count = int(sample_count)
    if isinstance(sample_count, bool) or count != sample_count or count < 1:
        raise CapabilityEvidenceError("Distribution sample count must be a positive integer")
    sampler = getattr(model, "sample_distribution", None)
    if callable(sampler):
        raw = _call_with_supported_keywords(
            sampler,
            (batch_or_context, count),
            {"generator": generator},
        )
    else:
        # During the architecture transition the formal head may already be
        # present while the top-level convenience method is not.  This remains
        # V6 continuous sampling, not a fallback to the old Scenario head.
        output = model(batch_or_context)
        if not isinstance(output, Mapping) or not isinstance(
            output.get("distribution"), Mapping
        ):
            raise CapabilityEvidenceError(
                "formal V6 model lacks model.sample_distribution and outputs['distribution']"
            )
        distribution = output["distribution"]
        world = distribution.get("world_latent")
        head = getattr(model, "distribution_head", None)
        if head is None:
            head = getattr(model, "distribution_generator", None)
        head_sampler = getattr(head, "sample", None)
        if world is not None and callable(head_sampler):
            revealed_zonal, revealed_mask = _revealed_from_batch(batch_or_context)
            raw = _call_with_supported_keywords(
                head_sampler,
                (world, count),
                {
                    "generator": generator,
                    "revealed_zonal": revealed_zonal,
                    "revealed_mask": revealed_mask,
                },
            )
        else:
            raw = _call_with_supported_keywords(
                model,
                (batch_or_context,),
                {
                    "distribution_sample_count": count,
                    "distribution_generator": generator,
                },
            )
    if not isinstance(raw, Mapping):
        raise CapabilityEvidenceError("Distribution sampler must return a mapping")
    if isinstance(raw.get("distribution"), Mapping) and any(
        key in raw["distribution"] for key in ("samples_zonal", "samples")
    ):
        raw = raw["distribution"]
    if not any(key in raw for key in ("samples_zonal", "samples")):
        raise CapabilityEvidenceError(
            "Distribution sampler did not return samples_zonal/samples"
        )
    if raw.get("fixed_particle_capacity") is True:
        raise CapabilityEvidenceError("formal Distribution sampler declares fixed capacity")
    if raw.get("sampling_budget") is not None and int(raw["sampling_budget"]) != count:
        raise CapabilityEvidenceError("Distribution sampler reported the wrong sampling budget")
    return raw


def _samples_and_weights(
    sampled: Mapping[str, Any], *, batch_size: int, sample_count: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    value = sampled.get("samples_zonal", sampled.get("samples"))
    samples = torch.as_tensor(value, device=device)
    expected = (batch_size, sample_count, HORIZON, CHANNELS)
    if samples.shape != expected:
        raise CapabilityEvidenceError(
            f"Distribution samples must be {expected}, got {tuple(samples.shape)}"
        )
    raw_weights = sampled.get("weights")
    weights = (
        samples.new_full((batch_size, sample_count), 1.0 / float(sample_count))
        if raw_weights is None
        else torch.as_tensor(raw_weights, device=device, dtype=samples.dtype)
    )
    if weights.shape != (batch_size, sample_count):
        raise CapabilityEvidenceError("Distribution weights have an invalid shape")
    if (
        not bool(torch.isfinite(samples).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
    ):
        raise CapabilityEvidenceError("Distribution samples/weights must be finite and nonnegative")
    mass = weights.sum(-1)
    if not bool(
        torch.allclose(mass, torch.ones_like(mass), rtol=0.0, atol=1.0e-6)
    ):
        raise CapabilityEvidenceError(
            "Distribution weights must already sum to one; formal gate does not repair them"
        )
    return samples, weights / mass[:, None]


_DEPLOYMENT_BATCH_KEYS = frozenset(
    {
        "market",
        "market_features",
        "market_inputs",
        "market_raw",
        "issue_workload_raw",
        "issue_pv_raw",
        "forecast_anchors",
        "forecast_anchor_metadata",
        "forecast_residual_caps",
        "forecast_residual_cap_metadata",
        "causal_da_anchor_available",
        "historical_market",
        "historical_market_available",
        "workload",
        "workload_features",
        "workload_inputs",
        "physical_static",
        "physical",
        "physical_features",
        "physical_dynamic",
        "state_sequence",
        "state",
        "state_features",
        "state_dt",
        "workload_dt",
        "day_of_week",
        "month",
        "holiday",
        "dst",
        "scenario_residual_caps_zonal",
        "scenario_residual_score_scales",
        "distribution_support",
        "distribution_support_lower_zonal",
        "distribution_support_upper_zonal",
        "distribution_support_contract_sha256",
        "distribution_training_contract_metadata",
        "policy_memory",
    }
)


def _deployment_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Strip every training-only/future-realization alias before causal audit."""

    result = {key: value for key, value in batch.items() if key in _DEPLOYMENT_BATCH_KEYS}
    forbidden = {
        "targets",
        "scenario_target_zonal",
        "rollout_targets",
        "realized",
        "realized_da",
        "realized_rt",
        "realized_workload",
        "realized_pv",
        "realized_pue",
    }
    if forbidden.intersection(result):
        raise CapabilityEvidenceError("deployment audit batch retained a future target alias")
    return result


def _perturb_future_state(
    batch: Mapping[str, Any], suffix: Tensor
) -> dict[str, Any]:
    """Perturb future rolling-state aliases so an unmasked read cannot pass."""

    result = dict(batch)
    for key in ("state_sequence", "state", "state_features"):
        value = result.get(key)
        if not isinstance(value, Tensor) or value.ndim < 3 or value.shape[1] != HORIZON:
            continue
        expansion = suffix
        while expansion.ndim < value.ndim:
            expansion = expansion.unsqueeze(-1)
        # Deterministic finite perturbation; a correctly causal model masks it.
        result[key] = torch.where(expansion, value + value.new_tensor(0.375), value)
    return result


def _declared_support_bounds(
    distribution: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    try:
        lower = torch.as_tensor(
            distribution["support_lower_zonal"], device=device, dtype=dtype
        )
        upper = torch.as_tensor(
            distribution["support_upper_zonal"], device=device, dtype=dtype
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CapabilityEvidenceError(
            "Distribution latent must declare support_lower_zonal/support_upper_zonal"
        ) from exc
    expected = (batch_size, HORIZON, CHANNELS)
    if (
        lower.shape != expected
        or upper.shape != expected
        or not bool(torch.isfinite(lower).all())
        or not bool(torch.isfinite(upper).all())
        or bool((lower > upper).any())
    ):
        raise CapabilityEvidenceError(
            f"Distribution declared support must be two finite {expected} tensors"
        )
    return lower, upper


def _support_violation(samples: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    below = (lower[:, None] - samples).clamp_min(0.0)
    above = (samples - upper[:, None]).clamp_min(0.0)
    return torch.maximum(below, above).flatten(1).amax(1)


def _project_to_canonical_support(
    values: Tensor, lower: Tensor, upper: Tensor
) -> Tensor:
    """Apply exactly the same free-variable support map to a baseline draw."""

    lower_n = lower[:, None]
    upper_n = upper[:, None]
    da = torch.clamp(values[..., 0:4], min=lower_n[..., 0:4], max=upper_n[..., 0:4])
    dart = torch.clamp(
        values[..., 8:12], min=lower_n[..., 8:12], max=upper_n[..., 8:12]
    )
    difference = da - dart
    increase = (lower_n[..., 4:8] - difference).clamp_min(0.0)
    da_increase = torch.minimum(
        increase, (upper_n[..., 0:4] - da).clamp_min(0.0)
    )
    da = da + da_increase
    dart = dart - torch.minimum(
        (increase - da_increase).clamp_min(0.0),
        (dart - lower_n[..., 8:12]).clamp_min(0.0),
    )
    difference = da - dart
    decrease = (difference - upper_n[..., 4:8]).clamp_min(0.0)
    da_decrease = torch.minimum(
        decrease, (da - lower_n[..., 0:4]).clamp_min(0.0)
    )
    da = da - da_decrease
    dart = dart + torch.minimum(
        (decrease - da_decrease).clamp_min(0.0),
        (upper_n[..., 8:12] - dart).clamp_min(0.0),
    )
    rt, dart = fp_safe_close_da_rt_dart(
        da,
        dart,
        rt_lower=lower_n[..., 4:8],
        rt_upper=upper_n[..., 4:8],
        dart_lower=lower_n[..., 8:12],
        dart_upper=upper_n[..., 8:12],
    )
    workload = torch.clamp(
        values[..., 12:13], min=lower_n[..., 12:13], max=upper_n[..., 12:13]
    )
    pv = torch.clamp(
        values[..., 13:17], min=lower_n[..., 13:17], max=upper_n[..., 13:17]
    )
    pue = torch.clamp(
        values[..., 17:21], min=lower_n[..., 17:21], max=upper_n[..., 17:21]
    )
    return torch.cat((da, rt, dart, workload, pv, pue), dim=-1)


def _weighted_quantiles(
    samples: Tensor, weights: Tensor, quantiles: tuple[float, ...]
) -> tuple[Tensor, ...]:
    flat = samples.flatten(2)
    ordered, indices = flat.sort(dim=1)
    ordered_weights = weights[:, :, None].expand_as(flat).gather(1, indices)
    cumulative = ordered_weights.cumsum(1)
    shape = samples.shape[:1] + samples.shape[2:]
    result: list[Tensor] = []
    for quantile in quantiles:
        position = (cumulative < float(quantile)).sum(1).clamp_max(samples.shape[1] - 1)
        result.append(ordered.gather(1, position[:, None]).squeeze(1).reshape(shape))
    return tuple(result)


def _energy_score(samples: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    flat = samples.flatten(2)
    observation = target.flatten(1)
    first = (weights * torch.linalg.vector_norm(flat - observation[:, None], dim=-1)).sum(-1)
    second = flat.new_zeros((flat.shape[0],))
    chunk = 32
    for start in range(0, flat.shape[1], chunk):
        distance = torch.cdist(flat[:, start : start + chunk], flat)
        second = second + (
            weights[:, start : start + chunk, None]
            * weights[:, None, :]
            * distance
        ).sum((1, 2))
    return first - 0.5 * second


def _marginal_crps(samples: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    flat = samples.flatten(2)
    observation = target.flatten(1)
    first = (weights[:, :, None] * (flat - observation[:, None]).abs()).sum(1)
    ordered, indices = flat.sort(dim=1)
    ordered_weights = weights[:, :, None].expand_as(flat).gather(1, indices)
    cumulative = ordered_weights.cumsum(1)
    second = (
        ordered_weights
        * ordered
        * (2.0 * cumulative - ordered_weights - 1.0)
    ).sum(1)
    return (first - second).mean(1)


def _variogram_score(samples: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    values: list[Tensor] = []
    for lag in (1, 4, 16):
        realized = (target[:, lag:] - target[:, :-lag]).abs().sqrt()
        generated = (samples[:, :, lag:] - samples[:, :, :-lag]).abs().sqrt()
        expected = (weights[:, :, None, None] * generated).sum(1)
        values.append((realized - expected).square().flatten(1).mean(1))
    return torch.stack(values, dim=1).mean(1)


def _interval_metrics(
    samples: Tensor, weights: Tensor, target: Tensor, level: float
) -> tuple[Tensor, Tensor, Tensor]:
    alpha = 1.0 - float(level)
    lower, upper = _weighted_quantiles(
        samples, weights, (alpha / 2.0, 1.0 - alpha / 2.0)
    )
    coverage = ((target >= lower) & (target <= upper)).float().flatten(1).mean(1)
    width = (upper - lower).flatten(1).mean(1)
    score = (
        upper
        - lower
        + (2.0 / alpha) * (lower - target).clamp_min(0.0)
        + (2.0 / alpha) * (target - upper).clamp_min(0.0)
    ).flatten(1).mean(1)
    return coverage, width, score


def _tail_score(samples: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    values: list[Tensor] = []
    for quantile in (0.05, 0.95):
        prediction = _weighted_quantiles(samples, weights, (quantile,))[0]
        residual = target - prediction
        values.append(
            torch.maximum(quantile * residual, (quantile - 1.0) * residual)
            .flatten(1)
            .mean(1)
        )
    return torch.stack(values, dim=1).mean(1)


def _average_pairwise_distance(samples: Tensor) -> Tensor:
    flat = samples.flatten(2)
    total = flat.new_zeros((flat.shape[0],))
    pairs = 0
    chunk = 32
    for start in range(0, flat.shape[1], chunk):
        current = flat[:, start : start + chunk]
        distance = torch.cdist(current, flat) / math.sqrt(max(flat.shape[-1], 1))
        row_indices = torch.arange(start, start + current.shape[1], device=flat.device)
        mask = row_indices[:, None] != torch.arange(flat.shape[1], device=flat.device)[None]
        total = total + distance[:, mask].sum(1)
        pairs += int(mask.sum())
    return total / float(max(pairs, 1))


def _metric_vectors(samples: Tensor, weights: Tensor, target: Tensor) -> dict[str, Tensor]:
    coverage_90, width_90, interval_90 = _interval_metrics(
        samples, weights, target, 0.90
    )
    coverage_95, width_95, interval_95 = _interval_metrics(
        samples, weights, target, 0.95
    )
    return {
        "energy_score": _energy_score(samples, weights, target),
        "marginal_crps": _marginal_crps(samples, weights, target),
        "variogram_score": _variogram_score(samples, weights, target),
        "tail_score": _tail_score(samples, weights, target),
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "interval_width_90": width_90,
        "interval_width_95": width_95,
        "interval_score_90": interval_90,
        "interval_score_95": interval_95,
        "effective_sample_size": weights.square().sum(-1).reciprocal(),
        "average_pairwise_sample_distance": _average_pairwise_distance(samples),
        "minimum_feature_std": samples.std(dim=1, unbiased=False).flatten(1).amin(1),
        "weight_entropy": -(weights * weights.clamp_min(1.0e-12).log()).sum(-1),
    }


def _identity_residual(samples: Tensor) -> Tensor:
    return (
        samples[..., 0:4] - samples[..., 4:8] - samples[..., 8:12]
    ).abs().flatten(1).amax(1)


def _tensor_values(value: Any) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, Mapping):
        result: list[Tensor] = []
        for item in value.values():
            result.extend(_tensor_values(item))
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = []
        for item in fields(value):
            result.extend(_tensor_values(getattr(value, item.name)))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_tensor_values(item))
        return result
    return []


def _distribution_mapping(output: Mapping[str, Any]) -> Mapping[str, Any]:
    value = output.get("distribution")
    if not isinstance(value, Mapping):
        raise CapabilityEvidenceError("model output lacks outputs['distribution']")
    if any(key in value for key in ("samples_zonal", "samples", "particles_zonal")):
        raise CapabilityEvidenceError(
            "deterministic outputs['distribution'] must not contain a fixed sample axis"
        )
    if value.get("fixed_particle_capacity") is True:
        raise CapabilityEvidenceError("Distribution latent declares fixed particle capacity")
    tensors = _tensor_values(value)
    if not tensors or not all(bool(torch.isfinite(item).all()) for item in tensors):
        raise CapabilityEvidenceError("Distribution parameters/latent are absent or non-finite")
    return value


def _center_zonal(output: Mapping[str, Any], distribution: Mapping[str, Any]) -> Tensor:
    for value in (
        distribution.get("center_zonal"),
        output.get("forecast_zonal"),
        (output.get("forecasts") or {}).get("zonal")
        if isinstance(output.get("forecasts"), Mapping)
        else None,
    ):
        if isinstance(value, Tensor):
            return value
    raise CapabilityEvidenceError("Distribution output lacks center_zonal")


def _max_tensor_difference(left: Any, right: Any) -> Tensor:
    left_values = _tensor_values(left)
    right_values = _tensor_values(right)
    if len(left_values) != len(right_values) or not left_values:
        raise CapabilityEvidenceError("causal-invariance tensor structures differ")
    reference = left_values[0]
    result = reference.new_zeros(())
    for first, second in zip(left_values, right_values, strict=True):
        if first.shape != second.shape:
            raise CapabilityEvidenceError("causal-invariance tensor shapes differ")
        result = torch.maximum(result, (first - second).abs().max())
    return result


def _policy_action_difference(
    first: Mapping[str, Any], second: Mapping[str, Any], batch_size: int, reference: Tensor
) -> Tensor:
    left = first.get("policy", first.get("actions"))
    right = second.get("policy", second.get("actions"))
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise CapabilityEvidenceError("Distribution causality audit requires policy/actions")
    result = reference.new_zeros((batch_size,))
    observed = 0
    for key in (
        "day_ahead",
        "battery",
        "battery_plan",
        "compute",
        "compute_plan",
        "migration",
        "fallback_confidence",
        "policy_confidence",
    ):
        if key not in left and key not in right:
            continue
        first_value = torch.as_tensor(left[key], device=reference.device)
        second_value = torch.as_tensor(right[key], device=reference.device)
        if first_value.shape != second_value.shape or first_value.shape[0] != batch_size:
            raise CapabilityEvidenceError(f"policy action {key!r} has an invalid shape")
        result = torch.maximum(
            result,
            (first_value - second_value).abs().reshape(batch_size, -1).amax(1),
        )
        observed += 1
    if not observed:
        raise CapabilityEvidenceError("Distribution causality audit found no policy action")
    return result


def _issued_day_ahead(output: Mapping[str, Any]) -> Tensor:
    policy = output.get("policy", output.get("actions"))
    if not isinstance(policy, Mapping) or not isinstance(policy.get("day_ahead"), Tensor):
        raise CapabilityEvidenceError("origin-zero policy did not issue day_ahead")
    return policy["day_ahead"].detach()


def _audit_causal_invariance(
    model: nn.Module,
    batch: Mapping[str, Any],
    target: Tensor,
    origins: Tensor,
    *,
    sample_count: int,
    sampling_seed: int,
) -> dict[str, Tensor]:
    batch_size = target.shape[0]
    periods = torch.arange(HORIZON, device=target.device)[None]
    zero_mask = periods.eq(0).expand(batch_size, -1)
    # Use the actual deployment allowlist, never the training batch.  This
    # removes targets/scenario_target/rollout realizations that a defective
    # model could otherwise read while the audit only perturbed ``revealed``.
    deployment = _deployment_batch(batch)
    origin_zero = dict(deployment)
    origin_zero["origin"] = torch.zeros(batch_size, dtype=torch.long, device=target.device)
    origin_zero["revealed"] = {"zonal": target, "mask": zero_mask}
    issued_da = _issued_day_ahead(model(origin_zero))

    mask = periods <= origins[:, None]
    legal = dict(deployment)
    legal["origin"] = origins
    legal["revealed"] = {"zonal": target, "mask": mask}
    legal["issued_da_schedule"] = issued_da
    first = model(legal)
    first_distribution = _distribution_mapping(first)
    first_sampled = sample_distribution(
        model,
        legal,
        sample_count,
        generator=_make_generator(target, sampling_seed),
    )
    first_samples, _ = _samples_and_weights(
        first_sampled,
        batch_size=batch_size,
        sample_count=sample_count,
        device=target.device,
    )
    prefix_mask = mask[:, None, :, None]
    prefix_identity = (
        (first_samples - target[:, None])
        .abs()
        .masked_fill(~prefix_mask, 0.0)
        .flatten(1)
        .amax(1)
    )

    alternative = target.clone()
    suffix = ~mask
    zone_shift = target.new_tensor((17.0, 29.0, 43.0, 61.0))[None, None]
    alternative[..., 0:4] = torch.where(
        suffix[..., None], alternative[..., 0:4] + zone_shift, alternative[..., 0:4]
    )
    alternative[..., 8:12] = torch.where(
        suffix[..., None], alternative[..., 8:12] - 0.5 * zone_shift, alternative[..., 8:12]
    )
    # Re-close only the perturbed suffix.  Recomputing the already-revealed
    # prefix in float32 changes otherwise identical RT values by one or two
    # ULPs and creates a false leakage signal under the bitwise prefix audit.
    alternative[..., 4:8] = torch.where(
        suffix[..., None],
        alternative[..., 0:4] - alternative[..., 8:12],
        alternative[..., 4:8],
    )
    positive_shift = target.new_tensor(
        (3.0, 0.7, 1.1, 1.7, 2.3, 0.03, 0.04, 0.05, 0.06)
    )[None, None]
    alternative[..., 12:21] = torch.where(
        suffix[..., None],
        (alternative[..., 12:21] + positive_shift).clamp_min(1.0e-6),
        alternative[..., 12:21],
    )
    # Keep every legal issue-time input fixed, alter the revealed future
    # coherently, and also alter any future state-sequence suffix.  Thus a
    # hidden unmasked future-state read cannot pass simply because a second
    # target alias happened to remain unchanged.
    perturbed = _perturb_future_state(deployment, suffix)
    perturbed["origin"] = origins
    perturbed["revealed"] = {"zonal": alternative, "mask": mask}
    perturbed["issued_da_schedule"] = issued_da
    second = model(perturbed)
    second_distribution = _distribution_mapping(second)
    second_sampled = sample_distribution(
        model,
        perturbed,
        sample_count,
        generator=_make_generator(target, sampling_seed),
    )
    second_samples, _ = _samples_and_weights(
        second_sampled,
        batch_size=batch_size,
        sample_count=sample_count,
        device=target.device,
    )
    prefix_sample_invariance = (
        (first_samples - second_samples)
        .abs()
        .masked_fill(~prefix_mask, 0.0)
        .flatten(1)
        .amax(1)
    )
    complete_sample_invariance = (first_samples - second_samples).abs().flatten(1).amax(1)
    latent_scalar = _max_tensor_difference(first_distribution, second_distribution)
    return {
        "revealed_prefix_identity": prefix_identity,
        "future_action_invariance": _policy_action_difference(
            first, second, batch_size, target
        ),
        "future_distribution_latent_invariance": latent_scalar.expand(batch_size),
        "future_prefix_sample_invariance": prefix_sample_invariance,
        "future_complete_sample_invariance": complete_sample_invariance,
    }


class DistributionCapabilityMaterializer:
    """Trainer-compatible Level-2 evaluator for one immutable V6 candidate."""

    def __init__(
        self,
        config: Mapping[str, Any],
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        corpus: PreloadedR4Corpus | None = None,
        training_contract: ScenarioTrainingContract,
        distribution_support_contract: (
            str | Path | Mapping[str, Any] | DistributionSupportContract | None
        ) = None,
        distribution_training_contract_path: str | Path | None = None,
        distribution_training_contract_sha256: str | None = None,
        sampling_contract_path: str | Path,
        sampling_contract_sha256: str | None = None,
        batch_size: int = 4,
        audit_origin: int = 47,
        seed: int = 20240813,
    ) -> None:
        self.config = config
        self.repository_root = Path(repository_root).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.corpus = corpus or PreloadedR4Corpus(config, self.repository_root)
        # V5 supplies only immutable train-residual lineage.  Active V6 values
        # come from a distinct support artifact under the independent root.
        self.residual_score_contract = training_contract
        self.training_contract = training_contract  # narrow caller compatibility
        if distribution_support_contract is None:
            lineage = config.get("lineage", {})
            raw_support = (
                lineage.get("distribution_support_artifact")
                if isinstance(lineage, Mapping)
                else None
            )
            if raw_support is None:
                raise CapabilityEvidenceError(
                    "V6 config lacks lineage.distribution_support_artifact"
                )
            support_path = Path(str(raw_support))
            distribution_support_contract = (
                support_path
                if support_path.is_absolute()
                else self.repository_root / support_path
            )
        self.support_contract = load_distribution_support_contract(
            distribution_support_contract,
            expected_source_residual_sha256=training_contract.artifact_sha256,
        )
        source_caps = training_contract.raw_p99_caps.to(dtype=torch.float32).clone()
        source_caps[4:8] = source_caps[0:4] + source_caps[8:12]
        if not torch.equal(
            self.support_contract.score_scales,
            training_contract.score_scales.to(dtype=torch.float32),
        ) or not torch.equal(self.support_contract.raw_p99_caps, source_caps):
            raise CapabilityEvidenceError(
                "V6 support scales/caps differ from their frozen train-residual source"
            )
        self.batch_size = int(batch_size)
        self.audit_origin = int(audit_origin)
        self.seed = int(seed)
        self.sampling_contract_path = Path(sampling_contract_path).resolve()
        if self.batch_size < 1:
            raise ValueError("Distribution capability batch_size must be positive")
        if not 0 <= self.audit_origin < HORIZON:
            raise ValueError("Distribution audit origin must lie in [0,95]")
        if not self.sampling_contract_path.is_file():
            raise CapabilityEvidenceError("preflight-frozen Distribution sampling contract is missing")
        self.sampling_contract = read_json(self.sampling_contract_path)
        if (
            self.sampling_contract.get("schema") != SAMPLING_CONTRACT_SCHEMA
            or self.sampling_contract.get("status") != "FROZEN_BEFORE_TRAINING"
            or int(self.sampling_contract.get("formal_probability_samples", -1))
            != FORMAL_PROBABILITY_SAMPLES
            or int(self.sampling_contract.get("formal_empirical_risk_samples", -1))
            != FORMAL_EMPIRICAL_RISK_SAMPLES
            or self.sampling_contract.get("formal_selection_2024_opened") is not False
            or self.sampling_contract.get("locked_test_2025_opened") is not False
        ):
            raise CapabilityEvidenceError("invalid V6 formal sampling contract")
        if sampling_contract_sha256 is None:
            self.sampling_contract_sha256 = sha256_file(
                self.sampling_contract_path
            )
        else:
            self.sampling_contract_sha256 = str(
                sampling_contract_sha256
            ).lower()
            if not _is_sha256(self.sampling_contract_sha256):
                raise CapabilityEvidenceError(
                    "V6 Distribution sampling contract SHA-256 is malformed"
                )
        raw_distribution_contract = (
            self.artifact_root / "data" / "distribution_training_contract.json"
            if distribution_training_contract_path is None
            else Path(distribution_training_contract_path)
        )
        self.distribution_training_contract_path = Path(
            raw_distribution_contract
        ).resolve()
        self.require_checkpoint_distribution_contract_binding = (
            True
        )
        if not self.distribution_training_contract_path.is_file():
            raise CapabilityEvidenceError("frozen V6 Distribution training contract is missing")
        self.distribution_training_contract_sha256 = (
            sha256_file(self.distribution_training_contract_path)
            if distribution_training_contract_sha256 is None
            else str(distribution_training_contract_sha256).lower()
        )
        if not _is_sha256(self.distribution_training_contract_sha256):
            raise CapabilityEvidenceError("V6 Distribution training contract SHA-256 is malformed")
        # Explicit digests are candidate-registry bindings.  Their content was
        # hashed once at preflight/candidate creation; do not repeatedly hash a
        # large dependency graph here.
        if distribution_training_contract_sha256 is None:
            pass
        self._baseline_cache: tuple[dict[str, Any], Path, str] | None = None

    @property
    def capability_root(self) -> Path:
        return self.artifact_root / "capability" / "distribution"

    def _baseline(self) -> tuple[dict[str, Any], Path, str]:
        if self._baseline_cache is not None:
            return self._baseline_cache
        train = tuple(self.corpus.records("train"))
        validation = tuple(self.corpus.records("validation"))
        train_dates, _ = _validate_split_dates(train, validation)
        signature = _corpus_signature(train, train_dates)
        path = self.capability_root / "frozen_training_residual_baseline_n256.pt"
        if path.exists():
            try:
                value = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                value = torch.load(path, map_location="cpu")
            if not isinstance(value, dict):
                raise CapabilityEvidenceError("frozen Distribution baseline is malformed")
            if (
                value.get("schema") != BASELINE_SCHEMA
                or value.get("corpus_sha256") != signature
                or int(value.get("formal_probability_samples", -1))
                != FORMAL_PROBABILITY_SAMPLES
                or value.get("sampling_contract_sha256")
                != self.sampling_contract_sha256
                or value.get("distribution_support_contract_sha256")
                != self.support_contract.artifact_sha256
                or value.get("source_residual_score_contract_sha256")
                != self.residual_score_contract.artifact_sha256
            ):
                raise CapabilityEvidenceError(
                    "frozen Distribution baseline does not match the V6 contract"
                )
            if tuple(value.get("train_dates", ())) != train_dates:
                raise CapabilityEvidenceError("frozen Distribution baseline date lineage changed")
        else:
            targets = torch.stack([_target(record) for record in train])
            pue_center = targets[..., 17:21].median(dim=0).values
            centers = torch.stack([_issue_center(record, pue_center) for record in train])
            residuals = targets - centers
            scale = residuals.abs().flatten(0, 1).median(dim=0).values
            scale = torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))
            value = {
                "schema": BASELINE_SCHEMA,
                "name": BASELINE_NAME,
                "fit_split": "2022-01-01_to_2023-06-30_only",
                "train_dates": train_dates,
                "corpus_sha256": signature,
                "formal_probability_samples": FORMAL_PROBABILITY_SAMPLES,
                "sampling_contract_sha256": self.sampling_contract_sha256,
                "distribution_support_contract_sha256": (
                    self.support_contract.artifact_sha256
                ),
                "source_residual_score_contract_sha256": (
                    self.residual_score_contract.artifact_sha256
                ),
                "sampling_seed": self.seed,
                "trajectory_sampling": "whole_day_joint_residual_without_replacement",
                "normalization": "train_residual_channel_median_absolute_scale",
                "pue_center": pue_center,
                "residuals": residuals,
                "scale": scale,
            }
            atomic_torch_save(path, value)
        residuals = torch.as_tensor(value.get("residuals"))
        scale = torch.as_tensor(value.get("scale"))
        pue_center = torch.as_tensor(value.get("pue_center"))
        if residuals.shape != (543, HORIZON, CHANNELS):
            raise CapabilityEvidenceError("frozen Distribution residual pool has an invalid shape")
        if scale.shape != (CHANNELS,) or bool((scale <= 0).any()) or not bool(
            torch.isfinite(scale).all()
        ):
            raise CapabilityEvidenceError("frozen Distribution metric scale is invalid")
        if pue_center.shape != (HORIZON, 4):
            raise CapabilityEvidenceError("frozen Distribution PUE center is invalid")
        self._baseline_cache = (value, path, sha256_file(path))
        return self._baseline_cache

    def _registered_seed(self, dates: Sequence[str], checkpoint_sha: str, purpose: str) -> int:
        digest = sha256_json(
            {
                "schema": SCHEMA,
                "seed": self.seed,
                "dates": tuple(dates),
                "checkpoint_sha256": checkpoint_sha,
                "purpose": purpose,
                "sampling_contract_sha256": self.sampling_contract_sha256,
            }
        )
        return int(digest[:16], 16) & ((1 << 63) - 1)

    def _baseline_indices(self, operating_date: str, baseline_sha: str) -> Tensor:
        seed = self._registered_seed((operating_date,), baseline_sha, "baseline_residual_indices")
        return torch.randperm(543, generator=torch.Generator().manual_seed(seed))[
            :FORMAL_PROBABILITY_SAMPLES
        ]

    def _materialize(
        self,
        model: nn.Module,
        checkpoint: Path,
        checkpoint_sha: str,
        epoch: int,
        baseline: Mapping[str, Any],
        baseline_path: Path,
        baseline_sha: str,
    ) -> dict[str, Any]:
        train = tuple(self.corpus.records("train"))
        validation = tuple(self.corpus.records("validation"))
        _, validation_dates = _validate_split_dates(train, validation)
        residuals = torch.as_tensor(baseline["residuals"], dtype=torch.float32)
        scale_cpu = self.support_contract.score_scales.to(dtype=torch.float32)
        pue_center = torch.as_tensor(baseline["pue_center"], dtype=torch.float32)
        if not torch.equal(
            scale_cpu,
            torch.as_tensor(baseline["scale"], dtype=torch.float32),
        ):
            raise CapabilityEvidenceError(
                "Distribution gate score scale differs from the train-only contract"
            )
        device = _model_device(model)
        was_training = model.training
        model.eval()
        rows: list[dict[str, Any]] = []
        rng_reproducible: bool | None = None
        try:
            with torch.inference_mode():
                for start in range(0, len(validation), self.batch_size):
                    records = validation[start : start + self.batch_size]
                    dates = validation_dates[start : start + self.batch_size]
                    batch = move_batch(_collate(records), device)
                    batch["origin"] = torch.zeros(len(records), dtype=torch.long, device=device)
                    batch = attach_distribution_support_contract(
                        batch, self.support_contract, device=device
                    )
                    output = model(batch)
                    if not isinstance(output, Mapping):
                        raise CapabilityEvidenceError("Distribution model output must be a mapping")
                    distribution = _distribution_mapping(output)
                    center = _center_zonal(output, distribution)
                    sampling_seed = self._registered_seed(dates, checkpoint_sha, "learned_probability")
                    sampled = sample_distribution(
                        model,
                        batch,
                        FORMAL_PROBABILITY_SAMPLES,
                        generator=_make_generator(center, sampling_seed),
                    )
                    samples, weights = _samples_and_weights(
                        sampled,
                        batch_size=len(records),
                        sample_count=FORMAL_PROBABILITY_SAMPLES,
                        device=device,
                    )
                    support_lower, support_upper = _declared_support_bounds(
                        distribution,
                        batch_size=len(records),
                        device=device,
                        dtype=samples.dtype,
                    )
                    if rng_reproducible is None:
                        repeated = sample_distribution(
                            model,
                            batch,
                            FORMAL_PROBABILITY_SAMPLES,
                            generator=_make_generator(center, sampling_seed),
                        )
                        repeated_samples, repeated_weights = _samples_and_weights(
                            repeated,
                            batch_size=len(records),
                            sample_count=FORMAL_PROBABILITY_SAMPLES,
                            device=device,
                        )
                        rng_reproducible = bool(
                            torch.equal(samples, repeated_samples)
                            and torch.equal(weights, repeated_weights)
                        )
                    target = torch.as_tensor(
                        batch["scenario_target_zonal"], device=device, dtype=samples.dtype
                    )
                    scale = scale_cpu.to(device=device, dtype=samples.dtype)
                    normalized_samples = samples / scale
                    normalized_target = target / scale
                    learned_metrics = _metric_vectors(
                        normalized_samples, weights, normalized_target
                    )
                    learned_identity = _identity_residual(samples)
                    caps = torch.as_tensor(
                        batch["scenario_residual_caps_zonal"],
                        device=device,
                        dtype=samples.dtype,
                    )
                    if caps.shape == (HORIZON, CHANNELS):
                        caps = caps[None].expand(len(records), -1, -1)
                    if caps.shape != (len(records), HORIZON, CHANNELS):
                        raise CapabilityEvidenceError("Distribution support caps have an invalid shape")
                    residual_cap_excess = (
                        (samples - center[:, None]).abs() - caps[:, None]
                    ).clamp_min(0.0).flatten(1).amax(1)
                    support_excess = _support_violation(
                        samples, support_lower, support_upper
                    )
                    positivity_violation = torch.stack(
                        (
                            (-samples[..., 12]).clamp_min(0.0).flatten(1).amax(1),
                            (-samples[..., 13:17]).clamp_min(0.0).flatten(1).amax(1),
                            (1.0e-4 - samples[..., 17:21]).clamp_min(0.0).flatten(1).amax(1),
                        ),
                        dim=1,
                    ).amax(1)
                    audit_origins = (
                        torch.arange(len(records), device=device, dtype=torch.long)
                        + len(rows)
                        + self.audit_origin
                    ).remainder(HORIZON)
                    audit_seed = self._registered_seed(dates, checkpoint_sha, "causal_invariance")
                    causal = _audit_causal_invariance(
                        model,
                        batch,
                        target,
                        audit_origins,
                        sample_count=FORMAL_PROBABILITY_SAMPLES,
                        sampling_seed=audit_seed,
                    )
                    baseline_samples: list[Tensor] = []
                    for record, operating_date in zip(records, dates, strict=True):
                        center_baseline = _issue_center(record, pue_center)
                        indices = self._baseline_indices(operating_date, baseline_sha)
                        current = center_baseline[None] + residuals.index_select(0, indices)
                        baseline_samples.append(current)
                    baseline_tensor = torch.stack(baseline_samples).to(
                        device=device, dtype=samples.dtype
                    )
                    # Fair paired comparison: baseline and learned draws use
                    # the same day's canonical absolute support and structural
                    # RT=DA-DART map.  The V5 N=20 artifact is never involved.
                    baseline_tensor = _project_to_canonical_support(
                        baseline_tensor, support_lower, support_upper
                    )
                    baseline_support_excess = _support_violation(
                        baseline_tensor, support_lower, support_upper
                    )
                    baseline_identity = _identity_residual(baseline_tensor)
                    baseline_weights = samples.new_full(
                        (len(records), FORMAL_PROBABILITY_SAMPLES),
                        1.0 / FORMAL_PROBABILITY_SAMPLES,
                    )
                    baseline_metrics = _metric_vectors(
                        baseline_tensor / scale, baseline_weights, normalized_target
                    )
                    parameters_finite = all(
                        bool(torch.isfinite(item).all())
                        for item in _tensor_values(distribution)
                    )
                    for local, operating_date in enumerate(dates):
                        row: dict[str, Any] = {
                            "date": operating_date,
                            "checkpoint_sha256": checkpoint_sha,
                            "baseline_name": BASELINE_NAME,
                            "baseline_sha256": baseline_sha,
                            "distribution_training_contract_sha256": (
                                self.distribution_training_contract_sha256
                            ),
                            "distribution_support_contract_sha256": (
                                self.support_contract.artifact_sha256
                            ),
                            "formal_sampling_contract_sha256": self.sampling_contract_sha256,
                            "formal_probability_sample_count": FORMAL_PROBABILITY_SAMPLES,
                            "formal_empirical_risk_sample_count": FORMAL_EMPIRICAL_RISK_SAMPLES,
                            "empirical_risk_replay_status": "NOT_RUN_DISTRIBUTION_STAGE",
                            "fixed_rng_reproducible": bool(rng_reproducible),
                            "finite_distribution_parameters": parameters_finite,
                            "finite_joint_samples": bool(torch.isfinite(samples[local]).all()),
                            "support_bounds_passed": bool(
                                float(support_excess[local]) <= 1.0e-6
                                and float(residual_cap_excess[local]) <= 1.0e-6
                                and float(positivity_violation[local]) <= 1.0e-6
                            ),
                            "baseline_support_contract_same": True,
                            "baseline_support_excess_max_abs": float(
                                baseline_support_excess[local]
                            ),
                            "baseline_da_rt_dart_identity_max_abs": float(
                                baseline_identity[local]
                            ),
                            "sample_diversity_passed": bool(
                                float(learned_metrics["average_pairwise_sample_distance"][local])
                                > 1.0e-6
                                and float(learned_metrics["minimum_feature_std"][local]) > 1.0e-6
                            ),
                            "revealed_prefix_identity_passed": bool(
                                float(causal["revealed_prefix_identity"][local]) <= 1.0e-6
                            ),
                            "future_leakage_audit_origin": int(audit_origins[local]),
                            "future_leakage_audit_status": (
                                "PASSED"
                                if max(
                                    float(causal["future_action_invariance"][local]),
                                    float(causal["future_distribution_latent_invariance"][local]),
                                    float(causal["future_prefix_sample_invariance"][local]),
                                    float(causal["future_complete_sample_invariance"][local]),
                                )
                                <= 1.0e-6
                                else "FAILED"
                            ),
                            "future_action_invariance_max_abs": float(
                                causal["future_action_invariance"][local]
                            ),
                            "future_distribution_latent_invariance_max_abs": float(
                                causal["future_distribution_latent_invariance"][local]
                            ),
                            "future_revealed_prefix_sample_invariance_max_abs": float(
                                causal["future_prefix_sample_invariance"][local]
                            ),
                            "future_complete_distribution_sample_invariance_max_abs": float(
                                causal["future_complete_sample_invariance"][local]
                            ),
                            "model_revealed_prefix_identity_max_abs": float(
                                causal["revealed_prefix_identity"][local]
                            ),
                            "model_da_rt_dart_identity_max_abs": float(learned_identity[local]),
                            "model_support_excess_max_abs": float(support_excess[local]),
                            "model_residual_cap_excess_max_abs": float(
                                residual_cap_excess[local]
                            ),
                            "model_positive_support_violation_max_abs": float(
                                positivity_violation[local]
                            ),
                            "formal_selection_2024_opened": False,
                            "locked_test_2025_opened": False,
                            "active_pyomo_loaded": False,
                            "active_gurobipy_loaded": False,
                        }
                        for name, values in learned_metrics.items():
                            row[f"model_{name}"] = float(values[local])
                        for name in (
                            "energy_score",
                            "marginal_crps",
                            "variogram_score",
                            "tail_score",
                            "interval_score_90",
                            "interval_score_95",
                        ):
                            row[f"baseline_{name}"] = float(baseline_metrics[name][local])
                        row["evidence_sha256"] = sha256_json(row)
                        rows.append(row)
        finally:
            model.train(was_training)
        decision = evaluate_distribution(rows, seed=self.seed)
        return {
            "schema": SCHEMA,
            "phase": "distribution",
            "epoch": int(epoch),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "baseline": str(baseline_path),
            "baseline_sha256": baseline_sha,
            "distribution_training_contract": str(
                self.distribution_training_contract_path
            ),
            "distribution_training_contract_sha256": (
                self.distribution_training_contract_sha256
            ),
            "distribution_support_contract": str(self.support_contract.path),
            "distribution_support_contract_sha256": (
                self.support_contract.artifact_sha256
            ),
            "formal_sampling_contract": str(self.sampling_contract_path),
            "formal_sampling_contract_sha256": self.sampling_contract_sha256,
            "validation_window": "2023-H2",
            "validation_days": len(rows),
            "formal_probability_sample_count": FORMAL_PROBABILITY_SAMPLES,
            "formal_empirical_risk_sample_count": FORMAL_EMPIRICAL_RISK_SAMPLES,
            "empirical_risk_replay_status": "NOT_RUN_DISTRIBUTION_STAGE",
            "metric_space": "train_residual_channel_median_absolute_scaled_joint_space",
            "baseline_fairness": "same_N256_same_dates_same_metric_definitions_registered_rng",
            "audit_origin_schedule": "cyclic_all_96_origins_across_179_dates",
            "future_leakage_audit_status": (
                "PASSED"
                if len(rows) == EXPECTED_VALIDATION_DAYS
                and {int(row["future_leakage_audit_origin"]) for row in rows}
                == set(range(HORIZON))
                and all(row["future_leakage_audit_status"] == "PASSED" for row in rows)
                else "FAILED"
            ),
            "formal_selection_2024_opened": False,
            "locked_test_2025_opened": False,
            "pyomo_loaded": False,
            "gurobipy_loaded": False,
            "external_solver": False,
            "daily_rows": rows,
            "gate": decision.as_evaluator_payload(),
        }

    def __call__(
        self,
        *,
        phase: str,
        horizon: int | None,
        checkpoint: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        epoch: int | None = None,
        model: nn.Module | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        _assert_solver_free_active_imports()
        if str(phase).lower() != "distribution" or horizon not in (None, HORIZON):
            raise CapabilityEvidenceError(
                "Distribution materializer requires phase=distribution and horizon None/K=96"
            )
        if model is None:
            raise CapabilityEvidenceError(
                "Distribution materializer requires the reloaded candidate model"
            )
        supplied = checkpoint_path if checkpoint_path is not None else checkpoint
        if supplied is None:
            raise CapabilityEvidenceError(
                "Distribution materializer requires a candidate checkpoint"
            )
        source = Path(supplied).resolve()
        digest, payload = _load_checkpoint_binding(
            source, model, epoch, checkpoint_sha256
        )
        payload_contract_path = payload.get("distribution_training_contract_path")
        payload_contract_sha = str(
            payload.get("distribution_training_contract_sha256", "")
        ).lower()
        if self.require_checkpoint_distribution_contract_binding:
            if (
                Path(str(payload_contract_path)).resolve()
                != self.distribution_training_contract_path
                or payload_contract_sha != self.distribution_training_contract_sha256
            ):
                raise CapabilityEvidenceError(
                    "candidate checkpoint is bound to a different Distribution training contract"
                )
        completed = int(payload.get("epochs_completed", int(payload.get("epoch", -1)) + 1))
        baseline, baseline_path, baseline_sha = self._baseline()
        destination = (
            self.capability_root / "checkpoints" / digest / "daily_evidence.json"
        )
        binding_path = destination.with_suffix(".sha256.json")
        if destination.exists():
            if not binding_path.is_file():
                raise CapabilityEvidenceError("cached Distribution evidence lacks its binding")
            binding = read_json(binding_path)
            stat = destination.stat()
            if (
                binding.get("checkpoint_sha256") != digest
                or binding.get("evidence_path") != str(destination.resolve())
                or int(binding.get("evidence_size_bytes", -1)) != stat.st_size
                or int(binding.get("evidence_mtime_ns", -1)) != stat.st_mtime_ns
                or not _is_sha256(binding.get("evidence_sha256"))
            ):
                raise CapabilityEvidenceError("cached Distribution evidence binding is invalid")
            materialized = read_json(destination)
            if (
                materialized.get("schema") != SCHEMA
                or materialized.get("checkpoint_sha256") != digest
                or materialized.get("baseline_sha256") != baseline_sha
                or materialized.get("formal_sampling_contract_sha256")
                != self.sampling_contract_sha256
                or materialized.get("distribution_training_contract_sha256")
                != self.distribution_training_contract_sha256
                or materialized.get("distribution_support_contract_sha256")
                != self.support_contract.artifact_sha256
            ):
                raise CapabilityEvidenceError("persisted Distribution evidence binding is invalid")
            rows = materialized.get("daily_rows")
            if not isinstance(rows, list):
                raise CapabilityEvidenceError("persisted Distribution daily evidence is malformed")
            result = evaluate_distribution(rows, seed=self.seed).as_evaluator_payload()
            evidence_sha = str(binding["evidence_sha256"])
        else:
            materialized = self._materialize(
                model,
                source,
                digest,
                completed,
                baseline,
                baseline_path,
                baseline_sha,
            )
            atomic_json(destination, materialized)
            evidence_sha = sha256_file(destination)
            stat = destination.stat()
            atomic_json(
                binding_path,
                {
                    "schema": "dcvpp.r4.v6.distribution-evidence-binding.v1",
                    "checkpoint_sha256": digest,
                    "evidence_path": str(destination.resolve()),
                    "evidence_sha256": evidence_sha,
                    "evidence_size_bytes": stat.st_size,
                    "evidence_mtime_ns": stat.st_mtime_ns,
                },
            )
            result = dict(materialized["gate"])
        result.update(
            {
                "evidence_path": str(destination),
                "evidence_sha256": evidence_sha,
                "checkpoint_sha256": digest,
                "baseline_sha256": baseline_sha,
                "formal_sampling_contract_sha256": self.sampling_contract_sha256,
                "distribution_training_contract_sha256": (
                    self.distribution_training_contract_sha256
                ),
                "distribution_support_contract_sha256": (
                    self.support_contract.artifact_sha256
                ),
                "formal_probability_sample_count": FORMAL_PROBABILITY_SAMPLES,
                "formal_empirical_risk_sample_count": FORMAL_EMPIRICAL_RISK_SAMPLES,
            }
        )
        return result


__all__ = [
    "BASELINE_NAME",
    "BASELINE_SCHEMA",
    "DistributionCapabilityMaterializer",
    "FORMAL_EMPIRICAL_RISK_SAMPLES",
    "FORMAL_PROBABILITY_SAMPLES",
    "SAMPLING_CONTRACT_SCHEMA",
    "SCHEMA",
    "sample_distribution",
]
