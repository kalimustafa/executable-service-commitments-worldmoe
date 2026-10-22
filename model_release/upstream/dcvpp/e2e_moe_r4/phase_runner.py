"""Default executable phase adapter for the independent formal R4 lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .artifact_io import atomic_json, read_json
from .capability_evaluation import build_capability_evaluator
from .capability_materialization import LiveRepresentationCapabilityEvaluator
from .config import load_r4_config
from .decision_capability_materialization import LiveDecisionCapabilityEvaluator
from .distribution_capability_materialization import (
    DistributionCapabilityMaterializer,
    DistributionSupportContract,
    SAMPLING_CONTRACT_SCHEMA,
)
from .formal_protocol import ProtocolContext, register_phase_runner
from .hashing import sha256_file
from .physical_twin import build_formal_r4_model
from .scenario_capability_materialization import ScenarioCapabilityMaterializer
from .scenario_training_contract import (
    ScenarioTrainingContract,
    ScenarioTrainingContractError,
    attach_scenario_training_contract,
    load_scenario_training_contract,
    materialize_scenario_training_contract,
)
from .trainer import (
    AtomicRunStore,
    DISTRIBUTIONAL_PHASE_ORDER,
    PHYSICAL_BATCH_FALLBACK,
    CheapValidationPolicy,
    DistributionCurriculumContract,
    R4PhaseTrainer,
    R4ProtocolError,
    build_phase_loss,
    distribution_curriculum_contract,
    phase_spec,
)
from .training_data import PreloadedR4Corpus


def _phase_directory(context: ProtocolContext, phase: str, seed: int) -> Path:
    return Path(context.output_root) / f"seed={int(seed)}" / f"phase={phase}"


def _predecessor(
    context: ProtocolContext,
    phase: str,
    seed: int,
    *,
    config: Mapping[str, Any],
    normalization_sha256: str,
) -> Path | None:
    architecture = str(config.get("project", {}).get("architecture_version", ""))
    is_distributional = architecture == "continuous_distributional_joint_moe_r4_v6"
    order = (
        DISTRIBUTIONAL_PHASE_ORDER
        if is_distributional
        else ("representation", "scenario", "policy", "a1", "a2")
    )
    index = order.index(phase)
    if index == 0:
        return None
    if is_distributional and phase == "distribution":
        lineage = config.get("lineage")
        source = (
            lineage.get("representation_selected_checkpoint")
            if isinstance(lineage, Mapping)
            else None
        )
        if not source:
            raise R4ProtocolError(
                "V6 Distribution requires lineage.representation_selected_checkpoint"
            )
        path = Path(str(source))
        return (
            path.resolve()
            if path.is_absolute()
            else (Path(context.repository_root) / path).resolve()
        )
    previous = order[index - 1]
    previous_dir = _phase_directory(context, previous, seed)
    store = AtomicRunStore(
        previous_dir,
        lineage_id=context.lineage_id,
        config_sha256=context.config_sha256,
        config_path=context.config_path,
    )
    # This strict load rejects latest/provisional/candidate files and verifies
    # the immutable source hash plus the capability PASS envelope.
    payload = store.load_selected_checkpoint(previous, seed, map_location="cpu")
    if payload.get("normalization_sha256") != normalization_sha256:
        raise R4ProtocolError(
            "selected predecessor normalization hash differs from the current "
            "train-only artifact"
        )
    return store.selected_best_path(previous, seed)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise R4ProtocolError("formal R4 training requires the verified CUDA device")
    return torch.device("cuda:0")


def _load_predecessor(
    model: torch.nn.Module,
    path: Path | None,
    *,
    normalization_sha256: str,
) -> None:
    if path is None:
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise R4ProtocolError(f"predecessor checkpoint has no model_state: {path}")
    if payload.get("normalization_sha256") != normalization_sha256:
        raise R4ProtocolError("predecessor normalization hash mismatch")
    model.load_state_dict(state, strict=True)


_REPRESENTATION_TRUNK_PREFIXES = (
    "market_encoder.",
    "workload_encoder.",
    "physical_stream.static_adapter.",
    "physical_stream.dynamic_liquid.",
    "physical_stream.dynamic_gate",
    "state_encoder.",
    "cross_fusion.",
    "moe.",
    "forecast_decoder.",
)
_LEGACY_HEAD_PREFIXES = ("scenario_generator.", "risk.", "policy.")


@dataclass(frozen=True)
class V6DistributionSupportContract:
    """Stable adapter over the frozen V6 train-only support artifact.

    The artifact cannot embed its own file digest without a self-reference.
    Consumers therefore use ``artifact_sha256`` from this adapter (and from
    the enclosing Distribution training contract) while all scientific
    content remains directly available under stable field names.
    """

    path: Path
    artifact_sha256: str
    source_residual_score_contract_path: Path
    source_residual_score_contract_sha256: str
    score_scales: tuple[float, ...]
    residual_caps: tuple[tuple[float, ...], ...]
    lower_zonal: tuple[tuple[float, ...], ...]
    upper_zonal: tuple[tuple[float, ...], ...]
    fit_split: str = "train"
    formal_data_opened: bool = False

    @property
    def raw_p99_caps(self) -> tuple[float, ...]:
        """Legacy contract-compatible 21-channel cap view."""

        return self.residual_caps[0]

    @property
    def scenario_residual_score_scales(self) -> tuple[float, ...]:
        return self.score_scales

    def tensor_payload(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.lower_zonal, dtype=torch.float32, device=device),
            torch.tensor(self.upper_zonal, dtype=torch.float32, device=device),
            torch.tensor(self.residual_caps, dtype=torch.float32, device=device),
        )


def _load_v6_representation_trunk(
    model: torch.nn.Module,
    source: Path,
    *,
    seed: int,
    normalization_sha256: str,
    manifest_path: Path,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Import only the epoch-28 capability-passed Representation trunk."""

    if not source.is_file():
        raise R4ProtocolError(f"V6 Representation predecessor is missing: {source}")
    if source_sha256 is None:
        source_sha256 = sha256_file(source)  # one preflight read, never per epoch
    else:
        source_sha256 = str(source_sha256).lower()
        if len(source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_sha256
        ):
            raise R4ProtocolError("V6 Representation predecessor digest is malformed")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise R4ProtocolError("V6 Representation predecessor has no model_state")
    required = {
        "phase": "representation",
        "seed": int(seed),
        "selected_epoch": 28,
        "checkpoint_role": "capability_passed_selected_best",
        "capability_gate_status": "PASSED",
        "normalization_sha256": normalization_sha256,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise R4ProtocolError(
                f"V6 Representation predecessor {key} must equal {expected!r}"
            )
    if int(payload.get("epochs_completed", -1)) != 28:
        raise R4ProtocolError("V6 must reuse the selected Representation epoch 28")
    gate = payload.get("selection_gate")
    if not isinstance(gate, Mapping) or str(gate.get("status")) not in {"PASS", "PASSED"}:
        raise R4ProtocolError("V6 Representation predecessor lacks a PASS gate")
    source_state = payload["model_state"]
    inherited = {
        str(name): value
        for name, value in source_state.items()
        if str(name).startswith(_REPRESENTATION_TRUNK_PREFIXES)
    }
    if not inherited:
        raise R4ProtocolError("V6 predecessor import found no Representation trunk")
    forbidden = [
        name for name in inherited if name.startswith(_LEGACY_HEAD_PREFIXES)
    ]
    if forbidden:
        raise R4ProtocolError("V6 predecessor import included a legacy fixed-particle head")
    current_keys = set(model.state_dict())
    unknown = sorted(set(inherited) - current_keys)
    if unknown:
        raise R4ProtocolError(
            "V6 Representation trunk names/shapes changed before import: "
            f"{unknown[:5]}"
        )
    incompatible = model.load_state_dict(inherited, strict=False)
    if incompatible.unexpected_keys:
        raise R4ProtocolError(
            f"V6 Representation import had unexpected keys: {incompatible.unexpected_keys[:5]}"
        )
    inherited_names = sorted(inherited)
    manifest = {
        "schema": "dcvpp.r4.v6-representation-predecessor-import.v1",
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "source_epoch": 28,
        "source_capability_gate": "PASSED",
        "normalization_sha256": normalization_sha256,
        "inherited_prefixes": list(_REPRESENTATION_TRUNK_PREFIXES),
        "inherited_parameter_and_buffer_keys": inherited_names,
        "excluded_legacy_prefixes": list(_LEGACY_HEAD_PREFIXES),
        "excluded_fixed20_head": True,
        "fresh_distribution_head": True,
    }
    if manifest_path.is_file():
        if read_json(manifest_path) != manifest:
            raise R4ProtocolError(
                "frozen V6 Representation predecessor manifest changed"
            )
    else:
        atomic_json(manifest_path, manifest)
    return manifest


def _resolved_config_path(
    config: Mapping[str, Any], repository_root: Path, dotted: tuple[str, ...]
) -> Path:
    value: Any = config
    for key in dotted:
        value = value.get(key) if isinstance(value, Mapping) else None
    if not value:
        raise R4ProtocolError(f"missing config path {'.'.join(dotted)}")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _load_v6_residual_score_contract(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[list[float], list[float], Path, str]:
    path = _resolved_config_path(
        config, repository_root, ("lineage", "residual_score_contract")
    )
    if not path.is_file():
        raise R4ProtocolError(f"V6 residual score contract is missing: {path}")
    digest = sha256_file(path)  # one preflight read
    payload = read_json(path)
    scales = payload.get("scenario_residual_score_scales")
    caps = payload.get("scenario_residual_raw_p99_caps")
    if not isinstance(scales, list) or not isinstance(caps, list):
        raise R4ProtocolError("V6 residual score contract is malformed")
    if len(scales) != 21 or len(caps) != 21:
        raise R4ProtocolError("V6 residual score contract must contain 21 channels")
    numeric_scales = [float(value) for value in scales]
    numeric_caps = [float(value) for value in caps]
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric_scales + numeric_caps):
        raise R4ProtocolError("V6 residual score scales/caps must be finite positive")
    return numeric_scales, numeric_caps, path, digest


def _freeze_v6_distribution_support(
    config: Mapping[str, Any],
    repository_root: Path,
    corpus: PreloadedR4Corpus,
    model: torch.nn.Module,
    *,
    normalization_sha256: str,
    residual_score_path: Path,
    residual_score_sha256: str,
    score_scales: list[float],
    residual_caps: list[float],
) -> V6DistributionSupportContract:
    """Materialize one train-only physical/statistical support contract."""

    path = _resolved_config_path(
        config, repository_root, ("lineage", "distribution_support_artifact")
    )
    train = tuple(corpus.records("train"))
    if len(train) != 543:
        raise R4ProtocolError("Distribution support requires exactly 543 train days")
    targets = torch.stack(
        [torch.as_tensor(record["scenario_target_zonal"]).float() for record in train]
    )
    if targets.shape != (543, 96, 21) or not bool(torch.isfinite(targets).all()):
        raise R4ProtocolError("Distribution support source targets are invalid")
    lower = torch.empty((96, 21), dtype=torch.float32)
    upper = torch.empty_like(lower)
    global_min = targets.amin(dim=(0, 1))
    global_max = targets.amax(dim=(0, 1))
    global_range = (global_max - global_min).clamp_min(1.0e-6)
    # Free price channels use one train-only global envelope per zone.  RT is
    # never free and is induced from DA-DART below.
    for channel_slice in (slice(0, 4), slice(8, 12)):
        lo = global_min[channel_slice] - 0.05 * global_range[channel_slice]
        hi = global_max[channel_slice] + 0.05 * global_range[channel_slice]
        lower[:, channel_slice] = lo
        upper[:, channel_slice] = hi
    lower[:, 4:8] = lower[:, 0:4] - upper[:, 8:12]
    upper[:, 4:8] = upper[:, 0:4] - lower[:, 8:12]

    physical_stream = getattr(model, "physical_stream", None)
    core = getattr(physical_stream, "physical_twin_core", None)
    if core is None or not isinstance(getattr(core, "capacity_work_units", None), torch.Tensor):
        raise R4ProtocolError("Distribution support requires the frozen physical twin")
    workload_capacity = float(core.capacity_work_units.detach().cpu().sum())
    if not math.isfinite(workload_capacity) or workload_capacity <= 0.0:
        raise R4ProtocolError("frozen workload capacity is invalid")
    lower[:, 12] = 0.0
    upper[:, 12] = workload_capacity
    lower[:, 13:17] = 0.0
    pv_cap = targets[..., 13:17].amax(dim=(0, 1)).clamp_min(1.0e-6)
    upper[:, 13:17] = pv_cap

    with torch.inference_mode():
        multiplier = torch.ones((1, 4), dtype=torch.float32)
        low_state = core(multiplier, torch.zeros((1, 96, 4), dtype=torch.float32))["pue"]
        high_state = core(multiplier, torch.ones((1, 96, 4), dtype=torch.float32))["pue"]
    pue_low = torch.minimum(low_state, high_state).squeeze(0)
    pue_high = torch.maximum(low_state, high_state).squeeze(0)
    lower[:, 17:21] = pue_low
    upper[:, 17:21] = pue_high
    if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all()):
        raise R4ProtocolError("Distribution support contains non-finite bounds")
    if not bool(torch.all(lower < upper)):
        raise R4ProtocolError("Distribution support lower bounds must be below upper bounds")
    if len(score_scales) != 21 or len(residual_caps) != 21:
        raise R4ProtocolError("Distribution support requires 21 score scales/caps")
    residual_caps_zonal = torch.tensor(residual_caps, dtype=torch.float32).reshape(
        1, 21
    ).expand(96, -1).clone()
    # Preserve the exact DA/RT/DART structural envelope used by the new head.
    residual_caps_zonal[:, 4:8] = (
        residual_caps_zonal[:, 0:4] + residual_caps_zonal[:, 8:12]
    )
    payload = {
        "schema": "r4_v6_distribution_support.v1",
        "artifact_path": str(path.resolve()),
        "fit_split": "train",
        "fit_days": 543,
        "fit_date_start": "2022-01-01",
        "fit_date_end": "2023-06-30",
        "formal_data_opened": False,
        "normalization_sha256": normalization_sha256,
        "physical_sha256": corpus.physical_sha256,
        "source_residual_score_contract_path": str(residual_score_path.resolve()),
        "source_residual_score_contract_sha256": residual_score_sha256,
        "residual_score_contract_sha256": residual_score_sha256,
        "score_scales": [float(value) for value in score_scales],
        # The 21-channel vector is the immutable V5 residual-score lineage.
        # ``residual_caps_zonal`` is the active [96,21] model input.  Keeping
        # both names prevents a time-expanded support tensor from masquerading
        # as a newly fitted score contract.
        "residual_caps": [float(value) for value in residual_caps],
        "residual_caps_zonal": residual_caps_zonal.tolist(),
        "lower_zonal": lower.tolist(),
        "upper_zonal": upper.tolist(),
        "channel_order": (
            [f"da_z{index}" for index in range(1, 5)]
            + [f"rt_z{index}" for index in range(1, 5)]
            + [f"dart_z{index}" for index in range(1, 5)]
            + ["workload"]
            + [f"pv_z{index}" for index in range(1, 5)]
            + [f"pue_z{index}" for index in range(1, 5)]
        ),
        "price_support": "train_global_minmax_plus_5pct_train_range",
        "rt_support": "structurally_induced_from_da_minus_dart",
        "workload_support": "zero_to_frozen_total_capacity_work_units",
        "pv_support": "zero_to_train_only_registered_capacity_proxy",
        "pue_support": "frozen_physical_twin_endpoint_envelope",
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "future_years_opened": False,
    }
    if path.is_file():
        existing = read_json(path)
        if existing != payload:
            raise R4ProtocolError("frozen Distribution support artifact changed")
    else:
        atomic_json(path, payload)
    digest = sha256_file(path)  # exactly once after materialization/load
    return V6DistributionSupportContract(
        path=path.resolve(),
        artifact_sha256=digest,
        source_residual_score_contract_path=residual_score_path.resolve(),
        source_residual_score_contract_sha256=residual_score_sha256,
        score_scales=tuple(float(value) for value in score_scales),
        residual_caps=tuple(
            tuple(float(value) for value in row)
            for row in payload["residual_caps_zonal"]
        ),
        lower_zonal=tuple(
            tuple(float(value) for value in row) for row in payload["lower_zonal"]
        ),
        upper_zonal=tuple(
            tuple(float(value) for value in row) for row in payload["upper_zonal"]
        ),
    )


def _freeze_v6_distribution_training_contract(
    context: ProtocolContext,
    config: Mapping[str, Any],
    curriculum: DistributionCurriculumContract,
    *,
    predecessor_manifest: Mapping[str, Any],
    normalization_sha256: str,
    residual_score_path: Path,
    residual_score_sha256: str,
    support_contract: V6DistributionSupportContract,
    sampling_contract_path: Path,
    sampling_contract_sha256: str,
) -> tuple[Path, str]:
    path = Path(context.output_root) / "data" / "distribution_training_contract.json"
    payload = {
        **curriculum.as_manifest(),
        "architecture_version": "continuous_distributional_joint_moe_r4_v6",
        "training_semantics_revision": (
            "r4_distributional_world_latent_variable_n_curriculum_cycle2_v1"
        ),
        "config_sha256": context.config_sha256,
        "representation_predecessor": dict(predecessor_manifest),
        "normalization_sha256": normalization_sha256,
        "residual_score_contract": str(residual_score_path),
        "residual_score_contract_sha256": residual_score_sha256,
        "distribution_support_artifact": str(support_contract.path),
        "distribution_support_sha256": support_contract.artifact_sha256,
        "distribution_support_schema": "r4_v6_distribution_support.v1",
        "distribution_sampling_contract": str(sampling_contract_path.resolve()),
        "distribution_sampling_contract_sha256": sampling_contract_sha256,
        "source_residual_score_contract_sha256": (
            support_contract.source_residual_score_contract_sha256
        ),
        "score_scales": list(support_contract.score_scales),
        "residual_caps_shape": [96, 21],
        "support_bounds_shape": [96, 21],
        "fit_split": support_contract.fit_split,
        "formal_data_opened": support_contract.formal_data_opened,
        "formal_probability_samples": 256,
        "formal_empirical_risk_samples": 512,
        "sampling_budget_ablation": [20, 64, 128, 256, 512],
        "old_fixed20_head_loaded": False,
        "old_scenario_epoch200_loaded": False,
    }
    if path.is_file():
        if read_json(path) != payload:
            raise R4ProtocolError("frozen V6 Distribution training contract changed")
    else:
        atomic_json(path, payload)
    return path.resolve(), sha256_file(path)


def _freeze_v6_distribution_sampling_contract(
    context: ProtocolContext,
    config: Mapping[str, Any],
    curriculum: DistributionCurriculumContract,
) -> tuple[Path, str]:
    """Freeze sampling budgets once before the first Distribution update."""

    path = Path(context.output_root) / "data" / "distribution_sampling_contract.json"
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        raise R4ProtocolError("V6 sampling config must be a mapping")
    payload = {
        "schema": SAMPLING_CONTRACT_SCHEMA,
        "status": "FROZEN_BEFORE_TRAINING",
        "config_sha256": context.config_sha256,
        "architecture_version": "continuous_distributional_joint_moe_r4_v6",
        "train_samples": curriculum.train_samples,
        "cheap_validation_samples": curriculum.cheap_validation_samples,
        "formal_probability_samples": int(sampling.get("formal_probability", -1)),
        "formal_empirical_risk_samples": int(
            sampling.get("formal_empirical_risk", -1)
        ),
        "sampling_budget_ablation": [int(value) for value in sampling.get("ablation", ())],
        "policy_inference_default_samples": int(
            sampling.get("policy_inference_default_samples", -1)
        ),
        "physical_microbatch_days": curriculum.physical_microbatch_days,
        "scientific_optimizer_effective_days": 543,
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "fixed_particle_representation_capacity": False,
    }
    required = {
        "formal_probability_samples": 256,
        "formal_empirical_risk_samples": 512,
        "policy_inference_default_samples": 0,
    }
    for key, expected in required.items():
        if payload[key] != expected:
            raise R4ProtocolError(f"V6 sampling contract {key} must equal {expected}")
    if payload["sampling_budget_ablation"] != [20, 64, 128, 256, 512]:
        raise R4ProtocolError("V6 sampling ablation contract changed")
    if path.is_file():
        if read_json(path) != payload:
            raise R4ProtocolError("frozen V6 Distribution sampling contract changed")
    else:
        atomic_json(path, payload)
    return path.resolve(), sha256_file(path)


def _attach_v6_distribution_contract(
    batch: Mapping[str, Any],
    *,
    support_lower: torch.Tensor,
    support_upper: torch.Tensor,
    support_sha256: str,
    distribution_contract_path: Path,
    distribution_contract_sha256: str,
    residual_caps: torch.Tensor,
) -> dict[str, Any]:
    result = dict(batch)
    result["distribution_support"] = {
        "schema": "r4_v6_distribution_support.v1",
        "lower_zonal": support_lower,
        "upper_zonal": support_upper,
        "contract_sha256": support_sha256,
    }
    # The Distribution head intentionally reuses the canonical cap key so the
    # deterministic world-latent path and the variable-N sampler see one
    # identical frozen envelope.
    result["scenario_residual_caps_zonal"] = residual_caps
    result["distribution_training_contract_metadata"] = {
        "path": str(distribution_contract_path),
        "sha256": distribution_contract_sha256,
        "support_sha256": support_sha256,
    }
    return result


def _fit_train_only_loss_scales(
    corpus: PreloadedR4Corpus,
) -> tuple[dict[str, float], list[float]]:
    """Fit immutable target scales from the 543-day train split only."""

    records = corpus.records("train")
    if len(records) != 543:
        raise R4ProtocolError("loss-scale fit requires exactly 543 train records")
    representation: dict[str, float] = {}
    for name in ("da", "rt", "dart", "workload", "pv", "pue"):
        daily_scales: list[torch.Tensor] = []
        for record in records:
            targets = record.get("targets")
            value = targets.get(name) if isinstance(targets, Mapping) else None
            if not isinstance(value, torch.Tensor):
                raise R4ProtocolError(f"train record is missing target {name!r}")
            daily_scales.append(value.detach().cpu().float().abs().mean())
        scale = torch.stack(daily_scales).median().clamp_min(1.0e-3)
        representation[name] = float(scale)

    scenario_daily_scales: list[torch.Tensor] = []
    for record in records:
        value = record.get("scenario_target_zonal")
        if not isinstance(value, torch.Tensor) or value.shape != (96, 21):
            raise R4ProtocolError(
                "train record scenario_target_zonal must have shape [96,21]"
            )
        scenario_daily_scales.append(value.detach().cpu().float().abs().mean(dim=0))
    scenario = torch.stack(scenario_daily_scales).median(dim=0).values.clamp_min(1.0e-3)
    return representation, [float(value) for value in scenario]


def _train_only_statistics_payload(
    corpus: PreloadedR4Corpus,
    config: Mapping[str, Any],
    representation_scales: Mapping[str, float],
    scenario_feature_scales: list[float],
) -> dict[str, Any]:
    return {
        **corpus.statistics.as_json(),
        "fixed_training_cost_scale": corpus.fixed_training_cost_scale,
        "fixed_training_cost_scale_name": config["training"][
            "fixed_training_cost_scale"
        ],
        "physical_sha256": corpus.physical_sha256,
        "physical_twin_lineage": dict(corpus.physical_lineage),
        "train_days": len(corpus.records("train")),
        "validation_days": len(corpus.records("validation")),
        "training_loss_scaling": {
            "fit_split": "train",
            "fit_days": 543,
            "representation_method": (
                "per_head_median_daily_mean_absolute_target_floor_1e-3"
            ),
            "representation_scales": dict(representation_scales),
            "scenario_method": (
                "per_variable_median_daily_mean_absolute_target_floor_1e-3"
            ),
            "scenario_variable_order": (
                [f"da_z{index}" for index in range(1, 5)]
                + [f"rt_z{index}" for index in range(1, 5)]
                + [f"dart_z{index}" for index in range(1, 5)]
                + ["workload"]
                + [f"pv_z{index}" for index in range(1, 5)]
                + [f"pue_z{index}" for index in range(1, 5)]
            ),
            "scenario_feature_scales": list(scenario_feature_scales),
        },
    }


def prepare_v6_distribution_preflight_contracts(
    context: ProtocolContext,
    config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Materialize every V6 train-only contract exactly once in preflight.

    The formal protocol is the sole caller.  Gradient training later reads the
    frozen-preflight manifest and never re-fits support, reimports a legacy
    fixed-particle head, or rehashes the large predecessor checkpoint.
    """

    resolved_config = (
        load_r4_config(context.config_path) if config is None else dict(config)
    )
    architecture = str(
        resolved_config.get("project", {}).get("architecture_version", "")
    )
    if architecture != "continuous_distributional_joint_moe_r4_v6":
        raise R4ProtocolError("V6 preflight helper received a non-distributional config")
    repository_root = Path(context.repository_root).resolve()
    corpus = PreloadedR4Corpus(resolved_config, repository_root)
    representation_scales, corpus_scenario_scales = _fit_train_only_loss_scales(
        corpus
    )
    normalization_path = _resolved_config_path(
        resolved_config, repository_root, ("lineage", "train_only_normalization")
    )
    if not normalization_path.is_file():
        raise R4ProtocolError(
            f"V6 inherited train-only normalization is missing: {normalization_path}"
        )
    expected_statistics = _train_only_statistics_payload(
        corpus,
        resolved_config,
        representation_scales,
        corpus_scenario_scales,
    )
    if read_json(normalization_path) != expected_statistics:
        raise R4ProtocolError(
            "V6 inherited normalization differs from the current frozen train corpus"
        )
    normalization_sha256 = sha256_file(normalization_path)

    model = build_formal_r4_model(resolved_config, repository_root)
    hard_limit = int(
        resolved_config["model"]["parameter_budget"]["hard_stop_above_trainable"]
    )
    if model.parameter_count > hard_limit:
        raise R4ProtocolError(
            f"R4 parameter count {model.parameter_count} exceeds hard limit {hard_limit}"
        )
    source_checkpoint = _resolved_config_path(
        resolved_config,
        repository_root,
        ("lineage", "representation_selected_checkpoint"),
    )
    source_representation_state = _resolved_config_path(
        resolved_config,
        repository_root,
        ("lineage", "representation_state"),
    )
    if not source_representation_state.is_file():
        raise R4ProtocolError(
            f"V6 Representation state evidence is missing: {source_representation_state}"
        )
    source_representation_state_sha256 = sha256_file(
        source_representation_state
    )
    import_manifest_path = (
        Path(context.output_root) / "data" / "representation_predecessor_import.json"
    ).resolve()
    predecessor_manifest = _load_v6_representation_trunk(
        model,
        source_checkpoint,
        seed=int(resolved_config["protocol"]["active_seed"]),
        normalization_sha256=normalization_sha256,
        manifest_path=import_manifest_path,
    )
    import_manifest_sha256 = sha256_file(import_manifest_path)

    residual_contract_path = _resolved_config_path(
        resolved_config, repository_root, ("lineage", "residual_score_contract")
    )
    try:
        residual_contract = load_scenario_training_contract(
            residual_contract_path,
            expected_representation_sha256=str(
                predecessor_manifest["source_checkpoint_sha256"]
            ),
            expected_normalization_sha256=normalization_sha256,
        )
    except ScenarioTrainingContractError as exc:
        raise R4ProtocolError(str(exc)) from exc
    score_scales = [float(value) for value in residual_contract.score_scales]
    residual_caps = [float(value) for value in residual_contract.raw_p99_caps]
    support_contract = _freeze_v6_distribution_support(
        resolved_config,
        repository_root,
        corpus,
        model,
        normalization_sha256=normalization_sha256,
        residual_score_path=residual_contract.path,
        residual_score_sha256=residual_contract.artifact_sha256,
        score_scales=score_scales,
        residual_caps=residual_caps,
    )
    setter = getattr(model.distribution_head, "set_default_residual_caps", None)
    initializer = getattr(model.distribution_head, "initialize_scale_from_contract", None)
    if not callable(setter) or not callable(initializer):
        raise R4ProtocolError("V6 Distribution head lacks frozen scale/cap APIs")
    setter(residual_contract.raw_p99_caps)
    initialized_scale_fraction = initializer(
        residual_contract.score_scales, residual_contract.raw_p99_caps
    )
    if not bool(torch.isfinite(initialized_scale_fraction).all()):
        raise R4ProtocolError("V6 contract-derived scale initialization is non-finite")

    curriculum = distribution_curriculum_contract(resolved_config)
    sampling_path, sampling_sha256 = _freeze_v6_distribution_sampling_contract(
        context, resolved_config, curriculum
    )
    training_path, training_sha256 = _freeze_v6_distribution_training_contract(
        context,
        resolved_config,
        curriculum,
        predecessor_manifest=predecessor_manifest,
        normalization_sha256=normalization_sha256,
        residual_score_path=residual_contract.path,
        residual_score_sha256=residual_contract.artifact_sha256,
        support_contract=support_contract,
        sampling_contract_path=sampling_path,
        sampling_contract_sha256=sampling_sha256,
    )

    def frozen(path: Path, digest: str, role: str) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "sha256": str(digest).lower(),
            "role": role,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    return {
        "schema": "dcvpp.r4.v6-distribution-preflight-contracts.v1",
        "config_sha256": context.config_sha256,
        "frozen_artifacts": {
            "representation_predecessor_import": frozen(
                import_manifest_path,
                import_manifest_sha256,
                "epoch28_representation_prefix_allowlist_binding",
            ),
            "distribution_support": frozen(
                support_contract.path,
                support_contract.artifact_sha256,
                "train_only_distribution_support",
            ),
            "distribution_training_contract": frozen(
                training_path,
                training_sha256,
                "five_stage_variable_n_training_contract",
            ),
            "distribution_sampling_contract": frozen(
                sampling_path,
                sampling_sha256,
                "variable_n_sampling_budget_contract",
            ),
        },
        "normalization": frozen(
            normalization_path, normalization_sha256, "inherited_train_only_normalization"
        ),
        "source_checkpoint": frozen(
            source_checkpoint,
            str(predecessor_manifest["source_checkpoint_sha256"]),
            "capability_passed_representation_epoch28",
        ),
        "source_representation_state": frozen(
            source_representation_state,
            source_representation_state_sha256,
            "epoch28_full_179_day_representation_gate_evidence",
        ),
        "source_residual_contract": frozen(
            residual_contract.path,
            residual_contract.artifact_sha256,
            "read_only_v5_train_residual_score_lineage",
        ),
        "model_contract": {
            "architecture_version": architecture,
            "parameter_count": int(model.parameter_count),
            "policy_requires_samples": False,
            "fixed20_head_loaded": False,
            "distribution_sample_budgets": [20, 64, 128, 256, 512],
            "train_samples": curriculum.train_samples,
            "cheap_validation_samples": curriculum.cheap_validation_samples,
        },
        "data_contract": {
            "train_days": 543,
            "validation_days": 179,
            "formal_selection_2024_opened": False,
            "locked_test_2025_opened": False,
        },
    }


@dataclass(frozen=True)
class V6FrozenDistributionSetup:
    normalization_path: Path
    normalization_sha256: str
    predecessor_manifest: Mapping[str, Any]
    predecessor_sha256: str
    residual_contract: ScenarioTrainingContract
    support_contract: V6DistributionSupportContract
    distribution_training_contract_path: Path
    distribution_training_contract_sha256: str
    sampling_contract_path: Path
    sampling_contract_sha256: str


def _frozen_manifest_entry(
    payload: Mapping[str, Any], key: str
) -> tuple[Path, str]:
    entry = payload.get(key)
    if not isinstance(entry, Mapping):
        raise R4ProtocolError(f"frozen preflight manifest lacks {key!r}")
    path = Path(str(entry.get("path", ""))).resolve()
    digest = str(entry.get("sha256", "")).lower()
    if not path.is_file():
        raise R4ProtocolError(f"frozen preflight artifact is missing: {path}")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise R4ProtocolError(f"frozen preflight digest for {key!r} is malformed")
    stat = path.stat()
    if (
        "size_bytes" in entry
        and int(entry["size_bytes"]) != int(stat.st_size)
    ) or (
        "mtime_ns" in entry and int(entry["mtime_ns"]) != int(stat.st_mtime_ns)
    ):
        raise R4ProtocolError(
            f"frozen preflight artifact {key!r} changed after its one-time hash"
        )
    return path, digest


def _load_v6_frozen_distribution_setup(
    context: ProtocolContext,
    config: Mapping[str, Any],
    model: torch.nn.Module,
) -> V6FrozenDistributionSetup:
    """Load preflight-frozen V6 contracts with stat checks, not repeat hashes."""

    manifest_path = (
        Path(context.output_root) / "preflight" / "frozen_preflight_manifest.json"
    ).resolve()
    if not manifest_path.is_file():
        raise R4ProtocolError(
            "V6 Distribution requires preflight/frozen_preflight_manifest.json"
        )
    manifest = read_json(manifest_path)
    if str(manifest.get("config_sha256", "")).lower() != context.config_sha256:
        raise R4ProtocolError("frozen V6 preflight manifest config binding changed")
    if str(manifest.get("status", "FROZEN_BEFORE_TRAINING")) not in {
        "PASS",
        "PASSED",
        "FROZEN_BEFORE_TRAINING",
    }:
        raise R4ProtocolError("V6 frozen preflight manifest is not training-ready")
    frozen_artifacts = manifest.get("frozen_artifacts")
    if not isinstance(frozen_artifacts, Mapping):
        raise R4ProtocolError("V6 preflight manifest lacks frozen_artifacts")
    import_path, _ = _frozen_manifest_entry(
        frozen_artifacts, "representation_predecessor_import"
    )
    support_path, support_sha = _frozen_manifest_entry(
        frozen_artifacts, "distribution_support"
    )
    training_path, training_sha = _frozen_manifest_entry(
        frozen_artifacts, "distribution_training_contract"
    )
    sampling_path, sampling_sha = _frozen_manifest_entry(
        frozen_artifacts, "distribution_sampling_contract"
    )
    normalization_path, normalization_sha = _frozen_manifest_entry(
        manifest, "normalization"
    )
    source_checkpoint, predecessor_sha = _frozen_manifest_entry(
        manifest, "source_checkpoint"
    )
    residual_path, residual_sha = _frozen_manifest_entry(
        manifest, "source_residual_contract"
    )

    configured_predecessor = _resolved_config_path(
        config,
        Path(context.repository_root),
        ("lineage", "representation_selected_checkpoint"),
    )
    configured_normalization = _resolved_config_path(
        config,
        Path(context.repository_root),
        ("lineage", "train_only_normalization"),
    )
    if source_checkpoint != configured_predecessor:
        raise R4ProtocolError("V6 preflight predecessor path differs from config")
    if normalization_path != configured_normalization:
        raise R4ProtocolError("V6 preflight normalization path differs from config")
    predecessor_manifest = _load_v6_representation_trunk(
        model,
        source_checkpoint,
        seed=int(config["protocol"]["active_seed"]),
        normalization_sha256=normalization_sha,
        manifest_path=import_path,
        source_sha256=predecessor_sha,
    )

    residual_payload = read_json(residual_path)
    residual_contract = ScenarioTrainingContract(
        residual_path, residual_sha, residual_payload
    )
    score_scales = residual_contract.score_scales
    residual_caps = residual_contract.raw_p99_caps
    if (
        score_scales.shape != (21,)
        or residual_caps.shape != (21,)
        or not bool(torch.isfinite(score_scales).all())
        or not bool(torch.isfinite(residual_caps).all())
        or bool((score_scales <= 0).any())
        or bool((residual_caps <= 0).any())
    ):
        raise R4ProtocolError("frozen V6 residual score contract is malformed")

    support_payload = read_json(support_path)
    if support_payload.get("schema") != "r4_v6_distribution_support.v1":
        raise R4ProtocolError("frozen V6 support schema changed")
    support_contract = V6DistributionSupportContract(
        path=support_path,
        artifact_sha256=support_sha,
        source_residual_score_contract_path=residual_path,
        source_residual_score_contract_sha256=residual_sha,
        score_scales=tuple(float(value) for value in support_payload["score_scales"]),
        residual_caps=tuple(
            tuple(float(value) for value in row)
            for row in support_payload["residual_caps_zonal"]
        ),
        lower_zonal=tuple(
            tuple(float(value) for value in row)
            for row in support_payload["lower_zonal"]
        ),
        upper_zonal=tuple(
            tuple(float(value) for value in row)
            for row in support_payload["upper_zonal"]
        ),
        fit_split=str(support_payload.get("fit_split", "")),
        formal_data_opened=bool(support_payload.get("formal_data_opened", True)),
    )
    support_tensors = support_contract.tensor_payload(torch.device("cpu"))
    if (
        support_tensors[0].shape != (96, 21)
        or support_tensors[1].shape != (96, 21)
        or support_tensors[2].shape != (96, 21)
        or support_contract.fit_split != "train"
        or support_contract.formal_data_opened
        or support_contract.source_residual_score_contract_sha256 != residual_sha
    ):
        raise R4ProtocolError("frozen V6 support payload is not train-only [96,21]")

    sampling_payload = read_json(sampling_path)
    if (
        sampling_payload.get("schema") != SAMPLING_CONTRACT_SCHEMA
        or sampling_payload.get("status") != "FROZEN_BEFORE_TRAINING"
        or int(sampling_payload.get("train_samples", -1)) != 32
        or int(sampling_payload.get("cheap_validation_samples", -1)) != 64
        or int(sampling_payload.get("formal_probability_samples", -1)) != 256
        or int(sampling_payload.get("formal_empirical_risk_samples", -1)) != 512
    ):
        raise R4ProtocolError("frozen V6 sampling contract changed")
    training_payload = read_json(training_path)
    if (
        training_payload.get("schema")
        != "dcvpp.r4.v6-distribution-training-contract.v1"
        or training_payload.get("distribution_support_sha256") != support_sha
        or training_payload.get("distribution_sampling_contract_sha256")
        != sampling_sha
        or training_payload.get("normalization_sha256") != normalization_sha
        or training_payload.get("residual_score_contract_sha256") != residual_sha
    ):
        raise R4ProtocolError("frozen V6 Distribution training contract changed")

    setter = getattr(model.distribution_head, "set_default_residual_caps", None)
    initializer = getattr(model.distribution_head, "initialize_scale_from_contract", None)
    if not callable(setter) or not callable(initializer):
        raise R4ProtocolError("V6 Distribution head lacks frozen scale/cap APIs")
    setter(residual_caps)
    initializer(score_scales, residual_caps)
    return V6FrozenDistributionSetup(
        normalization_path=normalization_path,
        normalization_sha256=normalization_sha,
        predecessor_manifest=predecessor_manifest,
        predecessor_sha256=predecessor_sha,
        residual_contract=residual_contract,
        support_contract=support_contract,
        distribution_training_contract_path=training_path,
        distribution_training_contract_sha256=training_sha,
        sampling_contract_path=sampling_path,
        sampling_contract_sha256=sampling_sha,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    phase: str,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    """One exact linear-warmup/cosine step per 543-day optimizer update."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise R4ProtocolError("training config must be a mapping")
    if training.get("scheduler") != "cosine_with_linear_warmup":
        raise R4ProtocolError("formal R4 requires cosine_with_linear_warmup")
    fraction = float(training.get("warmup_fraction", 0.0))
    if not 0.0 < fraction < 1.0:
        raise R4ProtocolError("warmup_fraction must lie strictly in (0,1)")
    total_steps = int(phase_spec(phase, config).epochs)
    warmup_steps = max(1, int(math.ceil(total_steps * fraction)))
    cosine_steps = max(total_steps - warmup_steps, 1)

    def multiplier(step: int) -> float:
        current = int(step)
        if current < warmup_steps:
            return float(current + 1) / float(warmup_steps)
        progress = min(max((current - warmup_steps) / cosine_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _choose_physical_batch(
    model: torch.nn.Module,
    corpus: PreloadedR4Corpus,
    phase: str,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    behavioral_baseline: torch.nn.Module | None = None,
    representation_scales: Mapping[str, float] | None = None,
    scenario_feature_scales: list[float] | None = None,
    scenario_training_contract: ScenarioTrainingContract | None = None,
    distribution_support_lower: torch.Tensor | None = None,
    distribution_support_upper: torch.Tensor | None = None,
    distribution_support_sha256: str | None = None,
    distribution_training_contract_path: Path | None = None,
    distribution_training_contract_sha256: str | None = None,
    distribution_residual_caps: torch.Tensor | None = None,
    pretraining_probe_path: Path | None = None,
) -> int:
    """Probe the registered fallback ladder without writing a training step."""

    configured = [
        int(value) for value in config["training"]["physical_microbatch_fallback"]
    ]
    if phase == "scenario":
        # Same-machine formal profiling established B128 as the largest
        # measured healthy Scenario batch (about 2.1 GiB reserved and 1.6 s
        # for a complete forward/backward).  B512 spilled the RTX 3080 into
        # WDDM shared memory and did not finish its forward in 30 s.  Physical
        # chunking does not change the registered 543-day mean gradient or the
        # single optimizer update, so skip unmeasured/unsafe larger chunks.
        configured = [value for value in configured if value <= 128]
        if not configured:
            raise R4ProtocolError("Scenario fallback ladder has no batch <=128")
    uses_v6_distribution_contract = distribution_training_contract_path is not None
    if phase == "distribution":
        configured = [32]
        if pretraining_probe_path is not None and pretraining_probe_path.is_file():
            persisted_probe = read_json(pretraining_probe_path)
            if (
                persisted_probe.get("status") != "PASS"
                or int(persisted_probe.get("physical_microbatch_days", -1)) != 32
                or int(persisted_probe.get("train_samples", -1)) != 32
                or persisted_probe.get("distribution_training_contract_sha256")
                != distribution_training_contract_sha256
                or persisted_probe.get("formal_selection_2024_opened") is not False
                or persisted_probe.get("locked_test_2025_opened") is not False
            ):
                raise R4ProtocolError("persisted V6 pretraining probe contract changed")
            return 32
    loss = build_phase_loss(
        phase,
        config,
        fixed_training_cost_scale=(
            corpus.fixed_training_cost_scale
            if phase in {"policy", "joint_policy", "a1", "a2"}
            else None
        ),
        representation_scales=representation_scales,
        scenario_feature_scales=scenario_feature_scales,
    )
    # Memory safety must be established at the formal K=96 path.  A K=8
    # success says nothing about the full decision rollout retained for
    # backward.  A1 additionally attaches the frozen selected-policy baseline.
    horizon = 96
    for size in configured:
        size = int(size)
        try:
            is_cuda_probe = device.type == "cuda"
            if is_cuda_probe:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            probe_started = time.perf_counter()
            batch = next(
                corpus.batches("train", size, device=device, max_days=size, origin=0)
            )
            if uses_v6_distribution_contract and phase != "representation":
                if (
                    distribution_support_lower is None
                    or distribution_support_upper is None
                    or distribution_support_sha256 is None
                    or distribution_training_contract_path is None
                    or distribution_training_contract_sha256 is None
                    or distribution_residual_caps is None
                ):
                    raise R4ProtocolError(
                        "Distribution physical-batch probe lacks its frozen contracts"
                    )
                batch = _attach_v6_distribution_contract(
                    batch,
                    support_lower=distribution_support_lower,
                    support_upper=distribution_support_upper,
                    support_sha256=distribution_support_sha256,
                    distribution_contract_path=distribution_training_contract_path,
                    distribution_contract_sha256=(
                        distribution_training_contract_sha256
                    ),
                    residual_caps=distribution_residual_caps,
                )
                batch["distribution_train_samples"] = int(
                    config.get("sampling", {}).get("train", 32)
                )
            elif phase != "representation":
                if scenario_training_contract is None:
                    raise R4ProtocolError(
                        f"phase {phase} physical-batch probe lacks Scenario contract"
                    )
                batch = attach_scenario_training_contract(
                    batch, scenario_training_contract, device=device
                )
            if phase == "a1":
                if behavioral_baseline is None:
                    raise R4ProtocolError(
                        "A1 physical-batch probe requires the frozen selected-policy baseline"
                    )
                batch = _attach_a1_behavioral_baseline(
                    batch, behavioral_baseline, horizon
                )
            model.zero_grad(set_to_none=True)
            value = loss(model, batch, horizon)["total"]
            if not bool(torch.isfinite(value)):
                raise R4ProtocolError("physical-batch probe produced non-finite loss")
            value.backward()
            finite_gradient_parameters = 0
            nonzero_gradient_parameters = 0
            for parameter in model.parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise R4ProtocolError(
                        "physical-batch probe produced a non-finite gradient"
                    )
                finite_gradient_parameters += 1
                if bool(torch.any(parameter.grad != 0)):
                    nonzero_gradient_parameters += 1
            if finite_gradient_parameters == 0 or nonzero_gradient_parameters == 0:
                raise R4ProtocolError(
                    "physical-batch probe found no finite nonzero trainable gradient"
                )
            if is_cuda_probe:
                torch.cuda.synchronize(device)
            probe_seconds = time.perf_counter() - probe_started
            peak_reserved = (
                int(torch.cuda.max_memory_reserved(device)) if is_cuda_probe else 0
            )
            total_memory = (
                int(torch.cuda.get_device_properties(device).total_memory)
                if is_cuda_probe else 2**63 - 1
            )
            # A batch that spills a 10 GiB RTX 3080 into WDDM shared memory can
            # report 100% utilization while running orders of magnitude more
            # slowly.  The fallback ladder preserves the same 543-day mean
            # gradient and one optimizer step; only the physical chunking
            # changes.  Keep a full GiB headroom and reject a probe that has
            # already exceeded the preregistered two-minute health budget.
            healthy_memory = (not is_cuda_probe) or peak_reserved <= min(
                int(total_memory * 0.875), total_memory - 1024**3
            )
            healthy_time = (not is_cuda_probe) or probe_seconds <= 120.0
            if (
                healthy_memory
                and healthy_time
                and phase == "distribution"
                and pretraining_probe_path is not None
            ):
                atomic_json(
                    pretraining_probe_path,
                    {
                        "schema": "dcvpp.r4.v6-distribution-pretraining-probe.v1",
                        "status": "PASS",
                        "physical_microbatch_days": size,
                        "scientific_optimizer_effective_days": 543,
                        "scientific_accumulation_steps": math.ceil(543 / size),
                        "train_samples": 32,
                        "loss_finite": True,
                        "gradient_finite": True,
                        "nonzero_gradient_parameters": nonzero_gradient_parameters,
                        "finite_gradient_parameters": finite_gradient_parameters,
                        "probe_loss": float(value.detach().cpu()),
                        "peak_cuda_reserved_bytes": peak_reserved,
                        "cuda_total_memory_bytes": (
                            int(total_memory) if is_cuda_probe else 0
                        ),
                        "probe_seconds": probe_seconds,
                        "distribution_training_contract_sha256": (
                            distribution_training_contract_sha256
                        ),
                        "distribution_support_sha256": distribution_support_sha256,
                        "formal_selection_2024_opened": False,
                        "locked_test_2025_opened": False,
                        "optimizer_step_executed": False,
                    },
                )
            model.zero_grad(set_to_none=True)
            del batch, value
            if is_cuda_probe:
                torch.cuda.empty_cache()
            if healthy_memory and healthy_time:
                return size
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    raise R4ProtocolError("all registered physical microbatches OOM on the RTX 3080")


def _cheap_validator(
    corpus: PreloadedR4Corpus,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    behavioral_baseline: torch.nn.Module | None = None,
    representation_scales: Mapping[str, float] | None = None,
    scenario_feature_scales: list[float] | None = None,
    scenario_training_contract: ScenarioTrainingContract | None = None,
    physical_batch_size: int = 1,
    distribution_support_lower: torch.Tensor | None = None,
    distribution_support_upper: torch.Tensor | None = None,
    distribution_support_sha256: str | None = None,
    distribution_training_contract_path: Path | None = None,
    distribution_training_contract_sha256: str | None = None,
    distribution_residual_caps: torch.Tensor | None = None,
) -> Any:
    def evaluate(
        model: torch.nn.Module,
        phase: str,
        horizon: int,
        *,
        distribution_stage_index: int | None = None,
    ) -> float | Mapping[str, float]:
        loss = build_phase_loss(
            phase,
            config,
            fixed_training_cost_scale=(
                corpus.fixed_training_cost_scale
                if phase in {"policy", "joint_policy", "a1", "a2"}
                else None
            ),
            representation_scales=representation_scales,
            scenario_feature_scales=scenario_feature_scales,
        )
        if phase == "distribution":
            if distribution_stage_index is None:
                raise R4ProtocolError(
                    "Distribution cheap validation requires its S1--S5 stage index"
                )
            loss.set_distribution_stage(distribution_stage_index)
            contract = distribution_curriculum_contract(config)
            loss.set_distribution_sample_count(contract.cheap_validation_samples)
        if phase == "representation":
            validation_days, batch_size = 179, min(int(physical_batch_size), 179)
        elif phase == "scenario":
            validation_days, batch_size = 179, min(int(physical_batch_size), 16)
        elif phase == "distribution":
            # Level1 is deliberately cheap and never substitutes for the
            # separate 179-day Level2 gate.
            validation_days, batch_size = 32, min(int(physical_batch_size), 8)
        else:
            # A fixed, preregistered chronological feedback panel is used for
            # decision gradients; the 179x96 hard gate remains phase-end only.
            validation_days, batch_size = 16, 1
        weighted_sum = 0.0
        metric_sums: dict[str, float] = {}
        observed_days = 0
        uses_v6_distribution_contract = distribution_training_contract_path is not None
        for batch in corpus.batches(
            "validation",
            batch_size,
            device=device,
            max_days=validation_days,
            origin=0,
        ):
            if uses_v6_distribution_contract and phase != "representation":
                if (
                    distribution_support_lower is None
                    or distribution_support_upper is None
                    or distribution_support_sha256 is None
                    or distribution_training_contract_path is None
                    or distribution_training_contract_sha256 is None
                    or distribution_residual_caps is None
                ):
                    raise R4ProtocolError(
                        "Distribution cheap validation lacks its frozen contracts"
                    )
                batch = _attach_v6_distribution_contract(
                    batch,
                    support_lower=distribution_support_lower,
                    support_upper=distribution_support_upper,
                    support_sha256=distribution_support_sha256,
                    distribution_contract_path=distribution_training_contract_path,
                    distribution_contract_sha256=(
                        distribution_training_contract_sha256
                    ),
                    residual_caps=distribution_residual_caps,
                )
                batch["distribution_train_samples"] = int(
                    config.get("sampling", {}).get("train", 32)
                )
            elif phase != "representation":
                if scenario_training_contract is None:
                    raise R4ProtocolError(
                        f"phase {phase} cheap validation lacks Scenario contract"
                    )
                batch = attach_scenario_training_contract(
                    batch, scenario_training_contract, device=device
                )
            if phase == "a1":
                if behavioral_baseline is None:
                    raise R4ProtocolError("A1 cheap validation requires frozen selected-policy baseline")
                batch = _attach_a1_behavioral_baseline(
                    batch, behavioral_baseline, horizon
                )
            values = loss(model, batch, horizon)
            value = values["total"]
            raw_days = batch.get("num_days") if isinstance(batch, Mapping) else None
            days = int(raw_days.item()) if isinstance(raw_days, torch.Tensor) else int(raw_days)
            weighted_sum += float(value.detach().cpu()) * days
            if phase == "distribution":
                for name, metric in values.items():
                    if not isinstance(metric, torch.Tensor):
                        continue
                    metric_sums[name] = metric_sums.get(name, 0.0) + (
                        float(metric.detach().mean().cpu()) * days
                    )
            observed_days += days
        score = weighted_sum / max(observed_days, 1)
        if observed_days != validation_days or not math.isfinite(score):
            raise R4ProtocolError("cheap validation produced no finite R4 losses")
        if phase == "distribution":
            metrics = {
                name: value / observed_days for name, value in metric_sums.items()
            }
            metrics["total"] = score
            metrics["cheap_validation_days"] = float(observed_days)
            metrics["sample_count"] = float(
                distribution_curriculum_contract(config).cheap_validation_samples
            )
            return metrics
        return score
    return evaluate


def _attach_a1_behavioral_baseline(
    batch: Mapping[str, Any],
    baseline_model: torch.nn.Module,
    horizon: int,
) -> dict[str, Any]:
    """Attach same-batch actions/cost from the frozen selected policy."""

    if baseline_model.training or any(
        parameter.requires_grad for parameter in baseline_model.parameters()
    ):
        raise R4ProtocolError("A1 behavioral baseline must be frozen and in eval mode")
    with torch.inference_mode():
        output = baseline_model.forward_training_rollout(batch, int(horizon))
    rollout = output.get("training_rollout") if isinstance(output, Mapping) else None
    if not isinstance(rollout, Mapping):
        raise R4ProtocolError("selected-policy baseline did not produce a rollout")
    required = ("day_ahead", "battery", "compute", "realized_settlement_cost")
    if any(not isinstance(rollout.get(name), torch.Tensor) for name in required):
        raise R4ProtocolError("selected-policy baseline rollout is incomplete")
    result = dict(batch)
    targets = batch.get("rollout_targets")
    if not isinstance(targets, Mapping):
        raise R4ProtocolError("A1 batch is missing rollout_targets")
    result["rollout_targets"] = {
        **dict(targets),
        "baseline_actions": {
            name: rollout[name].detach()
            for name in ("day_ahead", "battery", "compute")
        },
        "baseline_realized_cost": rollout["realized_settlement_cost"].detach(),
        "baseline_source": "selected_policy_checkpoint",
    }
    return result


def _evidence_evaluator(
    phase_dir: Path,
    phase: str,
    seed: int,
    corpus: PreloadedR4Corpus,
    config: Mapping[str, Any],
    repository_root: str | Path,
    artifact_root: str | Path,
    scenario_training_contract: ScenarioTrainingContract | None = None,
    distribution_support_contract: V6DistributionSupportContract | None = None,
    distribution_training_contract_path: Path | None = None,
    distribution_training_contract_sha256: str | None = None,
    distribution_sampling_contract_path: Path | None = None,
    distribution_sampling_contract_sha256: str | None = None,
) -> Any | None:
    """Register only already-materialized full hard-gate evidence.

    Training may run without this artifact, but the phase then terminates
    truthfully as NOT_READY/FAILED rather than inferring capability from loss.
    A separate evaluator materializes this file from raw predictions/runtime
    records after the trained candidate exists.
    """

    if phase == "representation":
        return LiveRepresentationCapabilityEvaluator(
            corpus, phase_dir, seed=int(seed)
        )
    if phase == "scenario":
        if scenario_training_contract is None:
            raise R4ProtocolError("Scenario capability gate lacks training contract")
        return ScenarioCapabilityMaterializer(
            config,
            repository_root,
            artifact_root,
            corpus=corpus,
            training_contract=scenario_training_contract,
            seed=int(seed),
        )
    if phase == "distribution":
        if (
            scenario_training_contract is None
            or distribution_support_contract is None
            or distribution_training_contract_path is None
            or distribution_training_contract_sha256 is None
            or distribution_sampling_contract_path is None
            or distribution_sampling_contract_sha256 is None
        ):
            raise R4ProtocolError(
                "Distribution Level2 gate lacks its preflight-frozen contracts"
            )
        support_payload = read_json(distribution_support_contract.path)
        support = DistributionSupportContract(
            distribution_support_contract.path,
            distribution_support_contract.artifact_sha256,
            support_payload,
        )
        return DistributionCapabilityMaterializer(
            config,
            repository_root,
            artifact_root,
            corpus=corpus,
            training_contract=scenario_training_contract,
            distribution_support_contract=support,
            distribution_training_contract_path=distribution_training_contract_path,
            distribution_training_contract_sha256=(
                distribution_training_contract_sha256
            ),
            sampling_contract_path=distribution_sampling_contract_path,
            sampling_contract_sha256=distribution_sampling_contract_sha256,
            seed=int(seed),
        )
    if phase in {"policy", "joint_policy", "a1", "a2"}:
        return LiveDecisionCapabilityEvaluator(
            corpus,
            phase_dir,
            config=config,
            repository_root=repository_root,
            seed=int(seed),
        )
    path = phase_dir / "capability_daily.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise R4ProtocolError(f"invalid capability daily evidence: {path}")
    baseline = payload.get("baseline") if isinstance(payload, Mapping) else None
    return build_capability_evaluator(
        phase, rows, baseline=baseline, seed=int(seed)
    )


def _adopt_v6_representation(
    context: ProtocolContext,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> Mapping[str, Any]:
    """Adopt the already-passed epoch-28 evidence without data/model training."""

    manifest_path = (
        Path(context.output_root) / "preflight" / "frozen_preflight_manifest.json"
    ).resolve()
    if not manifest_path.is_file():
        raise R4ProtocolError("V6 Representation adoption requires frozen preflight")
    manifest = read_json(manifest_path)
    if manifest.get("config_sha256") != context.config_sha256:
        raise R4ProtocolError("V6 Representation adoption config binding changed")
    checkpoint_path, checkpoint_sha = _frozen_manifest_entry(
        manifest, "source_checkpoint"
    )
    normalization_path, normalization_sha = _frozen_manifest_entry(
        manifest, "normalization"
    )
    if isinstance(manifest.get("source_representation_state"), Mapping):
        state_path, state_sha = _frozen_manifest_entry(
            manifest, "source_representation_state"
        )
    else:
        historical = manifest.get("historical_lineage")
        if not isinstance(historical, Mapping):
            raise R4ProtocolError(
                "V6 Representation adoption lacks frozen state lineage"
            )
        state_path = Path(
            str(historical.get("representation_state_path", ""))
        ).resolve()
        state_sha = str(
            historical.get("representation_state_sha256", "")
        ).lower()
        if (
            not state_path.is_file()
            or len(state_sha) != 64
            or sha256_file(state_path) != state_sha
        ):
            raise R4ProtocolError(
                "V6 Representation state historical-lineage binding changed"
            )
    state = read_json(state_path)
    gate = state.get("capability_gate")
    if (
        state.get("phase") != "representation"
        or int(state.get("seed", -1))
        != int(config["protocol"]["active_seed"])
        or int(state.get("selected_candidate_epoch", -1)) != 28
        or int(state.get("epochs_completed", -1)) < 28
        or state.get("capability_gate_status") != "PASSED"
        or not isinstance(gate, Mapping)
        or gate.get("status") != "PASS"
        or int(gate.get("evaluated_days", -1)) != 179
        or len(tuple(gate.get("dates", ()))) != 179
    ):
        raise R4ProtocolError(
            "V6 Representation adoption lacks the epoch-28 full 179-day PASS gate"
        )
    if state.get("normalization_sha256") != normalization_sha:
        raise R4ProtocolError("V6 adopted Representation normalization binding changed")
    configured_checkpoint = _resolved_config_path(
        config,
        Path(context.repository_root),
        ("lineage", "representation_selected_checkpoint"),
    )
    if configured_checkpoint != checkpoint_path:
        raise R4ProtocolError("V6 adopted Representation checkpoint differs from config")
    result = {
        "status": "COMPLETE",
        "phase": "representation",
        "seed": int(seed),
        "epochs_completed": 28,
        "selected_candidate_epoch": 28,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_role": "inherited_capability_passed_selected_best",
        "capability_gate": dict(gate),
        "capability_gate_status": "PASSED",
        "adoption_mode": "reuse_without_retraining",
        "optimizer_steps": 0,
        "trainable_parameters": [],
        "normalization_artifact": str(normalization_path),
        "normalization_sha256": normalization_sha,
        "source_representation_state": str(state_path),
        "source_representation_state_sha256": state_sha,
        "architecture_version": "continuous_distributional_joint_moe_r4_v6",
        "experiment_id": str(config["project"]["experiment_id"]),
        "training_semantics_revision": str(
            config["project"]["training_semantics_revision"]
        ),
        "method": "continuous_distributional_world_joint_moe_r4_v6",
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "scientific_completion": True,
    }
    phase_store = AtomicRunStore(
        _phase_directory(context, "representation", seed),
        lineage_id=context.lineage_id,
        config_sha256=context.config_sha256,
        config_path=context.config_path,
    )
    phase_store.write_state(result)
    return result


def default_phase_runner(
    *,
    context: ProtocolContext,
    phase: str,
    seed: int,
    resume: bool = True,
) -> Mapping[str, Any]:
    config = load_r4_config(context.config_path)
    is_distributional = (
        str(config.get("project", {}).get("architecture_version", ""))
        == "continuous_distributional_joint_moe_r4_v6"
    )
    if is_distributional and phase == "representation":
        return _adopt_v6_representation(context, config, seed=int(seed))
    phase_dir = _phase_directory(context, phase, seed)
    phase_dir.mkdir(parents=True, exist_ok=True)
    corpus = PreloadedR4Corpus(config, context.repository_root)
    representation_scales, scenario_feature_scales = _fit_train_only_loss_scales(
        corpus
    )
    statistics_payload = _train_only_statistics_payload(
        corpus, config, representation_scales, scenario_feature_scales
    )
    statistics_path = (
        _resolved_config_path(
            config,
            Path(context.repository_root),
            ("lineage", "train_only_normalization"),
        )
        if is_distributional
        else Path(context.output_root) / "data" / "train_only_normalization.json"
    )
    if statistics_path.is_file():
        persisted_statistics = read_json(statistics_path)
        if persisted_statistics != statistics_payload:
            raise R4ProtocolError(
                "refusing to overwrite a changed Representation normalization artifact"
            )
    elif phase == "representation":
        atomic_json(statistics_path, statistics_payload)
    else:
        raise R4ProtocolError(
            "Scenario and decision phases require the existing Representation normalization"
        )
    model = build_formal_r4_model(config, context.repository_root)
    hard_limit = int(config["model"]["parameter_budget"]["hard_stop_above_trainable"])
    if model.parameter_count > hard_limit:
        raise R4ProtocolError(
            f"R4 parameter count {model.parameter_count} exceeds hard limit {hard_limit}"
        )
    v6_setup: V6FrozenDistributionSetup | None = None
    if is_distributional:
        v6_setup = _load_v6_frozen_distribution_setup(context, config, model)
        normalization_sha256 = v6_setup.normalization_sha256
    else:
        normalization_sha256 = sha256_file(statistics_path)
    predecessor = _predecessor(
        context,
        phase,
        seed,
        config=config,
        normalization_sha256=normalization_sha256,
    )
    # V6 Distribution already imported only the epoch-28 Representation trunk
    # through the preflight allowlist.  Later V6 phases load the complete
    # selected predecessor from the same architecture.  The V5 path remains
    # strict/full-state for archived reproducibility.
    if not (is_distributional and phase == "distribution"):
        _load_predecessor(
            model,
            predecessor,
            normalization_sha256=normalization_sha256,
        )
    behavioral_baseline: torch.nn.Module | None = None
    if phase == "a1":
        if predecessor is None:
            raise R4ProtocolError("A1 requires the selected policy checkpoint")
        behavioral_baseline = copy.deepcopy(model).eval()
        behavioral_baseline.requires_grad_(False)
    device = _device()
    model.to(device)
    if behavioral_baseline is not None:
        behavioral_baseline.to(device).eval()
    scenario_training_contract: ScenarioTrainingContract | None = None
    if is_distributional:
        assert v6_setup is not None
        scenario_training_contract = v6_setup.residual_contract
        scenario_feature_scales = [
            float(value) for value in scenario_training_contract.score_scales
        ]
    elif phase != "representation":
        if predecessor is None:
            raise R4ProtocolError(f"phase {phase} requires a selected predecessor")
        contract_path = Path(context.output_root) / "data" / "scenario_training_contract.json"
        scenario_gate = config.get("capability_gates", {}).get("scenario", {})
        if not isinstance(scenario_gate, Mapping):
            raise R4ProtocolError("capability_gates.scenario must be a mapping")
        configured_baseline = scenario_gate.get("frozen_baseline_artifact")
        baseline_path = (
            Path(context.output_root)
            / "capability"
            / "scenario"
            / "frozen_training_residual_baseline.pt"
            if configured_baseline is None
            else (
                Path(configured_baseline)
                if Path(str(configured_baseline)).is_absolute()
                else Path(context.repository_root) / str(configured_baseline)
            )
        ).resolve()
        expected_baseline_sha = scenario_gate.get("frozen_baseline_sha256")
        if not baseline_path.is_file():
            raise R4ProtocolError(
                f"frozen Scenario residual baseline is missing: {baseline_path}"
            )
        actual_baseline_sha = sha256_file(baseline_path)
        if (
            expected_baseline_sha is not None
            and actual_baseline_sha != str(expected_baseline_sha).lower()
        ):
            raise R4ProtocolError("frozen Scenario residual baseline SHA-256 changed")
        try:
            if phase == "scenario" and not contract_path.is_file():
                scenario_training_contract = materialize_scenario_training_contract(
                    model,
                    corpus,
                    predecessor,
                    baseline_path,
                    contract_path,
                    device=device,
                    batch_size=128,
                    expected_baseline_sha256=actual_baseline_sha,
                )
            else:
                scenario_training_contract = load_scenario_training_contract(
                    contract_path,
                    expected_representation_sha256=(
                        sha256_file(predecessor) if phase == "scenario" else None
                    ),
                    expected_normalization_sha256=normalization_sha256,
                    expected_baseline_sha256=actual_baseline_sha,
                )
        except ScenarioTrainingContractError as exc:
            raise R4ProtocolError(str(exc)) from exc
        setter = getattr(model.scenario_generator, "set_default_residual_caps", None)
        if not callable(setter):
            raise R4ProtocolError(
                "Scenario generator lacks set_default_residual_caps contract API"
            )
        setter(scenario_training_contract.raw_p99_caps)
        if behavioral_baseline is not None:
            baseline_setter = getattr(
                behavioral_baseline.scenario_generator,
                "set_default_residual_caps",
                None,
            )
            if not callable(baseline_setter):
                raise R4ProtocolError(
                    "A1 behavioral baseline lacks Scenario cap contract API"
                )
            baseline_setter(scenario_training_contract.raw_p99_caps)
            learned_caps = model.scenario_generator.default_raw_unit_residual_caps_zonal
            baseline_caps = (
                behavioral_baseline.scenario_generator
                .default_raw_unit_residual_caps_zonal
            )
            if not torch.equal(learned_caps, baseline_caps):
                raise R4ProtocolError(
                    "A1 learned and behavioral models use different Scenario caps"
                )
        scenario_feature_scales = [
            float(value) for value in scenario_training_contract.score_scales
        ]
    model.configure_trainable_phase(phase)
    phase_config = config["training"]["phases"][phase]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(phase_config["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    distribution_support_lower: torch.Tensor | None = None
    distribution_support_upper: torch.Tensor | None = None
    distribution_residual_caps: torch.Tensor | None = None
    if v6_setup is not None:
        (
            distribution_support_lower,
            distribution_support_upper,
            distribution_residual_caps,
        ) = v6_setup.support_contract.tensor_payload(device)
    physical_batch = _choose_physical_batch(
        model,
        corpus,
        phase,
        config,
        device,
        behavioral_baseline=behavioral_baseline,
        representation_scales=representation_scales,
        scenario_feature_scales=scenario_feature_scales,
        scenario_training_contract=scenario_training_contract,
        distribution_support_lower=distribution_support_lower,
        distribution_support_upper=distribution_support_upper,
        distribution_support_sha256=(
            None if v6_setup is None else v6_setup.support_contract.artifact_sha256
        ),
        distribution_training_contract_path=(
            None
            if v6_setup is None
            else v6_setup.distribution_training_contract_path
        ),
        distribution_training_contract_sha256=(
            None
            if v6_setup is None
            else v6_setup.distribution_training_contract_sha256
        ),
        distribution_residual_caps=distribution_residual_caps,
        pretraining_probe_path=(
            phase_dir / "pretraining_probe.json"
            if is_distributional and phase == "distribution"
            else None
        ),
    )
    scheduler = _build_scheduler(optimizer, phase, config)
    store = AtomicRunStore(
        phase_dir,
        lineage_id=context.lineage_id,
        config_sha256=context.config_sha256,
        config_path=context.config_path,
    )
    evaluator = _evidence_evaluator(
        phase_dir,
        phase,
        seed,
        corpus,
        config,
        context.repository_root,
        context.output_root,
        scenario_training_contract=scenario_training_contract,
        distribution_support_contract=(
            None if v6_setup is None else v6_setup.support_contract
        ),
        distribution_training_contract_path=(
            None
            if v6_setup is None
            else v6_setup.distribution_training_contract_path
        ),
        distribution_training_contract_sha256=(
            None
            if v6_setup is None
            else v6_setup.distribution_training_contract_sha256
        ),
        distribution_sampling_contract_path=(
            None if v6_setup is None else v6_setup.sampling_contract_path
        ),
        distribution_sampling_contract_sha256=(
            None if v6_setup is None else v6_setup.sampling_contract_sha256
        ),
    )
    trainer = R4PhaseTrainer(
        model,
        optimizer,
        store=store,
        cheap_validator=_cheap_validator(
            corpus,
            config,
            device,
            behavioral_baseline=behavioral_baseline,
            representation_scales=representation_scales,
            scenario_feature_scales=scenario_feature_scales,
            scenario_training_contract=scenario_training_contract,
            physical_batch_size=physical_batch,
            distribution_support_lower=distribution_support_lower,
            distribution_support_upper=distribution_support_upper,
            distribution_support_sha256=(
                None
                if v6_setup is None
                else v6_setup.support_contract.artifact_sha256
            ),
            distribution_training_contract_path=(
                None
                if v6_setup is None
                else v6_setup.distribution_training_contract_path
            ),
            distribution_training_contract_sha256=(
                None
                if v6_setup is None
                else v6_setup.distribution_training_contract_sha256
            ),
            distribution_residual_caps=distribution_residual_caps,
        ),
        phase_end_evaluator=evaluator,
        physical_batch_size=physical_batch,
        cheap_validation_policy=CheapValidationPolicy(max_days=16),
        expected_training_dates=[
            str(item["operating_date"]) for item in corpus.records("train")
        ],
        predecessor_checkpoint=predecessor,
        predecessor_sha256=(
            v6_setup.predecessor_sha256
            if v6_setup is not None and phase == "distribution"
            else None
        ),
        scheduler=scheduler,
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        normalization_sha256=normalization_sha256,
        scenario_training_contract_path=(
            None
            if scenario_training_contract is None or is_distributional
            else scenario_training_contract.path
        ),
        scenario_training_contract_sha256=(
            None
            if scenario_training_contract is None or is_distributional
            else scenario_training_contract.artifact_sha256
        ),
        distribution_training_contract_path=(
            None
            if v6_setup is None
            else v6_setup.distribution_training_contract_path
        ),
        distribution_training_contract_sha256=(
            None
            if v6_setup is None
            else v6_setup.distribution_training_contract_sha256
        ),
        distribution_contract_preverified=v6_setup is not None,
    )
    loss = build_phase_loss(
        phase,
        config,
        fixed_training_cost_scale=(
            corpus.fixed_training_cost_scale
            if phase in {"policy", "joint_policy", "a1", "a2"}
            else None
        ),
        representation_scales=representation_scales,
        scenario_feature_scales=scenario_feature_scales,
    )

    def batches(epoch: int, horizon: int):
        for batch in corpus.batches(
            "train",
            physical_batch,
            device=device,
            shuffle_seed=int(seed) + int(epoch),
            origin=0,
        ):
            if is_distributional and phase != "representation":
                if (
                    v6_setup is None
                    or distribution_support_lower is None
                    or distribution_support_upper is None
                    or distribution_residual_caps is None
                ):
                    raise R4ProtocolError(
                        f"phase {phase} training batch lacks frozen V6 contracts"
                    )
                batch = _attach_v6_distribution_contract(
                    batch,
                    support_lower=distribution_support_lower,
                    support_upper=distribution_support_upper,
                    support_sha256=v6_setup.support_contract.artifact_sha256,
                    distribution_contract_path=(
                        v6_setup.distribution_training_contract_path
                    ),
                    distribution_contract_sha256=(
                        v6_setup.distribution_training_contract_sha256
                    ),
                    residual_caps=distribution_residual_caps,
                )
                batch["distribution_train_samples"] = int(
                    config["sampling"]["train"]
                )
            elif phase != "representation":
                if scenario_training_contract is None:
                    raise R4ProtocolError(
                        f"phase {phase} training batch lacks Scenario contract"
                    )
                batch = attach_scenario_training_contract(
                    batch, scenario_training_contract, device=device
                )
            if phase == "a1":
                assert behavioral_baseline is not None
                batch = _attach_a1_behavioral_baseline(
                    batch, behavioral_baseline, horizon
                )
            yield batch

    result = trainer.run(phase, seed, batches, loss, resume=resume)
    return {
        **asdict(result),
        "physical_batch_size": physical_batch,
        "configured_effective_batch_size": 543 if is_distributional else 512,
        "scientific_effective_batch_days": 543,
        "legacy_configured_effective_batch_size": 512 if is_distributional else None,
        "optimizer_effective_batch_days": 543,
        "parameter_count": model.parameter_count,
        "normalization_artifact": str(statistics_path.resolve()),
        "normalization_sha256": normalization_sha256,
        "scenario_training_contract_path": (
            None
            if scenario_training_contract is None or is_distributional
            else str(scenario_training_contract.path)
        ),
        "scenario_training_contract_sha256": (
            None
            if scenario_training_contract is None or is_distributional
            else scenario_training_contract.artifact_sha256
        ),
        "distribution_training_contract_path": (
            None
            if v6_setup is None
            else str(v6_setup.distribution_training_contract_path)
        ),
        "distribution_training_contract_sha256": (
            None
            if v6_setup is None
            else v6_setup.distribution_training_contract_sha256
        ),
        "train_distribution_samples": (
            None if v6_setup is None else int(config["sampling"]["train"])
        ),
        "cheap_validation_distribution_samples": (
            None
            if v6_setup is None
            else int(config["sampling"]["cheap_validation"])
        ),
        "predecessor_checkpoint": None if predecessor is None else str(predecessor),
        "formal_selection_2024_opened": False,
        "locked_test_2025_opened": False,
        "scientific_completion": result.status == "COMPLETE",
    }


def register_default_phase_runner() -> None:
    """Explicit CLI hook; importing formal_protocol alone remains fail closed."""

    register_phase_runner(default_phase_runner)


__all__ = [
    "default_phase_runner",
    "prepare_v6_distribution_preflight_contracts",
    "register_default_phase_runner",
]
