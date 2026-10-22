"""Strict governing configuration contract for the formal R4 program."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .hashing import sha256_file, sha256_json


ARCHITECTURE_VERSION = "hybrid_cfc_transformer_moe_r4"
EXPERIMENT_ID = "e2e_moe_hybrid_r4_formal_v1"
TRAINING_SEMANTICS_REVISION = "r4_paper_paired_nmae_physics_linked_pue_v5"
FORMAL_OUTPUT_ROOT = "artifacts/e2e_moe_hybrid_r4_v5_formal"
DISTRIBUTIONAL_ARCHITECTURE_VERSION = "continuous_distributional_joint_moe_r4_v6"
DISTRIBUTIONAL_EXPERIMENT_ID = "e2e_moe_distributional_r4_v6_formal_v1"
DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION = (
    "r4_distributional_world_latent_variable_n_curriculum_cycle2_v1"
)
DISTRIBUTIONAL_FORMAL_OUTPUT_ROOT = "artifacts/e2e_moe_distributional_r4_v6_formal"
DISTRIBUTIONAL_PHASE_ORDER = (
    "representation",
    "distribution",
    "joint_policy",
    "a1",
    "a2",
)
FORMAL_SEEDS = (20240813, 20240829, 20240917)
PHASE_ORDER = ("representation", "scenario", "policy", "a1", "a2")
FORBIDDEN_ACTIVE_IMPORTS = frozenset(
    {"pyomo", "gurobipy", "cvxpy", "cvxpylayers", "optnet"}
)


class R4ConfigError(ValueError):
    """Raised before any data is opened when the R4 contract is violated."""


def load_r4_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one formal R4 YAML without mutating the source."""

    source = Path(path).resolve()
    if not source.is_file():
        raise R4ConfigError(f"R4 config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise R4ConfigError("R4 config root must be a mapping")
    config = deepcopy(dict(payload))
    validate_r4_config(config)
    config["_config_path"] = str(source)
    config["_config_sha256"] = sha256_file(source)
    return config


def validate_r4_config(config: Mapping[str, Any]) -> None:
    """Fail closed on every invariant that defines the formal R4 method."""

    project = config.get("project")
    if isinstance(project, Mapping) and project.get("architecture_version") == DISTRIBUTIONAL_ARCHITECTURE_VERSION:
        _validate_distributional_r4_config(config)
        return

    _expect(config, "schema_version", "dcvpp.e2e_moe_hybrid_r4.formal.v1")
    _expect(config, "project.version", "R4")
    _expect(config, "project.architecture_version", ARCHITECTURE_VERSION)
    _expect(config, "project.training_semantics_revision", TRAINING_SEMANTICS_REVISION)
    _expect(config, "project.experiment_id", EXPERIMENT_ID)
    _expect(config, "project.output_root", FORMAL_OUTPUT_ROOT)
    _expect(config, "project.preserve_r3_read_only", True)
    _expect(config, "project.initialize_from_r3", False)

    _expect(config, "protocol.seeds", list(FORMAL_SEEDS))
    _expect(config, "protocol.active_seed", FORMAL_SEEDS[0])
    _expect(config, "protocol.mainline_first", True)
    _expect(config, "protocol.continue_after_seed1_only_on", "R4_MAINLINE_PASS")
    _expect(config, "protocol.splits.train.start", "2022-01-01")
    _expect(config, "protocol.splits.train.end", "2023-06-30")
    _expect(config, "protocol.splits.validation.start", "2023-07-01")
    _expect(config, "protocol.splits.validation.end", "2023-12-31")
    _expect(config, "protocol.splits.final_temporal_test.start", "2024-01-01")
    _expect(config, "protocol.splits.final_temporal_test.end", "2024-12-31")
    _expect(
        config,
        "protocol.splits.final_temporal_test.sealed_until_all_methods_frozen",
        True,
    )
    _expect(config, "protocol.splits.final_temporal_test.execute_once", True)
    _expect(config, "protocol.splits.locked.year", 2025)
    _expect(config, "protocol.splits.locked.reject_path_before_open", True)

    for field in (
        "data.frozen_physical_twin",
        "data.frozen_physical_surrogate",
        "data.physical_calibration_manifest",
    ):
        value = _get(config, field)
        if not isinstance(value, str) or not value.strip():
            raise R4ConfigError(f"{field} must be one explicit artifact path")

    _expect(config, "model.market.input_features", 16)
    _expect(config, "model.market.history_tokens", 7)
    _expect(config, "model.market.total_tokens", 103)
    _expect(config, "model.market.transformer_layers", 2)
    _expect(config, "model.market.attention_heads", 4)
    _expect(config, "model.market.feedforward_width", 384)
    _expect(config, "model.latent_width", 128)
    _expect(config, "model.moe.experts", 8)
    _expect(config, "model.moe.top_k", 2)
    _expect(config, "model.moe.token_level", True)
    _expect(config, "model.moe.expert_hidden_width", 256)
    _expect(config, "model.scenarios.count", 20)
    _expect(config, "model.scenarios.formal_output_count_fixed", 20)
    _expect(config, "model.scenarios.weighted", True)
    _expect(config, "model.forecasts.exact_identity", "rt = da - dart")
    _expect(config, "model.policy.freeze_day_ahead_after_origin_zero", True)
    _expect(config, "model.policy.empirical_migration", "fixed_zero")
    _expect(
        config,
        "model.physical.dynamic_features",
        ["pv_issue_mean_z4", "pv_issue_std_z4", "issue_utilization_z4"],
    )

    _expect(config, "runtime.external_solver", False)
    _expect(config, "runtime.rolling_origins", 96)
    _expect(config, "runtime.independent_validator", True)
    configured_forbidden = {
        str(name).lower() for name in _get(config, "runtime.forbidden_active_imports")
    }
    if configured_forbidden != FORBIDDEN_ACTIVE_IMPORTS:
        raise R4ConfigError(
            "runtime.forbidden_active_imports must exactly register the R4 solver ban"
        )

    _expect(config, "training.optimizer", "adamw")
    _expect(config, "training.effective_batch_size_days", 512)
    _expect(config, "training.optimizer_effective_batch_days", 543)
    _expect(
        config,
        "training.gradient_accumulation",
        "one_step_per_complete_543_day_coverage_cycle",
    )
    _expect(
        config,
        "training.physical_microbatch_fallback",
        [512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
    )
    _expect(config, "training.unbiased_partial_tail_normalization", True)
    _expect(config, "training.full_hard_validation_during_gradient_epochs", False)
    _expect(config, "training.phases.representation.max_epochs", 800)
    _expect(config, "training.phases.representation.patience", 30)
    _expect(config, "training.phases.scenario.max_epochs", 200)
    _expect(config, "training.phases.scenario.patience", 15)
    _expect(config, "training.phases.policy.max_epochs", 320)
    _expect(config, "training.phases.policy.k96_patience", 20)
    _expect(config, "training.phases.a1.max_epochs", 120)
    _expect(config, "training.phases.a1.patience", 15)
    _expect(config, "training.phases.a2.max_epochs", 180)
    _expect(config, "training.phases.a2.patience", 20)
    for phase in ("representation", "scenario", "policy", "a1", "a2"):
        weights = _get(config, f"training.phases.{phase}.loss_weights")
        if not isinstance(weights, Mapping) or not weights:
            raise R4ConfigError(f"phase {phase} requires explicit loss_weights")
        if any(not isinstance(value, (int, float)) or float(value) < 0 for value in weights.values()):
            raise R4ConfigError(f"phase {phase} loss weights must be nonnegative numbers")
    curriculum = _get(config, "training.phases.policy.curriculum")
    expected_curriculum = ((8, 40), (24, 60), (48, 80), (96, 140))
    actual_curriculum = tuple(
        (int(stage["origins"]), int(stage["max_epochs"])) for stage in curriculum
    )
    if actual_curriculum != expected_curriculum:
        raise R4ConfigError(
            f"policy curriculum must be {expected_curriculum}, got {actual_curriculum}"
        )
    if sum(limit for _, limit in actual_curriculum) != 320:
        raise R4ConfigError("policy curriculum must sum to 320 maximum epochs")

    _expect(config, "capability_gates.common.validation_dates_expected", 179)
    _expect(config, "capability_gates.common.bootstrap.block_days", 7)
    _expect(config, "capability_gates.common.bootstrap.replicates", 10000)
    _expect(
        config,
        "capability_gates.representation.decision_rule",
        "paired_daily_nmae_model_minus_baseline_one_sided_95_upper_lt_zero",
    )
    _expect(
        config,
        "capability_gates.representation.noninferiority_margin_fraction",
        0.0,
    )
    _expect(config, "capability_gates.scenario.minimum_effective_sample_size", 10.0)
    _expect(config, "capability_gates.policy.validation_origins", 17184)
    _expect(config, "capability_gates.policy.feasibility_fraction", 1.0)
    _expect(config, "capability_gates.policy.maximum_non_exogenous_rejected_origins", 0)
    _expect(config, "statistics.primary_unit", "day")
    _expect(config, "statistics.replicates", 10000)
    _expect(config, "statistics.block_days", 7)

    minimum = int(_get(config, "model.parameter_budget.expected_min_trainable"))
    expected_max = int(_get(config, "model.parameter_budget.expected_max_trainable"))
    hard_max = int(_get(config, "model.parameter_budget.hard_stop_above_trainable"))
    if not 0 < minimum <= expected_max < hard_max == 5_000_000:
        raise R4ConfigError("invalid R4 parameter budget ordering")


def _validate_distributional_r4_config(config: Mapping[str, Any]) -> None:
    """Validate the independent V6 continuous-distribution formal contract.

    V5 remains a read-only, reproducible historical lineage.  V6 deliberately
    shares only its capability-passed Representation and train-only contracts;
    the fixed-particle scenario/risk/policy heads are not predecessors.
    """

    _expect(config, "schema_version", "dcvpp.e2e_moe_distributional_r4.formal.v1")
    _expect(config, "project.version", "R4-V6")
    _expect(config, "project.architecture_version", DISTRIBUTIONAL_ARCHITECTURE_VERSION)
    _expect(
        config,
        "project.training_semantics_revision",
        DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION,
    )
    _expect(config, "project.experiment_id", DISTRIBUTIONAL_EXPERIMENT_ID)
    _expect(config, "project.output_root", DISTRIBUTIONAL_FORMAL_OUTPUT_ROOT)
    _expect(config, "project.preserve_v5_read_only", True)
    _expect(config, "project.reuse_selected_representation_only", True)
    _expect(config, "project.resume_fixed_particle_scenario", False)

    _expect(config, "protocol.seeds", list(FORMAL_SEEDS))
    _expect(config, "protocol.active_seed", FORMAL_SEEDS[0])
    _expect(config, "protocol.phase_order", list(DISTRIBUTIONAL_PHASE_ORDER))
    _expect(config, "protocol.splits.train.start", "2022-01-01")
    _expect(config, "protocol.splits.train.end", "2023-06-30")
    _expect(config, "protocol.splits.validation.start", "2023-07-01")
    _expect(config, "protocol.splits.validation.end", "2023-12-31")
    _expect(config, "protocol.splits.final_temporal_test.start", "2024-01-01")
    _expect(config, "protocol.splits.final_temporal_test.end", "2024-12-31")
    _expect(
        config,
        "protocol.splits.final_temporal_test.sealed_until_all_methods_frozen",
        True,
    )
    _expect(config, "protocol.splits.final_temporal_test.execute_once", True)
    _expect(config, "protocol.splits.locked.year", 2025)
    _expect(config, "protocol.splits.locked.reject_path_before_open", True)

    for field in (
        "data.selection_bundle_root",
        "data.official_settlement_root",
        "data.frozen_physical_twin",
        "data.frozen_physical_surrogate",
        "data.physical_calibration_manifest",
        "lineage.representation_selected_checkpoint",
        "lineage.train_only_normalization",
        "lineage.v5_failed_scenario_state",
        "lineage.distribution_support_artifact",
        "lineage.residual_score_contract",
    ):
        value = _get(config, field)
        if not isinstance(value, str) or not value.strip():
            raise R4ConfigError(f"{field} must be one explicit artifact path")
    _expect(config, "lineage.allow_representation_prefixes_only", True)
    _expect(config, "lineage.reject_legacy_scenario_risk_policy_heads", True)
    _expect(config, "lineage.distribution_support_hash_source", "frozen_preflight_manifest")

    _expect(config, "model.representation_base_architecture", ARCHITECTURE_VERSION)
    _expect(config, "model.latent_width", 128)
    _expect(config, "model.moe.experts", 8)
    _expect(config, "model.moe.top_k", 2)
    _expect(config, "model.distribution.representation_capacity_independent_of_sample_count", True)
    _expect(config, "model.distribution.continuous_residual_latent", True)
    _expect(config, "model.distribution.variable_sample_count", True)
    _expect(config, "model.distribution.policy_requires_samples", False)
    _expect(config, "model.distribution.exact_identity", "rt = da - dart")
    _expect(config, "model.policy.empirical_migration", "fixed_zero")

    sampling = _get(config, "sampling")
    if not isinstance(sampling, Mapping):
        raise R4ConfigError("sampling must be a mapping")
    _expect(config, "sampling.train", 32)
    _expect(config, "sampling.cheap_validation", 64)
    _expect(config, "sampling.formal_probability", 256)
    _expect(config, "sampling.formal_empirical_risk", 512)
    _expect(config, "sampling.ablation", [20, 64, 128, 256, 512])

    _expect(config, "runtime.external_solver", False)
    _expect(config, "runtime.rolling_origins", 96)
    _expect(config, "runtime.independent_validator", True)
    configured_forbidden = {
        str(name).lower() for name in _get(config, "runtime.forbidden_active_imports")
    }
    if configured_forbidden != FORBIDDEN_ACTIVE_IMPORTS:
        raise R4ConfigError(
            "runtime.forbidden_active_imports must exactly register the solver ban"
        )

    _expect(config, "training.optimizer", "adamw")
    _expect(config, "training.physical_microbatch_days", 32)
    _expect(config, "training.optimizer_effective_batch_days", 543)
    _expect(
        config,
        "training.gradient_accumulation",
        "one_step_per_complete_543_day_coverage_cycle",
    )
    _expect(config, "training.full_hard_validation_during_gradient_epochs", False)
    stages = _get(config, "training.phases.distribution.curriculum")
    if not isinstance(stages, list) or [stage.get("id") for stage in stages] != [
        "S1", "S2", "S3", "S4", "S5"
    ]:
        raise R4ConfigError("distribution curriculum must be ordered S1..S5")
    total_max = 0
    required_losses = {
        "center_scale", "energy_score", "marginal_crps", "coverage90", "coverage95",
        "interval", "tail", "dependence", "diversity", "identity",
    }
    for stage in stages:
        minimum = int(stage.get("min_epochs", 0))
        maximum = int(stage.get("max_epochs", 0))
        patience = int(stage.get("patience", 0))
        if not 0 < minimum <= maximum or patience <= 0:
            raise R4ConfigError(f"invalid distribution curriculum stage: {stage}")
        weights = stage.get("loss_weights")
        if not isinstance(weights, Mapping) or set(weights) != required_losses:
            raise R4ConfigError(
                f"stage {stage.get('id')} must define exactly {sorted(required_losses)}"
            )
        if any(not isinstance(v, (int, float)) or float(v) < 0 for v in weights.values()):
            raise R4ConfigError("distribution loss weights must be nonnegative")
        total_max += maximum
    _expect(config, "training.phases.distribution.max_epochs", total_max)
    _expect(config, "training.phases.distribution.train_samples", 32)
    _expect(config, "training.phases.distribution.cheap_validation_samples", 64)
    _expect(
        config,
        "training.phases.distribution.candidate_selection.level2",
        "full_frozen_179_day_distribution_gate",
    )
    _expect(config, "training.phases.distribution.candidate_selection.level3", "pass_only")

    _expect(config, "capability_gates.common.validation_dates_expected", 179)
    _expect(config, "capability_gates.distribution.formal_sample_count", 256)
    _expect(config, "capability_gates.distribution.formal_empirical_risk_samples", 512)
    _expect(config, "capability_gates.distribution.coverage_levels", [0.90, 0.95])
    _expect(config, "capability_gates.distribution.max_absolute_coverage_error", 0.03)
    _expect(config, "capability_gates.distribution.require_no_future_leakage", True)
    _expect(config, "capability_gates.distribution.require_identity_tolerance", 1.0e-6)
    _expect(config, "capability_gates.distribution.require_support_bounds", True)
    _expect(config, "capability_gates.distribution.require_finite", True)

    _expect(config, "audit.hash_policy", "freeze_once_then_manifest_compare")
    _expect(config, "audit.full_gate_frequency", "level2_candidates_only")
    _expect(config, "audit.hash_large_checkpoint", "once_at_candidate_write")


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove loader metadata before serializing a governing config copy."""

    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def canonical_config_digest(config: Mapping[str, Any]) -> str:
    return sha256_json(public_config(config))


def _get(config: Mapping[str, Any], dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise R4ConfigError(f"missing required R4 config field: {dotted}")
        current = current[part]
    return current


def _expect(config: Mapping[str, Any], dotted: str, expected: Any) -> None:
    actual = _get(config, dotted)
    if actual != expected:
        raise R4ConfigError(
            f"R4 config field {dotted!r} must equal {expected!r}, got {actual!r}"
        )


__all__ = [
    "ARCHITECTURE_VERSION",
    "DISTRIBUTIONAL_ARCHITECTURE_VERSION",
    "DISTRIBUTIONAL_EXPERIMENT_ID",
    "DISTRIBUTIONAL_FORMAL_OUTPUT_ROOT",
    "DISTRIBUTIONAL_PHASE_ORDER",
    "DISTRIBUTIONAL_TRAINING_SEMANTICS_REVISION",
    "EXPERIMENT_ID",
    "FORBIDDEN_ACTIVE_IMPORTS",
    "FORMAL_SEEDS",
    "FORMAL_OUTPUT_ROOT",
    "PHASE_ORDER",
    "R4ConfigError",
    "TRAINING_SEMANTICS_REVISION",
    "canonical_config_digest",
    "load_r4_config",
    "public_config",
    "validate_r4_config",
]
