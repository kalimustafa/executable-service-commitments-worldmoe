"""Fail-closed configuration contract for VPP-WorldMoE R5-V1/R5-V2.x.

The new namespace deliberately has no dependency on the R4-V8 model contract.
It reuses only the registered causal daily corpus and frozen physical evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


LEGACY_SCHEMA_VERSION = "dcvpp.vpp_world_moe.r5.v1"
SCHEMA_VERSION = "dcvpp.vpp_world_moe.r5.v2"
R5_V21_SCHEMA_VERSION = "dcvpp.vpp_world_moe.r5.v2.1"
R5_V22_SCHEMA_VERSION = "dcvpp.vpp_world_moe.r5.v2.2"
R5_V23_SCHEMA_VERSION = "dcvpp.vpp_world_moe.r5.v2.3"
R5_V24_SCHEMA_VERSION = "dcvpp.vpp_world_moe.r5.v2.4"
R5_V25_SCHEMA_VERSION = "dcvpp.vpp_world_moe.r5.v2.5"
LEGACY_ARCHITECTURE_VERSION = "causal_action_conditioned_world_moe_r5_v1"
ARCHITECTURE_VERSION = "causal_action_conditioned_world_moe_r5_v2_proper_score"
R5_V21_ARCHITECTURE_VERSION = (
    "causal_action_conditioned_world_moe_r5_v2_1_calibrated_training"
)
R5_V22_ARCHITECTURE_VERSION = "causal_continuous_action_conditioned_world_moe_r5_v2_2"
R5_V23_ARCHITECTURE_VERSION = (
    "causal_continuous_action_conditioned_world_moe_r5_v2_3_calibrated_u0"
)
R5_V24_ARCHITECTURE_VERSION = (
    "causal_continuous_action_conditioned_world_moe_r5_v2_4_atom_aware_u0"
)
R5_V25_ARCHITECTURE_VERSION = (
    "causal_action_conditioned_distribution_world_moe_r5_v2_5_low_rank_student_t"
)
EXPERT_ASSIGNMENT_SCHEMA = "7_general_top2_plus_conditional_tail_residual_v1"
PHASE_ORDER = ("w0", "w1", "u0", "p1", "j1", "c1")
SPLITS = {
    "train": ("2022-01-01", "2023-12-31", 722, "development_fit"),
    "validation": (
        "2024-01-01",
        "2024-12-31",
        364,
        "development_validation",
    ),
    "trainval_refit": (
        "2022-01-01",
        "2024-12-31",
        1086,
        "frozen_method_refit",
    ),
    "final_test": (
        "2025-01-01",
        "2025-12-31",
        362,
        "unique_final_test",
    ),
}
CONTEXTS = {
    "fast": (96, 0.25),
    "medium": (168, 1.0),
    "slow": (56, 24.0),
}
MODEL_SHAPE = {
    "horizon": 96,
    "nodes": 4,
    "entities_per_node": 8,
    "global_slots": 8,
    "input_dim": 8,
    "graph_edge_dim": 6,
    "constraint_dim": 8,
    "action_dim": 6,
}

ENTITY_NAMES = (
    "market",
    "battery",
    "pv",
    "cooling",
    "workload_arrival",
    "workload_queue",
    "compute",
    "grid_constraint",
)
GLOBAL_SLOT_NAMES = (
    "day_ahead_regime",
    "real_time_tail",
    "weather",
    "aggregate_workload",
    "system_constraint",
    "risk",
    "action_context",
    "calendar",
)
ACTION_FEATURE_NAMES = (
    "day_ahead_mw",
    "battery_mw",
    "compute_service_units",
    "migration_in_units",
    "migration_out_units",
    "pv_curtailment_mw",
)
CONSTRAINT_SLACK_NAMES = (
    "soc_fraction",
    "grid_headroom",
    "compute_headroom",
    "queue_or_deadline_slack",
    "pv_headroom",
    "sla_slack",
    "migration_in_headroom",
    "migration_out_headroom",
)


@dataclass(frozen=True)
class WorldMoEConfig:
    """In-memory neural architecture configuration.

    Structural dimensions are locked across the registered R5 lineages.  An
    input may contain fewer than 96 fast tokens for rolling inference and unit
    tests, but never more.
    """

    input_dim: int = 8
    graph_edge_dim: int = 6
    constraint_dim: int = 8
    action_dim: int = 6
    horizon: int = 96
    num_nodes: int = 4
    entities_per_node: int = 8
    global_slots: int = 8
    d_model: int = 256
    num_blocks: int = 10
    num_heads: int = 8
    num_regimes: int = 8
    dynamics_regime_temperature: float = 0.75
    stochastic_dim: int = 32
    num_experts: int = 8
    top_k: int = 2
    general_experts: int = 7
    regular_top_k: int = 2
    conditional_tail_expert: bool = True
    expert_assignment_schema: str = EXPERT_ASSIGNMENT_SCHEMA
    tail_capacity_fraction: float = 0.05
    tail_capacity_max_fraction: float = 1.0
    tail_min_tokens: int = 32
    tail_risk_threshold_work_units: float = 1.0e-4
    tail_activation_probability_threshold: float = 0.50
    tail_route_aux_weight: float = 0.05
    tail_route_aux_temperature: float = 1.0
    queue_occurrence_train_prevalence: float = 0.019967682659626007
    tail_route_train_prevalence: float = 0.019967682659626007
    tail_route_positive_weight: float = 49.08092411354107
    tail_smooth_activation_temperature: float = 1.0
    tail_residual_min_scale: float = 0.08
    tail_residual_max_scale: float = 0.12
    moe_block_indices: tuple[int, ...] = (1, 3, 5, 7, 9)
    expert_hidden_dim: int = 1024
    ffn_hidden_dim: int = 1024
    dropout: float = 0.0
    rollout_paths: int = 20
    dynamic_particle_count_enabled: bool = False
    maximum_dynamic_particle_count: int = 256
    planner_candidates: int = 8
    planner_refinements: int = 3
    planner_temperature: float = 0.25
    cvar_alpha: float = 0.95
    cvar_weight: float = 0.25
    queue_weight: float = 2.0
    migration_weight: float = 0.02
    battery_degradation_weight: float = 0.01
    unserved_power_weight: float = 100.0
    physics_residual_cap: float = 0.1
    minimum_scale: float = 1.0e-4
    support_atom_enabled: bool = False
    support_atom_prevalence_source: str = "train_2022_2023_only"
    support_atom_global_prevalence_role: str = (
        "diagnostic_evidence_only_not_head_initialization"
    )
    support_atom_workload_zero_prevalence: float = 0.1657222985
    support_atom_pv_zero_prevalence: float = 0.4820570055
    support_atom_pv_capacity_prevalence: float = 0.0038276
    support_atom_pue_minimum_prevalence: float = 0.0045787546
    support_atom_pue_maximum_prevalence: float = 0.1657222985
    support_atom_train_fit_artifact: str | None = None
    support_atom_train_fit_sha256: str | None = None
    support_atom_workload_zero_train_prevalence: float | None = None
    support_atom_pv_daylight_zero_prevalence: float | None = None
    support_atom_pv_daylight_capacity_prevalence: float | None = None
    support_atom_pv_night_zero_train_prevalence: float | None = None
    support_atom_pv_night_positive_train_prevalence: float | None = None
    support_atom_pue_residual_minimum_prevalence: float | None = None
    support_atom_pue_residual_maximum_prevalence: float | None = None
    support_atom_pue_residual_maximum_role: str = "structural_only_no_residual_head"
    support_atom_pue_maximum_given_zero_workload_prevalence: float | None = None
    pv_night_capacity_fraction_threshold: float = 1.0e-4
    analytic_mixed_support_enabled: bool = False
    pv_night_support_mode: str = "deterministic_zero"
    u0_conditioning_encoder_sample_stochastic: bool = True
    u0_action_conditioned: bool = False
    u0_action_source: str = "zero_action_legacy"
    innovation_distribution: str = "diagonal_gaussian"
    innovation_rank: int = 0
    innovation_student_t_df: float = 5.0
    innovation_scale_parameterization: str = "diagonal_log_scale"
    innovation_action_conditioned: bool = False
    cost_risk_critic_enabled: bool = False
    cost_risk_critic_parameterization: str = "disabled"
    cost_risk_critic_hidden_dim: int = 128
    cost_risk_critic_num_layers: int = 2
    cost_risk_critic_quantile_count: int = 64
    cost_risk_critic_distillation_weight: float = 0.0
    cost_risk_critic_train_phases: tuple[str, ...] = ("p1", "j1")
    cost_risk_critic_optimizer_group: str = "planner"
    gradient_checkpointing: bool = True
    # ``None`` preserves the legacy all-training-phases behavior.  A tuple
    # lets the trainer enable activation checkpointing only where the measured
    # memory/throughput trade-off warrants its recomputation cost.
    gradient_checkpointing_phases: tuple[str, ...] | None = None
    planner_candidate_chunk: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorldMoEConfig":
        """Build from the registered YAML model mapping and ignore metadata."""

        aliases = {
            "nodes": "num_nodes",
            "blocks": "num_blocks",
            "heads": "num_heads",
            "regimes": "num_regimes",
            "experts": "num_experts",
            "moe_blocks": "moe_block_indices",
        }
        allowed = {item.name for item in fields(cls)}
        resolved: dict[str, Any] = {}
        for raw_name, item in value.items():
            name = aliases.get(str(raw_name), str(raw_name))
            if name in allowed:
                resolved[name] = item
        if "moe_block_indices" in resolved:
            resolved["moe_block_indices"] = tuple(
                int(item) for item in resolved["moe_block_indices"]
            )
        if resolved.get("gradient_checkpointing_phases") is not None:
            resolved["gradient_checkpointing_phases"] = tuple(
                str(item).lower() for item in resolved["gradient_checkpointing_phases"]
            )
        if "cost_risk_critic_train_phases" in resolved:
            resolved["cost_risk_critic_train_phases"] = tuple(
                str(item).lower()
                for item in resolved["cost_risk_critic_train_phases"]
            )
        return cls(**resolved)

    def __post_init__(self) -> None:
        locked = {
            "num_nodes": (self.num_nodes, 4),
            "entities_per_node": (self.entities_per_node, 8),
            "global_slots": (self.global_slots, 8),
            "d_model": (self.d_model, 256),
            "num_blocks": (self.num_blocks, 10),
            "num_regimes": (self.num_regimes, 8),
            "num_experts": (self.num_experts, 8),
            "top_k": (self.top_k, 2),
            "general_experts": (self.general_experts, 7),
            "regular_top_k": (self.regular_top_k, 2),
            "conditional_tail_expert": (self.conditional_tail_expert, True),
            "rollout_paths": (self.rollout_paths, 20),
            "planner_candidates": (self.planner_candidates, 8),
            "planner_refinements": (self.planner_refinements, 3),
            "action_dim": (self.action_dim, len(ACTION_FEATURE_NAMES)),
            "constraint_dim": (
                self.constraint_dim,
                len(CONSTRAINT_SLACK_NAMES),
            ),
        }
        changed = [
            name for name, (actual, expected) in locked.items() if actual != expected
        ]
        if changed:
            detail = ", ".join(
                f"{name}={locked[name][0]} (required {locked[name][1]})"
                for name in changed
            )
            raise ValueError(f"WorldMoE structural contract changed: {detail}")
        if self.moe_block_indices != (1, 3, 5, 7, 9):
            raise ValueError("exactly five WorldMoE blocks 1,3,5,7,9 must contain MoE")
        if self.expert_assignment_schema != EXPERT_ASSIGNMENT_SCHEMA:
            raise ValueError("expert assignment schema changed")
        if self.general_experts + 1 != self.num_experts:
            raise ValueError("one of eight experts must be the conditional tail expert")
        if (
            not 0.0
            < self.tail_capacity_fraction
            <= self.tail_capacity_max_fraction
            <= 1.0
        ):
            raise ValueError(
                "tail capacity fractions must satisfy 0 < minimum <= maximum <= 1"
            )
        if self.tail_min_tokens < 1:
            raise ValueError("tail_min_tokens must be positive")
        if self.tail_risk_threshold_work_units <= 0.0:
            raise ValueError("tail risk threshold must be positive")
        if not 0.0 < self.tail_activation_probability_threshold < 1.0:
            raise ValueError("tail activation probability threshold must lie in (0,1)")
        if self.tail_route_aux_weight <= 0.0:
            raise ValueError("tail route auxiliary weight must be positive")
        if self.tail_route_aux_temperature <= 0.0:
            raise ValueError("tail route auxiliary temperature must be positive")
        if not 0.0 < self.queue_occurrence_train_prevalence < 1.0:
            raise ValueError("queue occurrence train prevalence must lie in (0,1)")
        if not 0.0 < self.tail_route_train_prevalence < 1.0:
            raise ValueError("tail route train prevalence must lie in (0,1)")
        expected_positive_weight = (
            1.0 - self.tail_route_train_prevalence
        ) / self.tail_route_train_prevalence
        if abs(self.tail_route_positive_weight - expected_positive_weight) > 1.0e-5:
            raise ValueError(
                "tail route positive weight disagrees with train prevalence"
            )
        if self.tail_smooth_activation_temperature <= 0.0:
            raise ValueError("tail smooth activation temperature must be positive")
        if not (
            0.0 < self.tail_residual_min_scale < self.tail_residual_max_scale <= 1.0
        ):
            raise ValueError("tail residual bounds must satisfy 0 < min < max <= 1")
        if self.horizon < 1 or self.horizon > 96:
            raise ValueError("horizon must lie in [1,96]")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.dynamics_regime_temperature <= 0.0:
            raise ValueError("dynamics regime temperature must be positive")
        if self.input_dim < 1 or self.graph_edge_dim < 1:
            raise ValueError("input and edge dimensions must be positive")
        support_prevalences = {
            "workload_zero": self.support_atom_workload_zero_prevalence,
            "pv_zero": self.support_atom_pv_zero_prevalence,
            "pv_capacity": self.support_atom_pv_capacity_prevalence,
            "pue_minimum": self.support_atom_pue_minimum_prevalence,
            "pue_maximum": self.support_atom_pue_maximum_prevalence,
        }
        if any(not 0.0 < value < 1.0 for value in support_prevalences.values()):
            raise ValueError("support-atom train prevalences must lie in (0,1)")
        if (
            self.support_atom_enabled
            and self.support_atom_prevalence_source != "train_2022_2023_only"
        ):
            raise ValueError(
                "support-atom prevalences must be frozen on 2022-2023 train"
            )
        if self.support_atom_enabled and self.support_atom_global_prevalence_role != (
            "diagnostic_evidence_only_not_head_initialization"
        ):
            raise ValueError("global support-atom prevalence role is invalid")
        resolved_prevalences = (
            self.support_atom_workload_zero_train_prevalence,
            self.support_atom_pv_daylight_zero_prevalence,
            self.support_atom_pv_daylight_capacity_prevalence,
            self.support_atom_pv_night_zero_train_prevalence,
            self.support_atom_pv_night_positive_train_prevalence,
            self.support_atom_pue_residual_minimum_prevalence,
        )
        if any(
            value is not None and not 0.0 < value < 1.0
            for value in resolved_prevalences
        ):
            raise ValueError(
                "resolved conditional support-atom prevalences must lie in (0,1)"
            )
        if self.support_atom_pue_residual_maximum_prevalence is not None:
            raise ValueError("PUE residual maximum has no train event or residual head")
        if self.support_atom_pue_residual_maximum_role != (
            "structural_only_no_residual_head"
        ):
            raise ValueError("PUE residual maximum role is invalid")
        if (
            self.support_atom_pue_maximum_given_zero_workload_prevalence is not None
            and self.support_atom_pue_maximum_given_zero_workload_prevalence != 1.0
        ):
            raise ValueError(
                "PUE workload-zero structural maximum prevalence must be 1"
            )
        if not 0.0 < self.pv_night_capacity_fraction_threshold < 1.0:
            raise ValueError("PV causal night threshold must lie in (0,1)")
        if self.analytic_mixed_support_enabled:
            if self.pv_night_support_mode != "mixed_zero_rare_positive":
                raise ValueError("analytic mixed support requires the registered PV night mode")
            night_zero = self.support_atom_pv_night_zero_train_prevalence
            night_positive = self.support_atom_pv_night_positive_train_prevalence
            if night_zero is None or night_positive is None or abs(
                night_zero + night_positive - 1.0
            ) > 1.0e-12:
                raise ValueError(
                    "PV night zero and rare-positive train prevalences must sum to 1"
                )
        elif self.pv_night_support_mode != "deterministic_zero":
            raise ValueError("PV night mixed support requires analytic mixed support")
        if not isinstance(self.u0_conditioning_encoder_sample_stochastic, bool):
            raise ValueError("U0 conditioning encoder sampling flag must be boolean")
        if self.u0_action_conditioned:
            if self.u0_action_source != "causal_packet_action_sequence":
                raise ValueError("action-conditioned U0 requires causal packet actions")
        elif self.u0_action_source != "zero_action_legacy":
            raise ValueError("legacy U0 action source changed without enabling conditioning")
        if self.innovation_rank < 0:
            raise ValueError("innovation_rank must be nonnegative")
        if self.maximum_dynamic_particle_count < self.rollout_paths:
            raise ValueError(
                "maximum_dynamic_particle_count cannot be below rollout_paths"
            )
        if not isinstance(self.dynamic_particle_count_enabled, bool):
            raise ValueError("dynamic particle-count flag must be boolean")
        if self.innovation_student_t_df <= 2.0:
            raise ValueError("Student-t innovation df must exceed 2")
        if self.innovation_rank:
            if (
                self.innovation_distribution != "low_rank_student_t"
                or self.innovation_scale_parameterization
                != "diagonal_plus_low_rank_factor"
                or not self.innovation_action_conditioned
            ):
                raise ValueError(
                    "low-rank innovation requires action-conditioned Student-t factors"
                )
        elif (
            self.innovation_distribution != "diagonal_gaussian"
            or self.innovation_scale_parameterization != "diagonal_log_scale"
            or self.innovation_action_conditioned
        ):
            raise ValueError("rank-zero legacy innovation contract changed")
        if self.cost_risk_critic_enabled:
            if (
                self.cost_risk_critic_parameterization
                != "monotone_fixed_grid_quantile"
                or self.cost_risk_critic_hidden_dim < 1
                or self.cost_risk_critic_num_layers < 1
                or self.cost_risk_critic_quantile_count < 2
                or self.cost_risk_critic_distillation_weight <= 0.0
                or self.cost_risk_critic_train_phases != ("p1", "j1")
                or self.cost_risk_critic_optimizer_group != "planner"
            ):
                raise ValueError("cost-risk critic contract is invalid")
        elif (
            self.cost_risk_critic_parameterization != "disabled"
            or self.cost_risk_critic_distillation_weight != 0.0
        ):
            raise ValueError("disabled cost-risk critic contract changed")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.gradient_checkpointing_phases is not None:
            phases = tuple(self.gradient_checkpointing_phases)
            allowed_phases = {"w0", "w1", "u0", "p1", "j1"}
            if (
                not phases
                or len(phases) != len(set(phases))
                or any(phase not in allowed_phases for phase in phases)
            ):
                raise ValueError(
                    "gradient_checkpointing_phases must be unique trainable phases"
                )
        if not 0.5 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie in (0.5,1)")
        if self.planner_candidate_chunk < 1:
            raise ValueError("planner_candidate_chunk must be positive")

    @property
    def local_slots(self) -> int:
        return self.num_nodes * self.entities_per_node

    @property
    def total_slots(self) -> int:
        return self.local_slots + self.global_slots


class WorldMoEConfigError(ValueError):
    """Raised before a weakened WorldMoE scientific contract can execute."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorldMoEConfigError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise WorldMoEConfigError(f"{label} must be a sequence")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorldMoEConfigError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _validate_no_retired_runtime(value: Any, path: str = "configuration") -> None:
    """Reject keys that would reintroduce a separate optimization runtime."""

    retired = ("gurobi", "solver", "oracle", "fallback")
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered == "o0" or any(word in lowered for word in retired):
                raise WorldMoEConfigError(
                    f"{path}.{key} reintroduces a retired runtime path"
                )
            _validate_no_retired_runtime(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _validate_no_retired_runtime(child, f"{path}[{index}]")


def _validate_split(splits: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _mapping(splits.get(name), f"protocol.splits.{name}")
    start, end, days, role = SPLITS[name]
    if (
        str(value.get("start")) != start
        or str(value.get("end")) != end
        or int(value.get("days", -1)) != days
        or str(value.get("role")) != role
    ):
        raise WorldMoEConfigError(
            f"{name} must be {start}..{end}, days={days}, role={role}"
        )
    return value


def _validate_physical_migration_separation(path: Path) -> None:
    if not path.is_file():
        raise WorldMoEConfigError(f"frozen physical artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise WorldMoEConfigError("frozen physical artifact must be a mapping")
    link_fields = {
        "migration_capacity",
        "migration_latency",
        "route_capacity",
        "route_latency",
        "network_bandwidth",
        "network_latency",
    }
    present = sorted(link_fields.intersection(str(key) for key in payload))
    if present:
        raise WorldMoEConfigError(
            "migration evidence must remain separate from the frozen physical twin: "
            + ", ".join(present)
        )


def _validate_migration_evidence(
    migration: Mapping[str, Any],
    *,
    root: Path,
    output_root: Path,
    paths: dict[str, Path],
    allow_frozen_source_lineage: bool = False,
) -> None:
    """Lock the direct-route, as-of-origin migration evidence contract."""

    availability = _mapping(
        migration.get("availability"), "data.migration.availability"
    )
    capacity = _mapping(migration.get("capacity"), "data.migration.capacity")
    routes = [
        [str(item) for item in route]
        for route in migration.get("enabled_routes", ())
        if isinstance(route, Sequence) and not isinstance(route, (str, bytes))
    ]
    if (
        migration.get("enabled") is not True
        or migration.get("mode") != "evidence_backed_operational_proxy_sensitivity"
        or migration.get("latency_source") != "ripe_atlas_direct_anchor_mesh_ipv4"
        or migration.get("capacity_source")
        != "ub_nysernet_published_100gbps_upper_bound"
        or migration.get("action_mask") != "ripe_asof_origin_gated"
        or routes != [["WEST", "N.Y.C."], ["N.Y.C.", "WEST"]]
    ):
        raise WorldMoEConfigError(
            "migration must use the registered direct WEST/NYC evidence adapter"
        )
    if (
        availability.get("selection") != "latest_result_timestamp_lte_origin"
        or availability.get("future_horizon")
        != "hold_forward_latest_origin_observation"
        or float(availability.get("max_age_minutes", -1.0)) != 30.0
        or float(availability.get("max_packet_loss_rate", -1.0)) != 0.05
        or int(availability.get("min_sent_packets", -1)) != 3
        or availability.get("future_results_allowed") is not False
        or availability.get("require_simultaneous_reverse_anchor_measurement")
        is not False
        or availability.get("directed_routes_independently_gated") is not True
    ):
        raise WorldMoEConfigError("migration as-of availability contract was weakened")
    if (
        capacity.get("evidence_role")
        != "operational_proxy_sensitivity_not_execution_log"
        or float(capacity.get("nominal_gbps", -1.0)) != 100.0
        or int(capacity.get("interval_seconds", -1)) != 900
        or capacity.get("measured_end_to_end_bandwidth") is not False
    ):
        raise WorldMoEConfigError("migration capacity proxy boundary was weakened")

    registered_paths = {
        "migration_evidence": migration.get("evidence_path"),
        "migration_manifest": migration.get("manifest_path"),
        "migration_workload_events": capacity.get("workload_events_path"),
        "migration_work_unit_bridge": capacity.get("work_unit_bridge_path"),
    }
    for name, raw in registered_paths.items():
        resolved = _resolve(root, raw, f"data.migration.{name}")
        if not resolved.is_file():
            raise WorldMoEConfigError(
                f"required migration evidence is missing: {resolved}"
            )
        paths[name] = resolved
    for name in ("migration_evidence", "migration_manifest"):
        if (
            not paths[name].is_relative_to(output_root)
            and not allow_frozen_source_lineage
        ):
            raise WorldMoEConfigError(
                f"{name} must belong to the WorldMoE output lineage"
            )
    if allow_frozen_source_lineage and migration.get("frozen_source_lineage") != (
        "R5-V1_development_only_migration_evidence"
    ):
        raise WorldMoEConfigError(
            "R5-V2 reused migration evidence must declare its frozen R5-V1 lineage"
        )
    manifest = json.loads(paths["migration_manifest"].read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or (
        manifest.get("schema") != "dcvpp.vpp_world_moe.migration-evidence.v1"
        or manifest.get("enabled_routes") != [["WEST", "N.Y.C."], ["N.Y.C.", "WEST"]]
    ):
        raise WorldMoEConfigError("migration evidence manifest is incompatible")
    window = manifest.get("development_window")
    if not isinstance(window, Mapping) or (
        window.get("stop_exclusive") != "2025-01-01T00:00:00+00:00"
        or window.get("opened_final_test_2025") is not False
    ):
        raise WorldMoEConfigError("migration evidence opened the sealed 2025 period")


def _float_mapping(value: Any, label: str) -> dict[str, float]:
    return {str(key): float(item) for key, item in _mapping(value, label).items()}


def _validate_r5_v2_objective(
    training: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    revision: str = "r5_v2",
) -> None:
    """Lock the R5-V2 proper-score and risk-training protocol."""

    if revision not in {
        "r5_v2",
        "r5_v2_1",
        "r5_v2_2",
        "r5_v2_3",
        "r5_v2_4",
        "r5_v2_5",
    }:
        raise WorldMoEConfigError("unknown R5-V2.x objective revision")
    is_v21 = revision in {
        "r5_v2_1",
        "r5_v2_2",
        "r5_v2_3",
        "r5_v2_4",
        "r5_v2_5",
    }
    is_v22 = revision in {"r5_v2_2", "r5_v2_3", "r5_v2_4", "r5_v2_5"}
    is_v23 = revision in {"r5_v2_3", "r5_v2_4", "r5_v2_5"}
    is_v24 = revision == "r5_v2_4"
    is_v25 = revision == "r5_v2_5"
    expected_protocol_revision = (
        "r5_v2_5_distribution_native_action_conditioned_u0"
        if is_v25
        else "r5_v2_4_u0_atom_aware_conditional_calibration"
        if is_v24
        else "r5_v2_3_u0_fair_ensemble_calibrated_dynamics"
        if is_v23
        else "r5_v2_2_w1_continuous_action_handoff_efficiency"
        if is_v22
        else "r5_v2_1_typed_anchor_calibrated_tail_pcgrad"
        if is_v21
        else None
    )
    if (
        expected_protocol_revision is not None
        and training.get("protocol_revision") != expected_protocol_revision
    ):
        raise WorldMoEConfigError(
            f"{revision.replace('_', '-').upper()} protocol revision is not registered"
        )
    if training.get("loss_profile") != "proper_joint_risk_r5_v2":
        raise WorldMoEConfigError("R5-V2 must use the registered proper-score loss")
    if training.get("comparison_profile") != (
        "matched_architecture_legacy_loss_from_scratch"
    ):
        raise WorldMoEConfigError("R5-V2 legacy comparison must be a fresh matched run")
    precision = _mapping(training.get("precision"), "training.precision")
    if precision != {
        "neural_forward": "bf16",
        "scientific_scores": "strict_float32",
        "cvar": "strict_float32",
        "physics": "strict_float32",
    }:
        raise WorldMoEConfigError("R5-V2 scientific boundaries must remain strict FP32")
    runtime = _mapping(training.get("cuda_runtime"), "training.cuda_runtime")
    if (
        runtime.get("allow_tf32") is not False
        or runtime.get("float32_matmul_precision") != "highest"
    ):
        raise WorldMoEConfigError("R5-V2 strict scientific path forbids global TF32")

    objective = _mapping(
        training.get("probabilistic_objective"),
        "training.probabilistic_objective",
    )
    quantiles = tuple(float(value) for value in objective.get("quantiles", ()))
    if quantiles != (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        raise WorldMoEConfigError("R5-V2 quantile grid changed")
    required = {
        "monotone_parameterization": "base_plus_softplus_positive_increments",
        "dart_definition": "deterministic_da_minus_rt_per_joint_path",
        "tail_thresholds": "train_2022_2023_only_fixed",
        "reduction": "global_numerator_over_valid_count",
        "mae_rmse_role": "final_reporting_only",
    }
    if any(objective.get(key) != value for key, value in required.items()):
        raise WorldMoEConfigError("R5-V2 probabilistic objective was weakened")
    if _float_mapping(
        objective.get("target_family_weights"),
        "training.probabilistic_objective.target_family_weights",
    ) != {"price": 0.25, "workload_volume": 0.25, "pv": 0.25, "pue": 0.25}:
        raise WorldMoEConfigError("R5-V2 target-family balance changed")
    if _float_mapping(
        objective.get("price_family_weights"),
        "training.probabilistic_objective.price_family_weights",
    ) != {"da": 0.25, "rt": 0.25, "dart": 0.50}:
        raise WorldMoEConfigError("R5-V2 price-family balance changed")

    phase_objectives = _mapping(
        training.get("phase_objectives"), "training.phase_objectives"
    )
    expected_weights = {
        "w0": {"quantile": 1.0, "count": 0.25, "state": 0.25, "moe": 0.01},
        "w1": {"latent": 0.5, "rollout": 0.5, "state": 0.1, "moe": 0.01},
        "u0": (
            {
                "crps": 0.30,
                "energy": 0.30,
                "variogram": 0.15,
                "tail_crps": 0.15,
                "mixture_nll": 0.0,
                "atom_brier": 0.10,
                "moe": 0.01,
            }
            if is_v24 or is_v25
            else {
                "crps": 0.30,
                "energy": 0.30,
                "variogram": 0.15,
                "tail_crps": 0.15,
                "mixture_nll": 0.0,
                "moe": 0.01,
            }
            if is_v23
            else {
                "crps": 0.30,
                "energy": 0.30,
                "variogram": 0.15,
                "tail_crps": 0.15,
                "mixture_nll": 0.10,
                "moe": 0.01,
            }
        ),
    }
    for phase, expected in expected_weights.items():
        value = _mapping(
            phase_objectives.get(phase), f"training.phase_objectives.{phase}"
        )
        if _float_mapping(value.get("weights"), f"{phase}.weights") != expected:
            raise WorldMoEConfigError(f"R5-V2 {phase.upper()} weights changed")
    p1 = _mapping(phase_objectives.get("p1"), "training.phase_objectives.p1")
    chunks = _mapping(p1.get("path_chunks"), "training.phase_objectives.p1.path_chunks")
    if (
        float(p1.get("cvar_alpha", -1.0)) != 0.95
        or float(p1.get("cvar_weight", -1.0)) != 0.25
        or p1.get("cvar_form") != "rockafellar_uryasev"
        or p1.get("constraint_method") != "raw_action_augmented_lagrangian"
        or chunks
        != {"paths_per_chunk": 20, "fixed_seed_chunks": 5, "aggregate_paths": 100}
    ):
        raise WorldMoEConfigError("R5-V2 P1 risk contract changed")
    j1 = _mapping(phase_objectives.get("j1"), "training.phase_objectives.j1")
    retired_j1_keys = {
        "noninferiority_ratio",
        "barrier_weight",
        "exogenous_policy_gradient",
    }.intersection(j1)
    if retired_j1_keys:
        raise WorldMoEConfigError(
            f"R5-V2 J1 contains retired controls: {sorted(retired_j1_keys)}"
        )
    if (
        j1.get("primary") != "policy"
        or j1.get("shared_gradient_method") != "pcgrad"
        or tuple(j1.get("objectives", ())) != ("world", "uncertainty", "policy")
        or j1.get("scalar_loss_aggregation") != "prohibited"
        or j1.get("parameter_scope") != "end_to_end_all_trainable"
    ):
        raise WorldMoEConfigError("R5-V2 J1 joint-training contract changed")

    selection = _mapping(
        training.get("checkpoint_selection"), "training.checkpoint_selection"
    )
    early = _mapping(selection.get("early_stop"), "checkpoint_selection.early_stop")
    if (
        selection.get("formal_validation_only") is not True
        or selection.get("batch_reduction") != "global_numerator_over_valid_count"
        or tuple(selection.get("prohibited_metrics", ())) != ("mae", "rmse")
        or int(early.get("stable_formal_evaluations", -1))
        != (3 if is_v23 else 2 if is_v21 else 3)
        or float(early.get("relative_improvement_below", -1.0)) != 0.001
    ):
        raise WorldMoEConfigError("R5-V2 checkpoint-selection contract changed")
    phases = _mapping(training.get("phases"), "training.phases")
    w0 = _mapping(phases.get("w0"), "training.phases.w0")
    def efficient_schedule_enabled(phase_name: str, phase_config: Mapping[str, Any]) -> bool:
        raw = _mapping(phase_config.get("efficient_schedule", {}), f"{phase_name}.efficient_schedule")
        if raw.get("enabled", False) is not True:
            return False
        expected = {
            "enabled": True,
            "first_formal_epoch": 32 if phase_name == "w0" else 24,
            "recovery_interval_seconds": 1800,
            "progress_interval_seconds": 60,
            "fixed_budget": phase_name == "w0",
            "early_stop_start_epoch": 128 if phase_name == "w0" else 64,
            "early_stop_evaluations": 4,
            "relative_improvement": 0.001,
            "defer_w1_reference": phase_name == "w1",
        }
        if phase_name == "w0" and int(phase_config.get("max_epochs", -1)) == 96:
            expected["early_stop_start_epoch"] = 96
        if phase_name == "w1" and (
            raw.get("early_stop_start_epoch"), raw.get("early_stop_evaluations")
        ) == (56, 2):
            expected.update(early_stop_start_epoch=56, early_stop_evaluations=2)
        if phase_name == "w1" and (
            raw.get("first_formal_epoch"), raw.get("early_stop_start_epoch"),
            raw.get("early_stop_evaluations"), raw.get("relative_improvement"),
        ) == (48, 0, 2, 0.0):
            expected.update(
                first_formal_epoch=48, early_stop_start_epoch=0,
                early_stop_evaluations=2, relative_improvement=0.0,
            )
        if "selection_policy" in raw:
            if raw["selection_policy"] not in {"capability_gated", "validation_score_only"}:
                raise WorldMoEConfigError(f"unknown {phase_name.upper()} efficient selection policy")
            expected["selection_policy"] = raw["selection_policy"]
        if "checkpoint_policy" in raw:
            if raw["checkpoint_policy"] not in {"milestones", "best_only"}:
                raise WorldMoEConfigError(f"unknown {phase_name.upper()} checkpoint policy")
            if raw["checkpoint_policy"] == "best_only" and raw.get("selection_policy") != "validation_score_only":
                raise WorldMoEConfigError("best_only requires validation_score_only selection")
            expected["checkpoint_policy"] = raw["checkpoint_policy"]
        if dict(raw) != expected:
            raise WorldMoEConfigError(f"approved {phase_name.upper()} efficient schedule changed")
        return True

    efficient_w0 = efficient_schedule_enabled("w0", w0)
    w0_budget = _mapping(
        w0.get("target_update_budgets"),
        "training.phases.w0.target_update_budgets",
    )
    expected_schedule = (
        {
            "short_epoch_factor": 8,
            "formal_validation_interval_short_epochs": 8,
            "min_epochs": 16,
            "max_epochs": 96,
            "patience": 16,
            "minimum": 122,
            "maximum": 732,
            "patience_equivalent": 122,
        }
        if is_v21
        else {
            "short_epoch_factor": 8,
            "formal_validation_interval_short_epochs": 8,
            "min_epochs": 16,
            "max_epochs": 320,
            "patience": 24,
            "minimum": 122,
            "maximum": 2440,
            "patience_equivalent": 183,
        }
    )
    if efficient_w0:
        if not is_v21:
            raise WorldMoEConfigError("efficient W0 requires the calibrated R5-V2.1 objective")
        expected_schedule = {
            **expected_schedule,
            "formal_validation_interval_short_epochs": 16,
            "max_epochs": 96 if int(w0.get("max_epochs", -1)) == 96 else 128,
            "maximum": 732 if int(w0.get("max_epochs", -1)) == 96 else 976,
        }
    actual_schedule = {
        "short_epoch_factor": int(w0.get("short_epoch_factor", -1)),
        "formal_validation_interval_short_epochs": int(
            w0.get("formal_validation_interval_short_epochs", -1)
        ),
        "min_epochs": int(w0.get("min_epochs", -1)),
        "max_epochs": int(w0.get("max_epochs", -1)),
        "patience": int(w0.get("patience", -1)),
        **{
            name: int(w0_budget.get(name, -1))
            for name in ("minimum", "maximum", "patience_equivalent")
        },
    }
    if actual_schedule != expected_schedule:
        raise WorldMoEConfigError("R5-V2 W0 short-epoch schedule changed")
    if is_v21:
        optimizer_contract = {
            "learning_rate": 3.0e-5,
            "head_learning_rate": 1.0e-4,
            "router_learning_rate": 5.0e-5,
            "tail_learning_rate": 3.0e-5,
            "matrix_weight_decay": 1.0e-2,
            "warmup_updates": 30,
            "cosine_decay_updates": 366,
            "minimum_learning_rate_ratio": 0.1,
            "adamw_beta2": 0.95,
        }
        if (
            any(
                abs(float(w0.get(name, float("nan"))) - expected) > 1.0e-12
                for name, expected in optimizer_contract.items()
            )
            or w0.get("w0_pcgrad") is not True
        ):
            raise WorldMoEConfigError("R5-V2.1 W0 optimizer contract changed")
        if (
            int(w0.get("early_formal_interval_short_epochs", -1)) != (16 if efficient_w0 else 4)
            or int(w0.get("early_formal_until_short_epoch", -1)) != (0 if efficient_w0 else 32)
        ):
            raise WorldMoEConfigError("R5-V2.1 early formal cadence changed")
    if is_v22:
        w1 = _mapping(phases.get("w1"), "training.phases.w1")
        efficient_w1 = efficient_schedule_enabled("w1", w1)
        w1_contract = {
            "batch_size": 12,
            "validation_batch_size": 12,
            "formal_validation_interval_short_epochs": 4 if efficient_w1 else 2,
            "learning_rate": 2.5e-5,
            "head_learning_rate": 7.0e-5,
            "router_learning_rate": 2.0e-5,
            "tail_learning_rate": 2.5e-5,
            "weight_decay": 0.0,
            "matrix_weight_decay": 1.0e-2,
            "warmup_updates": 64,
            "cosine_decay_updates": 1790,
            "minimum_learning_rate_ratio": 0.1,
            "adamw_beta2": 0.95,
        }
        if efficient_w1 and int(w1.get("max_epochs", -1)) != 80:
            raise WorldMoEConfigError("approved W1 efficient budget must be 80 epochs")
        if any(
            abs(float(w1.get(name, float("nan"))) - expected) > 1.0e-12
            for name, expected in w1_contract.items()
        ):
            raise WorldMoEConfigError("R5-V2.2 W1 efficiency contract changed")
        u0 = _mapping(phases.get("u0"), "training.phases.u0")
        if (
            int(u0.get("batch_size", -1)) != 2
            or int(u0.get("validation_batch_size", -1)) != 3
            or int(u0.get("formal_validation_interval_short_epochs", -1)) != 2
        ):
            raise WorldMoEConfigError("R5-V2.2 U0 efficiency contract changed")
        checkpointing = _mapping(
            training.get("checkpointing"), "training.checkpointing"
        )
        if (
            int(checkpointing.get("fixed_update_interval", -1)) != 32
            or float(checkpointing.get("max_wall_clock_seconds", -1.0)) != 180.0
            or int(checkpointing.get("sentinel_validation_batches", -1)) != 4
        ):
            raise WorldMoEConfigError("R5-V2.2 checkpoint cadence changed")
        if tuple(model.get("gradient_checkpointing_phases", ())) != ("w1",):
            raise WorldMoEConfigError(
                "R5-V2.2 gradient checkpointing phase contract changed"
            )
    if is_v23:
        objective_contract = (
            {
                "training_paths": 16,
                "sentinel_paths": 32,
                "formal_paths": 100,
                "formal_seed_chunks": 5,
                "formal_paths_per_seed_chunk": 20,
                "ensemble_score_estimator": "fair_off_diagonal_v1",
                "formal_coverage_interval": "rank_corrected_order_statistics_v1",
                "coverage_family_reduction": "equal_price_workload_pv_pue",
                "scenario_role": "dynamic_score_particles_not_distribution_definition",
                "common_random_numbers": "fixed_formal_seed_offsets",
                "analytic_marginal_distribution": True,
                "joint_innovation_distribution": "low_rank_student_t",
            }
            if is_v25
            else {
                "training_paths": 20,
                "formal_paths": 100,
                "formal_seed_chunks": 5,
                "ensemble_score_estimator": "fair_off_diagonal_v1",
                "formal_coverage_interval": "rank_corrected_order_statistics_v1",
                "coverage_family_reduction": "equal_price_workload_pv_pue",
            }
        )
        if any(
            objective.get(name) != expected
            for name, expected in objective_contract.items()
        ):
            raise WorldMoEConfigError("R5-V2.3 finite-ensemble objective changed")
        validation = _mapping(
            training.get("uncertainty_validation"),
            "training.uncertainty_validation",
        )
        if (
            tuple(int(value) for value in validation.get("seed_offsets", ()))
            != (0, 104729, 209759, 314159, 419431)
            or validation.get("concatenate_before_scoring") is not True
            or validation.get("coverage_method") != "rank_corrected_order_statistics_v1"
            or validation.get("family_reduction") != "equal_price_workload_pv_pue"
            or float(validation.get("global_point_error_max", -1.0)) != 0.02
            or float(validation.get("family_point_error_max", -1.0)) != 0.04
            or float(validation.get("global_bootstrap_ucb_max", -1.0)) != 0.03
            or float(validation.get("family_bootstrap_ucb_max", -1.0)) != 0.05
            or int(validation.get("bootstrap_resamples", -1)) != 2000
            or int(validation.get("validation_days", -1)) != 364
            or validation.get("zero_update_reference_required") is not True
            or validation.get("tail_route_aucpr_required") is not True
            or validation.get("tail_route_all_blocks_required") is not True
            or validation.get("locked_2025_opened") is not False
        ):
            raise WorldMoEConfigError("R5-V2.3 formal uncertainty validation changed")
        if is_v25 and (
            int(validation.get("training_paths", -1)) != 16
            or int(validation.get("sentinel_paths", -1)) != 32
            or int(validation.get("formal_paths", -1)) != 100
            or int(validation.get("formal_paths_per_seed_chunk", -1)) != 20
            or validation.get("common_random_numbers") is not True
        ):
            raise WorldMoEConfigError("R5-V2.5 dynamic-particle validation changed")
        parent = _mapping(
            training.get("parent_checkpoint"), "training.parent_checkpoint"
        )
        expected_parent = {
            "run_id": "r5v22-w1-continuous-action-formal-20260824-seed20240813",
            "phase": "w1",
            "role": "formal_primary_best",
            "epoch": 56,
            "update": 3360,
            "sha256": "a97003e1ec3d92c3d814653b743c21fd646da70c1b85778e1bd71edbb3f1f250",
            "gate_status": "PASS",
            "load_model_only": True,
            "load_optimizer": False,
            "load_rng": False,
        }
        if dict(parent) != expected_parent:
            raise WorldMoEConfigError("R5-V2.3 W1 epoch-56 parent lineage changed")
    if is_v24 or is_v25:
        expected_run_id = (
            "r5v25-u0-distribution-native-from-w1e56-seed20240813"
            if is_v25
            else "r5v24-u0-atom-aware-from-w1e56-seed20240813"
        )
        if training.get("run_id") != expected_run_id:
            raise WorldMoEConfigError(
                f"R5-V2.{5 if is_v25 else 4} run lineage changed"
            )
        support_atom_contract = {
            "support_atom_enabled": True,
            "support_atom_prevalence_source": "train_2022_2023_only",
            "support_atom_global_prevalence_role": (
                "diagnostic_evidence_only_not_head_initialization"
            ),
            "support_atom_workload_zero_prevalence": 0.1657222985,
            "support_atom_pv_zero_prevalence": 0.4820570055,
            "support_atom_pv_capacity_prevalence": 0.0038276,
            "support_atom_pue_minimum_prevalence": 0.0045787546,
            "support_atom_pue_maximum_prevalence": 0.1657222985,
            "support_atom_train_fit_artifact": (
                "artifacts/vpp_world_moe_r5_v2_4_2025final/support_atom_train_fit.json"
            ),
            "support_atom_train_fit_sha256": (
                "9575bdd67fc162936f35211d881fbab38c8485e040d67f94ab1179ec14c821e8"
            ),
            "support_atom_workload_zero_train_prevalence": 0.25613169436749766,
            "support_atom_pv_daylight_zero_prevalence": 0.003116651825467498,
            "support_atom_pv_daylight_capacity_prevalence": 0.00889080587711487,
            "support_atom_pv_night_zero_train_prevalence": 0.9965843720038351,
            "support_atom_pue_residual_minimum_prevalence": 0.018328516844779767,
            "support_atom_pue_residual_maximum_prevalence": None,
            "support_atom_pue_residual_maximum_role": (
                "structural_only_no_residual_head"
            ),
            "support_atom_pue_maximum_given_zero_workload_prevalence": 1.0,
            "pv_night_capacity_fraction_threshold": 1.0e-4,
            "u0_conditioning_encoder_sample_stochastic": False,
        }
        if any(
            model.get(name) != expected
            for name, expected in support_atom_contract.items()
        ):
            raise WorldMoEConfigError("R5-V2.4+ support-atom model contract changed")

        u0 = _mapping(phases.get("u0"), "training.phases.u0")
        optimizer_contract = {
            "learning_rate": 1.0e-5,
            "head_learning_rate": 6.0e-5,
            "router_learning_rate": 2.0e-5,
            "tail_learning_rate": 3.0e-5,
            "weight_decay": 0.0,
            "matrix_weight_decay": 1.0e-2,
            "warmup_updates": 64,
            "cosine_decay_updates": 5054,
            "minimum_learning_rate_ratio": 0.1,
            "adamw_beta2": 0.95,
            "formal_validation_interval_short_epochs": 2,
        }
        if any(
            abs(float(u0.get(name, float("nan"))) - expected) > 1.0e-12
            for name, expected in optimizer_contract.items()
        ):
            raise WorldMoEConfigError("R5-V2.4 U0 optimizer contract changed")
        expected_optimizer_groups = {
            "backbone": 1.0e-5,
            "support_atom_heads": 6.0e-5,
            "stochastic_scale": 6.0e-5,
            "router": 2.0e-5,
            "tail_expert": 3.0e-5,
        }
        if is_v25:
            expected_optimizer_groups["innovation_heads"] = 6.0e-5
        if _float_mapping(
            u0.get("optimizer_group_contract"),
            "training.phases.u0.optimizer_group_contract",
        ) != expected_optimizer_groups:
            raise WorldMoEConfigError("R5-V2.4+ U0 grouped learning rates changed")

        u0_objective = _mapping(
            phase_objectives.get("u0"), "training.phase_objectives.u0"
        )
        support_atoms = _mapping(
            u0_objective.get("support_atoms"),
            "training.phase_objectives.u0.support_atoms",
        )
        global_evidence = _float_mapping(
            support_atoms.get("global_evidence_only"),
            "training.phase_objectives.u0.support_atoms.global_evidence_only",
        )
        if global_evidence != {
            "workload_zero": 0.1657222985,
            "pv_zero": 0.4820570055,
            "pv_capacity": 0.0038276,
            "pue_minimum": 0.0045787546,
            "pue_maximum": 0.1657222985,
        }:
            raise WorldMoEConfigError("R5-V2.4+ global support evidence changed")
        conditional_priors = _mapping(
            support_atoms.get("conditional_train_resolved"),
            "training.phase_objectives.u0.support_atoms.conditional_train_resolved",
        )
        expected_conditional_priors = {
            "artifact": (
                "artifacts/vpp_world_moe_r5_v2_4_2025final/support_atom_train_fit.json"
            ),
            "artifact_sha256": (
                "9575bdd67fc162936f35211d881fbab38c8485e040d67f94ab1179ec14c821e8"
            ),
            "workload_zero": 0.25613169436749766,
            "pv_daylight_zero": 0.003116651825467498,
            "pv_daylight_capacity": 0.00889080587711487,
            "pv_night_zero": 0.9965843720038351,
            "pue_residual_minimum": 0.018328516844779767,
            "pue_residual_maximum": None,
            "pue_residual_maximum_role": "structural_only_no_residual_head",
            "pue_maximum_given_zero_workload": 1.0,
        }
        if (
            support_atoms.get("enabled") is not True
            or support_atoms.get("prevalence_source") != "train_2022_2023_only"
            or support_atoms.get("head_initialization_from_global_evidence")
            != "prohibited"
            or support_atoms.get("missing_conditional_train_stat_action")
            != "fail_closed"
            or dict(conditional_priors) != expected_conditional_priors
            or float(support_atoms.get("pv_night_capacity_fraction_threshold", -1.0))
            != 1.0e-4
            or u0_objective.get("conditioning_encoder_sample_stochastic") is not False
        ):
            raise WorldMoEConfigError("R5-V2.4+ train-resolved support atoms changed")

        failure = _mapping(
            early.get("failure_plateau"),
            "training.checkpoint_selection.early_stop.failure_plateau",
        )
        if (
            int(failure.get("formal_evaluations", -1)) != 3
            or float(failure.get("relative_improvement_below", -1.0)) != 0.001
            or tuple(failure.get("jointly_stable_metrics", ()))
            != ("gate_gap", "proper_score")
            or failure.get("terminal_status") != "FAIL_NO_RELEASE"
        ):
            raise WorldMoEConfigError("R5-V2.4+ failure-plateau contract changed")

        conditional_coverage = _mapping(
            validation.get("conditional_coverage"),
            "training.uncertainty_validation.conditional_coverage",
        )
        moving_block = _mapping(
            conditional_coverage.get("moving_block_bootstrap"),
            "training.uncertainty_validation.conditional_coverage.moving_block_bootstrap",
        )
        expected_conditioning = (
            "analytic_mixed_support_cdf" if is_v25 else "support_atom_state"
        )
        if (
            conditional_coverage.get("enabled") is not True
            or tuple(conditional_coverage.get("families", ()))
            != ("workload", "pv", "pue")
            or conditional_coverage.get("conditioning") != expected_conditioning
            or conditional_coverage.get("conditioning_encoder_sample_stochastic")
            is not False
            or (
                is_v25
                and conditional_coverage.get("interior_distribution")
                != "direct_conditional_not_filtered_unconditional_particles"
            )
            or moving_block
            != {
                "method": "moving_block_bootstrap",
                "block_length_days": 14,
                "resamples": 2000,
                "confidence": 0.95,
            }
        ):
            raise WorldMoEConfigError("R5-V2.4+ conditional-coverage contract changed")

        diagnostic_probe = _mapping(
            training.get("diagnostic_repair_probe"),
            "training.diagnostic_repair_probe",
        )
        expected_probe = {
            "run_id": "r5v23-u0-recalibrated-from-w1e56-seed20240813",
            "phase": "u0",
            "role": "score_best",
            "epoch": 14,
            "update": 5054,
            "sha256": "20761385f4d0aa08bd07f35611694159a1d905d44253eac143e0de145b0a0146",
            "eligible": False,
            "weights_only": True,
            "smoke_only": True,
            "formal_parent_allowed": False,
            "p1_release_authority": False,
        }
        if dict(diagnostic_probe) != expected_probe:
            raise WorldMoEConfigError("R5-V2.4+ epoch-14 diagnostic probe changed")

        if is_v25:
            v25_model_contract = {
                "dynamic_particle_count_enabled": True,
                "maximum_dynamic_particle_count": 256,
                "analytic_mixed_support_enabled": True,
                "pv_night_support_mode": "mixed_zero_rare_positive",
                "support_atom_pv_night_positive_train_prevalence": (
                    0.0034156279961649
                ),
                "u0_action_conditioned": True,
                "u0_action_source": "causal_packet_action_sequence",
                "innovation_distribution": "low_rank_student_t",
                "innovation_rank": 8,
                "innovation_student_t_df": 5.0,
                "innovation_scale_parameterization": (
                    "diagonal_plus_low_rank_factor"
                ),
                "innovation_action_conditioned": True,
                "cost_risk_critic_enabled": True,
                "cost_risk_critic_parameterization": "monotone_fixed_grid_quantile",
                "cost_risk_critic_hidden_dim": 128,
                "cost_risk_critic_num_layers": 2,
                "cost_risk_critic_quantile_count": 64,
                "cost_risk_critic_distillation_weight": 0.50,
                "cost_risk_critic_train_phases": ["p1", "j1"],
                "cost_risk_critic_optimizer_group": "planner",
            }
            if any(
                model.get(name) != expected
                for name, expected in v25_model_contract.items()
            ):
                raise WorldMoEConfigError("R5-V2.5 distribution model contract changed")
            if int(u0.get("sample_count", -1)) != 16:
                raise WorldMoEConfigError("R5-V2.5 U0 must use 16 dynamic particles")
            u0_distribution_contract = {
                "distribution_representation": (
                    "analytic_mixed_marginals_plus_low_rank_student_t_joint_innovation"
                ),
                "action_conditioning": "causal_packet_action_sequence",
                "dynamic_particle_role": "proper_joint_score_monte_carlo_only",
                "analytic_mixed_support": True,
                "conditional_coverage_source": (
                    "analytic_conditional_interior_distribution"
                ),
            }
            if any(
                u0_objective.get(name) != expected
                for name, expected in u0_distribution_contract.items()
            ):
                raise WorldMoEConfigError("R5-V2.5 U0 distribution objective changed")
            parent_upgrade = _mapping(
                training.get("parent_model_upgrade"),
                "training.parent_model_upgrade",
            )
            expected_missing_prefixes = (
                "dynamics.innovation_",
                "planner.dynamics.innovation_",
                "observation_queries.pv_night_positive_",
                "dynamics.observation_queries.pv_night_positive_",
                "planner.observation_queries.pv_night_positive_",
                "planner.dynamics.observation_queries.pv_night_positive_",
                "observation_queries.workload_interior_quantile_query.",
                "observation_queries.pv_interior_quantile_query.",
                "observation_queries.pue_interior_quantile_query.",
                "dynamics.observation_queries.workload_interior_quantile_query.",
                "dynamics.observation_queries.pv_interior_quantile_query.",
                "dynamics.observation_queries.pue_interior_quantile_query.",
                "planner.observation_queries.workload_interior_quantile_query.",
                "planner.observation_queries.pv_interior_quantile_query.",
                "planner.observation_queries.pue_interior_quantile_query.",
                "planner.dynamics.observation_queries.workload_interior_quantile_query.",
                "planner.dynamics.observation_queries.pv_interior_quantile_query.",
                "planner.dynamics.observation_queries.pue_interior_quantile_query.",
                "cost_risk_critic.",
            )
            if (
                parent_upgrade.get("strict") is not False
                or parent_upgrade.get("unexpected_keys") != "prohibited"
                or parent_upgrade.get("missing_parameter_policy")
                != "registered_new_modules_only"
                or parent_upgrade.get("initialization")
                != "registered_per_module_deterministic_initialization"
                or parent_upgrade.get("innovation_initialization") != "exact_zero"
                or parent_upgrade.get("neural_head_initialization")
                != "construction_seed_20240813"
                or tuple(parent_upgrade.get("allowed_missing_prefixes", ()))
                != expected_missing_prefixes
            ):
                raise WorldMoEConfigError("R5-V2.5 parent upgrade policy changed")
    if float(model.get("cvar_alpha", -1.0)) != 0.95:
        raise WorldMoEConfigError("R5-V2 model CVaR alpha must be 0.95")


def validate_world_moe_config(
    config: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    """Validate immutable data, architecture, and final-test boundaries."""

    _validate_no_retired_runtime(config)
    root = Path(repository_root).resolve()
    schema_version = str(config.get("schema_version"))
    if schema_version not in {
        LEGACY_SCHEMA_VERSION,
        SCHEMA_VERSION,
        R5_V21_SCHEMA_VERSION,
        R5_V22_SCHEMA_VERSION,
        R5_V23_SCHEMA_VERSION,
        R5_V24_SCHEMA_VERSION,
        R5_V25_SCHEMA_VERSION,
    }:
        raise WorldMoEConfigError("unexpected WorldMoE schema_version")
    is_r5_v2 = schema_version == SCHEMA_VERSION
    is_r5_v21 = schema_version == R5_V21_SCHEMA_VERSION
    is_r5_v22 = schema_version == R5_V22_SCHEMA_VERSION
    is_r5_v23 = schema_version == R5_V23_SCHEMA_VERSION
    is_r5_v24 = schema_version == R5_V24_SCHEMA_VERSION
    is_r5_v25 = schema_version == R5_V25_SCHEMA_VERSION
    is_calibrated_v2 = (
        is_r5_v21 or is_r5_v22 or is_r5_v23 or is_r5_v24 or is_r5_v25
    )
    is_r5_v2_family = (
        is_r5_v2
        or is_r5_v21
        or is_r5_v22
        or is_r5_v23
        or is_r5_v24
        or is_r5_v25
    )

    project = _mapping(config.get("project"), "project")
    expected_architecture = (
        R5_V25_ARCHITECTURE_VERSION
        if is_r5_v25
        else R5_V24_ARCHITECTURE_VERSION
        if is_r5_v24
        else R5_V23_ARCHITECTURE_VERSION
        if is_r5_v23
        else R5_V22_ARCHITECTURE_VERSION
        if is_r5_v22
        else R5_V21_ARCHITECTURE_VERSION
        if is_r5_v21
        else ARCHITECTURE_VERSION
        if is_r5_v2
        else LEGACY_ARCHITECTURE_VERSION
    )
    if project.get("architecture_version") != expected_architecture:
        raise WorldMoEConfigError("configuration is not a registered R5 WorldMoE")
    expected_version = (
        "R5-V2.5"
        if is_r5_v25
        else "R5-V2.4"
        if is_r5_v24
        else "R5-V2.3"
        if is_r5_v23
        else "R5-V2.2"
        if is_r5_v22
        else "R5-V2.1"
        if is_r5_v21
        else "R5-V2"
        if is_r5_v2
        else "R5-V1"
    )
    if project.get("version") != expected_version:
        raise WorldMoEConfigError("project version does not match schema version")
    if project.get("implementation_namespace") != "dcvpp.vpp_world_moe":
        raise WorldMoEConfigError("WorldMoE requires its independent namespace")
    output_root = _resolve(root, project.get("output_root"), "project.output_root")
    if "world_moe" not in output_root.name.lower():
        raise WorldMoEConfigError("WorldMoE requires an independent output lineage")

    protocol = _mapping(config.get("protocol"), "protocol")
    if (
        tuple(str(value).lower() for value in protocol.get("phase_order", ()))
        != PHASE_ORDER
    ):
        raise WorldMoEConfigError(f"protocol.phase_order must be {list(PHASE_ORDER)}")
    splits = _mapping(protocol.get("splits"), "protocol.splits")
    train = _validate_split(splits, "train")
    validation = _validate_split(splits, "validation")
    refit = _validate_split(splits, "trainval_refit")
    final = _validate_split(splits, "final_test")
    if int(refit["days"]) != int(train["days"]) + int(validation["days"]):
        raise WorldMoEConfigError("trainval_refit must close train plus validation")
    if refit.get("allowed_only_after_method_selection") is not True:
        raise WorldMoEConfigError("refit requires a frozen method selection")
    if (
        int(final.get("calendar_days", -1)) != 365
        or int(final.get("missing_or_pending_days", -1)) != 3
        or final.get("sealed_until_all_methods_frozen") is not True
        or final.get("execute_once") is not True
    ):
        raise WorldMoEConfigError("2025 must remain sealed and execute exactly once")
    locked = _mapping(protocol.get("locked"), "protocol.locked")
    if (
        int(locked.get("year", -1)) != 2025
        or locked.get("reject_path_before_open") is not True
        or locked.get("release_only_after_refit_freeze") is not True
    ):
        raise WorldMoEConfigError("2025 fail-closed release contract was weakened")

    data = _mapping(config.get("data"), "data")
    if data.get("verify_daily_hashes") is not False:
        raise WorldMoEConfigError(
            "registered daily records must be reused without repetitive hash work"
        )
    paths = {
        "output_root": output_root,
        "selection_bundle_root": _resolve(
            root, data.get("selection_bundle_root"), "data.selection_bundle_root"
        ),
        "official_settlement_root": _resolve(
            root,
            data.get("official_settlement_root"),
            "data.official_settlement_root",
        ),
        "frozen_physical_twin": _resolve(
            root, data.get("frozen_physical_twin"), "data.frozen_physical_twin"
        ),
        "frozen_physical_surrogate": _resolve(
            root,
            data.get("frozen_physical_surrogate"),
            "data.frozen_physical_surrogate",
        ),
        "physical_calibration_manifest": _resolve(
            root,
            data.get("physical_calibration_manifest"),
            "data.physical_calibration_manifest",
        ),
    }
    for name in (
        "selection_bundle_root",
        "official_settlement_root",
        "frozen_physical_twin",
        "frozen_physical_surrogate",
        "physical_calibration_manifest",
    ):
        if not paths[name].exists():
            raise WorldMoEConfigError(f"required data path is missing: {paths[name]}")

    contexts = _mapping(data.get("causal_contexts"), "data.causal_contexts")
    for name, (tokens, hours) in CONTEXTS.items():
        value = _mapping(contexts.get(name), f"data.causal_contexts.{name}")
        if (
            int(value.get("tokens", -1)) != tokens
            or abs(float(value.get("resolution_hours", -1.0)) - hours) > 1e-12
            or value.get("history_cutoff")
            != "source_day_completed_no_later_than_issue_time"
        ):
            raise WorldMoEConfigError(f"{name} causal context contract was weakened")

    weather = _mapping(data.get("weather"), "data.weather")
    weather_transform = _mapping(
        weather.get("unit_transform"), "data.weather.unit_transform"
    )
    weather_source_window = _mapping(
        weather.get("source_window"), "data.weather.source_window"
    )
    weather_coverage = _mapping(
        weather.get("coverage_gate"), "data.weather.coverage_gate"
    )
    if (
        weather.get("enabled") is not True
        or weather.get("primary_source") != "nasa_power_lag_persistence"
        or weather.get("source_product")
        != "NASA_POWER_referenced_hourly_reconstruction"
        or weather.get("operational_weather_forecast_claimed") is not False
        or weather.get("evidence_role") != "reconstructed"
        or weather.get("quality") != "frozen_derived"
        or weather_source_window
        != {
            "start": "2021-01-01T00:00:00Z",
            "end_exclusive": "2025-01-01T00:00:00Z",
        }
        or tuple(weather.get("required_features", ()))
        != ("air_temperature_2m_k", "downward_shortwave_wm2")
        or tuple(weather.get("site_order", ()))
        != ("WEST", "CAPITL", "N.Y.C.", "LONGIL")
        or int(weather.get("slot", -1)) != 34
        or int(weather.get("source_id", -1)) != 9
        or weather.get("refresh") != "per_rolling_origin"
        or weather.get("availability")
        != "complete_source_day_max_deployable_available_at_utc_lte_origin"
        or weather.get("source_day_policy")
        != "latest_complete_visible_day_strictly_before_operating_day"
        or weather.get("coherent_source_day_required") is not True
        or weather.get("feature_complete_mask_required") is not True
        or weather.get("affects_day_ahead_commitment") is not False
        or weather.get("future_realized_values_allowed") is not False
        or weather.get("final_test_2025_opened") is not False
        or weather_coverage.get("formal_non_smoke_required") is not True
        or weather_coverage.get("criterion") != "origin0_all_96_slots_complete_per_day"
        or int(weather_coverage.get("train_required_days", -1)) != 722
        or int(weather_coverage.get("validation_required_days", -1)) != 364
    ):
        raise WorldMoEConfigError("weather rolling-origin causal contract was weakened")
    if (
        abs(float(weather_transform.get("temperature_center_k", 0.0)) - 273.15) > 1e-12
        or abs(float(weather_transform.get("temperature_scale_k", 0.0)) - 40.0) > 1e-12
        or abs(
            float(weather_transform.get("downward_shortwave_scale_wm2", 0.0)) - 1000.0
        )
        > 1e-12
    ):
        raise WorldMoEConfigError("weather deterministic unit transform changed")
    paths["weather_source"] = _resolve(
        root, weather.get("source_path"), "data.weather.source_path"
    )
    if not paths["weather_source"].is_file():
        raise WorldMoEConfigError(
            f"registered NASA POWER weather source is missing: {paths['weather_source']}"
        )

    migration = _mapping(data.get("migration"), "data.migration")
    _validate_migration_evidence(
        migration,
        root=root,
        output_root=output_root,
        paths=paths,
        allow_frozen_source_lineage=is_r5_v2_family,
    )
    _validate_physical_migration_separation(paths["frozen_physical_twin"])

    normalization = _mapping(data.get("normalization"), "data.normalization")
    if (
        normalization.get("fit_split") != "train_2022_2023_only"
        or normalization.get("validation_statistics_allowed") is not False
        or normalization.get("final_test_statistics_allowed") is not False
    ):
        raise WorldMoEConfigError("normalization must be fitted on 2022--2023 only")
    paths["normalization_artifact"] = _resolve(
        root,
        normalization.get("artifact_path"),
        "data.normalization.artifact_path",
    )
    paths["data_lineage"] = _resolve(
        root, data.get("lineage_path"), "data.lineage_path"
    )
    frozen_v21_data_root = (
        root / "artifacts" / "vpp_world_moe_r5_v2_1_2025final" / "data"
    ).resolve()
    for name in ("normalization_artifact", "data_lineage"):
        belongs_to_current_lineage = paths[name].is_relative_to(output_root)
        reuses_frozen_v21_data = (
            (is_r5_v22 or is_r5_v23 or is_r5_v24 or is_r5_v25)
            and data.get("frozen_source_lineage")
            == "R5-V2.1_train_only_typed_data_contract"
            and paths[name].is_relative_to(frozen_v21_data_root)
            and paths[name].is_file()
        )
        if not belongs_to_current_lineage and not reuses_frozen_v21_data:
            raise WorldMoEConfigError(f"{name} must belong to WorldMoE output root")

    model = _mapping(config.get("model"), "model")
    for key, expected in MODEL_SHAPE.items():
        if int(model.get(key, -1)) != expected:
            raise WorldMoEConfigError(f"model.{key} must be {expected}")
    if (
        int(model.get("d_model", -1)) != 256
        or int(model.get("blocks", -1)) != 10
        or int(model.get("heads", -1)) != 8
        or int(model.get("experts", -1)) != 8
        or int(model.get("top_k", -1)) != 2
        or tuple(int(value) for value in model.get("moe_blocks", ())) != (1, 3, 5, 7, 9)
    ):
        raise WorldMoEConfigError("registered WorldMoE depth or sparse routing changed")
    if is_r5_v2_family:
        tail_contract = {
            "general_experts": 7,
            "regular_top_k": 2,
            "conditional_tail_expert": True,
            "expert_assignment_schema": EXPERT_ASSIGNMENT_SCHEMA,
            "tail_capacity_fraction": (
                0.019967682659626007
                if is_r5_v23 or is_r5_v24 or is_r5_v25
                else 0.05
            ),
            "tail_capacity_max_fraction": 1.0,
            "tail_min_tokens": 11 if is_r5_v23 or is_r5_v24 or is_r5_v25 else 32,
            "tail_risk_threshold_work_units": 1.0e-4,
            "tail_activation_probability_threshold": (
                0.05 if is_calibrated_v2 else 0.50
            ),
            "tail_route_aux_weight": 0.02 if is_calibrated_v2 else 0.05,
            "tail_route_aux_temperature": 1.0,
            "tail_gate_input": "causal_issue_time_only",
            "tail_teacher_role": "auxiliary_supervision_and_diagnostics_only",
        }
        if is_calibrated_v2:
            tail_contract.update(
                {
                    "queue_occurrence_train_prevalence": 0.019967682659626007,
                    "tail_route_train_prevalence": 0.019967682659626007,
                    "tail_route_positive_weight": 49.08092411354107,
                    "tail_smooth_activation_temperature": 1.0,
                    "tail_residual_min_scale": 0.08,
                    "tail_residual_max_scale": 0.12,
                }
            )
        if any(model.get(name) != expected for name, expected in tail_contract.items()):
            raise WorldMoEConfigError(
                "R5-V2.x conditional tail expert assignment contract changed"
            )
        expected_dropout = 0.10 if is_calibrated_v2 else 0.0
        if abs(float(model.get("dropout", -1.0)) - expected_dropout) > 1.0e-12:
            raise WorldMoEConfigError("R5-V2.x dropout contract changed")
    if int(model.get("rollout_paths", -1)) != 20:
        raise WorldMoEConfigError("WorldMoE requires 20 internal imagination paths")
    if (
        int(model.get("expert_hidden_dim", -1)) != 1024
        or int(model.get("ffn_hidden_dim", -1)) != 1024
    ):
        raise WorldMoEConfigError(
            "WorldMoE expert and dense FFN widths must both be 1024"
        )
    if is_r5_v24 or is_r5_v25:
        support_fit_path = _resolve(
            root,
            model.get("support_atom_train_fit_artifact"),
            "model.support_atom_train_fit_artifact",
        )
        if not support_fit_path.is_file():
            raise WorldMoEConfigError(
                f"R5-V2.4 support-atom train fit is missing: {support_fit_path}"
            )
        support_fit_bytes = support_fit_path.read_bytes()
        expected_support_fit_sha256 = (
            "9575bdd67fc162936f35211d881fbab38c8485e040d67f94ab1179ec14c821e8"
        )
        if (
            model.get("support_atom_train_fit_sha256") != expected_support_fit_sha256
            or hashlib.sha256(support_fit_bytes).hexdigest()
            != expected_support_fit_sha256
        ):
            raise WorldMoEConfigError("R5-V2.4 support-atom train fit SHA-256 changed")
        raw_support_fit = json.loads(support_fit_bytes.decode("utf-8"))
        fit_prevalence = _float_mapping(
            _mapping(raw_support_fit, "support-atom train fit").get("prevalence"),
            "support-atom train fit prevalence",
        )
        expected_fit_prevalence = {
            "workload_zero": 0.25613169436749766,
            "pv_zero_global": 0.4815039242843952,
            "pv_capacity_global": 0.0046095914127423825,
            "pv_daylight_zero": 0.003116651825467498,
            "pv_daylight_capacity": 0.00889080587711487,
            "pv_night_zero": 0.9965843720038351,
            "pue_minimum_global": 0.013634002770083102,
            "pue_maximum_global": 0.25613169436749766,
            "pue_residual_minimum_given_nonzero_workload": 0.018328516844779767,
            "pue_residual_maximum_given_nonzero_workload": 0.0,
            "pue_maximum_given_zero_workload": 1.0,
        }
        if (
            raw_support_fit.get("schema")
            != "dcvpp.vpp-world-moe.support-atoms.train-fit.v1"
            or raw_support_fit.get("fit_split") != "train_2022_2023_only"
            or int(raw_support_fit.get("fit_days", -1)) != 722
            or raw_support_fit.get("opened_final_test_2025") is not False
            or float(raw_support_fit.get("night_shortwave_threshold_normalized", -1.0))
            != 1.0e-4
            or fit_prevalence != expected_fit_prevalence
        ):
            raise WorldMoEConfigError("R5-V2.4 support-atom train fit changed")
        paths["support_atom_train_fit"] = support_fit_path

    training = _mapping(config.get("training"), "training")
    if is_r5_v2_family:
        _validate_r5_v2_objective(
            training,
            model,
            revision=(
                "r5_v2_4"
                if is_r5_v24
                else "r5_v2_5"
                if is_r5_v25
                else "r5_v2_3"
                if is_r5_v23
                else "r5_v2_2"
                if is_r5_v22
                else "r5_v2_1"
                if is_r5_v21
                else "r5_v2"
            ),
        )
        capability = _mapping(config.get("capability_gates"), "capability_gates")
        world_state_gate = _mapping(
            capability.get("world_state"), "capability_gates.world_state"
        )
        registered_w0_gate = {
            "queue_occurrence_threshold_work_units": 1.0e-4,
            "queue_occurrence_probability_threshold": 0.50,
            "queue_positive_target_count_min": 1.0,
            "queue_paired_confidence": 0.95,
            "queue_occurrence_paired_upper_max": 0.0,
            "queue_severity_paired_upper_max": 0.0,
            "soc_paired_upper_max": 0.0,
        }
        if any(
            name not in world_state_gate
            or abs(float(world_state_gate[name]) - expected) > 1.0e-12
            for name, expected in registered_w0_gate.items()
        ):
            raise WorldMoEConfigError("R5-V2 W0 state capability thresholds changed")
        if is_r5_v25:
            uncertainty_gate = _mapping(
                capability.get("uncertainty"), "capability_gates.uncertainty"
            )
            if (
                int(uncertainty_gate.get("paths", -1)) != 100
                or int(uncertainty_gate.get("training_paths", -1)) != 16
                or uncertainty_gate.get("scenario_role")
                != "dynamic_score_particles_not_distribution_definition"
                or uncertainty_gate.get("analytic_mixed_support_required") is not True
            ):
                raise WorldMoEConfigError("R5-V2.5 uncertainty capability changed")
    phases = _mapping(training.get("phases"), "training.phases")
    if tuple(str(key).lower() for key in phases) != PHASE_ORDER:
        raise WorldMoEConfigError(
            f"training.phases must be ordered as {list(PHASE_ORDER)}"
        )
    for phase in PHASE_ORDER:
        _mapping(phases.get(phase), f"training.phases.{phase}")
    target_horizons = tuple(
        int(value)
        for value in _sequence(
            training.get("target_horizons"), "training.target_horizons"
        )
    )
    if target_horizons != (1, 4, 16, 96):
        raise WorldMoEConfigError("world dynamics targets must be h=1/4/16/96")
    return paths


def _deep_merge_config(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        previous = result.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge_config(previous, value)
        else:
            result[key] = value
    return result


def _read_config_with_base(
    source: Path, *, seen: frozenset[Path] = frozenset()
) -> dict[str, Any]:
    resolved = source.resolve()
    if resolved in seen:
        raise WorldMoEConfigError("configuration inheritance contains a cycle")
    raw = dict(
        _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            "configuration",
        )
    )
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (resolved.parent / str(parent)).resolve()
    if not parent_path.is_file():
        raise WorldMoEConfigError(f"base configuration does not exist: {parent_path}")
    base = _read_config_with_base(parent_path, seen=seen | {resolved})
    return _deep_merge_config(base, raw)


def load_world_moe_config(
    path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    source = Path(path).resolve()
    config = _read_config_with_base(source)
    root = source.parent.parent.resolve()
    return config, root, validate_world_moe_config(config, root)


__all__ = [
    "ACTION_FEATURE_NAMES",
    "ARCHITECTURE_VERSION",
    "CONSTRAINT_SLACK_NAMES",
    "CONTEXTS",
    "ENTITY_NAMES",
    "EXPERT_ASSIGNMENT_SCHEMA",
    "GLOBAL_SLOT_NAMES",
    "LEGACY_ARCHITECTURE_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "MODEL_SHAPE",
    "PHASE_ORDER",
    "R5_V21_ARCHITECTURE_VERSION",
    "R5_V21_SCHEMA_VERSION",
    "R5_V22_ARCHITECTURE_VERSION",
    "R5_V22_SCHEMA_VERSION",
    "R5_V23_ARCHITECTURE_VERSION",
    "R5_V23_SCHEMA_VERSION",
    "R5_V24_ARCHITECTURE_VERSION",
    "R5_V24_SCHEMA_VERSION",
    "R5_V25_ARCHITECTURE_VERSION",
    "R5_V25_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SPLITS",
    "WorldMoEConfig",
    "WorldMoEConfigError",
    "load_world_moe_config",
    "validate_world_moe_config",
]
