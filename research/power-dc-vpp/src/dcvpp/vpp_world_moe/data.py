"""Causal multi-scale data adapter for VPP-WorldMoE.

Only the registered 2022--2024 development corpus is reused.  Fast tokens are
issue-time features, while medium and slow tokens contain fully completed days
whose completion timestamp is no later than the current issue timestamp.
Future realized values are exposed only under explicit target keys.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from dcvpp.e2e_moe_r4_v7.corpus import PreloadedV7DevelopmentCorpus

from .config import WorldMoEConfig
from .lineage import (
    DATA_LINEAGE_SCHEMA,
    NORMALIZATION_SCHEMA,
    materialize_lineage_pair,
)
from .migration import (
    MigrationEvidenceError,
    MigrationEvidenceStore,
    MigrationSnapshot,
)
from .nasa_power_weather import (
    NASA_POWER_LAG_SOURCE,
    NasaPowerLagPersistenceStore,
)
from .packets import WorldConstraintPacket, WorldInputPacket
from .weather import NYISO_WEATHER_SITES, WeatherQueryResult, WorldWeatherStore


FAST_STEPS = 96
MEDIUM_STEPS = 168
SLOW_STEPS = 56
NODES = 4
ENTITIES_PER_NODE = 8
GLOBAL_SLOTS = 8
TOTAL_ENTITIES = NODES * ENTITIES_PER_NODE + GLOBAL_SLOTS
INPUT_DIM = 8
GRAPH_EDGE_DIM = 6
CONSTRAINT_DIM = 8
ACTION_DIM = 6
PUBLIC_CHANNELS = 21
DT_HOURS = 0.25
QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
# One registered definition is used by the corpus targets, the W0 hurdle loss,
# the capability gate, and the tail-expert teacher signal.  Values at or below
# this tolerance are numerical residue from the passive queue trace, not queue
# events.
QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS = 1.0e-4
TARGET_FAMILY_SLICES = {
    "price": slice(0, 12),
    "workload": slice(12, 13),
    "pv": slice(13, 17),
    "pue": slice(17, 21),
}
PUBLIC_CHANNEL_NAMES = (
    *(f"da_{node}" for node in ("WEST", "CAPITL", "NYC", "LONGIL")),
    *(f"rt_{node}" for node in ("WEST", "CAPITL", "NYC", "LONGIL")),
    *(f"dart_{node}" for node in ("WEST", "CAPITL", "NYC", "LONGIL")),
    "workload_volume",
    *(f"pv_{node}" for node in ("WEST", "CAPITL", "NYC", "LONGIL")),
    *(f"pue_{node}" for node in ("WEST", "CAPITL", "NYC", "LONGIL")),
)

# Entity ids follow config.ENTITY_NAMES (0..7) for each node and
# config.GLOBAL_SLOT_NAMES (8..15) for the eight global slots.
ENTITY_IDS = torch.tensor(
    [kind for _node in range(NODES) for kind in range(ENTITIES_PER_NODE)]
    + list(range(ENTITIES_PER_NODE, ENTITIES_PER_NODE + GLOBAL_SLOTS)),
    dtype=torch.long,
)
NODE_IDS = torch.tensor(
    [node for node in range(NODES) for _ in range(ENTITIES_PER_NODE)]
    + [NODES] * GLOBAL_SLOTS,
    dtype=torch.long,
)

SOURCE_MISSING = 0
SOURCE_ISSUE_MARKET = 1
SOURCE_ISSUE_WORKLOAD = 2
SOURCE_ISSUE_PV = 3
SOURCE_FROZEN_PHYSICAL = 4
SOURCE_CAUSAL_DERIVED = 5
SOURCE_COMPLETED_HISTORY = 6
SOURCE_CALENDAR = 7
SOURCE_CAUSAL_BASELINE_ACTION = 8
SOURCE_CAUSAL_DERIVED_WEATHER = 9
# Compatibility name for already-materialised HRRR ablation fixtures.
SOURCE_HRRR_WEATHER = SOURCE_CAUSAL_DERIVED_WEATHER

QUALITY_MISSING = 0
QUALITY_REGISTERED_ISSUE = 1
QUALITY_FROZEN_DERIVED = 2
QUALITY_COMPLETED_OBSERVED = 3
QUALITY_EXACT_STRUCTURAL = 4
QUALITY_MASKED_UNSUPPORTED = 5
QUALITY_CAUSAL_FORECAST = 6

WEATHER_SLOT = 34
WEATHER_FEATURES = ("air_temperature_2m_k", "downward_shortwave_wm2")
WEATHER_TEMPERATURE_CENTER_K = 273.15
WEATHER_TEMPERATURE_SCALE_K = 40.0
WEATHER_SHORTWAVE_SCALE_WM2 = 1000.0

EDGE_SELF = 0
EDGE_LOCAL_RELATION = 1
EDGE_CROSS_NODE_SAME_TYPE = 2
EDGE_GLOBAL_BRIDGE = 3
EDGE_GLOBAL_RELATION = 4
EDGE_MIGRATION_UNAVAILABLE = 5


class WorldDataError(ValueError):
    """WorldMoE data would violate a shape, evidence, or temporal boundary."""


def _tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu().to(dtype=dtype).clone()
    # torch.tensor copies read-only NumPy views before exposing writable storage.
    return torch.tensor(value, dtype=dtype)


def _close_public_price_target(value: Tensor) -> Tensor:
    if value.shape[-1] != PUBLIC_CHANNELS:
        raise WorldDataError("public target must end in 21 channels")
    result = value.float().clone()
    result[..., 8:12] = result[..., :4] - result[..., 4:8]
    return result


def _issue_time(record: Mapping[str, Any]) -> datetime:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise WorldDataError("record provenance is missing")
    raw = str(provenance.get("issue_time_utc", "")).replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise WorldDataError("issue_time_utc is invalid") from exc
    if result.tzinfo is None:
        raise WorldDataError("issue_time_utc must be timezone-aware")
    return result.astimezone(timezone.utc)


def _operating_date(record: Mapping[str, Any]) -> date:
    try:
        result = date.fromisoformat(str(record.get("operating_date")))
    except ValueError as exc:
        raise WorldDataError("record operating_date is invalid") from exc
    if result.year >= 2025:
        raise WorldDataError("locked 2025 record rejected before adapter construction")
    return result


def _completion_time(record: Mapping[str, Any]) -> datetime:
    return datetime.combine(
        _operating_date(record) + timedelta(days=1), time.min, tzinfo=timezone.utc
    )


def _physical_vector(physical: Any, name: str) -> Tensor:
    value = _tensor(getattr(physical, name))
    if value.shape != (NODES,) or not bool(torch.isfinite(value).all()):
        raise WorldDataError(f"physical.{name} must contain four finite values")
    return value


@dataclass(frozen=True)
class WorldNormalizationStatistics:
    """Robust typed feature statistics and envelopes fitted on train only.

    ``*_scale`` accepted the legacy ``[8]`` feature-only layout in R5-V2.  New
    corpora fit a separate ``[40,8]`` scale for every typed slot/feature pair
    and provide the matching median centers.  Keeping the legacy layout valid
    lets old packets and small compatibility fixtures continue to load, while
    every newly materialised R5-V2.1 corpus uses the typed contract.
    """

    fast_scale: Tensor
    medium_scale: Tensor
    slow_scale: Tensor
    pv_capacity: Tensor
    pue_min: Tensor
    pue_max: Tensor
    pue_issue_center: Tensor
    fit_dates: tuple[str, ...]
    fast_center: Tensor | None = None
    medium_center: Tensor | None = None
    slow_center: Tensor | None = None
    da_residual_cap: Tensor | None = None

    def validate(self) -> "WorldNormalizationStatistics":
        if (
            len(self.fit_dates) != 722
            or self.fit_dates != tuple(sorted(self.fit_dates))
            or len(set(self.fit_dates)) != 722
            or any(not day.startswith(("2022-", "2023-")) for day in self.fit_dates)
        ):
            raise WorldDataError("normalization requires the 722 ordered train dates")
        for name in ("fast_scale", "medium_scale", "slow_scale"):
            value = getattr(self, name)
            if (
                value.shape
                not in {
                    (INPUT_DIM,),
                    (TOTAL_ENTITIES, INPUT_DIM),
                }
                or not bool(torch.isfinite(value).all())
                or bool((value <= 0).any())
            ):
                raise WorldDataError(f"{name} must be finite positive [8] or [40,8]")
            center = getattr(self, name.replace("scale", "center"))
            if center is not None and (
                center.shape != value.shape or not bool(torch.isfinite(center).all())
            ):
                raise WorldDataError(
                    f"{name.replace('scale', 'center')} must match {name}"
                )
        for name in ("pv_capacity", "pue_min", "pue_max"):
            value = getattr(self, name)
            if value.shape != (NODES,) or not bool(torch.isfinite(value).all()):
                raise WorldDataError(f"{name} must contain four finite values")
        if bool((self.pv_capacity <= 0).any()):
            raise WorldDataError("train-derived PV envelope must be positive")
        if bool((self.pue_min < 1.0).any()) or bool(
            (self.pue_max < self.pue_min).any()
        ):
            raise WorldDataError("train-derived PUE envelope is invalid")
        if self.pue_issue_center.shape != (FAST_STEPS, NODES):
            raise WorldDataError("PUE issue center must be [96,4]")
        if self.da_residual_cap is not None and (
            self.da_residual_cap.shape != (NODES,)
            or not bool(torch.isfinite(self.da_residual_cap).all())
            or bool((self.da_residual_cap <= 0).any())
        ):
            raise WorldDataError("DA residual cap must be finite positive [4]")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validate()

        def legacy_feature_scale(value: Tensor) -> list[float]:
            # Preserve the original artifact keys and [8] shape for consumers
            # that only understand R5-V2.  R5-V2.1 readers use the explicit
            # slot-feature keys below.
            if value.ndim == 1:
                return value.tolist()
            return value.median(dim=0).values.tolist()

        def center(value: Tensor, explicit: Tensor | None) -> Tensor:
            return torch.zeros_like(value) if explicit is None else explicit

        fast_center = center(self.fast_scale, self.fast_center)
        medium_center = center(self.medium_scale, self.medium_center)
        slow_center = center(self.slow_scale, self.slow_center)
        payload = {
            "schema": NORMALIZATION_SCHEMA,
            "fit_split": "train_2022_2023_only",
            "fit_start": self.fit_dates[0],
            "fit_end": self.fit_dates[-1],
            "fit_days": len(self.fit_dates),
            "fit_dates": list(self.fit_dates),
            "used_validation_statistics": False,
            "opened_final_test_2025": False,
            "normalization_layout": (
                "typed_slot_feature_40x8"
                if self.fast_scale.ndim == 2
                else "legacy_feature_only_8"
            ),
            "center_statistic": "train_median_by_typed_slot_feature",
            "scale_statistic": (
                "train_mad_by_typed_slot_feature_with_iqr_or_unit_fallback"
            ),
            "fast_feature_scale": legacy_feature_scale(self.fast_scale),
            "medium_feature_scale": legacy_feature_scale(self.medium_scale),
            "slow_feature_scale": legacy_feature_scale(self.slow_scale),
            "fast_slot_feature_center": fast_center.tolist(),
            "fast_slot_feature_scale": self.fast_scale.tolist(),
            "medium_slot_feature_center": medium_center.tolist(),
            "medium_slot_feature_scale": self.medium_scale.tolist(),
            "slow_slot_feature_center": slow_center.tolist(),
            "slow_slot_feature_scale": self.slow_scale.tolist(),
            "pv_capacity_train_max_by_zone": self.pv_capacity.tolist(),
            "pue_train_min_by_zone": self.pue_min.tolist(),
            "pue_train_max_by_zone": self.pue_max.tolist(),
            "pue_issue_center_train_median_96x4": self.pue_issue_center.tolist(),
        }
        if self.da_residual_cap is not None:
            payload["forecast_residual_caps"] = {"da": self.da_residual_cap.tolist()}
            payload["forecast_residual_cap_metadata"] = {
                "schema": "dcvpp.vpp_world_moe.forecast_residual_caps.r5.v2.1",
                "fit_split": "train_2022_2023_only",
                "quantile": 0.95,
                "units": "raw_price_units_by_zone",
                "opened_final_test_2025": False,
            }
        return payload


@dataclass(frozen=True)
class WorldTargetStatistics:
    """Frozen proper-score metadata fitted exclusively on 2022--2023 targets."""

    center: Tensor
    scale: Tensor
    tail_thresholds: Tensor
    state_center: Tensor
    state_scale: Tensor
    queue_event_prevalence: Tensor
    baseline_daily_cost: Tensor
    fit_dates: tuple[str, ...]
    state_training_baseline_soc: Tensor | None = None
    state_training_baseline_queue_occurrence: Tensor | None = None
    state_training_baseline_queue_severity: Tensor | None = None
    state_gate_baseline_soc_persistence: Tensor | None = None
    state_training_baseline_soc_reference: str = "unspecified"

    def validate(self) -> "WorldTargetStatistics":
        if (
            len(self.fit_dates) != 722
            or self.fit_dates != tuple(sorted(self.fit_dates))
            or len(set(self.fit_dates)) != 722
            or any(not day.startswith(("2022-", "2023-")) for day in self.fit_dates)
        ):
            raise WorldDataError("target scoring metadata requires 722 train dates")
        for name in ("center", "scale"):
            value = getattr(self, name)
            if value.shape != (PUBLIC_CHANNELS,) or not bool(
                torch.isfinite(value).all()
            ):
                raise WorldDataError(f"target {name} must be finite [21]")
        if bool((self.scale <= 0).any()):
            raise WorldDataError("target score scales must be positive")
        if self.tail_thresholds.shape != (PUBLIC_CHANNELS, 2) or not bool(
            torch.isfinite(self.tail_thresholds).all()
        ):
            raise WorldDataError("tail thresholds must be finite [21,2]")
        for name in ("state_center", "state_scale"):
            value = getattr(self, name)
            if value.shape != (8,) or not bool(torch.isfinite(value).all()):
                raise WorldDataError(f"target {name} must be finite [8]")
        if bool((self.state_scale <= 0).any()):
            raise WorldDataError("state score scales must be positive")
        if (
            self.queue_event_prevalence.shape != ()
            or not bool(torch.isfinite(self.queue_event_prevalence))
            or not bool(
                (self.queue_event_prevalence > 0) & (self.queue_event_prevalence < 1)
            )
        ):
            raise WorldDataError(
                "train queue-event prevalence must be a finite scalar in (0,1)"
            )
        if (
            self.baseline_daily_cost.shape != ()
            or not bool(torch.isfinite(self.baseline_daily_cost))
            or not bool(self.baseline_daily_cost > 0)
        ):
            raise WorldDataError(
                "train-frozen baseline daily cost must be positive scalar"
            )
        for name in (
            "state_training_baseline_soc",
            "state_training_baseline_queue_occurrence",
            "state_training_baseline_queue_severity",
        ):
            value = getattr(self, name)
            if value is not None and (
                value.shape != ()
                or not bool(torch.isfinite(value))
                or not bool(value > 0)
            ):
                raise WorldDataError(f"{name} must be a finite positive scalar")
        if self.state_gate_baseline_soc_persistence is not None and (
            self.state_gate_baseline_soc_persistence.shape != ()
            or not bool(torch.isfinite(self.state_gate_baseline_soc_persistence))
            or bool(self.state_gate_baseline_soc_persistence < 0)
        ):
            raise WorldDataError(
                "state_gate_baseline_soc_persistence must be finite nonnegative scalar"
            )
        return self

    def family_mapping(
        self, value: Tensor, *, device: torch.device
    ) -> dict[str, Tensor]:
        return {
            family: value[channel_slice].to(device=device, dtype=torch.float32)
            for family, channel_slice in TARGET_FAMILY_SLICES.items()
        }

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema": "dcvpp.vpp_world_moe.target_score_statistics.r5.v2",
            "fit_split": "train_2022_2023_only",
            "fit_dates": list(self.fit_dates),
            "opened_final_test_2025": False,
            "quantile_levels": list(QUANTILE_LEVELS),
            "channel_names": list(PUBLIC_CHANNEL_NAMES),
            "family_slices": {
                "price": [0, 12],
                "workload": [12, 13],
                "pv": [13, 17],
                "pue": [17, 21],
            },
            "transforms": {
                "price": "signed_median_mad",
                "workload": "log1p_then_median_mad",
                "pv": "divide_by_train_node_capacity",
                "pue": "map_to_train_physical_bounds",
                "soc": "unit_interval",
                "queue": (
                    "occurrence_q_gt_1e-4_plus_positive_only_log1p_train_severity_scale"
                ),
            },
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "tail_threshold_quantiles": [0.05, 0.95],
            "tail_thresholds": self.tail_thresholds.tolist(),
            "state_center": self.state_center.tolist(),
            "state_scale": self.state_scale.tolist(),
            "queue_occurrence_threshold_work_units": (
                QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS
            ),
            "aggregate_queue_event_rule": "any_node_q_gt_threshold",
            "train_aggregate_queue_event_prevalence": float(
                self.queue_event_prevalence
            ),
            "train_frozen_baseline_daily_cost": float(self.baseline_daily_cost),
            "baseline_daily_cost_statistic": "mean_over_all_722_train_days_once",
        }
        baselines = {
            "soc": self.state_training_baseline_soc,
            "queue_occurrence": self.state_training_baseline_queue_occurrence,
            "queue_severity": self.state_training_baseline_queue_severity,
        }
        if all(value is not None for value in baselines.values()):
            payload["state_training_baselines"] = {
                name: float(value) for name, value in baselines.items()
            }
            payload["state_training_baseline_metadata"] = {
                "fit_split": "train_2022_2023_only",
                "soc": self.state_training_baseline_soc_reference,
                "queue_occurrence": "per_node_zero_probability_brier",
                "queue_severity": "positive_only_zero_severity_normalized_smooth_l1",
                "used_validation_statistics": False,
                "opened_final_test_2025": False,
            }
        if self.state_gate_baseline_soc_persistence is not None:
            payload["state_gate_baselines"] = {
                "soc_persistence": float(self.state_gate_baseline_soc_persistence)
            }
            payload["state_gate_baseline_metadata"] = {
                "fit_split": "train_2022_2023_only",
                "soc_persistence": (
                    "issue_time_persistence_normalized_smooth_l1_raw_unfloored"
                ),
                "training_normalizer": False,
                "opened_final_test_2025": False,
            }
        return payload


@dataclass(frozen=True)
class GraphContract:
    edge_index: Tensor
    edge_type: Tensor
    edge_features: Tensor
    edge_mask: Tensor

    def validate(self) -> "GraphContract":
        count = int(self.edge_index.shape[-1])
        if self.edge_index.shape != (2, count):
            raise WorldDataError("graph edge_index must be [2,E]")
        if self.edge_type.shape != (count,) or self.edge_features.shape != (
            count,
            GRAPH_EDGE_DIM,
        ):
            raise WorldDataError("graph type/features have inconsistent edge counts")
        if self.edge_mask.shape != (count,) or self.edge_mask.dtype is not torch.bool:
            raise WorldDataError("graph edge_mask must be bool [E]")
        return self


def _build_graph_contract(
    supported_migration_routes: Sequence[tuple[int, int]] = (),
) -> GraphContract:
    supported = {
        (int(source), int(destination))
        for source, destination in supported_migration_routes
    }
    edges: list[tuple[int, int, int]] = []
    for slot in range(TOTAL_ENTITIES):
        edges.append((slot, slot, EDGE_SELF))
    for node in range(NODES):
        slots = range(node * ENTITIES_PER_NODE, (node + 1) * ENTITIES_PER_NODE)
        for source in slots:
            for destination in slots:
                if source != destination:
                    edges.append((source, destination, EDGE_LOCAL_RELATION))
    for kind in range(ENTITIES_PER_NODE):
        slots = [node * ENTITIES_PER_NODE + kind for node in range(NODES)]
        for source in slots:
            for destination in slots:
                if source != destination:
                    edges.append((source, destination, EDGE_CROSS_NODE_SAME_TYPE))
    globals_ = range(NODES * ENTITIES_PER_NODE, TOTAL_ENTITIES)
    locals_ = range(NODES * ENTITIES_PER_NODE)
    for global_slot in globals_:
        for local_slot in locals_:
            edges.append((global_slot, local_slot, EDGE_GLOBAL_BRIDGE))
            edges.append((local_slot, global_slot, EDGE_GLOBAL_BRIDGE))
        for other in globals_:
            if global_slot != other:
                edges.append((global_slot, other, EDGE_GLOBAL_RELATION))
    # Every directed compute-to-compute route is represented.  Only routes with
    # direct evidence enter the static topology; rolling availability is still
    # gated by the dynamic as-of-origin mask.
    for source_node in range(NODES):
        for destination_node in range(NODES):
            if source_node != destination_node:
                edges.append(
                    (
                        source_node * ENTITIES_PER_NODE + 6,
                        destination_node * ENTITIES_PER_NODE + 6,
                        EDGE_MIGRATION_UNAVAILABLE,
                    )
                )
    edge_index = torch.tensor([(a, b) for a, b, _ in edges], dtype=torch.long).T
    edge_type = torch.tensor([kind for _, _, kind in edges], dtype=torch.long)
    features = torch.zeros(len(edges), GRAPH_EDGE_DIM, dtype=torch.float32)
    features[:, 0] = edge_type.eq(EDGE_SELF)
    features[:, 1] = edge_type.eq(EDGE_LOCAL_RELATION)
    features[:, 2] = edge_type.eq(EDGE_CROSS_NODE_SAME_TYPE)
    features[:, 3] = torch.isin(
        edge_type, torch.tensor([EDGE_GLOBAL_BRIDGE, EDGE_GLOBAL_RELATION])
    )
    features[:, 4] = edge_type.ne(EDGE_MIGRATION_UNAVAILABLE)
    features[:, 5] = edge_type.eq(EDGE_MIGRATION_UNAVAILABLE)
    mask = edge_type.ne(EDGE_MIGRATION_UNAVAILABLE)
    for index, (source, destination, kind) in enumerate(edges):
        if kind != EDGE_MIGRATION_UNAVAILABLE:
            continue
        source_node = source // ENTITIES_PER_NODE
        destination_node = destination // ENTITIES_PER_NODE
        available_topology = (source_node, destination_node) in supported
        features[index, 4] = float(available_topology)
        mask[index] = available_topology
    return GraphContract(edge_index, edge_type, features, mask).validate()


def _decode_issue_distributions(
    record: Mapping[str, Any], legacy_statistics: Any
) -> tuple[Tensor, Tensor, Tensor]:
    workload = _tensor(record["workload"])
    workload_scale = _tensor(legacy_statistics.workload_scale)
    if workload.shape != (FAST_STEPS, 5) or workload_scale.shape != (5,):
        raise WorldDataError("legacy workload distribution has an invalid shape")
    workload_raw = torch.expm1(workload * workload_scale)
    physical_dynamic = _tensor(record["physical_dynamic"])
    pv_scale = _tensor(legacy_statistics.pv_scale)
    if physical_dynamic.shape != (FAST_STEPS, 12) or pv_scale.shape != (8,):
        raise WorldDataError("legacy physical dynamics have an invalid shape")
    pv_distribution = torch.expm1(physical_dynamic[:, :8] * pv_scale)
    return workload_raw, pv_distribution[:, :4], pv_distribution[:, 4:8]


def _fit_empirical_envelopes(
    train_records: Sequence[Mapping[str, Any]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    targets = _close_public_price_target(
        torch.stack(
            [_tensor(record["scenario_target_zonal"]) for record in train_records]
        )
    )
    if targets.shape != (722, FAST_STEPS, PUBLIC_CHANNELS):
        raise WorldDataError("train target stack must be [722,96,21]")
    pv = targets[..., 13:17]
    pue = targets[..., 17:21]
    pv_capacity = pv.amax(dim=(0, 1)).clamp_min(1.0e-6)
    pue_min = pue.amin(dim=(0, 1)).clamp_min(1.0)
    pue_max = torch.maximum(pue.amax(dim=(0, 1)), pue_min)
    pue_issue_center = pue.median(dim=0).values
    return pv_capacity, pue_min, pue_max, pue_issue_center


def _median_mad(value: Tensor) -> tuple[Tensor, Tensor]:
    center = value.median(dim=0).values
    mad = (value - center).abs().median(dim=0).values
    return center, (1.4826 * mad).clamp_min(1.0e-6)


def _queue_event_supervision(
    queue: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return per-node and aggregate queue-event targets and valid masks."""

    if queue.ndim < 2 or queue.shape[-1] != NODES:
        raise WorldDataError("queue event supervision requires [...,4] values")
    valid = torch.isfinite(queue)
    if bool((queue[valid] < 0).any()):
        raise WorldDataError("queue event supervision requires nonnegative values")
    occurrence = valid & queue.gt(QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS)
    aggregate_mask = valid.all(dim=-1, keepdim=True)
    aggregate_event = occurrence.any(dim=-1, keepdim=True) & aggregate_mask
    return occurrence, valid, aggregate_event, aggregate_mask


def _positive_queue_log_statistics(queue: Tensor) -> tuple[Tensor, Tensor]:
    """Fit queue-severity statistics on positive train observations only.

    Queue occurrence is a separate Bernoulli target.  Including the structural
    zeros in a median/MAD severity fit makes both statistics exactly zero for
    this sparse corpus and previously collapsed the scale to ``1e-6``.  The
    positive-only fit keeps severity conditional on occurrence and uses an IQR
    or positive-magnitude fallback when the positive MAD itself is degenerate.
    """

    if queue.ndim < 2 or queue.shape[-1] != NODES:
        raise WorldDataError("queue severity fitting requires [...,4] values")
    if not bool(torch.isfinite(queue).all()) or bool((queue < 0).any()):
        raise WorldDataError(
            "queue severity fitting requires finite nonnegative values"
        )
    transformed = torch.log1p(queue.float()).reshape(-1, NODES)
    centers: list[Tensor] = []
    scales: list[Tensor] = []
    for node in range(NODES):
        raw_node = queue.float().reshape(-1, NODES)[:, node]
        positive = transformed[:, node][
            raw_node > QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS
        ]
        if positive.numel() == 0:
            raise WorldDataError(
                f"queue node {node} has no positive train severity observations"
            )
        center = positive.median()
        mad_scale = 1.4826 * (positive - center).abs().median()
        quartiles = torch.quantile(
            positive, torch.tensor((0.25, 0.75), device=positive.device)
        )
        iqr_scale = (quartiles[1] - quartiles[0]) / 1.349
        fallback = torch.maximum(iqr_scale, center.abs()).clamp_min(1.0e-3)
        scale = torch.where(mad_scale > 1.0e-3, mad_scale, fallback)
        centers.append(center)
        scales.append(scale)
    return torch.stack(centers).float(), torch.stack(scales).float()


def _fit_train_baseline_daily_cost(
    train_records: Sequence[Mapping[str, Any]],
) -> Tensor:
    """Fit one immutable daily-cost denominator over the complete train split."""

    if not train_records:
        raise WorldDataError("baseline daily cost requires nonempty train records")
    values: list[Tensor] = []
    for record in train_records:
        operating_date = str(record.get("operating_date", ""))
        if not operating_date.startswith(("2022-", "2023-")):
            raise WorldDataError(
                "baseline daily cost may use only 2022-2023 train days"
            )
        rollout_targets = record.get("rollout_targets")
        if (
            not isinstance(rollout_targets, Mapping)
            or "baseline_realized_cost" not in rollout_targets
        ):
            raise WorldDataError("train record omitted baseline_realized_cost")
        value = _tensor(rollout_targets["baseline_realized_cost"]).reshape(-1)
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            raise WorldDataError(
                "baseline_realized_cost must be one finite daily value"
            )
        values.append(value[0])
    result = torch.stack(values).mean().float()
    if not bool(result > 0):
        raise WorldDataError("mean train baseline daily cost must be positive")
    return result


def _fit_soc_training_baseline(
    soc_target: Tensor, soc_scale: Tensor
) -> tuple[Tensor, Tensor, str]:
    """Separate the raw persistence gate from a nondegenerate train scale."""

    if soc_target.ndim != 3 or soc_target.shape[-1] != NODES:
        raise WorldDataError("SOC baseline fitting requires [days,time,4]")
    scale = soc_scale.reshape(1, 1, NODES)
    persistence = soc_target[:, :1].expand_as(soc_target)
    persistence_gate = F.smooth_l1_loss(
        persistence / scale,
        soc_target / scale,
        reduction="mean",
    ).float()
    climatology = soc_target.reshape(-1, NODES).median(dim=0).values
    climatology_baseline = F.smooth_l1_loss(
        climatology.reshape(1, 1, NODES).expand_as(soc_target) / scale,
        soc_target / scale,
        reduction="mean",
    ).float()
    fixed_half_baseline = F.smooth_l1_loss(
        torch.full_like(soc_target, 0.5) / scale,
        soc_target / scale,
        reduction="mean",
    ).float()
    if bool(climatology_baseline > 1.0e-6):
        return (
            climatology_baseline,
            persistence_gate,
            "train_per_node_median_climatology_smooth_l1",
        )
    if bool(fixed_half_baseline > 1.0e-6):
        return (
            fixed_half_baseline,
            persistence_gate,
            "fixed_0p5_unit_soc_smooth_l1",
        )
    empty_battery_baseline = (
        F.smooth_l1_loss(
            torch.zeros_like(soc_target),
            soc_target / scale,
            reduction="mean",
        )
        .float()
        .clamp_min(1.0e-6)
    )
    return (
        empty_battery_baseline,
        persistence_gate,
        (
            "empty_battery_0p0_unit_soc_smooth_l1_fallback_after_"
            "degenerate_train_climatology_and_fixed_0p5"
        ),
    )


def _fit_target_statistics(
    train_records: Sequence[Mapping[str, Any]],
    *,
    pv_capacity: Tensor,
    pue_min: Tensor,
    pue_max: Tensor,
) -> WorldTargetStatistics:
    targets = torch.stack(
        [_tensor(record["scenario_target_zonal"]) for record in train_records]
    )
    states = torch.stack(
        [_tensor(record["state_sequence"]) for record in train_records]
    )
    if targets.shape != (722, FAST_STEPS, PUBLIC_CHANNELS):
        raise WorldDataError("proper-score fitting requires train targets [722,96,21]")
    if states.shape != (722, FAST_STEPS, 49):
        raise WorldDataError("proper-score fitting requires train states [722,96,49]")
    flat = targets.reshape(-1, PUBLIC_CHANNELS)
    center = torch.zeros(PUBLIC_CHANNELS)
    scale = torch.ones(PUBLIC_CHANNELS)
    center[:12], scale[:12] = _median_mad(flat[:, :12])
    workload = torch.log1p(flat[:, 12:13].clamp_min(0.0))
    center[12:13], scale[12:13] = _median_mad(workload)
    scale[13:17] = pv_capacity.clamp_min(1.0e-6)
    center[17:21] = pue_min
    scale[17:21] = (pue_max - pue_min).clamp_min(1.0e-6)
    tail_thresholds = (
        torch.quantile(flat, torch.tensor((0.05, 0.95)), dim=0)
        .transpose(0, 1)
        .contiguous()
    )

    state_center = torch.zeros(8)
    state_scale = torch.ones(8)
    state_center[4:8], state_scale[4:8] = _positive_queue_log_statistics(
        states[..., 4:8]
    )
    (
        queue_occurrence,
        queue_valid,
        aggregate_queue_event,
        aggregate_queue_mask,
    ) = _queue_event_supervision(states[..., 4:8])
    queue_event_prevalence = aggregate_queue_event[aggregate_queue_mask].float().mean()
    # Training-only denominators put the three heterogeneous state proper
    # scores on comparable scales without changing any of their optima.  The
    # raw ScoreTerms remain untouched for validation and paper reporting.
    soc_target = states[..., :4].float()
    # Persistence is the scientific gate comparator, not automatically a safe
    # training denominator.  In this corpus every train SOC target is exactly
    # 0.5, so both per-node train climatology and the fixed 0.5 reference are
    # perfect and therefore degenerate.  Fall back deterministically to the
    # empty-battery boundary (0.0), which is target-independent, physical, and
    # yields a stable unit-interval Smooth-L1 scale.  This fallback is fitted
    # without validation/final-test access and is recorded explicitly below.
    (
        soc_training_baseline,
        soc_persistence_gate_baseline,
        soc_training_reference,
    ) = _fit_soc_training_baseline(soc_target, state_scale[:4])
    queue_occurrence_baseline = (
        queue_occurrence[queue_valid].float().square().mean().clamp_min(1.0e-6)
    )
    severity_mask = queue_valid & queue_occurrence
    if not bool(severity_mask.any()):
        raise WorldDataError("queue-severity baseline requires positive train events")
    queue_scale = state_scale[4:8].reshape(1, 1, NODES)
    severity_values = F.smooth_l1_loss(
        torch.zeros_like(states[..., 4:8].float()),
        torch.log1p(states[..., 4:8].float().clamp_min(0.0)) / queue_scale,
        reduction="none",
    )
    queue_severity_baseline = (
        severity_values[severity_mask].mean().float().clamp_min(1.0e-6)
    )
    fit_dates = tuple(str(record["operating_date"]) for record in train_records)
    return WorldTargetStatistics(
        center=center,
        scale=scale,
        tail_thresholds=tail_thresholds,
        state_center=state_center,
        state_scale=state_scale,
        queue_event_prevalence=queue_event_prevalence,
        baseline_daily_cost=_fit_train_baseline_daily_cost(train_records),
        fit_dates=fit_dates,
        state_training_baseline_soc=soc_training_baseline,
        state_training_baseline_queue_occurrence=queue_occurrence_baseline,
        state_training_baseline_queue_severity=queue_severity_baseline,
        state_gate_baseline_soc_persistence=soc_persistence_gate_baseline,
        state_training_baseline_soc_reference=soc_training_reference,
    ).validate()


def _allocate_total(total: Tensor, capacity: Tensor) -> Tensor:
    if total.shape != (FAST_STEPS,) or capacity.shape != (NODES,):
        raise WorldDataError("workload allocation requires [96] and four capacities")
    clipped = total.clamp(0.0, float(capacity.sum()))
    return clipped[:, None] * capacity[None] / capacity.sum()


def _mark(
    mask: Tensor,
    source: Tensor,
    quality: Tensor,
    slot: int,
    source_id: int,
    quality_id: int,
) -> None:
    mask[:, slot] = True
    source[:, slot] = int(source_id)
    quality[:, slot] = int(quality_id)


def _inject_origin_queue_risk(
    fast_values: Tensor,
    fast_mask: Tensor,
    source_ids: Tensor,
    quality_ids: Tensor,
    constraints: Sequence[Mapping[str, Tensor]],
    *,
    feature_scale: Tensor,
    feature_center: Tensor | None = None,
) -> None:
    """Inject the held issue-time queue ratio into aggregate risk token 37."""

    batch, time_steps = fast_values.shape[:2]
    if (
        fast_values.shape != (batch, time_steps, TOTAL_ENTITIES, INPUT_DIM)
        or fast_mask.shape != (batch, time_steps, TOTAL_ENTITIES)
        or source_ids.shape != fast_mask.shape
        or quality_ids.shape != fast_mask.shape
        or len(constraints) != batch
        or feature_scale.shape
        not in {
            (INPUT_DIM,),
            (TOTAL_ENTITIES, INPUT_DIM),
        }
    ):
        raise WorldDataError("origin queue-risk injection has inconsistent shapes")
    ratios = torch.stack(
        [
            item["initial_queue"].sum()
            / item["compute_capacity"].sum().clamp_min(1.0e-6)
            for item in constraints
        ]
    )
    center = _normalization_center(feature_scale, feature_center)
    slot_center = _slot_feature_statistic(center, 37, 6)
    slot_scale = _slot_feature_statistic(feature_scale, 37, 6)
    fast_values[:, :, 37, 6] = (ratios[:, None] - slot_center) / slot_scale.clamp_min(
        1.0e-6
    )
    fast_mask[:, :, 37] = True
    source_ids[:, :, 37] = SOURCE_CAUSAL_DERIVED
    quality_ids[:, :, 37] = QUALITY_FROZEN_DERIVED


def _raw_fast_context(
    record: Mapping[str, Any],
    *,
    physical: Any,
    legacy_statistics: Any,
    deadline_intervals: int,
    pv_capacity: Tensor,
    pue_min: Tensor,
    pue_max: Tensor,
    pue_issue_center: Tensor,
) -> dict[str, Tensor]:
    values = torch.zeros(FAST_STEPS, TOTAL_ENTITIES, INPUT_DIM)
    mask = torch.zeros(FAST_STEPS, TOTAL_ENTITIES, dtype=torch.bool)
    source = torch.zeros(FAST_STEPS, TOTAL_ENTITIES, dtype=torch.long)
    quality = torch.zeros(FAST_STEPS, TOTAL_ENTITIES, dtype=torch.long)
    market = _tensor(record["market_raw"])
    if market.shape != (FAST_STEPS, 16):
        raise WorldDataError("market_raw must be [96,16]")
    workload, issue_pv, pv_std = _decode_issue_distributions(record, legacy_statistics)
    capacity = _physical_vector(physical, "capacity_work_units")
    battery_power = _physical_vector(physical, "battery_power_mw")
    battery_energy = _physical_vector(physical, "battery_energy_mwh")
    grid_limit = _physical_vector(physical, "grid_import_limit_mw")
    idle = _physical_vector(physical, "idle_mw")
    work_power = _physical_vector(physical, "work_mw_per_unit")
    multiplier = _tensor(record["physical_static"])
    initial_soc = _tensor(record["rollout_targets"]["initial_soc"])
    arrival_mean = _allocate_total(workload[:, 0], capacity)
    arrival_std = workload[:, 1:2] * capacity[None] / capacity.sum()
    arrival_q10 = workload[:, 2:3] * capacity[None] / capacity.sum()
    arrival_q50 = workload[:, 3:4] * capacity[None] / capacity.sum()
    arrival_q90 = workload[:, 4:5] * capacity[None] / capacity.sum()
    issue_grid = _tensor(record["rollout_targets"]["baseline_actions"]["day_ahead"]).T
    time_fraction = torch.arange(FAST_STEPS, dtype=torch.float32) / FAST_STEPS
    total_capacity = capacity.sum().clamp_min(1.0e-6)
    deadline = int(deadline_intervals)
    if not 1 <= deadline <= FAST_STEPS:
        raise WorldDataError("deadline_intervals must lie in [1,96]")
    deadline_q90_utilization = torch.stack(
        [
            workload[index : min(index + deadline, FAST_STEPS), 4].mean()
            / total_capacity
            for index in range(FAST_STEPS)
        ]
    )

    for node in range(NODES):
        base = node * ENTITIES_PER_NODE
        da = market[:, node]
        rt = market[:, 4 + node]
        rt_std = market[:, 12 + node].clamp_min(0.0)
        dart = da - rt
        values[:, base + 0, 0] = da
        values[:, base + 0, 1] = rt
        values[:, base + 0, 2] = dart
        values[:, base + 0, 3] = rt_std
        values[:, base + 0, 4] = rt - 1.28155 * rt_std
        values[:, base + 0, 5] = rt + 1.28155 * rt_std
        values[:, base + 0, 6] = dart.abs()
        _mark(
            mask,
            source,
            quality,
            base + 0,
            SOURCE_ISSUE_MARKET,
            QUALITY_REGISTERED_ISSUE,
        )

        soc_fraction = initial_soc[node] / battery_energy[node].clamp_min(1.0e-6)
        values[:, base + 1, 0] = soc_fraction
        values[:, base + 1, 1] = initial_soc[node]
        values[:, base + 1, 2] = battery_power[node]
        values[:, base + 1, 3] = battery_energy[node]
        values[:, base + 1, 4] = 1.0 - soc_fraction
        values[:, base + 1, 5] = multiplier[node]
        _mark(
            mask,
            source,
            quality,
            base + 1,
            SOURCE_FROZEN_PHYSICAL,
            QUALITY_FROZEN_DERIVED,
        )

        pv_mean = issue_pv[:, node].clamp_min(0.0)
        std = pv_std[:, node].clamp_min(0.0)
        values[:, base + 2, 0] = pv_mean
        values[:, base + 2, 1] = std
        values[:, base + 2, 2] = (pv_mean - 1.28155 * std).clamp_min(0.0)
        values[:, base + 2, 3] = pv_mean
        values[:, base + 2, 4] = (pv_mean + 1.28155 * std).clamp_max(pv_capacity[node])
        values[:, base + 2, 5] = pv_capacity[node]
        values[:, base + 2, 6] = (pv_capacity[node] - pv_mean).clamp_min(0.0)
        _mark(
            mask,
            source,
            quality,
            base + 2,
            SOURCE_ISSUE_PV,
            QUALITY_REGISTERED_ISSUE,
        )

        values[:, base + 3, 0] = pue_issue_center[:, node]
        values[:, base + 3, 1] = pue_min[node]
        values[:, base + 3, 2] = pue_max[node]
        values[:, base + 3, 3] = idle[node]
        values[:, base + 3, 4] = work_power[node]
        values[:, base + 3, 5] = multiplier[node]
        _mark(
            mask,
            source,
            quality,
            base + 3,
            SOURCE_FROZEN_PHYSICAL,
            QUALITY_FROZEN_DERIVED,
        )

        values[:, base + 4, 0] = arrival_mean[:, node]
        values[:, base + 4, 1] = arrival_std[:, node]
        values[:, base + 4, 2] = arrival_q10[:, node]
        values[:, base + 4, 3] = arrival_q50[:, node]
        values[:, base + 4, 4] = arrival_q90[:, node]
        values[:, base + 4, 5] = capacity[node]
        values[:, base + 4, 6] = (capacity[node] - arrival_mean[:, node]).clamp_min(0.0)
        _mark(
            mask,
            source,
            quality,
            base + 4,
            SOURCE_ISSUE_WORKLOAD,
            QUALITY_REGISTERED_ISSUE,
        )

        # Queue is exactly zero at issue under the registered passive-EDF state.
        values[:, base + 5, 3] = capacity[node]
        values[:, base + 5, 5] = capacity[node]
        _mark(
            mask,
            source,
            quality,
            base + 5,
            SOURCE_CAUSAL_DERIVED,
            QUALITY_EXACT_STRUCTURAL,
        )

        values[:, base + 6, 0] = arrival_mean[:, node]
        values[:, base + 6, 1] = arrival_std[:, node]
        values[:, base + 6, 2] = capacity[node]
        values[:, base + 6, 3] = arrival_mean[:, node] / capacity[node]
        values[:, base + 6, 4] = (capacity[node] - arrival_mean[:, node]).clamp_min(0.0)
        values[:, base + 6, 5] = work_power[node]
        _mark(
            mask,
            source,
            quality,
            base + 6,
            SOURCE_CAUSAL_DERIVED,
            QUALITY_FROZEN_DERIVED,
        )

        grid = issue_grid[:, node].clamp(0.0, grid_limit[node])
        values[:, base + 7, 0] = grid
        values[:, base + 7, 1] = grid_limit[node]
        values[:, base + 7, 2] = (grid_limit[node] - grid).clamp_min(0.0)
        values[:, base + 7, 3] = grid / grid_limit[node].clamp_min(1.0e-6)
        values[:, base + 7, 4] = 0.0  # structural no-export boundary
        _mark(
            mask,
            source,
            quality,
            base + 7,
            SOURCE_CAUSAL_DERIVED,
            QUALITY_EXACT_STRUCTURAL,
        )

    dart = market[:, :4] - market[:, 4:8]
    values[:, 32, 0] = market[:, :4].mean(-1)
    values[:, 32, 1] = market[:, 4:8].mean(-1)
    values[:, 32, 2] = dart.mean(-1)
    values[:, 32, 3] = market[:, 12:16].mean(-1)
    _mark(mask, source, quality, 32, SOURCE_ISSUE_MARKET, QUALITY_REGISTERED_ISSUE)
    values[:, 33, 0] = market[:, 12:16].amax(-1)
    values[:, 33, 1] = dart.abs().amax(-1)
    values[:, 33, 2] = dart.std(-1, unbiased=False)
    _mark(mask, source, quality, 33, SOURCE_ISSUE_MARKET, QUALITY_REGISTERED_ISSUE)
    # Weather is not present in this authoritative daily bundle and stays masked.
    quality[:, 34] = QUALITY_MASKED_UNSUPPORTED
    values[:, 35, 0] = workload[:, 0]
    values[:, 35, 1:5] = workload[:, 1:5]
    values[:, 35, 5] = capacity.sum()
    _mark(mask, source, quality, 35, SOURCE_ISSUE_WORKLOAD, QUALITY_REGISTERED_ISSUE)
    values[:, 36, 0] = (
        (grid_limit[None] - issue_grid.clamp_min(0.0)).clamp_min(0.0).mean(-1)
    )
    values[:, 36, 1] = (capacity[None] - arrival_mean).clamp_min(0.0).mean(-1)
    values[:, 36, 2] = (pv_capacity[None] - issue_pv).clamp_min(0.0).mean(-1)
    _mark(mask, source, quality, 36, SOURCE_CAUSAL_DERIVED, QUALITY_FROZEN_DERIVED)
    values[:, 37, 0] = market[:, 12:16].mean(-1)
    values[:, 37, 1] = dart.abs().mean(-1)
    # Aggregate risk token.  Every value is available at issue time: registered
    # workload forecast distribution, frozen compute capacity, deadline policy,
    # and (in _packet) the held initial queue.  No realized queue target enters
    # this token.
    values[:, 37, 2] = workload[:, 0] / total_capacity
    values[:, 37, 3] = workload[:, 4] / total_capacity
    values[:, 37, 4] = (workload[:, 4] - total_capacity).clamp_min(0.0) / total_capacity
    values[:, 37, 5] = deadline_q90_utilization
    values[:, 37, 6] = 0.0  # origin-specific initial queue ratio set in _packet
    _mark(mask, source, quality, 37, SOURCE_CAUSAL_DERIVED, QUALITY_FROZEN_DERIVED)
    # Action context is filled only from a causal prefix in batch construction.
    source[:, 38] = SOURCE_CAUSAL_BASELINE_ACTION
    quality[:, 38] = QUALITY_MISSING
    operating = _operating_date(record)
    values[:, 39, 0] = operating.weekday() / 6.0
    values[:, 39, 1] = (operating.month - 1) / 11.0
    values[:, 39, 2] = float(record.get("holiday", 0))
    values[:, 39, 3] = float(record.get("dst", 0))
    _mark(mask, source, quality, 39, SOURCE_CALENDAR, QUALITY_EXACT_STRUCTURAL)
    values[..., 7] = time_fraction[:, None]
    values = torch.where(mask[..., None], values, torch.zeros_like(values))
    return {"values": values, "mask": mask, "source": source, "quality": quality}


def _completed_day_context(
    record: Mapping[str, Any],
    *,
    physical: Any,
    pv_capacity: Tensor,
    pue_min: Tensor,
    pue_max: Tensor,
) -> tuple[Tensor, Tensor]:
    target = _tensor(record["scenario_target_zonal"])
    state = _tensor(record["state_sequence"])
    if target.shape != (FAST_STEPS, PUBLIC_CHANNELS) or state.shape != (FAST_STEPS, 49):
        raise WorldDataError("completed record target/state shape is invalid")
    capacity = _physical_vector(physical, "capacity_work_units")
    grid_limit = _physical_vector(physical, "grid_import_limit_mw")
    energy = _physical_vector(physical, "battery_energy_mwh")
    idle = _physical_vector(physical, "idle_mw")
    work_power = _physical_vector(physical, "work_mw_per_unit")
    multiplier = _tensor(record["physical_static"])
    work = _allocate_total(target[:, 12], capacity)
    pv = target[:, 13:17].clamp_min(0.0)
    load = multiplier[None] * (idle[None] + work_power[None] * work)
    grid = (load - pv).clamp_min(0.0)
    values = torch.zeros(FAST_STEPS, TOTAL_ENTITIES, INPUT_DIM)
    mask = torch.zeros(FAST_STEPS, TOTAL_ENTITIES, dtype=torch.bool)
    for node in range(NODES):
        base = node * ENTITIES_PER_NODE
        values[:, base + 0, 0] = target[:, node]
        values[:, base + 0, 1] = target[:, 4 + node]
        values[:, base + 0, 2] = target[:, 8 + node]
        values[:, base + 0, 3] = (
            target[:, 4 + node].diff(prepend=target[:1, 4 + node]).abs()
        )
        values[:, base + 0, 6] = target[:, 8 + node].abs()
        values[:, base + 1, 0] = state[:, node]
        values[:, base + 1, 1] = state[:, node] * energy[node]
        values[:, base + 1, 3] = energy[node]
        values[:, base + 1, 4] = 1.0 - state[:, node]
        values[:, base + 2, 0] = pv[:, node]
        values[:, base + 2, 3] = pv[:, node]
        values[:, base + 2, 5] = pv_capacity[node]
        values[:, base + 2, 6] = (pv_capacity[node] - pv[:, node]).clamp_min(0.0)
        values[:, base + 3, 0] = target[:, 17 + node]
        values[:, base + 3, 1] = pue_min[node]
        values[:, base + 3, 2] = pue_max[node]
        values[:, base + 3, 3] = idle[node]
        values[:, base + 3, 4] = work_power[node]
        values[:, base + 4, 0] = work[:, node]
        values[:, base + 4, 5] = capacity[node]
        values[:, base + 5, 0] = state[:, 4 + node]
        values[:, base + 5, 1] = state[:, 8 + node]
        values[:, base + 5, 5] = capacity[node]
        values[:, base + 6, 0] = state[:, 20 + node]
        values[:, base + 6, 2] = capacity[node]
        values[:, base + 6, 3] = state[:, 20 + node] / capacity[node]
        values[:, base + 7, 0] = grid[:, node]
        values[:, base + 7, 1] = grid_limit[node]
        values[:, base + 7, 2] = (grid_limit[node] - grid[:, node]).clamp_min(0.0)
        mask[:, base : base + ENTITIES_PER_NODE] = True
    values[:, 32, 0] = target[:, :4].mean(-1)
    values[:, 32, 1] = target[:, 4:8].mean(-1)
    values[:, 32, 2] = target[:, 8:12].mean(-1)
    values[:, 33, 0] = (
        target[:, 4:8].diff(dim=0, prepend=target[:1, 4:8]).abs().amax(-1)
    )
    values[:, 33, 1] = target[:, 8:12].abs().amax(-1)
    values[:, 35, 0] = target[:, 12]
    values[:, 35, 5] = capacity.sum()
    values[:, 36, 0] = (grid_limit[None] - grid).clamp_min(0.0).mean(-1)
    values[:, 36, 1] = (capacity[None] - work).clamp_min(0.0).mean(-1)
    values[:, 37, 0] = target[:, 8:12].abs().mean(-1)
    operating = _operating_date(record)
    values[:, 39, 0] = operating.weekday() / 6.0
    values[:, 39, 1] = (operating.month - 1) / 11.0
    mask[:, (32, 33, 35, 36, 37, 39)] = True
    values[..., 7] = torch.arange(FAST_STEPS, dtype=torch.float32)[:, None] / FAST_STEPS
    return torch.where(mask[..., None], values, torch.zeros_like(values)), mask


def _aggregate_completed_day(
    values: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hourly_values = values.reshape(24, 4, TOTAL_ENTITIES, INPUT_DIM).mean(dim=1)
    hourly_mask = mask.reshape(24, 4, TOTAL_ENTITIES).all(dim=1)
    hourly_values[..., 7] = torch.arange(24, dtype=torch.float32)[:, None] / 24.0
    daily_values = values.mean(dim=0, keepdim=True)
    daily_mask = mask.all(dim=0, keepdim=True)
    daily_values[..., 7] = 0.0
    return hourly_values, hourly_mask, daily_values, daily_mask


def _history_context(
    sources: Sequence[Mapping[str, Any]],
    cache: Mapping[str, tuple[Tensor, Tensor, Tensor, Tensor]],
    *,
    cutoff_day: date | None = None,
) -> dict[str, Tensor | tuple[str, ...]]:
    medium_values = torch.zeros(MEDIUM_STEPS, TOTAL_ENTITIES, INPUT_DIM)
    medium_mask = torch.zeros(MEDIUM_STEPS, TOTAL_ENTITIES, dtype=torch.bool)
    slow_values = torch.zeros(SLOW_STEPS, TOTAL_ENTITIES, INPUT_DIM)
    slow_mask = torch.zeros(SLOW_STEPS, TOTAL_ENTITIES, dtype=torch.bool)
    source_by_date = {str(record["operating_date"]): record for record in sources}
    if cutoff_day is None:
        medium_selected = list(sources[-7:])
        slow_selected = list(sources[-SLOW_STEPS:])
        medium_slots = list(enumerate(medium_selected, start=7 - len(medium_selected)))
        slow_slots = list(
            enumerate(slow_selected, start=SLOW_STEPS - len(slow_selected))
        )
    else:
        medium_slots = []
        slow_slots = []
        for index in range(7):
            day = (cutoff_day - timedelta(days=6 - index)).isoformat()
            if day in source_by_date:
                medium_slots.append((index, source_by_date[day]))
        for index in range(SLOW_STEPS):
            day = (cutoff_day - timedelta(days=SLOW_STEPS - 1 - index)).isoformat()
            if day in source_by_date:
                slow_slots.append((index, source_by_date[day]))
    for index, source_record in medium_slots:
        day = str(source_record["operating_date"])
        hourly_values, hourly_mask, _, _ = cache[day]
        start = index * 24
        medium_values[start : start + 24] = hourly_values
        medium_mask[start : start + 24] = hourly_mask
    for index, source_record in slow_slots:
        day = str(source_record["operating_date"])
        _, _, daily_values, daily_mask = cache[day]
        slow_values[index] = daily_values[0]
        slow_mask[index] = daily_mask[0]
    medium_values[..., 7] = torch.arange(MEDIUM_STEPS)[:, None] / MEDIUM_STEPS
    slow_values[..., 7] = torch.arange(SLOW_STEPS)[:, None] / SLOW_STEPS
    medium_values = torch.where(
        medium_mask[..., None], medium_values, torch.zeros_like(medium_values)
    )
    slow_values = torch.where(
        slow_mask[..., None], slow_values, torch.zeros_like(slow_values)
    )
    return {
        "medium_values": medium_values,
        "medium_mask": medium_mask,
        "medium_source_ids": torch.where(
            medium_mask,
            torch.tensor(SOURCE_COMPLETED_HISTORY),
            torch.tensor(SOURCE_MISSING),
        ).to(torch.long),
        "medium_quality_ids": torch.where(
            medium_mask,
            torch.tensor(QUALITY_COMPLETED_OBSERVED),
            torch.tensor(QUALITY_MISSING),
        ).to(torch.long),
        "slow_values": slow_values,
        "slow_mask": slow_mask,
        "slow_source_ids": torch.where(
            slow_mask,
            torch.tensor(SOURCE_COMPLETED_HISTORY),
            torch.tensor(SOURCE_MISSING),
        ).to(torch.long),
        "slow_quality_ids": torch.where(
            slow_mask,
            torch.tensor(QUALITY_COMPLETED_OBSERVED),
            torch.tensor(QUALITY_MISSING),
        ).to(torch.long),
        "medium_source_dates": tuple(
            str(record["operating_date"]) for _, record in medium_slots
        ),
        "slow_source_dates": tuple(
            str(record["operating_date"]) for _, record in slow_slots
        ),
    }


def _robust_slot_feature_statistics(
    records: Sequence[Mapping[str, Any]], values_key: str, mask_key: str
) -> tuple[Tensor, Tensor]:
    """Fit train-only median/MAD statistics for every typed slot and feature.

    A feature index does not have one global semantic meaning: feature zero is
    price in the market entity, SOC in the battery entity, PV in the solar
    entity, and workload in the compute entity.  Pooling those values produced
    the former ``[8]`` scale and let absolute prices dominate the shared value
    projection.  This fit deliberately keeps all forty typed slots separate.
    """

    if not records:
        raise WorldDataError("typed normalization requires nonempty train records")
    prepared: list[tuple[Tensor, Tensor]] = []
    for record in records:
        values = torch.as_tensor(record[values_key], dtype=torch.float32).detach().cpu()
        mask = torch.as_tensor(record[mask_key], dtype=torch.bool).detach().cpu()
        if values.ndim != 3 or values.shape[-2:] != (
            TOTAL_ENTITIES,
            INPUT_DIM,
        ):
            raise WorldDataError(
                f"{values_key} must be [time,{TOTAL_ENTITIES},{INPUT_DIM}]"
            )
        if mask.shape != values.shape[:-1]:
            raise WorldDataError(f"{mask_key} does not match {values_key}")
        prepared.append((values, mask))
    centers = torch.zeros(TOTAL_ENTITIES, INPUT_DIM, dtype=torch.float32)
    scales = torch.ones(TOTAL_ENTITIES, INPUT_DIM, dtype=torch.float32)
    for slot in range(TOTAL_ENTITIES):
        observed: list[Tensor] = []
        for values, mask in prepared:
            selected = values[:, slot][mask[:, slot]]
            if selected.numel():
                observed.append(selected)
        if not observed:
            continue
        values = torch.cat(observed, dim=0).float()
        center = values.median(dim=0).values
        mad = 1.4826 * (values - center).abs().median(dim=0).values
        quartiles = torch.quantile(
            values, torch.tensor((0.25, 0.75), dtype=values.dtype), dim=0
        )
        iqr_scale = (quartiles[1] - quartiles[0]) / 1.349
        # Constant typed channels are centered to zero.  A unit/magnitude
        # fallback avoids amplifying a small validation shift by 1e6.
        fallback = torch.maximum(iqr_scale, center.abs()).clamp_min(1.0)
        scale = torch.where(mad > 1.0e-6, mad, fallback)
        centers[slot] = center
        scales[slot] = scale
    return centers, scales


def _normalization_center(scale: Tensor, center: Tensor | None) -> Tensor:
    return torch.zeros_like(scale) if center is None else center


def _slot_feature_statistic(value: Tensor, slot: int, feature: int) -> Tensor:
    if value.shape == (INPUT_DIM,):
        return value[feature]
    if value.shape == (TOTAL_ENTITIES, INPUT_DIM):
        return value[slot, feature]
    raise WorldDataError("normalization statistic must be [8] or [40,8]")


def _normalize_record(
    raw: Mapping[str, Any], statistics: WorldNormalizationStatistics
) -> dict[str, Any]:
    result = dict(raw)
    for prefix, scale, center in (
        ("fast", statistics.fast_scale, statistics.fast_center),
        ("medium", statistics.medium_scale, statistics.medium_center),
        ("slow", statistics.slow_scale, statistics.slow_center),
    ):
        resolved_center = _normalization_center(scale, center)
        values = (_tensor(result[f"{prefix}_values"]) - resolved_center) / scale
        mask = torch.as_tensor(result[f"{prefix}_mask"], dtype=torch.bool)
        result[f"{prefix}_values"] = torch.where(
            mask[..., None], values, torch.zeros_like(values)
        )
    return result


def _causal_da_anchor(
    record: Mapping[str, Any],
    completed_history: Sequence[Mapping[str, Any]],
) -> tuple[Tensor, dict[str, Any]]:
    """Return the registered previous-complete-day DA path in raw units.

    The R4 corpus already registers this anchor.  The deterministic fallback is
    retained for older fixture records: use the latest history day whose full
    operating trajectory completed before issue, and use the issue-time DA only
    for the first in-window train day.  No target-day realization is consulted.
    """

    issue = _issue_time(record)
    raw_registered = record.get("forecast_anchors")
    if isinstance(raw_registered, Mapping) and raw_registered.get("da") is not None:
        registered = _tensor(raw_registered["da"])
        if registered.shape != (FAST_STEPS, NODES) or not bool(
            torch.isfinite(registered).all()
        ):
            raise WorldDataError("registered forecast_anchors.da must be finite [96,4]")
        raw_metadata = record.get("forecast_anchor_metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        return registered, {
            "schema": "dcvpp.vpp_world_moe.causal_price_anchor.r5.v2.1",
            "da_source": str(
                metadata.get(
                    "da_source",
                    "registered_latest_fully_completed_operating_day_before_issue",
                )
            ),
            "source_operating_date": metadata.get("source_operating_date"),
            "normalization": "none_raw_target_units",
            "issue_time_utc": issue.isoformat(),
            "opened_final_test_2025": False,
        }

    for source in reversed(completed_history):
        if _completion_time(source) <= issue:
            target = _tensor(source["scenario_target_zonal"])
            if target.shape != (FAST_STEPS, PUBLIC_CHANNELS):
                raise WorldDataError("completed DA anchor source must be [96,21]")
            return target[:, :NODES].clone(), {
                "schema": "dcvpp.vpp_world_moe.causal_price_anchor.r5.v2.1",
                "da_source": "latest_fully_completed_operating_day_before_issue",
                "source_operating_date": str(source["operating_date"]),
                "normalization": "none_raw_target_units",
                "issue_time_utc": issue.isoformat(),
                "opened_final_test_2025": False,
            }

    market = _tensor(record["market_raw"])
    if market.shape != (FAST_STEPS, 16):
        raise WorldDataError("DA anchor warmup requires market_raw [96,16]")
    return market[:, :NODES].clone(), {
        "schema": "dcvpp.vpp_world_moe.causal_price_anchor.r5.v2.1",
        "da_source": "first_train_day_issue_time_da_warmup",
        "source_operating_date": None,
        "normalization": "none_raw_target_units",
        "issue_time_utc": issue.isoformat(),
        "opened_final_test_2025": False,
    }


def _fit_da_residual_cap(train_records: Sequence[Mapping[str, Any]]) -> Tensor:
    """Fit the 95% absolute DA-anchor residual envelope on 2022--2023 only."""

    residuals: list[Tensor] = []
    for record in train_records:
        if not str(record.get("operating_date", "")).startswith(("2022-", "2023-")):
            raise WorldDataError("DA residual cap may use only 2022-2023 train days")
        anchor_mapping = record.get("forecast_anchors")
        if not isinstance(anchor_mapping, Mapping) or anchor_mapping.get("da") is None:
            raise WorldDataError("train record omitted causal DA anchor")
        anchor = _tensor(anchor_mapping["da"])
        target = _tensor(record["base_record"]["scenario_target_zonal"])[..., :NODES]
        if anchor.shape != target.shape or anchor.shape != (FAST_STEPS, NODES):
            raise WorldDataError("DA anchor and target must both be [96,4]")
        residuals.append((target - anchor).abs())
    if len(residuals) != 722:
        raise WorldDataError("DA residual cap requires all 722 train days")
    values = torch.stack(residuals).reshape(-1, NODES)
    return torch.quantile(values, 0.95, dim=0).float().clamp_min(1.0e-3)


def _same_split_next_day_map(
    records: Sequence[Mapping[str, Any]],
    split_by_date: Mapping[str, str],
) -> dict[str, Mapping[str, Any] | None]:
    by_date = {str(record["operating_date"]): record for record in records}
    result: dict[str, Mapping[str, Any] | None] = {}
    for record in records:
        current_text = str(record["operating_date"])
        current = date.fromisoformat(current_text)
        if current.year >= 2025:
            raise WorldDataError("locked 2025 record rejected before h96 pairing")
        next_text = (current + timedelta(days=1)).isoformat()
        # Never construct or inspect a 2025 record path.  This is an in-memory
        # lookup only and remains absent in the development inventory.
        candidate = (
            None
            if current.year == 2024 and current.month == 12 and current.day == 31
            else by_date.get(next_text)
        )
        if candidate is not None and split_by_date.get(next_text) == split_by_date.get(
            current_text
        ):
            result[current_text] = candidate
        else:
            result[current_text] = None
    return result


def _dynamic_graph(
    fast_values: Tensor,
    fast_mask: Tensor,
    graph: GraphContract,
    *,
    migration_edge_features: Tensor | None = None,
    migration_edge_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    # fast_values [B,T,40,F]; the last two graph channels are causal source
    # level and source-to-destination difference, not fabricated link facts.
    source_index = graph.edge_index[0]
    destination_index = graph.edge_index[1]
    source_value = fast_values[:, :, source_index, 0]
    destination_value = fast_values[:, :, destination_index, 0]
    result = (
        graph.edge_features[None, None]
        .expand(fast_values.shape[0], fast_values.shape[1], -1, -1)
        .clone()
    )
    result[..., 4] = source_value
    result[..., 5] = destination_value - source_value
    dynamic_mask = (
        fast_mask[:, :, source_index]
        & fast_mask[:, :, destination_index]
        & graph.edge_mask[None, None]
    )
    route_edges = graph.edge_type.eq(EDGE_MIGRATION_UNAVAILABLE)
    if migration_edge_features is not None or migration_edge_mask is not None:
        if migration_edge_features is None or migration_edge_mask is None:
            raise WorldDataError("migration graph features and mask must be paired")
        expected_features = (
            fast_values.shape[0],
            fast_values.shape[1],
            NODES,
            NODES,
            GRAPH_EDGE_DIM,
        )
        expected_mask = expected_features[:-1]
        if (
            migration_edge_features.shape != expected_features
            or migration_edge_mask.shape != expected_mask
            or migration_edge_mask.dtype is not torch.bool
        ):
            raise WorldDataError("migration graph evidence has an invalid shape")
        route_source = source_index[route_edges] // ENTITIES_PER_NODE
        route_destination = destination_index[route_edges] // ENTITIES_PER_NODE
        result[:, :, route_edges] = migration_edge_features[
            :, :, route_source, route_destination
        ]
        dynamic_mask[:, :, route_edges] = (
            migration_edge_mask[:, :, route_source, route_destination]
            & graph.edge_mask[route_edges][None, None]
        )
    result = torch.where(dynamic_mask[..., None], result, torch.zeros_like(result))
    return result, dynamic_mask


def _causal_actions_and_slacks(
    record: Mapping[str, Any],
    *,
    origin: int,
    physical: Any,
    statistics: WorldNormalizationStatistics,
    migration_snapshot: MigrationSnapshot | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    if not 0 <= int(origin) < FAST_STEPS:
        raise WorldDataError("origin must lie in [0,95]")
    state = _tensor(record["state_sequence"])
    capacity = _physical_vector(physical, "capacity_work_units")
    energy = _physical_vector(physical, "battery_energy_mwh")
    power = _physical_vector(physical, "battery_power_mw")
    grid_limit = _physical_vector(physical, "grid_import_limit_mw")
    idle = _physical_vector(physical, "idle_mw")
    work_power = _physical_vector(physical, "work_mw_per_unit")
    issue_pv = _tensor(record["issue_pv_raw"])
    issued_da = _tensor(record["rollout_targets"]["baseline_actions"]["day_ahead"]).T
    actions = torch.zeros(FAST_STEPS, NODES, ACTION_DIM)
    action_mask = torch.zeros(FAST_STEPS, NODES, ACTION_DIM, dtype=torch.bool)
    actions[..., 0] = issued_da
    action_mask[..., 0] = True
    if origin >= 1:
        prefix = slice(1, origin + 1)
        actions[prefix, :, 1] = state[prefix, 16:20]
        actions[prefix, :, 2] = state[prefix, 20:24]
        # Passive execution uses all available PV: curtailment is exactly zero.
        action_mask[prefix, :, 1] = True
        action_mask[prefix, :, 2] = True
        action_mask[prefix, :, 5] = True
    # No historical executed migration log exists, so previous migration action
    # channels remain masked even when a route is currently feasible.
    actions = torch.where(action_mask, actions, torch.zeros_like(actions))

    held = state.clone()
    held[origin + 1 :] = state[origin]
    slacks = torch.zeros(FAST_STEPS, NODES, CONSTRAINT_DIM)
    slacks[..., 0] = held[:, 0:4].clamp(0.0, 1.0)
    slacks[..., 1] = (grid_limit[None] - held[:, 12:16]).clamp_min(0.0) / grid_limit[
        None
    ].clamp_min(1.0e-6)
    slacks[..., 2] = (capacity[None] - held[:, 20:24]).clamp_min(0.0) / capacity[
        None
    ].clamp_min(1.0e-6)
    slacks[..., 3] = (1.0 - held[:, 8:12]).clamp(0.0, 1.0)
    pv_headroom = (statistics.pv_capacity[None] - issue_pv).clamp_min(
        0.0
    ) / statistics.pv_capacity[None]
    slacks[..., 4] = pv_headroom
    slacks[..., 5] = (1.0 - held[:, 8:12]).clamp(0.0, 1.0)
    migration = migration_snapshot or MigrationSnapshot.unavailable()
    if bool(migration.allowed.any()):
        route_scale = migration.capacity.max().clamp_min(1.0e-8)
        inbound = migration.capacity.sum(dim=0) / route_scale
        outbound = migration.capacity.sum(dim=1) / route_scale
        slacks[..., 6] = inbound[None].clamp(0.0, 1.0)
        slacks[..., 7] = outbound[None].clamp(0.0, 1.0)
    current_soc = held[origin, 0:4] * energy
    constraints = {
        "battery_power_limit": power,
        "battery_energy_capacity": energy,
        "initial_soc": current_soc,
        "compute_capacity": capacity,
        "initial_queue": held[origin, 4:8].clamp_min(0.0),
        "pv_capacity": statistics.pv_capacity,
        "grid_import_limit": grid_limit,
        "grid_export_limit": torch.zeros(NODES),
        "migration_capacity": migration.capacity,
        "migration_allowed": migration.allowed,
        "pue_min": statistics.pue_min,
        "pue_max": statistics.pue_max,
        "idle_power": idle,
        "work_power_per_unit": work_power,
        "charge_efficiency": torch.full((NODES,), 0.95),
        "discharge_efficiency": torch.full((NODES,), 0.95),
        "dt_hours": torch.tensor(DT_HOURS),
    }
    return actions, action_mask, slacks, constraints


def to_world_input_packet(value: Any) -> WorldInputPacket:
    """Return the typed packet accepted by :class:`VPPWorldMoE`."""

    if isinstance(value, WorldInputPacket):
        return value
    if isinstance(value, Mapping):
        for alias in ("world_input", "input_packet", "packet"):
            candidate = value.get(alias)
            if isinstance(candidate, WorldInputPacket):
                return candidate
        raw_constraints = value.get("constraints")
        if isinstance(raw_constraints, WorldConstraintPacket):
            constraints = raw_constraints
        elif isinstance(raw_constraints, Mapping):
            constraints = WorldConstraintPacket(**dict(raw_constraints))
        else:
            raise WorldDataError("WorldInput mapping omitted typed constraints")
        names = {
            "fast_values",
            "fast_mask",
            "entity_ids",
            "node_ids",
            "source_ids",
            "quality_ids",
            "graph_edge_index",
            "graph_edge_type",
            "graph_edge_features",
            "graph_edge_mask",
            "previous_actions",
            "action_mask",
            "constraint_slacks",
            "constraints",
            "origin_index",
            "dt_hours",
            "medium_values",
            "medium_mask",
            "medium_source_ids",
            "medium_quality_ids",
            "slow_values",
            "slow_mask",
            "slow_source_ids",
            "slow_quality_ids",
            "dynamic_graph_edge_features",
            "dynamic_graph_edge_mask",
            "known_future_mask",
        }
        selected = {key: item for key, item in value.items() if key in names}
        selected["constraints"] = constraints
        return WorldInputPacket(**selected)
    raise WorldDataError("value is not convertible to WorldInputPacket")


def _explicit_masked_target(
    record: Mapping[str, Any],
    *,
    value_keys: Sequence[str],
    expected: tuple[int, ...],
    integral: bool = False,
) -> tuple[Tensor, Tensor]:
    """Read a target only when its explicit validity mask accompanies it."""

    for value_key in value_keys:
        mask_key = f"{value_key}_mask"
        if value_key not in record or mask_key not in record:
            continue
        value = _tensor(record[value_key])
        mask = (
            torch.as_tensor(record[mask_key], dtype=torch.bool).detach().cpu().clone()
        )
        if value.shape == expected[:-1] and expected[-1] == 1:
            value = value.unsqueeze(-1)
        if mask.shape == expected[:-1] and expected[-1] == 1:
            mask = mask.unsqueeze(-1)
        if value.shape != expected or mask.shape != expected:
            raise WorldDataError(f"{value_key} and mask must have shape {expected}")
        if bool(mask.any()):
            selected = value[mask]
            if not bool(torch.isfinite(selected).all()) or bool((selected < 0).any()):
                raise WorldDataError(f"{value_key} contains invalid labelled values")
            if integral and not bool(
                torch.isclose(selected, selected.round(), atol=1.0e-5, rtol=0.0).all()
            ):
                raise WorldDataError(f"{value_key} is not a genuine count target")
        return torch.where(mask, value, torch.zeros_like(value)), mask
    return torch.zeros(expected), torch.zeros(expected, dtype=torch.bool)


def _future_supervision(
    current_targets: Tensor,
    next_targets: Tensor,
    current_states: Tensor,
    next_states: Tensor,
    *,
    origin: int,
) -> dict[str, Any]:
    """Create explicit 1/4/16/96 targets; h96 is always next-day truth."""

    if current_targets.ndim != 3 or current_targets.shape[1:] != (
        FAST_STEPS,
        PUBLIC_CHANNELS,
    ):
        raise WorldDataError("current observation targets must be [B,96,21]")
    if next_targets.shape != current_targets.shape:
        raise WorldDataError("next-day observation targets must match current targets")
    if (
        current_states.shape[1:] != (FAST_STEPS, 8)
        or next_states.shape != current_states.shape
    ):
        raise WorldDataError("current/next state targets must be [B,96,8]")
    if not 0 <= int(origin) < FAST_STEPS:
        raise WorldDataError("future supervision origin must lie in [0,95]")
    public = torch.cat((current_targets, next_targets), dim=1).float()
    states = torch.cat((current_states, next_states), dim=1).float()
    public_valid = torch.isfinite(public)
    state_valid = torch.isfinite(states)
    observation_targets: dict[str, Tensor] = {}
    observation_masks: dict[str, Tensor] = {}
    state_targets: dict[str, dict[str, Tensor]] = {}
    state_masks: dict[str, dict[str, Tensor]] = {}
    target_indices: dict[str, Tensor] = {}
    for horizon in (1, 4, 16):
        key = f"h{horizon}"
        indices = torch.arange(horizon, FAST_STEPS, dtype=torch.long)
        observation_targets[key] = public[:, indices]
        observation_masks[key] = public_valid[:, indices]
        state_targets[key] = {
            "soc": states[:, indices, :4],
            "queue": states[:, indices, 4:8],
        }
        state_masks[key] = {
            "soc": state_valid[:, indices, :4],
            "queue": state_valid[:, indices, 4:8],
        }
        target_indices[key] = indices
    next_index = FAST_STEPS + int(origin)
    h96_index = torch.tensor((next_index,), dtype=torch.long)
    observation_targets["h96"] = public[:, h96_index]
    observation_masks["h96"] = public_valid[:, h96_index]
    state_targets["h96"] = {
        "soc": states[:, h96_index, :4],
        "queue": states[:, h96_index, 4:8],
    }
    state_masks["h96"] = {
        "soc": state_valid[:, h96_index, :4],
        "queue": state_valid[:, h96_index, 4:8],
    }
    target_indices["h96"] = h96_index
    return {
        "future_target_sequence": public,
        "future_target_mask": public_valid,
        "future_state_sequence": states,
        "future_state_mask": state_valid,
        "observation_targets": observation_targets,
        "observation_target_masks": observation_masks,
        "state_rollout_targets": state_targets,
        "state_rollout_masks": state_masks,
        "observation_target_indices": target_indices,
    }


def _w1_teacher_action_summary(
    current_states: Tensor,
    next_states: Tensor,
    *,
    origin: int,
) -> tuple[Tensor, Tensor]:
    """Build a 96-step continuous passive/actual action teacher trajectory.

    The 49-column passive EDF state trace stores interval ``t-1`` execution in
    state row ``t``.  This function is deliberately separate from the 8-column
    SOC/queue target packet, so action supervision cannot be mistaken for a
    causal model input.
    """

    if (
        current_states.ndim != 3
        or current_states.shape[1] != FAST_STEPS
        or current_states.shape[-1] < 40
        or next_states.shape != current_states.shape
    ):
        raise WorldDataError(
            "W1 action teacher requires matching [B,96,S>=40] state traces"
        )
    if not 0 <= int(origin) < FAST_STEPS:
        raise WorldDataError("W1 action teacher origin must lie in [0,95]")
    full = torch.cat((current_states, next_states), dim=1).float()
    action_state = full[:, 1:]
    batch = action_state.shape[0]
    battery = action_state[..., 16:20].mean(dim=-1)
    compute = action_state[..., 20:24].mean(dim=-1)
    migration = action_state[..., 24:40].reshape(batch, 191, NODES, NODES)
    migration_total = migration.sum(dim=(-1, -2)) / float(NODES)
    grid = action_state[..., 12:16].mean(dim=-1)
    summary = torch.stack(
        (
            battery,
            compute,
            migration_total,
            grid,
            torch.zeros_like(grid),
            torch.zeros_like(grid),
        ),
        dim=-1,
    )
    # Row 96 is the next day's reset state, not an executed boundary action.
    # Interpolate its passive baseline instead of injecting a false zero ramp.
    summary = summary.clone()
    summary[:, 95] = 0.5 * (summary[:, 94] + summary[:, 96])
    indices = int(origin) + torch.arange(FAST_STEPS, device=summary.device)
    selected = summary[:, indices]
    valid = torch.isfinite(selected)
    selected = torch.where(valid, selected, torch.zeros_like(selected))
    return selected, valid


class WorldMoECorpus:
    """Preloaded 722/364 WorldMoE corpus with train-only feature fitting."""

    def __init__(
        self,
        config: Mapping[str, Any],
        repository_root: str | Path,
        *,
        base_corpus: PreloadedV7DevelopmentCorpus | None = None,
        weather_store: WorldWeatherStore | NasaPowerLagPersistenceStore | None = None,
        weather_coverage_gate: bool = False,
        materialize: bool = True,
    ) -> None:
        self.config = dict(config)
        self.repository_root = Path(repository_root).resolve()
        self.model_config = WorldMoEConfig.from_mapping(
            config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
        )
        self.base = base_corpus or PreloadedV7DevelopmentCorpus(
            config, self.repository_root
        )
        self.physical = self.base.physical
        data_config = config.get("data")
        if not isinstance(data_config, Mapping):
            raise WorldDataError("config.data is required")
        self.deadline_intervals = int(data_config.get("deadline_intervals", 4))
        if self.deadline_intervals <= 0 or self.deadline_intervals > FAST_STEPS:
            raise WorldDataError("data.deadline_intervals must lie in [1,96]")
        weather_config = data_config.get("weather", {})
        if not isinstance(weather_config, Mapping):
            raise WorldDataError("config.data.weather must be a mapping")
        self.weather_source = str(
            weather_config.get("primary_source", NASA_POWER_LAG_SOURCE)
        )
        if self.weather_source != NASA_POWER_LAG_SOURCE:
            raise WorldDataError(
                "formal weather primary_source must be nasa_power_lag_persistence"
            )
        self.weather_store = weather_store
        if self.weather_store is None and weather_config.get("enabled") is True:
            raw_source_path = weather_config.get("source_path")
            if not isinstance(raw_source_path, str) or not raw_source_path.strip():
                raise WorldDataError(
                    "enabled weather requires data.weather.source_path"
                )
            source_path = Path(raw_source_path)
            if not source_path.is_absolute():
                source_path = self.repository_root / source_path
            self.weather_store = NasaPowerLagPersistenceStore(source_path)
        migration_config = data_config.get("migration")
        if not isinstance(migration_config, Mapping):
            raise WorldDataError("config.data.migration is required")
        self.migration: MigrationEvidenceStore | None = None
        if migration_config.get("enabled") is True:
            availability = migration_config.get("availability")
            if not isinstance(availability, Mapping):
                raise WorldDataError("migration availability thresholds are required")
            evidence_path = Path(str(migration_config.get("evidence_path")))
            manifest_path = Path(str(migration_config.get("manifest_path")))
            if not evidence_path.is_absolute():
                evidence_path = self.repository_root / evidence_path
            if not manifest_path.is_absolute():
                manifest_path = self.repository_root / manifest_path
            try:
                self.migration = MigrationEvidenceStore.from_artifacts(
                    evidence_path,
                    manifest_path,
                    max_age_minutes=float(availability.get("max_age_minutes")),
                    max_packet_loss_rate=float(
                        availability.get("max_packet_loss_rate")
                    ),
                    min_sent_packets=int(availability.get("min_sent_packets")),
                )
            except (MigrationEvidenceError, TypeError, ValueError) as exc:
                raise WorldDataError(
                    "registered migration evidence could not be loaded"
                ) from exc
        self.graph = _build_graph_contract(
            tuple(self.migration.supported_routes) if self.migration else ()
        )
        train = list(self.base.records("train"))
        validation = list(self.base.records("validation"))
        if len(train) != 722 or len(validation) != 364:
            raise WorldDataError(
                f"WorldMoE requires 722/364 days, got {len(train)}/{len(validation)}"
            )
        all_records = sorted(
            [*train, *validation], key=lambda item: str(item["operating_date"])
        )
        if any(_operating_date(record).year >= 2025 for record in all_records):
            raise WorldDataError("2025 entered the development record inventory")
        self._split_by_date = {
            str(record["operating_date"]): "train" for record in train
        } | {str(record["operating_date"]): "validation" for record in validation}
        pv_capacity, pue_min, pue_max, pue_center = _fit_empirical_envelopes(train)

        completed_cache: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        for record in all_records:
            values, mask = _completed_day_context(
                record,
                physical=self.physical,
                pv_capacity=pv_capacity,
                pue_min=pue_min,
                pue_max=pue_max,
            )
            completed_cache[str(record["operating_date"])] = _aggregate_completed_day(
                values, mask
            )

        raw_records: list[dict[str, Any]] = []
        completed: list[Mapping[str, Any]] = []
        for record in all_records:
            issue = _issue_time(record)
            eligible = [item for item in completed if _completion_time(item) <= issue]
            da_anchor, da_anchor_metadata = _causal_da_anchor(record, eligible)
            history = _history_context(
                eligible,
                completed_cache,
                cutoff_day=issue.date() - timedelta(days=1),
            )
            fast = _raw_fast_context(
                record,
                physical=self.physical,
                legacy_statistics=self.base.statistics,
                deadline_intervals=self.deadline_intervals,
                pv_capacity=pv_capacity,
                pue_min=pue_min,
                pue_max=pue_max,
                pue_issue_center=pue_center,
            )
            raw_records.append(
                {
                    "operating_date": str(record["operating_date"]),
                    "split": self._split_by_date[str(record["operating_date"])],
                    "fast_values": fast["values"],
                    "fast_mask": fast["mask"],
                    "source_ids": fast["source"],
                    "quality_ids": fast["quality"],
                    "forecast_anchors": {"da": da_anchor},
                    "forecast_anchor_metadata": da_anchor_metadata,
                    **history,
                    "base_record": record,
                    "issue_time_utc": issue.isoformat(),
                }
            )
            completed.append(record)

        raw_train = [record for record in raw_records if record["split"] == "train"]
        fit_dates = tuple(str(record["operating_date"]) for record in raw_train)
        fast_center, fast_scale = _robust_slot_feature_statistics(
            raw_train, "fast_values", "fast_mask"
        )
        medium_center, medium_scale = _robust_slot_feature_statistics(
            raw_train, "medium_values", "medium_mask"
        )
        slow_center, slow_scale = _robust_slot_feature_statistics(
            raw_train, "slow_values", "slow_mask"
        )
        self.statistics = WorldNormalizationStatistics(
            fast_scale=fast_scale,
            medium_scale=medium_scale,
            slow_scale=slow_scale,
            pv_capacity=pv_capacity,
            pue_min=pue_min,
            pue_max=pue_max,
            pue_issue_center=pue_center,
            fit_dates=fit_dates,
            fast_center=fast_center,
            medium_center=medium_center,
            slow_center=slow_center,
            da_residual_cap=_fit_da_residual_cap(raw_train),
        ).validate()
        self.target_statistics = _fit_target_statistics(
            train,
            pv_capacity=pv_capacity,
            pue_min=pue_min,
            pue_max=pue_max,
        )
        normalized = [
            _normalize_record(record, self.statistics) for record in raw_records
        ]
        self._by_date = {str(record["operating_date"]): record for record in normalized}
        self._records = {
            "train": tuple(
                record for record in normalized if record["split"] == "train"
            ),
            "validation": tuple(
                record for record in normalized if record["split"] == "validation"
            ),
        }
        self._records["trainval"] = self._records["train"] + self._records["validation"]
        self._next_day = _same_split_next_day_map(normalized, self._split_by_date)
        self.weather_coverage: dict[str, dict[str, int]] = {}
        for split_name in ("train", "validation"):
            complete = 0
            partial = 0
            missing = 0
            for record in self._records[split_name]:
                _, weather_mask = self._weather_context(record, origin=0)
                available = int(weather_mask.sum().item())
                if available == FAST_STEPS:
                    complete += 1
                elif available > 0:
                    partial += 1
                else:
                    missing += 1
            self.weather_coverage[split_name] = {
                "days": len(self._records[split_name]),
                "complete_origin0_days": complete,
                "partial_origin0_days": partial,
                "missing_origin0_days": missing,
            }
        if weather_coverage_gate:
            failures = {
                split: summary
                for split, summary in self.weather_coverage.items()
                if summary["complete_origin0_days"] != summary["days"]
            }
            if failures:
                raise WorldDataError(
                    f"formal weather coverage gate failed before W0: {failures}"
                )
        self.weather_artifact_count = (
            self.weather_store.artifact_count() if self.weather_store is not None else 0
        )
        split_lineage = getattr(self.base, "split_lineage", {})
        self.data_lineage = {
            "schema": DATA_LINEAGE_SCHEMA,
            "complete_days": {
                "train": 722,
                "validation": 364,
                "trainval_refit": 1086,
            },
            "split_contract": dict(split_lineage),
            "normalization_fit": "train_2022_2023_only",
            "proper_score_metadata": {
                "fit": "train_2022_2023_only",
                "quantiles": list(QUANTILE_LEVELS),
                "tail_threshold_quantiles": [0.05, 0.95],
                "validation_statistics_allowed": False,
                "final_test_2025_opened": False,
                "workload_count": "true_label_and_mask_only_no_volume_proxy",
                "deadline_intervals": self.deadline_intervals,
                "expired_work_without_label": "masked_not_approximated_by_queue",
            },
            "causal_contexts": {
                "fast": "96x15min_issue_time",
                "medium": "168x1h_completed_history",
                "slow": "56x1day_completed_history",
                "history_cutoff": "source_day_completed_no_later_than_issue_time",
                "aggregate_risk_slot_37": {
                    "feature_0": "mean_registered_rt_standard_deviation",
                    "feature_1": "mean_absolute_issue_time_dart",
                    "feature_2": "aggregate_arrival_mean_over_compute_capacity",
                    "feature_3": "aggregate_arrival_q90_over_compute_capacity",
                    "feature_4": "positive_q90_capacity_exceedance_ratio",
                    "feature_5": "deadline_window_q90_utilization",
                    "feature_6": "held_initial_queue_over_compute_capacity",
                    "feature_7": "time_fraction",
                    "realized_queue_target_visible": False,
                },
            },
            "h96_supervision": {
                "source": "exact_same_split_next_calendar_day",
                "unavailable_pairs_are_filtered": True,
                "same_day_substitute_allowed": False,
                "train_pairs": sum(
                    self._next_day[str(record["operating_date"])] is not None
                    for record in self._records["train"]
                ),
                "validation_pairs": sum(
                    self._next_day[str(record["operating_date"])] is not None
                    for record in self._records["validation"]
                ),
            },
            "previous_actions": {
                "source": "causal_passive_edf_baseline_trace",
                "empirical_executed_dispatch_available": False,
                "future_prefix_values_visible": False,
                "day_ahead_commitment": "issue_time_derived_schedule",
            },
            "physical": {
                "artifact_metadata": dict(getattr(self.base, "physical_lineage", {})),
                "battery_efficiency_contract": "registered_runtime_value_0.95",
                "grid_export": "structural_zero_import_only_runtime",
            },
            "migration": {
                "available": self.migration is not None,
                "enabled_routes": [["WEST", "N.Y.C."], ["N.Y.C.", "WEST"]],
                "unsupported_route_count": 10,
                "capacity_source": "ub_nysernet_published_100gbps_upper_bound",
                "latency_source": "ripe_atlas_direct_anchor_mesh_ipv4",
                "capacity_evidence_role": (
                    "operational_proxy_sensitivity_not_execution_log"
                ),
                "action_mask": "ripe_asof_origin_gated",
                "selection": "latest_result_timestamp_lte_origin",
                "future_results_allowed": False,
                "final_test_2025_opened": False,
                "manifest_schema": (
                    self.migration.manifest.get("schema")
                    if self.migration is not None
                    else None
                ),
            },
            "weather": {
                "adapter_enabled": self.weather_store is not None,
                "artifact_count": self.weather_artifact_count,
                "available": (
                    self.weather_artifact_count > 0
                    and any(
                        summary["complete_origin0_days"] > 0
                        for summary in self.weather_coverage.values()
                    )
                ),
                "primary_source": "NASA POWER referenced hourly reconstruction",
                "adapter": "NASA_POWER_hourly_reconstruction_lag_persistence_v1",
                "evidence_role": "reconstructed",
                "quality": "frozen_derived",
                "operational_weather_forecast_claimed": False,
                "source_lineage": (
                    self.weather_store.lineage()
                    if isinstance(self.weather_store, NasaPowerLagPersistenceStore)
                    else {}
                ),
                "optional_ablation": "NOAA_HRRR_and_NBM_not_formal_gate",
                "weather_refresh": "per_rolling_origin",
                "as_of": (
                    "complete_source_day_max_deployable_available_at_utc_lte_"
                    "rolling_origin"
                ),
                "source_day_policy": (
                    "latest_complete_visible_historical_day_strictly_before_"
                    "operating_day"
                ),
                "temporal_alignment": "same_time_of_day_lag_persistence_96x15min",
                "slot": WEATHER_SLOT,
                "channel_mapping": [
                    "TMP_WEST",
                    "TMP_CAPITL",
                    "TMP_N.Y.C.",
                    "TMP_LONGIL",
                    "DSWRF_WEST",
                    "DSWRF_CAPITL",
                    "DSWRF_N.Y.C.",
                    "DSWRF_LONGIL",
                ],
                "deterministic_unit_transform": {
                    "temperature": "(kelvin-273.15)/40",
                    "downward_shortwave": "watt_per_m2/1000",
                    "fit_on_validation_or_final_test": False,
                },
                "packet_mask": "all_2_features_x_4_sites_required_per_token",
                "future_realized_nasa_values_allowed": False,
                "coverage_gate": {
                    "formal_non_smoke_required": True,
                    "criterion": "origin0_all_96_slots_complete_per_day",
                    "per_split": self.weather_coverage,
                },
                "affects_day_ahead_commitment": False,
                "final_test_2025_opened": False,
            },
            "final_test_2025": {
                "status": "SEALED_ONE_SHOT",
                "opened": False,
                "development_adapter_constructs_2025_paths": False,
            },
        }
        if materialize:
            self.materialize_data_artifacts()

    def records(
        self, split: str, *, require_h96: bool = False
    ) -> Sequence[Mapping[str, Any]]:
        name = str(split).lower()
        if name == "trainval_refit":
            name = "trainval"
        if name not in self._records:
            raise WorldDataError("split must be train, validation, or trainval_refit")
        records = self._records[name]
        if require_h96:
            return tuple(
                record
                for record in records
                if self._next_day[str(record["operating_date"])] is not None
            )
        return records

    def world_input_packet(self, value: Any) -> WorldInputPacket:
        """Runner-compatible adapter for an already collated mapping."""

        return to_world_input_packet(value).validate(self.model_config)

    def materialize_data_artifacts(self) -> tuple[Path, Path]:
        data = self.config.get("data")
        if not isinstance(data, Mapping):
            raise WorldDataError("config.data is required to materialize lineage")
        normalization = data.get("normalization")
        if not isinstance(normalization, Mapping):
            raise WorldDataError("config.data.normalization is required")
        normalization_path = Path(str(normalization.get("artifact_path")))
        lineage_path = Path(str(data.get("lineage_path")))
        if not normalization_path.is_absolute():
            normalization_path = self.repository_root / normalization_path
        if not lineage_path.is_absolute():
            lineage_path = self.repository_root / lineage_path
        return materialize_lineage_pair(
            normalization_path=normalization_path,
            normalization={
                **self.statistics.as_dict(),
                "target_score_statistics": self.target_statistics.as_dict(),
            },
            lineage_path=lineage_path,
            lineage=self.data_lineage,
        )

    def _weather_context(
        self,
        record: Mapping[str, Any],
        *,
        origin: int,
    ) -> tuple[Tensor, Tensor]:
        values = torch.zeros(FAST_STEPS, INPUT_DIM)
        mask = torch.zeros(FAST_STEPS, dtype=torch.bool)
        if self.weather_store is None or not 0 <= origin < FAST_STEPS:
            return values, mask
        operating_start = datetime.combine(
            _operating_date(record["base_record"]), time.min, tzinfo=timezone.utc
        )
        decision_origin = operating_start + timedelta(minutes=15 * int(origin))
        valid_times = tuple(
            operating_start + timedelta(minutes=15 * step)
            for step in range(origin, FAST_STEPS)
        )
        if isinstance(self.weather_store, NasaPowerLagPersistenceStore):
            dense = self.weather_store.query_causal_array_as_of(
                decision_origin,
                valid_times=valid_times,
                source=self.weather_source,
                zones=tuple(NYISO_WEATHER_SITES),
                features=WEATHER_FEATURES,
            )
            if dense.values.shape != (
                len(valid_times),
                NODES,
                len(WEATHER_FEATURES),
            ):
                return values, mask
            raw = torch.from_numpy(dense.values)
            feature_mask = torch.from_numpy(dense.mask)
        else:
            result: WeatherQueryResult = self.weather_store.query_causal_hold_as_of(
                decision_origin,
                valid_times=valid_times,
                source=self.weather_source,
                zones=tuple(NYISO_WEATHER_SITES),
                features=WEATHER_FEATURES,
            )
            cube = result.to_cube()
            if cube.values.shape != (
                1,
                len(valid_times),
                NODES,
                len(WEATHER_FEATURES),
            ):
                return values, mask
            raw = torch.from_numpy(cube.values[0])
            feature_mask = torch.from_numpy(cube.mask[0])
        temperature = raw[..., 0]
        shortwave = raw[..., 1]
        plausible = feature_mask.clone()
        plausible[..., 0] &= temperature.ge(180.0) & temperature.le(340.0)
        plausible[..., 1] &= shortwave.ge(-5.0) & shortwave.le(1600.0)
        complete = plausible.all(dim=(-1, -2))
        transformed = torch.cat(
            (
                (temperature - WEATHER_TEMPERATURE_CENTER_K)
                / WEATHER_TEMPERATURE_SCALE_K,
                shortwave.clamp_min(0.0) / WEATHER_SHORTWAVE_SCALE_WM2,
            ),
            dim=-1,
        ).to(torch.float32)
        transformed = torch.where(
            complete[:, None], transformed, torch.zeros_like(transformed)
        )
        values[origin:] = transformed
        mask[origin:] = complete
        return values, mask

    def _packet(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        origin: int,
        device: torch.device,
    ) -> WorldInputPacket:
        fast_values = torch.stack(
            [_tensor(record["fast_values"]) for record in records]
        )
        fast_mask = torch.stack(
            [
                torch.as_tensor(record["fast_mask"], dtype=torch.bool)
                for record in records
            ]
        )
        source_ids = torch.stack(
            [
                torch.as_tensor(record["source_ids"], dtype=torch.long)
                for record in records
            ]
        )
        quality_ids = torch.stack(
            [
                torch.as_tensor(record["quality_ids"], dtype=torch.long)
                for record in records
            ]
        )
        for index, record in enumerate(records):
            weather_values, weather_mask = self._weather_context(
                record,
                origin=origin,
            )
            fast_values[index, :, WEATHER_SLOT] = weather_values
            fast_mask[index, :, WEATHER_SLOT] = weather_mask
            source_ids[index, :, WEATHER_SLOT] = torch.where(
                weather_mask,
                torch.tensor(SOURCE_CAUSAL_DERIVED_WEATHER),
                torch.tensor(SOURCE_MISSING),
            )
            quality_ids[index, :, WEATHER_SLOT] = torch.where(
                weather_mask,
                torch.tensor(QUALITY_FROZEN_DERIVED),
                torch.tensor(QUALITY_MISSING),
            )
        actions: list[Tensor] = []
        action_masks: list[Tensor] = []
        slacks: list[Tensor] = []
        constraints: list[dict[str, Tensor]] = []
        migration_features: list[Tensor] = []
        migration_masks: list[Tensor] = []
        for record in records:
            if self.migration is None:
                day_features = torch.zeros(FAST_STEPS, NODES, NODES, GRAPH_EDGE_DIM)
                day_mask = torch.zeros(FAST_STEPS, NODES, NODES, dtype=torch.bool)
                migration_snapshot = MigrationSnapshot.unavailable()
            else:
                day_features, day_mask, migration_snapshot = (
                    self.migration.causal_day_series(
                        _operating_date(record["base_record"]),
                        origin=origin,
                        steps=FAST_STEPS,
                    )
                )
            action, action_mask, slack, constraint = _causal_actions_and_slacks(
                record["base_record"],
                origin=origin,
                physical=self.physical,
                statistics=self.statistics,
                migration_snapshot=migration_snapshot,
            )
            actions.append(action)
            action_masks.append(action_mask)
            slacks.append(slack)
            constraints.append(constraint)
            migration_features.append(day_features)
            migration_masks.append(day_mask)
        # fast_values are already train-normalized at this boundary.  Feature 6
        # was structurally zero while fitting origin-0 records, so applying the
        # registered feature scale here preserves the same normalization rule.
        _inject_origin_queue_risk(
            fast_values,
            fast_mask,
            source_ids,
            quality_ids,
            constraints,
            feature_scale=self.statistics.fast_scale,
            feature_center=self.statistics.fast_center,
        )
        previous_actions = torch.stack(actions)
        action_mask = torch.stack(action_masks)
        constraint_slacks = torch.stack(slacks)
        # Slot 38 is the typed action context.  Only the causal action prefix is visible.
        prefix_available = action_mask.any(dim=(-1, -2))
        fast_center = _normalization_center(
            self.statistics.fast_scale, self.statistics.fast_center
        )
        for feature, action_channel in enumerate((0, 1, 2)):
            center = _slot_feature_statistic(fast_center, 38, feature)
            scale = _slot_feature_statistic(
                self.statistics.fast_scale, 38, feature
            ).clamp_min(1.0e-6)
            fast_values[:, :, 38, feature] = (
                previous_actions[..., action_channel].mean(-1) - center
            ) / scale
        fast_mask[:, :, 38] = prefix_available
        source_ids[:, :, 38] = SOURCE_CAUSAL_BASELINE_ACTION
        quality_ids[:, :, 38] = torch.where(
            prefix_available,
            torch.tensor(QUALITY_FROZEN_DERIVED),
            torch.tensor(QUALITY_MISSING),
        )
        dynamic_features, dynamic_mask = _dynamic_graph(
            fast_values,
            fast_mask,
            self.graph,
            migration_edge_features=torch.stack(migration_features),
            migration_edge_mask=torch.stack(migration_masks),
        )
        constraint_packet = WorldConstraintPacket(
            **{
                name: torch.stack([item[name] for item in constraints]).to(device)
                for name in constraints[0]
            }
        )
        packet = WorldInputPacket(
            fast_values=fast_values.to(device),
            fast_mask=fast_mask.to(device),
            medium_values=torch.stack(
                [_tensor(record["medium_values"]) for record in records]
            ).to(device),
            medium_mask=torch.stack(
                [
                    torch.as_tensor(record["medium_mask"], dtype=torch.bool)
                    for record in records
                ]
            ).to(device),
            medium_source_ids=torch.stack(
                [
                    torch.as_tensor(record["medium_source_ids"], dtype=torch.long)
                    for record in records
                ]
            ).to(device),
            medium_quality_ids=torch.stack(
                [
                    torch.as_tensor(record["medium_quality_ids"], dtype=torch.long)
                    for record in records
                ]
            ).to(device),
            slow_values=torch.stack(
                [_tensor(record["slow_values"]) for record in records]
            ).to(device),
            slow_mask=torch.stack(
                [
                    torch.as_tensor(record["slow_mask"], dtype=torch.bool)
                    for record in records
                ]
            ).to(device),
            slow_source_ids=torch.stack(
                [
                    torch.as_tensor(record["slow_source_ids"], dtype=torch.long)
                    for record in records
                ]
            ).to(device),
            slow_quality_ids=torch.stack(
                [
                    torch.as_tensor(record["slow_quality_ids"], dtype=torch.long)
                    for record in records
                ]
            ).to(device),
            entity_ids=ENTITY_IDS.to(device),
            node_ids=NODE_IDS.to(device),
            source_ids=source_ids.to(device),
            quality_ids=quality_ids.to(device),
            graph_edge_index=self.graph.edge_index.to(device),
            graph_edge_type=self.graph.edge_type.to(device),
            graph_edge_features=self.graph.edge_features.to(device),
            graph_edge_mask=self.graph.edge_mask.to(device),
            dynamic_graph_edge_features=dynamic_features.to(device),
            dynamic_graph_edge_mask=dynamic_mask.to(device),
            known_future_mask=fast_mask.clone().to(device),
            previous_actions=previous_actions.to(device),
            action_mask=action_mask.to(device),
            constraint_slacks=constraint_slacks.to(device),
            constraints=constraint_packet,
            origin_index=torch.full(
                (len(records),), int(origin), dtype=torch.long, device=device
            ),
            dt_hours=torch.full((len(records), FAST_STEPS), DT_HOURS, device=device),
        )
        return packet.validate(self.model_config)

    def batches(
        self,
        split: str,
        batch_size: int,
        *,
        device: torch.device | str,
        shuffle_seed: int | None = None,
        max_days: int | None = None,
        origin: int = 0,
        require_h96: bool = False,
        include_next_day: bool = False,
        shard_index: int = 0,
        shard_count: int = 1,
        start_batch: int = 0,
        batch_ordinals: Sequence[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        records = list(self.records(split, require_h96=require_h96))
        if shuffle_seed is not None:
            generator = torch.Generator().manual_seed(int(shuffle_seed))
            order = torch.randperm(len(records), generator=generator).tolist()
            records = [records[index] for index in order]
        if max_days is not None:
            records = records[: int(max_days)]
        size = int(batch_size)
        if size <= 0:
            raise WorldDataError("batch_size must be positive")
        count = int(shard_count)
        index = int(shard_index)
        offset = int(start_batch)
        if count <= 0 or index < 0 or index >= count:
            raise WorldDataError("batch shard must satisfy 0 <= index < count")
        if offset < 0:
            raise WorldDataError("start_batch must be nonnegative")
        if require_h96 and not include_next_day:
            raise WorldDataError("require_h96 requires include_next_day")
        starts = list(range(0, len(records), size))
        if batch_ordinals is not None:
            if count != 1 or index != 0 or offset != 0:
                raise WorldDataError(
                    "explicit batch ordinals cannot be combined with sharding"
                )
            selected_ordinals = tuple(int(value) for value in batch_ordinals)
            if tuple(sorted(set(selected_ordinals))) != selected_ordinals:
                raise WorldDataError("batch_ordinals must be sorted and unique")
            if any(value < 0 or value >= len(starts) for value in selected_ordinals):
                raise WorldDataError("batch_ordinals contains an out-of-range batch")
            starts = [starts[value] for value in selected_ordinals]
        else:
            # Partition the already-defined full-epoch batches.  Across all shards
            # every original optimizer update appears exactly once, including the
            # final short batch, while skipped shards never construct GPU packets.
            shard_start = len(starts) * index // count
            shard_stop = len(starts) * (index + 1) // count
            starts = starts[shard_start:shard_stop][offset:]
        target_device = torch.device(device)
        horizons = (1, 4, 16, 96)
        for start in starts:
            selected = records[start : start + size]
            packet = self._packet(selected, origin=origin, device=target_device)
            next_records = [
                self._next_day[str(record["operating_date"])] for record in selected
            ]
            all_next = all(record is not None for record in next_records)
            if require_h96 and not all_next:
                raise WorldDataError("W1 batch admitted a missing h96 next-day target")
            current_targets = _close_public_price_target(
                torch.stack(
                    [
                        _tensor(record["base_record"]["scenario_target_zonal"])
                        for record in selected
                    ]
                )
            )
            future_sequence = torch.zeros(len(selected), 192, PUBLIC_CHANNELS)
            future_mask = torch.zeros(
                len(selected), 192, PUBLIC_CHANNELS, dtype=torch.bool
            )
            future_sequence[:, :96] = current_targets
            future_mask[:, :96] = torch.isfinite(current_targets)
            for index, next_record in enumerate(next_records):
                if include_next_day and next_record is not None:
                    future_sequence[index, 96:] = _close_public_price_target(
                        _tensor(next_record["base_record"]["scenario_target_zonal"])
                    )
                    future_mask[index, 96:] = torch.isfinite(
                        future_sequence[index, 96:]
                    )
            states = torch.stack(
                [
                    _tensor(record["base_record"]["state_sequence"])
                    for record in selected
                ]
            )
            soc = states[..., :4]
            queue = states[..., 4:8]
            (
                queue_occurrence,
                queue_valid,
                queue_risk_event_target,
                queue_risk_event_mask,
            ) = _queue_event_supervision(queue)
            # The passive teacher trace represents one VPP backlog and is
            # replicated across nodes.  Route supervision must therefore use
            # one aggregate event rather than counting that event four times.
            # This target remains outside WorldInputPacket and is never visible
            # to the causal forward/inference path.
            quantile_targets = {
                family: current_targets[..., channel_slice].to(target_device)
                for family, channel_slice in TARGET_FAMILY_SLICES.items()
            }
            quantile_masks = {
                family: torch.isfinite(value)
                for family, value in quantile_targets.items()
            }
            quantile_scales = self.target_statistics.family_mapping(
                self.target_statistics.scale, device=target_device
            )
            quantile_centers = self.target_statistics.family_mapping(
                self.target_statistics.center, device=target_device
            )
            tail_thresholds = self.target_statistics.family_mapping(
                self.target_statistics.tail_thresholds, device=target_device
            )
            count_targets: list[Tensor] = []
            count_masks: list[Tensor] = []
            expired_targets: list[Tensor] = []
            expired_masks: list[Tensor] = []
            for record in selected:
                base_record = record["base_record"]
                count_target, count_mask = _explicit_masked_target(
                    base_record,
                    value_keys=("workload_count_target", "arrival_count_target"),
                    expected=(FAST_STEPS, 1),
                    integral=True,
                )
                expired_target, expired_mask = _explicit_masked_target(
                    base_record,
                    value_keys=(
                        "deadline_expired_work_target",
                        "sla_expired_work_target",
                    ),
                    expected=(FAST_STEPS, NODES),
                )
                count_targets.append(count_target)
                count_masks.append(count_mask)
                expired_targets.append(expired_target)
                expired_masks.append(expired_mask)
            reconstruction_targets = {
                "public_21": current_targets.to(target_device),
                "soc": soc.to(target_device),
                "queue": queue.to(target_device),
            }
            da_anchors = torch.stack(
                [_tensor(record["forecast_anchors"]["da"]) for record in selected]
            ).to(target_device)
            if self.statistics.da_residual_cap is None:
                raise WorldDataError(
                    "formal corpus omitted train-frozen DA residual cap"
                )
            state_training_baselines = {
                "soc": self.target_statistics.state_training_baseline_soc,
                "queue_occurrence": (
                    self.target_statistics.state_training_baseline_queue_occurrence
                ),
                "queue_severity": (
                    self.target_statistics.state_training_baseline_queue_severity
                ),
            }
            if any(value is None for value in state_training_baselines.values()):
                raise WorldDataError("formal corpus omitted state training baselines")
            output: dict[str, Any] = {
                "world_input": packet,
                "input_packet": packet,
                "operating_dates": tuple(
                    str(record["operating_date"]) for record in selected
                ),
                "origin": int(origin),
                "forecast_anchors": {"da": da_anchors},
                "forecast_anchor_metadata": {
                    "schema": "dcvpp.vpp_world_moe.causal_price_anchor.r5.v2.1",
                    "da_source": "latest_fully_completed_operating_day_before_issue",
                    "per_record": tuple(
                        dict(record["forecast_anchor_metadata"]) for record in selected
                    ),
                    "normalization": "none_raw_target_units",
                    "opened_final_test_2025": False,
                },
                "forecast_residual_caps": {
                    "da": self.statistics.da_residual_cap.to(target_device)
                },
                "forecast_residual_cap_metadata": {
                    "schema": "dcvpp.vpp_world_moe.forecast_residual_caps.r5.v2.1",
                    "fit_split": "train_2022_2023_only",
                    "quantile": 0.95,
                    "units": "raw_price_units_by_zone",
                    "opened_final_test_2025": False,
                },
                "causal_da_anchor_available": torch.tensor(
                    [
                        record["forecast_anchor_metadata"].get("da_source")
                        != "first_train_day_issue_time_da_warmup"
                        for record in selected
                    ],
                    dtype=torch.bool,
                    device=target_device,
                ),
                "quantile_levels": torch.tensor(
                    QUANTILE_LEVELS, dtype=torch.float32, device=target_device
                ),
                "quantile_targets": quantile_targets,
                "quantile_masks": quantile_masks,
                "quantile_scales": quantile_scales,
                "quantile_centers": quantile_centers,
                "state_centers": {
                    "soc": self.target_statistics.state_center[:4].to(target_device),
                    "queue": self.target_statistics.state_center[4:8].to(target_device),
                },
                "state_scales": {
                    "soc": self.target_statistics.state_scale[:4].to(target_device),
                    "queue": self.target_statistics.state_scale[4:8].to(target_device),
                    "queue_severity": self.target_statistics.state_scale[4:8].to(
                        target_device
                    ),
                },
                "state_training_baselines": {
                    name: value.to(target_device)  # type: ignore[union-attr]
                    for name, value in state_training_baselines.items()
                },
                "state_training_baseline_metadata": {
                    "fit_split": "train_2022_2023_only",
                    "soc": (
                        self.target_statistics.state_training_baseline_soc_reference
                    ),
                    "used_validation_statistics": False,
                    "opened_final_test_2025": False,
                },
                "state_gate_baselines": {
                    "soc_persistence": (
                        self.target_statistics.state_gate_baseline_soc_persistence
                        if self.target_statistics.state_gate_baseline_soc_persistence
                        is not None
                        else torch.zeros((), dtype=torch.float32)
                    ).to(target_device)
                },
                "tail_thresholds": tail_thresholds,
                "target_metadata": {
                    "fit_split": "train_2022_2023_only",
                    "channel_names": PUBLIC_CHANNEL_NAMES,
                    "family_slices": {
                        "price": (0, 12),
                        "workload": (12, 13),
                        "pv": (13, 17),
                        "pue": (17, 21),
                    },
                    "family_total_weights": {
                        "price": 0.25,
                        "workload": 0.25,
                        "pv": 0.25,
                        "pue": 0.25,
                    },
                    "price_subgroup_weights": {
                        "da": 0.25,
                        "rt": 0.25,
                        "dart": 0.50,
                    },
                    "transforms": {
                        "price": "signed_median_mad",
                        "workload": "log1p_then_median_mad",
                        "pv": "capacity_fraction",
                        "pue": "physical_bounds_fraction",
                    },
                    "opened_final_test_2025": False,
                    "train_frozen_baseline_daily_cost": float(
                        self.target_statistics.baseline_daily_cost
                    ),
                    "baseline_daily_cost_fit": "single_mean_over_all_722_train_days",
                    "queue_occurrence_threshold_work_units": (
                        QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS
                    ),
                    "aggregate_queue_event_rule": "any_node_q_gt_threshold",
                    "queue_route_teacher_visibility": "loss_only_never_model_input",
                    "aggregate_risk_slot_37": {
                        "0": "mean_registered_rt_standard_deviation",
                        "1": "mean_absolute_issue_time_dart",
                        "2": "aggregate_arrival_mean_over_compute_capacity",
                        "3": "aggregate_arrival_q90_over_compute_capacity",
                        "4": "positive_q90_capacity_exceedance_ratio",
                        "5": "deadline_window_q90_utilization",
                        "6": "held_initial_queue_over_compute_capacity",
                        "7": "time_fraction",
                    },
                },
                "train_frozen_baseline_daily_cost": (
                    self.target_statistics.baseline_daily_cost.to(target_device)
                ),
                "workload_volume_target": current_targets[..., 12:13].to(target_device),
                "workload_volume_mask": torch.isfinite(current_targets[..., 12:13]).to(
                    target_device
                ),
                "workload_count_target": torch.stack(count_targets).to(target_device),
                "workload_count_mask": torch.stack(count_masks).to(target_device),
                "deadline_intervals": torch.full(
                    (len(selected),),
                    self.deadline_intervals,
                    dtype=torch.long,
                    device=target_device,
                ),
                "deadline_pressure_target": states[..., 8:12].to(target_device),
                "deadline_pressure_mask": torch.isfinite(states[..., 8:12]).to(
                    target_device
                ),
                "deadline_expired_work_target": torch.stack(expired_targets).to(
                    target_device
                ),
                "deadline_expired_work_mask": torch.stack(expired_masks).to(
                    target_device
                ),
                "reconstruction_targets": reconstruction_targets,
                "reconstruction_masks": {
                    key: torch.isfinite(value)
                    for key, value in reconstruction_targets.items()
                },
                "queue_occurrence_target": queue_occurrence.to(target_device),
                "queue_occurrence_mask": queue_valid.to(target_device),
                "queue_severity_target": queue.to(target_device),
                "queue_severity_mask": (queue_valid & queue_occurrence).to(
                    target_device
                ),
                "queue_risk_event_target": queue_risk_event_target.to(target_device),
                "queue_risk_event_mask": queue_risk_event_mask.to(target_device),
                "queue_risk_event_train_prevalence": (
                    self.target_statistics.queue_event_prevalence.to(target_device)
                ),
                "stochastic_target": current_targets.to(target_device),
                "scenario_target": current_targets.to(target_device),
                "future_target_sequence": future_sequence.to(target_device),
                "future_target_mask": future_mask.to(target_device),
                "target_horizons": horizons,
                "h96_target_available": future_mask[:, 96 + int(origin)]
                .all(-1)
                .to(target_device)
                if origin < 96
                else torch.zeros(len(selected), dtype=torch.bool, device=target_device),
            }
            if include_next_day and all_next:
                typed_next = [record for record in next_records if record is not None]
                next_da_anchors = torch.stack(
                    [_tensor(record["forecast_anchors"]["da"]) for record in typed_next]
                ).to(target_device)
                output["future_forecast_anchors"] = {"da": next_da_anchors}
                next_targets = _close_public_price_target(
                    torch.stack(
                        [
                            _tensor(record["base_record"]["scenario_target_zonal"])
                            for record in typed_next
                        ]
                    )
                )
                next_states = torch.stack(
                    [
                        _tensor(record["base_record"]["state_sequence"])
                        for record in typed_next
                    ]
                )
                supervision = _future_supervision(
                    current_targets,
                    next_targets,
                    states[..., :8],
                    next_states[..., :8],
                    origin=origin,
                )
                teacher_action_summary, teacher_action_mask = (
                    _w1_teacher_action_summary(
                        states,
                        next_states,
                        origin=origin,
                    )
                )
                output.update(
                    {
                        key: (
                            {
                                nested_key: (
                                    {
                                        state_key: state_value.to(target_device)
                                        for state_key, state_value in nested_value.items()
                                    }
                                    if isinstance(nested_value, Mapping)
                                    else nested_value.to(target_device)
                                )
                                for nested_key, nested_value in value.items()
                            }
                            if isinstance(value, Mapping)
                            else value.to(target_device)
                        )
                        for key, value in supervision.items()
                    }
                )
                output["observation_masks"] = output["observation_target_masks"]
                output["teacher_action_summary"] = teacher_action_summary.to(
                    target_device
                )
                output["teacher_action_mask"] = teacher_action_mask.to(target_device)
                output["teacher_action_metadata"] = {
                    "source": "causal_passive_edf_baseline_execution_trace",
                    "empirical_dispatch_claimed": False,
                    "coordinates": (
                        "battery_mean",
                        "compute_mean",
                        "migration_total_per_node",
                        "grid_mean",
                        "pv_curtailment_passive_zero",
                        "unserved_power_passive_zero",
                    ),
                    "alignment": "state_t_plus_1_records_interval_t",
                    "continuous_actions": True,
                    "delta_actions_consumed": True,
                    "visible_to_causal_input": False,
                }
                target_packet = self._packet(
                    typed_next, origin=origin, device=target_device
                )
                output["target_packet"] = target_packet
                output["next_day_packet"] = target_packet
            yield output

    def batch_count(
        self,
        split: str,
        batch_size: int,
        *,
        require_h96: bool = False,
        max_days: int | None = None,
    ) -> int:
        """Return a batch count without constructing any CPU/GPU packets."""

        size = int(batch_size)
        if size <= 0:
            raise WorldDataError("batch_size must be positive")
        count = len(self.records(split, require_h96=require_h96))
        if max_days is not None:
            count = min(count, max(0, int(max_days)))
        return (count + size - 1) // size


__all__ = [
    "ACTION_DIM",
    "CONSTRAINT_DIM",
    "ENTITY_IDS",
    "FAST_STEPS",
    "GRAPH_EDGE_DIM",
    "GraphContract",
    "INPUT_DIM",
    "MEDIUM_STEPS",
    "NODE_IDS",
    "PUBLIC_CHANNEL_NAMES",
    "QUEUE_OCCURRENCE_THRESHOLD_WORK_UNITS",
    "QUANTILE_LEVELS",
    "SLOW_STEPS",
    "TARGET_FAMILY_SLICES",
    "TOTAL_ENTITIES",
    "WorldDataError",
    "WorldMoECorpus",
    "WorldNormalizationStatistics",
    "WorldTargetStatistics",
    "to_world_input_packet",
]
