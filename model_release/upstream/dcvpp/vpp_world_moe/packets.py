"""Typed tensor packets used by the VPP-WorldMoE core."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from torch import Tensor

from .config import WorldMoEConfig


NODES = 4
ENTITIES_PER_NODE = 8
GLOBAL_SLOTS = 8
TOTAL_SLOTS = 40
PUBLIC_CHANNELS = 21
WEATHER_SLOT = 34


def _shape(name: str, value: Tensor, expected: tuple[int | None, ...]) -> None:
    if value.ndim != len(expected) or any(
        item is not None and actual != item
        for actual, item in zip(value.shape, expected, strict=True)
    ):
        raise ValueError(f"{name} must have shape {expected}; got {tuple(value.shape)}")


def _strict_finite(name: str, value: Tensor) -> None:
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _bool(name: str, value: Tensor, expected: tuple[int | None, ...]) -> None:
    _shape(name, value, expected)
    if value.dtype is not torch.bool:
        raise ValueError(f"{name} must be bool")


def _long(name: str, value: Tensor, expected: tuple[int | None, ...]) -> None:
    _shape(name, value, expected)
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be integral")


def _detach_dataclass(instance: Any) -> Any:
    values: dict[str, Any] = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        if isinstance(value, Tensor):
            values[field.name] = value.detach()
        elif isinstance(value, dict):
            values[field.name] = {
                key: item.detach() if isinstance(item, Tensor) else item
                for key, item in value.items()
            }
        elif hasattr(value, "detached"):
            values[field.name] = value.detached()
        else:
            values[field.name] = value
    return replace(instance, **values)


def close_price_identity(da: Tensor, rt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return DA, RT and the only permitted DART definition."""

    if da.shape != rt.shape or da.shape[-1] != NODES:
        raise ValueError("DA and RT must share a final four-node dimension")
    return da, rt, da - rt


@dataclass(frozen=True)
class WorldConstraintPacket:
    """Physical limits consumed by neural decoders and the analytic shield."""

    battery_power_limit: Tensor
    battery_energy_capacity: Tensor
    initial_soc: Tensor
    compute_capacity: Tensor
    initial_queue: Tensor
    pv_capacity: Tensor
    grid_import_limit: Tensor
    grid_export_limit: Tensor
    migration_capacity: Tensor
    migration_allowed: Tensor
    pue_min: Tensor
    pue_max: Tensor
    idle_power: Tensor
    work_power_per_unit: Tensor
    charge_efficiency: Tensor
    discharge_efficiency: Tensor
    dt_hours: Tensor
    # Optional causal physical support for stochastic PV paths.  ``True``
    # means positive/capacity PV is physically admissible at that interval;
    # ``False`` is a known night interval and therefore forces an exact atom
    # at zero.  The model derives this field only from issue-time weather
    # values and their availability mask.  Legacy W0/W1 packets leave it None.
    pv_support_mask: Tensor | None = None

    def validate(
        self, batch: int | None = None, *, strict_values: bool = False
    ) -> "WorldConstraintPacket":
        size = self.battery_power_limit.shape[0] if batch is None else int(batch)
        for name in (
            "battery_power_limit",
            "battery_energy_capacity",
            "initial_soc",
            "compute_capacity",
            "initial_queue",
            "pv_capacity",
            "grid_import_limit",
            "grid_export_limit",
            "pue_min",
            "pue_max",
            "idle_power",
            "work_power_per_unit",
            "charge_efficiency",
            "discharge_efficiency",
        ):
            value = getattr(self, name)
            _shape(f"constraints.{name}", value, (size, NODES))
        _shape(
            "constraints.migration_capacity",
            self.migration_capacity,
            (size, NODES, NODES),
        )
        _bool(
            "constraints.migration_allowed",
            self.migration_allowed,
            (size, NODES, NODES),
        )
        _shape("constraints.dt_hours", self.dt_hours, (size,))
        if self.pv_support_mask is not None:
            _bool(
                "constraints.pv_support_mask",
                self.pv_support_mask,
                (size, None, NODES),
            )
        if not strict_values:
            return self
        nonnegative = (
            self.battery_power_limit,
            self.battery_energy_capacity,
            self.initial_soc,
            self.compute_capacity,
            self.initial_queue,
            self.pv_capacity,
            self.grid_import_limit,
            self.grid_export_limit,
            self.migration_capacity,
            self.idle_power,
            self.work_power_per_unit,
            self.dt_hours,
        )
        if any(bool((value < 0).any()) for value in nonnegative):
            raise ValueError(
                "physical capacities, states and dt_hours must be nonnegative"
            )
        if bool((self.initial_soc > self.battery_energy_capacity).any()):
            raise ValueError("initial SOC exceeds battery energy capacity")
        if bool((self.pue_min < 1.0).any()) or bool(
            (self.pue_max < self.pue_min).any()
        ):
            raise ValueError("PUE limits must satisfy 1 <= min <= max")
        for efficiency in (self.charge_efficiency, self.discharge_efficiency):
            if bool(((efficiency <= 0.0) | (efficiency > 1.0)).any()):
                raise ValueError("battery efficiencies must lie in (0,1]")
        diagonal = torch.diagonal(self.migration_allowed, dim1=-2, dim2=-1)
        if bool(diagonal.any()):
            raise ValueError("self-migration must be disabled")
        return self

    def detached(self) -> "WorldConstraintPacket":
        return _detach_dataclass(self)


@dataclass(frozen=True)
class WorldInputPacket:
    """Causal, typed, multi-scale event stream for one or more issue origins."""

    fast_values: Tensor
    fast_mask: Tensor
    entity_ids: Tensor
    node_ids: Tensor
    source_ids: Tensor
    quality_ids: Tensor
    graph_edge_index: Tensor
    graph_edge_type: Tensor
    graph_edge_features: Tensor
    graph_edge_mask: Tensor
    previous_actions: Tensor
    action_mask: Tensor
    constraint_slacks: Tensor
    constraints: WorldConstraintPacket
    origin_index: Tensor
    dt_hours: Tensor
    medium_values: Tensor | None = None
    medium_mask: Tensor | None = None
    medium_source_ids: Tensor | None = None
    medium_quality_ids: Tensor | None = None
    slow_values: Tensor | None = None
    slow_mask: Tensor | None = None
    slow_source_ids: Tensor | None = None
    slow_quality_ids: Tensor | None = None
    dynamic_graph_edge_features: Tensor | None = None
    dynamic_graph_edge_mask: Tensor | None = None
    known_future_mask: Tensor | None = None

    def validate(
        self, config: WorldMoEConfig, *, strict_values: bool = False
    ) -> "WorldInputPacket":
        batch, time, slots, width = self.fast_values.shape
        _shape(
            "fast_values",
            self.fast_values,
            (batch, time, TOTAL_SLOTS, config.input_dim),
        )
        _bool("fast_mask", self.fast_mask, (batch, time, TOTAL_SLOTS))
        if time > config.horizon:
            raise ValueError("fast event stream exceeds the configured 96-step horizon")
        _long("entity_ids", self.entity_ids, (TOTAL_SLOTS,))
        _long("node_ids", self.node_ids, (TOTAL_SLOTS,))
        if strict_values:
            if bool(((self.entity_ids < 0) | (self.entity_ids >= 16)).any()):
                raise ValueError("entity_ids must lie in [0,15]")
            if bool(((self.node_ids < 0) | (self.node_ids > NODES)).any()):
                raise ValueError("node_ids must use nodes 0..3 and global node id 4")
            expected_nodes = torch.cat(
                (
                    torch.arange(NODES, device=self.node_ids.device).repeat_interleave(
                        ENTITIES_PER_NODE
                    ),
                    torch.full((GLOBAL_SLOTS,), NODES, device=self.node_ids.device),
                )
            ).to(self.node_ids.dtype)
            if not torch.equal(self.node_ids, expected_nodes):
                raise ValueError(
                    "node_ids violate the fixed 4x8 plus 8-global slot layout"
                )
        _long("source_ids", self.source_ids, (batch, time, TOTAL_SLOTS))
        _long("quality_ids", self.quality_ids, (batch, time, TOTAL_SLOTS))
        if self.known_future_mask is not None:
            _bool(
                "known_future_mask",
                self.known_future_mask,
                (batch, time, TOTAL_SLOTS),
            )
        _shape(
            "previous_actions",
            self.previous_actions,
            (batch, time, NODES, config.action_dim),
        )
        _bool("action_mask", self.action_mask, (batch, time, NODES, config.action_dim))
        _shape(
            "constraint_slacks",
            self.constraint_slacks,
            (batch, time, NODES, config.constraint_dim),
        )
        _shape("dt_hours", self.dt_hours, (batch, time))
        if strict_values and bool((self.dt_hours <= 0).any()):
            raise ValueError("all event intervals must be positive")
        _long("origin_index", self.origin_index, (batch,))
        if strict_values and bool(
            ((self.origin_index < 0) | (self.origin_index >= time)).any()
        ):
            raise ValueError("origin_index lies outside the event stream")
        edge_count = self.graph_edge_index.shape[-1]
        _long("graph_edge_index", self.graph_edge_index, (2, edge_count))
        _long("graph_edge_type", self.graph_edge_type, (edge_count,))
        _shape(
            "graph_edge_features",
            self.graph_edge_features,
            (edge_count, config.graph_edge_dim),
        )
        _bool("graph_edge_mask", self.graph_edge_mask, (edge_count,))
        if strict_values and bool(
            ((self.graph_edge_index < 0) | (self.graph_edge_index >= TOTAL_SLOTS)).any()
        ):
            raise ValueError("graph edge index lies outside the 40 slots")
        if self.dynamic_graph_edge_features is not None:
            _shape(
                "dynamic_graph_edge_features",
                self.dynamic_graph_edge_features,
                (batch, time, edge_count, config.graph_edge_dim),
            )
        if self.dynamic_graph_edge_mask is not None:
            _bool(
                "dynamic_graph_edge_mask",
                self.dynamic_graph_edge_mask,
                (batch, time, edge_count),
            )
        if (self.dynamic_graph_edge_features is None) != (
            self.dynamic_graph_edge_mask is None
        ):
            raise ValueError(
                "dynamic graph features and mask must be both present or both absent"
            )
        for scale, values, mask, source, quality in (
            (
                "medium",
                self.medium_values,
                self.medium_mask,
                self.medium_source_ids,
                self.medium_quality_ids,
            ),
            (
                "slow",
                self.slow_values,
                self.slow_mask,
                self.slow_source_ids,
                self.slow_quality_ids,
            ),
        ):
            if values is None and mask is None:
                continue
            if values is None or mask is None:
                raise ValueError(
                    f"{scale} values and mask must be both present or absent"
                )
            _shape(f"{scale}_values", values, (batch, None, TOTAL_SLOTS, width))
            _bool(f"{scale}_mask", mask, (batch, values.shape[1], TOTAL_SLOTS))
            if (source is None) != (quality is None):
                raise ValueError(
                    f"{scale} source and quality ids must be both present or absent"
                )
            if source is not None and quality is not None:
                _long(
                    f"{scale}_source_ids",
                    source,
                    (batch, values.shape[1], TOTAL_SLOTS),
                )
                _long(
                    f"{scale}_quality_ids",
                    quality,
                    (batch, values.shape[1], TOTAL_SLOTS),
                )
        self.constraints.validate(batch, strict_values=strict_values)
        return self

    def detached(self) -> "WorldInputPacket":
        return _detach_dataclass(self)


@dataclass(frozen=True)
class RouterDiagnostics:
    block_index: int
    logits: Tensor
    probabilities: Tensor
    topk_indices: Tensor
    topk_weights: Tensor
    expert_load: Tensor
    balance_loss: Tensor
    z_loss: Tensor
    tail_route_logits: Tensor
    tail_route_probabilities: Tensor
    tail_active_mask: Tensor
    tail_selected_count: Tensor
    tail_capacity: Tensor
    tail_load: Tensor
    tail_utilization: Tensor
    # Optional compact expert capture.  The default path leaves these unset so
    # legacy callers retain the same memory and construction contract.
    capture_slot: int | None = None
    captured_pre_state: Tensor | None = None
    captured_selected_expert_states: Tensor | None = None
    captured_tail_state: Tensor | None = None

    def validate(self, batch: int, time: int) -> "RouterDiagnostics":
        _shape("router.logits", self.logits, (batch, time, TOTAL_SLOTS, 7))
        _shape(
            "router.probabilities", self.probabilities, (batch, time, TOTAL_SLOTS, 7)
        )
        _long("router.topk_indices", self.topk_indices, (batch, time, TOTAL_SLOTS, 2))
        _shape("router.topk_weights", self.topk_weights, (batch, time, TOTAL_SLOTS, 2))
        _shape("router.expert_load", self.expert_load, (7,))
        _shape("router.balance_loss", self.balance_loss, ())
        _shape("router.z_loss", self.z_loss, ())
        _shape("router.tail_route_logits", self.tail_route_logits, (batch, time))
        _shape(
            "router.tail_route_probabilities",
            self.tail_route_probabilities,
            (batch, time),
        )
        _bool("router.tail_active_mask", self.tail_active_mask, (batch, time))
        _long("router.tail_selected_count", self.tail_selected_count, (batch,))
        _long("router.tail_capacity", self.tail_capacity, (batch,))
        _shape("router.tail_load", self.tail_load, ())
        _shape("router.tail_utilization", self.tail_utilization, ())
        captured = (
            self.captured_pre_state,
            self.captured_selected_expert_states,
            self.captured_tail_state,
        )
        if self.capture_slot is None:
            if any(value is not None for value in captured):
                raise ValueError("router capture tensors require capture_slot")
        else:
            if (
                isinstance(self.capture_slot, bool)
                or not 0 <= self.capture_slot < TOTAL_SLOTS
            ):
                raise ValueError(
                    "router capture_slot lies outside the typed slot range"
                )
            if any(value is None for value in captured):
                raise ValueError("router compact capture is incomplete")
            assert self.captured_pre_state is not None
            assert self.captured_selected_expert_states is not None
            assert self.captured_tail_state is not None
            _shape(
                "router.captured_pre_state",
                self.captured_pre_state,
                (batch, time, None),
            )
            width = self.captured_pre_state.shape[-1]
            _shape(
                "router.captured_selected_expert_states",
                self.captured_selected_expert_states,
                (batch, time, self.topk_indices.shape[-1], width),
            )
            _shape(
                "router.captured_tail_state",
                self.captured_tail_state,
                (batch, time, width),
            )
        return self


@dataclass(frozen=True)
class WorldStatePacket:
    slots: Tensor
    current_slots: Tensor
    pooled_timeline: Tensor
    pooled_state: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    stochastic_mean: Tensor
    stochastic_log_scale: Tensor
    physics_balance_residual: Tensor
    router_diagnostics: tuple[RouterDiagnostics, ...]
    origin_index: Tensor

    def validate(self, config: WorldMoEConfig) -> "WorldStatePacket":
        batch, time = self.slots.shape[:2]
        _shape("state.slots", self.slots, (batch, time, TOTAL_SLOTS, config.d_model))
        _shape(
            "state.current_slots",
            self.current_slots,
            (batch, TOTAL_SLOTS, config.d_model),
        )
        _shape(
            "state.pooled_timeline", self.pooled_timeline, (batch, time, config.d_model)
        )
        _shape("state.pooled_state", self.pooled_state, (batch, config.d_model))
        _shape(
            "state.regime_logits",
            self.regime_logits,
            (batch, time, TOTAL_SLOTS, config.num_regimes),
        )
        _shape(
            "state.regime_probabilities",
            self.regime_probabilities,
            (batch, time, TOTAL_SLOTS, config.num_regimes),
        )
        _shape(
            "state.stochastic_mean",
            self.stochastic_mean,
            (batch, time, TOTAL_SLOTS, config.stochastic_dim),
        )
        _shape(
            "state.stochastic_log_scale",
            self.stochastic_log_scale,
            (batch, time, TOTAL_SLOTS, config.stochastic_dim),
        )
        _shape(
            "state.physics_balance_residual",
            self.physics_balance_residual,
            (batch, time, NODES),
        )
        _long("state.origin_index", self.origin_index, (batch,))
        if len(self.router_diagnostics) != 5:
            raise ValueError(
                "WorldMoE state must contain diagnostics from five MoE blocks"
            )
        for item in self.router_diagnostics:
            item.validate(batch, time)
        return self

    @property
    def router_aux_loss(self) -> Tensor:
        return sum(
            (
                item.balance_loss + 1.0e-3 * item.z_loss
                for item in self.router_diagnostics
            ),
            self.slots.new_zeros(()),
        )

    @property
    def queue_tail_route_logits(self) -> Tensor:
        """Aggregate five causal tail gates without consulting event truth."""

        return torch.stack(
            tuple(item.tail_route_logits for item in self.router_diagnostics), dim=0
        ).mean(dim=0)[..., None]

    @property
    def queue_tail_route_probabilities(self) -> Tensor:
        # Each MoE block already converts its weighted-BCE ranking logit to a
        # calibrated event probability.  Applying sigmoid to the mean raw logit
        # would silently undo that correction.
        return torch.stack(
            tuple(item.tail_route_probabilities for item in self.router_diagnostics),
            dim=0,
        ).mean(dim=0)[..., None]

    def detached(self) -> "WorldStatePacket":
        return _detach_dataclass(self)


@dataclass(frozen=True)
class WorldObservationPacket:
    da: Tensor
    rt: Tensor
    dart: Tensor
    arrivals: Tensor
    service: Tensor
    queue: Tensor
    pv: Tensor
    it_power: Tensor
    cooling_power: Tensor
    facility_power: Tensor
    pue: Tensor
    soc: Tensor
    public_21: Tensor

    def validate(
        self,
        batch: int | None = None,
        paths: int | None = None,
        *,
        strict_values: bool = False,
    ) -> "WorldObservationPacket":
        # Supports both [B,T,...] reconstructions and [B,P,T,...] imagination.
        prefix = self.da.shape[:-2]
        time = self.da.shape[-2]
        if batch is not None and prefix[0] != batch:
            raise ValueError("observation batch mismatch")
        if paths is not None and (len(prefix) != 2 or prefix[1] != paths):
            raise ValueError("observation path dimension mismatch")
        for name in (
            "da",
            "rt",
            "dart",
            "arrivals",
            "service",
            "pv",
            "it_power",
            "cooling_power",
            "facility_power",
            "pue",
            "soc",
        ):
            _shape(f"observation.{name}", getattr(self, name), (*prefix, time, NODES))
        _shape("observation.queue", self.queue, (*prefix, time + 1, NODES))
        _shape(
            "observation.public_21", self.public_21, (*prefix, time, PUBLIC_CHANNELS)
        )
        if strict_values and not torch.equal(self.dart, self.da - self.rt):
            raise ValueError("DART is not deterministically closed as DA-RT")
        if strict_values and (
            bool((self.arrivals < 0).any())
            or bool((self.service < 0).any())
            or bool((self.queue < 0).any())
        ):
            raise ValueError("workload arrival/service/queue must be nonnegative")
        if strict_values and bool((self.pv < 0).any()):
            raise ValueError("PV must be nonnegative")
        return self

    def detached(self) -> "WorldObservationPacket":
        return _detach_dataclass(self)


@dataclass(frozen=True)
class WorldActionPacket:
    battery: Tensor
    compute: Tensor
    migration: Tensor
    grid: Tensor
    pv_used: Tensor
    pv_curtailed: Tensor
    unserved_power: Tensor
    soc: Tensor
    queue: Tensor
    power_balance_residual: Tensor
    migration_balance_residual: Tensor
    constraint_violation: Tensor
    shield_intervention: Tensor

    @property
    def horizon(self) -> int:
        return int(self.battery.shape[-2])

    def validate(self, *, strict_values: bool = False) -> "WorldActionPacket":
        prefix = self.battery.shape[:-2]
        horizon = self.battery.shape[-2]
        for name in (
            "battery",
            "compute",
            "grid",
            "pv_used",
            "pv_curtailed",
            "unserved_power",
            "power_balance_residual",
            "constraint_violation",
            "shield_intervention",
        ):
            _shape(f"actions.{name}", getattr(self, name), (*prefix, horizon, NODES))
        _shape("actions.migration", self.migration, (*prefix, horizon, NODES, NODES))
        _shape("actions.soc", self.soc, (*prefix, horizon + 1, NODES))
        _shape("actions.queue", self.queue, (*prefix, horizon + 1, NODES))
        _shape(
            "actions.migration_balance_residual",
            self.migration_balance_residual,
            (*prefix, horizon),
        )
        if strict_values and (
            bool((self.compute < 0).any()) or bool((self.migration < 0).any())
        ):
            raise ValueError("compute and migration actions must be nonnegative")
        return self

    def detached(self) -> "WorldActionPacket":
        return _detach_dataclass(self)


@dataclass(frozen=True)
class WorldRolloutPacket:
    latent_paths: Tensor
    observations: WorldObservationPacket
    actions: WorldActionPacket
    path_cost: Tensor
    weights: Tensor
    regime_indices: Tensor
    expected_cost: Tensor
    cvar_cost: Tensor
    support_probabilities: dict[str, Tensor] | None = None

    def validate(
        self, config: WorldMoEConfig, *, strict_values: bool = False
    ) -> "WorldRolloutPacket":
        batch, paths, horizon = self.latent_paths.shape[:3]
        _shape(
            "rollout.latent_paths",
            self.latent_paths,
            (batch, paths, horizon, config.d_model),
        )
        self.observations.validate(
            batch=batch, paths=paths, strict_values=strict_values
        )
        self.actions.validate(strict_values=strict_values)
        _shape("rollout.path_cost", self.path_cost, (batch, paths))
        _shape("rollout.weights", self.weights, (batch, paths))
        _long("rollout.regime_indices", self.regime_indices, (batch, paths, horizon))
        _shape("rollout.expected_cost", self.expected_cost, (batch,))
        _shape("rollout.cvar_cost", self.cvar_cost, (batch,))
        if self.support_probabilities is not None:
            expected_support = {
                "workload_zero": (batch, paths, horizon, 1),
                "pv_zero": (batch, paths, horizon, NODES),
                "pv_capacity": (batch, paths, horizon, NODES),
                "pue_min": (batch, paths, horizon, NODES),
                "pue_max": (batch, paths, horizon, NODES),
            }
            if set(self.support_probabilities) != set(expected_support):
                raise ValueError(
                    "rollout support probabilities must expose the registered "
                    "workload/PV/PUE atom keys"
                )
            for name, expected in expected_support.items():
                value = self.support_probabilities[name]
                _shape(f"rollout.support_probabilities.{name}", value, expected)
                if strict_values and (
                    not bool(torch.isfinite(value).all())
                    or bool(((value < 0.0) | (value > 1.0)).any())
                ):
                    raise ValueError(
                        f"rollout support probability {name} must lie in [0,1]"
                    )
        if strict_values and bool((self.weights < 0).any()):
            raise ValueError("rollout weights must be nonnegative")
        return self

    def detached(self) -> "WorldRolloutPacket":
        return _detach_dataclass(self)


@dataclass(frozen=True)
class WorldPlanPacket:
    actions: WorldActionPacket
    raw_actions: Tensor
    candidate_scores: Tensor
    selection_weights: Tensor
    candidate_latents: Tensor
    refinement_trace: tuple[Tensor, ...]
    rollout: WorldRolloutPacket

    @property
    def battery(self) -> Tensor:
        return self.actions.battery

    @property
    def compute(self) -> Tensor:
        return self.actions.compute

    @property
    def migration(self) -> Tensor:
        return self.actions.migration

    @property
    def grid(self) -> Tensor:
        return self.actions.grid

    @property
    def soc(self) -> Tensor:
        return self.actions.soc

    @property
    def queue(self) -> Tensor:
        return self.actions.queue

    @property
    def constraint_violation(self) -> Tensor:
        return self.actions.constraint_violation

    @property
    def power_balance_residual(self) -> Tensor:
        return self.actions.power_balance_residual

    def validate(
        self, config: WorldMoEConfig, *, strict_values: bool = False
    ) -> "WorldPlanPacket":
        self.actions.validate(strict_values=strict_values)
        batch = self.actions.battery.shape[0]
        _shape(
            "plan.raw_actions",
            self.raw_actions,
            (batch, self.actions.horizon, 24),
        )
        _shape(
            "plan.candidate_scores",
            self.candidate_scores,
            (batch, config.planner_refinements, config.planner_candidates),
        )
        _shape(
            "plan.selection_weights",
            self.selection_weights,
            (batch, config.planner_candidates),
        )
        _shape(
            "plan.candidate_latents",
            self.candidate_latents,
            (batch, config.planner_candidates, config.d_model),
        )
        if len(self.refinement_trace) != config.planner_refinements:
            raise ValueError("planner must execute exactly three refinement rounds")
        self.rollout.validate(config, strict_values=strict_values)
        return self

    def detached(self) -> "WorldPlanPacket":
        return _detach_dataclass(self)


__all__ = [
    "ENTITIES_PER_NODE",
    "GLOBAL_SLOTS",
    "NODES",
    "PUBLIC_CHANNELS",
    "RouterDiagnostics",
    "TOTAL_SLOTS",
    "WorldActionPacket",
    "WorldConstraintPacket",
    "WorldInputPacket",
    "WorldObservationPacket",
    "WorldPlanPacket",
    "WorldRolloutPacket",
    "WorldStatePacket",
    "close_price_identity",
]
