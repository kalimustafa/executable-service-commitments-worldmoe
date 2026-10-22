"""Constraint-native neural action decoder and deterministic analytic shield."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from .config import WorldMoEConfig
from .packets import (
    NODES,
    WorldActionPacket,
    WorldConstraintPacket,
    WorldObservationPacket,
)


RAW_ACTION_DIM = 2 + NODES * NODES


def repeat_constraints(
    constraints: WorldConstraintPacket, repeats: int
) -> WorldConstraintPacket:
    values = {}
    for name in constraints.__dataclass_fields__:
        value = getattr(constraints, name)
        values[name] = (
            None
            if value is None
            else value.repeat_interleave(int(repeats), dim=0)
        )
    return replace(constraints, **values)


def repeat_observations(
    observations: WorldObservationPacket, repeats: int
) -> WorldObservationPacket:
    values = {
        name: getattr(observations, name).repeat_interleave(int(repeats), dim=0)
        for name in observations.__dataclass_fields__
    }
    return replace(observations, **values)


def _lindley_compute_and_queue(
    arrivals: Tensor,
    compute_capacity: Tensor,
    initial_queue: Tensor,
) -> tuple[Tensor, Tensor]:
    """Parallel form of ``q_t=max(0,q_(t-1)+arrivals_t-capacity_t)``."""

    unconstrained_queue = initial_queue[:, None] + torch.cumsum(
        arrivals - compute_capacity,
        dim=1,
    )
    running_floor = torch.minimum(
        torch.cummin(unconstrained_queue, dim=1).values,
        torch.zeros_like(unconstrained_queue),
    )
    queue_after_compute = unconstrained_queue - running_floor
    queue_trace = torch.cat((initial_queue[:, None], queue_after_compute), dim=1)
    available_queue = queue_trace[:, :-1] + arrivals
    compute = torch.minimum(compute_capacity, available_queue)
    return compute, queue_trace


class ConstraintNativeActionDecoder(nn.Module):
    """Decode battery, compute and migration without an optimization runtime."""

    raw_action_dim = RAW_ACTION_DIM

    def __init__(self, config: WorldMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.battery_temperature = nn.Parameter(torch.tensor(1.0))
        self.compute_temperature = nn.Parameter(torch.tensor(1.0))
        # Runtime evidence only; plain Python state deliberately stays outside
        # state_dict so every existing W0/W1/U0 checkpoint remains identical.
        self.last_zero_action_decode_path = "not_run"

    def decode_zero_actions(
        self,
        reference: WorldObservationPacket,
        constraints: WorldConstraintPacket,
    ) -> WorldActionPacket:
        """Decode U0's fixed zero action without a 96-step Python loop.

        The legacy general decoder adds a grid-driven compute reduction back
        to its queue twice.  Consequently, a binding grid limit is not a
        Lindley recurrence.  The frozen physical contract admits a static,
        conservative proof that this branch cannot bind: even maximum PUE,
        zero PV and the zero-logit compute request fit under the import limit.
        Only proved-safe batches use the parallel scan; all others execute the
        unchanged general decoder.
        """

        if reference.da.ndim != 3 or reference.da.shape[-1] != NODES:
            raise ValueError("zero-action reference must be [B,H,4]")
        batch, horizon, _ = reference.da.shape
        constraints.validate(batch)
        if reference.arrivals.shape != (batch, horizon, NODES):
            raise ValueError("reference observations must match action horizon")

        dtype = torch.float32
        device = reference.da.device
        power_limit = constraints.battery_power_limit.to(device=device, dtype=dtype)
        compute_limit = constraints.compute_capacity.to(device=device, dtype=dtype)
        idle_power = constraints.idle_power.to(device=device, dtype=dtype)
        work_power = constraints.work_power_per_unit.to(device=device, dtype=dtype)
        pue_max = constraints.pue_max.to(device=device, dtype=dtype)
        grid_import_limit = constraints.grid_import_limit.to(
            device=device, dtype=dtype
        )
        maximum_zero_action_facility = (
            idle_power + 0.5 * work_power * compute_limit
        ) * pue_max
        if not bool((maximum_zero_action_facility <= grid_import_limit).all()):
            self.last_zero_action_decode_path = "generic_fallback"
            raw_actions = torch.zeros(
                batch,
                horizon,
                RAW_ACTION_DIM,
                device=device,
                dtype=dtype,
            )
            return self.forward(raw_actions, reference, constraints)

        self.last_zero_action_decode_path = "lindley_fast"
        zero_logit = torch.zeros(batch, horizon, 1, device=device, dtype=dtype)
        battery_request = (
            torch.tanh(
                zero_logit / self.battery_temperature.abs().clamp_min(0.25)
            )
            * power_limit[:, None]
        )
        compute_request = (
            torch.sigmoid(
                zero_logit / self.compute_temperature.abs().clamp_min(0.25)
            )
            * compute_limit[:, None]
        )
        arrivals = reference.arrivals.float()
        compute, queue_trace = _lindley_compute_and_queue(
            arrivals,
            compute_request,
            constraints.initial_queue.to(device=device, dtype=dtype),
        )

        battery = battery_request
        migration = torch.zeros(
            batch,
            horizon,
            NODES,
            NODES,
            device=device,
            dtype=dtype,
        )
        bounded_soc = torch.minimum(
            constraints.initial_soc.to(device=device, dtype=dtype).clamp_min(0.0),
            constraints.battery_energy_capacity.to(device=device, dtype=dtype),
        )
        soc_trace = torch.cat(
            (
                constraints.initial_soc.to(device=device, dtype=dtype)[:, None],
                bounded_soc[:, None].expand(-1, horizon, -1),
            ),
            dim=1,
        )

        pue = reference.pue.float()
        pv_available = reference.pv.float()
        it_power = idle_power[:, None] + work_power[:, None] * compute
        facility_power = it_power * pue
        net = facility_power - pv_available - battery
        export_limit = constraints.grid_export_limit.to(
            device=device, dtype=dtype
        )[:, None]
        curtailment = torch.minimum(
            (-export_limit - net).clamp_min(0.0),
            pv_available,
        )
        pv_used = pv_available - curtailment
        net = facility_power - pv_used - battery
        grid = torch.maximum(
            torch.minimum(net, grid_import_limit[:, None]),
            -export_limit,
        )
        unserved = (net - grid).clamp_min(0.0)
        supplied = grid + pv_used + battery + unserved
        power_residual = facility_power - supplied
        migration_residual = torch.zeros(
            batch, horizon, device=device, dtype=dtype
        )
        intervention = (
            (battery_request - battery).abs()
            + (compute_request - compute).abs()
            + curtailment
            + unserved
        )

        return WorldActionPacket(
            battery=battery,
            compute=compute,
            migration=migration,
            grid=grid,
            pv_used=pv_used,
            pv_curtailed=curtailment,
            unserved_power=unserved,
            soc=soc_trace,
            queue=queue_trace,
            power_balance_residual=power_residual,
            migration_balance_residual=migration_residual,
            constraint_violation=unserved,
            shield_intervention=intervention,
        ).validate()

    def forward(
        self,
        raw_actions: Tensor,
        reference: WorldObservationPacket,
        constraints: WorldConstraintPacket,
    ) -> WorldActionPacket:
        if raw_actions.ndim != 3 or raw_actions.shape[-1] != RAW_ACTION_DIM:
            raise ValueError(f"raw_actions must be [B,H,{RAW_ACTION_DIM}]")
        # This module is the analytic shield.  Promote its complete recurrence
        # to FP32 even when the surrounding neural planner uses BF16 autocast.
        raw_actions = raw_actions.float()
        batch, horizon, _ = raw_actions.shape
        constraints.validate(batch)
        if reference.da.shape[:2] != (batch, horizon):
            raise ValueError("reference observations must match action horizon")
        dtype = raw_actions.dtype
        dt = constraints.dt_hours[:, None].to(dtype)
        power_limit = constraints.battery_power_limit.to(dtype)
        energy_capacity = constraints.battery_energy_capacity.to(dtype)
        charge_efficiency = constraints.charge_efficiency.to(dtype)
        discharge_efficiency = constraints.discharge_efficiency.to(dtype)
        compute_capacity = constraints.compute_capacity.to(dtype)
        migration_capacity = constraints.migration_capacity.to(dtype)
        migration_allowed = constraints.migration_allowed

        soc = constraints.initial_soc.to(dtype)
        queue = constraints.initial_queue.to(dtype)
        soc_trace = [soc]
        queue_trace = [queue]
        battery_values: list[Tensor] = []
        compute_values: list[Tensor] = []
        migration_values: list[Tensor] = []
        grid_values: list[Tensor] = []
        pv_used_values: list[Tensor] = []
        curtailment_values: list[Tensor] = []
        unserved_values: list[Tensor] = []
        power_residual_values: list[Tensor] = []
        migration_residual_values: list[Tensor] = []
        violation_values: list[Tensor] = []
        intervention_values: list[Tensor] = []

        battery_request = (
            torch.tanh(
                raw_actions[..., 0, None]
                / self.battery_temperature.abs().clamp_min(0.25)
            )
            * power_limit[:, None]
        )
        compute_request = (
            torch.sigmoid(
                raw_actions[..., 1, None]
                / self.compute_temperature.abs().clamp_min(0.25)
            )
            * compute_capacity[:, None]
        )
        migration_logits = raw_actions[..., 2:].reshape(batch, horizon, NODES, NODES)

        for index in range(horizon):
            request = battery_request[:, index]
            maximum_discharge = torch.minimum(
                power_limit,
                soc * discharge_efficiency / dt,
            )
            maximum_charge = torch.minimum(
                power_limit,
                (energy_capacity - soc).clamp_min(0.0) / (charge_efficiency * dt),
            )
            battery = torch.where(
                request >= 0.0,
                torch.minimum(request, maximum_discharge),
                -torch.minimum(-request, maximum_charge),
            )
            discharge = battery.clamp_min(0.0)
            charge = (-battery).clamp_min(0.0)
            soc = (
                soc
                + charge * charge_efficiency * dt
                - discharge / discharge_efficiency * dt
            ).clamp_min(0.0)
            soc = torch.minimum(soc, energy_capacity)

            available_queue = queue + reference.arrivals[:, index]
            # A zero neural request must mean "do not migrate".  A sigmoid
            # would map the planner's neutral logit to half of every available
            # route, creating a large fictitious transfer before the policy has
            # learned anything.  Positive tanh keeps an exact zero baseline,
            # preserves a smooth positive branch, and remains capacity bounded.
            migration_fraction = torch.tanh(migration_logits[:, index]).clamp_min(0.0)
            migration = migration_fraction * migration_capacity
            migration = migration * migration_allowed.to(dtype)
            diagonal = torch.eye(NODES, device=raw_actions.device, dtype=torch.bool)[
                None
            ]
            migration = migration.masked_fill(diagonal, 0.0)
            outgoing = migration.sum(dim=-1)
            row_scale = torch.minimum(
                torch.ones_like(outgoing),
                available_queue / outgoing.clamp_min(1.0e-8),
            )
            migration = migration * row_scale[..., None]
            outgoing = migration.sum(dim=-1)
            incoming = migration.sum(dim=-2)
            migrated_queue = (available_queue - outgoing + incoming).clamp_min(0.0)
            requested_compute = compute_request[:, index]
            compute = torch.minimum(requested_compute, migrated_queue)

            it_power = (
                constraints.idle_power.to(dtype)
                + constraints.work_power_per_unit.to(dtype) * compute
            )
            pue = reference.pue[:, index]
            facility_power = it_power * pue
            pv_available = reference.pv[:, index]
            net = facility_power - pv_available - battery
            import_excess = (net - constraints.grid_import_limit.to(dtype)).clamp_min(
                0.0
            )
            marginal_power = (
                constraints.work_power_per_unit.to(dtype) * pue
            ).clamp_min(1.0e-6)
            compute_reduction = torch.minimum(compute, import_excess / marginal_power)
            compute = compute - compute_reduction
            migrated_queue = migrated_queue + compute_reduction
            it_power = (
                constraints.idle_power.to(dtype)
                + constraints.work_power_per_unit.to(dtype) * compute
            )
            facility_power = it_power * pue
            net = facility_power - pv_available - battery

            export_limit = constraints.grid_export_limit.to(dtype)
            curtailment = (-export_limit - net).clamp_min(0.0)
            curtailment = torch.minimum(curtailment, pv_available)
            pv_used = pv_available - curtailment
            net = facility_power - pv_used - battery
            grid = torch.maximum(
                torch.minimum(net, constraints.grid_import_limit.to(dtype)),
                -export_limit,
            )
            unserved = (net - grid).clamp_min(0.0)
            supplied = grid + pv_used + battery + unserved
            power_residual = facility_power - supplied
            queue = (migrated_queue - compute).clamp_min(0.0)
            migration_residual = incoming.sum(dim=-1) - outgoing.sum(dim=-1)
            intervention = (
                (request - battery).abs()
                + (requested_compute - compute).abs()
                + curtailment
                + unserved
            )
            violation = unserved

            battery_values.append(battery)
            compute_values.append(compute)
            migration_values.append(migration)
            grid_values.append(grid)
            pv_used_values.append(pv_used)
            curtailment_values.append(curtailment)
            unserved_values.append(unserved)
            power_residual_values.append(power_residual)
            migration_residual_values.append(migration_residual)
            violation_values.append(violation)
            intervention_values.append(intervention)
            soc_trace.append(soc)
            queue_trace.append(queue)

        return WorldActionPacket(
            battery=torch.stack(battery_values, dim=1),
            compute=torch.stack(compute_values, dim=1),
            migration=torch.stack(migration_values, dim=1),
            grid=torch.stack(grid_values, dim=1),
            pv_used=torch.stack(pv_used_values, dim=1),
            pv_curtailed=torch.stack(curtailment_values, dim=1),
            unserved_power=torch.stack(unserved_values, dim=1),
            soc=torch.stack(soc_trace, dim=1),
            queue=torch.stack(queue_trace, dim=1),
            power_balance_residual=torch.stack(power_residual_values, dim=1),
            migration_balance_residual=torch.stack(migration_residual_values, dim=1),
            constraint_violation=torch.stack(violation_values, dim=1),
            shield_intervention=torch.stack(intervention_values, dim=1),
        ).validate()


__all__ = [
    "ConstraintNativeActionDecoder",
    "RAW_ACTION_DIM",
    "repeat_constraints",
    "repeat_observations",
]
