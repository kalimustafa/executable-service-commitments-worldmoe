"""Compact action-conditioned stochastic imagination dynamics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import WorldMoEConfig
from .observations import WorldObservationQueries
from .packets import (
    NODES,
    WorldActionPacket,
    WorldConstraintPacket,
    WorldRolloutPacket,
)


def _as_path_tensor(value: Tensor, paths: int, *, label: str) -> Tensor:
    """Return an FP32 ``[B,P,T,N]`` value without copying shared actions."""

    value32 = value.float()
    if value32.ndim == 3:
        value32 = value32[:, None]
    if value32.ndim != 4:
        raise ValueError(f"{label} must be [B,T,N] or [B,P,T,N]")
    if value32.shape[1] == 1:
        value32 = value32.expand(-1, int(paths), -1, -1)
    elif value32.shape[1] != int(paths):
        raise ValueError(f"{label} path count differs from the rollout")
    return value32


def deadline_expired_work(
    arrivals: Tensor,
    service: Tensor,
    initial_queue: Tensor,
    *,
    deadline_intervals: int,
) -> Tensor:
    """Compute only work that has truly crossed its service deadline.

    A positive queue is *not* an SLA violation.  Work becomes due only after
    ``deadline_intervals`` have elapsed.  The reduction is VPP-wide because
    migration changes location but cannot reset job age.  The returned tensor
    is the incremental newly-expired amount at every path/time step.
    """

    arrivals32 = arrivals.float()
    service32 = service.float()
    if arrivals32.ndim != 4:
        raise ValueError("arrivals must be [B,P,T,N]")
    service32 = _as_path_tensor(
        service32, int(arrivals32.shape[1]), label="service"
    )
    if arrivals32.shape != service32.shape:
        raise ValueError("arrival and service shapes differ")
    delay = int(deadline_intervals)
    if delay < 1:
        raise ValueError("deadline_intervals must be positive")
    batch, paths, horizon, _ = arrivals32.shape
    initial = initial_queue.float()
    if initial.shape != (batch, NODES):
        raise ValueError("initial_queue must be [B,4]")
    # FIFO cohorts make expiry irreversible: late service cannot erase an SLA
    # miss that occurred earlier, and a later cohort is counted independently.
    cohorts: list[Tensor] = []
    expired_values: list[Tensor] = []
    arrival_total = arrivals32.sum(dim=-1)
    service_total = service32.sum(dim=-1)
    for index in range(horizon):
        newly_arrived = arrival_total[..., index]
        if index == 0:
            newly_arrived = newly_arrived + initial.sum(dim=-1)[:, None]
        cohorts.append(newly_arrived)
        remaining = service_total[..., index]
        updated: list[Tensor] = []
        for cohort in cohorts:
            served = torch.minimum(cohort.clamp_min(0.0), remaining.clamp_min(0.0))
            updated.append((cohort - served).clamp_min(0.0))
            remaining = (remaining - served).clamp_min(0.0)
        cohorts = updated
        if len(cohorts) >= delay:
            expired_values.append(cohorts.pop(0))
        else:
            expired_values.append(torch.zeros(batch, paths, device=arrivals32.device))
    return torch.stack(expired_values, dim=-1)


def two_settlement_cost_components(
    *,
    da_price: Tensor,
    rt_price: Tensor,
    day_ahead_commitment: Tensor,
    grid: Tensor,
    battery: Tensor,
    migration: Tensor,
    pv_curtailed: Tensor,
    unserved_power: Tensor,
    arrivals: Tensor,
    service: Tensor,
    initial_queue: Tensor,
    dt_hours: Tensor,
    deadline_intervals: int,
    battery_degradation_weight: float,
    migration_weight: float,
    curtailment_weight: float,
    sla_weight: float,
) -> dict[str, Tensor]:
    """Return the complete differentiable two-settlement daily path cost.

    All terms remain differentiable.  R5-V2 joint training deliberately allows
    World, Uncertainty, and Policy gradients to update the unified model.
    """

    da = da_price.float()
    rt = rt_price.float()
    if da.ndim != 4 or rt.shape != da.shape:
        raise ValueError("DA and RT prices must be matching [B,P,T,N] tensors")
    batch, paths, horizon, nodes = da.shape
    if nodes != NODES:
        raise ValueError("two-settlement paths require four nodes")
    q_da = _as_path_tensor(
        day_ahead_commitment, paths, label="day_ahead_commitment"
    )
    grid32 = _as_path_tensor(grid, paths, label="grid")
    battery32 = _as_path_tensor(battery, paths, label="battery")
    curtailed32 = _as_path_tensor(pv_curtailed, paths, label="pv_curtailed")
    unserved32 = _as_path_tensor(unserved_power, paths, label="unserved_power")
    arrivals32 = _as_path_tensor(arrivals.float(), paths, label="arrivals")
    service32 = _as_path_tensor(service.float(), paths, label="service")
    if any(
        value.shape != da.shape
        for value in (
            q_da,
            grid32,
            battery32,
            curtailed32,
            unserved32,
            arrivals32,
            service32,
        )
    ):
        raise ValueError("two-settlement cost tensors must share [B,P,T,4]")
    dt = dt_hours.float()
    if dt.shape != (batch,):
        raise ValueError("dt_hours must be [B]")
    dt4 = dt[:, None, None, None]
    market = (da * q_da + rt * (grid32 - q_da)).mul(dt4).sum(dim=(-1, -2))
    degradation = (
        battery32.abs().mul(dt4).sum(dim=(-1, -2))
        * float(battery_degradation_weight)
    )
    migration32 = migration.float()
    if migration32.ndim == 4:
        migration32 = migration32[:, None]
    if migration32.ndim != 5 or migration32.shape[:3] != (
        batch,
        paths,
        horizon,
    ):
        if migration32.ndim == 5 and migration32.shape[1] == 1:
            migration32 = migration32.expand(-1, paths, -1, -1, -1)
        else:
            raise ValueError("migration must be [B,T,4,4] or [B,P,T,4,4]")
    if migration32.shape[1] == 1:
        migration32 = migration32.expand(-1, paths, -1, -1, -1)
    migration_cost = (
        migration32.mul(dt[:, None, None, None, None]).sum(dim=(-1, -2, -3))
        * float(migration_weight)
    )
    curtailment = (
        curtailed32.mul(dt4).sum(dim=(-1, -2)) * float(curtailment_weight)
    )
    expired = deadline_expired_work(
        arrivals32,
        service32,
        initial_queue,
        deadline_intervals=deadline_intervals,
    )
    # Power shortfall remains an energy-service violation, while SLA is only
    # newly expired workload.  Both are economically material but not conflated.
    sla = (
        expired.sum(dim=-1)
        + unserved32.mul(dt4).sum(dim=(-1, -2))
    ) * float(sla_weight)
    return {
        "market": market.float(),
        "degradation": degradation.float(),
        "migration": migration_cost.float(),
        "curtailment": curtailment.float(),
        "sla": sla.float(),
    }


def complete_two_settlement_path_cost(
    *,
    da_price: Tensor,
    rt_price: Tensor,
    day_ahead_commitment: Tensor,
    actions: WorldActionPacket,
    arrivals: Tensor,
    initial_queue: Tensor,
    dt_hours: Tensor,
    deadline_intervals: int = 4,
    battery_degradation_weight: float = 0.01,
    migration_weight: float = 0.02,
    curtailment_weight: float = 1.0,
    sla_weight: float = 100.0,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Complete market, degradation, migration, curtailment and true-SLA cost."""

    components = two_settlement_cost_components(
        da_price=da_price,
        rt_price=rt_price,
        day_ahead_commitment=day_ahead_commitment,
        grid=actions.grid,
        battery=actions.battery,
        migration=actions.migration,
        pv_curtailed=actions.pv_curtailed,
        unserved_power=actions.unserved_power,
        arrivals=arrivals,
        service=actions.compute,
        initial_queue=initial_queue,
        dt_hours=dt_hours,
        deadline_intervals=deadline_intervals,
        battery_degradation_weight=battery_degradation_weight,
        migration_weight=migration_weight,
        curtailment_weight=curtailment_weight,
        sla_weight=sla_weight,
    )
    total = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return total.float(), components


def normalize_daily_path_cost(path_cost: Tensor, baseline_daily_cost: Tensor) -> Tensor:
    """Normalize by a positive train-frozen daily baseline; never batch-fit."""

    costs = path_cost.float()
    baseline = baseline_daily_cost.float()
    if baseline.ndim == 0:
        baseline = baseline.expand(costs.shape[0])
    if baseline.shape == (costs.shape[0], 1):
        baseline = baseline.squeeze(-1)
    if baseline.shape != (costs.shape[0],):
        raise ValueError("train-frozen baseline daily cost must be scalar or [B]")
    if not bool(torch.isfinite(baseline).all()) or bool((baseline <= 0).any()):
        raise ValueError("train-frozen baseline daily cost must be finite and positive")
    return costs / baseline[:, None]


class CompactActionConditionedWorldDynamics(nn.Module):
    """Roll a compact world state along action-conditioned stochastic futures.

    ``innovation_rank == 0`` preserves the registered independent-Gaussian
    sampler for legacy lineages.  Positive ranks enable R5-V2.5's unit-variance
    Student-t innovation: an idiosyncratic component plus trainable low-rank
    global and temporally persistent factors.  The deterministic W1 transition
    remains unchanged because it consumes the residual mean only.
    """

    action_summary_dim = 6
    max_path_count = 256

    @staticmethod
    def _candidate_common_random(
        shape: torch.Size,
        *,
        group_size: int | None,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        normal: bool = False,
        bank_path_count: int | None = None,
    ) -> Tensor:
        """Draw common random numbers across one origin's policy candidates.

        The default ``None`` path executes the exact registered U0 sampler.
        Candidate sharing is a P1 search-only variance reduction: flattened
        rows are ordered ``[origin, candidate]``, so each candidate sees the
        same innovations while retaining its own action-conditioned logits.
        P1 draws the registered maximum 20-path bank on every tier and slices
        its prefix, making the 5/10/20 screening futures strictly nested.
        """

        if group_size is None:
            factory = torch.randn if normal else torch.rand
            return factory(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        group = int(group_size)
        if group < 1 or int(shape[0]) % group:
            raise ValueError("candidate common-random group is incompatible with batch")
        requested_paths = int(shape[1])
        bank_paths = (
            requested_paths if bank_path_count is None else int(bank_path_count)
        )
        if bank_paths < requested_paths:
            raise ValueError("candidate random bank is smaller than requested paths")
        grouped_shape = (
            int(shape[0]) // group,
            1,
            bank_paths,
            *shape[2:],
        )
        factory = torch.randn if normal else torch.rand
        values = factory(
            grouped_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        values = values[:, :, :requested_paths]
        return values.expand(-1, group, -1, *shape[2:]).reshape(shape)

    @staticmethod
    def _candidate_common_gamma(
        shape: torch.Size,
        *,
        concentration: float,
        group_size: int | None,
        generator: torch.Generator,
        device: torch.device,
        bank_path_count: int | None = None,
    ) -> Tensor:
        """Draw Gamma(shape, 1) values with the candidate-sharing contract.

        PyTorch distributions do not accept an explicit generator.  The
        primitive used here does, which keeps resumed/formal seed chunks bitwise
        reproducible without touching process-global RNG state.
        """

        alpha = float(concentration)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("gamma concentration must be finite and positive")

        def draw(current_shape: torch.Size) -> Tensor:
            parameter = torch.full(
                current_shape,
                alpha,
                device=device,
                dtype=torch.float32,
            )
            return torch._standard_gamma(parameter, generator=generator)

        if group_size is None:
            return draw(shape)
        group = int(group_size)
        if group < 1 or int(shape[0]) % group:
            raise ValueError("candidate common-random group is incompatible with batch")
        requested_paths = int(shape[1])
        bank_paths = (
            requested_paths if bank_path_count is None else int(bank_path_count)
        )
        if bank_paths < requested_paths:
            raise ValueError("candidate random bank is smaller than requested paths")
        grouped_shape = torch.Size(
            (int(shape[0]) // group, 1, bank_paths, *shape[2:])
        )
        values = draw(grouped_shape)[:, :, :requested_paths]
        return values.expand(-1, group, -1, *shape[2:]).reshape(shape)

    def __init__(
        self, config: WorldMoEConfig, observation_queries: WorldObservationQueries
    ) -> None:
        super().__init__()
        self.config = config
        self.observation_queries = observation_queries
        self.max_path_count = int(
            getattr(config, "maximum_dynamic_particle_count", self.max_path_count)
        )
        if self.max_path_count < 2:
            raise ValueError("maximum_dynamic_particle_count must be at least two")
        width = config.d_model
        self.state_norm = nn.LayerNorm(width)
        self.action_projection = nn.Linear(self.action_summary_dim, width)
        # Keep the registered 6->D action projection checkpoint-compatible and
        # learn action continuity on a separate residual path.  Initializing
        # the delta path at zero preserves the pre-upgrade transition exactly,
        # while a non-constant teacher trajectory immediately gives the new
        # weights a gradient.
        self.action_delta_projection = nn.Linear(
            self.action_summary_dim, width, bias=False
        )
        nn.init.zeros_(self.action_delta_projection.weight)
        self.drift = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.inverse_tau = nn.Linear(width, width)
        self.regime_head = nn.Linear(2 * width, config.num_regimes)
        self.regime_embedding = nn.Parameter(
            torch.randn(config.num_regimes, width) / math.sqrt(width)
        )
        self.residual_mean = nn.Linear(2 * width, config.stochastic_dim)
        self.residual_log_scale = nn.Linear(2 * width, config.stochastic_dim)
        self.residual_projection = nn.Linear(config.stochastic_dim, width)
        self.output_norm = nn.LayerNorm(width)
        self.regime_gate = nn.Parameter(torch.tensor(-1.5))
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))

        # Rank zero is the exact legacy sampler.  R5-V2.5 YAML explicitly
        # enables a positive rank, while the dataclass default stays zero so
        # older checkpoints/configs remain behaviorally compatible.
        self.innovation_rank = int(getattr(config, "innovation_rank", 0))
        if self.innovation_rank < 0 or self.innovation_rank > config.stochastic_dim:
            raise ValueError(
                "innovation_rank must lie between zero and stochastic_dim"
            )
        self.innovation_student_t_df = float(
            getattr(config, "innovation_student_t_df", 5.0)
        )
        if self.innovation_rank > 0 and (
            not math.isfinite(self.innovation_student_t_df)
            or self.innovation_student_t_df <= 2.0
        ):
            raise ValueError(
                "innovation_student_t_df must be finite and greater than two"
            )
        if self.innovation_rank > 0:
            # Zero loadings make a parent checkpoint upgrade stable: the first
            # R5-V2.5 forward is a unit-variance heavy-tailed diagonal model,
            # and the low-rank dependence grows only through learned evidence.
            self.innovation_factor_loadings = nn.Parameter(
                torch.zeros(self.innovation_rank, config.stochastic_dim)
            )
            self.innovation_global_logit = nn.Parameter(torch.zeros(()))
            self.innovation_temporal_logit = nn.Parameter(torch.zeros(()))
            self.innovation_temporal_persistence_logit = nn.Parameter(
                torch.zeros(())
            )
        else:
            self.register_parameter("innovation_factor_loadings", None)
            self.register_parameter("innovation_global_logit", None)
            self.register_parameter("innovation_temporal_logit", None)
            self.register_parameter(
                "innovation_temporal_persistence_logit", None
            )

    @staticmethod
    def action_summary(actions: WorldActionPacket) -> Tensor:
        migration_total = actions.migration.sum(dim=(-1, -2)) / float(NODES)
        return torch.stack(
            (
                actions.battery.mean(dim=-1),
                actions.compute.mean(dim=-1),
                migration_total,
                actions.grid.mean(dim=-1),
                actions.pv_curtailed.mean(dim=-1),
                actions.unserved_power.mean(dim=-1),
            ),
            dim=-1,
        )

    @staticmethod
    def _action_scale(constraints: WorldConstraintPacket) -> Tensor:
        """Return physical scales for the six continuous action coordinates."""

        battery = constraints.battery_power_limit.float().mean(dim=-1)
        compute = constraints.compute_capacity.float().mean(dim=-1)
        migration = constraints.migration_capacity.float().sum(dim=(-1, -2)) / float(
            NODES
        )
        grid = constraints.grid_import_limit.float().mean(dim=-1)
        # Curtailment and unserved power share the electrical import scale.
        scale = torch.stack(
            (battery, compute, migration, grid, grid, grid), dim=-1
        )
        # Migration is structurally unavailable in some registered records.
        # A unit denominator keeps its exact-zero coordinate well-defined.
        return scale.clamp_min(1.0)

    def normalize_action_summary(
        self,
        action_summary: Tensor,
        constraints: WorldConstraintPacket,
    ) -> Tensor:
        """Normalize a differentiable ``[B,...,6]`` continuous trajectory."""

        if action_summary.ndim < 3 or action_summary.shape[-1] != self.action_summary_dim:
            raise ValueError("action_summary must be [B,...,6]")
        if action_summary.shape[0] != constraints.dt_hours.shape[0]:
            raise ValueError("action_summary batch differs from constraints")
        scale = self._action_scale(constraints).to(
            device=action_summary.device, dtype=action_summary.dtype
        )
        while scale.ndim < action_summary.ndim:
            scale = scale.unsqueeze(1)
        return action_summary / scale

    def _straight_through_regime(
        self,
        regime_logits: Tensor,
        gumbel: Tensor,
        *,
        state_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Return the original hard regime with a logits-only soft gradient.

        The sampled index and gathered embedding are exactly the hard-forward
        computation used by the registered model.  The zero-valued surrogate
        term supplies a straight-through gradient to ``regime_head`` without
        adding a second gradient to ``regime_embedding``.
        """

        perturbed = regime_logits.float() + gumbel.float()
        regime_index = torch.argmax(perturbed, dim=-1)
        hard_regime = self.regime_embedding[regime_index].to(state_dtype)
        if not torch.is_grad_enabled() or not regime_logits.requires_grad:
            return hard_regime, regime_index
        temperature = float(self.config.dynamics_regime_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("dynamics regime temperature must be finite and positive")
        soft_weight = torch.softmax(perturbed / temperature, dim=-1)
        soft_regime = torch.matmul(
            soft_weight,
            self.regime_embedding.detach().float(),
        ).to(state_dtype)
        regime = hard_regime + (soft_regime - soft_regime.detach())
        return regime, regime_index

    def _deterministic_step(
        self,
        state: Tensor,
        action: Tensor,
        action_delta: Tensor,
        dt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One differentiable mean transition shared by W1 and imagination.

        W1 uses the expected regime embedding and residual mean rather than a
        sampled discrete regime.  All transition parameters are nevertheless
        exactly the same parameters consumed by stochastic U0/P1 rollouts.
        The returned scalar process log-scale lets W1 train the stochastic
        residual scale without inventing a second dynamics module.
        """

        action_embedding = self.action_projection(action)
        action_embedding = action_embedding + self.action_delta_projection(
            action_delta
        )
        normalized = self.state_norm(state)
        joined = torch.cat((normalized, action_embedding), dim=-1)
        drive = torch.tanh(self.drift(joined))
        inverse_tau = F.softplus(self.inverse_tau(normalized)).clamp_max(20.0)
        while dt.ndim < state.ndim:
            dt = dt.unsqueeze(-1)
        alpha = -torch.expm1(-(inverse_tau * dt).clamp_max(20.0))
        liquid = normalized + alpha * (drive - normalized)

        regime_probability = F.softmax(self.regime_head(joined).float(), dim=-1)
        regime = torch.matmul(
            regime_probability.to(self.regime_embedding.dtype),
            self.regime_embedding,
        ).to(state.dtype)
        residual_mean = self.residual_mean(joined)
        residual_log_scale = self.residual_log_scale(joined).clamp(-7.0, 2.0)
        residual = self.residual_projection(residual_mean)
        next_state = self.output_norm(
            liquid
            + torch.sigmoid(self.regime_gate) * regime
            + torch.sigmoid(self.residual_gate) * residual
        )
        process_log_scale = residual_log_scale.float().mean(dim=-1, keepdim=True)
        return next_state, process_log_scale

    def parallel_action_response(
        self,
        state_timeline: Tensor,
        action_summary: Tensor,
        constraints: WorldConstraintPacket,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate independent lightweight action responses over one timeline.

        ``state_timeline`` supplies the state at every response position, so no
        recurrent scan is required.  Raw six-coordinate actions are normalized
        with the same physical scales as the registered W1 rollout, and both
        action level and first difference enter the shared transition
        parameters.  The result is a deterministic, fully vectorized
        ``[B,T,D]`` response plus FP32 ``[B,T,1]`` process log-scale; it does
        not execute MoE routing, stochastic paths, observation decoding, or
        economic rollout work.
        """

        if (
            state_timeline.ndim != 3
            or state_timeline.shape[-1] != self.config.d_model
        ):
            raise ValueError("state_timeline must be [B,T,D]")
        batch, horizon, _ = state_timeline.shape
        if horizon < 1:
            raise ValueError("state_timeline must contain at least one time step")
        if action_summary.shape != (batch, horizon, self.action_summary_dim):
            raise ValueError("action_summary must be matching raw [B,T,6]")
        if not bool(torch.isfinite(state_timeline).all()) or not bool(
            torch.isfinite(action_summary).all()
        ):
            raise ValueError("parallel action response inputs must be finite")
        constraints.validate(batch)

        actions = self.normalize_action_summary(action_summary.float(), constraints)
        action_delta = torch.cat(
            (
                torch.zeros_like(actions[:, :1]),
                actions[:, 1:] - actions[:, :-1],
            ),
            dim=1,
        )
        state_dtype = state_timeline.dtype
        action_embedding = self.action_projection(actions.to(state_dtype))
        action_embedding = action_embedding + self.action_delta_projection(
            action_delta.to(state_dtype)
        )
        normalized = self.state_norm(state_timeline)
        joined = torch.cat((normalized, action_embedding), dim=-1)
        drive = torch.tanh(self.drift(joined))
        inverse_tau = F.softplus(self.inverse_tau(normalized)).clamp_max(20.0)
        dt = constraints.dt_hours.to(
            device=state_timeline.device,
            dtype=state_dtype,
        )[:, None, None]
        alpha = -torch.expm1(-(inverse_tau * dt).clamp_max(20.0))
        liquid = normalized + alpha * (drive - normalized)

        regime_probability = F.softmax(self.regime_head(joined).float(), dim=-1)
        regime = torch.matmul(
            regime_probability.to(self.regime_embedding.dtype),
            self.regime_embedding,
        ).to(state_dtype)
        residual_mean = self.residual_mean(joined)
        residual_log_scale = self.residual_log_scale(joined).clamp(-7.0, 2.0)
        residual = self.residual_projection(residual_mean)
        response = self.output_norm(
            liquid
            + torch.sigmoid(self.regime_gate) * regime
            + torch.sigmoid(self.residual_gate) * residual
        )
        process_log_scale = residual_log_scale.float().mean(dim=-1, keepdim=True)
        return response, process_log_scale

    def deterministic_teacher_rollout(
        self,
        initial_state: Tensor,
        action_summary: Tensor,
        constraints: WorldConstraintPacket,
    ) -> tuple[Tensor, Tensor]:
        """Roll one continuous teacher action path for W1.

        ``action_summary`` is the actual/passive-baseline action trajectory in
        raw units.  Both :math:`a_t` and :math:`a_t-a_{t-1}` enter every step;
        no action is discretized.  The method returns ``[B,T,D]`` latent means
        and ``[B,T,1]`` process log-scales.
        """

        if initial_state.ndim != 2 or initial_state.shape[-1] != self.config.d_model:
            raise ValueError("initial_state must be [B,256]")
        if action_summary.ndim != 3 or action_summary.shape[:1] != initial_state.shape[:1]:
            raise ValueError("teacher action_summary must be [B,T,6]")
        if action_summary.shape[-1] != self.action_summary_dim:
            raise ValueError("teacher action_summary must end in six coordinates")
        actions = self.normalize_action_summary(action_summary.float(), constraints)
        state = initial_state
        previous = actions[:, 0]
        latent_values: list[Tensor] = []
        scale_values: list[Tensor] = []
        for index in range(actions.shape[1]):
            action = actions[:, index]
            delta = action - previous
            state, process_log_scale = self._deterministic_step(
                state,
                action.to(state.dtype),
                delta.to(state.dtype),
                constraints.dt_hours.to(device=state.device, dtype=state.dtype),
            )
            latent_values.append(state)
            scale_values.append(process_log_scale)
            previous = action
        return torch.stack(latent_values, dim=1), torch.stack(scale_values, dim=1)

    def _controlled_queue_cost(
        self,
        observations: object,
        actions: WorldActionPacket,
        constraints: WorldConstraintPacket,
    ) -> Tensor:
        arrivals = getattr(observations, "arrivals").float()
        batch, paths, horizon, _ = arrivals.shape
        queue = constraints.initial_queue[:, None].expand(-1, paths, -1)
        queue_cost = arrivals.new_zeros(batch, paths)
        migration = actions.migration.float()[:, None]
        compute = actions.compute.float()[:, None]
        for index in range(horizon):
            incoming = migration[:, :, index].sum(dim=-2)
            outgoing = migration[:, :, index].sum(dim=-1)
            queue = (
                queue
                + arrivals[:, :, index]
                + incoming
                - outgoing
                - compute[:, :, index]
            ).clamp_min(0.0)
            queue_cost = queue_cost + queue.square().mean(dim=-1)
        return queue_cost / float(horizon)

    def forward(
        self,
        initial_state: Tensor,
        actions: WorldActionPacket,
        constraints: WorldConstraintPacket,
        *,
        path_count: int,
        seed: int | None = None,
        day_ahead_commitment: Tensor | None = None,
        deadline_intervals: int = 4,
        candidate_screening: bool = False,
        common_random_group_size: int | None = None,
        path_seed_offsets: Sequence[int] | None = None,
        compute_path_cost: bool = True,
        teacher_action_summary: Tensor | None = None,
    ) -> WorldRolloutPacket:
        paths = int(path_count)
        if paths < 2 or paths > self.max_path_count:
            raise ValueError(
                f"path_count must lie in [2,{self.max_path_count}]"
            )
        seed_offsets = (
            None
            if path_seed_offsets is None
            else tuple(int(value) for value in path_seed_offsets)
        )
        formal_ensemble = seed_offsets is not None
        formal_chunk_paths = paths
        if formal_ensemble:
            if torch.is_grad_enabled():
                raise ValueError("formal U0 rollout is evaluation-only")
            if candidate_screening:
                raise ValueError("formal U0 paths cannot use candidate screening")
            if common_random_group_size is not None:
                raise ValueError("formal U0 paths cannot use common random groups")
            if seed is None:
                raise ValueError("formal U0 paths require a fixed base seed")
            if not seed_offsets or len(seed_offsets) != len(set(seed_offsets)):
                raise ValueError(
                    "formal U0 rollout requires one or more unique seed offsets"
                )
            if len(seed_offsets) > paths:
                raise ValueError(
                    "formal seed chunk count cannot exceed path_count"
                )
            if paths % len(seed_offsets):
                raise ValueError(
                    "formal path_count must be divisible by the seed chunk count"
                )
            formal_chunk_paths = paths // len(seed_offsets)
        if common_random_group_size is not None and not candidate_screening:
            raise ValueError("common random groups are P1 candidate-screening only")
        actions.validate()
        batch, horizon = actions.battery.shape[:2]
        if initial_state.shape != (batch, self.config.d_model):
            raise ValueError("initial_state must be [B,256]")
        generators: tuple[torch.Generator, ...]
        if formal_ensemble:
            assert seed_offsets is not None and seed is not None
            generated: list[torch.Generator] = []
            for offset in seed_offsets:
                chunk_generator = torch.Generator(device=initial_state.device)
                chunk_generator.manual_seed(int(seed) + offset)
                generated.append(chunk_generator)
            generators = tuple(generated)
            generator = None
        else:
            generator = torch.Generator(device=initial_state.device)
            if seed is not None:
                generator.manual_seed(int(seed))
            generators = ()

        def path_random(
            shape: torch.Size,
            *,
            dtype: torch.dtype,
            normal: bool = False,
        ) -> Tensor:
            if not formal_ensemble:
                assert generator is not None
                return self._candidate_common_random(
                    shape,
                    group_size=common_random_group_size,
                    generator=generator,
                    device=initial_state.device,
                    dtype=dtype,
                    normal=normal,
                    bank_path_count=max(self.config.rollout_paths, paths),
                )
            chunk_shape = torch.Size(
                (shape[0], formal_chunk_paths, *shape[2:])
            )
            return torch.cat(
                tuple(
                    self._candidate_common_random(
                        chunk_shape,
                        group_size=None,
                        generator=chunk_generator,
                        device=initial_state.device,
                        dtype=dtype,
                        normal=normal,
                        bank_path_count=formal_chunk_paths,
                    )
                    for chunk_generator in generators
                ),
                dim=1,
            )

        def path_gamma(shape: torch.Size, *, concentration: float) -> Tensor:
            if not formal_ensemble:
                assert generator is not None
                return self._candidate_common_gamma(
                    shape,
                    concentration=concentration,
                    group_size=common_random_group_size,
                    generator=generator,
                    device=initial_state.device,
                    bank_path_count=max(self.config.rollout_paths, paths),
                )
            chunk_shape = torch.Size(
                (shape[0], formal_chunk_paths, *shape[2:])
            )
            return torch.cat(
                tuple(
                    self._candidate_common_gamma(
                        chunk_shape,
                        concentration=concentration,
                        group_size=None,
                        generator=chunk_generator,
                        device=initial_state.device,
                        bank_path_count=formal_chunk_paths,
                    )
                    for chunk_generator in generators
                ),
                dim=1,
            )

        def path_student_t(shape: torch.Size, *, dtype: torch.dtype) -> Tensor:
            """Unit-variance multivariate Student-t with a shared last-axis scale."""

            normal = path_random(shape, dtype=torch.float32, normal=True)
            scale_shape = torch.Size((*shape[:-1], 1))
            chi_square = 2.0 * path_gamma(
                scale_shape,
                concentration=0.5 * self.innovation_student_t_df,
            )
            # z * sqrt((df-2)/chi2) has variance one for df > 2.  Sharing the
            # scale over the last coordinate gives a true multivariate-t draw.
            multiplier = torch.sqrt(
                (self.innovation_student_t_df - 2.0)
                / chi_square.clamp_min(torch.finfo(torch.float32).tiny)
            )
            return (normal * multiplier).to(dtype=dtype)

        state = initial_state[:, None].expand(-1, paths, -1)
        raw_action_summary = (
            self.action_summary(actions)
            if teacher_action_summary is None
            else teacher_action_summary
        )
        if raw_action_summary.shape != (batch, horizon, self.action_summary_dim):
            raise ValueError("teacher_action_summary must be [B,T,6]")
        if not bool(torch.isfinite(raw_action_summary).all()):
            raise ValueError("teacher_action_summary must contain only finite values")
        action_summary = self.normalize_action_summary(
            raw_action_summary.float(), constraints
        )
        action_delta = torch.cat(
            (
                torch.zeros_like(action_summary[:, :1]),
                action_summary[:, 1:] - action_summary[:, :-1],
            ),
            dim=1,
        )
        # Both projections are independent across time.  One timeline GEMM per
        # projection preserves the exact action-conditioned recurrence while
        # replacing 2*T tiny launches with two efficient batched operations.
        action_embeddings = self.action_projection(action_summary)
        action_embeddings = action_embeddings + self.action_delta_projection(
            action_delta
        )
        dt = constraints.dt_hours[:, None, None].to(state.dtype)

        idiosyncratic_innovations: Tensor | None = None
        global_factors: Tensor | None = None
        temporal_drivers: Tensor | None = None
        temporal_factor_state: Tensor | None = None
        regime_uniform_bank: Tensor | None = None
        if self.innovation_rank > 0:
            idiosyncratic_innovations = path_student_t(
                torch.Size(
                    (
                        batch,
                        paths,
                        horizon,
                        self.config.stochastic_dim,
                    )
                ),
                dtype=state.dtype,
            )
            global_factors = path_student_t(
                torch.Size((batch, paths, self.innovation_rank)),
                dtype=state.dtype,
            )
            temporal_drivers = path_student_t(
                torch.Size((batch, paths, horizon, self.innovation_rank)),
                dtype=state.dtype,
            )
            temporal_factor_state = torch.zeros_like(global_factors)
            # One timeline draw replaces 96 small RNG launches in R5-V2.5.
            # Rank-zero legacy mode retains its historical interleaved draw
            # order exactly for checkpoint-comparison reproducibility.
            regime_uniform_bank = path_random(
                torch.Size((batch, paths, horizon, self.config.num_regimes)),
                dtype=torch.float32,
            )
        latent_values: list[Tensor] = []
        regime_values: list[Tensor] = []
        for index in range(horizon):
            action = action_embeddings[:, index, None]
            action = action.expand(-1, paths, -1)
            normalized = self.state_norm(state)
            joined = torch.cat((normalized, action), dim=-1)
            drive = torch.tanh(self.drift(joined))
            inverse_tau = F.softplus(self.inverse_tau(normalized)).clamp_max(20.0)
            alpha = -torch.expm1(-(inverse_tau * dt).clamp_max(20.0))
            liquid = normalized + alpha * (drive - normalized)

            regime_logits = self.regime_head(joined)
            uniform = (
                regime_uniform_bank[:, :, index]
                if regime_uniform_bank is not None
                else path_random(
                    regime_logits.shape,
                    dtype=torch.float32,
                )
            ).clamp(1.0e-6, 1.0 - 1.0e-6)
            gumbel = -torch.log(-torch.log(uniform))
            regime, regime_index = self._straight_through_regime(
                regime_logits,
                gumbel,
                state_dtype=state.dtype,
            )

            mean = self.residual_mean(joined)
            log_scale = self.residual_log_scale(joined).clamp(-7.0, 2.0)
            if self.innovation_rank > 0:
                assert idiosyncratic_innovations is not None
                assert global_factors is not None
                assert temporal_drivers is not None
                assert temporal_factor_state is not None
                assert self.innovation_factor_loadings is not None
                assert self.innovation_global_logit is not None
                assert self.innovation_temporal_logit is not None
                assert self.innovation_temporal_persistence_logit is not None
                persistence = 0.98 * torch.sigmoid(
                    self.innovation_temporal_persistence_logit.float()
                )
                temporal_factor_state = (
                    persistence * temporal_factor_state.float()
                    + torch.sqrt((1.0 - persistence.square()).clamp_min(1.0e-6))
                    * temporal_drivers[:, :, index].float()
                ).to(state.dtype)
                factors = (
                    torch.sigmoid(self.innovation_global_logit.float())
                    * global_factors.float()
                    + torch.sigmoid(self.innovation_temporal_logit.float())
                    * temporal_factor_state.float()
                )
                bounded_loading = torch.tanh(
                    self.innovation_factor_loadings.float()
                ) / math.sqrt(float(self.innovation_rank))
                low_rank = F.linear(factors, bounded_loading.t()).to(mean.dtype)
                noise = idiosyncratic_innovations[:, :, index].to(mean.dtype)
                noise = noise + low_rank
            else:
                noise = path_random(
                    mean.shape,
                    dtype=mean.dtype,
                    normal=True,
                )
            residual = self.residual_projection(mean + torch.exp(log_scale) * noise)
            state = self.output_norm(
                liquid
                + torch.sigmoid(self.regime_gate) * regime
                + torch.sigmoid(self.residual_gate) * residual
            )
            latent_values.append(state)
            regime_values.append(regime_index)
        latent_paths = torch.stack(latent_values, dim=2)
        regime_indices = torch.stack(regime_values, dim=2)
        support_probabilities: dict[str, Tensor] | None = None
        if bool(getattr(self.config, "support_atom_enabled", False)):
            support_uniforms = {
                "workload_zero": path_random(
                    torch.Size((batch, paths, horizon, 1)),
                    dtype=torch.float32,
                ),
                "pv_atom": path_random(
                    torch.Size((batch, paths, horizon, NODES)),
                    dtype=torch.float32,
                ),
                "pue_min": path_random(
                    torch.Size((batch, paths, horizon, NODES)),
                    dtype=torch.float32,
                ),
            }
            observations, support_probabilities = (
                self.observation_queries.from_latent_timeline_with_support(
                    latent_paths,
                    constraints,
                    support_uniforms,
                )
            )
        else:
            observations = self.observation_queries.from_latent_timeline(
                latent_paths, constraints
            )

        # Risk, cost and tail reductions are scientific FP32 operations.  The
        # latent transition above remains eligible for neural BF16 autocast.
        # U0 consumes only the generated observation paths.  Running the P1
        # two-settlement objective here would also execute the quadratic FIFO
        # deadline recurrence despite its result having no edge to any U0 loss
        # or release gate.  Preserve the rollout packet schema with explicit
        # FP32 zero placeholders; policy/planning callers keep the default True
        # branch and therefore cannot silently lose their economic objective.
        if compute_path_cost:
            commitment = (
                torch.zeros_like(actions.grid)
                if day_ahead_commitment is None
                else day_ahead_commitment
            )
            path_cost, _ = complete_two_settlement_path_cost(
                da_price=observations.da,
                rt_price=observations.rt,
                day_ahead_commitment=commitment,
                actions=actions,
                arrivals=observations.arrivals,
                initial_queue=constraints.initial_queue,
                dt_hours=constraints.dt_hours,
                deadline_intervals=deadline_intervals,
                battery_degradation_weight=self.config.battery_degradation_weight,
                migration_weight=self.config.migration_weight,
                curtailment_weight=1.0,
                sla_weight=self.config.unserved_power_weight,
            )
        else:
            path_cost = torch.zeros(
                batch,
                paths,
                device=latent_paths.device,
                dtype=torch.float32,
            )
        weights = torch.full(
            (batch, paths),
            1.0 / float(paths),
            device=latent_paths.device,
            dtype=torch.float32,
        )
        expected = (weights * path_cost).sum(dim=-1)
        tail_count = max(1, math.ceil((1.0 - self.config.cvar_alpha) * paths))
        cvar = torch.topk(path_cost, tail_count, dim=-1).values.mean(dim=-1)
        return WorldRolloutPacket(
            latent_paths=latent_paths,
            observations=observations,
            actions=actions,
            path_cost=path_cost,
            weights=weights,
            regime_indices=regime_indices,
            expected_cost=expected,
            cvar_cost=cvar,
            support_probabilities=support_probabilities,
        ).validate(self.config)


__all__ = [
    "CompactActionConditionedWorldDynamics",
    "complete_two_settlement_path_cost",
    "deadline_expired_work",
    "normalize_daily_path_cost",
    "two_settlement_cost_components",
]
