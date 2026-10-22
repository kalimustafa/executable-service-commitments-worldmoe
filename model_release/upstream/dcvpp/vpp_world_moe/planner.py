"""Eight-candidate, three-refinement latent planner for VPP-WorldMoE."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .actions import (
    RAW_ACTION_DIM,
    ConstraintNativeActionDecoder,
    repeat_constraints,
    repeat_observations,
)
from .config import WorldMoEConfig
from .dynamics import (
    CompactActionConditionedWorldDynamics,
    deadline_expired_work,
)
from .observations import WorldObservationQueries
from .packets import (
    WorldConstraintPacket,
    WorldPlanPacket,
    WorldStatePacket,
)


class HierarchicalLatentPlanner(nn.Module):
    """Learned proposal/refinement planner with analytic constraint decoding."""

    score_feature_dim = 5
    candidate_path_schedule = (5, 10, 20)
    candidate_keep_schedule = (8, 4, 2)
    policy_moe_roles = (
        "balanced_cost",
        "tail_cvar",
        "sla_queue",
        "battery_arbitrage",
        "pv_absorption",
        "migration_relief",
        "shield_feasibility",
        "tail_emergency",
    )
    # Columns: expected cost, CVaR, terminal queue, unserved power, shield.
    # These are fixed role semantics, not trainable state. Candidate embedding
    # index i is therefore a stable Policy-MoE expert role across checkpoints.
    policy_role_feature_weights = (
        (0.35, 0.35, 0.10, 0.10, 0.10),
        (0.05, 0.75, 0.05, 0.10, 0.05),
        (0.05, 0.10, 0.50, 0.30, 0.05),
        (0.70, 0.10, 0.05, 0.05, 0.10),
        (0.55, 0.10, 0.05, 0.10, 0.20),
        (0.10, 0.10, 0.50, 0.20, 0.10),
        (0.10, 0.10, 0.05, 0.25, 0.50),
        (0.05, 0.45, 0.20, 0.25, 0.05),
    )
    policy_role_tail_affinity = (-0.40, 0.85, 0.75, -0.10, 0.20, 0.45, 0.35, 1.20)
    policy_role_uncertainty_affinity = (
        -0.25,
        0.80,
        0.35,
        -0.10,
        0.15,
        0.25,
        0.45,
        1.00,
    )
    raw_constraint_names = (
        "soc_lower",
        "soc_upper",
        "grid_import",
        "grid_export",
        "queue_service",
        "migration_queue",
        "power_balance",
        "sla_deadline",
    )

    def __init__(
        self,
        config: WorldMoEConfig,
        observation_queries: WorldObservationQueries,
        dynamics: CompactActionConditionedWorldDynamics,
        decoder: ConstraintNativeActionDecoder,
    ) -> None:
        super().__init__()
        self.config = config
        self.observation_queries = observation_queries
        self.dynamics = dynamics
        self.decoder = decoder
        width = config.d_model
        self.candidate_embedding = nn.Parameter(
            torch.randn(config.planner_candidates, width) / width**0.5
        )
        self.time_embedding = nn.Parameter(
            torch.randn(config.horizon, width) / width**0.5
        )
        self.reference_transition = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, RAW_ACTION_DIM),
        )
        # DA commitment is a first-class policy decision.  It cannot be
        # reconstructed from real-time grid exchange after settlement.
        self.day_ahead_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 4),
        )
        self.cvar_eta_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self.register_buffer(
            "constraint_multipliers_buffer",
            torch.zeros(len(self.raw_constraint_names), dtype=torch.float32),
        )
        self.register_buffer(
            "constraint_penalties_buffer",
            torch.ones(len(self.raw_constraint_names), dtype=torch.float32),
        )
        self.refinement = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width + self.score_feature_dim),
                nn.Linear(width + self.score_feature_dim, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            for _ in range(config.planner_refinements)
        )
        self.refinement_norm = nn.ModuleList(
            nn.LayerNorm(width) for _ in range(config.planner_refinements)
        )

    def _reference_latent(self, state: WorldStatePacket, horizon: int) -> Tensor:
        time = self.time_embedding[:horizon].to(state.pooled_state.dtype)
        base = state.pooled_state[:, None] + time[None]
        return state.pooled_state[:, None] + self.reference_transition(base)

    def _raw_actions(self, candidate: Tensor, horizon: int) -> Tensor:
        time = self.time_embedding[:horizon].to(candidate.dtype)
        hidden = candidate[:, :, None] + time[None, None]
        return self.action_head(hidden)

    def _day_ahead_commitment(
        self,
        candidate: Tensor,
        constraints: WorldConstraintPacket,
        horizon: int,
    ) -> Tensor:
        time = self.time_embedding[:horizon].to(candidate.dtype)
        hidden = candidate[:, :, None] + time[None, None]
        request = torch.tanh(self.day_ahead_head(hidden).float())
        import_limit = constraints.grid_import_limit.float()[:, None, None]
        export_limit = constraints.grid_export_limit.float()[:, None, None]
        return torch.where(
            request >= 0.0,
            request * import_limit,
            request * export_limit,
        )

    @staticmethod
    def _gather_candidates(value: Tensor, indices: Tensor) -> Tensor:
        if value.shape[:1] != indices.shape[:1] or indices.ndim != 2:
            raise ValueError("candidate gather indices must be [B,K]")
        expanded = indices
        for _ in range(value.ndim - 2):
            expanded = expanded.unsqueeze(-1)
        return value.gather(
            1, expanded.expand(-1, -1, *value.shape[2:])
        )

    @staticmethod
    def _scatter_candidates(base: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        expanded = indices
        for _ in range(base.ndim - 2):
            expanded = expanded.unsqueeze(-1)
        return base.scatter(
            1, expanded.expand(-1, -1, *base.shape[2:]), values
        )

    @staticmethod
    def _normalize_candidate_values(value: Tensor) -> Tensor:
        """Candidate-axis normalization that cannot saturate on raw daily cost."""

        value32 = value.float()
        centered = value32 - value32.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
        return torch.tanh((centered / scale).clamp(-5.0, 5.0))

    def _policy_moe_context(self, state: WorldStatePacket) -> tuple[Tensor, Tensor]:
        tail = state.queue_tail_route_probabilities.float().squeeze(-1).amax(dim=1)
        batch_index = torch.arange(
            state.pooled_state.shape[0], device=state.pooled_state.device
        )
        origin = state.origin_index.to(device=batch_index.device, dtype=torch.long)
        probability = state.regime_probabilities.float()[batch_index, origin]
        entropy = -(
            probability.clamp_min(1.0e-8) * probability.clamp_min(1.0e-8).log()
        ).sum(dim=-1).mean(dim=-1)
        entropy = entropy / torch.log(
            entropy.new_tensor(float(self.config.num_regimes))
        )
        return tail.clamp(0.0, 1.0), entropy.clamp(0.0, 1.0)

    def _role_conditioned_scores(
        self,
        base_score: Tensor,
        normalized_features: Tensor,
        role_indices: Tensor,
        *,
        tail_probability: Tensor,
        mixture_uncertainty: Tensor,
    ) -> Tensor:
        base = self._normalize_candidate_values(base_score)
        weights = base.new_tensor(self.policy_role_feature_weights)[role_indices]
        specialist = (weights * normalized_features).sum(dim=-1)
        tail_affinity = base.new_tensor(self.policy_role_tail_affinity)[role_indices]
        uncertainty_affinity = base.new_tensor(
            self.policy_role_uncertainty_affinity
        )[role_indices]
        context_bonus = (
            0.35 * (tail_probability[:, None] - 0.5) * tail_affinity
            + 0.20
            * (mixture_uncertainty[:, None] - 0.5)
            * uncertainty_affinity
        )
        return base + 0.25 * specialist - context_bonus

    @staticmethod
    def _normalized_day_ahead(
        commitment: Tensor, constraints: WorldConstraintPacket
    ) -> Tensor:
        positive_scale = constraints.grid_import_limit.float()[:, None]
        negative_scale = constraints.grid_export_limit.float()[:, None]
        return torch.where(
            commitment >= 0.0,
            commitment / positive_scale.clamp_min(1.0e-6),
            commitment / negative_scale.clamp_min(1.0e-6),
        ).clamp(-1.0, 1.0)

    def _normalized_ramp_actions(
        self,
        raw_decoder: Tensor,
        day_ahead_commitment: Tensor,
        constraints: WorldConstraintPacket,
    ) -> Tensor:
        """Return 24 true controllable pre-shield coordinates in unit scale."""

        raw32 = raw_decoder.float()
        battery = torch.tanh(
            raw32[..., 0, None]
            / self.decoder.battery_temperature.abs().clamp_min(0.25)
        ).expand(-1, -1, 4)
        compute = torch.sigmoid(
            raw32[..., 1, None]
            / self.decoder.compute_temperature.abs().clamp_min(0.25)
        ).expand(-1, -1, 4)
        migration = torch.tanh(raw32[..., 2:].reshape(*raw32.shape[:-1], 4, 4))
        migration = migration.clamp_min(0.0) * constraints.migration_allowed[:, None]
        off_diagonal = ~torch.eye(4, dtype=torch.bool, device=raw32.device)
        migration = migration[..., off_diagonal]
        day_ahead = self._normalized_day_ahead(day_ahead_commitment, constraints)
        return torch.cat((battery, compute, day_ahead, migration), dim=-1)

    def _previous_ramp_context(
        self,
        state: WorldStatePacket,
        constraints: WorldConstraintPacket,
        previous_actions: Tensor | None,
        action_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Map the last causal action at the decision origin to ramp coordinates."""

        batch = state.pooled_state.shape[0]
        zero = state.pooled_state.new_zeros(batch, 24, dtype=torch.float32)
        invalid = torch.zeros(batch, 24, dtype=torch.bool, device=zero.device)
        if previous_actions is None or action_mask is None:
            return zero, invalid
        if (
            previous_actions.ndim != 4
            or previous_actions.shape[0] != batch
            or previous_actions.shape[-2:] != (4, 6)
            or action_mask.shape != previous_actions.shape
            or action_mask.dtype is not torch.bool
        ):
            raise ValueError("previous action context must be matching [B,T,4,6]")
        batch_index = torch.arange(batch, device=previous_actions.device)
        origin = state.origin_index.to(device=previous_actions.device, dtype=torch.long)
        value = previous_actions.float()[batch_index, origin]
        valid = action_mask[batch_index, origin]
        battery = value[..., 1] / constraints.battery_power_limit.float().clamp_min(
            1.0e-6
        )
        compute = value[..., 2] / constraints.compute_capacity.float().clamp_min(
            1.0e-6
        )
        day_ahead = self._normalized_day_ahead(
            value[..., 0][:, None], constraints
        ).squeeze(1)
        result = torch.cat(
            (battery, compute, day_ahead, zero[:, 12:]), dim=-1
        ).clamp(-1.0, 1.0)
        result_mask = torch.cat(
            (valid[..., 1], valid[..., 2], valid[..., 0], invalid[:, 12:]), dim=-1
        )
        return result, result_mask

    def constraint_parameters(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        multipliers = {
            name: self.constraint_multipliers_buffer[index]
            for index, name in enumerate(self.raw_constraint_names)
        }
        penalties = {
            name: self.constraint_penalties_buffer[index]
            for index, name in enumerate(self.raw_constraint_names)
        }
        return multipliers, penalties

    @torch.no_grad()
    def update_augmented_lagrangian(
        self, violations: dict[str, Tensor]
    ) -> None:
        """Perform the projected dual update after a completed optimizer step."""

        means = []
        for name in self.raw_constraint_names:
            if name not in violations:
                raise ValueError(f"missing raw constraint violation {name}")
            means.append(violations[name].float().clamp_min(0.0).mean())
        observed = torch.stack(means).to(self.constraint_multipliers_buffer.device)
        self.constraint_multipliers_buffer.copy_(
            (
                self.constraint_multipliers_buffer
                + self.constraint_penalties_buffer * observed
            ).clamp_min(0.0)
        )

    def raw_constraint_violations(
        self,
        raw_decoder: Tensor,
        raw_requests: Tensor,
        reference: object,
        constraints: WorldConstraintPacket,
        *,
        deadline_intervals: int = 4,
    ) -> dict[str, Tensor]:
        """Measure requested-action violations before the analytic shield."""

        raw = raw_decoder.float()
        requests = raw_requests.float()
        if raw.ndim != 3 or raw.shape[-1] != RAW_ACTION_DIM:
            raise ValueError("raw decoder actions must be [B,T,18]")
        if requests.shape != raw.shape[:-1] + (24,):
            raise ValueError("raw physical requests must be [B,T,24]")
        battery, compute, grid, pv_used, curtailed, unserved = requests.split(4, dim=-1)
        del pv_used, curtailed, unserved
        dt = constraints.dt_hours.float()[:, None]
        capacity = constraints.battery_energy_capacity.float()
        efficiency_in = constraints.charge_efficiency.float()
        efficiency_out = constraints.discharge_efficiency.float()
        soc_delta = (
            (-battery).clamp_min(0.0) * efficiency_in[:, None] * dt[..., None]
            - battery.clamp_min(0.0)
            / efficiency_out[:, None]
            * dt[..., None]
        )
        soc = constraints.initial_soc.float()[:, None] + soc_delta.cumsum(dim=1)

        migration_logits = raw[..., 2:].reshape(*raw.shape[:-1], 4, 4)
        migration = (
            torch.tanh(migration_logits).clamp_min(0.0)
            * constraints.migration_capacity.float()[:, None]
            * constraints.migration_allowed[:, None].float()
        )
        diagonal = torch.eye(4, dtype=torch.bool, device=raw.device)
        migration = migration.masked_fill(diagonal[None, None], 0.0)
        arrivals = getattr(reference, "arrivals").float()
        outgoing = migration.sum(dim=-1)
        incoming = migration.sum(dim=-2)
        queue_delta = arrivals - outgoing + incoming - compute
        queue_after = (
            constraints.initial_queue.float()[:, None]
            + queue_delta.cumsum(dim=1)
        )
        queue_before = torch.cat(
            (constraints.initial_queue.float()[:, None], queue_after[:, :-1]), dim=1
        )
        available = queue_before + arrivals
        migration_queue = outgoing - available
        # migrated = queue_after + compute, hence compute - migrated = -queue_after.
        queue_service = -queue_after

        facility = (
            constraints.idle_power.float()[:, None]
            + constraints.work_power_per_unit.float()[:, None] * compute
        ) * getattr(reference, "pue").float()
        power_residual = facility - (
            grid + getattr(reference, "pv").float() + battery
        )
        sla = deadline_expired_work(
            arrivals[:, None],
            compute[:, None],
            constraints.initial_queue,
            deadline_intervals=deadline_intervals,
        ).squeeze(1)[..., None].expand(-1, -1, 4) / 4.0
        return {
            "soc_lower": -soc,
            "soc_upper": soc - capacity[:, None],
            "grid_import": grid - constraints.grid_import_limit.float()[:, None],
            "grid_export": -constraints.grid_export_limit.float()[:, None] - grid,
            "queue_service": queue_service,
            "migration_queue": migration_queue,
            "power_balance": power_residual.abs(),
            "sla_deadline": sla,
        }

    def _score_candidates(
        self,
        state: WorldStatePacket,
        raw: Tensor,
        day_ahead_commitment: Tensor,
        reference: object,
        constraints: WorldConstraintPacket,
        *,
        seed: int | None,
        path_count: int,
    ) -> tuple[Tensor, Tensor]:
        batch, candidates, horizon, _ = raw.shape
        flat_raw = raw.reshape(batch * candidates, horizon, RAW_ACTION_DIM)
        flat_commitment = day_ahead_commitment.reshape(
            batch * candidates, horizon, 4
        )
        flat_constraints = repeat_constraints(constraints, candidates)
        flat_reference = repeat_observations(reference, candidates)
        actions = self.decoder(flat_raw, flat_reference, flat_constraints)
        initial = state.pooled_state[:, None].expand(-1, candidates, -1)
        initial = initial.reshape(batch * candidates, -1)
        rollout = self.dynamics(
            initial,
            actions,
            flat_constraints,
            path_count=int(path_count),
            seed=seed,
            day_ahead_commitment=flat_commitment,
            candidate_screening=True,
            common_random_group_size=candidates,
        )
        score = rollout.expected_cost + self.config.cvar_weight * rollout.cvar_cost
        score = score.reshape(batch, candidates)
        features = torch.stack(
            (
                rollout.expected_cost,
                rollout.cvar_cost,
                actions.queue[:, -1].mean(dim=-1),
                actions.unserved_power.mean(dim=(-1, -2)),
                actions.shield_intervention.mean(dim=(-1, -2)),
            ),
            dim=-1,
        ).reshape(batch, candidates, self.score_feature_dim)
        return score, self._normalize_candidate_values(features)

    def forward(
        self,
        state: WorldStatePacket,
        constraints: WorldConstraintPacket,
        *,
        horizon: int,
        path_count: int,
        seed: int | None = None,
        previous_actions: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> WorldPlanPacket:
        if path_count != self.config.rollout_paths:
            raise ValueError("latent planner requires the registered 20 paths")
        if not 1 <= horizon <= self.config.horizon:
            raise ValueError("planner horizon lies outside [1,96]")
        if (
            self.config.planner_candidates != len(self.policy_moe_roles)
            or self.config.planner_refinements != len(self.candidate_path_schedule)
        ):
            raise ValueError("Policy-MoE requires eight roles and three screening tiers")
        reference_latent = self._reference_latent(state, horizon)
        reference = self.observation_queries.from_latent_timeline(
            reference_latent, constraints
        )
        batch = state.pooled_state.shape[0]
        candidate_bank = (
            state.pooled_state[:, None]
            + self.candidate_embedding[None].to(state.pooled_state.dtype)
        )
        active_indices = torch.arange(
            self.config.planner_candidates,
            device=state.pooled_state.device,
            dtype=torch.long,
        )[None].expand(batch, -1)
        active_features = state.pooled_state.new_zeros(
            batch, self.config.planner_candidates, self.score_feature_dim
        )
        tail_probability, mixture_uncertainty = self._policy_moe_context(state)
        score_trace: list[Tensor] = []
        refinement_trace: list[Tensor] = []
        tier_roles: list[Tensor] = []
        final_raw: Tensor | None = None
        final_commitment: Tensor | None = None
        final_conditioned_score: Tensor | None = None
        for index, (screen_paths, expected_active) in enumerate(
            zip(
                self.candidate_path_schedule,
                self.candidate_keep_schedule,
                strict=True,
            )
        ):
            if active_indices.shape[1] != int(expected_active):
                raise RuntimeError("Policy-MoE screening cardinality drifted")
            active_candidate = self._gather_candidates(candidate_bank, active_indices)
            update_input = torch.cat(
                (active_candidate, active_features.to(active_candidate.dtype)), dim=-1
            )
            active_candidate = self.refinement_norm[index](
                active_candidate + self.refinement[index](update_input)
            )
            candidate_bank = self._scatter_candidates(
                candidate_bank, active_indices, active_candidate
            )
            raw = self._raw_actions(active_candidate, horizon)
            day_ahead_commitment = self._day_ahead_commitment(
                active_candidate, constraints, horizon
            )
            # Candidate evaluation is deliberately detached.  It keeps the
            # hierarchical search within a 10 GB card; the exact final scored
            # actions remain differentiable for end-to-end policy learning.
            with torch.no_grad():
                base_score, features = self._score_candidates(
                    state,
                    raw,
                    day_ahead_commitment,
                    reference,
                    constraints,
                    # Common seeds across tiers make score changes attributable
                    # to actions rather than a newly sampled future.
                    seed=seed,
                    path_count=int(screen_paths),
                )
                conditioned_score = self._role_conditioned_scores(
                    base_score,
                    features,
                    active_indices,
                    tail_probability=tail_probability,
                    mixture_uncertainty=mixture_uncertainty,
                )
            sentinel = torch.full(
                (batch, self.config.planner_candidates),
                1.0e30,
                dtype=conditioned_score.dtype,
                device=conditioned_score.device,
            )
            score_trace.append(
                sentinel.scatter(1, active_indices, conditioned_score)
            )
            raw_trace = raw.new_zeros(
                batch,
                self.config.planner_candidates,
                horizon,
                RAW_ACTION_DIM,
            )
            refinement_trace.append(
                self._scatter_candidates(raw_trace, active_indices, raw)
            )
            tier_roles.append(active_indices)
            if index + 1 < self.config.planner_refinements:
                next_count = int(self.candidate_keep_schedule[index + 1])
                selected_local = torch.topk(
                    conditioned_score, next_count, dim=-1, largest=False
                ).indices
                active_indices = active_indices.gather(1, selected_local)
                active_features = self._gather_candidates(features, selected_local)
            else:
                final_raw = raw
                final_commitment = day_ahead_commitment
                final_conditioned_score = conditioned_score
        if (
            final_raw is None
            or final_commitment is None
            or final_conditioned_score is None
        ):
            raise RuntimeError("Policy-MoE produced no final candidate tier")
        scores = torch.stack(score_trace, dim=1)
        active_selection = torch.softmax(
            -self._normalize_candidate_values(final_conditioned_score)
            / self.config.planner_temperature,
            dim=-1,
        )
        selection = torch.zeros(
            batch,
            self.config.planner_candidates,
            dtype=active_selection.dtype,
            device=active_selection.device,
        ).scatter(1, active_indices, active_selection)
        # The blended actions below are exactly the two candidates evaluated
        # in the final 20-path tier; no unscored post-refinement action leaks.
        blended_raw = torch.einsum(
            "bc,bcht->bht", active_selection.to(final_raw.dtype), final_raw
        )
        blended_commitment = torch.einsum(
            "bc,bchn->bhn", active_selection.float(), final_commitment.float()
        )
        actions = self.decoder(blended_raw, reference, constraints)
        blended_raw32 = blended_raw.float()
        raw_battery = (
            torch.tanh(
                blended_raw32[..., 0, None]
                / self.decoder.battery_temperature.abs().clamp_min(0.25)
            )
            * constraints.battery_power_limit[:, None]
        )
        raw_compute = (
            torch.sigmoid(
                blended_raw32[..., 1, None]
                / self.decoder.compute_temperature.abs().clamp_min(0.25)
            )
            * constraints.compute_capacity[:, None]
        )
        raw_it_power = (
            constraints.idle_power[:, None]
            + constraints.work_power_per_unit[:, None] * raw_compute
        )
        raw_grid = raw_it_power * reference.pue - reference.pv - raw_battery
        zero = torch.zeros_like(raw_grid)
        raw_action_requests = torch.cat(
            (raw_battery, raw_compute, raw_grid, reference.pv, zero, zero), dim=-1
        )
        ramp_actions = self._normalized_ramp_actions(
            blended_raw, blended_commitment, constraints
        )
        previous_ramp_action, previous_ramp_mask = self._previous_ramp_context(
            state,
            constraints,
            previous_actions,
            action_mask,
        )
        rollout = self.dynamics(
            state.pooled_state,
            actions,
            constraints,
            path_count=path_count,
            seed=None if seed is None else int(seed) + self.config.planner_refinements,
            day_ahead_commitment=blended_commitment,
        )
        result = WorldPlanPacket(
            actions=actions,
            raw_actions=ramp_actions,
            candidate_scores=scores,
            selection_weights=selection,
            candidate_latents=candidate_bank,
            refinement_trace=tuple(refinement_trace),
            rollout=rollout,
        ).validate(self.config)
        raw_violations = self.raw_constraint_violations(
            blended_raw,
            raw_action_requests,
            reference,
            constraints,
        )
        multipliers, penalties = self.constraint_parameters()
        metadata = {
            "day_ahead_commitment": blended_commitment.float(),
            "cvar_eta": self.cvar_eta_head(state.pooled_state).squeeze(-1).float(),
            "raw_constraint_violations": raw_violations,
            "constraint_multipliers": multipliers,
            "constraint_penalties": penalties,
            "shield_correction": actions.shield_intervention.float(),
            "previous_ramp_action": previous_ramp_action,
            "previous_ramp_mask": previous_ramp_mask,
            "policy_moe": {
                "role_names": self.policy_moe_roles,
                "tier_role_indices": tuple(value.detach() for value in tier_roles),
                "tier_path_counts": self.candidate_path_schedule,
                "selection_weights": selection.detach(),
                "tail_probability": tail_probability.detach(),
                "mixture_uncertainty": mixture_uncertainty.detach(),
                "final_role_indices": active_indices.detach(),
                "final_role_scores": final_conditioned_score.detach(),
                "final_risk_training": "trainer_fixed_5x20_equals_100_path_RU_CVaR",
                "selection_source": "final_two_candidates_scored_at_20_paths",
            },
            "scenario_cost_components": "available_via_complete_two_settlement_runtime",
        }
        # WorldPlanPacket intentionally stays a compact transport dataclass.
        # Metadata is attached immutably for model._planner_mapping, which
        # explicitly checks this registered extension point.
        object.__setattr__(result, "metadata", metadata)
        return result


__all__ = ["HierarchicalLatentPlanner"]
