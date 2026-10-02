"""Paired 2024 evaluation and same-checkpoint ablations for R-ECMoE.

The evaluator opens only the registered development-validation split.  One W1
expert packet is encoded per batch and reused by every registered decision
variant.  These are explicitly same-checkpoint interventions, not separately
trained models.  No optimizer, solver, candidate search, test split, or file
hash is involved.
"""

from __future__ import annotations

from contextlib import nullcontext
import csv
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .r_ecmoe import (
    COST_QUANTILES,
    ConstraintAssemblyResult,
    RECMoE,
    RECMoEOutput,
    RiskConditionPacket,
    W1ExpertPacket,
)
from .r_ecmoe_replay import RealizedHistoricalReplay
from .r_ecmoe_runtime import RECMoEInferenceRuntime


EVALUATION_DECISIONS = (
    (0, 0, 96),
    (1, 0, 96),
    (1, 4, 92),
    (1, 16, 80),
    (1, 48, 48),
    (1, 95, 1),
)
HARD_RESIDUAL_EXCLUSIONS = {"critical_shortfall", "target_soc_tracking"}


class RECMoEEvaluationError(RuntimeError):
    """Formal 2024 evaluation cannot preserve its paired evidence contract."""


@dataclass(frozen=True)
class EvaluationVariant:
    name: str
    active_top_k: int | None
    tail_mode: str
    route_prior_beta: float
    safe_tensor: bool = False
    expert_state_source: str = "selected_expert_states"
    coordinator_mode: str = "sparse_cross_expert"
    risk_condition_mode: str = "risk_token_plus_causal"
    description: str = ""

    @property
    def tail_enabled(self) -> bool:
        return self.tail_mode != "off"

    def validate(self) -> "EvaluationVariant":
        if not self.name or not self.description:
            raise RECMoEEvaluationError(
                "every evaluation variant requires a name and intervention description"
            )
        if self.tail_mode not in {"off", "conditional", "always_on"}:
            raise RECMoEEvaluationError(
                f"variant {self.name} has an unsupported tail intervention"
            )
        if self.expert_state_source not in {
            "selected_expert_states",
            "pooled_latent",
        }:
            raise RECMoEEvaluationError(
                f"variant {self.name} has an unsupported expert-state source"
            )
        if self.coordinator_mode not in {
            "sparse_cross_expert",
            "direct_weighted_mixture",
        }:
            raise RECMoEEvaluationError(
                f"variant {self.name} has an unsupported coordinator intervention"
            )
        if self.risk_condition_mode not in {
            "risk_token_plus_causal",
            "causal_only",
        }:
            raise RECMoEEvaluationError(
                f"variant {self.name} has an unsupported risk-condition intervention"
            )
        if (
            isinstance(self.route_prior_beta, bool)
            or not math.isfinite(float(self.route_prior_beta))
            or float(self.route_prior_beta) < 0.0
        ):
            raise RECMoEEvaluationError(
                f"variant {self.name} has an invalid route-prior coefficient"
            )
        if self.safe_tensor:
            if (
                self.active_top_k is not None
                or self.tail_mode != "off"
                or float(self.route_prior_beta) != 0.0
                or self.expert_state_source != "selected_expert_states"
                or self.coordinator_mode != "sparse_cross_expert"
                or self.risk_condition_mode != "risk_token_plus_causal"
            ):
                raise RECMoEEvaluationError(
                    "safe tensor cannot silently carry a learned intervention"
                )
        elif self.active_top_k not in {1, 2}:
            raise RECMoEEvaluationError(
                f"learned variant {self.name} requires Top-1 or Top-2"
            )
        return self

    def evidence_definition(self) -> dict[str, Any]:
        self.validate()
        return {
            "variant": self.name,
            "comparison_design": "same_checkpoint_intervention",
            "separately_retrained": False,
            "active_top_k": self.active_top_k,
            "tail_mode": self.tail_mode,
            "route_prior_beta": float(self.route_prior_beta),
            "safe_tensor": self.safe_tensor,
            "expert_state_source": self.expert_state_source,
            "coordinator_mode": self.coordinator_mode,
            "risk_condition_mode": self.risk_condition_mode,
            "description": self.description,
        }


REGISTERED_VARIANTS = (
    EvaluationVariant(
        "safe_tensor",
        None,
        "off",
        0.0,
        True,
        description="deterministic zero-residual neural safe tensor",
    ),
    EvaluationVariant(
        "top1",
        1,
        "conditional",
        1.0,
        description="same weights with deployment Top-1 continuation",
    ),
    EvaluationVariant(
        "top2",
        2,
        "conditional",
        1.0,
        description="registered main Top-2 conditional-tail configuration",
    ),
    EvaluationVariant(
        "no_decision_tail",
        2,
        "off",
        1.0,
        description="conditional decision-tail contribution set to zero",
    ),
    EvaluationVariant(
        "no_route_prior",
        2,
        "conditional",
        0.0,
        description="W1 route-logit prior coefficient set to zero",
    ),
    EvaluationVariant(
        "tail_always_on",
        2,
        "always_on",
        1.0,
        description="tail adapter forced on for every decision-timeline token",
    ),
    EvaluationVariant(
        "pooled_latent_instead_of_expert_states",
        2,
        "conditional",
        1.0,
        expert_state_source="pooled_latent",
        description="selected expert states replaced by repeated pooled W1 latent",
    ),
    EvaluationVariant(
        "no_sparse_coordinator",
        2,
        "conditional",
        1.0,
        coordinator_mode="direct_weighted_mixture",
        description=(
            "learned sparse coordinator replaced by a parameter-free route-weighted "
            "adapter mixture plus active tail residual"
        ),
    ),
    EvaluationVariant(
        "no_risk_condition",
        2,
        "conditional",
        1.0,
        risk_condition_mode="causal_only",
        description=(
            "learned risk/stage token removed while causal timing and physical "
            "execution constraints remain"
        ),
    ),
)


def _validate_variant_registry() -> None:
    names = [variant.validate().name for variant in REGISTERED_VARIANTS]
    if len(names) != len(set(names)) or names.count("top2") != 1:
        raise RECMoEEvaluationError(
            "evaluation variants must be unique and contain exactly one Top-2 main"
        )


_validate_variant_registry()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RECMoEEvaluationError(f"cannot write an empty evaluation table: {path}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise RECMoEEvaluationError(f"evaluation CSV schema changed within {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_ready(value) for key, value in row.items()})
    os.replace(temporary, path)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise RECMoEEvaluationError("percentile requires nonempty values")
    ordered = sorted(float(value) for value in values)
    position = probability * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _cvar95(values: Sequence[float]) -> float:
    if not values:
        raise RECMoEEvaluationError("CVaR95 requires realized daily costs")
    tail_count = max(1, int(math.ceil(0.05 * len(values))))
    tail = sorted((float(value) for value in values), reverse=True)[:tail_count]
    return sum(tail) / float(tail_count)


def moving_block_bootstrap_indices(
    days: int,
    *,
    block_days: int,
    resamples: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Fixed-seed circular moving blocks over already aggregated days."""

    if days < 2 or block_days < 1 or block_days > days or resamples < 1:
        raise RECMoEEvaluationError("invalid moving-block bootstrap dimensions")
    generator = random.Random(int(seed))
    blocks = math.ceil(days / block_days)
    result = []
    for _ in range(int(resamples)):
        indices: list[int] = []
        for _ in range(blocks):
            start = generator.randrange(days)
            indices.extend((start + offset) % days for offset in range(block_days))
        result.append(tuple(indices[:days]))
    return tuple(result)


def _autocast(runtime: RECMoEInferenceRuntime) -> Any:
    device = runtime.device
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=runtime.autocast_dtype,
        enabled=True,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_raw(packet: W1ExpertPacket) -> dict[str, Tensor]:
    batch, timeline = packet.state_latent.shape[:2]
    raw = packet.state_latent.new_zeros((batch, timeline, 36), dtype=torch.float32)
    return {
        "joined": raw,
        "day_ahead": raw[..., 0:4],
        "battery": raw[..., 4:8],
        "compute": raw[..., 8:12],
        "migration": raw[..., 12:28].reshape(batch, timeline, 4, 4),
        "pv": raw[..., 28:32],
        "target_soc": raw[..., 32:36],
    }


def _complete_output(
    model: RECMoE,
    packet: W1ExpertPacket,
    risk: RiskConditionPacket,
    assembled: ConstraintAssemblyResult,
    decision_state: Tensor | None,
    route_weights: Tensor | None,
    selected_ids: Tensor | None,
    tail_active: Tensor,
    risk_context: Tensor | None,
) -> RECMoEOutput:
    """Complete a post-decision object without using any realized target.

    The operating-cost slot is deliberately a zero placeholder.  Formal
    operating outcomes are populated only by :class:`RealizedHistoricalReplay`
    after every registered variant has already selected its routes and raw
    structured actions.
    """

    batch, timeline = packet.state_latent.shape[:2]
    if decision_state is not None and risk_context is not None:
        response, response_scale = model.apply_action_response(
            packet, assembled.action_summary
        )
        (
            cost_quantiles,
            expected_cost,
            predicted_upper_tail,
            sla_risk,
            constraint_risk,
            grid_risk,
        ) = model.risk_value_head(
            decision_state,
            response,
            response_scale,
            packet,
            risk_context,
            assembled.decision_active_mask,
            risk.tail_level,
        )
    else:
        response = torch.zeros_like(packet.state_latent)
        response_scale = packet.state_latent.new_zeros(
            (batch, timeline, 1), dtype=torch.float32
        )
        cost_quantiles = packet.state_latent.new_zeros(
            (batch, len(COST_QUANTILES)), dtype=torch.float32
        )
        expected_cost = packet.state_latent.new_zeros((batch,), dtype=torch.float32)
        predicted_upper_tail = expected_cost.clone()
        sla_risk = expected_cost.clone()
        constraint_risk = expected_cost.clone()
        grid_risk = expected_cost.clone()
    if route_weights is None:
        route_weights = packet.state_latent.new_empty((batch, timeline, 0))
    if selected_ids is None:
        selected_ids = torch.empty(
            batch,
            timeline,
            0,
            device=packet.state_latent.device,
            dtype=torch.long,
        )
    return RECMoEOutput(
        expert_packet=packet,
        actions=assembled.actions,
        day_ahead_commitment=assembled.day_ahead_commitment,
        target_soc=assembled.target_soc,
        raw_structured_actions=assembled.raw_structured_actions,
        constraint_residuals=assembled.residuals,
        response_latent=response,
        response_log_scale=response_scale,
        cost_quantiles=cost_quantiles,
        expected_cost=expected_cost,
        upper_tail_cost=predicted_upper_tail,
        sla_risk=sla_risk,
        constraint_risk=constraint_risk,
        grid_risk=grid_risk,
        differentiable_operating_cost=packet.state_latent.new_zeros(
            (batch,), dtype=torch.float32
        ),
        route_weights=route_weights,
        selected_expert_ids=selected_ids,
        tail_decision_active=tail_active,
        decision_active_mask=assembled.decision_active_mask,
        deadline_expired_work=assembled.deadline_expired_work,
        diagnostics={"outcome_basis": "decision_only_pending_realized_replay"},
    )


def _execute_variant(
    model: RECMoE,
    packet: W1ExpertPacket,
    risk: RiskConditionPacket,
    variant: EvaluationVariant,
) -> RECMoEOutput:
    """Execute one same-checkpoint intervention without another W1 encode.

    Only already-computed W1 packet tensors and frozen R weights are used.  The
    intervention is applied before raw actions are generated and before any
    realized target is admitted to the historical replay.
    """

    variant.validate()

    if variant.safe_tensor:
        assembled = model.assembler(_safe_raw(packet), packet, risk)
        inactive = torch.zeros_like(assembled.decision_active_mask)
        return _complete_output(
            model,
            packet,
            risk,
            assembled,
            None,
            None,
            None,
            inactive,
            None,
        )

    original_router_config = model.continuation_router.config
    model.continuation_router.config = replace(
        original_router_config,
        active_top_k=int(variant.active_top_k),
        route_prior_beta=float(variant.route_prior_beta),
    )
    try:
        batch, timeline = packet.state_latent.shape[:2]
        risk.validate(batch, timeline)
        causal_context = model.causal_context_projection(
            packet.causal_context.float()
        )
        if variant.risk_condition_mode == "risk_token_plus_causal":
            risk_context = model.risk_tokenizer(risk, time=timeline)
            risk_context = risk_context + causal_context.to(risk_context.dtype)
        else:
            # Origin and interval length are causal operational coordinates,
            # not tunable risk preferences, so retain only their projection.
            risk_context = causal_context

        selected_ids, selected_states, weights, _ = model.continuation_router(
            packet, risk_context
        )
        if variant.expert_state_source == "pooled_latent":
            selected_states = packet.state_latent[:, :, None].expand(
                -1, -1, selected_ids.shape[-1], -1
            )
        adapted = model.adapter_bank(
            selected_states, selected_ids, risk_context
        )

        if variant.tail_mode == "off":
            tail_active = torch.zeros(
                batch,
                timeline,
                device=packet.state_latent.device,
                dtype=torch.bool,
            )
            tail_state = torch.zeros_like(packet.tail_state)
        elif variant.tail_mode == "always_on":
            tail_active = torch.ones(
                batch,
                timeline,
                device=packet.state_latent.device,
                dtype=torch.bool,
            )
            # Apply the already-trained tail adapter to every timeline token.
            # This deliberately changes only the gate, never adapter weights.
            tail_state = model.tail_adapter.adapter(
                packet.tail_state, risk_context
            )
            tail_state = torch.where(
                tail_active[..., None], tail_state, torch.zeros_like(tail_state)
            )
        else:
            neutral_risk_weight = (
                torch.ones_like(risk.risk_weight)
                if variant.risk_condition_mode == "causal_only"
                else risk.risk_weight
            )
            tail_state, tail_active = model.tail_adapter(
                packet, risk_context, neutral_risk_weight
            )

        if variant.coordinator_mode == "sparse_cross_expert":
            decision_state = model.coordinator(
                adapted,
                weights,
                tail_state,
                tail_active,
                risk_context,
                packet.state_latent,
            )
        else:
            # Parameter-free replacement for the learned sparse coordinator.
            # Route weights and tail activity remain unchanged, so this isolates
            # the learned cross-expert mixing block rather than deleting MoE.
            decision_state = (
                adapted * weights.to(adapted.dtype)[..., None]
            ).sum(dim=2) + torch.where(
                tail_active[..., None], tail_state, torch.zeros_like(tail_state)
            )

        assembled = model.assembler(
            model.action_heads(decision_state), packet, risk
        )
        return _complete_output(
            model,
            packet,
            risk,
            assembled,
            decision_state,
            weights,
            selected_ids,
            tail_active,
            risk_context,
        )
    finally:
        model.continuation_router.config = original_router_config


def _batch_scale(value: Tensor) -> Tensor:
    numeric = value.detach().float().abs()
    return numeric.flatten(1).mean(1).clamp_min(1.0e-6)


def _normalized_tv(value: Tensor, scale: Tensor, active: Tensor) -> Tensor:
    if value.shape[1] < 2:
        return value.new_zeros((value.shape[0],), dtype=torch.float32)
    pair = active[:, 1:] & active[:, :-1]
    expanded = pair.reshape(*pair.shape, *([1] * (value.ndim - 2))).float()
    difference = (value.detach().float()[:, 1:] - value.detach().float()[:, :-1])
    numerator = (difference.abs() * expanded).flatten(1).sum(1)
    count = expanded.expand_as(difference).flatten(1).sum(1).clamp_min(1.0)
    return numerator / count / _batch_scale(scale)


def _route_gini(route_load: Tensor) -> Tensor:
    """Return the seven-route Gini coefficient for each sample.

    ``route_load`` is already normalized over the active decision horizon.  A
    zero-load row is defined as zero rather than producing an undefined ratio.
    """

    values = route_load.detach().float().clamp_min(0.0)
    if values.ndim != 2 or values.shape[-1] != 7:
        raise RECMoEEvaluationError("route Gini requires [B,7] route loads")
    numerator = (values[:, :, None] - values[:, None, :]).abs().sum(dim=(1, 2))
    denominator = 2.0 * values.shape[-1] * values.sum(dim=-1)
    return torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(1.0e-12),
        torch.zeros_like(denominator),
    )


def _masked_time_mean(value: Tensor, mask: Tensor) -> Tensor:
    if value.shape != mask.shape or value.ndim != 2:
        raise RECMoEEvaluationError("masked timeline mean requires matching [B,T]")
    numeric = value.detach().float()
    weight = mask.detach().bool().float()
    return (numeric * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _verify_replay_boundary(
    decision: RECMoEOutput,
    realized: RECMoEOutput,
) -> None:
    if realized.diagnostics.get("outcome_basis") != "realized_historical_replay":
        raise RECMoEEvaluationError(
            "formal outcomes must come from realized historical replay"
        )
    protected = (
        "day_ahead_commitment",
        "target_soc",
        "raw_structured_actions",
        "route_weights",
        "selected_expert_ids",
    )
    for name in protected:
        before = getattr(decision, name)
        after = getattr(realized, name)
        if before.data_ptr() != after.data_ptr():
            raise RECMoEEvaluationError(
                f"realized replay altered protected decision tensor {name}"
            )


def _execution_rows(
    decision: RECMoEOutput,
    realized: RECMoEOutput,
    *,
    variant: EvaluationVariant,
    dates: Sequence[str],
    stage_id: int,
    origin: int,
    horizon: int,
    cost_scale: float,
    equivalent_latency_ms_per_sample: float,
    queue_event_target: Tensor | None = None,
    queue_event_mask: Tensor | None = None,
) -> list[dict[str, Any]]:
    _verify_replay_boundary(decision, realized)
    actions = realized.actions
    source_constraints = realized.expert_packet.constraints
    active = realized.decision_active_mask.detach().bool()
    dt = source_constraints.dt_hours.detach().float()[:, None, None]
    deadline = realized.deadline_expired_work.detach().float().flatten(1).sum(1)
    unserved = (
        actions.unserved_power.detach().float() * active[..., None].float() * dt
    ).flatten(1).sum(1)
    battery_energy = (
        actions.battery.detach().float().abs() * active[..., None].float() * dt
    ).flatten(1).sum(1)
    compute_energy = (
        actions.compute.detach().float() * active[..., None].float() * dt
    ).flatten(1).sum(1)
    migration_energy = (
        actions.migration.detach().float()
        * active[..., None, None].float()
        * dt[..., None]
    ).flatten(1).sum(1)
    pv_curtailed_energy = (
        actions.pv_curtailed.detach().float() * active[..., None].float() * dt
    ).flatten(1).sum(1)
    grid = actions.grid.detach().float()
    grid_import = grid.clamp_min(0.0)
    grid_export = (-grid).clamp_min(0.0)
    grid_import_energy = (
        grid_import * active[..., None].float() * dt
    ).flatten(1).sum(1)
    grid_export_energy = (
        grid_export * active[..., None].float() * dt
    ).flatten(1).sum(1)
    import_limit = source_constraints.grid_import_limit.detach().float().clamp_min(
        1.0e-6
    )
    export_limit = source_constraints.grid_export_limit.detach().float().clamp_min(
        1.0e-6
    )
    grid_peak_import_ratio = (
        grid_import / import_limit[:, None]
    ).masked_fill(~active[..., None], 0.0).flatten(1).amax(1)
    grid_peak_export_ratio = (
        grid_export / export_limit[:, None]
    ).masked_fill(~active[..., None], 0.0).flatten(1).amax(1)
    if grid.shape[1] >= 2:
        ramp_mask = active[:, 1:] & active[:, :-1]
        grid_ramp = (grid[:, 1:] - grid[:, :-1]).abs() / (
            import_limit + export_limit
        )[:, None]
        grid_peak_ramp_ratio = grid_ramp.masked_fill(
            ~ramp_mask[..., None], 0.0
        ).flatten(1).amax(1)
    else:
        grid_peak_ramp_ratio = grid.new_zeros((grid.shape[0],))
    # A rebound is a V-shaped aggregate import excursion: a decrease into one
    # interval followed immediately by an increase.  Report the smaller leg so
    # a one-sided ramp is not mislabeled as rebound.
    aggregate_import = grid_import.sum(dim=-1)
    if grid.shape[1] >= 3:
        rebound_mask = active[:, :-2] & active[:, 1:-1] & active[:, 2:]
        down = (aggregate_import[:, :-2] - aggregate_import[:, 1:-1]).clamp_min(0.0)
        up = (aggregate_import[:, 2:] - aggregate_import[:, 1:-1]).clamp_min(0.0)
        rebound = torch.minimum(down, up) / import_limit.sum(dim=-1)[:, None]
        grid_rebound_ratio = rebound.masked_fill(~rebound_mask, 0.0).amax(dim=1)
    else:
        grid_rebound_ratio = grid.new_zeros((grid.shape[0],))
    compute = actions.compute.detach().float().clamp_min(0.0)
    compute_total = compute.sum(dim=-1)
    compute_share = compute / compute_total[..., None].clamp_min(1.0e-8)
    compute_hhi = compute_share.square().sum(dim=-1)
    compute_concentration_hhi = _masked_time_mean(
        compute_hhi,
        active & (compute_total > 1.0e-8),
    )
    terminal_unfinished_work = actions.queue.detach().float()[:, -1].sum(dim=-1)
    battery_capacity = source_constraints.battery_energy_capacity.detach().float()
    terminal_soc_fraction = (
        actions.soc.detach().float()[:, -1]
        / battery_capacity.clamp_min(1.0e-6)
    ).mean(dim=-1)
    power_scale = (
        source_constraints.grid_import_limit.detach().float()
        + source_constraints.grid_export_limit.detach().float()
    )
    tv = torch.stack(
        (
            _normalized_tv(
                actions.battery, source_constraints.battery_power_limit, active
            ),
            _normalized_tv(actions.compute, source_constraints.compute_capacity, active),
            _normalized_tv(actions.grid, power_scale, active),
            _normalized_tv(
                actions.migration, source_constraints.migration_capacity, active
            ),
        ),
        dim=-1,
    ).mean(-1)

    hard = torch.zeros(len(dates), device=actions.battery.device)
    for name, value in realized.constraint_residuals.items():
        if name in HARD_RESIDUAL_EXCLUSIONS:
            continue
        candidate = value.detach().float().abs().flatten(1).amax(1)
        hard = torch.maximum(hard, candidate)

    route_entropy = torch.zeros_like(hard)
    route_load = torch.zeros(len(dates), 7, device=hard.device)
    if decision.route_weights.shape[-1] > 0:
        weights = decision.route_weights.detach().float()
        ids = decision.selected_expert_ids.detach().long()
        active_count = active.float().sum(1).clamp_min(1.0)
        route_entropy = (
            -(
                weights * torch.log(weights.clamp_min(1.0e-8))
            ).sum(-1)
            * active.float()
        ).sum(1) / active_count
        route_load.scatter_add_(
            1,
            ids.reshape(len(dates), -1),
            (weights * active[..., None].float()).reshape(len(dates), -1),
        )
        route_load = route_load / active_count[:, None]
    route_gini = _route_gini(route_load)
    tail_rate = (
        (decision.tail_decision_active.detach().bool() & active).float().sum(1)
        / active.float().sum(1).clamp_min(1.0)
    )
    tail_probability = decision.expert_packet.tail_probability.detach().float()[..., 0]
    if queue_event_target is None:
        tail_target = (
            actions.queue.detach().float()[:, 1:].sum(dim=-1) > 1.0e-4
        ).float()
        tail_mask = active
        tail_target_source = "realized_replay_queue_fallback"
    else:
        tail_target = queue_event_target.detach().to(
            device=tail_probability.device, dtype=torch.float32
        )
        if tail_target.ndim == 3 and tail_target.shape[-1] == 1:
            tail_target = tail_target[..., 0]
        if tail_target.shape != tail_probability.shape:
            raise RECMoEEvaluationError(
                "queue-event target must match [B,T] tail probabilities"
            )
        if queue_event_mask is None:
            tail_mask = active
        else:
            tail_mask = queue_event_mask.detach().to(
                device=tail_probability.device, dtype=torch.bool
            )
            if tail_mask.ndim == 3 and tail_mask.shape[-1] == 1:
                tail_mask = tail_mask[..., 0]
            if tail_mask.shape != tail_probability.shape:
                raise RECMoEEvaluationError(
                    "queue-event mask must match [B,T] tail probabilities"
                )
            tail_mask = tail_mask & active
        tail_target_source = "registered_queue_risk_event_target"
    tail_probability_mean = _masked_time_mean(tail_probability, tail_mask)
    tail_event_rate = _masked_time_mean(tail_target, tail_mask)
    tail_probability_brier = _masked_time_mean(
        (tail_probability - tail_target).square(), tail_mask
    )
    tail_calibration_gap = (tail_probability_mean - tail_event_rate).abs()
    active_horizon = active.float().sum(1)
    if bool((active_horizon <= 0.0).any()):
        raise RECMoEEvaluationError("realized replay produced an empty active horizon")
    expected_horizon = active_horizon.new_full(active_horizon.shape, float(horizon))
    if not bool(torch.equal(active_horizon, expected_horizon)):
        raise RECMoEEvaluationError(
            "realized replay active horizon differs from the registered decision"
        )
    normalization = float(cost_scale) * active_horizon / 96.0
    realized_cost = realized.differentiable_operating_cost.detach().float()
    normalized_cost = realized_cost / normalization
    rows = []
    for index, operating_date in enumerate(dates):
        row = {
            "operating_date": operating_date,
            "variant": variant.name,
            "comparison_design": "same_checkpoint_intervention",
            "separately_retrained": False,
            "intervention_description": variant.description,
            "stage_id": int(stage_id),
            "decision_origin": int(origin),
            "horizon": int(horizon),
            "active_horizon": int(active_horizon[index].cpu()),
            "outcome_basis": "realized_historical_replay",
            "realized_operating_cost": float(realized_cost[index].cpu()),
            "cost_normalization_denominator": float(normalization[index].cpu()),
            "normalized_operating_cost": float(normalized_cost[index].cpu()),
            "deadline_expired_work": float(deadline[index].cpu()),
            "unserved_energy": float(unserved[index].cpu()),
            "hard_residual_max": float(hard[index].cpu()),
            "action_total_variation": float(tv[index].cpu()),
            "battery_throughput": float(battery_energy[index].cpu()),
            "compute_energy": float(compute_energy[index].cpu()),
            "migration_energy": float(migration_energy[index].cpu()),
            "pv_curtailed_energy": float(pv_curtailed_energy[index].cpu()),
            "grid_import_energy": float(grid_import_energy[index].cpu()),
            "grid_export_energy": float(grid_export_energy[index].cpu()),
            "grid_peak_import_ratio": float(grid_peak_import_ratio[index].cpu()),
            "grid_peak_export_ratio": float(grid_peak_export_ratio[index].cpu()),
            "grid_peak_ramp_ratio": float(grid_peak_ramp_ratio[index].cpu()),
            "grid_rebound_ratio": float(grid_rebound_ratio[index].cpu()),
            "compute_concentration_hhi": float(
                compute_concentration_hhi[index].cpu()
            ),
            "terminal_unfinished_work": float(
                terminal_unfinished_work[index].cpu()
            ),
            "terminal_soc_fraction": float(terminal_soc_fraction[index].cpu()),
            "route_entropy": float(route_entropy[index].cpu()),
            "route_gini": float(route_gini[index].cpu()),
            "tail_activation_rate": float(tail_rate[index].cpu()),
            "tail_probability_mean": float(tail_probability_mean[index].cpu()),
            "tail_event_rate": float(tail_event_rate[index].cpu()),
            "tail_probability_brier": float(tail_probability_brier[index].cpu()),
            "tail_calibration_gap": float(tail_calibration_gap[index].cpu()),
            "tail_calibration_target": tail_target_source,
            "equivalent_latency_ms_per_sample": float(
                equivalent_latency_ms_per_sample
            ),
        }
        for expert in range(7):
            row[f"route_load_{expert}"] = float(route_load[index, expert].cpu())
        rows.append(row)
    return rows


class RECMoE2024Evaluator:
    """Same-checkpoint paired evaluation over the complete 2024 validation set."""

    def __init__(
        self,
        runtime: RECMoEInferenceRuntime,
        corpus: Any,
        output_root: str | Path,
        *,
        batch_size: int = 16,
        bootstrap_seed: int = 20240813,
        bootstrap_resamples: int = 2000,
        block_days: int = 7,
        expected_validation_days: int = 364,
    ) -> None:
        if not isinstance(runtime, RECMoEInferenceRuntime):
            raise RECMoEEvaluationError("runtime must be RECMoEInferenceRuntime")
        if not callable(getattr(corpus, "batches", None)):
            raise RECMoEEvaluationError("corpus must expose validation batches")
        if batch_size < 1 or expected_validation_days < 2:
            raise RECMoEEvaluationError("evaluation batch/day counts must be positive")
        self.runtime = runtime
        self.model = runtime.model
        self.corpus = corpus
        self.output_root = Path(output_root).resolve()
        self.batch_size = int(batch_size)
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.block_days = int(block_days)
        self.expected_validation_days = int(expected_validation_days)
        self.cost_scale: float | None = None
        self.r_checkpoint_path, self.r_checkpoint = self._eligible_best_checkpoint()
        self.realized_replay = RealizedHistoricalReplay.from_model(self.model)

    def _eligible_best_checkpoint(self) -> tuple[Path, Mapping[str, Any]]:
        value = self.runtime.checkpoint_load.get("r_checkpoint_path")
        if not value:
            raise RECMoEEvaluationError("2024 evaluation requires one R best checkpoint")
        path = Path(str(value)).resolve()
        if path.name != "best_decision_value.pt" or not path.is_file():
            raise RECMoEEvaluationError(
                "2024 evaluation accepts only best_decision_value.pt"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise RECMoEEvaluationError("R best checkpoint is malformed")
        metrics = payload.get("metrics")
        required = {
            "schema": "dcvpp.r_ecmoe.runner.v1",
            "phase": "r_ecmoe",
            "locked_2025_opened": False,
            "external_solver": False,
            "candidate_search": False,
            "hash_computed": False,
        }
        if any(payload.get(name) != expected for name, expected in required.items()):
            raise RECMoEEvaluationError("R best checkpoint violates evaluation lineage")
        if (
            not isinstance(metrics, Mapping)
            or metrics.get("validation_scope") != "full"
            or metrics.get("selection_eligible") is not True
        ):
            raise RECMoEEvaluationError("R best is not an eligible full validation")
        if metrics.get("outcome_basis") != "realized_historical_replay":
            raise RECMoEEvaluationError(
                "R best lacks realized historical replay outcome evidence"
            )
        for name in ("empirical_cvar95", "worst_origin_sla"):
            value = metrics.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RECMoEEvaluationError(
                    f"R best lacks finite realized replay metric {name}"
                )
        return path, payload

    def _cost_scale(self, batch: Mapping[str, Any]) -> float:
        if self.cost_scale is not None:
            return self.cost_scale
        value = batch.get("train_frozen_baseline_daily_cost")
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise RECMoEEvaluationError("validation batch omitted train-only cost scale")
        numeric = float(value.detach().float().cpu())
        expected = self.r_checkpoint.get("train_frozen_cost_scale")
        if (
            not math.isfinite(numeric)
            or numeric <= 0.0
            or expected is None
            or not math.isclose(numeric, float(expected), rel_tol=1.0e-7, abs_tol=1.0e-7)
        ):
            raise RECMoEEvaluationError("evaluation cost scale differs from R training")
        self.cost_scale = numeric
        return numeric

    def _timed_encode(
        self,
        packet: Any,
        da_anchor: Tensor,
        da_cap: Tensor,
    ) -> tuple[W1ExpertPacket, float]:
        before = int(self.model.w1_backbone_calls)
        _synchronize(self.runtime.device)
        started = time.perf_counter()
        with _autocast(self.runtime):
            expert_packet = self.model.encode_expert_context(
                packet, da_anchor=da_anchor, da_residual_cap=da_cap
            )
        _synchronize(self.runtime.device)
        elapsed = (time.perf_counter() - started) * 1000.0
        if int(self.model.w1_backbone_calls) - before != 1:
            raise RECMoEEvaluationError("each evaluation batch must encode W1 once")
        return expert_packet, elapsed

    def _timed_variant(
        self,
        packet: W1ExpertPacket,
        risk: RiskConditionPacket,
        variant: EvaluationVariant,
    ) -> tuple[RECMoEOutput, float]:
        _synchronize(self.runtime.device)
        started = time.perf_counter()
        with _autocast(self.runtime):
            execution = _execute_variant(self.model, packet, risk, variant)
        _synchronize(self.runtime.device)
        return execution, (time.perf_counter() - started) * 1000.0

    def _aggregate_days(
        self, decision_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in decision_rows:
            grouped.setdefault(
                (str(row["operating_date"]), str(row["variant"])), []
            ).append(row)
        metrics = (
            "active_horizon",
            "realized_operating_cost",
            "cost_normalization_denominator",
            "normalized_operating_cost",
            "deadline_expired_work",
            "unserved_energy",
            "action_total_variation",
            "battery_throughput",
            "compute_energy",
            "migration_energy",
            "pv_curtailed_energy",
            "grid_import_energy",
            "grid_export_energy",
            "grid_peak_import_ratio",
            "grid_peak_export_ratio",
            "grid_peak_ramp_ratio",
            "grid_rebound_ratio",
            "compute_concentration_hhi",
            "terminal_unfinished_work",
            "terminal_soc_fraction",
            "route_entropy",
            "route_gini",
            "tail_activation_rate",
            "tail_probability_mean",
            "tail_event_rate",
            "tail_probability_brier",
            "tail_calibration_gap",
            "equivalent_latency_ms_per_sample",
            *(f"route_load_{index}" for index in range(7)),
        )
        rows = []
        for (operating_date, variant), values in sorted(grouped.items()):
            if len(values) != len(EVALUATION_DECISIONS):
                raise RECMoEEvaluationError(
                    f"{operating_date}/{variant} does not have six decisions"
                )
            row: dict[str, Any] = {
                "operating_date": operating_date,
                "variant": variant,
                "comparison_design": "same_checkpoint_intervention",
                "separately_retrained": False,
                "intervention_description": next(
                    item.description
                    for item in REGISTERED_VARIANTS
                    if item.name == variant
                ),
                "decision_count": len(values),
                "outcome_basis": "realized_historical_replay",
                "tail_calibration_target": str(
                    values[0]["tail_calibration_target"]
                ),
                "hard_residual_max": max(
                    float(value["hard_residual_max"]) for value in values
                ),
            }
            for metric in metrics:
                row[metric] = sum(float(value[metric]) for value in values) / float(
                    len(values)
                )
            rows.append(row)
        return rows

    def _summaries(
        self, daily_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        by_variant: dict[str, list[Mapping[str, Any]]] = {}
        for row in daily_rows:
            by_variant.setdefault(str(row["variant"]), []).append(row)
        rows = []
        for variant in (value.name for value in REGISTERED_VARIANTS):
            values = sorted(by_variant.get(variant, ()), key=lambda item: item["operating_date"])
            if len(values) != self.expected_validation_days:
                raise RECMoEEvaluationError(f"variant {variant} omitted validation days")
            costs = [float(value["normalized_operating_cost"]) for value in values]
            row = {
                "variant": variant,
                "comparison_design": "same_checkpoint_intervention",
                "separately_retrained": False,
                "intervention_description": next(
                    item.description
                    for item in REGISTERED_VARIANTS
                    if item.name == variant
                ),
                "days": len(values),
                "outcome_basis": "realized_historical_replay",
                "mean_realized_operating_cost": sum(
                    float(value["realized_operating_cost"]) for value in values
                )
                / len(values),
                "mean_normalized_operating_cost": sum(costs) / len(costs),
                "median_normalized_operating_cost": _percentile(costs, 0.50),
                "cvar95_normalized_operating_cost": _cvar95(costs),
                "mean_deadline_expired_work": sum(
                    float(value["deadline_expired_work"]) for value in values
                )
                / len(values),
                "mean_unserved_energy": sum(
                    float(value["unserved_energy"]) for value in values
                )
                / len(values),
                "maximum_hard_residual": max(
                    float(value["hard_residual_max"]) for value in values
                ),
                "mean_action_total_variation": sum(
                    float(value["action_total_variation"]) for value in values
                )
                / len(values),
                "mean_battery_throughput": sum(
                    float(value["battery_throughput"]) for value in values
                )
                / len(values),
                "mean_compute_energy": sum(
                    float(value["compute_energy"]) for value in values
                )
                / len(values),
                "mean_migration_energy": sum(
                    float(value["migration_energy"]) for value in values
                )
                / len(values),
                "mean_pv_curtailed_energy": sum(
                    float(value["pv_curtailed_energy"]) for value in values
                )
                / len(values),
                "mean_grid_import_energy": sum(
                    float(value["grid_import_energy"]) for value in values
                )
                / len(values),
                "mean_grid_export_energy": sum(
                    float(value["grid_export_energy"]) for value in values
                )
                / len(values),
                "mean_grid_peak_import_ratio": sum(
                    float(value["grid_peak_import_ratio"]) for value in values
                )
                / len(values),
                "mean_grid_peak_export_ratio": sum(
                    float(value["grid_peak_export_ratio"]) for value in values
                )
                / len(values),
                "mean_grid_peak_ramp_ratio": sum(
                    float(value["grid_peak_ramp_ratio"]) for value in values
                )
                / len(values),
                "mean_grid_rebound_ratio": sum(
                    float(value["grid_rebound_ratio"]) for value in values
                )
                / len(values),
                "mean_compute_concentration_hhi": sum(
                    float(value["compute_concentration_hhi"]) for value in values
                )
                / len(values),
                "mean_terminal_unfinished_work": sum(
                    float(value["terminal_unfinished_work"]) for value in values
                )
                / len(values),
                "mean_terminal_soc_fraction": sum(
                    float(value["terminal_soc_fraction"]) for value in values
                )
                / len(values),
                "mean_route_entropy": sum(
                    float(value["route_entropy"]) for value in values
                )
                / len(values),
                "mean_route_gini": sum(
                    float(value["route_gini"]) for value in values
                )
                / len(values),
                "mean_tail_activation_rate": sum(
                    float(value["tail_activation_rate"]) for value in values
                )
                / len(values),
                "mean_tail_probability": sum(
                    float(value["tail_probability_mean"]) for value in values
                )
                / len(values),
                "mean_tail_event_rate": sum(
                    float(value["tail_event_rate"]) for value in values
                )
                / len(values),
                "mean_tail_probability_brier": sum(
                    float(value["tail_probability_brier"]) for value in values
                )
                / len(values),
                "mean_tail_calibration_gap": sum(
                    float(value["tail_calibration_gap"]) for value in values
                )
                / len(values),
                "tail_calibration_target": str(
                    values[0]["tail_calibration_target"]
                ),
                "latency_mean_ms_per_sample": sum(
                    float(value["equivalent_latency_ms_per_sample"])
                    for value in values
                )
                / len(values),
                "latency_p95_ms_per_sample": _percentile(
                    [
                        float(value["equivalent_latency_ms_per_sample"])
                        for value in values
                    ],
                    0.95,
                ),
            }
            for expert in range(7):
                row[f"mean_route_load_{expert}"] = sum(
                    float(value[f"route_load_{expert}"]) for value in values
                ) / len(values)
            rows.append(row)
        return rows

    def _paired_comparisons(
        self, daily_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        by_variant = {
            variant.name: {
                str(row["operating_date"]): row
                for row in daily_rows
                if row["variant"] == variant.name
            }
            for variant in REGISTERED_VARIANTS
        }
        dates = sorted(by_variant["top2"])
        blocks = moving_block_bootstrap_indices(
            len(dates),
            block_days=self.block_days,
            resamples=self.bootstrap_resamples,
            seed=self.bootstrap_seed,
        )
        tolerance = float(
            self.r_checkpoint.get("resolved_config", {}).get(
                "hard_residual_tolerance", 1.0e-5
            )
        )
        rows = []
        for comparator in (
            variant.name for variant in REGISTERED_VARIANTS if variant.name != "top2"
        ):
            top2 = by_variant["top2"]
            other = by_variant[comparator]
            if set(top2) != set(other) or len(top2) != self.expected_validation_days:
                raise RECMoEEvaluationError("paired variants do not share exact dates")
            deltas = [
                float(top2[day]["normalized_operating_cost"])
                - float(other[day]["normalized_operating_cost"])
                for day in dates
            ]
            bootstrap = [
                sum(deltas[index] for index in sample) / float(len(sample))
                for sample in blocks
            ]
            rows.append(
                {
                    "primary_variant": "top2",
                    "comparator": comparator,
                    "comparison_design": "same_checkpoint_intervention",
                    "separately_retrained": False,
                    "outcome_basis": "realized_historical_replay",
                    "days": len(dates),
                    "cost_delta_top2_minus_comparator": sum(deltas) / len(deltas),
                    "cost_delta_ci95_lower": _percentile(bootstrap, 0.025),
                    "cost_delta_ci95_upper": _percentile(bootstrap, 0.975),
                    "top2_daily_cost_win_rate": sum(value < 0.0 for value in deltas)
                    / len(deltas),
                    "deadline_delta": sum(
                        float(top2[day]["deadline_expired_work"])
                        - float(other[day]["deadline_expired_work"])
                        for day in dates
                    )
                    / len(dates),
                    "unserved_delta": sum(
                        float(top2[day]["unserved_energy"])
                        - float(other[day]["unserved_energy"])
                        for day in dates
                    )
                    / len(dates),
                    "top2_hard_violation_rate": sum(
                        float(top2[day]["hard_residual_max"]) > tolerance
                        for day in dates
                    )
                    / len(dates),
                    "comparator_hard_violation_rate": sum(
                        float(other[day]["hard_residual_max"]) > tolerance
                        for day in dates
                    )
                    / len(dates),
                    "bootstrap_block_days": self.block_days,
                    "bootstrap_resamples": self.bootstrap_resamples,
                    "bootstrap_seed": self.bootstrap_seed,
                }
            )
        return rows

    def run(self) -> Mapping[str, Any]:
        _validate_variant_registry()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        decision_rows: list[dict[str, Any]] = []
        reference_dates: tuple[str, ...] | None = None
        w1_encode_calls = 0
        variant_decision_calls = 0
        realized_replay_calls = 0
        with torch.inference_mode():
            for stage_id, origin, horizon in EVALUATION_DECISIONS:
                dates_for_decision: list[str] = []
                for batch in self.corpus.batches(
                    "validation",
                    self.batch_size,
                    device=self.runtime.device,
                    shuffle_seed=None,
                    max_days=None,
                    origin=origin,
                    require_h96=False,
                    include_next_day=False,
                ):
                    dates = tuple(str(value) for value in batch.get("operating_dates", ()))
                    packet = batch.get("world_input")
                    if (
                        not dates
                        or packet is None
                        or len(dates) != int(packet.fast_values.shape[0])
                    ):
                        raise RECMoEEvaluationError(
                            "validation batch omitted operating dates or world_input"
                        )
                    if any(date.fromisoformat(value).year != 2024 for value in dates):
                        raise RECMoEEvaluationError("evaluation attempted to open non-2024 data")
                    if not bool((packet.origin_index == origin).all()):
                        raise RECMoEEvaluationError("validation packet origin mismatch")
                    dates_for_decision.extend(dates)
                    da_anchor = batch.get("forecast_anchors", {}).get("da")
                    da_cap = batch.get("forecast_residual_caps", {}).get("da")
                    if not isinstance(da_anchor, Tensor) or not isinstance(da_cap, Tensor):
                        raise RECMoEEvaluationError("validation batch omitted DA anchors")
                    risk = RiskConditionPacket.defaults(
                        len(dates),
                        device=self.runtime.device,
                        horizon=horizon,
                        stage=stage_id,
                    )
                    expert_packet, encode_ms = self._timed_encode(
                        packet, da_anchor, da_cap
                    )
                    w1_encode_calls += 1
                    # Every same-checkpoint variant finishes routing and raw-action
                    # generation before the batch's realized target is admitted.
                    decisions: list[
                        tuple[EvaluationVariant, RECMoEOutput, float]
                    ] = []
                    for variant in REGISTERED_VARIANTS:
                        decision, decision_ms = self._timed_variant(
                            expert_packet, risk, variant
                        )
                        variant_decision_calls += 1
                        decisions.append((variant, decision, decision_ms))
                    scale = self._cost_scale(batch)
                    queue_event_target = batch.get("queue_risk_event_target")
                    queue_event_mask = batch.get("queue_risk_event_mask")
                    if queue_event_target is not None and not isinstance(
                        queue_event_target, Tensor
                    ):
                        raise RECMoEEvaluationError(
                            "queue_risk_event_target must be a Tensor when present"
                        )
                    if queue_event_mask is not None and not isinstance(
                        queue_event_mask, Tensor
                    ):
                        raise RECMoEEvaluationError(
                            "queue_risk_event_mask must be a Tensor when present"
                        )
                    for variant, decision, decision_ms in decisions:
                        realized = self.realized_replay(
                            decision,
                            batch=batch,
                            risk=risk,
                            source_split="validation",
                        )
                        realized_replay_calls += 1
                        decision_rows.extend(
                            _execution_rows(
                                decision,
                                realized,
                                variant=variant,
                                dates=dates,
                                stage_id=stage_id,
                                origin=origin,
                                horizon=horizon,
                                cost_scale=scale,
                                equivalent_latency_ms_per_sample=(
                                    encode_ms + decision_ms
                                )
                                / len(dates),
                                queue_event_target=queue_event_target,
                                queue_event_mask=queue_event_mask,
                            )
                        )
                if (
                    len(dates_for_decision) != self.expected_validation_days
                    or len(set(dates_for_decision)) != self.expected_validation_days
                ):
                    raise RECMoEEvaluationError(
                        "evaluation did not consume the complete unique validation year"
                    )
                ordered_dates = tuple(dates_for_decision)
                if reference_dates is None:
                    reference_dates = ordered_dates
                elif ordered_dates != reference_dates:
                    raise RECMoEEvaluationError(
                        "DA/RT decisions do not share identical ordered 2024 dates"
                    )
        daily_rows = self._aggregate_days(decision_rows)
        summaries = self._summaries(daily_rows)
        comparisons = self._paired_comparisons(daily_rows)
        _atomic_csv(self.output_root / "per_day_variant.csv", daily_rows)
        _atomic_csv(self.output_root / "variant_summary.csv", summaries)
        _atomic_csv(self.output_root / "paired_comparisons.csv", comparisons)
        meta = {
            "schema": "dcvpp.r_ecmoe.evaluation_2024.v1",
            "recorded_at_utc": _utc_now(),
            "split": "validation",
            "year": 2024,
            "validation_days": self.expected_validation_days,
            "decisions_per_day": len(EVALUATION_DECISIONS),
            "variants": [variant.name for variant in REGISTERED_VARIANTS],
            "variant_definitions": [
                variant.evidence_definition() for variant in REGISTERED_VARIANTS
            ],
            "comparison_design": "same_checkpoint_intervention_only",
            "shared_checkpoint_for_all_variants": True,
            "separately_retrained_variant_claims": False,
            "w1_packet_reused_across_variants": True,
            "paired_daily_rows": len(daily_rows),
            "w1_encode_calls": w1_encode_calls,
            "variant_decision_calls": variant_decision_calls,
            "realized_replay_calls": realized_replay_calls,
            "decision_variant_rows_before_daily_aggregation": len(decision_rows),
            "batch_size": self.batch_size,
            "best_checkpoint": str(self.r_checkpoint_path),
            "w1_checkpoint": str(self.runtime.checkpoint_path),
            "run_id": self.r_checkpoint.get("run_id"),
            "best_validation_scope": "full",
            "best_selection_eligible": True,
            "outcome_basis": "realized_historical_replay",
            "cost_normalization": (
                "train_frozen_baseline_daily_cost * active_horizon / 96"
            ),
            "behavior_metric_definitions": {
                "grid_peak_import_ratio": (
                    "maximum active grid import divided elementwise by the "
                    "registered import limit"
                ),
                "grid_peak_ramp_ratio": (
                    "maximum active one-step absolute grid ramp divided by "
                    "the registered import-plus-export range"
                ),
                "grid_rebound_ratio": (
                    "maximum V-shaped aggregate-import excursion, using the "
                    "smaller of the down/up legs, divided by total import limit"
                ),
                "compute_concentration_hhi": (
                    "mean active-time Herfindahl index of four-site compute shares"
                ),
                "terminal_unfinished_work": (
                    "continuous-cohort terminal queue sum; not task-level integer count"
                ),
                "route_gini": "Gini coefficient of seven-expert active-horizon load",
                "tail_probability_brier": (
                    "Brier score of final-W1 tail probability against the registered "
                    "causal queue-event target when present"
                ),
                "tail_calibration_gap": (
                    "absolute gap between mean tail probability and event prevalence"
                ),
            },
            "realized_replay_excluded_from_inference_latency": True,
            "bootstrap": {
                "method": "circular_moving_block_over_daily_paired_differences",
                "block_days": self.block_days,
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
            },
            "locked_2025_opened": False,
            "external_solver": False,
            "candidate_search": False,
            "hash_computed": False,
        }
        ablation = {
            "schema": "dcvpp.r_ecmoe.ablation_summary.v1",
            "primary_variant": "top2",
            "safe_baseline": "safe_tensor",
            "comparison_design": "same_checkpoint_intervention_only",
            "shared_checkpoint_for_all_variants": True,
            "separately_retrained_variant_claims": False,
            "variant_definitions": [
                variant.evidence_definition() for variant in REGISTERED_VARIANTS
            ],
            "outcome_basis": "realized_historical_replay",
            "variant_summaries": summaries,
            "paired_comparisons": comparisons,
            "interpretation": (
                "negative top2-minus-comparator cost favors the registered Top-2 model"
            ),
            "locked_2025_opened": False,
            "external_solver": False,
            "candidate_search": False,
            "hash_computed": False,
        }
        _atomic_json(self.output_root / "ablation_summary.json", ablation)
        _atomic_json(self.output_root / "evaluation_2024_meta.json", meta)
        return meta


__all__ = [
    "EVALUATION_DECISIONS",
    "EvaluationVariant",
    "HARD_RESIDUAL_EXCLUSIONS",
    "RECMoE2024Evaluator",
    "RECMoEEvaluationError",
    "REGISTERED_VARIANTS",
    "moving_block_bootstrap_indices",
]
