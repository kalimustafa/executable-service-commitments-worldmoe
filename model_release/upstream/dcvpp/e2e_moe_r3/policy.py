"""Causal neural day-ahead and rolling real-time policy heads for R3."""

from __future__ import annotations

from collections.abc import Mapping
import warnings

import torch
from torch import Tensor, nn


def _site_tensor(
    value: Tensor,
    *,
    name: str,
    batch: int,
    sites: int,
    reference: Tensor,
) -> Tensor:
    """Return a finite ``[batch, sites]`` state tensor.

    Aggregate queue and deadline signals may be supplied as ``[batch]`` or
    ``[batch, 1]`` and are broadcast to the registered sites.  Site-resolved
    values must already have the exact site width.
    """

    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 1 and result.shape[0] == batch:
        result = result.unsqueeze(-1)
    if result.shape == (batch, 1):
        result = result.expand(batch, sites)
    if result.shape != (batch, sites):
        raise ValueError(
            f"{name} must have shape [batch], [batch, 1], or "
            f"[batch, {sites}], got {tuple(result.shape)}"
        )
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _signed_log1p(value: Tensor) -> Tensor:
    """Stable scale compression that preserves the sign of physical state."""

    return value.sign() * torch.log1p(value.abs())


class NeuralRollingPolicy(nn.Module):
    """Emit state-conditioned actions for exactly one rolling origin.

    ``forward_origin`` is the paper-facing interface.  It requires the current
    origin, SOC, backlog, deadline pressure, previous executed action, and an
    explicit revelation mask.  The mask is rejected if it exposes any period
    after the current origin.  Day-ahead decisions are issued only at origin
    zero; callers retain that schedule for later origins.

    The head uses a fixed number of differentiable refinement steps.  Its
    outputs are bounded neural decisions; an external deterministic constraint
    shield remains responsible for enforcing the full physical contract.

    ``forward`` is retained as a compatibility seam for representation and
    scope tests that do not yet construct rolling state.  It does not represent
    the formal rolling policy.
    """

    action_names = (
        "day_ahead",
        "grid",
        "real_time",
        "battery",
        "compute",
        "migration",
    )

    def __init__(
        self,
        hidden_dim: int,
        *,
        horizon: int = 96,
        action_dim: int = 1,
        num_sites: int = 4,
        max_grid_action: float = 1.0,
        max_battery_action: float = 1.0,
        max_compute_action: float = 1.0,
        max_migration_action: float = 1.0,
        origin_embedding_dim: int = 16,
        context_hidden_dim: int | None = None,
        refinement_steps: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or horizon < 1 or action_dim < 1 or num_sites < 1:
            raise ValueError("policy dimensions must be positive")
        if origin_embedding_dim < 1:
            raise ValueError("origin_embedding_dim must be positive")
        if refinement_steps < 1:
            raise ValueError("refinement_steps must be at least one")
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.num_sites = int(num_sites)
        self.max_grid_action = float(max_grid_action)
        self.max_battery_action = float(max_battery_action)
        self.max_compute_action = float(max_compute_action)
        self.max_migration_action = float(max_migration_action)
        self.origin_embedding_dim = int(origin_embedding_dim)
        self.context_hidden_dim = int(context_hidden_dim or hidden_dim)
        self.refinement_steps = int(refinement_steps)

        # Previous actions exclude the day-ahead schedule, which is issued once
        # and retained by the rolling runtime: grid, battery, compute, migration.
        self.previous_action_dim = (
            3 * self.num_sites + self.num_sites * self.num_sites
        )
        # SOC, backlog and deadline pressure are site-resolved.  Three mask
        # summaries retain origin/revelation information without admitting a
        # future observation vector.
        self.rolling_state_feature_dim = (
            3 * self.num_sites + self.previous_action_dim + 3
        )
        self.current_action_dim = (
            3 * self.num_sites
            + self.num_sites * self.num_sites
            + self.num_sites
        )

        self.origin_embedding = nn.Embedding(self.horizon, self.origin_embedding_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(
                self.rolling_state_feature_dim + self.origin_embedding_dim,
                self.context_hidden_dim,
            ),
            nn.LayerNorm(self.context_hidden_dim),
            nn.GELU(),
            nn.Linear(self.context_hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        # Trained with the policy and frozen during A1.  Frozen parameters do
        # not sever autograd: decision losses still propagate through this
        # bridge into the A1-permitted DART point/calibration/scenario heads.
        self.dart_to_policy_bridge = nn.Sequential(
            nn.Linear(3 * self.horizon, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # This projection lets the causal state condition the existing Top-2
        # MoE routing before the decision latent is formed.
        self.moe_context_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        policy_hidden = max(hidden_dim, 32)
        self.policy_core = nn.Sequential(
            nn.Linear(hidden_dim * 2, policy_hidden),
            nn.LayerNorm(policy_hidden),
            nn.GELU(),
        )
        self.joint_action_head = nn.Linear(policy_hidden, self.current_action_dim)
        self.refinement = nn.Sequential(
            nn.Linear(policy_hidden + self.current_action_dim, policy_hidden),
            nn.LayerNorm(policy_hidden),
            nn.GELU(),
            nn.Linear(policy_hidden, self.current_action_dim),
        )
        self.origin_day_ahead_head = nn.Linear(
            policy_hidden + self.current_action_dim,
            self.num_sites * self.horizon,
        )

        # Compatibility-only heads preserve the legacy tensor shapes used by
        # existing training losses.  The rolling runtime uses forward_origin.
        self.day_ahead_head = nn.Linear(
            policy_hidden, self.horizon * self.action_dim
        )
        self.real_time_head = nn.Linear(policy_hidden, self.action_dim)
        self.battery_head = nn.Linear(policy_hidden, self.action_dim)
        self.compute_head = nn.Linear(policy_hidden, self.num_sites)
        self.migration_head = nn.Linear(
            policy_hidden, self.num_sites * self.num_sites
        )

    def _previous_action_tensor(
        self,
        previous_action: Mapping[str, Tensor] | Tensor,
        *,
        batch: int,
        reference: Tensor,
    ) -> Tensor:
        if isinstance(previous_action, Mapping):
            grid_value = previous_action.get("grid")
            if grid_value is None:
                grid_value = previous_action.get("real_time")
            if grid_value is None:
                raise KeyError("previous_action requires 'grid' or 'real_time'")
            parts = [
                _site_tensor(
                    grid_value,
                    name="previous_action.grid",
                    batch=batch,
                    sites=self.num_sites,
                    reference=reference,
                ),
                _site_tensor(
                    previous_action["battery"],
                    name="previous_action.battery",
                    batch=batch,
                    sites=self.num_sites,
                    reference=reference,
                ),
                _site_tensor(
                    previous_action["compute"],
                    name="previous_action.compute",
                    batch=batch,
                    sites=self.num_sites,
                    reference=reference,
                ),
            ]
            migration = torch.as_tensor(
                previous_action["migration"],
                device=reference.device,
                dtype=reference.dtype,
            )
            expected = (batch, self.num_sites, self.num_sites)
            if migration.shape != expected:
                raise ValueError(
                    "previous_action.migration must have shape "
                    f"{expected}, got {tuple(migration.shape)}"
                )
            if not torch.isfinite(migration).all():
                raise ValueError("previous_action.migration contains non-finite values")
            result = torch.cat((*parts, migration.flatten(start_dim=1)), dim=-1)
        else:
            result = torch.as_tensor(
                previous_action, device=reference.device, dtype=reference.dtype
            )
            expected = (batch, self.previous_action_dim)
            if result.shape != expected:
                raise ValueError(
                    f"previous_action must have shape {expected}, got "
                    f"{tuple(result.shape)}"
                )
            if not torch.isfinite(result).all():
                raise ValueError("previous_action contains non-finite values")
        return result

    def encode_origin_context(
        self,
        latent: Tensor,
        *,
        origin: Tensor | int,
        current_soc: Tensor,
        backlog: Tensor,
        deadline_pressure: Tensor,
        previous_action: Mapping[str, Tensor] | Tensor,
        revealed_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Validate and encode the complete causal rolling state.

        Returns ``(context, origin_index, causal_mask)``.  No value located
        after ``origin`` can pass the mask validation.
        """

        if latent.ndim != 2 or latent.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"latent must have shape [batch, {self.hidden_dim}], got "
                f"{tuple(latent.shape)}"
            )
        batch = latent.shape[0]
        origin_index = torch.as_tensor(origin, device=latent.device)
        if origin_index.ndim == 0:
            origin_index = origin_index.expand(batch)
        if origin_index.shape != (batch,):
            raise ValueError(
                f"origin must be scalar or shape [batch], got {tuple(origin_index.shape)}"
            )
        if origin_index.dtype.is_floating_point:
            if not torch.equal(origin_index, origin_index.round()):
                raise ValueError("origin must contain integer indices")
        origin_index = origin_index.to(dtype=torch.long)
        if ((origin_index < 0) | (origin_index >= self.horizon)).any():
            raise ValueError(f"origin must lie in [0, {self.horizon - 1}]")

        mask = torch.as_tensor(revealed_mask, device=latent.device)
        if mask.shape != (batch, self.horizon):
            raise ValueError(
                "revealed_mask must have shape "
                f"{(batch, self.horizon)}, got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("revealed_mask must be boolean or binary")
            mask = mask.to(dtype=torch.bool)
        periods = torch.arange(self.horizon, device=latent.device).unsqueeze(0)
        future = periods > origin_index.unsqueeze(1)
        if torch.any(mask & future):
            raise ValueError("revealed_mask exposes information after the current origin")
        current_revealed = mask.gather(1, origin_index.unsqueeze(1)).squeeze(1)
        if not torch.all(current_revealed):
            raise ValueError("revealed_mask must include the current origin")

        soc = _site_tensor(
            current_soc,
            name="current_soc",
            batch=batch,
            sites=self.num_sites,
            reference=latent,
        )
        if (soc < 0).any():
            raise ValueError("current_soc must be nonnegative")
        queue = _site_tensor(
            backlog,
            name="backlog",
            batch=batch,
            sites=self.num_sites,
            reference=latent,
        )
        if (queue < 0).any():
            raise ValueError("backlog must be nonnegative")
        pressure = _site_tensor(
            deadline_pressure,
            name="deadline_pressure",
            batch=batch,
            sites=self.num_sites,
            reference=latent,
        )
        if (pressure < 0).any():
            raise ValueError("deadline_pressure must be nonnegative")
        previous = self._previous_action_tensor(
            previous_action, batch=batch, reference=latent
        )

        denominator = float(max(self.horizon - 1, 1))
        origin_fraction = origin_index.to(latent.dtype) / denominator
        revealed_fraction = mask.to(latent.dtype).mean(dim=1)
        mask_summary = torch.stack(
            (
                origin_fraction,
                revealed_fraction,
                current_revealed.to(latent.dtype),
            ),
            dim=-1,
        )
        rolling_state = torch.cat(
            (
                _signed_log1p(soc),
                _signed_log1p(queue),
                _signed_log1p(pressure),
                _signed_log1p(previous),
                mask_summary,
            ),
            dim=-1,
        )
        context = self.context_encoder(
            torch.cat((rolling_state, self.origin_embedding(origin_index)), dim=-1)
        )
        return context, origin_index, mask

    def condition_moe_input(self, fused_latent: Tensor, context: Tensor) -> Tensor:
        """Condition the upstream Top-2 routing input on current rolling state."""

        if fused_latent.shape != context.shape:
            raise ValueError(
                "fused_latent and context must share [batch, hidden] shape, got "
                f"{tuple(fused_latent.shape)} and {tuple(context.shape)}"
            )
        return fused_latent + self.moe_context_projection(context)

    def encode_dart_signal(
        self,
        dart_forecast: Tensor | None,
        dart_scenarios: Tensor | None,
        scenario_weights: Tensor | None,
        *,
        reference: Tensor,
    ) -> Tensor:
        """Bridge DART point, expectation, and upper tail into policy space."""

        batch = reference.shape[0]
        if dart_forecast is None and dart_scenarios is None:
            return reference.new_zeros((batch, self.hidden_dim))
        if dart_forecast is None or dart_scenarios is None or scenario_weights is None:
            raise ValueError(
                "dart_forecast, dart_scenarios, and scenario_weights must be "
                "provided together"
            )
        point = torch.as_tensor(
            dart_forecast, device=reference.device, dtype=reference.dtype
        )
        particles = torch.as_tensor(
            dart_scenarios, device=reference.device, dtype=reference.dtype
        )
        weights = torch.as_tensor(
            scenario_weights, device=reference.device, dtype=reference.dtype
        )
        if point.shape != (batch, self.horizon):
            raise ValueError(
                f"dart_forecast must have shape {(batch, self.horizon)}, got "
                f"{tuple(point.shape)}"
            )
        if particles.ndim != 3 or particles.shape[0] != batch or particles.shape[2] != self.horizon:
            raise ValueError(
                "dart_scenarios must have shape [batch, scenarios, horizon], got "
                f"{tuple(particles.shape)}"
            )
        if weights.shape != particles.shape[:2]:
            raise ValueError(
                f"scenario_weights must have shape {tuple(particles.shape[:2])}, "
                f"got {tuple(weights.shape)}"
            )
        normalized = weights.clamp_min(0.0)
        normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(1e-8)
        expectation = (particles * normalized.unsqueeze(-1)).sum(dim=1)
        # A smooth upper-tail proxy preserves gradients to every particle and
        # avoids sorting discontinuities in the bridge.
        tail_weights = torch.softmax(particles / 10.0, dim=1)
        upper_tail = (particles * tail_weights).sum(dim=1)
        features = torch.cat(
            (
                _signed_log1p(point),
                _signed_log1p(expectation),
                _signed_log1p(upper_tail),
            ),
            dim=-1,
        )
        return self.dart_to_policy_bridge(features)

    def _refined_raw_action(self, hidden: Tensor) -> Tensor:
        raw = self.joint_action_head(hidden)
        for _ in range(self.refinement_steps - 1):
            correction = self.refinement(torch.cat((hidden, torch.tanh(raw)), dim=-1))
            raw = raw + correction / float(self.refinement_steps)
        return raw

    def _decode_joint_action(
        self,
        raw: Tensor,
        hidden: Tensor,
        origin_index: Tensor,
    ) -> dict[str, Tensor]:
        batch = raw.shape[0]
        z = self.num_sites
        cursor = 0
        grid_raw = raw[:, cursor : cursor + z]
        cursor += z
        battery_raw = raw[:, cursor : cursor + z]
        cursor += z
        compute_raw = raw[:, cursor : cursor + z]
        cursor += z
        migration_logits = raw[:, cursor : cursor + z * z].reshape(batch, z, z)
        cursor += z * z
        migration_gate = torch.sigmoid(raw[:, cursor : cursor + z])

        grid = torch.sigmoid(grid_raw) * self.max_grid_action
        # A single signed action constructively prevents simultaneous charge
        # and discharge.  The shield decomposes it into nonnegative flows.
        battery = torch.tanh(battery_raw) * self.max_battery_action
        compute = torch.sigmoid(compute_raw) * self.max_compute_action

        if z > 1:
            eye = torch.eye(z, device=raw.device, dtype=torch.bool).unsqueeze(0)
            masked_logits = migration_logits.masked_fill(eye, -torch.inf)
            migration = torch.softmax(masked_logits, dim=-1)
            migration = migration.masked_fill(eye, 0.0)
            migration = (
                migration
                * migration_gate.unsqueeze(-1)
                * self.max_migration_action
            )
        else:
            migration = raw.new_zeros((batch, 1, 1))

        day_ahead_raw = self.origin_day_ahead_head(
            torch.cat((hidden, raw), dim=-1)
        )
        day_ahead = torch.sigmoid(day_ahead_raw).reshape(
            batch, z, self.horizon
        ) * self.max_grid_action
        issue_mask = origin_index.eq(0)
        day_ahead = day_ahead * issue_mask[:, None, None].to(day_ahead.dtype)

        return {
            "day_ahead": day_ahead,
            "day_ahead_issue_mask": issue_mask,
            "grid": grid,
            "real_time": grid,
            "battery": battery,
            "battery_charge_mode": battery.ge(0).to(battery.dtype),
            "compute": compute,
            "migration": migration,
            "migration_logits": migration_logits,
            "migration_gate": migration_gate,
        }

    def forward_origin(
        self,
        latent: Tensor,
        *,
        origin: Tensor | int,
        current_soc: Tensor,
        backlog: Tensor,
        deadline_pressure: Tensor,
        previous_action: Mapping[str, Tensor] | Tensor,
        revealed_mask: Tensor,
        context: Tensor | None = None,
        dart_forecast: Tensor | None = None,
        dart_scenarios: Tensor | None = None,
        scenario_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return t0-only DA and current 4-site rolling actions.

        Shapes for the default four-site contract are ``day_ahead [B,4,96]``,
        ``grid/battery/compute [B,4]``, and ``migration [B,4,4]``.
        """

        encoded_context, origin_index, _ = self.encode_origin_context(
            latent,
            origin=origin,
            current_soc=current_soc,
            backlog=backlog,
            deadline_pressure=deadline_pressure,
            previous_action=previous_action,
            revealed_mask=revealed_mask,
        )
        if context is not None:
            if context.shape != encoded_context.shape:
                raise ValueError(
                    f"context must have shape {tuple(encoded_context.shape)}, got "
                    f"{tuple(context.shape)}"
                )
            # The model may pass the same already-computed context after using
            # it to condition MoE routing.  Re-encoding above deliberately
            # retains validation of every explicit causal input.
            encoded_context = context
        encoded_context = encoded_context + self.encode_dart_signal(
            dart_forecast,
            dart_scenarios,
            scenario_weights,
            reference=latent,
        )
        hidden = self.policy_core(torch.cat((latent, encoded_context), dim=-1))
        raw = self._refined_raw_action(hidden)
        return self._decode_joint_action(raw, hidden, origin_index)

    def forward(
        self,
        latent: Tensor,
        *,
        dart_forecast: Tensor | None = None,
        dart_scenarios: Tensor | None = None,
        scenario_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compatibility output for non-rolling training code.

        Formal execution must call :meth:`forward_origin` with explicit state.
        """

        if latent.ndim != 2 or latent.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"latent must have shape [batch, {self.hidden_dim}], got "
                f"{tuple(latent.shape)}"
            )
        batch = latent.shape[0]
        context = self.encode_dart_signal(
            dart_forecast,
            dart_scenarios,
            scenario_weights,
            reference=latent,
        )
        hidden = self.policy_core(torch.cat((latent, context), dim=-1))
        day_ahead = torch.tanh(self.day_ahead_head(hidden)).reshape(
            batch, self.horizon, self.action_dim
        )
        real_time = torch.tanh(self.real_time_head(hidden))
        battery = torch.tanh(self.battery_head(hidden))
        compute = torch.softmax(self.compute_head(hidden), dim=-1)
        migration_logits = self.migration_head(hidden).reshape(
            batch, self.num_sites, self.num_sites
        )
        if self.num_sites > 1:
            eye = torch.eye(
                self.num_sites, device=latent.device, dtype=torch.bool
            ).unsqueeze(0)
            migration = torch.softmax(
                migration_logits.masked_fill(eye, -torch.inf), dim=-1
            ).masked_fill(eye, 0.0)
        else:
            migration = migration_logits.new_zeros(migration_logits.shape)
        return {
            "day_ahead": day_ahead * self.max_grid_action,
            "real_time": real_time * self.max_grid_action,
            "battery": battery * self.max_battery_action,
            "compute": compute * self.max_compute_action,
            "migration": migration * self.max_migration_action,
            "migration_logits": migration_logits,
        }


class FeasibilityFallbackHead(nn.Module):
    """Predict policy validity and confidence in accepting its action."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        head_hidden = max(hidden_dim // 2, 8)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, latent: Tensor) -> dict[str, Tensor]:
        logits = self.network(latent)
        feasibility = torch.sigmoid(logits[..., 0])
        fallback_confidence = torch.sigmoid(logits[..., 1])
        return {
            "feasibility": feasibility,
            "feasibility_probability": feasibility,
            "fallback_confidence": fallback_confidence,
            "feasibility_logits": logits[..., 0],
            "fallback_logits": logits[..., 1],
        }


def should_use_safe_policy(
    *,
    tail_risk_probability: Tensor,
    fallback_confidence: Tensor,
    feasibility_probability: Tensor,
    expert_disagreement: Tensor | None = None,
    sla_stress: Tensor | None = None,
    ood_score: Tensor | None = None,
    risk_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> Tensor:
    """Return entries that should use the certified conservative policy path."""

    fallback = (
        (tail_risk_probability >= risk_threshold)
        | (fallback_confidence < confidence_threshold)
        | (feasibility_probability < confidence_threshold)
    )
    for value in (expert_disagreement, sla_stress, ood_score):
        if value is not None:
            fallback = fallback | (value >= risk_threshold)
    return fallback


def should_full_optimize(
    *,
    tail_risk_probability: Tensor,
    fallback_confidence: Tensor,
    feasibility_probability: Tensor,
    expert_disagreement: Tensor | None = None,
    sla_stress: Tensor | None = None,
    ood_score: Tensor | None = None,
    risk_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> Tensor:
    """Deprecated alias for :func:`should_use_safe_policy`.

    The name is retained only for old checkpoints/callers.  Solver-free R3 has
    no optimization fallback; true entries select the certified safe policy.
    """

    warnings.warn(
        "should_full_optimize is deprecated; use should_use_safe_policy",
        DeprecationWarning,
        stacklevel=2,
    )
    return should_use_safe_policy(
        tail_risk_probability=tail_risk_probability,
        fallback_confidence=fallback_confidence,
        feasibility_probability=feasibility_probability,
        expert_disagreement=expert_disagreement,
        sla_stress=sla_stress,
        ood_score=ood_score,
        risk_threshold=risk_threshold,
        confidence_threshold=confidence_threshold,
    )


__all__ = [
    "FeasibilityFallbackHead",
    "NeuralRollingPolicy",
    "should_full_optimize",
    "should_use_safe_policy",
]
