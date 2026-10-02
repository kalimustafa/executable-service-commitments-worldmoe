"""Unified VPP-WorldMoE public model API."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actions import ConstraintNativeActionDecoder
from .backbone import MultiScaleCausalWorldBackbone
from .config import WorldMoEConfig
from .dynamics import CompactActionConditionedWorldDynamics
from .observations import QUANTILE_LEVELS, WorldObservationQueries
from .packets import (
    WEATHER_SLOT,
    WorldActionPacket,
    WorldConstraintPacket,
    WorldInputPacket,
    WorldObservationPacket,
    WorldPlanPacket,
    WorldRolloutPacket,
    WorldStatePacket,
)
from .risk_critic import TwinMonotoneCostQuantileCritic
from .sparse_moe import EXPERT_ASSIGNMENT_SCHEMA
from .tokenizer import UnifiedTypedEventTokenizer


WORLD_PARAMETER_GROUPS = (
    "encoder",
    "exogenous_dynamics",
    "action_dynamics",
    "stochastic",
    "planner",
    "shared",
    "router",
    "physics",
)

LATENT_HORIZONS = (1, 4, 16, 96)

# The released W1 checkpoints predate R-ECMoE and therefore contain the
# candidate planner below the ``planner.`` namespace.  R-ECMoE never calls that
# module: it consumes the W1 sparse expert packet and emits one action directly.
# Keep this allow-list deliberately structural rather than accepting arbitrary
# ``planner.*`` keys, so a misspelled/new core key cannot be hidden as legacy
# compatibility state.
_LEGACY_PLANNER_ALIAS_PREFIXES = (
    ("planner.observation_queries.", "observation_queries."),
    ("planner.dynamics.", "dynamics."),
    ("planner.decoder.", "action_decoder."),
)
_LEGACY_PLANNER_UNUSED_EXACT_KEYS = frozenset(
    {
        "planner.candidate_embedding",
        "planner.time_embedding",
        "planner.constraint_multipliers_buffer",
        "planner.constraint_penalties_buffer",
    }
)
_LEGACY_PLANNER_UNUSED_PREFIXES = (
    "planner.reference_transition.",
    "planner.action_head.",
    "planner.day_ahead_head.",
    "planner.cvar_eta_head.",
    "planner.refinement.",
    "planner.refinement_norm.",
)


def load_plannerless_released_w1_state(
    model: "VPPWorldMoE",
    state: Mapping[str, Any],
) -> tuple[str, ...]:
    """Strictly load released W1 core state into a plannerless model.

    The only omitted checkpoint entries are the enumerated, unused candidate
    planner parameters and three aliases of already-present canonical modules.
    Alias tensors must equal their canonical checkpoint tensors exactly.  All
    remaining keys are loaded with ``strict=True``; any other missing or
    unexpected key is therefore fatal.
    """

    if bool(getattr(model, "legacy_planner_enabled", True)) or hasattr(
        model, "planner"
    ):
        raise ValueError("released-W1 plannerless loading requires planner disabled")
    if not isinstance(state, Mapping):
        raise TypeError("released W1 model_state must be a mapping")

    filtered: dict[str, Any] = {}
    omitted: list[str] = []
    for raw_name, value in state.items():
        if not isinstance(raw_name, str):
            raise RuntimeError("released W1 model_state keys must be strings")
        name = raw_name
        if not name.startswith("planner."):
            filtered[name] = value
            continue

        alias_target: str | None = None
        for legacy_prefix, canonical_prefix in _LEGACY_PLANNER_ALIAS_PREFIXES:
            if name.startswith(legacy_prefix):
                alias_target = canonical_prefix + name.removeprefix(legacy_prefix)
                break
        if alias_target is not None:
            if alias_target not in state:
                raise RuntimeError(
                    f"legacy planner alias {name!r} lacks canonical key "
                    f"{alias_target!r}"
                )
            canonical = state[alias_target]
            if not isinstance(value, Tensor) or not isinstance(canonical, Tensor):
                raise RuntimeError(
                    f"legacy planner alias {name!r} is not tensor state"
                )
            if not torch.equal(value, canonical):
                raise RuntimeError(
                    f"legacy planner alias {name!r} differs from canonical key "
                    f"{alias_target!r}"
                )
            omitted.append(name)
            continue

        if name in _LEGACY_PLANNER_UNUSED_EXACT_KEYS or name.startswith(
            _LEGACY_PLANNER_UNUSED_PREFIXES
        ):
            omitted.append(name)
            continue
        raise RuntimeError(f"unrecognized planner checkpoint key: {name!r}")

    incompatible = model.load_state_dict(filtered, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        # ``strict=True`` currently raises before this branch, but retain the
        # explicit invariant if PyTorch's return contract changes.
        raise RuntimeError(
            "plannerless released-W1 core load differs: "
            f"missing={tuple(incompatible.missing_keys)}, "
            f"unexpected={tuple(incompatible.unexpected_keys)}"
        )
    return tuple(omitted)


def _public_scale(independent: Tensor) -> Tensor:
    if independent.shape[-1] != 17:
        raise ValueError("independent public scale must end in 17")
    da = independent[..., :4]
    rt = independent[..., 4:8]
    dart = torch.sqrt(da.square() + rt.square()).clamp_min(1.0e-8)
    return torch.cat(
        (
            da,
            rt,
            dart,
            independent[..., 8:9],
            independent[..., 9:13],
            independent[..., 13:17],
        ),
        dim=-1,
    )


class VPPWorldMoE(nn.Module):
    """One causal world state spanning perception, uncertainty and action."""

    architecture_version = "causal_action_conditioned_world_moe_r5_v2"
    online_neural_only = True

    def __init__(
        self,
        config: WorldMoEConfig | None = None,
        *,
        include_legacy_planner: bool = True,
    ) -> None:
        super().__init__()
        self.config = WorldMoEConfig() if config is None else config
        self.tokenizer = UnifiedTypedEventTokenizer(self.config)
        self.backbone = MultiScaleCausalWorldBackbone(self.config)
        self.observation_queries = WorldObservationQueries(self.config)
        occurrence_prevalence = float(
            getattr(
                self.config,
                "queue_occurrence_train_prevalence",
                0.019967682659626007,
            )
        )
        if not 0.0 < occurrence_prevalence < 1.0:
            raise ValueError("queue occurrence train prevalence must lie in (0,1)")
        occurrence_output = self.observation_queries.queue_occurrence_query[-1]
        if not isinstance(occurrence_output, nn.Linear):
            raise RuntimeError("queue occurrence query must end in a Linear head")
        # Start from the frozen train prior instead of an arbitrary 0.5 event
        # probability.  A tiny nonzero weight keeps symmetry broken while the
        # rare-event proper score begins adapting the shared representation.
        nn.init.normal_(occurrence_output.weight, mean=0.0, std=1.0e-4)
        nn.init.constant_(
            occurrence_output.bias,
            math.log(occurrence_prevalence / (1.0 - occurrence_prevalence)),
        )
        self.dynamics = CompactActionConditionedWorldDynamics(
            self.config, self.observation_queries
        )
        self.action_decoder = ConstraintNativeActionDecoder(self.config)
        self.legacy_planner_enabled = bool(include_legacy_planner)
        if self.legacy_planner_enabled:
            # Lazy import keeps the candidate-search implementation out of the
            # formal R-ECMoE module graph and construction path.  The default
            # remains True so historical W0/W1/U0/P1/J1 callers are unchanged.
            from .planner import HierarchicalLatentPlanner

            self.planner = HierarchicalLatentPlanner(
                self.config,
                self.observation_queries,
                self.dynamics,
                self.action_decoder,
            )
        self.cost_risk_critic = (
            TwinMonotoneCostQuantileCritic(
                self.config.d_model,
                horizon=self.config.horizon,
            )
            if bool(getattr(self.config, "cost_risk_critic_enabled", False))
            else None
        )
        self.latent_prior_heads = nn.ModuleDict(
            {
                f"h{horizon}": nn.Sequential(
                    nn.LayerNorm(self.config.d_model),
                    nn.Linear(self.config.d_model, self.config.d_model),
                    nn.SiLU(),
                    nn.Linear(self.config.d_model, 2 * self.config.d_model),
                )
                for horizon in LATENT_HORIZONS
            }
        )
        self.latent_posterior_heads = nn.ModuleDict(
            {
                f"h{horizon}": nn.Sequential(
                    nn.LayerNorm(self.config.d_model),
                    nn.Linear(self.config.d_model, self.config.d_model),
                    nn.SiLU(),
                    nn.Linear(self.config.d_model, 2 * self.config.d_model),
                )
                for horizon in LATENT_HORIZONS
            }
        )
        self.posterior_target_encoder = nn.Sequential(
            nn.LayerNorm(29),
            nn.Linear(29, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        # Retained only so every registered W0/W1 checkpoint loads strictly.
        # R5-V2.3 takes stochastic scale from the action-conditioned dynamics
        # that actually generate scenarios; this legacy parallel head must not
        # influence U0 samples, losses, or checkpoint selection.
        self.distribution_scale_head = nn.Linear(self.config.d_model, 17)
        self.latent_mixture_logits_head = nn.Linear(
            self.config.d_model, self.config.num_regimes
        )
        self.latent_mixture_parameter_head = nn.Linear(
            self.config.d_model,
            2 * self.config.num_regimes * self.config.stochastic_dim,
        )

    def encode(
        self,
        packet: WorldInputPacket,
        *,
        sample_stochastic: bool | None = None,
        phase: str | None = None,
        capture_expert_context: bool = False,
    ) -> WorldStatePacket:
        packet.validate(self.config)
        tokens, action_context, slack_context = self.tokenizer(packet)
        phase_name = None if phase is None else str(phase).lower()
        w0_deterministic = phase_name == "w0"
        u0_deterministic = phase_name == "u0" and not bool(
            getattr(
                self.config,
                "u0_conditioning_encoder_sample_stochastic",
                True,
            )
        )
        if w0_deterministic or u0_deterministic:
            sample = False
        else:
            sample = (
                self.training if sample_stochastic is None else bool(sample_stochastic)
            )
        return self.backbone(
            tokens,
            action_context,
            slack_context,
            packet,
            sample_stochastic=sample,
            apply_dynamic_residuals=not w0_deterministic,
            smooth_tail_activation=w0_deterministic,
            capture_expert_context=capture_expert_context,
        )

    def _constraints_with_pv_support(
        self,
        packet: WorldInputPacket,
        *,
        horizon: int,
    ) -> WorldConstraintPacket:
        """Attach the issue-time causal daylight support used by U0/P1.

        A known, near-zero shortwave feature is the only condition that may
        declare night.  Missing/non-finite weather remains conservatively
        daylight-capable, so absence can never manufacture a deterministic
        zero atom.  No target or post-origin realized field is consulted.
        """

        if not bool(getattr(self.config, "support_atom_enabled", False)):
            return packet.constraints
        if packet.fast_values.shape[-1] < 8:
            raise ValueError("PV support requires WEATHER_SLOT shortwave features 4:8")
        if horizon > packet.fast_values.shape[1]:
            raise ValueError("PV support horizon exceeds the issue-time packet")
        shortwave = packet.fast_values[:, :horizon, WEATHER_SLOT, 4:8].float()
        known = packet.fast_mask[:, :horizon, WEATHER_SLOT]
        if packet.known_future_mask is not None:
            known = known & packet.known_future_mask[:, :horizon, WEATHER_SLOT]
        known = known.unsqueeze(-1) & torch.isfinite(shortwave)
        threshold = float(self.config.pv_night_capacity_fraction_threshold)
        daylight = (~known) | (shortwave > threshold)
        return replace(packet.constraints, pv_support_mask=daylight)

    def query_observations(
        self,
        state: WorldStatePacket,
        constraints: Any,
        *,
        da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> WorldObservationPacket:
        return self.observation_queries.from_state(
            state,
            constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )

    def imagine(
        self,
        state: WorldStatePacket,
        actions: WorldActionPacket,
        constraints: Any,
        *,
        horizon: int | None = None,
        path_count: int | None = None,
        seed: int | None = None,
        path_seed_offsets: Sequence[int] | None = None,
        compute_path_cost: bool = True,
        teacher_action_summary: Tensor | None = None,
    ) -> WorldRolloutPacket:
        resolved_horizon = actions.horizon if horizon is None else int(horizon)
        if resolved_horizon != actions.horizon:
            raise ValueError("imagination horizon differs from decoded actions")
        return self.dynamics(
            state.pooled_state,
            actions,
            constraints,
            path_count=(
                self.config.rollout_paths if path_count is None else int(path_count)
            ),
            seed=seed,
            path_seed_offsets=path_seed_offsets,
            compute_path_cost=compute_path_cost,
            teacher_action_summary=teacher_action_summary,
        )

    def plan(
        self,
        state: WorldStatePacket,
        constraints: Any,
        *,
        horizon: int | None = None,
        path_count: int | None = None,
        seed: int | None = None,
        previous_actions: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> WorldPlanPacket:
        planner = getattr(self, "planner", None)
        if planner is None:
            raise RuntimeError(
                "legacy candidate planner is disabled for this plannerless W1"
            )
        return planner(
            state,
            constraints,
            horizon=self.config.horizon if horizon is None else int(horizon),
            path_count=(
                self.config.rollout_paths if path_count is None else int(path_count)
            ),
            seed=seed,
            previous_actions=previous_actions,
            action_mask=action_mask,
        )

    def _default_rollout(
        self,
        state: WorldStatePacket,
        packet: WorldInputPacket,
        *,
        horizon: int,
        path_count: int,
        seed: int | None,
        path_seed_offsets: Sequence[int] | None = None,
        compute_path_cost: bool = True,
        teacher_action_summary: Tensor | None = None,
    ) -> WorldRolloutPacket:
        constraints = self._constraints_with_pv_support(packet, horizon=horizon)
        latent = state.pooled_state[:, None].expand(-1, horizon, -1)
        reference = self.observation_queries.from_latent_timeline(latent, constraints)
        actions = self.action_decoder.decode_zero_actions(
            reference,
            constraints,
        )
        return self.imagine(
            state,
            actions,
            constraints,
            horizon=horizon,
            path_count=path_count,
            seed=seed,
            path_seed_offsets=path_seed_offsets,
            compute_path_cost=compute_path_cost,
            teacher_action_summary=teacher_action_summary,
        )

    @staticmethod
    def _gaussian_parameters(
        head: nn.Module,
        value: Tensor,
        *,
        anchor_mean: bool = False,
        process_log_scale: Tensor | None = None,
    ) -> dict[str, Tensor]:
        parameters = head(value).float()
        mean, log_scale = parameters.chunk(2, dim=-1)
        if anchor_mean:
            # Keep the learned W0/dynamics state as the posterior/prior anchor;
            # the Gaussian head learns a bounded correction instead of being
            # free to regenerate and discard that state.
            mean = value.float() + 0.1 * torch.tanh(mean)
        if process_log_scale is not None:
            if process_log_scale.shape != value.shape[:-1] + (1,):
                raise ValueError("process_log_scale must match latent positions")
            # This makes the W1 prior likelihood train the same stochastic
            # residual scale that U0/P1 sample during imagination.
            log_scale = log_scale + 0.1 * torch.tanh(process_log_scale.float())
        return {"mean": mean, "log_scale": log_scale.clamp(-7.0, 3.0)}

    def predict_latent_distributions(
        self,
        state: WorldStatePacket,
        *,
        target_state: WorldStatePacket | None = None,
        posterior_target_latents: Mapping[str, Tensor] | None = None,
        teacher_action_summary: Tensor | None = None,
        constraints: Any | None = None,
        require_all: bool = False,
    ) -> tuple[
        dict[str, dict[str, Tensor]],
        dict[str, dict[str, Tensor]],
        dict[str, Tensor],
    ]:
        priors: dict[str, dict[str, Tensor]] = {}
        posteriors: dict[str, dict[str, Tensor]] = {}
        targets: dict[str, Tensor] = {}
        time = state.pooled_timeline.shape[1]
        dynamics_timeline: Tensor | None = None
        dynamics_log_scale: Tensor | None = None
        if teacher_action_summary is not None:
            if constraints is None:
                raise ValueError("W1 teacher actions require physical constraints")
            if teacher_action_summary.shape[:2] != (state.pooled_timeline.shape[0], 96):
                raise ValueError("W1 teacher action trajectory must be [B,96,6]")
            batch_index = torch.arange(
                state.pooled_timeline.shape[0], device=state.pooled_timeline.device
            )
            initial = state.pooled_timeline[
                batch_index, state.origin_index.to(device=batch_index.device)
            ]
            dynamics_timeline, dynamics_log_scale = (
                self.dynamics.deterministic_teacher_rollout(
                    initial, teacher_action_summary, constraints
                )
            )
            if time != 96:
                raise ValueError(
                    "formal W1 action dynamics require a 96-token timeline"
                )
        for horizon in LATENT_HORIZONS[:-1]:
            key = f"h{horizon}"
            if time <= horizon:
                if require_all:
                    raise ValueError(f"W1 lacks an in-window {key} target")
                continue
            source = (
                state.pooled_timeline[:, :-horizon]
                if dynamics_timeline is None
                else dynamics_timeline[:, horizon - 1 : time - 1]
            )
            source_process_scale = (
                None
                if dynamics_log_scale is None
                else dynamics_log_scale[:, horizon - 1 : time - 1]
            )
            target = (
                state.pooled_timeline[:, horizon:].detach()
                if posterior_target_latents is None
                else posterior_target_latents[key]
            )
            priors[key] = self._gaussian_parameters(
                self.latent_prior_heads[key],
                source,
                anchor_mean=dynamics_timeline is not None,
                process_log_scale=source_process_scale,
            )
            posteriors[key] = self._gaussian_parameters(
                self.latent_posterior_heads[key], target, anchor_mean=True
            )
            if dynamics_timeline is not None:
                priors[key]["dynamics_mean"] = source
                priors[key]["process_log_scale"] = source_process_scale
            targets[key] = target
        if target_state is not None:
            batch_index = torch.arange(
                state.pooled_timeline.shape[0], device=state.pooled_timeline.device
            )
            source = (
                state.pooled_timeline[
                    batch_index, state.origin_index.to(device=batch_index.device)
                ][:, None]
                if dynamics_timeline is None
                else dynamics_timeline[:, -1:]
            )
            source_process_scale = (
                None if dynamics_log_scale is None else dynamics_log_scale[:, -1:]
            )
            target = (
                target_state.pooled_timeline[
                    batch_index, target_state.origin_index.to(device=batch_index.device)
                ][:, None].detach()
                if posterior_target_latents is None
                else posterior_target_latents["h96"]
            )
            priors["h96"] = self._gaussian_parameters(
                self.latent_prior_heads["h96"],
                source,
                anchor_mean=dynamics_timeline is not None,
                process_log_scale=source_process_scale,
            )
            posteriors["h96"] = self._gaussian_parameters(
                self.latent_posterior_heads["h96"], target, anchor_mean=True
            )
            if dynamics_timeline is not None:
                priors["h96"]["dynamics_mean"] = source
                priors["h96"]["process_log_scale"] = source_process_scale
            targets["h96"] = target
        elif require_all:
            raise ValueError(
                "W1 requires a real causally consecutive target_packet for h96"
            )
        return priors, posteriors, targets

    def encode_posterior_targets(
        self,
        future_target_sequence: Tensor,
        future_state_sequence: Tensor,
        *,
        origin_index: Tensor,
    ) -> dict[str, Tensor]:
        """Encode realized public/state targets for the W1 posterior only."""

        if future_target_sequence.ndim != 3 or future_target_sequence.shape[1:] != (
            192,
            21,
        ):
            raise ValueError("future_target_sequence must be [B,192,21]")
        if (
            future_state_sequence.ndim != 3
            or future_state_sequence.shape[:2] != future_target_sequence.shape[:2]
            or future_state_sequence.shape[-1] < 8
        ):
            raise ValueError("future_state_sequence must be [B,192,S] with S>=8")
        if not bool(torch.isfinite(future_target_sequence).all()) or not bool(
            torch.isfinite(future_state_sequence).all()
        ):
            raise ValueError("realized W1 posterior targets must be finite")
        if origin_index.shape != (future_target_sequence.shape[0],):
            raise ValueError("posterior target origin_index must be [B]")
        realized = torch.cat(
            (future_target_sequence.float(), future_state_sequence[..., :8].float()),
            dim=-1,
        )
        encoded = self.posterior_target_encoder(realized)
        result = {
            f"h{horizon}": encoded[:, horizon:96] for horizon in LATENT_HORIZONS[:-1]
        }
        batch_index = torch.arange(encoded.shape[0], device=encoded.device)
        h96_index = 96 + origin_index.to(device=encoded.device)
        result["h96"] = encoded[batch_index, h96_index][:, None]
        return result

    def predict_latent_horizons(
        self,
        state: WorldStatePacket,
        *,
        target_state: WorldStatePacket | None = None,
        require_all: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compatibility view backed by the explicit R5-V2 Gaussian packets."""

        priors, _, targets = self.predict_latent_distributions(
            state, target_state=target_state, require_all=require_all
        )
        return {key: value["mean"] for key, value in priors.items()}, targets

    def _w1_observation_predictions(
        self,
        priors: Mapping[str, Mapping[str, Tensor]],
        packet: WorldInputPacket,
        *,
        da_anchor: Tensor | None = None,
        future_da_anchor: Tensor | None = None,
        da_residual_cap: Tensor | None = None,
    ) -> tuple[
        dict[str, dict[str, Tensor]],
        dict[str, dict[str, Tensor]],
        dict[str, dict[str, Tensor]],
    ]:
        quantiles: dict[str, dict[str, Tensor]] = {}
        joint_paths: dict[str, dict[str, Tensor]] = {}
        states: dict[str, dict[str, Tensor]] = {}
        for key, parameters in priors.items():
            latent = parameters["mean"]
            horizon = int(key.removeprefix("h"))
            aligned_anchor = None
            if da_anchor is not None:
                if horizon == 96:
                    if future_da_anchor is None:
                        raise ValueError(
                            "h96 anchored decoding requires next-day DA anchor"
                        )
                    batch_index = torch.arange(
                        latent.shape[0], device=future_da_anchor.device
                    )
                    origin_index = packet.origin_index.to(
                        device=future_da_anchor.device, dtype=torch.long
                    )
                    aligned_anchor = future_da_anchor[
                        batch_index, origin_index
                    ].unsqueeze(1)
                else:
                    aligned_anchor = da_anchor[:, horizon:]
                if aligned_anchor.shape[-2] != latent.shape[-2]:
                    raise ValueError(
                        f"{key} DA anchor length differs from latent prediction"
                    )
            decoded = self.observation_queries.quantiles_from_latent_timeline(
                latent,
                packet.constraints,
                da_anchor=aligned_anchor,
                da_residual_cap=da_residual_cap,
            )
            joint = decoded.pop("joint_price_paths")
            quantiles[key] = {
                name: value
                for name, value in decoded.items()
                if name in {"price", "workload", "pv", "pue", "da", "rt", "dart"}
                and isinstance(value, Tensor)
            }
            joint_paths[key] = dict(joint)  # type: ignore[arg-type]
            # These are endpoint latents, not a fresh queue trajectory that
            # starts again from issue-time initial_queue.  Reuse W0's typed
            # hurdle heads to avoid h4/h16 initial-state misalignment and the
            # clamp-at-zero dead gradient of the operational recurrence.
            queue_distribution = (
                self.observation_queries.queue_distribution_from_latent_timeline(latent)
            )
            states[key] = {
                "soc": self.observation_queries.soc_from_latent_timeline(latent),
                "queue": queue_distribution["mean"],
                "queue_distribution": queue_distribution,
            }
        return quantiles, joint_paths, states

    def _latent_mixture(self, state: WorldStatePacket) -> dict[str, Tensor]:
        timeline = state.pooled_timeline
        logits = self.latent_mixture_logits_head(timeline).float()
        raw = (
            self.latent_mixture_parameter_head(timeline)
            .float()
            .reshape(
                *timeline.shape[:-1],
                self.config.num_regimes,
                2 * self.config.stochastic_dim,
            )
        )
        means, log_scales = raw.chunk(2, dim=-1)
        target = state.stochastic_mean.float().mean(dim=-2).detach()
        return {
            "log_weights": F.log_softmax(logits, dim=-1),
            "means": means,
            "log_scales": log_scales.clamp(-7.0, 3.0),
            "target": target,
        }

    @staticmethod
    def _w1_latent_diagnostics(
        priors: Mapping[str, Mapping[str, Tensor]],
        posteriors: Mapping[str, Mapping[str, Tensor]],
    ) -> dict[str, Tensor]:
        raw_values: list[Tensor] = []
        for key in ("h1", "h4", "h16", "h96"):
            prior = priors[key]
            posterior = posteriors[key]
            p_mean = prior["mean"].float()
            q_mean = posterior["mean"].float()
            p_logvar = 2.0 * prior["log_scale"].float()
            q_logvar = 2.0 * posterior["log_scale"].float()
            raw = 0.5 * (
                p_logvar
                - q_logvar
                + (q_logvar - p_logvar).exp()
                + (q_mean - p_mean).square() * (-p_logvar).exp()
                - 1.0
            ).sum(dim=-1)
            raw_values.append(raw.reshape(-1))
        joined = torch.cat(raw_values)
        finite = torch.isfinite(joined).all()
        floor_fraction = (joined <= 1.0 + 1.0e-6).float().mean()
        # Free bits are an optimization threshold, not a capability failure.
        # The explicit raw-KL path remains trainable below one nat, so the
        # relevant degeneracy check is finite positive raw information.
        non_degenerate = finite & (joined.mean() > 1.0e-6)
        return {
            "raw_kl_mean": joined.mean(),
            "free_bit_floor_fraction": floor_fraction,
            "non_degenerate": non_degenerate.float(),
        }

    @staticmethod
    def _w1_queue_diagnostics(
        states: Mapping[str, Mapping[str, Tensor]],
    ) -> dict[str, Tensor]:
        probabilities: list[Tensor] = []
        severities: list[Tensor] = []
        for key in ("h1", "h4", "h16", "h96"):
            distribution = states[key]["queue_distribution"]
            probabilities.append(distribution["occurrence_probability"].reshape(-1))
            severities.append(distribution["positive_severity"].reshape(-1))
        probability = torch.cat(probabilities).float()
        severity = torch.cat(severities).float()
        finite = torch.isfinite(probability).all() & torch.isfinite(severity).all()
        varying = (probability.std(unbiased=False) > 1.0e-7) | (
            severity.std(unbiased=False) > 1.0e-7
        )
        return {
            "occurrence_probability_mean": probability.mean(),
            "positive_severity_mean": severity.mean(),
            "prediction_non_degenerate": (finite & varying).float(),
        }

    @staticmethod
    def _reconstruction_mapping(
        observations: WorldObservationPacket,
    ) -> dict[str, Tensor]:
        return {
            "da": observations.da,
            "rt": observations.rt,
            "dart": observations.dart,
            "workload_arrivals": observations.arrivals,
            "workload_service": observations.service,
            "workload_queue": observations.queue[:, 1:],
            "queue": observations.queue[:, 1:],
            "soc": observations.soc,
            "workload": observations.arrivals.sum(dim=-1, keepdim=True),
            "pv": observations.pv,
            "pue": observations.pue,
            "public_21": observations.public_21,
        }

    @staticmethod
    def _planner_mapping(plan: WorldPlanPacket) -> dict[str, Any]:
        def member(source: Any, *names: str) -> Any | None:
            if source is None:
                return None
            for name in names:
                if isinstance(source, Mapping) and source.get(name) is not None:
                    return source[name]
                value = getattr(source, name, None)
                if value is not None:
                    return value
            return None

        actions = plan.actions
        rollout = plan.rollout
        metadata = member(plan, "metadata", "training_metadata")
        rollout_metadata = member(rollout, "metadata", "training_metadata")

        def scientific(*names: str) -> Any | None:
            for source in (plan, rollout, metadata, rollout_metadata):
                value = member(source, *names)
                if value is not None:
                    return value
            return None

        shielded_actions = torch.cat(
            (
                actions.battery,
                actions.compute,
                actions.grid,
                actions.pv_used,
                actions.pv_curtailed,
                actions.unserved_power,
            ),
            dim=-1,
        )
        shield_correction = scientific("shield_correction", "shield_intervention")
        if shield_correction is None:
            shield_correction = actions.shield_intervention
        return {
            "packet": plan,
            # Raw path cost remains a diagnostic only.  R5-V2 losses must use
            # normalized_scenario_costs and must not silently fall back here.
            "scenario_costs": rollout.path_cost,
            "scenario_weights": rollout.weights,
            "day_ahead_commitment": scientific("day_ahead_commitment"),
            "normalized_scenario_costs": scientific(
                "normalized_scenario_costs", "normalized_path_costs"
            ),
            "cvar_eta": scientific("cvar_eta", "eta"),
            "raw_constraint_violations": scientific(
                "raw_constraint_violations", "pre_shield_constraint_violations"
            ),
            "constraint_multipliers": scientific(
                "constraint_multipliers", "lagrange_multipliers"
            ),
            "constraint_penalties": scientific(
                "constraint_penalties", "penalty_coefficients"
            ),
            "shield_correction": shield_correction,
            "previous_ramp_action": scientific(
                "previous_ramp_action", "ramp_initial_action"
            ),
            "previous_ramp_mask": scientific("previous_ramp_mask"),
            "policy_moe": scientific("policy_moe"),
            "shielded_actions": shielded_actions,
            "shielded_action_components": {
                "battery": actions.battery,
                "compute": actions.compute,
                "migration": actions.migration,
                "grid": actions.grid,
            },
            "raw_actions": plan.raw_actions,
            "constraint_violations": actions.constraint_violation,
            "sla_shortfall": actions.queue[:, 1:],
            "terminal_violation": (
                actions.queue[:, -1] + actions.constraint_violation[:, -1]
            ),
            "shield_intervention": actions.shield_intervention,
            "candidate_scores": plan.candidate_scores,
            "selection_weights": plan.selection_weights,
        }

    def _u0_fast_result(
        self,
        state: WorldStatePacket,
        packet: WorldInputPacket,
        *,
        horizon: int,
        paths: int,
        seed: int | None,
        path_seed_offsets: Sequence[int] | None,
        teacher_action_summary: Tensor | None,
    ) -> Mapping[str, Any]:
        """Run only the stochastic objects consumed by U0 loss and gates.

        U0 scores generated public paths, the causal MoE routes that produced
        their initial state, and the legacy mixture diagnostic.  Issue-state
        reconstruction/quantile heads, W1 prior/posterior heads, and economic
        path costs have no edge to that objective.  Omitting those computations
        changes no parameter registration and therefore keeps every W0/W1
        checkpoint strictly loadable.
        """

        constraints = self._constraints_with_pv_support(packet, horizon=horizon)
        rollout = self._default_rollout(
            state,
            packet,
            horizon=horizon,
            path_count=paths,
            seed=seed,
            path_seed_offsets=path_seed_offsets,
            compute_path_cost=False,
            teacher_action_summary=teacher_action_summary,
        )
        observations = rollout.observations
        scenarios = observations.public_21
        distribution_mean = scenarios.mean(dim=1)
        empirical_scale = scenarios.std(dim=1, unbiased=False)
        distribution_log_scale = torch.log(
            empirical_scale.clamp_min(self.config.minimum_scale)
        )
        mixture = self._latent_mixture(state)
        analytic_distribution: Mapping[str, Any] = {}
        if bool(getattr(self.config, "analytic_mixed_support_enabled", False)):
            if teacher_action_summary is None:
                raise ValueError(
                    "distribution-native U0 requires a continuous teacher action path"
                )
            deterministic_timeline, _ = self.dynamics.deterministic_teacher_rollout(
                state.pooled_state,
                teacher_action_summary,
                constraints,
            )
            analytic_distribution = (
                self.observation_queries.mixed_distribution_from_latent_timeline(
                    deterministic_timeline,
                    constraints,
                )
            )
        analytic_quantiles = analytic_distribution.get("quantiles", {})
        analytic_joint_price = analytic_distribution.get("joint_price_paths", {})
        return {
            "state": state,
            "observations": observations,
            "reconstructions": self._reconstruction_mapping(observations),
            # These empty views preserve the public mapping contract while
            # proving that no W1 head was executed on the U0 fast path.
            "latent_predictions": {},
            "latent_targets": {},
            "latent_prior": {},
            "latent_posterior": {},
            "quantile_predictions": analytic_quantiles,
            "quantile_levels": torch.tensor(
                QUANTILE_LEVELS,
                dtype=torch.float32,
                device=state.pooled_timeline.device,
            ),
            "count_distribution": None,
            "queue_distribution": None,
            "state_predictions": {},
            "observation_quantile_predictions": {},
            "observation_joint_price_paths": {},
            "state_rollout_predictions": {},
            "w1_latent_diagnostics": None,
            "w1_queue_diagnostics": None,
            "observation_predictions": observations,
            "rollout": rollout,
            "support_probabilities": rollout.support_probabilities or {},
            "analytic_distribution": analytic_distribution,
            "mixed_distribution": analytic_distribution,
            "plan": None,
            "planner": None,
            "scenarios": scenarios,
            "joint_price_paths": (
                analytic_joint_price
                if analytic_joint_price
                else {
                    "da": observations.da,
                    "rt": observations.rt,
                    "dart": observations.dart,
                }
            ),
            "distribution_mean": distribution_mean,
            "distribution_log_scale": distribution_log_scale,
            "distribution_scale_available": True,
            "distribution_scale_source": "dynamics_residual_empirical",
            "distribution_scale_head_role": "legacy_diagnostic_only",
            "mixture_log_weights": mixture["log_weights"],
            "mixture_means": mixture["means"],
            "mixture_log_scales": mixture["log_scales"],
            "mixture_target": mixture["target"],
            "latent_mixture": {
                "logits": mixture["log_weights"],
                "log_weights": mixture["log_weights"],
                "means": mixture["means"],
                "log_scales": mixture["log_scales"],
            },
            "latent_mixture_target": mixture["target"],
            "router_aux_loss": state.router_aux_loss,
            "expert_assignment_schema": EXPERT_ASSIGNMENT_SCHEMA,
            "queue_tail_route_logits": state.queue_tail_route_logits,
            "queue_tail_route_probabilities": state.queue_tail_route_probabilities,
            "queue_tail_route_positive_weight": float(
                getattr(self.config, "tail_route_positive_weight", 49.08092411354107)
            ),
            "moe_tail_diagnostics": [
                {
                    "block_index": diagnostic.block_index,
                    "tail_route_logits": diagnostic.tail_route_logits[..., None],
                    "tail_route_probabilities": (
                        diagnostic.tail_route_probabilities[..., None]
                    ),
                    "tail_route_positive_weight": float(
                        getattr(
                            self.config,
                            "tail_route_positive_weight",
                            49.08092411354107,
                        )
                    ),
                    "tail_active_mask": diagnostic.tail_active_mask[..., None],
                    "tail_activation_rate": diagnostic.tail_load,
                    "tail_utilization": diagnostic.tail_utilization,
                    "tail_capacity": diagnostic.tail_capacity,
                    "tail_selected_count": diagnostic.tail_selected_count,
                    "causal_probability_mean": (
                        diagnostic.tail_route_probabilities.mean()
                    ),
                }
                for diagnostic in state.router_diagnostics
            ],
            "physics_residual": state.physics_balance_residual,
            "u0_scenario_only_fast_path": True,
            "rollout_cost_available": False,
        }

    def _p1_fast_result(
        self,
        state: WorldStatePacket,
        packet: WorldInputPacket,
        *,
        horizon: int,
        paths: int,
        seed: int | None,
    ) -> Mapping[str, Any]:
        """Run only objects consumed by formal P1 loss and diagnostics."""

        planning_constraints = self._constraints_with_pv_support(
            packet, horizon=horizon
        )
        plan_packet = self.plan(
            state,
            planning_constraints,
            horizon=horizon,
            path_count=paths,
            seed=seed,
            previous_actions=packet.previous_actions,
            action_mask=packet.action_mask,
        )
        rollout = plan_packet.rollout
        observations = rollout.observations
        planner_mapping = self._planner_mapping(plan_packet)
        if self.cost_risk_critic is not None:
            selected_action_summary = self.dynamics.normalize_action_summary(
                self.dynamics.action_summary(plan_packet.actions),
                planning_constraints,
            )
            cost_distribution = self.cost_risk_critic(
                state.pooled_state,
                selected_action_summary,
            )
            planner_mapping["cost_distribution"] = cost_distribution
            planner_mapping["cost_quantiles"] = cost_distribution["quantiles"]
            planner_mapping["cost_quantile_levels"] = cost_distribution["levels"]
            planner_mapping["critic_expected_cost"] = cost_distribution["expected_cost"]
            planner_mapping["critic_cvar95"] = cost_distribution["cvar95"]
        return {
            "state": state,
            "observations": observations,
            "observation_predictions": observations,
            "rollout": rollout,
            "support_probabilities": rollout.support_probabilities or {},
            "plan": plan_packet,
            "planner": planner_mapping,
            "scenarios": observations.public_21,
            # P1 validation still proves the exact per-path price closure.
            "joint_price_paths": {
                "da": observations.da,
                "rt": observations.rt,
                "dart": observations.dart,
            },
            "router_aux_loss": state.router_aux_loss,
            "expert_assignment_schema": EXPERT_ASSIGNMENT_SCHEMA,
            "queue_tail_route_logits": state.queue_tail_route_logits,
            "queue_tail_route_probabilities": state.queue_tail_route_probabilities,
            "queue_tail_route_positive_weight": float(
                getattr(self.config, "tail_route_positive_weight", 49.08092411354107)
            ),
            "moe_tail_diagnostics": [
                {
                    "block_index": diagnostic.block_index,
                    "tail_route_logits": diagnostic.tail_route_logits[..., None],
                    "tail_route_probabilities": (
                        diagnostic.tail_route_probabilities[..., None]
                    ),
                    "tail_route_positive_weight": float(
                        getattr(
                            self.config,
                            "tail_route_positive_weight",
                            49.08092411354107,
                        )
                    ),
                    "tail_active_mask": diagnostic.tail_active_mask[..., None],
                    "tail_activation_rate": diagnostic.tail_load,
                    "tail_utilization": diagnostic.tail_utilization,
                    "tail_capacity": diagnostic.tail_capacity,
                    "tail_selected_count": diagnostic.tail_selected_count,
                    "causal_probability_mean": (
                        diagnostic.tail_route_probabilities.mean()
                    ),
                }
                for diagnostic in state.router_diagnostics
            ],
            "physics_residual": state.physics_balance_residual,
            "p1_fast_path": True,
        }

    def forward(
        self,
        packet: WorldInputPacket,
        *,
        phase: str = "joint",
        horizon: int | None = None,
        sample_count: int | None = None,
        seed: int | None = None,
        path_seed_offsets: Sequence[int] | None = None,
        plan: bool | None = None,
        target_packet: WorldInputPacket | None = None,
        count_label_mask: Tensor | None = None,
        future_target_sequence: Tensor | None = None,
        future_state_sequence: Tensor | None = None,
        teacher_action_summary: Tensor | None = None,
        teacher_action_mask: Tensor | None = None,
        forecast_anchors: Mapping[str, Tensor] | None = None,
        forecast_residual_caps: Mapping[str, Tensor] | None = None,
        future_forecast_anchors: Mapping[str, Tensor] | None = None,
    ) -> Mapping[str, Any]:
        phase_name = str(phase).lower()
        if phase_name == "joint":
            phase_name = "j1"
        allowed = {"w0", "w1", "u0", "p1", "j1", "c1", "encode"}
        if phase_name not in allowed:
            raise ValueError(f"unknown WorldMoE phase: {phase}")
        resolved_horizon = (
            min(packet.fast_values.shape[1], self.config.horizon)
            if horizon is None
            else int(horizon)
        )
        if not 1 <= resolved_horizon <= self.config.horizon:
            raise ValueError("forward horizon lies outside [1,96]")
        paths = self.config.rollout_paths if sample_count is None else int(sample_count)
        formal_ensemble = path_seed_offsets is not None
        if formal_ensemble:
            offsets = tuple(int(value) for value in path_seed_offsets)
            if phase_name != "u0" or plan is True:
                raise ValueError("100-path seed chunks are formal U0 rollout-only")
            if torch.is_grad_enabled():
                raise ValueError("100-path formal U0 rollout is evaluation-only")
            if paths != 5 * self.config.rollout_paths:
                raise ValueError("formal U0 rollout requires exactly 100 paths")
            path_seed_offsets = offsets
        elif phase_name == "u0" and bool(
            getattr(self.config, "dynamic_particle_count_enabled", False)
        ):
            maximum_paths = int(
                getattr(self.config, "maximum_dynamic_particle_count", 256)
            )
            if not 2 <= paths <= maximum_paths:
                raise ValueError(
                    "distribution-native U0 path count lies outside its registered range"
                )
        elif paths != self.config.rollout_paths:
            raise ValueError("WorldMoE uses exactly 20 internal paths")

        da_anchor = None if forecast_anchors is None else forecast_anchors.get("da")
        da_residual_cap = (
            None if forecast_residual_caps is None else forecast_residual_caps.get("da")
        )
        future_da_anchor = (
            None
            if future_forecast_anchors is None
            else future_forecast_anchors.get("da")
        )
        if da_anchor is not None and not isinstance(da_anchor, Tensor):
            raise TypeError("forecast_anchors.da must be a Tensor")
        if da_residual_cap is not None and not isinstance(da_residual_cap, Tensor):
            raise TypeError("forecast_residual_caps.da must be a Tensor")
        if future_da_anchor is not None and not isinstance(future_da_anchor, Tensor):
            raise TypeError("future_forecast_anchors.da must be a Tensor")

        state = self.encode(packet, phase=phase_name)
        if phase_name == "u0" and bool(
            getattr(self.config, "u0_action_conditioned", False)
        ):
            expected_action_shape = (state.pooled_timeline.shape[0], 96, 6)
            if teacher_action_summary is None or teacher_action_mask is None:
                raise ValueError(
                    "distribution-native U0 requires the registered teacher action path"
                )
            if (
                teacher_action_summary.shape != expected_action_shape
                or teacher_action_mask.shape != expected_action_shape
                or teacher_action_mask.dtype is not torch.bool
                or not bool(teacher_action_mask.all())
                or not bool(torch.isfinite(teacher_action_summary).all())
            ):
                raise ValueError(
                    "U0 teacher actions must be finite, fully valid [B,96,6] tensors"
                )
        if phase_name == "p1" and plan is True:
            return self._p1_fast_result(
                state,
                packet,
                horizon=resolved_horizon,
                paths=paths,
                seed=seed,
            )
        if phase_name == "u0" and plan is not True:
            return self._u0_fast_result(
                state,
                packet,
                horizon=resolved_horizon,
                paths=paths,
                seed=seed,
                path_seed_offsets=path_seed_offsets,
                teacher_action_summary=teacher_action_summary,
            )
        observations = self.query_observations(
            state,
            packet.constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )
        target_state = None
        if target_packet is not None:
            with torch.no_grad():
                target_state = self.encode(
                    target_packet,
                    sample_stochastic=False,
                    phase=phase_name,
                )
        posterior_target_latents = None
        if phase_name in {"w1", "j1"}:
            if (
                future_target_sequence is None
                or future_state_sequence is None
                or teacher_action_summary is None
                or teacher_action_mask is None
            ):
                raise ValueError(
                    "W1/J1 lacks a real causally consecutive future_target_sequence "
                    "or a continuous teacher action trajectory"
                )
            expected_action_shape = (state.pooled_timeline.shape[0], 96, 6)
            if teacher_action_summary.shape != expected_action_shape:
                raise ValueError("teacher_action_summary must be [B,96,6]")
            if (
                teacher_action_mask.shape != expected_action_shape
                or teacher_action_mask.dtype is not torch.bool
                or not bool(teacher_action_mask.all())
            ):
                raise ValueError("formal W1 requires a fully valid teacher action mask")
            if not bool(torch.isfinite(teacher_action_summary).all()):
                raise ValueError("teacher_action_summary contains non-finite values")
        latent_prior, latent_posterior, latent_targets = (
            self.predict_latent_distributions(
                state,
                target_state=target_state,
                posterior_target_latents=posterior_target_latents,
                teacher_action_summary=teacher_action_summary,
                constraints=packet.constraints,
                require_all=phase_name in {"w1", "j1"},
            )
        )
        latent_predictions = {
            key: parameters["mean"] for key, parameters in latent_prior.items()
        }
        reconstructions = self._reconstruction_mapping(observations)
        quantile_output = self.observation_queries.quantiles_from_state(
            state,
            packet.constraints,
            da_anchor=da_anchor,
            da_residual_cap=da_residual_cap,
        )
        joint_price_paths = quantile_output.pop("joint_price_paths")
        quantile_predictions = {
            key: value
            for key, value in quantile_output.items()
            if isinstance(value, Tensor)
        }
        count_distribution = self.observation_queries.count_distribution_from_state(
            state, count_label_mask
        )
        queue_distribution = self.observation_queries.queue_distribution_from_state(
            state
        )
        state_predictions = {
            "soc": observations.soc,
            "queue": queue_distribution["mean"],
        }
        observation_quantile_predictions: dict[str, dict[str, Tensor]] = {}
        observation_joint_price_paths: dict[str, dict[str, Tensor]] = {}
        state_rollout_predictions: dict[str, dict[str, Tensor]] = {}
        if phase_name in {"w1", "j1"}:
            (
                observation_quantile_predictions,
                observation_joint_price_paths,
                state_rollout_predictions,
            ) = self._w1_observation_predictions(
                latent_prior,
                packet,
                da_anchor=da_anchor,
                future_da_anchor=future_da_anchor,
                da_residual_cap=da_residual_cap,
            )
        w1_latent_diagnostics = (
            self._w1_latent_diagnostics(latent_prior, latent_posterior)
            if phase_name in {"w1", "j1"}
            else None
        )
        w1_queue_diagnostics = (
            self._w1_queue_diagnostics(state_rollout_predictions)
            if phase_name in {"w1", "j1"}
            else None
        )
        mixture = self._latent_mixture(state)
        distribution_mean = observations.public_21
        # No stochastic distribution exists until a dynamics rollout is
        # materialized.  Keep the historical tensor contract for W0/W1 while
        # explicitly marking the value unavailable below.
        distribution_log_scale = torch.zeros_like(distribution_mean)
        distribution_scale_available = False

        want_plan = phase_name in {"p1", "j1", "c1"} if plan is None else bool(plan)
        plan_packet: WorldPlanPacket | None = None
        rollout: WorldRolloutPacket | None = None
        if want_plan:
            planning_constraints = self._constraints_with_pv_support(
                packet,
                horizon=resolved_horizon,
            )
            plan_packet = self.plan(
                state,
                planning_constraints,
                horizon=resolved_horizon,
                path_count=paths,
                seed=seed,
                previous_actions=packet.previous_actions,
                action_mask=packet.action_mask,
            )
            rollout = plan_packet.rollout
        elif phase_name == "u0" or sample_count is not None:
            rollout = self._default_rollout(
                state,
                packet,
                horizon=resolved_horizon,
                path_count=paths,
                seed=seed,
                path_seed_offsets=path_seed_offsets,
            )
        scenarios = None
        if rollout is not None:
            scenarios = rollout.observations.public_21
            distribution_mean = scenarios.mean(dim=1)
            empirical_scale = scenarios.std(dim=1, unbiased=False)
            distribution_log_scale = torch.log(
                empirical_scale.clamp_min(self.config.minimum_scale)
            )
            distribution_scale_available = True
        result: dict[str, Any] = {
            "state": state,
            "observations": observations,
            "reconstructions": reconstructions,
            "latent_predictions": latent_predictions,
            "latent_targets": latent_targets,
            "latent_prior": latent_prior,
            "latent_posterior": latent_posterior,
            "w1_teacher_action_summary": teacher_action_summary,
            "w1_teacher_action_delta": (
                None
                if teacher_action_summary is None
                else torch.cat(
                    (
                        torch.zeros_like(teacher_action_summary[:, :1]),
                        teacher_action_summary[:, 1:] - teacher_action_summary[:, :-1],
                    ),
                    dim=1,
                )
            ),
            "quantile_predictions": quantile_predictions,
            "quantile_levels": torch.tensor(
                QUANTILE_LEVELS,
                dtype=torch.float32,
                device=state.pooled_timeline.device,
            ),
            "joint_price_paths": joint_price_paths,
            "count_distribution": count_distribution,
            "queue_distribution": queue_distribution,
            "state_predictions": state_predictions,
            "observation_quantile_predictions": observation_quantile_predictions,
            "observation_joint_price_paths": observation_joint_price_paths,
            "state_rollout_predictions": state_rollout_predictions,
            "w1_latent_diagnostics": w1_latent_diagnostics,
            "w1_queue_diagnostics": w1_queue_diagnostics,
            "observation_predictions": (
                None if rollout is None else rollout.observations
            ),
            "rollout": rollout,
            "support_probabilities": (
                {} if rollout is None else rollout.support_probabilities or {}
            ),
            "plan": plan_packet,
            "planner": (
                None if plan_packet is None else self._planner_mapping(plan_packet)
            ),
            "scenarios": scenarios,
            "distribution_mean": distribution_mean,
            "distribution_log_scale": distribution_log_scale,
            "distribution_scale_available": distribution_scale_available,
            "distribution_scale_source": "dynamics_residual_empirical",
            "distribution_scale_head_role": "legacy_diagnostic_only",
            "mixture_log_weights": mixture["log_weights"],
            "mixture_means": mixture["means"],
            "mixture_log_scales": mixture["log_scales"],
            "mixture_target": mixture["target"],
            "latent_mixture": {
                "logits": mixture["log_weights"],
                "log_weights": mixture["log_weights"],
                "means": mixture["means"],
                "log_scales": mixture["log_scales"],
            },
            "latent_mixture_target": mixture["target"],
            "router_aux_loss": state.router_aux_loss,
            "expert_assignment_schema": EXPERT_ASSIGNMENT_SCHEMA,
            "queue_tail_route_logits": state.queue_tail_route_logits,
            "queue_tail_route_probabilities": (state.queue_tail_route_probabilities),
            "queue_tail_route_positive_weight": float(
                getattr(self.config, "tail_route_positive_weight", 49.08092411354107)
            ),
            "moe_tail_diagnostics": [
                {
                    "block_index": diagnostic.block_index,
                    "tail_route_logits": diagnostic.tail_route_logits[..., None],
                    "tail_route_probabilities": (
                        diagnostic.tail_route_probabilities[..., None]
                    ),
                    "tail_route_positive_weight": float(
                        getattr(
                            self.config,
                            "tail_route_positive_weight",
                            49.08092411354107,
                        )
                    ),
                    "tail_active_mask": diagnostic.tail_active_mask[..., None],
                    "tail_activation_rate": diagnostic.tail_load,
                    "tail_utilization": diagnostic.tail_utilization,
                    "tail_capacity": diagnostic.tail_capacity,
                    "tail_selected_count": diagnostic.tail_selected_count,
                    "causal_probability_mean": (
                        diagnostic.tail_route_probabilities.mean()
                    ),
                }
                for diagnostic in state.router_diagnostics
            ],
            "physics_residual": state.physics_balance_residual,
        }
        return result

    def parameter_groups(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {name: [] for name in WORLD_PARAMETER_GROUPS}
        for name, _ in self.named_parameters():
            if name.startswith(("tokenizer.", "observation_queries.")) or any(
                part in name for part in (".local_attention.", ".graph_attention.")
            ):
                group = "encoder"
            elif (
                name.startswith(
                    (
                        "latent_prior_heads.",
                        "latent_posterior_heads.",
                        "posterior_target_encoder.",
                    )
                )
                or ".temporal_ssm." in name
            ):
                group = "exogenous_dynamics"
            elif (
                ".stochastic_update." in name
                or name.startswith("distribution_scale_head.")
                or name.startswith(
                    ("latent_mixture_logits_head.", "latent_mixture_parameter_head.")
                )
                or name.startswith(
                    (
                        "dynamics.regime_",
                        "dynamics.residual_",
                    )
                )
            ):
                group = "stochastic"
            elif name.startswith("dynamics.") or ".liquid_transition." in name:
                group = "action_dynamics"
            elif name.startswith("planner."):
                group = "planner"
            elif name.startswith("cost_risk_critic."):
                group = "planner"
            elif any(
                marker in name for marker in (".moe.router.", ".moe.tail_router.")
            ):
                group = "router"
            elif name.startswith("action_decoder."):
                group = "planner"
            elif (
                ".moe." in name
                or ".feed_forward." in name
                or ".physics_residual." in name
                or name.startswith("backbone.final_norm.")
                or name.startswith("backbone.pool_score.")
            ):
                group = "shared"
            else:
                raise RuntimeError(f"unregistered WorldMoE parameter: {name}")
            groups[group].append(name)
        result = {key: tuple(value) for key, value in groups.items()}
        flattened = [name for values in result.values() for name in values]
        available = [name for name, _ in self.named_parameters()]
        if (
            tuple(result) != WORLD_PARAMETER_GROUPS
            or len(flattened) != len(set(flattened))
            or set(flattened) != set(available)
        ):
            raise RuntimeError(
                "WorldMoE parameter groups are not disjoint and exhaustive"
            )
        return result

    def world_parameter_groups(self) -> dict[str, tuple[str, ...]]:
        return self.parameter_groups()

    def configure_trainable_groups(
        self, active_groups: Sequence[str]
    ) -> tuple[str, ...]:
        active = {str(item) for item in active_groups}
        unknown = active - set(WORLD_PARAMETER_GROUPS)
        if unknown:
            raise ValueError(f"unknown WorldMoE parameter groups: {sorted(unknown)}")
        groups = self.parameter_groups()
        enabled = {name for group in active for name in groups[group]}
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in enabled)
        return tuple(sorted(enabled))


__all__ = [
    "LATENT_HORIZONS",
    "VPPWorldMoE",
    "WORLD_PARAMETER_GROUPS",
]
