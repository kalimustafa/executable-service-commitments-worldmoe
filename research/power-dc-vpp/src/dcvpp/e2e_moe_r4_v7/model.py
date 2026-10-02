"""Independent R4-V7 baseline-anchored joint world-policy model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dcvpp.e2e_moe_r4.liquid import ClosedFormLiquidSequence
from dcvpp.e2e_moe_r4.model import (
    ARCHITECTURE_VERSION as REPRESENTATION_ARCHITECTURE_VERSION,
    CrossDomainFusion,
    MarketTransformerEncoder,
    PhysicalStream,
    R4Config as RepresentationConfig,
    TokenTop2MoE,
    ZonalForecastDecoder,
)

from .config import ARCHITECTURE_VERSION
from .data import FrozenResidualAnchor, issue_center_from_batch
from .distribution import (
    BaselineAnchoredConditionalTransport,
    BaselineAnchoredWorld,
)
from .policy import JointWorldPolicy, PhysicalActionLimits


HORIZON = 96
CHANNELS = 21
ZONES = 4


@dataclass(frozen=True)
class V7ModelConfig:
    """Compact neural configuration; scientific invariants remain fixed."""

    horizon: int = HORIZON
    history_days: int = 7
    num_sites: int = ZONES
    market_dim: int = 16
    workload_dim: int = 5
    physical_static_dim: int = 4
    physical_dynamic_dim: int = 12
    state_dim: int = 49
    hidden_dim: int = 128
    market_layers: int = 2
    market_heads: int = 4
    market_ff_dim: int = 384
    fusion_heads: int = 4
    fusion_ff_dim: int = 256
    expert_hidden_dim: int = 256
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.1
    router_jitter: float = 0.05
    physical_residual_bound: float = 0.1
    forecast_da_residual_cap: float = 0.1
    forecast_pv_residual_cap: float = 0.01
    forecast_pue_residual_cap: float = 0.04
    distribution_dependence_rank: int = 4
    distribution_regime_count: int = 8
    distribution_transport_cap_fraction: float = 0.5
    policy_refinement_steps: int = 2
    max_grid_mw: float = 1.0
    max_battery_mw: float = 1.0
    max_compute_work: float = 1.0

    def __post_init__(self) -> None:
        if (self.horizon, self.num_sites, self.hidden_dim) != (96, 4, 128):
            raise ValueError("V7 fixes horizon=96, sites=4, and hidden_dim=128")
        if (self.num_experts, self.top_k) != (8, 2):
            raise ValueError("V7 requires eight Top-2 experts")
        if self.policy_refinement_steps != 2:
            raise ValueError("V7 requires exactly two Policy refinements")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "V7ModelConfig":
        if raw is None:
            return cls()
        root: Mapping[str, Any] = raw
        model = root.get("model")
        if isinstance(model, Mapping):
            root = model
        market = root.get("market", {})
        workload = root.get("workload", {})
        physical = root.get("physical", {})
        fusion = root.get("fusion", {})
        moe = root.get("moe", {})
        distribution = root.get("distribution", {})
        policy = root.get("policy", {})
        forecasts = root.get("forecasts", {})
        sections = (market, workload, physical, fusion, moe, distribution, policy, forecasts)
        if not all(isinstance(item, Mapping) for item in sections):
            raise ValueError("V7 model sections must be mappings")
        caps = forecasts.get(
            "bounded_neural_residual_raw_unit_fallback_caps",
            forecasts.get("bounded_neural_residuals", {}),
        )
        if not isinstance(caps, Mapping):
            caps = {}
        values: dict[str, Any] = {
            "horizon": root.get("sequence_length", root.get("horizon", 96)),
            "history_days": market.get("history_tokens", 7),
            "market_dim": market.get("input_features", 16),
            "workload_dim": len(workload.get("input_features"))
            if isinstance(workload.get("input_features"), (list, tuple))
            else workload.get("input_features", 5),
            "physical_dynamic_dim": physical.get("input_features", 12),
            "hidden_dim": root.get("latent_width", root.get("hidden_dim", 128)),
            "market_layers": market.get("transformer_layers", 2),
            "market_heads": market.get("attention_heads", 4),
            "market_ff_dim": market.get("feedforward_width", 384),
            "fusion_heads": fusion.get("attention_heads", 4),
            "fusion_ff_dim": fusion.get("feedforward_width", 256),
            "expert_hidden_dim": moe.get("expert_hidden_width", 256),
            "num_experts": moe.get("experts", 8),
            "top_k": moe.get("top_k", 2),
            "dropout": market.get("dropout", 0.1),
            "router_jitter": moe.get("router_jitter_std_train", 0.05),
            "physical_residual_bound": physical.get("residual_bound_fraction", 0.1),
            "forecast_da_residual_cap": caps.get("da", 0.1),
            "forecast_pv_residual_cap": caps.get("pv", 0.01),
            "forecast_pue_residual_cap": caps.get("pue", 0.04),
            "distribution_dependence_rank": distribution.get("dependence_rank", 4),
            "distribution_regime_count": distribution.get("regimes", 8),
            "distribution_transport_cap_fraction": distribution.get(
                "transport_cap_fraction", 0.5
            ),
            "policy_refinement_steps": policy.get("residual_refinement_steps", 2),
            "max_grid_mw": policy.get("max_grid_mw", root.get("max_grid_mw", 1.0)),
            "max_battery_mw": policy.get(
                "max_battery_mw", root.get("max_battery_mw", 1.0)
            ),
            "max_compute_work": policy.get(
                "max_compute_work", root.get("max_compute_work", 1.0)
            ),
        }
        # Direct compact mappings remain useful for isolated architecture tests.
        known = {field.name for field in fields(cls)}
        for key in known:
            if key in root and not isinstance(root[key], Mapping):
                values[key] = root[key]
        return cls(**values)

    def representation_config(self) -> RepresentationConfig:
        return RepresentationConfig(
            architecture_version=REPRESENTATION_ARCHITECTURE_VERSION,
            horizon=self.horizon,
            history_days=self.history_days,
            num_sites=self.num_sites,
            market_dim=self.market_dim,
            workload_dim=self.workload_dim,
            physical_static_dim=self.physical_static_dim,
            physical_dynamic_dim=self.physical_dynamic_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            market_layers=self.market_layers,
            market_heads=self.market_heads,
            market_ff_dim=self.market_ff_dim,
            fusion_heads=self.fusion_heads,
            fusion_ff_dim=self.fusion_ff_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout=self.dropout,
            router_jitter=self.router_jitter,
            physical_residual_bound=self.physical_residual_bound,
            forecast_da_residual_cap=self.forecast_da_residual_cap,
            forecast_pv_residual_cap=self.forecast_pv_residual_cap,
            forecast_pue_residual_cap=self.forecast_pue_residual_cap,
        )


class BaselineAnchoredJointWorldPolicyMoEV7(nn.Module):
    """V7 model with separate truth and decision optimizer boundaries."""

    architecture_version = ARCHITECTURE_VERSION
    representation_architecture_version = REPRESENTATION_ARCHITECTURE_VERSION

    REPRESENTATION_PREFIXES = (
        "market_encoder.",
        "workload_encoder.",
        "physical_stream.static_adapter.",
        "physical_stream.dynamic_liquid.",
        "physical_stream.dynamic_gate",
        "state_encoder.",
        "cross_fusion.",
        "moe.",
        "forecast_decoder.",
    )
    A1_PREFIXES = (
        "forecast_decoder.dart_head.",
        "forecast_decoder.dart_calibration_",
        "distribution.truth_dart_adapter.",
        "distribution.decision_adapter.dart_bridge.",
    )
    A2_EXTRA_PREFIXES = (
        "moe.router.",
        "moe.condition.",
        "forecast_decoder.dart_head.",
        "forecast_decoder.dart_calibration_",
    )

    def __init__(
        self,
        residual_anchor: FrozenResidualAnchor,
        config: V7ModelConfig | Mapping[str, Any] | None = None,
        *,
        physical_twin_core: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, V7ModelConfig)
            else V7ModelConfig.from_mapping(config)
        )
        self.residual_anchor = residual_anchor
        rep = self.config.representation_config()
        # These modules are exactly the capability-passed Representation and
        # forecast component classes.  No predecessor world or policy object is
        # instantiated in V7.
        self.market_encoder = MarketTransformerEncoder(rep)
        self.workload_encoder = ClosedFormLiquidSequence(
            rep.workload_dim,
            rep.hidden_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
        )
        self.physical_stream = PhysicalStream(rep, physical_twin_core)
        self.state_encoder = ClosedFormLiquidSequence(
            rep.state_dim,
            rep.hidden_dim,
            tau_min_hours=0.25,
            tau_max_hours=24.0,
        )
        self.cross_fusion = CrossDomainFusion(rep)
        self.moe = TokenTop2MoE(rep)
        self.forecast_decoder = ZonalForecastDecoder(
            rep.hidden_dim,
            da_residual_cap=rep.forecast_da_residual_cap,
            pv_residual_cap=rep.forecast_pv_residual_cap,
            pue_residual_cap=rep.forecast_pue_residual_cap,
        )
        self.distribution = BaselineAnchoredConditionalTransport(
            residual_anchor,
            hidden_dim=self.config.hidden_dim,
            dependence_rank=self.config.distribution_dependence_rank,
            regime_count=self.config.distribution_regime_count,
            transport_cap_fraction=self.config.distribution_transport_cap_fraction,
        )
        self.policy = JointWorldPolicy(
            self.config.hidden_dim,
            heads=self.config.market_heads,
            expert_hidden_dim=self.config.expert_hidden_dim,
            refinement_steps=self.config.policy_refinement_steps,
            max_grid_mw=self.config.max_grid_mw,
            max_battery_mw=self.config.max_battery_mw,
            max_compute_work=self.config.max_compute_work,
        )
        self.physical_stream.freeze_twin()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def train(self, mode: bool = True) -> "BaselineAnchoredJointWorldPolicyMoEV7":
        super().train(mode)
        self.physical_stream.freeze_twin()
        return self

    @staticmethod
    def _assert_action_scope(value: Mapping[str, Any]) -> None:
        stack: list[Mapping[str, Any]] = [value]
        while stack:
            current = stack.pop()
            for key, item in current.items():
                if "migration" in str(key).lower():
                    raise ValueError("V7 received a field outside its three-action scope")
                if isinstance(item, Mapping):
                    stack.append(item)

    @staticmethod
    def _assert_issue_time(batch: Mapping[str, Any]) -> None:
        if int(torch.as_tensor(batch.get("origin", 0)).max().item()) != 0:
            raise ValueError("day-ahead issuance must occur at issue time")
        forbidden = {
            "revealed",
            "revealed_zonal",
            "revealed_mask",
            "realized_prefix",
            "current_day_realized_prefix",
        }
        present = sorted(forbidden.intersection(batch))
        if present:
            raise ValueError(
                "day-ahead issuance cannot read a realized current-day prefix: "
                + ", ".join(present)
            )

    def _model_tensor(self, value: Any) -> Tensor:
        reference = next(self.parameters())
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _required(batch: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if batch.get(name) is not None:
                return batch[name]
        raise KeyError(f"missing required model input; expected one of {names}")

    def _workload_features(self, raw: Any) -> Tensor:
        workload = self._model_tensor(raw)
        if workload.ndim != 3 or workload.shape[1] != HORIZON:
            raise ValueError("workload must be [batch,96,features]")
        if workload.shape[-1] == self.config.workload_dim:
            return workload
        if workload.shape[-1] == 2 and self.config.workload_dim == 5:
            mean, std = workload.unbind(-1)
            return torch.stack(
                (mean, std, mean - 1.2815516 * std, mean, mean + 1.2815516 * std),
                dim=-1,
            )
        raise ValueError(f"workload must end in {self.config.workload_dim} features")

    def _forecast_anchors(
        self, batch: Mapping[str, Any], reference: Tensor
    ) -> dict[str, Tensor]:
        raw = batch.get("forecast_anchors", {})
        if not isinstance(raw, Mapping):
            raise ValueError("forecast_anchors must be a mapping")
        da = raw.get("da")
        pv = raw.get("pv")
        if da is None and batch.get("market_raw") is not None:
            da = torch.as_tensor(batch["market_raw"])[..., :4]
        if pv is None and batch.get("issue_pv_raw") is not None:
            pv = batch["issue_pv_raw"]
        result: dict[str, Tensor] = {}
        expected = (reference.shape[0], HORIZON, ZONES)
        for name, value in (("da", da), ("pv", pv)):
            if value is None:
                continue
            tensor = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype
            )
            if tensor.shape != expected or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"forecast anchor {name} must be finite {expected}")
            result[name] = tensor
        if self.physical_stream.formal_artifact_backed and set(result) != {"da", "pv"}:
            raise KeyError("formal V7 requires causal DA and PV forecast anchors")
        return result

    def _forecast_caps(
        self, batch: Mapping[str, Any], reference: Tensor
    ) -> dict[str, Tensor]:
        raw = batch.get("forecast_residual_caps", {})
        if not isinstance(raw, Mapping):
            raise ValueError("forecast_residual_caps must be a mapping")
        result: dict[str, Tensor] = {}
        for name, value in raw.items():
            if name not in {"da", "pv", "pue"}:
                continue
            tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
            if not bool(torch.isfinite(tensor).all()) or bool((tensor <= 0).any()):
                raise ValueError(f"forecast_residual_caps.{name} must be positive")
            result[name] = tensor
        return result

    def prepare_issue_context(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Encode only issue-time exogenous inputs; no current-day prefix."""

        self._assert_action_scope(batch)
        market = self._model_tensor(
            self._required(batch, "market", "market_features", "market_inputs")
        )
        workload = self._workload_features(
            self._required(batch, "workload", "workload_features", "workload_inputs")
        )
        physical_static = self._model_tensor(
            self._required(batch, "physical_static", "physical", "physical_features")
        )
        physical_dynamic = batch.get("physical_dynamic")
        if "physical_static" not in batch and physical_static.ndim == 3:
            physical_dynamic = physical_static
            physical_static = physical_static[:, 0, : self.config.physical_static_dim]
        physical_dynamic_tensor = (
            None if physical_dynamic is None else self._model_tensor(physical_dynamic)
        )
        historical = batch.get("historical_market")
        historical_tensor = None if historical is None else self._model_tensor(historical)
        market_output = self.market_encoder(
            market,
            historical_market=historical_tensor,
            historical_market_available=batch.get("historical_market_available"),
            day_of_week=batch.get("day_of_week"),
            month=batch.get("month"),
            holiday=batch.get("holiday"),
            dst=batch.get("dst"),
        )
        workload_tokens, _ = self.workload_encoder(
            workload, dt=batch.get("workload_dt")
        )
        physical = self.physical_stream(physical_static, physical_dynamic_tensor)
        anchor_center = batch.get("anchor_center_zonal")
        if anchor_center is None:
            anchor_center = issue_center_from_batch(dict(batch), self.residual_anchor)
        anchor_center = torch.as_tensor(
            anchor_center, device=market.device, dtype=market.dtype
        )
        if anchor_center.shape != (market.shape[0], HORIZON, CHANNELS):
            raise ValueError("anchor_center_zonal must be [batch,96,21]")
        return {
            "_v7_issue_context": True,
            "architecture_version": self.architecture_version,
            "market_tokens": market_output["tokens"],
            "market_context_tokens": market_output["context_tokens"],
            "workload_tokens": workload_tokens,
            "physical_tokens": physical["tokens"],
            "physical_static": physical_static,
            "pue_core": physical["pue_core"],
            "forecast_anchors": self._forecast_anchors(batch, market),
            "forecast_residual_caps": self._forecast_caps(batch, market),
            "anchor_center_zonal": anchor_center,
            "state_dt": batch.get("state_dt"),
            "initial_state": batch.get(
                "initial_state",
                batch.get("state_sequence", batch.get("state", batch.get("state_features"))),
            ),
            "default_limits": batch.get("limits", batch.get("physical_limits")),
        }

    def _snapshot_mapping(
        self, value: Mapping[str, Any], reference: Tensor, index: Tensor
    ) -> Tensor:
        batch = reference.shape[0]

        def snapshot(name: str, width: int, *aliases: str) -> Tensor:
            raw = next(
                (value[key] for key in (name, *aliases) if value.get(key) is not None),
                None,
            )
            if raw is None:
                return reference.new_zeros((batch, width))
            tensor = torch.as_tensor(raw, device=reference.device, dtype=reference.dtype)
            if tensor.ndim == 0:
                tensor = tensor.reshape(1, 1).expand(batch, width)
            elif tensor.ndim == 1:
                if tensor.shape == (width,):
                    tensor = tensor[None].expand(batch, -1)
                elif tensor.shape == (batch,) and width == 1:
                    tensor = tensor[:, None]
            elif tensor.ndim == 3 and tensor.shape[1] == HORIZON:
                tensor = tensor[torch.arange(batch, device=reference.device), index]
            if tensor.shape == (batch, 1) and width > 1:
                tensor = tensor.expand(-1, width)
            if tensor.shape != (batch, width):
                raise ValueError(f"rolling state {name} must resolve to [batch,{width}]")
            return tensor

        return torch.cat(
            (
                snapshot("current_soc", 4, "soc"),
                snapshot("backlog", 4),
                snapshot("deadline_pressure", 4),
                snapshot("previous_grid", 4),
                snapshot("previous_battery", 4),
                snapshot("previous_compute", 4),
                snapshot("rt_innovation", 4, "revealed_rt"),
                snapshot("workload_innovation", 1, "revealed_workload"),
                snapshot("pv_innovation", 4, "revealed_pv"),
            ),
            dim=-1,
        )

    def _state_sequence(
        self,
        rolling_state: Any,
        reference: Tensor,
        origin: int | Tensor,
        *,
        issue_only: bool = False,
    ) -> tuple[Tensor, Tensor]:
        batch = reference.shape[0]
        index = torch.as_tensor(origin, device=reference.device)
        if index.ndim == 0:
            index = index.expand(batch)
        if index.shape != (batch,):
            raise ValueError("origin must be scalar or [batch]")
        index = index.long()
        if bool(((index < 0) | (index >= HORIZON)).any()):
            raise ValueError("origin must lie in [0,95]")
        if rolling_state is None:
            sequence = reference.new_zeros((batch, HORIZON, self.config.state_dim))
        elif isinstance(rolling_state, Mapping):
            self._assert_action_scope(rolling_state)
            snapshot = self._snapshot_mapping(rolling_state, reference, index)
            snapshot = F.pad(snapshot, (0, self.config.state_dim - snapshot.shape[-1]))
            sequence = reference.new_zeros((batch, HORIZON, self.config.state_dim))
            sequence[torch.arange(batch, device=reference.device), index] = snapshot
        else:
            tensor = torch.as_tensor(
                rolling_state, device=reference.device, dtype=reference.dtype
            )
            if tensor.ndim == 2:
                if tensor.shape[0] != batch or tensor.shape[-1] > self.config.state_dim:
                    raise ValueError("rolling state snapshot must be [batch,features]")
                tensor = F.pad(tensor, (0, self.config.state_dim - tensor.shape[-1]))
                sequence = reference.new_zeros((batch, HORIZON, self.config.state_dim))
                sequence[torch.arange(batch, device=reference.device), index] = tensor
            elif tensor.ndim == 3:
                if tensor.shape[:2] != (batch, HORIZON) or tensor.shape[-1] > self.config.state_dim:
                    raise ValueError("rolling state sequence must be [batch,96,features]")
                sequence = F.pad(tensor, (0, self.config.state_dim - tensor.shape[-1]))
                if issue_only:
                    first = sequence[:, 0].clone()
                    sequence = torch.zeros_like(sequence)
                    sequence[:, 0] = first
            else:
                raise ValueError("rolling state must be a mapping, [batch,F], or [batch,96,F]")
        periods = torch.arange(HORIZON, device=reference.device)[None]
        causal = sequence * (periods <= index[:, None]).unsqueeze(-1).to(sequence.dtype)
        return causal, index

    def _decode_forecasts(
        self, context: Mapping[str, Any], tokens: Tensor
    ) -> dict[str, Any]:
        forecasts = self.forecast_decoder(
            tokens,
            context["pue_core"],
            da_anchor=context["forecast_anchors"].get("da"),
            pv_anchor=context["forecast_anchors"].get("pv"),
            residual_caps=context["forecast_residual_caps"],
        )
        if self.physical_stream.formal_artifact_backed:
            pue = self.physical_stream.pue_from_total_workload(
                context["physical_static"], forecasts["workload"]
            )
            zonal = torch.cat(
                (
                    forecasts["da"],
                    forecasts["rt"],
                    forecasts["dart"],
                    forecasts["workload"],
                    forecasts["pv"],
                    pue,
                ),
                dim=-1,
            )
            forecasts = {**forecasts, "pue": pue, "zonal": zonal}
        return forecasts

    def _world(
        self,
        context: Mapping[str, Any],
        rolling_state: Any,
        *,
        origin: int | Tensor,
        issue_only: bool = False,
    ) -> dict[str, Any]:
        state_sequence, index = self._state_sequence(
            rolling_state,
            context["market_tokens"],
            origin,
            issue_only=issue_only,
        )
        state_tokens, state_memory = self.state_encoder(
            state_sequence, dt=context.get("state_dt")
        )
        fusion = self.cross_fusion(
            context["market_tokens"],
            context["workload_tokens"],
            context["physical_tokens"],
            state_tokens,
        )
        routing = self.moe(
            fusion["tokens"], context["market_tokens"], state_tokens, origin=index
        )
        forecasts = self._decode_forecasts(context, routing["tokens"])
        world = self.distribution(
            routing["tokens"],
            context["anchor_center_zonal"],
            representation_center_zonal=forecasts["zonal"],
            origin=index,
        )
        state_hidden = state_tokens.gather(
            1, index[:, None, None].expand(-1, 1, self.config.hidden_dim)
        ).squeeze(1)
        return {
            "world": world,
            "state_hidden": state_hidden,
            "state_tokens": state_tokens,
            "state_memory": state_memory,
            "fusion": fusion,
            "routing": routing,
            "forecasts": forecasts,
            "origin_index": index,
        }

    @staticmethod
    def _distribution_output(
        world_output: Mapping[str, Any], sampled: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        world: BaselineAnchoredWorld = world_output["world"]
        result = {
            **world.as_dict(),
            "state_hidden": world_output["state_hidden"],
            "token_features": world_output["routing"]["tokens"],
        }
        if sampled is not None:
            result.update(sampled)
        return result

    @staticmethod
    def _revealed(batch: Mapping[str, Any]) -> tuple[Any | None, Any | None]:
        nested = batch.get("revealed")
        if isinstance(nested, Mapping):
            return (
                nested.get("zonal", nested.get("revealed_zonal")),
                nested.get("mask", nested.get("revealed_mask")),
            )
        return batch.get("revealed_zonal"), batch.get("revealed_mask")

    def forward_distribution(
        self,
        batch: Mapping[str, Any],
        sample_count: int | None = None,
        *,
        base_indices: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run Distribution only; Policy is not constructed or evaluated."""

        self._assert_action_scope(batch)
        context = self.prepare_issue_context(batch)
        origin = batch.get("origin", 0)
        rolling_state = batch.get(
            "rolling_state",
            batch.get("state_sequence", batch.get("state", batch.get("state_features"))),
        )
        world_output = self._world(context, rolling_state, origin=origin)
        sampled = None
        if sample_count is not None:
            revealed, revealed_mask = self._revealed(batch)
            sampled = self.distribution.sample(
                world_output["world"],
                int(sample_count),
                base_indices=base_indices,
                noise=noise,
                generator=generator,
                seed=seed,
                revealed_zonal=revealed,
                revealed_mask=revealed_mask,
            )
        distribution = self._distribution_output(world_output, sampled)
        return {
            "architecture_version": self.architecture_version,
            "issue_context": context,
            "distribution": distribution,
            "forecasts": world_output["forecasts"],
            "representation": {
                "fusion": world_output["fusion"],
                "routing": world_output["routing"],
                "state_tokens": world_output["state_tokens"],
            },
            "state_hidden": world_output["state_hidden"],
        }

    def forward_issue(
        self,
        batch: Mapping[str, Any],
        *,
        limits: Mapping[str, Any] | PhysicalActionLimits | None = None,
        sample_count: int | None = None,
        base_indices: Tensor | None = None,
        noise: Tensor | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Issue and freeze DA before any realized current-day value exists."""

        self._assert_action_scope(batch)
        self._assert_issue_time(batch)
        context = self.prepare_issue_context(batch)
        world_output = self._world(
            context, context.get("initial_state"), origin=0, issue_only=True
        )
        memory = self.distribution.decision_memory(
            world_output["world"], detach_truth=True
        )
        policy = self.policy(
            memory,
            world_output["state_hidden"],
            origin=0,
            limits=limits if limits is not None else context.get("default_limits"),
        )
        sampled = None
        if sample_count is not None:
            sampled = self.distribution.sample(
                world_output["world"],
                int(sample_count),
                base_indices=base_indices,
                noise=noise,
                seed=seed,
            )
        issued_context = {
            **context,
            "issued_day_ahead": policy["day_ahead"],
        }
        return {
            "architecture_version": self.architecture_version,
            "issue_context": issued_context,
            "issued_day_ahead": policy["day_ahead"],
            "distribution": self._distribution_output(world_output, sampled),
            "forecasts": world_output["forecasts"],
            "policy": policy,
            "actions": policy["free_actions"],
            "critic": policy["critic"],
        }

    def forward_origin(
        self,
        issue_context: Mapping[str, Any],
        rolling_state: Any,
        origin: int | Tensor,
        *,
        limits: Mapping[str, Any] | PhysicalActionLimits | None = None,
        issued_day_ahead: Tensor | None = None,
        sample_count: int | None = None,
        base_indices: Tensor | None = None,
        noise: Tensor | None = None,
        seed: int | None = None,
        revealed_zonal: Tensor | None = None,
        revealed_mask: Tensor | None = None,
        detach_distribution_core: bool = True,
    ) -> dict[str, Any]:
        """Recondition the future world and act at one rolling origin."""

        context = issue_context.get("issue_context", issue_context)
        if not isinstance(context, Mapping) or not context.get("_v7_issue_context"):
            raise ValueError("forward_origin requires a V7 issue_context")
        self._assert_action_scope(context)
        # Decision training deliberately truncates gradients at the dynamic
        # world-state boundary.  Representation and Distribution-truth are
        # frozen for a decision update; retaining their 96-token graph at each
        # sampled origin only wastes VRAM and accidentally creates long BPTT
        # through the analytic shield.  The Decision adapter and Policy below
        # remain fully differentiable.
        if detach_distribution_core:
            with torch.no_grad():
                world_output = self._world(context, rolling_state, origin=origin)
        else:
            world_output = self._world(context, rolling_state, origin=origin)
        frozen_da = issued_day_ahead
        if frozen_da is None:
            frozen_da = context.get("issued_day_ahead")
        memory = self.distribution.decision_memory(
            world_output["world"], detach_truth=detach_distribution_core
        )
        policy = self.policy(
            memory,
            world_output["state_hidden"],
            origin=origin,
            limits=limits if limits is not None else context.get("default_limits"),
            issued_day_ahead=frozen_da,
        )
        sampled = None
        if sample_count is not None:
            sampled = self.distribution.sample(
                world_output["world"],
                int(sample_count),
                base_indices=base_indices,
                noise=noise,
                seed=seed,
                revealed_zonal=revealed_zonal,
                revealed_mask=revealed_mask,
            )
        return {
            "architecture_version": self.architecture_version,
            "distribution": self._distribution_output(world_output, sampled),
            "forecasts": world_output["forecasts"],
            "policy": policy,
            "actions": policy["free_actions"],
            "critic": policy["critic"],
            "state_hidden": world_output["state_hidden"],
        }

    @staticmethod
    def _extract_world(value: Any) -> BaselineAnchoredWorld:
        if isinstance(value, BaselineAnchoredWorld):
            return value
        if isinstance(value, Mapping):
            if isinstance(value.get("world"), BaselineAnchoredWorld):
                return value["world"]
            if isinstance(value.get("world_latent"), BaselineAnchoredWorld):
                return value["world_latent"]
            nested = value.get("distribution")
            if nested is not None:
                return BaselineAnchoredJointWorldPolicyMoEV7._extract_world(nested)
        raise ValueError("could not find a V7 BaselineAnchoredWorld")

    def sample_distribution(
        self,
        world_or_output: Any,
        sample_count: int,
        *,
        base_indices: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        revealed_zonal: Tensor | None = None,
        revealed_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        world = self._extract_world(world_or_output)
        return self.distribution.sample(
            world,
            sample_count,
            base_indices=base_indices,
            noise=noise,
            generator=generator,
            seed=seed,
            revealed_zonal=revealed_zonal,
            revealed_mask=revealed_mask,
        )

    def forward_decision(
        self,
        batch_or_context: Mapping[str, Any],
        *,
        distribution: Any | None = None,
        origin: int | Tensor | None = None,
        rolling_state: Any | None = None,
        limits: Mapping[str, Any] | PhysicalActionLimits | None = None,
        issued_day_ahead: Tensor | None = None,
        detach_distribution_core: bool = True,
    ) -> dict[str, Any]:
        """Run the decision optimizer boundary with an optional frozen world."""

        root = batch_or_context
        context = root.get("issue_context", root)
        if not isinstance(context, Mapping) or not context.get("_v7_issue_context"):
            context = self.prepare_issue_context(root)
        resolved_origin = root.get("origin", 0) if origin is None else origin
        if rolling_state is None:
            rolling_state = root.get(
                "rolling_state",
                root.get("state_sequence", root.get("state", context.get("initial_state"))),
            )
        if distribution is None:
            world_output = self._world(
                context, rolling_state, origin=resolved_origin
            )
            world = world_output["world"]
            state_hidden = world_output["state_hidden"]
            distribution_output = self._distribution_output(world_output)
            forecasts = world_output["forecasts"]
        else:
            world = self._extract_world(distribution)
            candidate = distribution.get("state_hidden") if isinstance(distribution, Mapping) else None
            if candidate is None and isinstance(root.get("distribution"), Mapping):
                candidate = root["distribution"].get("state_hidden")
            if candidate is None:
                state_sequence, index = self._state_sequence(
                    rolling_state, context["market_tokens"], resolved_origin
                )
                state_tokens, _ = self.state_encoder(
                    state_sequence, dt=context.get("state_dt")
                )
                state_hidden = state_tokens.gather(
                    1, index[:, None, None].expand(-1, 1, self.config.hidden_dim)
                ).squeeze(1)
            else:
                state_hidden = candidate
            distribution_output = (
                distribution.get("distribution", distribution)
                if isinstance(distribution, Mapping)
                else world.as_dict()
            )
            forecasts = root.get("forecasts")
        memory = self.distribution.decision_memory(
            world, detach_truth=detach_distribution_core
        )
        frozen_da = issued_day_ahead
        if frozen_da is None:
            frozen_da = context.get("issued_day_ahead", root.get("issued_day_ahead"))
        policy = self.policy(
            memory,
            state_hidden,
            origin=resolved_origin,
            limits=limits
            if limits is not None
            else root.get("limits", context.get("default_limits")),
            issued_day_ahead=frozen_da,
        )
        return {
            "architecture_version": self.architecture_version,
            "issue_context": context,
            "distribution": distribution_output,
            "forecasts": forecasts,
            "policy": policy,
            "actions": policy["free_actions"],
            "critic": policy["critic"],
            "state_hidden": state_hidden,
            "distribution_core_detached": bool(detach_distribution_core),
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        phase: str = "joint",
        sample_count: int | None = None,
        base_indices: Tensor | None = None,
        noise: Tensor | None = None,
        seed: int | None = None,
        detach_distribution_core: bool = True,
        limits: Mapping[str, Any] | PhysicalActionLimits | None = None,
        issued_day_ahead: Tensor | None = None,
    ) -> dict[str, Any]:
        phase = str(phase).lower()
        if phase == "distribution":
            return self.forward_distribution(
                batch,
                sample_count,
                base_indices=base_indices,
                noise=noise,
                seed=seed,
            )
        if phase == "decision":
            return self.forward_decision(
                batch,
                origin=batch.get("origin", 0),
                limits=limits,
                issued_day_ahead=issued_day_ahead,
                detach_distribution_core=detach_distribution_core,
            )
        if phase != "joint":
            raise ValueError("phase must be distribution, decision, or joint")
        distribution_output = self.forward_distribution(
            batch,
            sample_count,
            base_indices=base_indices,
            noise=noise,
            seed=seed,
        )
        decision = self.forward_decision(
            distribution_output,
            distribution=distribution_output["distribution"],
            origin=batch.get("origin", 0),
            rolling_state=batch.get(
                "rolling_state",
                batch.get("state_sequence", batch.get("state", batch.get("state_features"))),
            ),
            limits=limits if limits is not None else batch.get("limits"),
            issued_day_ahead=issued_day_ahead,
            detach_distribution_core=detach_distribution_core,
        )
        return {
            **distribution_output,
            "policy": decision["policy"],
            "actions": decision["actions"],
            "critic": decision["critic"],
            "distribution_core_detached": decision["distribution_core_detached"],
        }

    def distribution_truth_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, _ in self.named_parameters()
            if name.startswith("distribution.truth_")
        )

    def decision_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, _ in self.named_parameters()
            if name.startswith("distribution.decision_adapter.")
            or name.startswith("policy.")
        )

    def representation_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, _ in self.named_parameters()
            if any(name.startswith(prefix) for prefix in self.REPRESENTATION_PREFIXES)
        )

    def a1_parameter_names(self) -> tuple[str, ...]:
        """Strict DART-only adapter allowlist for the A1 ablation."""

        return tuple(
            name
            for name, _ in self.named_parameters()
            if any(name.startswith(prefix) for prefix in self.A1_PREFIXES)
        )

    def a2_parameter_names(self) -> tuple[str, ...]:
        """Joint adapters, Decision MoE/Policy and representation router only."""

        selected = set(self.distribution_truth_parameter_names())
        selected.update(self.decision_parameter_names())
        selected.update(
            name
            for name, _ in self.named_parameters()
            if any(name.startswith(prefix) for prefix in self.A2_EXTRA_PREFIXES)
        )
        return tuple(sorted(selected))

    def configure_trainable_phase(self, phase: str) -> tuple[str, ...]:
        phase = phase.lower()
        truth = set(self.distribution_truth_parameter_names())
        decision = set(self.decision_parameter_names())
        representation = set(self.representation_parameter_names())
        if phase in {"d0", "distribution", "calibration_restore"}:
            selected = truth
        elif phase in {"p0", "decision", "policy"}:
            selected = decision
        elif phase in {"j1", "j2", "j3", "joint"}:
            selected = truth | decision
        elif phase == "representation":
            selected = representation
        elif phase == "a1":
            selected = set(self.a1_parameter_names())
        elif phase == "a2":
            selected = set(self.a2_parameter_names())
        else:
            raise ValueError(f"unknown V7 phase {phase!r}")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in selected)
        self.physical_stream.freeze_twin()
        return tuple(sorted(selected))

    def load_representation_predecessor(
        self,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        expected_epoch: int = 28,
        require_capability_passed: bool = True,
    ) -> dict[str, Any]:
        """Allowlist-load only the selected Representation/forecast trunk."""

        if isinstance(checkpoint, Mapping):
            payload = dict(checkpoint)
            source = "in_memory"
        else:
            source_path = Path(checkpoint).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            raw = torch.load(source_path, map_location="cpu", weights_only=False)
            if not isinstance(raw, Mapping):
                raise ValueError("Representation checkpoint payload is malformed")
            payload = dict(raw)
            source = str(source_path)
        if payload.get("phase") not in {None, "representation"}:
            raise ValueError("only a Representation checkpoint may seed V7")
        epoch = payload.get(
            "selected_candidate_epoch",
            payload.get(
                "selected_epoch", payload.get("epochs_completed", payload.get("epoch"))
            ),
        )
        if epoch is None or int(epoch) != int(expected_epoch):
            raise ValueError(f"V7 requires selected Representation epoch {expected_epoch}")
        if require_capability_passed and (
            payload.get("capability_gate_status") != "PASSED"
            or payload.get("checkpoint_role") != "capability_passed_selected_best"
        ):
            raise ValueError("Representation predecessor must be the capability-passed selected best")
        if payload.get("architecture_version", REPRESENTATION_ARCHITECTURE_VERSION) != (
            REPRESENTATION_ARCHITECTURE_VERSION
        ):
            raise ValueError("Representation predecessor has the wrong architecture")
        raw_state = payload.get("model_state")
        if not isinstance(raw_state, Mapping):
            raise ValueError("Representation checkpoint has no model_state mapping")
        current = self.state_dict()
        allowed: dict[str, Tensor] = {}
        rejected: list[str] = []
        for raw_name, raw_value in raw_state.items():
            name = str(raw_name)
            if any(name.startswith(prefix) for prefix in self.REPRESENTATION_PREFIXES):
                if name not in current:
                    raise ValueError(f"unknown Representation trunk key {name!r}")
                tensor = torch.as_tensor(raw_value)
                if tensor.shape != current[name].shape:
                    raise ValueError(f"Representation key {name!r} has incompatible shape")
                allowed[name] = tensor
            else:
                rejected.append(name)
        required = set(self.representation_parameter_names())
        missing = sorted(required - set(allowed))
        if missing:
            raise ValueError(
                "Representation predecessor misses allowlisted parameters: "
                + ", ".join(missing[:5])
            )
        self.load_state_dict(allowed, strict=False)
        self.physical_stream.freeze_twin()
        return {
            "source": source,
            "source_architecture": REPRESENTATION_ARCHITECTURE_VERSION,
            "target_architecture": self.architecture_version,
            "selected_representation_epoch": int(epoch),
            "loaded_key_count": len(allowed),
            "loaded_keys": tuple(sorted(allowed)),
            "rejected_predecessor_keys": tuple(sorted(rejected)),
            "old_distribution_loaded": False,
            "old_policy_loaded": False,
        }


R4V7Model = BaselineAnchoredJointWorldPolicyMoEV7


__all__ = [
    "ARCHITECTURE_VERSION",
    "BaselineAnchoredJointWorldPolicyMoEV7",
    "R4V7Model",
    "V7ModelConfig",
]
