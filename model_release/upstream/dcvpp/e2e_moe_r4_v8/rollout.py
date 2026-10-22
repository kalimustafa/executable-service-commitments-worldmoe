"""V8 typed-packet adapter for the proven V7 differentiable rollout.

Distribution scenarios are detached before they become exogenous rollout
worlds.  Policy tensors remain attached, so post-shield cost gradients reach
only the V8 decision bridge, actor, critic and Policy routers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass
from typing import Any

import torch
from torch import Tensor, nn

from dcvpp.e2e_moe_r4_v7.rollout import differentiable_training_rollout

from .packets import DistributionSamplePacket, DistributionTruthPacket, PolicyPacket


ROLLOUT_ADAPTER_SCHEMA = "dcvpp.r4.v8.v7-rollout-adapter.v1"


class V8RolloutError(RuntimeError):
    """Raised when typed V8 outputs cannot satisfy the V7 rollout contract."""


def _critic_mapping(packet: PolicyPacket) -> dict[str, Tensor]:
    return {
        "expected_cost": packet.critic[:, 0],
        "var": packet.critic[:, 1],
        "cvar": packet.critic[:, 2],
        "sla_risk": packet.critic[:, 3],
        "shield_intervention_risk": packet.critic[:, 4],
    }


def _policy_output_mapping(value: Any, *, issue: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V8RolloutError("V8 forward output must be a mapping")
    packet = value.get("policy")
    if isinstance(packet, PolicyPacket):
        critic = _critic_mapping(packet)
        free_actions = {
            "day_ahead": packet.day_ahead_plan,
            "battery": packet.current_battery,
            "compute": packet.current_compute,
        }
        policy = {
            "day_ahead": packet.day_ahead_plan,
            "battery": packet.current_battery,
            "compute": packet.current_compute,
            "battery_plan": packet.battery_plan,
            "compute_plan": packet.compute_plan,
            "free_actions": free_actions,
            "actions": free_actions,
            "critic": critic,
        }
        result: dict[str, Any] = {
            "policy": policy,
            "actions": free_actions,
            "critic": critic,
            "battery_plan": packet.battery_plan,
            "compute_plan": packet.compute_plan,
        }
        if issue:
            context = value.get("issue_context")
            if not isinstance(context, Mapping):
                raise V8RolloutError("V8 forward_issue omitted issue_context")
            result["issue_context"] = context
            result["issued_day_ahead"] = packet.day_ahead_plan
        return result

    # Compatibility with a future V8 wrapper that already emits the V7 mapping.
    if isinstance(packet, Mapping):
        result = dict(value)
        if issue and not isinstance(result.get("issue_context"), Mapping):
            raise V8RolloutError("V8 forward_issue omitted issue_context")
        return result
    raise V8RolloutError("V8 output lacks a typed or mapping Policy payload")


class _V8PolicyAdapter:
    """Expose V8 typed Policy packets through the V7 action mapping contract."""

    def __init__(self, model: nn.Module, *, detach_shared: bool) -> None:
        self.model = model
        self.detach_shared = bool(detach_shared)

    @staticmethod
    def _issue_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(batch)
        if result.get("initial_state") is not None and not any(
            result.get(name) is not None
            for name in ("state", "state_sequence", "state_features")
        ):
            result["state"] = result["initial_state"]
        return result

    def forward_issue(
        self, batch: Mapping[str, Any], *, limits: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        callback = getattr(self.model, "forward_issue", None)
        if not callable(callback):
            raise V8RolloutError("V8 model must expose forward_issue")
        output = callback(
            self._issue_batch(batch),
            limits=limits,
            detach_shared=self.detach_shared,
        )
        return _policy_output_mapping(output, issue=True)

    def forward_origin(
        self,
        issue_context: Mapping[str, Any],
        rolling_state: Any,
        origin: int,
        *,
        limits: Mapping[str, Any] | None = None,
        issued_day_ahead: Tensor | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        callback = getattr(self.model, "forward_origin", None)
        if not callable(callback):
            raise V8RolloutError("V8 model must expose forward_origin")
        output = callback(
            issue_context=issue_context,
            rolling_state=rolling_state,
            origin=origin,
            limits=limits,
            issued_day_ahead=issued_day_ahead,
            detach_shared=self.detach_shared,
        )
        return _policy_output_mapping(output, issue=False)


def _packet_tensors(value: Any) -> tuple[Tensor, Tensor, Tensor | None] | None:
    if isinstance(value, DistributionSamplePacket):
        packet = value.validate()
        return packet.public_samples, packet.weights, packet.base_indices
    # Duck-typed support for the registered DistributionScenarioPacket name
    # used by the trainer-facing protocol and future packet revisions.
    if is_dataclass(value) or value.__class__.__name__ == "DistributionScenarioPacket":
        samples = getattr(value, "public_samples", getattr(value, "samples", None))
        weights = getattr(value, "weights", None)
        indices = getattr(value, "base_indices", None)
        if isinstance(samples, Tensor) and isinstance(weights, Tensor):
            return samples, weights, indices if isinstance(indices, Tensor) else None
    return None


def _materialize_scenarios(
    model: nn.Module,
    distribution_output: Any,
    *,
    sample_count: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    direct = _packet_tensors(distribution_output)
    truth: DistributionTruthPacket | None = None
    if direct is None and isinstance(distribution_output, Mapping):
        candidate = distribution_output.get("samples")
        direct = _packet_tensors(candidate)
        if direct is None and isinstance(candidate, Tensor):
            weights = distribution_output.get("weights")
            if weights is None:
                weights = candidate.new_full(
                    candidate.shape[:2], 1.0 / float(candidate.shape[1])
                )
            direct = (candidate, torch.as_tensor(weights), None)
        possible_truth = distribution_output.get("distribution")
        if isinstance(possible_truth, DistributionTruthPacket):
            truth = possible_truth
        nested = distribution_output.get("distribution")
        if direct is None and isinstance(nested, Mapping):
            nested_candidate = nested.get("samples")
            direct = _packet_tensors(nested_candidate)
            if direct is None and isinstance(nested_candidate, Tensor):
                weights = nested.get("weights")
                if weights is None:
                    weights = nested_candidate.new_full(
                        nested_candidate.shape[:2],
                        1.0 / float(nested_candidate.shape[1]),
                    )
                direct = (nested_candidate, torch.as_tensor(weights), None)
    elif isinstance(distribution_output, DistributionTruthPacket):
        truth = distribution_output

    if direct is None and truth is not None:
        distribution_plane = getattr(model, "distribution", None)
        sampler = getattr(distribution_plane, "sample", None)
        if not callable(sampler):
            raise V8RolloutError("V8 model cannot sample its DistributionTruthPacket")
        with torch.no_grad():
            direct = _packet_tensors(
                sampler(truth, int(sample_count), seed=int(seed))
            )
    if direct is None:
        raise V8RolloutError(
            "distribution_output lacks typed scenarios or a sampleable truth packet"
        )

    samples, weights, indices = direct
    samples = samples.detach()
    weights = weights.to(device=samples.device, dtype=samples.dtype).detach()
    if samples.ndim != 4 or samples.shape[2:] != (96, 21):
        raise V8RolloutError("V8 rollout samples must be [day,world,96,21]")
    if weights.ndim == 1 and weights.shape[0] == samples.shape[1]:
        weights = weights[None].expand(samples.shape[0], -1)
    if weights.shape != samples.shape[:2]:
        raise V8RolloutError("V8 rollout weights must be [day,world]")
    if indices is not None:
        indices = indices.to(samples.device)
        if indices.ndim == 1 and indices.shape[0] == samples.shape[1]:
            indices = indices[None].expand(samples.shape[0], -1)
        if indices.shape != samples.shape[:2]:
            raise V8RolloutError("V8 rollout base_indices must be [day,world]")
    requested = int(sample_count)
    available = int(samples.shape[1])
    if requested < 2:
        raise V8RolloutError("sample_count must be at least two")
    if available < requested:
        raise V8RolloutError(
            f"Distribution supplied {available} worlds but {requested} were requested"
        )
    if available > requested:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        selected_cpu = torch.randperm(available, generator=generator)[:requested]
        selected = selected_cpu.to(samples.device)
        samples = samples.index_select(1, selected)
        weights = weights.index_select(1, selected)
        if indices is not None:
            indices = indices.index_select(1, selected)
    if bool((weights < 0).any()) or bool((weights.sum(1) <= 0).any()):
        raise V8RolloutError("scenario weights must be nonnegative with positive rows")
    weights = weights / weights.sum(1, keepdim=True)
    return samples, weights, None if indices is None else indices.detach()


def distribution_scenarios_to_mapping(
    model: nn.Module,
    distribution_output: Any,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, Tensor]:
    """Convert a typed V8 scenario/truth output to the V7 rollout mapping."""

    samples, weights, indices = _materialize_scenarios(
        model,
        distribution_output,
        sample_count=sample_count,
        seed=seed,
    )
    result = {
        "samples": samples,
        "samples_zonal": samples,
        "weights": weights,
    }
    if indices is not None:
        result["base_indices"] = indices
    return result


def rollout_fn(
    model: nn.Module,
    batch: Mapping[str, Any],
    distribution_output: Any,
    *,
    stage_id: str,
    horizon: int,
    origin_budget: int | None,
    sample_count: int,
    seed: int,
) -> Mapping[str, Any]:
    """Run the complete differentiable V7 shield rollout with V8 packets."""

    distribution = distribution_scenarios_to_mapping(
        model,
        distribution_output,
        sample_count=int(sample_count),
        seed=int(seed),
    )
    result = differentiable_training_rollout(
        _V8PolicyAdapter(
            model,
            detach_shared=str(stage_id).lower() not in {"j1", "j2", "j3"},
        ),
        batch,
        distribution,
        horizon=int(horizon),
        phase=str(stage_id),
        origin_budget=origin_budget,
    )
    return {
        **result,
        "v8_rollout_adapter_schema": ROLLOUT_ADAPTER_SCHEMA,
        "requested_sample_count": int(sample_count),
        "rollout_seed": int(seed),
    }


__all__ = [
    "ROLLOUT_ADAPTER_SCHEMA",
    "V8RolloutError",
    "distribution_scenarios_to_mapping",
    "rollout_fn",
]
