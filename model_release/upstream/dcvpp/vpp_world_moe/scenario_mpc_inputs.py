"""Causal native-W1 joint scenarios for the batch-one scenario MPC interface.

Only a frozen W1 expert packet, issue-time APIC anchors/caps and declared risk
preferences enter this adapter. Realized trajectories and target-bearing batch
mappings are deliberately absent from its public interface.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .packets import WorldActionPacket
from .r_ecmoe import RiskConditionPacket, W1ExpertPacket
from .r_ecmoe_evaluation import _safe_raw


SCENARIO_COUNT = 20
TIMELINE = 96
NODES = 4
PHYSICAL_VECTOR_NAMES = (
    "battery_power_limit", "battery_energy_capacity", "initial_soc", "compute_capacity",
    "initial_queue", "pv_capacity", "grid_import_limit", "grid_export_limit", "pue_min",
    "pue_max", "idle_power", "work_power_per_unit", "charge_efficiency", "discharge_efficiency",
)
FORECAST_NAMES = ("da", "rt", "arrivals", "pv", "pue")


class ScenarioMPCInputError(ValueError):
    """A causal packet or native rollout cannot satisfy the exact SMPC contract."""


def _finite_tensor(value: Any, name: str, shape: tuple[int, ...] | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise ScenarioMPCInputError(f"{name} must be a tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ScenarioMPCInputError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ScenarioMPCInputError(f"{name} contains nonfinite values")
    return value


def _numpy(value: Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float64).numpy().copy()


def _slice_actions(actions: WorldActionPacket, origin: int, horizon: int) -> WorldActionPacket:
    """Slice the time axis after B, including the extra SOC/queue endpoint."""
    if not isinstance(actions, WorldActionPacket):
        raise ScenarioMPCInputError("the registered assembler must return WorldActionPacket actions")
    actions.validate()
    stop = origin + horizon
    result = {}
    for field in fields(WorldActionPacket):
        value = getattr(actions, field.name)
        full_time = TIMELINE + 1 if field.name in {"soc", "queue"} else TIMELINE
        if value.shape[0] != 1 or value.shape[1] != full_time:
            raise ScenarioMPCInputError(f"assembler action {field.name} has the wrong batch/time axes")
        _finite_tensor(value, f"actions.{field.name}")
        result[field.name] = value[:, origin:stop + (field.name in {"soc", "queue"})].contiguous()
    return WorldActionPacket(**result).validate()


def _issue_anchor(value: Tensor, origin: int, horizon: int, device: torch.device) -> tuple[Tensor, str]:
    value = _finite_tensor(value, "issue DA anchor").to(device=device, dtype=torch.float32)
    if value.shape == (1, TIMELINE, NODES):
        return value[:, origin:origin + horizon].contiguous(), "slice_B_T_4_on_time_axis"
    if value.shape == (1, NODES):
        return value[:, None, :].expand(1, horizon, NODES).contiguous(), "broadcast_B_4_over_active_time"
    raise ScenarioMPCInputError("issue DA anchor must be [1,96,4] or [1,4]; scenario axes are not accepted")


def _issue_cap(value: Tensor, device: torch.device) -> tuple[Tensor, str]:
    value = _finite_tensor(value, "issue DA residual cap").to(device=device, dtype=torch.float32)
    if value.ndim == 0 or value.shape in {(NODES,), (1, NODES)}:
        cap, mode = value, "native_scalar_or_node_cap"
    elif value.shape == (1, TIMELINE, NODES):
        # The native APIC cap broadcasts node/batch axes, not time. Explicitly
        # collapse only a truly time-constant supplied cap; never reinterpret
        # H=20 as the 20-scenario dimension.
        cap = value[:, 0, :]
        if not torch.equal(value, cap[:, None, :].expand_as(value)):
            raise ScenarioMPCInputError(
                "native W1 APIC decoder does not support time-varying [1,96,4] caps; "
                "refusing to map time onto scenarios or replace the decoder"
            )
        mode = "explicitly_collapsed_time_constant_B_T_4_cap"
    else:
        raise ScenarioMPCInputError("issue DA cap must be scalar, [4], [1,4], or time-constant [1,96,4]")
    if bool((cap <= 0).any()):
        raise ScenarioMPCInputError("issue DA residual caps must be positive")
    return cap.contiguous(), mode


def _input_hash(packet: W1ExpertPacket, risk: RiskConditionPacket, anchor: Tensor, cap: Tensor, seed: int) -> str:
    digest = hashlib.sha256()

    def add(name: str, value: Tensor) -> None:
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

    add("source_state.pooled_state", packet.source_state.pooled_state)
    add("origin_index", packet.origin_index)
    add("previous_actions", packet.previous_actions)
    add("valid_action_masks", packet.valid_action_masks)
    for group, instance in (("reference", packet.reference), ("constraints", packet.constraints), ("risk", risk)):
        for field in fields(instance):
            value = getattr(instance, field.name)
            if isinstance(value, Tensor):
                add(f"{group}.{field.name}", value)
    add("provided_issue_da_anchor", anchor)
    add("provided_issue_da_residual_cap", cap)
    digest.update(json.dumps({"scenario_seed": seed, "path_count": SCENARIO_COUNT}, sort_keys=True).encode())
    return digest.hexdigest()


@torch.inference_mode()
def generate_w1_joint_scenarios(
    model: Any,
    packet: W1ExpertPacket,
    risk: RiskConditionPacket,
    da_anchor: Tensor,
    da_residual_cap: Tensor,
    scenario_seed: int = 20240813,
) -> dict[str, Any]:
    """Return 20 native joint paths and causal physical inputs for one suffix.

    Forecasts are NumPy float64 arrays [20,96,4]. Only origin:origin+horizon
    contains decoded paths; inactive entries are zero (PUE uses its physical
    minimum). Weights are the normalized native W1 Monte Carlo weights, not
    learned weights from a historical R3 scenario model.
    """
    if not isinstance(packet, W1ExpertPacket) or not isinstance(risk, RiskConditionPacket):
        raise ScenarioMPCInputError("only W1ExpertPacket and RiskConditionPacket inputs are accepted")
    if packet.state_latent.ndim != 3 or packet.state_latent.shape[:2] != (1, TIMELINE):
        raise ScenarioMPCInputError("scenario generation requires batch 1 and a 96-step expert timeline")
    if isinstance(scenario_seed, bool) or int(scenario_seed) != scenario_seed:
        raise ScenarioMPCInputError("scenario_seed must be an integer")
    seed = int(scenario_seed)
    width = int(packet.state_latent.shape[-1])
    _finite_tensor(packet.source_state.pooled_state, "causal pooled W1 state", (1, width))
    _finite_tensor(packet.origin_index, "origin_index", (1,))
    origin = int(packet.origin_index[0].detach().cpu())
    if origin != float(packet.origin_index[0].detach().cpu()):
        raise ScenarioMPCInputError("origin_index must be integral")
    risk.validate(1, TIMELINE)
    horizon, stage = int(risk.horizon[0].detach().cpu()), int(risk.stage[0].detach().cpu())
    if origin < 0 or horizon < 1 or origin + horizon > TIMELINE or stage not in (0, 1):
        raise ScenarioMPCInputError("invalid causal stage/origin/horizon")
    if stage == 0 and origin != 0:
        raise ScenarioMPCInputError("the registered DA decision starts at origin 0")
    packet.constraints.validate(1, strict_values=True)
    _finite_tensor(packet.previous_actions, "causal previous actions", (1, TIMELINE, NODES, 6))
    if packet.valid_action_masks.shape != packet.previous_actions.shape or packet.valid_action_masks.dtype != torch.bool:
        raise ScenarioMPCInputError("causal action masks must be boolean [1,96,4,6]")
    if not bool(packet.valid_action_masks[:, origin:origin + horizon, :, 0].all()):
        raise ScenarioMPCInputError("active-window published DA quantities are unavailable; no realized fallback is permitted")
    world = model.world_model
    maximum_paths = getattr(getattr(world, "dynamics", None), "max_path_count", None)
    if maximum_paths is not None and int(maximum_paths) < SCENARIO_COUNT:
        raise ScenarioMPCInputError(f"native W1 supports only {maximum_paths} paths, not the required 20")
    queries = getattr(world, "observation_queries", None)
    if not callable(getattr(queries, "from_latent_timeline", None)):
        raise ScenarioMPCInputError("native world_model.observation_queries.from_latent_timeline is unavailable")
    if bool(getattr(world.config, "support_atom_enabled", False)):
        raise ScenarioMPCInputError("this helper is for the legacy continuous W1 decoder, not a support-atom U0 lineage")
    active_anchor, anchor_mode = _issue_anchor(da_anchor, origin, horizon, packet.state_latent.device)
    active_cap, cap_mode = _issue_cap(da_residual_cap, packet.state_latent.device)
    causal_hash = _input_hash(packet, risk, da_anchor, da_residual_cap, seed)
    assembled = model.assembler(_safe_raw(packet), packet, risk)
    actions = _slice_actions(assembled.actions, origin, horizon)
    rollout = world.imagine(packet.source_state, actions, packet.constraints,
                            horizon=horizon, path_count=SCENARIO_COUNT, seed=seed, compute_path_cost=False)
    latent = _finite_tensor(rollout.latent_paths, "native joint latent paths", (1, SCENARIO_COUNT, horizon, width))
    # Native imagine's internal observation call omits APIC anchors. Decode the
    # same sampled latent paths again with causal issue anchors; no new noise,
    # target statistics or independently sampled marginal paths are introduced.
    decoded = queries.from_latent_timeline(latent, packet.constraints,
                                          da_anchor=active_anchor, da_residual_cap=active_cap)
    active = {name: _numpy(_finite_tensor(getattr(decoded, name), f"decoded.{name}",
                                         (1, SCENARIO_COUNT, horizon, NODES)))[0]
              for name in FORECAST_NAMES}
    dart = _finite_tensor(decoded.dart, "decoded.dart", (1, SCENARIO_COUNT, horizon, NODES))
    forward_identity = float(((decoded.da - decoded.rt) - dart).abs().max().detach().cpu())
    if forward_identity != 0.0:
        raise ScenarioMPCInputError("anchored native W1 observations do not preserve the forward-dtype APIC identity")
    if (active["arrivals"] < 0).any() or (active["pv"] < 0).any() or (active["pue"] < 1).any():
        raise ScenarioMPCInputError("native decoded paths violate nonnegative workload/PV or PUE >= 1 support")
    native_weights = _numpy(_finite_tensor(rollout.weights, "native rollout weights", (1, SCENARIO_COUNT)))[0]
    weight_sum = float(native_weights.sum())
    if (native_weights < 0).any() or weight_sum <= 0:
        raise ScenarioMPCInputError("native rollout weights must be nonnegative with positive total mass")
    weights = native_weights / weight_sum
    if not np.allclose(weights, 1.0 / SCENARIO_COUNT, rtol=0.0, atol=1e-10):
        raise ScenarioMPCInputError("native weights are not uniform W1 Monte Carlo weights; refusing a false weighting-method label")
    constraints = {name: _numpy(getattr(packet.constraints, name))[0] for name in PHYSICAL_VECTOR_NAMES}
    constraints["dt"] = float(packet.constraints.dt_hours[0].detach().cpu())
    if constraints["dt"] <= 0:
        raise ScenarioMPCInputError("physical dt must be positive")
    scenarios = {name: np.zeros((SCENARIO_COUNT, TIMELINE, NODES), dtype=np.float64) for name in FORECAST_NAMES}
    scenarios["pue"][:] = np.maximum(constraints["pue_min"], 1.0)
    for name in FORECAST_NAMES:
        scenarios[name][:, origin:origin + horizon] = active[name]
    active_mask = np.zeros(TIMELINE, dtype=bool)
    active_mask[origin:origin + horizon] = True
    spread = {name: np.std(values, axis=0, ddof=0) for name, values in active.items()}
    diagnostics = {
        "scenario_source": "native_W1_action_conditioned_joint_latent_paths",
        "weighting_method": "uniform_native_W1_monte_carlo",
        "legacy_R3_learned_scenario_weights": False,
        "scenario_seed": seed, "path_count": SCENARIO_COUNT,
        "native_max_path_count": maximum_paths,
        "native_weight_sum_before_float64_normalization": weight_sum,
        "normalized_weight_sum": float(weights.sum()),
        "native_latent_shape": list(latent.shape),
        "decoded_active_shapes": {name: [1, *values.shape] for name, values in active.items()},
        "forecast_shapes": {name: list(values.shape) for name, values in scenarios.items()},
        "action_shapes_after_time_slice": {field.name: list(getattr(actions, field.name).shape) for field in fields(WorldActionPacket)},
        "anchor_mode": anchor_mode, "decoded_anchor_shape": list(active_anchor.shape),
        "cap_mode": cap_mode, "decoded_cap_shape": list(active_cap.shape),
        "apic_forward_dtype_identity_max_abs": forward_identity,
        "apic_float64_reexpression_identity_max_abs": float(np.abs(active["da"] - active["rt"] - _numpy(dart)[0]).max()),
        "scenario_axis_std_mean": {name: float(values.mean()) for name, values in spread.items()},
        "scenario_axis_std_max": {name: float(values.max()) for name, values in spread.items()},
        "all_decoded_paths_identical": all(bool((values == 0).all()) for values in spread.values()),
        "inactive_fill": "zero for da/rt/arrivals/pv; physical pue_min >= 1 for pue",
        "source_input_sha256": causal_hash,
        "input_hash_scope": "pooled W1 source state, origin, reference observations, physical constraints, previous actions/masks, risk preferences, issue anchors/caps and sampling seed; excludes model weights",
        "reference_actions": "registered assembler applied to _safe_raw(packet), then time-sliced",
        "compute_path_cost": False, "external_perturbations_added": False,
        "realized_targets_or_batch_mapping_accepted": False,
    }
    return {"scenarios": scenarios, "weights": weights, "constraints": constraints,
            "previous_da": _numpy(packet.previous_actions[..., 0])[0],
            "origin": origin, "stage": stage, "horizon": horizon,
            "action_range": float(risk.action_range[0].detach().cpu()), "active_mask": active_mask,
            "diagnostics": diagnostics}
